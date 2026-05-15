"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampling execution helper for pipeline orchestrators.
Runs the sampler loop, integrates preview callbacks, applies LoRAs, and triggers post-sample hooks and diagnostics (including ER-SDE option
propagation into the sampler and diagnostic metadata dumps), with explicit control over txt2img image-conditioning fallback injection, exact
same-latent boundary-state capture/resume for top-level swap-model continuations, the internal img2img fixed-step execution seam used by
hires continuations, and optional request-scoped denoiser sessions entered only after canonical LoRA activation.
Also wraps the active sampling pass with the shared IP-Adapter stage when `processing.ip_adapter` is configured.

Symbols (top-level; keep in sync; no ghosts):
- `_maybe_dump_latents` (function): Dump latents to disk when enabled via env flags (debug diagnostics + effective ER-SDE metadata).
- `execute_sampling_result` (function): Execute sampling and return the sampled latents plus an optional captured boundary state for exact same-latent resume.
- `execute_sampling` (function): Execute sampling given processing + plan + conditioning payload and return the sampled latents (supports explicit opt-out of default txt2img image-conditioning fallback).
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import math
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from apps.backend.core import devices
from apps.backend.core.rng import ImageRNG
from apps.backend.core.state import state as backend_state
from apps.backend.patchers.lora_apply import apply_loras_to_engine
from apps.backend.runtime.model_registry.capabilities import ENGINE_SURFACES, semantic_engine_for_engine_id
from apps.backend.runtime.live_preview import (
    LivePreviewMethod,
    decode_preview_image,
    debug_preview_factors_enabled,
    live_preview_method,
    maybe_log_preview_factors,
    preview_interval_steps,
)
from apps.backend.runtime.diagnostics.error_summary import summarize_exception_for_console
from apps.backend.runtime.logging import emit_backend_message
from apps.backend.runtime.processing.conditioners import txt2img_conditioning
from apps.backend.runtime.processing.datatypes import ConditioningPayload, PromptContext, SamplingPlan
from apps.backend.runtime.sampling.context import build_sampling_context
from apps.backend.runtime.sampling.driver import SamplingBoundaryState, SamplingResult
from apps.backend.infra.config.env_flags import env_flag
from apps.backend.infra.config.repo_root import get_repo_root

from .ip_adapter import apply_processing_ip_adapter
from .scripts import collect_lora_selections, run_before_sampling_hooks, run_post_sample_hooks


def _maybe_dump_latents(
    latents: torch.Tensor,
    processing: Any,
    plan: SamplingPlan,
    prompt_context: PromptContext,
) -> None:
    if not env_flag("CODEX_DUMP_LATENTS", default=False):
        return

    path_hint = os.getenv("CODEX_DUMP_LATENTS_PATH")
    if path_hint:
        target = Path(path_hint).expanduser()
    else:
        target = get_repo_root() / "logs" / "diagnostics"
    if not target.suffix:
        timestamp = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        target = target / f"latents-{timestamp}.pt"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "latents": latents.detach().cpu(),
            "metadata": {
                "timestamp_utc": _dt.datetime.utcnow().isoformat(timespec="seconds"),
                "width": int(getattr(processing, "width", 0) or 0),
                "height": int(getattr(processing, "height", 0) or 0),
                "steps": int(plan.steps),
                "guidance_scale": float(plan.guidance_scale),
                "sampler": plan.sampler_name,
                "scheduler": plan.scheduler_name,
                "prompts": prompt_context.prompts,
                "negative_prompts": prompt_context.negative_prompts,
                "seeds": plan.seeds,
                "subseeds": plan.subseeds,
            },
        }
        effective_sampler_name = plan.sampler_name
        if isinstance(effective_sampler_name, str) and effective_sampler_name.strip().lower() == "er sde":
            if plan.er_sde is None:
                payload["metadata"]["er_sde"] = {
                    "solver_type": "er_sde",
                    "max_stage": 3,
                    "eta": 1.0,
                    "s_noise": 1.0,
                }
            else:
                payload["metadata"]["er_sde"] = {
                    "solver_type": plan.er_sde.solver_type,
                    "max_stage": int(plan.er_sde.max_stage),
                    "eta": float(plan.er_sde.eta),
                    "s_noise": float(plan.er_sde.s_noise),
                }
        torch.save(payload, target)
        emit_backend_message("[diagnostics] dumped latents", logger=__name__, target=str(target))
    except Exception as exc:  # noqa: BLE001
        emit_backend_message(
            "Failed to dump latents",
            logger=__name__,
            level="ERROR",
            target=str(target),
            error=summarize_exception_for_console(exc),
        )


def execute_sampling_result(
    processing: Any,
    plan: SamplingPlan,
    payload: ConditioningPayload,
    prompt_context: PromptContext,
    prompt_loras: Sequence[Any],
    *,
    rng: ImageRNG,
    noise: torch.Tensor | None = None,
    image_conditioning: torch.Tensor | None = None,
    allow_txt2img_conditioning_fallback: bool = True,
    img2img_fix_steps: bool = False,
    init_latent: torch.Tensor | None = None,
    start_at_step: int | None = None,
    denoise_strength: float | None = None,
    pre_denoiser_hook: Callable[[torch.Tensor, torch.Tensor, int, int | None], torch.Tensor] | None = None,
    post_denoiser_hook: Callable[[torch.Tensor, torch.Tensor, int, int | None], torch.Tensor] | None = None,
    post_step_hook: Callable[[torch.Tensor, int, int | None], None] | None = None,
    post_sample_hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
    capture_boundary_state_at_step: int | None = None,
    resume_boundary_state: SamplingBoundaryState | None = None,
) -> SamplingResult:
    """Execute the sampler using the provided configuration."""
    exact_resume_active = resume_boundary_state is not None
    exact_resume_noise_baseline: torch.Tensor | None = None
    if exact_resume_active:
        if noise is not None:
            raise RuntimeError("Exact boundary resume does not accept explicit initial-noise overrides.")
        noise = resume_boundary_state.latent.detach().clone()
        exact_resume_noise_baseline = noise.detach().clone()
    elif noise is None:
        noise = rng.next()

    model = processing.sd_model
    if hasattr(model, "codex_objects_original") and model.codex_objects_original is not None:
        model.codex_objects = model.codex_objects_original.shallow_copy()

    run_before_sampling_hooks(processing, prompt_context, plan.seeds, plan.subseeds)

    merged = collect_lora_selections(prompt_loras)
    if merged:
        engine_id = str(getattr(model, "engine_id", "") or "").strip()
        if engine_id:
            try:
                semantic_engine = semantic_engine_for_engine_id(engine_id)
            except KeyError:
                semantic_engine = None
            if semantic_engine is not None and not ENGINE_SURFACES[semantic_engine].supports_lora:
                raise RuntimeError(
                    f"LoRA selections are unsupported for engine '{engine_id}'. "
                    "Remove LoRA prompt tags and lora_sha entries for this request."
                )
    if hasattr(model, "codex_objects_after_applying_lora") and model.codex_objects_after_applying_lora is not None:
        stats = apply_loras_to_engine(model, merged)
        if merged:
            emit_backend_message(
                "[native] Applied LoRA(s)",
                logger=__name__,
                files=stats.files,
                params_touched=stats.params_touched,
            )
        model.codex_objects = model.codex_objects_after_applying_lora.shallow_copy()
    elif merged:
        raise RuntimeError(
            "Prompt/global LoRA selections were provided, but the active model does not expose "
            "`codex_objects_after_applying_lora`."
        )

    sampling_session_factory_attr = "_codex_exact_inpaint_sampling_session_factory"
    sampling_session_factory = getattr(processing, sampling_session_factory_attr, None)
    if sampling_session_factory is None:
        sampling_session = contextlib.nullcontext()
    else:
        if not callable(sampling_session_factory):
            raise TypeError(
                f"{sampling_session_factory_attr} must be a callable returning a context manager; "
                f"got {type(sampling_session_factory).__name__}."
            )
        sampling_session = sampling_session_factory()
        if not hasattr(sampling_session, "__enter__") or not hasattr(sampling_session, "__exit__"):
            raise TypeError(
                f"{sampling_session_factory_attr} must return a context manager; "
                f"got {type(sampling_session).__name__}."
            )

    with sampling_session:
        with apply_processing_ip_adapter(processing):
            if processing.scripts is not None:
                processing.scripts.process_before_every_sampling(
                    processing,
                    x=noise,
                    noise=noise,
                    c=payload.conditioning,
                    uc=payload.unconditional,
                )

            if exact_resume_active:
                if getattr(processing, "modified_noise", None) is not None:
                    processing.modified_noise = None
                    raise RuntimeError(
                        "Exact boundary resume rejects `processing.modified_noise`; initial-noise overrides are unsupported."
                    )
                if exact_resume_noise_baseline is not None and not torch.equal(noise, exact_resume_noise_baseline):
                    raise RuntimeError(
                        "Exact boundary resume rejects in-place initial-noise mutation during sampling hooks/scripts."
                    )
            elif getattr(processing, "modified_noise", None) is not None:
                noise = processing.modified_noise
                processing.modified_noise = None

            preview_method = live_preview_method(default=LivePreviewMethod.FULL)
            preview_interval = int(preview_interval_steps(default=0))
            debug_factors = debug_preview_factors_enabled()
            preview_emitted = False
            progress_owner_token = str(getattr(processing, "_codex_progress_owner_token", "") or "")

            def _preview_cb(denoised_latent: torch.Tensor, step: int, total: int | None) -> None:
                nonlocal preview_emitted
                if preview_interval <= 0:
                    return
                # Skip preview decode on the final step only when at least one preview
                # was already emitted; this preserves low-overhead long runs while still
                # allowing short runs to emit one terminal preview.
                try:
                    if total is not None and int(total) > 0 and int(step) >= int(total) and preview_emitted:
                        if debug_factors:
                            maybe_log_preview_factors(processing, denoised_latent, step=int(step), total=int(total))
                        return
                except Exception:
                    # If step/total are malformed, fall back to best-effort preview.
                    pass
                preview = decode_preview_image(processing, denoised_latent, method=preview_method)
                if preview is None:
                    return
                backend_state.set_current_image(
                    preview,
                    sampling_step=int(step),
                    owner_token=progress_owner_token,
                )
                preview_emitted = True
                if debug_factors:
                    maybe_log_preview_factors(processing, denoised_latent, step=int(step), total=int(total or 0))

            if image_conditioning is None and allow_txt2img_conditioning_fallback:
                image_conditioning = txt2img_conditioning(
                    processing.sd_model,
                    noise,
                    processing.width,
                    processing.height,
                )

            if denoise_strength is not None and math.isclose(float(denoise_strength), 0.0):
                if capture_boundary_state_at_step is not None or resume_boundary_state is not None:
                    raise RuntimeError("Exact boundary capture/resume is invalid when denoise_strength resolves to 0.0.")
                samples = init_latent if init_latent is not None else torch.zeros_like(noise)
                if post_sample_hook is not None:
                    samples = post_sample_hook(samples)
                samples = run_post_sample_hooks(processing, samples)
                _maybe_dump_latents(samples, processing, plan, prompt_context)
                devices.torch_gc()
                return SamplingResult(samples=samples)

            sampler_name = plan.sampler_name
            scheduler_name = plan.scheduler_name
            context = build_sampling_context(
                processing.sd_model,
                sampler_name=sampler_name,
                scheduler_name=scheduler_name,
                steps=int(plan.steps),
                noise_source=plan.noise_settings.source.value,
                eta_noise_seed_delta=plan.noise_settings.eta_noise_seed_delta,
                height=int(getattr(processing, "height", 0) or 0) or None,
                width=int(getattr(processing, "width", 0) or 0) or None,
                device=noise.device,
                dtype=noise.dtype,
            )

            result = processing.sampler.sample_result(
                processing,
                noise,
                payload.conditioning,
                payload.unconditional,
                image_conditioning=image_conditioning,
                init_latent=init_latent,
                resume_boundary_state=resume_boundary_state,
                capture_boundary_state_at_step=capture_boundary_state_at_step,
                start_at_step=start_at_step,
                denoise_strength=denoise_strength,
                img2img_fix_steps=img2img_fix_steps,
                pre_denoiser_hook=pre_denoiser_hook,
                post_denoiser_hook=post_denoiser_hook,
                preview_callback=_preview_cb,
                post_step_hook=post_step_hook,
                post_sample_hook=post_sample_hook,
                context=context,
                er_sde_options=plan.er_sde,
            )

            if result.boundary_state is not None:
                devices.torch_gc()
                return result

            samples = run_post_sample_hooks(processing, result.samples)
            _maybe_dump_latents(samples, processing, plan, prompt_context)
            devices.torch_gc()
            return SamplingResult(samples=samples)


def execute_sampling(
    processing: Any,
    plan: SamplingPlan,
    payload: ConditioningPayload,
    prompt_context: PromptContext,
    prompt_loras: Sequence[Any],
    *,
    rng: ImageRNG,
    noise: torch.Tensor | None = None,
    image_conditioning: torch.Tensor | None = None,
    allow_txt2img_conditioning_fallback: bool = True,
    img2img_fix_steps: bool = False,
    init_latent: torch.Tensor | None = None,
    start_at_step: int | None = None,
    denoise_strength: float | None = None,
    pre_denoiser_hook: Callable[[torch.Tensor, torch.Tensor, int, int | None], torch.Tensor] | None = None,
    post_denoiser_hook: Callable[[torch.Tensor, torch.Tensor, int, int | None], torch.Tensor] | None = None,
    post_step_hook: Callable[[torch.Tensor, int, int | None], None] | None = None,
    post_sample_hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    return execute_sampling_result(
        processing,
        plan,
        payload,
        prompt_context,
        prompt_loras,
        rng=rng,
        noise=noise,
        image_conditioning=image_conditioning,
        allow_txt2img_conditioning_fallback=allow_txt2img_conditioning_fallback,
        img2img_fix_steps=img2img_fix_steps,
        init_latent=init_latent,
        start_at_step=start_at_step,
        denoise_strength=denoise_strength,
        pre_denoiser_hook=pre_denoiser_hook,
        post_denoiser_hook=post_denoiser_hook,
        post_step_hook=post_step_hook,
        post_sample_hook=post_sample_hook,
    ).samples
