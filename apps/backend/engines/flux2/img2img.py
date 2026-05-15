"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: FLUX.2-specific img2img/inpaint wrapper using truthful image-latent conditioning.
Owns the backend-side FLUX.2 img2img seam because the shared canonical img2img route only knows classic init-latent denoise
or Flux Kontext semantics. This wrapper keeps the existing event/progress envelope, injects VAE-encoded `image_latents`
to match `Flux2KleinPipeline(image=...)`, and reuses the shared masked bundle/full-res composite path when FLUX.2 inpaint
is requested instead of pretending SD-family `image_conditioning` semantics apply directly. Partial denoise now uses
clean `image_latents` plus sampler-native continuation from `init_latent`, and unmasked hires reuses shared hires prep
while keeping masked hires fail-loud; hires prompt parsing is LoRA-only, base/request LoRAs stay inherited unless prompt-local hires tags
override them, explicit hires request overrides stay request-owned, the wrapper seeds a per-run progress-owner token before pre-sampling
VAE encode begins, and the shared generic inpaint-mode hook owner is reused instead of re-implementing it here.

Symbols (top-level; keep in sync; no ghosts):
- `_resolve_flux2_sampling_inputs` (function): Resolves noise/init-latent/denoise args for FLUX.2 continuation sampling.
- `generate_flux2_img2img` (function): Run FLUX.2 image-conditioned img2img and return sampled latents.
- `run_flux2_img2img` (function): Canonical event-stream wrapper for the FLUX.2 img2img seam.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

from dataclasses import replace
import json
import logging
import math
import threading
from typing import Any, Iterator, Sequence

import torch

from apps.backend.core.requests import Img2ImgRequest
from apps.backend.engines.util.adapters import build_img2img_processing
from apps.backend.runtime.processing.conditioners import encode_image_batch, resolve_processing_encode_seed
from apps.backend.runtime.pipeline_stages.image_init import prepare_init_bundle
from apps.backend.runtime.pipeline_stages.hires_fix import (
    prepare_hires_latents_and_conditioning,
    resolve_hires_family_strategy,
)
from apps.backend.runtime.pipeline_stages.masked_img2img import (
    apply_inpaint_full_res_composite,
    prepare_masked_img2img_bundle,
    resolve_mask_enforcer_hooks,
)
from apps.backend.runtime.pipeline_stages.prompt_context import (
    apply_prompt_context,
    build_hires_prompt_context,
    build_prompt_context,
)
from apps.backend.runtime.pipeline_stages.sampling_execute import execute_sampling
from apps.backend.runtime.pipeline_stages.sampling_plan import (
    build_sampling_plan,
    ensure_sampler_and_rng,
    resolve_sampler_scheduler_override,
)
from apps.backend.runtime.pipeline_stages.scripts import run_process_scripts
from apps.backend.runtime.processing.datatypes import GenerationResult, PromptContext, SamplingPlan
from apps.backend.runtime.processing.models import CodexProcessingImg2Img
from apps.backend.runtime.text_processing import (
    clear_last_extra_generation_params,
    snapshot_last_extra_generation_params,
)
from apps.backend.runtime.sampling.driver import CodexSampler
from apps.backend.use_cases._image_streaming import (
    _build_common_info,
    _decode_generation_output,
    _ImageProgressProfile,
    _iter_image_progress_events,
    _resolve_seed_plan,
    _resolve_progress_owner_token,
    _run_inference_worker,
    _seed_progress_owner_token,
)
from apps.backend.use_cases.img2img import _compute_conditioning_payload


logger = get_backend_logger("backend.engines.flux2.img2img")


def _conditioning_cache_hit_metadata(processing: CodexProcessingImg2Img) -> dict[str, object]:
    return {"conditioning_cache_hit": bool(getattr(processing, "_codex_conditioning_cache_hit", False))}


def _derive_seeds(processing: CodexProcessingImg2Img) -> tuple[list[int], list[int], float]:
    seeds = list(getattr(processing, "seeds", []) or [])
    if not seeds:
        seeds = [int(getattr(processing, "seed", -1) or -1)]
    subseeds = list(getattr(processing, "subseeds", []) or [])
    strength = float(getattr(processing, "subseed_strength", 0.0) or 0.0)
    return seeds, subseeds, strength


def _hires_enabled(processing: CodexProcessingImg2Img) -> bool:
    return bool(getattr(getattr(processing, "hires", None), "enabled", False))


def _reject_unsupported_processing(processing: CodexProcessingImg2Img) -> None:
    if getattr(processing, "init_image", None) is None:
        raise ValueError("FLUX.2 img2img requires init_image.")
    if bool(getattr(processing, "mask_region_split", False)):
        raise NotImplementedError(
            "FLUX.2 masked img2img does not support mask_region_split; multi-pass region splitting is not wired in this seam."
        )
    if processing.has_mask() and _hires_enabled(processing):
        raise NotImplementedError(
            "FLUX.2 masked img2img hires does not support masks/inpaint in this backend seam yet."
        )


def _resolve_flux2_sampling_inputs(
    *,
    init_latent: torch.Tensor,
    rng: Any,
    denoise_strength: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float | None]:
    if not CodexSampler.uses_img2img_continuation(denoise_strength):
        return None, None, None
    noise = rng.next().to(init_latent)
    return noise, init_latent, float(denoise_strength)


def _read_denoise_strength(processing: CodexProcessingImg2Img, *, default: float = 1.0) -> float:
    raw_value = getattr(processing, "denoising_strength", default)
    if raw_value is None:
        return float(default)
    return float(raw_value)


def _resolve_flux2_hires_execution(
    processing: CodexProcessingImg2Img,
    *,
    emit_plan_event: bool,
) -> tuple[str, int, int, int, float]:
    if not _hires_enabled(processing):
        raise RuntimeError("Hires execution was requested but processing.hires.enabled is false.")
    model = getattr(processing, "sd_model", None)
    engine_id = str(getattr(model, "engine_id", "") or "").strip()
    if engine_id == "":
        raise RuntimeError("Hires is enabled but processing.sd_model.engine_id is unavailable.")
    hires_strategy = resolve_hires_family_strategy(engine_id)
    setattr(processing, "_codex_hires_strategy", hires_strategy)

    hi_cfg = processing.hires
    upscaler_id = hi_cfg.require_upscaler_id()
    from apps.backend.runtime.vision.upscalers.specs import LATENT_UPSCALE_MODES

    if upscaler_id not in LATENT_UPSCALE_MODES and not upscaler_id.startswith("spandrel:"):
        raise ValueError(
            f"Invalid 'hires.upscaler': {upscaler_id!r}. "
            "Expected a 'latent:*' or 'spandrel:*' upscaler id from GET /api/upscalers."
        )

    target_width, target_height = hi_cfg.resolve_target_dimensions(
        base_width=int(processing.width),
        base_height=int(processing.height),
    )
    steps = hi_cfg.resolve_second_pass_steps(base_steps=int(processing.steps))

    denoise = float(hi_cfg.denoise)
    if emit_plan_event:
        emit_backend_event(
            "pipeline.hires.plan",
            logger=_LOG.name,
            mode=str(getattr(processing, "_codex_mode", "img2img") or "img2img"),
            stage="hires.plan",
            task_id=getattr(processing, "_codex_task_id", None),
            correlation_id=getattr(processing, "_codex_correlation_id", None),
            correlation_source=getattr(processing, "_codex_correlation_source", None),
            engine_id=engine_id,
            strategy=hires_strategy,
            upscaler_id=upscaler_id,
            target_width=target_width,
            target_height=target_height,
            steps=int(steps),
            denoise=float(denoise),
        )
    return upscaler_id, int(target_width), int(target_height), int(steps), float(denoise)


def _build_flux2_hr_prompt_context(
    processing: CodexProcessingImg2Img,
    base_context: PromptContext,
) -> PromptContext:
    hi_cfg = processing.hires
    return build_hires_prompt_context(
        prompt_seed=hi_cfg.prompt if hi_cfg.prompt else base_context.prompts,
        negative_seed=(
            [hi_cfg.negative_prompt]
            if hi_cfg.negative_prompt
            else list(base_context.negative_prompts)
        ),
        base_context=base_context,
    )


def _run_flux2_hires_pass(
    processing: CodexProcessingImg2Img,
    plan: SamplingPlan,
    base_samples: torch.Tensor,
    base_context: PromptContext,
    *,
    upscaler_id: str,
    target_width: int,
    target_height: int,
    steps: int,
    denoise: float,
) -> torch.Tensor:
    hi_cfg = processing.hires

    original = {
        "prompts": processing.prompts,
        "negative_prompts": getattr(processing, "negative_prompts", []),
        "width": processing.width,
        "height": processing.height,
        "guidance_scale": processing.guidance_scale,
        "distilled_guidance_scale": getattr(processing, "distilled_guidance_scale", 3.5),
        "cfg_scale": getattr(processing, "cfg_scale", processing.guidance_scale),
        "steps": processing.steps,
        "denoising_strength": getattr(processing, "denoising_strength", 0.75),
        "sampler_name": getattr(processing, "sampler_name", None),
        "scheduler": getattr(processing, "scheduler", None),
        "sampler": getattr(processing, "sampler", None),
    }

    hi_prompt_context = _build_flux2_hr_prompt_context(processing, base_context)

    try:
        processing.prompts = hi_prompt_context.prompts
        processing.negative_prompts = hi_prompt_context.negative_prompts
        processing.width = target_width
        processing.height = target_height
        effective_target_width = int(processing.width)
        effective_target_height = int(processing.height)
        processing.guidance_scale = float(hi_cfg.cfg or processing.guidance_scale)
        processing.distilled_guidance_scale = float(
            getattr(hi_cfg, "distilled_cfg", getattr(processing, "distilled_guidance_scale", 3.5))
            or getattr(processing, "distilled_guidance_scale", 3.5)
        )
        processing.cfg_scale = processing.guidance_scale
        processing.steps = int(steps)
        processing.denoising_strength = denoise
        hires_runtime_plan = replace(
            plan,
            steps=int(processing.steps),
            guidance_scale=float(processing.guidance_scale),
        )
        hires_sampler, hires_scheduler = resolve_sampler_scheduler_override(
            base_sampler=str(hires_runtime_plan.sampler_name or ""),
            base_scheduler=str(hires_runtime_plan.scheduler_name or ""),
            sampler_override=getattr(hi_cfg, "sampler_name", None),
            scheduler_override=getattr(hi_cfg, "scheduler", None),
        )
        processing.sampler_name = hires_sampler
        processing.scheduler = hires_scheduler
        processing.sampler = CodexSampler(processing.sd_model, algorithm=hires_sampler)
        processing._codex_effective_hires_sampling = {
            "sampler": hires_sampler,
            "scheduler": hires_scheduler,
            "steps": int(steps),
            "denoise": float(denoise),
            "width": int(effective_target_width),
            "height": int(effective_target_height),
        }
        request_contract = getattr(getattr(processing, "sd_model", None), "_apply_runtime_request_contract", None)
        if callable(request_contract):
            request_contract(processing)
        processing.prepare_prompt_data()

        hires_inputs = prepare_hires_latents_and_conditioning(
            processing,
            base_samples=base_samples,
            base_decoded=None,
            target_width=int(effective_target_width),
            target_height=int(effective_target_height),
            upscaler_id=upscaler_id,
            tile=getattr(hi_cfg, "tile", None),
        )
        if hires_inputs.continuation_mode not in {"image_latents", "image_latents_denoise"}:
            raise RuntimeError(
                "FLUX.2 hires expected image-latents continuation from the shared hires prep stage; "
                f"got {hires_inputs.continuation_mode!r}."
            )
        if hires_inputs.image_conditioning is not None:
            raise RuntimeError("FLUX.2 hires must not produce SD-style image_conditioning tensors.")
        hires_init_latent = getattr(hires_inputs, "init_latent", None)
        sampling_init_latent = (
            hires_init_latent
            if isinstance(hires_init_latent, torch.Tensor)
            else hires_inputs.latents
        )

        hires_payload = _compute_conditioning_payload(
            processing,
            hi_prompt_context,
            hi_prompt_context.prompts,
            conditioning=None,
            unconditional_conditioning=None,
        )
        if not isinstance(hires_payload.conditioning, dict):
            raise TypeError(
                "FLUX.2 hires requires dict conditioning so image_latents can be injected truthfully; "
                f"got {type(hires_payload.conditioning).__name__}."
            )
        hires_payload.conditioning["image_latents"] = hires_inputs.latents
        if isinstance(hires_payload.unconditional, dict):
            hires_payload.unconditional["image_latents"] = hires_inputs.latents

        hr_plan = replace(
            hires_runtime_plan,
            sampler_name=hires_sampler,
            scheduler_name=hires_scheduler,
            steps=int(processing.steps),
            guidance_scale=float(processing.guidance_scale),
        )
        rng = ensure_sampler_and_rng(processing, hr_plan)
        noise = rng.next().to(sampling_init_latent)
        return execute_sampling(
            processing,
            hr_plan,
            hires_payload,
            hi_prompt_context,
            hi_prompt_context.loras,
            rng=rng,
            noise=noise,
            init_latent=sampling_init_latent,
            denoise_strength=denoise,
            start_at_step=0,
            img2img_fix_steps=True,
            allow_txt2img_conditioning_fallback=False,
        )
    finally:
        processing.prompts = original["prompts"]
        processing.negative_prompts = original["negative_prompts"]
        processing.width = original["width"]
        processing.height = original["height"]
        processing.guidance_scale = original["guidance_scale"]
        processing.distilled_guidance_scale = original["distilled_guidance_scale"]
        processing.cfg_scale = original["cfg_scale"]
        processing.steps = original["steps"]
        processing.denoising_strength = original["denoising_strength"]
        processing.sampler_name = original["sampler_name"]
        processing.scheduler = original["scheduler"]
        processing.sampler = original["sampler"]


def generate_flux2_img2img(
    processing: CodexProcessingImg2Img,
    conditioning: Any,
    unconditional_conditioning: Any,
    prompts: Sequence[str],
    *,
    seeds: Sequence[int] | None = None,
    subseeds: Sequence[int] | None = None,
    subseed_strength: float | None = None,
) -> GenerationResult:
    if not isinstance(processing, CodexProcessingImg2Img):
        raise TypeError("generate_flux2_img2img expects CodexProcessingImg2Img")

    setattr(processing, "_codex_pipeline_mode", "img2img")
    setattr(processing, "_codex_conditioning_cache_hit", False)
    _reject_unsupported_processing(processing)
    request_contract = getattr(getattr(processing, "sd_model", None), "_apply_runtime_request_contract", None)
    if callable(request_contract):
        request_contract(processing)

    prompt_context = build_prompt_context(processing, prompts)
    apply_prompt_context(processing, prompt_context)

    seed_list, subseed_list, subseed_value = _derive_seeds(processing)
    if seeds is not None:
        seed_list = list(seeds)
    if subseeds is not None:
        subseed_list = list(subseeds)
    if subseed_strength is not None:
        subseed_value = float(subseed_strength)

    plan = build_sampling_plan(processing, seed_list, subseed_list, subseed_value)
    rng = ensure_sampler_and_rng(processing, plan)

    processing.seeds = list(plan.seeds)
    processing.subseeds = list(plan.subseeds)
    processing.guidance_scale = plan.guidance_scale
    processing.cfg_scale = plan.guidance_scale
    processing.steps = plan.steps
    processing.prepare_prompt_data()

    run_process_scripts(processing)
    hires_execution = _resolve_flux2_hires_execution(processing, emit_plan_event=True) if _hires_enabled(processing) else None

    pre_denoiser_hook = None
    post_denoiser_hook = None
    post_step_hook = None
    post_sample_hook = None
    full_res_plan = None
    encode_seed = resolve_processing_encode_seed(processing)
    if processing.has_mask():
        enforcement = getattr(processing, "inpaint_mode", None)
        masked_bundle, enforcer = prepare_masked_img2img_bundle(
            processing,
            plan,
            enforce_mode=enforcement,
            include_image_conditioning=False,
        )
        image_latents = encode_image_batch(
            processing.sd_model,
            masked_bundle.init_tensor,
            stage="engines.flux2.img2img.mask_conditioning_latents",
            encode_seed=encode_seed,
        )
        continuation_latent = masked_bundle.init_latent
        full_res_plan = masked_bundle.full_res
        hooks = resolve_mask_enforcer_hooks(enforcer, enforce_mode=enforcement)
        pre_denoiser_hook = hooks.pre_denoiser
        post_denoiser_hook = hooks.post_denoiser
        post_step_hook = hooks.post_step
        post_sample_hook = hooks.post_sample
    else:
        init_bundle = prepare_init_bundle(processing)
        image_latents = init_bundle.latents
        continuation_latent = init_bundle.latents
    processing.init_latent = continuation_latent
    processing.image_conditioning = None
    payload = _compute_conditioning_payload(
        processing,
        prompt_context,
        prompts,
        conditioning,
        unconditional_conditioning,
    )
    if not isinstance(payload.conditioning, dict):
        raise TypeError(
            "FLUX.2 img2img requires dict conditioning so image_latents can be injected truthfully; "
            f"got {type(payload.conditioning).__name__}."
        )
    payload.conditioning["image_latents"] = image_latents
    if isinstance(payload.unconditional, dict):
        payload.unconditional["image_latents"] = image_latents
    denoise_strength = float(getattr(processing, "denoising_strength", 1.0) or 1.0)
    noise, init_latent, sampling_denoise_strength = _resolve_flux2_sampling_inputs(
        init_latent=continuation_latent,
        rng=rng,
        denoise_strength=denoise_strength,
    )

    samples = execute_sampling(
        processing,
        plan,
        payload,
        prompt_context,
        prompt_context.loras,
        rng=rng,
        noise=noise,
        init_latent=init_latent,
        denoise_strength=sampling_denoise_strength,
        start_at_step=0,
        img2img_fix_steps=False,
        allow_txt2img_conditioning_fallback=False,
        pre_denoiser_hook=pre_denoiser_hook,
        post_denoiser_hook=post_denoiser_hook,
        post_step_hook=post_step_hook,
        post_sample_hook=post_sample_hook,
    )

    if hires_execution is not None:
        if full_res_plan is not None:
            raise NotImplementedError(
                "FLUX.2 masked img2img hires does not support masks/inpaint in this backend seam yet."
            )
        upscaler_id, target_width, target_height, steps, denoise = hires_execution
        samples = _run_flux2_hires_pass(
            processing,
            plan,
            samples,
            prompt_context,
            upscaler_id=upscaler_id,
            target_width=int(target_width),
            target_height=int(target_height),
            steps=int(steps),
            denoise=float(denoise),
        )

    metadata = dict(_conditioning_cache_hit_metadata(processing))
    if full_res_plan is not None:
        metadata["full_res_plan"] = full_res_plan
    return GenerationResult(
        samples=samples,
        decoded=None,
        metadata=metadata,
    )


def run_flux2_img2img(*, engine: Any, request: Any) -> Iterator["InferenceEvent"]:
    """Run FLUX.2 img2img as a canonical event stream."""

    from apps.backend.core.requests import ResultEvent

    if not isinstance(request, Img2ImgRequest):
        raise TypeError("run_flux2_img2img expects Img2ImgRequest")

    engine.ensure_loaded()

    proc = build_img2img_processing(request)
    proc.sd_model = engine
    task_context = str(threading.current_thread().name or "").strip() or "unknown-thread"
    setattr(proc, "_codex_pipeline_mode", "img2img")
    task_id: str | None = None
    marker = "-task-"
    if marker in task_context:
        candidate = task_context.split(marker, 1)[1].strip()
        if candidate:
            task_id = candidate
    if task_id is not None:
        setattr(proc, "_codex_task_id", task_id)
        setattr(proc, "_codex_correlation_id", task_id)
        setattr(proc, "_codex_hires_correlation_id", task_id)
        setattr(proc, "_codex_correlation_source", "task_id")
    progress_owner_token = _resolve_progress_owner_token(task_context=task_context, task_id=task_id)
    setattr(proc, "_codex_progress_owner_token", progress_owner_token)
    _seed_progress_owner_token(progress_owner_token=progress_owner_token)

    base_seed, seeds, subseeds, subseed_strength = _resolve_seed_plan(
        seed=getattr(request, "seed", None),
        batch_total=proc.batch_total,
    )
    proc.seed = base_seed
    proc.seeds = list(seeds)
    proc.subseed = -1
    proc.subseeds = list(subseeds)

    prompts = list(getattr(proc, "prompts", []) or []) or [proc.prompt]
    smart_flags = {
        "smart_offload": bool(getattr(proc, "smart_offload", False)),
        "smart_fallback": bool(getattr(proc, "smart_fallback", False)),
        "smart_cache": bool(getattr(proc, "smart_cache", False)),
    }

    def _generate() -> dict[str, object]:
        import time

        cleanup_targets: list[Any] = [engine]
        sampling_start = 0.0
        sampling_end = 0.0
        active_decode_engine: Any = engine

        try:
            clear_last_extra_generation_params()
            sampling_start = time.perf_counter()
            output = generate_flux2_img2img(
                proc,
                conditioning=None,
                unconditional_conditioning=None,
                prompts=prompts,
                seeds=seeds,
                subseeds=subseeds,
                subseed_strength=subseed_strength,
            )
            sampling_end = time.perf_counter()

            output_decode_engine = getattr(output, "decode_engine", None)
            active_decode_engine = output_decode_engine if output_decode_engine is not None else getattr(proc, "sd_model", None)
            if active_decode_engine is None:
                active_decode_engine = engine
            if active_decode_engine is not None and not any(existing is active_decode_engine for existing in cleanup_targets):
                cleanup_targets.append(active_decode_engine)

            images, decode_ms = _decode_generation_output(
                engine=active_decode_engine,
                output=output,
                task_label="img2img",
            )
            full_res_plan = output.metadata.get("full_res_plan")
            if full_res_plan is not None:
                images = apply_inpaint_full_res_composite(images, plan=full_res_plan)

            all_seeds = list(getattr(proc, "all_seeds", []) or []) or list(seeds)
            seed_value = int(all_seeds[0]) if all_seeds else int(base_seed)

            extra_params: dict[str, object] = {}
            try:
                extra_params.update(snapshot_last_extra_generation_params())
                extra_params.update(getattr(proc, "extra_generation_params", {}) or {})
            except Exception:  # noqa: BLE001
                extra_params = getattr(proc, "extra_generation_params", {}) or {}

            timings: dict[str, float] = {
                "sampling_ms": max(0.0, (sampling_end - sampling_start) * 1000.0),
                "decode_ms": float(decode_ms),
            }
            img2img_mode = "masked_image_latents" if full_res_plan is not None else "image_latents"
            mode_info: dict[str, object] = {
                "img2img_mode": img2img_mode,
                "denoise_strength": _read_denoise_strength(proc, default=1.0),
            }
            if bool(getattr(getattr(proc, "hires", None), "enabled", False)):
                try:
                    mode_info["hires"] = getattr(proc, "hires", None).as_dict()
                except Exception:  # noqa: BLE001
                    pass
                effective_hires_sampling = getattr(proc, "_codex_effective_hires_sampling", None)
                if isinstance(effective_hires_sampling, dict) and effective_hires_sampling:
                    mode_info["effective_hires_sampling"] = dict(effective_hires_sampling)

            info = _build_common_info(
                engine_id=engine.engine_id,
                task="img2img",
                proc=proc,
                seed=seed_value,
                all_seeds=all_seeds,
                extra_params=extra_params,
                timings_ms=timings,
                mode_info=mode_info,
            )
            return {"images": images, "info": json.dumps(info)}
        finally:
            processing_model = getattr(proc, "sd_model", None)
            if processing_model is not None and not any(existing is processing_model for existing in cleanup_targets):
                cleanup_targets.append(processing_model)
            for target in cleanup_targets:
                post_cleanup = getattr(target, "_post_txt2img_cleanup", None)
                if callable(post_cleanup):
                    post_cleanup()

    done, outcome = _run_inference_worker(
        name=f"{engine.engine_id}-img2img-worker",
        fn=_generate,
        runtime_overrides=smart_flags,
    )

    img2img_mode = "masked_image_latents" if proc.has_mask() else "image_latents"
    yield from _iter_image_progress_events(
        done=done,
        outcome=outcome,
        progress_owner_token=progress_owner_token,
        profile=_ImageProgressProfile(
            encode_weight=10.0,
            sampling_weight=80.0,
            decode_weight=10.0,
            extra_data={"img2img_mode": img2img_mode},
            include_sampling_block_alias=True,
        ),
    )

    if outcome.error is not None:
        raise outcome.error

    payload = outcome.output
    if not isinstance(payload, dict):
        raise RuntimeError(
            "FLUX.2 img2img worker returned invalid payload type; expected dict with 'images' and 'info'. "
            f"Got {type(payload).__name__}."
        )
    images = payload.get("images")
    info = payload.get("info")
    if not isinstance(images, list):
        raise RuntimeError("FLUX.2 img2img worker payload field 'images' must be list.")
    if not isinstance(info, str):
        raise RuntimeError("FLUX.2 img2img worker payload field 'info' must be JSON string.")
    yield ResultEvent(payload={"images": images, "info": info})


__all__ = ["generate_flux2_img2img", "run_flux2_img2img"]
