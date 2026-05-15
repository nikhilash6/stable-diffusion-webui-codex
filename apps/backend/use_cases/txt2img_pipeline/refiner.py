"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Latent model-swap stages for the txt2img pipeline.
Implements the generic first-pass `swap_model` stage plus SDXL refiner stages (global + hires), loading the selected engine, rebuilding
conditioning, and running an additional sampling pass over existing latents through the shared sampler without prompt-control ownership.
The top-level `swap_model` seam now resumes only from an exact sampler-owned boundary state and enforces same-family compatibility before resume.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_swap_pointer_remaining_steps` (function): Validate a `switch_at_step` pointer against total steps and return the truthful remaining-step count.
- `SwapModelStage` (dataclass): Executable generic first-pass model-swap stage with selector-authoritative engine loading.
- `GlobalSwapModelStage` (class): First-pass swap-model stage for the global (base) scope.
- `RefinerStage` (dataclass): Executable SDXL refiner stage implementing shared refiner sampling logic and selector-authoritative engine loading.
- `GlobalRefinerStage` (class): Refiner stage for the global (base) scope.
- `HiresRefinerStage` (class): Refiner stage for the hires scope.
"""
# // tags: refiner, pipeline, sdxl, hires, swap_model

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from dataclasses import dataclass
from typing import Callable

import torch

from apps.backend.core.rng import ImageRNG
from apps.backend.core.engine_loader import EngineLoadOptions, load_engine as _load_engine
from apps.backend.runtime.model_registry.capabilities import primary_family_for_engine_id
from apps.backend.runtime.processing.datatypes import ConditioningPayload, PromptContext
from apps.backend.runtime.processing.models import CodexProcessingTxt2Img, RefinerConfig, SwapStageConfig
from apps.backend.runtime.pipeline_stages.sampling_execute import execute_sampling, execute_sampling_result
from apps.backend.runtime.pipeline_stages.sampling_plan import build_sampling_plan, ensure_sampler, ensure_sampler_and_rng
from apps.backend.runtime.sampling.driver import SamplingBoundaryState


RefinerConditioningFn = Callable[[CodexProcessingTxt2Img, PromptContext], tuple[object, object]]
RefinerLogFn = Callable[[object, object], None]
RefinerTensorLogFn = Callable[[str, torch.Tensor | None], None]


def resolve_swap_pointer_remaining_steps(*, label: str, swap_at_step: int, total_steps: int) -> int:
    if total_steps < 2:
        raise RuntimeError(f"{label} requires total steps >= 2 for swap semantics (got {total_steps}).")
    if swap_at_step < 1 or swap_at_step >= total_steps:
        raise RuntimeError(f"{label} 'switch_at_step' must be in [1, {total_steps - 1}] (got {swap_at_step}).")
    return int(total_steps - swap_at_step)


@dataclass(slots=True)
class RefinerStage:
    """Executable refiner stage with shared behaviour for global/hires scopes."""

    config: RefinerConfig | None
    label: str

    def is_enabled(self) -> bool:
        cfg = self.config
        return bool(cfg and cfg.enabled and cfg.swap_at_step > 0)

    def run(
        self,
        *,
        processing: CodexProcessingTxt2Img,
        prompt_context: PromptContext,
        noise_settings,
        samples: torch.Tensor,
        compute_conditioning: RefinerConditioningFn,
        log_conditioning: RefinerLogFn,
        log_tensor_stats: RefinerTensorLogFn,
    ) -> torch.Tensor:
        if not self.is_enabled():
            return samples

        cfg = self.config
        assert cfg is not None  # satisfies type-checkers

        selection = cfg.selection
        model_name = selection.require_model_ref(context=self.label)
        if not model_name:
            raise RuntimeError(
                f"{self.label} is enabled but no refiner model was specified. "
                "Provide a valid SDXL refiner checkpoint."
            )

        seed_value = int(cfg.seed)
        if seed_value < 0:
            seed_value = int(torch.randint(0, 2**31 - 1, (1,)).item())

        original_steps = int(processing.steps)
        swap_at_step = int(cfg.swap_at_step)
        effective_refiner_steps = resolve_swap_pointer_remaining_steps(
            label=self.label,
            swap_at_step=swap_at_step,
            total_steps=original_steps,
        )

        logger = get_backend_logger(f"{__name__}.refiner.{self.label.replace(' ', '_').lower()}")
        logger.info(
            "[refiner] starting %s model=%s swap_at_step=%d remaining_steps=%d cfg=%.3f seed=%d",
            self.label,
            model_name,
            swap_at_step,
            effective_refiner_steps,
            cfg.cfg,
            seed_value,
        )

        load_opts = EngineLoadOptions(
            device=None,
            dtype=None,
            attention_backend=None,
            accelerator=None,
            vae_path=selection.vae_path,
            vae_source=selection.vae_source,
            tenc_path=list(selection.tenc_path) if isinstance(selection.tenc_path, tuple) else selection.tenc_path,
            text_encoder_override=dict(selection.text_encoder_override) if selection.text_encoder_override else None,
            checkpoint_core_only=selection.checkpoint_core_only,
            model_format=selection.model_format,
        )
        try:
            refiner_engine = _load_engine(model_name, options=load_opts)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load SDXL refiner engine '{model_name}': {exc}") from exc

        original_sd_model = processing.sd_model
        original_width = processing.width
        original_height = processing.height
        original_cfg = processing.guidance_scale
        original_cfg_scale = getattr(processing, "cfg_scale", processing.guidance_scale)

        try:
            processing.sd_model = refiner_engine
            latent_h, latent_w = samples.shape[-2], samples.shape[-1]
            processing.width = latent_w * 8
            processing.height = latent_h * 8
            processing.guidance_scale = cfg.cfg
            processing.cfg_scale = cfg.cfg
            processing.steps = original_steps

            plan = build_sampling_plan(
                processing,
                seeds=[seed_value],
                subseeds=[seed_value],
                subseed_strength=0.0,
                noise_settings=noise_settings,
            )
            rng = ensure_sampler_and_rng(processing, plan, latent_channels=samples.shape[1])
            noise = rng.next().to(samples)

            cond_ref, uncond_ref = compute_conditioning(processing, prompt_context)
            if cond_ref is None or uncond_ref is None:
                raise RuntimeError(
                    f"Failed to build conditioning for {self.label.lower()}; get_learned_conditioning returned None."
                )

            payload = ConditioningPayload(conditioning=cond_ref, unconditional=uncond_ref)
            log_conditioning(cond_ref, uncond_ref)

            processing.update_extra_param(
                self.label,
                {
                    "model": model_name,
                    "swap_at_step": int(swap_at_step),
                    "effective_refiner_steps": int(effective_refiner_steps),
                    "total_steps": int(original_steps),
                    "cfg": float(cfg.cfg),
                    "seed": int(seed_value),
                },
            )

            samples_refined = execute_sampling(
                processing,
                plan,
                payload,
                prompt_context,
                prompt_context.loras,
                rng=rng,
                noise=noise,
                init_latent=samples,
                start_at_step=swap_at_step,
                denoise_strength=1.0,
            )
            setattr(processing, "_codex_last_decode_engine", refiner_engine)
            log_tensor_stats(f"{self.label.lower().replace(' ', '_')}_samples", samples_refined)
            return samples_refined
        finally:
            processing.sd_model = original_sd_model
            processing.width = original_width
            processing.height = original_height
            processing.guidance_scale = original_cfg
            processing.cfg_scale = original_cfg_scale
            processing.steps = original_steps


@dataclass(slots=True)
class SwapModelStage:
    """Executable first-pass model-swap stage with persistent engine ownership."""

    config: SwapStageConfig | None
    label: str

    def is_enabled(self) -> bool:
        cfg = self.config
        return bool(cfg and cfg.enabled and cfg.swap_at_step > 0)

    def run(
        self,
        *,
        processing: CodexProcessingTxt2Img,
        prompt_context: PromptContext,
        noise_settings,
        boundary_state: SamplingBoundaryState,
        compute_conditioning: RefinerConditioningFn,
        log_conditioning: RefinerLogFn,
        log_tensor_stats: RefinerTensorLogFn,
    ) -> torch.Tensor:
        samples = boundary_state.latent
        if not self.is_enabled():
            return samples

        cfg = self.config
        assert cfg is not None  # satisfies type-checkers

        selection = cfg.selection
        model_name = selection.require_model_ref(context=self.label)
        if not model_name:
            raise RuntimeError(
                f"{self.label} is enabled but no swap model was specified. "
                "Provide a valid checkpoint for the first-pass swap stage."
            )

        if int(cfg.seed) != -1:
            raise RuntimeError(
                f"{self.label} exact resume does not support explicit stage seed overrides. "
                "Leave `swap_model.seed` at -1 so the continuation inherits the already-captured base RNG continuity."
            )

        original_steps = int(processing.steps)
        swap_at_step = int(cfg.swap_at_step)
        remaining_steps = resolve_swap_pointer_remaining_steps(
            label=self.label,
            swap_at_step=swap_at_step,
            total_steps=original_steps,
        )
        if int(boundary_state.completed_steps) != swap_at_step:
            raise RuntimeError(
                f"{self.label} resume boundary mismatch: captured completed_steps={boundary_state.completed_steps} "
                f"but runtime switch_at_step={swap_at_step}."
            )

        logger = get_backend_logger(f"{__name__}.swap_model.{self.label.replace(' ', '_').lower()}")
        logger.info(
            "[swap_model] starting %s model=%s swap_at_step=%d remaining_steps=%d cfg=%.3f seed=inherited",
            self.label,
            model_name,
            swap_at_step,
            remaining_steps,
            cfg.cfg,
        )

        load_opts = EngineLoadOptions(
            device=None,
            dtype=None,
            attention_backend=None,
            accelerator=None,
            vae_path=selection.vae_path,
            vae_source=selection.vae_source,
            tenc_path=list(selection.tenc_path) if isinstance(selection.tenc_path, tuple) else selection.tenc_path,
            text_encoder_override=dict(selection.text_encoder_override) if selection.text_encoder_override else None,
            checkpoint_core_only=selection.checkpoint_core_only,
            model_format=selection.model_format,
            zimage_variant=selection.zimage_variant,
        )
        try:
            swap_engine = _load_engine(model_name, options=load_opts)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load swap_model engine '{model_name}': {exc}") from exc

        original_sd_model = processing.sd_model
        base_engine_id = str(boundary_state.engine_id or "")
        swap_engine_id = str(getattr(swap_engine, "engine_id", "") or "")
        try:
            base_family = primary_family_for_engine_id(base_engine_id)
        except KeyError as exc:
            raise RuntimeError(
                f"{self.label} exact resume requires boundary-state-only family proof, but the captured engine_id "
                f"{base_engine_id or '<empty>'!r} has no primary family mapping."
            ) from exc
        try:
            swap_family = primary_family_for_engine_id(swap_engine_id)
        except KeyError as exc:
            raise RuntimeError(
                f"{self.label} exact resume requires a swap engine with a primary family mapping, but got "
                f"{swap_engine_id or '<empty>'!r}."
            ) from exc
        if swap_family != base_family:
            raise RuntimeError(
                f"{self.label} exact resume requires same-family engines; "
                f"captured={base_engine_id or '<unknown>'} ({base_family.value}) "
                f"swap={swap_engine_id or '<unknown>'} ({swap_family.value})."
            )
        original_width = processing.width
        original_height = processing.height
        original_cfg = processing.guidance_scale
        original_cfg_scale = getattr(processing, "cfg_scale", processing.guidance_scale)
        original_sampler = getattr(processing, "sampler", None)
        original_rng = getattr(processing, "rng", None)
        if not isinstance(original_rng, ImageRNG):
            raise RuntimeError(
                f"{self.label} exact resume requires the captured base `processing.rng` ImageRNG; "
                f"got {type(original_rng).__name__}."
            )
        inherited_seeds = [int(seed) for seed in original_rng.seeds]
        inherited_subseeds = [int(seed) for seed in original_rng.subseeds]
        inherited_subseed_strength = float(original_rng.subseed_strength)
        primary_seed = inherited_seeds[0] if inherited_seeds else -1

        active_sd_model = original_sd_model
        active_sampler = original_sampler
        active_rng = original_rng
        try:
            processing.sd_model = swap_engine
            latent_h, latent_w = samples.shape[-2], samples.shape[-1]
            processing.width = latent_w * 8
            processing.height = latent_h * 8
            processing.guidance_scale = cfg.cfg
            processing.cfg_scale = cfg.cfg
            processing.steps = original_steps

            plan = build_sampling_plan(
                processing,
                seeds=inherited_seeds,
                subseeds=inherited_subseeds,
                subseed_strength=inherited_subseed_strength,
                noise_settings=noise_settings,
            )
            ensure_sampler(processing, plan)

            cond_swap, uncond_swap = compute_conditioning(processing, prompt_context)
            if cond_swap is None or uncond_swap is None:
                raise RuntimeError(
                    f"Failed to build conditioning for {self.label.lower()}; get_learned_conditioning returned None."
                )

            payload = ConditioningPayload(conditioning=cond_swap, unconditional=uncond_swap)
            log_conditioning(cond_swap, uncond_swap)

            processing.update_extra_param(
                self.label,
                {
                    "model": model_name,
                    "swap_at_step": int(swap_at_step),
                    "remaining_steps": int(remaining_steps),
                    "total_steps": int(original_steps),
                    "cfg": float(cfg.cfg),
                    "seed": int(primary_seed),
                    "seed_mode": "inherited_base_rng",
                },
            )

            swapped_samples = execute_sampling_result(
                processing,
                plan,
                payload,
                prompt_context,
                prompt_context.loras,
                rng=original_rng,
                resume_boundary_state=boundary_state,
                start_at_step=swap_at_step,
                denoise_strength=1.0,
            ).samples
            active_sd_model = swap_engine
            active_sampler = getattr(processing, "sampler", None)
            active_rng = original_rng
            setattr(processing, "_codex_last_decode_engine", swap_engine)
            log_tensor_stats(f"{self.label.lower().replace(' ', '_')}_samples", swapped_samples)
            return swapped_samples
        finally:
            processing.sd_model = active_sd_model
            processing.sampler = active_sampler
            processing.rng = active_rng
            processing.width = original_width
            processing.height = original_height
            processing.guidance_scale = original_cfg
            processing.cfg_scale = original_cfg_scale
            processing.steps = original_steps


class GlobalSwapModelStage(SwapModelStage):
    def __init__(self, config: SwapStageConfig | None):
        super().__init__(config=config, label="Swap Model")


class GlobalRefinerStage(RefinerStage):
    def __init__(self, config: RefinerConfig | None):
        super().__init__(config=config, label="Refiner")


class HiresRefinerStage(RefinerStage):
    def __init__(self, config: RefinerConfig | None):
        super().__init__(config=config, label="Hires Refiner")
