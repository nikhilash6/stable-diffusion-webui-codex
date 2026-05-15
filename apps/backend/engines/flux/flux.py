"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux diffusion engine (txt2img/img2img) using the Codex Flux runtime.
Assembles the runtime via `CodexFluxFamilyFactory`, manages conditioning caching, and exposes the hooks required by shared txt2img/img2img workflows (encode/decode/conditioning + optional core streaming controller).
When smart offload is enabled, CLIP patcher unload is stage-scoped (only unload when this call loaded it) to avoid unload/reload within the conditioning stage.

Symbols (top-level; keep in sync; no ghosts):
- `_FluxPromptList` (class): Prompt list wrapper carrying distilled CFG scale + negative/smart-cache flags for conditioning.
- `Flux` (class): Codex diffusion engine implementation for Flux (runtime assembly, conditioning, sampling integration).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Any, Iterable, List, Mapping, Optional

import torch

from apps.backend.core.engine_interface import EngineCapabilities
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.common.capabilities_presets import (
    DEFAULT_IMAGE_DEVICES,
    DEFAULT_IMAGE_PRECISION,
    IMAGE_TASKS,
)
from apps.backend.engines.common.model_scopes import stage_scoped_model_load
from apps.backend.engines.common.prompt_wrappers import PromptListBase
from apps.backend.engines.common.runtime_lifecycle import require_runtime
from apps.backend.engines.common.tensor_tree import detach_to_cpu, move_to_device
from apps.backend.engines.flux._clip_skip import apply_flux_clip_skip
from apps.backend.engines.flux.factory import CodexFluxFamilyFactory
from apps.backend.engines.flux.spec import FLUX_SPEC, FluxEngineRuntime
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.models.loader import DiffusionModelBundle
from apps.backend.runtime.model_registry.specs import ModelFamily

logger = get_backend_logger("backend.engines.flux")

_FLUX_FACTORY = CodexFluxFamilyFactory(spec=FLUX_SPEC)


class _FluxPromptList(PromptListBase):
    def __init__(
        self,
        items: Iterable[str],
        *,
        distilled_cfg_scale: float,
        is_negative_prompt: bool,
        smart_cache: bool | None,
    ) -> None:
        self.distilled_cfg_scale = float(distilled_cfg_scale)
        super().__init__(items, is_negative_prompt=is_negative_prompt, smart_cache=smart_cache)


class Flux(CodexDiffusionEngine):
    """Codex native Flux engine."""

    engine_id = "flux1"
    expected_family = ModelFamily.FLUX
    _factory = _FLUX_FACTORY

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[FluxEngineRuntime] = None
        self._guidance_default = FLUX_SPEC.distilled_cfg_scale_default

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=IMAGE_TASKS,
            model_types=("flux1",),
            devices=DEFAULT_IMAGE_DEVICES,
            precision=DEFAULT_IMAGE_PRECISION,
            extras={
                "samplers": ("euler", "euler a", "ddim", "dpm++ 2m"),
                "schedulers": ("simple", "beta", "normal"),
            },
        )

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        assembly = self._factory.assemble(bundle, options=options)
        runtime = assembly.runtime
        self._runtime = runtime
        self.use_distilled_cfg_scale = runtime.use_distilled_cfg
        logger.debug("Flux runtime prepared (distilled cfg=%s)", runtime.use_distilled_cfg)

        # Note: Streaming is handled in assemble_flux_runtime() via _maybe_enable_streaming_core
        # Check if streaming was enabled and store controller reference
        from apps.backend.runtime.families.flux.streaming import StreamedFluxCore
        core_model = getattr(runtime.denoiser.model, "diffusion_model", runtime.denoiser.model)
        if isinstance(core_model, StreamedFluxCore):
            self._streaming_controller = core_model.controller
        else:
            self._streaming_controller = None

        return assembly.codex_objects

    def _on_unload(self) -> None:
        self._runtime = None
        self._streaming_controller = None

    def _require_runtime(self) -> FluxEngineRuntime:
        return require_runtime(self._runtime, label=self.engine_id)

    def _prepare_prompt_wrappers(
        self,
        texts: list[str],
        proc: Any,
        *,
        is_negative: bool,
    ) -> _FluxPromptList:
        distilled = getattr(proc, "distilled_guidance_scale", None)
        if distilled is None:
            distilled = getattr(proc, "distilled_cfg", None)
        try:
            distilled_value = float(distilled) if distilled is not None else float(self._guidance_default)
        except Exception:
            distilled_value = float(self._guidance_default)
        smart_flag = getattr(proc, "smart_cache", None)
        smart_value = None if smart_flag is None else bool(smart_flag)
        return _FluxPromptList(
            [str(t or "") for t in texts],
            distilled_cfg_scale=distilled_value,
            is_negative_prompt=is_negative,
            smart_cache=smart_value,
        )

    def set_clip_skip(self, clip_skip: int) -> None:
        runtime = self._require_runtime()
        apply_flux_clip_skip(
            engine=self,
            runtime=runtime,
            clip_skip=clip_skip,
            logger=logger,
            label=self.engine_id,
        )

    @torch.no_grad()
    def get_learned_conditioning(self, prompt: List[str]):
        runtime = self._require_runtime()
        clip_patcher = self.codex_objects.text_encoders["clip"].patcher
        with stage_scoped_model_load(
            clip_patcher,
            smart_offload_enabled=self.smart_offload_enabled,
            manager=memory_management.manager,
        ):
            texts = tuple(str(x or "") for x in prompt)
            is_negative = bool(getattr(prompt, "is_negative_prompt", False))
            smart_flag = getattr(prompt, "smart_cache", None)
            use_cache = bool(smart_flag) if smart_flag is not None else self.smart_cache_enabled
            distilled_cfg_scale = getattr(prompt, "distilled_cfg_scale", self._guidance_default)
            try:
                distilled_cfg_scale = float(distilled_cfg_scale)
            except Exception:
                distilled_cfg_scale = float(self._guidance_default)
            cache_key = (texts, is_negative, distilled_cfg_scale)

            cached = self._get_cached_cond(cache_key, bucket_name="flux.conditioning", enabled=use_cache)
            if isinstance(cached, dict):
                target_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
                cond = move_to_device(cached, device=target_device)
                logger.debug("[flux] conditioning cache hit for %d prompts", len(prompt))
                return cond

            # Cache miss - compute conditioning
            clip_branch = runtime.text.clip_text
            cond_l, pooled_l = (clip_branch(prompt) if clip_branch is not None else (None, None))
            cond_t5 = runtime.text.t5_text(prompt)
            cond = {"crossattn": cond_t5}

            if pooled_l is not None:
                cond["vector"] = pooled_l

            if self.use_distilled_cfg_scale:
                cond["guidance"] = torch.full((len(prompt),), float(distilled_cfg_scale), dtype=torch.float32)
                logger.info(
                    "[flux] guidance enabled: scale=%.2f shape=%s", distilled_cfg_scale, tuple(cond["guidance"].shape)
                )
            else:
                logger.info("[flux] guidance disabled (schnell variant)")

            # Debug: log all cond keys and shapes
            cond_info = {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__ for k, v in cond.items()}
            logger.info("[flux] conditioning dict: %s", cond_info)

            if use_cache:
                self._set_cached_cond(cache_key, detach_to_cpu(cond), enabled=use_cache)

            return cond

    @torch.no_grad()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        token_count = len(runtime.text.t5_text.tokenize([prompt])[0])
        return token_count, max(255, token_count)
