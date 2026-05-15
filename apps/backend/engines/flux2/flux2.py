"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: FLUX.2 Klein backend engine for the truthful 4B/base-4B image-generation slice.
Assembles the FLUX.2 runtime via the factory seam, exposes Qwen3 prompt conditioning as sampler-ready
`{"crossattn", "vector"}` payloads, wires the dedicated FLUX.2 image-conditioned img2img wrapper, and overrides
first-stage latent encode/decode for normalized external 32-channel FLUX.2 latents while honoring the shared optional
img2img `encode_seed` contract.

Symbols (top-level; keep in sync; no ghosts):
- `_Flux2PromptList` (class): Prompt wrapper carrying negative/smart-cache flags for FLUX.2 conditioning.
- `Flux2Engine` (class): Backend diffusion engine registered under engine id `flux2`.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import math
from typing import Any, Iterable, Mapping

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.common.model_scopes import stage_scoped_model_load
from apps.backend.engines.common.prompt_wrappers import PromptListBase
from apps.backend.engines.common.runtime_lifecycle import require_runtime
from apps.backend.engines.common.tensor_tree import detach_to_cpu
from apps.backend.engines.flux2.factory import CodexFlux2Factory
from apps.backend.engines.flux2.spec import FLUX2_SPEC, Flux2EngineRuntime
from apps.backend.runtime.families.flux2.runtime import decode_flux2_external_latents, encode_flux2_external_latents
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.model_registry.capabilities import ENGINE_SURFACES, SemanticEngine
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.models.loader import DiffusionModelBundle

logger = get_backend_logger("backend.engines.flux2")
_FLUX2_FACTORY = CodexFlux2Factory(spec=FLUX2_SPEC)


class _Flux2PromptList(PromptListBase):
    def __init__(
        self,
        items: Iterable[str],
        *,
        is_negative_prompt: bool,
        smart_cache: bool | None,
    ) -> None:
        super().__init__(items, is_negative_prompt=is_negative_prompt, smart_cache=smart_cache)


class Flux2Engine(CodexDiffusionEngine):
    """Truthful FLUX.2 Klein engine (4B/base-4B; txt2img + image-conditioned img2img)."""

    engine_id = "flux2"
    expected_family = ModelFamily.FLUX2

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Flux2EngineRuntime | None = None
        self._distilled_variant = False

    def capabilities(self) -> EngineCapabilities:
        surface = ENGINE_SURFACES[SemanticEngine.FLUX2]
        tasks: list[TaskType] = []
        if surface.supports_txt2img:
            tasks.append(TaskType.TXT2IMG)
        if surface.supports_img2img:
            tasks.append(TaskType.IMG2IMG)
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=tuple(tasks),
            model_types=("flux2",),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
            extras={
                "samplers": ("euler", "dpm++ 2m"),
                "schedulers": ("simple",),
            },
        )

    def _prepare_prompt_wrappers(
        self,
        texts: list[str],
        proc: Any,
        *,
        is_negative: bool,
    ) -> _Flux2PromptList:
        self._apply_runtime_request_contract(proc)
        prepared_texts = [str(t or "") for t in texts]
        if self._distilled_variant and is_negative:
            if any(text.strip() for text in prepared_texts):
                raise NotImplementedError(
                    "FLUX.2-klein-4B does not support user negative prompts in this backend seam. "
                    "Leave the negative prompt empty and use distilled guidance only."
                )
            prepared_texts = [""] * len(prepared_texts)
        smart_flag = getattr(proc, "smart_cache", None)
        smart_value = None if smart_flag is None else bool(smart_flag)
        return _Flux2PromptList(
            prepared_texts,
            is_negative_prompt=is_negative,
            smart_cache=smart_value,
        )

    @property
    def required_text_encoders(self) -> tuple[str, ...]:
        return ("qwen3",)

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        assembly = _FLUX2_FACTORY.assemble(bundle, options=options)
        runtime = assembly.runtime
        self._runtime = runtime
        self._distilled_variant = bool(runtime.use_distilled_cfg)
        self.use_distilled_cfg_scale = runtime.use_distilled_cfg
        self._device = str(runtime.device)
        self._dtype = str(runtime.core_compute_dtype)
        logger.debug(
            "FLUX.2 runtime assembled: variant=%s distilled_cfg=%s core_dtype=%s",
            runtime.variant,
            runtime.use_distilled_cfg,
            runtime.core_compute_dtype,
        )
        return assembly.codex_objects

    def _on_unload(self) -> None:
        self._runtime = None
        self._distilled_variant = False

    def _require_runtime(self) -> Flux2EngineRuntime:
        return require_runtime(self._runtime, label=self.engine_id)

    def _resolve_effective_guidance_scale(self, proc: Any) -> float:
        default_guidance = float(getattr(proc, "guidance_scale", 4.0) or 4.0)
        if not self._distilled_variant:
            if not math.isfinite(default_guidance) or default_guidance <= 0.0:
                raise ValueError(f"FLUX.2 guidance_scale must be a finite value > 0, got {default_guidance!r}.")
            return default_guidance

        distilled = getattr(proc, "distilled_guidance_scale", None)
        if distilled is None:
            distilled = getattr(proc, "distilled_cfg", None)
        value = default_guidance if distilled is None else float(distilled)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"FLUX.2 distilled guidance must be a finite value > 0, got {value!r}.")
        return value

    def _apply_runtime_request_contract(self, proc: Any) -> None:
        if not self._distilled_variant:
            return
        negative_prompt = getattr(proc, "negative_prompt", None)
        if isinstance(negative_prompt, str) and negative_prompt.strip():
            raise NotImplementedError(
                "FLUX.2-klein-4B does not support user negative prompts in this backend seam. "
                "Leave the negative prompt empty and use distilled guidance only."
            )
        guidance = self._resolve_effective_guidance_scale(proc)
        proc.guidance_scale = guidance
        proc.cfg_scale = guidance

    @staticmethod
    def _build_inert_vector(*, crossattn: torch.Tensor) -> torch.Tensor:
        if not isinstance(crossattn, torch.Tensor) or crossattn.ndim != 3:
            raise TypeError(
                "FLUX.2 inert pooled/vector placeholder requires a 3D crossattn tensor; "
                f"got {type(crossattn).__name__} shape={getattr(crossattn, 'shape', None)}."
            )
        batch = int(crossattn.shape[0])
        return torch.zeros((batch, 1), device=crossattn.device, dtype=crossattn.dtype)

    @torch.no_grad()
    def get_learned_conditioning(self, prompt: list[str]):
        runtime = self._require_runtime()
        qwen_patcher = self.codex_objects.text_encoders["qwen3"].patcher

        texts = tuple(str(x or "") for x in prompt)
        is_negative = bool(getattr(prompt, "is_negative_prompt", False))
        smart_flag = getattr(prompt, "smart_cache", None)
        use_cache = bool(smart_flag) if smart_flag is not None else self.smart_cache_enabled
        max_length = int(getattr(runtime.text.qwen3_text, "max_length", 512) or 512)
        cache_key = (texts, is_negative, max_length)

        cached = self._get_cached_cond(cache_key, bucket_name="flux2.conditioning", enabled=use_cache)
        if isinstance(cached, dict):
            target_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
            core_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE)
            return {
                "crossattn": cached["crossattn"].to(device=target_device, dtype=core_dtype),
                "vector": cached["vector"].to(device=target_device, dtype=core_dtype),
            }
        if cached is not None:
            raise RuntimeError(
                "FLUX.2 conditioning cache returned unsupported payload type: "
                f"{type(cached).__name__}."
            )

        with stage_scoped_model_load(
            qwen_patcher,
            smart_offload_enabled=self.smart_offload_enabled,
            manager=memory_management.manager,
        ):
            cond = runtime.text.qwen3_text(list(texts))
            if not isinstance(cond, torch.Tensor) or cond.ndim != 3:
                raise RuntimeError(
                    "FLUX.2 text encoder returned invalid conditioning tensor: "
                    f"type={type(cond).__name__} shape={getattr(cond, 'shape', None)}."
                )
            target_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
            core_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE)
            cond = cond.to(device=target_device, dtype=core_dtype)
            conditioning = {
                "crossattn": cond,
                "vector": self._build_inert_vector(crossattn=cond),
            }
            if use_cache:
                self._set_cached_cond(cache_key, detach_to_cpu(conditioning), enabled=use_cache)
            return conditioning

    @torch.no_grad()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        runtime = self._require_runtime()
        return runtime.text.qwen3_text.prompt_lengths(prompt)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor, *, encode_seed: int | None = None) -> torch.Tensor:
        vae_target = self._vae_memory_target()
        memory_management.manager.load_model(
            vae_target,
            source="engines.flux2.encode_first_stage",
            stage="encode_first_stage",
            component_hint="vae",
        )
        unload_vae = self.smart_offload_enabled
        try:
            sample = encode_flux2_external_latents(
                self.codex_objects.vae,
                x.movedim(1, -1) * 0.5 + 0.5,
                encode_seed=encode_seed,
            )
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.manager.unload_model(
                    vae_target,
                    source="engines.flux2.encode_first_stage",
                    stage="encode_first_stage",
                    component_hint="vae",
                )

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        vae_target = self._vae_memory_target()
        denoiser_target = self._denoiser_memory_target()
        preserve_denoiser_residency = (
            denoiser_target is not None
            and denoiser_target is not vae_target
            and memory_management.manager.is_model_loaded(denoiser_target)
        )
        if preserve_denoiser_residency:
            memory_management.manager.load_models(
                [vae_target, denoiser_target],
                source="engines.flux2.decode_first_stage",
                stage="decode_first_stage",
                component_hint="vae",
                event_reason="preserve_denoiser_residency",
            )
        else:
            memory_management.manager.load_model(
                vae_target,
                source="engines.flux2.decode_first_stage",
                stage="decode_first_stage",
                component_hint="vae",
            )
        unload_vae = self.smart_offload_enabled
        try:
            sample = decode_flux2_external_latents(self.codex_objects.vae, x)
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.manager.unload_model(
                    vae_target,
                    source="engines.flux2.decode_first_stage",
                    stage="decode_first_stage",
                    component_hint="vae",
                )

    def img2img(self, request: Any, **kwargs: Any) -> Iterable[Any]:
        """FLUX.2 img2img wrapper (image_latents conditioning, no classic init-latent denoise semantics)."""

        from apps.backend.engines.flux2.img2img import run_flux2_img2img

        yield from run_flux2_img2img(engine=self, request=request)


__all__ = ["Flux2Engine"]
