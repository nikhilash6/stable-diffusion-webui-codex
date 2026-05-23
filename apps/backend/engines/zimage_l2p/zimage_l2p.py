"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z-Image L2P pixel-space engine facade.
Loads the L2P denoiser + external Qwen3-4B text encoder, builds masked Qwen conditioning, and exposes the exact pixel txt2img hook used by
the canonical txt2img pipeline. L2P has no VAE and does not implement img2img/inpaint/hires/LoRA surfaces.

Symbols (top-level; keep in sync; no ghosts):
- `_ZImageL2PPromptList` (class): Prompt wrapper carrying CFG/cache metadata for L2P conditioning.
- `ZImageL2PEngine` (class): `CodexDiffusionEngine` implementation registered as exact engine id `zimage_l2p`.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.core.rng import ImageRNG
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.common.model_scopes import stage_scoped_model_load
from apps.backend.engines.common.prompt_wrappers import PromptListBase
from apps.backend.engines.common.runtime_lifecycle import require_runtime
from apps.backend.engines.common.tensor_tree import detach_to_cpu, move_to_device
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.models.loader import DiffusionModelBundle
from apps.backend.runtime.processing.datatypes import PromptContext, SamplingPlan
from apps.backend.runtime.logging import get_backend_logger

from .factory import CodexZImageL2PFactory
from .spec import ZIMAGE_L2P_SPEC, ZImageL2PEngineRuntime
from .standalone_sampler import sample_zimage_l2p_pixel_txt2img


logger = get_backend_logger("backend.engines.zimage_l2p")
_ZIMAGE_L2P_FACTORY = CodexZImageL2PFactory(spec=ZIMAGE_L2P_SPEC)


class _ZImageL2PPromptList(PromptListBase):
    """List-like prompt wrapper used to carry per-run L2P metadata."""

    def __init__(
        self,
        items: list[str],
        *,
        cfg_scale: float,
        is_negative_prompt: bool,
        smart_cache: bool | None,
    ) -> None:
        self.cfg_scale = float(cfg_scale)
        super().__init__(items, is_negative_prompt=is_negative_prompt, smart_cache=smart_cache)


class ZImageL2PEngine(CodexDiffusionEngine):
    """Z-Image L2P exact pixel-space txt2img engine."""

    engine_id = "zimage_l2p"
    expected_family = ModelFamily.ZIMAGE_L2P
    requires_vae = False

    def __init__(self) -> None:
        super().__init__()
        self._runtime: ZImageL2PEngineRuntime | None = None
        self._device = str(memory_management.manager.mount_device())
        self._dtype = "fp32"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG,),
            model_types=("zimage_l2p",),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
            extras={"size": "1024x1024", "requires_vae": False},
        )

    @property
    def required_text_encoders(self) -> tuple[str, ...]:
        return ("qwen3_4b",)

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        self.use_distilled_cfg_scale = False
        assembly = _ZIMAGE_L2P_FACTORY.assemble(bundle, options=options)
        runtime = assembly.runtime
        runtime_device = getattr(runtime, "device", None)
        if runtime_device is None:
            raise RuntimeError("Z-Image L2P runtime contract violation: assembled runtime is missing `device`.")
        expected_core_device = str(memory_management.manager.get_device(DeviceRole.CORE))
        if str(runtime_device) != expected_core_device:
            raise RuntimeError(
                "Z-Image L2P runtime contract violation: assembled runtime device does not match memory-manager CORE "
                f"device (runtime={runtime_device}, expected={expected_core_device})."
            )
        runtime_dtype = getattr(runtime, "core_compute_dtype", None)
        if runtime_dtype not in {"bf16", "fp16", "fp32"}:
            raise RuntimeError(
                "Z-Image L2P runtime contract violation: invalid core_compute_dtype "
                f"{runtime_dtype!r}."
            )
        self._runtime = runtime
        self._device = str(runtime_device)
        self._dtype = str(runtime_dtype)
        logger.debug("Z-Image L2P runtime assembled")
        return assembly.codex_objects

    def _on_unload(self) -> None:
        self._runtime = None

    def _require_runtime(self) -> ZImageL2PEngineRuntime:
        return require_runtime(self._runtime, label=self.engine_id)

    def _prepare_prompt_wrappers(
        self,
        texts: list[str],
        proc: Any,
        *,
        is_negative: bool,
    ) -> _ZImageL2PPromptList:
        raw_cfg = getattr(proc, "guidance_scale", None)
        try:
            cfg_scale = float(raw_cfg) if raw_cfg is not None else ZIMAGE_L2P_SPEC.default_cfg_scale
        except Exception:
            cfg_scale = ZIMAGE_L2P_SPEC.default_cfg_scale
        smart_flag = getattr(proc, "smart_cache", None)
        smart_value = None if smart_flag is None else bool(smart_flag)
        return _ZImageL2PPromptList(
            [str(text or "") for text in texts],
            cfg_scale=cfg_scale,
            is_negative_prompt=is_negative,
            smart_cache=smart_value,
        )

    @torch.no_grad()
    def get_learned_conditioning(self, prompts: list[str]) -> list[torch.Tensor]:
        runtime = self._require_runtime()
        qwen_patcher = self.codex_objects.text_encoders["qwen3_4b"].patcher

        texts = tuple(str(item or "") for item in prompts)
        is_negative = bool(getattr(prompts, "is_negative_prompt", False))
        smart_flag = getattr(prompts, "smart_cache", None)
        use_cache = bool(smart_flag) if smart_flag is not None else self.smart_cache_enabled
        max_length = int(getattr(runtime.text.qwen3_text, "max_length", 512) or 512)
        cache_key = (texts, is_negative, max_length)

        cached = self._get_cached_cond(cache_key, bucket_name="zimage_l2p.conditioning", enabled=use_cache)
        if isinstance(cached, list) and all(isinstance(item, torch.Tensor) for item in cached):
            target_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
            core_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE)
            return move_to_device(cached, device=target_device, dtype=core_dtype)

        with stage_scoped_model_load(
            qwen_patcher,
            smart_offload_enabled=self.smart_offload_enabled,
            manager=memory_management.manager,
        ):
            cond = runtime.text.qwen3_text.encode_masked(list(texts))
            if not isinstance(cond, list) or not cond:
                raise RuntimeError("Z-Image L2P Qwen text encoder returned empty masked conditioning.")
            core_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE)
            cond = [tensor.to(dtype=core_dtype) for tensor in cond]
            if use_cache:
                self._set_cached_cond(cache_key, detach_to_cpu(cond), enabled=use_cache)
            return cond

    @torch.no_grad()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        runtime = self._require_runtime()
        tokens = runtime.text.qwen3_text.tokenize([prompt])
        return len(tokens[0]), max(512, len(tokens[0]))

    def encode_first_stage(self, x: torch.Tensor, *, encode_seed: int | None = None) -> torch.Tensor:
        del x, encode_seed
        raise NotImplementedError("zimage_l2p encode_first_stage not yet implemented")

    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        del x
        raise NotImplementedError("zimage_l2p decode_first_stage not yet implemented")

    @torch.inference_mode()
    def sample_pixel_txt2img(
        self,
        *,
        cond: object,
        uncond: object | None,
        sampling_plan: SamplingPlan,
        prompt_context: PromptContext,
        rng: ImageRNG,
        width: int,
        height: int,
        progress_callback: Any | None = None,
    ) -> torch.Tensor:
        del prompt_context
        runtime = self._require_runtime()
        if int(width) != 1024 or int(height) != 1024:
            raise RuntimeError(f"Z-Image L2P first tranche supports exactly 1024x1024; got {width}x{height}.")

        denoiser_patcher = runtime.denoiser
        with stage_scoped_model_load(
            denoiser_patcher,
            smart_offload_enabled=self.smart_offload_enabled,
            manager=memory_management.manager,
        ):
            sampler_model = getattr(denoiser_patcher, "model", None)
            model = getattr(sampler_model, "diffusion_model", None)
            if model is None:
                raise RuntimeError("Z-Image L2P denoiser patcher is missing the wrapped diffusion_model.")
            core_device = memory_management.manager.get_device(DeviceRole.CORE)
            noise = rng.next().to(device=core_device, dtype=torch.float32)
            if tuple(int(value) for value in noise.shape) != (1, 3, 1024, 1024):
                raise RuntimeError(
                    "Z-Image L2P expected canonical pixel noise shape [1,3,1024,1024]; "
                    f"got {tuple(noise.shape)}."
                )
            return sample_zimage_l2p_pixel_txt2img(
                model=model,
                noise=noise,
                cond=cond,
                uncond=uncond,
                steps=int(sampling_plan.steps),
                guidance_scale=float(sampling_plan.guidance_scale),
                flow_shift=float(ZIMAGE_L2P_SPEC.flow_shift),
                progress_callback=progress_callback,
            )


__all__ = ["ZImageL2PEngine"]
