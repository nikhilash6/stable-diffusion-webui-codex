"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDXL diffusion engine implementation (base + refiner) for the backend orchestrator.
Implements SDXL runtime assembly, conditioning, and smart-cache integration; mode wrappers are inherited from `CodexDiffusionEngine` (Option A) and delegate to `apps/backend/use_cases/`.
When smart offload is enabled, CLIP patcher unload is stage-scoped (only unload when this call loaded it) to avoid unload/reload between cond+uncond.

Symbols (top-level; keep in sync; no ghosts):
- `_tensor_stats` (function): Computes basic tensor statistics for debug logging (shape/dtype/device + min/max/mean/std).
- `_validate_conditioning_payload` (function): Validates conditioning-related payload fields against the assembled runtime/spec.
- `_SDXLPrompt` (class): Prompt marker type used for internal prompt/meta handling.
- `_prompt_meta` (function): Computes metadata for a prompt batch (length/count flags) used in caching and diagnostics.
- `_smart_cache_from_prompts` (function): Determines smart-cache behavior hints from prompt content and runtime settings.
- `StableDiffusionXL` (class): Main SDXL engine (loads bundles, assembles runtime, encodes conditioning; mode wrappers delegate to use-cases).
- `StableDiffusionXLRefiner` (class): SDXL refiner engine (second-stage refinement runtime; similar lifecycle to the base engine).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch

from apps.backend.core.engine_interface import EngineCapabilities
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.common.capabilities_presets import (
    DEFAULT_IMAGE_DEVICES,
    DEFAULT_IMAGE_PRECISION,
    IMAGE_TASKS,
)
from apps.backend.engines.common.model_scopes import stage_scoped_model_load
from apps.backend.engines.common.runtime_lifecycle import require_runtime
from apps.backend.engines.common.tensor_tree import detach_to_cpu, move_to_device
from apps.backend.engines.sd._clip_skip import apply_sd_clip_skip
from apps.backend.engines.sd.factory import CodexSDFamilyFactory
from apps.backend.engines.sd.spec import SDXL_REFINER_SPEC, SDXL_SPEC, SDEngineRuntime
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.memory.smart_offload import (
    smart_cache_enabled,
    record_smart_cache_hit,
    record_smart_cache_miss,
)
from apps.backend.runtime.common.nn.unet.layers import Timestep
from apps.backend.runtime.models.loader import DiffusionModelBundle
from apps.backend.runtime.model_registry.specs import ModelFamily


# note: no extra device assertions here; diagnostics should be captured upstream

logger = get_backend_logger("backend.engines.sd.sdxl")

_SDXL_FACTORY = CodexSDFamilyFactory(spec=SDXL_SPEC)
_SDXL_REFINER_FACTORY = CodexSDFamilyFactory(spec=SDXL_REFINER_SPEC)


def _tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    if tensor is None:
        return {"shape": None, "dtype": None, "device": None}
    with torch.no_grad():
        data = tensor.detach()
        stats_tensor = data.float()
        return {
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
            "device": str(data.device),
            "min": float(stats_tensor.min().item()),
            "max": float(stats_tensor.max().item()),
            "mean": float(stats_tensor.mean().item()),
            "std": float(stats_tensor.std(unbiased=False).item()),
        }


_SDXL_DEFAULT_CROP_LEFT = 0
_SDXL_DEFAULT_CROP_TOP = 0


def _validate_conditioning_payload(runtime: SDEngineRuntime, payload: Mapping[str, Any], *, label: str) -> None:
    """Fail fast when CLIP/conditioning outputs are malformed.

    Ensures shapes match the UNet config and that no NaN/Inf values sneak into
    the sampling loop, which would otherwise produce noisy “golesma” results.
    """

    def _require_tensor(key: str) -> torch.Tensor:
        value = payload.get(key) if isinstance(payload, Mapping) else None
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"SDXL conditioning '{label}' missing tensor '{key}' (got {type(value).__name__}).")
        return value

    cross = _require_tensor("crossattn")
    vector = _require_tensor("vector")

    if cross.ndim != 3:
        raise RuntimeError(
            f"SDXL conditioning '{label}' crossattn must be 3D (B, S, C); got shape={tuple(cross.shape)}."
        )
    if vector.ndim != 2:
        raise RuntimeError(
            f"SDXL conditioning '{label}' vector must be 2D (B, F); got shape={tuple(vector.shape)}."
        )

    for name, tensor in (("crossattn", cross), ("vector", vector)):
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"SDXL conditioning '{label}' contains non-finite values in '{name}'. "
                "Check CLIP weights/conversion before sampling."
            )

    cfg = getattr(runtime.unet.model, "diffusion_model", None)
    if cfg is not None:
        cfg = getattr(cfg, "codex_config", None)

    expected_ctx = getattr(cfg, "context_dim", None) if cfg is not None else None
    if isinstance(expected_ctx, int) and int(cross.shape[-1]) != expected_ctx:
        raise RuntimeError(
            f"SDXL conditioning '{label}' context dim {int(cross.shape[-1])} does not match UNet context_dim={expected_ctx}."
        )

    expected_adm = getattr(cfg, "adm_in_channels", None) if cfg is not None else None
    if isinstance(expected_adm, int) and int(vector.shape[1]) != expected_adm:
        raise RuntimeError(
            f"SDXL conditioning '{label}' ADM vector dim {int(vector.shape[1])} does not match adm_in_channels={expected_adm}."
        )


class _SDXLPrompt(str):
    """String subclass that carries SDXL spatial metadata for conditioning."""

    __slots__ = (
        "width",
        "height",
        "target_width",
        "target_height",
        "crop_left",
        "crop_top",
        "is_negative_prompt",
        "smart_cache",
    )

    def __new__(
        cls,
        text: str,
        *,
        width: int,
        height: int,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        crop_left: int = 0,
        crop_top: int = 0,
        is_negative_prompt: bool = False,
        smart_cache: Optional[bool] = None,
    ) -> "_SDXLPrompt":
        obj = super().__new__(cls, text or "")
        obj.width = int(width or 1024)
        obj.height = int(height or 1024)
        obj.target_width = int(target_width or obj.width)
        obj.target_height = int(target_height or obj.height)
        obj.crop_left = int(crop_left or 0)
        obj.crop_top = int(crop_top or 0)
        obj.is_negative_prompt = bool(is_negative_prompt)
        obj.smart_cache = None if smart_cache is None else bool(smart_cache)
        return obj


def _prompt_meta(prompts: Sequence[str]) -> Tuple[int, int, int, int, int, int, bool]:
    reference: Any = prompts
    if isinstance(prompts, (list, tuple)) and prompts:
        reference = prompts[0]

    def _meta(attr: str, default: int) -> int:
        value = getattr(reference, attr, None)
        if value is None:
            value = getattr(prompts, attr, None)
        return int(value if value not in (None, "") else default)

    width = _meta("width", 1024)
    height = _meta("height", 1024)
    target_width = _meta("target_width", width)
    target_height = _meta("target_height", height)
    crop_left = _meta("crop_left", _SDXL_DEFAULT_CROP_LEFT)
    crop_top = _meta("crop_top", _SDXL_DEFAULT_CROP_TOP)
    is_negative = bool(getattr(reference, "is_negative_prompt", getattr(prompts, "is_negative_prompt", False)))
    return width, height, target_width, target_height, crop_left, crop_top, is_negative


def _smart_cache_from_prompts(prompts: Sequence[str]) -> Optional[bool]:
    """Extract Smart Cache override from wrapped prompts when present."""
    try:
        if isinstance(prompts, (list, tuple)) and prompts:
            value = getattr(prompts[0], "smart_cache", None)
        else:
            value = getattr(prompts, "smart_cache", None)
        if value is None:
            return None
        return bool(value)
    except Exception:
        return None

def _sdxl_force_zero_negative_prompt(prompts: Sequence[str], *, is_negative: bool) -> bool:
    if not is_negative:
        return False
    return all(str(x or "").strip() == "" for x in prompts)


def _sdxl_embed_key(
    *,
    height: int,
    width: int,
    target_height: int,
    target_width: int,
    crop_top: int,
    crop_left: int,
) -> tuple[int, int, int, int, int, int]:
    return (
        int(height),
        int(width),
        int(target_height),
        int(target_width),
        int(crop_top),
        int(crop_left),
    )


def _sdxl_get_embed_flat(
    *,
    embedder: Timestep,
    embed_cache: dict[tuple[int, int, int, int, int, int], torch.Tensor],
    embed_key: tuple[int, int, int, int, int, int],
    batch: int,
    dtype_like: torch.Tensor,
    use_cache: bool,
    bucket: str,
) -> torch.Tensor:
    flat = None
    if use_cache:
        flat = embed_cache.get(embed_key)
        if flat is not None:
            record_smart_cache_hit(bucket)
        else:
            record_smart_cache_miss(bucket)

    if flat is None:
        embed_values = [
            embedder(torch.tensor([embed_key[0]])),
            embedder(torch.tensor([embed_key[1]])),
            embedder(torch.tensor([embed_key[4]])),
            embedder(torch.tensor([embed_key[5]])),
            embedder(torch.tensor([embed_key[2]])),
            embedder(torch.tensor([embed_key[3]])),
        ]
        flat_tensor = torch.flatten(torch.cat(embed_values)).unsqueeze(dim=0).detach()
        if use_cache:
            embed_cache.clear()
            # Store cached embeddings on CPU to avoid pinning VRAM.
            embed_cache[embed_key] = flat_tensor.to("cpu")
        flat = flat_tensor

    return flat.repeat(int(batch), 1).to(dtype_like)


class StableDiffusionXL(CodexDiffusionEngine):
    """Codex-native SDXL base engine."""

    engine_id = "sdxl"
    expected_family = ModelFamily.SDXL

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[SDEngineRuntime] = None
        self.embedder = Timestep(256)
        # Cache textual CLIP embeddings by prompt text + polarity (cond/uncond).
        # Spatial metadata is applied per-call via embed_values.
        # Cached tensors are stored on CPU to avoid pinning VRAM between jobs.
        self._cond_cache: dict[
            tuple,
            tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor],
        ] = {}
        # Cache spatial embedding vectors keyed by (h, w, target_h, target_w, crop_top, crop_left).
        # Cached tensors are stored on CPU and moved back to the text encoder device on use.
        self._embed_cache: dict[tuple[int, int, int, int, int, int], torch.Tensor] = {}

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=IMAGE_TASKS,
            model_types=("sdxl",),
            devices=DEFAULT_IMAGE_DEVICES,
            precision=DEFAULT_IMAGE_PRECISION,
        )

    # load() behavior inherited from CodexDiffusionEngine

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        assembly = _SDXL_FACTORY.assemble(bundle, options=dict(options))
        runtime = assembly.runtime
        self._runtime = runtime
        self.register_model_family("sdxl")
        # New runtime / weights invalidate any cached conditioning.
        self._cond_cache.clear()
        self._embed_cache.clear()

        logger.debug(
            "StableDiffusionXL runtime prepared with branches=%s clip_skip=%d",
            runtime.classic_order,
            runtime.classic_engine("clip_l").clip_skip,
        )

        return assembly.codex_objects

    def _on_unload(self) -> None:
        self._runtime = None
        self._embed_cache.clear()

    def _require_runtime(self) -> SDEngineRuntime:
        return require_runtime(self._runtime, label=self.engine_id)

    def set_clip_skip(self, clip_skip: int) -> None:
        runtime = self._require_runtime()
        apply_sd_clip_skip(
            engine=self,
            runtime=runtime,
            clip_skip=clip_skip,
            logger=logger,
            label="SDXL",
        )

    def _prepare_prompt_wrappers(
        self,
        texts: Sequence[str],
        proc: Any,
        *,
        is_negative: bool,
    ) -> List[_SDXLPrompt]:
        width = int(getattr(proc, "width", 1024) or 1024)
        height = int(getattr(proc, "height", 1024) or 1024)
        hires_cfg = getattr(proc, "hires", None)
        target_width = int(getattr(hires_cfg, "resize_x", 0) or width)
        target_height = int(getattr(hires_cfg, "resize_y", 0) or height)
        crop_left = int(getattr(proc, "sdxl_crop_left", _SDXL_DEFAULT_CROP_LEFT) or 0)
        crop_top = int(getattr(proc, "sdxl_crop_top", _SDXL_DEFAULT_CROP_TOP) or 0)
        smart_cache = getattr(proc, "smart_cache", None)

        wrappers: List[_SDXLPrompt] = []
        for entry in texts:
            raw_text = str(entry or "")
            entry_width = int(getattr(entry, "width", width) or width)
            entry_height = int(getattr(entry, "height", height) or height)
            entry_target_width = int(getattr(entry, "target_width", target_width) or target_width)
            entry_target_height = int(getattr(entry, "target_height", target_height) or target_height)
            entry_crop_left = int(getattr(entry, "crop_left", crop_left) or crop_left)
            entry_crop_top = int(getattr(entry, "crop_top", crop_top) or crop_top)
            wrappers.append(
                _SDXLPrompt(
                    raw_text,
                    width=entry_width,
                    height=entry_height,
                    target_width=entry_target_width,
                    target_height=entry_target_height,
                    crop_left=entry_crop_left,
                    crop_top=entry_crop_top,
                    is_negative_prompt=is_negative,
                    smart_cache=smart_cache,
                )
            )
        return wrappers

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
            width, height, target_width, target_height, crop_left, crop_top, is_negative = _prompt_meta(prompt)
            label = "uncond" if is_negative else "cond"
            smart_cache = _smart_cache_from_prompts(prompt)
            use_cache = smart_cache_enabled() if smart_cache is None else bool(smart_cache)

            cache_key = (texts, bool(is_negative))
            te_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)

            cached = self._get_cached_cond(cache_key, bucket_name="sdxl.base.text", enabled=use_cache)
            if cached is not None:
                cond_l, pooled_l, cond_g, pooled_g = move_to_device(cached, device=te_device)
            else:
                out_l = runtime.classic_engine("clip_l")(prompt)
                pooled_l = None
                if isinstance(out_l, tuple) and len(out_l) == 2:
                    cond_l, pooled_l = out_l
                else:
                    cond_l = out_l
                    pooled_l = getattr(cond_l, "pooled", None)

                out_g = runtime.classic_engine("clip_g")(prompt)
                if isinstance(out_g, tuple) and len(out_g) == 2:
                    cond_g, pooled_g = out_g
                else:
                    pooled_g = getattr(out_g, "pooled", None)
                    cond_g = out_g
                    if pooled_g is None:
                        raise RuntimeError(
                            "SDXL CLIP-G did not provide a pooled embedding; cannot build conditioning vector."
                        )
                if use_cache:
                    self._set_cached_cond(
                        cache_key,
                        detach_to_cpu((cond_l, pooled_l, cond_g, pooled_g)),
                        enabled=use_cache,
                    )

            embed_key = _sdxl_embed_key(
                height=height,
                width=width,
                target_height=target_height,
                target_width=target_width,
                crop_top=crop_top,
                crop_left=crop_left,
            )
            flat = _sdxl_get_embed_flat(
                embedder=self.embedder,
                embed_cache=self._embed_cache,
                embed_key=embed_key,
                batch=pooled_g.shape[0],
                dtype_like=pooled_g,
                use_cache=use_cache,
                bucket="sdxl.base.embed",
            )

            force_zero_negative_prompt = _sdxl_force_zero_negative_prompt(prompt, is_negative=bool(is_negative))

            if force_zero_negative_prompt:
                if pooled_l is not None:
                    pooled_l = torch.zeros_like(pooled_l)
                pooled_g = torch.zeros_like(pooled_g)
                cond_l = torch.zeros_like(cond_l)
                cond_g = torch.zeros_like(cond_g)

            cond = {
                "crossattn": torch.cat([cond_l, cond_g], dim=2),
                "vector": torch.cat([pooled_g, flat], dim=1),
            }

            _validate_conditioning_payload(runtime, cond, label=label)

            logger.debug("Generated SDXL conditioning for %d prompts.", len(prompt))
            return cond

    @torch.no_grad()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        engine = runtime.classic_engine("clip_l")
        _, token_count = engine.process_texts([prompt])
        target = engine.get_target_prompt_token_count(token_count)
        return token_count, target

    def _decode_debug_stats_enabled(self) -> bool:
        from apps.backend.infra.config.env_flags import env_flag

        return env_flag("CODEX_SDXL_DEBUG_DECODE_STATS", False) and logger.isEnabledFor(logging.DEBUG)

    def _log_decode_stats(self, stage: str, tensor: torch.Tensor) -> None:
        try:
            logger.debug("[sdxl.decode] %s stats=%s", stage, _tensor_stats(tensor))
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("[sdxl.decode] failed to compute stats for stage=%s", stage, exc_info=True)



class StableDiffusionXLRefiner(CodexDiffusionEngine):
    """Codex-native SDXL refiner engine."""

    engine_id = "sdxl_refiner"
    expected_family = ModelFamily.SDXL_REFINER

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[SDEngineRuntime] = None
        self.embedder = Timestep(256)
        # Cached tensors are stored on CPU to avoid pinning VRAM between jobs.
        self._cond_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._embed_cache: dict[tuple[int, int, int, int, int, int], torch.Tensor] = {}

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=IMAGE_TASKS,
            model_types=("sdxl_refiner",),
            devices=DEFAULT_IMAGE_DEVICES,
            precision=DEFAULT_IMAGE_PRECISION,
        )

    # load() behavior inherited from CodexDiffusionEngine

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        assembly = _SDXL_REFINER_FACTORY.assemble(bundle, options=dict(options))
        runtime = assembly.runtime
        self._runtime = runtime
        self.register_model_family("sdxl")
        self._cond_cache.clear()
        self._embed_cache.clear()

        logger.debug(
            "StableDiffusionXLRefiner runtime prepared with clip_skip=%d",
            runtime.classic_engine("clip_g").clip_skip,
        )

        return assembly.codex_objects

    def _on_unload(self) -> None:
        self._runtime = None
        self._embed_cache.clear()

    def _require_runtime(self) -> SDEngineRuntime:
        return require_runtime(self._runtime, label=self.engine_id)

    def set_clip_skip(self, clip_skip: int) -> None:
        runtime = self._require_runtime()
        apply_sd_clip_skip(
            engine=self,
            runtime=runtime,
            clip_skip=clip_skip,
            logger=logger,
            label="SDXL refiner",
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
            width, height, target_width, target_height, crop_left, crop_top, is_negative = _prompt_meta(prompt)
            label = "uncond" if is_negative else "cond"
            smart_cache = _smart_cache_from_prompts(prompt)
            use_cache = smart_cache_enabled() if smart_cache is None else bool(smart_cache)

            cache_key = (texts, bool(is_negative))
            te_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)

            cached = self._get_cached_cond(cache_key, bucket_name="sdxl.refiner.text", enabled=use_cache)
            if cached is not None:
                cond_g, pooled = move_to_device(cached, device=te_device)
            else:
                cond_g, pooled = runtime.classic_engine("clip_g")(prompt)
                if use_cache:
                    self._set_cached_cond(
                        cache_key,
                        detach_to_cpu((cond_g, pooled)),
                        enabled=use_cache,
                    )

            embed_key = _sdxl_embed_key(
                height=height,
                width=width,
                target_height=target_height,
                target_width=target_width,
                crop_top=crop_top,
                crop_left=crop_left,
            )
            flat = _sdxl_get_embed_flat(
                embedder=self.embedder,
                embed_cache=self._embed_cache,
                embed_key=embed_key,
                batch=pooled.shape[0],
                dtype_like=pooled,
                use_cache=use_cache,
                bucket="sdxl.refiner.embed",
            )

            force_zero_negative_prompt = _sdxl_force_zero_negative_prompt(prompt, is_negative=bool(is_negative))

            if force_zero_negative_prompt:
                pooled = torch.zeros_like(pooled)
                cond_g = torch.zeros_like(cond_g)

            cond = {
                "crossattn": cond_g,
                "vector": torch.cat([pooled, flat], dim=1),
            }

            _validate_conditioning_payload(runtime, cond, label=label)

            logger.debug("Generated SDXL refiner conditioning for %d prompts.", len(prompt))
            return cond

    @torch.no_grad()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        engine = runtime.classic_engine("clip_g")
        _, token_count = engine.process_texts([prompt])
        target = engine.get_target_prompt_token_count(token_count)
        return token_count, target

    def _decode_debug_stats_enabled(self) -> bool:
        from apps.backend.infra.config.env_flags import env_flag

        return env_flag("CODEX_SDXL_DEBUG_DECODE_STATS", False) and logger.isEnabledFor(logging.DEBUG)

    def _log_decode_stats(self, stage: str, tensor: torch.Tensor) -> None:
        try:
            logger.debug("[sdxl_refiner.decode] %s stats=%s", stage, _tensor_stats(tensor))
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("[sdxl_refiner.decode] failed to compute stats for stage=%s", stage, exc_info=True)
