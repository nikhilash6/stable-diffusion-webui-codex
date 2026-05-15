"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Live preview helpers (decode strategies + preview-factor fitting/logging).
Provides live preview decoding (full VAE vs cheap approximation) and an optional least-squares fitting tool to derive latent→RGB factors for
debugging preview quality.

Symbols (top-level; keep in sync; no ghosts):
- `debug_preview_factors_enabled` (function): Indicates whether preview factor fitting logs are enabled.
- `debug_preview_factors_sample_limit` (function): Returns the pixel sample cap used for factor fitting.
- `LivePreviewMethod` (enum): Preview decode strategy (`Full` VAE vs `Approx cheap`).
- `preview_runtime_overrides` (function): Context manager for per-thread preview overrides (interval + method).
- `preview_interval_steps` (function): Returns the effective preview interval (thread overrides first, env fallback).
- `live_preview_method` (function): Returns the effective preview method (thread overrides first, env fallback).
- `live_preview_method_from_env` (function): Reads `CODEX_LIVE_PREVIEW_METHOD` into a `LivePreviewMethod`.
- `live_preview_method_to_env` (function): Converts a `LivePreviewMethod` into an env-friendly string.
- `_tensor_to_pil_rgb` (function): Converts a tensor image into a PIL RGB image.
- `_PreviewProjectionSpec` (dataclass): Internal latent->RGB projection contract for cheap preview profiles.
- `_preview_family_id` (function): Resolves normalized model family id from runtime processing/model metadata.
- `_preview_profile_from_family_or_engine` (function): Resolves canonical preview profile id for a model family/engine id.
- `_preview_profile_id` (function): Resolves preview profile id from family + latent channel count.
- `_canonical_preview_profile_id` (function): Normalizes legacy/alias profile ids to canonical profile ids.
- `_projection_for_profile` (function): Returns internal projection spec for a profile/channels pair when available.
- `_projection_is_valid` (function): Validates projection shape and finiteness contract.
- `_warn_projection_skip_once` (function): Emits deduplicated skip warnings for missing/invalid cheap projections.
- `decode_preview_image` (function): Decodes a denoised latent into a preview image using the selected method.
- `PreviewFactorsFit` (dataclass): Fit result container for latent→RGB factors and bias (with MSE and VAE metadata).
- `fit_preview_factors` (function): Fits latent→RGB factors via least squares against a decoded VAE image (debug tool).
- `maybe_log_preview_factors` (function): Logs preview-factor fits once per job when enabled.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Optional

from apps.backend.core.state import state as backend_state
from apps.backend.infra.config.env_flags import env_flag, env_int
from apps.backend.runtime.model_registry.specs import ModelFamily

logger = get_backend_logger(__name__)

_LATENT_RGB_FACTORS_SD15: tuple[tuple[float, float, float], ...] = (
    (0.3512, 0.2297, 0.3227),
    (0.3250, 0.4974, 0.2350),
    (-0.2829, 0.1762, 0.2721),
    (-0.2120, -0.2616, -0.7177),
)

_LATENT_RGB_FACTORS_SDXL: tuple[tuple[float, float, float], ...] = (
    (0.3920, 0.4054, 0.4549),
    (-0.2634, -0.0196, 0.0653),
    (0.0568, 0.1687, -0.0755),
    (-0.3112, -0.2359, -0.2076),
)

_PREVIEW_PROFILE_SD15 = "sd15"
_PREVIEW_PROFILE_SDXL = "sdxl"
_PREVIEW_PROFILE_FLOW16 = "flow16"

_PREVIEW_ZERO_BIAS: tuple[float, float, float] = (0.0, 0.0, 0.0)
_LATENT_RGB_FACTORS_FLOW16_BOOTSTRAP: tuple[tuple[float, float, float], ...] = (
    (0.3920, 0.4054, 0.4549),
    (-0.2634, -0.0196, 0.0653),
    (0.0568, 0.1687, -0.0755),
    (-0.3112, -0.2359, -0.2076),
    (0.1960, 0.2027, 0.2274),
    (-0.1317, -0.0098, 0.0326),
    (0.0284, 0.08435, -0.03775),
    (-0.1556, -0.11795, -0.1038),
    (0.0980, 0.10135, 0.1137),
    (-0.06585, -0.0049, 0.016325),
    (0.0142, 0.042175, -0.018875),
    (-0.0778, -0.058975, -0.0519),
    (0.0490, 0.050675, 0.05685),
    (-0.032925, -0.00245, 0.0081625),
    (0.0071, 0.0210875, -0.0094375),
    (-0.0389, -0.0294875, -0.02595),
)

_PREVIEW_PROFILE_BY_FAMILY_OR_ENGINE_ID: dict[str, str] = {
    ModelFamily.SD15.value: _PREVIEW_PROFILE_SD15,
    ModelFamily.SD20.value: _PREVIEW_PROFILE_SD15,
    ModelFamily.SDXL.value: _PREVIEW_PROFILE_SDXL,
    ModelFamily.SDXL_REFINER.value: _PREVIEW_PROFILE_SDXL,
    ModelFamily.SD3.value: _PREVIEW_PROFILE_FLOW16,
    ModelFamily.SD35.value: _PREVIEW_PROFILE_FLOW16,
    ModelFamily.FLUX.value: _PREVIEW_PROFILE_FLOW16,
    ModelFamily.FLUX_KONTEXT.value: _PREVIEW_PROFILE_FLOW16,
    ModelFamily.CHROMA.value: _PREVIEW_PROFILE_FLOW16,
    ModelFamily.ZIMAGE.value: _PREVIEW_PROFILE_FLOW16,
    ModelFamily.ANIMA.value: _PREVIEW_PROFILE_FLOW16,
    ModelFamily.AURA.value: _PREVIEW_PROFILE_FLOW16,
    ModelFamily.WAN22_5B.value: _PREVIEW_PROFILE_FLOW16,
    ModelFamily.WAN22_14B.value: _PREVIEW_PROFILE_FLOW16,
    ModelFamily.WAN22_ANIMATE.value: _PREVIEW_PROFILE_FLOW16,
    ModelFamily.QWEN_IMAGE.value: _PREVIEW_PROFILE_FLOW16,
    ModelFamily.HUNYUAN.value: _PREVIEW_PROFILE_FLOW16,
    ModelFamily.SVD.value: _PREVIEW_PROFILE_FLOW16,
    "flux1_fill": _PREVIEW_PROFILE_FLOW16,
    "flux1_chroma": _PREVIEW_PROFILE_FLOW16,
    "hunyuan_video": _PREVIEW_PROFILE_FLOW16,
}

_PREVIEW_PROFILE_ALIASES: dict[str, str] = {
    "anima": _PREVIEW_PROFILE_FLOW16,
}

_DEBUG_PREVIEW_FACTORS_LAST_JOB_TS: str | None = None

_THREAD_OVERRIDES = threading.local()


@dataclass(frozen=True)
class _PreviewProjectionSpec:
    profile_id: str
    channels: int
    factors: tuple[tuple[float, float, float], ...]
    bias: tuple[float, float, float] = _PREVIEW_ZERO_BIAS


_PREVIEW_PROJECTIONS: dict[tuple[str, int], _PreviewProjectionSpec] = {
    (_PREVIEW_PROFILE_SD15, 4): _PreviewProjectionSpec(
        profile_id=_PREVIEW_PROFILE_SD15,
        channels=4,
        factors=_LATENT_RGB_FACTORS_SD15,
    ),
    (_PREVIEW_PROFILE_SDXL, 4): _PreviewProjectionSpec(
        profile_id=_PREVIEW_PROFILE_SDXL,
        channels=4,
        factors=_LATENT_RGB_FACTORS_SDXL,
    ),
    (_PREVIEW_PROFILE_FLOW16, 16): _PreviewProjectionSpec(
        profile_id=_PREVIEW_PROFILE_FLOW16,
        channels=16,
        factors=_LATENT_RGB_FACTORS_FLOW16_BOOTSTRAP,
    ),
}


def _get_override(name: str) -> object | None:
    return getattr(_THREAD_OVERRIDES, name, None)


@contextmanager
def preview_runtime_overrides(
    *,
    interval_steps: int | None = None,
    method: "LivePreviewMethod" | None = None,
) -> Iterator[None]:
    """Temporarily override preview settings for the current thread.

    This avoids mutating `os.environ` (process-global) for per-task settings. It is safe for
    concurrent generation because worker threads carry independent thread-local overrides.
    """

    prev_interval = _get_override("preview_interval_steps")
    prev_method = _get_override("live_preview_method")
    _THREAD_OVERRIDES.preview_interval_steps = interval_steps
    _THREAD_OVERRIDES.live_preview_method = method
    try:
        yield
    finally:
        _THREAD_OVERRIDES.preview_interval_steps = prev_interval
        _THREAD_OVERRIDES.live_preview_method = prev_method


def debug_preview_factors_enabled() -> bool:
    return env_flag("CODEX_DEBUG_PREVIEW_FACTORS", default=False)


def debug_preview_factors_sample_limit() -> int:
    return env_int("CODEX_DEBUG_PREVIEW_FACTORS_SAMPLES", default=4096, min_value=256)


class LivePreviewMethod(str, Enum):
    FULL = "Full"
    APPROX_CHEAP = "Approx cheap"

    @staticmethod
    def from_string(value: str | None, *, default: "LivePreviewMethod" = FULL) -> "LivePreviewMethod":
        key = (value or "").strip().lower()
        if key in {"full", "vae", ""}:
            return LivePreviewMethod.FULL
        if key in {"approx cheap", "approx_cheap", "approx-cheap", "cheap"}:
            return LivePreviewMethod.APPROX_CHEAP
        return default


def live_preview_method_from_env(*, default: LivePreviewMethod = LivePreviewMethod.FULL) -> LivePreviewMethod:
    return LivePreviewMethod.from_string(os.getenv("CODEX_LIVE_PREVIEW_METHOD"), default=default)


def live_preview_method(*, default: LivePreviewMethod = LivePreviewMethod.FULL) -> LivePreviewMethod:
    """Return the effective live preview method (thread overrides first)."""

    override = _get_override("live_preview_method")
    if override is not None:
        if isinstance(override, LivePreviewMethod):
            return override
        return LivePreviewMethod.from_string(str(override), default=default)
    return live_preview_method_from_env(default=default)


def live_preview_method_to_env(method: LivePreviewMethod) -> str:
    return method.value


def preview_interval_steps(*, default: int = 0) -> int:
    """Return the effective preview interval in steps (thread overrides first)."""

    override = _get_override("preview_interval_steps")
    if override is not None:
        try:
            return max(0, int(override))
        except Exception:
            return max(0, int(default))
    return env_int("CODEX_PREVIEW_INTERVAL", default=default, min_value=0)


def _tensor_to_pil_rgb(tensor: Any) -> Any:
    import numpy as np
    from PIL import Image

    arr = tensor.detach().cpu().float().clamp(-1, 1)
    arr = ((arr + 1.0) * 0.5).mul(255.0).byte().movedim(0, -1).numpy()
    return Image.fromarray(np.asarray(arr), mode="RGB")


def _preview_family_id(processing: Any) -> str | None:
    model = getattr(processing, "sd_model", None)
    if model is None:
        return None

    expected_family = getattr(model, "expected_family", None)
    if expected_family is not None:
        expected_value = getattr(expected_family, "value", expected_family)
        key = str(expected_value).strip().lower()
        return key or None

    engine_id = str(getattr(model, "engine_id", "") or "").strip().lower()
    if engine_id:
        return engine_id
    return None


def _preview_profile_from_family_or_engine(family_or_engine_id: str | None) -> str | None:
    if family_or_engine_id is None:
        return None
    key = str(family_or_engine_id).strip().lower()
    if key == "":
        return None
    return _PREVIEW_PROFILE_BY_FAMILY_OR_ENGINE_ID.get(key, key)


def _preview_profile_id(processing: Any, *, channels: int) -> str:
    resolved_channels = int(channels)
    family_profile = _preview_profile_from_family_or_engine(_preview_family_id(processing))
    if resolved_channels == 4:
        if family_profile == _PREVIEW_PROFILE_SDXL or bool(getattr(getattr(processing, "sd_model", None), "is_sdxl", False)):
            return _PREVIEW_PROFILE_SDXL
        return _PREVIEW_PROFILE_SD15
    if family_profile is not None:
        return family_profile
    if resolved_channels == 16:
        return _PREVIEW_PROFILE_FLOW16
    return "unknown"


def _canonical_preview_profile_id(profile_id: str | None) -> str | None:
    if profile_id is None:
        return None
    normalized = str(profile_id).strip().lower()
    if normalized == "":
        return None
    return _PREVIEW_PROFILE_ALIASES.get(normalized, normalized)


def _projection_for_profile(profile_id: str, *, channels: int) -> _PreviewProjectionSpec | None:
    canonical_profile_id = _canonical_preview_profile_id(profile_id)
    if canonical_profile_id is None:
        return None
    return _PREVIEW_PROJECTIONS.get((canonical_profile_id, int(channels)))


def _projection_is_valid(spec: _PreviewProjectionSpec) -> bool:
    if int(spec.channels) <= 0:
        return False
    if len(spec.factors) != int(spec.channels):
        return False
    if len(spec.bias) != 3:
        return False
    for row in spec.factors:
        if len(row) != 3:
            return False
        if any(not math.isfinite(float(value)) for value in row):
            return False
    if any(not math.isfinite(float(value)) for value in spec.bias):
        return False
    return True


def _warn_projection_skip_once(
    processing: Any,
    *,
    method: LivePreviewMethod,
    profile_id: str,
    channels: int,
    reason: str,
) -> None:
    dedupe_key = (method.value, str(profile_id), int(channels), str(reason))
    dedupe_bucket_name = "_codex_preview_warned_projection_skips"
    try:
        dedupe_bucket = getattr(processing, dedupe_bucket_name, None)
        if dedupe_bucket is None:
            dedupe_bucket = set()
            setattr(processing, dedupe_bucket_name, dedupe_bucket)
        if isinstance(dedupe_bucket, set):
            if dedupe_key in dedupe_bucket:
                return
            dedupe_bucket.add(dedupe_key)
    except Exception:
        pass
    logger.warning(
        "Live preview method '%s' cannot resolve projection (%s) for profile=%s channels=%d; skipping preview.",
        method.value,
        reason,
        profile_id,
        int(channels),
    )


def decode_preview_image(processing: Any, denoised_latent: Any, *, method: LivePreviewMethod) -> Any | None:
    import torch
    import torch.nn.functional as F

    if not isinstance(denoised_latent, torch.Tensor):
        return None
    if denoised_latent.ndim != 4:
        return None

    if method == LivePreviewMethod.APPROX_CHEAP:
        channels = int(denoised_latent.shape[1])
        profile_id = _preview_profile_id(processing, channels=channels)
        projection = _projection_for_profile(profile_id, channels=channels)
        if projection is None:
            _warn_projection_skip_once(
                processing,
                method=method,
                profile_id=profile_id,
                channels=channels,
                reason="missing",
            )
            return None
        if not _projection_is_valid(projection):
            _warn_projection_skip_once(
                processing,
                method=method,
                profile_id=profile_id,
                channels=channels,
                reason="invalid",
            )
            return None
        mat = torch.tensor(projection.factors, device=denoised_latent.device, dtype=denoised_latent.dtype)
        rgb_small = torch.einsum("blhw,lr->brhw", denoised_latent, mat)
        if projection.bias != _PREVIEW_ZERO_BIAS:
            bias = torch.tensor(projection.bias, device=denoised_latent.device, dtype=denoised_latent.dtype).view(1, 3, 1, 1)
            rgb_small = rgb_small + bias
        rgb = F.interpolate(rgb_small, scale_factor=8, mode="bilinear", align_corners=False)
        return _tensor_to_pil_rgb(rgb[0])

    if method == LivePreviewMethod.FULL:
        from apps.backend.runtime.processing.conditioners import decode_latent_batch

        decoded = decode_latent_batch(processing.sd_model, denoised_latent)
        return _tensor_to_pil_rgb(decoded[0])

    logger.warning("Unknown live preview method '%s'; skipping preview.", method.value)
    return None


@dataclass(frozen=True)
class PreviewFactorsFit:
    model_name: str
    channels: int
    step: int
    total: int
    scale_h: int
    scale_w: int
    sample_count: int
    mse: float
    factors: tuple[tuple[float, float, float], ...]
    bias: tuple[float, float, float]
    vae_meta: dict[str, object]


def fit_preview_factors(
    processing: Any,
    denoised_latent: Any,
    *,
    step: int,
    total: int,
    sample_limit: int,
) -> Optional[PreviewFactorsFit]:
    import torch
    import torch.nn.functional as F

    from apps.backend.runtime.processing.conditioners import decode_latent_batch

    if not isinstance(denoised_latent, torch.Tensor) or denoised_latent.ndim != 4:
        return None

    try:
        decoded = decode_latent_batch(processing.sd_model, denoised_latent).detach().float()
    except Exception as exc:
        logger.warning("[preview-factors] decode_first_stage failed: %s", exc)
        return None

    if decoded.ndim != 4 or decoded.shape[1] != 3:
        logger.warning("[preview-factors] unexpected decoded shape=%s; skipping.", tuple(decoded.shape))
        return None

    latent_h, latent_w = int(denoised_latent.shape[-2]), int(denoised_latent.shape[-1])
    if latent_h <= 0 or latent_w <= 0:
        return None

    decoded_small = F.interpolate(decoded, size=(latent_h, latent_w), mode="area")

    channels = int(denoised_latent.shape[1])
    latent = denoised_latent.detach().float()[0].movedim(0, -1).reshape(-1, channels)
    rgb = decoded_small[0].movedim(0, -1).reshape(-1, 3)

    n_pixels = int(latent.shape[0])
    if n_pixels <= 0:
        return None

    sample_n = min(int(sample_limit), n_pixels)
    idx = torch.linspace(0, n_pixels - 1, steps=sample_n, device=latent.device).long()
    latent_s = latent.index_select(0, idx)
    rgb_s = rgb.index_select(0, idx)

    ones = torch.ones((sample_n, 1), device=latent_s.device, dtype=latent_s.dtype)
    latent_aug = torch.cat([latent_s, ones], dim=1)

    try:
        sol = torch.linalg.lstsq(latent_aug, rgb_s).solution  # (C+1, 3)
    except Exception as exc:
        logger.warning("[preview-factors] lstsq failed: %s", exc)
        return None

    pred = latent_aug @ sol
    mse = (pred - rgb_s).pow(2).mean().item()

    factors_rows = tuple(tuple(float(v) for v in row) for row in sol[:-1].detach().cpu().tolist())
    bias_row = tuple(float(v) for v in sol[-1].detach().cpu().tolist())

    scale_h = int(round(decoded.shape[-2] / float(latent_h))) if latent_h else 0
    scale_w = int(round(decoded.shape[-1] / float(latent_w))) if latent_w else 0

    vae_meta: dict[str, object] = {}
    try:
        vae = getattr(getattr(processing.sd_model, "codex_objects", None), "vae", None)
        fs = getattr(vae, "first_stage_model", None)
        for key in ("scaling_factor", "shift_factor", "latents_mean", "latents_std"):
            if hasattr(fs, key):
                vae_meta[key] = getattr(fs, key)
    except Exception:
        vae_meta = {}

    return PreviewFactorsFit(
        model_name=type(processing.sd_model).__name__,
        channels=channels,
        step=int(step),
        total=int(total),
        scale_h=scale_h,
        scale_w=scale_w,
        sample_count=int(sample_n),
        mse=float(mse),
        factors=factors_rows,
        bias=bias_row,
        vae_meta=vae_meta or {},
    )


def maybe_log_preview_factors(processing: Any, denoised_latent: Any, *, step: int, total: int) -> None:
    global _DEBUG_PREVIEW_FACTORS_LAST_JOB_TS

    if not debug_preview_factors_enabled():
        return

    job_ts = str(getattr(backend_state, "job_timestamp", "") or "")
    if job_ts and _DEBUG_PREVIEW_FACTORS_LAST_JOB_TS == job_ts:
        return

    fit = fit_preview_factors(
        processing,
        denoised_latent,
        step=int(step),
        total=int(total),
        sample_limit=debug_preview_factors_sample_limit(),
    )
    if fit is None:
        return

    logger.info(
        "[preview-factors] model=%s channels=%d step=%d/%d scale=%dx%d samples=%d mse=%.6g vae=%s",
        fit.model_name,
        fit.channels,
        fit.step,
        fit.total,
        fit.scale_h,
        fit.scale_w,
        fit.sample_count,
        fit.mse,
        fit.vae_meta,
    )
    logger.info("[preview-factors] factors = %s", tuple(tuple(round(v, 6) for v in row) for row in fit.factors))
    logger.info("[preview-factors] bias = %s", tuple(round(v, 6) for v in fit.bias))

    _DEBUG_PREVIEW_FACTORS_LAST_JOB_TS = job_ts or "(unknown)"


__all__ = [
    "LivePreviewMethod",
    "PreviewFactorsFit",
    "debug_preview_factors_enabled",
    "debug_preview_factors_sample_limit",
    "decode_preview_image",
    "fit_preview_factors",
    "live_preview_method",
    "live_preview_method_from_env",
    "live_preview_method_to_env",
    "maybe_log_preview_factors",
    "preview_interval_steps",
    "preview_runtime_overrides",
]
