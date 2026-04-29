"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Request→processing adapters for txt2img/img2img (hires/swap-model/refiner/smart flags).
Builds Codex processing objects from API request DTOs, including Hires, first-pass swap-model, Refiner, IP-Adapter, and SUPIR mode configs (with hires
tile config), per-job smart runtime flags, and strict pass-through overrides like `extras.er_sde` for sampler runtime wiring.
The img2img adapter now transfers the canonical `inpaint_mode` owner into `CodexProcessingImg2Img` without aliasing removed `mask_enforcement` names.

Symbols (top-level; keep in sync; no ghosts):
- `_require_non_negative_int` (function): Validates an explicit integer `>= 0` for fail-loud request→processing transfer seams.
- `_build_swap_model_config` (function): Builds a typed `SwapModelConfig` from request payload data.
- `_build_swap_stage_config` (function): Builds a typed `SwapStageConfig` for the global first-pass `swap_model` stage.
- `_build_hires_config` (function): Builds a `CodexHiresConfig` from request payload data (including hires tile config + nested hires refiner).
- `_build_ip_adapter_config` (function): Builds a typed `IpAdapterConfig` from request payload data.
- `_build_refiner_config` (function): Builds a `RefinerConfig` from request payload data.
- `build_txt2img_processing` (function): Converts a `Txt2ImgRequest` into a fully-populated `CodexProcessingTxt2Img`.
- `build_img2img_processing` (function): Converts an `Img2ImgRequest` into a fully-populated `CodexProcessingImg2Img` (including inpaint mask wiring + typed SUPIR ownership).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import math
from typing import Any, Mapping
import logging

from apps.backend.core.requests import Img2ImgRequest, Txt2ImgRequest
from apps.backend.runtime.adapters.ip_adapter.types import IpAdapterConfig, IpAdapterSourceConfig
from apps.backend.runtime.families.supir.config import parse_supir_mode_config
from apps.backend.runtime.processing.models import (
    CodexHiresConfig,
    CodexProcessingImg2Img,
    CodexProcessingTxt2Img,
    RefinerConfig,
    SwapStageConfig,
    SwapModelConfig,
)
from apps.backend.runtime.vision.upscalers.specs import tile_config_from_payload

_log = get_backend_logger(__name__)


def _parse_batch_count(extras: Mapping[str, Any] | None) -> int:
    if not isinstance(extras, Mapping):
        return 1
    raw = extras.get("batch_count")
    if raw is None:
        return 1
    if isinstance(raw, bool):
        raise ValueError("Invalid 'extras.batch_count': expected integer >= 1, got boolean.")
    if isinstance(raw, int):
        parsed = raw
    elif isinstance(raw, float):
        if not raw.is_integer():
            raise ValueError(f"Invalid 'extras.batch_count': expected integer >= 1, got {raw!r}.")
        parsed = int(raw)
    elif isinstance(raw, str):
        token = raw.strip()
        if not token:
            raise ValueError("Invalid 'extras.batch_count': expected integer >= 1, got empty string.")
        try:
            parsed = int(token, 10)
        except ValueError as exc:
            raise ValueError(f"Invalid 'extras.batch_count': expected integer >= 1, got {raw!r}.") from exc
    else:
        raise ValueError(f"Invalid 'extras.batch_count': expected integer >= 1, got {type(raw).__name__}.")
    if parsed < 1:
        raise ValueError(f"Invalid 'extras.batch_count': expected integer >= 1, got {parsed}.")
    return parsed


def _parse_optional_finite_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid '{field}': expected finite number, got {value!r}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Invalid '{field}': expected finite number, got {value!r}.")
    return parsed


def _require_non_negative_int(value: Any, *, field: str) -> int:
    if value is None:
        raise ValueError(f"Invalid '{field}': expected integer >= 0, got None.")
    if isinstance(value, bool):
        raise ValueError(f"Invalid '{field}': expected integer >= 0, got boolean.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Invalid '{field}': expected integer >= 0, got {value!r}.")
        parsed = int(value)
    else:
        raise ValueError(f"Invalid '{field}': expected integer >= 0, got {type(value).__name__}.")
    if parsed < 0:
        raise ValueError(f"Invalid '{field}': expected integer >= 0, got {parsed}.")
    return parsed


def _build_hires_config(data: Mapping[str, Any] | None, *, default_cfg: float, default_distilled: float) -> CodexHiresConfig:
    payload = data or {}
    enabled = bool(payload.get("enable", False))
    if not enabled:
        return CodexHiresConfig(
            enabled=False,
            scale=1.0,
            denoise=0.0,
            upscaler=None,
            second_pass_steps=0,
            resize_x=0,
            resize_y=0,
            prompt="",
            negative_prompt="",
            cfg=default_cfg,
            distilled_cfg=default_distilled,
            sampler_name=None,
            scheduler=None,
            swap_model=None,
            refiner=None,
        )

    prompt_value = payload.get("prompt", "")
    negative_value = payload.get("negative_prompt", "")
    cfg_value = payload.get("cfg", default_cfg)
    distilled_value = payload.get("distilled_cfg", default_distilled)
    sampler_name = payload.get("sampler_name")
    scheduler = payload.get("scheduler")
    required_active_fields = ("scale", "denoise", "upscaler", "steps", "resize_x", "resize_y")
    for required_field in required_active_fields:
        if required_field not in payload or payload.get(required_field) is None:
            raise ValueError(f"'hires.{required_field}' is required when hires is enabled.")
    upscaler_value = payload["upscaler"]
    if not isinstance(upscaler_value, str) or not upscaler_value.strip():
        raise ValueError("'hires.upscaler' must be a non-empty string when hires is enabled.")

    raw_swap_model = payload.get("swap_model")
    if raw_swap_model is not None and not isinstance(raw_swap_model, Mapping):
        raise ValueError("'hires.swap_model' must be an object when provided.")
    swap_model_cfg = _build_swap_model_config(raw_swap_model)
    if raw_swap_model is not None and swap_model_cfg is None:
        raise ValueError("'hires.swap_model' requires 'model' or 'model_sha' when provided.")
    refiner_cfg = _build_refiner_config(payload.get("refiner"), default_cfg=cfg_value, context="hires.refiner")
    tile_cfg = tile_config_from_payload(payload.get("tile"), context="hires.tile")

    return CodexHiresConfig(
        enabled=enabled,
        scale=float(payload["scale"]),
        denoise=float(payload["denoise"]),
        upscaler=upscaler_value.strip(),
        tile=tile_cfg,
        second_pass_steps=int(payload["steps"]),
        resize_x=int(payload["resize_x"]),
        resize_y=int(payload["resize_y"]),
        prompt=str(prompt_value),
        negative_prompt=str(negative_value),
        cfg=float(cfg_value),
        distilled_cfg=float(distilled_value),
        sampler_name=sampler_name,
        scheduler=scheduler,
        swap_model=swap_model_cfg,
        refiner=refiner_cfg if refiner_cfg.enabled else None,
    )


def _build_swap_model_config(data: Mapping[str, Any] | None) -> SwapModelConfig | None:
    payload = data or {}
    model_raw = payload.get("model")
    model_name = str(model_raw).strip() if model_raw is not None else ""
    model_sha_raw = payload.get("model_sha")
    model_sha = str(model_sha_raw).strip().lower() if model_sha_raw is not None else ""
    if not model_name and not model_sha:
        return None

    raw_tenc_path = payload.get("tenc_path")
    tenc_path: str | tuple[str, ...] | None = None
    if isinstance(raw_tenc_path, str):
        normalized = raw_tenc_path.strip()
        tenc_path = normalized or None
    elif isinstance(raw_tenc_path, (list, tuple)):
        normalized_paths = tuple(
            str(entry).strip()
            for entry in raw_tenc_path
            if isinstance(entry, str) and str(entry).strip()
        )
        tenc_path = normalized_paths or None

    text_encoder_override_raw = payload.get("text_encoder_override")
    text_encoder_override = (
        {str(key): value for key, value in text_encoder_override_raw.items()}
        if isinstance(text_encoder_override_raw, Mapping)
        else None
    )

    model_format_raw = payload.get("model_format")
    model_format = str(model_format_raw).strip().lower() if isinstance(model_format_raw, str) else None
    if model_format not in {None, "checkpoint", "diffusers", "gguf"}:
        raise ValueError(f"Invalid swap-model model_format: {model_format_raw!r}.")

    zimage_variant_raw = payload.get("zimage_variant")
    zimage_variant = str(zimage_variant_raw).strip().lower() if isinstance(zimage_variant_raw, str) else None
    if zimage_variant not in {None, "turbo", "base"}:
        raise ValueError(f"Invalid swap-model zimage_variant: {zimage_variant_raw!r}.")

    vae_source_raw = payload.get("vae_source")
    vae_source = str(vae_source_raw).strip().lower() if isinstance(vae_source_raw, str) else None
    if vae_source not in {None, "built_in", "external"}:
        raise ValueError(f"Invalid swap-model vae_source: {vae_source_raw!r}.")

    checkpoint_core_only = payload.get("checkpoint_core_only")
    if checkpoint_core_only is not None and not isinstance(checkpoint_core_only, bool):
        raise ValueError("swap_model.checkpoint_core_only must be a boolean when provided.")

    vae_path_raw = payload.get("vae_path")
    vae_path = str(vae_path_raw).strip() if isinstance(vae_path_raw, str) else None

    return SwapModelConfig(
        model=model_name or None,
        model_sha=model_sha or None,
        checkpoint_core_only=checkpoint_core_only,
        model_format=model_format,  # type: ignore[arg-type]
        zimage_variant=zimage_variant,  # type: ignore[arg-type]
        vae_source=vae_source,  # type: ignore[arg-type]
        vae_path=vae_path or None,
        tenc_path=tenc_path,
        text_encoder_override=text_encoder_override,
    )


def _build_refiner_config(data: Mapping[str, Any] | None, *, default_cfg: float, context: str) -> RefinerConfig:
    payload = data or {}
    enabled = bool(payload.get("enable", False))
    swap_at_step = int(payload.get("switch_at_step", 0) or 0)
    cfg = float(payload.get("cfg", default_cfg))
    seed = int(payload.get("seed", -1))
    selection = _build_swap_model_config(payload)
    if selection is not None and selection.zimage_variant is not None:
        raise ValueError(f"'{context}.zimage_variant' is unsupported.")
    if enabled and swap_at_step <= 0:
        raise ValueError(f"'{context}.switch_at_step' must be >= 1 when '{context}.enable' is true.")
    if enabled and selection is None:
        raise ValueError(f"'{context}' requires 'model' or 'model_sha' when enabled.")

    return RefinerConfig(
        enabled=enabled,
        swap_at_step=swap_at_step,
        cfg=cfg,
        seed=seed,
        selection=selection or SwapModelConfig(),
    )


def _build_ip_adapter_config(data: Mapping[str, Any] | None, *, context: str) -> IpAdapterConfig | None:
    payload = data or {}
    enabled = bool(payload.get("enabled", False))
    if not enabled:
        return None

    model_raw = payload.get("model")
    if not isinstance(model_raw, str) or not model_raw.strip():
        raise ValueError(f"'{context}.model' is required when '{context}.enabled' is true.")
    image_encoder_raw = payload.get("image_encoder")
    if not isinstance(image_encoder_raw, str) or not image_encoder_raw.strip():
        raise ValueError(f"'{context}.image_encoder' is required when '{context}.enabled' is true.")

    weight = _parse_optional_finite_float(payload.get("weight"), field=f"{context}.weight")
    if weight is None:
        weight = 1.0
    if weight < 0.0:
        raise ValueError(f"'{context}.weight' must be >= 0.0.")

    start_at = _parse_optional_finite_float(payload.get("start_at"), field=f"{context}.start_at")
    if start_at is None:
        start_at = 0.0
    end_at = _parse_optional_finite_float(payload.get("end_at"), field=f"{context}.end_at")
    if end_at is None:
        end_at = 1.0
    if not 0.0 <= start_at <= 1.0:
        raise ValueError(f"'{context}.start_at' must be between 0.0 and 1.0.")
    if not 0.0 <= end_at <= 1.0:
        raise ValueError(f"'{context}.end_at' must be between 0.0 and 1.0.")
    if start_at > end_at:
        raise ValueError(f"'{context}.start_at' must be <= '{context}.end_at'.")

    source_payload = payload.get("source")
    if not isinstance(source_payload, Mapping):
        raise ValueError(f"'{context}.source' must be an object when '{context}.enabled' is true.")
    source_kind_raw = source_payload.get("kind")
    source_kind = str(source_kind_raw or "").strip()
    if source_kind not in {"uploaded", "same_as_init"}:
        raise ValueError(f"Unsupported '{context}.source.kind': {source_kind_raw!r}.")
    reference_image_data = source_payload.get("reference_image_data")
    if source_kind == "uploaded":
        if not isinstance(reference_image_data, str) or not reference_image_data.strip():
            raise ValueError(f"'{context}.source.reference_image_data' is required when kind='uploaded'.")
    elif source_kind == "same_as_init" and reference_image_data is not None:
        raise ValueError(f"'{context}.source.reference_image_data' is invalid when kind='same_as_init'.")

    return IpAdapterConfig(
        enabled=True,
        model=model_raw.strip(),
        image_encoder=image_encoder_raw.strip(),
        weight=float(weight),
        start_at=float(start_at),
        end_at=float(end_at),
        source=IpAdapterSourceConfig(
            kind=source_kind,
            reference_image_data=reference_image_data.strip() if isinstance(reference_image_data, str) else None,
        ),
    )


def _build_swap_stage_config(data: Mapping[str, Any] | None, *, default_cfg: float, context: str) -> SwapStageConfig | None:
    payload = data or {}
    enabled = bool(payload.get("enable", False))
    swap_at_step = int(payload.get("switch_at_step", 0) or 0)
    cfg = float(payload.get("cfg", default_cfg))
    seed = int(payload.get("seed", -1))
    selection = _build_swap_model_config(payload)
    if not enabled:
        return None
    if swap_at_step <= 0:
        raise ValueError(f"'{context}.switch_at_step' must be >= 1 when '{context}.enable' is true.")
    if selection is None:
        raise ValueError(f"'{context}' requires 'model' or 'model_sha' when enabled.")
    return SwapStageConfig(
        enabled=True,
        swap_at_step=swap_at_step,
        cfg=cfg,
        seed=seed,
        selection=selection,
    )


def build_txt2img_processing(req: Txt2ImgRequest) -> CodexProcessingTxt2Img:
    _log.debug(
        "build_txt2img_processing: size=%dx%d steps=%s sampler=%s scheduler=%s cfg=%s seed=%s hr=%s",
        req.width,
        req.height,
        req.steps,
        req.sampler,
        req.scheduler,
        req.guidance_scale,
        req.seed,
        bool(req.hires),
    )
    metadata = dict(req.metadata or {})
    extras = req.extras if isinstance(req.extras, Mapping) else {}
    iterations = _parse_batch_count(extras)
    if getattr(req, "clip_skip", None) is not None:
        metadata["clip_skip"] = int(req.clip_skip)
    smart_offload = bool(getattr(req, "smart_offload", False))
    smart_fallback = bool(getattr(req, "smart_fallback", False))
    smart_cache = bool(getattr(req, "smart_cache", False))
    distilled_cfg = _parse_optional_finite_float(metadata.get("distilled_cfg_scale"), field="metadata.distilled_cfg_scale")
    if distilled_cfg is None:
        distilled_cfg = 3.5

    hires_cfg = _build_hires_config(
        req.hires if isinstance(req.hires, dict) else {},
        default_cfg=req.guidance_scale or 7.0,
        default_distilled=distilled_cfg,
    )
    processing = CodexProcessingTxt2Img(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        batch_size=req.batch_size or 1,
        iterations=iterations,
        guidance_scale=req.guidance_scale or 7.0,
        distilled_guidance_scale=distilled_cfg,
        width=req.width,
        height=req.height,
        steps=req.steps or 20,
        sampler_name=req.sampler,
        scheduler=req.scheduler,
        seed=-1 if req.seed is None else int(req.seed),
        metadata=metadata,
        smart_offload=smart_offload,
        smart_fallback=smart_fallback,
        smart_cache=smart_cache,
    )
    processing.swap_model = _build_swap_stage_config(
        extras.get("swap_model"),
        default_cfg=processing.guidance_scale,
        context="extras.swap_model",
    )
    processing.ip_adapter = _build_ip_adapter_config(
        extras.get("ip_adapter"),
        context="extras.ip_adapter",
    )
    refiner_cfg = _build_refiner_config(
        extras.get("refiner"),
        default_cfg=processing.guidance_scale,
        context="extras.refiner",
    )
    processing.refiner = refiner_cfg if refiner_cfg.enabled else None
    if hires_cfg.enabled:
        processing.enable_hires(cfg=hires_cfg)
    for key, value in extras.items():
        if key == "ip_adapter":
            continue
        if key == "er_sde" and isinstance(value, Mapping):
            processing.update_override(key, dict(value))
        else:
            processing.update_override(key, value)
        if key == "eta_noise_seed_delta":
            try:
                processing.eta_noise_seed_delta = int(value)
            except Exception:
                processing.eta_noise_seed_delta = value
    return processing


def build_img2img_processing(req: Img2ImgRequest) -> CodexProcessingImg2Img:
    _log.debug(
        "build_img2img_processing: size=%sx%s steps=%s sampler=%s scheduler=%s cfg=%s denoise=%s has_init=%s has_mask=%s",
        req.width,
        req.height,
        req.steps,
        req.sampler,
        req.scheduler,
        req.guidance_scale,
        getattr(req, "denoise_strength", None),
        bool(getattr(req, "init_image", None)),
        bool(getattr(req, "mask", None)),
    )
    width = req.width
    height = req.height
    if getattr(req, "init_image", None) is not None:
        try:
            w, h = req.init_image.size  # type: ignore[attr-defined]
            width = width or w
            height = height or h
        except Exception:
            pass

    metadata = dict(req.metadata or {})
    extras = req.extras if isinstance(req.extras, Mapping) else {}
    supir_config = parse_supir_mode_config(extras.get("supir"))
    iterations = _parse_batch_count(extras)
    if getattr(req, "clip_skip", None) is not None:
        metadata["clip_skip"] = int(req.clip_skip)
    smart_offload = bool(getattr(req, "smart_offload", False))
    smart_fallback = bool(getattr(req, "smart_fallback", False))
    smart_cache = bool(getattr(req, "smart_cache", False))

    distilled_cfg = _parse_optional_finite_float(metadata.get("distilled_cfg_scale"), field="metadata.distilled_cfg_scale")
    if distilled_cfg is None:
        distilled_cfg = 3.5

    image_cfg_scale = _parse_optional_finite_float(metadata.get("image_cfg_scale"), field="metadata.image_cfg_scale")
    per_step_blend_strength = _parse_optional_finite_float(
        req.per_step_blend_strength,
        field="Img2ImgRequest.per_step_blend_strength",
    )
    if per_step_blend_strength is None:
        raise ValueError("Img2ImgRequest.per_step_blend_strength must be provided explicitly")
    if per_step_blend_strength < 0.0 or per_step_blend_strength > 1.0:
        raise ValueError("Img2ImgRequest.per_step_blend_strength must be between 0.0 and 1.0")
    per_step_blend_steps = _require_non_negative_int(
        req.per_step_blend_steps,
        field="Img2ImgRequest.per_step_blend_steps",
    )

    mask_round = bool(getattr(req, "mask_round", True))
    processing = CodexProcessingImg2Img(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        batch_size=req.batch_size or 1,
        iterations=iterations,
        guidance_scale=req.guidance_scale or 7.0,
        distilled_guidance_scale=distilled_cfg,
        width=width,
        height=height,
        steps=req.steps or 20,
        sampler_name=req.sampler,
        scheduler=req.scheduler,
        seed=-1 if req.seed is None else int(req.seed),
        init_image=req.init_image,
        mask=req.mask,
        inpaint_mode=getattr(req, "inpaint_mode", None),
        per_step_blend_strength=per_step_blend_strength,
        per_step_blend_steps=per_step_blend_steps,
        mask_region_split=bool(getattr(req, "mask_region_split", False)),
        mask_blur=int(getattr(req, "mask_blur", 4) or 0),
        mask_blur_x=int(getattr(req, "mask_blur_x", getattr(req, "mask_blur", 4)) or 0),
        mask_blur_y=int(getattr(req, "mask_blur_y", getattr(req, "mask_blur", 4)) or 0),
        mask_round=mask_round,
        inpainting_fill=int(getattr(req, "inpainting_fill", 0) or 0),
        inpaint_full_res_padding=int(getattr(req, "inpaint_full_res_padding", 0) or 0),
        inpainting_mask_invert=int(getattr(req, "inpainting_mask_invert", 0) or 0),
        denoising_strength=float(req.denoise_strength),
        image_cfg_scale=image_cfg_scale,
        metadata=metadata,
        smart_offload=smart_offload,
        smart_fallback=smart_fallback,
        smart_cache=smart_cache,
        supir=supir_config,
    )
    processing.ip_adapter = _build_ip_adapter_config(
        extras.get("ip_adapter"),
        context="img2img_extras.ip_adapter",
    )
    if req.hires:
        hires_cfg = _build_hires_config(
            req.hires,
            default_cfg=processing.guidance_scale,
            default_distilled=processing.distilled_guidance_scale,
        )
        if hires_cfg.enabled:
            processing.enable_hires(hires_cfg)
    for key, value in extras.items():
        if key in {"ip_adapter", "supir"}:
            continue
        if key == "er_sde" and isinstance(value, Mapping):
            processing.update_override(key, dict(value))
        else:
            processing.update_override(key, value)
        if key == "eta_noise_seed_delta":
            try:
                processing.eta_noise_seed_delta = int(value)
            except Exception:
                processing.eta_noise_seed_delta = value
    return processing
