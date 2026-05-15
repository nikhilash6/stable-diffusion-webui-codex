"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Netflix VOID runtime assembly scaffold, native text-encoder hydration, request admission guard, and explicit not-yet-implemented execution seam.
Rehydrates the typed Netflix VOID bundle-planning contract into a small runtime container that carries the resolved
base bundle, literal overlay pair, normalized device/dtype options, and the family-owned T5 text-encoder runtime.
Also owns fail-loud request admission checks for the native vid2vid lane before the actual Pass 1 -> warped-noise ->
Pass 2 runtime lands.

Symbols (top-level; keep in sync; no ghosts):
- `NetflixVoidNativeComponents` (dataclass): Loaded runtime scaffold carrying resolved bundle inputs, native text encoder, and normalized options.
- `build_netflix_void_native_components` (function): Build the runtime scaffold from typed bundle inputs and engine options.
- `_validate_netflix_void_vid2vid_request` (function): Reject request fields that do not belong to the native Netflix VOID contract.
- `run_netflix_void_vid2vid` (function): Native Netflix VOID vid2vid execution seam (currently fail-loud).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch

from apps.backend.core.requests import InferenceEvent, Vid2VidRequest
from apps.backend.core.strict_values import parse_bool_value
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole

from .model import NetflixVoidBundleInputs
from .native import NetflixVoidTextEncoderRuntime, load_netflix_void_text_encoder_runtime
from .preprocess import prepare_netflix_void_vid2vid_inputs


@dataclass(frozen=True, slots=True)
class NetflixVoidNativeComponents:
    bundle_inputs: NetflixVoidBundleInputs
    device: str
    dtype: str
    owner_device: torch.device
    owner_dtype: torch.dtype
    text_encoder_runtime: NetflixVoidTextEncoderRuntime
    core_streaming_enabled: bool


def _normalize_device_label(raw_value: object) -> str:
    normalized = str(raw_value or "auto").strip().lower()
    if normalized in {"", "auto", "cpu", "cuda"}:
        return normalized or "auto"
    raise RuntimeError(f"Netflix VOID runtime device must be one of auto|cpu|cuda, got {raw_value!r}.")


def _normalize_dtype_label(raw_value: object) -> str:
    normalized = str(raw_value or "").strip().lower()
    if not normalized:
        return ""
    if normalized in {"fp16", "bf16", "fp32"}:
        return normalized
    raise RuntimeError(f"Netflix VOID runtime dtype must be one of fp16|bf16|fp32 when provided, got {raw_value!r}.")


def _torch_dtype_from_label(label: str) -> torch.dtype:
    normalized = _normalize_dtype_label(label)
    if not normalized:
        raise RuntimeError("Netflix VOID runtime dtype label must not be empty when converting to torch.dtype.")
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[normalized]


def _resolve_owner_device(raw_value: object, *, role: DeviceRole) -> torch.device:
    normalized = _normalize_device_label(raw_value)
    if normalized == "auto":
        return memory_management.manager.get_device(role)
    return torch.device(normalized)


def _resolve_owner_dtype(raw_value: object, *, role: DeviceRole) -> torch.dtype:
    normalized = _normalize_dtype_label(raw_value)
    if normalized:
        return _torch_dtype_from_label(normalized)
    resolved = memory_management.manager.dtype_for_role(role)
    if not isinstance(resolved, torch.dtype):
        raise RuntimeError(
            f"Netflix VOID runtime owner dtype for role {role.value!r} must be torch.dtype, got {type(resolved).__name__}."
        )
    return resolved


def _dtype_label(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }
    if dtype not in mapping:
        raise RuntimeError(f"Netflix VOID runtime does not support owner dtype {dtype!r}.")
    return mapping[dtype]


def build_netflix_void_native_components(
    *,
    bundle_inputs: NetflixVoidBundleInputs,
    engine_options: Mapping[str, Any],
) -> NetflixVoidNativeComponents:
    owner_device = _resolve_owner_device(engine_options.get("device"), role=DeviceRole.CORE)
    owner_dtype = _resolve_owner_dtype(engine_options.get("dtype"), role=DeviceRole.CORE)
    text_encoder_device = _resolve_owner_device(engine_options.get("device"), role=DeviceRole.TEXT_ENCODER)
    text_encoder_dtype = _resolve_owner_dtype(engine_options.get("dtype"), role=DeviceRole.TEXT_ENCODER)
    return NetflixVoidNativeComponents(
        bundle_inputs=bundle_inputs,
        device=str(owner_device),
        dtype=_dtype_label(owner_dtype),
        owner_device=owner_device,
        owner_dtype=owner_dtype,
        text_encoder_runtime=load_netflix_void_text_encoder_runtime(
            base_bundle=bundle_inputs.base_bundle,
            device=text_encoder_device,
            torch_dtype=text_encoder_dtype,
        ),
        core_streaming_enabled=parse_bool_value(
            engine_options.get("core_streaming_enabled"),
            field="engine_options.core_streaming_enabled",
            default=False,
        ),
    )


def _validate_netflix_void_vid2vid_request(request: Vid2VidRequest) -> None:
    video_path = str(getattr(request, "video_path", "") or "").strip()
    if not video_path:
        raise RuntimeError("Netflix VOID vid2vid requires 'video_path'.")

    mask_video_path = str(getattr(request, "mask_video_path", "") or "").strip()
    if not mask_video_path:
        raise RuntimeError("Netflix VOID vid2vid requires 'mask_video_path' (quadmask video).")

    if getattr(request, "reference_image", None) is not None:
        raise RuntimeError("Netflix VOID vid2vid does not support 'reference_image'.")

    for field_name in ("pose_video_path", "face_video_path", "background_video_path"):
        raw_value = str(getattr(request, field_name, "") or "").strip()
        if raw_value:
            raise RuntimeError(f"Netflix VOID vid2vid does not support '{field_name}'.")

    animate_mode = str(getattr(request, "animate_mode", "animate") or "animate").strip().lower()
    if animate_mode not in {"", "animate"}:
        raise RuntimeError("Netflix VOID vid2vid does not support 'animate_mode'.")

    segment_frame_length = int(getattr(request, "segment_frame_length", 77) or 77)
    if segment_frame_length != 77:
        raise RuntimeError("Netflix VOID vid2vid does not support 'segment_frame_length'.")

    previous_conditioning_frames = int(getattr(request, "prev_segment_conditioning_frames", 1) or 1)
    if previous_conditioning_frames != 1:
        raise RuntimeError("Netflix VOID vid2vid does not support 'prev_segment_conditioning_frames'.")

    if getattr(request, "motion_encode_batch_size", None) is not None:
        raise RuntimeError("Netflix VOID vid2vid does not support 'motion_encode_batch_size'.")


def run_netflix_void_vid2vid(*, native: NetflixVoidNativeComponents, request: Vid2VidRequest) -> Iterator[InferenceEvent]:
    del native
    _validate_netflix_void_vid2vid_request(request)
    prepared_inputs = prepare_netflix_void_vid2vid_inputs(request)
    del prepared_inputs
    raise NotImplementedError("netflix_void native Pass1->Pass2 vid2vid execution is not yet implemented")
