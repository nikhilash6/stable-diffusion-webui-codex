"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF VAE IO helpers (I2V condition encode + decode to frames).
Loads WAN VAE weights via explicit native lanes (`2d_native` or `3d_native`), applies latent normalization, and converts between latents and RGB frames for the WAN22 GGUF runtime.
I2V init-image preprocessing is deterministic and no-stretch (`scale + crop + resize`) across tensor/PIL/ndarray paths before VAE encode,
with optional image scale (`img2vid_image_scale > 0` when provided; omitted = auto-fit minimum) and normalized crop offsets (`x/y` in `[0,1]`)
aligned to the frontend guide projection contract.
Includes strict finite checks and explicit dtype/device retry logic (no silent fallbacks). Model keyspace ownership and lane detection are delegated to `runtime/state_dict/keymap_wan22_vae.py`.

Symbols (top-level; keep in sync; no ghosts):
- `WAN22VAEContractError` (exception): Deterministic WAN VAE path/config contract failure (non-retryable by dtype fallback loops).
- `WanVAEDecodeSession` (dataclass): Reusable loaded WAN VAE decode session (device/dtype/lane) for multi-chunk decode passes.
- `_is_cuda_device_name` (function): Canonical CUDA device-type check that accepts indexed forms (`cuda:0`, etc.) for retry/cleanup paths.
- `_format_exception_message` (function): Stable exception-text formatter used by retry/fallback logs and terminal fail-loud error messages.
- `load_vae` (function): Loads the WAN VAE component (from directory bundles or single-file weights with sibling/override config dir).
- `_cuda_bf16_supported` (function): Best-effort BF16 support probe for CUDA (used for dtype fallbacks).
- `_vae_dtype_candidates` (function): Ordered dtype candidates for VAE encode/decode attempts (requested dtype first).
- `open_vae_decode_session` (function): Loads WAN VAE once with fallback dtype candidates for repeated decode calls.
- `close_vae_decode_session` (function): Offloads and tears down a decode session loaded by `open_vae_decode_session`.
- `vae_encode_video_condition` (function): Encodes the Diffusers-style I2V conditioning video into latents (deterministic mode, no-stretch init-image preprocessing).
- `vae_decode_video` (function): Decodes video latents to frames; can validate the expected output frame count.
- `decode_latents_to_frames` (function): Validates strict WAN latent-channel decode input (no implicit slicing) and returns frames (optional frame-count validation).
"""

from __future__ import annotations

import os
import math
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Optional

import torch

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.checkpoint.io import load_torch_file
from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime.common.vae_ldm import AutoencoderKL_LDM, sanitize_ldm_vae_config
from apps.backend.runtime.common.vae_codex3d import (
    AutoencoderCodex3D,
    sanitize_codex3d_vae_config,
)
from apps.backend.runtime.state_dict.keymap_wan22_vae import resolve_wan22_vae_keyspace

from .config import RunConfig, as_torch_dtype, resolve_device_name
from .diagnostics import cuda_empty_cache, get_logger, log_numerics_enabled, summarize_numerics, warn_fallback
from .wan_latent_norms import resolve_norm

_SUPPORTED_WAN_VAE_LATENT_CHANNELS = (16, 48)
_SUPPORTED_WAN_VAE_LANES = ("2d_native", "3d_native")


class WAN22VAEContractError(RuntimeError):
    """Deterministic WAN22 VAE path/config contract failure."""


@dataclass
class WanVAEDecodeSession:
    vae: Any
    lane: str
    decode_device: str
    decode_dtype: torch.dtype


def _require_offload_device(*, context: str) -> torch.device:
    manager = getattr(memory_management, "manager", None)
    if manager is None or not hasattr(manager, "offload_device"):
        raise WAN22VAEContractError(
            f"WAN22 GGUF: {context} requires an active memory manager with offload_device()."
        )
    offload_device = manager.offload_device()
    if not isinstance(offload_device, torch.device):
        raise WAN22VAEContractError(
            f"WAN22 GGUF: {context} expected offload_device() -> torch.device, "
            f"got {type(offload_device).__name__}."
        )
    return offload_device


def _is_cuda_device_name(device_name: str) -> bool:
    try:
        return torch.device(str(device_name)).type == "cuda"
    except Exception:
        return str(device_name).strip().lower().startswith("cuda")


def _format_exception_message(exc: BaseException | None) -> str:
    if exc is None:
        return "<none>"
    text = " ".join(str(exc).split()).strip()
    if text:
        return text
    parts = [" ".join(str(item).split()).strip() for item in getattr(exc, "args", ()) if str(item).strip()]
    if parts:
        return "; ".join(parts)
    return repr(exc)


def _infer_latent_channels_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    candidates: tuple[tuple[str, int], ...] = (
        ("post_quant_conv.weight", 0),
        ("quant_conv.weight", 0),
        ("encoder.conv_out.weight", 0),
        ("decoder.conv_in.weight", 1),
        ("conv2.weight", 0),
        ("conv1.weight", 0),
        ("encoder.head.2.weight", 0),
        ("decoder.conv1.weight", 1),
    )
    for key, axis in candidates:
        tensor = state_dict.get(key)
        if not torch.is_tensor(tensor):
            continue
        shape = tuple(int(dim) for dim in tensor.shape)
        if len(shape) < 2 or axis >= len(shape):
            continue
        value = int(shape[axis])
        if key in {"quant_conv.weight", "encoder.conv_out.weight", "conv1.weight", "encoder.head.2.weight"} and value % 2 == 0:
            value = value // 2
        if value > 0:
            return value
    return None


def load_vae(
    vae_path: Optional[str],
    *,
    torch_dtype: torch.dtype,
    enable_tiling: bool = False,
    config_dir_override: Optional[str] = None,
) -> Any:
    if not vae_path:
        raise WAN22VAEContractError(
            "WAN22 GGUF: wan_vae_path is required when running the GGUF runtime "
            "(VAE bundle directory path missing)."
        )

    path = os.path.expanduser(str(vae_path))

    def _instantiate_with_state_dict(state_dict_path: str, config_dir: str) -> Any:
        try:
            raw_state_dict = load_torch_file(state_dict_path, device="cpu")
            if not isinstance(raw_state_dict, MutableMapping):
                raise WAN22VAEContractError(
                    "WAN22 GGUF: VAE checkpoint loader returned non-mutable-mapping state_dict "
                    f"(type={type(raw_state_dict).__name__})."
                )
            resolved_vae = resolve_wan22_vae_keyspace(raw_state_dict)
            lane = str(resolved_vae.metadata.get("lane", "")).strip()
            if lane not in _SUPPORTED_WAN_VAE_LANES:
                raise WAN22VAEContractError(
                    "WAN22 GGUF: unsupported VAE lane "
                    f"{lane!r} (supported={list(_SUPPORTED_WAN_VAE_LANES)})."
                )
            resolved_style_obj = resolved_vae.style
            resolved_style = (
                resolved_style_obj.value if hasattr(resolved_style_obj, "value") else str(resolved_style_obj)
            )
            resolved_state_dict_view = resolved_vae.view

            if lane == "2d_native":
                config = AutoencoderKL_LDM.load_config(config_dir)
                native_config = sanitize_ldm_vae_config(config)
                inferred_latent_channels = _infer_latent_channels_from_state_dict(resolved_state_dict_view)
                if inferred_latent_channels is not None:
                    configured_channels = native_config.get("latent_channels")
                    if configured_channels is None:
                        native_config["latent_channels"] = int(inferred_latent_channels)
                    elif int(configured_channels) != int(inferred_latent_channels):
                        raise WAN22VAEContractError(
                            "WAN22 GGUF: VAE config/state_dict latent channel mismatch "
                            f"(lane=2d_native keyspace_style={resolved_style} "
                            f"config latent_channels={int(configured_channels)} "
                            f"inferred={int(inferred_latent_channels)})."
                        )
                vae = AutoencoderKL_LDM.from_config(native_config)
                missing, unexpected = safe_load_state_dict(
                    vae,
                    resolved_state_dict_view,
                    log_name="WAN22 VAE (2d_native)",
                )
                if missing or unexpected:
                    raise WAN22VAEContractError(
                        "WAN22 GGUF: native VAE load failed strict validation "
                        f"(lane=2d_native missing={len(missing)} unexpected={len(unexpected)} "
                        f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]})."
                    )
                setattr(vae, "_codex_vae_lane", "2d_native")
                setattr(vae, "_codex_vae_style", resolved_style)
                return vae

            config = AutoencoderCodex3D.load_config(config_dir)

            native_config = sanitize_codex3d_vae_config(config)
            inferred_latent_channels = _infer_latent_channels_from_state_dict(resolved_state_dict_view)
            if inferred_latent_channels is not None:
                configured_channels = native_config.get("z_dim")
                if configured_channels is None:
                    native_config["z_dim"] = int(inferred_latent_channels)
                elif int(configured_channels) != int(inferred_latent_channels):
                    raise WAN22VAEContractError(
                        "WAN22 GGUF: VAE config/state_dict latent channel mismatch "
                        f"(lane=3d_native keyspace_style={resolved_style} config z_dim={int(configured_channels)} "
                        f"inferred={int(inferred_latent_channels)})."
                    )
            vae = AutoencoderCodex3D.from_config(native_config)
            missing, unexpected = safe_load_state_dict(
                vae,
                resolved_state_dict_view,
                log_name="WAN22 VAE (3d_native)",
            )
            if missing or unexpected:
                raise WAN22VAEContractError(
                    "WAN22 GGUF: native VAE load failed strict validation "
                    f"(lane=3d_native keyspace_style={resolved_style} missing={len(missing)} "
                    f"unexpected={len(unexpected)} missing_sample={missing[:10]} "
                    f"unexpected_sample={unexpected[:10]})."
                )
            setattr(vae, "_codex_vae_lane", "3d_native")
            setattr(vae, "_codex_vae_style", resolved_style)
            return vae
        except WAN22VAEContractError:
            raise
        except Exception as exc:
            raise WAN22VAEContractError(
                "WAN22 GGUF: failed to load native VAE lane from checkpoint "
                f"path={state_dict_path!r} config_dir={config_dir!r}: {exc}"
            ) from exc

    if os.path.isdir(path):
        weights_candidates = (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
            "model.safetensors",
            "model.bin",
            "pytorch_model.bin",
        )
        state_dict_path = None
        for name in weights_candidates:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                state_dict_path = candidate
                break
        if state_dict_path is None:
            raise WAN22VAEContractError(f"WAN22 GGUF: no VAE weights file found under directory: {path}")
        vae = _instantiate_with_state_dict(state_dict_path, path).to(dtype=torch_dtype)
        if enable_tiling and hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
            except Exception as exc:
                raise WAN22VAEContractError(
                    "WAN22 GGUF: failed to enable VAE tiling after directory load "
                    f"(weights={state_dict_path!r} config_dir={path!r})."
                ) from exc
        return vae
    if os.path.isfile(path):
        config_dirs: list[str] = []
        if isinstance(config_dir_override, str) and str(config_dir_override).strip():
            config_dirs.append(os.path.expanduser(str(config_dir_override).strip()))
        config_dirs.append(os.path.dirname(path))
        chosen_config_dir: str | None = None
        for config_dir in config_dirs:
            config_path = os.path.join(config_dir, "config.json")
            if os.path.isfile(config_path):
                chosen_config_dir = config_dir
                break
        if not chosen_config_dir:
            raise WAN22VAEContractError(
                "WAN22 GGUF: single-file VAE load requires config.json at sibling path "
                "or provided metadata config directory. "
                f"VAE file={path} checked_config_dirs={config_dirs}"
            )
        vae = _instantiate_with_state_dict(path, chosen_config_dir).to(dtype=torch_dtype)
        if enable_tiling and hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
            except Exception as exc:
                raise WAN22VAEContractError(
                    "WAN22 GGUF: failed to enable VAE tiling after single-file load "
                    f"(weights={path!r} config_dir={chosen_config_dir!r})."
                ) from exc
        return vae
    raise WAN22VAEContractError(f"WAN22 GGUF: VAE path not found: {path}")


def _retrieve_latents(encoder_output: Any, *, sample_mode: str) -> torch.Tensor:
    dist = getattr(encoder_output, "latent_dist", None)
    if dist is not None:
        mode = str(sample_mode or "").strip().lower()
        if mode in {"mode", "argmax"}:
            return dist.mode()
        if mode in {"sample", ""}:
            return dist.sample()
        raise ValueError(f"Unsupported VAE sample_mode: {sample_mode!r} (expected 'mode' or 'sample')")
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    if torch.is_tensor(encoder_output):
        return encoder_output
    if isinstance(encoder_output, (tuple, list)) and encoder_output and torch.is_tensor(encoder_output[0]):
        return encoder_output[0]
    raise AttributeError(
        "VAE encode output has neither latent_dist nor latents "
        f"(type={type(encoder_output).__name__})."
    )


def _normalize_img2vid_image_scale(raw_value: Any) -> float | None:
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise RuntimeError(
            "WAN22 GGUF: img2vid image scale must be a finite number > 0 when provided "
            f"(got {type(raw_value).__name__})."
        )
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"WAN22 GGUF: img2vid image scale must be finite and > 0 (got {raw_value!r}).")
    return value


def _normalize_img2vid_crop_offset(raw_value: Any, *, field_name: str) -> float:
    if raw_value is None or raw_value == "":
        return 0.5
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise RuntimeError(
            f"WAN22 GGUF: {field_name} must be a finite number in [0,1] "
            f"(got {type(raw_value).__name__})."
        )
    value = float(raw_value)
    if not math.isfinite(value):
        raise RuntimeError(f"WAN22 GGUF: {field_name} must be finite in [0,1] (got {raw_value!r}).")
    if value < 0.0 or value > 1.0:
        raise RuntimeError(f"WAN22 GGUF: {field_name} must be in [0,1] (got {value!r}).")
    return float(value)


def _resolve_no_stretch_crop_window(
    *,
    source_width: int,
    source_height: int,
    frame_width: int,
    frame_height: int,
    image_scale: float | None,
) -> tuple[int, int, float]:
    src_w = int(source_width)
    src_h = int(source_height)
    dst_w = int(frame_width)
    dst_h = int(frame_height)
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        raise RuntimeError(
            "WAN22 GGUF: invalid no-stretch geometry "
            f"(source={src_w}x{src_h} frame={dst_w}x{dst_h})."
        )

    min_scale = max(float(dst_w) / float(src_w), float(dst_h) / float(src_h))
    effective_scale = float(min_scale) if image_scale is None else float(image_scale)
    if float(effective_scale) + 1e-9 < float(min_scale):
        raise RuntimeError(
            "WAN22 GGUF: img2vid image scale is too small for requested frame "
            f"(image_scale={float(effective_scale):.6f} min_required={float(min_scale):.6f})."
        )

    crop_width = max(1, min(src_w, int(float(dst_w) / float(effective_scale) + 0.5)))
    crop_height = max(1, min(src_h, int(float(dst_h) / float(effective_scale) + 0.5)))
    if crop_width <= 0 or crop_height <= 0 or crop_width > src_w or crop_height > src_h:
        raise RuntimeError(
            "WAN22 GGUF: no-stretch crop window exceeds source bounds "
            f"(source={src_w}x{src_h} crop={crop_width}x{crop_height} image_scale={float(effective_scale):.6f})."
        )
    return int(crop_width), int(crop_height), float(effective_scale)


def _crop_resize_hw_no_stretch(
    x: torch.Tensor,
    *,
    height: int,
    width: int,
    image_scale: float | None,
    crop_offset_x: float,
    crop_offset_y: float,
) -> torch.Tensor:
    if x.ndim != 4:
        raise RuntimeError(
            "WAN22 GGUF: no-stretch crop+resize expects 4D tensor [B,C,H,W], "
            f"got shape={tuple(getattr(x, 'shape', ()))!r}."
        )
    target_h = int(height)
    target_w = int(width)
    if target_h <= 0 or target_w <= 0:
        raise RuntimeError(
            "WAN22 GGUF: no-stretch crop+resize target size must be positive "
            f"(height={target_h} width={target_w})."
        )
    normalized_image_scale = _normalize_img2vid_image_scale(image_scale)
    normalized_crop_offset_x = _normalize_img2vid_crop_offset(crop_offset_x, field_name="img2vid_crop_offset_x")
    normalized_crop_offset_y = _normalize_img2vid_crop_offset(crop_offset_y, field_name="img2vid_crop_offset_y")

    _, _, src_h, src_w = x.shape
    src_h = int(src_h)
    src_w = int(src_w)
    if src_h <= 0 or src_w <= 0:
        raise RuntimeError(
            "WAN22 GGUF: no-stretch crop+resize source size must be positive "
            f"(height={src_h} width={src_w})."
        )
    if src_h == target_h and src_w == target_w:
        return x

    crop_w, crop_h, effective_image_scale = _resolve_no_stretch_crop_window(
        source_width=src_w,
        source_height=src_h,
        frame_width=target_w,
        frame_height=target_h,
        image_scale=normalized_image_scale,
    )
    if crop_w > src_w or crop_h > src_h:
        raise RuntimeError(
            "WAN22 GGUF: no-stretch crop window exceeds source bounds "
            f"(source={src_w}x{src_h} crop={crop_w}x{crop_h} image_scale={float(effective_image_scale):.6f})."
        )

    slack_x = max(0, int(src_w) - int(crop_w))
    slack_y = max(0, int(src_h) - int(crop_h))
    left = (
        min(slack_x, max(0, int(slack_x * float(normalized_crop_offset_x) + 0.5)))
        if slack_x > 0
        else 0
    )
    top = (
        min(slack_y, max(0, int(slack_y * float(normalized_crop_offset_y) + 0.5)))
        if slack_y > 0
        else 0
    )
    bottom = top + crop_h
    right = left + crop_w
    if bottom > src_h or right > src_w:
        raise RuntimeError(
            "WAN22 GGUF: no-stretch crop window exceeds source bounds "
            f"(source={src_h}x{src_w} crop=(top={top}, left={left}, h={crop_h}, w={crop_w}) "
            f"image_scale={float(effective_image_scale):.6f})."
        )

    cropped = x[:, :, top:bottom, left:right]
    got_h = int(cropped.shape[-2])
    got_w = int(cropped.shape[-1])
    if got_h != crop_h or got_w != crop_w:
        raise RuntimeError(
            "WAN22 GGUF: no-stretch crop produced unexpected shape "
            f"(expected={crop_h}x{crop_w} got={got_h}x{got_w})."
        )
    if crop_h == target_h and crop_w == target_w:
        return cropped

    import torch.nn.functional as F

    return F.interpolate(cropped, size=(target_h, target_w), mode="bilinear", align_corners=False)


def _prepare_init_image_tensor(
    init_image: Any,
    *,
    device: str,
    torch_dtype: torch.dtype,
    height: int,
    width: int,
    image_scale: float,
    crop_offset_x: float,
    crop_offset_y: float,
) -> torch.Tensor:
    dev_name = resolve_device_name(device)
    target_device = torch.device(dev_name)
    target_h = int(height)
    target_w = int(width)
    if target_h <= 0 or target_w <= 0:
        raise RuntimeError(
            "WAN22 GGUF: init_image target dimensions must be positive "
            f"(height={target_h} width={target_w})."
        )

    if hasattr(init_image, "to"):
        t = init_image
        if hasattr(t, "ndim") and int(t.ndim) == 5:
            t = t[:, :, 0, ...]
        if hasattr(t, "ndim") and int(t.ndim) == 3:
            t = t.unsqueeze(0)
        if not hasattr(t, "ndim") or int(getattr(t, "ndim", 0)) != 4:
            raise RuntimeError(
                "WAN22 GGUF: init_image tensor must be 4D [B,C,H,W] (or 5D [B,C,T,H,W]); "
                f"got {getattr(t, 'shape', None)}"
            )
        t = t.to(device=target_device, dtype=torch_dtype)
        t = _crop_resize_hw_no_stretch(
            t,
            height=target_h,
            width=target_w,
            image_scale=image_scale,
            crop_offset_x=crop_offset_x,
            crop_offset_y=crop_offset_y,
        )
        return t

    from PIL import Image
    import numpy as np

    if isinstance(init_image, Image.Image):
        img = init_image.convert("RGB")
        arr = np.array(img).astype("float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = t.to(device=target_device, dtype=torch_dtype)
        t = _crop_resize_hw_no_stretch(
            t,
            height=target_h,
            width=target_w,
            image_scale=image_scale,
            crop_offset_x=crop_offset_x,
            crop_offset_y=crop_offset_y,
        )
        return t * 2.0 - 1.0

    arr = np.asarray(init_image).astype("float32")
    if arr.ndim == 3 and arr.shape[2] in (1, 3):
        arr = arr / 255.0 if arr.max() > 1.0 else arr
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        t = torch.from_numpy(arr).unsqueeze(0)
    else:
        raise RuntimeError("WAN22 GGUF: unsupported init_image array shape")

    t = t.to(device=target_device, dtype=torch_dtype)
    t = _crop_resize_hw_no_stretch(
        t,
        height=target_h,
        width=target_w,
        image_scale=image_scale,
        crop_offset_x=crop_offset_x,
        crop_offset_y=crop_offset_y,
    )
    return t * 2.0 - 1.0


def _cuda_bf16_supported() -> bool:
    if not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
        return False
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


def _vae_dtype_candidates(*, device: str, preferred: torch.dtype) -> list[torch.dtype]:
    dev = resolve_device_name(device)
    if dev.startswith("cuda") and torch.cuda.is_available():
        out: list[torch.dtype] = [preferred]
        if preferred != torch.float16:
            out.append(torch.float16)
        if _cuda_bf16_supported():
            bf16 = getattr(torch, "bfloat16", torch.float16)
            if preferred != bf16:
                out.append(bf16)
        if preferred != torch.float32:
            out.append(torch.float32)
        # Deduplicate while preserving order.
        seen: set[torch.dtype] = set()
        uniq: list[torch.dtype] = []
        for dt in out:
            if dt in seen:
                continue
            seen.add(dt)
            uniq.append(dt)
        return uniq

    # CPU: default to float32 for stability (BF16 is optional and hardware-dependent).
    return [torch.float32]


def _assert_supported_wan_vae_latent_channels(channels: int, *, context: str) -> None:
    if int(channels) in _SUPPORTED_WAN_VAE_LATENT_CHANNELS:
        return
    supported = ", ".join(str(value) for value in _SUPPORTED_WAN_VAE_LATENT_CHANNELS)
    raise RuntimeError(
        f"WAN22 GGUF: {context} supports latent channels [{supported}] only (got C={int(channels)}). "
        "If C includes mask/image channels (e.g., I2V model-input state), pass pure VAE latents to decode."
    )


def _resolve_loaded_vae_lane(vae: Any) -> str:
    raw_lane = getattr(vae, "_codex_vae_lane", None)
    if raw_lane is None:
        raise RuntimeError(
            "WAN22 GGUF: loaded VAE is missing required lane marker `_codex_vae_lane`. "
            "Runtime-loaded WAN VAEs must be stamped by `load_vae()` before encode/decode."
        )
    lane = str(raw_lane).strip().lower()
    if lane not in _SUPPORTED_WAN_VAE_LANES:
        raise RuntimeError(
            "WAN22 GGUF: loaded VAE exposes unsupported lane marker "
            f"{lane!r} (supported={list(_SUPPORTED_WAN_VAE_LANES)})."
        )
    return lane


def _to_frame_batch_4d(video_tensor: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Convert `[B,C,T,H,W]` tensor to frame-batched `[B*T,C,H,W]`."""
    if video_tensor.ndim != 5:
        raise RuntimeError(
            f"WAN22 GGUF: expected 5D video tensor [B,C,T,H,W], got shape={tuple(video_tensor.shape)}."
        )
    b, c, t, h, w = video_tensor.shape
    batched = video_tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    return batched, int(b), int(t)


def _from_frame_batch_4d(frame_tensor: torch.Tensor, *, batch: int, frames: int) -> torch.Tensor:
    """Convert frame-batched `[B*T,C,H,W]` tensor back to `[B,C,T,H,W]`."""
    if frame_tensor.ndim != 4:
        raise RuntimeError(
            f"WAN22 GGUF: expected 4D frame-batch tensor [B*T,C,H,W], got shape={tuple(frame_tensor.shape)}."
        )
    bt, c, h, w = frame_tensor.shape
    expected = int(batch) * int(frames)
    if int(bt) != expected:
        raise RuntimeError(
            "WAN22 GGUF: frame-batch reshape mismatch "
            f"(B*T={expected} expected, got {int(bt)})."
        )
    return frame_tensor.view(int(batch), int(frames), int(c), int(h), int(w)).permute(0, 2, 1, 3, 4)


def open_vae_decode_session(
    *,
    device: str,
    dtype: str,
    vae_dir: str | None = None,
    vae_config_dir: str | None = None,
    logger: Any = None,
) -> WanVAEDecodeSession:
    log = get_logger(logger)
    dev_name = resolve_device_name(device)
    target_device = str(torch.device(dev_name))
    preferred = as_torch_dtype(dtype)
    dtypes = _vae_dtype_candidates(device=device, preferred=preferred)
    last_exc: Exception | None = None

    for attempt_index, torch_dtype in enumerate(dtypes):
        if attempt_index > 0:
            warn_fallback(
                logger,
                component="VAE decode session",
                detail=f"retrying session load with dtype={torch_dtype} device={target_device}",
                reason=_format_exception_message(last_exc),
            )
        try:
            vae = load_vae(
                vae_dir,
                torch_dtype=torch_dtype,
                enable_tiling=bool(memory_management.manager.vae_always_tiled),
                config_dir_override=vae_config_dir,
            )
            vae = vae.to(device=target_device, dtype=torch_dtype)
            lane = _resolve_loaded_vae_lane(vae)
            log.info(
                "[wan22.gguf] VAE decode session loaded: device=%s dtype=%s lane=%s",
                target_device,
                str(torch_dtype),
                lane,
            )
            return WanVAEDecodeSession(
                vae=vae,
                lane=lane,
                decode_device=str(target_device),
                decode_dtype=torch_dtype,
            )
        except WAN22VAEContractError:
            raise
        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError("WAN22 GGUF: failed to initialize VAE decode session.") from last_exc


def close_vae_decode_session(session: WanVAEDecodeSession | None, *, logger: Any = None) -> None:
    if session is None:
        return
    try:
        session.vae.to(_require_offload_device(context="VAE decode session close"))
    except Exception as exc:
        raise WAN22VAEContractError("WAN22 GGUF: failed to offload VAE decode session.") from exc
    try:
        del session.vae
    except Exception as exc:
        raise WAN22VAEContractError("WAN22 GGUF: failed to release VAE decode session state.") from exc
    cuda_empty_cache(logger, label="after-vae-decode-session")


def vae_encode_video_condition(
    init_image: Any,
    *,
    num_frames: int,
    height: int,
    width: int,
    device: str,
    dtype: str,
    img2vid_image_scale: float | None = None,
    img2vid_crop_offset_x: float = 0.5,
    img2vid_crop_offset_y: float = 0.5,
    vae_dir: str | None = None,
    vae_config_dir: str | None = None,
    logger: Any = None,
    offload_after: bool = True,
) -> torch.Tensor:
    log = get_logger(logger)

    if int(num_frames) <= 0:
        raise RuntimeError(f"WAN22 GGUF: invalid num_frames={num_frames} for I2V video_condition")

    dev_name = resolve_device_name(device)
    target = str(torch.device(dev_name))
    preferred = as_torch_dtype(dtype)
    dtypes = _vae_dtype_candidates(device=device, preferred=preferred)
    last_exc: Exception | None = None
    last_lane: str | None = None
    last_image_shape: tuple[int, ...] | None = None
    last_encode_input_shape: tuple[int, ...] | None = None

    for attempt_idx, torch_dtype in enumerate(dtypes):
        vae = None
        try:
            if attempt_idx > 0:
                warn_fallback(
                    logger,
                    component="VAE encode",
                    detail=f"retrying with dtype={torch_dtype} device={target}",
                    reason=_format_exception_message(last_exc),
                )
            vae = load_vae(
                vae_dir,
                torch_dtype=torch_dtype,
                enable_tiling=bool(memory_management.manager.vae_always_tiled),
                config_dir_override=vae_config_dir,
            )
            vae = vae.to(device=target, dtype=torch_dtype)
            lane = _resolve_loaded_vae_lane(vae)
            last_lane = lane
            keyspace_style = str(getattr(vae, "_codex_vae_style", "unknown")).strip().lower() or "unknown"
            log.info("[wan22.gguf] VAE lane=%s keyspace_style=%s", lane, keyspace_style)

            image = _prepare_init_image_tensor(
                init_image,
                device=device,
                torch_dtype=torch_dtype,
                height=height,
                width=width,
                image_scale=img2vid_image_scale,
                crop_offset_x=img2vid_crop_offset_x,
                crop_offset_y=img2vid_crop_offset_y,
            )
            if image.ndim != 4:
                raise RuntimeError(
                    "WAN22 GGUF: expected preprocessed init image to be 4D [B,C,H,W], "
                    f"got {tuple(image.shape)}"
                )
            last_image_shape = tuple(int(dim) for dim in image.shape)

            # Diffusers-style I2V conditioning video: first frame is the init image; remaining frames are 0 (i.e., 0.5 gray).
            image = image.unsqueeze(2)  # [B,C,1,H,W]
            regulation = lambda posterior: posterior.mode()
            with torch.no_grad():
                if lane == "2d_native":
                    batch_size = int(image.shape[0])
                    first_frame_batch = image[:, :, 0, :, :]
                    if int(num_frames) > 1:
                        zero_frame_batch = first_frame_batch.new_zeros(first_frame_batch.shape)
                        encode_input = torch.cat((first_frame_batch, zero_frame_batch), dim=0)
                    else:
                        encode_input = first_frame_batch
                    last_encode_input_shape = tuple(int(dim) for dim in encode_input.shape)
                    try:
                        encoded_out = vae.encode(encode_input, regulation=regulation)
                    except TypeError:
                        encoded_out = vae.encode(encode_input)
                    encoded_raw = _retrieve_latents(encoded_out, sample_mode="mode")
                    if encoded_raw.ndim == 4:
                        first_latents = encoded_raw[:batch_size]
                        if int(num_frames) > 1:
                            expected = int(batch_size) * 2
                            if int(encoded_raw.shape[0]) != expected:
                                raise RuntimeError(
                                    "WAN22 GGUF: 2d_native encode expected 2*B outputs for first+zero frames "
                                    f"(expected={expected} got={int(encoded_raw.shape[0])})."
                                )
                            zero_latents = encoded_raw[batch_size:]
                            encoded = first_latents.new_empty(
                                (
                                    int(batch_size),
                                    int(first_latents.shape[1]),
                                    int(num_frames),
                                    int(first_latents.shape[2]),
                                    int(first_latents.shape[3]),
                                )
                            )
                            encoded[:, :, :1, :, :] = first_latents.unsqueeze(2)
                            encoded[:, :, 1:, :, :] = zero_latents.unsqueeze(2).expand(
                                -1,
                                -1,
                                int(num_frames) - 1,
                                -1,
                                -1,
                            )
                        else:
                            if int(encoded_raw.shape[0]) != int(batch_size):
                                raise RuntimeError(
                                    "WAN22 GGUF: 2d_native encode expected B outputs for single-frame conditioning "
                                    f"(expected={int(batch_size)} got={int(encoded_raw.shape[0])})."
                                )
                            encoded = first_latents.unsqueeze(2)
                    else:
                        raise RuntimeError(
                            "WAN22 GGUF: VAE encode produced unsupported tensor rank "
                            f"(lane=2d_native shape={tuple(encoded_raw.shape)})."
                        )
                else:
                    video_condition = image.new_zeros(
                        (image.shape[0], image.shape[1], int(num_frames), int(height), int(width))
                    )
                    video_condition[:, :, :1, :, :] = image
                    last_encode_input_shape = tuple(int(dim) for dim in video_condition.shape)
                    try:
                        encoded_out = vae.encode(video_condition, regulation=regulation)
                    except TypeError:
                        encoded_out = vae.encode(video_condition)
                    encoded_raw = _retrieve_latents(encoded_out, sample_mode="mode")
                    if encoded_raw.ndim == 5:
                        encoded = encoded_raw
                    elif encoded_raw.ndim == 4:
                        encoded = encoded_raw.unsqueeze(2)
                    else:
                        raise RuntimeError(
                            "WAN22 GGUF: VAE encode produced unsupported tensor rank "
                            f"(lane=3d_native shape={tuple(encoded_raw.shape)})."
                        )
            latent_channels = int(encoded.shape[1])
            _assert_supported_wan_vae_latent_channels(latent_channels, context="VAE encode")
            norm = resolve_norm(None, channels=latent_channels)
            log.info("[wan22.gguf] VAE latent norm=%s channels=%d", norm.name, norm.channels)
            encoded = norm.process_in_(encoded)

            if not torch.isfinite(encoded).all():
                n_bad = int((~torch.isfinite(encoded)).sum().item())
                raise RuntimeError(
                    "WAN22 GGUF: VAE encode produced non-finite latents "
                    f"(bad={n_bad} dtype={torch_dtype} device={target}; {summarize_numerics(encoded, name='encoded')})."
                )

            if log_numerics_enabled():
                log.info(
                    "[wan22.gguf] VAE encode ok: device=%s dtype=%s %s",
                    target,
                    str(torch_dtype),
                    summarize_numerics(encoded, name="encoded"),
                )

            return encoded
        except torch.OutOfMemoryError as exc:
            last_exc = exc
            if _is_cuda_device_name(target):
                warn_fallback(
                    logger,
                    component="VAE encode",
                    detail=f"OOM on CUDA at dtype={torch_dtype}; retrying remaining CUDA dtypes",
                    reason="cuda_oom",
                )
                cuda_empty_cache(logger, label="vae-encode-oom")
                continue
            raise
        except WAN22VAEContractError:
            raise
        except Exception as exc:
            last_exc = exc
        finally:
            if offload_after and vae is not None:
                try:
                    vae.to(_require_offload_device(context="VAE encode offload"))
                except Exception as exc:
                    raise WAN22VAEContractError(
                        "WAN22 GGUF: failed to offload VAE after encode stage."
                    ) from exc
                del vae
                cuda_empty_cache(logger, label="after-vae-encode")

    attempted_dtypes = ", ".join(str(dt) for dt in dtypes)
    raise RuntimeError(
        "WAN22 GGUF: VAE encode failed for all dtype fallbacks "
        f"(device={target} attempted_dtypes=[{attempted_dtypes}] "
        f"lane={last_lane or 'unknown'} init_image_shape={last_image_shape} "
        f"encode_input_shape={last_encode_input_shape} "
        f"num_frames={int(num_frames)} size={int(height)}x{int(width)} "
        f"last_error={type(last_exc).__name__ if last_exc is not None else '<none>'}: "
        f"{_format_exception_message(last_exc)})."
    ) from last_exc


def vae_decode_video(
    video_latents: Any,
    *,
    model_dir: str,
    device: str,
    dtype: str,
    vae_dir: str | None = None,
    vae_config_dir: str | None = None,
    logger: Any = None,
    offload_after: bool = True,
    expected_frames: int | None = None,
    decode_session: WanVAEDecodeSession | None = None,
) -> list[object]:
    _ = model_dir  # kept for signature symmetry (callers pass stage dir; current VAE loads from explicit path)
    log = get_logger(logger)

    if hasattr(video_latents, "ndim"):
        if video_latents.ndim == 4:
            video_latents = video_latents.unsqueeze(2)
        elif video_latents.ndim != 5:
            raise RuntimeError(
                f"WAN22 VAE decode expects 4D or 5D latents; got shape={tuple(getattr(video_latents,'shape',()))}"
            )

    b, c, t_lat, h, w = video_latents.shape

    _assert_supported_wan_vae_latent_channels(int(c), context="VAE decode")
    norm = resolve_norm(None, channels=int(c))

    from PIL import Image

    dev_name = resolve_device_name(device)
    target = str(torch.device(dev_name))
    preferred = as_torch_dtype(dtype)
    dtypes = _vae_dtype_candidates(device=device, preferred=preferred)
    last_exc: Exception | None = None
    decode_meta: dict[str, Any] | None = None

    class _DecodeNonFiniteError(RuntimeError):
        """Internal decode sentinel used to preserve dtype retry flow."""

    def _decode_attempt(
        *,
        attempt_device: str,
        torch_dtype: torch.dtype,
        session: WanVAEDecodeSession | None = None,
    ) -> tuple[list[Image.Image], dict[str, Any]]:
        vae: Any
        lane: str
        local_vae_loaded = False
        attempt_frames: list[Image.Image] = []
        chunk_count = 0
        first_chunk_shape: tuple[int, ...] | None = None
        last_chunk_shape: tuple[int, ...] | None = None
        expected_chunk_layout: tuple[int, int, int, int] | None = None

        def _extract_decode_tensor(decoded: Any) -> torch.Tensor:
            sample = getattr(decoded, "sample", None)
            if sample is not None:
                return sample
            if torch.is_tensor(decoded):
                return decoded
            if isinstance(decoded, (tuple, list)) and decoded and torch.is_tensor(decoded[0]):
                return decoded[0]
            raise RuntimeError(
                "WAN22 GGUF: VAE decode output has no tensor sample "
                f"(type={type(decoded).__name__})."
            )

        def _append_decoded_chunk(
            img_chunk: torch.Tensor,
            *,
            lane_name: str,
            chunk_label: str,
        ) -> None:
            nonlocal chunk_count, first_chunk_shape, last_chunk_shape, expected_chunk_layout
            if not torch.is_tensor(img_chunk):
                raise RuntimeError(
                    "WAN22 GGUF: VAE decode chunk has non-tensor output "
                    f"(lane={lane_name} chunk={chunk_label} type={type(img_chunk).__name__})."
                )
            if img_chunk.ndim == 4:
                if lane_name == "3d_native":
                    img_chunk = img_chunk.unsqueeze(2)
                else:
                    raise RuntimeError(
                        "WAN22 GGUF: VAE decode produced unsupported tensor rank "
                        f"(lane={lane_name} chunk={chunk_label} shape={tuple(img_chunk.shape)})."
                    )
            if img_chunk.ndim != 5:
                raise RuntimeError(
                    "WAN22 GGUF: VAE decode produced unsupported tensor rank "
                    f"(lane={lane_name} chunk={chunk_label} shape={tuple(img_chunk.shape)})."
                )
            out_batch = int(img_chunk.shape[0])
            out_channels = int(img_chunk.shape[1])
            out_time = int(img_chunk.shape[2])
            out_h = int(img_chunk.shape[3])
            out_w = int(img_chunk.shape[4])
            if out_batch < 1 or out_channels != 3:
                raise RuntimeError(
                    "WAN22 GGUF: VAE decode produced unexpected chunk shape "
                    f"(lane={lane_name} chunk={chunk_label} shape={tuple(img_chunk.shape)}; expected B>=1 and C=3)."
                )
            chunk_layout = (out_batch, out_channels, out_h, out_w)
            if expected_chunk_layout is None:
                expected_chunk_layout = chunk_layout
            elif chunk_layout != expected_chunk_layout:
                raise RuntimeError(
                    "WAN22 GGUF: VAE decode produced inconsistent chunk layout "
                    f"(lane={lane_name} chunk={chunk_label} layout={chunk_layout} expected={expected_chunk_layout})."
                )
            if not torch.isfinite(img_chunk).all():
                n_bad = int((~torch.isfinite(img_chunk)).sum().item())
                raise _DecodeNonFiniteError(
                    "WAN22 GGUF: VAE decode produced non-finite outputs "
                    f"(lane={lane_name} chunk={chunk_label} bad={n_bad}; {summarize_numerics(img_chunk, name='vae_out_chunk')})."
                )
            chunk_shape = tuple(int(v) for v in img_chunk.shape)
            if first_chunk_shape is None:
                first_chunk_shape = chunk_shape
            last_chunk_shape = chunk_shape
            chunk_count += 1
            if log_numerics_enabled():
                log.info(
                    "[wan22.gguf] VAE decode chunk ok lane=%s chunk=%s: %s",
                    lane_name,
                    chunk_label,
                    summarize_numerics(img_chunk, name="vae_out_chunk"),
                )
            for ti in range(out_time):
                x = img_chunk[0, :, ti, :, :].detach()
                if x.ndim != 3:
                    raise RuntimeError(
                        f"WAN22 GGUF: VAE decode produced unexpected frame tensor rank: shape={tuple(x.shape)}; expected [C,H,W]"
                    )
                x = (x + 1.0) * 0.5
                x = x.clamp(0, 1)
                frame_hwc = x.permute(1, 2, 0)
                frame_uint8 = frame_hwc.mul(255).to(dtype=torch.uint8)
                arr = frame_uint8.cpu().numpy()
                attempt_frames.append(Image.fromarray(arr))

        if session is not None:
            vae = session.vae
            lane = str(session.lane)
            attempt_device = str(session.decode_device)
            torch_dtype = session.decode_dtype
        else:
            vae = load_vae(
                vae_dir,
                torch_dtype=torch_dtype,
                enable_tiling=bool(memory_management.manager.vae_always_tiled),
                config_dir_override=vae_config_dir,
            )
            vae = vae.to(device=attempt_device, dtype=torch_dtype)
            lane = _resolve_loaded_vae_lane(vae)
            local_vae_loaded = True
        lat = video_latents.to(device=attempt_device, dtype=torch_dtype)
        lat = norm.process_out_(lat)
        if not torch.isfinite(lat).all():
            n_bad = int((~torch.isfinite(lat)).sum().item())
            raise RuntimeError(
                "WAN22 GGUF: non-finite latents after unnormalize; refusing to decode "
                f"(bad={n_bad} dtype={torch_dtype} device={attempt_device}; {summarize_numerics(lat, name='lat_unnorm')})."
            )
        with torch.no_grad():
            if lane == "2d_native":
                lat_batched, batch_size, frame_count = _to_frame_batch_4d(lat)
                if int(batch_size) < 1:
                    raise RuntimeError(
                        "WAN22 GGUF: VAE decode produced invalid batch size for 2D lane "
                        f"(batch_size={batch_size} frame_count={frame_count})."
                    )
                if int(batch_size) == 1 and int(frame_count) > 1:
                    decode_chunk_frames = min(int(frame_count), 8)
                    for frame_start in range(0, int(frame_count), int(decode_chunk_frames)):
                        frame_stop = min(int(frame_count), int(frame_start + decode_chunk_frames))
                        frame_slice = frame_stop - frame_start
                        decoded = vae.decode(lat_batched[frame_start:frame_stop, :, :, :])
                        img_chunk = _extract_decode_tensor(decoded)
                        if img_chunk.ndim == 4:
                            img_chunk = _from_frame_batch_4d(img_chunk, batch=1, frames=frame_slice)
                        elif img_chunk.ndim != 5 or int(img_chunk.shape[2]) != int(frame_slice):
                            raise RuntimeError(
                                "WAN22 GGUF: 2D VAE decode produced unexpected chunk shape "
                                f"(chunk={frame_start}:{frame_stop} shape={tuple(img_chunk.shape)} expected_t={frame_slice})."
                            )
                        _append_decoded_chunk(
                            img_chunk,
                            lane_name=lane,
                            chunk_label=f"{frame_start}:{frame_stop}",
                        )
                else:
                    decoded = vae.decode(lat_batched)
                    img_chunk = _extract_decode_tensor(decoded)
                    if img_chunk.ndim == 4:
                        img_chunk = _from_frame_batch_4d(img_chunk, batch=batch_size, frames=frame_count)
                    elif img_chunk.ndim != 5 or int(img_chunk.shape[2]) != int(frame_count):
                        raise RuntimeError(
                            "WAN22 GGUF: 2D VAE decode produced unexpected tensor shape "
                            f"(shape={tuple(img_chunk.shape)} expected_t={frame_count})."
                        )
                    _append_decoded_chunk(img_chunk, lane_name=lane, chunk_label="full")
            else:
                stream_ok = False
                try:
                    decoded_stream = vae.decode(
                        lat,
                        chunk_callback=lambda chunk, idx: _append_decoded_chunk(
                            chunk,
                            lane_name=lane,
                            chunk_label=str(int(idx)),
                        ),
                    )
                    stream_ok = chunk_count > 0
                    if not stream_ok:
                        img_chunk = _extract_decode_tensor(decoded_stream)
                        if img_chunk.ndim == 4:
                            img_chunk = img_chunk.unsqueeze(2)
                        elif img_chunk.ndim != 5:
                            raise RuntimeError(
                                "WAN22 GGUF: VAE decode produced unsupported tensor rank "
                                f"(lane={lane} shape={tuple(img_chunk.shape)})."
                            )
                        _append_decoded_chunk(img_chunk, lane_name=lane, chunk_label="stream-fallback")
                        stream_ok = chunk_count > 0
                except TypeError as exc:
                    if "chunk_callback" not in str(exc):
                        raise
                if not stream_ok:
                    decoded = vae.decode(lat)
                    img_chunk = _extract_decode_tensor(decoded)
                    if img_chunk.ndim == 4:
                        img_chunk = img_chunk.unsqueeze(2)
                    elif img_chunk.ndim != 5:
                        raise RuntimeError(
                            "WAN22 GGUF: VAE decode produced unsupported tensor rank "
                            f"(lane={lane} shape={tuple(img_chunk.shape)})."
                        )
                    _append_decoded_chunk(img_chunk, lane_name=lane, chunk_label="full")
        if local_vae_loaded and offload_after:
            try:
                vae.to(_require_offload_device(context="VAE decode offload"))
            except Exception as exc:
                raise WAN22VAEContractError(
                    "WAN22 GGUF: failed to offload VAE after decode stage."
                ) from exc
            del vae
            cuda_empty_cache(logger, label="after-vae-decode")
        if not attempt_frames:
            raise RuntimeError(
                "WAN22 GGUF: VAE decode produced zero output frames "
                f"(lane={lane} latent_T={int(t_lat)} device={attempt_device} dtype={torch_dtype})."
            )
        return attempt_frames, {
            "lane": lane,
            "chunk_count": int(chunk_count),
            "first_chunk_shape": first_chunk_shape,
            "last_chunk_shape": last_chunk_shape,
        }

    decoded_frames: list[Image.Image] | None = None
    if decode_session is not None:
        try:
            decoded_frames, decode_meta = _decode_attempt(
                attempt_device=decode_session.decode_device,
                torch_dtype=decode_session.decode_dtype,
                session=decode_session,
            )
        except WAN22VAEContractError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "WAN22 GGUF: VAE decode failed with an active decode session "
                f"(device={decode_session.decode_device} dtype={decode_session.decode_dtype} lane={decode_session.lane}; "
                f"cause={type(exc).__name__}: {_format_exception_message(exc)})."
            ) from exc
    else:
        for attempt_idx, torch_dtype in enumerate(dtypes):
            if attempt_idx > 0:
                warn_fallback(
                    logger,
                    component="VAE decode",
                    detail=f"retrying with dtype={torch_dtype} device={target}",
                    reason=_format_exception_message(last_exc),
                )
            try:
                decoded_frames, decode_meta = _decode_attempt(attempt_device=target, torch_dtype=torch_dtype)
                break
            except torch.OutOfMemoryError as exc:
                last_exc = exc
                if _is_cuda_device_name(target):
                    warn_fallback(
                        logger,
                        component="VAE decode",
                        detail=f"OOM on CUDA at dtype={torch_dtype}; retrying remaining CUDA dtypes",
                        reason="cuda_oom",
                    )
                    cuda_empty_cache(logger, label="vae-decode-oom")
                    continue
                raise
            except _DecodeNonFiniteError as exc:
                last_exc = exc
                continue
            except WAN22VAEContractError:
                raise
            except Exception as exc:
                last_exc = exc
                # Continue to next dtype.
                continue

    if not decoded_frames:
        attempted_dtypes = ", ".join(str(dt) for dt in dtypes)
        raise RuntimeError(
            "WAN22 GGUF: VAE decode failed for all dtype fallbacks "
            f"(device={target} attempted_dtypes=[{attempted_dtypes}] "
            f"latent_shape={tuple(video_latents.shape)} "
            f"last_error={type(last_exc).__name__ if last_exc is not None else '<none>'}: "
            f"{_format_exception_message(last_exc)})."
        ) from last_exc
    t_out = int(len(decoded_frames))
    if expected_frames is not None and int(expected_frames) != t_out:
        raise RuntimeError(
            "WAN22 GGUF: VAE decode time dimension mismatch: "
            f"expected T={int(expected_frames)} got T={t_out} (latent_T={int(t_lat)})."
        )
    log.info(
        "[wan22.gguf] VAE decode output frames=%d lane=%s chunks=%s first_chunk_shape=%s last_chunk_shape=%s",
        t_out,
        (decode_meta or {}).get("lane"),
        (decode_meta or {}).get("chunk_count"),
        (decode_meta or {}).get("first_chunk_shape"),
        (decode_meta or {}).get("last_chunk_shape"),
    )
    return decoded_frames


def decode_latents_to_frames(
    *,
    latents: torch.Tensor,
    model_dir: str,
    cfg: RunConfig,
    logger: Any = None,
    debug_preview: bool = False,
    expected_frames: int | None = None,
    decode_session: WanVAEDecodeSession | None = None,
) -> list[object]:
    log = get_logger(logger)
    x = latents
    log.info("[wan22.gguf] decode latents: shape=%s", tuple(x.shape))
    _ = debug_preview  # debug-only clamp removed (no env-driven behavior)

    c = int(x.shape[1])
    _assert_supported_wan_vae_latent_channels(c, context="decode")

    if not torch.isfinite(x).all():
        n_bad = int((~torch.isfinite(x)).sum().item())
        raise RuntimeError(
            "WAN22 GGUF: decode input latents are non-finite; aborting before VAE decode "
            f"(bad={n_bad}; {summarize_numerics(x, name='latents_in')})."
        )

    if log_numerics_enabled():
        log.info("[wan22.gguf] decode latents (pre-VAE): %s", summarize_numerics(x, name="latents_in"))

    return vae_decode_video(
        x,
        model_dir=model_dir,
        device=resolve_device_name(cfg.device),
        dtype=cfg.dtype,
        vae_dir=cfg.vae_dir,
        vae_config_dir=cfg.vae_config_dir,
        logger=logger,
        expected_frames=expected_frames,
        decode_session=decode_session,
    )
