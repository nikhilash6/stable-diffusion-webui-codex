"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native LTX2 txt2vid/img2vid execution helpers.
Owns direct native execution against loaded LTX2 components (text encoder, connectors, transformer, VAEs, vocoder,
and native FlowMatch-Euler scheduler), including deterministic generation-boundary cleanup for streamed transformers,
truthful request-owned seed/guidance handling, explicit latent-stage sampling/decode primitives for the `two_stage`
runtime path, keeps public stage sampling pinned to the locked `native.transformer` owner, returns raw `(video, audio)`
tuples that runtime.py can normalize into the family-local result contract,
and threads the explicit zero-timestep decode input required by timestep-conditioned LTX video VAEs.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2NativeLatentStageResult` (dataclass): Native latent-stage bridge contract for two-stage orchestration.
- `_resolve_guidance_scale` (function): Preserve explicit `cfg_scale=0` while defaulting only missing guidance values.
- `sample_ltx2_txt2vid_native` (function): Execute the native txt2vid sampler and return the latent-stage result.
- `sample_ltx2_img2vid_native` (function): Execute the native img2vid sampler and return the latent-stage result.
- `decode_ltx2_native_stage_result` (function): Decode a native latent-stage result into raw `(video, audio)` outputs.
- `run_ltx2_txt2vid_native` (function): Execute the native LTX2 txt2vid path and return raw `(video, audio)`.
- `run_ltx2_img2vid_native` (function): Execute the native LTX2 img2vid path and return raw `(video, audio)`.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from apps.backend.runtime.model_registry.ltx2_execution import LTX2_PROFILE_DISTILLED
from .scheduler import Ltx2FlowMatchEulerScheduler
from .text import encode_ltx2_prompt_pair


@dataclass(frozen=True, slots=True)
class Ltx2NativeLatentStageResult:
    video_latents_unpacked_unnormalized: torch.Tensor
    audio_latents_packed_normalized: torch.Tensor
    latent_num_frames: int
    latent_height: int
    latent_width: int
    audio_num_frames: int
    latent_mel_bins: int
    batch_size: int


def _resolve_device(native: Any) -> torch.device:
    return torch.device(str(getattr(native, "device_label", "cpu") or "cpu"))

def _resolve_execution_profile(request: Any) -> str | None:
    extras = getattr(request, "extras", None)
    if not isinstance(extras, Mapping):
        return None
    normalized = str(extras.get("ltx_execution_profile") or "").strip()
    return normalized or None


def _resolve_guidance_scale(request: Any) -> float:
    guidance_scale = getattr(request, "guidance_scale", None)
    if guidance_scale is None:
        if _resolve_execution_profile(request) == LTX2_PROFILE_DISTILLED:
            return 1.0
        return 4.0
    return float(guidance_scale)


def _build_scheduler_config(
    native: Any,
    *,
    preserve_custom_sigmas: bool,
) -> dict[str, Any]:
    scheduler_config = dict(getattr(native, "scheduler_config", {}) or {})
    if preserve_custom_sigmas:
        scheduler_config["use_dynamic_shifting"] = False
        scheduler_config["shift_terminal"] = None
    return scheduler_config

def _resolve_dtype(native: Any) -> torch.dtype:
    native_dtype = getattr(native, "torch_dtype", None)
    if isinstance(native_dtype, torch.dtype):
        return native_dtype
    return torch.float32


def _module_dtype(module: Any, fallback: torch.dtype) -> torch.dtype:
    module_dtype = getattr(module, "dtype", None)
    if isinstance(module_dtype, torch.dtype):
        return module_dtype
    return fallback


def _randn_tensor(
    shape: Sequence[int],
    *,
    generator: torch.Generator | Sequence[torch.Generator] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(generator, Sequence) and not isinstance(generator, (bytes, str)):
        if len(generator) != int(shape[0]):
            raise RuntimeError(
                "LTX2 expected one generator per batch item when generator is a sequence; "
                f"got batch={int(shape[0])} generators={len(generator)}."
            )
        samples = [
            torch.randn(tuple(int(dim) for dim in shape[1:]), generator=item, device=device, dtype=dtype)
            for item in generator
        ]
        return torch.stack(samples, dim=0)
    return torch.randn(tuple(int(dim) for dim in shape), generator=generator, device=device, dtype=dtype)


def _rescale_noise_cfg(
    noise_cfg: torch.Tensor,
    noise_pred_text: torch.Tensor,
    *,
    guidance_rescale: float,
) -> torch.Tensor:
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1.0 - guidance_rescale) * noise_cfg


def _transformer_cache_context(transformer: Any):
    cache_context = getattr(transformer, "cache_context", None)
    if callable(cache_context):
        return cache_context("cond_uncond")
    return nullcontext()


@contextmanager
def _transformer_streaming_lifecycle(transformer: Any):
    reset_controller = getattr(transformer, "reset_controller", None)
    move_all_to_storage = getattr(transformer, "move_all_to_storage", None)
    if callable(reset_controller):
        reset_controller()
    try:
        yield
    finally:
        if callable(move_all_to_storage):
            move_all_to_storage()


def _retrieve_latents(
    encoder_output: Any,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise RuntimeError("LTX2 VAE encode output does not expose latents or latent_dist.")


def _pack_video_latents(latents: torch.Tensor, *, patch_size: int, patch_size_t: int) -> torch.Tensor:
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // int(patch_size_t)
    post_patch_height = height // int(patch_size)
    post_patch_width = width // int(patch_size)
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        int(patch_size_t),
        post_patch_height,
        int(patch_size),
        post_patch_width,
        int(patch_size),
    )
    return latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)


def _unpack_video_latents(
    latents: torch.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int,
    patch_size_t: int,
) -> torch.Tensor:
    batch_size = int(latents.size(0))
    latents = latents.reshape(batch_size, num_frames, height, width, -1, int(patch_size_t), int(patch_size), int(patch_size))
    return latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)


def _normalize_video_latents(
    latents: torch.Tensor,
    *,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - latents_mean) * float(scaling_factor) / latents_std


def _denormalize_video_latents(
    latents: torch.Tensor,
    *,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return latents * latents_std / float(scaling_factor) + latents_mean


def _pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    return latents.transpose(1, 2).flatten(2, 3)


def _unpack_audio_latents(
    latents: torch.Tensor,
    *,
    latent_length: int,
    num_mel_bins: int,
) -> torch.Tensor:
    batch_size = int(latents.size(0))
    latents = latents.reshape(batch_size, latent_length, -1, num_mel_bins)
    return latents.permute(0, 2, 1, 3).contiguous()


def _normalize_audio_latents(
    latents: torch.Tensor,
    *,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
) -> torch.Tensor:
    return (latents - latents_mean.to(latents.device, latents.dtype)) / latents_std.to(latents.device, latents.dtype)


def _denormalize_audio_latents(
    latents: torch.Tensor,
    *,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
) -> torch.Tensor:
    return (latents * latents_std.to(latents.device, latents.dtype)) + latents_mean.to(latents.device, latents.dtype)


def _create_noised_state(
    latents: torch.Tensor,
    *,
    noise_scale: float | torch.Tensor,
    generator: torch.Generator | Sequence[torch.Generator] | None,
) -> torch.Tensor:
    noise = _randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    return noise_scale * noise + (1.0 - noise_scale) * latents


def _packed_conditioning_mask(conditioning_mask: torch.Tensor, *, patch_size: int, patch_size_t: int) -> torch.Tensor:
    packed = _pack_video_latents(conditioning_mask, patch_size=patch_size, patch_size_t=patch_size_t)
    if int(packed.shape[-1]) != 1:
        raise RuntimeError(
            "LTX2 conditioning-mask packing expected a singleton feature dimension; "
            f"got shape={tuple(int(dim) for dim in packed.shape)!r}."
        )
    return packed.squeeze(-1)


def _prepare_txt2vid_video_latents(
    *,
    native: Any,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
    num_frames: int,
    noise_scale: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | Sequence[torch.Generator] | None,
    latents: torch.Tensor | None = None,
) -> torch.Tensor:
    spatial_ratio = int(getattr(native.vae, "spatial_compression_ratio", 32) or 32)
    temporal_ratio = int(getattr(native.vae, "temporal_compression_ratio", 8) or 8)
    patch_size = int(getattr(native.transformer.config, "patch_size", 1) or 1)
    patch_size_t = int(getattr(native.transformer.config, "patch_size_t", 1) or 1)

    if latents is not None:
        if latents.ndim == 5:
            latents = _normalize_video_latents(
                latents,
                latents_mean=native.vae.latents_mean,
                latents_std=native.vae.latents_std,
                scaling_factor=float(getattr(native.vae.config, "scaling_factor", 1.0) or 1.0),
            )
            latents = _pack_video_latents(latents, patch_size=patch_size, patch_size_t=patch_size_t)
        if latents.ndim != 3:
            raise RuntimeError(
                "LTX2 txt2vid latents must be packed [batch, seq, features] or unpacked [batch, channels, frames, height, width]; "
                f"got shape={tuple(int(dim) for dim in latents.shape)!r}."
            )
        return _create_noised_state(latents, noise_scale=float(noise_scale), generator=generator).to(device=device, dtype=dtype)

    latent_height = int(height) // spatial_ratio
    latent_width = int(width) // spatial_ratio
    latent_num_frames = (int(num_frames) - 1) // temporal_ratio + 1
    shape = (int(batch_size), int(num_channels_latents), latent_num_frames, latent_height, latent_width)
    latents = _randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    return _pack_video_latents(latents, patch_size=patch_size, patch_size_t=patch_size_t)


def _prepare_audio_latents(
    *,
    native: Any,
    batch_size: int,
    num_channels_latents: int,
    audio_latent_length: int,
    num_mel_bins: int,
    noise_scale: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | Sequence[torch.Generator] | None,
    latents: torch.Tensor | None = None,
) -> torch.Tensor:
    mel_ratio = int(getattr(native.audio_vae, "mel_compression_ratio", 4) or 4)
    latent_mel_bins = int(num_mel_bins) // mel_ratio

    if latents is not None:
        if latents.ndim == 4:
            latents = _pack_audio_latents(latents)
        if latents.ndim != 3:
            raise RuntimeError(
                "LTX2 audio latents must be packed [batch, seq, features] or unpacked [batch, channels, length, mel_bins]; "
                f"got shape={tuple(int(dim) for dim in latents.shape)!r}."
            )
        latents = _normalize_audio_latents(
            latents,
            latents_mean=native.audio_vae.latents_mean,
            latents_std=native.audio_vae.latents_std,
        )
        return _create_noised_state(latents, noise_scale=float(noise_scale), generator=generator).to(device=device, dtype=dtype)

    shape = (int(batch_size), int(num_channels_latents), int(audio_latent_length), int(latent_mel_bins))
    latents = _randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    return _pack_audio_latents(latents)


def _coerce_image_batch(
    init_image: Any,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    height: int,
    width: int,
) -> torch.Tensor:
    def _normalize_tensor_image(tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = tensor
        if image_tensor.ndim == 5:
            image_tensor = image_tensor[:, :, 0, ...]
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        elif image_tensor.ndim == 3:
            if int(image_tensor.shape[0]) in {1, 3, 4}:
                image_tensor = image_tensor.unsqueeze(0)
            elif int(image_tensor.shape[-1]) in {1, 3, 4}:
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            else:
                raise RuntimeError(
                    "LTX2 init_image tensor must be CHW or HWC when 3D; "
                    f"got shape={tuple(int(dim) for dim in image_tensor.shape)!r}."
                )
        elif image_tensor.ndim != 4:
            raise RuntimeError(
                "LTX2 init_image tensor must be 4D [batch, channels, height, width] or convertible from 2D/3D/5D; "
                f"got shape={tuple(int(dim) for dim in image_tensor.shape)!r}."
            )
        if int(image_tensor.shape[1]) == 4:
            image_tensor = image_tensor[:, :3, ...]
        if int(image_tensor.shape[1]) == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)
        if int(image_tensor.shape[1]) != 3:
            raise RuntimeError(
                "LTX2 init_image tensor must have 1, 3, or 4 channels after normalization; "
                f"got channels={int(image_tensor.shape[1])}."
            )
        image_tensor = image_tensor.to(device=device, dtype=torch.float32)
        if torch.is_floating_point(image_tensor):
            if float(image_tensor.min().item()) < 0.0:
                image_tensor = image_tensor.clamp(-1.0, 1.0)
            else:
                image_tensor = image_tensor / 255.0 if float(image_tensor.max().item()) > 1.0 else image_tensor
                image_tensor = image_tensor.clamp(0.0, 1.0) * 2.0 - 1.0
        else:
            image_tensor = image_tensor / 255.0
            image_tensor = image_tensor.clamp(0.0, 1.0) * 2.0 - 1.0
        image_tensor = F.interpolate(image_tensor, size=(int(height), int(width)), mode="bilinear", align_corners=False)
        return image_tensor.to(dtype=dtype)

    if isinstance(init_image, (list, tuple)):
        if not init_image:
            raise RuntimeError("LTX2 img2vid init_image sequence must not be empty.")
        images = [_coerce_image_batch(item, batch_size=1, device=device, dtype=dtype, height=height, width=width) for item in init_image]
        image_tensor = torch.cat(images, dim=0)
    elif isinstance(init_image, torch.Tensor):
        image_tensor = _normalize_tensor_image(init_image)
    elif isinstance(init_image, Image.Image):
        image = init_image.convert("RGB").resize((int(width), int(height)), Image.BILINEAR)
        array = np.asarray(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        image_tensor = image_tensor * 2.0 - 1.0
        image_tensor = image_tensor.to(dtype=dtype)
    else:
        array = np.asarray(init_image)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3:
            raise RuntimeError(
                "LTX2 init_image array must be 2D grayscale or 3D HWC/CHW; "
                f"got shape={tuple(int(dim) for dim in array.shape)!r}."
            )
        if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
            tensor = torch.from_numpy(array).unsqueeze(0)
        else:
            if array.shape[-1] == 4:
                array = array[..., :3]
            elif array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=2)
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        image_tensor = _normalize_tensor_image(tensor)

    if int(image_tensor.shape[0]) == 1 and int(batch_size) > 1:
        image_tensor = image_tensor.repeat(int(batch_size), 1, 1, 1)
    if int(image_tensor.shape[0]) != int(batch_size):
        raise RuntimeError(
            "LTX2 init_image batch does not match prompt batch; "
            f"got image_batch={int(image_tensor.shape[0])} prompt_batch={int(batch_size)}."
        )
    return image_tensor


def _prepare_img2vid_video_latents(
    *,
    native: Any,
    init_image: Any,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
    num_frames: int,
    noise_scale: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | Sequence[torch.Generator] | None,
    latents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_ratio = int(getattr(native.vae, "spatial_compression_ratio", 32) or 32)
    temporal_ratio = int(getattr(native.vae, "temporal_compression_ratio", 8) or 8)
    patch_size = int(getattr(native.transformer.config, "patch_size", 1) or 1)
    patch_size_t = int(getattr(native.transformer.config, "patch_size_t", 1) or 1)

    latent_height = int(height) // spatial_ratio
    latent_width = int(width) // spatial_ratio
    latent_num_frames = (int(num_frames) - 1) // temporal_ratio + 1

    shape = (int(batch_size), int(num_channels_latents), latent_num_frames, latent_height, latent_width)
    mask_shape = (int(batch_size), 1, latent_num_frames, latent_height, latent_width)

    if latents is not None:
        conditioning_mask = latents.new_zeros(mask_shape)
        conditioning_mask[:, :, 0] = 1.0
        if latents.ndim == 5:
            if init_image is not None:
                image_tensor = _coerce_image_batch(
                    init_image,
                    batch_size=int(batch_size),
                    device=device,
                    dtype=dtype,
                    height=int(height),
                    width=int(width),
                )
                if isinstance(generator, Sequence) and not isinstance(generator, (bytes, str)):
                    if len(generator) != int(batch_size):
                        raise RuntimeError(
                            "LTX2 img2vid expected one generator per batch item when generator is a sequence; "
                            f"got batch={int(batch_size)} generators={len(generator)}."
                        )
                    init_latents = [
                        _retrieve_latents(native.vae.encode(image_tensor[index : index + 1].unsqueeze(2)), generator[index], "argmax")
                        for index in range(int(batch_size))
                    ]
                else:
                    init_latents = [
                        _retrieve_latents(native.vae.encode(image_tensor[index : index + 1].unsqueeze(2)), generator, "argmax")
                        for index in range(int(batch_size))
                    ]
                conditioned_init_latents = torch.cat(init_latents, dim=0).to(device=device, dtype=dtype)
                latents = latents.clone()
                latents[:, :, :1] = conditioned_init_latents
            latents = _normalize_video_latents(
                latents,
                latents_mean=native.vae.latents_mean,
                latents_std=native.vae.latents_std,
                scaling_factor=float(getattr(native.vae.config, "scaling_factor", 1.0) or 1.0),
            )
            latents = _create_noised_state(
                latents,
                noise_scale=float(noise_scale) * (1.0 - conditioning_mask),
                generator=generator,
            )
            latents = _pack_video_latents(latents, patch_size=patch_size, patch_size_t=patch_size_t)
        conditioning_mask = _packed_conditioning_mask(conditioning_mask, patch_size=patch_size, patch_size_t=patch_size_t)
        if latents.ndim != 3 or tuple(latents.shape[:2]) != tuple(conditioning_mask.shape):
            raise RuntimeError(
                "LTX2 img2vid packed latents do not align with the conditioning mask; "
                f"got latents={tuple(int(dim) for dim in latents.shape)!r} conditioning_mask={tuple(int(dim) for dim in conditioning_mask.shape)!r}."
            )
        return latents.to(device=device, dtype=dtype), conditioning_mask

    image_tensor = _coerce_image_batch(
        init_image,
        batch_size=int(batch_size),
        device=device,
        dtype=dtype,
        height=int(height),
        width=int(width),
    )

    if isinstance(generator, Sequence) and not isinstance(generator, (bytes, str)):
        if len(generator) != int(batch_size):
            raise RuntimeError(
                "LTX2 img2vid expected one generator per batch item when generator is a sequence; "
                f"got batch={int(batch_size)} generators={len(generator)}."
            )
        init_latents = [
            _retrieve_latents(native.vae.encode(image_tensor[index : index + 1].unsqueeze(2)), generator[index], "argmax")
            for index in range(int(batch_size))
        ]
    else:
        init_latents = [
            _retrieve_latents(native.vae.encode(image_tensor[index : index + 1].unsqueeze(2)), generator, "argmax")
            for index in range(int(batch_size))
        ]

    init_latents = torch.cat(init_latents, dim=0).to(device=device, dtype=dtype)
    init_latents = _normalize_video_latents(
        init_latents,
        latents_mean=native.vae.latents_mean,
        latents_std=native.vae.latents_std,
        scaling_factor=float(getattr(native.vae.config, "scaling_factor", 1.0) or 1.0),
    )
    init_latents = init_latents.repeat(1, 1, latent_num_frames, 1, 1)

    conditioning_mask = torch.zeros(mask_shape, device=device, dtype=dtype)
    conditioning_mask[:, :, 0] = 1.0
    noise = _randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = init_latents * conditioning_mask + noise * (1.0 - conditioning_mask)

    conditioning_mask = _packed_conditioning_mask(conditioning_mask, patch_size=patch_size, patch_size_t=patch_size_t)
    latents = _pack_video_latents(latents, patch_size=patch_size, patch_size_t=patch_size_t)
    return latents, conditioning_mask


def _resolve_batch_size(prompt: str | Sequence[str]) -> int:
    if isinstance(prompt, str):
        return 1
    return len(prompt)


def _validate_run_args(*, width: int, height: int, num_frames: int, num_inference_steps: int, frame_rate: float) -> None:
    if int(width) <= 0 or int(height) <= 0:
        raise RuntimeError(f"LTX2 width/height must be positive; got width={width!r} height={height!r}.")
    if int(width) % 32 != 0 or int(height) % 32 != 0:
        raise RuntimeError(f"LTX2 width/height must be divisible by 32; got width={width!r} height={height!r}.")
    if int(num_frames) <= 0:
        raise RuntimeError(f"LTX2 num_frames must be > 0; got {num_frames!r}.")
    if int(num_inference_steps) <= 0:
        raise RuntimeError(f"LTX2 num_inference_steps must be > 0; got {num_inference_steps!r}.")
    if not np.isfinite(float(frame_rate)) or float(frame_rate) <= 0.0:
        raise RuntimeError(f"LTX2 frame_rate must be finite > 0; got {frame_rate!r}.")


def _resolve_sigma_schedule(
    *,
    num_inference_steps: int,
    sigmas: Sequence[float] | None,
) -> tuple[int, np.ndarray]:
    if sigmas is None:
        steps = int(num_inference_steps)
        sigma_values = np.linspace(1.0, 1.0 / float(steps), steps, dtype=np.float32)
        return steps, sigma_values

    sigma_values = np.asarray(list(sigmas), dtype=np.float32)
    if sigma_values.ndim != 1 or int(sigma_values.size) <= 0:
        raise RuntimeError(
            "LTX2 sigma schedule override must be a non-empty 1D sequence of finite floats; "
            f"got shape={tuple(int(dim) for dim in sigma_values.shape)!r}."
        )
    if not np.isfinite(sigma_values).all():
        raise RuntimeError("LTX2 sigma schedule override must contain only finite floats.")
    return int(sigma_values.size), sigma_values


@torch.no_grad()
def _sample_ltx2_native_latents(
    *,
    native: Any,
    prompt: str | Sequence[str],
    negative_prompt: str | Sequence[str] | None,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    guidance_scale: float,
    init_image: Any | None,
    generator: torch.Generator | Sequence[torch.Generator] | None,
    guidance_rescale: float = 0.0,
    noise_scale: float = 0.0,
    latents: torch.Tensor | None = None,
    audio_latents: torch.Tensor | None = None,
    sigmas: Sequence[float] | None = None,
    attention_kwargs: Mapping[str, Any] | None = None,
    max_sequence_length: int = 1024,
    prompt_scale_factor: int = 8,
) -> Ltx2NativeLatentStageResult:
    effective_transformer = native.transformer
    preserve_custom_sigmas = sigmas is not None
    effective_num_inference_steps, sigma_values = _resolve_sigma_schedule(
        num_inference_steps=int(num_inference_steps),
        sigmas=sigmas,
    )
    _validate_run_args(
        width=int(width),
        height=int(height),
        num_frames=int(num_frames),
        num_inference_steps=int(effective_num_inference_steps),
        frame_rate=float(frame_rate),
    )

    device = _resolve_device(native)
    execution_dtype = _resolve_dtype(native)
    batch_size = _resolve_batch_size(prompt)
    do_classifier_free_guidance = float(guidance_scale) > 1.0

    encoded_prompt = encode_ltx2_prompt_pair(
        native=native,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=float(guidance_scale),
        num_videos_per_prompt=1,
        max_sequence_length=int(max_sequence_length),
        scale_factor=int(prompt_scale_factor),
        device=device,
        dtype=execution_dtype,
    )

    spatial_ratio = int(getattr(native.vae, "spatial_compression_ratio", 32) or 32)
    temporal_ratio = int(getattr(native.vae, "temporal_compression_ratio", 8) or 8)
    mel_ratio = int(getattr(native.audio_vae, "mel_compression_ratio", 4) or 4)
    audio_temporal_ratio = int(getattr(native.audio_vae, "temporal_compression_ratio", 4) or 4)
    patch_size = int(getattr(effective_transformer.config, "patch_size", 1) or 1)
    patch_size_t = int(getattr(effective_transformer.config, "patch_size_t", 1) or 1)
    audio_sampling_rate = int(getattr(getattr(native.audio_vae, "config", None), "sample_rate", 16000) or 16000)
    audio_hop_length = int(getattr(getattr(native.audio_vae, "config", None), "mel_hop_length", 160) or 160)

    latent_num_frames = (int(num_frames) - 1) // temporal_ratio + 1
    latent_height = int(height) // spatial_ratio
    latent_width = int(width) // spatial_ratio
    video_sequence_length = latent_num_frames * latent_height * latent_width

    num_channels_latents = int(getattr(effective_transformer.config, "in_channels", 0) or 0)
    if num_channels_latents <= 0:
        raise RuntimeError("LTX2 transformer config is missing a positive in_channels value.")

    if init_image is None:
        video_latents = _prepare_txt2vid_video_latents(
            native=native,
            batch_size=int(batch_size),
            num_channels_latents=num_channels_latents,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            noise_scale=float(noise_scale),
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        conditioning_mask = None
    else:
        video_latents, conditioning_mask = _prepare_img2vid_video_latents(
            native=native,
            init_image=init_image,
            batch_size=int(batch_size),
            num_channels_latents=num_channels_latents,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            noise_scale=float(noise_scale),
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

    duration_s = float(num_frames) / float(frame_rate)
    audio_latents_per_second = float(audio_sampling_rate) / float(audio_hop_length) / float(audio_temporal_ratio)
    audio_num_frames = max(1, int(round(duration_s * audio_latents_per_second)))
    num_mel_bins = int(getattr(getattr(native.audio_vae, "config", None), "mel_bins", 64) or 64)
    latent_mel_bins = int(num_mel_bins) // mel_ratio
    num_channels_latents_audio = int(getattr(getattr(native.audio_vae, "config", None), "latent_channels", 8) or 8)
    audio_latents_tensor = _prepare_audio_latents(
        native=native,
        batch_size=int(batch_size),
        num_channels_latents=num_channels_latents_audio,
        audio_latent_length=int(audio_num_frames),
        num_mel_bins=int(num_mel_bins),
        noise_scale=float(noise_scale),
        dtype=torch.float32,
        device=device,
        generator=generator,
        latents=audio_latents,
    )

    scheduler_config = _build_scheduler_config(native, preserve_custom_sigmas=preserve_custom_sigmas)
    scheduler = Ltx2FlowMatchEulerScheduler.from_config(scheduler_config)
    audio_scheduler = Ltx2FlowMatchEulerScheduler.from_config(scheduler_config)
    scheduler.set_timesteps(
        int(effective_num_inference_steps),
        device=device,
        sigmas=sigma_values,
        sequence_length=int(video_sequence_length),
    )
    audio_scheduler.set_timesteps(
        int(effective_num_inference_steps),
        device=device,
        sigmas=sigma_values,
        sequence_length=int(video_sequence_length),
    )

    video_coords = effective_transformer.rope.prepare_video_coords(
        int(video_latents.shape[0]),
        int(latent_num_frames),
        int(latent_height),
        int(latent_width),
        video_latents.device,
        fps=float(frame_rate),
    )
    audio_coords = effective_transformer.audio_rope.prepare_audio_coords(
        int(audio_latents_tensor.shape[0]),
        int(audio_num_frames),
        audio_latents_tensor.device,
    )

    conditioning_mask_for_model = None
    if conditioning_mask is not None:
        conditioning_mask_for_model = conditioning_mask
        if do_classifier_free_guidance:
            conditioning_mask_for_model = torch.cat([conditioning_mask_for_model, conditioning_mask_for_model], dim=0)

    for timestep in scheduler.timesteps:
        latent_model_input = torch.cat([video_latents, video_latents], dim=0) if do_classifier_free_guidance else video_latents
        latent_model_input = latent_model_input.to(dtype=encoded_prompt.video_prompt_embeds.dtype)
        audio_model_input = (
            torch.cat([audio_latents_tensor, audio_latents_tensor], dim=0)
            if do_classifier_free_guidance
            else audio_latents_tensor
        )
        audio_model_input = audio_model_input.to(dtype=encoded_prompt.audio_prompt_embeds.dtype)

        timestep_batch = timestep.expand(int(latent_model_input.shape[0]))
        transformer_kwargs: dict[str, Any] = {
            "hidden_states": latent_model_input,
            "audio_hidden_states": audio_model_input,
            "encoder_hidden_states": encoded_prompt.video_prompt_embeds,
            "audio_encoder_hidden_states": encoded_prompt.audio_prompt_embeds,
            "encoder_attention_mask": encoded_prompt.attention_mask,
            "audio_encoder_attention_mask": encoded_prompt.attention_mask,
            "num_frames": int(latent_num_frames),
            "height": int(latent_height),
            "width": int(latent_width),
            "fps": float(frame_rate),
            "audio_num_frames": int(audio_num_frames),
            "video_coords": video_coords,
            "audio_coords": audio_coords,
            "attention_kwargs": None if attention_kwargs is None else dict(attention_kwargs),
            "return_dict": False,
        }
        if conditioning_mask_for_model is None:
            transformer_kwargs["timestep"] = timestep_batch
        else:
            transformer_kwargs["timestep"] = timestep_batch.unsqueeze(-1) * (1.0 - conditioning_mask_for_model)
            transformer_kwargs["audio_timestep"] = timestep_batch

        with _transformer_cache_context(effective_transformer):
            noise_pred_video, noise_pred_audio = effective_transformer(**transformer_kwargs)
        noise_pred_video = noise_pred_video.float()
        noise_pred_audio = noise_pred_audio.float()

        if do_classifier_free_guidance:
            noise_pred_video_uncond, noise_pred_video_text = noise_pred_video.chunk(2)
            noise_pred_video = noise_pred_video_uncond + float(guidance_scale) * (
                noise_pred_video_text - noise_pred_video_uncond
            )
            noise_pred_audio_uncond, noise_pred_audio_text = noise_pred_audio.chunk(2)
            noise_pred_audio = noise_pred_audio_uncond + float(guidance_scale) * (
                noise_pred_audio_text - noise_pred_audio_uncond
            )
            if float(guidance_rescale) > 0.0:
                noise_pred_video = _rescale_noise_cfg(
                    noise_pred_video,
                    noise_pred_video_text,
                    guidance_rescale=float(guidance_rescale),
                )
                noise_pred_audio = _rescale_noise_cfg(
                    noise_pred_audio,
                    noise_pred_audio_text,
                    guidance_rescale=float(guidance_rescale),
                )

        if conditioning_mask is None:
            video_latents = scheduler.step(noise_pred_video, timestep, video_latents, return_dict=False)[0]
        else:
            unpacked_noise_pred_video = _unpack_video_latents(
                noise_pred_video,
                num_frames=int(latent_num_frames),
                height=int(latent_height),
                width=int(latent_width),
                patch_size=patch_size,
                patch_size_t=patch_size_t,
            )
            unpacked_video_latents = _unpack_video_latents(
                video_latents,
                num_frames=int(latent_num_frames),
                height=int(latent_height),
                width=int(latent_width),
                patch_size=patch_size,
                patch_size_t=patch_size_t,
            )
            predicted_latents = scheduler.step(
                unpacked_noise_pred_video[:, :, 1:],
                timestep,
                unpacked_video_latents[:, :, 1:],
                return_dict=False,
            )[0]
            unpacked_video_latents = torch.cat([unpacked_video_latents[:, :, :1], predicted_latents], dim=2)
            video_latents = _pack_video_latents(
                unpacked_video_latents,
                patch_size=patch_size,
                patch_size_t=patch_size_t,
            )

        audio_latents_tensor = audio_scheduler.step(
            noise_pred_audio,
            timestep,
            audio_latents_tensor,
            return_dict=False,
        )[0]

    unpacked_video_latents = _unpack_video_latents(
        video_latents,
        num_frames=int(latent_num_frames),
        height=int(latent_height),
        width=int(latent_width),
        patch_size=patch_size,
        patch_size_t=patch_size_t,
    )
    unpacked_video_latents = _denormalize_video_latents(
        unpacked_video_latents,
        latents_mean=native.vae.latents_mean,
        latents_std=native.vae.latents_std,
        scaling_factor=float(getattr(native.vae.config, "scaling_factor", 1.0) or 1.0),
    )

    unpacked_audio_latents = _denormalize_audio_latents(
        audio_latents_tensor,
        latents_mean=native.audio_vae.latents_mean,
        latents_std=native.audio_vae.latents_std,
    )
    unpacked_audio_latents = _unpack_audio_latents(
        unpacked_audio_latents,
        latent_length=int(audio_num_frames),
        num_mel_bins=int(latent_mel_bins),
    )
    return Ltx2NativeLatentStageResult(
        video_latents_unpacked_unnormalized=unpacked_video_latents,
        audio_latents_packed_normalized=audio_latents_tensor,
        latent_num_frames=int(latent_num_frames),
        latent_height=int(latent_height),
        latent_width=int(latent_width),
        audio_num_frames=int(audio_num_frames),
        latent_mel_bins=int(latent_mel_bins),
        batch_size=int(batch_size),
    )


@torch.no_grad()
def decode_ltx2_native_stage_result(
    *,
    native: Any,
    stage_result: Ltx2NativeLatentStageResult,
) -> tuple[Any, Any]:
    device = _resolve_device(native)
    execution_dtype = _resolve_dtype(native)
    vae_dtype = _module_dtype(native.vae, execution_dtype)
    audio_vae_dtype = _module_dtype(native.audio_vae, execution_dtype)
    unpacked_audio_latents = _denormalize_audio_latents(
        stage_result.audio_latents_packed_normalized,
        latents_mean=native.audio_vae.latents_mean,
        latents_std=native.audio_vae.latents_std,
    )
    unpacked_audio_latents = _unpack_audio_latents(
        unpacked_audio_latents,
        latent_length=int(stage_result.audio_num_frames),
        num_mel_bins=int(stage_result.latent_mel_bins),
    )
    decoded_video_latents = stage_result.video_latents_unpacked_unnormalized.to(dtype=execution_dtype)

    decode_timestep = None
    if bool(getattr(getattr(native.vae, "config", None), "timestep_conditioning", False)):
        decode_timestep = torch.zeros(int(stage_result.batch_size), device=device, dtype=decoded_video_latents.dtype)

    video = native.vae.decode(
        decoded_video_latents.to(dtype=vae_dtype),
        timestep=decode_timestep,
        return_dict=False,
    )[0]
    generated_mel_spectrograms = native.audio_vae.decode(
        unpacked_audio_latents.to(dtype=audio_vae_dtype),
        return_dict=False,
    )[0]
    audio = native.vocoder(generated_mel_spectrograms)
    return video, audio


@torch.no_grad()
def _run_ltx2_native(
    *,
    native: Any,
    prompt: str | Sequence[str],
    negative_prompt: str | Sequence[str] | None,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    guidance_scale: float,
    init_image: Any | None,
    generator: torch.Generator | Sequence[torch.Generator] | None,
    guidance_rescale: float = 0.0,
    noise_scale: float = 0.0,
    latents: torch.Tensor | None = None,
    audio_latents: torch.Tensor | None = None,
    sigmas: Sequence[float] | None = None,
    attention_kwargs: Mapping[str, Any] | None = None,
    max_sequence_length: int = 1024,
    prompt_scale_factor: int = 8,
) -> tuple[Any, Any]:
    stage_result = _sample_ltx2_native_latents(
        native=native,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        init_image=init_image,
        generator=generator,
        guidance_rescale=guidance_rescale,
        noise_scale=noise_scale,
        latents=latents,
        audio_latents=audio_latents,
        sigmas=sigmas,
        attention_kwargs=attention_kwargs,
        max_sequence_length=max_sequence_length,
        prompt_scale_factor=prompt_scale_factor,
    )
    return decode_ltx2_native_stage_result(native=native, stage_result=stage_result)


@torch.no_grad()
def sample_ltx2_txt2vid_native(
    *,
    native: Any,
    request: Any,
    plan: Any,
    width: int | None = None,
    height: int | None = None,
    num_frames: int | None = None,
    frame_rate: float | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    noise_scale: float = 0.0,
    latents: torch.Tensor | None = None,
    audio_latents: torch.Tensor | None = None,
    sigmas: Sequence[float] | None = None,
    generator: torch.Generator | Sequence[torch.Generator] | None = None,
) -> Ltx2NativeLatentStageResult:
    with _transformer_streaming_lifecycle(native.transformer):
        return _sample_ltx2_native_latents(
            native=native,
            prompt=("" if getattr(request, "prompt", None) is None else getattr(request, "prompt")),
            negative_prompt=getattr(request, "negative_prompt", None),
            width=int(getattr(plan, "width", 0) if width is None else width),
            height=int(getattr(plan, "height", 0) if height is None else height),
            num_frames=int(getattr(plan, "frames", 0) if num_frames is None else num_frames),
            frame_rate=float(int(getattr(plan, "fps", 0)) if frame_rate is None else frame_rate),
            num_inference_steps=int(getattr(plan, "steps", 0) if num_inference_steps is None else num_inference_steps),
            guidance_scale=float(_resolve_guidance_scale(request) if guidance_scale is None else guidance_scale),
            init_image=None,
            generator=generator,
            noise_scale=float(noise_scale),
            latents=latents,
            audio_latents=audio_latents,
            sigmas=sigmas,
        )


@torch.no_grad()
def run_ltx2_txt2vid_native(
    *,
    native: Any,
    request: Any,
    plan: Any,
    generator: torch.Generator | Sequence[torch.Generator] | None = None,
) -> tuple[Any, Any]:
    with _transformer_streaming_lifecycle(native.transformer):
        return _run_ltx2_native(
            native=native,
            prompt=("" if getattr(request, "prompt", None) is None else getattr(request, "prompt")),
            negative_prompt=getattr(request, "negative_prompt", None),
            width=int(getattr(plan, "width", 0) or 0),
            height=int(getattr(plan, "height", 0) or 0),
            num_frames=int(getattr(plan, "frames", 0) or 0),
            frame_rate=float(int(getattr(plan, "fps", 0) or 0)),
            num_inference_steps=int(getattr(plan, "steps", 0) or 0),
            guidance_scale=_resolve_guidance_scale(request),
            init_image=None,
            generator=generator,
        )


@torch.no_grad()
def sample_ltx2_img2vid_native(
    *,
    native: Any,
    request: Any,
    plan: Any,
    width: int | None = None,
    height: int | None = None,
    num_frames: int | None = None,
    frame_rate: float | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    noise_scale: float = 0.0,
    latents: torch.Tensor | None = None,
    audio_latents: torch.Tensor | None = None,
    sigmas: Sequence[float] | None = None,
    generator: torch.Generator | Sequence[torch.Generator] | None = None,
) -> Ltx2NativeLatentStageResult:
    init_image = getattr(request, "init_image", None)
    if init_image is None:
        raise RuntimeError("LTX2 native img2vid requires `request.init_image`.")
    with _transformer_streaming_lifecycle(native.transformer):
        return _sample_ltx2_native_latents(
            native=native,
            prompt=("" if getattr(request, "prompt", None) is None else getattr(request, "prompt")),
            negative_prompt=getattr(request, "negative_prompt", None),
            width=int(getattr(plan, "width", 0) if width is None else width),
            height=int(getattr(plan, "height", 0) if height is None else height),
            num_frames=int(getattr(plan, "frames", 0) if num_frames is None else num_frames),
            frame_rate=float(int(getattr(plan, "fps", 0)) if frame_rate is None else frame_rate),
            num_inference_steps=int(getattr(plan, "steps", 0) if num_inference_steps is None else num_inference_steps),
            guidance_scale=float(_resolve_guidance_scale(request) if guidance_scale is None else guidance_scale),
            init_image=init_image,
            generator=generator,
            noise_scale=float(noise_scale),
            latents=latents,
            audio_latents=audio_latents,
            sigmas=sigmas,
        )


@torch.no_grad()
def run_ltx2_img2vid_native(
    *,
    native: Any,
    request: Any,
    plan: Any,
    generator: torch.Generator | Sequence[torch.Generator] | None = None,
) -> tuple[Any, Any]:
    init_image = getattr(request, "init_image", None)
    if init_image is None:
        raise RuntimeError("LTX2 native img2vid requires `request.init_image`.")
    with _transformer_streaming_lifecycle(native.transformer):
        return _run_ltx2_native(
            native=native,
            prompt=("" if getattr(request, "prompt", None) is None else getattr(request, "prompt")),
            negative_prompt=getattr(request, "negative_prompt", None),
            width=int(getattr(plan, "width", 0) or 0),
            height=int(getattr(plan, "height", 0) or 0),
            num_frames=int(getattr(plan, "frames", 0) or 0),
            frame_rate=float(int(getattr(plan, "fps", 0) or 0)),
            num_inference_steps=int(getattr(plan, "steps", 0) or 0),
            guidance_scale=_resolve_guidance_scale(request),
            init_image=init_image,
            generator=generator,
        )


__all__ = [
    "Ltx2NativeLatentStageResult",
    "decode_ltx2_native_stage_result",
    "sample_ltx2_img2vid_native",
    "sample_ltx2_txt2vid_native",
    "run_ltx2_img2vid_native",
    "run_ltx2_txt2vid_native",
]
