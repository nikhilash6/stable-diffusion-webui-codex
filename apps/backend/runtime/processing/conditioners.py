"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Preprocessing helpers for building conditioning tensors and decoding latents.
Implements request-driven txt2img/img2img conditioning helpers used before sampling, including deterministic img2img
VAE-posterior encode seed plumbing. Plain UNets still use zero img2img conditioning; model-class inpaint UNets build
masked `c_concat` conditioning from the runtime channel contract, while latent-mask enforcement semantics stay owned by
the masked-img2img pipeline stages.

Symbols (top-level; keep in sync; no ghosts):
- `normalize_torch_manual_seed` (function): Validates torch manual-seed bounds and remaps negative seeds to their runtime `uint64` representation.
- `resolve_processing_encode_seed` (function): Resolves the canonical img2img VAE-posterior seed from a processing object.
- `decode_latent_batch` (function): Decode a batch of latents via `sd_model.decode_first_stage`, with optional smart-offload pre-VAE guard.
- `encode_image_batch` (function): Encode a BCHW image tensor via `sd_model.encode_first_stage`, with optional smart-offload pre-VAE guard
  and deterministic posterior seed.
- `txt2img_conditioning` (function): Build the zero image-conditioning tensor used by request-driven txt2img paths.
- `img2img_conditioning` (function): Build runtime-channel-owned img2img conditioning (zero tensor for plain UNets, masked `c_concat` for inpaint-channel UNets).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from apps.backend.runtime.memory.smart_offload_invariants import enforce_smart_offload_pre_vae_residency

_MIN_TORCH_SEED = -(1 << 63)
_MAX_TORCH_SEED = (1 << 64) - 1


def normalize_torch_manual_seed(value: int) -> int:
    resolved = int(value)
    if resolved < _MIN_TORCH_SEED or resolved > _MAX_TORCH_SEED:
        raise ValueError(
            f"Seed {resolved} is outside torch.Generator.manual_seed range "
            f"[{_MIN_TORCH_SEED}, {_MAX_TORCH_SEED}]."
        )
    if resolved < 0:
        return (1 << 64) + resolved
    return resolved


def resolve_processing_encode_seed(processing: Any) -> int | None:
    raw_seeds = list(getattr(processing, "seeds", []) or [])
    candidate = raw_seeds[0] if raw_seeds else getattr(processing, "seed", None)
    if candidate is None:
        return None
    return normalize_torch_manual_seed(int(candidate))


def decode_latent_batch(
    sd_model: Any,
    latents: torch.Tensor,
    *,
    target_device=None,
    stage: str | None = None,
) -> torch.Tensor:
    if isinstance(stage, str) and stage.strip():
        enforce_smart_offload_pre_vae_residency(sd_model, stage=stage.strip())
    decoded = sd_model.decode_first_stage(latents)
    if target_device is not None:
        decoded = decoded.to(target_device)
    return decoded


def encode_image_batch(
    sd_model: Any,
    images: torch.Tensor,
    *,
    encode_seed: int | None = None,
    target_device=None,
    stage: str | None = None,
) -> torch.Tensor:
    if isinstance(stage, str) and stage.strip():
        enforce_smart_offload_pre_vae_residency(sd_model, stage=stage.strip())
    latents = sd_model.encode_first_stage(images, encode_seed=encode_seed)
    if target_device is not None:
        latents = latents.to(target_device)
    return latents


def txt2img_conditioning(sd_model: Any, latents: torch.Tensor, width: int, height: int) -> torch.Tensor:
    del sd_model, width, height
    return latents.new_zeros(latents.shape[0], 5, 1, 1)


def img2img_conditioning(
    sd_model: Any,
    source_image: torch.Tensor,
    latent_image: torch.Tensor,
    *,
    image_mask: Optional[Any] = None,
    round_mask: bool = True,
    precomputed_conditioning_latent: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if latent_image.ndim != 4:
        raise ValueError(f"latent_image must be BCHW; got shape={tuple(latent_image.shape)}")
    if source_image.ndim != 4:
        raise ValueError(f"source_image must be BCHW; got shape={tuple(source_image.shape)}")

    diffusion_model = getattr(
        getattr(getattr(getattr(sd_model, "codex_objects", None), "denoiser", None), "model", None),
        "diffusion_model",
        None,
    )
    codex_config = getattr(diffusion_model, "codex_config", None)
    unet_in_channels = getattr(codex_config, "in_channels", None)
    if not isinstance(unet_in_channels, int) or unet_in_channels <= 0:
        raise ValueError("img2img conditioning requires a positive integer UNet codex_config.in_channels")

    latent_channels = int(latent_image.shape[1])
    if unet_in_channels == latent_channels:
        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)
    if unet_in_channels != latent_channels + 5:
        raise ValueError(
            "Unsupported img2img conditioning channel contract: "
            f"unet_in_channels={unet_in_channels} latent_channels={latent_channels}"
        )

    def _expand_batch(tensor: torch.Tensor, *, batch: int) -> torch.Tensor:
        if int(tensor.shape[0]) == batch:
            return tensor
        if int(tensor.shape[0]) != 1:
            raise ValueError(f"conditioning batch mismatch: got={int(tensor.shape[0])} expected 1 or {batch}")
        return tensor.expand(batch, *tensor.shape[1:])

    source_image = source_image.to(device=latent_image.device, dtype=torch.float32)
    source_image = _expand_batch(source_image, batch=int(latent_image.shape[0]))

    conditioning_mask: torch.Tensor
    if image_mask is None:
        conditioning_mask = source_image.new_ones((1, 1, source_image.shape[2], source_image.shape[3]))
    elif torch.is_tensor(image_mask):
        conditioning_mask = image_mask.detach()
        if conditioning_mask.ndim == 2:
            conditioning_mask = conditioning_mask.unsqueeze(0).unsqueeze(0)
        elif conditioning_mask.ndim == 3:
            conditioning_mask = conditioning_mask.unsqueeze(1)
        elif conditioning_mask.ndim != 4:
            raise ValueError(f"image_mask tensor must be HW/BHW/BCHW; got shape={tuple(conditioning_mask.shape)}")
        if int(conditioning_mask.shape[1]) != 1:
            raise ValueError(
                "image_mask tensor must have a single channel for img2img conditioning; "
                f"got shape={tuple(conditioning_mask.shape)}"
            )
        conditioning_mask = conditioning_mask.to(device=source_image.device, dtype=source_image.dtype)
    else:
        array = torch.from_numpy(np.array(image_mask.convert("L")).astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)
        conditioning_mask = array.to(device=source_image.device, dtype=source_image.dtype)

    conditioning_mask = _expand_batch(conditioning_mask, batch=int(source_image.shape[0]))
    if tuple(conditioning_mask.shape[2:]) != tuple(source_image.shape[2:]):
        raise ValueError(
            "image_mask spatial shape must match source_image for img2img conditioning: "
            f"mask={tuple(conditioning_mask.shape[2:])} source={tuple(source_image.shape[2:])}"
        )
    if round_mask:
        conditioning_mask = torch.round(conditioning_mask)

    conditioning_image = torch.lerp(
        source_image,
        source_image * (1.0 - conditioning_mask),
        1.0,
    )
    if precomputed_conditioning_latent is not None and image_mask is None:
        conditioning_latent = precomputed_conditioning_latent.to(device=latent_image.device, dtype=latent_image.dtype)
    else:
        conditioning_latent = encode_image_batch(
            sd_model,
            conditioning_image,
            target_device=latent_image.device,
            stage="runtime.processing.conditioners.img2img_conditioning.encode",
        ).to(dtype=latent_image.dtype)

    conditioning_mask = torch.nn.functional.interpolate(
        conditioning_mask,
        size=latent_image.shape[-2:],
        mode="nearest",
    )
    conditioning_mask = _expand_batch(conditioning_mask, batch=int(conditioning_latent.shape[0]))
    return torch.cat([conditioning_mask.to(dtype=latent_image.dtype), conditioning_latent], dim=1)
