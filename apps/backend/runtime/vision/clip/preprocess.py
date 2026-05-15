"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Preprocessing helpers for CLIP vision encoders in backend runtime.
Normalizes a batch of images (expected in BHWC) to the encoder input format (BCHW) using resize/crop and mean/std stats.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger for clip vision preprocessing.
- `_build_processor` (function): Caches a CLIPImageProcessor configured for a canonical clip-vision preprocess spec.
- `preprocess_image` (function): Resizes/crops and normalizes an image batch for a given preprocess spec.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from functools import lru_cache

import torch
from transformers import CLIPImageProcessor

from .errors import ClipVisionInputError
from .specs import ClipVisionPreprocessSpec

logger = get_backend_logger("backend.runtime.vision.clip.preprocess")


@lru_cache(maxsize=8)
def _build_processor(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> CLIPImageProcessor:
    return CLIPImageProcessor(
        do_resize=True,
        do_center_crop=True,
        do_rescale=True,
        do_normalize=True,
        do_convert_rgb=False,
        size={"shortest_edge": int(image_size)},
        crop_size={"height": int(image_size), "width": int(image_size)},
        image_mean=list(mean),
        image_std=list(std),
    )


def preprocess_image(
    image: torch.Tensor,
    spec: ClipVisionPreprocessSpec,
    *,
    crop: bool = True,
) -> torch.Tensor:
    """Normalize a BHWC image tensor into canonical CLIP BCHW pixel values."""
    if image.ndim != 4:
        raise ClipVisionInputError(f"Expected image tensor with 4 dims (batch, height, width, channels); got {image.shape}.")
    if image.shape[-1] < 3:
        raise ClipVisionInputError("Clip vision encoder requires RGB channels; received fewer than 3.")

    device = image.device
    dtype = image.dtype
    logger.debug(
        "Preprocessing clip vision batch: shape=%s dtype=%s device=%s crop=%s",
        tuple(image.shape),
        dtype,
        device,
        crop,
    )
    image = image[..., :3]  # enforce RGB
    processor = _build_processor(
        int(spec.image_size),
        tuple(float(v) for v in spec.mean),
        tuple(float(v) for v in spec.std),
    )
    images_np = (
        torch.clamp((image.detach().to(device="cpu", dtype=torch.float32) * 255.0).round(), 0.0, 255.0)
        .to(torch.uint8)
        .numpy()
    )
    processed = processor(
        images=[sample for sample in images_np],
        do_center_crop=bool(crop),
        size={"shortest_edge": int(spec.image_size)} if crop else {"height": int(spec.image_size), "width": int(spec.image_size)},
        crop_size={"height": int(spec.image_size), "width": int(spec.image_size)},
        return_tensors="pt",
        input_data_format="channels_last",
    ).pixel_values
    if processed.ndim != 4 or processed.shape[1] != 3:
        raise ClipVisionInputError(
            "CLIPImageProcessor returned invalid pixel_values shape "
            f"{tuple(processed.shape)} for clip vision preprocessing."
        )
    return processed.to(device=device)
