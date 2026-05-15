"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Init-image preparation helpers for img2img-style pipelines.
Encodes an init image into tensors/latents and returns a structured `InitImageBundle` for downstream pipelines.
When `processing.width/height` are provided, the init image is resized to target output resolution before VAE encode,
using truthful pixel-space resize semantics from `override_settings.resize_mode`.

Symbols (top-level; keep in sync; no ghosts):
- `_lanczos_resample` (function): Returns the PIL LANCZOS resample constant for the current Pillow version.
- `_normalize_resize_mode` (function): Resolves runtime resize-mode overrides to the supported pixel-space modes.
- `_resize_and_fill` (function): Fits the image inside the target box and fills the margins by stretching edge pixels.
- `_resize_init_image` (function): Applies the requested pixel-space resize strategy before VAE encode.
- `prepare_init_bundle` (function): Converts a processing init image into a tensor/latent bundle (optionally includes a mask).
- `__all__` (constant): Explicit export list for the module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from apps.backend.core import devices
from apps.backend.runtime.processing.conditioners import encode_image_batch, resolve_processing_encode_seed
from apps.backend.runtime.processing.datatypes import InitImageBundle


_PIXEL_RESIZE_MODES = {"just_resize", "crop_and_resize", "resize_and_fill"}


def _lanczos_resample() -> int:
    return Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def _normalize_resize_mode(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "just_resize"
    if raw in _PIXEL_RESIZE_MODES:
        return raw
    raise ValueError(
        "Unknown img2img resize mode override "
        f"{raw!r}; expected one of: {', '.join(sorted(_PIXEL_RESIZE_MODES))}"
    )


def _resize_and_fill(image: Image.Image, width: int, height: int) -> Image.Image:
    ratio = width / height
    src_ratio = image.width / image.height
    resample = _lanczos_resample()

    resized_width = width if ratio < src_ratio else max(1, image.width * height // image.height)
    resized_height = height if ratio >= src_ratio else max(1, image.height * width // image.width)
    resized = image.resize((resized_width, resized_height), resample=resample)

    result = Image.new("RGB", (width, height))
    offset_x = width // 2 - resized_width // 2
    offset_y = height // 2 - resized_height // 2
    right_fill = width - (offset_x + resized_width)
    bottom_fill = height - (offset_y + resized_height)
    result.paste(resized, box=(offset_x, offset_y))

    if ratio < src_ratio:
        if offset_y > 0:
            top = resized.crop((0, 0, resized_width, 1)).resize((resized_width, offset_y), resample=resample)
            result.paste(top, box=(offset_x, 0))
        if bottom_fill > 0:
            bottom = resized.crop((0, resized_height - 1, resized_width, resized_height)).resize(
                (resized_width, bottom_fill),
                resample=resample,
            )
            result.paste(bottom, box=(offset_x, offset_y + resized_height))
    elif ratio > src_ratio:
        if offset_x > 0:
            left = resized.crop((0, 0, 1, resized_height)).resize((offset_x, resized_height), resample=resample)
            result.paste(left, box=(0, offset_y))
        if right_fill > 0:
            right = resized.crop((resized_width - 1, 0, resized_width, resized_height)).resize(
                (right_fill, resized_height),
                resample=resample,
            )
            result.paste(right, box=(offset_x + resized_width, offset_y))
    return result


def _resize_init_image(image: Image.Image, width: int, height: int, resize_mode: str) -> Image.Image:
    resample = _lanczos_resample()
    if resize_mode == "crop_and_resize":
        return ImageOps.fit(image, (width, height), method=resample, centering=(0.5, 0.5))
    if resize_mode == "resize_and_fill":
        return _resize_and_fill(image, width, height)
    return image.resize((width, height), resample=resample)


def prepare_init_bundle(processing: Any) -> InitImageBundle:
    """Encode the init image into tensor/latents for downstream pipelines."""
    image = getattr(processing, "init_image", None)
    if image is None:
        raise ValueError("img2img requires processing.init_image")

    if not isinstance(image, Image.Image):
        raise TypeError(
            "img2img requires processing.init_image to be a PIL.Image.Image; "
            f"got {type(image).__name__}."
        )

    target_width = int(getattr(processing, "width", 0) or 0)
    target_height = int(getattr(processing, "height", 0) or 0)
    overrides = getattr(processing, "override_settings", {}) or {}
    resize_mode = _normalize_resize_mode(overrides.get("resize_mode"))

    prepared_image = image.convert("RGB")
    if target_width > 0 and target_height > 0 and prepared_image.size != (target_width, target_height):
        prepared_image = _resize_init_image(prepared_image, target_width, target_height, resize_mode)
    setattr(processing, "init_image", prepared_image)

    array = np.array(prepared_image).astype(np.float32) / 255.0
    array = array * 2.0 - 1.0
    array = np.moveaxis(array, 2, 0)
    tensor = torch.from_numpy(np.expand_dims(array, axis=0)).to(
        devices.default_device(), dtype=torch.float32
    )
    latents = encode_image_batch(
        processing.sd_model,
        tensor,
        encode_seed=resolve_processing_encode_seed(processing),
        stage="runtime.pipeline_stages.image_init.prepare_init_bundle.encode",
    )
    bundle = InitImageBundle(
        tensor=tensor,
        latents=latents,
        mask=getattr(processing, "mask", None),
        mode="latent",
    )
    return bundle


__all__ = ["prepare_init_bundle"]
