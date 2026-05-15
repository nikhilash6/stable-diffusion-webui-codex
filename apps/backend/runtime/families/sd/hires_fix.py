"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SD-family hires-fix helpers (second pass init preparation).
Prepares the hi-res pass init latents and request-driven image-conditioning, routing either:
- latent interpolation upscalers (`latent:*`), or
- external SR models via the global upscalers runtime (`spandrel:*`).
When `resize_x/resize_y` change the aspect ratio, the hires-fix stage uses a “fill then crop” plan to avoid stretching.

Symbols (top-level; keep in sync; no ghosts):
- `prepare_hires_latents_and_conditioning` (function): Build hires init latents + image-conditioning for SD/SDXL.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from apps.backend.runtime.pipeline_stages.hires_fix import HiresFillCropPlan
from apps.backend.runtime.processing.conditioners import (
    decode_latent_batch,
    encode_image_batch,
    img2img_conditioning,
    txt2img_conditioning,
)
from apps.backend.runtime.vision.upscalers.registry import upscale_image_tensor, upscale_latent_tensor
from apps.backend.runtime.vision.upscalers.specs import LATENT_UPSCALE_MODES, TileConfig


def prepare_hires_latents_and_conditioning(
    sd_model: Any,
    *,
    base_samples: torch.Tensor,
    base_decoded: torch.Tensor | None,
    target_width: int,
    target_height: int,
    upscaler_id: str,
    tile: TileConfig,
    image_mask: Any | None = None,
    round_mask: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    resize_plan: HiresFillCropPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare hires init latents + image-conditioning for SD-family pipelines."""

    if not isinstance(upscaler_id, str) or not upscaler_id.strip():
        raise ValueError("Missing hires upscaler id")

    uid = upscaler_id.strip()
    tw = int(target_width)
    th = int(target_height)
    if tw <= 0 or th <= 0:
        raise ValueError("target_width/target_height must be positive")

    if resize_plan is None:
        resize_plan = HiresFillCropPlan(
            base_width=int(base_samples.shape[-1]) * 8,
            base_height=int(base_samples.shape[-2]) * 8,
            target_width=tw,
            target_height=th,
            internal_width=tw,
            internal_height=th,
            crop_left=0,
            crop_top=0,
        )
    if resize_plan.target_width != tw or resize_plan.target_height != th:
        raise ValueError("resize_plan target size mismatch")
    internal_w = int(resize_plan.internal_width)
    internal_h = int(resize_plan.internal_height)
    crop_left = int(resize_plan.crop_left)
    crop_top = int(resize_plan.crop_top)

    # Latent upscalers: scale latents directly (no external model).
    if uid in LATENT_UPSCALE_MODES:
        internal_latent_w = max(1, int(internal_w) // 8)
        internal_latent_h = max(1, int(internal_h) // 8)
        upscaled_latents = upscale_latent_tensor(
            base_samples,
            upscaler_id=uid,
            target_width=internal_latent_w,
            target_height=internal_latent_h,
        )

        if internal_w == tw and internal_h == th and crop_left == 0 and crop_top == 0:
            latents = upscaled_latents
            image_conditioning = txt2img_conditioning(sd_model, latents, int(tw), int(th))
            return latents, image_conditioning

        decoded = decode_latent_batch(
            sd_model,
            upscaled_latents,
            stage="hires.prepare.latent.crop_decode",
        ).to(dtype=torch.float32)
        cropped = decoded[:, :, crop_top : crop_top + th, crop_left : crop_left + tw]
        latents = encode_image_batch(sd_model, cropped, stage="hires.prepare.latent.crop_encode")
        image_conditioning = img2img_conditioning(
            sd_model,
            cropped,
            latents,
            image_mask=image_mask,
            round_mask=round_mask,
        )

        return latents, image_conditioning

    # External SR models (spandrel): decode → SR in pixel space → re-encode.
    if uid.startswith("spandrel:"):
        decoded = base_decoded
        if decoded is None:
            decoded = decode_latent_batch(
                sd_model,
                base_samples,
                stage="hires.prepare.spandrel.decode_base",
            ).to(dtype=torch.float32)
        else:
            decoded = decoded.to(dtype=torch.float32)

        # Convert decoded [-1..1] to pixel [0..1] for SR.
        pixel_01 = decoded.add(1.0).mul(0.5).clamp(0.0, 1.0)
        upscaled_01 = upscale_image_tensor(
            pixel_01,
            upscaler_id=uid,
            target_width=int(internal_w),
            target_height=int(internal_h),
            tile=tile,
            progress_callback=progress_callback,
        )
        if internal_w != tw or internal_h != th or crop_left != 0 or crop_top != 0:
            upscaled_01 = upscaled_01[:, :, crop_top : crop_top + th, crop_left : crop_left + tw]

        # Convert back to [-1..1] for VAE encode and image-conditioning.
        tensor = upscaled_01.mul(2.0).sub(1.0)
        latents = encode_image_batch(sd_model, tensor, stage="hires.prepare.spandrel.encode_upscaled")
        image_conditioning = img2img_conditioning(
            sd_model,
            tensor,
            latents,
            image_mask=image_mask,
            round_mask=round_mask,
        )
        return latents, image_conditioning

    raise ValueError(f"Unsupported hires upscaler id: {uid!r}")


__all__ = ["prepare_hires_latents_and_conditioning"]
