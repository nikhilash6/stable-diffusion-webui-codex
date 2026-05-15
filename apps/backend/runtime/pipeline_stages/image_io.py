"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Image/latent conversion helpers used by pipeline orchestrators.
Provides small utilities to convert between decoded latent tensors and PIL images, and to decode samples for hi-res stages with
upscaler-aware gating (latent hires skips redundant base decode).

Symbols (top-level; keep in sync; no ghosts):
- `latents_to_pil` (function): Convert decoded latent tensors into RGB PIL images.
- `pil_to_tensor` (function): Convert PIL images into a normalized tensor suitable for img2img/conditioning.
- `maybe_decode_for_hr` (function): Decode samples to RGB only when the resolved hires upscaler requires pixel-space input, preserving engine output dtype.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from apps.backend.core import devices
from apps.backend.runtime.processing.conditioners import decode_latent_batch


def latents_to_pil(decoded: torch.Tensor) -> list[Image.Image]:
    """Convert decoded latent tensor into RGB PIL images."""
    images: list[Image.Image] = []
    for sample in decoded:
        arr = sample.detach().cpu().float().clamp(-1, 1)
        arr = ((arr + 1.0) * 0.5).mul(255.0).byte().movedim(0, -1).numpy()
        images.append(Image.fromarray(arr, mode="RGB"))
    return images


def pil_to_tensor(images: Sequence[Image.Image]) -> torch.Tensor:
    """Convert a sequence of PIL images into a normalized tensor."""
    arrays = []
    for img in images:
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        arr = np.moveaxis(arr, 2, 0)
        arrays.append(arr)
    tensor = torch.from_numpy(np.stack(arrays, axis=0))
    return tensor.to(devices.default_device(), dtype=torch.float32)


def maybe_decode_for_hr(
    processing: Any,
    samples: torch.Tensor,
    *,
    hires_upscaler_id: str | None = None,
) -> torch.Tensor | None:
    """Decode samples to RGB only when hires preparation needs pixel-space input."""
    hires_cfg = getattr(processing, "hires", None)
    if not bool(getattr(hires_cfg, "enabled", False)):
        return None

    upscaler_id = hires_upscaler_id
    if not isinstance(upscaler_id, str) or not upscaler_id.strip():
        candidate = getattr(hires_cfg, "upscaler", None) if hires_cfg is not None else None
        upscaler_id = str(candidate).strip() if isinstance(candidate, str) else ""

    if upscaler_id.startswith("latent:"):
        return None

    devices.torch_gc()
    decoded = decode_latent_batch(
        processing.sd_model,
        samples,
        stage="hires.prepare.base_decode",
    )
    return decoded
