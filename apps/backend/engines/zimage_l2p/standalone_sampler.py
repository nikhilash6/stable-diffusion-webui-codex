"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Pixel-space FlowMatch sampler for Z-Image L2P.
Implements the upstream Z-Image scheduler math for exact 1024x1024 L2P txt2img without VAE decode or latent-space fallthrough.

Symbols (top-level; keep in sync; no ghosts):
- `build_zimage_l2p_sigmas` (function): Builds shifted FlowMatch sigma schedule for L2P.
- `sample_zimage_l2p_pixel_txt2img` (function): Runs classic-CFG L2P pixel-space Euler sampling and returns decoded RGB tensors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


def build_zimage_l2p_sigmas(
    *,
    steps: int,
    denoising_strength: float = 1.0,
    shift: float = 3.0,
    device: torch.device | str,
) -> torch.Tensor:
    """Build the shifted sigma schedule used by the upstream L2P Z-Image scheduler."""

    if int(steps) <= 0:
        raise RuntimeError(f"Z-Image L2P requires positive steps; got {steps}.")
    sigma_start = float(denoising_strength)
    if sigma_start <= 0.0 or sigma_start > 1.0:
        raise RuntimeError(f"Z-Image L2P denoising_strength must be in (0, 1]; got {sigma_start}.")
    raw = torch.linspace(sigma_start, 0.0, int(steps) + 1, dtype=torch.float32, device=device)[:-1]
    return float(shift) * raw / (1.0 + (float(shift) - 1.0) * raw)


def _require_prompt_embeddings(value: object, *, label: str, batch_size: int) -> list[torch.Tensor]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(f"Z-Image L2P {label} conditioning must be a list of per-prompt tensors.")
    tensors = list(value)
    if len(tensors) != int(batch_size):
        raise RuntimeError(
            f"Z-Image L2P {label} conditioning batch mismatch: "
            f"got {len(tensors)} tensors for batch={batch_size}."
        )
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or int(tensor.shape[-1]) != 2560:
            raise RuntimeError(
                f"Z-Image L2P {label} conditioning tensor {index} must be [tokens,2560]; "
                f"got type={type(tensor).__name__} shape={getattr(tensor, 'shape', None)}."
            )
    return tensors


def _move_embeddings(
    embeddings: Sequence[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    return [tensor.to(device=device, dtype=dtype) for tensor in embeddings]


@torch.inference_mode()
def sample_zimage_l2p_pixel_txt2img(
    *,
    model: torch.nn.Module,
    noise: torch.Tensor,
    cond: object,
    uncond: object | None,
    steps: int,
    guidance_scale: float,
    flow_shift: float = 3.0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> torch.Tensor:
    """Run L2P pixel-space Euler sampling and return decoded RGB `[B,3,H,W]` in `[-1, 1]`."""

    if noise.ndim != 4 or int(noise.shape[1]) != 3:
        raise RuntimeError(f"Z-Image L2P noise must be [B,3,H,W]; got {tuple(noise.shape)}.")
    batch_size = int(noise.shape[0])
    height = int(noise.shape[2])
    width = int(noise.shape[3])
    if height != 1024 or width != 1024:
        raise RuntimeError(f"Z-Image L2P first tranche supports exactly 1024x1024; got {width}x{height}.")

    device = noise.device
    model_dtype = getattr(model, "computation_dtype", None)
    if not isinstance(model_dtype, torch.dtype):
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32

    cond_list = _move_embeddings(
        _require_prompt_embeddings(cond, label="positive", batch_size=batch_size),
        device=device,
        dtype=model_dtype,
    )
    uncond_list = None
    cfg_scale = float(guidance_scale)
    if cfg_scale > 1.0:
        uncond_list = _move_embeddings(
            _require_prompt_embeddings(uncond, label="negative", batch_size=batch_size),
            device=device,
            dtype=model_dtype,
        )

    sample = noise.to(device=device, dtype=torch.float32)
    sigmas = build_zimage_l2p_sigmas(steps=int(steps), shift=float(flow_shift), device=device)
    sigma_next = torch.cat([sigmas[1:], torch.zeros((1,), dtype=sigmas.dtype, device=device)], dim=0)

    for step_index, sigma in enumerate(sigmas):
        sigma_batch = sigma.repeat(batch_size).to(device=device)
        model_input = sample.to(dtype=model_dtype)
        cond_pred = model(model_input, sigma_batch, prompt_embeds=cond_list).float()
        if uncond_list is not None:
            uncond_pred = model(model_input, sigma_batch, prompt_embeds=uncond_list).float()
            model_output = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        else:
            model_output = cond_pred
        sample = sample + model_output * (sigma_next[step_index] - sigma)
        if progress_callback is not None:
            progress_callback(
                {
                    "step": int(step_index + 1),
                    "steps": int(steps),
                    "sigma": float(sigma.detach().cpu().item()),
                }
            )

    return sample.clamp(-1.0, 1.0)


__all__ = ["build_zimage_l2p_sigmas", "sample_zimage_l2p_pixel_txt2img"]
