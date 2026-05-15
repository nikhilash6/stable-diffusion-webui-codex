"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR color-fix tensor operators.
Provides the bounded tensor-only color-fix routines used by native SUPIR mode after decode.
The implementation is adapted to Codex style and intentionally stays tensor-native (no PIL transforms)
so the canonical image egress path can reuse decoded tensors without a second conversion hop.

Symbols (top-level; keep in sync; no ghosts):
- `adaptive_instance_normalization` (function): Match the target tensor statistics to the reference tensor.
- `wavelet_reconstruction` (function): Recompose target high frequencies with the reference low-frequency color structure.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _calc_mean_std(feat: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    if feat.ndim != 4:
        raise ValueError(f"SUPIR color-fix expects BCHW tensors; got shape={tuple(feat.shape)}")
    batch, channels = feat.shape[:2]
    feat_var = feat.reshape(batch, channels, -1).var(dim=2, unbiased=False) + float(eps)
    feat_std = feat_var.sqrt().reshape(batch, channels, 1, 1)
    feat_mean = feat.reshape(batch, channels, -1).mean(dim=2).reshape(batch, channels, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(target_feat: Tensor, reference_feat: Tensor) -> Tensor:
    if tuple(target_feat.shape) != tuple(reference_feat.shape):
        raise ValueError(
            "SUPIR AdaIN color-fix requires matching BCHW tensors: "
            f"target={tuple(target_feat.shape)} reference={tuple(reference_feat.shape)}"
        )
    reference_mean, reference_std = _calc_mean_std(reference_feat)
    target_mean, target_std = _calc_mean_std(target_feat)
    normalized = (target_feat - target_mean.expand_as(target_feat)) / target_std.expand_as(target_feat)
    return normalized * reference_std.expand_as(target_feat) + reference_mean.expand_as(target_feat)


def _wavelet_blur(image: Tensor, radius: int) -> Tensor:
    if image.ndim != 4:
        raise ValueError(f"SUPIR wavelet blur expects BCHW tensors; got shape={tuple(image.shape)}")
    if int(image.shape[1]) != 3:
        raise ValueError(f"SUPIR wavelet blur expects RGB tensors with 3 channels; got shape={tuple(image.shape)}")
    kernel = torch.tensor(
        [
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625],
        ],
        dtype=image.dtype,
        device=image.device,
    )
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    padded = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel, groups=3, dilation=radius)


def _wavelet_decomposition(image: Tensor, *, levels: int = 5) -> tuple[Tensor, Tensor]:
    high_freq = torch.zeros_like(image)
    low_freq = image
    for level in range(int(levels)):
        radius = 2**level
        low_freq = _wavelet_blur(low_freq, radius)
        high_freq = high_freq + (image - low_freq)
        image = low_freq
    return high_freq, low_freq


def wavelet_reconstruction(target_feat: Tensor, reference_feat: Tensor) -> Tensor:
    if tuple(target_feat.shape) != tuple(reference_feat.shape):
        raise ValueError(
            "SUPIR wavelet color-fix requires matching BCHW tensors: "
            f"target={tuple(target_feat.shape)} reference={tuple(reference_feat.shape)}"
        )
    target_high, _target_low = _wavelet_decomposition(target_feat)
    _reference_high, reference_low = _wavelet_decomposition(reference_feat)
    return target_high + reference_low


__all__ = ["adaptive_instance_normalization", "wavelet_reconstruction"]
