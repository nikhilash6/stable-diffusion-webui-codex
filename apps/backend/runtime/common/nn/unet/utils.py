"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Utility helpers for the UNet implementation (conv/pool factories, embeddings, and control application).

Symbols (top-level; keep in sync; no ghosts):
- `checkpoint` (function): Gradient checkpoint stub (raises `NotImplementedError` when enabled).
- `exists` (function): True when a value is not `None`.
- `default` (function): Return a fallback when the provided value is `None`.
- `conv_nd` (function): Build a Conv layer for 2D/3D tensors.
- `avg_pool_nd` (function): Build an AvgPool layer for 1D/2D/3D tensors.
- `apply_control` (function): Apply a residual from a `control` dict to a tensor (best-effort; logs on mismatch).
- `timestep_embedding` (function): Sinusoidal timestep embedding helper.
- `ensure_sequence` (function): Normalize a scalar/iterable into an integer tuple, optionally padding to a target length.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging

import math
from typing import Iterable, Tuple

import torch
from einops import repeat
from torch import nn

_log = get_backend_logger("backend.runtime.common.nn.unet.utils")


def checkpoint(function, args, parameters, enable: bool = False):
    if enable:
        raise NotImplementedError("Gradient checkpointing is not implemented in Codex yet.")
    return function(*args)


def exists(value):
    return value is not None


def default(value, fallback):
    return value if exists(value) else fallback


def conv_nd(dims, *args, **kwargs):
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported convolution dimensionality: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"Unsupported pooling dimensionality: {dims}")


def apply_control(tensor, control, name):
    if control is not None and name in control and control[name]:
        ctrl = control[name].pop()
        if ctrl is not None:
            try:
                tensor = tensor + ctrl
            except Exception:
                _log.warning("control could not be applied tensor=%s ctrl=%s", tensor.shape, ctrl.shape)
    return tensor


def timestep_embedding(timesteps, dim, max_period: int = 10000, repeat_only: bool = False):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(
            start=0, end=half, dtype=torch.float32, device=timesteps.device
        ) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def ensure_sequence(value, *, length: int | None = None, fill: int | None = None) -> Tuple[int, ...]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        seq = tuple(int(v) for v in value)
    else:
        seq = (int(value),)
    if length is not None and len(seq) < length:
        if len(seq) == 1:
            seq = tuple(seq[0] for _ in range(length))
        elif fill is not None:
            seq = seq + tuple(fill for _ in range(length - len(seq)))
    return seq
