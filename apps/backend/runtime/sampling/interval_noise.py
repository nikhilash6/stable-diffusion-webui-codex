"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Deterministic nested-interval noise helper for native stochastic samplers.
Builds correlated normalized interval noises for nested sigma intervals using driver-owned seeded draws so
native stochastic lanes can preserve Brownian nested-interval geometry without ambient randomness.

Symbols (top-level; keep in sync; no ghosts):
- `compose_nested_interval_noises` (function): Derive correlated normalized noises for nested `[start, mid]` and `[start, end]` intervals.
- `__all__` (constant): Explicit export list for this module.
"""

from __future__ import annotations

import math

import torch


def compose_nested_interval_noises(
    *,
    interval_start: float,
    interval_mid: float,
    interval_end: float,
    first_draw: torch.Tensor,
    second_draw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not math.isfinite(interval_start) or not math.isfinite(interval_mid) or not math.isfinite(interval_end):
        raise RuntimeError(
            "Nested interval noise requires finite interval geometry values "
            f"(start={interval_start}, mid={interval_mid}, end={interval_end})."
        )
    interval_lower = min(interval_start, interval_end)
    interval_upper = max(interval_start, interval_end)
    if not (interval_lower < interval_mid < interval_upper):
        raise RuntimeError(
            "Nested interval noise requires strict nested geometry with mid inside [start, end] "
            f"(start={interval_start}, mid={interval_mid}, end={interval_end})."
        )
    if tuple(first_draw.shape) != tuple(second_draw.shape):
        raise RuntimeError(
            "Nested interval noise draw shape mismatch: "
            f"start_mid={tuple(first_draw.shape)} mid_end={tuple(second_draw.shape)}."
        )
    if not first_draw.is_floating_point() or not second_draw.is_floating_point():
        raise RuntimeError("Nested interval noise draws must be floating-point tensors.")
    if not bool(torch.all(torch.isfinite(first_draw))):
        raise RuntimeError("Nested interval noise received non-finite values in the start-mid draw tensor.")
    if not bool(torch.all(torch.isfinite(second_draw))):
        raise RuntimeError("Nested interval noise received non-finite values in the mid-end draw tensor.")

    start_mid_length = abs(interval_mid - interval_start)
    mid_end_length = abs(interval_end - interval_mid)
    start_end_length = abs(interval_end - interval_start)
    if start_mid_length <= 0.0 or mid_end_length <= 0.0 or start_end_length <= 0.0:
        raise RuntimeError(
            "Nested interval noise requires strictly positive interval lengths "
            f"(start_mid={start_mid_length}, mid_end={mid_end_length}, start_end={start_end_length})."
        )
    length_sum = start_mid_length + mid_end_length
    if not math.isclose(length_sum, start_end_length, rel_tol=1e-6, abs_tol=1e-9):
        raise RuntimeError(
            "Nested interval noise geometry is inconsistent: interval lengths do not compose "
            f"(start_mid + mid_end={length_sum}, start_end={start_end_length})."
        )

    start_mid_weight = math.sqrt(start_mid_length / start_end_length)
    mid_end_weight = math.sqrt(mid_end_length / start_end_length)
    interval_noise_start_end = (
        first_draw * start_mid_weight
        + second_draw * mid_end_weight
    )
    if not bool(torch.all(torch.isfinite(interval_noise_start_end))):
        raise RuntimeError("Nested interval noise composition produced non-finite start-end interval noise.")
    return first_draw, interval_noise_start_end


__all__ = ["compose_nested_interval_noises"]
