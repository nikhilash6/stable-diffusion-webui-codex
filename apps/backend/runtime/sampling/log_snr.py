"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared half-logSNR and exponential-integrator helpers for native sampling families.
Provides CONST-aware first-sigma offset handling, half-logSNR forward/inverse conversions, and stable phi helpers reused by
native SA-Solver and SEEDS-family driver lanes.

Symbols (top-level; keep in sync; no ghosts):
- `HALF_LOG_SNR_PERCENT_OFFSET` (constant): Default percent offset used to avoid the CONST first-sigma singular boundary.
- `offset_first_sigma_for_snr` (function): Adjust the first sigma to avoid invalid CONST half-logSNR singularities.
- `sigma_to_half_log_snr` (function): Convert sigma ladders to half-logSNR values with prediction-type-aware rules.
- `half_log_snr_to_sigma` (function): Invert half-logSNR values back to sigma values with prediction-type-aware rules.
- `ei_h_phi_1` (function): Compute the `h * phi_1(h)` term used by exponential-integrator solvers.
- `ei_h_phi_2` (function): Compute the `h * phi_2(h)` term used by exponential-integrator solvers.
"""

from __future__ import annotations

import math
from typing import Any

import torch


HALF_LOG_SNR_PERCENT_OFFSET = 1e-4


def offset_first_sigma_for_snr(
    sigmas: torch.Tensor,
    *,
    prediction_type: str | None,
    predictor: Any,
    percent_offset: float = HALF_LOG_SNR_PERCENT_OFFSET,
) -> torch.Tensor:
    if sigmas.ndim != 1:
        raise RuntimeError(f"Half-logSNR helpers expect a 1D sigma schedule, got shape={tuple(sigmas.shape)}.")
    if int(sigmas.numel()) <= 1:
        return sigmas
    if not bool(torch.all(torch.isfinite(sigmas))):
        raise RuntimeError("Half-logSNR helpers require a finite sigma schedule before applying the first-sigma offset.")
    if prediction_type != "const":
        return sigmas
    if not math.isfinite(percent_offset) or not (0.0 < percent_offset < 1.0):
        raise RuntimeError(
            f"Half-logSNR first-sigma offset must be finite and inside (0, 1); got {percent_offset!r}."
        )
    if float(sigmas[0]) < 1.0:
        return sigmas
    percent_to_sigma = getattr(predictor, "percent_to_sigma", None)
    if not callable(percent_to_sigma):
        raise RuntimeError(
            "CONST half-logSNR conversion requires predictor.percent_to_sigma(...) for the first-sigma offset."
        )
    shifted_sigma = float(percent_to_sigma(percent_offset))
    if not math.isfinite(shifted_sigma) or not (0.0 < shifted_sigma < 1.0):
        raise RuntimeError(
            "predictor.percent_to_sigma(...) returned an invalid first-sigma offset "
            f"{shifted_sigma!r} for percent_offset={percent_offset}."
        )
    adjusted = sigmas.clone()
    adjusted[0] = shifted_sigma
    return adjusted


def sigma_to_half_log_snr(sigmas: torch.Tensor, *, prediction_type: str | None) -> torch.Tensor:
    if sigmas.ndim != 1:
        raise RuntimeError(f"Half-logSNR helpers expect a 1D sigma schedule, got shape={tuple(sigmas.shape)}.")
    if not bool(torch.all(torch.isfinite(sigmas))):
        raise RuntimeError("Half-logSNR helpers require a finite sigma schedule for forward conversion.")
    sigmas_fp32 = sigmas.to(dtype=torch.float32)
    if prediction_type == "const":
        sigma_safe = sigmas_fp32.clamp(min=1e-6, max=1.0 - 1e-6)
        half_log_snr = -torch.logit(sigma_safe)
    else:
        sigma_safe = sigmas_fp32.clamp(min=1e-12)
        half_log_snr = -torch.log(sigma_safe)
    if not bool(torch.all(torch.isfinite(half_log_snr))):
        raise RuntimeError("Half-logSNR forward conversion produced non-finite values.")
    return half_log_snr


def half_log_snr_to_sigma(half_log_snr: torch.Tensor, *, prediction_type: str | None) -> torch.Tensor:
    if not bool(torch.all(torch.isfinite(half_log_snr))):
        raise RuntimeError("Half-logSNR inverse conversion requires finite inputs.")
    half_log_snr_fp32 = half_log_snr.to(dtype=torch.float32)
    if prediction_type == "const":
        sigma = torch.sigmoid(-half_log_snr_fp32)
    else:
        sigma = torch.exp(-half_log_snr_fp32)
    if not bool(torch.all(torch.isfinite(sigma))):
        raise RuntimeError("Half-logSNR inverse conversion produced non-finite sigma values.")
    if bool(torch.any(sigma <= 0.0)):
        raise RuntimeError("Half-logSNR inverse conversion produced non-positive sigma values.")
    return sigma.to(dtype=half_log_snr.dtype)


def ei_h_phi_1(h: torch.Tensor) -> torch.Tensor:
    if not bool(torch.all(torch.isfinite(h))):
        raise RuntimeError("Exponential-integrator phi_1 requires finite inputs.")
    result = torch.expm1(h)
    if not bool(torch.all(torch.isfinite(result))):
        raise RuntimeError("Exponential-integrator phi_1 produced non-finite values.")
    return result


def ei_h_phi_2(h: torch.Tensor) -> torch.Tensor:
    if not bool(torch.all(torch.isfinite(h))):
        raise RuntimeError("Exponential-integrator phi_2 requires finite inputs.")
    abs_h = torch.abs(h)
    safe_h = torch.where(abs_h <= 1e-6, torch.ones_like(h), h)
    raw = (torch.expm1(h) - h) / safe_h
    series = 0.5 + (h / 6.0) + (h.square() / 24.0)
    result = torch.where(abs_h <= 1e-6, series, raw)
    if not bool(torch.all(torch.isfinite(result))):
        raise RuntimeError("Exponential-integrator phi_2 produced non-finite values.")
    return result


__all__ = [
    "HALF_LOG_SNR_PERCENT_OFFSET",
    "ei_h_phi_1",
    "ei_h_phi_2",
    "half_log_snr_to_sigma",
    "offset_first_sigma_for_snr",
    "sigma_to_half_log_snr",
]
