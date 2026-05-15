"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native DEIS coefficient construction for the sampling driver.
Builds the tabulated DEIS multistep coefficients used by the native `deis` sampler branch, keeping the
runtime independent from external sampler imports while matching the vendored reference coefficient path.

Symbols (top-level; keep in sync; no ghosts):
- `build_deis_coefficients` (function): Build cached DEIS coefficient tuples for a sigma ladder.
"""

from __future__ import annotations

from functools import lru_cache
import math

import torch


_DEFAULT_DEIS_EPSILON_S = 1e-3
_DEFAULT_DEIS_SIGMA_MIN = 0.002
_DEFAULT_DEIS_SIGMA_MAX = 80.0
_DEFAULT_DEIS_MAX_ORDER = 3
_DEFAULT_DEIS_INTEGRATION_POINTS = 10_000


def _deis_alpha(beta_0: float, beta_1: float, times: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * times.square() * (beta_1 - beta_0) - times * beta_0)


def _deis_integrand(beta_0: float, beta_1: float, taus: torch.Tensor) -> torch.Tensor:
    taus_work = taus.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        alpha = _deis_alpha(beta_0, beta_1, taus_work)
        log_alpha = alpha.log()
        d_log_alpha_dtau = torch.autograd.grad(log_alpha.sum(), taus_work)[0]
    integrand = -0.5 * d_log_alpha_dtau / torch.sqrt(alpha * (1.0 - alpha))
    if not bool(torch.all(torch.isfinite(integrand))):
        raise RuntimeError("DEIS coefficient construction produced non-finite integrand values.")
    return integrand.detach()


def _deis_lagrange_poly(previous_times: torch.Tensor, basis_index: int, taus: torch.Tensor) -> torch.Tensor:
    polynomial = torch.ones_like(taus)
    basis_time = previous_times[basis_index]
    for other_index, other_time in enumerate(previous_times):
        if other_index == basis_index:
            continue
        denominator = basis_time - other_time
        if abs(float(denominator)) <= 1e-12:
            raise RuntimeError(
                "DEIS coefficient construction encountered duplicate time nodes "
                f"(basis_index={basis_index}, other_index={other_index})."
            )
        polynomial = polynomial * ((taus - other_time) / denominator)
    return polynomial


def _deis_edm_to_t(
    sigmas: torch.Tensor,
    *,
    epsilon_s: float = _DEFAULT_DEIS_EPSILON_S,
    sigma_min: float = _DEFAULT_DEIS_SIGMA_MIN,
    sigma_max: float = _DEFAULT_DEIS_SIGMA_MAX,
) -> tuple[torch.Tensor, float, float]:
    if sigmas.ndim != 1 or int(sigmas.numel()) < 2:
        raise RuntimeError(
            "DEIS coefficient construction requires a 1D sigma ladder with at least two entries "
            f"(got shape={tuple(sigmas.shape)})."
        )
    if not bool(torch.all(torch.isfinite(sigmas))):
        raise RuntimeError("DEIS coefficient construction requires finite sigma values.")

    sigma_values = sigmas.detach().to(device="cpu", dtype=torch.float64)
    vp_beta_d = 2.0 * (
        math.log(sigma_min**2 + 1.0) / epsilon_s - math.log(sigma_max**2 + 1.0)
    ) / (epsilon_s - 1.0)
    vp_beta_min = math.log(sigma_max**2 + 1.0) - 0.5 * vp_beta_d
    t_steps = (
        torch.sqrt(vp_beta_min**2 + 2.0 * vp_beta_d * torch.log(sigma_values.square() + 1.0))
        - vp_beta_min
    ) / vp_beta_d
    if not bool(torch.all(torch.isfinite(t_steps))):
        raise RuntimeError("DEIS coefficient construction produced non-finite time nodes.")
    return t_steps, vp_beta_min, vp_beta_d + vp_beta_min


@lru_cache(maxsize=32)
def _build_deis_coefficients_cached(
    sigma_key: tuple[float, ...],
    max_order: int,
    integration_points: int,
) -> tuple[tuple[float, ...], ...]:
    sigma_steps = torch.tensor(sigma_key, dtype=torch.float64)
    t_steps, beta_0, beta_1 = _deis_edm_to_t(sigma_steps)
    coefficients: list[tuple[float, ...]] = []
    for index, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        if float(sigma_steps[index + 1]) <= 0.0:
            coefficients.append(())
            continue
        order = min(index + 1, max_order)
        if order == 1:
            coefficients.append(())
            continue
        taus = torch.linspace(float(t_cur), float(t_next), integration_points, dtype=torch.float64)
        dtau = (float(t_next) - float(t_cur)) / float(integration_points)
        previous_times = t_steps[[index - offset for offset in range(order)]]
        integrand = _deis_integrand(beta_0, beta_1, taus)
        step_coefficients: list[float] = []
        for basis_index in range(order):
            polynomial = _deis_lagrange_poly(previous_times, basis_index, taus)
            step_coefficients.append(float(torch.sum(integrand * polynomial).item() * dtau))
        coefficients.append(tuple(step_coefficients))
    return tuple(coefficients)


def build_deis_coefficients(
    sigmas: torch.Tensor,
    *,
    max_order: int = _DEFAULT_DEIS_MAX_ORDER,
    integration_points: int = _DEFAULT_DEIS_INTEGRATION_POINTS,
) -> tuple[tuple[float, ...], ...]:
    if max_order < 1 or max_order > 4:
        raise ValueError(f"DEIS max_order must be in [1, 4]; got {max_order}.")
    if integration_points < 2:
        raise ValueError(
            f"DEIS integration_points must be >= 2; got {integration_points}."
        )
    sigma_key = tuple(float(value) for value in sigmas.detach().cpu().tolist())
    if not sigma_key:
        raise RuntimeError("DEIS coefficient construction requires a non-empty sigma ladder.")
    return _build_deis_coefficients_cached(sigma_key, max_order, integration_points)


__all__ = ["build_deis_coefficients"]
