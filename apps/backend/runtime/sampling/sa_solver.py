"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Pure SA-Solver coefficient and tau helpers for the native sampling driver.
Provides stochastic Adams coefficient construction and the default sigma-interval tau function used by the native
`sa-solver` / `sa-solver pece` runtime lane. Shared half-logSNR helpers live in `log_snr.py` and are consumed alongside
this module by the driver.

Symbols (top-level; keep in sync; no ghosts):
- `SA_SOLVER_DEFAULT_PREDICTOR_ORDER` (constant): Reference default predictor order for SA-Solver.
- `SA_SOLVER_DEFAULT_CORRECTOR_ORDER` (constant): Reference default corrector order for SA-Solver.
- `SA_SOLVER_DEFAULT_S_NOISE` (constant): Reference default stochastic noise multiplier for SA-Solver.
- `SA_SOLVER_DEFAULT_SIMPLE_ORDER_2` (constant): Reference default simple-order-2 toggle for SA-Solver.
- `SA_SOLVER_DEFAULT_ETA` (constant): Reference default tau interval strength for SA-Solver.
- `compute_exponential_coeffs` (function): Build the factored exponential-integral coefficients used by SA-Solver.
- `compute_simple_stochastic_adams_b_coeffs` (function): Compute the simplified order-2 SA-Solver Adams coefficients.
- `compute_stochastic_adams_b_coeffs` (function): Compute the general SA-Solver stochastic Adams coefficients.
- `get_tau_interval_func` (function): Build the default sigma-interval stochasticity window for SA-Solver.
"""

from __future__ import annotations

import math
from typing import Callable, Union

import torch


SA_SOLVER_DEFAULT_PREDICTOR_ORDER = 3
SA_SOLVER_DEFAULT_CORRECTOR_ORDER = 4
SA_SOLVER_DEFAULT_S_NOISE = 1.0
SA_SOLVER_DEFAULT_SIMPLE_ORDER_2 = False
SA_SOLVER_DEFAULT_ETA = 1.0


def compute_exponential_coeffs(
    s: torch.Tensor,
    t: torch.Tensor,
    solver_order: int,
    tau_t: float,
) -> torch.Tensor:
    if solver_order <= 0:
        raise RuntimeError(f"SA-Solver requires solver_order >= 1; got {solver_order}.")
    if not math.isfinite(tau_t):
        raise RuntimeError(f"SA-Solver received non-finite tau_t={tau_t}.")
    if not bool(torch.all(torch.isfinite(s))) or not bool(torch.all(torch.isfinite(t))):
        raise RuntimeError("SA-Solver exponential coefficients require finite lambda endpoints.")
    tau_mul = 1.0 + tau_t**2
    h = t - s
    p = torch.arange(solver_order, dtype=s.dtype, device=s.device)
    product_terms_factored = t.pow(p) - s.pow(p) * (-tau_mul * h).exp()
    recursive_depth_mat = p.unsqueeze(1) - p.unsqueeze(0)
    log_factorial = (p + 1).lgamma()
    recursive_coeff_mat = log_factorial.unsqueeze(1) - log_factorial.unsqueeze(0)
    if tau_t > 0.0:
        recursive_coeff_mat = recursive_coeff_mat - (recursive_depth_mat * math.log(tau_mul))
    signs = torch.where(recursive_depth_mat % 2 == 0, 1.0, -1.0)
    recursive_coeff_mat = (recursive_coeff_mat.exp() * signs).tril()
    coeffs = recursive_coeff_mat @ product_terms_factored
    if not bool(torch.all(torch.isfinite(coeffs))):
        raise RuntimeError("SA-Solver exponential coefficient construction produced non-finite values.")
    return coeffs


def compute_simple_stochastic_adams_b_coeffs(
    sigma_next: torch.Tensor,
    curr_lambdas: torch.Tensor,
    lambda_s: torch.Tensor,
    lambda_t: torch.Tensor,
    tau_t: float,
    *,
    is_corrector_step: bool = False,
) -> torch.Tensor:
    if curr_lambdas.ndim != 1 or int(curr_lambdas.numel()) != 2:
        raise RuntimeError(
            "SA-Solver simple order-2 coefficients require exactly two lambda history entries."
        )
    tau_mul = 1.0 + tau_t**2
    h = lambda_t - lambda_s
    alpha_t = sigma_next * lambda_t.exp()
    if is_corrector_step:
        b_1 = alpha_t * (0.5 * tau_mul * h)
        b_2 = alpha_t * (-h * tau_mul).expm1().neg() - b_1
    else:
        denominator = curr_lambdas[-2] - lambda_s
        if abs(float(denominator)) <= 1e-12:
            raise RuntimeError("SA-Solver simple order-2 predictor denominator collapsed.")
        b_2 = alpha_t * (0.5 * tau_mul * h.pow(2)) / denominator
        b_1 = alpha_t * (-h * tau_mul).expm1().neg() - b_2
    coeffs = torch.stack([b_2, b_1])
    if not bool(torch.all(torch.isfinite(coeffs))):
        raise RuntimeError("SA-Solver simple stochastic Adams coefficients became non-finite.")
    return coeffs


def compute_stochastic_adams_b_coeffs(
    sigma_next: torch.Tensor,
    curr_lambdas: torch.Tensor,
    lambda_s: torch.Tensor,
    lambda_t: torch.Tensor,
    tau_t: float,
    *,
    simple_order_2: bool = False,
    is_corrector_step: bool = False,
) -> torch.Tensor:
    if curr_lambdas.ndim != 1:
        raise RuntimeError(
            f"SA-Solver stochastic Adams coefficients require a 1D lambda history, got shape={tuple(curr_lambdas.shape)}."
        )
    num_timesteps = int(curr_lambdas.numel())
    if num_timesteps <= 0:
        raise RuntimeError("SA-Solver stochastic Adams coefficients require at least one lambda history entry.")
    if simple_order_2 and num_timesteps == 2:
        return compute_simple_stochastic_adams_b_coeffs(
            sigma_next,
            curr_lambdas,
            lambda_s,
            lambda_t,
            tau_t,
            is_corrector_step=is_corrector_step,
        )
    exp_integral_coeffs = compute_exponential_coeffs(lambda_s, lambda_t, num_timesteps, tau_t)
    vandermonde_matrix_t = torch.vander(curr_lambdas, num_timesteps, increasing=True).T
    try:
        lagrange_integrals = torch.linalg.solve(
            vandermonde_matrix_t.to(dtype=torch.float64),
            exp_integral_coeffs.to(dtype=torch.float64),
        ).to(dtype=sigma_next.dtype)
    except RuntimeError as exc:
        raise RuntimeError(
            "SA-Solver stochastic Adams coefficient solve failed "
            f"(num_timesteps={num_timesteps}, tau_t={tau_t})."
        ) from exc
    alpha_t = sigma_next * lambda_t.exp()
    coeffs = alpha_t * lagrange_integrals
    if not bool(torch.all(torch.isfinite(coeffs))):
        raise RuntimeError("SA-Solver stochastic Adams coefficients became non-finite.")
    return coeffs


def get_tau_interval_func(
    start_sigma: float,
    end_sigma: float,
    *,
    eta: float = SA_SOLVER_DEFAULT_ETA,
) -> Callable[[Union[torch.Tensor, float]], float]:
    if not math.isfinite(start_sigma) or not math.isfinite(end_sigma):
        raise RuntimeError(
            f"SA-Solver tau interval requires finite sigma bounds; got start={start_sigma}, end={end_sigma}."
        )
    if start_sigma <= 0.0 or end_sigma <= 0.0:
        raise RuntimeError(
            f"SA-Solver tau interval requires strictly positive sigma bounds; got start={start_sigma}, end={end_sigma}."
        )
    if start_sigma < end_sigma:
        raise RuntimeError(
            f"SA-Solver tau interval requires start_sigma >= end_sigma; got start={start_sigma}, end={end_sigma}."
        )
    if not math.isfinite(eta):
        raise RuntimeError(f"SA-Solver eta must be finite; got {eta!r}.")

    def tau_func(sigma: Union[torch.Tensor, float]) -> float:
        sigma_value = float(sigma.item()) if isinstance(sigma, torch.Tensor) else float(sigma)
        if not math.isfinite(sigma_value):
            raise RuntimeError(f"SA-Solver tau interval received non-finite sigma {sigma_value!r}.")
        if eta <= 0.0:
            return 0.0
        return eta if start_sigma >= sigma_value >= end_sigma else 0.0

    return tau_func


__all__ = [
    "SA_SOLVER_DEFAULT_CORRECTOR_ORDER",
    "SA_SOLVER_DEFAULT_ETA",
    "SA_SOLVER_DEFAULT_PREDICTOR_ORDER",
    "SA_SOLVER_DEFAULT_SIMPLE_ORDER_2",
    "SA_SOLVER_DEFAULT_S_NOISE",
    "compute_exponential_coeffs",
    "compute_simple_stochastic_adams_b_coeffs",
    "compute_stochastic_adams_b_coeffs",
    "get_tau_interval_func",
]
