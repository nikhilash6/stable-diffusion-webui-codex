"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sigma schedule construction utilities for diffusion samplers.
Defines canonical scheduler names and builds sigma schedules (Karras, exponential, DDIM, beta, align-your-steps, etc.).
SIMPLE schedules are predictor-aware and support explicit mode selection (legacy shifted-linspace vs tail-downsample sigma selection), and
`linear_quadratic` now follows the parity-target piecewise shape with strict input guards.

Symbols (top-level; keep in sync; no ghosts):
- `SchedulerName` (enum): Canonical scheduler names for sigma schedule construction (strict, no silent fallback).
- `_append_zero` (function): Appends a terminal sigma=0 to a sigma schedule tensor.
- `_karras_schedule` (function): Builds a Karras sigma schedule.
- `_polyexponential_schedule` (function): Builds a polyexponential sigma schedule.
- `_exponential_schedule` (function): Builds an exponential sigma schedule.
- `_uniform_schedule_from_predictor` (function): Builds a uniform schedule from a predictor function.
- `_simple_schedule_from_predictor` (function): Builds a simple schedule from a predictor function.
- `_sgm_uniform_schedule` (function): Builds an SGM-uniform schedule (SGM-style).
- `_ddim_uniform_schedule` (function): Builds a DDIM-uniform schedule.
- `_normal_schedule` (function): Builds a normal schedule (optionally SGM variant).
- `_beta_continued_fraction` (function): Evaluate the continued-fraction form used by the regularized incomplete beta function.
- `_regularized_incomplete_beta` (function): Evaluate the regularized incomplete beta CDF for scalar quantile inversion.
- `_inverse_regularized_incomplete_beta` (function): Numerically invert the regularized incomplete beta CDF with bisection.
- `_predictor_sigma_ladder` (function): Validates and returns predictor sigma ladders for ladder-backed schedulers.
- `_beta_schedule` (function): Builds a beta schedule.
- `_linear_quadratic_schedule` (function): Builds a linear-quadratic schedule.
- `_kl_optimal_schedule` (function): Builds a KL-optimal schedule.
- `_align_your_steps_schedule` (function): Builds the “align your steps” schedule variants (SDXL aware).
- `_turbo_schedule` (function): Builds a turbo schedule.
- `build_sigma_schedule` (function): Main scheduler entrypoint; selects the schedule builder and returns the sigma tensor.
- `__all__` (constant): Export list for the schedule helpers.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import torch
from apps.backend.runtime.sampling_adapters.prediction import SIMPLE_SCHEDULE_MODE_TAIL_DOWNSAMPLE_SIGMAS


class SchedulerName(str, Enum):
    """Canonical scheduler names for sigma schedule construction.

    Notes
    - This controls ONLY the sigma schedule construction, not the integrator
      (which is selected via the sampler kind).
    - Values are strict and fail-fast: unknown scheduler names raise with a clear message.
    """

    SIMPLE = "simple"
    KARRAS = "karras"
    EULER_DISCRETE = "euler_discrete"
    EXPONENTIAL = "exponential"
    POLYEXPONENTIAL = "polyexponential"
    UNIFORM = "uniform"
    SGM_UNIFORM = "sgm_uniform"
    DDIM = "ddim"
    DDIM_UNIFORM = "ddim_uniform"
    NORMAL = "normal"
    BETA = "beta"
    LINEAR_QUADRATIC = "linear_quadratic"
    KL_OPTIMAL = "kl_optimal"
    TURBO = "turbo"
    ALIGN_YOUR_STEPS = "align_your_steps"
    ALIGN_YOUR_STEPS_GITS = "align_your_steps_gits"
    ALIGN_YOUR_STEPS_11 = "align_your_steps_11"
    ALIGN_YOUR_STEPS_32 = "align_your_steps_32"

    @staticmethod
    def from_string(name: str) -> "SchedulerName":
        if not isinstance(name, str):
            raise TypeError("scheduler name must be a string")
        if not name:
            raise ValueError("scheduler name must not be empty")
        try:
            return SchedulerName(name)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported scheduler '{name}'. Supported: {[m.value for m in SchedulerName]}"
            ) from exc


def _append_zero(sigmas: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    terminal = torch.zeros(1, device=device, dtype=dtype)
    return torch.cat([sigmas.to(device=device, dtype=dtype), terminal])


def _karras_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rho: float = 7.0,
) -> torch.Tensor:
    ramp = torch.linspace(0, 1, steps, device=device, dtype=dtype)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + (min_inv - max_inv) * ramp) ** rho
    return _append_zero(sigmas, device=device, dtype=dtype)


def _polyexponential_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rho: float = 1.0,
) -> torch.Tensor:
    ramp = torch.linspace(1, 0, steps, device=device, dtype=dtype) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return _append_zero(sigmas, device=device, dtype=dtype)


def _exponential_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ramp = torch.linspace(math.log(sigma_max), math.log(sigma_min), steps, device=device, dtype=dtype)
    sigmas = ramp.exp()
    return _append_zero(sigmas, device=device, dtype=dtype)


def _uniform_schedule_from_predictor(steps: int, predictor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigmas = getattr(predictor, "sigmas", None)
    if sigmas is None:
        raise RuntimeError("predictor does not expose 'sigmas' needed for uniform schedule")
    sigmas = torch.as_tensor(sigmas, device=device, dtype=dtype)
    # Evenly sample sigmas from the predictor ladder (skip sigma=inf at index 0)
    stride = max(int(math.floor(len(sigmas) / max(steps, 1))), 1)
    ladder = sigmas[1::stride][:steps]
    if ladder.numel() < steps:
        ladder = torch.nn.functional.interpolate(
            ladder.view(1, 1, -1), size=steps, mode="linear", align_corners=False
        ).view(-1)
    return _append_zero(ladder, device=device, dtype=dtype)


def _simple_schedule_from_predictor(
    steps: int,
    predictor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    flow_shift: float | None = None,
) -> torch.Tensor:
    """Predictor-ladder 'simple' schedule built via predictor sigma/timestep.

    Behavior depends on predictor configuration:
    - If `predictor.simple_schedule_mode == "tail_downsample_sigmas"`, build a SIMPLE ladder by downsampling
      `predictor.sigmas` from the **end** and appending a terminal 0. This mode now fails loud when
      `steps > len(predictor.sigmas)` because tail downsampling beyond the base ladder would duplicate the head sigma.
    - Otherwise, use the legacy Codex behavior:
      - special-case `prediction_type="const"` for FlowMatch-style shifted linspace ladders,
      - else construct a linear ladder of timesteps and map via `predictor.sigma(t)`,
      - append a terminal 0.
    """

    mode = getattr(predictor, "simple_schedule_mode", None)
    if isinstance(mode, str) and mode.strip().lower() == SIMPLE_SCHEDULE_MODE_TAIL_DOWNSAMPLE_SIGMAS:
        base_sigmas = getattr(predictor, "sigmas", None)
        if base_sigmas is None:
            raise RuntimeError("predictor is missing sigmas ladder required for tail-downsample simple schedule")
        steps_i = int(steps)
        if steps_i <= 0:
            raise ValueError("steps must be >= 1 for tail-downsample simple schedule")
        sigmas = torch.as_tensor(base_sigmas, device=device, dtype=dtype)
        if sigmas.ndim != 1:
            raise RuntimeError(
                "predictor.sigmas must be a 1D ladder for tail-downsample simple schedule; "
                f"got shape={tuple(sigmas.shape)}."
            )
        if sigmas.numel() == 0:
            raise RuntimeError("predictor.sigmas is empty; cannot build tail-downsample simple sigma schedule")
        if not torch.isfinite(sigmas).all().item():
            raise RuntimeError("predictor.sigmas must contain only finite values for tail-downsample simple schedule.")
        if (sigmas[1:] < sigmas[:-1]).any().item():
            raise RuntimeError(
                "predictor.sigmas must be monotonically non-decreasing (sigma_min -> sigma_max) "
                "for tail-downsample simple schedule."
            )
        total = int(sigmas.numel())
        if steps_i > total:
            raise ValueError(
                "Tail-downsample simple sigma schedule requires steps <= len(predictor.sigmas); "
                f"got steps={steps_i} predictor_sigmas={total}. Downsampling beyond the base ladder "
                "would duplicate the head sigma and break RF/CONST ancestral steps."
            )
        # The tail-downsample index formula uses `floor(i * total / steps)` for i in [0..steps-1],
        # then selects from the ladder tail toward the head.
        indices = [total - 1 - ((i * total) // steps_i) for i in range(steps_i)]
        ladder = sigmas[torch.as_tensor(indices, device=device, dtype=torch.long)]
        return _append_zero(ladder, device=device, dtype=dtype)

    # Special case: FlowMatchEulerPrediction (Z Image Turbo) expects the diffusers
    # `ZImagePipeline` schedule shape:
    # - The pipeline sets `scheduler.sigma_min = 0.0` and calls `set_timesteps(N)`.
    # - This yields base sigmas linearly spaced from 1.0 -> 0.0 (inclusive),
    #   then applies the `shift` transform once (and appends a terminal 0).
    # - Because 0.0 is already present as the final inference sigma, the schedule
    #   ends with a double-zero tail (dt=0 for the last step). This matches the
    #   upstream recommendation `num_inference_steps=9` to get 8 effective steps.
    pred_type = getattr(predictor, "prediction_type", None)
    if isinstance(pred_type, str) and pred_type.lower() == "const":
        shift_value: float | None
        if flow_shift is not None:
            shift_value = float(flow_shift)
        else:
            mu = getattr(predictor, "mu", None)
            try:
                shift_value = float(mu) if mu is not None else None
            except Exception:  # noqa: BLE001 - defensive
                shift_value = None

        pseudo = getattr(predictor, "pseudo_timestep_range", None)
        if shift_value is not None and shift_value > 0.0:
            # Z Image Turbo parity: when pseudo_timestep_range=1000, diffusers forces
            # sigma_min=0.0, yielding a double-zero tail after appending terminal 0.
            include_zero = int(pseudo or 0) == 1000
            if include_zero:
                base = torch.linspace(1.0, 0.0, int(steps), device=device, dtype=dtype)
            else:
                denom: float | None = None
                try:
                    sigmas = getattr(predictor, "sigmas", None)
                    if sigmas is not None:
                        total = int(len(sigmas))
                        if total > 0:
                            denom = float(total)
                except Exception:  # noqa: BLE001 - best-effort
                    denom = None
                if denom is None:
                    try:
                        denom = float(int(pseudo or 0)) if int(pseudo or 0) > 0 else None
                    except Exception:  # noqa: BLE001 - best-effort
                        denom = None
                if denom is None:
                    denom = float(max(int(steps), 1))
                base = torch.linspace(1.0, 1.0 / denom, int(steps), device=device, dtype=dtype)
            if shift_value == 1.0:
                shifted = base
            else:
                shifted = shift_value * base / (1.0 + (shift_value - 1.0) * base)
            return _append_zero(shifted, device=device, dtype=dtype)

    base_sigmas = getattr(predictor, "sigmas", None)
    if base_sigmas is None:
        raise RuntimeError("predictor does not expose 'sigmas' needed for simple scheduler")
    total = int(len(base_sigmas))
    if total == 0:
        raise RuntimeError("predictor.sigmas is empty; cannot build simple sigma schedule")

    t_max = total - 1
    t = torch.linspace(float(t_max), 0.0, int(steps), device=device, dtype=dtype)
    ladder = predictor.sigma(t)
    return _append_zero(ladder, device=device, dtype=dtype)


def _sgm_uniform_schedule(steps: int, predictor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    start = predictor.sigma_to_t(torch.as_tensor(float(predictor.sigma_max), device=device, dtype=dtype))
    end = predictor.sigma_to_t(torch.as_tensor(float(predictor.sigma_min), device=device, dtype=dtype))
    timesteps = torch.linspace(start, end, steps + 1, device=device, dtype=dtype)[:-1]
    sigmas = predictor.t_to_sigma(timesteps)
    return _append_zero(sigmas, device=device, dtype=dtype)


def _ddim_uniform_schedule(steps: int, predictor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigmas = torch.as_tensor(getattr(predictor, "sigmas", []), device=device, dtype=dtype)
    if sigmas.numel() == 0:
        raise RuntimeError("predictor is missing sigmas ladder required for DDIM uniform schedule")
    stride = max(len(sigmas) // max(steps, 1), 1)
    ladder = sigmas[1::stride]
    ladder = ladder.flip(0)[:steps]
    return _append_zero(ladder, device=device, dtype=dtype)


def _normal_schedule(steps: int, predictor, *, device: torch.device, dtype: torch.dtype, sgm: bool = False) -> torch.Tensor:
    start = predictor.timestep(torch.as_tensor(float(predictor.sigma_max), device=device, dtype=dtype))
    end = predictor.timestep(torch.as_tensor(float(predictor.sigma_min), device=device, dtype=dtype))
    if sgm:
        timesteps = torch.linspace(start, end, steps + 1, device=device, dtype=dtype)[:-1]
    else:
        timesteps = torch.linspace(start, end, steps, device=device, dtype=dtype)
    sigmas = predictor.sigma(timesteps)
    return _append_zero(sigmas, device=device, dtype=dtype)


def _beta_continued_fraction(
    alpha: float,
    beta: float,
    x: float,
    *,
    max_iterations: int = 200,
    epsilon: float = 3.0e-14,
) -> float:
    if not (0.0 < x < 1.0):
        raise ValueError(f"Beta continued fraction requires x in (0, 1); got {x}.")
    qab = alpha + beta
    qap = alpha + 1.0
    qam = alpha - 1.0

    machine_min = 1.0e-30
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < machine_min:
        d = machine_min
    d = 1.0 / d
    fraction = d

    for iteration in range(1, max_iterations + 1):
        m = float(iteration)
        m2 = 2.0 * m

        aa = m * (beta - m) * x / ((qam + m2) * (alpha + m2))
        d = 1.0 + aa * d
        if abs(d) < machine_min:
            d = machine_min
        c = 1.0 + aa / c
        if abs(c) < machine_min:
            c = machine_min
        d = 1.0 / d
        fraction *= d * c

        aa = -(alpha + m) * (qab + m) * x / ((alpha + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < machine_min:
            d = machine_min
        c = 1.0 + aa / c
        if abs(c) < machine_min:
            c = machine_min
        d = 1.0 / d
        delta = d * c
        fraction *= delta

        if abs(delta - 1.0) <= epsilon:
            return fraction

    raise RuntimeError(
        f"Beta continued fraction did not converge for alpha={alpha}, beta={beta}, x={x}."
    )


def _regularized_incomplete_beta(alpha: float, beta: float, x: float) -> float:
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"regularized incomplete beta requires alpha > 0; got {alpha}.")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"regularized incomplete beta requires beta > 0; got {beta}.")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_front = (
        math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + alpha * math.log(x)
        + beta * math.log1p(-x)
    )
    front = math.exp(log_front)
    threshold = (alpha + 1.0) / (alpha + beta + 2.0)
    if x < threshold:
        return front * _beta_continued_fraction(alpha, beta, x) / alpha
    return 1.0 - front * _beta_continued_fraction(beta, alpha, 1.0 - x) / beta


def _inverse_regularized_incomplete_beta(
    probability: float,
    alpha: float,
    beta: float,
    *,
    iterations: int = 80,
) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0

    low = 0.0
    high = 1.0
    for _ in range(iterations):
        midpoint = 0.5 * (low + high)
        cdf_value = _regularized_incomplete_beta(alpha, beta, midpoint)
        if cdf_value < probability:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def _predictor_sigma_ladder(
    predictor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    schedule_label: str,
) -> torch.Tensor:
    sigmas_ladder = getattr(predictor, "sigmas", None)
    if sigmas_ladder is None:
        raise RuntimeError(f"predictor does not expose 'sigmas' needed for {schedule_label} scheduler")
    sigmas_ladder = torch.as_tensor(sigmas_ladder, device=device, dtype=dtype)
    if sigmas_ladder.ndim != 1:
        raise RuntimeError(
            f"predictor.sigmas for {schedule_label} scheduler must be 1D; got shape {tuple(sigmas_ladder.shape)}."
        )
    if sigmas_ladder.numel() < 2:
        raise RuntimeError(
            f"predictor.sigmas must expose at least two ladder entries for {schedule_label} scheduler."
        )
    if not bool(torch.all(torch.isfinite(sigmas_ladder))):
        raise RuntimeError(
            f"predictor.sigmas contains non-finite values; cannot build {schedule_label} schedule."
        )
    if bool(torch.any(sigmas_ladder[1:] < sigmas_ladder[:-1])):
        raise RuntimeError(
            f"predictor.sigmas must be monotonically non-decreasing (sigma_min -> sigma_max) for {schedule_label} scheduler."
        )
    return sigmas_ladder


def _beta_schedule(
    steps: int,
    predictor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    alpha: float = 0.6,
    beta: float = 0.6,
) -> torch.Tensor:
    total_steps = int(steps)
    if total_steps <= 0:
        raise ValueError("beta scheduler requires steps >= 1.")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"beta scheduler alpha must be finite and > 0; got {alpha}.")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"beta scheduler beta must be finite and > 0; got {beta}.")
    sigmas_ladder = _predictor_sigma_ladder(
        predictor,
        device=device,
        dtype=dtype,
        schedule_label="beta",
    )

    total_timesteps = int(sigmas_ladder.numel()) - 1
    picked_timesteps: list[int] = []
    last_timestep: int | None = None
    for step_index in range(total_steps):
        probability = 1.0 - (float(step_index) / float(total_steps))
        quantile = _inverse_regularized_incomplete_beta(probability, alpha, beta)
        if not math.isfinite(quantile):
            raise RuntimeError(
                f"Beta scheduler inverse CDF produced non-finite quantile at step {step_index}."
            )
        timestep = int(round(quantile * float(total_timesteps)))
        timestep = max(0, min(total_timesteps, timestep))
        if last_timestep is not None and timestep > last_timestep:
            raise RuntimeError(
                "beta scheduler produced a non-monotonic timestep ladder "
                f"(prev={last_timestep}, current={timestep}, step_index={step_index})."
            )
        picked_timesteps.append(timestep)
        last_timestep = timestep
    if len(picked_timesteps) != total_steps:
        raise RuntimeError(
            "beta scheduler failed to produce the requested non-terminal step count "
            f"(requested={total_steps}, produced={len(picked_timesteps)})."
        )
    index_tensor = torch.tensor(picked_timesteps, device=device, dtype=torch.long)
    sigmas = sigmas_ladder.index_select(0, index_tensor).to(device=device, dtype=dtype)
    if int(sigmas.numel()) != total_steps:
        raise RuntimeError(
            "beta scheduler picked sigma count does not match requested steps "
            f"(requested={total_steps}, produced={int(sigmas.numel())})."
        )
    schedule = _append_zero(sigmas, device=device, dtype=dtype)
    expected_total = total_steps + 1
    if int(schedule.numel()) != expected_total:
        raise RuntimeError(
            "beta scheduler failed terminal-zero schedule length invariant "
            f"(requested_non_terminal={total_steps}, expected_total={expected_total}, "
            f"produced_total={int(schedule.numel())})."
        )
    return schedule


def _linear_quadratic_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    threshold_noise: float = 0.025,
    linear_steps: int | None = None,
) -> torch.Tensor:
    if steps == 1:
        sigmas = torch.tensor([sigma_max], device=device, dtype=dtype)
        return _append_zero(sigmas, device=device, dtype=dtype)
    if not math.isfinite(threshold_noise):
        raise ValueError("threshold_noise must be finite for linear_quadratic schedule")
    if linear_steps is None:
        linear_steps = steps // 2
    linear_steps_int = int(linear_steps)
    if linear_steps_int <= 0:
        raise ValueError("linear_steps must be >= 1 for linear_quadratic schedule")
    if linear_steps_int >= steps:
        raise ValueError("linear_steps must be < steps for linear_quadratic schedule")
    quadratic_steps = steps - linear_steps_int
    linear_indices = torch.arange(linear_steps_int, device=device, dtype=dtype)
    linear = linear_indices * (float(threshold_noise) / float(linear_steps_int))
    threshold_noise_step_diff = float(linear_steps_int) - float(threshold_noise) * float(steps)
    quadratic_coef = threshold_noise_step_diff / (float(linear_steps_int) * float(quadratic_steps**2))
    linear_coef = float(threshold_noise) / float(linear_steps_int) - 2.0 * threshold_noise_step_diff / float(
        quadratic_steps**2
    )
    const = quadratic_coef * float(linear_steps_int**2)
    quadratic_indices = torch.arange(linear_steps_int, steps, device=device, dtype=dtype)
    quadratic = quadratic_coef * quadratic_indices.pow(2) + linear_coef * quadratic_indices + const
    sigma_schedule = 1.0 - torch.cat([linear, quadratic])
    sigmas = sigma_schedule * float(sigma_max)
    return _append_zero(sigmas, device=device, dtype=dtype)


def _kl_optimal_schedule(steps: int, sigma_min: float, sigma_max: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(steps, device=device, dtype=dtype) / max(float(steps - 1), 1.0)
    sigmas = torch.tan(idx * math.atan(sigma_min) + (1 - idx) * math.atan(sigma_max))
    return _append_zero(sigmas, device=device, dtype=dtype)


def _align_your_steps_schedule(kind: SchedulerName, steps: int, *, is_sdxl: bool, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Tables mirrored from the reference sd_schedulers tables; trailing zero will be appended separately
    if kind is SchedulerName.ALIGN_YOUR_STEPS:
        sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029] if is_sdxl else [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]
    elif kind is SchedulerName.ALIGN_YOUR_STEPS_GITS:
        sigmas = [14.615, 4.734, 2.567, 1.529, 0.987, 0.652, 0.418, 0.268, 0.179, 0.127, 0.029] if is_sdxl else [14.615, 4.617, 2.507, 1.236, 0.702, 0.402, 0.240, 0.156, 0.104, 0.094, 0.029]
    elif kind is SchedulerName.ALIGN_YOUR_STEPS_11:
        sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029] if is_sdxl else [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]
    else:  # ALIGN_YOUR_STEPS_32
        if is_sdxl:
            sigmas = [14.61500000000000000, 11.14916180000000000, 8.505221270000000000, 6.488271510000000000, 5.437074020000000000, 4.603986190000000000, 3.898547040000000000, 3.274074570000000000, 2.743965270000000000, 2.299686590000000000, 1.954485140000000000, 1.671087150000000000, 1.428781520000000000, 1.231810090000000000, 1.067896490000000000, 0.925794430000000000, 0.802908860000000000, 0.696601210000000000, 0.604369030000000000, 0.528525520000000000, 0.467733440000000000, 0.413933790000000000, 0.362581860000000000, 0.310085170000000000, 0.265189250000000000, 0.223264610000000000, 0.176538770000000000, 0.139591920000000000, 0.105873810000000000, 0.055193690000000000, 0.028773340000000000, 0.015000000000000000]
        else:
            sigmas = [14.61500000000000000, 11.23951352000000000, 8.643630810000000000, 6.647294240000000000, 5.572508620000000000, 4.716485460000000000, 3.991960650000000000, 3.519560900000000000, 3.134904660000000000, 2.792287880000000000, 2.487736280000000000, 2.216638650000000000, 1.975083510000000000, 1.779317200000000000, 1.614753350000000000, 1.465409530000000000, 1.314849000000000000, 1.166424970000000000, 1.034755470000000000, 0.915737440000000000, 0.807481690000000000, 0.712023610000000000, 0.621739000000000000, 0.530652020000000000, 0.452909600000000000, 0.374914550000000000, 0.274618190000000000, 0.201152900000000000, 0.141058730000000000, 0.066828810000000000, 0.031661210000000000, 0.015000000000000000]
    sigmas = torch.as_tensor(sigmas, device=device, dtype=dtype)
    if sigmas.numel() != steps:
        sigmas = torch.nn.functional.interpolate(sigmas.view(1, 1, -1), size=steps, mode="linear", align_corners=False).view(-1)
    return _append_zero(sigmas, device=device, dtype=dtype)


def _turbo_schedule(steps: int, predictor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    timesteps = torch.flip(torch.arange(1, steps + 1, device=device, dtype=dtype) * float(1000.0 / steps) - 1, (0,)).round().clamp(0, 999)
    sigmas = predictor.sigma(timesteps)
    return _append_zero(sigmas, device=device, dtype=dtype)


def build_sigma_schedule(
    scheduler_name: str,
    steps: int,
    *,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    dtype: torch.dtype,
    predictor: Optional[object] = None,
    flow_shift: float | None = None,
    is_sdxl: bool = False,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be >= 1")

    kind = SchedulerName.from_string(scheduler_name)

    if kind is SchedulerName.SIMPLE:
        if predictor is None:
            raise RuntimeError("predictor required for simple scheduler")
        return _simple_schedule_from_predictor(steps, predictor, device=device, dtype=dtype, flow_shift=flow_shift)
    if kind in (SchedulerName.KARRAS, SchedulerName.EULER_DISCRETE):
        return _karras_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    if kind is SchedulerName.EXPONENTIAL:
        return _exponential_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    if kind is SchedulerName.POLYEXPONENTIAL:
        return _polyexponential_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    if kind is SchedulerName.UNIFORM:
        if predictor is None:
            raise RuntimeError("predictor required for uniform scheduler")
        return _uniform_schedule_from_predictor(steps, predictor, device=device, dtype=dtype)
    if kind is SchedulerName.SGM_UNIFORM:
        if predictor is None:
            raise RuntimeError("predictor required for sgm_uniform scheduler")
        return _sgm_uniform_schedule(steps, predictor, device=device, dtype=dtype)
    if kind is SchedulerName.DDIM:
        if predictor is None:
            raise RuntimeError("predictor required for ddim scheduler")
        sigs = _ddim_uniform_schedule(steps, predictor, device=device, dtype=dtype)
        return sigs
    if kind is SchedulerName.DDIM_UNIFORM:
        if predictor is None:
            raise RuntimeError("predictor required for ddim_uniform scheduler")
        return _ddim_uniform_schedule(steps, predictor, device=device, dtype=dtype)
    if kind is SchedulerName.NORMAL:
        if predictor is None:
            raise RuntimeError("predictor required for normal scheduler")
        return _normal_schedule(steps, predictor, device=device, dtype=dtype, sgm=False)
    if kind is SchedulerName.BETA:
        if predictor is None:
            raise RuntimeError("predictor required for beta scheduler")
        return _beta_schedule(steps, predictor, device=device, dtype=dtype)
    if kind is SchedulerName.LINEAR_QUADRATIC:
        return _linear_quadratic_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    if kind is SchedulerName.KL_OPTIMAL:
        return _kl_optimal_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    if kind in {
        SchedulerName.ALIGN_YOUR_STEPS,
        SchedulerName.ALIGN_YOUR_STEPS_GITS,
        SchedulerName.ALIGN_YOUR_STEPS_11,
        SchedulerName.ALIGN_YOUR_STEPS_32,
    }:
        return _align_your_steps_schedule(kind, steps, is_sdxl=is_sdxl, device=device, dtype=dtype)
    if kind is SchedulerName.TURBO:
        if predictor is None:
            raise RuntimeError("predictor required for turbo scheduler")
        return _turbo_schedule(steps, predictor, device=device, dtype=dtype)

    raise ValueError(f"Unsupported scheduler '{scheduler_name}' after normalization")


__all__ = [
    "SchedulerName",
    "build_sigma_schedule",
]
