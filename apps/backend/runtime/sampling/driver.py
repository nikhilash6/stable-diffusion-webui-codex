"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampling driver (native-only) for diffusion runtimes.
Selects sampler implementations from specs, compiles conditioning, handles cancellation/precision fallback, and runs the sampling loop
while emitting timeline/diagnostic hooks (and optional global profiling sections via `CODEX_PROFILE`), including native ER-SDE stage updates,
    native `heun` / `heunpp2` / `lms` / `ddpm` / `ipndm` / `ipndm v` / `deis` / `res multistep*` / `restart` / `gradient estimation` / `gradient estimation cfg++` /
    `dpm++ 2m cfg++` / `sa-solver` / `sa-solver pece` / `seeds 2` / `seeds 3` / `dpm 2` / `dpm 2 ancestral` / `dpm++ 2s ancestral` / `dpm++ 2s ancestral cfg++` /
    `dpm fast` execution, dedicated `dpm++ sde` / `dpm++ 2m sde` / `dpm++ 2m sde heun` / `dpm++ 2m sde gpu` / `dpm++ 2m sde heun gpu` / `dpm++ 3m sde`,
and dedicated `uni-pc` / `uni-pc bh2`
multistep predictor/corrector handling, strict runtime option validation (`solver_type`, `max_stage`, `eta`, `s_noise`) and optional guidance policy wiring
(APG/rescale/trunc/renorm), emits explicit runtime telemetry for console block-progress activation state, and owns the
shared img2img denoise-step split between proportional base execution and internal fixed-step hires continuations while
reaffirming the processing-owned raw progress-owner token when sampling starts.

Symbols (top-level; keep in sync; no ghosts):
- `_SamplingCancelled` (exception): Raised when an in-flight sampling run is cancelled (checked via backend state).
- `_raise_if_cancelled` (function): Checks cancellation state and raises `_SamplingCancelled` when requested.
- `_PrecisionFallbackRequest` (exception): Signals the caller to retry sampling with a different precision policy.
- `_resolve_guidance_policy` (function): Resolves and validates optional guidance policy overrides (env + request extras) for APG/rescale/trunc/renorm.
- `SamplingBoundaryState` (dataclass): Opaque sampler-owned boundary carrier for exact same-latent mid-schedule resume.
- `SamplingResult` (dataclass): Sampler return object carrying final latents plus an optional captured boundary state.
- `CodexSampler` (class): Main sampler driver; builds `SamplingContext`, resolves sampler specs, runs the native sampler loop, and integrates
  memory-management/timeline diagnostics.
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_event, emit_backend_message, get_backend_logger

# tags: sampling, diagnostics

from dataclasses import dataclass
from typing import Any, Optional, Callable, List, Mapping
import math
import os

import torch

from apps.backend.core.philox import PhiloxGenerator
from apps.backend.core.rng import ImageRNG, NoiseSettings
from apps.backend.infra.config.env_flags import env_flag, env_int
from apps.backend.infra.config.bootstrap_env import get_bootstrap_env

from .inner_loop import sampling_function_inner, sampling_prepare, sampling_cleanup
from .block_progress import (
    BLOCK_PROGRESS_CALLBACK_KEY,
    RichBlockProgressController,
    validate_block_progress_payload,
)
from .condition import compile_conditions
from .context import SamplingContext, build_sampling_context
from .deis import build_deis_coefficients
from .interval_noise import compose_nested_interval_noises
from .log_snr import (
    ei_h_phi_2,
    ei_h_phi_1,
    half_log_snr_to_sigma,
    offset_first_sigma_for_snr,
    sigma_to_half_log_snr,
)
from .registry import get_sampler_spec
from .sa_solver import (
    SA_SOLVER_DEFAULT_CORRECTOR_ORDER,
    SA_SOLVER_DEFAULT_ETA,
    SA_SOLVER_DEFAULT_PREDICTOR_ORDER,
    SA_SOLVER_DEFAULT_SIMPLE_ORDER_2,
    SA_SOLVER_DEFAULT_S_NOISE,
    compute_stochastic_adams_b_coeffs,
    get_tau_interval_func,
)
from ..sampling_adapters.extra import RestartStepPlan, build_restart_step_plan
from ...core.state import state as backend_state
from apps.backend.engines.util.schedulers import SamplerKind
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.memory.smart_offload_invariants import enforce_smart_offload_pre_sampling_residency
from apps.backend.runtime.diagnostics.timeline import timeline
from apps.backend.runtime.diagnostics.profiler import profiler
from apps.backend.patchers.base import set_model_options_post_cfg_function

class _SamplingCancelled(Exception):
    """Signal that sampling was cancelled externally."""


def _raise_if_cancelled() -> None:
    if backend_state.should_stop:
        raise _SamplingCancelled("cancelled")



_MAX_LMS_ORDER = 4
_MAX_IPNDM_ORDER = 4


@dataclass(slots=True)
class SamplingBoundaryState:
    """Opaque exact-resume carrier for one fixed-step sampling boundary."""

    completed_steps: int
    total_steps: int
    sampler_name: str
    scheduler_name: str
    prediction_type: str | None
    engine_id: str
    full_sigmas: torch.Tensor
    latent: torch.Tensor
    old_denoised: torch.Tensor | None = None
    older_denoised: torch.Tensor | None = None
    old_denoised_d: torch.Tensor | None = None
    gradient_estimation_prev_d: torch.Tensor | None = None
    t_prev: float | None = None
    h_prev: float | None = None
    h_prev_2: float | None = None
    eps_history: tuple[torch.Tensor, ...] = ()
    res_multistep_old_sigma_down: float | None = None
    uni_pc_history_denoised: tuple[torch.Tensor, ...] = ()
    uni_pc_history_lambdas: tuple[float, ...] = ()
    sa_solver_pred_list: tuple[torch.Tensor, ...] = ()
    sa_solver_prev_noise: torch.Tensor | None = None
    sa_solver_prev_h: float = 0.0
    sa_solver_prev_tau_t: float = 0.0
    guidance_apg_momentum_buffer: torch.Tensor | None = None
    seeded_step_rng: ImageRNG | None = None


@dataclass(slots=True)
class SamplingResult:
    """Sampling return object with an optional captured boundary state."""

    samples: torch.Tensor
    boundary_state: SamplingBoundaryState | None = None


_EXACT_BOUNDARY_RESUME_SUPPORTED_SAMPLERS: frozenset[SamplerKind] = frozenset(
    {
        SamplerKind.EULER,
        SamplerKind.EULER_A,
        SamplerKind.EULER_CFG_PP,
        SamplerKind.EULER_A_CFG_PP,
        SamplerKind.HEUN,
        SamplerKind.HEUNPP2,
        SamplerKind.LMS,
        SamplerKind.DDIM,
        SamplerKind.DDPM,
        SamplerKind.DPM2M,
        SamplerKind.DPM2M_CFG_PP,
        SamplerKind.DPMPP_SDE,
        SamplerKind.DPM2S_ANCESTRAL,
        SamplerKind.DPM2S_ANCESTRAL_CFG_PP,
        SamplerKind.DPM3M_SDE,
        SamplerKind.DPM2,
        SamplerKind.DPM2_ANCESTRAL,
        SamplerKind.IPNDM,
        SamplerKind.IPNDM_V,
        SamplerKind.DEIS,
        SamplerKind.RES_MULTISTEP,
        SamplerKind.RES_MULTISTEP_CFG_PP,
        SamplerKind.RES_MULTISTEP_ANCESTRAL,
        SamplerKind.RES_MULTISTEP_ANCESTRAL_CFG_PP,
        SamplerKind.GRADIENT_ESTIMATION,
        SamplerKind.GRADIENT_ESTIMATION_CFG_PP,
        SamplerKind.SA_SOLVER,
        SamplerKind.SA_SOLVER_PECE,
        SamplerKind.SEEDS_2,
        SamplerKind.SEEDS_3,
        SamplerKind.UNI_PC,
        SamplerKind.UNI_PC_BH2,
        SamplerKind.ER_SDE,
    }
)

_EXACT_BOUNDARY_RESUME_UNSUPPORTED_REASONS: dict[SamplerKind, str] = {
    SamplerKind.DPM_FAST: (
        "Sampler 'dpm fast' is unsupported for exact top-level swap_model resume because its dedicated fast-solver path "
        "does not expose an honest mid-schedule boundary-state seam yet."
    ),
    SamplerKind.DPM_ADAPTIVE: (
        "Sampler 'dpm adaptive' is unsupported for exact top-level swap_model resume because adaptive execution does not "
        "have an honest fixed total-step boundary contract."
    ),
    SamplerKind.RESTART: (
        "Sampler 'restart' is unsupported for exact top-level swap_model resume because its renoise planner does not expose "
        "an honest mid-schedule boundary-state seam yet."
    ),
    SamplerKind.DPM2M_SDE: (
        "Sampler 'dpm++ 2m sde' is unsupported for exact top-level swap_model resume because its half-logSNR "
        "runtime state is not restored from the captured boundary yet."
    ),
    SamplerKind.DPM2M_SDE_HEUN: (
        "Sampler 'dpm++ 2m sde heun' is unsupported for exact top-level swap_model resume because its half-logSNR "
        "runtime state is not restored from the captured boundary yet."
    ),
    SamplerKind.DPM2M_SDE_GPU: (
        "Sampler 'dpm++ 2m sde gpu' is unsupported for exact top-level swap_model resume because its half-logSNR "
        "runtime state is not restored from the captured boundary yet."
    ),
    SamplerKind.DPM2M_SDE_HEUN_GPU: (
        "Sampler 'dpm++ 2m sde heun gpu' is unsupported for exact top-level swap_model resume because its half-logSNR "
        "runtime state is not restored from the captured boundary yet."
    ),
}

_SEEDED_STEP_RNG_SAMPLERS: frozenset[SamplerKind] = frozenset(
    {
        SamplerKind.EULER_A,
        SamplerKind.EULER_A_CFG_PP,
        SamplerKind.DDPM,
        SamplerKind.ER_SDE,
        SamplerKind.DPMPP_SDE,
        SamplerKind.DPM2M_SDE,
        SamplerKind.DPM2M_SDE_HEUN,
        SamplerKind.DPM2M_SDE_GPU,
        SamplerKind.DPM2M_SDE_HEUN_GPU,
        SamplerKind.DPM3M_SDE,
        SamplerKind.DPM2_ANCESTRAL,
        SamplerKind.DPM2S_ANCESTRAL,
        SamplerKind.DPM2S_ANCESTRAL_CFG_PP,
        SamplerKind.RES_MULTISTEP_ANCESTRAL,
        SamplerKind.RES_MULTISTEP_ANCESTRAL_CFG_PP,
        SamplerKind.SA_SOLVER,
        SamplerKind.SA_SOLVER_PECE,
        SamplerKind.SEEDS_2,
        SamplerKind.SEEDS_3,
    }
)

_IPNDM_MULTISTEP_COEFFS: dict[int, tuple[float, ...]] = {
    1: (1.0,),
    2: (3.0 / 2.0, -1.0 / 2.0),
    3: (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0),
    4: (55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0),
}


def _poly_mul(poly_a: list[float], poly_b: list[float]) -> list[float]:
    if not poly_a or not poly_b:
        raise RuntimeError("Polynomial multiplication requires non-empty coefficient vectors.")
    product = [0.0] * (len(poly_a) + len(poly_b) - 1)
    for index_a, value_a in enumerate(poly_a):
        for index_b, value_b in enumerate(poly_b):
            product[index_a + index_b] += value_a * value_b
    return product


def _integrate_polynomial(coefficients: list[float], start: float, end: float) -> float:
    if not math.isfinite(start) or not math.isfinite(end):
        raise RuntimeError(f"Polynomial integration bounds must be finite (start={start}, end={end}).")
    total = 0.0
    for power, coefficient in enumerate(coefficients):
        exponent = power + 1
        total += coefficient * ((end**exponent) - (start**exponent)) / float(exponent)
    return total


def _compute_lms_coefficients(sigma_nodes: list[float], sigma_next: float) -> list[float]:
    if not sigma_nodes:
        raise RuntimeError("LMS coefficient construction requires at least one sigma node.")
    if len(sigma_nodes) > _MAX_LMS_ORDER:
        raise RuntimeError(
            f"LMS coefficient construction supports at most {_MAX_LMS_ORDER} sigma nodes; got {len(sigma_nodes)}."
        )
    if not all(math.isfinite(node) for node in sigma_nodes):
        raise RuntimeError(f"LMS coefficient construction received non-finite sigma nodes: {sigma_nodes}")
    if not math.isfinite(sigma_next):
        raise RuntimeError(f"LMS coefficient construction received non-finite sigma_next={sigma_next}.")

    sigma_start = float(sigma_nodes[0])
    coefficients: list[float] = []
    for basis_index, sigma_basis in enumerate(sigma_nodes):
        denominator = 1.0
        basis_poly = [1.0]
        for other_index, sigma_other in enumerate(sigma_nodes):
            if other_index == basis_index:
                continue
            spacing = float(sigma_basis) - float(sigma_other)
            if abs(spacing) <= 1e-12:
                raise RuntimeError(
                    "LMS coefficient construction encountered duplicate sigma nodes "
                    f"(basis={sigma_basis}, other={sigma_other})."
                )
            denominator *= spacing
            basis_poly = _poly_mul(basis_poly, [-float(sigma_other), 1.0])
        normalized_basis = [value / denominator for value in basis_poly]
        coefficients.append(_integrate_polynomial(normalized_basis, sigma_start, float(sigma_next)))
    return coefficients


def _compute_ipndm_derivative(eps_history: list[torch.Tensor]) -> torch.Tensor:
    if not eps_history:
        raise RuntimeError("IPNDM derivative construction requires at least one epsilon history entry.")
    order = min(len(eps_history), _MAX_IPNDM_ORDER)
    coeffs = _IPNDM_MULTISTEP_COEFFS.get(order)
    if coeffs is None:
        raise RuntimeError(f"IPNDM derivative construction received unsupported order={order}.")
    history_terms = list(reversed(eps_history[-order:]))
    return sum(coeff * term for coeff, term in zip(coeffs, history_terms))


def _compute_ipndm_v_derivative(
    sigmas_run: torch.Tensor,
    local_step_index: int,
    eps_history: list[torch.Tensor],
) -> torch.Tensor:
    if not eps_history:
        raise RuntimeError("IPNDM-V derivative construction requires at least one epsilon history entry.")
    if local_step_index < 0 or local_step_index + 1 >= int(sigmas_run.numel()):
        raise RuntimeError(
            "IPNDM-V derivative construction received an out-of-range local_step_index "
            f"(local_step_index={local_step_index}, sigma_count={int(sigmas_run.numel())})."
        )

    def _step_delta(newer: float, older: float, *, label: str) -> float:
        if not math.isfinite(newer) or not math.isfinite(older):
            raise RuntimeError(
                f"IPNDM-V derivative construction requires finite sigma nodes for {label} "
                f"(newer={newer}, older={older})."
            )
        delta = newer - older
        if abs(delta) <= 1e-12:
            raise RuntimeError(
                f"IPNDM-V derivative construction received a zero-width interval for {label} "
                f"(newer={newer}, older={older})."
            )
        return delta

    order = min(len(eps_history), _MAX_IPNDM_ORDER)
    d_cur = eps_history[-1]
    if order == 1:
        return d_cur

    t_cur = float(sigmas_run[local_step_index])
    t_next = float(sigmas_run[local_step_index + 1])
    h_n = _step_delta(t_next, t_cur, label="h_n")
    t_prev_1 = float(sigmas_run[local_step_index - 1])
    h_n_1 = _step_delta(t_cur, t_prev_1, label="h_n_1")

    if order == 2:
        coeff1 = (2.0 + (h_n / h_n_1)) / 2.0
        coeff2 = -(h_n / h_n_1) / 2.0
        return coeff1 * d_cur + coeff2 * eps_history[-2]

    t_prev_2 = float(sigmas_run[local_step_index - 2])
    h_n_2 = _step_delta(t_prev_1, t_prev_2, label="h_n_2")
    temp1 = (1.0 - h_n / (3.0 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2.0
    coeff1 = (2.0 + (h_n / h_n_1)) / 2.0 + temp1
    coeff2 = -(h_n / h_n_1) / 2.0 - (1.0 + h_n_1 / h_n_2) * temp1
    coeff3 = temp1 * h_n_1 / h_n_2
    if order == 3:
        return coeff1 * d_cur + coeff2 * eps_history[-2] + coeff3 * eps_history[-3]

    t_prev_3 = float(sigmas_run[local_step_index - 3])
    h_n_3 = _step_delta(t_prev_2, t_prev_3, label="h_n_3")
    temp2 = (
        (1.0 - h_n / (3.0 * (h_n + h_n_1))) / 2.0
        + (1.0 - h_n / (2.0 * (h_n + h_n_1))) * h_n / (6.0 * (h_n + h_n_1 + h_n_2))
    ) * (
        h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)
    ) / (
        h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3)
    )
    coeff1 = (2.0 + (h_n / h_n_1)) / 2.0 + temp1 + temp2
    coeff2 = -(h_n / h_n_1) / 2.0 - (1.0 + h_n_1 / h_n_2) * temp1 - (
        1.0 + (h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3)))
    ) * temp2
    coeff3 = temp1 * h_n_1 / h_n_2 + (
        (h_n_1 / h_n_2)
        + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * (1.0 + h_n_2 / h_n_3)
    ) * temp2
    coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2
    return coeff1 * d_cur + coeff2 * eps_history[-2] + coeff3 * eps_history[-3] + coeff4 * eps_history[-4]


def _compute_ancestral_sigmas(sigma_from: float, sigma_to: float, *, eta: float = 1.0) -> tuple[float, float]:
    if not math.isfinite(sigma_from) or not math.isfinite(sigma_to):
        raise RuntimeError(
            "Ancestral sigma conversion requires finite sigma values "
            f"(sigma_from={sigma_from}, sigma_to={sigma_to})."
        )
    if sigma_from < 0.0 or sigma_to < 0.0:
        raise RuntimeError(
            "Ancestral sigma conversion requires non-negative sigma values "
            f"(sigma_from={sigma_from}, sigma_to={sigma_to})."
        )
    if eta < 0.0:
        raise RuntimeError(f"Ancestral sigma conversion requires eta >= 0; got {eta}.")
    if sigma_to == 0.0 or eta == 0.0:
        return float(sigma_to), 0.0
    if sigma_from <= 0.0:
        raise RuntimeError(
            "Ancestral sigma conversion requires sigma_from > 0 when sigma_to > 0 "
            f"(sigma_from={sigma_from}, sigma_to={sigma_to})."
        )

    variance_ratio = (sigma_to**2) * max((sigma_from**2) - (sigma_to**2), 0.0) / max(sigma_from**2, 1e-12)
    if variance_ratio < -1e-12:
        raise RuntimeError(
            "Ancestral sigma conversion produced negative variance ratio "
            f"(sigma_from={sigma_from}, sigma_to={sigma_to}, ratio={variance_ratio})."
        )
    sigma_up = min(float(sigma_to), float(eta) * math.sqrt(max(variance_ratio, 0.0)))
    sigma_down_sq = (sigma_to**2) - (sigma_up**2)
    if sigma_down_sq < -1e-12:
        raise RuntimeError(
            "Ancestral sigma conversion produced negative sigma_down^2 "
            f"(sigma_from={sigma_from}, sigma_to={sigma_to}, sigma_up={sigma_up})."
        )
    sigma_down = math.sqrt(max(sigma_down_sq, 0.0))
    return sigma_down, sigma_up


class _PrecisionFallbackRequest(Exception):
    """Internal control flow exception used to trigger a sampling retry."""


_GUIDANCE_POLICY_KEY = "codex_guidance_policy"
_GUIDANCE_STEP_INDEX_KEY = "codex_guidance_step_index"
_GUIDANCE_TOTAL_STEPS_KEY = "codex_guidance_total_steps"
_GUIDANCE_APG_MOMENTUM_BUFFER_KEY = "codex_guidance_apg_momentum_buffer"
_GUIDANCE_WARNED_SAMPLER_CFG_KEY = "codex_guidance_sampler_cfg_warned"

_GUIDANCE_ALLOWED_KEYS = {
    "apg_enabled",
    "apg_start_step",
    "apg_eta",
    "apg_momentum",
    "apg_norm_threshold",
    "apg_rescale",
    "guidance_rescale",
    "cfg_trunc_ratio",
    "renorm_cfg",
}

_MAX_DENOISE_REBUILD_STEPS = 10000
_ER_SDE_CONST_SNR_PERCENT_OFFSET = 1e-4
_SEEDS_2_DEFAULT_ETA = 1.0
_SEEDS_2_DEFAULT_S_NOISE = 1.0
_SEEDS_2_DEFAULT_R = 0.5
_SEEDS_3_DEFAULT_ETA = 1.0
_SEEDS_3_DEFAULT_S_NOISE = 1.0
_SEEDS_3_DEFAULT_R_1 = 1.0 / 3.0
_SEEDS_3_DEFAULT_R_2 = 2.0 / 3.0
_DPMPP_SDE_DEFAULT_ETA = 1.0
_DPMPP_SDE_DEFAULT_S_NOISE = 1.0
_DPMPP_SDE_DEFAULT_R = 0.5


def _read_env_text(name: str) -> str | None:
    value = get_bootstrap_env(name)
    if value is None:
        value = os.getenv(name)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _read_env_float(name: str, default: float) -> float:
    text = _read_env_text(name)
    if text is None:
        return float(default)
    try:
        value = float(text)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{name} must be a float; got: {text!r}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite; got: {text!r}")
    return value


def _read_env_bool(name: str, default: bool) -> bool:
    text = _read_env_text(name)
    if text is None:
        return bool(default)
    normalized = text.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean (1/0/true/false/yes/no/on/off); got: {text!r}")


def _read_env_nonnegative_int(name: str, default: int) -> int:
    text = _read_env_text(name)
    if text is None:
        return int(default)
    try:
        value = int(text)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{name} must be an integer >= 0; got: {text!r}") from exc
    if value < 0:
        raise RuntimeError(f"{name} must be >= 0; got: {text!r}")
    return value


def _read_env_optional_float(name: str) -> float | None:
    text = _read_env_text(name)
    if text is None:
        return None
    try:
        value = float(text)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{name} must be a float; got: {text!r}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite; got: {text!r}")
    return value


def _resolve_guidance_policy(processing: Any) -> dict[str, Any] | None:
    policy: dict[str, Any] = {
        "apg_enabled": _read_env_bool("CODEX_GUIDANCE_APG_ENABLED", default=False),
        "apg_start_step": _read_env_nonnegative_int("CODEX_GUIDANCE_APG_START_STEP", default=0),
        "apg_eta": _read_env_float("CODEX_GUIDANCE_APG_ETA", default=0.0),
        "apg_momentum": _read_env_float("CODEX_GUIDANCE_APG_MOMENTUM", default=0.0),
        "apg_norm_threshold": _read_env_float("CODEX_GUIDANCE_APG_NORM_THRESHOLD", default=15.0),
        "apg_rescale": _read_env_float("CODEX_GUIDANCE_APG_RESCALE", default=0.0),
        "guidance_rescale": _read_env_float("CODEX_GUIDANCE_RESCALE", default=0.0),
        "cfg_trunc_ratio": _read_env_optional_float("CODEX_GUIDANCE_CFG_TRUNC_RATIO"),
        "renorm_cfg": _read_env_float("CODEX_GUIDANCE_RENORM_CFG", default=0.0),
    }

    overrides = getattr(processing, "override_settings", {})
    if isinstance(overrides, Mapping):
        guidance_override = overrides.get("guidance")
        if guidance_override is not None:
            if not isinstance(guidance_override, Mapping):
                raise RuntimeError(
                    "override_settings.guidance must be an object when provided "
                    f"(got {type(guidance_override).__name__})."
                )
            unknown = sorted(str(key) for key in guidance_override.keys() if str(key) not in _GUIDANCE_ALLOWED_KEYS)
            if unknown:
                raise RuntimeError(
                    "Unexpected override_settings.guidance key(s): "
                    + ", ".join(unknown)
                )
            for key in _GUIDANCE_ALLOWED_KEYS:
                if key not in guidance_override:
                    continue
                raw_value = guidance_override[key]
                if key == "apg_enabled":
                    if not isinstance(raw_value, bool):
                        raise RuntimeError("override_settings.guidance.apg_enabled must be boolean.")
                    policy[key] = raw_value
                    continue
                if key == "apg_start_step":
                    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                        raise RuntimeError("override_settings.guidance.apg_start_step must be an integer >= 0.")
                    if isinstance(raw_value, float) and not raw_value.is_integer():
                        raise RuntimeError("override_settings.guidance.apg_start_step must be an integer >= 0.")
                    value_int = int(raw_value)
                    if value_int < 0:
                        raise RuntimeError("override_settings.guidance.apg_start_step must be >= 0.")
                    policy[key] = value_int
                    continue
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    raise RuntimeError(f"override_settings.guidance.{key} must be numeric.")
                value_float = float(raw_value)
                if not math.isfinite(value_float):
                    raise RuntimeError(f"override_settings.guidance.{key} must be finite.")
                policy[key] = value_float

    apg_enabled = bool(policy.get("apg_enabled", False))
    apg_start_step = int(policy.get("apg_start_step", 0) or 0)
    apg_eta = float(policy.get("apg_eta", 0.0) or 0.0)
    apg_momentum = float(policy.get("apg_momentum", 0.0) or 0.0)
    apg_norm_threshold = float(policy.get("apg_norm_threshold", 0.0) or 0.0)
    apg_rescale = float(policy.get("apg_rescale", 0.0) or 0.0)
    guidance_rescale = float(policy.get("guidance_rescale", 0.0) or 0.0)
    cfg_trunc_ratio = policy.get("cfg_trunc_ratio")
    renorm_cfg = float(policy.get("renorm_cfg", 0.0) or 0.0)

    if apg_start_step < 0:
        raise RuntimeError(f"Invalid guidance apg_start_step={apg_start_step}; expected >= 0.")
    if apg_momentum < 0.0 or apg_momentum >= 1.0:
        raise RuntimeError(f"Invalid guidance apg_momentum={apg_momentum}; expected range [0, 1).")
    if apg_norm_threshold < 0.0:
        raise RuntimeError(f"Invalid guidance apg_norm_threshold={apg_norm_threshold}; expected >= 0.")
    if guidance_rescale < 0.0 or guidance_rescale > 1.0:
        raise RuntimeError(f"Invalid guidance guidance_rescale={guidance_rescale}; expected range [0, 1].")
    if apg_rescale < 0.0 or apg_rescale > 1.0:
        raise RuntimeError(f"Invalid guidance apg_rescale={apg_rescale}; expected range [0, 1].")
    if renorm_cfg < 0.0:
        raise RuntimeError(f"Invalid guidance renorm_cfg={renorm_cfg}; expected >= 0.")
    if cfg_trunc_ratio is not None:
        cfg_trunc_ratio = float(cfg_trunc_ratio)
        if cfg_trunc_ratio < 0.0 or cfg_trunc_ratio > 1.0:
            raise RuntimeError(f"Invalid guidance cfg_trunc_ratio={cfg_trunc_ratio}; expected range [0, 1].")

    active = (
        apg_enabled
        or guidance_rescale > 0.0
        or apg_rescale > 0.0
        or renorm_cfg > 0.0
        or cfg_trunc_ratio is not None
    )
    if not active:
        return None

    return {
        "apg_enabled": apg_enabled,
        "apg_start_step": apg_start_step,
        "apg_eta": apg_eta,
        "apg_momentum": apg_momentum,
        "apg_norm_threshold": apg_norm_threshold,
        "apg_rescale": apg_rescale,
        "guidance_rescale": guidance_rescale,
        "cfg_trunc_ratio": cfg_trunc_ratio,
        "renorm_cfg": renorm_cfg,
    }

class CodexSampler:
    """Native sampler for diffusion runtimes with strict runtime contracts.

    - Uses the model's predictor to derive a sigma schedule from sigma_max→sigma_min.
    - Calls `sampling_function_inner` (CFG and condition assembly) each step.
    - Updates `backend_state` for progress reporting.
    - Implements k-diffusion anchored sampler families and dedicated native solver variants.
    """

    def __init__(self, sd_model: Any, *, algorithm: str | None = None) -> None:
        self.sd_model = sd_model
        self.algorithm = (algorithm or "euler a").strip().lower()
        self._logger_name = get_backend_logger(f"{__name__}.sampler").name
        self._log_enabled = env_flag("CODEX_LOG_SAMPLER", default=False)
        self._log_sigmas = env_flag("CODEX_LOG_SIGMAS", default=False)

    def _emit_event(self, event: str, /, **fields: object) -> None:
        emit_backend_event(event, logger=self._logger_name, **fields)

    @staticmethod
    def _compact_series(values: list[float]) -> str:
        if not values:
            return "none"
        return "/".join(f"{value:.6g}" for value in values)

    def _summarize_sigmas(self, sigmas: torch.Tensor, *, window: int = 6) -> str:
        try:
            values = [float(v) for v in sigmas.detach().cpu().tolist()]
        except Exception:
            return "<unavailable>"
        if len(values) <= window * 2:
            return ",".join(f"{v:.6g}" for v in values)
        head = ",".join(f"{v:.6g}" for v in values[:window])
        tail = ",".join(f"{v:.6g}" for v in values[-window:])
        return f"{head},...,{tail}"

    @staticmethod
    def _normalize_denoise_strength(denoise_strength: float | None) -> float | None:
        if denoise_strength is None:
            return None
        if isinstance(denoise_strength, bool):
            raise ValueError("denoise_strength must be a float in [0, 1]")
        try:
            value = float(denoise_strength)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"denoise_strength must be numeric; got {type(denoise_strength).__name__}") from exc
        if not math.isfinite(value):
            raise ValueError("denoise_strength must be finite")
        if value < 0.0 or value > 1.0:
            raise ValueError("denoise_strength must be in [0, 1]")
        return value

    @classmethod
    def uses_img2img_continuation(cls, denoise_strength: float | None) -> bool:
        normalized = cls._normalize_denoise_strength(denoise_strength)
        if normalized is None:
            return False
        return not math.isclose(normalized, 1.0, rel_tol=0.0, abs_tol=0.0)

    @staticmethod
    def _forge_img2img_denoise_fraction(denoise_strength: float) -> float:
        return min(float(denoise_strength), 0.999)

    @classmethod
    def _nonflow_partial_denoise_effective_steps(cls, *, steps: int, denoise_strength: float) -> int:
        t_enc = int(cls._forge_img2img_denoise_fraction(denoise_strength) * float(steps))
        return max(1, min(int(steps), t_enc + 1))

    @classmethod
    def _nonflow_partial_denoise_start_index(cls, *, steps: int, denoise_strength: float) -> int:
        effective_steps = cls._nonflow_partial_denoise_effective_steps(
            steps=int(steps),
            denoise_strength=float(denoise_strength),
        )
        return max(int(steps) - effective_steps, 0)

    @staticmethod
    def _flow_partial_denoise_start_index(*, steps: int, denoise_strength: float) -> int:
        init_timestep = min(float(steps) * float(denoise_strength), float(steps))
        return int(max(float(steps) - init_timestep, 0.0))

    @classmethod
    def _flow_partial_denoise_effective_steps(cls, *, steps: int, denoise_strength: float) -> int:
        return int(steps) - cls._flow_partial_denoise_start_index(steps=int(steps), denoise_strength=float(denoise_strength))

    @staticmethod
    def _discard_penultimate_sigma_ladder(sigmas: torch.Tensor) -> torch.Tensor:
        if sigmas.ndim != 1:
            raise RuntimeError(f"Discard-penultimate sigma ladder must be 1D; got shape={tuple(sigmas.shape)}")
        sigma_count = int(sigmas.numel())
        if sigma_count < 3:
            raise RuntimeError(f"Discard-penultimate samplers require at least 3 sigma entries; got {sigma_count}.")
        return torch.cat([sigmas[:-2], sigmas[-1:]])

    def _build_flow_partial_denoise_sigmas(
        self,
        *,
        active_context: SamplingContext,
        noise: torch.Tensor,
        steps: int,
        denoise_strength: float,
        discard_penultimate_sigma: bool = False,
    ) -> torch.Tensor:
        base_sigmas = active_context.sigmas.to(device=noise.device, dtype=torch.float32)
        if base_sigmas.ndim != 1:
            raise RuntimeError(f"Flow partial denoise sigma schedule must be 1D; got shape={tuple(base_sigmas.shape)}")
        flow_steps = int(steps)
        if discard_penultimate_sigma:
            base_sigmas = self._discard_penultimate_sigma_ladder(base_sigmas)
            flow_steps -= 1
        t_start = self._flow_partial_denoise_start_index(steps=int(flow_steps), denoise_strength=float(denoise_strength))
        effective_steps = int(flow_steps) - int(t_start)
        if effective_steps < 1:
            raise ValueError(
                "After adjusting the num_inference_steps by strength parameter: "
                f"{denoise_strength}, the number of pipeline steps is {effective_steps} which is < 1 "
                "and not appropriate for this pipeline."
            )
        tail = base_sigmas[int(t_start) :]
        expected_length = int(effective_steps) + 1
        if int(tail.numel()) != expected_length:
            raise RuntimeError(
                "Flow partial denoise sigma schedule length mismatch: "
                f"got={int(tail.numel())} expected={expected_length} "
                f"(steps={steps} denoise={denoise_strength} t_start={t_start})."
            )
        if self._log_enabled:
            self._emit_event(
                "sampling.denoise_schedule",
                steps=int(flow_steps),
                denoise=float(denoise_strength),
                mode="flow_trim",
                t_start=int(t_start),
                effective_steps=int(effective_steps),
                first=float(tail[0].item()),
                last=float(tail[-1].item()),
            )
        return tail

    def _build_fixed_step_partial_denoise_sigmas(
        self,
        *,
        processing: Any,
        model: Any,
        active_context: SamplingContext,
        noise: torch.Tensor,
        steps: int,
        denoise_strength: float,
    ) -> torch.Tensor:
        new_steps = int(float(steps) / float(denoise_strength))
        if new_steps < 1:
            raise RuntimeError(
                f"Partial denoise schedule rebuild produced invalid new_steps={new_steps} from steps={steps} denoise={denoise_strength}"
            )
        if new_steps > _MAX_DENOISE_REBUILD_STEPS:
            raise ValueError(
                f"denoise_strength={denoise_strength} expands schedule to new_steps={new_steps}, "
                f"above safety limit {_MAX_DENOISE_REBUILD_STEPS}"
            )
        denoise_context = build_sampling_context(
            self.sd_model,
            sampler_name=self.algorithm,
            scheduler_name=active_context.scheduler_name,
            steps=new_steps,
            noise_source=active_context.noise_settings.source.value,
            eta_noise_seed_delta=active_context.noise_settings.eta_noise_seed_delta,
            height=(int(getattr(processing, "height", 0) or 0) or None),
            width=(int(getattr(processing, "width", 0) or 0) or None),
            device=noise.device,
            dtype=noise.dtype,
            predictor=model,
            is_sdxl=bool(getattr(getattr(self.sd_model, "engine", None), "is_sdxl", False)),
        )
        denoise_sigmas = denoise_context.sigmas.to(device=noise.device, dtype=torch.float32)
        required = int(steps) + 1
        if denoise_sigmas.ndim != 1:
            raise RuntimeError(f"Partial denoise schedule must be 1D; got shape={tuple(denoise_sigmas.shape)}")
        if int(denoise_sigmas.numel()) < required:
            raise RuntimeError(
                f"Partial denoise schedule too short: got={int(denoise_sigmas.numel())} required={required} "
                f"(steps={steps} denoise={denoise_strength} new_steps={new_steps})"
            )
        tail = denoise_sigmas[-required:]
        if self._log_enabled:
            self._emit_event(
                "sampling.denoise_schedule",
                steps=int(steps),
                denoise=float(denoise_strength),
                mode="fixed_rebuild",
                new_steps=int(new_steps),
                selected=int(required),
                first=float(tail[0].item()),
                last=float(tail[-1].item()),
            )
        return tail

    def _rebind_unet_precision(self, dtype: torch.dtype) -> None:
        denoiser = self.sd_model.codex_objects.denoiser
        model = getattr(denoiser, "model", None)
        if model is None:
            return
        previous = getattr(model, "computation_dtype", None)
        if hasattr(model, "computation_dtype"):
            model.computation_dtype = dtype
        diffusion_model = getattr(model, "diffusion_model", None)
        if diffusion_model is not None:
            diffusion_model.to(dtype=dtype)
        emit_backend_message(
            "Diffusion core precision updated",
            logger=self._logger_name,
            previous=str(previous),
            next=str(dtype),
        )

    @staticmethod
    def _normalize_er_sde_solver_type(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("ER-SDE option 'solver_type' must be a string")
        normalized = value.strip().lower().replace("-", " ").replace("_", " ")
        solver_map = {
            "er sde": "er_sde",
            "reverse time sde": "reverse_time_sde",
            "ode": "ode",
        }
        solver_type = solver_map.get(normalized)
        if solver_type is None:
            raise ValueError(
                "ER-SDE option 'solver_type' must be one of: ER-SDE, Reverse-time SDE, ODE"
            )
        return solver_type

    @staticmethod
    def _resolve_er_sde_runtime_params(er_sde_options: Any) -> dict[str, Any]:
        allowed_keys = {"solver_type", "max_stage", "eta", "s_noise"}
        if er_sde_options is None:
            payload: dict[str, Any] = {}
        elif isinstance(er_sde_options, dict):
            raw_payload = dict(er_sde_options)
            unknown_keys = sorted(set(raw_payload.keys()) - allowed_keys)
            if unknown_keys:
                raise ValueError(f"Unexpected ER-SDE option key(s): {', '.join(unknown_keys)}")
            payload = {key: value for key, value in raw_payload.items() if value is not None}
        else:
            attr_keys: set[str] = set()
            if hasattr(er_sde_options, "__dict__"):
                attr_keys.update(str(key) for key in vars(er_sde_options).keys())
            slots_value = getattr(type(er_sde_options), "__slots__", ())
            if isinstance(slots_value, str):
                slots_iterable = (slots_value,)
            else:
                slots_iterable = tuple(slots_value) if isinstance(slots_value, (tuple, list)) else ()
            attr_keys.update(str(key) for key in slots_iterable)
            unknown_keys = sorted(
                key for key in attr_keys if key and not key.startswith("_") and key not in allowed_keys
            )
            if unknown_keys:
                raise ValueError(f"Unexpected ER-SDE option key(s): {', '.join(unknown_keys)}")
            payload = {}
            for key in allowed_keys:
                value = getattr(er_sde_options, key, None)
                if value is not None:
                    payload[key] = value

        solver_raw = payload.get("solver_type", "er_sde")
        solver_type = CodexSampler._normalize_er_sde_solver_type(solver_raw)

        max_stage_raw = payload.get("max_stage", 3)
        if isinstance(max_stage_raw, bool) or not isinstance(max_stage_raw, (int, float)):
            raise ValueError("ER-SDE option 'max_stage' must be an integer in [1, 3]")
        if isinstance(max_stage_raw, float) and not max_stage_raw.is_integer():
            raise ValueError("ER-SDE option 'max_stage' must be an integer in [1, 3]")
        max_stage = int(max_stage_raw)
        if max_stage < 1 or max_stage > 3:
            raise ValueError("ER-SDE option 'max_stage' must be in [1, 3]")

        eta_raw = payload.get("eta", 1.0)
        if isinstance(eta_raw, bool) or not isinstance(eta_raw, (int, float)):
            raise ValueError("ER-SDE option 'eta' must be numeric")
        eta = float(eta_raw)
        if not math.isfinite(eta):
            raise ValueError("ER-SDE option 'eta' must be finite")
        if eta < 0.0:
            raise ValueError("ER-SDE option 'eta' must be >= 0")

        s_noise_raw = payload.get("s_noise", 1.0)
        if isinstance(s_noise_raw, bool) or not isinstance(s_noise_raw, (int, float)):
            raise ValueError("ER-SDE option 's_noise' must be numeric")
        s_noise = float(s_noise_raw)
        if not math.isfinite(s_noise):
            raise ValueError("ER-SDE option 's_noise' must be finite")
        if s_noise < 0.0:
            raise ValueError("ER-SDE option 's_noise' must be >= 0")

        if solver_type == "ode" or (solver_type == "reverse_time_sde" and eta == 0.0):
            eta = 0.0
            s_noise = 0.0

        return {
            "solver_type": solver_type,
            "max_stage": max_stage,
            "eta": eta,
            "s_noise": s_noise,
        }

    @staticmethod
    def _er_sde_noise_scaler(
        values: torch.Tensor,
        *,
        solver_type: str,
        eta: float,
    ) -> torch.Tensor:
        if solver_type == "er_sde":
            return values * (torch.exp(values.pow(0.3)) + 10.0)
        return values.pow(eta + 1.0)

    @staticmethod
    def _compute_er_sde_lambdas(sigmas: torch.Tensor, *, prediction_type: str | None) -> torch.Tensor:
        if sigmas.ndim != 1:
            raise RuntimeError(f"ER-SDE expects a 1D sigma schedule, got shape={tuple(sigmas.shape)}")
        sigmas_fp32 = sigmas.to(dtype=torch.float32)
        if prediction_type == "const":
            if int(sigmas_fp32.numel()) > 1 and float(sigmas_fp32[0]) >= 1.0:
                sigmas_fp32 = sigmas_fp32.clone()
                sigmas_fp32[0] = float(_ER_SDE_CONST_SNR_PERCENT_OFFSET)
            sigma_safe = sigmas_fp32.clamp(min=1e-6, max=1.0 - 1e-6)
            half_log_snr = -torch.logit(sigma_safe)
        else:
            sigma_safe = sigmas_fp32.clamp(min=1e-12)
            half_log_snr = -torch.log(sigma_safe)
        return torch.exp(-half_log_snr)

    @staticmethod
    def _resolve_uni_pc_bh_coefficients(
        *,
        order: int,
        previous_rks: list[float],
        hh: float,
        variant: str,
    ) -> tuple[float, list[float] | None, list[float]]:
        if order <= 0:
            raise RuntimeError(f"UNI_PC coefficient solve requires order >= 1; got {order}.")
        if not math.isfinite(hh):
            raise RuntimeError(f"UNI_PC coefficient solve received non-finite hh={hh}.")
        if abs(hh) <= 1e-12:
            raise RuntimeError("UNI_PC coefficient solve cannot proceed with hh≈0.")
        if variant not in {"bh1", "bh2"}:
            raise RuntimeError(f"UNI_PC coefficient solve received unsupported variant={variant!r}.")
        if len(previous_rks) != max(0, order - 1):
            raise RuntimeError(
                "UNI_PC coefficient solve received inconsistent rk history "
                f"(order={order}, previous_rks={previous_rks})."
            )
        for rk in previous_rks:
            if not math.isfinite(rk):
                raise RuntimeError(f"UNI_PC coefficient solve received non-finite rk={rk}.")

        b_h = hh if variant == "bh1" else math.expm1(hh)
        if abs(b_h) <= 1e-12:
            raise RuntimeError("UNI_PC coefficient solve cannot proceed with B_h≈0.")

        if order == 1:
            return b_h, None, [0.5]

        rks = [*previous_rks, 1.0]
        h_phi_k = (math.expm1(hh) / hh) - 1.0
        factorial_i = 1.0
        rows: list[list[float]] = []
        rhs: list[float] = []
        for power in range(1, order + 1):
            rows.append([rk ** (power - 1) for rk in rks])
            rhs.append((h_phi_k * factorial_i) / b_h)
            factorial_i *= float(power + 1)
            h_phi_k = (h_phi_k / hh) - (1.0 / factorial_i)

        matrix = torch.tensor(rows, dtype=torch.float64)
        vector = torch.tensor(rhs, dtype=torch.float64)
        try:
            if order == 2:
                predictor = [0.5]
            else:
                predictor = torch.linalg.solve(matrix[:-1, :-1], vector[:-1]).tolist()
            corrector = torch.linalg.solve(matrix, vector).tolist()
        except RuntimeError as exc:
            raise RuntimeError(
                "UNI_PC coefficient solve failed "
                f"(variant={variant}, order={order}, hh={hh}, previous_rks={previous_rks})."
            ) from exc

        if not all(math.isfinite(value) for value in predictor):
            raise RuntimeError(
                "UNI_PC predictor coefficients became non-finite "
                f"(variant={variant}, order={order}, predictor={predictor})."
            )
        if not all(math.isfinite(value) for value in corrector):
            raise RuntimeError(
                "UNI_PC corrector coefficients became non-finite "
                f"(variant={variant}, order={order}, corrector={corrector})."
            )
        return b_h, predictor, corrector

    @staticmethod
    def _clone_noise_settings(settings: NoiseSettings) -> NoiseSettings:
        return NoiseSettings(
            source=settings.source,
            eta_noise_seed_delta=int(settings.eta_noise_seed_delta or 0),
            force_device=settings.force_device,
        )

    @staticmethod
    def _resolve_processing_seed_list(processing: Any, *, batch_size: int) -> list[int]:
        seed_values = list(getattr(processing, "all_seeds", []) or []) or list(getattr(processing, "seeds", []) or [])
        if not seed_values:
            seed_value = int(getattr(processing, "seed", -1))
            if seed_value < 0:
                raise RuntimeError(
                    "Deterministic sampler step noise requires explicit per-sample seeds; processing is missing seeds."
                )
            if batch_size != 1:
                raise RuntimeError(
                    "Deterministic sampler step noise requires per-sample seeds for batched runs; "
                    f"got batch_size={batch_size} with only processing.seed."
                )
            seed_values = [seed_value]
        normalized = [int(value) for value in seed_values]
        if len(normalized) != batch_size:
            raise RuntimeError(
                "Deterministic sampler step noise seed count mismatch: "
                f"got seeds={len(normalized)} batch_size={batch_size}."
            )
        return normalized

    def _build_seeded_step_rng(
        self,
        *,
        processing: Any,
        active_context: SamplingContext,
        noise: torch.Tensor,
    ) -> ImageRNG:
        batch_size = int(noise.shape[0])
        latent_shape = tuple(int(dim) for dim in noise.shape[1:])
        template_rng = getattr(processing, "rng", None)
        rng_target_device = noise.device
        if isinstance(template_rng, ImageRNG):
            template_shape = tuple(int(dim) for dim in template_rng.shape)
            if template_shape != latent_shape:
                raise RuntimeError(
                    "processing.rng shape mismatch for deterministic sampler step noise: "
                    f"rng_shape={template_shape} noise_shape={latent_shape}."
                )
            seeds = [int(seed) for seed in template_rng.seeds]
            if len(seeds) != batch_size:
                raise RuntimeError(
                    "processing.rng seed count mismatch for deterministic sampler step noise: "
                    f"rng_seeds={len(seeds)} batch_size={batch_size}."
                )
            subseeds = [int(seed) for seed in template_rng.subseeds]
            subseed_strength = float(template_rng.subseed_strength)
            seed_resize_from_h = int(template_rng.seed_resize_from_h)
            seed_resize_from_w = int(template_rng.seed_resize_from_w)
            settings = self._clone_noise_settings(template_rng.settings)
            rng_target_device = template_rng.device
        else:
            seeds = self._resolve_processing_seed_list(processing, batch_size=batch_size)
            subseeds = [int(seed) for seed in (getattr(processing, "all_subseeds", []) or []) or (getattr(processing, "subseeds", []) or [])]
            subseed_strength = float(getattr(processing, "subseed_strength", 0.0) or 0.0)
            seed_resize_from_h = int(getattr(processing, "seed_resize_from_h", 0) or 0)
            seed_resize_from_w = int(getattr(processing, "seed_resize_from_w", 0) or 0)
            settings = self._clone_noise_settings(active_context.noise_settings)

        # Clone the shared ImageRNG policy; `core.rng` drives determinism through
        # `torch.Generator(...).manual_seed(...)` / Philox and applies `eta_noise_seed_delta`
        # immediately after the initial latent noise draw.
        step_rng = ImageRNG(
            latent_shape,
            seeds,
            subseeds=subseeds,
            subseed_strength=subseed_strength,
            seed_resize_from_h=seed_resize_from_h,
            seed_resize_from_w=seed_resize_from_w,
            settings=settings,
            device=rng_target_device,
        )
        initial_noise = step_rng.next()
        if tuple(initial_noise.shape) != tuple(noise.shape):
            raise RuntimeError(
                "Deterministic sampler step noise bootstrap shape mismatch: "
                f"bootstrap={tuple(initial_noise.shape)} noise={tuple(noise.shape)}."
            )
        return step_rng

    @staticmethod
    def _next_seeded_step_noise(step_rng: ImageRNG, reference: torch.Tensor) -> torch.Tensor:
        sampled = step_rng.next()
        if tuple(sampled.shape) != tuple(reference.shape):
            raise RuntimeError(
                "Deterministic sampler step noise shape mismatch: "
                f"sampled={tuple(sampled.shape)} reference={tuple(reference.shape)}."
            )
        return sampled.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _clone_boundary_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value.detach().clone()

    @classmethod
    def _clone_image_rng(cls, value: ImageRNG | None) -> ImageRNG | None:
        if value is None:
            return None

        cloned = ImageRNG(
            tuple(int(dim) for dim in value.shape),
            [int(seed) for seed in value.seeds],
            subseeds=[int(seed) for seed in value.subseeds],
            subseed_strength=float(value.subseed_strength),
            seed_resize_from_h=int(value.seed_resize_from_h),
            seed_resize_from_w=int(value.seed_resize_from_w),
            settings=cls._clone_noise_settings(value.settings),
            device=value.device,
        )
        cloned._is_first = bool(value._is_first)

        if len(cloned._generators) != len(value._generators):
            raise RuntimeError(
                "ImageRNG generator count mismatch while cloning seeded-step RNG state: "
                f"cloned={len(cloned._generators)} source={len(value._generators)}."
            )

        restored_generators: list[PhiloxGenerator | torch.Generator] = []
        for source_generator, cloned_generator in zip(value._generators, cloned._generators, strict=True):
            if isinstance(source_generator, PhiloxGenerator):
                restored_generators.append(
                    PhiloxGenerator(
                        seed=int(source_generator.seed),
                        offset=int(source_generator.offset),
                    )
                )
                continue

            if isinstance(cloned_generator, PhiloxGenerator):
                raise RuntimeError("ImageRNG clone produced incompatible generator types during state restore.")

            cloned_generator.set_state(source_generator.get_state().clone())
            restored_generators.append(cloned_generator)

        cloned._generators = restored_generators
        return cloned

    @classmethod
    def _clone_boundary_tensor_tuple(cls, values: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(cls._clone_boundary_tensor(value) for value in values if value is not None)

    def _ensure_exact_boundary_resume_supported(self, sampler_kind: SamplerKind) -> None:
        reason = _EXACT_BOUNDARY_RESUME_UNSUPPORTED_REASONS.get(sampler_kind)
        if reason is not None:
            raise RuntimeError(reason)
        if sampler_kind not in _EXACT_BOUNDARY_RESUME_SUPPORTED_SAMPLERS:
            raise RuntimeError(
                f"Sampler '{sampler_kind.value}' is unsupported for exact top-level swap_model resume "
                "until its boundary-state continuation is implemented explicitly."
            )

    def _build_sampling_boundary_state(
        self,
        *,
        processing: Any,
        active_context: SamplingContext,
        steps: int,
        current_step: int,
        prediction_type: str | None,
        sigmas: torch.Tensor,
        latent: torch.Tensor,
        old_denoised: torch.Tensor | None,
        older_denoised: torch.Tensor | None,
        old_denoised_d: torch.Tensor | None,
        gradient_estimation_prev_d: torch.Tensor | None,
        t_prev: float | None,
        h_prev: float | None,
        h_prev_2: float | None,
        eps_history: list[torch.Tensor],
        res_multistep_old_sigma_down: float | None,
        uni_pc_history_denoised: list[torch.Tensor],
        uni_pc_history_lambdas: list[float],
        sa_solver_pred_list: list[torch.Tensor],
        sa_solver_prev_noise: torch.Tensor | None,
        sa_solver_prev_h: float,
        sa_solver_prev_tau_t: float,
        guidance_apg_momentum_buffer: torch.Tensor | None,
        seeded_step_rng: ImageRNG | None,
    ) -> SamplingBoundaryState:
        engine_id = str(getattr(getattr(processing, "sd_model", None), "engine_id", "") or "")
        return SamplingBoundaryState(
            completed_steps=int(current_step),
            total_steps=int(steps),
            sampler_name=str(self.algorithm),
            scheduler_name=str(active_context.scheduler_name),
            prediction_type=prediction_type,
            engine_id=engine_id,
            full_sigmas=sigmas.detach().to(device="cpu", dtype=torch.float32).clone(),
            latent=self._clone_boundary_tensor(latent),
            old_denoised=self._clone_boundary_tensor(old_denoised),
            older_denoised=self._clone_boundary_tensor(older_denoised),
            old_denoised_d=self._clone_boundary_tensor(old_denoised_d),
            gradient_estimation_prev_d=self._clone_boundary_tensor(gradient_estimation_prev_d),
            t_prev=t_prev,
            h_prev=h_prev,
            h_prev_2=h_prev_2,
            eps_history=self._clone_boundary_tensor_tuple(eps_history),
            res_multistep_old_sigma_down=res_multistep_old_sigma_down,
            uni_pc_history_denoised=self._clone_boundary_tensor_tuple(uni_pc_history_denoised),
            uni_pc_history_lambdas=tuple(float(value) for value in uni_pc_history_lambdas),
            sa_solver_pred_list=self._clone_boundary_tensor_tuple(sa_solver_pred_list),
            sa_solver_prev_noise=self._clone_boundary_tensor(sa_solver_prev_noise),
            sa_solver_prev_h=float(sa_solver_prev_h),
            sa_solver_prev_tau_t=float(sa_solver_prev_tau_t),
            guidance_apg_momentum_buffer=self._clone_boundary_tensor(guidance_apg_momentum_buffer),
            seeded_step_rng=self._clone_image_rng(seeded_step_rng),
        )

    def _validate_resume_boundary_state(
        self,
        *,
        boundary_state: SamplingBoundaryState,
        noise: torch.Tensor,
        sigmas: torch.Tensor,
        active_context: SamplingContext,
        steps: int,
        start_at_step: int | None,
        prediction_type: str | None,
    ) -> None:
        if int(boundary_state.total_steps) != int(steps):
            raise RuntimeError(
                "resume_boundary_state total-step mismatch: "
                f"state_total_steps={boundary_state.total_steps} runtime_total_steps={steps}."
            )
        if int(boundary_state.completed_steps) < 1 or int(boundary_state.completed_steps) >= int(steps):
            raise RuntimeError(
                "resume_boundary_state completed_steps is outside the valid pointer range: "
                f"completed_steps={boundary_state.completed_steps} total_steps={steps}."
            )
        if start_at_step is not None and int(start_at_step) != int(boundary_state.completed_steps):
            raise RuntimeError(
                "resume_boundary_state/start_at_step mismatch: "
                f"resume_boundary_state.completed_steps={boundary_state.completed_steps} start_at_step={start_at_step}."
            )
        if str(boundary_state.sampler_name) != str(self.algorithm):
            raise RuntimeError(
                "resume_boundary_state sampler mismatch: "
                f"state_sampler={boundary_state.sampler_name!r} runtime_sampler={self.algorithm!r}."
            )
        if str(boundary_state.scheduler_name) != str(active_context.scheduler_name):
            raise RuntimeError(
                "resume_boundary_state scheduler mismatch: "
                f"state_scheduler={boundary_state.scheduler_name!r} runtime_scheduler={active_context.scheduler_name!r}."
            )
        if boundary_state.prediction_type != prediction_type:
            raise RuntimeError(
                "resume_boundary_state prediction-type mismatch: "
                f"state_prediction={boundary_state.prediction_type!r} runtime_prediction={prediction_type!r}."
            )
        expected_shape = tuple(int(dim) for dim in noise.shape)
        state_shape = tuple(int(dim) for dim in boundary_state.latent.shape)
        if state_shape != expected_shape:
            raise RuntimeError(
                "resume_boundary_state latent shape mismatch: "
                f"state_shape={state_shape} runtime_shape={expected_shape}."
            )
        runtime_sigmas = sigmas.detach().to(device="cpu", dtype=torch.float32)
        if tuple(boundary_state.full_sigmas.shape) != tuple(runtime_sigmas.shape):
            raise RuntimeError(
                "resume_boundary_state sigma schedule shape mismatch: "
                f"state_shape={tuple(boundary_state.full_sigmas.shape)} runtime_shape={tuple(runtime_sigmas.shape)}."
            )
        if not torch.allclose(boundary_state.full_sigmas, runtime_sigmas, rtol=1e-6, atol=1e-6):
            raise RuntimeError(
                "resume_boundary_state sigma schedule mismatch: exact top-level swap_model resume requires "
                "identical full schedules across both same-family engines."
            )

    @torch.no_grad()
    def sample_result(
        self,
        processing: Any,
        noise: torch.Tensor,
        cond: Any,
        uncond: Optional[Any],
        image_conditioning: Optional[torch.Tensor] = None,
        *,
        init_latent: Optional[torch.Tensor] = None,
        resume_boundary_state: SamplingBoundaryState | None = None,
        capture_boundary_state_at_step: int | None = None,
        start_at_step: int | None = None,
        denoise_strength: float | None = None,
        img2img_fix_steps: bool = False,
        pre_denoiser_hook: Optional[Callable[[torch.Tensor, torch.Tensor, int, int | None], torch.Tensor]] = None,
        post_denoiser_hook: Optional[Callable[[torch.Tensor, torch.Tensor, int, int | None], torch.Tensor]] = None,
        preview_callback: Optional[Callable[[torch.Tensor, int, int | None], None]] = None,
        post_step_hook: Optional[Callable[[torch.Tensor, int, int | None], None]] = None,
        post_sample_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        context: SamplingContext | None = None,
        er_sde_options: Any = None,
    ) -> SamplingResult:
        return self._sample_impl(
            processing,
            noise,
            cond,
            uncond,
            image_conditioning=image_conditioning,
            init_latent=init_latent,
            resume_boundary_state=resume_boundary_state,
            capture_boundary_state_at_step=capture_boundary_state_at_step,
            start_at_step=start_at_step,
            denoise_strength=denoise_strength,
            img2img_fix_steps=img2img_fix_steps,
            pre_denoiser_hook=pre_denoiser_hook,
            post_denoiser_hook=post_denoiser_hook,
            preview_callback=preview_callback,
            post_step_hook=post_step_hook,
            post_sample_hook=post_sample_hook,
            context=context,
            er_sde_options=er_sde_options,
        )

    @torch.no_grad()
    def sample(
        self,
        processing: Any,
        noise: torch.Tensor,
        cond: Any,
        uncond: Optional[Any],
        image_conditioning: Optional[torch.Tensor] = None,
        *,
        init_latent: Optional[torch.Tensor] = None,
        start_at_step: int | None = None,
        denoise_strength: float | None = None,
        img2img_fix_steps: bool = False,
        pre_denoiser_hook: Optional[Callable[[torch.Tensor, torch.Tensor, int, int | None], torch.Tensor]] = None,
        post_denoiser_hook: Optional[Callable[[torch.Tensor, torch.Tensor, int, int | None], torch.Tensor]] = None,
        preview_callback: Optional[Callable[[torch.Tensor, int, int | None], None]] = None,
        post_step_hook: Optional[Callable[[torch.Tensor, int, int | None], None]] = None,
        post_sample_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        context: SamplingContext | None = None,
        er_sde_options: Any = None,
    ) -> torch.Tensor:
        return self._sample_impl(
            processing,
            noise,
            cond,
            uncond,
            image_conditioning=image_conditioning,
            init_latent=init_latent,
            start_at_step=start_at_step,
            denoise_strength=denoise_strength,
            img2img_fix_steps=img2img_fix_steps,
            pre_denoiser_hook=pre_denoiser_hook,
            post_denoiser_hook=post_denoiser_hook,
            preview_callback=preview_callback,
            post_step_hook=post_step_hook,
            post_sample_hook=post_sample_hook,
            context=context,
            er_sde_options=er_sde_options,
        ).samples

    def _sample_impl(
        self,
        processing: Any,
        noise: torch.Tensor,
        cond: Any,
        uncond: Optional[Any],
        image_conditioning: Optional[torch.Tensor] = None,
        *,
        init_latent: Optional[torch.Tensor] = None,
        resume_boundary_state: SamplingBoundaryState | None = None,
        capture_boundary_state_at_step: int | None = None,
        start_at_step: int | None = None,
        denoise_strength: float | None = None,
        img2img_fix_steps: bool = False,
        pre_denoiser_hook: Optional[Callable[[torch.Tensor, torch.Tensor, int, int | None], torch.Tensor]] = None,
        post_denoiser_hook: Optional[Callable[[torch.Tensor, torch.Tensor, int, int | None], torch.Tensor]] = None,
        preview_callback: Optional[Callable[[torch.Tensor, int, int | None], None]] = None,
        post_step_hook: Optional[Callable[[torch.Tensor, int, int | None], None]] = None,
        post_sample_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        context: SamplingContext | None = None,
        er_sde_options: Any = None,
    ) -> SamplingResult:
        base_noise = noise.detach()
        base_context = context
        warned_full_preview = False

        spec = get_sampler_spec(self.algorithm)
        captured_boundary_state: SamplingBoundaryState | None = None

        while True:
            denoiser = self.sd_model.codex_objects.denoiser
            model = denoiser.model

            steps = int(getattr(processing, "steps", 20))
            cfg_scale = float(getattr(processing, "cfg_scale", 7.0))
            
            # Flux Dev uses distilled CFG via guidance embedding - disable traditional CFG
            # to avoid double scaling which corrupts the output (sand texture artifact)
            if hasattr(self.sd_model, "use_distilled_cfg_scale") and self.sd_model.use_distilled_cfg_scale:
                if cfg_scale != 1.0:
                    emit_backend_message(
                        "[flux] Distilled CFG active: forcing cfg_scale=1.0",
                        logger=self._logger_name,
                        previous_cfg_scale=cfg_scale,
                    )
                cfg_scale = 1.0

            if steps <= 0:
                raise ValueError("steps must be >= 1")
            if noise.ndim != 4:
                raise ValueError(f"noise must be BCHW; got shape={tuple(noise.shape)}")

            target_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE)
            noise = base_noise.to(dtype=target_dtype)

            if resume_boundary_state is not None and init_latent is not None:
                raise RuntimeError("resume_boundary_state and init_latent are mutually exclusive.")
            if resume_boundary_state is not None and capture_boundary_state_at_step is not None:
                raise RuntimeError("resume_boundary_state and capture_boundary_state_at_step are mutually exclusive.")
            if init_latent is not None and init_latent.dtype != noise.dtype:
                init_latent = init_latent.to(dtype=noise.dtype)
            normalized_denoise = self._normalize_denoise_strength(denoise_strength)
            exact_boundary_resume_requested = (
                resume_boundary_state is not None or capture_boundary_state_at_step is not None
            )
            if exact_boundary_resume_requested:
                if bool(img2img_fix_steps):
                    raise RuntimeError(
                        "Exact top-level swap_model boundary capture/resume is incompatible with img2img fixed-step continuation."
                    )
                if normalized_denoise is not None and not math.isclose(float(normalized_denoise), 1.0):
                    raise RuntimeError(
                        "Exact top-level swap_model boundary capture/resume requires full-strength denoise (1.0)."
                    )
                if capture_boundary_state_at_step is not None:
                    capture_step = int(capture_boundary_state_at_step)
                    if capture_step < 1 or capture_step >= steps:
                        raise RuntimeError(
                            "capture_boundary_state_at_step must be inside the original total-step pointer range "
                            f"[1, {steps - 1}] (got {capture_step})."
                        )
                    if int(start_at_step or 0) != 0:
                        raise RuntimeError(
                            "capture_boundary_state_at_step must run on the original full schedule; do not combine it "
                            f"with start_at_step={start_at_step}."
                        )

            block_progress_controller: RichBlockProgressController | None = None
            retry = False
            prepared = False
            state_started = False
            active_context = base_context
            guidance_policy: dict[str, Any] | None = None

            try:
                allow_vae_resident = False
                preview_interval = 0
                res_multistep_cfg_pp_capture: torch.Tensor | None = None
                res_multistep_cfg_pp_saved_post_cfg_functions: list[Any] | None = None
                res_multistep_cfg_pp_disable_cfg1_missing = object()
                res_multistep_cfg_pp_saved_disable_cfg1: Any = res_multistep_cfg_pp_disable_cfg1_missing
                gradient_estimation_cfg_pp_capture: torch.Tensor | None = None
                gradient_estimation_cfg_pp_saved_post_cfg_functions: list[Any] | None = None
                gradient_estimation_cfg_pp_disable_cfg1_missing = object()
                gradient_estimation_cfg_pp_saved_disable_cfg1: Any = gradient_estimation_cfg_pp_disable_cfg1_missing
                euler_cfg_pp_capture: torch.Tensor | None = None
                euler_cfg_pp_saved_post_cfg_functions: list[Any] | None = None
                euler_cfg_pp_disable_cfg1_missing = object()
                euler_cfg_pp_saved_disable_cfg1: Any = euler_cfg_pp_disable_cfg1_missing
                dpm2m_cfg_pp_capture: torch.Tensor | None = None
                dpm2m_cfg_pp_saved_post_cfg_functions: list[Any] | None = None
                dpm2m_cfg_pp_disable_cfg1_missing = object()
                dpm2m_cfg_pp_saved_disable_cfg1: Any = dpm2m_cfg_pp_disable_cfg1_missing
                dpm2s_ancestral_cfg_pp_capture: torch.Tensor | None = None
                dpm2s_ancestral_cfg_pp_saved_post_cfg_functions: list[Any] | None = None
                dpm2s_ancestral_cfg_pp_disable_cfg1_missing = object()
                dpm2s_ancestral_cfg_pp_saved_disable_cfg1: Any = dpm2s_ancestral_cfg_pp_disable_cfg1_missing
                try:
                    if base_context is not None:
                        preview_interval = max(0, int(getattr(base_context, "preview_interval", 0) or 0))
                    else:
                        from apps.backend.runtime.live_preview import preview_interval_steps

                        preview_interval = preview_interval_steps(default=0)
                except Exception:
                    preview_interval = 0

                if preview_callback is not None and preview_interval > 0:
                    try:
                        from apps.backend.runtime.live_preview import LivePreviewMethod, live_preview_method

                        method = live_preview_method(default=LivePreviewMethod.FULL)
                        allow_vae_resident = method == LivePreviewMethod.FULL
                        if allow_vae_resident and not warned_full_preview:
                            emit_backend_message(
                                "Live preview FULL uses VAE decode during sampling. This increases VRAM usage and can reduce generation performance; prefer 'Approx cheap' when possible.",
                                logger=self._logger_name,
                                level="WARNING",
                            )
                            warned_full_preview = True
                    except Exception:
                        allow_vae_resident = False

                enforce_smart_offload_pre_sampling_residency(
                    self.sd_model,
                    stage="sampling.prepare",
                    allow_vae_resident=allow_vae_resident,
                )

                sampling_prepare(denoiser, noise)
                prepared = True

                discard_penultimate_sigma = spec.kind in {
                    SamplerKind.DPM2,
                    SamplerKind.DPM2_ANCESTRAL,
                    SamplerKind.UNI_PC,
                    SamplerKind.UNI_PC_BH2,
                }
                requested_steps = int(steps)
                schedule_steps = requested_steps + 1 if discard_penultimate_sigma else requested_steps

                scheduler_name = getattr(processing, "scheduler", None)
                if scheduler_name in (None, ""):
                    scheduler_name = spec.default_scheduler
                if not isinstance(scheduler_name, str) or not scheduler_name:
                    raise ValueError("Scheduler name must be a non-empty string")
                if not spec.is_supported_scheduler(scheduler_name):
                    raise ValueError(
                        f"Scheduler '{scheduler_name}' is not supported by sampler '{spec.name}'. "
                        f"Allowed: {sorted(spec.allowed_schedulers)}"
                    )
                if active_context is None:
                    active_context = build_sampling_context(
                        self.sd_model,
                        sampler_name=self.algorithm,
                        scheduler_name=scheduler_name,
                        steps=schedule_steps,
                        noise_source=getattr(processing, "noise_source", None),
                        eta_noise_seed_delta=int(getattr(processing, "eta_noise_seed_delta", 0) or 0),
                        height=(int(getattr(processing, "height", 0) or 0) or None),
                        width=(int(getattr(processing, "width", 0) or 0) or None),
                        device=noise.device,
                        dtype=noise.dtype,
                        predictor=model,
                        is_sdxl=bool(getattr(getattr(self.sd_model, "engine", None), "is_sdxl", False)),
                    )

                # Keep sigma ladder in fp32 for numeric stability; casting to bf16/fp16
                # quantizes the schedule and can produce severe quality regressions.
                schedule_steps = int(active_context.steps)
                flow_partial_denoise = (
                    normalized_denoise is not None
                    and self.uses_img2img_continuation(normalized_denoise)
                    and str(getattr(active_context, "prediction_type", "") or "").lower() == "const"
                )
                fixed_step_partial_denoise = (
                    normalized_denoise is not None
                    and bool(img2img_fix_steps)
                    and self.uses_img2img_continuation(normalized_denoise)
                    and not flow_partial_denoise
                )
                proportional_partial_denoise = (
                    normalized_denoise is not None
                    and not bool(img2img_fix_steps)
                    and not flow_partial_denoise
                )
                if normalized_denoise is None or proportional_partial_denoise:
                    sigmas = active_context.sigmas.to(device=noise.device, dtype=torch.float32)
                    if proportional_partial_denoise and self._log_enabled:
                        proportional_steps = self._nonflow_partial_denoise_effective_steps(
                            steps=int(requested_steps),
                            denoise_strength=float(normalized_denoise),
                        )
                        self._emit_event(
                            "sampling.denoise_schedule",
                            steps=int(requested_steps),
                            denoise=float(normalized_denoise),
                            mode="nonflow_proportional",
                            selected=int(proportional_steps + 1),
                        )
                elif fixed_step_partial_denoise:
                    sigmas = self._build_fixed_step_partial_denoise_sigmas(
                        processing=processing,
                        model=model,
                        active_context=active_context,
                        noise=noise,
                        steps=schedule_steps,
                        denoise_strength=normalized_denoise,
                    )
                else:
                    sigmas = self._build_flow_partial_denoise_sigmas(
                        active_context=active_context,
                        noise=noise,
                        steps=schedule_steps,
                        denoise_strength=normalized_denoise,
                        discard_penultimate_sigma=discard_penultimate_sigma,
                    )

                if sigmas.ndim != 1 or int(sigmas.numel()) < 2:
                    raise RuntimeError(f"sigma schedule must be 1D with at least 2 entries; got shape={tuple(sigmas.shape)}")
                sigma_count = int(sigmas.numel())
                if sigma_count > schedule_steps + 1:
                    raise RuntimeError(
                            "sigma schedule length unexpectedly exceeds requested build length: "
                            f"got {sigma_count}, max_expected {schedule_steps + 1} (schedule_steps={schedule_steps})."
                        )
                if discard_penultimate_sigma and not flow_partial_denoise:
                    sigmas = self._discard_penultimate_sigma_ladder(sigmas)
                sigma_count = int(sigmas.numel())
                total_schedule_steps = sigma_count - 1
                if total_schedule_steps <= 0:
                    raise RuntimeError(
                        f"sigma schedule must expose at least one denoise step; got sigma_count={sigma_count}."
                    )
                expected_total_schedule_steps = int(requested_steps)
                if flow_partial_denoise:
                    expected_total_schedule_steps = self._flow_partial_denoise_effective_steps(
                        steps=int(schedule_steps) - (1 if discard_penultimate_sigma else 0),
                        denoise_strength=float(normalized_denoise),
                    )
                ddpm_beta_dedup = (
                    active_context.sampler_kind is SamplerKind.DDPM and active_context.scheduler_name == "beta"
                )
                if total_schedule_steps != expected_total_schedule_steps:
                    if not ddpm_beta_dedup:
                        raise RuntimeError(
                            "post-adjust sigma schedule length mismatch: "
                            f"got {sigma_count}, expected {expected_total_schedule_steps + 1} "
                            f"(requested_steps={requested_steps} effective_expected={expected_total_schedule_steps})."
                        )
                    if total_schedule_steps > expected_total_schedule_steps:
                        raise RuntimeError(
                            "beta scheduler produced more steps than requested for ddpm: "
                            f"requested_steps={expected_total_schedule_steps}, effective_steps={total_schedule_steps}."
                        )
                steps = total_schedule_steps

                if self._log_sigmas or self._log_enabled:
                    schedule_first = float(sigmas[0]) if len(sigmas) > 0 else float("nan")
                    schedule_last = float(sigmas[-1]) if len(sigmas) > 0 else float("nan")
                    schedule_summary = self._summarize_sigmas(sigmas)
                    sigma_min_val = float("nan") if active_context.sigma_min is None else float(active_context.sigma_min)
                    sigma_max_val = float("nan") if active_context.sigma_max is None else float(active_context.sigma_max)
                    self._emit_event(
                        "sampling.sigma_schedule",
                        length=len(sigmas) - 1,
                        predict_min=sigma_min_val,
                        predict_max=sigma_max_val,
                        first=schedule_first,
                        last=schedule_last,
                        ladder=schedule_summary,
                    )

                sampler_kind = active_context.sampler_kind
                if exact_boundary_resume_requested:
                    self._ensure_exact_boundary_resume_supported(sampler_kind)

                start_idx = int(start_at_step or 0)
                if proportional_partial_denoise:
                    start_idx += self._nonflow_partial_denoise_start_index(
                        steps=int(steps),
                        denoise_strength=float(normalized_denoise),
                    )
                start_idx = max(0, min(start_idx, steps - 1))
                sigmas_run = sigmas[start_idx:]
                restart_step_plan: list[RestartStepPlan] | None = None
                if sampler_kind is SamplerKind.RESTART:
                    if active_context.scheduler_name != "karras":
                        raise RuntimeError(
                            f"Restart requires scheduler 'karras'; got {active_context.scheduler_name!r}."
                        )
                    restart_step_plan = build_restart_step_plan(sigmas_run)
                    if not restart_step_plan:
                        raise RuntimeError("Restart planner produced an empty execution plan.")
                resume_prediction_type = getattr(active_context, "prediction_type", None)
                if resume_prediction_type is None:
                    resume_prediction_type = getattr(getattr(model, "predictor", None), "prediction_type", None)
                if isinstance(resume_prediction_type, str):
                    resume_prediction_type = resume_prediction_type.lower()
                exact_resume_active = resume_boundary_state is not None
                if exact_resume_active:
                    self._validate_resume_boundary_state(
                        boundary_state=resume_boundary_state,
                        noise=noise,
                        sigmas=sigmas,
                        active_context=active_context,
                        steps=steps,
                        start_at_step=start_at_step,
                        prediction_type=resume_prediction_type,
                    )
                    start_idx = int(resume_boundary_state.completed_steps)
                    sigmas_run = sigmas
                    x = resume_boundary_state.latent.to(device=noise.device, dtype=noise.dtype)
                elif init_latent is not None:
                    sigma0 = sigmas_run[:1].to(device=noise.device, dtype=noise.dtype)
                    init_latent = init_latent.to(device=noise.device, dtype=noise.dtype)
                    x = model.predictor.noise_scaling(
                        sigma0,
                        noise,
                        init_latent,
                        max_denoise=False,
                    )
                else:
                    # Keep x in the core dtype (bf16/fp16) while preserving the sigma ladder precision.
                    sigma0 = sigmas_run[:1].to(dtype=noise.dtype)
                    x = model.predictor.noise_scaling(sigma0, noise, torch.zeros_like(noise))

                if self._log_enabled:
                    try:
                        smax = float(sigmas[0].item()) if hasattr(sigmas[0], "item") else float(sigmas[0])
                        smin = float(sigmas[-1].item()) if hasattr(sigmas[-1], "item") else float(sigmas[-1])
                        head = [float(v) for v in sigmas[: min(4, len(sigmas))].detach().cpu().tolist()]
                    except Exception:
                        smax = float("nan")
                        smin = float("nan")
                        head = []
                    pred_type = getattr(model.predictor, "prediction_type", None)
                    sigma_data = getattr(model.predictor, "sigma_data", None)
                    self._emit_event(
                        "sampling.plan.prepare",
                        algorithm=self.algorithm,
                        scheduler=active_context.scheduler_name,
                        steps=steps,
                        cfg_scale=float(cfg_scale),
                        prediction=pred_type or getattr(active_context, "prediction_type", None) or "<unknown>",
                        sigma_max=smax,
                        sigma_min=smin,
                        sigma_data=float(sigma_data) if sigma_data is not None else "n/a",
                        head=self._compact_series(head),
                    )

                compiled_cond = compile_conditions(cond)
                compiled_uncond = compile_conditions(uncond) if uncond is not None else None
                log_cfg_delta = False
                cfg_delta_steps = 0
                if self._log_enabled:
                    log_cfg_delta = env_flag("CODEX_LOG_CFG_DELTA", default=False)
                    if log_cfg_delta:
                        cfg_delta_steps = env_int("CODEX_LOG_CFG_DELTA_N", default=2, min_value=0)

                if isinstance(image_conditioning, torch.Tensor):
                    if (
                        image_conditioning.shape[0] == noise.shape[0]
                        and image_conditioning.shape[2] == noise.shape[2]
                        and image_conditioning.shape[3] == noise.shape[3]
                    ):
                        from .condition import Condition

                        for entry in compiled_cond:
                            entry["model_conds"]["c_concat"] = Condition(image_conditioning)
                        if compiled_uncond is not None:
                            for entry in compiled_uncond:
                                entry["model_conds"]["c_concat"] = Condition(image_conditioning)

                run_total_steps = (
                    steps
                    if exact_resume_active
                    else (len(restart_step_plan) if restart_step_plan is not None else steps - start_idx)
                )
                reported_total_steps: int | None = None if sampler_kind is SamplerKind.DPM_ADAPTIVE else run_total_steps
                guidance_policy = _resolve_guidance_policy(processing)
                if guidance_policy is None:
                    denoiser.model_options.pop(_GUIDANCE_POLICY_KEY, None)
                    denoiser.model_options.pop(_GUIDANCE_STEP_INDEX_KEY, None)
                    denoiser.model_options.pop(_GUIDANCE_TOTAL_STEPS_KEY, None)
                    denoiser.model_options.pop(_GUIDANCE_APG_MOMENTUM_BUFFER_KEY, None)
                    denoiser.model_options.pop(_GUIDANCE_WARNED_SAMPLER_CFG_KEY, None)
                else:
                    if sampler_kind is SamplerKind.DPM_ADAPTIVE and guidance_policy.get("cfg_trunc_ratio") is not None:
                        raise RuntimeError(
                            "Sampler 'dpm adaptive' does not support guidance cfg_trunc_ratio because "
                            "adaptive runs do not have an honest fixed total-step contract."
                        )
                    denoiser.model_options[_GUIDANCE_POLICY_KEY] = guidance_policy
                    if reported_total_steps is None:
                        denoiser.model_options.pop(_GUIDANCE_TOTAL_STEPS_KEY, None)
                    else:
                        denoiser.model_options[_GUIDANCE_TOTAL_STEPS_KEY] = reported_total_steps
                    denoiser.model_options.pop(_GUIDANCE_WARNED_SAMPLER_CFG_KEY, None)
                    self._emit_event(
                        "guidance.policy",
                        apg_enabled=bool(guidance_policy.get("apg_enabled", False)),
                        start_step=int(guidance_policy.get("apg_start_step", 0) or 0),
                        cfg_trunc_ratio=guidance_policy.get("cfg_trunc_ratio"),
                        guidance_rescale=float(guidance_policy.get("guidance_rescale", 0.0) or 0.0),
                        apg_rescale=float(guidance_policy.get("apg_rescale", 0.0) or 0.0),
                        renorm_cfg=float(guidance_policy.get("renorm_cfg", 0.0) or 0.0),
                    )
                if exact_resume_active:
                    assert resume_boundary_state is not None
                    guidance_apg_momentum_buffer = self._clone_boundary_tensor(
                        resume_boundary_state.guidance_apg_momentum_buffer
                    )
                    if guidance_apg_momentum_buffer is not None:
                        denoiser.model_options[_GUIDANCE_APG_MOMENTUM_BUFFER_KEY] = guidance_apg_momentum_buffer.to(
                            device=x.device,
                            dtype=x.dtype,
                        )
                    else:
                        denoiser.model_options.pop(_GUIDANCE_APG_MOMENTUM_BUFFER_KEY, None)
                progress_owner_token = str(getattr(processing, "_codex_progress_owner_token", "") or "")
                backend_state.start(
                    job_count=1,
                    sampling_steps=reported_total_steps,
                    progress_owner_token=progress_owner_token,
                )
                state_started = True
                transformer_options = denoiser.model_options.get("transformer_options", None)
                if not isinstance(transformer_options, dict):
                    raise RuntimeError(
                        "denoiser.model_options['transformer_options'] must be a dict for block progress wiring "
                        f"(got {type(transformer_options).__name__})."
                    )
                block_progress_controller = RichBlockProgressController(enabled=active_context.enable_progress)
                console_block_progress_active = bool(getattr(block_progress_controller, "is_active", False))
                if self._log_enabled:
                    self._emit_event(
                        "sampling.block_progress.console",
                        enabled=console_block_progress_active,
                        env_flag="CODEX_PROGRESS_BAR",
                    )

                def _on_block_progress(block_index: int, total_blocks: int) -> None:
                    normalized_index, normalized_total = validate_block_progress_payload(
                        block_index=block_index,
                        total_blocks=total_blocks,
                    )

                    backend_state.update_sampling_block(
                        block_index=normalized_index,
                        total_blocks=normalized_total,
                        owner_token=progress_owner_token,
                    )
                    if block_progress_controller is not None:
                        block_progress_controller.update(
                            block_index=normalized_index,
                            total_blocks=normalized_total,
                        )

                transformer_options[BLOCK_PROGRESS_CALLBACK_KEY] = _on_block_progress
                backend_state.reset_sampling_blocks(owner_token=progress_owner_token)

                strict = True
                import time as _time

                preview_interval = active_context.preview_interval
                t0 = _time.perf_counter()

                prediction_type = getattr(active_context, "prediction_type", None)
                if prediction_type is None:
                    prediction_type = getattr(getattr(model, "predictor", None), "prediction_type", None)
                if isinstance(prediction_type, str):
                    prediction_type = prediction_type.lower()
                profile_meta = {
                    "algorithm": self.algorithm,
                    "sampler_kind": sampler_kind.value,
                    "scheduler": active_context.scheduler_name,
                    "steps": reported_total_steps,
                    "requested_steps": run_total_steps,
                    "cfg_scale": float(cfg_scale),
                    "device": str(noise.device),
                    "noise_dtype": str(noise.dtype),
                    "x_dtype": str(x.dtype),
                    "model_compute_dtype": str(getattr(model, "computation_dtype", None)),
                }
                profile_name = f"sampling-{sampler_kind.value}-{active_context.scheduler_name}"

                if self._log_enabled:
                    head = []
                    try:
                        head = [float(v) for v in sigmas_run[: min(4, len(sigmas_run))].detach().cpu().tolist()]
                    except Exception:
                        head = []
                    self._emit_event(
                        "sampling.plan.run",
                        algorithm=sampler_kind.value,
                        scheduler=active_context.scheduler_name,
                        steps=run_total_steps,
                        cfg_scale=float(cfg_scale),
                        head=self._compact_series(head),
                    )

                old_denoised: Optional[torch.Tensor] = None
                older_denoised: Optional[torch.Tensor] = None
                old_denoised_d: Optional[torch.Tensor] = None
                gradient_estimation_prev_d: Optional[torch.Tensor] = None
                t_prev: float | None = None
                h_prev: float | None = None
                h_prev_2: float | None = None
                eps_history: List[torch.Tensor] = []
                deis_coefficients: tuple[tuple[float, ...], ...] | None = None
                res_multistep_old_sigma_down: float | None = None
                uni_pc_history_denoised: list[torch.Tensor] = []
                uni_pc_history_lambdas: list[float] = []
                uni_pc_order_cap = 1
                er_sde_params: dict[str, Any] | None = None
                er_sde_lambdas: torch.Tensor | None = None
                er_sde_point_indices: torch.Tensor | None = None
                sa_solver_sigmas: torch.Tensor | None = None
                sa_solver_lambdas: torch.Tensor | None = None
                sa_solver_tau_func: Callable[[float | torch.Tensor], float] | None = None
                sa_solver_pred_list: list[torch.Tensor] = []
                sa_solver_prev_noise: torch.Tensor | None = None
                sa_solver_prev_h = 0.0
                sa_solver_prev_tau_t = 0.0
                sa_solver_predictor_order = SA_SOLVER_DEFAULT_PREDICTOR_ORDER
                sa_solver_corrector_order = SA_SOLVER_DEFAULT_CORRECTOR_ORDER
                sa_solver_s_noise = SA_SOLVER_DEFAULT_S_NOISE
                sa_solver_simple_order_2 = SA_SOLVER_DEFAULT_SIMPLE_ORDER_2
                sa_solver_use_pece = sampler_kind is SamplerKind.SA_SOLVER_PECE
                sa_solver_lower_order_to_end = False
                seeds2_sigmas: torch.Tensor | None = None
                seeds2_lambdas: torch.Tensor | None = None
                seeds2_eta = _SEEDS_2_DEFAULT_ETA
                seeds2_s_noise = _SEEDS_2_DEFAULT_S_NOISE
                seeds2_r = _SEEDS_2_DEFAULT_R
                seeds3_sigmas: torch.Tensor | None = None
                seeds3_lambdas: torch.Tensor | None = None
                seeds3_eta = _SEEDS_3_DEFAULT_ETA
                seeds3_s_noise = _SEEDS_3_DEFAULT_S_NOISE
                seeds3_r_1 = _SEEDS_3_DEFAULT_R_1
                seeds3_r_2 = _SEEDS_3_DEFAULT_R_2
                dpm_sde_sigmas: torch.Tensor | None = None
                dpm_sde_lambdas: torch.Tensor | None = None
                dpm_sde_eta = _DPMPP_SDE_DEFAULT_ETA
                dpm_sde_s_noise = _DPMPP_SDE_DEFAULT_S_NOISE
                dpm_sde_r = _DPMPP_SDE_DEFAULT_R
                dpm2m_sde_sigmas: torch.Tensor | None = None
                dpm2m_sde_lambdas: torch.Tensor | None = None
                dpm2m_sde_eta = 1.0
                dpm2m_sde_s_noise = 1.0
                dpm3m_sde_sigmas: torch.Tensor | None = None
                dpm3m_sde_lambdas: torch.Tensor | None = None
                dpm3m_sde_eta = 1.0
                dpm3m_sde_s_noise = 1.0
                lms_sigmas_run: list[float] | None = None
                seeded_step_rng: ImageRNG | None = None
                if exact_resume_active:
                    assert resume_boundary_state is not None
                    old_denoised = self._clone_boundary_tensor(resume_boundary_state.old_denoised)
                    if old_denoised is not None:
                        old_denoised = old_denoised.to(device=x.device, dtype=x.dtype)
                    older_denoised = self._clone_boundary_tensor(resume_boundary_state.older_denoised)
                    if older_denoised is not None:
                        older_denoised = older_denoised.to(device=x.device, dtype=x.dtype)
                    old_denoised_d = self._clone_boundary_tensor(resume_boundary_state.old_denoised_d)
                    if old_denoised_d is not None:
                        old_denoised_d = old_denoised_d.to(device=x.device, dtype=x.dtype)
                    gradient_estimation_prev_d = self._clone_boundary_tensor(
                        resume_boundary_state.gradient_estimation_prev_d
                    )
                    if gradient_estimation_prev_d is not None:
                        gradient_estimation_prev_d = gradient_estimation_prev_d.to(device=x.device, dtype=x.dtype)
                    t_prev = resume_boundary_state.t_prev
                    h_prev = resume_boundary_state.h_prev
                    h_prev_2 = resume_boundary_state.h_prev_2
                    eps_history = [
                        tensor.to(device=x.device, dtype=x.dtype)
                        for tensor in self._clone_boundary_tensor_tuple(resume_boundary_state.eps_history)
                    ]
                    res_multistep_old_sigma_down = resume_boundary_state.res_multistep_old_sigma_down
                    uni_pc_history_denoised = [
                        tensor.to(device=x.device, dtype=x.dtype)
                        for tensor in self._clone_boundary_tensor_tuple(resume_boundary_state.uni_pc_history_denoised)
                    ]
                    uni_pc_history_lambdas = [float(value) for value in resume_boundary_state.uni_pc_history_lambdas]
                    sa_solver_pred_list = [
                        tensor.to(device=x.device, dtype=x.dtype)
                        for tensor in self._clone_boundary_tensor_tuple(resume_boundary_state.sa_solver_pred_list)
                    ]
                    sa_solver_prev_noise = self._clone_boundary_tensor(resume_boundary_state.sa_solver_prev_noise)
                    if sa_solver_prev_noise is not None:
                        sa_solver_prev_noise = sa_solver_prev_noise.to(device=x.device, dtype=x.dtype)
                    sa_solver_prev_h = float(resume_boundary_state.sa_solver_prev_h)
                    sa_solver_prev_tau_t = float(resume_boundary_state.sa_solver_prev_tau_t)
                    seeded_step_rng = self._clone_image_rng(resume_boundary_state.seeded_step_rng)
                if sampler_kind in {SamplerKind.UNI_PC, SamplerKind.UNI_PC_BH2}:
                    uni_pc_order_cap = max(1, min(3, int(sigmas_run.numel()) - 2))
                if sampler_kind is SamplerKind.ER_SDE:
                    er_sde_params = self._resolve_er_sde_runtime_params(er_sde_options)
                    er_sde_lambdas = self._compute_er_sde_lambdas(
                        sigmas_run,
                        prediction_type=getattr(active_context, "prediction_type", None),
                    )
                    er_sde_point_indices = torch.arange(
                        0.0,
                        200.0,
                        dtype=torch.float32,
                        device=x.device,
                    )
                if sampler_kind is SamplerKind.DPMPP_SDE:
                    if active_context.scheduler_name != "karras":
                        raise RuntimeError(
                            f"DPM++ SDE currently requires scheduler 'karras'; got {active_context.scheduler_name!r}."
                        )
                    if not math.isfinite(dpm_sde_eta) or dpm_sde_eta < 0.0:
                        raise RuntimeError(
                            f"DPM++ SDE default eta must be finite and >= 0; got {dpm_sde_eta!r}."
                        )
                    if not math.isfinite(dpm_sde_s_noise) or dpm_sde_s_noise < 0.0:
                        raise RuntimeError(
                            f"DPM++ SDE default s_noise must be finite and >= 0; got {dpm_sde_s_noise!r}."
                        )
                    if not math.isfinite(dpm_sde_r) or not (0.0 < dpm_sde_r < 1.0):
                        raise RuntimeError(
                            f"DPM++ SDE midpoint ratio must be finite and inside (0, 1); got {dpm_sde_r!r}."
                        )
                    predictor = getattr(model, "predictor", None)
                    dpm_sde_sigmas = offset_first_sigma_for_snr(
                        sigmas_run,
                        prediction_type=prediction_type,
                        predictor=predictor,
                    )
                    dpm_sde_lambdas = sigma_to_half_log_snr(
                        dpm_sde_sigmas,
                        prediction_type=prediction_type,
                    )
                if (not exact_resume_active) and sampler_kind in {
                    SamplerKind.DPM2M_SDE,
                    SamplerKind.DPM2M_SDE_HEUN,
                    SamplerKind.DPM2M_SDE_GPU,
                    SamplerKind.DPM2M_SDE_HEUN_GPU,
                }:
                    if active_context.scheduler_name != "exponential":
                        raise RuntimeError(
                            f"DPM++ 2M SDE currently requires scheduler 'exponential'; got {active_context.scheduler_name!r}."
                        )
                    if not math.isfinite(dpm2m_sde_eta) or dpm2m_sde_eta < 0.0:
                        raise RuntimeError(
                            f"DPM++ 2M SDE default eta must be finite and >= 0; got {dpm2m_sde_eta!r}."
                        )
                    if not math.isfinite(dpm2m_sde_s_noise) or dpm2m_sde_s_noise < 0.0:
                        raise RuntimeError(
                            f"DPM++ 2M SDE default s_noise must be finite and >= 0; got {dpm2m_sde_s_noise!r}."
                        )
                    predictor = getattr(model, "predictor", None)
                    dpm2m_sde_sigmas = offset_first_sigma_for_snr(
                        sigmas_run,
                        prediction_type=prediction_type,
                        predictor=predictor,
                    )
                    dpm2m_sde_lambdas = sigma_to_half_log_snr(
                        dpm2m_sde_sigmas,
                        prediction_type=prediction_type,
                    )
                if sampler_kind is SamplerKind.DPM3M_SDE:
                    if active_context.scheduler_name != "exponential":
                        raise RuntimeError(
                            f"DPM++ 3M SDE currently requires scheduler 'exponential'; got {active_context.scheduler_name!r}."
                        )
                    if not math.isfinite(dpm3m_sde_eta) or dpm3m_sde_eta < 0.0:
                        raise RuntimeError(
                            f"DPM++ 3M SDE default eta must be finite and >= 0; got {dpm3m_sde_eta!r}."
                        )
                    if not math.isfinite(dpm3m_sde_s_noise) or dpm3m_sde_s_noise < 0.0:
                        raise RuntimeError(
                            f"DPM++ 3M SDE default s_noise must be finite and >= 0; got {dpm3m_sde_s_noise!r}."
                        )
                    predictor = getattr(model, "predictor", None)
                    dpm3m_sde_sigmas = offset_first_sigma_for_snr(
                        sigmas_run,
                        prediction_type=prediction_type,
                        predictor=predictor,
                    )
                    dpm3m_sde_lambdas = sigma_to_half_log_snr(
                        dpm3m_sde_sigmas,
                        prediction_type=prediction_type,
                    )
                if sampler_kind in {SamplerKind.SA_SOLVER, SamplerKind.SA_SOLVER_PECE}:
                    if active_context.scheduler_name != "karras":
                        raise RuntimeError(
                            f"SA-Solver currently requires scheduler 'karras'; got {active_context.scheduler_name!r}."
                        )
                    predictor = getattr(model, "predictor", None)
                    if predictor is None:
                        raise RuntimeError(
                            "SA-Solver requires model.predictor to construct the default tau interval."
                        )
                    sa_solver_sigmas = offset_first_sigma_for_snr(
                        sigmas_run,
                        prediction_type=prediction_type,
                        predictor=predictor,
                    )
                    sa_solver_lambdas = sigma_to_half_log_snr(
                        sa_solver_sigmas,
                        prediction_type=prediction_type,
                    )
                    percent_to_sigma = getattr(predictor, "percent_to_sigma", None)
                    if not callable(percent_to_sigma):
                        raise RuntimeError(
                            "SA-Solver requires predictor.percent_to_sigma(...) to construct the default tau interval."
                        )
                    start_sigma = float(percent_to_sigma(0.2))
                    end_sigma = float(percent_to_sigma(0.8))
                    sa_solver_tau_func = get_tau_interval_func(
                        start_sigma,
                        end_sigma,
                        eta=SA_SOLVER_DEFAULT_ETA,
                    )
                    sa_solver_lower_order_to_end = bool(float(sa_solver_sigmas[-1]) == 0.0)
                if sampler_kind is SamplerKind.SEEDS_2:
                    if active_context.scheduler_name != "karras":
                        raise RuntimeError(
                            f"SEEDS-2 currently requires scheduler 'karras'; got {active_context.scheduler_name!r}."
                        )
                    if not math.isfinite(seeds2_eta) or seeds2_eta < 0.0:
                        raise RuntimeError(f"SEEDS-2 default eta must be finite and >= 0; got {seeds2_eta!r}.")
                    if not math.isfinite(seeds2_s_noise) or seeds2_s_noise < 0.0:
                        raise RuntimeError(
                            f"SEEDS-2 default s_noise must be finite and >= 0; got {seeds2_s_noise!r}."
                        )
                    if not math.isfinite(seeds2_r) or not (0.0 < seeds2_r <= 1.0):
                        raise RuntimeError(f"SEEDS-2 default r must be finite and in (0, 1]; got {seeds2_r!r}.")
                    predictor = getattr(model, "predictor", None)
                    seeds2_sigmas = offset_first_sigma_for_snr(
                        sigmas_run,
                        prediction_type=prediction_type,
                        predictor=predictor,
                    )
                    seeds2_lambdas = sigma_to_half_log_snr(
                        seeds2_sigmas,
                        prediction_type=prediction_type,
                    )
                if sampler_kind is SamplerKind.SEEDS_3:
                    if active_context.scheduler_name != "karras":
                        raise RuntimeError(
                            f"SEEDS-3 currently requires scheduler 'karras'; got {active_context.scheduler_name!r}."
                        )
                    if not math.isfinite(seeds3_eta) or seeds3_eta < 0.0:
                        raise RuntimeError(f"SEEDS-3 default eta must be finite and >= 0; got {seeds3_eta!r}.")
                    if not math.isfinite(seeds3_s_noise) or seeds3_s_noise < 0.0:
                        raise RuntimeError(
                            f"SEEDS-3 default s_noise must be finite and >= 0; got {seeds3_s_noise!r}."
                        )
                    if not math.isfinite(seeds3_r_1) or not (0.0 < seeds3_r_1 < 1.0):
                        raise RuntimeError(
                            f"SEEDS-3 default r_1 must be finite and in (0, 1); got {seeds3_r_1!r}."
                        )
                    if not math.isfinite(seeds3_r_2) or not (0.0 < seeds3_r_2 <= 1.0):
                        raise RuntimeError(
                            f"SEEDS-3 default r_2 must be finite and in (0, 1]; got {seeds3_r_2!r}."
                        )
                    if not (seeds3_r_1 < seeds3_r_2):
                        raise RuntimeError(
                            f"SEEDS-3 default ratios must satisfy r_1 < r_2; got r_1={seeds3_r_1!r} r_2={seeds3_r_2!r}."
                        )
                    predictor = getattr(model, "predictor", None)
                    seeds3_sigmas = offset_first_sigma_for_snr(
                        sigmas_run,
                        prediction_type=prediction_type,
                        predictor=predictor,
                    )
                    seeds3_lambdas = sigma_to_half_log_snr(
                        seeds3_sigmas,
                        prediction_type=prediction_type,
                    )
                if sampler_kind is SamplerKind.LMS:
                    lms_sigmas_run = [float(value) for value in sigmas_run.detach().cpu().tolist()]
                if sampler_kind is SamplerKind.DEIS:
                    deis_coefficients = build_deis_coefficients(sigmas_run, max_order=3)
                if sampler_kind in {
                    SamplerKind.RES_MULTISTEP_CFG_PP,
                    SamplerKind.RES_MULTISTEP_ANCESTRAL_CFG_PP,
                }:
                    res_multistep_cfg_pp_saved_post_cfg_functions = list(
                        denoiser.model_options.get("sampler_post_cfg_function", [])
                    )
                    res_multistep_cfg_pp_saved_disable_cfg1 = denoiser.model_options.get(
                        "disable_cfg1_optimization",
                        res_multistep_cfg_pp_disable_cfg1_missing,
                    )

                    def _capture_res_multistep_cfg_pp(args: Mapping[str, Any]) -> torch.Tensor:
                        nonlocal res_multistep_cfg_pp_capture
                        denoised_payload = args.get("denoised")
                        if not isinstance(denoised_payload, torch.Tensor):
                            raise RuntimeError(
                                "Residual multistep cfg++ post-CFG hook requires tensor `denoised` output."
                            )
                        uncond_denoised = args.get("uncond_denoised")
                        if not isinstance(uncond_denoised, torch.Tensor):
                            raise RuntimeError(
                                "Residual multistep cfg++ requires tensor `uncond_denoised` capture from sampler post-CFG hook."
                            )
                        res_multistep_cfg_pp_capture = uncond_denoised
                        return denoised_payload

                    denoiser.model_options["sampler_post_cfg_function"] = list(
                        res_multistep_cfg_pp_saved_post_cfg_functions
                    )
                    set_model_options_post_cfg_function(
                        denoiser.model_options,
                        _capture_res_multistep_cfg_pp,
                        disable_cfg1_optimization=True,
                    )
                if sampler_kind is SamplerKind.GRADIENT_ESTIMATION_CFG_PP:
                    gradient_estimation_cfg_pp_saved_post_cfg_functions = list(
                        denoiser.model_options.get("sampler_post_cfg_function", [])
                    )
                    gradient_estimation_cfg_pp_saved_disable_cfg1 = denoiser.model_options.get(
                        "disable_cfg1_optimization",
                        gradient_estimation_cfg_pp_disable_cfg1_missing,
                    )

                    def _capture_gradient_estimation_cfg_pp(args: Mapping[str, Any]) -> torch.Tensor:
                        nonlocal gradient_estimation_cfg_pp_capture
                        denoised_payload = args.get("denoised")
                        if not isinstance(denoised_payload, torch.Tensor):
                            raise RuntimeError(
                                "Gradient estimation cfg++ post-CFG hook requires tensor `denoised` output."
                            )
                        uncond_denoised = args.get("uncond_denoised")
                        if not isinstance(uncond_denoised, torch.Tensor):
                            raise RuntimeError(
                                "Gradient estimation cfg++ requires tensor `uncond_denoised` capture from sampler post-CFG hook."
                            )
                        gradient_estimation_cfg_pp_capture = uncond_denoised
                        return denoised_payload

                    denoiser.model_options["sampler_post_cfg_function"] = list(
                        gradient_estimation_cfg_pp_saved_post_cfg_functions
                    )
                    set_model_options_post_cfg_function(
                        denoiser.model_options,
                        _capture_gradient_estimation_cfg_pp,
                        disable_cfg1_optimization=True,
                    )
                if sampler_kind in {SamplerKind.EULER_CFG_PP, SamplerKind.EULER_A_CFG_PP}:
                    euler_cfg_pp_saved_post_cfg_functions = list(
                        denoiser.model_options.get("sampler_post_cfg_function", [])
                    )
                    euler_cfg_pp_saved_disable_cfg1 = denoiser.model_options.get(
                        "disable_cfg1_optimization",
                        euler_cfg_pp_disable_cfg1_missing,
                    )

                    def _capture_euler_cfg_pp(args: Mapping[str, Any]) -> torch.Tensor:
                        nonlocal euler_cfg_pp_capture
                        denoised_payload = args.get("denoised")
                        if not isinstance(denoised_payload, torch.Tensor):
                            raise RuntimeError(
                                "Euler cfg++ post-CFG hook requires tensor `denoised` output."
                            )
                        uncond_denoised = args.get("uncond_denoised")
                        if not isinstance(uncond_denoised, torch.Tensor):
                            raise RuntimeError(
                                "Euler cfg++ requires tensor `uncond_denoised` capture from sampler post-CFG hook."
                            )
                        euler_cfg_pp_capture = uncond_denoised
                        return denoised_payload

                    denoiser.model_options["sampler_post_cfg_function"] = list(
                        euler_cfg_pp_saved_post_cfg_functions
                    )
                    set_model_options_post_cfg_function(
                        denoiser.model_options,
                        _capture_euler_cfg_pp,
                        disable_cfg1_optimization=True,
                    )
                if sampler_kind is SamplerKind.DPM2M_CFG_PP:
                    dpm2m_cfg_pp_saved_post_cfg_functions = list(
                        denoiser.model_options.get("sampler_post_cfg_function", [])
                    )
                    dpm2m_cfg_pp_saved_disable_cfg1 = denoiser.model_options.get(
                        "disable_cfg1_optimization",
                        dpm2m_cfg_pp_disable_cfg1_missing,
                    )

                    def _capture_dpm2m_cfg_pp(args: Mapping[str, Any]) -> torch.Tensor:
                        nonlocal dpm2m_cfg_pp_capture
                        denoised_payload = args.get("denoised")
                        if not isinstance(denoised_payload, torch.Tensor):
                            raise RuntimeError(
                                "DPM++ 2M cfg++ post-CFG hook requires tensor `denoised` output."
                            )
                        uncond_denoised = args.get("uncond_denoised")
                        if not isinstance(uncond_denoised, torch.Tensor):
                            raise RuntimeError(
                                "DPM++ 2M cfg++ requires tensor `uncond_denoised` capture from sampler post-CFG hook."
                            )
                        dpm2m_cfg_pp_capture = uncond_denoised
                        return denoised_payload

                    denoiser.model_options["sampler_post_cfg_function"] = list(
                        dpm2m_cfg_pp_saved_post_cfg_functions
                    )
                    set_model_options_post_cfg_function(
                        denoiser.model_options,
                        _capture_dpm2m_cfg_pp,
                        disable_cfg1_optimization=True,
                    )
                if sampler_kind is SamplerKind.DPM2S_ANCESTRAL_CFG_PP:
                    dpm2s_ancestral_cfg_pp_saved_post_cfg_functions = list(
                        denoiser.model_options.get("sampler_post_cfg_function", [])
                    )
                    dpm2s_ancestral_cfg_pp_saved_disable_cfg1 = denoiser.model_options.get(
                        "disable_cfg1_optimization",
                        dpm2s_ancestral_cfg_pp_disable_cfg1_missing,
                    )

                    def _capture_dpm2s_ancestral_cfg_pp(args: Mapping[str, Any]) -> torch.Tensor:
                        nonlocal dpm2s_ancestral_cfg_pp_capture
                        denoised_payload = args.get("denoised")
                        if not isinstance(denoised_payload, torch.Tensor):
                            raise RuntimeError(
                                "DPM++ 2S ancestral cfg++ post-CFG hook requires tensor `denoised` output."
                            )
                        uncond_denoised = args.get("uncond_denoised")
                        if not isinstance(uncond_denoised, torch.Tensor):
                            raise RuntimeError(
                                "DPM++ 2S ancestral cfg++ requires tensor `uncond_denoised` capture from sampler post-CFG hook."
                            )
                        dpm2s_ancestral_cfg_pp_capture = uncond_denoised
                        return denoised_payload

                    denoiser.model_options["sampler_post_cfg_function"] = list(
                        dpm2s_ancestral_cfg_pp_saved_post_cfg_functions
                    )
                    set_model_options_post_cfg_function(
                        denoiser.model_options,
                        _capture_dpm2s_ancestral_cfg_pp,
                        disable_cfg1_optimization=True,
                    )
                if sampler_kind in _SEEDED_STEP_RNG_SAMPLERS:
                    if exact_resume_active:
                        if seeded_step_rng is None:
                            raise RuntimeError(
                                f"resume_boundary_state is missing seeded_step_rng for exact stochastic resume with sampler "
                                f"'{sampler_kind.value}'."
                            )
                    else:
                        seeded_step_rng = self._build_seeded_step_rng(
                            processing=processing,
                            active_context=active_context,
                            noise=base_noise,
                        )
                        for skip_index in range(start_idx):
                            sigma_skip_next = float(sigmas[skip_index + 1])
                            if sigma_skip_next <= 0.0:
                                continue
                            if sampler_kind in {
                                SamplerKind.DPM2M_SDE,
                                SamplerKind.DPM2M_SDE_HEUN,
                                SamplerKind.DPM2M_SDE_GPU,
                                SamplerKind.DPM2M_SDE_HEUN_GPU,
                            }:
                                if dpm2m_sde_lambdas is None:
                                    raise RuntimeError("DPM++ 2M SDE seeded noise RNG setup missing half-logSNR state.")
                                if dpm2m_sde_eta <= 0.0 or dpm2m_sde_s_noise <= 0.0:
                                    continue
                                lambda_skip_s = float(dpm2m_sde_lambdas[skip_index])
                                lambda_skip_t = float(dpm2m_sde_lambdas[skip_index + 1])
                                h_skip = lambda_skip_t - lambda_skip_s
                                if not math.isfinite(h_skip) or h_skip <= 0.0:
                                    raise RuntimeError(
                                        "DPM++ 2M SDE skip-burn produced an invalid half-logSNR step size "
                                        f"h={h_skip} at skip_index={skip_index}."
                                    )
                                noise_scale_sq = -math.expm1(-2.0 * h_skip * dpm2m_sde_eta)
                                if noise_scale_sq < -1e-8:
                                    raise RuntimeError(
                                        "DPM++ 2M SDE skip-burn produced a negative stochastic noise scale "
                                        f"({noise_scale_sq}) at skip_index={skip_index}."
                                    )
                                if noise_scale_sq <= 0.0:
                                    continue
                            if sampler_kind is SamplerKind.DPMPP_SDE:
                                if dpm_sde_eta <= 0.0 or dpm_sde_s_noise <= 0.0:
                                    continue
                                seeded_step_rng.next()
                            if sampler_kind is SamplerKind.DPM3M_SDE:
                                if dpm3m_sde_lambdas is None:
                                    raise RuntimeError("DPM++ 3M SDE seeded noise RNG setup missing half-logSNR state.")
                                if dpm3m_sde_eta <= 0.0 or dpm3m_sde_s_noise <= 0.0:
                                    continue
                                lambda_skip_s = float(dpm3m_sde_lambdas[skip_index])
                                lambda_skip_t = float(dpm3m_sde_lambdas[skip_index + 1])
                                h_skip = lambda_skip_t - lambda_skip_s
                                if not math.isfinite(h_skip) or h_skip <= 0.0:
                                    raise RuntimeError(
                                        "DPM++ 3M SDE skip-burn produced an invalid half-logSNR step size "
                                        f"h={h_skip} at skip_index={skip_index}."
                                    )
                                noise_scale_sq = -math.expm1(-2.0 * h_skip * dpm3m_sde_eta)
                                if noise_scale_sq < -1e-8:
                                    raise RuntimeError(
                                        "DPM++ 3M SDE skip-burn produced a negative stochastic noise scale "
                                        f"({noise_scale_sq}) at skip_index={skip_index}."
                                    )
                                if noise_scale_sq <= 0.0:
                                    continue
                            if sampler_kind is SamplerKind.ER_SDE:
                                if er_sde_params is None:
                                    raise RuntimeError("ER-SDE seeded noise RNG setup missing runtime parameters.")
                                if float(er_sde_params["s_noise"]) <= 0.0:
                                    continue
                            if sampler_kind in {SamplerKind.SA_SOLVER, SamplerKind.SA_SOLVER_PECE}:
                                if sa_solver_tau_func is None:
                                    raise RuntimeError("SA-Solver seeded noise RNG setup missing tau interval function.")
                                if float(sa_solver_tau_func(sigma_skip_next)) <= 0.0:
                                    continue
                            if sampler_kind is SamplerKind.SEEDS_2:
                                if seeds2_s_noise <= 0.0:
                                    continue
                                seeded_step_rng.next()
                            if sampler_kind is SamplerKind.SEEDS_3:
                                if seeds3_s_noise <= 0.0:
                                    continue
                                seeded_step_rng.next()
                                seeded_step_rng.next()
                            seeded_step_rng.next()

                def _dpm_sigma_from_time(time_value: torch.Tensor) -> torch.Tensor:
                    return torch.exp(-time_value)

                def _dpm_time_from_sigma(sigma_value: torch.Tensor) -> torch.Tensor:
                    if not bool(torch.all(torch.isfinite(sigma_value))):
                        raise RuntimeError(
                            "DPM solver received non-finite sigma values while converting to log-time."
                        )
                    if bool(torch.any(sigma_value <= 0.0)):
                        raise RuntimeError(
                            "DPM solver requires strictly positive sigmas while converting to log-time."
                        )
                    return -torch.log(sigma_value)

                def _dpm_select_solver_sigma_bounds() -> tuple[float, float]:
                    if sigmas_run.ndim != 1:
                        raise RuntimeError(
                            f"DPM solver expects a 1D sigma schedule; got shape={tuple(sigmas_run.shape)}."
                        )
                    sigma_count = int(sigmas_run.numel())
                    if sigma_count < 2:
                        raise RuntimeError(
                            "DPM solver requires at least two sigma entries (start and end)."
                        )
                    sigma_max_value = float(sigmas_run[0])
                    positive_indices = torch.nonzero(sigmas_run > 0.0, as_tuple=False).flatten()
                    if int(positive_indices.numel()) <= 0:
                        raise RuntimeError(
                            "DPM solver requires at least one strictly-positive sigma entry "
                            "in the active schedule."
                        )
                    sigma_min_value = float(sigmas_run[int(positive_indices[-1].item())])
                    if not math.isfinite(sigma_max_value) or not math.isfinite(sigma_min_value):
                        raise RuntimeError(
                            "DPM solver sigma bounds must be finite "
                            f"(sigma_max={sigma_max_value}, sigma_min={sigma_min_value})."
                        )

                def _cfg_pp_alpha_from_sigma(sigma_value: float) -> float:
                    if not math.isfinite(sigma_value) or sigma_value < 0.0:
                        raise RuntimeError(f"CFG++ alpha requires a finite non-negative sigma; got {sigma_value!r}.")
                    if prediction_type == "const":
                        sigma_safe = min(max(sigma_value, 1e-6), 1.0 - 1e-6)
                        alpha_value = 1.0 - sigma_safe
                    else:
                        alpha_value = 1.0
                    if not math.isfinite(alpha_value) or alpha_value <= 0.0:
                        raise RuntimeError(
                            f"CFG++ alpha projection produced an invalid value {alpha_value!r} for sigma={sigma_value!r}."
                        )
                    return alpha_value
                    if sigma_max_value <= 0.0 or sigma_min_value <= 0.0:
                        raise RuntimeError(
                            "DPM solver sigma bounds must be > 0 "
                            f"(sigma_max={sigma_max_value}, sigma_min={sigma_min_value})."
                        )
                    if sigma_max_value < sigma_min_value:
                        raise RuntimeError(
                            "DPM solver requires a descending sigma interval "
                            f"(sigma_max={sigma_max_value}, sigma_min={sigma_min_value})."
                        )
                    return sigma_min_value, sigma_max_value

                def _dpm_eval_eps(
                    latent_state: torch.Tensor,
                    *,
                    sigma_value: float,
                    current_step: int,
                    step_total: int | None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    nonlocal retry
                    if not math.isfinite(sigma_value) or sigma_value <= 0.0:
                        raise RuntimeError(
                            "DPM solver denoiser call requires a finite positive sigma "
                            f"(got {sigma_value})."
                        )
                    sigma_batch = torch.full(
                        (latent_state.shape[0],),
                        sigma_value,
                        device=latent_state.device,
                        dtype=torch.float32,
                    )
                    if guidance_policy is not None:
                        denoiser.model_options[_GUIDANCE_STEP_INDEX_KEY] = max(0, current_step - 1)
                        if step_total is None:
                            denoiser.model_options.pop(_GUIDANCE_TOTAL_STEPS_KEY, None)
                        else:
                            denoiser.model_options[_GUIDANCE_TOTAL_STEPS_KEY] = step_total
                    model_input = latent_state
                    if pre_denoiser_hook is not None:
                        model_input = pre_denoiser_hook(model_input, sigma_batch, current_step, step_total)
                        if not isinstance(model_input, torch.Tensor):
                            raise RuntimeError("pre_denoiser_hook must return a torch.Tensor")
                        if tuple(model_input.shape) != tuple(noise.shape):
                            raise RuntimeError(
                                "pre_denoiser_hook returned unexpected shape "
                                f"{tuple(model_input.shape)}; expected {tuple(noise.shape)}"
                            )
                    denoised_output = sampling_function_inner(
                        model,
                        model_input,
                        sigma_batch,
                        compiled_uncond,
                        compiled_cond,
                        cfg_scale,
                        denoiser.model_options,
                        seed=None,
                        return_full=False,
                    )
                    if post_denoiser_hook is not None:
                        denoised_output = post_denoiser_hook(
                            denoised_output,
                            sigma_batch,
                            current_step,
                            step_total,
                        )
                        if not isinstance(denoised_output, torch.Tensor):
                            raise RuntimeError("post_denoiser_hook must return a torch.Tensor")
                        if tuple(denoised_output.shape) != tuple(model_input.shape):
                            raise RuntimeError(
                                "post_denoiser_hook returned unexpected shape "
                                f"{tuple(denoised_output.shape)}; expected {tuple(model_input.shape)}"
                            )
                    epsilon_output = (model_input - denoised_output) / max(sigma_value, 1e-8)
                    if strict and (torch.isnan(epsilon_output).any() or torch.isnan(denoised_output).any()):
                        reason = f"NaN detected at sampling step {current_step}"
                        emit_backend_message(
                            "NaN encountered during sampling; attempting precision fallback.",
                            logger=self._logger_name,
                            level="WARNING",
                            step=current_step,
                            dtype=str(getattr(model, "computation_dtype", model_input.dtype)),
                        )
                        next_dtype = memory_management.manager.report_precision_failure(
                            DeviceRole.CORE,
                            location=f"sampler.step_{current_step}",
                            reason=reason,
                        )
                        if next_dtype is None:
                            hint = memory_management.manager.precision_hint(DeviceRole.CORE)
                            raise RuntimeError(
                                "Diffusion core produced NaNs at "
                                f"step {current_step} on {noise.device} with dtype "
                                f"{getattr(model, 'computation_dtype', model_input.dtype)}. {hint}"
                            )
                        self._rebind_unet_precision(next_dtype)
                        retry = True
                        raise _PrecisionFallbackRequest
                    return model_input, denoised_output, epsilon_output

                def _dpm_emit_step_callbacks(
                    *,
                    latent_state: torch.Tensor,
                    denoised_state: torch.Tensor,
                    current_step: int,
                    sigma_current: float,
                    sigma_next: float,
                    step_start_time: float,
                ) -> float:
                    if post_step_hook is not None:
                        post_step_hook(latent_state, current_step, reported_total_steps)
                    if preview_callback is not None and (
                        (preview_interval > 0 and (current_step % preview_interval == 0))
                        or (reported_total_steps is not None and current_step >= reported_total_steps)
                    ):
                        try:
                            preview_callback(denoised_state.detach(), current_step, reported_total_steps)
                        except Exception:
                            pass
                    log_interval = max(1, reported_total_steps // 5) if reported_total_steps is not None else 5
                    if self._log_enabled and (
                        current_step == 1
                        or (reported_total_steps is not None and current_step == reported_total_steps)
                        or current_step % log_interval == 0
                    ):
                        self._emit_event(
                            "sampling.step",
                            step=current_step,
                            total_steps=reported_total_steps,
                            sigma=sigma_current,
                            sigma_next=sigma_next,
                            norm_x=float(latent_state.norm().item()),
                            norm_den=float(denoised_state.norm().item()),
                            dt_ms=(_time.perf_counter() - step_start_time) * 1000.0,
                        )
                        step_start_time = _time.perf_counter()
                    backend_state.tick(
                        sampling_step=(
                            max(1, min(current_step, reported_total_steps))
                            if reported_total_steps is not None
                            else max(1, current_step)
                        ),
                        owner_token=progress_owner_token,
                    )
                    backend_state.reset_sampling_blocks(owner_token=progress_owner_token)
                    profiler.step()
                    return step_start_time

                def _dpm_cached_eps(
                    cache: dict[str, dict[str, torch.Tensor]],
                    *,
                    cache_key: str,
                    latent_state: torch.Tensor,
                    time_value: torch.Tensor,
                    current_step: int,
                    step_total: int | None,
                ) -> dict[str, torch.Tensor]:
                    cached_payload = cache.get(cache_key)
                    if cached_payload is not None:
                        return cached_payload
                    sigma_value = float(_dpm_sigma_from_time(time_value).item())
                    model_input, denoised_output, epsilon_output = _dpm_eval_eps(
                        latent_state,
                        sigma_value=sigma_value,
                        current_step=current_step,
                        step_total=step_total,
                    )
                    payload = {
                        "model_input": model_input,
                        "denoised": denoised_output,
                        "epsilon": epsilon_output,
                    }
                    cache[cache_key] = payload
                    return payload

                def _dpm_solver_step_1(
                    latent_state: torch.Tensor,
                    *,
                    time_current: torch.Tensor,
                    time_next: torch.Tensor,
                    eps_cache: dict[str, dict[str, torch.Tensor]],
                    current_step: int,
                    step_total: int | None,
                ) -> torch.Tensor:
                    step_delta = float((time_next - time_current).item())
                    step_multiplier = math.expm1(step_delta)
                    sigma_next_value = float(_dpm_sigma_from_time(time_next).item())
                    eps_payload = _dpm_cached_eps(
                        eps_cache,
                        cache_key="epsilon",
                        latent_state=latent_state,
                        time_value=time_current,
                        current_step=current_step,
                        step_total=step_total,
                    )
                    state_base = eps_payload["model_input"]
                    epsilon_base = eps_payload["epsilon"]
                    return state_base - sigma_next_value * step_multiplier * epsilon_base

                def _dpm_solver_step_2(
                    latent_state: torch.Tensor,
                    *,
                    time_current: torch.Tensor,
                    time_next: torch.Tensor,
                    stage_ratio: float,
                    eps_cache: dict[str, dict[str, torch.Tensor]],
                    current_step: int,
                    step_total: int | None,
                ) -> torch.Tensor:
                    if stage_ratio == 0.0:
                        raise RuntimeError("DPM solver step-2 received stage_ratio=0.")
                    step_delta = float((time_next - time_current).item())
                    if abs(step_delta) <= 1e-12:
                        return latent_state
                    sigma_next_value = float(_dpm_sigma_from_time(time_next).item())
                    eps_payload = _dpm_cached_eps(
                        eps_cache,
                        cache_key="epsilon",
                        latent_state=latent_state,
                        time_value=time_current,
                        current_step=current_step,
                        step_total=step_total,
                    )
                    state_base = eps_payload["model_input"]
                    epsilon_base = eps_payload["epsilon"]
                    stage_time = time_current + stage_ratio * (time_next - time_current)
                    stage_multiplier = math.expm1(stage_ratio * step_delta)
                    stage_sigma = float(_dpm_sigma_from_time(stage_time).item())
                    state_stage = state_base - stage_sigma * stage_multiplier * epsilon_base
                    eps_stage_payload = _dpm_cached_eps(
                        eps_cache,
                        cache_key="epsilon_stage_1",
                        latent_state=state_stage,
                        time_value=stage_time,
                        current_step=current_step,
                        step_total=step_total,
                    )
                    epsilon_stage = eps_stage_payload["epsilon"]
                    base_multiplier = math.expm1(step_delta)
                    return (
                        state_base
                        - sigma_next_value * base_multiplier * epsilon_base
                        - sigma_next_value
                        / (2.0 * stage_ratio)
                        * base_multiplier
                        * (epsilon_stage - epsilon_base)
                    )

                def _dpm_solver_step_3(
                    latent_state: torch.Tensor,
                    *,
                    time_current: torch.Tensor,
                    time_next: torch.Tensor,
                    stage_ratio_1: float,
                    stage_ratio_2: float,
                    eps_cache: dict[str, dict[str, torch.Tensor]],
                    current_step: int,
                    step_total: int | None,
                ) -> torch.Tensor:
                    if stage_ratio_1 == 0.0 or stage_ratio_2 == 0.0:
                        raise RuntimeError("DPM solver step-3 received zero stage ratio.")
                    step_delta = float((time_next - time_current).item())
                    if abs(step_delta) <= 1e-12:
                        return latent_state
                    sigma_next_value = float(_dpm_sigma_from_time(time_next).item())
                    eps_payload = _dpm_cached_eps(
                        eps_cache,
                        cache_key="epsilon",
                        latent_state=latent_state,
                        time_value=time_current,
                        current_step=current_step,
                        step_total=step_total,
                    )
                    state_base = eps_payload["model_input"]
                    epsilon_base = eps_payload["epsilon"]

                    stage_time_1 = time_current + stage_ratio_1 * (time_next - time_current)
                    stage_multiplier_1 = math.expm1(stage_ratio_1 * step_delta)
                    stage_sigma_1 = float(_dpm_sigma_from_time(stage_time_1).item())
                    state_stage_1 = state_base - stage_sigma_1 * stage_multiplier_1 * epsilon_base
                    eps_stage_1_payload = _dpm_cached_eps(
                        eps_cache,
                        cache_key="epsilon_stage_1",
                        latent_state=state_stage_1,
                        time_value=stage_time_1,
                        current_step=current_step,
                        step_total=step_total,
                    )
                    epsilon_stage_1 = eps_stage_1_payload["epsilon"]

                    stage_time_2 = time_current + stage_ratio_2 * (time_next - time_current)
                    stage_multiplier_2 = math.expm1(stage_ratio_2 * step_delta)
                    stage_sigma_2 = float(_dpm_sigma_from_time(stage_time_2).item())
                    denominator = stage_ratio_2 * step_delta
                    if abs(denominator) <= 1e-12:
                        raise RuntimeError(
                            "DPM solver step-3 encountered a collapsed stage denominator "
                            f"(ratio={stage_ratio_2}, delta={step_delta})."
                        )
                    stage_correction = (
                        (stage_multiplier_2 / denominator) - 1.0
                    )
                    state_stage_2 = (
                        state_base
                        - stage_sigma_2 * stage_multiplier_2 * epsilon_base
                        - stage_sigma_2
                        * (stage_ratio_2 / stage_ratio_1)
                        * stage_correction
                        * (epsilon_stage_1 - epsilon_base)
                    )
                    eps_stage_2_payload = _dpm_cached_eps(
                        eps_cache,
                        cache_key="epsilon_stage_2",
                        latent_state=state_stage_2,
                        time_value=stage_time_2,
                        current_step=current_step,
                        step_total=step_total,
                    )
                    epsilon_stage_2 = eps_stage_2_payload["epsilon"]
                    base_expm1 = math.expm1(step_delta)
                    slope_correction = (base_expm1 / step_delta) - 1.0
                    return (
                        state_base
                        - sigma_next_value * base_expm1 * epsilon_base
                        - sigma_next_value
                        / stage_ratio_2
                        * slope_correction
                        * (epsilon_stage_2 - epsilon_base)
                    )

                def _run_native_dpm_fast(latent_state: torch.Tensor, *, start_time: float) -> tuple[torch.Tensor, float]:
                    solver_eta = 0.0
                    solver_s_noise = 1.0
                    sigma_min_value, sigma_max_value = _dpm_select_solver_sigma_bounds()
                    step_budget = int(run_total_steps)
                    if step_budget < 1:
                        raise RuntimeError("DPM fast requires run_total_steps >= 1.")

                    time_start = float(
                        _dpm_time_from_sigma(
                            torch.tensor(sigma_max_value, dtype=torch.float32, device=latent_state.device)
                        ).item()
                    )
                    time_end = float(
                        _dpm_time_from_sigma(
                            torch.tensor(sigma_min_value, dtype=torch.float32, device=latent_state.device)
                        ).item()
                    )
                    macro_steps = math.floor(step_budget / 3) + 1
                    time_schedule = torch.linspace(
                        time_start,
                        time_end,
                        macro_steps + 1,
                        dtype=torch.float32,
                        device=latent_state.device,
                    )
                    if step_budget % 3 == 0:
                        solver_orders = [3] * (macro_steps - 2) + [2, 1]
                    else:
                        solver_orders = [3] * (macro_steps - 1) + [step_budget % 3]
                    if len(solver_orders) != macro_steps:
                        raise RuntimeError(
                            "DPM fast internal order plan mismatch "
                            f"(orders={len(solver_orders)} macro_steps={macro_steps})."
                        )

                    progress_cursor = 0
                    for macro_index, solver_order in enumerate(solver_orders):
                        if backend_state.should_stop:
                            raise _SamplingCancelled("cancelled")
                        time_current = time_schedule[macro_index]
                        time_next = time_schedule[macro_index + 1]
                        time_next_solver = time_next
                        sigma_up = 0.0

                        if solver_eta != 0.0:
                            sigma_current_value = float(_dpm_sigma_from_time(time_current).item())
                            sigma_next_value = float(_dpm_sigma_from_time(time_next).item())
                            sigma_down_value = sigma_next_value
                            sigma_up = min(
                                sigma_next_value,
                                solver_eta
                                * math.sqrt(
                                    max(
                                        sigma_next_value**2
                                        * max(sigma_current_value**2 - sigma_next_value**2, 0.0)
                                        / max(sigma_current_value**2, 1e-8),
                                        0.0,
                                    )
                                ),
                            )
                            sigma_down_squared = sigma_next_value**2 - sigma_up**2
                            if sigma_down_squared < 0.0:
                                raise RuntimeError(
                                    "DPM fast produced negative sigma_down^2 "
                                    f"(sigma_current={sigma_current_value}, sigma_next={sigma_next_value}, sigma_up={sigma_up})."
                                )
                            sigma_down_value = math.sqrt(sigma_down_squared)
                            time_next_solver = torch.minimum(
                                time_schedule[-1],
                                _dpm_time_from_sigma(
                                    torch.tensor(
                                        sigma_down_value,
                                        dtype=torch.float32,
                                        device=latent_state.device,
                                    )
                                ),
                            )
                            sigma_next_solver = float(_dpm_sigma_from_time(time_next_solver).item())
                            sigma_up_sq = sigma_next_value**2 - sigma_next_solver**2
                            if sigma_up_sq < -1e-8:
                                raise RuntimeError(
                                    "DPM fast produced negative sigma_up^2 after time conversion "
                                    f"(sigma_next={sigma_next_value}, sigma_next_solver={sigma_next_solver})."
                                )
                            sigma_up = math.sqrt(max(sigma_up_sq, 0.0))

                        eps_cache: dict[str, dict[str, torch.Tensor]] = {}
                        callback_step = max(1, min(progress_cursor + 1, run_total_steps))
                        callback_payload = _dpm_cached_eps(
                            eps_cache,
                            cache_key="epsilon",
                            latent_state=latent_state,
                            time_value=time_current,
                            current_step=callback_step,
                            step_total=run_total_steps,
                        )
                        callback_denoised = callback_payload["denoised"]

                        if solver_order == 1:
                            latent_state = _dpm_solver_step_1(
                                latent_state,
                                time_current=time_current,
                                time_next=time_next_solver,
                                eps_cache=eps_cache,
                                current_step=callback_step,
                                step_total=run_total_steps,
                            )
                        elif solver_order == 2:
                            latent_state = _dpm_solver_step_2(
                                latent_state,
                                time_current=time_current,
                                time_next=time_next_solver,
                                stage_ratio=0.5,
                                eps_cache=eps_cache,
                                current_step=callback_step,
                                step_total=run_total_steps,
                            )
                        elif solver_order == 3:
                            latent_state = _dpm_solver_step_3(
                                latent_state,
                                time_current=time_current,
                                time_next=time_next_solver,
                                stage_ratio_1=1.0 / 3.0,
                                stage_ratio_2=2.0 / 3.0,
                                eps_cache=eps_cache,
                                current_step=callback_step,
                                step_total=run_total_steps,
                            )
                        else:
                            raise RuntimeError(f"DPM fast produced unsupported solver order={solver_order}.")

                        if sigma_up != 0.0:
                            raise RuntimeError(
                                "DPM fast stochastic eta path is unsupported without explicit deterministic noise contract."
                            )
                        if solver_s_noise < 0.0:
                            raise RuntimeError(f"DPM fast received invalid s_noise={solver_s_noise}.")
                        progress_cursor = min(run_total_steps, progress_cursor + int(solver_order))
                        sigma_current_event = float(_dpm_sigma_from_time(time_current).item())
                        sigma_next_event = float(_dpm_sigma_from_time(time_next).item())
                        start_time = _dpm_emit_step_callbacks(
                            latent_state=latent_state,
                            denoised_state=callback_denoised,
                            current_step=max(1, progress_cursor),
                            sigma_current=sigma_current_event,
                            sigma_next=sigma_next_event,
                            step_start_time=start_time,
                        )

                    return latent_state, start_time

                def _run_native_dpm_adaptive(
                    latent_state: torch.Tensor,
                    *,
                    start_time: float,
                ) -> tuple[torch.Tensor, float]:
                    solver_order = 3
                    solver_rtol = 0.05
                    solver_atol = 0.0078
                    solver_h_init = 0.05
                    solver_pcoeff = 0.0
                    solver_icoeff = 1.0
                    solver_dcoeff = 0.0
                    solver_accept_safety = 0.81
                    solver_eta = 0.0
                    solver_s_noise = 1.0

                    sigma_min_value, sigma_max_value = _dpm_select_solver_sigma_bounds()
                    time_start = float(
                        _dpm_time_from_sigma(
                            torch.tensor(sigma_max_value, dtype=torch.float32, device=latent_state.device)
                        ).item()
                    )
                    time_end = float(
                        _dpm_time_from_sigma(
                            torch.tensor(sigma_min_value, dtype=torch.float32, device=latent_state.device)
                        ).item()
                    )
                    forward_direction = time_end > time_start
                    if not forward_direction and solver_eta != 0.0:
                        raise RuntimeError("DPM adaptive requires eta=0 for reverse sampling.")
                    step_size = abs(float(solver_h_init)) * (1.0 if forward_direction else -1.0)
                    if not math.isfinite(step_size) or step_size == 0.0:
                        raise RuntimeError(
                            "DPM adaptive requires a finite non-zero initial step size "
                            f"(h_init={solver_h_init})."
                        )
                    if solver_order not in {2, 3}:
                        raise RuntimeError(f"DPM adaptive requires solver_order in {{2,3}}; got {solver_order}.")
                    if solver_rtol <= 0.0 or solver_atol <= 0.0:
                        raise RuntimeError(
                            "DPM adaptive requires positive tolerances "
                            f"(rtol={solver_rtol}, atol={solver_atol})."
                        )
                    if solver_accept_safety <= 0.0:
                        raise RuntimeError(
                            "DPM adaptive requires accept_safety > 0 "
                            f"(got {solver_accept_safety})."
                        )
                    if solver_s_noise < 0.0:
                        raise RuntimeError(f"DPM adaptive received invalid s_noise={solver_s_noise}.")

                    pid_weight_1 = (solver_pcoeff + solver_icoeff + solver_dcoeff) / float(solver_order)
                    pid_weight_2 = -(solver_pcoeff + 2.0 * solver_dcoeff) / float(solver_order)
                    pid_weight_3 = solver_dcoeff / float(solver_order)
                    pid_errors: list[float] = []
                    pid_epsilon = 1e-8

                    def _pid_limiter(raw_factor: float) -> float:
                        return 1.0 + math.atan(raw_factor - 1.0)

                    def _pid_propose_step(error_value: float) -> bool:
                        nonlocal step_size
                        if not math.isfinite(error_value) or error_value < 0.0:
                            raise RuntimeError(
                                f"DPM adaptive received invalid error estimate: {error_value}."
                            )
                        inverse_error = 1.0 / (error_value + pid_epsilon)
                        if not pid_errors:
                            pid_errors.extend([inverse_error, inverse_error, inverse_error])
                        pid_errors[0] = inverse_error
                        raw_factor = (
                            pid_errors[0] ** pid_weight_1
                            * pid_errors[1] ** pid_weight_2
                            * pid_errors[2] ** pid_weight_3
                        )
                        limited_factor = _pid_limiter(raw_factor)
                        if not math.isfinite(limited_factor) or limited_factor <= 0.0:
                            raise RuntimeError(
                                f"DPM adaptive PID produced invalid step factor: {limited_factor}."
                            )
                        accept_step = limited_factor >= solver_accept_safety
                        if accept_step:
                            pid_errors[2] = pid_errors[1]
                            pid_errors[1] = pid_errors[0]
                        step_size *= limited_factor
                        if not math.isfinite(step_size) or step_size == 0.0:
                            raise RuntimeError(
                                f"DPM adaptive PID produced invalid next step size: {step_size}."
                            )
                        return accept_step

                    current_time = time_start
                    previous_low_order_state = latent_state
                    accepted_steps = 0
                    iteration_count = 0
                    estimated_iterations = max(
                        1,
                        int(math.ceil(abs(time_end - current_time) / max(abs(step_size), 1e-8))),
                    )
                    max_iterations = max(64, estimated_iterations * 64)

                    while (
                        current_time < time_end - 1e-5
                        if forward_direction
                        else current_time > time_end + 1e-5
                    ):
                        iteration_count += 1
                        if iteration_count > max_iterations:
                            raise RuntimeError(
                                "DPM adaptive exceeded the iteration safety limit "
                                f"(iterations={iteration_count}, max={max_iterations})."
                            )
                        if backend_state.should_stop:
                            raise _SamplingCancelled("cancelled")

                        proposed_time = (
                            min(time_end, current_time + step_size)
                            if forward_direction
                            else max(time_end, current_time + step_size)
                        )
                        time_current = torch.tensor(
                            current_time,
                            dtype=torch.float32,
                            device=latent_state.device,
                        )
                        time_next = torch.tensor(
                            proposed_time,
                            dtype=torch.float32,
                            device=latent_state.device,
                        )
                        time_next_solver = time_next
                        sigma_up = 0.0

                        if solver_eta != 0.0:
                            sigma_current_value = float(_dpm_sigma_from_time(time_current).item())
                            sigma_next_value = float(_dpm_sigma_from_time(time_next).item())
                            sigma_down_value = sigma_next_value
                            sigma_up = min(
                                sigma_next_value,
                                solver_eta
                                * math.sqrt(
                                    max(
                                        sigma_next_value**2
                                        * max(sigma_current_value**2 - sigma_next_value**2, 0.0)
                                        / max(sigma_current_value**2, 1e-8),
                                        0.0,
                                    )
                                ),
                            )
                            sigma_down_squared = sigma_next_value**2 - sigma_up**2
                            if sigma_down_squared < 0.0:
                                raise RuntimeError(
                                    "DPM adaptive produced negative sigma_down^2 "
                                    f"(sigma_current={sigma_current_value}, sigma_next={sigma_next_value}, sigma_up={sigma_up})."
                                )
                            sigma_down_value = math.sqrt(sigma_down_squared)
                            time_next_solver = torch.minimum(
                                torch.tensor(time_end, dtype=torch.float32, device=latent_state.device),
                                _dpm_time_from_sigma(
                                    torch.tensor(
                                        sigma_down_value,
                                        dtype=torch.float32,
                                        device=latent_state.device,
                                    )
                                ),
                            )
                            sigma_next_solver = float(_dpm_sigma_from_time(time_next_solver).item())
                            sigma_up_sq = sigma_next_value**2 - sigma_next_solver**2
                            if sigma_up_sq < -1e-8:
                                raise RuntimeError(
                                    "DPM adaptive produced negative sigma_up^2 after time conversion "
                                    f"(sigma_next={sigma_next_value}, sigma_next_solver={sigma_next_solver})."
                                )
                            sigma_up = math.sqrt(max(sigma_up_sq, 0.0))

                        eps_cache: dict[str, dict[str, torch.Tensor]] = {}
                        callback_step = max(1, accepted_steps + 1)
                        callback_payload = _dpm_cached_eps(
                            eps_cache,
                            cache_key="epsilon",
                            latent_state=latent_state,
                            time_value=time_current,
                            current_step=callback_step,
                            step_total=reported_total_steps,
                        )
                        callback_denoised = callback_payload["denoised"]

                        if solver_order == 2:
                            low_order_state = _dpm_solver_step_1(
                                latent_state,
                                time_current=time_current,
                                time_next=time_next_solver,
                                eps_cache=eps_cache,
                                current_step=callback_step,
                                step_total=reported_total_steps,
                            )
                            high_order_state = _dpm_solver_step_2(
                                latent_state,
                                time_current=time_current,
                                time_next=time_next_solver,
                                stage_ratio=0.5,
                                eps_cache=eps_cache,
                                current_step=callback_step,
                                step_total=reported_total_steps,
                            )
                        else:
                            low_order_state = _dpm_solver_step_2(
                                latent_state,
                                time_current=time_current,
                                time_next=time_next_solver,
                                stage_ratio=1.0 / 3.0,
                                eps_cache=eps_cache,
                                current_step=callback_step,
                                step_total=reported_total_steps,
                            )
                            high_order_state = _dpm_solver_step_3(
                                latent_state,
                                time_current=time_current,
                                time_next=time_next_solver,
                                stage_ratio_1=1.0 / 3.0,
                                stage_ratio_2=2.0 / 3.0,
                                eps_cache=eps_cache,
                                current_step=callback_step,
                                step_total=reported_total_steps,
                            )

                        atol_tensor = torch.full_like(low_order_state, solver_atol)
                        error_denominator = torch.maximum(
                            atol_tensor,
                            solver_rtol * torch.maximum(low_order_state.abs(), previous_low_order_state.abs()),
                        )
                        if bool(torch.any(error_denominator <= 0.0)):
                            raise RuntimeError("DPM adaptive produced non-positive error denominator.")
                        error_tensor = torch.linalg.norm((low_order_state - high_order_state) / error_denominator)
                        error_value = float(error_tensor.item()) / math.sqrt(float(latent_state.numel()))
                        accept_step = _pid_propose_step(error_value)

                        if accept_step:
                            previous_low_order_state = low_order_state
                            latent_state = high_order_state
                            if sigma_up != 0.0:
                                raise RuntimeError(
                                    "DPM adaptive stochastic eta path is unsupported without explicit deterministic noise contract."
                                )
                            current_time = proposed_time
                            accepted_steps += 1
                            sigma_current_event = float(_dpm_sigma_from_time(time_current).item())
                            sigma_next_event = float(_dpm_sigma_from_time(time_next).item())
                            start_time = _dpm_emit_step_callbacks(
                                latent_state=latent_state,
                                denoised_state=callback_denoised,
                                current_step=max(1, accepted_steps),
                                sigma_current=sigma_current_event,
                                sigma_next=sigma_next_event,
                                step_start_time=start_time,
                            )

                    if accepted_steps == 0:
                        sigma_terminal = float(sigmas_run[min(1, len(sigmas_run) - 1)])
                        start_time = _dpm_emit_step_callbacks(
                            latent_state=latent_state,
                            denoised_state=latent_state,
                            current_step=1,
                            sigma_current=float(sigmas_run[0]),
                            sigma_next=sigma_terminal,
                            step_start_time=start_time,
                        )
                    return latent_state, start_time

                def _run_native_restart(
                    latent_state: torch.Tensor,
                    *,
                    start_time: float,
                ) -> tuple[torch.Tensor, float]:
                    if restart_step_plan is None:
                        raise RuntimeError("Restart runner missing execution plan.")
                    restart_s_noise = 1.0
                    restart_step_rng: ImageRNG | None = None
                    if any(step.renoise_scale > 0.0 for step in restart_step_plan):
                        restart_step_rng = self._build_seeded_step_rng(
                            processing=processing,
                            active_context=active_context,
                            noise=base_noise,
                        )
                    for current_step, step_plan in enumerate(restart_step_plan, start=1):
                        if backend_state.should_stop:
                            raise _SamplingCancelled("cancelled")
                        if step_plan.renoise_scale > 0.0:
                            if restart_step_rng is None:
                                raise RuntimeError(
                                    "Restart execution planned renoise events without a deterministic step RNG."
                                )
                            restart_noise = self._next_seeded_step_noise(restart_step_rng, latent_state)
                            latent_state = latent_state + restart_noise * restart_s_noise * step_plan.renoise_scale
                        model_input, denoised_output, epsilon_output = _dpm_eval_eps(
                            latent_state,
                            sigma_value=step_plan.sigma_current,
                            current_step=current_step,
                            step_total=run_total_steps,
                        )
                        sigma_delta = step_plan.sigma_next - step_plan.sigma_current
                        if step_plan.sigma_next == 0.0:
                            latent_state = model_input + sigma_delta * epsilon_output
                        else:
                            euler_state = model_input + sigma_delta * epsilon_output
                            _, _, epsilon_candidate = _dpm_eval_eps(
                                euler_state,
                                sigma_value=step_plan.sigma_next,
                                current_step=current_step,
                                step_total=run_total_steps,
                            )
                            latent_state = model_input + 0.5 * sigma_delta * (epsilon_output + epsilon_candidate)
                        start_time = _dpm_emit_step_callbacks(
                            latent_state=latent_state,
                            denoised_state=denoised_output,
                            current_step=current_step,
                            sigma_current=step_plan.sigma_current,
                            sigma_next=step_plan.sigma_next,
                            step_start_time=start_time,
                        )
                    return latent_state, start_time

                with profiler.profile_run(profile_name, meta=profile_meta):
                    for i in range(start_idx, steps):
                        if backend_state.should_stop:
                            raise _SamplingCancelled("cancelled")
                        run_step_index = i - start_idx
                        history_step_index = i if exact_resume_active else run_step_index
                        if sampler_kind in {SamplerKind.DPM_FAST, SamplerKind.DPM_ADAPTIVE, SamplerKind.RESTART}:
                            if run_step_index != 0:
                                break
                            if sampler_kind is SamplerKind.DPM_FAST:
                                x, t0 = _run_native_dpm_fast(x, start_time=t0)
                            elif sampler_kind is SamplerKind.DPM_ADAPTIVE:
                                x, t0 = _run_native_dpm_adaptive(x, start_time=t0)
                            else:
                                x, t0 = _run_native_restart(x, start_time=t0)
                            break
                        backend_state.reset_sampling_blocks(owner_token=progress_owner_token)
                        if guidance_policy is not None:
                            denoiser.model_options[_GUIDANCE_STEP_INDEX_KEY] = history_step_index
                        with profiler.section(f"sampling.step/{history_step_index + 1}"):
                            sigma = sigmas[i]
                            sigma_next = sigmas[i + 1]
                            if sampler_kind is SamplerKind.SEEDS_2:
                                if seeds2_sigmas is None:
                                    raise RuntimeError("SEEDS-2 missing adjusted sigma schedule.")
                                sigma = seeds2_sigmas[history_step_index]
                                sigma_next = seeds2_sigmas[history_step_index + 1]
                            if sampler_kind is SamplerKind.SEEDS_3:
                                if seeds3_sigmas is None:
                                    raise RuntimeError("SEEDS-3 missing adjusted sigma schedule.")
                                sigma = seeds3_sigmas[history_step_index]
                                sigma_next = seeds3_sigmas[history_step_index + 1]
                            if sampler_kind in {SamplerKind.SA_SOLVER, SamplerKind.SA_SOLVER_PECE}:
                                if sa_solver_sigmas is None:
                                    raise RuntimeError("SA-Solver missing adjusted sigma schedule.")
                                sigma = sa_solver_sigmas[history_step_index]
                                sigma_next = sa_solver_sigmas[history_step_index + 1]
                            if sampler_kind is SamplerKind.DPMPP_SDE:
                                if dpm_sde_sigmas is None:
                                    raise RuntimeError("DPM++ SDE missing adjusted sigma schedule.")
                                sigma = dpm_sde_sigmas[history_step_index]
                                sigma_next = dpm_sde_sigmas[history_step_index + 1]
                            if sampler_kind is SamplerKind.DPM3M_SDE:
                                if dpm3m_sde_sigmas is None:
                                    raise RuntimeError("DPM++ 3M SDE missing adjusted sigma schedule.")
                                sigma = dpm3m_sde_sigmas[history_step_index]
                                sigma_next = dpm3m_sde_sigmas[history_step_index + 1]
                            if sampler_kind in {
                                SamplerKind.RES_MULTISTEP_CFG_PP,
                                SamplerKind.RES_MULTISTEP_ANCESTRAL_CFG_PP,
                            }:
                                res_multistep_cfg_pp_capture = None
                            if sampler_kind is SamplerKind.GRADIENT_ESTIMATION_CFG_PP:
                                gradient_estimation_cfg_pp_capture = None
                            if sampler_kind in {SamplerKind.EULER_CFG_PP, SamplerKind.EULER_A_CFG_PP}:
                                euler_cfg_pp_capture = None
                            if sampler_kind is SamplerKind.DPM2M_CFG_PP:
                                dpm2m_cfg_pp_capture = None
                            if sampler_kind is SamplerKind.DPM2S_ANCESTRAL_CFG_PP:
                                dpm2s_ancestral_cfg_pp_capture = None
                            sigma_batch = torch.full((x.shape[0],), float(sigma), device=x.device, dtype=torch.float32)
                            current_step = history_step_index + 1
                            step_index = history_step_index

                            if pre_denoiser_hook is not None:
                                x = pre_denoiser_hook(x, sigma_batch, current_step, run_total_steps)
                                if not isinstance(x, torch.Tensor):
                                    raise RuntimeError("pre_denoiser_hook must return a torch.Tensor")
                                if tuple(x.shape) != tuple(noise.shape):
                                    raise RuntimeError(
                                        "pre_denoiser_hook returned unexpected shape "
                                        f"{tuple(x.shape)}; expected {tuple(noise.shape)}"
                                    )

                            if log_cfg_delta and (i - start_idx) < cfg_delta_steps:
                                denoised, cond_pred, uncond_pred = sampling_function_inner(
                                    model,
                                    x,
                                    sigma_batch,
                                    compiled_uncond,
                                    compiled_cond,
                                    cfg_scale,
                                    denoiser.model_options,
                                    seed=None,
                                    return_full=True,
                                )
                                cfg1_optimization = math.isclose(cfg_scale, 1.0) and not denoiser.model_options.get(
                                    "disable_cfg1_optimization", False
                                )
                                if compiled_uncond is None or cfg1_optimization:
                                    self._emit_event(
                                        "sampling.cfg_delta",
                                        step=i + 1,
                                        total_steps=steps,
                                        sigma=float(sigma),
                                        cfg_scale=float(cfg_scale),
                                        uncond_used=False,
                                    )
                                else:
                                    try:
                                        delta_abs_mean = float((cond_pred - uncond_pred).detach().float().abs().mean().item())
                                    except Exception:
                                        delta_abs_mean = float("nan")
                                    self._emit_event(
                                        "sampling.cfg_delta",
                                        step=i + 1,
                                        total_steps=steps,
                                        sigma=float(sigma),
                                        cfg_scale=float(cfg_scale),
                                        delta_abs_mean=delta_abs_mean,
                                    )
                            else:
                                denoised = sampling_function_inner(
                                    model,
                                    x,
                                    sigma_batch,
                                    compiled_uncond,
                                    compiled_cond,
                                    cfg_scale,
                                    denoiser.model_options,
                                    seed=None,
                                    return_full=False,
                                )

                            if post_denoiser_hook is not None:
                                denoised = post_denoiser_hook(denoised, sigma_batch, current_step, run_total_steps)
                                if not isinstance(denoised, torch.Tensor):
                                    raise RuntimeError("post_denoiser_hook must return a torch.Tensor")
                                if tuple(denoised.shape) != tuple(x.shape):
                                    raise RuntimeError(
                                        "post_denoiser_hook returned unexpected shape "
                                        f"{tuple(denoised.shape)}; expected {tuple(x.shape)}"
                                    )

                            eps_source = denoised
                            if sampler_kind is SamplerKind.GRADIENT_ESTIMATION_CFG_PP:
                                if gradient_estimation_cfg_pp_capture is None:
                                    raise RuntimeError(
                                        "Gradient estimation cfg++ requires uncond_denoised capture, "
                                        "but the sampler post-CFG hook did not run."
                                    )
                                if tuple(gradient_estimation_cfg_pp_capture.shape) != tuple(x.shape):
                                    raise RuntimeError(
                                        "Gradient estimation cfg++ captured unexpected uncond_denoised shape "
                                        f"{tuple(gradient_estimation_cfg_pp_capture.shape)}; expected {tuple(x.shape)}"
                                    )
                                eps_source = gradient_estimation_cfg_pp_capture
                            eps = (x - eps_source) / max(float(sigma), 1e-8)
                            if strict and (torch.isnan(eps).any() or torch.isnan(denoised).any()):
                                reason = f"NaN detected at sampling step {i + 1}"
                                emit_backend_message(
                                    "NaN encountered during sampling; attempting precision fallback.",
                                    logger=self._logger_name,
                                    level="WARNING",
                                    step=i + 1,
                                    dtype=str(getattr(model, "computation_dtype", x.dtype)),
                                )
                                next_dtype = memory_management.manager.report_precision_failure(
                                    DeviceRole.CORE,
                                    location=f"sampler.step_{i + 1}",
                                    reason=reason,
                                )
                                if next_dtype is None:
                                    hint = memory_management.manager.precision_hint(DeviceRole.CORE)
                                    raise RuntimeError(
                                        f"Diffusion core produced NaNs at step {i + 1} on {noise.device} with dtype {getattr(model, 'computation_dtype', x.dtype)}. {hint}"
                                    )
                                self._rebind_unet_precision(next_dtype)
                                retry = True
                                raise _PrecisionFallbackRequest

                            eps_history.append(eps.detach())
                            if len(eps_history) > 4:
                                eps_history.pop(0)

                            def _evaluate_step_candidate(
                                x_candidate: torch.Tensor,
                                *,
                                sigma_candidate: float,
                            ) -> tuple[torch.Tensor, torch.Tensor]:
                                sigma_candidate_batch = torch.full(
                                    (x.shape[0],),
                                    sigma_candidate,
                                    device=x.device,
                                    dtype=torch.float32,
                                )
                                candidate_input = x_candidate
                                if pre_denoiser_hook is not None:
                                    candidate_input = pre_denoiser_hook(
                                        candidate_input,
                                        sigma_candidate_batch,
                                        current_step,
                                        run_total_steps,
                                    )
                                    if not isinstance(candidate_input, torch.Tensor):
                                        raise RuntimeError("pre_denoiser_hook must return a torch.Tensor")
                                    if tuple(candidate_input.shape) != tuple(x.shape):
                                        raise RuntimeError(
                                            "pre_denoiser_hook returned unexpected shape "
                                            f"{tuple(candidate_input.shape)}; expected {tuple(x.shape)}"
                                        )
                                denoised_candidate = sampling_function_inner(
                                    model,
                                    candidate_input,
                                    sigma_candidate_batch,
                                    compiled_uncond,
                                    compiled_cond,
                                    cfg_scale,
                                    denoiser.model_options,
                                    seed=None,
                                    return_full=False,
                                )
                                if post_denoiser_hook is not None:
                                    denoised_candidate = post_denoiser_hook(
                                        denoised_candidate,
                                        sigma_candidate_batch,
                                        current_step,
                                        run_total_steps,
                                    )
                                    if not isinstance(denoised_candidate, torch.Tensor):
                                        raise RuntimeError("post_denoiser_hook must return a torch.Tensor")
                                    if tuple(denoised_candidate.shape) != tuple(candidate_input.shape):
                                        raise RuntimeError(
                                            "post_denoiser_hook returned unexpected shape "
                                            f"{tuple(denoised_candidate.shape)}; expected {tuple(candidate_input.shape)}"
                                        )
                                return candidate_input, denoised_candidate

                            if sampler_kind is SamplerKind.EULER:
                                x = x - (float(sigma) - float(sigma_next)) * eps
                            elif sampler_kind is SamplerKind.EULER_CFG_PP:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if euler_cfg_pp_capture is None:
                                    raise RuntimeError(
                                        "Euler cfg++ requires uncond_denoised capture, "
                                        "but the sampler post-CFG hook did not run."
                                    )
                                if tuple(euler_cfg_pp_capture.shape) != tuple(x.shape):
                                    raise RuntimeError(
                                        "Euler cfg++ captured unexpected uncond_denoised shape "
                                        f"{tuple(euler_cfg_pp_capture.shape)}; expected {tuple(x.shape)}"
                                    )
                                if sigma_next_f <= 0.0:
                                    x = denoised
                                else:
                                    alpha_s = _cfg_pp_alpha_from_sigma(sigma_f)
                                    alpha_t = _cfg_pp_alpha_from_sigma(sigma_next_f)
                                    d = (x - alpha_s * euler_cfg_pp_capture) / max(sigma_f, 1e-8)
                                    x = alpha_t * denoised + sigma_next_f * d
                            elif sampler_kind is SamplerKind.EULER_A:
                                sigma = float(sigma)
                                sigma_next = float(sigma_next)
                                if prediction_type == "const":
                                    if sigma_next <= 0.0:
                                        x = denoised
                                    else:
                                        if sigma <= 0.0:
                                            raise RuntimeError(
                                                "Euler ancestral RF/CONST requires sigma > 0 before the terminal step."
                                            )
                                        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * 1.0
                                        sigma_down = sigma_next * downstep_ratio
                                        alpha_ip1 = 1.0 - sigma_next
                                        alpha_down = 1.0 - sigma_down
                                        if abs(alpha_down) <= 1e-12:
                                            raise RuntimeError(
                                                "Euler ancestral RF/CONST produced alpha_down=0; cannot compute renoise term."
                                            )
                                        renoise_sq = sigma_next**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2
                                        if renoise_sq < -1e-12:
                                            raise RuntimeError(
                                                "Euler ancestral RF/CONST produced negative renoise variance."
                                            )
                                        renoise_coeff = max(renoise_sq, 0.0) ** 0.5
                                        sigma_down_i_ratio = sigma_down / sigma
                                        x = sigma_down_i_ratio * x + (1.0 - sigma_down_i_ratio) * denoised
                                        if seeded_step_rng is None:
                                            raise RuntimeError("Euler ancestral RF/CONST missing deterministic noise RNG.")
                                        x = (
                                            (alpha_ip1 / alpha_down) * x
                                            + self._next_seeded_step_noise(seeded_step_rng, x) * 1.0 * renoise_coeff
                                        )
                                elif sigma_next <= 0.0:
                                    x = denoised
                                else:
                                    sigma_up_sq = max(sigma_next**2 * (sigma**2 - sigma_next**2) / max(sigma**2, 1e-8), 0.0)
                                    sigma_up = sigma_up_sq ** 0.5
                                    sigma_down = (max(sigma_next**2 - sigma_up_sq, 0.0)) ** 0.5
                                    x = denoised + sigma_down * eps
                                    if seeded_step_rng is None:
                                        raise RuntimeError("Euler ancestral missing deterministic noise RNG.")
                                    noise = self._next_seeded_step_noise(seeded_step_rng, x)
                                    x = x + sigma_up * noise
                            elif sampler_kind is SamplerKind.EULER_A_CFG_PP:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if euler_cfg_pp_capture is None:
                                    raise RuntimeError(
                                        "Euler ancestral cfg++ requires uncond_denoised capture, "
                                        "but the sampler post-CFG hook did not run."
                                    )
                                if tuple(euler_cfg_pp_capture.shape) != tuple(x.shape):
                                    raise RuntimeError(
                                        "Euler ancestral cfg++ captured unexpected uncond_denoised shape "
                                        f"{tuple(euler_cfg_pp_capture.shape)}; expected {tuple(x.shape)}"
                                    )
                                if sigma_next_f <= 0.0:
                                    x = denoised
                                else:
                                    alpha_s = _cfg_pp_alpha_from_sigma(sigma_f)
                                    alpha_t = _cfg_pp_alpha_from_sigma(sigma_next_f)
                                    d = (x - alpha_s * euler_cfg_pp_capture) / max(sigma_f, 1e-8)
                                    sigma_down_ratio, sigma_up_ratio = _compute_ancestral_sigmas(
                                        sigma_f / alpha_s,
                                        sigma_next_f / alpha_t,
                                        eta=1.0,
                                    )
                                    sigma_down = alpha_t * sigma_down_ratio
                                    x = alpha_t * denoised + sigma_down * d
                                    if seeded_step_rng is None:
                                        raise RuntimeError("Euler ancestral cfg++ missing deterministic noise RNG.")
                                    x = x + alpha_t * self._next_seeded_step_noise(seeded_step_rng, x) * sigma_up_ratio
                            elif sampler_kind is SamplerKind.HEUN:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                dt = sigma_next_f - sigma_f
                                if sigma_next_f <= 0.0:
                                    x = x + eps * dt
                                else:
                                    x_2 = x + eps * dt
                                    x_2_eval, denoised_2 = _evaluate_step_candidate(
                                        x_2,
                                        sigma_candidate=sigma_next_f,
                                    )
                                    d_2 = (x_2_eval - denoised_2) / max(sigma_next_f, 1e-8)
                                    x = x + 0.5 * (eps + d_2) * dt
                            elif sampler_kind is SamplerKind.HEUNPP2:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                local_step_index = step_index
                                final_step_index = int(sigmas_run.numel()) - 1
                                if final_step_index < 1:
                                    raise RuntimeError("HEUNPP2 requires at least one sigma transition.")
                                if sigma_f <= 0.0:
                                    raise RuntimeError(
                                        f"HEUNPP2 requires a strictly positive current sigma; got sigma={sigma_f}."
                                    )
                                dt = sigma_next_f - sigma_f
                                if local_step_index + 1 == final_step_index:
                                    x = x + eps * dt
                                elif local_step_index + 2 == final_step_index:
                                    if sigma_next_f <= 0.0:
                                        raise RuntimeError(
                                            "HEUNPP2 penultimate step requires a positive intermediate sigma before the terminal zero."
                                        )
                                    x_2 = x + eps * dt
                                    x_2_eval, denoised_2 = _evaluate_step_candidate(
                                        x_2,
                                        sigma_candidate=sigma_next_f,
                                    )
                                    d_2 = (x_2_eval - denoised_2) / max(sigma_next_f, 1e-8)
                                    weight_scale = 2.0 * float(sigmas_run[0])
                                    if not math.isfinite(weight_scale) or weight_scale <= 0.0:
                                        raise RuntimeError(
                                            f"HEUNPP2 produced invalid penultimate weight scale {weight_scale}."
                                        )
                                    w_2 = sigma_next_f / weight_scale
                                    w_1 = 1.0 - w_2
                                    x = x + (w_1 * eps + w_2 * d_2) * dt
                                else:
                                    sigma_next_next_f = float(sigmas_run[local_step_index + 2])
                                    if sigma_next_f <= 0.0 or sigma_next_next_f <= 0.0:
                                        raise RuntimeError(
                                            "HEUNPP2 ordinary steps require two positive lookahead sigmas before the terminal zero "
                                            f"(sigma_next={sigma_next_f}, sigma_next_next={sigma_next_next_f})."
                                        )
                                    x_2 = x + eps * dt
                                    x_2_eval, denoised_2 = _evaluate_step_candidate(
                                        x_2,
                                        sigma_candidate=sigma_next_f,
                                    )
                                    d_2 = (x_2_eval - denoised_2) / max(sigma_next_f, 1e-8)
                                    dt_2 = sigma_next_next_f - sigma_next_f
                                    x_3 = x_2 + d_2 * dt_2
                                    x_3_eval, denoised_3 = _evaluate_step_candidate(
                                        x_3,
                                        sigma_candidate=sigma_next_next_f,
                                    )
                                    d_3 = (x_3_eval - denoised_3) / max(sigma_next_next_f, 1e-8)
                                    weight_scale = 3.0 * float(sigmas_run[0])
                                    if not math.isfinite(weight_scale) or weight_scale <= 0.0:
                                        raise RuntimeError(
                                            f"HEUNPP2 produced invalid ordinary-step weight scale {weight_scale}."
                                        )
                                    w_2 = sigma_next_f / weight_scale
                                    w_3 = sigma_next_next_f / weight_scale
                                    w_1 = 1.0 - w_2 - w_3
                                    x = x + (w_1 * eps + w_2 * d_2 + w_3 * d_3) * dt
                            elif sampler_kind is SamplerKind.LMS:
                                if lms_sigmas_run is None:
                                    raise RuntimeError("LMS sigma ladder cache is not initialized.")
                                local_step_index = step_index
                                current_order = min(local_step_index + 1, _MAX_LMS_ORDER)
                                sigma_nodes = [
                                    float(lms_sigmas_run[local_step_index - offset]) for offset in range(current_order)
                                ]
                                sigma_target = float(lms_sigmas_run[local_step_index + 1])
                                coeffs = _compute_lms_coefficients(sigma_nodes, sigma_target)
                                derivatives = list(reversed(eps_history[-current_order:]))
                                x = x + sum(coeff * derivative for coeff, derivative in zip(coeffs, derivatives))
                            elif sampler_kind is SamplerKind.IPNDM:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if sigma_next_f <= 0.0:
                                    x = denoised
                                else:
                                    x = x + (sigma_next_f - sigma_f) * _compute_ipndm_derivative(eps_history)
                            elif sampler_kind is SamplerKind.IPNDM_V:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if sigma_next_f <= 0.0:
                                    x = denoised
                                else:
                                    local_step_index = step_index
                                    x = x + (sigma_next_f - sigma_f) * _compute_ipndm_v_derivative(
                                        sigmas_run,
                                        local_step_index,
                                        eps_history,
                                    )
                            elif sampler_kind is SamplerKind.DEIS:
                                if deis_coefficients is None:
                                    raise RuntimeError("DEIS coefficient cache is not initialized.")
                                local_step_index = step_index
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                order = min(3, local_step_index + 1)
                                if sigma_next_f <= 0.0:
                                    order = 1
                                if order == 1:
                                    x = x + (sigma_next_f - sigma_f) * eps
                                else:
                                    coeffs = deis_coefficients[local_step_index]
                                    if len(coeffs) != order:
                                        raise RuntimeError(
                                            "DEIS coefficient order mismatch "
                                            f"(step={local_step_index}, expected={order}, got={len(coeffs)})."
                                        )
                                    history_terms: list[torch.Tensor] = [eps]
                                    history_terms.extend(reversed(eps_history[:-1]))
                                    x = x + sum(coeff * term for coeff, term in zip(coeffs, history_terms))
                            elif sampler_kind in {
                                SamplerKind.RES_MULTISTEP,
                                SamplerKind.RES_MULTISTEP_CFG_PP,
                                SamplerKind.RES_MULTISTEP_ANCESTRAL,
                                SamplerKind.RES_MULTISTEP_ANCESTRAL_CFG_PP,
                            }:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                res_multistep_cfg_pp = sampler_kind in {
                                    SamplerKind.RES_MULTISTEP_CFG_PP,
                                    SamplerKind.RES_MULTISTEP_ANCESTRAL_CFG_PP,
                                }
                                res_multistep_eta = 1.0 if sampler_kind in {
                                    SamplerKind.RES_MULTISTEP_ANCESTRAL,
                                    SamplerKind.RES_MULTISTEP_ANCESTRAL_CFG_PP,
                                } else 0.0
                                sigma_down, sigma_up = _compute_ancestral_sigmas(
                                    sigma_f,
                                    sigma_next_f,
                                    eta=res_multistep_eta,
                                )
                                if sigma_down <= 0.0 or old_denoised is None or res_multistep_old_sigma_down is None:
                                    if res_multistep_cfg_pp:
                                        if res_multistep_cfg_pp_capture is None:
                                            raise RuntimeError(
                                                "Residual multistep cfg++ requires uncond_denoised capture, "
                                                "but the sampler post-CFG hook did not run."
                                            )
                                        if tuple(res_multistep_cfg_pp_capture.shape) != tuple(x.shape):
                                            raise RuntimeError(
                                                "Residual multistep cfg++ captured unexpected uncond_denoised shape "
                                                f"{tuple(res_multistep_cfg_pp_capture.shape)}; expected {tuple(x.shape)}"
                                            )
                                        uncond_eps = (x - res_multistep_cfg_pp_capture) / max(sigma_f, 1e-8)
                                        x = denoised + uncond_eps * sigma_down
                                    else:
                                        x = x + (sigma_down - sigma_f) * eps
                                else:
                                    previous_sigma = float(sigmas[i - 1])
                                    if sigma_f <= 0.0 or res_multistep_old_sigma_down <= 0.0 or previous_sigma <= 0.0:
                                        raise RuntimeError(
                                            "Residual multistep second-order update requires strictly positive "
                                            f"sigmas (sigma={sigma_f}, old_sigma_down={res_multistep_old_sigma_down}, previous_sigma={previous_sigma})."
                                        )
                                    t = -math.log(sigma_f)
                                    t_old = -math.log(res_multistep_old_sigma_down)
                                    t_next = -math.log(sigma_down)
                                    t_prev_local = -math.log(previous_sigma)
                                    h = t_next - t
                                    if abs(h) <= 1e-12:
                                        raise RuntimeError(
                                            "Residual multistep second-order update received degenerate step size "
                                            f"(sigma={sigma_f}, sigma_down={sigma_down})."
                                        )
                                    c2 = (t_prev_local - t_old) / h
                                    phi_arg = -h
                                    phi1_val = math.expm1(phi_arg) / phi_arg if abs(phi_arg) > 1e-12 else 1.0
                                    phi2_val = (phi1_val - 1.0) / phi_arg if abs(phi_arg) > 1e-12 else 0.5
                                    if abs(c2) <= 1e-12 or not math.isfinite(c2):
                                        b1 = 0.0
                                        b2 = 0.0
                                    else:
                                        b1 = phi1_val - (phi2_val / c2)
                                        b2 = phi2_val / c2
                                        if not math.isfinite(b1):
                                            b1 = 0.0
                                        if not math.isfinite(b2):
                                            b2 = 0.0
                                    if res_multistep_cfg_pp:
                                        if res_multistep_cfg_pp_capture is None:
                                            raise RuntimeError(
                                                "Residual multistep cfg++ requires uncond_denoised capture, "
                                                "but the sampler post-CFG hook did not run."
                                            )
                                        if tuple(res_multistep_cfg_pp_capture.shape) != tuple(x.shape):
                                            raise RuntimeError(
                                                "Residual multistep cfg++ captured unexpected uncond_denoised shape "
                                                f"{tuple(res_multistep_cfg_pp_capture.shape)}; expected {tuple(x.shape)}"
                                            )
                                        x = x + (denoised - res_multistep_cfg_pp_capture)
                                        x = math.exp(-h) * x + h * (
                                            b1 * res_multistep_cfg_pp_capture + b2 * old_denoised
                                        )
                                    else:
                                        x = math.exp(-h) * x + h * (b1 * denoised + b2 * old_denoised)
                                if sigma_next_f > 0.0 and sigma_up > 0.0:
                                    if seeded_step_rng is None:
                                        raise RuntimeError(
                                            "Residual multistep ancestral variants require deterministic step noise RNG."
                                        )
                                    x = x + self._next_seeded_step_noise(seeded_step_rng, x) * sigma_up
                                if res_multistep_cfg_pp:
                                    if res_multistep_cfg_pp_capture is None:
                                        raise RuntimeError(
                                            "Residual multistep cfg++ requires uncond_denoised capture, "
                                            "but the sampler post-CFG hook did not run."
                                        )
                                    old_denoised = res_multistep_cfg_pp_capture.detach()
                                else:
                                    old_denoised = denoised.detach()
                                res_multistep_old_sigma_down = sigma_down
                            elif sampler_kind in {
                                SamplerKind.GRADIENT_ESTIMATION,
                                SamplerKind.GRADIENT_ESTIMATION_CFG_PP,
                            }:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                dt = sigma_next_f - sigma_f
                                if sigma_next_f <= 0.0:
                                    x = denoised
                                else:
                                    if sampler_kind is SamplerKind.GRADIENT_ESTIMATION_CFG_PP:
                                        x = denoised + eps * sigma_next_f
                                    else:
                                        x = x + eps * dt
                                    if gradient_estimation_prev_d is not None:
                                        x = x + (eps - gradient_estimation_prev_d) * dt
                                gradient_estimation_prev_d = eps.detach()
                            elif sampler_kind is SamplerKind.DDPM:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if not math.isfinite(sigma_f) or not math.isfinite(sigma_next_f):
                                    raise RuntimeError(
                                        "DDPM requires finite sigma values "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                if sigma_f <= 0.0 or sigma_next_f < 0.0:
                                    raise RuntimeError(
                                        "DDPM requires a strictly positive current sigma and a non-negative next sigma "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                sigma_tensor = torch.as_tensor(sigma_f, device=x.device, dtype=x.dtype)
                                sigma_next_tensor = torch.as_tensor(sigma_next_f, device=x.device, dtype=x.dtype)
                                alpha_cumprod = 1.0 / ((sigma_tensor * sigma_tensor) + 1.0)
                                alpha_cumprod_prev = 1.0 / ((sigma_next_tensor * sigma_next_tensor) + 1.0)
                                alpha_cumprod_value = float(alpha_cumprod.item())
                                alpha_cumprod_prev_value = float(alpha_cumprod_prev.item())
                                if (
                                    not math.isfinite(alpha_cumprod_value)
                                    or not math.isfinite(alpha_cumprod_prev_value)
                                    or alpha_cumprod_value <= 0.0
                                    or alpha_cumprod_prev_value <= 0.0
                                ):
                                    raise RuntimeError(
                                        "DDPM produced invalid alpha_cumprod values "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                alpha = alpha_cumprod / alpha_cumprod_prev
                                alpha_value = float(alpha.item())
                                if not math.isfinite(alpha_value) or alpha_value <= 0.0:
                                    raise RuntimeError(
                                        "DDPM produced invalid alpha ratio "
                                        f"(alpha_cumprod={alpha_cumprod_value}, alpha_cumprod_prev={alpha_cumprod_prev_value})."
                                    )
                                one_minus_alpha_cumprod = 1.0 - alpha_cumprod
                                one_minus_alpha_cumprod_value = float(one_minus_alpha_cumprod.item())
                                if one_minus_alpha_cumprod_value <= 0.0 or not math.isfinite(one_minus_alpha_cumprod_value):
                                    raise RuntimeError(
                                        "DDPM produced non-positive 1-alpha_cumprod denominator "
                                        f"(alpha_cumprod={alpha_cumprod_value})."
                                    )
                                mean = torch.sqrt(1.0 / alpha) * (
                                    x / torch.sqrt(1.0 + sigma_tensor * sigma_tensor)
                                    - (1.0 - alpha) * eps / torch.sqrt(one_minus_alpha_cumprod)
                                )
                                if sigma_next_f > 0.0:
                                    variance = (1.0 - alpha) * (1.0 - alpha_cumprod_prev) / one_minus_alpha_cumprod
                                    variance_value = float(variance.item())
                                    if not math.isfinite(variance_value) or variance_value < -1e-12:
                                        raise RuntimeError(
                                            "DDPM produced negative variance "
                                            f"(alpha={alpha_value}, alpha_cumprod={alpha_cumprod_value}, "
                                            f"alpha_cumprod_prev={alpha_cumprod_prev_value})."
                                        )
                                    if seeded_step_rng is None:
                                        raise RuntimeError("DDPM missing deterministic noise RNG.")
                                    mean = (
                                        mean
                                        + self._next_seeded_step_noise(seeded_step_rng, x)
                                        * torch.sqrt(torch.clamp(variance, min=0.0))
                                    )
                                    x = mean * torch.sqrt(1.0 + sigma_next_tensor * sigma_next_tensor)
                                else:
                                    x = mean
                            elif sampler_kind is SamplerKind.DPM2:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if sigma_next_f <= 0.0:
                                    x = x + eps * (sigma_next_f - sigma_f)
                                else:
                                    sigma_mid = math.exp((math.log(max(sigma_f, 1e-12)) + math.log(max(sigma_next_f, 1e-12))) * 0.5)
                                    x_2 = x + eps * (sigma_mid - sigma_f)
                                    x_2_eval, denoised_2 = _evaluate_step_candidate(
                                        x_2,
                                        sigma_candidate=sigma_mid,
                                    )
                                    d_2 = (x_2_eval - denoised_2) / max(sigma_mid, 1e-8)
                                    x = x + d_2 * (sigma_next_f - sigma_f)
                            elif sampler_kind is SamplerKind.DPM2_ANCESTRAL:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if prediction_type == "const":
                                    if sigma_next_f <= 0.0:
                                        x = x + eps * (sigma_next_f - sigma_f)
                                    else:
                                        if sigma_f <= 0.0:
                                            raise RuntimeError(
                                                "DPM2 ancestral RF/CONST requires sigma > 0 before the terminal step."
                                            )
                                        downstep_ratio = 1.0 + (sigma_next_f / sigma_f - 1.0) * 1.0
                                        sigma_down = sigma_next_f * downstep_ratio
                                        alpha_ip1 = 1.0 - sigma_next_f
                                        alpha_down = 1.0 - sigma_down
                                        if abs(alpha_down) <= 1e-12:
                                            raise RuntimeError(
                                                "DPM2 ancestral RF/CONST produced alpha_down=0; cannot compute renoise term."
                                            )
                                        renoise_sq = sigma_next_f**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2
                                        if renoise_sq < -1e-12:
                                            raise RuntimeError(
                                                "DPM2 ancestral RF/CONST produced negative renoise variance."
                                            )
                                        if sigma_down <= 0.0:
                                            x = x + eps * (sigma_down - sigma_f)
                                        else:
                                            sigma_mid = math.exp(
                                                (math.log(max(sigma_f, 1e-12)) + math.log(max(sigma_down, 1e-12))) * 0.5
                                            )
                                            x_2 = x + eps * (sigma_mid - sigma_f)
                                            x_2_eval, denoised_2 = _evaluate_step_candidate(
                                                x_2,
                                                sigma_candidate=sigma_mid,
                                            )
                                            d_2 = (x_2_eval - denoised_2) / max(sigma_mid, 1e-8)
                                            x = x + d_2 * (sigma_down - sigma_f)
                                            if seeded_step_rng is None:
                                                raise RuntimeError("DPM2 ancestral RF/CONST missing deterministic noise RNG.")
                                            x = (
                                                (alpha_ip1 / alpha_down) * x
                                                + self._next_seeded_step_noise(seeded_step_rng, x)
                                                * math.sqrt(max(renoise_sq, 0.0))
                                            )
                                else:
                                    sigma_down, sigma_up = _compute_ancestral_sigmas(sigma_f, sigma_next_f, eta=1.0)
                                    if sigma_down <= 0.0:
                                        x = x + eps * (sigma_down - sigma_f)
                                    else:
                                        sigma_mid = math.exp((math.log(max(sigma_f, 1e-12)) + math.log(max(sigma_down, 1e-12))) * 0.5)
                                        x_2 = x + eps * (sigma_mid - sigma_f)
                                        x_2_eval, denoised_2 = _evaluate_step_candidate(
                                            x_2,
                                            sigma_candidate=sigma_mid,
                                        )
                                        d_2 = (x_2_eval - denoised_2) / max(sigma_mid, 1e-8)
                                        x = x + d_2 * (sigma_down - sigma_f)
                                        if sigma_next_f > 0.0:
                                            if seeded_step_rng is None:
                                                raise RuntimeError("DPM2 ancestral missing deterministic noise RNG.")
                                            x = x + self._next_seeded_step_noise(seeded_step_rng, x) * sigma_up
                            elif sampler_kind is SamplerKind.DPM2S_ANCESTRAL:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if prediction_type == "const":
                                    if sigma_next_f <= 0.0:
                                        x = x + eps * (sigma_next_f - sigma_f)
                                    else:
                                        if sigma_f <= 0.0:
                                            raise RuntimeError(
                                                "DPM++ 2S ancestral RF/CONST requires sigma > 0 before the terminal step."
                                            )
                                        downstep_ratio = 1.0 + (sigma_next_f / sigma_f - 1.0) * 1.0
                                        sigma_down = sigma_next_f * downstep_ratio
                                        alpha_ip1 = 1.0 - sigma_next_f
                                        alpha_down = 1.0 - sigma_down
                                        if abs(alpha_down) <= 1e-12:
                                            raise RuntimeError(
                                                "DPM++ 2S ancestral RF/CONST produced alpha_down=0; cannot compute renoise term."
                                            )
                                        renoise_sq = sigma_next_f**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2
                                        if renoise_sq < -1e-12:
                                            raise RuntimeError(
                                                "DPM++ 2S ancestral RF/CONST produced negative renoise variance."
                                            )
                                        if math.isclose(sigma_f, 1.0):
                                            sigma_s = 1.0 - _ER_SDE_CONST_SNR_PERCENT_OFFSET
                                        else:
                                            lambda_i = math.log(max(1.0 - sigma_f, 1e-12) / max(sigma_f, 1e-12))
                                            lambda_down = math.log(max(1.0 - sigma_down, 1e-12) / max(sigma_down, 1e-12))
                                            sigma_s = 1.0 / (math.exp(lambda_i + 0.5 * (lambda_down - lambda_i)) + 1.0)
                                        u = (sigma_s / sigma_f) * x + (1.0 - sigma_s / sigma_f) * denoised
                                        u_eval, denoised_2 = _evaluate_step_candidate(
                                            u,
                                            sigma_candidate=sigma_s,
                                        )
                                        x = (sigma_down / sigma_f) * x + (1.0 - sigma_down / sigma_f) * denoised_2
                                        if seeded_step_rng is None:
                                            raise RuntimeError("DPM++ 2S ancestral RF/CONST missing deterministic noise RNG.")
                                        x = (
                                            (alpha_ip1 / alpha_down) * x
                                            + self._next_seeded_step_noise(seeded_step_rng, x) * math.sqrt(max(renoise_sq, 0.0))
                                        )
                                else:
                                    sigma_down, sigma_up = _compute_ancestral_sigmas(sigma_f, sigma_next_f, eta=1.0)
                                    if sigma_down <= 0.0:
                                        x = x + eps * (sigma_down - sigma_f)
                                    else:
                                        t = -math.log(max(sigma_f, 1e-12))
                                        t_next = -math.log(max(sigma_down, 1e-12))
                                        h = t_next - t
                                        sigma_s = math.exp(-(t + 0.5 * h))
                                        x_2 = (sigma_s / sigma_f) * x - math.expm1(-0.5 * h) * denoised
                                        x_2_eval, denoised_2 = _evaluate_step_candidate(
                                            x_2,
                                            sigma_candidate=sigma_s,
                                        )
                                        x = (sigma_down / sigma_f) * x - math.expm1(-h) * denoised_2
                                        if sigma_next_f > 0.0:
                                            if seeded_step_rng is None:
                                                raise RuntimeError("DPM++ 2S ancestral missing deterministic noise RNG.")
                                            x = x + self._next_seeded_step_noise(seeded_step_rng, x) * sigma_up
                            elif sampler_kind is SamplerKind.DPM2S_ANCESTRAL_CFG_PP:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if dpm2s_ancestral_cfg_pp_capture is None:
                                    raise RuntimeError(
                                        "DPM++ 2S ancestral cfg++ requires uncond_denoised capture, "
                                        "but the sampler post-CFG hook did not run."
                                    )
                                if tuple(dpm2s_ancestral_cfg_pp_capture.shape) != tuple(x.shape):
                                    raise RuntimeError(
                                        "DPM++ 2S ancestral cfg++ captured unexpected uncond_denoised shape "
                                        f"{tuple(dpm2s_ancestral_cfg_pp_capture.shape)}; expected {tuple(x.shape)}"
                                    )
                                uncond_denoised = dpm2s_ancestral_cfg_pp_capture
                                if prediction_type == "const":
                                    if sigma_next_f <= 0.0:
                                        x = denoised
                                    else:
                                        if sigma_f <= 0.0:
                                            raise RuntimeError(
                                                "DPM++ 2S ancestral cfg++ RF/CONST requires sigma > 0 before the terminal step."
                                            )
                                        downstep_ratio = 1.0 + (sigma_next_f / sigma_f - 1.0) * 1.0
                                        sigma_down = sigma_next_f * downstep_ratio
                                        alpha_ip1 = 1.0 - sigma_next_f
                                        alpha_down = 1.0 - sigma_down
                                        if abs(alpha_down) <= 1e-12:
                                            raise RuntimeError(
                                                "DPM++ 2S ancestral cfg++ RF/CONST produced alpha_down=0; cannot compute renoise term."
                                            )
                                        renoise_sq = sigma_next_f**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2
                                        if renoise_sq < -1e-12:
                                            raise RuntimeError(
                                                "DPM++ 2S ancestral cfg++ RF/CONST produced negative renoise variance."
                                            )
                                        if sigma_down <= 0.0:
                                            d_uncond = (x - uncond_denoised) / max(sigma_f, 1e-8)
                                            x = denoised + d_uncond * sigma_down
                                        else:
                                            if math.isclose(sigma_f, 1.0):
                                                sigma_s = 1.0 - _ER_SDE_CONST_SNR_PERCENT_OFFSET
                                            else:
                                                lambda_bounds = sigma_to_half_log_snr(
                                                    torch.tensor(
                                                        [sigma_f, sigma_down],
                                                        dtype=torch.float32,
                                                        device=x.device,
                                                    ),
                                                    prediction_type=prediction_type,
                                                )
                                                lambda_mid = lambda_bounds[0] + 0.5 * (
                                                    lambda_bounds[1] - lambda_bounds[0]
                                                )
                                                sigma_mid_tensor = half_log_snr_to_sigma(
                                                    lambda_mid.reshape(1),
                                                    prediction_type=prediction_type,
                                                )
                                                sigma_s = float(sigma_mid_tensor[0].item())
                                            x_shifted = x + (denoised - uncond_denoised)
                                            u = (sigma_s / sigma_f) * x_shifted + (1.0 - sigma_s / sigma_f) * denoised
                                            u_eval, denoised_2 = _evaluate_step_candidate(
                                                u,
                                                sigma_candidate=sigma_s,
                                            )
                                            x = (sigma_down / sigma_f) * x_shifted + (1.0 - sigma_down / sigma_f) * denoised_2
                                            if seeded_step_rng is None:
                                                raise RuntimeError(
                                                    "DPM++ 2S ancestral cfg++ RF/CONST missing deterministic noise RNG."
                                                )
                                            x = (
                                                (alpha_ip1 / alpha_down) * x
                                                + self._next_seeded_step_noise(seeded_step_rng, x)
                                                * math.sqrt(max(renoise_sq, 0.0))
                                            )
                                else:
                                    sigma_down, sigma_up = _compute_ancestral_sigmas(sigma_f, sigma_next_f, eta=1.0)
                                    if sigma_down <= 0.0:
                                        d_uncond = (x - uncond_denoised) / max(sigma_f, 1e-8)
                                        x = denoised + d_uncond * sigma_down
                                    else:
                                        t = -math.log(max(sigma_f, 1e-12))
                                        t_next = -math.log(max(sigma_down, 1e-12))
                                        h = t_next - t
                                        sigma_s = math.exp(-(t + 0.5 * h))
                                        x_shifted = x + (denoised - uncond_denoised)
                                        x_2 = (sigma_s / sigma_f) * x_shifted - math.expm1(-0.5 * h) * denoised
                                        x_2_eval, denoised_2 = _evaluate_step_candidate(
                                            x_2,
                                            sigma_candidate=sigma_s,
                                        )
                                        x = (sigma_down / sigma_f) * x_shifted - math.expm1(-h) * denoised_2
                                        if sigma_next_f > 0.0:
                                            if seeded_step_rng is None:
                                                raise RuntimeError(
                                                    "DPM++ 2S ancestral cfg++ missing deterministic noise RNG."
                                                )
                                            x = x + self._next_seeded_step_noise(seeded_step_rng, x) * sigma_up
                            elif sampler_kind is SamplerKind.DPM2M:
                                # DPM-Solver++(2M) in log-sigma time (reference update form).
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if sigma_next_f <= 0.0:
                                    x = denoised
                                    old_denoised = denoised.detach()
                                    t_prev = None
                                else:
                                    t = -math.log(max(sigma_f, 1e-12))
                                    t_next = -math.log(max(sigma_next_f, 1e-12))
                                    h = t_next - t
                                    if old_denoised is None or t_prev is None:
                                        x = (sigma_next_f / sigma_f) * x - math.expm1(-h) * denoised
                                    else:
                                        h_last = t - t_prev
                                        r = h_last / h if abs(h) > 1e-12 else 1.0
                                        denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised
                                        x = (sigma_next_f / sigma_f) * x - math.expm1(-h) * denoised_d
                                    old_denoised = denoised.detach()
                                    t_prev = t
                            elif sampler_kind is SamplerKind.DPM2M_CFG_PP:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if dpm2m_cfg_pp_capture is None:
                                    raise RuntimeError(
                                        "DPM++ 2M cfg++ requires uncond_denoised capture, "
                                        "but the sampler post-CFG hook did not run."
                                    )
                                if tuple(dpm2m_cfg_pp_capture.shape) != tuple(x.shape):
                                    raise RuntimeError(
                                        "DPM++ 2M cfg++ captured unexpected uncond_denoised shape "
                                        f"{tuple(dpm2m_cfg_pp_capture.shape)}; expected {tuple(x.shape)}"
                                    )
                                if sigma_next_f <= 0.0:
                                    x = denoised
                                    old_denoised = dpm2m_cfg_pp_capture.detach()
                                    t_prev = None
                                else:
                                    t = -math.log(max(sigma_f, 1e-12))
                                    t_next = -math.log(max(sigma_next_f, 1e-12))
                                    h = t_next - t
                                    exp_neg_h = math.exp(-h)
                                    if old_denoised is None or t_prev is None:
                                        denoised_mix = -exp_neg_h * dpm2m_cfg_pp_capture
                                    else:
                                        h_last = t - t_prev
                                        r = h_last / h if abs(h) > 1e-12 else 1.0
                                        denoised_mix = (
                                            -exp_neg_h * dpm2m_cfg_pp_capture
                                            - math.expm1(-h) * (1.0 / (2.0 * r)) * (denoised - old_denoised)
                                        )
                                    x = denoised + denoised_mix + exp_neg_h * x
                                    old_denoised = dpm2m_cfg_pp_capture.detach()
                                    t_prev = t
                            elif sampler_kind is SamplerKind.DPMPP_SDE:
                                if dpm_sde_sigmas is None or dpm_sde_lambdas is None:
                                    raise RuntimeError("DPM++ SDE missing initialized half-logSNR runtime state.")
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if not math.isfinite(sigma_f) or not math.isfinite(sigma_next_f):
                                    raise RuntimeError(
                                        "DPM++ SDE requires finite sigma values "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                if sigma_f <= 0.0 or sigma_next_f < 0.0:
                                    raise RuntimeError(
                                        "DPM++ SDE requires a strictly positive current sigma and a non-negative next sigma "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                if sigma_next_f == 0.0:
                                    x = denoised
                                else:
                                    lambda_s = dpm_sde_lambdas[step_index]
                                    lambda_t = dpm_sde_lambdas[step_index + 1]
                                    h = lambda_t - lambda_s
                                    h_f = float(h)
                                    if not math.isfinite(h_f) or h_f <= 0.0:
                                        raise RuntimeError(
                                            "DPM++ SDE produced an invalid half-logSNR step size "
                                            f"h={h_f} at step={step_index + 1}."
                                        )
                                    lambda_mid = lambda_s + h * dpm_sde_r
                                    sigma_mid = half_log_snr_to_sigma(
                                        lambda_mid.reshape(1),
                                        prediction_type=prediction_type,
                                    )[0]
                                    sigma_mid_f = float(sigma_mid)
                                    if not math.isfinite(sigma_mid_f) or sigma_mid_f <= 0.0:
                                        raise RuntimeError(
                                            "DPM++ SDE produced an invalid midpoint sigma "
                                            f"(sigma_mid={sigma_mid_f}) at step={step_index + 1}."
                                        )
                                    alpha_s = sigma * lambda_s.exp()
                                    alpha_mid = sigma_mid * lambda_mid.exp()
                                    alpha_t = sigma_next * lambda_t.exp()
                                    alpha_s_f = float(alpha_s)
                                    alpha_mid_f = float(alpha_mid)
                                    alpha_t_f = float(alpha_t)
                                    if (
                                        not math.isfinite(alpha_s_f)
                                        or not math.isfinite(alpha_mid_f)
                                        or not math.isfinite(alpha_t_f)
                                        or alpha_s_f <= 0.0
                                        or alpha_mid_f <= 0.0
                                        or alpha_t_f <= 0.0
                                    ):
                                        raise RuntimeError(
                                            "DPM++ SDE produced non-positive alpha values "
                                            f"(alpha_s={alpha_s_f}, alpha_mid={alpha_mid_f}, alpha_t={alpha_t_f}) "
                                            f"at step={step_index + 1}."
                                        )

                                    lambda_s_f = float(lambda_s)
                                    lambda_t_f = float(lambda_t)
                                    lambda_mid_f = float(lambda_mid)
                                    sigma_down_mid, sigma_up_mid = _compute_ancestral_sigmas(
                                        math.exp(-lambda_s_f),
                                        math.exp(-lambda_mid_f),
                                        eta=dpm_sde_eta,
                                    )
                                    sigma_down_end, sigma_up_end = _compute_ancestral_sigmas(
                                        math.exp(-lambda_s_f),
                                        math.exp(-lambda_t_f),
                                        eta=dpm_sde_eta,
                                    )
                                    if sigma_down_mid <= 0.0 or sigma_down_end <= 0.0:
                                        raise RuntimeError(
                                            "DPM++ SDE produced non-positive ancestral sigma_down values "
                                            f"(mid={sigma_down_mid}, end={sigma_down_end}) at step={step_index + 1}."
                                        )
                                    h_mid = -math.log(sigma_down_mid) - lambda_s_f
                                    h_end = -math.log(sigma_down_end) - lambda_s_f
                                    if (
                                        not math.isfinite(h_mid)
                                        or not math.isfinite(h_end)
                                        or h_mid <= 0.0
                                        or h_end <= 0.0
                                    ):
                                        raise RuntimeError(
                                            "DPM++ SDE produced invalid projected half-logSNR step sizes "
                                            f"(h_mid={h_mid}, h_end={h_end}) at step={step_index + 1}."
                                        )
                                    x_2 = (
                                        (alpha_mid_f / alpha_s_f) * math.exp(-h_mid) * x
                                    ) - (alpha_mid * math.expm1(-h_mid) * denoised)

                                    interval_noise_end: torch.Tensor | None = None
                                    if dpm_sde_eta > 0.0 and dpm_sde_s_noise > 0.0:
                                        if seeded_step_rng is None:
                                            raise RuntimeError("DPM++ SDE missing deterministic noise RNG.")
                                        interval_noise_mid, interval_noise_end = compose_nested_interval_noises(
                                            interval_start=sigma_f,
                                            interval_mid=sigma_mid_f,
                                            interval_end=sigma_next_f,
                                            first_draw=self._next_seeded_step_noise(seeded_step_rng, x),
                                            second_draw=self._next_seeded_step_noise(seeded_step_rng, x),
                                        )
                                        x_2 = x_2 + (
                                            alpha_mid
                                            * interval_noise_mid
                                            * dpm_sde_s_noise
                                            * sigma_up_mid
                                        )

                                    _, denoised_2, _ = _dpm_eval_eps(
                                        x_2,
                                        sigma_value=sigma_mid_f,
                                        current_step=current_step,
                                        step_total=run_total_steps,
                                    )
                                    denoised_d = torch.lerp(
                                        denoised,
                                        denoised_2,
                                        1.0 / (2.0 * dpm_sde_r),
                                    )
                                    x = (
                                        (alpha_t_f / alpha_s_f) * math.exp(-h_end) * x
                                    ) - (alpha_t * math.expm1(-h_end) * denoised_d)
                                    if interval_noise_end is not None:
                                        x = x + (
                                            alpha_t
                                            * interval_noise_end
                                            * dpm_sde_s_noise
                                            * sigma_up_end
                                        )
                            elif sampler_kind in {
                                SamplerKind.DPM2M_SDE,
                                SamplerKind.DPM2M_SDE_HEUN,
                                SamplerKind.DPM2M_SDE_GPU,
                                SamplerKind.DPM2M_SDE_HEUN_GPU,
                            }:
                                if dpm2m_sde_sigmas is None or dpm2m_sde_lambdas is None:
                                    raise RuntimeError("DPM++ 2M SDE missing initialized half-logSNR runtime state.")
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if not math.isfinite(sigma_f) or not math.isfinite(sigma_next_f):
                                    raise RuntimeError(
                                        "DPM++ 2M SDE requires finite sigma values "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                if sigma_f <= 0.0 or sigma_next_f < 0.0:
                                    raise RuntimeError(
                                        "DPM++ 2M SDE requires a strictly positive current sigma and a non-negative next sigma "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                if sigma_next_f == 0.0:
                                    x = denoised
                                    old_denoised = denoised.detach()
                                    h_prev = None
                                else:
                                    lambda_s = dpm2m_sde_lambdas[step_index]
                                    lambda_t = dpm2m_sde_lambdas[step_index + 1]
                                    h = lambda_t - lambda_s
                                    h_f = float(h)
                                    if not math.isfinite(h_f) or h_f <= 0.0:
                                        raise RuntimeError(
                                            "DPM++ 2M SDE produced an invalid half-logSNR step size "
                                            f"h={h_f} at step={step_index + 1}."
                                        )
                                    h_eta = h * (dpm2m_sde_eta + 1.0)
                                    alpha_t = dpm2m_sde_sigmas[step_index + 1] * lambda_t.exp()
                                    alpha_t_f = float(alpha_t)
                                    if not math.isfinite(alpha_t_f) or alpha_t_f <= 0.0:
                                        raise RuntimeError(
                                            "DPM++ 2M SDE produced an invalid alpha_t value "
                                            f"alpha_t={alpha_t_f} at step={step_index + 1}."
                                        )
                                    phi_1 = -ei_h_phi_1(-h_eta)
                                    if not bool(torch.isfinite(phi_1)):
                                        raise RuntimeError(
                                            f"DPM++ 2M SDE produced a non-finite phi_1 value at step={step_index + 1}."
                                        )
                                    x = (sigma_next_f / sigma_f) * math.exp(-h_f * dpm2m_sde_eta) * x + alpha_t * phi_1 * denoised
                                    if old_denoised is not None and h_prev is not None:
                                        if not math.isfinite(h_prev) or abs(h_prev) <= 1e-12:
                                            raise RuntimeError(
                                                "DPM++ 2M SDE correction term requires a finite non-zero previous step size "
                                                f"(h_prev={h_prev}) at step={step_index + 1}."
                                            )
                                        r = h_prev / h_f
                                        if not math.isfinite(r) or abs(r) <= 1e-12:
                                            raise RuntimeError(
                                                "DPM++ 2M SDE correction term produced an invalid step ratio "
                                                f"r={r} at step={step_index + 1}."
                                            )
                                        if sampler_kind in {
                                            SamplerKind.DPM2M_SDE_HEUN,
                                            SamplerKind.DPM2M_SDE_HEUN_GPU,
                                        }:
                                            h_eta_f = float(h_eta)
                                            if not math.isfinite(h_eta_f) or abs(h_eta_f) <= 1e-12:
                                                raise RuntimeError(
                                                    "DPM++ 2M SDE Heun correction requires a finite non-zero stochastic step size "
                                                    f"(h_eta={h_eta_f}) at step={step_index + 1}."
                                                )
                                            correction_weight = (phi_1 / (-h_eta)) + 1.0
                                        else:
                                            correction_weight = 0.5 * phi_1
                                        if not bool(torch.isfinite(correction_weight)):
                                            raise RuntimeError(
                                                "DPM++ 2M SDE correction term produced a non-finite correction weight "
                                                f"at step={step_index + 1}."
                                            )
                                        x = x + alpha_t * correction_weight * (1.0 / r) * (denoised - old_denoised)
                                    if dpm2m_sde_eta > 0.0 and dpm2m_sde_s_noise > 0.0:
                                        noise_scale_sq = -math.expm1(-2.0 * h_f * dpm2m_sde_eta)
                                        if noise_scale_sq < -1e-8:
                                            raise RuntimeError(
                                                "DPM++ 2M SDE produced a negative stochastic noise scale "
                                                f"({noise_scale_sq}) at step={step_index + 1}."
                                            )
                                        noise_scale = math.sqrt(max(noise_scale_sq, 0.0))
                                        if not math.isfinite(noise_scale):
                                            raise RuntimeError(
                                                f"DPM++ 2M SDE produced a non-finite stochastic scale at step={step_index + 1}."
                                            )
                                        if noise_scale > 0.0:
                                            if seeded_step_rng is None:
                                                raise RuntimeError("DPM++ 2M SDE missing deterministic noise RNG.")
                                            x = x + (
                                                self._next_seeded_step_noise(seeded_step_rng, x)
                                                * sigma_next_f
                                                * noise_scale
                                                * dpm2m_sde_s_noise
                                            )
                                    old_denoised = denoised.detach()
                                    h_prev = h_f
                            elif sampler_kind is SamplerKind.DPM3M_SDE:
                                if dpm3m_sde_sigmas is None or dpm3m_sde_lambdas is None:
                                    raise RuntimeError("DPM++ 3M SDE missing initialized half-logSNR runtime state.")
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if not math.isfinite(sigma_f) or not math.isfinite(sigma_next_f):
                                    raise RuntimeError(
                                        "DPM++ 3M SDE requires finite sigma values "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                if sigma_f <= 0.0 or sigma_next_f < 0.0:
                                    raise RuntimeError(
                                        "DPM++ 3M SDE requires a strictly positive current sigma and a non-negative next sigma "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                if sigma_next_f == 0.0:
                                    x = denoised
                                else:
                                    lambda_s = dpm3m_sde_lambdas[step_index]
                                    lambda_t = dpm3m_sde_lambdas[step_index + 1]
                                    h = lambda_t - lambda_s
                                    h_f = float(h)
                                    if not math.isfinite(h_f) or h_f <= 0.0:
                                        raise RuntimeError(
                                            "DPM++ 3M SDE produced an invalid half-logSNR step size "
                                            f"h={h_f} at step={step_index + 1}."
                                        )
                                    h_eta = h * (dpm3m_sde_eta + 1.0)
                                    alpha_t = dpm3m_sde_sigmas[step_index + 1] * lambda_t.exp()
                                    alpha_t_f = float(alpha_t)
                                    if not math.isfinite(alpha_t_f) or alpha_t_f <= 0.0:
                                        raise RuntimeError(
                                            "DPM++ 3M SDE produced an invalid alpha_t value "
                                            f"alpha_t={alpha_t_f} at step={step_index + 1}."
                                        )
                                    x = (
                                        (sigma_next_f / sigma_f) * math.exp(-h_f * dpm3m_sde_eta) * x
                                    ) + alpha_t * (-h_eta).expm1().neg() * denoised
                                    h_eta_f = float(h_eta)
                                    if not math.isfinite(h_eta_f) or abs(h_eta_f) <= 1e-12:
                                        raise RuntimeError(
                                            "DPM++ 3M SDE requires a finite non-zero stochastic step size "
                                            f"(h_eta={h_eta_f}) at step={step_index + 1}."
                                        )
                                    phi_2 = h_eta.neg().expm1() / h_eta + 1.0
                                    if not bool(torch.isfinite(phi_2)):
                                        raise RuntimeError(
                                            f"DPM++ 3M SDE produced a non-finite phi_2 value at step={step_index + 1}."
                                        )
                                    if older_denoised is not None and old_denoised is not None and h_prev is not None and h_prev_2 is not None:
                                        if (
                                            not math.isfinite(h_prev)
                                            or abs(h_prev) <= 1e-12
                                            or not math.isfinite(h_prev_2)
                                            or abs(h_prev_2) <= 1e-12
                                        ):
                                            raise RuntimeError(
                                                "DPM++ 3M SDE correction requires finite non-zero previous step sizes "
                                                f"(h_prev={h_prev}, h_prev_2={h_prev_2}) at step={step_index + 1}."
                                            )
                                        r0 = h_prev / h_f
                                        r1 = h_prev_2 / h_f
                                        sum_r = r0 + r1
                                        if (
                                            not math.isfinite(r0)
                                            or abs(r0) <= 1e-12
                                            or not math.isfinite(r1)
                                            or abs(r1) <= 1e-12
                                            or not math.isfinite(sum_r)
                                            or abs(sum_r) <= 1e-12
                                        ):
                                            raise RuntimeError(
                                                "DPM++ 3M SDE correction produced invalid step ratios "
                                                f"(r0={r0}, r1={r1}) at step={step_index + 1}."
                                            )
                                        d1_0 = (denoised - old_denoised) / r0
                                        d1_1 = (old_denoised - older_denoised) / r1
                                        d1 = d1_0 + (d1_0 - d1_1) * r0 / sum_r
                                        d2 = (d1_0 - d1_1) / sum_r
                                        phi_3 = (phi_2 / h_eta) - 0.5
                                        if not bool(torch.isfinite(phi_3)):
                                            raise RuntimeError(
                                                f"DPM++ 3M SDE produced a non-finite phi_3 value at step={step_index + 1}."
                                            )
                                        x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
                                    elif old_denoised is not None and h_prev is not None:
                                        if not math.isfinite(h_prev) or abs(h_prev) <= 1e-12:
                                            raise RuntimeError(
                                                "DPM++ 3M SDE warm-start correction requires a finite non-zero previous step size "
                                                f"(h_prev={h_prev}) at step={step_index + 1}."
                                            )
                                        r = h_prev / h_f
                                        if not math.isfinite(r) or abs(r) <= 1e-12:
                                            raise RuntimeError(
                                                "DPM++ 3M SDE warm-start correction produced an invalid step ratio "
                                                f"r={r} at step={step_index + 1}."
                                            )
                                        d = (denoised - old_denoised) / r
                                        x = x + (alpha_t * phi_2) * d
                                    if dpm3m_sde_eta > 0.0 and dpm3m_sde_s_noise > 0.0:
                                        noise_scale_sq = -math.expm1(-2.0 * h_f * dpm3m_sde_eta)
                                        if noise_scale_sq < -1e-8:
                                            raise RuntimeError(
                                                "DPM++ 3M SDE produced a negative stochastic noise scale "
                                                f"({noise_scale_sq}) at step={step_index + 1}."
                                            )
                                        noise_scale = math.sqrt(max(noise_scale_sq, 0.0))
                                        if not math.isfinite(noise_scale):
                                            raise RuntimeError(
                                                f"DPM++ 3M SDE produced a non-finite stochastic scale at step={step_index + 1}."
                                            )
                                        if noise_scale > 0.0:
                                            if seeded_step_rng is None:
                                                raise RuntimeError("DPM++ 3M SDE missing deterministic noise RNG.")
                                            x = x + (
                                                self._next_seeded_step_noise(seeded_step_rng, x)
                                                * sigma_next_f
                                                * noise_scale
                                                * dpm3m_sde_s_noise
                                            )
                                    h_prev_2 = h_prev
                                    h_prev = h_f
                                older_denoised = old_denoised
                                old_denoised = denoised.detach()
                            elif sampler_kind is SamplerKind.ER_SDE:
                                if er_sde_params is None or er_sde_lambdas is None or er_sde_point_indices is None:
                                    raise RuntimeError("ER-SDE runtime state is not initialized")
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if sigma_next_f <= 0.0:
                                    x = denoised
                                    old_denoised = denoised.detach()
                                else:
                                    solver_type = str(er_sde_params["solver_type"])
                                    eta = float(er_sde_params["eta"])
                                    s_noise = float(er_sde_params["s_noise"])
                                    max_stage = int(er_sde_params["max_stage"])
                                    stage_used = min(max_stage, step_index + 1)

                                    er_lambda_s = er_sde_lambdas[step_index]
                                    er_lambda_t = er_sde_lambdas[step_index + 1]
                                    er_lambda_s_f = float(er_lambda_s)
                                    er_lambda_t_f = float(er_lambda_t)
                                    if (
                                        not math.isfinite(er_lambda_s_f)
                                        or not math.isfinite(er_lambda_t_f)
                                        or er_lambda_s_f <= 0.0
                                        or er_lambda_t_f <= 0.0
                                    ):
                                        raise RuntimeError(
                                            f"ER-SDE received invalid lambda values: {er_lambda_s_f}, {er_lambda_t_f}"
                                        )

                                    alpha_s = sigma_f / er_lambda_s_f
                                    alpha_t = sigma_next_f / er_lambda_t_f
                                    if (
                                        not math.isfinite(alpha_s)
                                        or not math.isfinite(alpha_t)
                                        or alpha_s <= 0.0
                                        or alpha_t <= 0.0
                                    ):
                                        raise RuntimeError(
                                            f"ER-SDE received invalid alpha values: {alpha_s}, {alpha_t}"
                                        )

                                    r_alpha = alpha_t / alpha_s
                                    noise_scale_s = self._er_sde_noise_scaler(
                                        er_lambda_s,
                                        solver_type=solver_type,
                                        eta=eta,
                                    )
                                    noise_scale_t = self._er_sde_noise_scaler(
                                        er_lambda_t,
                                        solver_type=solver_type,
                                        eta=eta,
                                    )
                                    noise_scale_s_f = float(noise_scale_s)
                                    noise_scale_t_f = float(noise_scale_t)
                                    if (
                                        not math.isfinite(noise_scale_s_f)
                                        or not math.isfinite(noise_scale_t_f)
                                        or noise_scale_s_f <= 0.0
                                        or noise_scale_t_f <= 0.0
                                    ):
                                        raise RuntimeError(
                                            f"ER-SDE noise scaler returned invalid values: {noise_scale_s_f}, {noise_scale_t_f}"
                                        )
                                    r = noise_scale_t_f / noise_scale_s_f
                                    if not math.isfinite(r):
                                        raise RuntimeError(f"ER-SDE produced non-finite ratio r={r}")

                                    # Stage 1 (Euler)
                                    x = r_alpha * r * x + alpha_t * (1.0 - r) * denoised

                                    if stage_used >= 2:
                                        if old_denoised is None:
                                            raise RuntimeError("ER-SDE stage-2 requires previous denoised state")
                                        dt = er_lambda_t_f - er_lambda_s_f
                                        lambda_step_size = -dt / 200.0
                                        lambda_pos = er_lambda_t + er_sde_point_indices * lambda_step_size
                                        scaled_pos = self._er_sde_noise_scaler(
                                            lambda_pos,
                                            solver_type=solver_type,
                                            eta=eta,
                                        )
                                        if not bool(torch.all(torch.isfinite(scaled_pos))):
                                            raise RuntimeError(
                                                f"ER-SDE stage-2 produced non-finite scaled positions at step={step_index + 1}"
                                            )
                                        if bool(torch.any(scaled_pos <= 0.0)):
                                            raise RuntimeError(
                                                f"ER-SDE stage-2 produced non-positive scaled positions at step={step_index + 1}"
                                            )
                                        s_term = float(torch.sum(1.0 / scaled_pos) * lambda_step_size)
                                        if not math.isfinite(s_term):
                                            raise RuntimeError(
                                                f"ER-SDE stage-2 produced non-finite integral term at step={step_index + 1}"
                                            )
                                        prev_gap = er_lambda_s_f - float(er_sde_lambdas[step_index - 1])
                                        if abs(prev_gap) <= 1e-12:
                                            raise RuntimeError(
                                                f"ER-SDE stage-2 denominator collapsed at step={step_index + 1}"
                                            )
                                        denoised_d = (denoised - old_denoised) / prev_gap
                                        x = x + alpha_t * (dt + s_term * noise_scale_t_f) * denoised_d

                                        if stage_used >= 3:
                                            if old_denoised_d is None:
                                                raise RuntimeError("ER-SDE stage-3 requires previous stage-2 state")
                                            s_u = float(
                                                torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                                            )
                                            if not math.isfinite(s_u):
                                                raise RuntimeError(
                                                    f"ER-SDE stage-3 produced non-finite integral term at step={step_index + 1}"
                                                )
                                            stage3_gap = (er_lambda_s_f - float(er_sde_lambdas[step_index - 2])) / 2.0
                                            if abs(stage3_gap) <= 1e-12:
                                                raise RuntimeError(
                                                    f"ER-SDE stage-3 denominator collapsed at step={step_index + 1}"
                                                )
                                            denoised_u = (denoised_d - old_denoised_d) / stage3_gap
                                            x = x + alpha_t * ((dt**2) / 2.0 + s_u * noise_scale_t_f) * denoised_u
                                        old_denoised_d = denoised_d.detach()

                                    if s_noise > 0.0:
                                        noise_term = er_lambda_t_f**2 - er_lambda_s_f**2 * (r**2)
                                        if noise_term < -1e-8:
                                            raise RuntimeError(
                                                f"ER-SDE produced negative stochastic term at step={step_index + 1}: {noise_term}"
                                            )
                                        noise_scale = math.sqrt(max(noise_term, 0.0))
                                        if not math.isfinite(noise_scale):
                                            raise RuntimeError(
                                                f"ER-SDE produced non-finite stochastic scale at step={step_index + 1}"
                                            )
                                        if seeded_step_rng is None:
                                            raise RuntimeError("ER-SDE missing deterministic noise RNG.")
                                        x = x + (
                                            alpha_t
                                            * self._next_seeded_step_noise(seeded_step_rng, x)
                                            * s_noise
                                            * noise_scale
                                        )
                                    old_denoised = denoised.detach()
                            elif sampler_kind is SamplerKind.SEEDS_2:
                                if seeds2_sigmas is None or seeds2_lambdas is None:
                                    raise RuntimeError("SEEDS-2 missing initialized half-logSNR runtime state.")
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if not math.isfinite(sigma_f) or not math.isfinite(sigma_next_f):
                                    raise RuntimeError(
                                        "SEEDS-2 requires finite sigma values "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                if sigma_f <= 0.0 or sigma_next_f < 0.0:
                                    raise RuntimeError(
                                        "SEEDS-2 requires a strictly positive current sigma and a non-negative next sigma "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                if sigma_next_f == 0.0:
                                    x = denoised
                                else:
                                    lambda_s = seeds2_lambdas[step_index]
                                    lambda_t = seeds2_lambdas[step_index + 1]
                                    h = lambda_t - lambda_s
                                    h_f = float(h)
                                    if not math.isfinite(h_f) or h_f <= 0.0:
                                        raise RuntimeError(
                                            f"SEEDS-2 produced invalid half-logSNR step size h={h_f} at step={step_index + 1}."
                                        )
                                    h_eta = h * (seeds2_eta + 1.0)
                                    lambda_s_1 = torch.lerp(lambda_s, lambda_t, seeds2_r)
                                    sigma_s_1 = half_log_snr_to_sigma(
                                        lambda_s_1,
                                        prediction_type=prediction_type,
                                    )
                                    sigma_s_1_f = float(sigma_s_1)
                                    if not math.isfinite(sigma_s_1_f) or sigma_s_1_f <= 0.0:
                                        raise RuntimeError(
                                            f"SEEDS-2 produced invalid intermediate sigma {sigma_s_1_f} at step={step_index + 1}."
                                        )
                                    alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
                                    alpha_t = seeds2_sigmas[step_index + 1] * lambda_t.exp()
                                    if not bool(torch.isfinite(alpha_s_1)) or not bool(torch.isfinite(alpha_t)):
                                        raise RuntimeError(
                                            f"SEEDS-2 produced non-finite alpha values at step={step_index + 1}."
                                        )

                                    x_2 = (
                                        (sigma_s_1_f / sigma_f) * math.exp(-seeds2_r * h_f * seeds2_eta) * x
                                    ) - (alpha_s_1 * ei_h_phi_1(-seeds2_r * h_eta) * denoised)
                                    if seeds2_s_noise > 0.0:
                                        if seeded_step_rng is None:
                                            raise RuntimeError("SEEDS-2 missing deterministic noise RNG.")
                                        noise_scale_1_sq = -math.expm1(-2.0 * seeds2_r * h_f * seeds2_eta)
                                        if noise_scale_1_sq < -1e-8:
                                            raise RuntimeError(
                                                "SEEDS-2 produced a negative first stochastic noise scale "
                                                f"({noise_scale_1_sq}) at step={step_index + 1}."
                                            )
                                        sde_noise = (
                                            math.sqrt(max(noise_scale_1_sq, 0.0))
                                            * self._next_seeded_step_noise(seeded_step_rng, x)
                                        )
                                        x_2 = x_2 + sde_noise * sigma_s_1_f * seeds2_s_noise
                                    else:
                                        sde_noise = None

                                    _, denoised_2, _ = _dpm_eval_eps(
                                        x_2,
                                        sigma_value=sigma_s_1_f,
                                        current_step=current_step,
                                        step_total=run_total_steps,
                                    )

                                    denoised_d = torch.lerp(
                                        denoised,
                                        denoised_2,
                                        1.0 / (2.0 * seeds2_r),
                                    )
                                    x = (
                                        (sigma_next_f / sigma_f) * math.exp(-h_f * seeds2_eta) * x
                                    ) - (alpha_t * ei_h_phi_1(-h_eta) * denoised_d)
                                    if seeds2_s_noise > 0.0:
                                        if seeded_step_rng is None or sde_noise is None:
                                            raise RuntimeError("SEEDS-2 stochastic path lost deterministic noise state.")
                                        segment_factor = (seeds2_r - 1.0) * h_f * seeds2_eta
                                        noise_scale_2_sq = -math.expm1(2.0 * segment_factor)
                                        if noise_scale_2_sq < -1e-8:
                                            raise RuntimeError(
                                                "SEEDS-2 produced a negative second stochastic noise scale "
                                                f"({noise_scale_2_sq}) at step={step_index + 1}."
                                            )
                                        sde_noise = (
                                            sde_noise * math.exp(segment_factor)
                                            + math.sqrt(max(noise_scale_2_sq, 0.0))
                                            * self._next_seeded_step_noise(seeded_step_rng, x)
                                        )
                                        x = x + sde_noise * sigma_next_f * seeds2_s_noise
                            elif sampler_kind is SamplerKind.SEEDS_3:
                                if seeds3_sigmas is None or seeds3_lambdas is None:
                                    raise RuntimeError("SEEDS-3 missing initialized half-logSNR runtime state.")
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if not math.isfinite(sigma_f) or not math.isfinite(sigma_next_f):
                                    raise RuntimeError(
                                        "SEEDS-3 requires finite sigma values "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                if sigma_f <= 0.0 or sigma_next_f < 0.0:
                                    raise RuntimeError(
                                        "SEEDS-3 requires a strictly positive current sigma and a non-negative next sigma "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                if sigma_next_f == 0.0:
                                    x = denoised
                                else:
                                    lambda_s = seeds3_lambdas[step_index]
                                    lambda_t = seeds3_lambdas[step_index + 1]
                                    h = lambda_t - lambda_s
                                    h_f = float(h)
                                    if not math.isfinite(h_f) or h_f <= 0.0:
                                        raise RuntimeError(
                                            f"SEEDS-3 produced invalid half-logSNR step size h={h_f} at step={step_index + 1}."
                                        )
                                    h_eta = h * (seeds3_eta + 1.0)
                                    lambda_s_1 = torch.lerp(lambda_s, lambda_t, seeds3_r_1)
                                    lambda_s_2 = torch.lerp(lambda_s, lambda_t, seeds3_r_2)
                                    sigma_s_1 = half_log_snr_to_sigma(
                                        lambda_s_1,
                                        prediction_type=prediction_type,
                                    )
                                    sigma_s_2 = half_log_snr_to_sigma(
                                        lambda_s_2,
                                        prediction_type=prediction_type,
                                    )
                                    sigma_s_1_f = float(sigma_s_1)
                                    sigma_s_2_f = float(sigma_s_2)
                                    if not math.isfinite(sigma_s_1_f) or sigma_s_1_f <= 0.0:
                                        raise RuntimeError(
                                            f"SEEDS-3 produced invalid first intermediate sigma {sigma_s_1_f} at step={step_index + 1}."
                                        )
                                    if not math.isfinite(sigma_s_2_f) or sigma_s_2_f <= 0.0:
                                        raise RuntimeError(
                                            f"SEEDS-3 produced invalid second intermediate sigma {sigma_s_2_f} at step={step_index + 1}."
                                        )
                                    alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
                                    alpha_s_2 = sigma_s_2 * lambda_s_2.exp()
                                    alpha_t = seeds3_sigmas[step_index + 1] * lambda_t.exp()
                                    if not bool(torch.isfinite(alpha_s_1)) or not bool(torch.isfinite(alpha_s_2)) or not bool(torch.isfinite(alpha_t)):
                                        raise RuntimeError(
                                            f"SEEDS-3 produced non-finite alpha values at step={step_index + 1}."
                                        )

                                    x_2 = (
                                        (sigma_s_1_f / sigma_f) * math.exp(-seeds3_r_1 * h_f * seeds3_eta) * x
                                    ) - (alpha_s_1 * ei_h_phi_1(-seeds3_r_1 * h_eta) * denoised)
                                    if seeds3_s_noise > 0.0:
                                        if seeded_step_rng is None:
                                            raise RuntimeError("SEEDS-3 missing deterministic noise RNG.")
                                        noise_scale_1_sq = -math.expm1(-2.0 * seeds3_r_1 * h_f * seeds3_eta)
                                        if noise_scale_1_sq < -1e-8:
                                            raise RuntimeError(
                                                "SEEDS-3 produced a negative first stochastic noise scale "
                                                f"({noise_scale_1_sq}) at step={step_index + 1}."
                                            )
                                        sde_noise = (
                                            math.sqrt(max(noise_scale_1_sq, 0.0))
                                            * self._next_seeded_step_noise(seeded_step_rng, x)
                                        )
                                        x_2 = x_2 + sde_noise * sigma_s_1_f * seeds3_s_noise
                                    else:
                                        sde_noise = None

                                    _, denoised_2, _ = _dpm_eval_eps(
                                        x_2,
                                        sigma_value=sigma_s_1_f,
                                        current_step=current_step,
                                        step_total=run_total_steps,
                                    )

                                    a3_2 = (seeds3_r_2 / seeds3_r_1) * ei_h_phi_2(-seeds3_r_2 * h_eta)
                                    a3_1 = ei_h_phi_1(-seeds3_r_2 * h_eta) - a3_2
                                    x_3 = (
                                        (sigma_s_2_f / sigma_f) * math.exp(-seeds3_r_2 * h_f * seeds3_eta) * x
                                    ) - (alpha_s_2 * (a3_1 * denoised + a3_2 * denoised_2))
                                    if seeds3_s_noise > 0.0:
                                        if seeded_step_rng is None or sde_noise is None:
                                            raise RuntimeError("SEEDS-3 stochastic path lost deterministic noise state.")
                                        segment_factor = (seeds3_r_1 - seeds3_r_2) * h_f * seeds3_eta
                                        noise_scale_2_sq = -math.expm1(2.0 * segment_factor)
                                        if noise_scale_2_sq < -1e-8:
                                            raise RuntimeError(
                                                "SEEDS-3 produced a negative second stochastic noise scale "
                                                f"({noise_scale_2_sq}) at step={step_index + 1}."
                                            )
                                        sde_noise = (
                                            sde_noise * math.exp(segment_factor)
                                            + math.sqrt(max(noise_scale_2_sq, 0.0))
                                            * self._next_seeded_step_noise(seeded_step_rng, x)
                                        )
                                        x_3 = x_3 + sde_noise * sigma_s_2_f * seeds3_s_noise

                                    _, denoised_3, _ = _dpm_eval_eps(
                                        x_3,
                                        sigma_value=sigma_s_2_f,
                                        current_step=current_step,
                                        step_total=run_total_steps,
                                    )

                                    b3 = ei_h_phi_2(-h_eta) / seeds3_r_2
                                    b1 = ei_h_phi_1(-h_eta) - b3
                                    x = (
                                        (sigma_next_f / sigma_f) * math.exp(-h_f * seeds3_eta) * x
                                    ) - (alpha_t * (b1 * denoised + b3 * denoised_3))
                                    if seeds3_s_noise > 0.0:
                                        if seeded_step_rng is None or sde_noise is None:
                                            raise RuntimeError("SEEDS-3 stochastic path lost deterministic noise state.")
                                        segment_factor = (seeds3_r_2 - 1.0) * h_f * seeds3_eta
                                        noise_scale_3_sq = -math.expm1(2.0 * segment_factor)
                                        if noise_scale_3_sq < -1e-8:
                                            raise RuntimeError(
                                                "SEEDS-3 produced a negative third stochastic noise scale "
                                                f"({noise_scale_3_sq}) at step={step_index + 1}."
                                            )
                                        sde_noise = (
                                            sde_noise * math.exp(segment_factor)
                                            + math.sqrt(max(noise_scale_3_sq, 0.0))
                                            * self._next_seeded_step_noise(seeded_step_rng, x)
                                        )
                                        x = x + sde_noise * sigma_next_f * seeds3_s_noise
                            elif sampler_kind is SamplerKind.DDIM:
                                x = denoised + float(sigma_next) * eps
                            elif sampler_kind in {SamplerKind.SA_SOLVER, SamplerKind.SA_SOLVER_PECE}:
                                if (
                                    sa_solver_sigmas is None
                                    or sa_solver_lambdas is None
                                    or sa_solver_tau_func is None
                                ):
                                    raise RuntimeError("SA-Solver missing initialized runtime state.")
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if not math.isfinite(sigma_f) or not math.isfinite(sigma_next_f):
                                    raise RuntimeError(
                                        "SA-Solver requires finite sigma values "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                if sigma_f <= 0.0 or sigma_next_f < 0.0:
                                    raise RuntimeError(
                                        "SA-Solver requires a strictly positive current sigma and a non-negative next sigma "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                max_used_order = max(sa_solver_predictor_order, sa_solver_corrector_order)
                                sa_solver_pred_list.append(denoised.detach())
                                sa_solver_pred_list = sa_solver_pred_list[-max_used_order:]
                                predictor_order_used = min(sa_solver_predictor_order, len(sa_solver_pred_list))
                                if step_index == 0 or (sigma_next_f == 0.0 and not sa_solver_use_pece):
                                    corrector_order_used = 0
                                else:
                                    corrector_order_used = min(
                                        sa_solver_corrector_order,
                                        len(sa_solver_pred_list),
                                    )
                                if sa_solver_lower_order_to_end:
                                    predictor_order_used = min(
                                        predictor_order_used,
                                        int(sa_solver_sigmas.numel()) - 2 - step_index,
                                    )
                                    corrector_order_used = min(
                                        corrector_order_used,
                                        int(sa_solver_sigmas.numel()) - 1 - step_index,
                                    )

                                x_current = x
                                if corrector_order_used > 0:
                                    prev_sigma_f = float(sa_solver_sigmas[step_index - 1])
                                    if prev_sigma_f <= 0.0:
                                        raise RuntimeError(
                                            "SA-Solver corrector requires a strictly positive previous sigma "
                                            f"(prev_sigma={prev_sigma_f})."
                                        )
                                    curr_lambdas = sa_solver_lambdas[
                                        step_index - corrector_order_used + 1 : step_index + 1
                                    ]
                                    if int(curr_lambdas.numel()) != corrector_order_used:
                                        raise RuntimeError(
                                            "SA-Solver corrector history window mismatch "
                                            f"(expected={corrector_order_used}, actual={int(curr_lambdas.numel())})."
                                        )
                                    b_coeffs = compute_stochastic_adams_b_coeffs(
                                        sa_solver_sigmas[step_index],
                                        curr_lambdas,
                                        sa_solver_lambdas[step_index - 1],
                                        sa_solver_lambdas[step_index],
                                        sa_solver_prev_tau_t,
                                        simple_order_2=sa_solver_simple_order_2,
                                        is_corrector_step=True,
                                    )
                                    pred_mat = torch.stack(
                                        sa_solver_pred_list[-corrector_order_used:],
                                        dim=1,
                                    )
                                    corr_res = torch.tensordot(
                                        pred_mat,
                                        b_coeffs.to(device=pred_mat.device, dtype=pred_mat.dtype),
                                        dims=([1], [0]),
                                    )
                                    x_current = (
                                        (sigma_f / prev_sigma_f)
                                        * math.exp(-(sa_solver_prev_tau_t**2) * sa_solver_prev_h)
                                        * x
                                    ) + corr_res
                                    if sa_solver_prev_tau_t > 0.0 and sa_solver_s_noise > 0.0:
                                        if sa_solver_prev_noise is None:
                                            raise RuntimeError(
                                                "SA-Solver corrector expected cached predictor noise from the previous step."
                                            )
                                        x_current = x_current + sa_solver_prev_noise
                                    if sa_solver_use_pece:
                                        corrected_state = x_current
                                        if pre_denoiser_hook is not None:
                                            corrected_state = pre_denoiser_hook(
                                                corrected_state,
                                                sigma_batch,
                                                current_step,
                                                run_total_steps,
                                            )
                                            if not isinstance(corrected_state, torch.Tensor):
                                                raise RuntimeError("pre_denoiser_hook must return a torch.Tensor")
                                            if tuple(corrected_state.shape) != tuple(x.shape):
                                                raise RuntimeError(
                                                    "pre_denoiser_hook returned unexpected shape "
                                                    f"{tuple(corrected_state.shape)}; expected {tuple(x.shape)}"
                                                )
                                        denoised = sampling_function_inner(
                                            model,
                                            corrected_state,
                                            sigma_batch,
                                            compiled_uncond,
                                            compiled_cond,
                                            cfg_scale,
                                            denoiser.model_options,
                                            seed=None,
                                            return_full=False,
                                        )
                                        if post_denoiser_hook is not None:
                                            denoised = post_denoiser_hook(
                                                denoised,
                                                sigma_batch,
                                                current_step,
                                                run_total_steps,
                                            )
                                            if not isinstance(denoised, torch.Tensor):
                                                raise RuntimeError("post_denoiser_hook must return a torch.Tensor")
                                            if tuple(denoised.shape) != tuple(corrected_state.shape):
                                                raise RuntimeError(
                                                    "post_denoiser_hook returned unexpected shape "
                                                    f"{tuple(denoised.shape)}; expected {tuple(corrected_state.shape)}"
                                                )
                                        if strict and torch.isnan(denoised).any():
                                            reason = f"NaN detected at sampling step {i + 1}"
                                            emit_backend_message(
                                                "NaN encountered during sampling; attempting precision fallback.",
                                                logger=self._logger_name,
                                                level="WARNING",
                                                step=i + 1,
                                                dtype=str(getattr(model, "computation_dtype", x.dtype)),
                                            )
                                            next_dtype = memory_management.manager.report_precision_failure(
                                                DeviceRole.CORE,
                                                location=f"sampler.step_{i + 1}",
                                                reason=reason,
                                            )
                                            if next_dtype is None:
                                                hint = memory_management.manager.precision_hint(DeviceRole.CORE)
                                                raise RuntimeError(
                                                    f"Diffusion core produced NaNs at step {i + 1} on {noise.device} "
                                                    f"with dtype {getattr(model, 'computation_dtype', x.dtype)}. {hint}"
                                                )
                                            self._rebind_unet_precision(next_dtype)
                                            retry = True
                                            raise _PrecisionFallbackRequest
                                        sa_solver_pred_list[-1] = denoised.detach()

                                if sigma_next_f == 0.0:
                                    x = denoised
                                    sa_solver_prev_noise = None
                                    sa_solver_prev_h = 0.0
                                    sa_solver_prev_tau_t = 0.0
                                else:
                                    tau_t = float(sa_solver_tau_func(sigma_next_f))
                                    if not math.isfinite(tau_t) or tau_t < 0.0:
                                        raise RuntimeError(
                                            f"SA-Solver produced invalid tau_t={tau_t} at step={step_index + 1}."
                                        )
                                    if predictor_order_used <= 0:
                                        raise RuntimeError(
                                            "SA-Solver predictor requires at least one history entry before the next sigma step."
                                        )
                                    curr_lambdas = sa_solver_lambdas[
                                        step_index - predictor_order_used + 1 : step_index + 1
                                    ]
                                    if int(curr_lambdas.numel()) != predictor_order_used:
                                        raise RuntimeError(
                                            "SA-Solver predictor history window mismatch "
                                            f"(expected={predictor_order_used}, actual={int(curr_lambdas.numel())})."
                                        )
                                    b_coeffs = compute_stochastic_adams_b_coeffs(
                                        sa_solver_sigmas[step_index + 1],
                                        curr_lambdas,
                                        sa_solver_lambdas[step_index],
                                        sa_solver_lambdas[step_index + 1],
                                        tau_t,
                                        simple_order_2=sa_solver_simple_order_2,
                                        is_corrector_step=False,
                                    )
                                    pred_mat = torch.stack(
                                        sa_solver_pred_list[-predictor_order_used:],
                                        dim=1,
                                    )
                                    pred_res = torch.tensordot(
                                        pred_mat,
                                        b_coeffs.to(device=pred_mat.device, dtype=pred_mat.dtype),
                                        dims=([1], [0]),
                                    )
                                    sa_solver_prev_h = float(
                                        sa_solver_lambdas[step_index + 1] - sa_solver_lambdas[step_index]
                                    )
                                    if not math.isfinite(sa_solver_prev_h) or sa_solver_prev_h <= 0.0:
                                        raise RuntimeError(
                                            f"SA-Solver produced invalid step size h={sa_solver_prev_h} at step={step_index + 1}."
                                        )
                                    x = (
                                        (sigma_next_f / sigma_f)
                                        * math.exp(-(tau_t**2) * sa_solver_prev_h)
                                        * x_current
                                    ) + pred_res
                                    if tau_t > 0.0 and sa_solver_s_noise > 0.0:
                                        if seeded_step_rng is None:
                                            raise RuntimeError("SA-Solver missing deterministic noise RNG.")
                                        noise_scale_sq = -math.expm1(-2.0 * (tau_t**2) * sa_solver_prev_h)
                                        if noise_scale_sq < -1e-8:
                                            raise RuntimeError(
                                                "SA-Solver produced a negative stochastic noise scale "
                                                f"({noise_scale_sq}) at step={step_index + 1}."
                                            )
                                        noise_scale = math.sqrt(max(noise_scale_sq, 0.0))
                                        sa_solver_prev_noise = (
                                            self._next_seeded_step_noise(seeded_step_rng, x)
                                            * sigma_next_f
                                            * noise_scale
                                            * sa_solver_s_noise
                                        )
                                        x = x + sa_solver_prev_noise
                                    else:
                                        sa_solver_prev_noise = None
                                    sa_solver_prev_tau_t = tau_t
                            elif sampler_kind in {SamplerKind.UNI_PC, SamplerKind.UNI_PC_BH2}:
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if not math.isfinite(sigma_f) or not math.isfinite(sigma_next_f):
                                    raise RuntimeError(
                                        "UNI_PC requires finite sigma values "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                if sigma_f < 0.0 or sigma_next_f < 0.0:
                                    raise RuntimeError(
                                        "UNI_PC requires non-negative sigmas "
                                        f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                    )
                                variant = "bh1" if sampler_kind is SamplerKind.UNI_PC else "bh2"
                                if sigma_f == 0.0:
                                    if sigma_next_f > 0.0:
                                        raise RuntimeError(
                                            "UNI_PC encountered non-monotonic zero-to-positive sigma transition "
                                            f"(sigma={sigma_f}, sigma_next={sigma_next_f})."
                                        )
                                    # SIMPLE/CONST flow ladders may end with a supported double-zero tail.
                                    # Treat the preterminal zero step as a terminal no-op.
                                    x = denoised
                                    uni_pc_history_denoised = []
                                    uni_pc_history_lambdas = []
                                else:
                                    sigma_solver_next = 1e-3 if sigma_next_f == 0.0 else sigma_next_f
                                    sigma_ratio = sigma_solver_next / sigma_f
                                    if not math.isfinite(sigma_ratio):
                                        raise RuntimeError(
                                            f"UNI_PC produced non-finite sigma ratio: sigma={sigma_f}, sigma_next={sigma_next_f}"
                                        )

                                    lambda_s = -math.log(max(sigma_f, 1e-12))
                                    lambda_t = -math.log(max(sigma_solver_next, 1e-12))
                                    h = lambda_t - lambda_s
                                    hh = -h
                                    if not math.isfinite(hh):
                                        raise RuntimeError(
                                            f"UNI_PC produced non-finite hh from sigma={sigma_f}, sigma_next={sigma_next_f}"
                                        )
                                    current_history_denoised = [*uni_pc_history_denoised, denoised.detach()]
                                    current_history_lambdas = [*uni_pc_history_lambdas, lambda_s]
                                    local_step_index = step_index
                                    remaining_active_steps = max(1, run_total_steps - local_step_index)
                                    step_order = min(
                                        uni_pc_order_cap,
                                        len(current_history_denoised),
                                        remaining_active_steps,
                                    )
                                    history_denoised = current_history_denoised[-step_order:]
                                    history_lambdas = current_history_lambdas[-step_order:]
                                    previous_rks: list[float] = []
                                    d1_terms: list[torch.Tensor] = []
                                    for history_index in range(1, step_order):
                                        lambda_prev = history_lambdas[-(history_index + 1)]
                                        rk = (lambda_prev - lambda_s) / h
                                        if not math.isfinite(rk):
                                            raise RuntimeError(
                                                "UNI_PC produced non-finite rk "
                                                f"(variant={variant}, lambda_prev={lambda_prev}, lambda_s={lambda_s}, h={h})."
                                            )
                                        if abs(rk) <= 1e-12:
                                            raise RuntimeError(
                                                "UNI_PC encountered degenerate rk≈0 "
                                                f"(variant={variant}, lambda_prev={lambda_prev}, lambda_s={lambda_s}, h={h})."
                                            )
                                        previous_rks.append(rk)
                                        d1_terms.append((history_denoised[-(history_index + 1)] - denoised) / rk)

                                    b_h, predictor_coeffs, corrector_coeffs = self._resolve_uni_pc_bh_coefficients(
                                        order=step_order,
                                        previous_rks=previous_rks,
                                        hh=hh,
                                        variant=variant,
                                    )
                                    if not math.isfinite(b_h):
                                        raise RuntimeError(
                                            f"UNI_PC produced non-finite B_h from hh={hh}."
                                        )

                                    x_base = sigma_ratio * x + (1.0 - sigma_ratio) * denoised

                                    x_pred = x_base
                                    if predictor_coeffs is not None and d1_terms:
                                        predictor_residual = sum(
                                            coeff * term for coeff, term in zip(predictor_coeffs, d1_terms)
                                        )
                                        x_pred = x_base - b_h * predictor_residual

                                    if current_step >= run_total_steps:
                                        x = x_pred
                                    else:
                                        sigma_next_batch = torch.full(
                                            (x.shape[0],),
                                            sigma_solver_next,
                                            device=x.device,
                                            dtype=torch.float32,
                                        )
                                        if pre_denoiser_hook is not None:
                                            x_pred = pre_denoiser_hook(x_pred, sigma_next_batch, current_step, run_total_steps)
                                            if not isinstance(x_pred, torch.Tensor):
                                                raise RuntimeError("pre_denoiser_hook must return a torch.Tensor")
                                            if tuple(x_pred.shape) != tuple(x.shape):
                                                raise RuntimeError(
                                                    "pre_denoiser_hook returned unexpected shape "
                                                    f"{tuple(x_pred.shape)}; expected {tuple(x.shape)}"
                                                )
                                        denoised_next = sampling_function_inner(
                                            model,
                                            x_pred,
                                            sigma_next_batch,
                                            compiled_uncond,
                                            compiled_cond,
                                            cfg_scale,
                                            denoiser.model_options,
                                            seed=None,
                                            return_full=False,
                                        )
                                        if post_denoiser_hook is not None:
                                            denoised_next = post_denoiser_hook(
                                                denoised_next,
                                                sigma_next_batch,
                                                current_step,
                                                run_total_steps,
                                            )
                                            if not isinstance(denoised_next, torch.Tensor):
                                                raise RuntimeError("post_denoiser_hook must return a torch.Tensor")
                                            if tuple(denoised_next.shape) != tuple(x_pred.shape):
                                                raise RuntimeError(
                                                    "post_denoiser_hook returned unexpected shape "
                                                    f"{tuple(denoised_next.shape)}; expected {tuple(x_pred.shape)}"
                                                )
                                        d1_t = denoised_next - denoised
                                        if len(corrector_coeffs) == 1:
                                            correction = corrector_coeffs[0] * d1_t
                                        else:
                                            correction = sum(
                                                coeff * term for coeff, term in zip(corrector_coeffs[:-1], d1_terms)
                                            ) + corrector_coeffs[-1] * d1_t
                                        x = x_base - b_h * correction
                                    uni_pc_history_denoised = current_history_denoised[-2:]
                                    uni_pc_history_lambdas = current_history_lambdas[-2:]
                            else:
                                raise NotImplementedError(f"Sampler '{sampler_kind.value}' is not implemented natively yet")

                            if post_step_hook is not None:
                                post_step_hook(x, current_step, run_total_steps)

                            if preview_callback is not None and (
                                (preview_interval > 0 and (current_step % preview_interval == 0))
                                or current_step == run_total_steps
                            ):
                                try:
                                    preview_callback(denoised.detach(), current_step, run_total_steps)
                                except Exception:
                                    pass

                            if self._log_enabled and (
                                i == 0 or (i + 1) == steps or (i + 1) % max(1, steps // 5) == 0
                            ):
                                eps_norm = float(eps.norm().item()) if hasattr(eps, "norm") else float("nan")
                                den_norm = float(denoised.norm().item()) if hasattr(denoised, "norm") else float("nan")
                                self._emit_event(
                                    "sampling.step",
                                    step=i + 1,
                                    total_steps=steps,
                                    sigma=float(sigma),
                                    sigma_next=float(sigma_next),
                                    norm_x=float(x.norm().item()),
                                    norm_eps=eps_norm,
                                    norm_den=den_norm,
                                    dt_ms=(_time.perf_counter() - t0) * 1000.0,
                                )
                                t0 = _time.perf_counter()

                            backend_state.tick(sampling_step=current_step, owner_token=progress_owner_token)
                            backend_state.reset_sampling_blocks(owner_token=progress_owner_token)
                            if (
                                capture_boundary_state_at_step is not None
                                and current_step == int(capture_boundary_state_at_step)
                            ):
                                captured_boundary_state = self._build_sampling_boundary_state(
                                    processing=processing,
                                    active_context=active_context,
                                    steps=steps,
                                    current_step=current_step,
                                    prediction_type=prediction_type,
                                    sigmas=sigmas,
                                    latent=x,
                                    old_denoised=old_denoised,
                                    older_denoised=older_denoised,
                                    old_denoised_d=old_denoised_d,
                                    gradient_estimation_prev_d=gradient_estimation_prev_d,
                                    t_prev=t_prev,
                                    h_prev=h_prev,
                                    h_prev_2=h_prev_2,
                                    eps_history=eps_history,
                                    res_multistep_old_sigma_down=res_multistep_old_sigma_down,
                                    uni_pc_history_denoised=uni_pc_history_denoised,
                                    uni_pc_history_lambdas=uni_pc_history_lambdas,
                                    sa_solver_pred_list=sa_solver_pred_list,
                                    sa_solver_prev_noise=sa_solver_prev_noise,
                                    sa_solver_prev_h=sa_solver_prev_h,
                                    sa_solver_prev_tau_t=sa_solver_prev_tau_t,
                                    guidance_apg_momentum_buffer=denoiser.model_options.get(
                                        _GUIDANCE_APG_MOMENTUM_BUFFER_KEY
                                    ),
                                    seeded_step_rng=seeded_step_rng,
                                )
                                break
                        profiler.step()

                sampling_cleanup(denoiser)
                prepared = False

                backend_state.end()
                state_started = False

                if captured_boundary_state is not None:
                    return SamplingResult(samples=x, boundary_state=captured_boundary_state)
                if post_sample_hook is not None:
                    x = post_sample_hook(x)

                return SamplingResult(samples=x)
            except _PrecisionFallbackRequest:
                emit_backend_message(
                    "Precision fallback requested for diffusion core; retrying with next dtype.",
                    logger=self._logger_name,
                    level="WARNING",
                )
            except _SamplingCancelled:
                emit_backend_message(
                    "Sampling cancelled by request; aborting current run.",
                    logger=self._logger_name,
                )
                raise RuntimeError("cancelled")
            finally:
                if block_progress_controller is not None:
                    block_progress_controller.close()
                    block_progress_controller = None
                if prepared:
                    sampling_cleanup(denoiser)
                if state_started:
                    backend_state.end()
                denoiser.model_options.pop(_GUIDANCE_POLICY_KEY, None)
                denoiser.model_options.pop(_GUIDANCE_STEP_INDEX_KEY, None)
                denoiser.model_options.pop(_GUIDANCE_TOTAL_STEPS_KEY, None)
                denoiser.model_options.pop(_GUIDANCE_APG_MOMENTUM_BUFFER_KEY, None)
                denoiser.model_options.pop(_GUIDANCE_WARNED_SAMPLER_CFG_KEY, None)
                if res_multistep_cfg_pp_saved_post_cfg_functions is not None:
                    if res_multistep_cfg_pp_saved_post_cfg_functions:
                        denoiser.model_options["sampler_post_cfg_function"] = list(
                            res_multistep_cfg_pp_saved_post_cfg_functions
                        )
                    else:
                        denoiser.model_options.pop("sampler_post_cfg_function", None)
                    if res_multistep_cfg_pp_saved_disable_cfg1 is res_multistep_cfg_pp_disable_cfg1_missing:
                        denoiser.model_options.pop("disable_cfg1_optimization", None)
                    else:
                        denoiser.model_options["disable_cfg1_optimization"] = res_multistep_cfg_pp_saved_disable_cfg1
                if gradient_estimation_cfg_pp_saved_post_cfg_functions is not None:
                    if gradient_estimation_cfg_pp_saved_post_cfg_functions:
                        denoiser.model_options["sampler_post_cfg_function"] = list(
                            gradient_estimation_cfg_pp_saved_post_cfg_functions
                        )
                    else:
                        denoiser.model_options.pop("sampler_post_cfg_function", None)
                    if gradient_estimation_cfg_pp_saved_disable_cfg1 is gradient_estimation_cfg_pp_disable_cfg1_missing:
                        denoiser.model_options.pop("disable_cfg1_optimization", None)
                    else:
                        denoiser.model_options["disable_cfg1_optimization"] = gradient_estimation_cfg_pp_saved_disable_cfg1
                if euler_cfg_pp_saved_post_cfg_functions is not None:
                    if euler_cfg_pp_saved_post_cfg_functions:
                        denoiser.model_options["sampler_post_cfg_function"] = list(
                            euler_cfg_pp_saved_post_cfg_functions
                        )
                    else:
                        denoiser.model_options.pop("sampler_post_cfg_function", None)
                    if euler_cfg_pp_saved_disable_cfg1 is euler_cfg_pp_disable_cfg1_missing:
                        denoiser.model_options.pop("disable_cfg1_optimization", None)
                    else:
                        denoiser.model_options["disable_cfg1_optimization"] = euler_cfg_pp_saved_disable_cfg1
                if dpm2m_cfg_pp_saved_post_cfg_functions is not None:
                    if dpm2m_cfg_pp_saved_post_cfg_functions:
                        denoiser.model_options["sampler_post_cfg_function"] = list(
                            dpm2m_cfg_pp_saved_post_cfg_functions
                        )
                    else:
                        denoiser.model_options.pop("sampler_post_cfg_function", None)
                    if dpm2m_cfg_pp_saved_disable_cfg1 is dpm2m_cfg_pp_disable_cfg1_missing:
                        denoiser.model_options.pop("disable_cfg1_optimization", None)
                    else:
                        denoiser.model_options["disable_cfg1_optimization"] = dpm2m_cfg_pp_saved_disable_cfg1
                if dpm2s_ancestral_cfg_pp_saved_post_cfg_functions is not None:
                    if dpm2s_ancestral_cfg_pp_saved_post_cfg_functions:
                        denoiser.model_options["sampler_post_cfg_function"] = list(
                            dpm2s_ancestral_cfg_pp_saved_post_cfg_functions
                        )
                    else:
                        denoiser.model_options.pop("sampler_post_cfg_function", None)
                    if dpm2s_ancestral_cfg_pp_saved_disable_cfg1 is dpm2s_ancestral_cfg_pp_disable_cfg1_missing:
                        denoiser.model_options.pop("disable_cfg1_optimization", None)
                    else:
                        denoiser.model_options["disable_cfg1_optimization"] = dpm2s_ancestral_cfg_pp_saved_disable_cfg1
                transformer_options = denoiser.model_options.get("transformer_options", None)
                if isinstance(transformer_options, dict):
                    transformer_options.pop(BLOCK_PROGRESS_CALLBACK_KEY, None)
                backend_state.clear_flags()

            if retry:
                memory_management.manager.soft_empty_cache(force=True)
                continue



__all__ = ["CodexSampler"]
