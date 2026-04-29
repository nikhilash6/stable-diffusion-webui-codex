"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampling plan construction helpers for pipeline orchestration.
Validates sampler/scheduler selection, resolves noise settings, builds a `SamplingPlan`, and prepares the active sampler with optional RNG installation.

Symbols (top-level; keep in sync; no ghosts):
- `_normalize_scheduler_name` (function): Validate a scheduler name for the given sampler.
- `resolve_sampler_scheduler_override` (function): Resolve sampler/scheduler for a derived plan (e.g., hires pass) with override semantics.
- `resolve_noise_settings` (function): Derive `NoiseSettings` for a run from processing overrides and env.
- `_resolve_er_sde_payload_options` (function): Build normalized typed ER-SDE options from processing overrides.
- `resolve_er_sde_options_for_sampler` (function): Attach ER-SDE options only when the active sampler is ER-SDE.
- `build_sampling_plan` (function): Build a `SamplingPlan` from processing state and explicit seeds/subseeds.
- `ensure_sampler` (function): Ensure `processing.sampler` exists for the current sampling plan without installing a fresh RNG.
- `ensure_sampler_and_rng` (function): Ensure `processing.sampler` and `processing.rng` exist for the current sampling plan.
"""

from __future__ import annotations

from typing import Any, Sequence

from apps.backend.core.rng import ImageRNG, NoiseSettings, NoiseSourceKind
from apps.backend.runtime.processing.datatypes import ErSdeOptions, SamplingPlan
from apps.backend.runtime.sampling import SUPPORTED_SCHEDULERS
from apps.backend.runtime.sampling.context import SchedulerName
from apps.backend.runtime.sampling.driver import CodexSampler
from apps.backend.runtime.sampling.registry import get_sampler_spec

def _normalize_scheduler_name(sampler: str, scheduler: str) -> str:
    if scheduler not in SUPPORTED_SCHEDULERS:
        raise ValueError(f"Scheduler '{scheduler}' is not supported")
    try:
        canonical_enum = SchedulerName.from_string(scheduler)
    except ValueError as exc:
        raise ValueError(f"Unsupported scheduler '{scheduler}'") from exc
    spec = get_sampler_spec(sampler)
    if not spec.is_supported_scheduler(canonical_enum.value):
        raise ValueError(f"Scheduler '{scheduler}' is not supported for sampler '{sampler}'")
    return canonical_enum.value


def resolve_sampler_scheduler_override(
    *,
    base_sampler: str,
    base_scheduler: str,
    sampler_override: str | None,
    scheduler_override: str | None,
) -> tuple[str, str]:
    """Resolve sampler/scheduler selection for a derived sampling plan (e.g., hires pass).

    Semantics:
    - If `sampler_override` is set, it becomes the sampler for the derived plan.
      - If `scheduler_override` is NOT set, the scheduler defaults to the sampler's default scheduler.
    - If only `scheduler_override` is set, it is validated against the base sampler.
    - If neither override is set, base sampler/scheduler are kept.
    """

    base_sampler_value = str(base_sampler or "").strip()
    base_scheduler_value = str(base_scheduler or "").strip()
    if not base_sampler_value:
        raise ValueError("base_sampler must be a non-empty sampler name")
    if not base_scheduler_value:
        raise ValueError("base_scheduler must be a non-empty scheduler name")

    def _normalize_override(value: str | None, *, kind: str) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{kind}_override must be a string when provided")
        normalized = value.strip()
        if normalized == "":
            return ""
        lowered = normalized.lower()
        if kind == "sampler" and lowered in {"use same sampler", "use same"}:
            raise ValueError(
                "sampler_override does not accept 'use same*' sentinels; "
                "omit sampler_override or pass an empty string to inherit the base sampler."
            )
        if kind == "scheduler" and lowered in {"use same scheduler", "use same"}:
            raise ValueError(
                "scheduler_override does not accept 'use same*' sentinels; "
                "omit scheduler_override or pass an empty string to inherit the base scheduler."
            )
        return normalized

    sampler_override_value = _normalize_override(sampler_override, kind="sampler")
    scheduler_override_value = _normalize_override(scheduler_override, kind="scheduler")

    sampler_name = sampler_override_value or base_sampler_value
    if scheduler_override_value:
        scheduler_name = scheduler_override_value
    elif sampler_override_value:
        scheduler_name = get_sampler_spec(sampler_name).default_scheduler
    else:
        scheduler_name = base_scheduler_value

    normalized_scheduler = _normalize_scheduler_name(sampler_name, scheduler_name)
    return sampler_name, normalized_scheduler


def resolve_noise_settings(processing: Any) -> NoiseSettings:
    """Inspect processing overrides/environment and return noise source settings."""
    source = None
    eta_delta = 0
    overrides = getattr(processing, "override_settings", {})
    if isinstance(overrides, dict):
        source = overrides.get("noise_source")
        eta_delta = overrides.get("eta_noise_seed_delta", eta_delta)
    metadata = getattr(processing, "metadata", {})
    if isinstance(metadata, dict):
        source = metadata.get("noise_source", source)
    if getattr(processing, "noise_source", None):
        source = processing.noise_source

    if source is None:
        source_kind = NoiseSourceKind.GPU
    else:
        if not isinstance(source, str):
            raise ValueError(
                "noise_source must be a string when provided "
                f"(got {type(source).__name__})."
            )
        normalized_source = source.strip()
        if not normalized_source:
            source_kind = NoiseSourceKind.GPU
        else:
            try:
                source_kind = NoiseSourceKind.from_string(normalized_source)
            except ValueError as exc:
                allowed = ", ".join(member.value for member in NoiseSourceKind)
                raise ValueError(
                    f"Invalid noise_source value {source!r}. "
                    f"Allowed: {allowed}."
                ) from exc

    delta = int(getattr(processing, "eta_noise_seed_delta", eta_delta) or eta_delta or 0)
    settings = NoiseSettings(source=source_kind, eta_noise_seed_delta=delta)
    processing.eta_noise_seed_delta = settings.eta_noise_seed_delta
    return settings


def _resolve_er_sde_payload_options(processing: Any) -> ErSdeOptions | None:
    """Resolve ER-SDE options from processing overrides with strict validation."""
    overrides = getattr(processing, "override_settings", {})
    if not isinstance(overrides, dict):
        return None
    if "er_sde" not in overrides:
        return None
    normalized = CodexSampler._resolve_er_sde_runtime_params(overrides.get("er_sde"))
    return ErSdeOptions(
        solver_type=str(normalized["solver_type"]),
        max_stage=int(normalized["max_stage"]),
        eta=float(normalized["eta"]),
        s_noise=float(normalized["s_noise"]),
    )


def resolve_er_sde_options_for_sampler(processing: Any, sampler_name: str | None) -> ErSdeOptions | None:
    """Resolve ER-SDE options only for an active ER-SDE sampler selection."""
    if not isinstance(sampler_name, str) or sampler_name.strip().lower() != "er sde":
        return None
    return _resolve_er_sde_payload_options(processing)


def build_sampling_plan(
    processing: Any,
    seeds: Sequence[int],
    subseeds: Sequence[int],
    subseed_strength: float,
    noise_settings: NoiseSettings | None = None,
) -> SamplingPlan:
    """Create a sampling plan for the generation run."""
    if noise_settings is None:
        noise_settings = resolve_noise_settings(processing)
    guidance = float(getattr(processing, "guidance_scale", 7.0) or 7.0)
    steps = int(getattr(processing, "steps", 20) or 20)
    sampler_name = getattr(processing, "sampler_name", None)
    scheduler_name = getattr(processing, "scheduler", None)
    if not isinstance(sampler_name, str) or not sampler_name:
        raise ValueError("processing.sampler_name must be set to a non-empty sampler name")
    if not isinstance(scheduler_name, str) or not scheduler_name:
        raise ValueError("processing.scheduler must be set to a non-empty scheduler name")
    normalized_scheduler = _normalize_scheduler_name(sampler_name, scheduler_name)
    processing.scheduler = normalized_scheduler
    return SamplingPlan(
        sampler_name=sampler_name,
        scheduler_name=normalized_scheduler,
        steps=steps,
        guidance_scale=guidance,
        seeds=list(seeds),
        subseeds=list(subseeds),
        subseed_strength=float(subseed_strength),
        noise_settings=noise_settings,
        er_sde=resolve_er_sde_options_for_sampler(processing, sampler_name),
    )


def ensure_sampler(processing: Any, plan: SamplingPlan) -> CodexSampler:
    """Ensure processing has a sampler configured for the current plan."""
    algo = plan.sampler_name
    if not isinstance(algo, str) or not algo:
        raise ValueError("SamplingPlan.sampler_name must be a non-empty sampler name")
    processing.sampler = CodexSampler(processing.sd_model, algorithm=algo)
    return processing.sampler


def ensure_sampler_and_rng(
    processing: Any,
    plan: SamplingPlan,
    *,
    latent_channels: int | None = None,
) -> ImageRNG:
    """Ensure processing has a sampler + RNG configured for the current plan."""
    ensure_sampler(processing, plan)
    if latent_channels is None:
        latent_channels = getattr(
            processing.sd_model.codex_objects_after_applying_lora.vae,
            "latent_channels",
            4,
        )
    shape = (
        latent_channels,
        processing.height // 8,
        processing.width // 8,
    )
    rng = ImageRNG(
        shape,
        plan.seeds,
        subseeds=plan.subseeds,
        subseed_strength=plan.subseed_strength,
        seed_resize_from_h=getattr(processing, "seed_resize_from_h", 0),
        seed_resize_from_w=getattr(processing, "seed_resize_from_w", 0),
        settings=plan.noise_settings,
    )
    processing.rng = rng
    return rng
