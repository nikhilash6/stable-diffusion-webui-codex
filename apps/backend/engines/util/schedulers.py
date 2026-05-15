"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampler/scheduler mapping for diffusers pipelines.
Maps UI-facing sampler/scheduler selections to a strict diffusers scheduler instance for the bridge-supported sampler slice
(`euler`, `euler a`, `heun`, `lms`, `ddim`, `dpm++ 2m`, `dpm++ 2m sde`, `dpm++ 2m sde heun`, `dpm 2`, `dpm 2 ancestral`, `uni-pc`) and fail-loudly rejects
native-only sampler variants that this bridge does not implement (no silent fallbacks).

Symbols (top-level; keep in sync; no ghosts):
- `apply_sampler_scheduler` (function): Applies a sampler/scheduler selection to a pipeline and returns the effective outcome.
"""

from __future__ import annotations

from typing import List, Union

from apps.backend.types.samplers import SamplerKind, ApplyOutcome


def apply_sampler_scheduler(pipe, sampler: Union[str, SamplerKind], scheduler: str) -> ApplyOutcome:
    """Strict mapping of sampler/scheduler to Diffusers pipeline.

    - Allowed: euler a, euler, heun, lms, ddim, dpm++ 2m, dpm++ 2m sde, dpm++ 2m sde heun, dpm 2, dpm 2 ancestral, uni-pc.
    - Explicitly rejected: `uni-pc bh2` and `dpm++ 2s ancestral` (native variants not implemented by this bridge).
    - On invalid or failed application, raises with the root cause; no fallbacks.
    """
    wanted_sampler = sampler.value if isinstance(sampler, SamplerKind) else sampler
    if not isinstance(wanted_sampler, str) or not wanted_sampler:
        raise ValueError("sampler must be a non-empty string")
    if not isinstance(scheduler, str) or not scheduler:
        raise ValueError("scheduler must be a non-empty string")
    wanted_scheduler = scheduler
    eff_sampler = wanted_sampler
    eff_scheduler = wanted_scheduler
    warnings: List[str] = []

    kind = SamplerKind.from_string(wanted_sampler)
    if kind is SamplerKind.UNI_PC_BH2:
        raise ValueError(
            "Sampler 'uni-pc bh2' is not implemented in this scheduler bridge; "
            "use 'uni-pc' on this path."
        )
    if kind is SamplerKind.DPM2S_ANCESTRAL:
        raise ValueError(
            "Sampler 'dpm++ 2s ancestral' is not implemented in this scheduler bridge; "
            "use a native sampling runtime path."
        )

    from diffusers import (
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        HeunDiscreteScheduler,
        LMSDiscreteScheduler,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        KDPM2DiscreteScheduler,
        KDPM2AncestralDiscreteScheduler,
        UniPCMultistepScheduler,
    )

    allowed = {
        SamplerKind.EULER: EulerDiscreteScheduler,
        SamplerKind.EULER_A: EulerAncestralDiscreteScheduler,
        SamplerKind.HEUN: HeunDiscreteScheduler,
        SamplerKind.LMS: LMSDiscreteScheduler,
        SamplerKind.DDIM: DDIMScheduler,
        SamplerKind.DPM2M: DPMSolverMultistepScheduler,
        SamplerKind.DPM2M_SDE: DPMSolverMultistepScheduler,
        SamplerKind.DPM2M_SDE_HEUN: DPMSolverMultistepScheduler,
        SamplerKind.DPM2: KDPM2DiscreteScheduler,
        SamplerKind.DPM2_ANCESTRAL: KDPM2AncestralDiscreteScheduler,
        SamplerKind.UNI_PC: UniPCMultistepScheduler,
    }

    target_cls = allowed.get(kind)
    if target_cls is None:
        raise ValueError(f"Unsupported sampler '{wanted_sampler}'")

    # Rebuild scheduler from config to preserve sigmas/timesteps defaults
    conf = getattr(pipe, "scheduler", None)
    conf = getattr(conf, "config", None)
    scheduler_overrides = {}
    if kind in {SamplerKind.DPM2M_SDE, SamplerKind.DPM2M_SDE_HEUN}:
        if wanted_scheduler != "exponential":
            raise ValueError(
                f"Sampler '{wanted_sampler}' requires scheduler 'exponential'; got '{wanted_scheduler}'."
            )
        scheduler_overrides = {
            "algorithm_type": "sde-dpmsolver++",
            "solver_order": 2,
            "solver_type": "heun" if kind is SamplerKind.DPM2M_SDE_HEUN else "midpoint",
            "use_exponential_sigmas": True,
            "use_karras_sigmas": False,
        }
    if conf is not None:
        pipe.scheduler = target_cls.from_config(conf, **scheduler_overrides)
    else:
        pipe.scheduler = target_cls(**scheduler_overrides)

    # Heuristics for options
    if isinstance(pipe.scheduler, DPMSolverMultistepScheduler) and not scheduler_overrides:
        pipe.scheduler.config.setdefault("use_karras_sigmas", True)
    if isinstance(pipe.scheduler, (EulerDiscreteScheduler, EulerAncestralDiscreteScheduler)):
        # trailing spacing is more compatible with SD style
        pipe.scheduler.config.setdefault("timestep_spacing", "trailing")

    eff_sampler = wanted_sampler
    eff_scheduler = type(pipe.scheduler).__name__
    return ApplyOutcome(wanted_sampler, wanted_scheduler, eff_sampler, eff_scheduler, warnings)


__all__ = ["apply_sampler_scheduler", "ApplyOutcome", "SamplerKind"]
