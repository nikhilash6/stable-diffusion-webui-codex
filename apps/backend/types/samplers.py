"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampler and scheduler type definitions.
Defines the canonical `SamplerKind` enum used for strict sampler parsing across API/runtime surfaces and an `ApplyOutcome` result container.
Executable native variants are added explicitly when supported; removed public sampler identifiers stay absent from this enum surface.

Symbols (top-level; keep in sync; no ghosts):
- `ApplyOutcome` (dataclass): Result of applying sampler/scheduler selection to a pipeline/request.
- `SamplerKind` (enum): Canonical sampler identifiers (no alias normalization; fail-fast on unknown values).
- `__all__` (constant): Explicit export list for this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass
class ApplyOutcome:
    """Result of applying sampler/scheduler to a pipeline."""
    sampler_in: str
    scheduler_in: str
    sampler_effective: str
    scheduler_effective: str
    warnings: List[str]


class SamplerKind(str, Enum):
    """Canonical sampler identifiers."""
    EULER = "euler"
    EULER_A = "euler a"
    EULER_CFG_PP = "euler cfg++"
    EULER_A_CFG_PP = "euler a cfg++"
    HEUN = "heun"
    HEUNPP2 = "heunpp2"
    LMS = "lms"
    DDIM = "ddim"
    DPM2M = "dpm++ 2m"
    DPM2M_CFG_PP = "dpm++ 2m cfg++"
    DPMPP_SDE = "dpm++ sde"
    DPM2M_SDE = "dpm++ 2m sde"
    DPM2M_SDE_HEUN = "dpm++ 2m sde heun"
    DPM2M_SDE_GPU = "dpm++ 2m sde gpu"
    DPM2M_SDE_HEUN_GPU = "dpm++ 2m sde heun gpu"
    UNI_PC = "uni-pc"
    UNI_PC_BH2 = "uni-pc bh2"
    DPM2S_ANCESTRAL = "dpm++ 2s ancestral"
    DPM2S_ANCESTRAL_CFG_PP = "dpm++ 2s ancestral cfg++"
    DPM3M_SDE = "dpm++ 3m sde"
    DPM2 = "dpm 2"
    DPM2_ANCESTRAL = "dpm 2 ancestral"
    DPM_FAST = "dpm fast"
    DPM_ADAPTIVE = "dpm adaptive"
    DDPM = "ddpm"
    IPNDM = "ipndm"
    IPNDM_V = "ipndm v"
    DEIS = "deis"
    RES_MULTISTEP = "res multistep"
    RES_MULTISTEP_CFG_PP = "res multistep cfg++"
    RES_MULTISTEP_ANCESTRAL = "res multistep ancestral"
    RES_MULTISTEP_ANCESTRAL_CFG_PP = "res multistep ancestral cfg++"
    GRADIENT_ESTIMATION = "gradient estimation"
    GRADIENT_ESTIMATION_CFG_PP = "gradient estimation cfg++"
    ER_SDE = "er sde"
    SEEDS_2 = "seeds 2"
    SEEDS_3 = "seeds 3"
    SA_SOLVER = "sa-solver"
    SA_SOLVER_PECE = "sa-solver pece"
    RESTART = "restart"

    @staticmethod
    def from_string(name: str) -> "SamplerKind":
        """Parse sampler name to enum (strict, no alias resolution)."""
        if not isinstance(name, str):
            raise TypeError("sampler name must be a string")
        if not name:
            raise ValueError("sampler name must not be empty")
        try:
            return SamplerKind(name)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported sampler '{name}'. Valid: {[m.value for m in SamplerKind]}"
            ) from exc


__all__ = ["SamplerKind", "ApplyOutcome"]
