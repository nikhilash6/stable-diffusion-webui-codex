"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampler/scheduler catalog and executable-support flags for the runtime/API surface.
Defines the canonical sampler/scheduler inventory, derives executable sampler support from the current runtime-backed implementation
surface (including native `ddpm`, `heunpp2`, `ipndm`, `ipndm v`, `deis`, `dpm++ sde`, `dpm++ 2m sde`, `dpm++ 2m sde heun`,
`dpm++ 2m sde gpu`, `dpm++ 2m sde heun gpu`, `dpm++ 3m sde`, `res multistep*`, `gradient estimation*`, `sa-solver*`, and `seeds*`),
and provides default scheduler selection rules consumed by the sampling registry and API exposure layers.

Symbols (top-level; keep in sync; no ghosts):
- `SAMPLER_OPTIONS` (constant): Canonical sampler inventory table (canonical name + optional scheduler allowlists + executable support flag).
- `SUPPORTED_SAMPLERS` (constant): Set of supported sampler canonical names.
- `SCHEDULER_OPTIONS` (constant): UI-facing scheduler option table (canonical name only).
- `SUPPORTED_SCHEDULERS` (constant): Set of supported scheduler canonical names.
- `SAMPLER_DEFAULT_SCHEDULER` (constant): Default scheduler per sampler (used by UI and sampling plan defaults).
"""

from __future__ import annotations

from typing import Dict, List, Set


SUPPORTED_SAMPLERS: Set[str] = {
    "euler",
    "euler a",
    "euler cfg++",
    "euler a cfg++",
    "heun",
    "heunpp2",
    "lms",
    "ddim",
    "ddpm",
    "dpm++ 2m",
    "dpm++ 2m cfg++",
    "dpm++ 2m sde",
    "dpm++ 2m sde heun",
    "dpm++ 2m sde gpu",
    "dpm++ 2m sde heun gpu",
    "dpm++ sde",
    "dpm++ 2s ancestral",
    "dpm++ 2s ancestral cfg++",
    "dpm++ 3m sde",
    "dpm 2",
    "dpm 2 ancestral",
    "dpm fast",
    "dpm adaptive",
    "ipndm",
    "ipndm v",
    "deis",
    "res multistep",
    "res multistep cfg++",
    "res multistep ancestral",
    "res multistep ancestral cfg++",
    "gradient estimation",
    "gradient estimation cfg++",
    "sa-solver",
    "sa-solver pece",
    "seeds 2",
    "seeds 3",
    "uni-pc",
    "uni-pc bh2",
    "er sde",
    "restart",
}

SAMPLER_OPTIONS: List[Dict[str, object]] = [
    {"name": "euler"},
    {"name": "euler a"},
    {"name": "euler cfg++", "schedulers": ["euler_discrete"]},
    {"name": "euler a cfg++", "schedulers": ["euler_discrete"]},
    {"name": "heun"},
    {"name": "heunpp2", "schedulers": ["karras"]},
    {"name": "lms"},
    {"name": "ddim", "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "dpm++ 2m"},
    {"name": "dpm++ 2m cfg++", "schedulers": ["karras"]},
    {"name": "dpm++ 2m sde", "schedulers": ["exponential"]},
    {"name": "dpm++ 2m sde heun", "schedulers": ["exponential"]},
    {"name": "dpm++ 2m sde gpu", "schedulers": ["exponential"]},
    {"name": "dpm++ 2m sde heun gpu", "schedulers": ["exponential"]},
    {"name": "dpm++ sde", "schedulers": ["karras"]},
    {"name": "dpm++ 2s ancestral"},
    {"name": "dpm++ 2s ancestral cfg++", "schedulers": ["karras"]},
    {"name": "dpm++ 3m sde", "schedulers": ["exponential"]},
    {"name": "dpm 2"},
    {"name": "dpm 2 ancestral"},
    {"name": "dpm fast"},
    {"name": "dpm adaptive", "schedulers": ["karras"]},
    {"name": "uni-pc", "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "uni-pc bh2", "schedulers": ["ddim", "ddim_uniform", "karras", "exponential", "simple", "euler_discrete"]},
    {"name": "ddpm"},
    {"name": "ipndm"},
    {"name": "ipndm v"},
    {"name": "deis"},
    {"name": "res multistep"},
    {"name": "res multistep cfg++"},
    {"name": "res multistep ancestral"},
    {"name": "res multistep ancestral cfg++"},
    {"name": "gradient estimation"},
    {"name": "gradient estimation cfg++"},
    {"name": "er sde"},
    {"name": "seeds 2", "schedulers": ["karras"]},
    {"name": "seeds 3", "schedulers": ["karras"]},
    {"name": "sa-solver", "schedulers": ["karras"]},
    {"name": "sa-solver pece", "schedulers": ["karras"]},
    {"name": "restart", "schedulers": ["karras"]},
]

for entry in SAMPLER_OPTIONS:
    entry["supported"] = str(entry["name"]) in SUPPORTED_SAMPLERS

SCHEDULER_OPTIONS: List[Dict[str, object]] = [
    {"name": "uniform", "supported": True},
    {
        "name": "karras",
        "supported": True,
    },
    {
        "name": "exponential",
        "supported": True,
    },
    {
        "name": "polyexponential",
        "supported": True,
    },
    {
        "name": "simple",
        "supported": True,
    },
    {
        "name": "euler_discrete",
        "supported": True,
    },
    {"name": "ddim", "supported": True},
    {
        "name": "sgm_uniform",
        "supported": True,
    },
    {
        "name": "ddim_uniform",
        "supported": True,
    },
    {
        "name": "beta",
        "supported": True,
    },
    {
        "name": "normal",
        "supported": True,
    },
    {
        "name": "linear_quadratic",
        "supported": True,
    },
    {
        "name": "kl_optimal",
        "supported": True,
    },
    {"name": "turbo", "supported": True},
    {"name": "align_your_steps", "supported": True},
    {"name": "align_your_steps_gits", "supported": True},
    {"name": "align_your_steps_11", "supported": True},
    {"name": "align_your_steps_32", "supported": True},
]

SUPPORTED_SCHEDULERS: Set[str] = {entry["name"] for entry in SCHEDULER_OPTIONS if entry.get("supported", True)}

# Default scheduler per sampler.
SAMPLER_DEFAULT_SCHEDULER: Dict[str, str] = {
    # Euler-family samplers default to the predictor ladder ("simple" schedule).
    "euler": "simple",
    "euler a": "simple",
    "euler cfg++": "euler_discrete",
    "euler a cfg++": "euler_discrete",
    "heun": "karras",
    "heunpp2": "karras",
    "lms": "karras",
    "ddim": "ddim",
    "dpm++ 2m": "karras",
    "dpm++ 2m cfg++": "karras",
    "dpm++ sde": "karras",
    "dpm++ 2m sde": "exponential",
    "dpm++ 2m sde heun": "exponential",
    "dpm++ 2m sde gpu": "exponential",
    "dpm++ 2m sde heun gpu": "exponential",
    "dpm++ 2s ancestral": "karras",
    "dpm++ 2s ancestral cfg++": "karras",
    "dpm++ 3m sde": "exponential",
    "dpm 2": "karras",
    "dpm 2 ancestral": "karras",
    "dpm fast": "karras",
    "dpm adaptive": "karras",
    "uni-pc": "simple",
    "uni-pc bh2": "simple",
    "ddpm": "beta",
    "ipndm": "karras",
    "ipndm v": "karras",
    "deis": "karras",
    "res multistep": "karras",
    "res multistep cfg++": "karras",
    "res multistep ancestral": "karras",
    "res multistep ancestral cfg++": "karras",
    "gradient estimation": "karras",
    "gradient estimation cfg++": "karras",
    "er sde": "exponential",
    "seeds 2": "karras",
    "seeds 3": "karras",
    "sa-solver": "karras",
    "sa-solver pece": "karras",
    "restart": "karras",
}

__all__ = [
    "SAMPLER_OPTIONS",
    "SUPPORTED_SAMPLERS",
    "SCHEDULER_OPTIONS",
    "SUPPORTED_SCHEDULERS",
    "SAMPLER_DEFAULT_SCHEDULER",
]
