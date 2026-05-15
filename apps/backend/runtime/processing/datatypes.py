"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared dataclasses for Codex generation workflows (conditioning, plans, results, metadata).
These structures are consumed by engines and workflow builders to describe runs without relying on legacy state containers.

Symbols (top-level; keep in sync; no ghosts):
- `ExtraNetworkDescriptor` (dataclass): Parsed extra network descriptor (e.g. LoRA path/weight + metadata).
- `PromptContext` (dataclass): Normalized prompt state after preprocessing (prompts/negatives/loras/request-owned clip_skip/metadata).
- `ConditioningPayload` (dataclass): Conditioning tensors assembled for a generation pass (cond/uncond + extras).
- `ErSdeOptions` (dataclass): Native ER-SDE runtime options (`solver_type`, `max_stage`, `eta`, `s_noise`).
- `SamplingPlan` (dataclass): Complete specification of a sampling run (sampler/scheduler/steps/seeds/noise settings + optional ER-SDE options).
- `InitImageBundle` (dataclass): Inputs derived from an initial image (pixels/latents + optional mask).
- `AppliedExtra` (dataclass): Record of applied extra network or post-processing effect.
- `GenerationResult` (dataclass): Outputs and diagnostics from a generation pass (samples/decoded + metadata/applied_extras/decode owner).
- `VideoPlan` (dataclass): Execution plan for video workflows (frames/fps/steps/scheduler + extras).
- `VideoResult` (dataclass): Result bundle for video workflows (frames + metadata).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from apps.backend.core.rng import NoiseSettings


_InitImageMode = Literal["pixel", "latent"]


@dataclass(slots=True)
class ExtraNetworkDescriptor:
    """Structured description of an extra network (e.g. LoRA) parsed from prompts."""

    path: str
    weight: float
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PromptContext:
    """Normalized prompt state after preprocessing."""

    prompts: list[str]
    negative_prompts: list[str]
    loras: Sequence[Any]
    clip_skip: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ConditioningPayload:
    """Conditioning tensors assembled for a generation pass."""

    conditioning: Any
    unconditional: Any
    extras: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ErSdeOptions:
    """ER-SDE runtime options applied when sampler is `er sde`."""

    solver_type: str
    max_stage: int
    eta: float
    s_noise: float


@dataclass(slots=True)
class SamplingPlan:
    """Complete specification of a sampling run."""

    sampler_name: str | None
    scheduler_name: str | None
    steps: int
    guidance_scale: float
    seeds: list[int]
    subseeds: list[int]
    subseed_strength: float
    noise_settings: NoiseSettings
    er_sde: ErSdeOptions | None = None


@dataclass(slots=True)
class InitImageBundle:
    """Inputs derived from an initial image (img2img/img2vid/hires)."""

    tensor: Any
    latents: Any | None
    mask: Any | None = None
    mode: _InitImageMode = "pixel"


@dataclass(slots=True)
class AppliedExtra:
    """Record of applied extra network or post-processing effect."""

    name: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    """Outputs and diagnostics from a generation pass."""

    samples: Any
    decoded: Any | None
    applied_extras: list[AppliedExtra] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    decode_engine: Any | None = None


@dataclass(slots=True)
class VideoPlan:
    """Execution plan for video generation tasks."""

    sampler_name: str | None
    scheduler_name: str | None
    steps: int
    frames: int
    fps: int
    width: int
    height: int
    guidance_scale: float | None
    extras: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class VideoResult:
    """Result bundle for video workflows."""

    frames: Sequence[Any]
    metadata: dict[str, object]
    video_meta: dict[str, object] | None = None
