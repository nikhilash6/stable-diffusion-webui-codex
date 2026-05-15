"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampling context construction utilities for diffusion samplers.
Builds per-run sampling state (sampler kind, noise settings, scheduler config) into a `SamplingContext`, including console block-progress enablement (`CODEX_PROGRESS_BAR`, default on).
Sigma schedule construction lives in `sigma_schedules.py`.

Symbols (top-level; keep in sync; no ghosts):
- `SamplingContext` (dataclass): Bundles sampling configuration/state for one run (sampler kind, scheduler, noise settings, etc.).
- `build_sampling_context` (function): Builds a `SamplingContext` from inputs (engine/runtime settings + request payload).
- `SchedulerName` (enum): Canonical scheduler names for sigma schedule construction (strict, no silent fallback).
- `build_sigma_schedule` (function): Main scheduler entrypoint; selects the schedule builder and returns the sigma tensor.
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from apps.backend.core.rng import NoiseSettings, NoiseSourceKind
from apps.backend.engines.util.schedulers import SamplerKind
from apps.backend.infra.config.env_flags import env_flag
from apps.backend.runtime.live_preview import preview_interval_steps
from apps.backend.runtime.sampling.flow_shift_resolver import resolve_flow_shift_for_sampling
from apps.backend.runtime.sampling.sigma_schedules import SchedulerName, build_sigma_schedule


@dataclass
class SamplingContext:
    sampler_kind: SamplerKind
    scheduler_name: str
    sigmas: torch.Tensor
    steps: int
    noise_settings: NoiseSettings
    preview_interval: int = 0
    enable_progress: bool = True
    prediction_type: str | None = None
    sigma_min: float | None = None
    sigma_max: float | None = None
    sigma_data: float | None = None
    flow_shift: float | None = None
    flow_shift_config_path: str | None = None
    flow_shift_repo_dir: str | None = None

    @property
    def device(self) -> torch.device:
        return self.sigmas.device


def build_sampling_context(
    sd_model,
    *,
    sampler_name: str,
    scheduler_name: str,
    steps: int,
    noise_source: str | None,
    eta_noise_seed_delta: int,
    height: int | None = None,
    width: int | None = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    predictor: Optional[object] = None,
    is_sdxl: bool = False,
) -> SamplingContext:
    sampler_kind = SamplerKind.from_string(sampler_name)
    predictor_container = predictor or getattr(sd_model.codex_objects.denoiser, "model", None)
    if predictor_container is None or getattr(predictor_container, "predictor", None) is None:
        raise RuntimeError("sd_model does not expose a predictor for sigma bounds")

    pred = predictor_container.predictor

    def _as_float(value: object | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value.item()) if hasattr(value, "item") else float(value)  # type: ignore[arg-type]
        except Exception:
            return None

    sigma_min = _as_float(getattr(pred, "sigma_min", None))
    sigma_max = _as_float(getattr(pred, "sigma_max", None))
    if sigma_min is None or sigma_max is None:
        raise RuntimeError("predictor is missing sigma_min/sigma_max required for sampling")
    if sigma_max < sigma_min:
        sigma_min, sigma_max = sigma_max, sigma_min
    prediction_type = getattr(pred, "prediction_type", None)
    if isinstance(prediction_type, str):
        prediction_type = prediction_type.lower()
    sigma_data = _as_float(getattr(pred, "sigma_data", None))

    noise_settings = NoiseSettings(
        source=NoiseSourceKind.from_string(noise_source) if noise_source else NoiseSourceKind.GPU,
        eta_noise_seed_delta=int(eta_noise_seed_delta or 0),
    )

    dev = device or predictor_container.diffusion_model.load_device
    dt = dtype or getattr(predictor_container.diffusion_model, "dtype", torch.float32)
    # Sigma schedules are numerically sensitive (timestep mapping, step sizes). Using
    # bf16/fp16 here quantizes the ladder and can severely degrade output quality
    # (e.g., SDXL "checkerboard/golesma" artifacts). Keep schedules in fp32.
    sigma_dtype = dt if dt not in (torch.float16, torch.bfloat16) else torch.float32

    flow_shift_value: float | None = None
    flow_shift_config_path: str | None = None
    flow_shift_repo_dir: str | None = None
    if prediction_type == "const":
        resolution = resolve_flow_shift_for_sampling(sd_model, pred, height=height, width=width)
        flow_shift_value = float(resolution.effective_shift)
        flow_shift_config_path = resolution.spec.config_path
        flow_shift_repo_dir = resolution.repo_dir

    sigmas = build_sigma_schedule(
        scheduler_name,
        steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=dev,
        dtype=sigma_dtype,
        predictor=pred,
        flow_shift=flow_shift_value,
        is_sdxl=is_sdxl or bool(getattr(sd_model, "is_sdxl", False)),
    )

    if prediction_type == "const" and flow_shift_value is not None:
        try:
            sigma_max = float(sigmas[0].detach().cpu().item())
            sigma_min = float(sigmas[-1].detach().cpu().item())
        except Exception:
            pass

    context = SamplingContext(
        sampler_kind=sampler_kind,
        scheduler_name=scheduler_name,
        sigmas=sigmas,
        steps=steps,
        noise_settings=noise_settings,
        preview_interval=preview_interval_steps(default=0),
        enable_progress=env_flag("CODEX_PROGRESS_BAR", default=True),
        prediction_type=prediction_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
        flow_shift=flow_shift_value,
        flow_shift_config_path=flow_shift_config_path,
        flow_shift_repo_dir=flow_shift_repo_dir,
    )

    emit_backend_message(
        "sampling-context",
        logger=__name__,
        level=logging.DEBUG,
        sampler=context.sampler_kind.value,
        scheduler=context.scheduler_name,
        steps=context.steps,
        noise_source=context.noise_settings.source.value,
        eta_delta=context.noise_settings.eta_noise_seed_delta,
        prediction=context.prediction_type,
        sigma_min=float(context.sigma_min) if context.sigma_min is not None else float("nan"),
        sigma_max=float(context.sigma_max) if context.sigma_max is not None else float("nan"),
    )
    return context


__all__ = [
    "SamplingContext",
    "build_sampling_context",
    "build_sigma_schedule",
    "SchedulerName",
]
