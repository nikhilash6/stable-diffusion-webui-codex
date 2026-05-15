"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Generic denoiser patcher wrapper for non-UNet architectures.
Wraps a denoiser model in `SamplerModel` and exposes it through the shared `ModelPatcher` base without ControlNet-specific features.

Symbols (top-level; keep in sync; no ghosts):
- `DenoiserPatcher` (class): Thin wrapper around `ModelPatcher` with a `from_model(...)` constructor for denoisers.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging

from apps.backend.runtime.sampling_adapters.sampler_model import SamplerModel
from .base import ModelPatcher

logger = get_backend_logger("backend.patchers.denoiser")


class DenoiserPatcher(ModelPatcher):
    """Codex-native denoiser patcher for non-UNet denoiser architectures."""

    @classmethod
    def from_model(cls, model, diffusers_scheduler, config, predictor=None):
        wrapped = SamplerModel(model=model, diffusers_scheduler=diffusers_scheduler, predictor=predictor, config=config)
        logger.debug("Wrapping denoiser model %s with SamplerModel", type(model).__name__)
        return cls(
            wrapped,
            load_device=wrapped.diffusion_model.load_device,
            offload_device=wrapped.diffusion_model.offload_device,
            current_device=wrapped.diffusion_model.initial_device,
        )
