"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Codex-native Stable Diffusion 3.5 engine entrypoint (temporarily disabled).
The module preserves SD3.5 conditioning helpers, but runtime assembly is blocked behind a fail-loud `NotImplementedError`
until the SD3.5 conditioning/keymap port is finalized.

Symbols (top-level; keep in sync; no ghosts):
- `_opts` (function): Loads SD3/SD35 environment flags (currently `CODEX_SD3_ENABLE_T5`) into a simple namespace.
- `StableDiffusion3` (class): SD 3.5 diffusion engine wiring runtime components to the Codex engine interface.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from types import SimpleNamespace
from typing import Any, List, Mapping, Optional

import torch

from apps.backend.core.engine_interface import EngineCapabilities
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.common.capabilities_presets import (
    DEFAULT_IMAGE_DEVICES,
    DEFAULT_IMAGE_PRECISION,
    IMAGE_TASKS,
)
from apps.backend.engines.common.model_scopes import stage_scoped_model_load
from apps.backend.engines.common.runtime_lifecycle import require_runtime
from apps.backend.engines.sd._clip_skip import apply_sd_clip_skip
from apps.backend.engines.sd.spec import SDEngineRuntime
from apps.backend.infra.config.env_flags import env_flag
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.models.loader import DiffusionModelBundle

logger = get_backend_logger("backend.engines.sd.sd35")


def _opts():
    enable_t5 = env_flag("CODEX_SD3_ENABLE_T5", default=True)
    return SimpleNamespace(sd3_enable_t5=enable_t5)


class StableDiffusion3(CodexDiffusionEngine):
    """Codex-native Stable Diffusion 3 engine."""

    engine_id = "sd35"
    expected_family = ModelFamily.SD35

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[SDEngineRuntime] = None

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=IMAGE_TASKS,
            model_types=("sd35", "sd3"),
            devices=DEFAULT_IMAGE_DEVICES,
            precision=DEFAULT_IMAGE_PRECISION,
        )

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        del bundle, options
        raise NotImplementedError(
            "Engine 'sd35' is temporarily disabled until SD3.5 conditioning/keymap port is finalized."
        )

    def _on_unload(self) -> None:
        self._runtime = None

    def _require_runtime(self) -> SDEngineRuntime:
        return require_runtime(self._runtime, label=self.engine_id)

    def set_clip_skip(self, clip_skip: int) -> None:
        runtime = self._require_runtime()
        apply_sd_clip_skip(
            engine=self,
            runtime=runtime,
            clip_skip=clip_skip,
            logger=logger,
            label=self.engine_id,
        )

    @torch.no_grad()
    def get_learned_conditioning(self, prompt: List[str]):
        runtime = self._require_runtime()
        clip_patcher = self.codex_objects.text_encoders["clip"].patcher
        with stage_scoped_model_load(
            clip_patcher,
            smart_offload_enabled=self.smart_offload_enabled,
            manager=memory_management.manager,
        ):
            cond_l, pooled_l = runtime.classic_engine("clip_l")(prompt)
            cond_g, pooled_g = runtime.classic_engine("clip_g")(prompt)

            opts = _opts()
            if opts.sd3_enable_t5:
                cond_t5 = runtime.t5_engine("t5xxl")(prompt)
            else:
                cond_t5 = torch.zeros((len(prompt), 256, 4096), device=cond_l.device, dtype=cond_l.dtype)

            is_negative_prompt = getattr(prompt, "is_negative_prompt", False)
            force_zero_negative_prompt = bool(is_negative_prompt) and all(str(x or "").strip() == "" for x in prompt)

            if force_zero_negative_prompt:
                pooled_l = torch.zeros_like(pooled_l)
                pooled_g = torch.zeros_like(pooled_g)
                cond_l = torch.zeros_like(cond_l)
                cond_g = torch.zeros_like(cond_g)
                cond_t5 = torch.zeros_like(cond_t5)

            cond_lg = torch.cat([cond_l, cond_g], dim=-1)
            if cond_lg.shape[-1] < 4096:
                pad = 4096 - cond_lg.shape[-1]
                cond_lg = torch.nn.functional.pad(cond_lg, (0, pad))

            cond = {
                "crossattn": torch.cat([cond_lg, cond_t5], dim=-2),
                "vector": torch.cat([pooled_l, pooled_g], dim=-1),
            }

            return cond

    @torch.no_grad()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        engine = runtime.t5_engine("t5xxl")
        token_count = len(engine.tokenize([prompt])[0])
        return token_count, max(255, token_count)
