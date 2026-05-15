"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared SD1/SD2 engine implementation (classic CLIP text encoder + UNet denoiser + VAE).
Centralizes SD-classic runtime wiring, clip-skip, and conditioning so SD15 and SD20 engines only specify spec + capabilities.

Symbols (top-level; keep in sync; no ghosts):
- `CodexSDClassicEngineBase` (class): Shared implementation for SD1/SD2 engines (build/runtime lifecycle + clip-skip + conditioning).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Any, List, Mapping, Optional

import torch

from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.common.model_scopes import stage_scoped_model_load
from apps.backend.engines.common.runtime_lifecycle import require_runtime
from apps.backend.engines.sd._clip_skip import apply_sd_clip_skip
from apps.backend.engines.sd.factory import CodexSDFamilyFactory
from apps.backend.engines.sd.spec import SDEngineRuntime
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models.loader import DiffusionModelBundle


class CodexSDClassicEngineBase(CodexDiffusionEngine):
    """Shared SD1/SD2 engine implementation (classic CLIP + UNet + VAE)."""

    engine_id: str
    _factory: CodexSDFamilyFactory
    _model_family: str

    def __init__(self) -> None:
        super().__init__(logger=get_backend_logger(f"backend.engines.sd.{self.engine_id}"))
        self._runtime: Optional[SDEngineRuntime] = None
        self._primary_branch: Optional[str] = None

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        assembly = self._factory.assemble(bundle, options=dict(options))
        runtime = assembly.runtime
        self._runtime = runtime
        self._primary_branch = runtime.classic_order[0] if runtime.classic_order else None
        self.register_model_family(self._model_family)

        self._logger.debug("%s runtime prepared with branches=%s", self.engine_id, runtime.classic_order)
        return assembly.codex_objects

    def _on_unload(self) -> None:
        self._runtime = None
        self._primary_branch = None

    def _require_runtime(self) -> SDEngineRuntime:
        return require_runtime(self._runtime, label=self.engine_id)

    def set_clip_skip(self, clip_skip: int) -> None:
        runtime = self._require_runtime()
        apply_sd_clip_skip(
            engine=self,
            runtime=runtime,
            clip_skip=clip_skip,
            logger=self._logger,
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
            conditioning = runtime.primary_classic()(prompt)
            self._logger.debug("Generated conditioning for %d prompts.", len(prompt))
            return conditioning

    @torch.no_grad()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        engine = runtime.primary_classic()
        _, token_count = engine.process_texts([prompt])
        target = engine.get_target_prompt_token_count(token_count)
        return token_count, target
