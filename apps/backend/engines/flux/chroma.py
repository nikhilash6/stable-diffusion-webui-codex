"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux.1 Chroma engine (Flux-family variant).
Assembles a `FluxEngineRuntime` from `CHROMA_SPEC` via the Flux family factory and exposes the `CodexDiffusionEngine` surface used by API/use-cases.
When smart offload is enabled, text encoder patcher unload is stage-scoped (only unload when this call loaded it).

Symbols (top-level; keep in sync; no ghosts):
- `Chroma` (class): Chroma diffusion engine (txt2img/img2img) wiring the Chroma runtime (Flux toolkit) to the Codex engine interface.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
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
from apps.backend.engines.flux.factory import CodexFluxFamilyFactory
from apps.backend.engines.flux.spec import CHROMA_SPEC, FluxEngineRuntime
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.models.loader import DiffusionModelBundle

logger = get_backend_logger("backend.engines.flux.chroma")

_CHROMA_FACTORY = CodexFluxFamilyFactory(spec=CHROMA_SPEC)


class Chroma(CodexDiffusionEngine):
    """Codex native Chroma engine built on Flux toolkit."""

    engine_id = "flux1_chroma"
    expected_family = ModelFamily.CHROMA

    @property
    def required_text_encoders(self) -> tuple[str, ...]:
        return ("t5",)

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[FluxEngineRuntime] = None

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=IMAGE_TASKS,
            model_types=("chroma",),
            devices=DEFAULT_IMAGE_DEVICES,
            precision=DEFAULT_IMAGE_PRECISION,
        )

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        assembly = _CHROMA_FACTORY.assemble(bundle, options=options)
        runtime = assembly.runtime
        self._runtime = runtime
        self.use_distilled_cfg_scale = runtime.use_distilled_cfg
        logger.debug("Chroma runtime prepared")

        return assembly.codex_objects

    def _on_unload(self) -> None:
        self._runtime = None

    def _require_runtime(self) -> FluxEngineRuntime:
        return require_runtime(self._runtime, label=self.engine_id)

    def set_clip_skip(self, clip_skip: int):
        logger.debug("Chroma ignores clip_skip (no CLIP branch)")

    @torch.no_grad()
    def get_learned_conditioning(self, prompt: List[str]):
        runtime = self._require_runtime()
        t5_patcher = self.codex_objects.text_encoders["t5"].patcher
        with stage_scoped_model_load(
            t5_patcher,
            smart_offload_enabled=self.smart_offload_enabled,
            manager=memory_management.manager,
        ):
            conditioning = runtime.text.t5_text(prompt)
            logger.debug("Chroma conditioning generated for %d prompts", len(prompt))
            return conditioning

    @torch.no_grad()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        token_count = len(runtime.text.t5_text.tokenize([prompt])[0])
        return token_count, max(255, token_count)
