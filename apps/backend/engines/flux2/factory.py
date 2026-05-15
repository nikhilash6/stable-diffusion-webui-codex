"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: FLUX.2 runtime factory helpers for consistent engine assembly.
Centralizes FLUX.2 runtime assembly and `CodexObjects` construction so the engine facade stays thin and the sampler/VAE
contract is owned by the spec/runtime layer.

Symbols (top-level; keep in sync; no ghosts):
- `CodexFlux2Assembly` (dataclass): Assembled FLUX.2 runtime + `CodexObjects` bundle.
- `CodexFlux2Factory` (class): Builder that assembles a `Flux2EngineRuntime` from a model bundle and produces `CodexObjects`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from apps.backend.engines.common.base import CodexObjects, TextEncoderHandle
from apps.backend.engines.flux2.spec import FLUX2_SPEC, Flux2EngineRuntime, Flux2EngineSpec, assemble_flux2_runtime
from apps.backend.runtime.models.loader import DiffusionModelBundle


@dataclass(frozen=True, slots=True)
class CodexFlux2Assembly:
    runtime: Flux2EngineRuntime
    codex_objects: CodexObjects


class CodexFlux2Factory:
    def __init__(self, *, spec: Flux2EngineSpec = FLUX2_SPEC) -> None:
        self._spec = spec

    def assemble(self, bundle: DiffusionModelBundle, *, options: Mapping[str, Any]) -> CodexFlux2Assembly:
        runtime = assemble_flux2_runtime(
            spec=self._spec,
            estimated_config=bundle.estimated_config,
            codex_components=bundle.components,
            bundle=bundle,
            engine_options=options,
        )
        codex_objects = CodexObjects(
            denoiser=runtime.denoiser,
            vae=runtime.vae,
            text_encoders={
                "qwen3": TextEncoderHandle(
                    patcher=runtime.qwen,
                    runtime=runtime.text.qwen3_text,
                )
            },
            clipvision=None,
        )
        return CodexFlux2Assembly(runtime=runtime, codex_objects=codex_objects)


__all__ = ["CodexFlux2Assembly", "CodexFlux2Factory"]
