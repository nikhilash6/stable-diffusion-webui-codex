"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Netflix VOID engine runtime factory helpers.
Centralizes the typed runtime assembly so the `NetflixVoidEngine` facade stays thin while the native Pass 1 -> Pass 2
runtime grows under one seam.

Symbols (top-level; keep in sync; no ghosts):
- `CodexNetflixVoidAssembly` (dataclass): Assembled Netflix VOID engine runtime bundle.
- `CodexNetflixVoidFactory` (class): Builder that assembles `NetflixVoidEngineRuntime` from a selected model reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from apps.backend.engines.netflix_void.spec import (
    NETFLIX_VOID_SPEC,
    NetflixVoidEngineRuntime,
    NetflixVoidEngineSpec,
    assemble_netflix_void_runtime,
)


@dataclass(frozen=True, slots=True)
class CodexNetflixVoidAssembly:
    runtime: NetflixVoidEngineRuntime


class CodexNetflixVoidFactory:
    def __init__(self, *, spec: NetflixVoidEngineSpec = NETFLIX_VOID_SPEC) -> None:
        self._spec = spec

    def assemble(self, model_ref: str, *, options: Mapping[str, Any]) -> CodexNetflixVoidAssembly:
        runtime = assemble_netflix_void_runtime(
            spec=self._spec,
            model_ref=model_ref,
            engine_options=options,
        )
        return CodexNetflixVoidAssembly(runtime=runtime)
