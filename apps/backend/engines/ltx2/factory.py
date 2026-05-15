"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LTX2 engine runtime factory helpers.
Centralizes the typed runtime assembly so the `Ltx2Engine` facade stays thin and future native generation assembly can
grow under one seam without rewriting engine lifecycle code.

Symbols (top-level; keep in sync; no ghosts):
- `CodexLtx2Assembly` (dataclass): Assembled LTX2 engine runtime bundle.
- `CodexLtx2Factory` (class): Builder that assembles `Ltx2EngineRuntime` from a `DiffusionModelBundle`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from apps.backend.engines.ltx2.spec import LTX2_SPEC, Ltx2EngineRuntime, Ltx2EngineSpec, assemble_ltx2_runtime
from apps.backend.runtime.models.loader import DiffusionModelBundle


@dataclass(frozen=True, slots=True)
class CodexLtx2Assembly:
    runtime: Ltx2EngineRuntime


class CodexLtx2Factory:
    def __init__(self, *, spec: Ltx2EngineSpec = LTX2_SPEC) -> None:
        self._spec = spec

    def assemble(self, bundle: DiffusionModelBundle, *, options: Mapping[str, Any]) -> CodexLtx2Assembly:
        runtime = assemble_ltx2_runtime(
            spec=self._spec,
            bundle=bundle,
            engine_options=options,
        )
        return CodexLtx2Assembly(runtime=runtime)

