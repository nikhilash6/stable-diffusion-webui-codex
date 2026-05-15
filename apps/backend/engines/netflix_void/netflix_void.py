"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend Netflix VOID video engine placeholder.
Keeps the `netflix_void` engine id importable as an explicit stub while the native runtime remains parked/not implemented.

Symbols (top-level; keep in sync; no ghosts):
- `NetflixVoidEngine` (class): Backend video engine placeholder registered under engine id `netflix_void`.
"""

from __future__ import annotations

from typing import Any, Iterator

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.core.requests import InferenceEvent, Vid2VidRequest
from apps.backend.engines.common.base_video import BaseVideoEngine
from apps.backend.runtime.model_registry.specs import ModelFamily


class NetflixVoidEngine(BaseVideoEngine):
    engine_id = "netflix_void"
    expected_family = ModelFamily.NETFLIX_VOID
    model_types: tuple[str, ...] = ("netflix_void",)
    runtime_note: str = "Netflix VOID placeholder (parked)"

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(),
            model_types=self.model_types,
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
            extras={"notes": self.runtime_note},
        )

    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        del model_ref, options
        raise NotImplementedError("netflix_void not yet implemented")

    def unload(self) -> None:  # type: ignore[override]
        self.mark_unloaded()

    def vid2vid(self, request: Vid2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        del request, kwargs
        raise NotImplementedError("netflix_void not yet implemented")
