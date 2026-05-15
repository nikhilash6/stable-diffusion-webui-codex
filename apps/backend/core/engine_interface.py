"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Abstract interfaces and capability descriptors for inference engines.
Defines the engine contract (`BaseInferenceEngine`) and typed capability metadata (`EngineCapabilities`) used by the orchestrator/registry.

Symbols (top-level; keep in sync; no ghosts):
- `TaskType` (enum): Supported task types (txt2img/img2img/inpaint/upscale/video).
- `EngineCapabilities` (dataclass): Declares tasks/model types/devices/precision supported by an engine.
- `BaseInferenceEngine` (class): Abstract base class for engine implementations (lifecycle + per-task iterator APIs).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping, Sequence

from .exceptions import UnsupportedTaskError


logger = get_backend_logger(__name__)


class TaskType(str, Enum):
    """Supported inference task types."""

    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"
    UPSCALE = "upscale"
    TXT2VID = "txt2vid"
    IMG2VID = "img2vid"
    VID2VID = "vid2vid"


@dataclass(frozen=True)
class EngineCapabilities:
    """Describes the capabilities of a concrete engine implementation."""

    engine_id: str
    tasks: Sequence[TaskType]
    model_types: Sequence[str]
    devices: Sequence[str] = ("cpu", "cuda")
    precision: Sequence[str] = ("fp32", "fp16")
    extras: Mapping[str, Any] = field(default_factory=dict)

    def supports(self, task: TaskType) -> bool:
        return task in self.tasks


class BaseInferenceEngine(ABC):
    """Base class for inference engines.

    Concrete engines are responsible for loading their weights, executing the
    requested task, and streaming progress/result events. All engines must be
    stateless with regards to requests; any persistent state (e.g. loaded
    weights) should be managed on the instance.
    """

    engine_id: str

    def __init__(self) -> None:
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def load(self, model_ref: str, **options: Any) -> None:
        """Prepare weights/resources for the given model reference."""

    @abstractmethod
    def unload(self) -> None:
        """Release heavy resources. Called when the engine is evicted."""

    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Return the static capabilities of this engine."""

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def status(self) -> Mapping[str, Any]:
        """Return a lightweight status report for debugging/logging."""

        return {
            "engine_id": getattr(self, "engine_id", "unknown"),
            "loaded": self._is_loaded,
        }

    def memory_usage(self) -> Mapping[str, Any]:
        """Return memory usage information (override if available)."""

        return {}

    # ------------------------------------------------------------------
    # Optional task implementations. Engines must override the ones they
    # support. Default implementation raises UnsupportedTaskError.
    # ------------------------------------------------------------------
    def txt2img(self, request: "Txt2ImgRequest", **kwargs: Any) -> Iterator["InferenceEvent"]:
        raise UnsupportedTaskError(f"Engine {self.engine_id} cannot run txt2img")

    def img2img(self, request: "Img2ImgRequest", **kwargs: Any) -> Iterator["InferenceEvent"]:
        raise UnsupportedTaskError(f"Engine {self.engine_id} cannot run img2img")

    def inpaint(self, request: "Img2ImgRequest", **kwargs: Any) -> Iterator["InferenceEvent"]:
        raise UnsupportedTaskError(f"Engine {self.engine_id} cannot run inpaint")

    def upscale(self, request: "Txt2ImgRequest", **kwargs: Any) -> Iterator["InferenceEvent"]:
        raise UnsupportedTaskError(f"Engine {self.engine_id} cannot run upscale")

    def txt2vid(self, request: "Txt2VidRequest", **kwargs: Any) -> Iterator["InferenceEvent"]:
        raise UnsupportedTaskError(f"Engine {self.engine_id} cannot run txt2vid")

    def img2vid(self, request: "Img2VidRequest", **kwargs: Any) -> Iterator["InferenceEvent"]:
        raise UnsupportedTaskError(f"Engine {self.engine_id} cannot run img2vid")

    def vid2vid(self, request: "Vid2VidRequest", **kwargs: Any) -> Iterator["InferenceEvent"]:
        raise UnsupportedTaskError(f"Engine {self.engine_id} cannot run vid2vid")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def ensure_loaded(self) -> None:
        if not self._is_loaded:
            raise RuntimeError("Engine must be loaded before use")

    def mark_loaded(self) -> None:
        self._is_loaded = True

    def mark_unloaded(self) -> None:
        self._is_loaded = False


# Deferred type checking imports (avoid circular dependencies at runtime)
if False:  # pragma: no cover
    from .requests import (
        Img2ImgRequest,
        Img2VidRequest,
        InferenceEvent,
        Txt2ImgRequest,
        Txt2VidRequest,
        Vid2VidRequest,
    )
