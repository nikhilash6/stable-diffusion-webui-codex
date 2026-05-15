"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 Animate 14B video engine placeholder for the GGUF runtime.
Preserves animate-lane img2vid while keeping vid2vid explicitly parked with strict GGUF asset validation and lazy runtime materialization.

Symbols (top-level; keep in sync; no ghosts):
- `Wan22Animate14BEngine` (class): `BaseVideoEngine` implementation for WAN22 14B animate lane;
  runs img2vid while parking vid2vid and enforces GGUF-only model loading.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Optional

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.core.exceptions import EngineLoadError
from apps.backend.core.requests import Img2VidRequest, InferenceEvent, Vid2VidRequest
from apps.backend.engines.common.base_video import BaseVideoEngine
from apps.backend.engines.wan22.wan22_common import WanComponents, unload_wan_components
from apps.backend.runtime.memory import memory_management
from apps.backend.use_cases.img2vid import run_img2vid as _run_i2v


class Wan22Animate14BEngine(BaseVideoEngine):
    engine_id = "wan22_14b_animate"
    model_types: tuple[str, ...] = ("wan-2.2-animate-14b",)
    runtime_note: str = "WAN 2.2 Animate 14B via GGUF runtime"

    def __init__(self) -> None:
        super().__init__()
        self._comp: Optional[WanComponents] = None

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.IMG2VID,),
            model_types=self.model_types,
            precision=("fp16", "bf16", "fp32"),
            extras={"notes": self.runtime_note},
        )

    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        self._logger.debug("[%s] before load()", self.engine_id)
        default_mount_device = str(memory_management.manager.mount_device())
        requested_device = options.get("device")
        if requested_device not in (None, "", "auto"):
            raise EngineLoadError(
                "WAN22 engine-local 'device' override is not supported; "
                "configure the runtime main device via launcher/API canonical selection."
            )
        dty = str(options.get("dtype", "fp16"))
        comp = WanComponents()
        engine_label = self.engine_id

        p = os.path.expanduser(str(model_ref or "")).strip()
        if not p:
            raise EngineLoadError(f"WAN22 {engine_label}: empty model_ref")
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        if os.path.isdir(p):
            raise EngineLoadError(
                f"WAN22 {engine_label} is GGUF-only (expected a .gguf file resolved from model_sha); "
                f"got a directory: {model_ref}"
            )
        if not str(p).lower().endswith(".gguf"):
            raise EngineLoadError(
                f"WAN22 {engine_label} is GGUF-only (expected a .gguf file resolved from model_sha); "
                f"got: {model_ref}"
            )
        if not os.path.isfile(p):
            alt = os.path.abspath(os.path.join("models", "Wan", model_ref))
            if os.path.isfile(alt) and str(alt).lower().endswith(".gguf"):
                p = alt
            else:
                raise EngineLoadError(f"WAN22 {engine_label} GGUF model not found: {model_ref}")
        path_base = os.path.basename(p).lower()
        if "5b" in path_base and "14b" not in path_base:
            raise EngineLoadError(
                f"WAN22 {engine_label} requires 14B GGUF weights; got a 5B-labeled file: {model_ref}"
            )

        comp.model_dir = p
        comp.dtype = dty
        comp.device = default_mount_device

        # GGUF path: keep load() lazy and defer runtime materialization to use-cases/runtime.
        comp.pipeline = None
        ref_base = os.path.basename(str(model_ref or "")).lower()
        if "animate" in ref_base and "14b" in ref_base:
            variant_hint = "wan22_14b_animate"
        elif "14b" in ref_base:
            variant_hint = "wan22_14b"
        elif "5b" in ref_base:
            variant_hint = "wan22_5b"
        else:
            variant_hint = "unknown"
        weights_hint = "14b" if "14b" in ref_base else ("5b" if "5b" in ref_base else "unknown")
        self._logger.info(
            "WAN22 GGUF runtime selected (dispatch=%s variant=%s weights_hint=%s) for %s (device=%s dtype=%s)",
            self.engine_id,
            variant_hint,
            weights_hint,
            p,
            comp.device,
            dty,
        )

        self._comp = comp
        self._logger.debug("[%s] after load()", self.engine_id)
        self.mark_loaded()

    def unload(self) -> None:  # type: ignore[override]
        self._logger.debug("[%s] before unload()", self.engine_id)
        unload_wan_components(self._comp, engine_id=self.engine_id, logger=self._logger)
        self._comp = None
        self._logger.debug("[%s] after unload()", self.engine_id)
        self.mark_unloaded()

    def vid2vid(self, request: Vid2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        del request, kwargs
        raise NotImplementedError("wan vid2vid not yet implemented")

    def img2vid(self, request: Img2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.debug("[%s] before img2vid()", self.engine_id)
        self.ensure_loaded()
        assert self._comp is not None
        yield from _run_i2v(engine=self, comp=self._comp, request=request)
        self._logger.debug("[%s] after img2vid()", self.engine_id)
