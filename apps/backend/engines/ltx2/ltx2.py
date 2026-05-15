"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend LTX2 video engine facade for the staged native runtime bring-up.
Registers the `ltx2` engine as a thin `BaseVideoEngine` adapter, carries the loader-produced typed LTX2 bundle contract
through load/unload, and delegates mode ownership to canonical `txt2vid` / `img2vid` use-cases backed by the native
LTX2 runtime assembly.

Symbols (top-level; keep in sync; no ghosts):
- `_LTX2_FACTORY` (constant): Factory used to assemble the minimal LTX2 engine runtime.
- `Ltx2Engine` (class): Backend video engine registered under engine id `ltx2`.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Any, Iterator

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.core.exceptions import EngineLoadError
from apps.backend.core.requests import Img2VidRequest, InferenceEvent, Txt2VidRequest
from apps.backend.engines.common.base_video import BaseVideoEngine
from apps.backend.engines.ltx2.factory import CodexLtx2Factory
from apps.backend.engines.ltx2.spec import LTX2_SPEC, Ltx2EngineRuntime
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.models.loader import DiffusionModelBundle, resolve_diffusion_bundle
from apps.backend.use_cases.img2vid import run_img2vid as _run_i2v
from apps.backend.use_cases.txt2vid import run_txt2vid as _run_t2v

logger = get_backend_logger("backend.engines.ltx2")
_LTX2_FACTORY = CodexLtx2Factory(spec=LTX2_SPEC)


class Ltx2Engine(BaseVideoEngine):
    engine_id = "ltx2"
    expected_family = ModelFamily.LTX2
    model_types: tuple[str, ...] = ("ltx2",)
    runtime_note: str = "LTX2 native backend-only runtime"

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Ltx2EngineRuntime | None = None
        self._current_bundle: DiffusionModelBundle | None = None

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2VID, TaskType.IMG2VID),
            model_types=self.model_types,
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
            extras={"notes": self.runtime_note},
        )

    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        self._logger.debug("[%s] before load()", self.engine_id)
        raw_options = dict(options)

        bundle_obj = raw_options.pop("_bundle", None)
        if bundle_obj is None:
            bundle = resolve_diffusion_bundle(
                model_ref,
                text_encoder_override=raw_options.get("text_encoder_override"),
                vae_path=raw_options.get("vae_path"),
                tenc_path=raw_options.get("tenc_path"),
                expected_family=self.expected_family,
            )
        else:
            if not isinstance(bundle_obj, DiffusionModelBundle):
                raise EngineLoadError("LTX2 load option `_bundle` must be a DiffusionModelBundle when provided.")
            bundle = bundle_obj

        if bundle.family is not self.expected_family:
            raise EngineLoadError(
                f"LTX2 engine expected family {self.expected_family.value!r}, got {bundle.family.value!r}."
            )

        assembly = _LTX2_FACTORY.assemble(bundle, options=raw_options)
        self._runtime = assembly.runtime
        self._current_bundle = bundle
        self.mark_loaded()
        self._logger.debug("[%s] after load()", self.engine_id)

    def unload(self) -> None:  # type: ignore[override]
        self._logger.debug("[%s] before unload()", self.engine_id)
        self._runtime = None
        self._current_bundle = None
        self.mark_unloaded()
        self._logger.debug("[%s] after unload()", self.engine_id)

    def _require_runtime(self) -> Ltx2EngineRuntime:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("ltx2 runtime is not initialised; call load() first.")
        return runtime

    def txt2vid(self, request: Txt2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self.ensure_loaded()
        runtime = self._require_runtime()
        yield from _run_t2v(engine=self, comp=runtime, request=request)

    def img2vid(self, request: Img2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self.ensure_loaded()
        runtime = self._require_runtime()
        yield from _run_i2v(engine=self, comp=runtime, request=request)
