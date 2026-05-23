"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z-Image L2P runtime factory helpers.
Centralizes L2P runtime assembly and `CodexObjects` construction so the engine facade stays a canonical txt2img hook provider.

Symbols (top-level; keep in sync; no ghosts):
- `CodexZImageL2PAssembly` (dataclass): Assembled L2P runtime + `CodexObjects` bundle.
- `CodexZImageL2PFactory` (class): Builder that assembles a `ZImageL2PEngineRuntime` from a model bundle and engine options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from apps.backend.engines.common.base import CodexObjects, TextEncoderHandle
from apps.backend.engines.zimage_l2p.spec import (
    ZImageL2PEngineRuntime,
    ZImageL2PEngineSpec,
    assemble_zimage_l2p_runtime,
)
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.models.loader import DiffusionModelBundle


@dataclass(frozen=True, slots=True)
class CodexZImageL2PAssembly:
    runtime: ZImageL2PEngineRuntime
    codex_objects: CodexObjects


class CodexZImageL2PFactory:
    """Assemble L2P runtimes and the corresponding engine component bundle."""

    def __init__(self, *, spec: ZImageL2PEngineSpec) -> None:
        self._spec = spec

    def assemble(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexZImageL2PAssembly:
        model_format = str(options.get("model_format") or "").strip().lower()
        core_device = memory_management.manager.get_device(DeviceRole.CORE)
        runtime = assemble_zimage_l2p_runtime(
            spec=self._spec,
            codex_components=bundle.components,
            estimated_config=bundle.estimated_config,
            model_path=str(bundle.model_ref),
            model_format=model_format,
            device=str(core_device),
            tenc_path=options.get("tenc_path") if isinstance(options.get("tenc_path"), str) else None,
        )
        codex_objects = CodexObjects(
            denoiser=runtime.denoiser,
            vae=None,
            text_encoders={
                "qwen3_4b": TextEncoderHandle(
                    patcher=runtime.qwen.patcher,
                    runtime=runtime.qwen,
                )
            },
            clipvision=None,
        )
        return CodexZImageL2PAssembly(runtime=runtime, codex_objects=codex_objects)


__all__ = ["CodexZImageL2PAssembly", "CodexZImageL2PFactory"]
