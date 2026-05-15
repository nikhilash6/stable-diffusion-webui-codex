"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed bundle-planning contracts for the native LTX2 runtime seam.
Defines immutable dataclasses for the parser-owned LTX2 component map, vendored metadata paths, and the external Gemma3
text-encoder asset that future runtime/engine assembly must consume without renaming bundle keys, including the exact
wrapped-vocoder config extracted from real LTX 2.3 audio bundles.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2VendorPaths` (dataclass): Normalized local vendor metadata paths (`model_index`, `tokenizer`, `connectors config`).
- `Ltx2TextEncoderAsset` (dataclass): Resolved external Gemma3 text-encoder asset path + tokenizer metadata.
- `Ltx2ComponentStates` (dataclass): Parser-owned LTX2 component state-dict bundle (`transformer`, `connectors`, `vae`, `audio_vae`, `vocoder`).
- `Ltx2BundleInputs` (dataclass): Typed planning result consumed by the future native LTX2 runtime/engine assembly seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from apps.backend.runtime.model_parser.specs import CodexEstimatedConfig
from apps.backend.runtime.model_registry.specs import ModelSignature


@dataclass(frozen=True, slots=True)
class Ltx2VendorPaths:
    repo_dir: str
    model_index_path: str
    tokenizer_dir: str
    connectors_config_path: str


@dataclass(frozen=True, slots=True)
class Ltx2TextEncoderAsset:
    alias: str
    path: str
    kind: str
    tokenizer_dir: str


@dataclass(frozen=True, slots=True)
class Ltx2ComponentStates:
    transformer: Mapping[str, Any]
    connectors: Mapping[str, Any]
    vae: Mapping[str, Any]
    audio_vae: Mapping[str, Any]
    vocoder: Mapping[str, Any]

    @classmethod
    def from_component_map(cls, components: Mapping[str, Mapping[str, Any]]) -> "Ltx2ComponentStates":
        expected = ("transformer", "connectors", "vae", "audio_vae", "vocoder")
        missing = [name for name in expected if name not in components]
        unexpected = sorted(set(components.keys()) - set(expected))
        if missing or unexpected:
            problems: list[str] = []
            if missing:
                problems.append(f"missing={missing!r}")
            if unexpected:
                problems.append(f"unexpected={unexpected!r}")
            raise RuntimeError(
                "LTX2 bundle planning requires exactly the parser-owned component names "
                f"{expected!r}; {' '.join(problems)}."
            )

        return cls(
            transformer=components["transformer"],
            connectors=components["connectors"],
            vae=components["vae"],
            audio_vae=components["audio_vae"],
            vocoder=components["vocoder"],
        )


@dataclass(frozen=True, slots=True)
class Ltx2BundleInputs:
    model_ref: str
    signature: ModelSignature
    estimated_config: CodexEstimatedConfig
    components: Ltx2ComponentStates
    text_encoder: Ltx2TextEncoderAsset
    vendor_paths: Ltx2VendorPaths
    vocoder_config: Mapping[str, Any] | None = None
