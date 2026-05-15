"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public LTX2 runtime-family surface for the backend-only native video seam.
Exports the immutable parser-owned component contract, vendor metadata resolution helpers, loader/runtime handoff
dataclasses, native runtime assembly helpers, and the family-local `frames + audio_asset + metadata` result contract.

Symbols (top-level; keep in sync; no ghosts):
- `LTX2_COMPONENT_NAMES` (constant): Immutable parser-owned LTX2 component names.
- `LTX2_REQUIRED_TEXT_ENCODER_SLOT` (constant): Required external Gemma3 text-encoder alias.
- `LTX2_VENDOR_REPO_ID` (constant): Canonical LTX2 metadata repo id.
- `Ltx2BundleInputs` (dataclass): Typed loader/runtime handoff contract for native assembly.
- `Ltx2ComponentStates` (dataclass): Parser-owned LTX2 component state bundle.
- `Ltx2TextEncoderAsset` (dataclass): External Gemma3 text-encoder asset contract.
- `Ltx2VendorPaths` (dataclass): Vendored metadata path contract.
- `Ltx2RunResult` (dataclass): Family-local video+audio generation result contract.
- `Ltx2NativeComponents` (dataclass): Loaded native LTX2 runtime components reused by txt2vid/img2vid.
- `build_ltx2_bundle_metadata` (function): Serialize stable loader metadata for runtime/engine assembly.
- `build_ltx2_native_components` (function): Assemble the loaded native LTX2 runtime components.
- `build_ltx2_run_result` (function): Normalize an LTX2 `frames + audio_asset + metadata` result.
- `prepare_ltx2_bundle_inputs` (function): Build the typed LTX2 bundle-planning contract from loader-side parser output.
- `require_ltx2_bundle_inputs` (function): Rehydrate the typed LTX2 contract from a generic diffusion bundle.
- `resolve_ltx2_vendor_paths` (function): Resolve fail-loud local vendored metadata paths for LTX2.
- `run_ltx2_txt2vid` (function): Execute the native LTX2 txt2vid runtime path.
- `run_ltx2_img2vid` (function): Execute the native LTX2 img2vid runtime path.
"""

from __future__ import annotations

from .config import LTX2_COMPONENT_NAMES, LTX2_REQUIRED_TEXT_ENCODER_SLOT, LTX2_VENDOR_REPO_ID, resolve_ltx2_vendor_paths
from .loader import build_ltx2_bundle_metadata, prepare_ltx2_bundle_inputs
from .model import Ltx2BundleInputs, Ltx2ComponentStates, Ltx2TextEncoderAsset, Ltx2VendorPaths
from .runtime import (
    Ltx2NativeComponents,
    Ltx2RunResult,
    build_ltx2_native_components,
    build_ltx2_run_result,
    require_ltx2_bundle_inputs,
    run_ltx2_img2vid,
    run_ltx2_txt2vid,
)

__all__ = [
    "LTX2_COMPONENT_NAMES",
    "LTX2_REQUIRED_TEXT_ENCODER_SLOT",
    "LTX2_VENDOR_REPO_ID",
    "Ltx2BundleInputs",
    "Ltx2ComponentStates",
    "Ltx2NativeComponents",
    "Ltx2RunResult",
    "Ltx2TextEncoderAsset",
    "Ltx2VendorPaths",
    "build_ltx2_bundle_metadata",
    "build_ltx2_native_components",
    "build_ltx2_run_result",
    "prepare_ltx2_bundle_inputs",
    "require_ltx2_bundle_inputs",
    "resolve_ltx2_vendor_paths",
    "run_ltx2_img2vid",
    "run_ltx2_txt2vid",
]
