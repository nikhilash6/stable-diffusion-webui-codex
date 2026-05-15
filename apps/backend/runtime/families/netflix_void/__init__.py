"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public Netflix VOID runtime-family surface for the native vid2vid scaffold.
Exports the immutable base/overlay bundle contracts, loader-side resolution helpers, default family constants, the
runtime scaffold with native text-encoder hydration, and the explicitly fail-loud vid2vid execution seam plus the
shared source-video/quadmask preprocessing owner.

Symbols (top-level; keep in sync; no ghosts):
- `NETFLIX_VOID_ENGINE_ID` (constant): Canonical internal engine/family id.
- `NETFLIX_VOID_PUBLIC_LABEL` (constant): Human-readable family label.
- `NETFLIX_VOID_PASS1_FILENAME` (constant): Literal Pass 1 overlay filename.
- `NETFLIX_VOID_PASS2_FILENAME` (constant): Literal Pass 2 overlay filename.
- `NETFLIX_VOID_BASE_REQUIRED_ENTRIES` (constant): Required base-bundle files/directories.
- `NetflixVoidBaseBundle` (dataclass): Resolved local base-bundle contract.
- `NetflixVoidOverlayPair` (dataclass): Resolved literal Pass 1 + Pass 2 overlay pair.
- `NetflixVoidBundleInputs` (dataclass): Typed loader/runtime handoff envelope.
- `NetflixVoidNativeComponents` (dataclass): Runtime scaffold carrying bundle inputs, native text encoder, and normalized options.
- `NetflixVoidPreparedInputs` (dataclass): Prepared source-video + quadmask tensors plus bounded metadata for native runtime use.
- `build_netflix_void_bundle_metadata` (function): Serialize stable bundle metadata for runtime/engine assembly.
- `build_netflix_void_native_components` (function): Build the runtime scaffold from typed bundle inputs.
- `prepare_netflix_void_vid2vid_inputs` (function): Load, validate, resize, quantize, and temporally pad one VOID vid2vid request.
- `prepare_netflix_void_bundle_inputs` (function): Build the typed Netflix VOID bundle envelope from local inventory.
- `resolve_netflix_void_base_bundle` (function): Resolve exactly one valid local base-bundle contract.
- `resolve_netflix_void_base_dirs` (function): List valid local base-bundle directories under `netflix_void_base`.
- `resolve_netflix_void_checkpoint_record` (function): Resolve and validate the selected Pass 1 overlay record.
- `run_netflix_void_vid2vid` (function): Native Netflix VOID vid2vid execution seam.
"""

from __future__ import annotations

from .config import (
    NETFLIX_VOID_BASE_REQUIRED_ENTRIES,
    NETFLIX_VOID_DEFAULT_HEIGHT,
    NETFLIX_VOID_DEFAULT_MAX_VIDEO_LENGTH,
    NETFLIX_VOID_DEFAULT_PASS1_CFG,
    NETFLIX_VOID_DEFAULT_PASS1_STEPS,
    NETFLIX_VOID_DEFAULT_PASS2_CFG,
    NETFLIX_VOID_DEFAULT_PASS2_STEPS,
    NETFLIX_VOID_DEFAULT_TEMPORAL_WINDOW,
    NETFLIX_VOID_DEFAULT_WIDTH,
    NETFLIX_VOID_ENGINE_ID,
    NETFLIX_VOID_PASS1_FILENAME,
    NETFLIX_VOID_PASS2_FILENAME,
    NETFLIX_VOID_PUBLIC_LABEL,
    netflix_void_base_dir_is_valid,
)
from .loader import (
    build_netflix_void_bundle_metadata,
    prepare_netflix_void_bundle_inputs,
    resolve_netflix_void_base_bundle,
    resolve_netflix_void_base_dirs,
    resolve_netflix_void_checkpoint_record,
)
from .model import NetflixVoidBaseBundle, NetflixVoidBundleInputs, NetflixVoidOverlayPair
from .preprocess import NetflixVoidPreparedInputs, prepare_netflix_void_vid2vid_inputs
from .runtime import NetflixVoidNativeComponents, build_netflix_void_native_components, run_netflix_void_vid2vid

__all__ = [
    "NETFLIX_VOID_BASE_REQUIRED_ENTRIES",
    "NETFLIX_VOID_DEFAULT_HEIGHT",
    "NETFLIX_VOID_DEFAULT_MAX_VIDEO_LENGTH",
    "NETFLIX_VOID_DEFAULT_PASS1_CFG",
    "NETFLIX_VOID_DEFAULT_PASS1_STEPS",
    "NETFLIX_VOID_DEFAULT_PASS2_CFG",
    "NETFLIX_VOID_DEFAULT_PASS2_STEPS",
    "NETFLIX_VOID_DEFAULT_TEMPORAL_WINDOW",
    "NETFLIX_VOID_DEFAULT_WIDTH",
    "NETFLIX_VOID_ENGINE_ID",
    "NETFLIX_VOID_PASS1_FILENAME",
    "NETFLIX_VOID_PASS2_FILENAME",
    "NETFLIX_VOID_PUBLIC_LABEL",
    "NetflixVoidBaseBundle",
    "NetflixVoidBundleInputs",
    "NetflixVoidNativeComponents",
    "NetflixVoidOverlayPair",
    "NetflixVoidPreparedInputs",
    "build_netflix_void_bundle_metadata",
    "build_netflix_void_native_components",
    "netflix_void_base_dir_is_valid",
    "prepare_netflix_void_vid2vid_inputs",
    "prepare_netflix_void_bundle_inputs",
    "resolve_netflix_void_base_bundle",
    "resolve_netflix_void_base_dirs",
    "resolve_netflix_void_checkpoint_record",
    "run_netflix_void_vid2vid",
]
