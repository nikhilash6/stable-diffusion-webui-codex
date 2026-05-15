"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: CLIP state-dict keyspace facade for runtime loaders.
Delegates CLIP key-style detection/keyspace resolution to canonical keymap ownership (`runtime/state_dict/keymap_sdxl_clip.py`) and returns
lazy normalized lookup views plus resolved layout metadata used by layout-aware module selection.

Symbols (top-level; keep in sync; no ghosts):
- `normalize_codex_clip_state_dict_with_layout` (function): Resolves a CLIP state dict into a normalized lookup view plus layout metadata.
- `normalize_codex_clip_state_dict` (function): Convenience wrapper returning only the normalized CLIP lookup view.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Literal

from apps.backend.runtime.state_dict.keymap_sdxl_clip import (
    ClipLayoutMetadata,
    clip_layout_metadata_from_resolved,
    resolve_clip_keyspace_with_layout,
)

_QKVImpl = Literal["auto", "split", "fused"]
_ProjectionOrientationTarget = Literal["auto", "linear", "matmul"]


def normalize_codex_clip_state_dict_with_layout(
    state_dict: MutableMapping[str, Any],
    *,
    num_layers: int,
    keep_projection: bool,
    qkv_impl: _QKVImpl = "auto",
    projection_orientation: _ProjectionOrientationTarget = "auto",
    layout_metadata: ClipLayoutMetadata | None = None,
    require_projection: bool = False,
) -> tuple[MutableMapping[str, Any], ClipLayoutMetadata]:
    resolved = resolve_clip_keyspace_with_layout(
        state_dict,
        num_layers=num_layers,
        keep_projection=keep_projection,
        qkv_impl=qkv_impl,
        projection_orientation=projection_orientation,
        layout_metadata=layout_metadata,
        require_projection=require_projection,
    )
    resolved_layout = clip_layout_metadata_from_resolved(resolved)
    return resolved.view, resolved_layout


def normalize_codex_clip_state_dict(
    state_dict: MutableMapping[str, Any],
    *,
    num_layers: int,
    keep_projection: bool,
    qkv_impl: _QKVImpl = "auto",
    projection_orientation: _ProjectionOrientationTarget = "auto",
    layout_metadata: ClipLayoutMetadata | None = None,
    require_projection: bool = False,
) -> MutableMapping[str, Any]:
    normalized, _layout = normalize_codex_clip_state_dict_with_layout(
        state_dict,
        num_layers=num_layers,
        keep_projection=keep_projection,
        qkv_impl=qkv_impl,
        projection_orientation=projection_orientation,
        layout_metadata=layout_metadata,
        require_projection=require_projection,
    )
    return normalized
