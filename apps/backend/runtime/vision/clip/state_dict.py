"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical CLIP vision state-dict normalization facade for runtime loaders.
Delegates supported CLIP vision source-style resolution to `runtime/state_dict/keymap_clip_vision.py` and returns the lazy canonical
lookup view plus resolved layout metadata used by `ClipVisionEncoder`.

Symbols (top-level; keep in sync; no ghosts):
- `normalize_clip_vision_state_dict_with_layout` (function): Resolves a CLIP vision state dict into a canonical lookup view plus layout metadata.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from apps.backend.runtime.state_dict.key_mapping import KeyMappingError
from apps.backend.runtime.state_dict.keymap_clip_vision import (
    ClipVisionLayoutMetadata,
    clip_vision_layout_metadata_from_resolved,
    resolve_clip_vision_keyspace_with_layout,
)
from .errors import ClipVisionLoadError


def normalize_clip_vision_state_dict_with_layout(
    state_dict: MutableMapping[str, Any],
) -> tuple[MutableMapping[str, Any], ClipVisionLayoutMetadata]:
    try:
        resolved = resolve_clip_vision_keyspace_with_layout(state_dict)
    except KeyMappingError as exc:
        raise ClipVisionLoadError(f"Clip vision keyspace resolution failed: {exc}") from exc
    resolved_layout = clip_vision_layout_metadata_from_resolved(resolved)
    return resolved.view, resolved_layout
