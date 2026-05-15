"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: CLIP state-dict conversion helpers for model parsing (SD1.5/SD2.x/SDXL).
Normalizes CLIP checkpoints through canonical keymap ownership (`runtime/state_dict/keymap_sdxl_clip.py`) and preserves native layout in
`auto` mode (no structural conversion required). Fixes position-id dtype when requested and canonicalizes projection/logit keys.

Symbols (top-level; keep in sync; no ghosts):
- `_ensure_position_ids_long` (function): Ensures `position_ids` tensors are `torch.long` (rounding when needed).
- `_with_prefix` (function): Adds a prefix to every key in a state dict mapping.
- `_strip_prefix` (function): Removes a key prefix when present (leaves non-matching keys unchanged).
- `convert_clip` (function): Generic CLIP converter (alias-aware; keymap-owned CLIP keyspace resolution + layout-aware projection policy).
- `convert_sd15_clip` (function): SD1.5 CLIP-L converter (drops heads reconstructed at runtime).
- `convert_sd20_clip` (function): SD2.x CLIP-H converter (keeps native projection layout in auto mode).
- `convert_sdxl_clip_l` (function): SDXL CLIP-L converter (drops runtime-reconstructed projection weights).
- `convert_sdxl_clip_g` (function): SDXL CLIP-G converter (keeps native projection layout; keeps logit_scale).
"""

from __future__ import annotations

from typing import Any, Dict, Literal

import torch

from apps.backend.runtime.models.clip_key_normalization import (
    normalize_codex_clip_state_dict_with_layout,
)
from apps.backend.runtime.state_dict.keymap_sdxl_clip import ClipLayoutMetadata


def _ensure_position_ids_long(sd: Dict[str, Any], key: str) -> None:
    value = sd.get(key)
    if isinstance(value, torch.Tensor) and value.dtype != torch.long:
        sd[key] = value.round().to(torch.long)


def _with_prefix(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in sd.items()}


def _strip_prefix(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    plen = len(prefix)
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
        else:
            out[k] = v
    return out


_QKVImpl = Literal["auto", "split", "fused"]
_ProjectionOrientation = Literal["auto", "linear", "matmul"]


def convert_clip(
    sd: Dict[str, Any],
    *,
    alias: str,
    layers: int,
    ensure_position_ids: bool = False,
    drop_logit_scale: bool = False,
    qkv_impl: _QKVImpl = "auto",
    projection_orientation: _ProjectionOrientation = "auto",
    layout_metadata: ClipLayoutMetadata | None = None,
) -> Dict[str, Any]:
    prefixed = _with_prefix(dict(sd), f"{alias}.")
    work, _layout = normalize_codex_clip_state_dict_with_layout(
        prefixed,
        num_layers=layers,
        keep_projection=True,
        qkv_impl=qkv_impl,
        projection_orientation=projection_orientation,
        layout_metadata=layout_metadata,
        require_projection=False,
    )
    if ensure_position_ids:
        _ensure_position_ids_long(work, "transformer.text_model.embeddings.position_ids")
    if drop_logit_scale:
        work.pop("logit_scale", None)
    return _strip_prefix(work, f"{alias}.")


def convert_sd15_clip(sd: Dict[str, Any]) -> Dict[str, Any]:
    converted = convert_clip(
        sd,
        alias="clip_l",
        layers=12,
        ensure_position_ids=True,
        drop_logit_scale=True,
    )
    # Remove heads reconstructed at runtime.
    converted.pop("transformer.text_projection.weight", None)
    return converted


def convert_sd20_clip(sd: Dict[str, Any]) -> Dict[str, Any]:
    return convert_clip(
        sd,
        alias="clip_h",
        layers=32,
        ensure_position_ids=True,
        drop_logit_scale=True,
        projection_orientation="auto",
    )


def convert_sdxl_clip_l(sd: Dict[str, Any]) -> Dict[str, Any]:
    converted = convert_clip(
        sd,
        alias="clip_l",
        layers=32,
        ensure_position_ids=True,
        drop_logit_scale=True,
    )
    converted.pop("transformer.text_projection.weight", None)
    return converted


def convert_sdxl_clip_g(sd: Dict[str, Any]) -> Dict[str, Any]:
    return convert_clip(
        sd,
        alias="clip_g",
        layers=32,
        ensure_position_ids=True,
        drop_logit_scale=False,
        projection_orientation="auto",
    )
