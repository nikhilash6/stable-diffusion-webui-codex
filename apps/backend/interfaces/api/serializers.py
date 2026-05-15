"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Serialization helpers for API responses.
Keeps checkpoint listing responses explicit and inventory-authoritative, and fails loud when canonical `short_hash` / `filename`
metadata is missing from `/api/models` checkpoint records.

Symbols (top-level; keep in sync; no ghosts):
- `_serialize_checkpoint` (function): Serializes a checkpoint record for `/api/models` responses (`hash`/`filename`/`format`/`core_only` and related metadata), rejecting missing canonical metadata.
"""

from __future__ import annotations

from typing import Any, Dict


def _serialize_checkpoint(info) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
    short_hash = getattr(info, "short_hash", None)
    if not isinstance(short_hash, str) or not short_hash.strip():
        raise RuntimeError("Checkpoint record is missing short_hash metadata for /api/models serialization.")
    filename = getattr(info, "filename", None)
    if not isinstance(filename, str) or not filename.strip():
        raise RuntimeError("Checkpoint record is missing filename metadata for /api/models serialization.")
    raw_format = getattr(info, "format", None)
    normalized_format = getattr(raw_format, "value", raw_format)
    if not isinstance(normalized_format, str) or not normalized_format.strip():
        raise RuntimeError("Checkpoint record is missing format metadata for /api/models serialization.")
    core_only = getattr(info, "core_only", None)
    if not isinstance(core_only, bool):
        raise RuntimeError("Checkpoint record is missing core_only metadata for /api/models serialization.")
    payload: Dict[str, Any] = {
        "title": info.title,
        "name": info.name,
        "model_name": info.model_name,
        "hash": short_hash.strip(),
        "filename": filename.strip(),
        "format": normalized_format.strip().lower(),
        "metadata": info.metadata,
        "core_only": core_only,
    }
    core_only_reason = getattr(info, "core_only_reason", None)
    if isinstance(core_only_reason, str) and core_only_reason.strip():
        payload["core_only_reason"] = core_only_reason.strip()
    family_hint = getattr(info, "family_hint", None)
    if isinstance(family_hint, str) and family_hint.strip():
        payload["family_hint"] = family_hint.strip()
    return payload
