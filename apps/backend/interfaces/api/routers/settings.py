"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Settings schema/values API routes.
Serves the settings schema (hardcoded registry or JSON fallback) and current stored values.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for settings endpoints.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException

from apps.backend.interfaces.api.json_store import _load_json


def build_router(
    *,
    codex_root: Path,
    settings_registry_ok: bool,
    schema_hardcoded: Callable[[], Dict[str, Any]] | None,
    field_index: Callable[[], Dict[str, Any]],
    opts_load_native: Callable[[], Dict[str, Any]],
) -> APIRouter:
    router = APIRouter()
    _settings_schema_cache: Optional[Dict[str, Any]] = None

    @router.get("/api/settings/schema")
    def settings_schema() -> Dict[str, Any]:
        nonlocal _settings_schema_cache
        if _settings_schema_cache is None:
            if settings_registry_ok and schema_hardcoded is not None:
                try:
                    _settings_schema_cache = schema_hardcoded()
                except Exception as exc:  # pragma: no cover
                    get_backend_logger("backend.api").warning("settings hardcoded schema failed: %s", exc)
                    _settings_schema_cache = None
            if _settings_schema_cache is None:
                schema_path = str(codex_root / "apps" / "backend" / "interfaces" / "schemas" / "settings_schema.json")
                _settings_schema_cache = _load_json(schema_path)
                if not _settings_schema_cache:
                    raise HTTPException(status_code=500, detail="settings schema not found (registry and JSON)")
        return _settings_schema_cache

    @router.get("/api/settings/values")
    def settings_values() -> Dict[str, Any]:
        try:
            vals = opts_load_native()
            idx = field_index() if settings_registry_ok else {}
            if idx:
                vals = {k: vals.get(k) for k in idx.keys()}
            return {"values": vals}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read values: {exc}")

    return router
