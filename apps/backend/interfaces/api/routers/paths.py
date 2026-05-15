"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Paths configuration API routes.
Exposes apps/paths.json for the UI and accepts updates for engine-specific keys, with fail-loud read/write/type validation.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for paths endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from apps.backend.interfaces.api.json_store import _load_json, _save_json
from apps.backend.services.model_catalog import current_models_revision, invalidate_model_catalog


def build_router(*, codex_root: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/api/paths")
    def get_paths() -> Dict[str, Any]:
        cfg_path_obj = codex_root / "apps" / "paths.json"
        if not cfg_path_obj.is_file():
            raise HTTPException(status_code=500, detail=f"paths config missing: {cfg_path_obj}")
        cfg_path = str(cfg_path_obj)
        try:
            raw = _load_json(cfg_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read paths config: {exc}") from exc

        paths: Dict[str, list[str]] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise HTTPException(status_code=500, detail=f"paths config contains a non-string key: {key!r}")
            if not isinstance(value, list):
                raise HTTPException(status_code=500, detail=f"paths config entry {key!r} must be a list")
            normalized_values: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    raise HTTPException(status_code=500, detail=f"paths config entry {key!r} contains a non-string value")
                normalized_values.append(item)
            paths[key] = normalized_values

        return {"paths": paths}

    @router.post("/api/paths")
    def set_paths(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict) or "paths" not in payload or not isinstance(payload["paths"], dict):
            raise HTTPException(status_code=400, detail='payload must be {"paths": {...}}')

        cfg_path_obj = codex_root / "apps" / "paths.json"
        if not cfg_path_obj.is_file():
            raise HTTPException(status_code=500, detail=f"paths config missing: {cfg_path_obj}")
        cfg_path = str(cfg_path_obj)
        try:
            current = _load_json(cfg_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read paths config: {exc}") from exc

        incoming = payload["paths"] or {}
        new_paths: Dict[str, Any] = dict(current)

        for key, value in incoming.items():
            if not isinstance(key, str):
                raise HTTPException(status_code=400, detail="paths keys must be strings")
            if value is None:
                new_paths[key] = []
            elif isinstance(value, list):
                normalized_values: list[str] = []
                for item in value:
                    if not isinstance(item, str):
                        raise HTTPException(status_code=400, detail=f"paths[{key!r}] entries must be strings")
                    normalized_values.append(item)
                new_paths[key] = normalized_values
            else:
                raise HTTPException(status_code=400, detail=f"paths[{key!r}] must be a list or null")

        changed = new_paths != current
        if changed:
            try:
                _save_json(cfg_path, new_paths)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"failed to write paths config: {exc}") from exc
            try:
                models_revision = int(invalidate_model_catalog(reason="api.paths.set"))
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"paths updated but model catalog invalidation failed: {exc}",
                ) from exc
        else:
            models_revision = int(current_models_revision())
        return {"ok": True, "changed": changed, "models_revision": models_revision}

    return router
