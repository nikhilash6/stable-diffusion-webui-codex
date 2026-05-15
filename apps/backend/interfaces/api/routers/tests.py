"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Bounded backend diagnostics routes.
Exposes narrow live-validation endpoints for runtime-owned diagnostics (currently SRAM split-KV
validation and the IP-Adapter conditioning probe) without overloading the system health surface
or turning the API into an arbitrary test-execution framework. This route family is
operator-facing: malformed payloads still fail with `400`, while expected execution outcomes are
returned as structured receipts.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for bounded diagnostics endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from apps.backend.interfaces.api.path_utils import _path_for_api, _path_from_api
from apps.backend.runtime.attention.sram.splitkv_validation import (
    SplitKvValidationInvalidRequest,
    parse_splitkv_validation_request,
    run_splitkv_validation,
)
from apps.backend.runtime.adapters.ip_adapter.probe import (
    IpAdapterProbeInvalidRequest,
    parse_ip_adapter_probe_request,
    run_ip_adapter_probe_subprocess,
)


def build_router() -> APIRouter:
    router = APIRouter()

    def _resolve_inventory_scoped_path(
        raw: object,
        *,
        field_name: str,
        inventory_key: str,
        inventory_label: str,
    ) -> str:
        if not isinstance(raw, str) or not raw.strip():
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be a non-empty string")
        raw_value = raw.strip()
        from apps.backend.inventory import cache as _inventory_cache

        inventory = _inventory_cache.get()
        try:
            raw_absolute = str(Path(raw_value).expanduser().resolve(strict=False))
        except Exception:
            raw_absolute = ""

        for item in inventory.get(inventory_key, []):
            if not isinstance(item, dict):
                continue
            item_path = item.get("path")
            if not isinstance(item_path, str) or not item_path.strip():
                continue
            absolute_path = str(Path(item_path).expanduser().resolve(strict=False))
            api_path = _path_for_api(item_path)
            if raw_value == api_path or (raw_absolute and raw_absolute == absolute_path):
                return absolute_path

        raise HTTPException(
            status_code=400,
            detail=(
                f"'{field_name}' must resolve to an inventory-backed {inventory_label} path from "
                f"'/api/models' ({inventory_key})."
            ),
        )

    def _normalize_ip_adapter_probe_payload(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="payload must be a JSON object")
        allowed_keys = {"model", "image_encoder", "source", "crop", "compare_official_encoder"}
        unknown_keys = sorted(str(key) for key in payload.keys() if key not in allowed_keys)
        if unknown_keys:
            raise HTTPException(status_code=400, detail=f"unknown payload keys: {', '.join(unknown_keys)}")
        source = payload.get("source")
        if not isinstance(source, dict):
            raise HTTPException(status_code=400, detail="'source' must be an object")
        source_allowed_keys = {"kind", "reference_image_data", "reference_image_path"}
        unknown_source_keys = sorted(str(key) for key in source.keys() if key not in source_allowed_keys)
        if unknown_source_keys:
            raise HTTPException(status_code=400, detail=f"unknown payload keys in 'source': {', '.join(unknown_source_keys)}")
        source_kind = source.get("kind")
        if not isinstance(source_kind, str) or not source_kind.strip():
            raise HTTPException(status_code=400, detail="'source.kind' must be a non-empty string")
        normalized: dict[str, Any] = {
            "model_path": _resolve_inventory_scoped_path(
                payload.get("model"),
                field_name="model",
                inventory_key="ip_adapter_models",
                inventory_label="IP-Adapter model",
            ),
            "image_encoder_path": _resolve_inventory_scoped_path(
                payload.get("image_encoder"),
                field_name="image_encoder",
                inventory_key="ip_adapter_image_encoders",
                inventory_label="IP-Adapter image encoder",
            ),
            "source_kind": source_kind.strip().lower(),
            "crop": payload.get("crop", True),
            "compare_official_encoder": payload.get("compare_official_encoder", False),
        }
        if normalized["source_kind"] == "path":
            try:
                normalized["reference_image_path"] = _path_from_api(source.get("reference_image_path"))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"'source.reference_image_path' {exc}") from exc
        else:
            normalized["reference_image_data"] = source.get("reference_image_data")
        return normalized

    @router.post("/api/tests/attention/sram/splitkv")
    def validate_attention_sram_splitkv(payload: Any = Body(default=None)) -> dict[str, Any]:
        try:
            request = parse_splitkv_validation_request(payload)
            return run_splitkv_validation(request).to_payload()
        except SplitKvValidationInvalidRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive sync route guard
            raise HTTPException(status_code=500, detail="internal error") from exc

    @router.post("/api/tests/ip-adapter/probe")
    def probe_ip_adapter(payload: Any = Body(default=None)) -> dict[str, Any]:
        try:
            normalized_payload = _normalize_ip_adapter_probe_payload(payload)
            request = parse_ip_adapter_probe_request(normalized_payload)
            return run_ip_adapter_probe_subprocess(request).to_payload()
        except IpAdapterProbeInvalidRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive sync route guard
            raise HTTPException(status_code=500, detail="internal error") from exc

    return router
