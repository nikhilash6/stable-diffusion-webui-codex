"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Options API routes for reading, updating, and validating settings.
Exposes the JSON-backed options store and registry-driven validation helpers. Enforces finite numeric values for number/slider settings
and emits apply metadata on `POST /api/options`. Conditional writes require `X-Codex-Expected-Revision` so stale clients fail loud instead
of silently overwriting newer option state. Memory-manager backend and dtype changes are persisted but require a backend restart
(launcher-owned) to take effect; this endpoint does not hot-apply memory reconfiguration.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for options endpoints.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, List

from fastapi import APIRouter, Body, Header, HTTPException


def build_router(
    *,
    opts_load_native: Callable[[], Dict[str, Any]],
    opts_snapshot,
    opts_set_many_if_revision: Callable[[Dict[str, Any]], tuple[list[str], int]]
    | Callable[[Dict[str, Any], int | None], tuple[list[str], int]],
    settings_registry_ok: bool,
    field_index: Callable[[], Dict[str, Any]],
    setting_type,
) -> APIRouter:
    router = APIRouter()
    VAE_BY_FAMILY_OPTION_KEY = "codex_vae_by_family"
    VAE_FAMILIES = ("sd15", "sdxl", "flux1", "flux2", "chroma", "zimage", "anima")
    hot_apply_reasons: Dict[str, str] = {
        "codex_smart_offload": "hot-applied immediately (effective for the next generation request).",
        "codex_smart_fallback": "hot-applied immediately (effective for the next generation request).",
        "codex_smart_cache": "hot-applied immediately (effective for the next generation request).",
        "codex_core_streaming": "hot-applied immediately (effective for the next generation request).",
        "codex_export_video": "hot-applied immediately (effective for the next generation request).",
        "codex_vae_by_family": "hot-applied immediately (UI preference persistence).",
    }

    @router.get("/api/options")
    def get_options() -> Dict[str, Any]:
        try:
            values = opts_load_native()
            revision = int(values.get("codex_options_revision", 0) or 0)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read options revision: {exc}") from exc
        return {"values": values, "revision": max(0, revision)}

    @router.get("/api/options/keys")
    def get_options_keys() -> Dict[str, Any]:
        """List supported option keys and basic metadata from the settings registry."""
        if not settings_registry_ok:
            return {"keys": [], "types": {}, "choices": {}}
        try:
            idx = field_index()
            keys = list(idx.keys())
            types = {}
            choices = {}
            for k, f in idx.items():
                t = getattr(getattr(f, "type", None), "name", None) or str(getattr(f, "type", None))
                types[k] = t
                ch = getattr(f, "choices", None)
                if isinstance(ch, list):
                    choices[k] = ch
            return {"keys": keys, "types": types, "choices": choices}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read registry: {exc}")

    @router.get("/api/options/snapshot")
    def get_options_snapshot() -> Dict[str, Any]:
        """Return a typed snapshot of current options (for UI defaults)."""
        try:
            return {"snapshot": opts_snapshot().as_dict()}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read snapshot: {exc}")

    @router.get("/api/options/defaults")
    def get_options_defaults() -> Dict[str, Any]:
        """Return default values from the settings registry and the current snapshot."""
        defaults: Dict[str, Any] = {}
        if settings_registry_ok:
            try:
                idx = field_index()
                for k, f in idx.items():
                    defaults[k] = getattr(f, "default", None)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"failed to read registry defaults: {exc}") from exc
        try:
            snap = opts_snapshot().as_dict()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read options snapshot: {exc}") from exc
        return {"defaults": defaults, "snapshot": snap}

    def _parse_checkbox_value(key: str, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value in (0, 1):
                return bool(int(value))
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("1", "true", "yes", "on"):
                return True
            if normalized in ("0", "false", "no", "off"):
                return False
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid value for {key}: expected bool or one of "
                f"('true','false','1','0','yes','no','on','off')."
            ),
        )

    def _normalize_vae_selection(raw: str) -> str:
        value = str(raw or "").strip()
        if not value:
            return ""
        lower = value.lower()
        if lower in {"automatic", "built in", "built-in"}:
            return "built-in"
        if lower == "none":
            return "none"
        return value

    def _validate_vae_by_family_value(value: Any) -> str:
        parsed: Any = value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid value for {VAE_BY_FAMILY_OPTION_KEY}: expected JSON object string "
                        "or object payload."
                    ),
                )
            try:
                parsed = json.loads(text)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for {VAE_BY_FAMILY_OPTION_KEY}: invalid JSON ({exc}).",
                ) from exc
        if not isinstance(parsed, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for {VAE_BY_FAMILY_OPTION_KEY}: expected object.",
            )
        out: Dict[str, str] = {}
        for family_raw, selected_raw in parsed.items():
            family = str(family_raw or "").strip().lower()
            if family not in VAE_FAMILIES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid value for {VAE_BY_FAMILY_OPTION_KEY}: unknown family "
                        f"'{family_raw}'. Allowed: {', '.join(VAE_FAMILIES)}."
                    ),
                )
            if not isinstance(selected_raw, str):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid value for {VAE_BY_FAMILY_OPTION_KEY}: family '{family}' must map to a string."
                    ),
                )
            normalized_selection = _normalize_vae_selection(selected_raw)
            if not normalized_selection:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid value for {VAE_BY_FAMILY_OPTION_KEY}: family '{family}' has empty selection."
                    ),
                )
            out[family] = normalized_selection
        return json.dumps(out, sort_keys=True, separators=(",", ":"))

    def _validate_options(payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid payload")
        if not settings_registry_ok:
            raise HTTPException(
                status_code=503,
                detail=(
                    "settings registry unavailable; options validation is disabled until backend restart "
                    "restores registry health"
                ),
            )
        try:
            idx = field_index()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"registry unavailable: {exc}")

        unknown = sorted(k for k in payload.keys() if k not in idx)
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown option key(s): {', '.join(unknown)}")

        out: Dict[str, Any] = {}
        for k, v in payload.items():
            f = idx[k]
            try:
                if k == VAE_BY_FAMILY_OPTION_KEY:
                    out[k] = _validate_vae_by_family_value(v)
                    continue
                if getattr(f, "choices", None) and isinstance(f.choices, list) and v not in f.choices:
                    raise HTTPException(status_code=400, detail=f"Invalid value for {k}: not in choices")
                if getattr(f, "type", None) in (setting_type.SLIDER, setting_type.NUMBER):
                    if isinstance(v, bool):
                        raise HTTPException(status_code=400, detail=f"Invalid value for {k}: boolean is not a numeric value")
                    num = float(v)
                    if not math.isfinite(num):
                        raise HTTPException(status_code=400, detail=f"Invalid value for {k}: must be finite")
                    lo = getattr(f, "min", None)
                    hi = getattr(f, "max", None)
                    if isinstance(lo, (int, float)) and num < lo:
                        raise HTTPException(status_code=400, detail=f"Invalid value for {k}: below min {lo}")
                    if isinstance(hi, (int, float)) and num > hi:
                        raise HTTPException(status_code=400, detail=f"Invalid value for {k}: above max {hi}")
                    out[k] = num
                elif getattr(f, "type", None) == setting_type.CHECKBOX:
                    out[k] = _parse_checkbox_value(k, v)
                else:
                    out[k] = v
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid value for {k}: {exc}") from exc
        return out

    def _parse_expected_revision(raw_value: str | None) -> int | None:
        if raw_value is None:
            raise HTTPException(
                status_code=428,
                detail="X-Codex-Expected-Revision header is required for POST /api/options.",
            )
        trimmed = str(raw_value).strip()
        if not trimmed:
            raise HTTPException(
                status_code=428,
                detail="X-Codex-Expected-Revision header is required for POST /api/options.",
            )
        if not trimmed.isdigit():
            raise HTTPException(status_code=400, detail="X-Codex-Expected-Revision must be a non-negative integer.")
        return max(0, int(trimmed))

    @router.post("/api/options")
    def set_options(
        payload: Dict[str, Any] = Body(...),
        expected_revision_header: str | None = Header(default=None, alias="X-Codex-Expected-Revision"),
    ) -> Dict[str, Any]:
        updates = _validate_options(payload)
        expected_revision = _parse_expected_revision(expected_revision_header)
        try:
            updated, revision = opts_set_many_if_revision(updates, expected_revision=expected_revision)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            expected = getattr(exc, "expected_revision", None)
            current = getattr(exc, "current_revision", None)
            if isinstance(expected, int) and isinstance(current, int):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": (
                            f"stale options write rejected: expected revision {expected}, current revision is "
                            f"{current}."
                        ),
                        "expected_revision": max(0, expected),
                        "current_revision": max(0, current),
                    },
                ) from exc
            raise HTTPException(
                status_code=500,
                detail="Failed to persist options.",
            ) from exc
        applied_now: List[str] = []
        restart_required: List[str] = []
        for key in updated:
            if key == "codex_options_revision":
                continue
            reason = hot_apply_reasons.get(key)
            if reason is not None:
                applied_now.append(f"{key}: {reason}")
                continue
            restart_required.append(f"{key}: not hot-applied; restart required.")
        return {
            "updated": updated,
            "revision": max(0, revision),
            "applied_now": applied_now,
            "restart_required": restart_required,
        }

    @router.post("/api/options/validate")
    def validate_options(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Dry-run options validation; returns accepted and rejected keys with reasons."""
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid payload")
        if not settings_registry_ok:
            raise HTTPException(
                status_code=503,
                detail=(
                    "settings registry unavailable; options validation is disabled until backend restart "
                    "restores registry health"
                ),
            )
        try:
            idx = field_index()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"registry unavailable: {exc}")
        accepted: Dict[str, Any] = {}
        rejected: Dict[str, str] = {}
        for k, v in payload.items():
            f = idx.get(k)
            if not f:
                rejected[k] = "unknown key"
                continue
            try:
                if k == VAE_BY_FAMILY_OPTION_KEY:
                    accepted[k] = _validate_vae_by_family_value(v)
                    continue
                if getattr(f, "choices", None) and isinstance(f.choices, list) and v not in f.choices:
                    rejected[k] = "not in choices"
                    continue
                if getattr(f, "type", None) in (setting_type.SLIDER, setting_type.NUMBER):
                    if isinstance(v, bool):
                        rejected[k] = "boolean is not a numeric value"
                        continue
                    num = float(v)
                    if not math.isfinite(num):
                        rejected[k] = "must be finite"
                        continue
                    lo = getattr(f, "min", None)
                    hi = getattr(f, "max", None)
                    if isinstance(lo, (int, float)) and num < lo:
                        rejected[k] = f"below min {lo}"
                        continue
                    if isinstance(hi, (int, float)) and num > hi:
                        rejected[k] = f"above max {hi}"
                        continue
                    accepted[k] = num
                elif getattr(f, "type", None) == setting_type.CHECKBOX:
                    accepted[k] = _parse_checkbox_value(k, v)
                else:
                    accepted[k] = v
            except HTTPException as exc:
                rejected[k] = str(exc.detail)
            except Exception:
                rejected[k] = "invalid value"
        return {"accepted": accepted, "rejected": rejected}

    return router
