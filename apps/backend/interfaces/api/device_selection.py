"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared explicit device selection helpers for API payloads (fail loud).
Resolves the configured runtime main device, applies payload validation against that invariant, and applies the
device switch only when the task actually starts running (single-flight-safe).

Symbols (top-level; keep in sync; no ghosts):
- `_cuda_available_for_fallback` (function): Returns whether CUDA is available for default main-device fallback.
- `_normalize_backend_label` (function): Normalizes runtime/backend labels into canonical API device keys.
- `GenerationRouteMode` (enum): Typed generation route/mode keys for route-level device policy lookup.
- `RouteDevicePolicy` (dataclass): Typed route-level device policy entry (label + optional allowlist).
- `generation_route_device_policy` (function): Resolve typed generation route/mode policy from the central policy matrix.
- `configured_main_device` (function): Resolves the active configured main device (live memory-manager authority first, then args/env/fallback when manager is unavailable).
- `parse_device_from_payload` (function): Validates payload device, enforces main-device invariant, and applies optional route policy allowlist checks.
- `apply_primary_device` (function): Applies the validated device via `memory_management.switch_primary_device`.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass
from typing import Any, Mapping


_ALLOWED_DEVICES = {"cpu", "cuda", "mps", "xpu", "directml"}
_WAN_VIDEO_ALLOWED_DEVICES = frozenset({"cpu", "cuda"})


class GenerationRouteMode(str, enum.Enum):
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    TXT2VID = "txt2vid"
    IMG2VID = "img2vid"
    VID2VID = "vid2vid"


@dataclass(frozen=True, slots=True)
class RouteDevicePolicy:
    route_mode: GenerationRouteMode
    route_label: str
    allowed_devices: frozenset[str] | None = None


_GENERATION_ROUTE_DEVICE_POLICY_MATRIX: dict[GenerationRouteMode, RouteDevicePolicy] = {
    GenerationRouteMode.TXT2IMG: RouteDevicePolicy(
        route_mode=GenerationRouteMode.TXT2IMG,
        route_label="txt2img",
        allowed_devices=None,
    ),
    GenerationRouteMode.IMG2IMG: RouteDevicePolicy(
        route_mode=GenerationRouteMode.IMG2IMG,
        route_label="img2img",
        allowed_devices=None,
    ),
    GenerationRouteMode.TXT2VID: RouteDevicePolicy(
        route_mode=GenerationRouteMode.TXT2VID,
        route_label="video generation",
        allowed_devices=_WAN_VIDEO_ALLOWED_DEVICES,
    ),
    GenerationRouteMode.IMG2VID: RouteDevicePolicy(
        route_mode=GenerationRouteMode.IMG2VID,
        route_label="video generation",
        allowed_devices=_WAN_VIDEO_ALLOWED_DEVICES,
    ),
    GenerationRouteMode.VID2VID: RouteDevicePolicy(
        route_mode=GenerationRouteMode.VID2VID,
        route_label="video generation",
        allowed_devices=_WAN_VIDEO_ALLOWED_DEVICES,
    ),
}


def generation_route_device_policy(route_mode: GenerationRouteMode) -> RouteDevicePolicy:
    try:
        return _GENERATION_ROUTE_DEVICE_POLICY_MATRIX[route_mode]
    except KeyError as exc:
        raise RuntimeError(f"Missing generation route device policy for mode {route_mode!r}.") from exc


def _cuda_available_for_fallback() -> bool:
    try:
        import torch  # type: ignore

        return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        return False


def _normalize_backend_label(raw: str) -> str:
    normalized = str(raw or "").strip().lower()
    if not normalized:
        raise ValueError("Empty device/backend label.")
    if normalized == "gpu":
        return "cuda"
    if normalized == "dml":
        return "directml"
    if normalized.startswith("cuda"):
        return "cuda"
    if normalized in _ALLOWED_DEVICES:
        return normalized
    raise ValueError(f"Unsupported device/backend label: {raw!r}")


def configured_main_device() -> str:
    mem_management = None
    try:
        from apps.backend.runtime.memory import memory_management as mem_management
    except ModuleNotFoundError as exc:
        missing_name = str(getattr(exc, "name", "") or "")
        if missing_name not in {
            "apps.backend.runtime.memory",
            "apps.backend.runtime.memory.memory_management",
        }:
            raise RuntimeError(
                "Failed to import runtime memory manager dependencies for primary device authority."
            ) from exc
        mem_management = None
    except Exception as exc:
        raise RuntimeError("Failed to import runtime memory manager for primary device authority.") from exc

    manager = getattr(mem_management, "manager", None) if mem_management is not None else None
    if manager is not None:
        if not hasattr(manager, "primary_device"):
            raise RuntimeError(
                "Runtime memory manager contract violation: missing primary_device() authority."
            )
        try:
            primary_device = manager.primary_device()
        except Exception as exc:
            raise RuntimeError("Failed to read primary device from runtime memory manager.") from exc
        return _normalize_backend_label(str(primary_device))
    if mem_management is not None:
        raise RuntimeError(
            "Runtime memory manager contract violation: imported memory module has no `manager` authority."
        )

    from apps.backend.infra.config import args as runtime_args

    candidates: list[str | None] = [
        getattr(runtime_args.args, "codex_main_device", None),
        os.getenv("CODEX_MAIN_DEVICE"),
    ]
    for raw in candidates:
        normalized = str(raw or "").strip().lower()
        if not normalized:
            continue
        if normalized == "auto":
            return "cuda" if _cuda_available_for_fallback() else "cpu"
        try:
            return _normalize_backend_label(normalized)
        except ValueError:
            continue
    return "cuda" if _cuda_available_for_fallback() else "cpu"


def parse_device_from_payload(
    payload: Mapping[str, Any],
    *,
    route_policy: RouteDevicePolicy | None = None,
) -> str:
    main = configured_main_device()
    for legacy_key in ("codex_device", "codex_diffusion_device"):
        if legacy_key in payload:
            raise ValueError(f"Unsupported legacy device key: '{legacy_key}'. Use 'device'.")
    raw = payload.get("device") or ""
    dev = str(raw).strip().lower()
    allowed = "|".join(sorted(_ALLOWED_DEVICES))
    if not dev:
        raise ValueError(f"Missing 'device' selection (allowed values: {allowed})")
    if dev not in _ALLOWED_DEVICES:
        raise ValueError(f"Invalid device (allowed: {allowed})")
    if dev != main:
        raise ValueError(
            f"Device override '{dev}' diverges from configured main device '{main}'. "
            "Set launcher main device and keep payload device aligned."
        )
    allowed_by_route = route_policy.allowed_devices if route_policy is not None else None
    if allowed_by_route is not None and dev not in allowed_by_route:
        route_allowed = "|".join(sorted(allowed_by_route))
        route_label = route_policy.route_label.strip() if isinstance(route_policy.route_label, str) else ""
        label = route_label or "This route"
        raise ValueError(
            f"{label} supports only {route_allowed} "
            "(or auto resolving to one of those backends)."
        )
    return dev


def apply_primary_device(device: str) -> None:
    from apps.backend.runtime.memory import memory_management as mem_management

    mem_management.switch_primary_device(str(device).strip().lower())


__all__ = [
    "GenerationRouteMode",
    "RouteDevicePolicy",
    "generation_route_device_policy",
    "configured_main_device",
    "parse_device_from_payload",
    "apply_primary_device",
]
