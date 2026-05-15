"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed SUPIR mode config for canonical SDXL img2img/inpaint ownership.
Defines the nested `img2img_extras.supir` request surface used by the canonical img2img pipeline and a strict parser that:
- accepts only the nested object shape,
- validates the tranche-1 public SUPIR controls, including the runtime-owned restore-window knob,
- rejects flat legacy `supir_*` aliases and unknown keys.

Symbols (top-level; keep in sync; no ghosts):
- `SupirColorFixMode` (type alias): Allowed SUPIR color-fix labels.
- `SupirModeConfig` (dataclass): Parsed SUPIR mode configuration (`img2img_extras.supir` -> typed fields).
- `parse_supir_mode_config` (function): Parse and validate a nested SUPIR config object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypeAlias

from .errors import SupirConfigError
from .samplers.registry import resolve_supir_sampler
from .samplers.types import SupirSamplerSpec
from .weights import SupirVariant

SupirColorFixMode: TypeAlias = Literal["None", "AdaIN", "Wavelet"]

_ALLOWED_KEYS = frozenset(
    {
        "enabled",
        "variant",
        "sampler",
        "controlScale",
        "restorationScale",
        "restoreCfgSTmin",
        "colorFix",
    }
)


@dataclass(frozen=True)
class SupirModeConfig:
    enabled: bool
    variant: SupirVariant
    sampler: SupirSamplerSpec
    control_scale: float = 0.8
    restoration_scale: float = 4.0
    restore_cfg_s_tmin: float = 0.05
    color_fix: SupirColorFixMode = "None"


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise SupirConfigError(f"{name} must be a bool")


def _as_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise SupirConfigError(f"{name} must be a float, not bool")
    try:
        parsed = float(value)
    except Exception as exc:  # noqa: BLE001
        raise SupirConfigError(f"{name} must be a float") from exc
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise SupirConfigError(f"{name} must be finite")
    return parsed


def _as_str(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise SupirConfigError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise SupirConfigError(f"{name} must be a non-empty string")
    return text


def _parse_variant(value: Any) -> SupirVariant:
    raw = _as_str(value, name="img2img_extras.supir.variant")
    try:
        return SupirVariant(raw)
    except Exception:  # noqa: BLE001
        allowed = ", ".join(v.value for v in SupirVariant)
        raise SupirConfigError(f"img2img_extras.supir.variant must be one of: {allowed}") from None


def _parse_color_fix(value: Any) -> SupirColorFixMode:
    raw = _as_str(value, name="img2img_extras.supir.colorFix")
    normalized = raw.lower()
    if normalized == "none":
        return "None"
    if normalized == "adain":
        return "AdaIN"
    if normalized == "wavelet":
        return "Wavelet"
    raise SupirConfigError("img2img_extras.supir.colorFix must be one of: None, AdaIN, Wavelet")


def parse_supir_mode_config(payload: Mapping[str, Any] | None) -> SupirModeConfig | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise SupirConfigError("img2img_extras.supir must be an object")

    flat_aliases = sorted(str(key) for key in payload.keys() if str(key).startswith("supir_"))
    if flat_aliases:
        raise SupirConfigError(
            "Flat SUPIR request aliases are not supported; send only nested 'img2img_extras.supir'. "
            f"Found: {flat_aliases}"
        )

    unknown = sorted(str(key) for key in payload.keys() if str(key) not in _ALLOWED_KEYS)
    if unknown:
        raise SupirConfigError(f"Unexpected img2img_extras.supir key(s): {', '.join(unknown)}")

    enabled = _as_bool(payload.get("enabled", False), name="img2img_extras.supir.enabled")
    if not enabled:
        return None

    variant = _parse_variant(payload.get("variant", SupirVariant.V0Q.value))
    sampler = resolve_supir_sampler(
        payload.get("sampler", "restore_euler_edm_stable"),
        include_dev=False,
    )
    control_scale = _as_float(payload.get("controlScale", 0.8), name="img2img_extras.supir.controlScale")
    restoration_scale = _as_float(
        payload.get("restorationScale", 4.0),
        name="img2img_extras.supir.restorationScale",
    )
    restore_cfg_s_tmin = _as_float(
        payload.get("restoreCfgSTmin", 0.05),
        name="img2img_extras.supir.restoreCfgSTmin",
    )
    color_fix = _parse_color_fix(payload.get("colorFix", "None"))

    if control_scale <= 0.0:
        raise SupirConfigError("img2img_extras.supir.controlScale must be > 0")
    if control_scale > 2.0:
        raise SupirConfigError("img2img_extras.supir.controlScale must be <= 2")
    if restoration_scale <= 0.0:
        raise SupirConfigError("img2img_extras.supir.restorationScale must be > 0")
    if restoration_scale > 6.0:
        raise SupirConfigError("img2img_extras.supir.restorationScale must be <= 6")
    if restore_cfg_s_tmin < 0.0:
        raise SupirConfigError("img2img_extras.supir.restoreCfgSTmin must be >= 0")
    if restore_cfg_s_tmin > 5.0:
        raise SupirConfigError("img2img_extras.supir.restoreCfgSTmin must be <= 5")
    return SupirModeConfig(
        enabled=True,
        variant=variant,
        sampler=sampler,
        control_scale=control_scale,
        restoration_scale=restoration_scale,
        restore_cfg_s_tmin=restore_cfg_s_tmin,
        color_fix=color_fix,
    )


__all__ = [
    "SupirColorFixMode",
    "SupirModeConfig",
    "parse_supir_mode_config",
]
