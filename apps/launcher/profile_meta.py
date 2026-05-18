"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher profile metadata, mode-profile constants, allocator constants, and strict manual API env parsing.
Owns persisted `meta.json` shape plus helper parsers that are shared by GUI, Docker TUI, and service startup code without depending on env-map storage.

Symbols (top-level; keep in sync; no ghosts):
- `DEFAULT_MODEL_NAME` (constant): Default launcher model overlay name.
- `DEFAULT_PYTORCH_CUDA_ALLOC_CONF` (constant): Default `PYTORCH_CUDA_ALLOC_CONF` applied by launcher env build when enabled.
- `ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY` (constant): Env key toggling default allocator config injection.
- `CODEX_CUDA_MALLOC_KEY` (constant): Env key toggling backend `--cuda-malloc` forwarding in launcher-managed runs.
- `CODEX_APP_MODE_PROFILE_ENV_KEY` (constant): Env key selecting launcher app mode profile.
- `LAUNCHER_MODE_PROFILE_CHOICES` (constant): Allowed launcher app mode profile values.
- `DEFAULT_MANUAL_API_ENV_TEXT` (constant): Suggested manual API env overlay text prefilled in launcher metadata.
- `_default_external_terminal_enabled` (function): Canonical default provider for launcher external-terminal preference.
- `normalize_mode_profile` (function): Strict parser/validator for launcher app mode profile values.
- `parse_manual_api_env_text` (function): Parses manual API env text (`KEY=VALUE` per line) with strict validation.
- `LauncherMeta` (dataclass): Persisted launcher UI metadata.
- `load_launcher_meta` (function): Loads `LauncherMeta` from disk or creates defaults.
- `write_launcher_meta` (function): Writes `LauncherMeta` to disk.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

DEFAULT_MODEL_NAME = "default"
DEFAULT_PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:256,garbage_collection_threshold:0.8"
ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY = "CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF"
CODEX_CUDA_MALLOC_KEY = "CODEX_CUDA_MALLOC"
CODEX_APP_MODE_PROFILE_ENV_KEY = "CODEX_APP_MODE_PROFILE"
LAUNCHER_MODE_PROFILE_DEV_SERVICE = "dev_service"
LAUNCHER_MODE_PROFILE_EMBEDDED = "embedded"
LAUNCHER_MODE_PROFILE_CHOICES: tuple[str, ...] = (
    LAUNCHER_MODE_PROFILE_DEV_SERVICE,
    LAUNCHER_MODE_PROFILE_EMBEDDED,
)
DEFAULT_LAUNCHER_MODE_PROFILE = LAUNCHER_MODE_PROFILE_DEV_SERVICE
DEFAULT_MANUAL_API_ENV_TEXT = "TORCH_CUDA_ARCH_LIST=8.6\nMAX_JOBS=4"
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _default_external_terminal_enabled() -> bool:
    return os.name == "nt"


def normalize_mode_profile(raw_value: str, *, source: str) -> str:
    normalized = str(raw_value or "").strip().lower()
    if normalized in LAUNCHER_MODE_PROFILE_CHOICES:
        return normalized
    allowed = ", ".join(LAUNCHER_MODE_PROFILE_CHOICES)
    raise ValueError(f"Invalid {source}: {raw_value!r}. Allowed values: {allowed}.")


def parse_manual_api_env_text(raw_text: str) -> Dict[str, str]:
    overlay: Dict[str, str] = {}
    seen_keys: Dict[str, tuple[str, int]] = {}
    normalized_text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    for line_number, raw_line in enumerate(normalized_text.split("\n"), start=1):
        stripped = str(raw_line).strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(
                "Invalid manual API env vars line "
                f"{line_number}: expected 'KEY=VALUE', got {raw_line!r}."
            )
        key_raw, value_raw = stripped.split("=", 1)
        key = str(key_raw).strip()
        value = str(value_raw).strip()
        if not key:
            raise ValueError(
                "Invalid manual API env vars line "
                f"{line_number}: empty env key in {raw_line!r}."
            )
        if not _ENV_KEY_RE.match(key):
            raise ValueError(
                "Invalid manual API env vars line "
                f"{line_number}: invalid env key {key!r}. "
                "Expected pattern [A-Za-z_][A-Za-z0-9_]*."
            )
        dedupe_key = key.lower()
        if dedupe_key in seen_keys:
            first_key, first_line = seen_keys[dedupe_key]
            raise ValueError(
                "Invalid manual API env vars: duplicate key "
                f"{key!r} at line {line_number} (already defined as {first_key!r} at line {first_line})."
            )
        seen_keys[dedupe_key] = (key, line_number)
        overlay[key] = value
    return overlay


@dataclass
class LauncherMeta:
    external_terminal: bool = field(default_factory=_default_external_terminal_enabled)
    sdpa_policy: str = "auto"
    tab_index: int = 0
    active_model: str = DEFAULT_MODEL_NAME
    window_geometry: str = ""
    show_advanced_controls: bool = False
    app_mode_profile: str = DEFAULT_LAUNCHER_MODE_PROFILE
    frontend_dev_typecheck: bool = False
    manual_api_env_enabled: bool = False
    manual_api_env_text: str = DEFAULT_MANUAL_API_ENV_TEXT


def load_launcher_meta(root: Path) -> LauncherMeta:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        meta = LauncherMeta()
        write_launcher_meta(root, meta)
        return meta
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read launcher meta {meta_path}: {exc}") from exc
    external_terminal_default = _default_external_terminal_enabled()
    raw_mode_profile = str(data.get("app_mode_profile", DEFAULT_LAUNCHER_MODE_PROFILE) or DEFAULT_LAUNCHER_MODE_PROFILE)
    try:
        app_mode_profile = normalize_mode_profile(
            raw_mode_profile,
            source=f"{meta_path} app_mode_profile",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return LauncherMeta(
        external_terminal=bool(data["external_terminal"]) if "external_terminal" in data else external_terminal_default,
        sdpa_policy=str(data.get("sdpa_policy", "auto")),
        tab_index=int(data.get("tab_index", 0)),
        active_model=str(data.get("active_model", DEFAULT_MODEL_NAME)),
        window_geometry=str(data.get("window_geometry", "") or ""),
        show_advanced_controls=bool(data.get("show_advanced_controls", False)),
        app_mode_profile=app_mode_profile,
        frontend_dev_typecheck=bool(data.get("frontend_dev_typecheck", False)),
        manual_api_env_enabled=bool(data.get("manual_api_env_enabled", False)),
        manual_api_env_text=str(data.get("manual_api_env_text", DEFAULT_MANUAL_API_ENV_TEXT) or ""),
    )


def write_launcher_meta(root: Path, meta: LauncherMeta) -> None:
    meta_path = root / "meta.json"
    payload = {
        "external_terminal": meta.external_terminal,
        "sdpa_policy": meta.sdpa_policy,
        "tab_index": meta.tab_index,
        "active_model": meta.active_model,
        "show_advanced_controls": bool(getattr(meta, "show_advanced_controls", False)),
        "app_mode_profile": normalize_mode_profile(
            str(getattr(meta, "app_mode_profile", DEFAULT_LAUNCHER_MODE_PROFILE) or DEFAULT_LAUNCHER_MODE_PROFILE),
            source="launcher meta app_mode_profile",
        ),
        "frontend_dev_typecheck": bool(getattr(meta, "frontend_dev_typecheck", False)),
        "manual_api_env_enabled": bool(getattr(meta, "manual_api_env_enabled", False)),
        "manual_api_env_text": str(getattr(meta, "manual_api_env_text", DEFAULT_MANUAL_API_ENV_TEXT) or ""),
    }
    window_geometry = str(getattr(meta, "window_geometry", "") or "").strip()
    if window_geometry:
        payload["window_geometry"] = window_geometry
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
