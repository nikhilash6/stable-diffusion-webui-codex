"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: JSON-backed options store for backend and launchers.
Provides a small, typed facade over `apps/settings_values.json` so the API, runtime helpers, and launcher profile defaults can share a single
source of truth without importing legacy/compat shims. Includes one runtime main-device option plus per-component storage/compute dtype override keys.

Symbols (top-level; keep in sync; no ghosts):
- `SETTINGS_PATH` (constant): Absolute path to `apps/settings_values.json` under the repo root.
- `OPTIONS_REVISION_KEY` (constant): Internal settings revision key persisted in `settings_values.json`.
- `_SETTINGS_LOCK_PATH` (constant): Sidecar lock file path used to serialize settings updates across processes.
- `_coerce_revision` (function): Normalizes revision values into a non-negative integer.
- `_coerce_bool` (function): Strict bool parser for persisted option values (fail-loud on invalid literals/types).
- `_exclusive_settings_lock` (function): Context manager that acquires process/thread update lock for read-modify-write operations.
- `_atomic_write_values` (function): Writes settings JSON via temp-file + fsync + replace.
- `load_values` (function): Reads the settings JSON from disk and returns a dict.
- `save_values` (function): Writes the settings JSON to disk (atomic overwrite).
- `get_revision` (function): Returns the normalized persisted options revision (non-negative int).
- `get_value` (function): Reads a single option value with a fallback default.
- `set_values` (function): Persists option updates, bumps `OPTIONS_REVISION_KEY`, and returns updated keys.
- `OptionsRevisionMismatchError` (class): Raised when a conditional options write sees a different current revision under the store lock.
- `set_values_if_revision` (function): Locked compare-and-set options write that fails loud when `expected_revision` is stale.
- `OptionsSnapshot` (class): Typed snapshot of option values used by runtime/engines/launchers.
- `get_snapshot` (function): Builds an `OptionsSnapshot` from persisted values.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from apps.backend.infra.config.repo_root import get_repo_root

SETTINGS_PATH = str(get_repo_root() / "apps" / "settings_values.json")
OPTIONS_REVISION_KEY = "codex_options_revision"
_SETTINGS_LOCK_PATH = f"{SETTINGS_PATH}.lock"
_SETTINGS_UPDATE_LOCK = threading.RLock()

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None


def _coerce_revision(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        revision = int(value)  # type: ignore[arg-type]
    except Exception:
        return 0
    return max(0, revision)


def _coerce_bool(value: object, *, key: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise RuntimeError(
        f"Invalid boolean setting '{key}': expected bool or one of "
        f"('true','false','1','0','yes','no','on','off'), got {value!r}."
    )


def load_values() -> Dict[str, Any]:
    if not os.path.exists(SETTINGS_PATH):
        return {}
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid settings file (expected object): {SETTINGS_PATH}")
    return data


@contextlib.contextmanager
def _exclusive_settings_lock():
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    lock_fd = os.open(_SETTINGS_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with _SETTINGS_UPDATE_LOCK:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _atomic_write_values(values: Mapping[str, Any]) -> None:
    directory = os.path.dirname(SETTINGS_PATH)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="settings_values.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(values), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, SETTINGS_PATH)
        try:
            directory_fd = os.open(directory, os.O_DIRECTORY)
        except Exception:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def save_values(values: Mapping[str, Any]) -> None:
    with _exclusive_settings_lock():
        _atomic_write_values(values)


def get_revision(values: Mapping[str, Any] | None = None) -> int:
    source = values if values is not None else load_values()
    return _coerce_revision(source.get(OPTIONS_REVISION_KEY))


def get_value(key: str, default: Any = None) -> Any:
    return load_values().get(key, default)


def set_values(payload: Mapping[str, Any]) -> list[str]:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    with _exclusive_settings_lock():
        data = load_values()
        updated: list[str] = []
        for k, v in payload.items():
            key = str(k)
            data[key] = v
            updated.append(key)
        if updated:
            data[OPTIONS_REVISION_KEY] = get_revision(data) + 1
            if OPTIONS_REVISION_KEY not in updated:
                updated.append(OPTIONS_REVISION_KEY)
        _atomic_write_values(data)
        return updated


class OptionsRevisionMismatchError(RuntimeError):
    def __init__(self, *, expected_revision: int, current_revision: int):
        super().__init__(
            f"stale options write rejected: expected revision {expected_revision}, current revision is {current_revision}."
        )
        self.expected_revision = max(0, int(expected_revision))
        self.current_revision = max(0, int(current_revision))


def set_values_if_revision(payload: Mapping[str, Any], expected_revision: int | None = None) -> tuple[list[str], int]:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    with _exclusive_settings_lock():
        data = load_values()
        current_revision = get_revision(data)
        if expected_revision is not None and current_revision != max(0, int(expected_revision)):
            raise OptionsRevisionMismatchError(
                expected_revision=max(0, int(expected_revision)),
                current_revision=current_revision,
            )
        updated: list[str] = []
        for k, v in payload.items():
            key = str(k)
            data[key] = v
            updated.append(key)
        if updated:
            next_revision = current_revision + 1
            data[OPTIONS_REVISION_KEY] = next_revision
            if OPTIONS_REVISION_KEY not in updated:
                updated.append(OPTIONS_REVISION_KEY)
            _atomic_write_values(data)
            return updated, next_revision
        return updated, current_revision


@dataclass
class OptionsSnapshot:
    codex_options_revision: int = 0
    codex_export_video: bool = False
    codex_main_device: str = "auto"
    codex_core_dtype: str = "auto"
    codex_core_compute_dtype: str = "auto"
    codex_te_dtype: str = "auto"
    codex_te_compute_dtype: str = "auto"
    codex_vae_dtype: str = "auto"
    codex_vae_compute_dtype: str = "auto"
    codex_smart_offload: bool = False
    codex_smart_fallback: bool = False
    codex_smart_cache: bool = True
    codex_core_streaming: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            OPTIONS_REVISION_KEY: self.codex_options_revision,
            "codex_export_video": self.codex_export_video,
            "codex_main_device": self.codex_main_device,
            "codex_core_dtype": self.codex_core_dtype,
            "codex_core_compute_dtype": self.codex_core_compute_dtype,
            "codex_te_dtype": self.codex_te_dtype,
            "codex_te_compute_dtype": self.codex_te_compute_dtype,
            "codex_vae_dtype": self.codex_vae_dtype,
            "codex_vae_compute_dtype": self.codex_vae_compute_dtype,
            "codex_smart_offload": self.codex_smart_offload,
            "codex_smart_fallback": self.codex_smart_fallback,
            "codex_smart_cache": self.codex_smart_cache,
            "codex_core_streaming": self.codex_core_streaming,
        }


def get_snapshot() -> OptionsSnapshot:
    v = load_values()

    def _str_value(key: str, default: str) -> str:
        raw = v.get(key)
        if raw is None:
            return default
        text = str(raw).strip()
        return text or default

    return OptionsSnapshot(
        codex_options_revision=get_revision(v),
        codex_export_video=_coerce_bool(v.get("codex_export_video"), key="codex_export_video", default=False),
        codex_main_device=_str_value("codex_main_device", "auto"),
        codex_core_dtype=_str_value("codex_core_dtype", "auto"),
        codex_core_compute_dtype=_str_value("codex_core_compute_dtype", "auto"),
        codex_te_dtype=_str_value("codex_te_dtype", "auto"),
        codex_te_compute_dtype=_str_value("codex_te_compute_dtype", "auto"),
        codex_vae_dtype=_str_value("codex_vae_dtype", "auto"),
        codex_vae_compute_dtype=_str_value("codex_vae_compute_dtype", "auto"),
        codex_smart_offload=_coerce_bool(v.get("codex_smart_offload"), key="codex_smart_offload", default=False),
        codex_smart_fallback=_coerce_bool(v.get("codex_smart_fallback"), key="codex_smart_fallback", default=False),
        codex_smart_cache=_coerce_bool(v.get("codex_smart_cache"), key="codex_smart_cache", default=True),
        codex_core_streaming=_coerce_bool(v.get("codex_core_streaming"), key="codex_core_streaming", default=False),
    )


__all__ = [
    "SETTINGS_PATH",
    "OPTIONS_REVISION_KEY",
    "OptionsSnapshot",
    "get_revision",
    "get_snapshot",
    "get_value",
    "load_values",
    "save_values",
    "set_values",
]
