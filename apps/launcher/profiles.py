"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher profile persistence (meta + env areas + per-model env overlays).
Implements the profile store used by the TUI/GUI launchers to load/save settings under `.sangoi/launcher/` (meta/areas/models) and to
expose a mapping-like interface for editing environment variables with per-area routing and migrations.
Defines defaults for performance-related env keys (GGUF dequant-cache/LoRA knobs, CFG batching, profiling flags) and task/runtime safety knobs (single-flight,
task cancel mode, task SSE buffer caps, safeweights), plus attention/bootstrap device policy keys (`CODEX_MAIN_DEVICE`, `CODEX_MOUNT_DEVICE`, `CODEX_OFFLOAD_DEVICE`) with CPU offload default, so runs are reproducible.
LoRA apply-mode profile defaults resolve unset launcher config to `online` while preserving explicit `merge` values.
Also stores API-only manual env overlay settings (`manual_api_env_enabled`, `manual_api_env_text`) plus launcher-owned frontend dev boot policy, and
validates overlay text parsing for fail-loud startup.

Symbols (top-level; keep in sync; no ghosts):
- `_default_area_env` (function): Builds default per-area env maps (debug/log/profiling flags + device defaults + GGUF dequant-cache/LoRA runtime knobs; default offload target is CPU).
- `_default_external_terminal_enabled` (function): Canonical default provider for launcher "external terminal" preference (enabled on Windows, disabled elsewhere).
- `_BOOTSTRAP_DEVICE_KEYS` (constant): Runtime-global launcher device keys that must stay scoped to `areas/core` (never model/non-core overlays).
- `DEFAULT_PYTORCH_CUDA_ALLOC_CONF` (constant): Default `PYTORCH_CUDA_ALLOC_CONF` applied by launchers when unset.
- `ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY` (constant): Env key toggling default allocator config injection when `PYTORCH_CUDA_ALLOC_CONF` is unset.
- `CODEX_CUDA_MALLOC_KEY` (constant): Env key toggling backend `--cuda-malloc` forwarding in launcher-managed runs.
- `CODEX_APP_MODE_PROFILE_ENV_KEY` (constant): Env key selecting launcher app mode profile (`dev_service|embedded`).
- `LAUNCHER_MODE_PROFILE_CHOICES` (constant): Allowed launcher app mode profiles.
- `normalize_mode_profile` (function): Strict parser/validator for launcher app mode profile values.
- `DEFAULT_MANUAL_API_ENV_TEXT` (constant): Suggested manual API env overlay text prefilled in launcher metadata.
- `parse_manual_api_env_text` (function): Parses manual API env text (`KEY=VALUE` per line) with strict, line-numbered validation.
- `LauncherMeta` (dataclass): Persisted launcher UI metadata (active model, tab index, terminal preference, sdpa policy, manual API env overlay,
  frontend dev boot policy).
- `_EnvironmentView` (class): `MutableMapping` view that routes env reads/writes into the underlying profile store (areas/models).
- `LauncherProfileStore` (dataclass): Main profile store; loads/saves meta/env maps, resolves key routing, and provides lookup helpers
  (contains nested helpers for container resolution and file IO).
- `_default_root` (function): Returns the default launcher storage root under repo `.sangoi/launcher/`.
- `_ensure_tree` (function): Ensures launcher storage directories exist.
- `_load_meta` (function): Loads `LauncherMeta` from disk (or defaults).
- `_write_meta` (function): Writes `LauncherMeta` to disk.
- `_load_areas` (function): Loads area env files from disk.
- `_load_models` (function): Loads per-model env files from disk.
- `_write_env_maps` (function): Writes env maps to disk (one file per map).
- `_read_env_file` (function): Reads a single env JSON file with optional defaults merge.
- `_stringify_dict` (function): Normalizes mapping values to strings for env storage.
- `_maybe_migrate_legacy` (function): Migrates legacy launcher layouts/keys when present.
- `_resolve_container_static` (function): Static resolution helper for routing an env key to an area/model container.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Tuple

from apps.backend.infra.config.repo_root import get_repo_root
from apps.launcher.settings import SettingValidationError, normalize_attention_env

LOGGER = logging.getLogger("codex.launcher.profiles")

ENV_PREFIX_AREAS: Dict[str, str] = {
    "CODEX_": "core",
}

_BOOTSTRAP_DEVICE_KEYS: frozenset[str] = frozenset(
    {
        "CODEX_MAIN_DEVICE",
        "CODEX_MOUNT_DEVICE",
        "CODEX_OFFLOAD_DEVICE",
        "CODEX_CORE_DEVICE",
        "CODEX_TE_DEVICE",
        "CODEX_VAE_DEVICE",
    }
)


def _default_area_env() -> Dict[str, Dict[str, str]]:
    """Compute default environment values partitioned by area."""
    core = {
        "CODEX_PIPELINE_DEBUG": os.getenv("CODEX_PIPELINE_DEBUG", "0"),
        "CODEX_CFG_BATCH_MODE": os.getenv("CODEX_CFG_BATCH_MODE", "fused"),
        "CODEX_SINGLE_FLIGHT": os.getenv("CODEX_SINGLE_FLIGHT", "1"),
        "CODEX_TASK_EVENT_BUFFER_MAX_EVENTS": os.getenv("CODEX_TASK_EVENT_BUFFER_MAX_EVENTS", "5000"),
        "CODEX_TASK_EVENT_BUFFER_MAX_MB": os.getenv("CODEX_TASK_EVENT_BUFFER_MAX_MB", "64"),
        "CODEX_TASK_CANCEL_DEFAULT_MODE": os.getenv("CODEX_TASK_CANCEL_DEFAULT_MODE", "immediate"),
        "CODEX_SAFE_WEIGHTS": os.getenv("CODEX_SAFE_WEIGHTS", "0"),
        "CODEX_PROFILE": os.getenv("CODEX_PROFILE", "0"),
        "CODEX_PROFILE_TRACE": os.getenv("CODEX_PROFILE_TRACE", "1"),
        "CODEX_PROFILE_RECORD_SHAPES": os.getenv("CODEX_PROFILE_RECORD_SHAPES", "0"),
        "CODEX_PROFILE_PROFILE_MEMORY": os.getenv("CODEX_PROFILE_PROFILE_MEMORY", "1"),
        "CODEX_PROFILE_WITH_STACK": os.getenv("CODEX_PROFILE_WITH_STACK", "0"),
        "CODEX_PROFILE_TOP_N": os.getenv("CODEX_PROFILE_TOP_N", "25"),
        "CODEX_PROFILE_MAX_STEPS": os.getenv("CODEX_PROFILE_MAX_STEPS", "0"),
        "CODEX_TRACE_INFERENCE_DEBUG": os.getenv("CODEX_TRACE_INFERENCE_DEBUG", "0"),
        "CODEX_TRACE_LOAD_PATCH_DEBUG": os.getenv("CODEX_TRACE_LOAD_PATCH_DEBUG", "0"),
        "CODEX_TRACE_CALL_DEBUG": os.getenv("CODEX_TRACE_CALL_DEBUG", "0"),
        "CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC": os.getenv("CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC", "10"),
        "CODEX_MAIN_DEVICE": "auto",
        "CODEX_MOUNT_DEVICE": "auto",
        "CODEX_OFFLOAD_DEVICE": "cpu",
        "CODEX_CORE_DEVICE": "auto",
        "CODEX_TE_DEVICE": "auto",
        "CODEX_VAE_DEVICE": "auto",
        "CODEX_GGUF_DEQUANT_CACHE": "off",
        "CODEX_ATTENTION_BACKEND": "pytorch",
        "CODEX_ATTENTION_SDPA_POLICY": "auto",
        "CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE": os.getenv("CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE", "hybrid"),
        "CODEX_LORA_APPLY_MODE": "online",
        "CODEX_LORA_ONLINE_MATH": "weight_merge",
        ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY: "1",
        CODEX_CUDA_MALLOC_KEY: "0",
    }
    return {"core": core}


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
    raise ValueError(
        f"Invalid {source}: {raw_value!r}. Allowed values: {allowed}."
    )


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


class _EnvironmentView(MutableMapping[str, str]):
    """Mutable mapping that routes env mutations to areas/models."""

    def __init__(self, store: "LauncherProfileStore") -> None:
        self._store = store

    def __getitem__(self, key: str) -> str:
        with self._store._lock:
            value = self._store.lookup_env(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: str) -> None:
        value = str(value)
        if key.startswith("PYTORCH_") and key.endswith("_ALLOC_CONF") and key != "PYTORCH_CUDA_ALLOC_CONF":
            raise KeyError(
                f"Unsupported allocator key {key!r}. Use 'PYTORCH_CUDA_ALLOC_CONF'."
            )
        if (
            key.startswith("CODEX_ENABLE_DEFAULT_PYTORCH_")
            and key.endswith("_ALLOC_CONF")
            and key != ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY
        ):
            raise KeyError(
                "Unsupported allocator toggle key "
                f"{key!r}. Use {ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY!r}."
            )
        with self._store._lock:
            target_map, target_kind = self._store.resolve_container_for_key(key)
            target_map[key] = value
        LOGGER.debug("Set env %s=%s (container=%s)", key, value, target_kind)

    def __delitem__(self, key: str) -> None:
        removed = False
        with self._store._lock:
            for area, mapping in self._store.areas.items():
                if key in mapping:
                    del mapping[key]
                    removed = True
                    LOGGER.debug("Deleted env %s from area %s", key, area)
                    break
            if not removed:
                model_map = self._store.models.get(self._store.meta.active_model, {})
                if key in model_map:
                    del model_map[key]
                    removed = True
                    LOGGER.debug("Deleted env %s from model %s", key, self._store.meta.active_model)
        if not removed:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store.build_env())

    def __len__(self) -> int:
        return len(self._store.build_env())

    def get(self, key: str, default: str | None = None) -> str | None:  # type: ignore[override]
        try:
            return self.__getitem__(key)
        except KeyError:
            return default


@dataclass
class LauncherProfileStore:
    root: Path
    meta: LauncherMeta
    areas: Dict[str, Dict[str, str]] = field(default_factory=dict)
    models: Dict[str, Dict[str, str]] = field(default_factory=dict)
    _env_view: _EnvironmentView | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    @classmethod
    def load(cls, root: Path | None = None) -> "LauncherProfileStore":
        root = root or _default_root()
        _ensure_tree(root)
        _maybe_migrate_legacy(root)
        meta = _load_meta(root)
        areas = _load_areas(root)
        models = _load_models(root)
        store = cls(root=root, meta=meta, areas=areas, models=models)
        changed = store._ensure_consistency()
        if changed:
            LOGGER.warning("Launcher profile sanitized at load; persisting normalized env maps to %s", root)
            store.save()
        return store

    @property
    def env(self) -> _EnvironmentView:
        if self._env_view is None:
            self._env_view = _EnvironmentView(self)
        return self._env_view

    def build_env(self) -> Dict[str, str]:
        with self._lock:
            env: Dict[str, str] = {}
            for mapping in self.areas.values():
                env.update(mapping)
            active_model = self.meta.active_model
            model_overlay = self.models.get(active_model, {})
            for key, value in model_overlay.items():
                if key.startswith("CODEX_"):
                    # CODEX runtime/bootstrap knobs are area-scoped (core); never let model overlays override them.
                    continue
                env[key] = value
            raw_enabled = str(env.get(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY, "1") or "").strip().lower()
            default_alloc_enabled = raw_enabled in {"", "1", "true", "yes", "on"}
            if default_alloc_enabled and not str(env.get("PYTORCH_CUDA_ALLOC_CONF", "") or "").strip():
                env["PYTORCH_CUDA_ALLOC_CONF"] = DEFAULT_PYTORCH_CUDA_ALLOC_CONF
            return env

    def build_manual_api_env_overlay(self) -> Dict[str, str]:
        with self._lock:
            enabled = bool(getattr(self.meta, "manual_api_env_enabled", False))
            raw_text = str(getattr(self.meta, "manual_api_env_text", DEFAULT_MANUAL_API_ENV_TEXT) or "")
        if not enabled:
            return {}
        try:
            return parse_manual_api_env_text(raw_text)
        except Exception as exc:
            raise ValueError(f"Invalid Manual Env Vars configuration: {exc}") from exc

    def lookup_env(self, key: str) -> str | None:
        with self._lock:
            if key == "PYTORCH_CUDA_ALLOC_CONF":
                return self.build_env().get(key)
            for prefix, area in ENV_PREFIX_AREAS.items():
                if key.startswith(prefix):
                    return self.areas.get(area, {}).get(key)
            active_model = self.meta.active_model
            if key in self.models.get(active_model, {}):
                return self.models[active_model][key]
            # As a safeguard allow lookups in other stored maps
            for mapping in self.areas.values():
                if key in mapping:
                    return mapping[key]
            for mapping in self.models.values():
                if key in mapping:
                    return mapping[key]
            return None

    def resolve_container_for_key(self, key: str) -> Tuple[Dict[str, str], str]:
        with self._lock:
            for prefix, area in ENV_PREFIX_AREAS.items():
                if key.startswith(prefix):
                    mapping = self.areas.setdefault(area, {})
                    return mapping, f"area:{area}"
            model = self.meta.active_model
            mapping = self.models.setdefault(model, {})
            return mapping, f"model:{model}"

    def save(self) -> None:
        LOGGER.debug("Persisting launcher profile to %s", self.root)
        _ensure_tree(self.root)
        with self._lock:
            changed = self._ensure_consistency()
            if changed:
                LOGGER.warning("Launcher profile normalized at save; writing canonical env maps to %s", self.root)
            if bool(getattr(self.meta, "manual_api_env_enabled", False)):
                self.build_manual_api_env_overlay()
            _write_meta(self.root, self.meta)
            _write_env_maps(self.root / "areas", self.areas)
            _write_env_maps(self.root / "models", self.models)

    def save_meta(self) -> None:
        """Persist launcher metadata (meta.json) without touching env maps."""
        LOGGER.debug("Persisting launcher meta to %s", self.root)
        _ensure_tree(self.root)
        with self._lock:
            _write_meta(self.root, self.meta)

    # ------------------------------------------------------------------ internal

    def _ensure_consistency(self) -> bool:
        changed = False
        normalized_mode_profile = normalize_mode_profile(
            str(getattr(self.meta, "app_mode_profile", DEFAULT_LAUNCHER_MODE_PROFILE) or DEFAULT_LAUNCHER_MODE_PROFILE),
            source="launcher meta app_mode_profile",
        )
        if getattr(self.meta, "app_mode_profile", None) != normalized_mode_profile:
            self.meta.app_mode_profile = normalized_mode_profile
            changed = True
        # Legacy migration: when old profiles only have component device keys,
        # seed CODEX_MAIN_DEVICE from core device before defaults are merged.
        for container in list(self.areas.values()):
            raw_main = str(container.get("CODEX_MAIN_DEVICE", "") or "").strip().lower()
            if not raw_main:
                legacy_core = str(container.get("CODEX_CORE_DEVICE", "") or "").strip().lower()
                if legacy_core:
                    container["CODEX_MAIN_DEVICE"] = legacy_core
                    changed = True
                    raw_main = legacy_core
            if raw_main:
                if not str(container.get("CODEX_MOUNT_DEVICE", "") or "").strip().lower():
                    container["CODEX_MOUNT_DEVICE"] = raw_main
                    changed = True
                if not str(container.get("CODEX_OFFLOAD_DEVICE", "") or "").strip().lower():
                    container["CODEX_OFFLOAD_DEVICE"] = "cpu"
                    changed = True

        defaults = _default_area_env()
        for area, values in defaults.items():
            if area not in self.areas:
                changed = True
            current = self.areas.setdefault(area, {})
            for key, default in values.items():
                if key not in current:
                    current[key] = default
                    changed = True
        if self.meta.active_model not in self.models:
            self.models[self.meta.active_model] = {}
            changed = True

        # Drop legacy env knobs (WAN_* and CODEX_* runtime settings).
        # Keep bootstrap-critical device defaults (CODEX_*_DEVICE) in launcher env so
        # the API can start in non-interactive spawns without prompting/fallbacks.
        for container in list(self.areas.values()) + list(self.models.values()):
            for key in list(container.keys()):
                if key.startswith("PYTORCH_") and key.endswith("_ALLOC_CONF") and key != "PYTORCH_CUDA_ALLOC_CONF":
                    container.pop(key, None)
                    changed = True
                    LOGGER.warning(
                        "Dropped unsupported launcher allocator key from persisted profile: %s "
                        "(use PYTORCH_CUDA_ALLOC_CONF).",
                        key,
                    )
                    continue
                if (
                    key.startswith("CODEX_ENABLE_DEFAULT_PYTORCH_")
                    and key.endswith("_ALLOC_CONF")
                    and key != ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY
                ):
                    container.pop(key, None)
                    changed = True
                    LOGGER.warning(
                        "Dropped unsupported launcher allocator toggle key from persisted profile: %s "
                        "(use %s).",
                        key,
                        ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
                    )
                    continue
                if key.startswith("WAN_"):
                    container.pop(key, None)
                    changed = True
                if key in {
                    "CODEX_ATTN_CHUNK_SIZE",
                    "CODEX_GGUF_CACHE_POLICY",
                    "CODEX_GGUF_CACHE_LIMIT_MB",
                    "CODEX_DIFFUSION_DEVICE",
                    "CODEX_DIFFUSION_DTYPE",
                    "CODEX_TE_DTYPE",
                    "CODEX_VAE_DTYPE",
                    "CODEX_SWAP_POLICY",
                    "CODEX_SWAP_METHOD",
                    "CODEX_GPU_PREFER_CONSTRUCT",
                    "CODEX_SMART_OFFLOAD",
                    "CODEX_PIN_SHARED_MEMORY",
                }:
                    container.pop(key, None)
                    changed = True
        core_env = self.areas.setdefault("core", {})
        try:
            normalize_attention_env(core_env)
        except SettingValidationError as exc:
            raise RuntimeError(f"Invalid launcher attention setting in area 'core': {exc}") from exc
        for area_name, container in self.areas.items():
            if area_name == "core":
                continue
            removed_device_keys: list[str] = []
            for key in _BOOTSTRAP_DEVICE_KEYS:
                if key in container:
                    container.pop(key, None)
                    removed_device_keys.append(key)
                    changed = True
            if removed_device_keys:
                LOGGER.warning(
                    "Dropped non-core launcher device override(s) from area '%s': %s. "
                    "Main/mount/offload/core/TE/VAE device keys are runtime-global in area 'core'.",
                    area_name,
                    ", ".join(sorted(removed_device_keys)),
                )
            removed_keys: list[str] = []
            for key in ("CODEX_ATTENTION_BACKEND", "CODEX_ATTENTION_SDPA_POLICY"):
                if key in container:
                    container.pop(key, None)
                    removed_keys.append(key)
                    changed = True
            if removed_keys:
                LOGGER.warning(
                    "Dropped non-core launcher attention override(s) from area '%s': %s. "
                    "Attention backend is a launcher runtime-global setting in area 'core'.",
                    area_name,
                    ", ".join(removed_keys),
                )
            if ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY in container:
                moved_value = str(container.pop(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY, "") or "").strip()
                changed = True
                core_current = str(core_env.get(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY, "") or "").strip()
                if moved_value and not core_current:
                    core_env[ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY] = moved_value
                    changed = True
                    LOGGER.warning(
                        "Moved non-core launcher allocator toggle from area '%s' to area 'core': %s.",
                        area_name,
                        ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
                    )
                else:
                    LOGGER.warning(
                        "Dropped non-core launcher allocator toggle from area '%s': %s.",
                        area_name,
                        ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
                    )
        for model_name, container in self.models.items():
            removed_device_keys: list[str] = []
            for key in _BOOTSTRAP_DEVICE_KEYS:
                if key in container:
                    container.pop(key, None)
                    removed_device_keys.append(key)
                    changed = True
            if removed_device_keys:
                LOGGER.warning(
                    "Dropped model-scoped launcher device override(s) for model '%s': %s. "
                    "Main/mount/offload/core/TE/VAE device keys are runtime-global in area 'core'.",
                    model_name,
                    ", ".join(sorted(removed_device_keys)),
                )
            removed_keys: list[str] = []
            for key in ("CODEX_ATTENTION_BACKEND", "CODEX_ATTENTION_SDPA_POLICY"):
                if key in container:
                    container.pop(key, None)
                    removed_keys.append(key)
                    changed = True
            if removed_keys:
                LOGGER.warning(
                    "Dropped model-scoped launcher attention override(s) for model '%s': %s. "
                    "Attention backend is a launcher runtime-global setting.",
                    model_name,
                    ", ".join(removed_keys),
                )
            if ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY in container:
                moved_value = str(container.pop(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY, "") or "").strip()
                changed = True
                core_current = str(core_env.get(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY, "") or "").strip()
                if moved_value and not core_current:
                    core_env[ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY] = moved_value
                    changed = True
                    LOGGER.warning(
                        "Moved model-scoped launcher allocator toggle for model '%s' to area 'core': %s.",
                        model_name,
                        ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
                    )
                else:
                    LOGGER.warning(
                        "Dropped model-scoped launcher allocator toggle for model '%s': %s.",
                        model_name,
                        ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
                    )
        if self.areas.pop("wan", None) is not None:
            changed = True
        # Device/dtype bootstrap settings live in launcher env and are forwarded as API CLI flags.
        return changed

def _default_root() -> Path:
    return get_repo_root() / ".sangoi" / "launcher"


def _ensure_tree(root: Path) -> None:
    (root / "areas").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)


def _load_meta(root: Path) -> LauncherMeta:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        meta = LauncherMeta()
        _write_meta(root, meta)
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


def _write_meta(root: Path, meta: LauncherMeta) -> None:
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


def _load_areas(root: Path) -> Dict[str, Dict[str, str]]:
    defaults = _default_area_env()
    areas_dir = root / "areas"
    areas: Dict[str, Dict[str, str]] = {}
    for area, default_env in defaults.items():
        areas[area] = _read_env_file(areas_dir / f"{area}.json", default_env)
    for path in areas_dir.glob("*.json"):
        area = path.stem
        if area in areas:
            continue
        areas[area] = _read_env_file(path)
    return areas


def _load_models(root: Path) -> Dict[str, Dict[str, str]]:
    models_dir = root / "models"
    models: Dict[str, Dict[str, str]] = {}
    default_path = models_dir / f"{DEFAULT_MODEL_NAME}.json"
    models[DEFAULT_MODEL_NAME] = _read_env_file(default_path, {})
    for path in models_dir.glob("*.json"):
        model = path.stem
        if model in models:
            continue
        models[model] = _read_env_file(path)
    return models


def _write_env_maps(base_dir: Path, mapping: Dict[str, Dict[str, str]]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    expected_files = {f"{name}.json" for name in mapping}
    for path in base_dir.glob("*.json"):
        if path.name in expected_files:
            continue
        path.unlink(missing_ok=True)
    for name, env in mapping.items():
        out_path = base_dir / f"{name}.json"
        out_path.write_text(json.dumps(_stringify_dict(env), indent=2, sort_keys=True))


def _read_env_file(path: Path, defaults: Dict[str, str] | None = None) -> Dict[str, str]:
    if not path.exists():
        if defaults is None:
            defaults = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_stringify_dict(defaults), indent=2, sort_keys=True))
        return dict(defaults)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read launcher environment file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Launcher environment file {path} must contain a JSON object.")
    result: Dict[str, str] = {}
    for key, value in data.items():
        result[str(key)] = str(value)
    return result


def _stringify_dict(data: Dict[str, str]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in data.items()}


def _maybe_migrate_legacy(root: Path) -> None:
    meta_path = root / "meta.json"
    if meta_path.exists():
        return
    legacy_path = root.parent / "tui-profile.json"
    if not legacy_path.exists():
        return
    try:
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read legacy profile {legacy_path}: {exc}") from exc

    LOGGER.info("Migrating legacy launcher profile from %s", legacy_path)
    legacy_env = {str(k): str(v) for k, v in (legacy.get("env") or {}).items()}
    external_terminal_default = _default_external_terminal_enabled()
    meta = LauncherMeta(
        external_terminal=(
            bool(legacy["external_terminal"]) if "external_terminal" in legacy else external_terminal_default
        ),
        sdpa_policy=str(legacy.get("sdpa_policy", "auto")),
        tab_index=int(legacy.get("tab_index", 0)),
        active_model=DEFAULT_MODEL_NAME,
    )

    defaults = _default_area_env()
    areas = {area: dict(values) for area, values in defaults.items()}
    models = {DEFAULT_MODEL_NAME: {}}

    for key, value in legacy_env.items():
        container, _ = _resolve_container_static(key, areas, models, meta.active_model)
        container[key] = value
    # ensure consistency and write through
    store = LauncherProfileStore(root=root, meta=meta, areas=areas, models=models)
    store._ensure_consistency()
    store.save()
    backup_path = legacy_path.with_suffix(".legacy-backup")
    legacy_path.rename(backup_path)
    LOGGER.info("Legacy profile migrated; backup saved to %s", backup_path)


def _resolve_container_static(
    key: str,
    areas: Dict[str, Dict[str, str]],
    models: Dict[str, Dict[str, str]],
    active_model: str,
) -> Tuple[Dict[str, str], str]:
    for prefix, area in ENV_PREFIX_AREAS.items():
        if key.startswith(prefix):
            return areas.setdefault(area, {}), f"area:{area}"
    return models.setdefault(active_model, {}), f"model:{active_model}"
