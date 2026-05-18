"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher profile store and env-map mutation view.
Owns loading, saving, consistency cleanup, legacy migration, and mapping-like env edits for segmented launcher profiles under `.sangoi/launcher/`.

Symbols (top-level; keep in sync; no ghosts):
- `_EnvironmentView` (class): Mutable mapping that routes env reads/writes into launcher profile area/model maps.
- `LauncherProfileStore` (dataclass): Main profile store; loads/saves meta/env maps, resolves key routing, and builds runtime env snapshots.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Tuple

from apps.launcher.profile_defaults import (
    BOOTSTRAP_DEVICE_KEYS,
    ENV_PREFIX_AREAS,
    default_area_env,
    default_launcher_root,
    ensure_launcher_tree,
    load_launcher_areas,
    load_launcher_models,
    resolve_container_static,
    write_env_maps,
)
from apps.launcher.profile_meta import (
    CODEX_CUDA_MALLOC_KEY,
    DEFAULT_LAUNCHER_MODE_PROFILE,
    DEFAULT_MANUAL_API_ENV_TEXT,
    DEFAULT_MODEL_NAME,
    DEFAULT_PYTORCH_CUDA_ALLOC_CONF,
    ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
    LauncherMeta,
    _default_external_terminal_enabled,
    load_launcher_meta,
    normalize_mode_profile,
    parse_manual_api_env_text,
    write_launcher_meta,
)
from apps.launcher.settings import SettingValidationError, normalize_attention_env

LOGGER = logging.getLogger("codex.launcher.profile_store")


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
            raise KeyError(f"Unsupported allocator key {key!r}. Use 'PYTORCH_CUDA_ALLOC_CONF'.")
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
        root = root or default_launcher_root()
        ensure_launcher_tree(root)
        _maybe_migrate_legacy(root)
        meta = load_launcher_meta(root)
        areas = load_launcher_areas(root)
        models = load_launcher_models(root)
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
        ensure_launcher_tree(self.root)
        with self._lock:
            changed = self._ensure_consistency()
            if changed:
                LOGGER.warning("Launcher profile normalized at save; writing canonical env maps to %s", self.root)
            if bool(getattr(self.meta, "manual_api_env_enabled", False)):
                self.build_manual_api_env_overlay()
            write_launcher_meta(self.root, self.meta)
            write_env_maps(self.root / "areas", self.areas)
            write_env_maps(self.root / "models", self.models)

    def save_meta(self) -> None:
        """Persist launcher metadata (meta.json) without touching env maps."""
        LOGGER.debug("Persisting launcher meta to %s", self.root)
        ensure_launcher_tree(self.root)
        with self._lock:
            write_launcher_meta(self.root, self.meta)

    def _ensure_consistency(self) -> bool:
        changed = False
        normalized_mode_profile = normalize_mode_profile(
            str(getattr(self.meta, "app_mode_profile", DEFAULT_LAUNCHER_MODE_PROFILE) or DEFAULT_LAUNCHER_MODE_PROFILE),
            source="launcher meta app_mode_profile",
        )
        if getattr(self.meta, "app_mode_profile", None) != normalized_mode_profile:
            self.meta.app_mode_profile = normalized_mode_profile
            changed = True
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

        defaults = default_area_env()
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
            for key in BOOTSTRAP_DEVICE_KEYS:
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
            for key in BOOTSTRAP_DEVICE_KEYS:
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
        return changed


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
    legacy_env = {str(key): str(value) for key, value in (legacy.get("env") or {}).items()}
    external_terminal_default = _default_external_terminal_enabled()
    meta = LauncherMeta(
        external_terminal=(
            bool(legacy["external_terminal"]) if "external_terminal" in legacy else external_terminal_default
        ),
        sdpa_policy=str(legacy.get("sdpa_policy", "auto")),
        tab_index=int(legacy.get("tab_index", 0)),
        active_model=DEFAULT_MODEL_NAME,
    )

    defaults = default_area_env()
    areas = {area: dict(values) for area, values in defaults.items()}
    models = {DEFAULT_MODEL_NAME: {}}

    for key, value in legacy_env.items():
        container, _ = resolve_container_static(key, areas, models, meta.active_model)
        container[key] = value
    store = LauncherProfileStore(root=root, meta=meta, areas=areas, models=models)
    store._ensure_consistency()
    store.save()
    backup_path = legacy_path.with_suffix(".legacy-backup")
    legacy_path.rename(backup_path)
    LOGGER.info("Legacy profile migrated; backup saved to %s", backup_path)
