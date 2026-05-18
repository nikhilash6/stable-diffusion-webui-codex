"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher profile env defaults, storage paths, and env-map JSON IO helpers.
Builds default `areas/core` env maps from the launcher setting registry and owns low-level read/write helpers for `.sangoi/launcher/areas` and `.sangoi/launcher/models`.

Symbols (top-level; keep in sync; no ghosts):
- `ENV_PREFIX_AREAS` (constant): Env-key prefix routing table for launcher env areas.
- `BOOTSTRAP_DEVICE_KEYS` (constant): Runtime-global launcher device keys scoped to `areas/core`.
- `default_area_env` (function): Builds default per-area env maps from the setting registry.
- `default_launcher_root` (function): Returns the default launcher storage root.
- `ensure_launcher_tree` (function): Ensures launcher storage directories exist.
- `load_launcher_areas` (function): Loads area env JSON maps from disk.
- `load_launcher_models` (function): Loads model env JSON maps from disk.
- `write_env_maps` (function): Writes env maps to disk and prunes stale JSON files.
- `resolve_container_static` (function): Resolves an env key to an area/model mapping during legacy migration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from apps.backend.infra.config.repo_root import get_repo_root
from apps.launcher.profile_meta import DEFAULT_MODEL_NAME
from apps.launcher.setting_registry import launcher_default_core_env

ENV_PREFIX_AREAS: Dict[str, str] = {
    "CODEX_": "core",
}

BOOTSTRAP_DEVICE_KEYS: frozenset[str] = frozenset(
    {
        "CODEX_MAIN_DEVICE",
        "CODEX_MOUNT_DEVICE",
        "CODEX_OFFLOAD_DEVICE",
        "CODEX_CORE_DEVICE",
        "CODEX_TE_DEVICE",
        "CODEX_VAE_DEVICE",
    }
)


def default_area_env() -> Dict[str, Dict[str, str]]:
    """Compute default environment values partitioned by area."""
    return {"core": launcher_default_core_env()}


def default_launcher_root() -> Path:
    return get_repo_root() / ".sangoi" / "launcher"


def ensure_launcher_tree(root: Path) -> None:
    (root / "areas").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)


def load_launcher_areas(root: Path) -> Dict[str, Dict[str, str]]:
    defaults = default_area_env()
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


def load_launcher_models(root: Path) -> Dict[str, Dict[str, str]]:
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


def write_env_maps(base_dir: Path, mapping: Dict[str, Dict[str, str]]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    expected_files = {f"{name}.json" for name in mapping}
    for path in base_dir.glob("*.json"):
        if path.name in expected_files:
            continue
        path.unlink(missing_ok=True)
    for name, env in mapping.items():
        out_path = base_dir / f"{name}.json"
        out_path.write_text(json.dumps(_stringify_dict(env), indent=2, sort_keys=True))


def resolve_container_static(
    key: str,
    areas: Dict[str, Dict[str, str]],
    models: Dict[str, Dict[str, str]],
    active_model: str,
) -> Tuple[Dict[str, str], str]:
    for prefix, area in ENV_PREFIX_AREAS.items():
        if key.startswith(prefix):
            return areas.setdefault(area, {}), f"area:{area}"
    return models.setdefault(active_model, {}), f"model:{active_model}"


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
    return {str(key): str(value) for key, value in data.items()}
