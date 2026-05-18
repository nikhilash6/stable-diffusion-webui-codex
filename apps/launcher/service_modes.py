"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher app-mode profile resolution and preflight checks.
Owns dev-service vs embedded mode metadata and validates frontend runtime prerequisites before launcher-managed services start.

Symbols (top-level; keep in sync; no ghosts):
- `LauncherModeProfile` (dataclass): Static metadata for one launcher app mode profile.
- `resolve_requested_mode_profile` (function): Resolves app mode from env, preferred meta value, or default.
- `assert_mode_preflight` (function): Validates mode-specific frontend packaging/service prerequisites.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from apps.launcher.profile_meta import (
    CODEX_APP_MODE_PROFILE_ENV_KEY,
    DEFAULT_LAUNCHER_MODE_PROFILE,
    LAUNCHER_MODE_PROFILE_CHOICES,
    LAUNCHER_MODE_PROFILE_DEV_SERVICE,
    LAUNCHER_MODE_PROFILE_EMBEDDED,
    normalize_mode_profile,
)


@dataclass(frozen=True, slots=True)
class LauncherModeProfile:
    key: str
    api_frontend_mode: str
    requires_ui_service: bool
    description: str


_MODE_PROFILE_MAP: dict[str, LauncherModeProfile] = {
    LAUNCHER_MODE_PROFILE_DEV_SERVICE: LauncherModeProfile(
        key=LAUNCHER_MODE_PROFILE_DEV_SERVICE,
        api_frontend_mode=LAUNCHER_MODE_PROFILE_DEV_SERVICE,
        requires_ui_service=True,
        description="Dual-process mode (API + Vite UI dev service).",
    ),
    LAUNCHER_MODE_PROFILE_EMBEDDED: LauncherModeProfile(
        key=LAUNCHER_MODE_PROFILE_EMBEDDED,
        api_frontend_mode=LAUNCHER_MODE_PROFILE_EMBEDDED,
        requires_ui_service=False,
        description="Embedded mode (API serves built SPA from apps/interface/dist).",
    ),
}


def resolve_requested_mode_profile(
    *,
    preferred_profile: str | None,
    env: Mapping[str, str],
) -> LauncherModeProfile:
    env_value = str(env.get(CODEX_APP_MODE_PROFILE_ENV_KEY, "") or "").strip()
    if env_value:
        return _resolve_mode_profile(
            env_value,
            source_label=f"env {CODEX_APP_MODE_PROFILE_ENV_KEY}",
        )
    if preferred_profile is not None and str(preferred_profile).strip():
        return _resolve_mode_profile(
            str(preferred_profile),
            source_label="launcher meta app_mode_profile",
        )
    return _resolve_mode_profile(
        DEFAULT_LAUNCHER_MODE_PROFILE,
        source_label="default launcher app mode profile",
    )


def assert_mode_preflight(profile: LauncherModeProfile, root: Path) -> None:
    if profile.key == LAUNCHER_MODE_PROFILE_EMBEDDED:
        _assert_embedded_packaging_contract(root)
        return
    if profile.key == LAUNCHER_MODE_PROFILE_DEV_SERVICE:
        _assert_dev_service_requirements(root)
        return
    raise RuntimeError(f"Unhandled launcher mode profile {profile.key!r}.")


def _resolve_mode_profile(raw_value: str, *, source_label: str) -> LauncherModeProfile:
    normalized = normalize_mode_profile(raw_value, source=source_label)
    profile = _MODE_PROFILE_MAP.get(normalized)
    if profile is None:
        allowed = ", ".join(LAUNCHER_MODE_PROFILE_CHOICES)
        raise RuntimeError(f"Unsupported launcher mode profile {normalized!r}. Allowed values: {allowed}.")
    return profile


def _assert_embedded_packaging_contract(root: Path) -> None:
    dist_dir = root / "apps" / "interface" / "dist"
    index_path = dist_dir / "index.html"
    assets_dir = dist_dir / "assets"
    if not dist_dir.is_dir():
        raise RuntimeError(
            "Embedded app mode requires a built frontend package at "
            f"{dist_dir}. Run 'npm run build' in apps/interface."
        )
    if not index_path.is_file():
        raise RuntimeError(
            "Embedded app mode packaging contract violation: missing "
            f"{index_path}."
        )
    if not assets_dir.is_dir():
        raise RuntimeError(
            "Embedded app mode packaging contract violation: missing assets directory "
            f"{assets_dir}."
        )
    js_assets = sorted(assets_dir.glob("*.js"))
    css_assets = sorted(assets_dir.glob("*.css"))
    if not js_assets:
        raise RuntimeError(
            "Embedded app mode packaging contract violation: no JavaScript bundle found under "
            f"{assets_dir}."
        )
    if not css_assets:
        raise RuntimeError(
            "Embedded app mode packaging contract violation: no CSS bundle found under "
            f"{assets_dir}."
        )

    try:
        index_html = index_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(
            f"Embedded app mode packaging contract violation: failed reading {index_path}: {exc}"
        ) from exc

    missing_refs: list[str] = []
    for asset in (js_assets[0].name, css_assets[0].name):
        if f"assets/{asset}" not in index_html:
            missing_refs.append(asset)
    if missing_refs:
        raise RuntimeError(
            "Embedded app mode packaging contract violation: index.html does not reference "
            f"asset(s): {', '.join(missing_refs)}."
        )


def _assert_dev_service_requirements(root: Path) -> None:
    interface_dir = root / "apps" / "interface"
    package_json = interface_dir / "package.json"
    node_modules = interface_dir / "node_modules"
    vite_pkg = node_modules / "vite" / "package.json"
    if not interface_dir.is_dir():
        raise RuntimeError(
            "Dev-service app mode requires frontend workspace at "
            f"{interface_dir}."
        )
    if not package_json.is_file():
        raise RuntimeError(
            "Dev-service app mode requires frontend package manifest: "
            f"{package_json}."
        )
    if not node_modules.is_dir():
        raise RuntimeError(
            "Dev-service app mode requires installed frontend dependencies: "
            f"{node_modules}. Run 'npm install' in apps/interface."
        )
    if not vite_pkg.is_file():
        raise RuntimeError(
            "Dev-service app mode requires Vite dependency at "
            f"{vite_pkg}. Run 'npm install' in apps/interface."
        )
