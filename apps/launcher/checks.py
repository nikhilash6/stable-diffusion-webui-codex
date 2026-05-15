"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher environment validation helpers.
Runs mode-aware preflight checks and returns structured results for UI display and diagnostics
(dev-service mode validates Node/npm/Vite; embedded mode validates built dist packaging contract).

Symbols (top-level; keep in sync; no ghosts):
- `LOGGER` (constant): Module logger for launcher checks.
- `MIN_NODE_MAJOR` (constant): Minimum supported Node.js major version.
- `CodexLaunchCheck` (dataclass): Structured check result (name/ok/detail).
- `_parse_semver` (function): Parses a semver string into integer tuples for comparison.
- `_expected_python_version` (function): Reads the repo-pinned Python version from `.python-version` when available.
- `_check_python_version` (function): Validates the running Python version against supported majors/minors.
- `_check_node` (function): Validates node/npm availability and minimum version.
- `_vite_requirement_satisfied` (function): Checks if an installed Vite version satisfies a package.json requirement string.
- `_codex_root` (function): Resolves the repo root used for frontend checks.
- `_check_vite` (function): Validates installed Vite version vs `apps/interface/package.json` requirement.
- `_resolve_mode_profile_for_checks` (function): Resolves launcher app mode profile for preflight execution.
- `_check_embedded_dist_contract` (function): Validates embedded-mode SPA packaging contract.
- `run_launch_checks` (function): Executes mode-aware launcher checks and returns results.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from apps.backend.infra.config.repo_root import get_repo_root
from apps.launcher.profiles import (
    CODEX_APP_MODE_PROFILE_ENV_KEY,
    DEFAULT_LAUNCHER_MODE_PROFILE,
    LAUNCHER_MODE_PROFILE_DEV_SERVICE,
    LAUNCHER_MODE_PROFILE_EMBEDDED,
    normalize_mode_profile,
)

LOGGER = logging.getLogger("codex.launcher.checks")
MIN_NODE_MAJOR = 18


@dataclass(frozen=True)
class CodexLaunchCheck:
    name: str
    ok: bool
    detail: str


def _parse_semver(version: str, components: int = 3) -> tuple[int, ...]:
    parts: list[int] = []
    for part in version.split(".")[:components]:
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits or 0))
    while len(parts) < components:
        parts.append(0)
    return tuple(parts)


def _expected_python_version(codex_root: Path) -> str | None:
    candidate = codex_root / ".python-version"
    if not candidate.exists():
        return None
    try:
        raw = candidate.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return raw or None


def _check_python_version() -> CodexLaunchCheck:
    major, minor, micro = sys.version_info[:3]
    codex_root = _codex_root()
    expected = _expected_python_version(codex_root)
    if expected is None:
        supported = (major == 3) and (minor == 12)
        detail = f"Detected Python {major}.{minor}.{micro}"
        if not supported:
            detail += " (expected 3.12.x)"
        return CodexLaunchCheck(name="python-version", ok=supported, detail=detail)

    expected_tuple = _parse_semver(expected, components=3)
    supported = (major, minor, micro) == expected_tuple
    detail = f"Detected Python {major}.{minor}.{micro}"
    if not supported:
        detail += f" (expected {expected})"
    return CodexLaunchCheck(name="python-version", ok=supported, detail=detail)


def _check_node() -> CodexLaunchCheck:
    npm = shutil.which("npm")
    node = shutil.which("node")
    if not (node and npm):
        missing = ", ".join(x for x, ref in (("node", node), ("npm", npm)) if ref is None)
        detail = f"Missing tool(s): {missing or 'unknown'}"
        return CodexLaunchCheck(name="node/npm", ok=False, detail=detail)

    try:
        raw_node_version = subprocess.check_output([node, "--version"], text=True, stderr=subprocess.STDOUT).strip()
        node_version = raw_node_version.lstrip("v")
    except Exception as exc:
        return CodexLaunchCheck(
            name="node/npm",
            ok=False,
            detail=f"node detected at {node} but version check failed: {exc}",
        )

    node_major = _parse_semver(node_version, components=1)[0]
    try:
        npm_version = subprocess.check_output([npm, "--version"], text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        npm_version = None

    ok = (node_major >= MIN_NODE_MAJOR) and (npm_version is not None)
    detail_bits = [
        f"node {node_version} (path={node}, requires >= {MIN_NODE_MAJOR})",
        f"npm {npm_version or 'unavailable'} (path={npm})",
    ]
    if node_major < MIN_NODE_MAJOR:
        detail_bits.append("upgrade Node.js to >=18.x")
    if npm_version is None:
        detail_bits.append("npm --version command failed")

    return CodexLaunchCheck(name="node/npm", ok=ok, detail="; ".join(detail_bits))


def _vite_requirement_satisfied(actual: str, requirement: str) -> bool:
    if not requirement:
        return True
    req = requirement.strip()
    if "||" in req:
        return any(_vite_requirement_satisfied(actual, part) for part in req.split("||"))
    if req.startswith("^"):
        target_major = _parse_semver(req[1:], components=1)[0]
        return _parse_semver(actual, components=1)[0] == target_major
    if req.startswith("~"):
        target = _parse_semver(req[1:], components=2)
        return _parse_semver(actual, components=2) == target
    if req.startswith(">="):
        target = _parse_semver(req[2:], components=3)
        return _parse_semver(actual, components=3) >= target
    return actual.startswith(req)


def _codex_root() -> Path:
    return get_repo_root()


def _check_vite(codex_root: Path) -> CodexLaunchCheck:
    interface_dir = codex_root / "apps" / "interface"
    package_json = interface_dir / "package.json"
    if not package_json.exists():
        return CodexLaunchCheck(
            name="vite",
            ok=False,
            detail="apps/interface/package.json not found; frontend workspace missing.",
        )
    try:
        package_spec = json.loads(package_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return CodexLaunchCheck(name="vite", ok=False, detail=f"Failed to read package.json: {exc}")

    dev_deps = package_spec.get("devDependencies") or {}
    requirement = str(dev_deps.get("vite", "")).strip()
    if not requirement:
        return CodexLaunchCheck(
            name="vite",
            ok=False,
            detail="vite is not listed under devDependencies; ensure frontend deps are declared.",
        )

    installed_pkg = interface_dir / "node_modules" / "vite" / "package.json"
    if not installed_pkg.exists():
        return CodexLaunchCheck(
            name="vite",
            ok=False,
            detail="vite not installed (apps/interface/node_modules/vite missing). Run 'npm install' in apps/interface.",
        )
    try:
        installed_spec = json.loads(installed_pkg.read_text(encoding="utf-8"))
        installed_version = str(installed_spec.get("version", "")).strip()
    except Exception as exc:
        return CodexLaunchCheck(name="vite", ok=False, detail=f"Failed to read installed Vite package: {exc}")

    if not installed_version:
        return CodexLaunchCheck(
            name="vite",
            ok=False,
            detail="Installed Vite package has no version field; reinstall dev dependencies.",
        )

    ok = _vite_requirement_satisfied(installed_version, requirement)
    detail = f"vite {installed_version} (expected {requirement})"
    if not ok:
        detail += " — reinstall dependencies (npm install in apps/interface)."
    return CodexLaunchCheck(name="vite", ok=ok, detail=detail)


def _resolve_mode_profile_for_checks(mode_profile: str | None) -> str:
    env_override = str(os.getenv(CODEX_APP_MODE_PROFILE_ENV_KEY, "") or "").strip()
    if env_override:
        return normalize_mode_profile(env_override, source=f"env {CODEX_APP_MODE_PROFILE_ENV_KEY}")
    if mode_profile is not None and str(mode_profile).strip():
        return normalize_mode_profile(
            str(mode_profile),
            source="launcher meta app_mode_profile",
        )
    return normalize_mode_profile(
        DEFAULT_LAUNCHER_MODE_PROFILE,
        source="default launcher app mode profile",
    )


def _check_embedded_dist_contract(codex_root: Path) -> CodexLaunchCheck:
    dist_dir = codex_root / "apps" / "interface" / "dist"
    index_path = dist_dir / "index.html"
    assets_dir = dist_dir / "assets"
    if not dist_dir.is_dir():
        return CodexLaunchCheck(
            name="embedded-dist",
            ok=False,
            detail=(
                f"Embedded mode requires {dist_dir}. "
                "Run 'npm run build' in apps/interface."
            ),
        )
    if not index_path.is_file():
        return CodexLaunchCheck(
            name="embedded-dist",
            ok=False,
            detail=f"Embedded mode packaging contract violation: missing {index_path}.",
        )
    if not assets_dir.is_dir():
        return CodexLaunchCheck(
            name="embedded-dist",
            ok=False,
            detail=f"Embedded mode packaging contract violation: missing {assets_dir}.",
        )
    js_assets = sorted(assets_dir.glob("*.js"))
    css_assets = sorted(assets_dir.glob("*.css"))
    if not js_assets:
        return CodexLaunchCheck(
            name="embedded-dist",
            ok=False,
            detail=f"Embedded mode packaging contract violation: no JS bundle in {assets_dir}.",
        )
    if not css_assets:
        return CodexLaunchCheck(
            name="embedded-dist",
            ok=False,
            detail=f"Embedded mode packaging contract violation: no CSS bundle in {assets_dir}.",
        )
    return CodexLaunchCheck(
        name="embedded-dist",
        ok=True,
        detail=(
            f"Found dist contract at {dist_dir} "
            f"(js={js_assets[0].name}, css={css_assets[0].name})."
        ),
    )


def run_launch_checks(*, mode_profile: str | None = None) -> List[CodexLaunchCheck]:
    """Execute all launcher environment checks."""
    root = _codex_root()
    resolved_mode = _resolve_mode_profile_for_checks(mode_profile)
    checks: list[CodexLaunchCheck] = [
        CodexLaunchCheck(name="app-mode-profile", ok=True, detail=f"{resolved_mode}"),
        _check_python_version(),
    ]
    if resolved_mode == LAUNCHER_MODE_PROFILE_DEV_SERVICE:
        checks.extend((_check_node(), _check_vite(root)))
    elif resolved_mode == LAUNCHER_MODE_PROFILE_EMBEDDED:
        checks.append(_check_embedded_dist_contract(root))
    else:
        checks.append(
            CodexLaunchCheck(
                name="app-mode-profile",
                ok=False,
                detail=f"Unhandled launcher app mode profile: {resolved_mode!r}",
            )
        )
    for check in checks:
        LOGGER.debug("Launch check %s ok=%s detail=%s", check.name, check.ok, check.detail)
    return checks
