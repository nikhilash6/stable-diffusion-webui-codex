"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher default API/UI service specifications.
Owns construction of launcher-managed service specs and app-mode-specific UI service inclusion; lifecycle stays in `service_process.py`.

Symbols (top-level; keep in sync; no ghosts):
- `build_ui_dev_service_command` (function): Builds the Vite dev-service command for launcher UI mode.
- `default_services` (function): Builds default launcher service handles for the selected app mode profile.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Dict, List

from apps.backend.infra.config.repo_root import get_repo_root
from apps.launcher.log_buffer import CodexLogBuffer
from apps.launcher.profile_meta import CODEX_APP_MODE_PROFILE_ENV_KEY
from apps.launcher.service_modes import assert_mode_preflight, resolve_requested_mode_profile
from apps.launcher.service_process import CodexServiceHandle, CodexServiceSpec


def build_ui_dev_service_command(*, frontend_dev_typecheck: bool) -> List[str]:
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    script_name = "dev:typecheck" if frontend_dev_typecheck else "dev:fast"
    return [npm_cmd, "run", script_name, "--", "--host"]


def default_services(
    log_buffer: CodexLogBuffer | None = None,
    *,
    mode_profile: str | None = None,
    frontend_dev_typecheck: bool = False,
) -> Dict[str, CodexServiceHandle]:
    root = _codex_root()
    py_exe = Path(sys.executable)
    api_port = os.getenv("API_PORT_OVERRIDE", "7850")
    web_port = os.getenv("WEB_PORT", "7860")
    resolved_mode = resolve_requested_mode_profile(
        preferred_profile=mode_profile,
        env=os.environ,
    )
    assert_mode_preflight(resolved_mode, root)
    api_spec = CodexServiceSpec(
        name="API",
        command=[
            str(py_exe),
            str(root / "apps" / "backend" / "interfaces" / "api" / "run_api.py"),
        ],
        cwd=root,
        base_env={
            "PYTHONUNBUFFERED": "1",
            "API_PORT_OVERRIDE": str(api_port),
            CODEX_APP_MODE_PROFILE_ENV_KEY: resolved_mode.api_frontend_mode,
        },
        allow_external_terminal=True,
    )
    services: Dict[str, CodexServiceHandle] = {
        "API": CodexServiceHandle(api_spec, log_buffer=log_buffer),
    }
    if resolved_mode.requires_ui_service:
        ui_spec = CodexServiceSpec(
            name="UI",
            command=build_ui_dev_service_command(frontend_dev_typecheck=frontend_dev_typecheck),
            cwd=root / "apps" / "interface",
            base_env={
                "FORCE_COLOR": "1",
                "API_HOST": "localhost",
                "API_PORT": str(api_port),
                "WEB_PORT": str(web_port),
                "SERVER_HOST": "localhost",
            },
            allow_external_terminal=os.name == "nt",
        )
        services["UI"] = CodexServiceHandle(ui_spec, log_buffer=log_buffer)
    return services


def _codex_root() -> Path:
    return get_repo_root()
