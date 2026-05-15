"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Non-UI controller for the Tk launcher.
Wraps launcher infrastructure (profiles, services, logs) behind a small imperative API so tabs don’t reach into internals.
Builds both working and cached-committed service-scoped environments so launcher tabs can stay truthful about save-gated settings without
per-poll disk reload, applies API-only manual env overlays only to API starts/restarts, resolves service URLs from committed-vs-live launcher
truth (including repo-root live UI pid files), and refreshes launcher-owned UI dev boot command selection from committed meta before UI
starts/restarts.

Symbols (top-level; keep in sync; no ghosts):
- `LauncherController` (class): Holds store/services/log_buffer and provides service + persistence helpers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from apps.launcher.log_buffer import CodexLogBuffer
from apps.launcher.profiles import LauncherProfileStore
from apps.launcher.services import CodexServiceHandle, ServiceStatus, build_ui_dev_service_command


@dataclass(slots=True)
class LauncherController:
    codex_root: Path
    store: LauncherProfileStore
    log_buffer: CodexLogBuffer
    services: Dict[str, CodexServiceHandle]
    _committed_store_snapshot: LauncherProfileStore | None = field(default=None, init=False, repr=False)
    _last_manual_api_url_error: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._refresh_committed_store_snapshot()

    def build_env(self) -> dict[str, str]:
        return self.store.build_env()

    def build_env_for_service(self, name: str, *, committed: bool = False) -> dict[str, str]:
        store = self._get_committed_store() if committed else self.store
        return self._build_env_for_service_from_store(name, store)

    def service_urls(self, name: str) -> tuple[str, str | None]:
        normalized_name = str(name or "").strip().upper()
        committed_store = self._get_committed_store()
        env = self._build_url_env_from_store(normalized_name, committed_store)
        service = self.services[normalized_name]
        if normalized_name == "API":
            if service.status in {ServiceStatus.STARTING, ServiceStatus.RUNNING} and service.effective_port is not None:
                host = self._browser_host(str(service.effective_host or env.get("API_HOST", "localhost")))
                api_port = int(service.effective_port)
            else:
                host = self._browser_host(str(env.get("API_HOST", "localhost")))
                api_port = self._parse_port(
                    env.get("API_PORT_OVERRIDE", service.spec.base_env.get("API_PORT_OVERRIDE", "7850")),
                    default=7850,
                )
            root_url = f"http://{host}:{api_port}"
            return root_url, f"{root_url}/docs"
        if normalized_name == "UI":
            base_port = self._parse_port(
                env.get("WEB_PORT", service.spec.base_env.get("WEB_PORT", "7860")),
                default=7860,
            )
            if service.status in {ServiceStatus.STARTING, ServiceStatus.RUNNING}:
                host = self._browser_host(str(service.effective_host or env.get("SERVER_HOST", "localhost")))
                runtime_port = self._resolve_ui_effective_port(
                    base_port=int(base_port),
                    pid_root=self.codex_root,
                    interface_cwd=Path(service.spec.cwd),
                    expected_launch_token=str(service.launch_token or "").strip() or None,
                )
                effective_port = int(runtime_port)
            else:
                host = self._browser_host(str(env.get("SERVER_HOST", "localhost")))
                effective_port = int(base_port)
            return f"http://{host}:{effective_port}", None
        return "-", None

    def committed_external_terminal(self) -> bool:
        return bool(getattr(self._get_committed_store().meta, "external_terminal", False))

    def working_external_terminal(self) -> bool:
        return bool(getattr(self.store.meta, "external_terminal", False))

    def working_frontend_dev_typecheck(self) -> bool:
        return bool(getattr(self.store.meta, "frontend_dev_typecheck", False))

    def update_external_terminal(self, enabled: bool) -> None:
        self.store.meta.external_terminal = bool(enabled)

    def update_frontend_dev_typecheck(self, enabled: bool) -> None:
        self.store.meta.frontend_dev_typecheck = bool(enabled)

    def update_manual_api_env_enabled(self, enabled: bool) -> None:
        self.store.meta.manual_api_env_enabled = bool(enabled)

    def update_manual_api_env_text(self, text: str) -> None:
        self.store.meta.manual_api_env_text = str(text)

    def validate_manual_api_env_text(self) -> None:
        self.store.build_manual_api_env_overlay()

    def manual_api_env_error(self) -> str | None:
        try:
            self.validate_manual_api_env_text()
        except Exception as exc:
            return str(exc)
        return None

    def _build_url_env_from_store(self, name: str, store: LauncherProfileStore) -> dict[str, str]:
        normalized_name = str(name or "").strip().upper()
        if normalized_name != "API":
            return self._build_env_for_service_from_store(normalized_name, store)
        try:
            env = self._build_env_for_service_from_store(normalized_name, store)
        except ValueError as exc:
            message = str(exc)
            if message != self._last_manual_api_url_error:
                self.log_buffer.log(
                    "launcher",
                    "Manual Env Vars invalid for API URL preview/open; ignoring overlay until fixed: "
                    f"{message}",
                    stream="event",
                )
                self._last_manual_api_url_error = message
            return store.build_env()
        self._last_manual_api_url_error = None
        return env

    def _build_env_for_service_from_store(self, name: str, store: LauncherProfileStore) -> dict[str, str]:
        env = store.build_env()
        normalized_name = str(name or "").strip().upper()
        if normalized_name == "API":
            env.update(store.build_manual_api_env_overlay())
            return env
        if normalized_name == "UI":
            api_service = self.services.get("API")
            if (
                api_service is not None
                and api_service.status in {ServiceStatus.STARTING, ServiceStatus.RUNNING}
                and api_service.effective_port is not None
            ):
                env["API_PORT"] = str(api_service.effective_port)
        return env

    @property
    def service_names(self) -> tuple[str, ...]:
        return tuple(self.services.keys())

    @property
    def external_terminal_supported(self) -> bool:
        return os.name == "nt"

    def start_service(self, name: str) -> None:
        normalized_name = str(name or "").strip().upper()
        committed_store = self._get_committed_store(refresh=True)
        if normalized_name == "UI":
            self._refresh_ui_service_command(store=committed_store)
        env = self._build_env_for_service_from_store(normalized_name, committed_store)
        external_terminal = bool(getattr(committed_store.meta, "external_terminal", False)) and self.external_terminal_supported
        self.services[normalized_name].start(env, external_terminal=external_terminal)

    def restart_service(self, name: str) -> None:
        normalized_name = str(name or "").strip().upper()
        committed_store = self._get_committed_store(refresh=True)
        if normalized_name == "UI":
            self._refresh_ui_service_command(store=committed_store)
        env = self._build_env_for_service_from_store(normalized_name, committed_store)
        external_terminal = bool(getattr(committed_store.meta, "external_terminal", False)) and self.external_terminal_supported
        self.services[normalized_name].restart(env, external_terminal=external_terminal)

    def stop_service(self, name: str, *, wait: float = 10.0) -> None:
        self.services[name].stop(wait=wait)

    def kill_service(self, name: str, *, wait: float = 10.0) -> None:
        self.services[name].kill(wait=wait)

    def start_all(self) -> None:
        for name in self.service_names:
            self.start_service(name)

    def stop_all(self, *, wait: float = 10.0) -> None:
        for name in reversed(self.service_names):
            self.stop_service(name, wait=wait)

    def persist_tab_index(self, tab_index: int) -> None:
        self.store.meta.tab_index = int(tab_index)
        self.store.save_meta()

    def persist_window_geometry(self, geometry: str) -> None:
        self.store.meta.window_geometry = str(geometry)
        self.store.save_meta()

    def persist_show_advanced_controls(self, enabled: bool) -> None:
        self.store.meta.show_advanced_controls = bool(enabled)
        self.store.save_meta()

    def save_settings(self) -> None:
        self.store.save()
        committed_store = self._refresh_committed_store_snapshot()
        self._refresh_ui_service_command(store=committed_store)

    def reload_store(self) -> None:
        self.store = LauncherProfileStore.load(root=self.store.root)
        committed_store = self._refresh_committed_store_snapshot()
        self._refresh_ui_service_command(store=committed_store)

    def _refresh_committed_store_snapshot(self) -> LauncherProfileStore:
        committed_store = LauncherProfileStore.load(root=self.store.root)
        self._committed_store_snapshot = committed_store
        return committed_store

    def _get_committed_store(self, *, refresh: bool = False) -> LauncherProfileStore:
        if refresh or self._committed_store_snapshot is None:
            return self._refresh_committed_store_snapshot()
        return self._committed_store_snapshot

    def _refresh_ui_service_command(self, *, store: LauncherProfileStore | None = None) -> None:
        ui_service = self.services.get("UI")
        if ui_service is None:
            return
        source_store = self.store if store is None else store
        ui_service.spec.command = build_ui_dev_service_command(
            frontend_dev_typecheck=bool(getattr(source_store.meta, "frontend_dev_typecheck", False))
        )

    @staticmethod
    def _browser_host(raw_host: str) -> str:
        host = str(raw_host or "localhost").strip() or "localhost"
        if host in {"0.0.0.0", "::", "[::]"}:
            return "localhost"
        return host

    @staticmethod
    def _parse_int(value: object, *, default: int) -> int:
        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError):
            return int(default)
        return int(parsed)

    @staticmethod
    def _parse_port(value: object, *, default: int) -> int:
        parsed = LauncherController._parse_int(value, default=default)
        if parsed < 1 or parsed > 65535:
            return int(default)
        return int(parsed)

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True
        except ProcessLookupError:
            return False
        except OSError:
            return False
        return True

    @staticmethod
    def _resolve_ui_effective_port(
        *,
        base_port: int,
        pid_root: Path,
        interface_cwd: Path,
        expected_launch_token: str | None = None,
    ) -> int:
        expected_cwd = str(interface_cwd.resolve())
        for candidate in (base_port + 20000, base_port + 10000, base_port):
            pid_path = pid_root / f".webui-ui-{candidate}.pid"
            if not pid_path.exists():
                continue
            try:
                payload = json.loads(pid_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(payload.get("service", "")).strip().lower() != "ui":
                continue
            if LauncherController._parse_port(payload.get("port"), default=-1) != candidate:
                continue
            raw_cwd = str(payload.get("cwd", "")).strip()
            if not raw_cwd:
                continue
            try:
                if str(Path(raw_cwd).resolve()) != expected_cwd:
                    continue
            except (TypeError, ValueError, OSError):
                continue
            if expected_launch_token is not None:
                receipt_token = str(payload.get("launcher_ui_token", "")).strip()
                if receipt_token != expected_launch_token:
                    continue
            pid = LauncherController._parse_int(payload.get("pid"), default=-1)
            if not LauncherController._pid_is_alive(pid):
                continue
            return int(candidate)
        return int(base_port)
