"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher service specs and process supervision (API + UI).
Defines service specs/handles, spawns subprocesses with environment overrides, streams logs into a shared buffer, performs strict port
availability checks (IPv4/IPv6), resolves launcher-owned API fallback ports before spawning the backend child, and captures live browser host
truth on running service handles for launcher endpoint resolution.
Maps launcher env toggles to backend CLI flags, including main/mount/offload bootstrap device flags, with offload defaulting to CPU when unset,
plus trace/profiler diagnostics toggles.
App startup mode is explicit via mode profiles (`dev_service` or `embedded`) with fail-loud preflight gates before mode activation.
When `CODEX_CUDA_MALLOC=1`, validates/ensures `PYTORCH_CUDA_ALLOC_CONF` includes `backend:cudaMallocAsync` before spawning the API process.
Launcher-owned frontend dev boot policy selects `npm run dev:fast` or `npm run dev:typecheck` for the UI service without introducing a new
runtime env var.

Symbols (top-level; keep in sync; no ghosts):
- `ServiceStatus` (enum): Launcher service lifecycle status.
- `CodexServiceSpec` (dataclass): Static service definition (command/cwd/env + external-terminal policy).
- `CodexServiceHandle` (dataclass): Runtime service handle; spawns/monitors subprocess, forwards stdout/stderr to a log buffer, and tracks live endpoint truth.
- `_codex_root` (function): Resolves the repo root used for service working directories.
- `default_services` (function): Builds default API+UI service handles with ports/env derived from the environment.
- `_env_truthy` (function): Normalizes launcher env booleans (`1/true/yes/on`) for CLI flag forwarding.
- `_sanitize_allocator_env_contract` (function): Removes unsupported allocator env keys before subprocess spawn (contract keys only: `PYTORCH_CUDA_ALLOC_CONF` + `CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF`).
- `_parse_pytorch_cuda_alloc_conf` (function): Parses `PYTORCH_CUDA_ALLOC_CONF` entries into strict `key:value` pairs.
- `_ensure_cuda_malloc_async_allocator_env` (function): Enforces allocator backend `cudaMallocAsync` when `CODEX_CUDA_MALLOC=1`.
- `_api_backend_args_from_env` (function): Builds backend CLI args for the API service from launcher env settings (device defaults, attention backend/SDPA policy, LoRA/runtime toggles; offload defaults to CPU when unset).
- `_parse_port_like_value` (function): Parses a port-like value into a valid TCP port or returns `None`.
- `_api_port_candidate_chain` (function): Returns the launcher/API fallback chain for a requested/base API port.
- `_resolve_api_runtime_port` (function): Resolves the effective launcher-owned API port before spawn and reports whether fallback was used.
- `build_ui_dev_service_command` (function): Builds the UI dev-service command from the launcher-owned frontend typecheck toggle.
- `_extract_cli_port` (function): Extracts a `--port` value from a command list.
- `_port_free_everywhere` (function): Validates a port is bindable on common IPv4/IPv6 local hosts.
- `_windows_no_activate` (function): Windows startupinfo helper to open consoles without stealing focus.
"""

from __future__ import annotations

import logging
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import errno
import socket
from contextlib import closing
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Iterator, List, Mapping, MutableMapping, Optional

from .log_buffer import CodexLogBuffer
from .profiles import (
    CODEX_APP_MODE_PROFILE_ENV_KEY,
    DEFAULT_LAUNCHER_MODE_PROFILE,
    LAUNCHER_MODE_PROFILE_CHOICES,
    LAUNCHER_MODE_PROFILE_DEV_SERVICE,
    LAUNCHER_MODE_PROFILE_EMBEDDED,
    normalize_mode_profile,
)
from .settings import DEVICE_CHOICES
from apps.backend.infra.config.repo_root import get_repo_root

LOGGER = logging.getLogger("codex.launcher.services")


class ServiceStatus(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass(slots=True)
class CodexServiceSpec:
    """Static definition of a service the launcher can run."""

    name: str
    command: List[str]
    cwd: Path
    base_env: Mapping[str, str] = field(default_factory=dict)
    allow_external_terminal: bool = False


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


@dataclass
class CodexServiceHandle:
    spec: CodexServiceSpec
    log_buffer: Optional[CodexLogBuffer] = None
    process: subprocess.Popen[str] | None = None
    status: ServiceStatus = ServiceStatus.STOPPED
    pid: Optional[int] = None
    started_at: Optional[float] = None
    last_exit_code: int | None = None
    process_group_id: int | None = None
    effective_port: int | None = None
    effective_host: str | None = None
    launch_token: str | None = None
    _stop_requested: bool = False
    _stop_reason: str | None = None
    _stdout_thread: Optional[threading.Thread] = None
    _stderr_thread: Optional[threading.Thread] = None
    _queue: Queue[str] = field(default_factory=Queue, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self, overrides: Mapping[str, str], *, external_terminal: bool = False) -> None:
        with self._lock:
            proc = self.process
            if proc and proc.poll() is None:
                LOGGER.info("Service %s already running (pid=%s)", self.spec.name, proc.pid)
                self.status = ServiceStatus.RUNNING
                self.pid = proc.pid
                return

        overrides_map = dict(overrides)

        env = os.environ.copy()
        env.update(self.spec.base_env)
        env.update(overrides_map)
        _sanitize_allocator_env_contract(env, scope_label=self.spec.name)

        command = list(self.spec.command)
        normalized_service_name = self.spec.name.upper()
        resolved_api_port: int | None = None
        resolved_browser_host: str | None = None
        resolved_launch_token: str | None = None
        if normalized_service_name == "API":
            resolved_mode = _resolve_requested_mode_profile(preferred_profile=None, env=env)
            _assert_mode_preflight(resolved_mode, self.spec.cwd)
            if _env_truthy(env.get("CODEX_CUDA_MALLOC")):
                _ensure_cuda_malloc_async_allocator_env(env)
            command.extend(_api_backend_args_from_env(env))
            resolved_api_port, requested_api_port, used_fallback = _resolve_api_runtime_port(command=command, env=env)
            env["API_PORT_OVERRIDE"] = str(resolved_api_port)
            raw_host = str(env.get("API_HOST", "localhost")).strip() or "localhost"
            resolved_browser_host = "localhost" if raw_host in {"0.0.0.0", "::", "[::]"} else raw_host
            if used_fallback:
                fallback_message = (
                    f"API port {requested_api_port} is busy; using fallback {resolved_api_port}."
                )
                if self.log_buffer:
                    self.log_buffer.log("launcher", fallback_message)
        elif normalized_service_name == "UI":
            raw_host = str(env.get("SERVER_HOST", "localhost")).strip() or "localhost"
            resolved_browser_host = "localhost" if raw_host in {"0.0.0.0", "::", "[::]"} else raw_host
            resolved_launch_token = secrets.token_hex(16)
            env["CODEX_LAUNCHER_UI_INSTANCE_TOKEN"] = resolved_launch_token
        flags = 0
        startupinfo = None
        use_external = external_terminal and self.spec.allow_external_terminal
        if use_external and os.name == "nt":
            flags |= getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            startupinfo = _windows_no_activate()
            command = ["cmd.exe", "/K", subprocess.list2cmdline(command)]
        elif os.name == "nt":
            # Needed for CTRL_BREAK_EVENT to be deliverable (best-effort graceful stop).
            flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
        stdin = subprocess.DEVNULL
        if use_external:
            stdout = None
            stderr = None
            stdin = None

        with self._lock:
            self.status = ServiceStatus.STARTING
            self.started_at = time.time()
            self._stop_requested = False
            self._stop_reason = None
            self.last_exit_code = None
            self.process_group_id = None
            self.effective_port = resolved_api_port
            self.effective_host = resolved_browser_host
            self.launch_token = resolved_launch_token
        try:
            popen_kwargs: dict[str, object] = {}
            if os.name != "nt" and not use_external:
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(
                command,
                cwd=str(self.spec.cwd),
                env=env,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                text=True if stdout is not None else False,
                bufsize=1,
                creationflags=flags,
                startupinfo=startupinfo,
                **popen_kwargs,
            )
        except Exception as exc:
            with self._lock:
                self.status = ServiceStatus.ERROR
                self.pid = None
                self.process = None
                self.effective_port = None
                self.effective_host = None
                self.launch_token = None
            LOGGER.error("Failed to start %s: %s", self.spec.name, exc)
            raise

        with self._lock:
            self.process = proc
            self.pid = proc.pid
        if os.name != "nt" and not use_external:
            # When `start_new_session=True`, the child becomes a new process group leader.
            with self._lock:
                self.process_group_id = proc.pid
        with self._lock:
            self.status = ServiceStatus.RUNNING
        if stdout is subprocess.PIPE and stderr is subprocess.PIPE:
            self._stdout_thread = threading.Thread(target=self._capture_output, args=("stdout",), daemon=True)
            self._stdout_thread.start()
            self._stderr_thread = threading.Thread(target=self._capture_output, args=("stderr",), daemon=True)
            self._stderr_thread.start()
        threading.Thread(target=self._wait_for_exit, args=(proc,), daemon=True).start()

    def stop(self, *, wait: float = 10.0) -> None:
        with self._lock:
            proc = self.process
            if not proc or proc.poll() is not None:
                self.status = ServiceStatus.STOPPED
                self.pid = None
                self.process = None
                self.process_group_id = None
                self.started_at = None
                self.effective_port = None
                self.effective_host = None
                self.launch_token = None
                return
            LOGGER.info("Stopping service %s", self.spec.name)
            self._stop_requested = True
            if not self._stop_reason:
                self._stop_reason = "stopped"
            reason = str(self._stop_reason or "stopped")
            pgid = self.process_group_id
        exit_code: int | None = None
        try:
            if os.name == "nt":
                try:
                    proc.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))  # type: ignore[attr-defined]
                except Exception:
                    proc.terminate()
            else:
                if pgid:
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    proc.terminate()
            exit_code = proc.wait(timeout=wait)
            with self._lock:
                self.last_exit_code = exit_code
        except Exception:
            LOGGER.warning("Terminate failed, killing %s", self.spec.name)
            try:
                if os.name != "nt" and pgid:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
        finally:
            with self._lock:
                if self.process is proc:
                    self.status = ServiceStatus.STOPPED
                    self.pid = None
                    self.process = None
                    self.process_group_id = None
                    self.started_at = None
                    self.effective_port = None
                    self.effective_host = None
                    self.launch_token = None
            if self.log_buffer and exit_code is not None:
                self.log_buffer.log(self.spec.name, f"{reason} (code {exit_code})", stream="event")

    def kill(self, *, wait: float = 10.0) -> None:
        with self._lock:
            proc = self.process
            if not proc or proc.poll() is not None:
                self.status = ServiceStatus.STOPPED
                self.pid = None
                self.process = None
                self.process_group_id = None
                self.started_at = None
                self.effective_port = None
                self.effective_host = None
                self.launch_token = None
                return
            LOGGER.warning("Killing service %s", self.spec.name)
            self._stop_requested = True
            if not self._stop_reason:
                self._stop_reason = "killed"
            reason = str(self._stop_reason or "killed")
            pgid = self.process_group_id
        exit_code: int | None = None
        try:
            if os.name != "nt" and pgid:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
            try:
                exit_code = proc.wait(timeout=wait)
                with self._lock:
                    self.last_exit_code = exit_code
            except Exception:
                pass
        finally:
            with self._lock:
                if self.process is proc:
                    self.status = ServiceStatus.STOPPED
                    self.pid = None
                    self.process = None
                    self.process_group_id = None
                    self.started_at = None
                    self.effective_port = None
                    self.effective_host = None
                    self.launch_token = None
            if self.log_buffer and exit_code is not None:
                self.log_buffer.log(self.spec.name, f"{reason} (code {exit_code})", stream="event")

    def restart(self, overrides: Mapping[str, str], *, external_terminal: bool = False) -> None:
        with self._lock:
            self._stop_requested = True
            self._stop_reason = "restarting"
        self.stop(wait=10.0)
        time.sleep(0.2)
        self.start(overrides, external_terminal=external_terminal)

    def iterate_live_output(self) -> Iterator[str]:
        while self.process and self.process.poll() is None:
            try:
                yield self._queue.get(timeout=0.1)
            except Empty:
                continue

    def _capture_output(self, stream_name: str) -> None:
        proc = self.process
        if not proc:
            return
        stream = getattr(proc, stream_name, None)
        if stream is None:
            return
        try:
            for line in stream:
                cleaned = (line.rstrip("\n") or " ")
                if self.log_buffer:
                    self.log_buffer.log(self.spec.name, cleaned, stream=stream_name)
                self._queue.put(cleaned)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _wait_for_exit(self, proc: subprocess.Popen) -> None:
        code = proc.wait()
        with self._lock:
            if self.process is not proc:
                return
            self.last_exit_code = code
            if self._stop_requested:
                status = ServiceStatus.STOPPED
                reason = str(self._stop_reason or "stopped")
                message = f"{reason} (code {code})"
            else:
                if code == 0:
                    status = ServiceStatus.STOPPED
                    message = "exited cleanly"
                else:
                    status = ServiceStatus.ERROR
                    message = f"exited with code {code}"
            self.status = status
            self.pid = None
            self.process = None
            self.process_group_id = None
            self.started_at = None
            self.effective_port = None
            self.effective_host = None
            self.launch_token = None
        if self.log_buffer and not self._stop_requested:
            self.log_buffer.log(self.spec.name, message, stream="event")


def _codex_root() -> Path:
    return get_repo_root()


def _resolve_mode_profile(raw_value: str, *, source_label: str) -> LauncherModeProfile:
    normalized = normalize_mode_profile(raw_value, source=source_label)
    profile = _MODE_PROFILE_MAP.get(normalized)
    if profile is None:
        allowed = ", ".join(LAUNCHER_MODE_PROFILE_CHOICES)
        raise RuntimeError(
            f"Unsupported launcher mode profile {normalized!r}. Allowed values: {allowed}."
        )
    return profile


def _resolve_requested_mode_profile(
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


def _assert_mode_preflight(profile: LauncherModeProfile, root: Path) -> None:
    if profile.key == LAUNCHER_MODE_PROFILE_EMBEDDED:
        _assert_embedded_packaging_contract(root)
        return
    if profile.key == LAUNCHER_MODE_PROFILE_DEV_SERVICE:
        _assert_dev_service_requirements(root)
        return
    raise RuntimeError(f"Unhandled launcher mode profile {profile.key!r}.")


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
    resolved_mode = _resolve_requested_mode_profile(
        preferred_profile=mode_profile,
        env=os.environ,
    )
    _assert_mode_preflight(resolved_mode, root)
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


def _env_truthy(value: object) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _sanitize_allocator_env_contract(env: MutableMapping[str, str], *, scope_label: str) -> None:
    supported_alloc_key = "PYTORCH_CUDA_ALLOC_CONF"
    supported_toggle_key = "CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF"
    unsupported_keys: list[str] = []
    for key in list(env.keys()):
        if key.startswith("PYTORCH_") and key.endswith("_ALLOC_CONF") and key != supported_alloc_key:
            unsupported_keys.append(key)
            continue
        if (
            key.startswith("CODEX_ENABLE_DEFAULT_PYTORCH_")
            and key.endswith("_ALLOC_CONF")
            and key != supported_toggle_key
        ):
            unsupported_keys.append(key)
    if unsupported_keys:
        keys = ", ".join(sorted(unsupported_keys))
        raise ValueError(
            f"Unsupported allocator env key(s) for {scope_label}: {keys}. "
            "Supported keys: PYTORCH_CUDA_ALLOC_CONF and "
            f"{supported_toggle_key}."
        )


def _parse_pytorch_cuda_alloc_conf(raw_conf: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for raw_entry in str(raw_conf or "").split(","):
        token = raw_entry.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                "Invalid PYTORCH_CUDA_ALLOC_CONF entry "
                f"{token!r}: expected 'key:value' format."
            )
        key, value = token.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                "Invalid PYTORCH_CUDA_ALLOC_CONF entry "
                f"{token!r}: expected non-empty 'key:value' parts."
            )
        entries.append((key, value))
    return entries


def _ensure_cuda_malloc_async_allocator_env(env: MutableMapping[str, str]) -> None:
    target_backend = "cudaMallocAsync"
    target_backend_norm = target_backend.lower()
    raw_alloc_conf = str(env.get("PYTORCH_CUDA_ALLOC_CONF", "") or "").strip()
    if not raw_alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = f"backend:{target_backend}"
        return

    entries = _parse_pytorch_cuda_alloc_conf(raw_alloc_conf)
    backend_index: int | None = None
    for index, (key, _value) in enumerate(entries):
        if key.strip().lower() == "backend":
            if backend_index is not None:
                raise ValueError(
                    "Invalid PYTORCH_CUDA_ALLOC_CONF: multiple 'backend' entries found. "
                    "Use exactly one backend directive."
                )
            backend_index = index

    if backend_index is None:
        entries.append(("backend", target_backend))
        env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(f"{key}:{value}" for key, value in entries)
        return

    configured_backend = entries[backend_index][1]
    if configured_backend.replace(" ", "").lower() != target_backend_norm:
        raise ValueError(
            "CODEX_CUDA_MALLOC=1 requires PYTORCH_CUDA_ALLOC_CONF backend:cudaMallocAsync, "
            f"but found backend:{configured_backend}. "
            "Set PYTORCH_CUDA_ALLOC_CONF with backend:cudaMallocAsync or disable CODEX_CUDA_MALLOC."
        )
    env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(f"{key}:{value}" for key, value in entries)


def _api_backend_args_from_env(env: Mapping[str, str]) -> List[str]:
    args: List[str] = []
    allowed_devices = set(DEVICE_CHOICES)

    def _append_device_arg(*, env_key: str, flag: str, fallback: str = "") -> None:
        raw_value = str(env.get(env_key, "") or "").strip().lower()
        if not raw_value:
            raw_value = str(fallback or "").strip().lower()
        if not raw_value:
            return
        if raw_value not in allowed_devices:
            allowed = ", ".join(sorted(allowed_devices))
            raise ValueError(f"{env_key} must be one of: {allowed} (got {raw_value!r}).")
        args.append(f"--{flag}={raw_value}")

    # Main device contract (single authority). When provided, mirror to all components.
    raw_main_device = str(env.get("CODEX_MAIN_DEVICE", "") or "").strip().lower()
    if raw_main_device:
        if raw_main_device not in allowed_devices:
            allowed = ", ".join(sorted(allowed_devices))
            raise ValueError(f"CODEX_MAIN_DEVICE must be one of: {allowed} (got {raw_main_device!r}).")
        args.append(f"--main-device={raw_main_device}")
        args.append(f"--core-device={raw_main_device}")
        args.append(f"--te-device={raw_main_device}")
        args.append(f"--vae-device={raw_main_device}")
    else:
        # Legacy per-component defaults (kept for backward-compatible launcher profiles).
        raw_core_device = str(env.get("CODEX_CORE_DEVICE", "") or "").strip().lower()
        if raw_core_device:
            if raw_core_device not in allowed_devices:
                allowed = ", ".join(sorted(allowed_devices))
                raise ValueError(f"CODEX_CORE_DEVICE must be one of: {allowed} (got {raw_core_device!r}).")
            args.append(f"--core-device={raw_core_device}")

        raw_te_device = str(env.get("CODEX_TE_DEVICE", "") or "").strip().lower()
        if raw_te_device:
            if raw_te_device not in allowed_devices:
                allowed = ", ".join(sorted(allowed_devices))
                raise ValueError(f"CODEX_TE_DEVICE must be one of: {allowed} (got {raw_te_device!r}).")
            args.append(f"--te-device={raw_te_device}")

        raw_vae_device = str(env.get("CODEX_VAE_DEVICE", "") or "").strip().lower()
        if raw_vae_device:
            if raw_vae_device not in allowed_devices:
                allowed = ", ".join(sorted(allowed_devices))
                raise ValueError(f"CODEX_VAE_DEVICE must be one of: {allowed} (got {raw_vae_device!r}).")
            args.append(f"--vae-device={raw_vae_device}")

    _append_device_arg(env_key="CODEX_MOUNT_DEVICE", flag="mount-device", fallback=raw_main_device)
    _append_device_arg(env_key="CODEX_OFFLOAD_DEVICE", flag="offload-device", fallback="cpu")

    raw_attention_backend = str(env.get("CODEX_ATTENTION_BACKEND", "") or "").strip().lower()
    if raw_attention_backend:
        if raw_attention_backend not in {"pytorch", "xformers", "split", "quad"}:
            raise ValueError(
                "CODEX_ATTENTION_BACKEND must be one of: pytorch, xformers, split, quad "
                f"(got {raw_attention_backend!r}).",
            )
        args.append(f"--attention-backend={raw_attention_backend}")
        raw_sdpa_policy = str(env.get("CODEX_ATTENTION_SDPA_POLICY", "") or "").strip().lower()
        if raw_attention_backend == "pytorch" and raw_sdpa_policy:
            if raw_sdpa_policy not in {"auto", "flash", "mem_efficient", "math"}:
                raise ValueError(
                    "CODEX_ATTENTION_SDPA_POLICY must be one of: auto, flash, mem_efficient, math "
                    f"(got {raw_sdpa_policy!r}).",
                )
            args.append(f"--attention-sdpa-policy={raw_sdpa_policy}")

    raw_lora_mode = str(env.get("CODEX_LORA_APPLY_MODE", "") or "").strip().lower()
    if raw_lora_mode:
        args.append(f"--lora-apply-mode={raw_lora_mode}")

    raw_lora_math = str(env.get("CODEX_LORA_ONLINE_MATH", "") or "").strip().lower()
    if raw_lora_math:
        args.append(f"--lora-online-math={raw_lora_math}")

    if _env_truthy(env.get("CODEX_CUDA_MALLOC")):
        args.append("--cuda-malloc")

    if _env_truthy(env.get("CODEX_TRACE_CONTRACT")):
        args.append("--trace-contract")
    if _env_truthy(env.get("CODEX_TRACE_PROFILER")):
        args.append("--trace-profiler")

    return args


def _extract_cli_port(command: List[str]) -> int | None:
    for idx, token in enumerate(command):
        if token == "--port" and idx + 1 < len(command):
            try:
                return int(command[idx + 1])
            except Exception:
                return None
        if token.startswith("--port="):
            try:
                return int(token.split("=", 1)[1])
            except Exception:
                return None
    return None


def _parse_port_like_value(raw_value: object) -> int | None:
    try:
        parsed = int(str(raw_value or "").strip())
    except (TypeError, ValueError):
        return None
    if parsed < 1 or parsed > 65535:
        return None
    return int(parsed)


def _api_port_candidate_chain(base_port: int) -> tuple[int, int, int]:
    normalized_base = int(base_port)
    return (
        normalized_base,
        normalized_base + 10000,
        normalized_base + 20000,
    )


def _resolve_api_runtime_port(*, command: List[str], env: Mapping[str, str]) -> tuple[int, int, bool]:
    requested_port = _extract_cli_port(command)
    if requested_port is None:
        requested_port = _parse_port_like_value(env.get("API_PORT_OVERRIDE") or env.get("API_PORT"))
    if requested_port is None:
        requested_port = 7850
    blocked_details: list[str] = []
    for index, candidate in enumerate(_api_port_candidate_chain(requested_port)):
        if candidate < 1 or candidate > 65535:
            blocked_details.append(f"{candidate} (out_of_range)")
            continue
        ok, blocked = _port_free_everywhere(candidate)
        if ok:
            return int(candidate), int(requested_port), bool(index != 0)
        blocked_details.append(f"{candidate} ({blocked or 'busy'})")
    blocked_summary = ", ".join(blocked_details)
    raise RuntimeError(
        "No free API port in launcher fallback chain. "
        f"Tried: {blocked_summary}. "
        "You may already have Codex running (WSL/Windows) or another service bound on IPv4/IPv6 localhost. "
        "Stop the other instance or set API_PORT_OVERRIDE/WEB_PORT to a free pair."
    )


def _port_free_everywhere(port: int) -> tuple[bool, str]:
    def _can_bind(family: int, host: str) -> tuple[bool, str]:
        try:
            with closing(socket.socket(family, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if family == socket.AF_INET6:
                    s.bind((host, port, 0, 0))
                else:
                    s.bind((host, port))
                return True, ""
        except OSError as exc:
            if getattr(exc, "errno", None) in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
                return True, ""
            code = getattr(exc, "errno", None)
            return False, f"host={host} errno={code}"

    for family, host in (
        (socket.AF_INET, "0.0.0.0"),
        (socket.AF_INET, "127.0.0.1"),
        (socket.AF_INET6, "::"),
        (socket.AF_INET6, "::1"),
    ):
        ok, detail = _can_bind(family, host)
        if not ok:
            return False, detail
    return True, ""


def _windows_no_activate():
    if os.name != "nt":
        return None
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
        startupinfo.wShowWindow = 4  # SW_SHOWNOACTIVATE
        return startupinfo
    except Exception:
        return None
