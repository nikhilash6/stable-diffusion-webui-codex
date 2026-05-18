"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher service process specifications, launch preparation, and lifecycle supervision.
Owns service handles and the pure `prepare_service_launch(...)` seam so startup command/env preparation can be smoke-tested without spawning processes.

Symbols (top-level; keep in sync; no ghosts):
- `ServiceStatus` (class): Launcher service lifecycle states.
- `CodexServiceSpec` (dataclass): Static definition of a launcher-managed service.
- `PreparedServiceLaunch` (dataclass): Fully prepared no-spawn service launch plan.
- `prepare_service_launch` (function): Builds final command/env/stdio/startupinfo for a service start.
- `CodexServiceHandle` (dataclass): Mutable process supervisor for one launcher-managed service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import logging
import os
from pathlib import Path
from queue import Empty, Queue
import secrets
import signal
import subprocess
import threading
import time
from typing import Iterator, List, Mapping, MutableMapping, Optional

from apps.launcher.api_args import api_backend_args_from_env
from apps.launcher.log_buffer import CodexLogBuffer
from apps.launcher.ports import resolve_api_runtime_port
from apps.launcher.service_env import (
    ensure_cuda_malloc_async_allocator_env,
    env_truthy,
    sanitize_allocator_env_contract,
)
from apps.launcher.service_modes import assert_mode_preflight, resolve_requested_mode_profile

LOGGER = logging.getLogger("codex.launcher.service_process")


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
class PreparedServiceLaunch:
    command: List[str]
    cwd: Path
    env: MutableMapping[str, str]
    stdin: object
    stdout: object
    stderr: object
    text: bool
    bufsize: int
    creationflags: int
    startupinfo: object | None
    popen_kwargs: Mapping[str, object]
    use_external: bool
    effective_port: int | None
    effective_host: str | None
    launch_token: str | None


def prepare_service_launch(
    spec: CodexServiceSpec,
    overrides: Mapping[str, str],
    *,
    external_terminal: bool = False,
    log_buffer: CodexLogBuffer | None = None,
) -> PreparedServiceLaunch:
    """Prepare service command/env/stdio without mutating a handle or spawning a process."""

    overrides_map = dict(overrides)
    env = os.environ.copy()
    env.update(spec.base_env)
    env.update(overrides_map)
    sanitize_allocator_env_contract(env, scope_label=spec.name)

    command = list(spec.command)
    normalized_service_name = spec.name.upper()
    resolved_api_port: int | None = None
    resolved_browser_host: str | None = None
    resolved_launch_token: str | None = None
    if normalized_service_name == "API":
        resolved_mode = resolve_requested_mode_profile(preferred_profile=None, env=env)
        assert_mode_preflight(resolved_mode, spec.cwd)
        if env_truthy(env.get("CODEX_CUDA_MALLOC")):
            ensure_cuda_malloc_async_allocator_env(env)
        command.extend(api_backend_args_from_env(env))
        resolved_api_port, requested_api_port, used_fallback = resolve_api_runtime_port(command=command, env=env)
        env["API_PORT_OVERRIDE"] = str(resolved_api_port)
        raw_host = str(env.get("API_HOST", "localhost")).strip() or "localhost"
        resolved_browser_host = "localhost" if raw_host in {"0.0.0.0", "::", "[::]"} else raw_host
        if used_fallback:
            fallback_message = f"API port {requested_api_port} is busy; using fallback {resolved_api_port}."
            if log_buffer:
                log_buffer.log("launcher", fallback_message)
    elif normalized_service_name == "UI":
        raw_host = str(env.get("SERVER_HOST", "localhost")).strip() or "localhost"
        resolved_browser_host = "localhost" if raw_host in {"0.0.0.0", "::", "[::]"} else raw_host
        resolved_launch_token = secrets.token_hex(16)
        env["CODEX_LAUNCHER_UI_INSTANCE_TOKEN"] = resolved_launch_token

    flags = 0
    startupinfo = None
    use_external = external_terminal and spec.allow_external_terminal
    if use_external and os.name == "nt":
        flags |= getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        startupinfo = _windows_no_activate()
        command = ["cmd.exe", "/K", subprocess.list2cmdline(command)]
    elif os.name == "nt":
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    stdout = subprocess.PIPE
    stderr = subprocess.PIPE
    stdin = subprocess.DEVNULL
    if use_external:
        stdout = None
        stderr = None
        stdin = None

    popen_kwargs: dict[str, object] = {}
    if os.name != "nt" and not use_external:
        popen_kwargs["start_new_session"] = True

    return PreparedServiceLaunch(
        command=command,
        cwd=spec.cwd,
        env=env,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        text=True if stdout is not None else False,
        bufsize=1,
        creationflags=flags,
        startupinfo=startupinfo,
        popen_kwargs=popen_kwargs,
        use_external=use_external,
        effective_port=resolved_api_port,
        effective_host=resolved_browser_host,
        launch_token=resolved_launch_token,
    )


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

        launch = prepare_service_launch(
            self.spec,
            overrides,
            external_terminal=external_terminal,
            log_buffer=self.log_buffer,
        )

        with self._lock:
            self.status = ServiceStatus.STARTING
            self.started_at = time.time()
            self._stop_requested = False
            self._stop_reason = None
            self.last_exit_code = None
            self.process_group_id = None
            self.effective_port = launch.effective_port
            self.effective_host = launch.effective_host
            self.launch_token = launch.launch_token
        try:
            proc = subprocess.Popen(
                launch.command,
                cwd=str(launch.cwd),
                env=launch.env,
                stdin=launch.stdin,
                stdout=launch.stdout,
                stderr=launch.stderr,
                text=launch.text,
                bufsize=launch.bufsize,
                creationflags=launch.creationflags,
                startupinfo=launch.startupinfo,
                **dict(launch.popen_kwargs),
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
        if os.name != "nt" and not launch.use_external:
            with self._lock:
                self.process_group_id = proc.pid
        with self._lock:
            self.status = ServiceStatus.RUNNING
        if launch.stdout is subprocess.PIPE and launch.stderr is subprocess.PIPE:
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
                cleaned = line.rstrip("\n") or " "
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


def _windows_no_activate() -> object | None:
    if os.name != "nt":
        return None
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
        startupinfo.wShowWindow = 4
        return startupinfo
    except Exception:
        return None
