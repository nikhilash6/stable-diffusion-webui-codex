# // tags: api-entrypoint, fastapi, task-router
"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: FastAPI entrypoint + uvicorn factory for the Codex WebUI backend.
This module builds the `/api/*` surface by assembling router modules (generation/tasks/models/options/tools/ui persistence/upscale/supir/tests), and mounts the built UI as SPA static files only when explicit embedded app mode is enabled (uses lifespan handlers for startup hooks; no deprecated `on_event`).
Bootstrap env overrides are published only when non-default to avoid pinning global defaults across test runs.
Bootstrap env publication includes LoRA loader policies (`CODEX_LORA_APPLY_MODE`, `CODEX_LORA_MERGE_MODE`, `CODEX_LORA_REFRESH_SIGNATURE`) from resolved runtime namespace values.
Startup settings normalization preserves `codex_options_revision` while pruning unknown keys and failing loud on invalid reliability-critical values
(including `codex_attention_backend`, checkbox settings, and non-finite numeric options).
Options router wiring includes the locked compare-and-set helper from `options_store` so conditional `POST /api/options` writes can reject stale
expected revisions atomically instead of relying on router-side prechecks.
Launcher/backend trace toggles (`--trace-contract`, `--trace-profiler`) are published via bootstrap env for runtime diagnostics modules.
Startup bootstrap logs allocator diagnostics (`PYTORCH_CUDA_ALLOC_CONF`, resolved allocator backend, and `--cuda-malloc` flag state) for fail-loud VRAM debugging.
Allocator bootstrap contract is `PYTORCH_CUDA_ALLOC_CONF` only.

Symbols (top-level; keep in sync; no ghosts):
- `_cli_arg_value` (function): Reads a CLI flag value from argv (supports `--flag value` and `--flag=value` forms).
- `_parse_trace_max` (function): Parses `--trace-debug-max-per-func` / `--trace-call-debug-max-per-func` into a non-negative int (or `None`).
- `_env_truthy` (function): Parses launcher/env boolean tokens (`1/true/yes/on`) from string values.
- `_emit_run_api_message` (function): Emits repo-owned bootstrap/API operational logs through the canonical backend wrapper.
- `_report_run_api_exception` (function): Dumps a handled bootstrap/API exception and emits one concise wrapper-owned summary line.
- `_trace_debug_logging_requested` (function): Resolves whether any trace-debug category requests DEBUG logging bootstrap.
- `_resolve_app_mode_profile` (function): Resolves explicit app mode profile (`dev_service|embedded`) from environment.
- `_assert_embedded_dist_contract` (function): Enforces embedded SPA packaging contract (`dist/index.html` + assets bundles).
- `ensure_initialized` (function): Performs early runtime bootstrap (repo root/sys.path, optional tracing/logging hooks) before serving.
- `_SuppressUvicornAccessNoiseFilter` (class): Logging filter to reduce uvicorn access-log spam for noisy endpoints.
- `_install_uvicorn_access_noise_filter` (function): Installs `_SuppressUvicornAccessNoiseFilter` when configured.
- `port_free` (function): Checks whether a TCP port is free on IPv4/IPv6 loopback/wildcard.
- `scan_range` (function): Scans a port range to find a free port.
- `pick_api_port_simple` (function): Picks a free port near a base port (and reports whether it was the base).
- `banner` (function): Logs the startup banner with the selected port.
- `_DummyRequest` (class): Minimal request shim used where a request-like object is needed without FastAPI internals.
- `build_app` (function): Constructs the FastAPI app; wires router modules, configures middleware, and mounts the UI SPA (lifespan-aware).
- `_bootstrap_runtime` (function): Bootstraps runtime settings/env before app creation (used by the uvicorn factory path).
- `_enable_trace_debug` (function): Enables global tracing/debug logging when requested via argv/env.
- `_try_disable_windows_power_throttling` (function): On Windows, disables process execution-speed throttling via `SetProcessInformation(ProcessPowerThrottling)`.
- `create_api_app` (function): Canonical uvicorn `--factory` entrypoint; calls bootstrap and returns the built FastAPI app.
- `main` (function): CLI entrypoint used by launchers (selects port, builds app, runs uvicorn).
"""

import asyncio
import errno
import math
import os
import socket
import sys
from pathlib import Path
from contextlib import closing, asynccontextmanager
from typing import Any, List, Mapping, Optional, Sequence, Tuple
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from apps.backend.infra.config.provenance import generation_provenance as _generation_provenance
from apps.backend.services.output_service import save_generated_images as _save_generated_images
from apps.backend.services.media_service import MediaService
from apps.backend.services.live_preview_service import LivePreviewService
from apps.backend.interfaces.api.path_utils import CODEX_ROOT
from apps.backend.interfaces.api.routers import generation, models, options, paths, settings, supir, system, tasks, tests, tools, ui, upscale
from apps.backend.services import options_store
from apps.backend.infra.config import args as config_args
from apps.backend.runtime.diagnostics.pipeline_debug import apply_env_flag as _apply_pipeline_debug_flag
from apps.backend.runtime.diagnostics.error_summary import summarize_exception_for_console
from apps.backend.runtime.memory import memory_management as mem_management
from apps.backend.runtime.models import api as model_api
from apps.backend.core.strict_values import parse_bool_value
from apps.backend.core.state import state as backend_state
from apps.backend.runtime.logging import build_backend_uvicorn_log_config, emit_backend_message

def _cli_arg_value(argv: Sequence[str], flag: str) -> Optional[str]:
    for idx, token in enumerate(argv):
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
        if token == flag and idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def _parse_trace_max(argv: Sequence[str]) -> Optional[int]:
    value = _cli_arg_value(argv, "--trace-debug-max-per-func")
    if value is None:
        value = _cli_arg_value(argv, "--trace-call-debug-max-per-func")
    if value is None:
        return None
    try:
        numeric = int(value)
    except Exception:
        return None
    return max(0, numeric)


def _env_truthy(raw: object) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _emit_run_api_message(
    message: str,
    /,
    *,
    level: int = logging.INFO,
    logger: str | None = None,
    **fields: object,
) -> None:
    emit_backend_message(
        message,
        logger=logger or __name__,
        level=level,
        **fields,
    )


def _report_run_api_exception(
    exc: BaseException,
    /,
    *,
    message: str,
    where: str,
    logger: str | None = None,
    **context_fields: object,
) -> None:
    try:
        from apps.backend.runtime.diagnostics.exception_hook import dump_exception as _dump_exception

        _dump_exception(
            type(exc),
            exc,
            exc.__traceback__,
            where=where,
            context=context_fields or None,
        )
    except Exception:
        pass
    _emit_run_api_message(
        f"{message}: {summarize_exception_for_console(exc)}",
        level=logging.ERROR,
        logger=logger,
    )


def _trace_debug_logging_requested(argv: Sequence[str], env: Mapping[str, str]) -> bool:
    call_trace_requested = (
        ("--trace-debug" in argv)
        or ("--trace-call-debug" in argv)
        or _env_truthy(env.get("CODEX_TRACE_CALL_DEBUG"))
    )
    if call_trace_requested:
        return True
    return _env_truthy(env.get("CODEX_TRACE_INFERENCE_DEBUG")) or _env_truthy(env.get("CODEX_TRACE_LOAD_PATCH_DEBUG"))

# Early trace hook: if any trace-debug source is requested (call/inference/load),
# force DEBUG logging visibility before importing FastAPI/uvicorn. Global
# function-call tracing remains call-trace-only.
try:
    startup_argv = tuple(sys.argv[1:])
    startup_env = os.environ
    call_trace_requested = (
        ("--trace-debug" in startup_argv)
        or ("--trace-call-debug" in startup_argv)
        or _env_truthy(startup_env.get("CODEX_TRACE_CALL_DEBUG"))
    )
    trace_debug_requested = _trace_debug_logging_requested(startup_argv, startup_env)
    if trace_debug_requested:
        startup_env["CODEX_LOG_DEBUG"] = "1"
        from apps.backend.runtime import logging as runtime_logging  # type: ignore

        runtime_logging.setup_logging(level="DEBUG")
    if call_trace_requested:
        from apps.backend.runtime.diagnostics import call_trace as _call_trace  # type: ignore

        max_per_func = _parse_trace_max(startup_argv)
        _call_trace.enable(max_calls_per_func=max_per_func)
except Exception:
    # Never block startup because of tracing/logging issues, but don't fail silently.
    _report_run_api_exception(
        sys.exc_info()[1] or RuntimeError("unknown trace-debug bootstrap failure"),
        message="startup: failed to bootstrap trace-debug logging/call-trace",
        where="interfaces.api.run_api.trace_bootstrap",
    )

try:
    from colorama import Fore, Style  # type: ignore

    def color_cyan(s: str) -> str: return Fore.CYAN + s + Style.RESET_ALL
except Exception:  # pragma: no cover - optional dependency missing

    def color_cyan(s: str) -> str: return s

# Install global exception hooks as early as possible so any startup errors are dumped
try:
    from apps.backend.runtime.diagnostics.exception_hook import install_exception_hooks as _install_exc_hooks
    _EXC_LOG_PATH = _install_exc_hooks(log_dir=str(CODEX_ROOT / 'logs'))
except Exception as exc:
    _report_run_api_exception(
        exc,
        message="startup: failed to install exception hooks",
        where="interfaces.api.run_api.install_exception_hooks",
    )
    _EXC_LOG_PATH = None


_initialized = False
_RUNTIME_NAMESPACE: Optional[Any] = None
_APP: Optional[FastAPI] = None
_WINDOWS_POWER_THROTTLING_ATTEMPTED = False
_APP_MODE_PROFILE_ENV_KEY = "CODEX_APP_MODE_PROFILE"
_APP_MODE_PROFILE_DEV_SERVICE = "dev_service"
_APP_MODE_PROFILE_EMBEDDED = "embedded"
_APP_MODE_PROFILE_CHOICES: tuple[str, ...] = (
    _APP_MODE_PROFILE_DEV_SERVICE,
    _APP_MODE_PROFILE_EMBEDDED,
)


def _resolve_app_mode_profile(env: Mapping[str, str]) -> str:
    raw_value = str(
        env.get(_APP_MODE_PROFILE_ENV_KEY, _APP_MODE_PROFILE_DEV_SERVICE)
        or _APP_MODE_PROFILE_DEV_SERVICE
    ).strip().lower()
    if raw_value in _APP_MODE_PROFILE_CHOICES:
        return raw_value
    allowed = ", ".join(_APP_MODE_PROFILE_CHOICES)
    raise RuntimeError(
        f"Invalid {_APP_MODE_PROFILE_ENV_KEY}={raw_value!r}. Allowed values: {allowed}."
    )


def _assert_embedded_dist_contract(ui_dist_dir: Path) -> None:
    index_path = ui_dist_dir / "index.html"
    assets_dir = ui_dist_dir / "assets"
    if not ui_dist_dir.is_dir():
        raise RuntimeError(
            "Embedded app mode requires a built frontend package at "
            f"{ui_dist_dir}. Run 'npm run build' in apps/interface."
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
    for asset_name in (js_assets[0].name, css_assets[0].name):
        if f"assets/{asset_name}" not in index_html:
            missing_refs.append(asset_name)
    if missing_refs:
        raise RuntimeError(
            "Embedded app mode packaging contract violation: index.html does not reference "
            f"asset(s): {', '.join(missing_refs)}."
        )


def ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return

    # Configure root logging so engine/runtime INFO logs are visible in console
    try:
        from apps.backend.runtime import logging as runtime_logging

        runtime_logging.setup_logging(level="INFO")
    except Exception as exc:
        raise RuntimeError("backend logging bootstrap failed in ensure_initialized()") from exc

    _initialized = True


def _try_disable_windows_power_throttling() -> None:
    """Disable Windows process execution-speed throttling (EcoQoS background throttle).

    This targets focus/background-driven throttling symptoms where GPU workloads appear to
    stall while the API process remains alive. It is Windows-only and best-effort.
    """

    global _WINDOWS_POWER_THROTTLING_ATTEMPTED
    if _WINDOWS_POWER_THROTTLING_ATTEMPTED:
        return
    _WINDOWS_POWER_THROTTLING_ATTEMPTED = True

    if os.name != "nt":
        return

    try:
        import ctypes
        from ctypes import wintypes

        # PROCESS_INFORMATION_CLASS::ProcessPowerThrottling (enum order from processthreadsapi.h).
        process_power_throttling = 4
        process_power_throttling_current_version = 1
        process_power_throttling_execution_speed = 0x1

        class _PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [
                ("Version", wintypes.DWORD),
                ("ControlMask", wintypes.DWORD),
                ("StateMask", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE

        if not hasattr(kernel32, "SetProcessInformation"):
            _emit_run_api_message(
                "startup: SetProcessInformation is unavailable on this Windows runtime; cannot disable process power throttling.",
                level=logging.WARNING,
            )
            return

        set_process_information = kernel32.SetProcessInformation
        set_process_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        set_process_information.restype = wintypes.BOOL

        state = _PROCESS_POWER_THROTTLING_STATE(
            Version=process_power_throttling_current_version,
            ControlMask=process_power_throttling_execution_speed,
            StateMask=0,  # Disable execution-speed throttling for this process.
        )
        ok = bool(
            set_process_information(
                get_current_process(),
                process_power_throttling,
                ctypes.byref(state),
                ctypes.sizeof(state),
            )
        )
        if not ok:
            win_err = int(ctypes.get_last_error())
            win_err_text = str(ctypes.WinError(win_err))
            _emit_run_api_message(
                "startup: failed to disable Windows process power throttling",
                level=logging.WARNING,
                error=win_err,
                detail=win_err_text,
            )
            return
        _emit_run_api_message("startup: disabled Windows process power throttling (execution_speed).")
    except Exception as exc:
        _report_run_api_exception(
            exc,
            message="startup: failed to apply Windows power-throttling patch",
            where="interfaces.api.run_api.windows_power_throttling",
        )

_UVICORN_ACCESS_NOISE_PREFIXES = (
    "/api/tools/convert-gguf/",
    "/api/tools/merge-safetensors/",
)
_UVICORN_ACCESS_NOISE_FILTER_INSTALLED = False


class _SuppressUvicornAccessNoiseFilter(logging.Filter):
    def __init__(self, suppress_path_prefixes: Optional[List[str]] = None) -> None:
        super().__init__()
        self._prefixes = tuple(suppress_path_prefixes or [])

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - depends on uvicorn internals
        if not self._prefixes:
            return True

        try:
            path: Optional[str] = None
            args = getattr(record, "args", None)
            if isinstance(args, tuple) and len(args) >= 3:
                path = str(args[2])
            elif isinstance(args, dict):
                raw = args.get("path") or args.get("raw_path")
                if raw is not None:
                    path = str(raw)

            if path is None:
                request_line = getattr(record, "request_line", None)
                if isinstance(request_line, str):
                    parts = request_line.split(" ")
                    if len(parts) >= 2:
                        path = parts[1]

            if path is None:
                msg = record.getMessage()
                for prefix in self._prefixes:
                    if prefix in msg:
                        return False
                return True

            path_only = path.split("?", 1)[0]
            return not path_only.startswith(self._prefixes)
        except Exception:
            return True


def _install_uvicorn_access_noise_filter() -> None:
    """Suppress noisy uvicorn access logs for high-frequency polling endpoints.

    The UI polls some tool endpoints (e.g. GGUF conversion and safetensors merge progress).
    Uvicorn logs every request at INFO, flooding the console during long-running jobs.
    """
    global _UVICORN_ACCESS_NOISE_FILTER_INSTALLED
    if _UVICORN_ACCESS_NOISE_FILTER_INSTALLED:
        return

    allow_tools = os.getenv("CODEX_UVICORN_ACCESS_LOG_TOOLS", "").strip().lower() in {"1", "true", "yes", "on"}
    if allow_tools:
        return

    logger = logging.getLogger("uvicorn.access")
    logger.addFilter(_SuppressUvicornAccessNoiseFilter(list(_UVICORN_ACCESS_NOISE_PREFIXES)))
    _UVICORN_ACCESS_NOISE_FILTER_INSTALLED = True


# This module is commonly loaded via `python -m uvicorn --factory ...:create_api_app`.
# Install the access-log filter at import time so it applies even when uvicorn is launched
# via CLI (no custom log_config passed).
_install_uvicorn_access_noise_filter()


def port_free(port: int, host: str = '0.0.0.0') -> bool:
    """Return True if a port looks free across common bind targets.

    This check is intentionally conservative to avoid split-brain situations
    where something is already bound on IPv6 loopback (::1) while we only test
    IPv4 (0.0.0.0). That exact setup can make `localhost` resolve to the wrong
    service with *no* obvious bind error.

    We treat the port as "busy" if any of these binds fail with EADDRINUSE:
    - IPv4 wildcard + loopback
    - IPv6 wildcard + loopback (when supported)
    """

    def _can_bind(family: int, bind_host: str) -> bool:
        try:
            with closing(socket.socket(family, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if family == socket.AF_INET6:
                    addr = (bind_host, port, 0, 0)
                else:
                    addr = (bind_host, port)
                s.bind(addr)
                return True
        except OSError as exc:
            # Ignore unsupported address families on systems without IPv6.
            if getattr(exc, "errno", None) in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
                return True
            return False

    bind_targets: list[tuple[int, str]] = []
    host_norm = str(host or "").strip().lower()
    if host_norm in {"", "0.0.0.0", "0", "*"}:
        bind_targets.extend([(socket.AF_INET, "0.0.0.0"), (socket.AF_INET, "127.0.0.1")])
    elif host_norm == "localhost":
        bind_targets.append((socket.AF_INET, "127.0.0.1"))
    else:
        bind_targets.append((socket.AF_INET, host))

    # Also check IPv6 loopback/wildcard (when available).
    bind_targets.extend([(socket.AF_INET6, "::"), (socket.AF_INET6, "::1")])

    return all(_can_bind(fam, h) for fam, h in bind_targets)


def scan_range(r: Tuple[int, int], host: str = '0.0.0.0') -> Optional[int]:
    start, end = int(r[0]), int(r[1])
    for p in range(start, end + 1):
        if port_free(p, host):
            return p
    return None


def pick_api_port_simple(base: int, host: str = '0.0.0.0') -> Tuple[int, bool]:
    # Try base -> base+10000 -> base+20000
    for i, candidate in enumerate((base, base + 10000, base + 20000)):
        if candidate < 1 or candidate > 65535:
            continue
        if port_free(candidate, host):
            return candidate, (i != 0)
    raise RuntimeError(f'No free API port among {base}, {base+10000}, {base+20000}')


def banner(port: int) -> None:
    msg = (
        "\n"
        "==============================================\n"
        "  PORT GUARD ACTIVATED — API Fallback        \n"
        "==============================================\n"
        f" Using API port {port}.                       \n"
        " Tip: set API_PORT or free blocked range.     \n"
        "==============================================\n"
    )
    _emit_run_api_message(color_cyan(msg))


class _DummyRequest:
    def __init__(self, username: str = "api") -> None:
        self.username = username


def build_app(*, app_mode_profile: str | None = None) -> FastAPI:
    ensure_initialized()

    # Native parameter helpers (replace legacy _txt2img/_img2img parsers)
    from apps.backend.services import param_utils as _p

    # Exception hooks for asyncio + HTTP middleware to dump unhandled route exceptions
    lifespan = None
    dump_current_exception = None
    try:
        from apps.backend.runtime.diagnostics.exception_hook import (
            attach_asyncio as _attach_asyncio,
            dump_current_exception as _dump_current_exception,
        )

        dump_current_exception = _dump_current_exception

        @asynccontextmanager
        async def _lifespan(app: FastAPI):  # pragma: no cover
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.get_event_loop()
                _attach_asyncio(loop)
            except Exception as exc:  # noqa: BLE001
                _report_run_api_exception(
                    exc,
                    message="startup: failed to attach asyncio exception hook",
                    where="interfaces.api.run_api.attach_asyncio_hook",
                )
            yield

        lifespan = _lifespan
    except Exception:
        lifespan = None
        dump_current_exception = None

    app = FastAPI(lifespan=lifespan) if lifespan is not None else FastAPI()

    # middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
    )

    media = MediaService()
    live_preview = LivePreviewService()
    # Native options facade (JSON-backed). Import early so helpers are available
    # to any route or startup function defined below.
    from apps.backend.services.options_store import (
        get_value as _opts_get,
        set_values_if_revision as _opts_set_many_if_revision,
        save_values as _opts_save_native,
        get_snapshot as _opts_snapshot,
        load_values as _opts_load_native,
    )
    _ui_dist_dir = CODEX_ROOT / "apps" / "interface" / "dist"
    resolved_app_mode_profile = _resolve_app_mode_profile(
        {_APP_MODE_PROFILE_ENV_KEY: app_mode_profile}
        if app_mode_profile is not None
        else os.environ
    )
    if dump_current_exception is not None:
        @app.middleware('http')
        async def _errors_middleware(request, call_next):  # type: ignore[no-untyped-def]  # pragma: no cover
            try:
                return await call_next(request)
            except Exception:
                dump_current_exception(
                    where="http",
                    context={"path": str(getattr(request, "url", "")), "method": getattr(request, "method", "")},
                )
                raise

    # Settings registry (hardcoded dataclasses/enums via codegen)
    try:
        # Generated by the internal settings registry generator.
        from apps.backend.interfaces.schemas.settings_registry import (  # type: ignore
            schema_to_json as _schema_hardcoded,
            field_index as _field_index,
            SettingType as _SettingType,
        )
        _settings_registry_ok = True
    except Exception as _e:  # pragma: no cover - optional during transition
        _emit_run_api_message(
            "settings: registry not available",
            level=logging.WARNING,
            reason=str(_e),
        )
        _settings_registry_ok = False
        _schema_hardcoded = None

        def _field_index() -> dict[str, Any]:
            return {}

        _SettingType = None

    # Load saved settings on startup and normalize them against the settings registry.
    def _apply_saved_settings() -> None:
        if not _settings_registry_ok:
            return
        saved = _opts_load_native()
        if not isinstance(saved, dict) or not saved:
            return
        idx = _field_index()
        revision_key = options_store.OPTIONS_REVISION_KEY
        normalized_revision = options_store.get_revision(saved)
        # Validate and normalize persisted values against schema, then re-save
        changed = False
        dropped: list[str] = []
        for k in list(saved.keys()):
            if k == revision_key:
                continue
            f = idx.get(k)
            if not f:
                dropped.append(k)
                saved.pop(k, None)
                changed = True
                continue
            try:
                if getattr(f, 'choices', None) and isinstance(f.choices, list) and saved[k] not in f.choices:
                    if k == "codex_attention_backend":
                        allowed = ", ".join(str(choice) for choice in f.choices)
                        raise RuntimeError(
                            "Invalid persisted setting codex_attention_backend="
                            f"'{saved[k]}' in {options_store.SETTINGS_PATH}. "
                            f"Allowed: {allowed}. "
                            "Edit settings_values.json and set codex_attention_backend to an allowed value.",
                        )
                    dropped.append(k)
                    saved.pop(k, None)
                    changed = True
                    continue
                if getattr(f, 'type', None) in (_SettingType.SLIDER, _SettingType.NUMBER):
                    v = float(saved[k])
                    if not math.isfinite(v):
                        raise ValueError(f"non-finite numeric value for {k}: {v!r}")
                    lo = getattr(f, 'min', None)
                    hi = getattr(f, 'max', None)
                    if isinstance(lo, (int, float)) and v < lo:
                        saved[k] = lo
                        changed = True
                    if isinstance(hi, (int, float)) and v > hi:
                        saved[k] = hi
                        changed = True
                if getattr(f, 'type', None) == _SettingType.CHECKBOX:
                    parsed_checkbox = parse_bool_value(saved[k], field=f"settings_values.{k}")
                    if not isinstance(saved[k], bool) or parsed_checkbox is not saved[k]:
                        saved[k] = parsed_checkbox
                        changed = True
            except Exception:
                if k == "codex_attention_backend" or getattr(f, 'type', None) == _SettingType.CHECKBOX:
                    raise
                dropped.append(k)
                saved.pop(k, None)
                changed = True
                continue
        previous_revision = saved.get(revision_key)
        if previous_revision != normalized_revision:
            saved[revision_key] = normalized_revision
            changed = True
        if changed:
            if dropped:
                _emit_run_api_message(
                    "settings: dropped invalid/unknown keys from settings_values.json",
                    level=logging.WARNING,
                    keys=", ".join(sorted(set(dropped))),
                )
            _opts_save_native(saved)

    # Apply saved settings early (after modules init) before serving
    try:
        _apply_saved_settings()
    except Exception as e:  # pragma: no cover
        _report_run_api_exception(
            e,
            message="settings: failed to validate saved settings",
            where="interfaces.api.run_api.settings_validation",
        )
        raise

    # Honour pipeline debug env flag
    _apply_pipeline_debug_flag()

    # Register routers
    app.include_router(system.build_router(app_version=options_store.get_snapshot().as_dict().get("app_version", "")))
    app.include_router(settings.build_router(
        codex_root=CODEX_ROOT,
        settings_registry_ok=_settings_registry_ok,
        schema_hardcoded=_schema_hardcoded,
        field_index=_field_index,
        opts_load_native=_opts_load_native,
    ))
    app.include_router(ui.build_router(
        codex_root=CODEX_ROOT,
        model_api=model_api,
    ))
    app.include_router(models.build_router(
        model_api=model_api,
    ))
    app.include_router(paths.build_router(codex_root=CODEX_ROOT))
    app.include_router(options.build_router(
        opts_load_native=_opts_load_native,
        opts_snapshot=_opts_snapshot,
        opts_set_many_if_revision=_opts_set_many_if_revision,
        settings_registry_ok=_settings_registry_ok,
        field_index=_field_index,
        setting_type=_SettingType,
    ))
    app.include_router(tasks.build_router(codex_root=CODEX_ROOT, backend_state=backend_state))
    app.include_router(tests.build_router())
    app.include_router(tools.build_router(codex_root=CODEX_ROOT))
    app.include_router(upscale.build_router(
        codex_root=CODEX_ROOT,
        opts_get=_opts_get,
        generation_provenance=_generation_provenance,
        save_generated_images=_save_generated_images,
    ))
    app.include_router(supir.build_router(
        codex_root=CODEX_ROOT,
        opts_get=_opts_get,
        generation_provenance=_generation_provenance,
        save_generated_images=_save_generated_images,
    ))
    app.include_router(generation.build_router(
        codex_root=CODEX_ROOT,
        media=media,
        live_preview=live_preview,
        opts_get=_opts_get,
        opts_snapshot=_opts_snapshot,
        generation_provenance=_generation_provenance,
        save_generated_images=_save_generated_images,
        param_utils=_p,
    ))

    # Serve built UI (Vite build) if present, with SPA fallback
    class SPAStaticFiles(StaticFiles):  # type: ignore[misc]
        async def get_response(self, path: str, scope):  # type: ignore[override]
            try:
                return await super().get_response(path, scope)
            except Exception as exc:
                try:
                    from starlette.exceptions import HTTPException as StarletteHTTPException  # type: ignore

                    if isinstance(exc, StarletteHTTPException) and exc.status_code == 404:
                        return await super().get_response("index.html", scope)
                except Exception:
                    pass
                raise

    # Mount UI dist after API routes when embedded mode is explicitly selected.
    if resolved_app_mode_profile == _APP_MODE_PROFILE_EMBEDDED:
        _assert_embedded_dist_contract(Path(_ui_dist_dir))
        app.mount("/", SPAStaticFiles(directory=str(_ui_dist_dir), html=True), name="ui")
    elif resolved_app_mode_profile == _APP_MODE_PROFILE_DEV_SERVICE:
        _emit_run_api_message(
            "startup: app mode profile selected; skipping embedded SPA mount.",
            profile=_APP_MODE_PROFILE_DEV_SERVICE,
        )
    else:  # pragma: no cover - guarded by _resolve_app_mode_profile
        raise RuntimeError(
            f"Unhandled {_APP_MODE_PROFILE_ENV_KEY} value {resolved_app_mode_profile!r}."
        )

    return app


def _bootstrap_runtime(argv: Sequence[str], env: Mapping[str, str], settings: Mapping[str, Any]) -> Any:
    global _RUNTIME_NAMESPACE
    if _RUNTIME_NAMESPACE is not None:
        return _RUNTIME_NAMESPACE

    def _allocator_backend(raw_alloc_conf: str) -> str | None:
        backend: str | None = None
        for raw_entry in str(raw_alloc_conf or "").split(","):
            token = raw_entry.strip()
            if not token:
                continue
            if ":" not in token:
                raise RuntimeError(
                    "Invalid PYTORCH_CUDA_ALLOC_CONF entry "
                    f"{token!r}: expected 'key:value' format."
                )
            key, value = token.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if not key or not value:
                raise RuntimeError(
                    "Invalid PYTORCH_CUDA_ALLOC_CONF entry "
                    f"{token!r}: expected non-empty 'key:value' parts."
                )
            if key == "backend":
                if backend is not None:
                    raise RuntimeError(
                        "Invalid PYTORCH_CUDA_ALLOC_CONF: multiple 'backend' entries found. "
                        "Use exactly one backend directive."
                    )
                backend = value
        return backend

    raw_alloc_conf = str(env.get("PYTORCH_CUDA_ALLOC_CONF", "") or "").strip()
    ns, runtime_config = config_args.initialize(
        argv=argv,
        env=env,
        settings=settings,
        strict=True,
    )
    ns.trace_inference_debug = _env_truthy(env.get("CODEX_TRACE_INFERENCE_DEBUG"))
    ns.trace_load_patch_debug = _env_truthy(env.get("CODEX_TRACE_LOAD_PATCH_DEBUG"))
    allocator_backend = _allocator_backend(raw_alloc_conf)
    _emit_run_api_message(
        "startup: PYTORCH_CUDA_ALLOC_CONF",
        alloc_conf=raw_alloc_conf or "<unset>",
        backend=allocator_backend or "<default>",
        cuda_malloc_flag=bool(getattr(ns, "cuda_malloc", False)),
    )
    _emit_run_api_message(
        "startup: attention backend bootstrap",
        requested_backend=str(getattr(ns, "attention_backend", "<unset>") or "<unset>"),
        requested_sdpa_policy=str(getattr(ns, "attention_sdpa_policy", "<unset>") or "<unset>"),
        resolved_backend=str(getattr(runtime_config.attention, "backend", "<unknown>")),
        flash=bool(getattr(runtime_config.attention, "enable_flash", False)),
        mem_efficient=bool(getattr(runtime_config.attention, "enable_mem_efficient", False)),
    )
    _try_disable_windows_power_throttling()
    # Publish resolved bootstrap values (after CLI/env/settings precedence) without mutating os.environ.
    try:
        from apps.backend.infra.config.bootstrap_env import set_bootstrap_env as _set_bootstrap_env

        if getattr(ns, "debug_preview_factors", False):
            _set_bootstrap_env("CODEX_DEBUG_PREVIEW_FACTORS", "1")
        if getattr(ns, "trace_contract", False):
            _set_bootstrap_env("CODEX_TRACE_CONTRACT", "1")
        if getattr(ns, "trace_profiler", False):
            _set_bootstrap_env("CODEX_TRACE_PROFILER", "1")
            _set_bootstrap_env("CODEX_PROFILE", "1")
        if _env_truthy(env.get("CODEX_TRACE_INFERENCE_DEBUG")):
            _set_bootstrap_env("CODEX_TRACE_INFERENCE_DEBUG", "1")
        if _env_truthy(env.get("CODEX_TRACE_LOAD_PATCH_DEBUG")):
            _set_bootstrap_env("CODEX_TRACE_LOAD_PATCH_DEBUG", "1")
        if getattr(ns, "trace_debug", False) or _env_truthy(env.get("CODEX_TRACE_INFERENCE_DEBUG")) or _env_truthy(env.get("CODEX_TRACE_LOAD_PATCH_DEBUG")):
            _set_bootstrap_env("CODEX_LOG_DEBUG", "1")
        mode = getattr(ns, "lora_apply_mode", None)
        if mode is not None:
            # Only publish non-default values. Publishing defaults would pin global state
            # and prevent per-test env overrides from taking effect.
            from apps.backend.infra.config.lora_apply_mode import DEFAULT_LORA_APPLY_MODE

            mode_value = str(mode).strip()
            if mode_value and mode_value != DEFAULT_LORA_APPLY_MODE.value:
                _set_bootstrap_env("CODEX_LORA_APPLY_MODE", mode_value)
        lora_merge_mode = getattr(ns, "lora_merge_mode", None)
        if lora_merge_mode is not None:
            from apps.backend.infra.config.lora_merge_mode import DEFAULT_LORA_MERGE_MODE

            mode_value = str(lora_merge_mode).strip()
            if mode_value and mode_value != DEFAULT_LORA_MERGE_MODE.value:
                _set_bootstrap_env("CODEX_LORA_MERGE_MODE", mode_value)
        lora_refresh_signature = getattr(ns, "lora_refresh_signature", None)
        if lora_refresh_signature is not None:
            from apps.backend.infra.config.lora_refresh_signature import DEFAULT_LORA_REFRESH_SIGNATURE_MODE

            signature_value = str(lora_refresh_signature).strip()
            if signature_value and signature_value != DEFAULT_LORA_REFRESH_SIGNATURE_MODE.value:
                _set_bootstrap_env("CODEX_LORA_REFRESH_SIGNATURE", signature_value)
    except Exception as exc:
        raise RuntimeError("startup: failed while publishing resolved bootstrap environment values.") from exc
    mem_management.reinitialize(runtime_config)
    # Pre-warm model inventory at process bootstrap so `/api/models/inventory`
    # is already hot when the UI first loads quicksettings. This avoids paying
    # the full filesystem scan cost on the first UI request.
    try:
        from apps.backend.inventory import cache as _inv_cache
        inv = _inv_cache.refresh()
        _emit_run_api_message(
            "inventory: initialized at startup",
            logger="inventory",
            vaes=len(inv.get("vaes", [])),
            text_encoders=len(inv.get("text_encoders", [])),
            loras=len(inv.get("loras", [])),
            wan22_gguf=len(inv.get("wan22", [])),
            metadata=len(inv.get("metadata", [])),
        )
    except Exception as e:
        _emit_run_api_message(
            "inventory: failed to initialize at startup",
            logger="inventory",
            level=logging.WARNING,
            reason=str(e),
        )
    _RUNTIME_NAMESPACE = ns
    return ns


def _enable_trace_debug(ns: Any) -> None:
    try:
        call_trace_requested = bool(getattr(ns, "trace_debug", False))
        trace_inference_debug = bool(getattr(ns, "trace_inference_debug", False))
        trace_load_patch_debug = bool(getattr(ns, "trace_load_patch_debug", False))
        trace_debug_requested = bool(
            call_trace_requested
            or trace_inference_debug
            or trace_load_patch_debug
        )
        if trace_debug_requested:
            from apps.backend.runtime import logging as runtime_logging  # type: ignore

            os.environ["CODEX_LOG_DEBUG"] = "1"
            runtime_logging.setup_logging(level="DEBUG")
        if call_trace_requested:
            from apps.backend.runtime.diagnostics import call_trace as _call_trace  # type: ignore

            _call_trace.enable(max_calls_per_func=getattr(ns, "trace_debug_max_per_func", None))
    except Exception as exc:
        _report_run_api_exception(
            exc,
            message="startup: failed to bootstrap trace-debug logging/call-trace",
            where="interfaces.api.run_api.enable_trace_debug",
        )


def create_api_app(*, argv: Optional[Sequence[str]] = None, env: Optional[Mapping[str, str]] = None) -> FastAPI:
    argv_seq = list(argv or [])
    env_map = env or os.environ
    settings = options_store.load_values()
    ns = _bootstrap_runtime(argv_seq, env_map, settings)
    _enable_trace_debug(ns)
    ensure_initialized()
    app_mode_profile = _resolve_app_mode_profile(env_map)
    # Build a fresh app each time to avoid stale/None globals under factory mode
    app = build_app(app_mode_profile=app_mode_profile)
    if app is None:
        raise RuntimeError("build_app() returned None")
    global _APP
    _APP = app
    return app

def main(argv: Optional[Sequence[str]] = None) -> None:
    try:
        ensure_initialized()
    except Exception as exc:
        _report_run_api_exception(
            exc,
            message="startup: failed to initialize backend logging",
            where="interfaces.api.run_api.main.ensure_initialized",
        )
        raise SystemExit(1) from exc

    host = '0.0.0.0'
    override = os.environ.get('API_PORT_OVERRIDE')
    used_fallback = False
    port: Optional[int] = None
    if override:
        try:
            candidate = int(override)
        except ValueError:
            candidate = None
        if candidate is not None:
            # If chosen override busy, hop by +10000
            for c in (candidate, candidate + 10000, candidate + 20000):
                if c < 1 or c > 65535:
                    continue
                if port_free(c, host):
                    port = c
                    used_fallback = (c != candidate)
                    break
        else:
            override = None  # force fallback logic
    if port is None:
        try:
            # default base 7850
            port, used_fallback = pick_api_port_simple(7850, host)
        except RuntimeError as e:
            _emit_run_api_message("[PORT GUARD]", level=logging.ERROR, detail=str(e))
            raise SystemExit(1)

    if used_fallback:
        banner(port)

    try:
        argv_seq = list(argv) if argv is not None else sys.argv[1:]
        api_app = create_api_app(argv=argv_seq, env=os.environ)
    except Exception as exc:
        _report_run_api_exception(
            exc,
            message="[INIT]",
            where="interfaces.api.run_api.main.create_api_app",
        )
        raise SystemExit(1) from exc

    log_config = build_backend_uvicorn_log_config(
        suppress_access_prefixes=list(_UVICORN_ACCESS_NOISE_PREFIXES),
        access_filter_factory="apps.backend.interfaces.api.run_api._SuppressUvicornAccessNoiseFilter",
        level="INFO",
    )
    uvicorn.run(api_app, host=host, port=port, log_level='info', log_config=log_config)


if __name__ == '__main__':
    main()
