"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Global Python function-call tracer for deep Codex debugging.
Implements a sys/thread profile hook that logs function calls under configurable module scopes with indentation and per-function call caps, and
provides env-driven enable/disable helpers to avoid flooding logs by default.

Symbols (top-level; keep in sync; no ghosts):
- `_profiler` (function): Profile hook used by `sys.setprofile` / `threading.setprofile` (logs `call`/`return` events).
- `_set_max_per_func` (function): Configures the per-function call cap (0 disables the cap).
- `_set_module_prefixes` (function): Configures module scope prefixes (`None` means trace any module).
- `_reset_counters` (function): Clears per-function call counters and “muted” notifications.
- `enable` (function): Enables global call tracing (optionally configuring caps) and installs the profiler hooks.
- `disable` (function): Disables call tracing and restores prior profiler hooks.
- `_env_trace_limit` (function): Reads the max-per-func cap from environment variables.
- `_env_trace_module_prefixes` (function): Reads module-scope prefixes from environment (`CODEX_TRACE_CALL_DEBUG_MODULE_PREFIXES`).
- `enable_from_env` (function): Enables tracing when env flags request it (launcher/API entrypoint integration).
"""

from __future__ import annotations

import os
import sys
import threading
from types import FrameType
from typing import Any, Callable, Optional, Tuple

from apps.backend.infra.config.env_flags import env_flag
from apps.backend.runtime.logging import configure_backend_root_for_call_trace, get_backend_logger

_logger = get_backend_logger("backend.calltrace")

_enabled: bool = False
_local = threading.local()
_prev_profile: Optional[Callable[..., Any]] = None

_DEFAULT_MAX_PER_FUNC = 10
_DEFAULT_MODULE_PREFIXES: tuple[str, ...] = ("apps.",)
_max_per_func: int = _DEFAULT_MAX_PER_FUNC
_module_prefixes: tuple[str, ...] | None = _DEFAULT_MODULE_PREFIXES
_call_counts: dict[Tuple[str, str], int] = {}
_muted_notified: set[Tuple[str, str]] = set()


def _set_module_prefixes(prefixes: tuple[str, ...] | None) -> None:
    global _module_prefixes
    if prefixes is None:
        _module_prefixes = None
        return
    normalized = tuple(str(prefix).strip() for prefix in prefixes if str(prefix).strip())
    _module_prefixes = normalized or _DEFAULT_MODULE_PREFIXES


def _env_trace_module_prefixes() -> tuple[str, ...] | None:
    raw = os.getenv("CODEX_TRACE_CALL_DEBUG_MODULE_PREFIXES")
    if raw is None:
        return _DEFAULT_MODULE_PREFIXES
    tokens = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not tokens:
        return _DEFAULT_MODULE_PREFIXES
    lowered = {token.lower() for token in tokens}
    if lowered & {"*", "all", "any"}:
        return None
    return tuple(tokens)


def _should_trace_module(module_name: object) -> bool:
    if not isinstance(module_name, str):
        return False
    prefixes = _module_prefixes
    if prefixes is None:
        return True
    return any(module_name.startswith(prefix) for prefix in prefixes)

def _profiler(frame: FrameType, event: str, arg: Any):  # pragma: no cover - runtime hook
    # Guard: prevent recursion while we log
    if getattr(_local, "busy", False):
        return _profiler

    if event == "call":
        try:
            mod = frame.f_globals.get("__name__", "<unknown>")

            if not _should_trace_module(mod):
                return _profiler
            _local.busy = True
            depth = getattr(_local, "depth", 0) + 1
            _local.depth = depth
            func = frame.f_code.co_name or "<unknown>"

            # Best-effort class name enrichment
            qn = func
            try:
                if "self" in frame.f_locals:
                    qn = f"{type(frame.f_locals['self']).__name__}.{func}"
                elif "cls" in frame.f_locals and hasattr(frame.f_locals["cls"], "__name__"):
                    qn = f"{frame.f_locals['cls'].__name__}.{func}"
            except Exception:
                pass

            key = (mod, qn)
            indent = " " * (depth - 1)

            if _max_per_func > 0:
                count = _call_counts.get(key, 0) + 1
                _call_counts[key] = count
                if count > _max_per_func:
                    if key not in _muted_notified:
                        _logger.debug("%sCALL %s.%s (muted after %d calls)", indent, mod, qn, _max_per_func)
                        _muted_notified.add(key)
                    return _profiler

            # Indent for readability but keep message short
            _logger.debug("%sCALL %s.%s", indent, mod, qn)
        except Exception:
            # Never raise from the profiler; keep tracing alive
            pass
        finally:
            _local.busy = False
        return _profiler
    elif event == "return":
        # Track depth to keep indentation balanced
        try:
            _local.depth = max(0, getattr(_local, "depth", 0) - 1)
        except Exception:
            _local.depth = 0
        return _profiler

    # Ignore other events (c_call/c_return, exceptions) to reduce noise
    return _profiler


def _set_max_per_func(value: Optional[int]) -> None:
    global _max_per_func
    if value is None:
        _max_per_func = _DEFAULT_MAX_PER_FUNC
        return
    try:
        numeric = int(value)
    except Exception:
        numeric = _DEFAULT_MAX_PER_FUNC
    _max_per_func = max(0, numeric)


def _reset_counters() -> None:
    _call_counts.clear()
    _muted_notified.clear()


def enable(*, max_calls_per_func: Optional[int] = None) -> None:
    """Enable global function-call tracing.

    Logging level must allow DEBUG for messages to be visible.
    """
    global _enabled, _prev_profile

    # Env fallback even when caller passes None (ensures UI/env overrides stick)
    if max_calls_per_func is None:
        max_calls_per_func = _env_trace_limit()
    _set_max_per_func(max_calls_per_func)
    _set_module_prefixes(_env_trace_module_prefixes())
    if _enabled:
        _reset_counters()
        _logger.debug(
            "call-trace limit set to %s per function (scope=%s)",
            "unlimited" if _max_per_func == 0 else _max_per_func,
            "all-modules" if _module_prefixes is None else ",".join(_module_prefixes),
        )
        return

    # Avoid tracing our own tracing/logging internals by bumping this logger level
    # if the root level is very low. We still emit debug from this logger.
    try:
        configure_backend_root_for_call_trace()
    except Exception:
        pass

    _reset_counters()

    _prev_profile = sys.getprofile()
    sys.setprofile(_profiler)
    threading.setprofile(_profiler)
    _enabled = True
    _logger.debug(
        "call-trace enabled (sys.setprofile, limit=%s per function, scope=%s)",
        "unlimited" if _max_per_func == 0 else _max_per_func,
        "all-modules" if _module_prefixes is None else ",".join(_module_prefixes),
    )


def disable() -> None:  # pragma: no cover - runtime hook
    global _enabled, _prev_profile
    if not _enabled:
        return
    # Restore previous profiler if present
    sys.setprofile(_prev_profile)
    threading.setprofile(_prev_profile)
    _prev_profile = None
    _enabled = False
    _reset_counters()
    _logger.debug("call-trace disabled")


def _env_trace_limit() -> Optional[int]:
    raw = os.getenv("CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC")
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def enable_from_env() -> None:
    """Enable when CODEX_TRACE_CALL_DEBUG=1 (or truthy)."""
    if env_flag("CODEX_TRACE_CALL_DEBUG", default=False):
        enable(max_calls_per_func=_env_trace_limit())


__all__ = ["enable", "disable", "enable_from_env"]
