"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Lightweight tracing helpers for torch ops and scoped sections.
Implements an env-driven trace mode that patches `torch.nn.Module.to` to emit debug logs, plus a context manager for scoped tracing.

Symbols (top-level; keep in sync; no ghosts):
- `enable` (function): Enables trace mode (reads `CODEX_TRACE_TORCH` and `CODEX_TRACE_LIMIT`) and patches `torch.nn.Module.to`.
- `disable` (function): Disables trace mode and restores the original `torch.nn.Module.to`.
- `event` (function): Emits a trace log event when tracing is enabled.
- `trace_section` (contextmanager): Context manager that enables tracing for the duration of a section and logs timing metadata.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import os
import time
from contextlib import contextmanager
from typing import Any

import torch

_log = get_backend_logger("backend.trace")
_enabled: bool = False
_limit: int = 500
_events: int = 0
_stack: list[str] = []
_orig_module_to = torch.nn.Module.to


def _maybe_log(msg: str, *args: Any) -> None:
    global _events
    if not _enabled:
        return
    if _events >= _limit:
        return
    _log.debug(msg, *args)
    _events += 1


def _format_device(dev: Any) -> str:
    try:
        if hasattr(dev, "type"):
            dtype = getattr(dev, "type", "")
            index = getattr(dev, "index", 0)
            return f"{dtype}:{index}".rstrip(":")
        return str(dev)
    except Exception:
        return str(dev)


def _patched_module_to(self: torch.nn.Module, *args: Any, **kwargs: Any):  # type: ignore[override]
    if _enabled:
        dev = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if len(args) >= 1 and dev is None:
            dev = args[0]
        if len(args) >= 2 and dtype is None:
            dtype = args[1]
        dtype_repr = getattr(dtype, "__repr__", lambda: str(dtype))
        _maybe_log("to(): mod=%s dev=%s dtype=%s", self.__class__.__name__, _format_device(dev), dtype_repr())
    return _orig_module_to(self, *args, **kwargs)


def enable(section: str = "") -> None:
    global _enabled, _limit, _events
    _enabled = bool(int(os.environ.get("CODEX_TRACE_TORCH", "0") or "0"))
    try:
        _limit = int(os.environ.get("CODEX_TRACE_LIMIT", "500"))
    except Exception:
        _limit = 500
    _events = 0
    if not _enabled:
        return
    _maybe_log("trace: enable section=%s limit=%d", section, _limit)
    torch.nn.Module.to = _patched_module_to  # type: ignore[assignment]


def disable() -> None:
    global _enabled
    if not _enabled:
        return
    torch.nn.Module.to = _orig_module_to  # type: ignore[assignment]
    _maybe_log("trace: disable (events=%d)", _events)
    _enabled = False


def event(what: str, **meta: Any) -> None:
    if not _enabled:
        return
    if _events >= _limit:
        return
    if meta:
        _maybe_log("event: %s %s", what, meta)
    else:
        _maybe_log("event: %s", what)


@contextmanager
def trace_section(name: str):
    enable(name)
    start = time.time()
    _stack.append(name)
    try:
        yield
    finally:
        dur = (time.time() - start) * 1000.0
        event("section_end", name=name, ms=f"{dur:.2f}")
        _stack.pop()
        disable()


__all__ = [
    "disable",
    "enable",
    "event",
    "trace_section",
]
