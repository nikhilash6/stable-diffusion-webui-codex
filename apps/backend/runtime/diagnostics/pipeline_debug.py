"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Lightweight pipeline debug toggle and decorator helpers.
Provides a process-wide debug flag, a best-effort env-driven enable hook (`CODEX_PIPELINE_DEBUG`), and a decorator that emits enter/exit/error tracing.

Symbols (top-level; keep in sync; no ghosts):
- `PIPELINE_DEBUG_ENABLED` (constant): Global on/off switch for pipeline debug logging.
- `set_pipeline_debug` (function): Enables/disables pipeline debug logging.
- `log` (function): Logs a message when pipeline debug is enabled.
- `apply_env_flag` (function): Reads `CODEX_PIPELINE_DEBUG` and toggles pipeline debug accordingly.
- `pipeline_trace` (function): Decorator helper that emits pipeline-debug logs and optional timeline spans.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import os
from functools import wraps
from typing import Any, Callable, TypeVar, cast


logger = get_backend_logger("backend.pipeline.debug")

PIPELINE_DEBUG_ENABLED = False

F = TypeVar("F", bound=Callable[..., Any])


def set_pipeline_debug(enabled: bool) -> None:
    global PIPELINE_DEBUG_ENABLED
    PIPELINE_DEBUG_ENABLED = bool(enabled)
    logger.info("pipeline debug %s", "ativado" if PIPELINE_DEBUG_ENABLED else "desativado")


def log(message: str) -> None:
    if PIPELINE_DEBUG_ENABLED:
        logger.info(message)


def apply_env_flag(raw_value: str | None = None) -> None:
    raw = raw_value if raw_value is not None else os.getenv("CODEX_PIPELINE_DEBUG", "0")
    normalized = str(raw).strip().lower()
    set_pipeline_debug(normalized in {"1", "true", "yes", "on"})


def pipeline_trace(func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        trace_name = f"{func.__module__}.{func.__qualname__}"
        debug_enabled = bool(PIPELINE_DEBUG_ENABLED)
        if debug_enabled:
            logger.info("[pipeline] enter %s", trace_name)

        try:
            from apps.backend.runtime.diagnostics.timeline import timeline
        except Exception:
            timeline = None

        timeline_enabled = timeline is not None and bool(getattr(timeline, "enabled", False))

        if not timeline_enabled:
            try:
                result = func(*args, **kwargs)
            except Exception:
                if debug_enabled:
                    logger.exception("[pipeline] error %s", trace_name)
                raise
            if debug_enabled:
                logger.info("[pipeline] exit %s", trace_name)
            return result

        stage_name = "pipeline"
        capture_active = getattr(timeline, "_active_capture", None) is not None

        if not capture_active:
            with timeline.capture(name="txt2img_pipeline"):
                timeline.enter(stage_name, trace_name)
                try:
                    result = func(*args, **kwargs)
                except Exception:
                    if debug_enabled:
                        logger.exception("[pipeline] error %s", trace_name)
                    raise
                finally:
                    timeline.exit(stage_name, trace_name)
                if debug_enabled:
                    logger.info("[pipeline] exit %s", trace_name)
                return result

        timeline.enter(stage_name, trace_name)
        try:
            result = func(*args, **kwargs)
        except Exception:
            if debug_enabled:
                logger.exception("[pipeline] error %s", trace_name)
            raise
        finally:
            timeline.exit(stage_name, trace_name)
        if debug_enabled:
            logger.info("[pipeline] exit %s", trace_name)
        return result

    return cast(F, wrapper)


__all__ = [
    "PIPELINE_DEBUG_ENABLED",
    "set_pipeline_debug",
    "log",
    "pipeline_trace",
    "apply_env_flag",
]
