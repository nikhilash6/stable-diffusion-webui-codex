"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Centralized backend logging setup and wrapper API (env-driven level, optional rich/tqdm integration).
Configures root logging once per interpreter, supports console/file handlers, and can wrap stream handlers to cooperate with tqdm progress
bars. Level filtering can also be controlled per-level via `CODEX_LOG_*` env vars.
Includes structured multiline rendering for dense telemetry events, a plain-message wrapper for repo-owned operational logs, and the
canonical uvicorn logging-config builder consumed by the API bootstrap seam.

Symbols (top-level; keep in sync; no ghosts):
- `BackendLoggerProxy` (class): Canonical repo-owned logger proxy that routes method-style calls through the wrapper family.
- `TqdmAwareHandler` (class): Proxy handler that cooperates with tqdm-managed progress bars.
- `_is_stream_handler` (function): Detects stream handlers including wrapped `TqdmAwareHandler`.
- `_parse_level` (function): Parses level names (including TRACE=5) and returns a logging level.
- `configure_backend_root_for_call_trace` (function): Applies the sanctioned backend-root logger mutations required by call tracing.
- `format_log_message` (function): Builds a consistent event-style log message with optional key/value context.
- `CodexLogHighlighter` (class): Regex-based Rich highlighter for structured `key=value` diagnostics.
- `get_backend_logger` (function): Returns a normalized backend logger (`backend.*`) from module or relative names.
- `emit_backend_message` (function): Canonical repo-owned human-readable operational-log emitter (supports stdlib-style message interpolation).
- `emit_backend_event` (function): Canonical global backend event emitter (single source of truth for event emission path).
- `build_backend_uvicorn_log_config` (function): Canonical server-log config builder for the uvicorn integration seam.
- `LevelFilter` (class): Env-driven log-level filter (CODEX_LOG_DEBUG/INFO/WARNING/ERROR).
- `setup_logging` (function): Idempotent root logger setup using env vars (level/format/file, optional Rich handler).
"""

from __future__ import annotations

import logging
import os
import re
import sys
import datetime
from typing import Any, Optional, Sequence

try:  # Optional; fall back to plain logs when missing
    from colorama import init as _colorama_init  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _colorama_init = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from rich.console import Console  # type: ignore
    from rich.highlighter import RegexHighlighter  # type: ignore
    from rich.logging import RichHandler  # type: ignore
    from rich.theme import Theme  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Console = None  # type: ignore[assignment]
    RegexHighlighter = None  # type: ignore[assignment]
    RichHandler = None  # type: ignore[assignment]
    Theme = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore[assignment]

if _colorama_init is not None:  # pragma: no cover - environment dependent
    _colorama_init(autoreset=True)

_CONFIGURED = False
_SAFE_LOG_TOKEN = re.compile(r"^[A-Za-z0-9._:/,-]+$")
_STRUCTURED_MEMORY_PREFIXES = ("memory_before_", "memory_after_", "memory_current_")
_STRUCTURED_FIELD_CHUNK_SIZE = 4
_STRUCTURED_INLINE_MAX_FIELDS = 6
_DEFAULT_DATEFMT = "%m/%d/%y %H:%M:%S"


if RegexHighlighter is not None:  # pragma: no cover - optional dependency
    class CodexLogHighlighter(RegexHighlighter):
        """Highlight backend diagnostic tokens (`key=value`) with stable styles."""

        base_style = "codexlog."
        highlights = [
            r"(?m)^(?:[A-Za-z_][A-Za-z0-9_.-]*\s\|\s)?(?P<event>[A-Za-z_][A-Za-z0-9_.-]*(?:\.[A-Za-z0-9_.-]+)+)\b(?=\s*(?:\||$))",
            r"(?P<tag>\[[A-Za-z0-9_:.@/-]+\])",
            r"(?<![\w.])(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)(?==)",
            r"=(?P<value>(?!torch\.[A-Za-z0-9_]+\b)(?!(?:cuda|cpu|mps|xpu)(?::\d+)?\b)(?!(?i:true|false)\b)(?![-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?\b)[^\s|]+)",
            r"=(?P<dtype>torch\.[A-Za-z0-9_]+)\b",
            r"=(?P<device>(?:cuda|cpu|mps|xpu)(?::\d+)?)\b",
            r"=(?P<bool>(?i:true|false))\b",
            r"=(?P<number>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\b",
            r"(?P<arrow>->)",
        ]
else:
    CodexLogHighlighter = None  # type: ignore[assignment,misc]


class BackendLoggerProxy:
    """Thin repo-owned logger facade that keeps callsites on the canonical wrapper seam."""

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @property
    def name(self) -> str:
        return self._logger.name

    def isEnabledFor(self, level: str | int) -> bool:
        return self._logger.isEnabledFor(_coerce_level_value(level))

    def log(
        self,
        level: str | int,
        message: str,
        /,
        *message_args: object,
        exc_info: Any = None,
        **fields: object,
    ) -> None:
        _emit_backend_message_to_logger(
            self._logger,
            message,
            *message_args,
            level=level,
            exc_info=exc_info,
            **fields,
        )

    def debug(self, message: str, /, *message_args: object, exc_info: Any = None, **fields: object) -> None:
        self.log(logging.DEBUG, message, *message_args, exc_info=exc_info, **fields)

    def info(self, message: str, /, *message_args: object, exc_info: Any = None, **fields: object) -> None:
        self.log(logging.INFO, message, *message_args, exc_info=exc_info, **fields)

    def warning(self, message: str, /, *message_args: object, exc_info: Any = None, **fields: object) -> None:
        self.log(logging.WARNING, message, *message_args, exc_info=exc_info, **fields)

    def error(self, message: str, /, *message_args: object, exc_info: Any = None, **fields: object) -> None:
        self.log(logging.ERROR, message, *message_args, exc_info=exc_info, **fields)

    def critical(self, message: str, /, *message_args: object, exc_info: Any = None, **fields: object) -> None:
        self.log(logging.CRITICAL, message, *message_args, exc_info=exc_info, **fields)

    def exception(self, message: str, /, *message_args: object, exc_info: Any = True, **fields: object) -> None:
        self.error(message, *message_args, exc_info=exc_info, **fields)


class TqdmAwareHandler(logging.Handler):
    """Proxy handler that cooperates with tqdm-managed progress bars."""

    def __init__(self, inner: logging.Handler) -> None:
        super().__init__()
        self.inner = inner

    def setFormatter(self, fmt: logging.Formatter) -> None:  # noqa: N802 (logging API)
        super().setFormatter(fmt)
        self.inner.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI integration
        if tqdm is not None and getattr(tqdm, "_instances", None):
            try:
                rendered = self.format(record)
                if RichHandler is not None and isinstance(self.inner, RichHandler):
                    timestamp = datetime.datetime.fromtimestamp(record.created).strftime("%m/%d/%y %H:%M:%S")
                    rendered = f"[{timestamp}] {record.levelname:<8s} {rendered}"
                target_stream = getattr(self.inner, "stream", sys.stderr)
                tqdm.write(rendered, file=target_stream)
                return
            except Exception:
                # Fall back to the wrapped handler on unexpected errors
                pass
        self.inner.emit(record)


def _is_stream_handler(handler: logging.Handler) -> bool:
    if isinstance(handler, logging.StreamHandler):
        return True
    return isinstance(handler, TqdmAwareHandler) and isinstance(handler.inner, logging.StreamHandler)


def _parse_level(value: Optional[str]) -> int:
    if not value:
        return logging.DEBUG
    v = value.strip().upper()
    mapping = {
        "TRACE": 5,  # custom: lower than DEBUG if someone sets it
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,
    }
    resolved = mapping.get(v, logging.DEBUG)
    logging.addLevelName(5, "TRACE")
    return resolved


def _coerce_level_value(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    return _parse_level(level)


def _env_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _should_force_plain() -> bool:
    env = os.environ.get("SD_WEBUI_NO_RICH") or os.environ.get("CODEX_LOG_NO_RICH")
    if not env:
        return False
    return env.strip().lower() in {"1", "true", "yes", "on"}


def _include_logger_name() -> bool:
    return _env_true("CODEX_LOG_INCLUDE_LOGGER_NAME", "0")


def _default_log_format(*, include_logger_name: bool) -> str:
    if include_logger_name:
        return "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s"
    return "[%(asctime)s] %(levelname)-8s %(message)s"


def _resolve_log_format(*, include_logger_name: bool) -> str:
    return os.environ.get("CODEX_LOG_FORMAT", _default_log_format(include_logger_name=include_logger_name))


def _render_log_token(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value if _SAFE_LOG_TOKEN.fullmatch(value) else repr(value)
    return repr(value)


def _render_log_fields(**fields: object) -> list[tuple[str, str]]:
    rendered_fields: list[tuple[str, str]] = []
    for key, value in fields.items():
        if value is None:
            continue
        safe_key = key if _SAFE_LOG_TOKEN.fullmatch(key) else repr(key)
        rendered_fields.append((safe_key, _render_log_token(value)))
    return rendered_fields


def _normalize_backend_logger_name(name: Optional[str]) -> str:
    if name is None:
        return "backend"
    if not isinstance(name, str):
        raise TypeError(
            "Backend logger selector must be None or a string name. "
            f"Got {type(name).__name__}."
        )

    normalized = name.strip()
    if not normalized:
        return "backend"
    normalized = normalized.strip(".")
    if not normalized:
        return "backend"

    if normalized in {"backend", "apps.backend"}:
        normalized = "backend"
    elif normalized.startswith("apps.backend."):
        normalized = "backend." + normalized[len("apps.backend.") :]
    elif not normalized.startswith("backend."):
        normalized = "backend." + normalized
    normalized = normalized.rstrip(".")
    normalized = ".".join(part for part in normalized.split(".") if part)
    return normalized or "backend"


def _resolve_backend_raw_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(_normalize_backend_logger_name(name))


def configure_backend_root_for_call_trace() -> None:
    """Apply the sanctioned backend-root logger mutations required by call tracing."""

    backend_root = _resolve_backend_raw_logger("backend")
    backend_root.setLevel(min(backend_root.level, logging.DEBUG))
    backend_root.propagate = False


def _resolved_level_name(level: str | int | None) -> str:
    if isinstance(level, int):
        return logging.getLevelName(level)
    if level is None:
        return logging.getLevelName(_parse_level(None))
    return logging.getLevelName(_parse_level(str(level)))


def format_log_message(event: str, /, **fields: object) -> str:
    """Build a consistent event-style log message.

    Output format:
    - without fields: `event`
    - with fields: `event | key=value key2='text value'`
    """

    rendered_fields = _render_log_fields(**fields)

    safe_event = event if _SAFE_LOG_TOKEN.fullmatch(event) else repr(event)
    if not rendered_fields:
        return safe_event

    has_memory_windows = any(
        key.startswith(_STRUCTURED_MEMORY_PREFIXES)
        for key, _value in rendered_fields
    )
    if not has_memory_windows and len(rendered_fields) <= _STRUCTURED_INLINE_MAX_FIELDS:
        inline = " ".join(f"{key}={value}" for key, value in rendered_fields)
        return f"{safe_event} | {inline}"

    context_tokens: list[str] = []
    memory_before_tokens: list[str] = []
    memory_after_tokens: list[str] = []
    memory_current_tokens: list[str] = []

    for key, value in rendered_fields:
        token = f"{key}={value}"
        if key.startswith("memory_before_"):
            suffix = key[len("memory_before_") :]
            memory_before_tokens.append(f"{suffix}={value}")
            continue
        if key.startswith("memory_after_"):
            suffix = key[len("memory_after_") :]
            memory_after_tokens.append(f"{suffix}={value}")
            continue
        if key.startswith("memory_current_"):
            suffix = key[len("memory_current_") :]
            memory_current_tokens.append(f"{suffix}={value}")
            continue
        context_tokens.append(token)

    def _append_group(lines: list[str], label: str, tokens: list[str]) -> None:
        if not tokens:
            return
        for start in range(0, len(tokens), _STRUCTURED_FIELD_CHUNK_SIZE):
            chunk = tokens[start : start + _STRUCTURED_FIELD_CHUNK_SIZE]
            lines.append(f"  {label}: {' '.join(chunk)}")

    lines: list[str] = [safe_event]
    _append_group(lines, "context", context_tokens)
    _append_group(lines, "memory_before", memory_before_tokens)
    _append_group(lines, "memory_after", memory_after_tokens)
    _append_group(lines, "memory_current", memory_current_tokens)
    return "\n".join(lines)


def _format_backend_message(message: str, /, **fields: object) -> str:
    rendered_fields = _render_log_fields(**fields)
    if not rendered_fields:
        return message
    inline = " ".join(f"{key}={value}" for key, value in rendered_fields)
    return f"{message} | {inline}"


def _emit_backend_message_to_logger(
    target: logging.Logger,
    message: str,
    /,
    *message_args: object,
    level: str | int,
    exc_info: Any = None,
    **fields: object,
) -> None:
    resolved_level = _coerce_level_value(level)
    if not fields:
        target.log(resolved_level, message, *message_args, exc_info=exc_info)
        return
    rendered_message = message % message_args if message_args else message
    target.log(resolved_level, _format_backend_message(rendered_message, **fields), exc_info=exc_info)


def get_backend_logger(name: Optional[str] = None) -> BackendLoggerProxy:
    """Return a backend logger with normalized namespace.

    Accepted forms:
    - `apps.backend`/`backend` -> `backend`
    - `apps.backend.<...>` -> `backend.<...>`
    - `backend.<...>`      -> unchanged
    - `<relative>`         -> `backend.<relative>`
    - empty/None           -> `backend`
    """

    return BackendLoggerProxy(_resolve_backend_raw_logger(name))


def emit_backend_message(
    message: str,
    /,
    *message_args: object,
    logger: Optional[str] = None,
    level: str | int = logging.INFO,
    exc_info: Any = None,
    **fields: object,
) -> None:
    """Emit a repo-owned human-readable operational log through the canonical path."""

    _emit_backend_message_to_logger(
        _resolve_backend_raw_logger(logger),
        message,
        *message_args,
        level=level,
        exc_info=exc_info,
        **fields,
    )


def emit_backend_event(
    event: str,
    /,
    *,
    logger: Optional[str] = None,
    level: str | int = logging.INFO,
    **fields: object,
) -> None:
    """Emit a backend event through the single canonical emission path.

    This function is the global source of truth for backend event emission.
    """

    _resolve_backend_raw_logger(logger).log(_coerce_level_value(level), format_log_message(event, **fields))


def build_backend_uvicorn_log_config(
    *,
    suppress_access_prefixes: Sequence[str] | None = None,
    access_filter_factory: str | None = None,
    level: str | int = "INFO",
) -> dict[str, Any]:
    """Return the canonical uvicorn logging config for backend bootstrap use."""

    include_logger_name = _include_logger_name()
    resolved_level_name = _resolved_level_name(level)
    filters: dict[str, Any] = {}
    access_handler_filters: list[str] = []
    if suppress_access_prefixes and access_filter_factory:
        filters["codex_access_noise"] = {
            "()": access_filter_factory,
            "suppress_path_prefixes": list(suppress_access_prefixes),
        }
        access_handler_filters.append("codex_access_noise")

    access_handler: dict[str, Any] = {
        "formatter": "access",
        "class": "logging.StreamHandler",
        "stream": "ext://sys.stderr",
    }
    if access_handler_filters:
        access_handler["filters"] = access_handler_filters

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": filters,
        "formatters": {
            "default": {
                "format": _resolve_log_format(include_logger_name=include_logger_name),
                "datefmt": _DEFAULT_DATEFMT,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "format": "[%(asctime)s] %(levelname)-8s %(client_addr)s - %(request_line)s %(status_code)s",
                "datefmt": _DEFAULT_DATEFMT,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": access_handler,
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": resolved_level_name, "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": resolved_level_name, "propagate": False},
            "uvicorn.access": {"handlers": ["access"], "level": resolved_level_name, "propagate": False},
        },
    }
    return config


class LevelFilter(logging.Filter):
    """Filter that enables/disables log levels based on individual env vars.
    
    Checks CODEX_LOG_DEBUG, CODEX_LOG_INFO, CODEX_LOG_WARNING, CODEX_LOG_ERROR.
    If set to "1"/"true"/"yes"/"on", that level is allowed.
    If set to "0"/"false"/"no"/"off", that level is blocked.
    If not set, defaults apply (DEBUG=0, INFO=1, WARNING=1, ERROR=1).
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._level_flags = self._read_flags()
    
    def _read_flags(self) -> dict[int, bool]:
        """Read level flags from environment."""
        def _is_enabled(env_var: str, default: str) -> bool:
            val = os.environ.get(env_var, default).strip().lower()
            return val in ("1", "true", "yes", "on")
        
        return {
            logging.DEBUG: _is_enabled("CODEX_LOG_DEBUG", "0"),
            logging.INFO: _is_enabled("CODEX_LOG_INFO", "1"),
            logging.WARNING: _is_enabled("CODEX_LOG_WARNING", "1"),
            logging.ERROR: _is_enabled("CODEX_LOG_ERROR", "1"),
            logging.CRITICAL: True,  # Always show critical
        }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if the record should be logged."""
        return self._level_flags.get(record.levelno, True)


def setup_logging(level: Optional[str] = None, *, install_tqdm_bridge: bool = True) -> None:
    """Initialize root logger based on env vars, only once.

    - Sets root level to env-provided level (default DEBUG).
    - Adds a stderr StreamHandler with a concise, actionable format.
    - Optionally adds a file handler if CODEX_LOG_FILE is set.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Determine level
    level_name = level if level is not None else (
        os.environ.get("CODEX_LOG_LEVEL")
        or os.environ.get("SDWEBUI_LOG_LEVEL")
        or os.environ.get("WEBUI_LOG_LEVEL")
    )
    resolved_level = _parse_level(level_name)

    include_logger_name = _include_logger_name()

    # Concise format: [MM/DD/YY HH:MM:SS] LEVEL     message
    fmt = _resolve_log_format(include_logger_name=include_logger_name)
    datefmt = _DEFAULT_DATEFMT

    root = logging.getLogger()
    root.setLevel(resolved_level)

    def _build_stream_handler() -> logging.Handler:
        level_filter = LevelFilter()
        formatter: logging.Formatter
        if not _should_force_plain() and RichHandler is not None and Console is not None:
            console_theme = None
            if Theme is not None:
                console_theme = Theme(
                    {
                        "codexlog.event": "bold bright_magenta",
                        "codexlog.tag": "bold bright_blue",
                        "codexlog.key": "bright_cyan",
                        "codexlog.dtype": "bright_green",
                        "codexlog.device": "cyan",
                        "codexlog.bool": "bright_magenta",
                        "codexlog.number": "bright_yellow",
                        "codexlog.value": "grey70",
                        "codexlog.arrow": "bright_black",
                    }
                )
            console = Console(
                color_system="auto",
                soft_wrap=True,
                highlight=False,
                emoji=False,
                stderr=True,
                theme=console_theme,
            )
            rich_highlighter = CodexLogHighlighter() if CodexLogHighlighter is not None else None
            inner: logging.Handler = RichHandler(
                console=console,
                show_time=True,
                show_path=_env_true("CODEX_LOG_RICH_SHOW_PATH", "0"),
                rich_tracebacks=_env_true("CODEX_LOG_RICH_TRACEBACKS", "1"),
                markup=False,
                highlighter=rich_highlighter,
                keywords=[],
            )
            formatter = logging.Formatter("%(name)s | %(message)s" if include_logger_name else "%(message)s")
        else:
            inner = logging.StreamHandler(stream=sys.stderr)
            formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        if install_tqdm_bridge and tqdm is not None:
            handler: logging.Handler = TqdmAwareHandler(inner)
        else:
            handler = inner
        handler.setLevel(resolved_level)
        handler.setFormatter(formatter)
        handler.addFilter(level_filter)
        return handler

    if not any(_is_stream_handler(h) for h in root.handlers):
        root.addHandler(_build_stream_handler())

    # Ensure a dedicated handler for the 'backend' logger hierarchy so DEBUG
    # logs are not filtered by third-party handlers (e.g., uvicorn/gradio)
    codex = logging.getLogger("backend")
    codex.setLevel(resolved_level)
    # mark our handler to avoid duplicates on re-entry
    has_codex = False
    for h in codex.handlers:
        if getattr(h, "_codex", False):
            has_codex = True
            break
    if not has_codex:
        h = _build_stream_handler()
        setattr(h, "_codex", True)
        codex.addHandler(h)
    # prevent double printing via root handlers
    codex.propagate = False

    log_file = os.environ.get("CODEX_LOG_FILE")
    file_handler: logging.FileHandler | None = None
    if log_file:
        abs_log_file = os.path.abspath(log_file)
        for h in root.handlers:
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_log_file:
                file_handler = h
                break

        if file_handler is None:
            try:
                fh = logging.FileHandler(abs_log_file, encoding="utf-8")
                fh.setLevel(resolved_level)
                fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
                fh.addFilter(LevelFilter())
                root.addHandler(fh)
                file_handler = fh
            except Exception:
                # If file handler fails, keep stderr logging only; do not crash startup
                logging.getLogger(__name__).exception("Failed to attach file handler: %s", log_file)

    # `backend` logger has `propagate=False`, so it won't reach root's file handler.
    # Attach the file handler explicitly so launcher users actually get backend logs.
    if file_handler is not None and not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == getattr(file_handler, "baseFilename", None)
        for h in codex.handlers
    ):
        codex.addHandler(file_handler)

    logging.getLogger(__name__).debug(
        "logging configured level=%s file=%s handlers=%d",
        logging.getLevelName(resolved_level),
        log_file or "<stderr-only>",
        len(root.handlers),
    )

    _CONFIGURED = True
