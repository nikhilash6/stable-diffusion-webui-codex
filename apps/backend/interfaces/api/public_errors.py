"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public-safe task and HTTP error normalization for API payloads/SSE.
Builds one public terminal error envelope (`message` + `code` + optional `error_id`) for async task channels while
preserving readable HTTP fallback details and actionable terminal classes like cancellation and out-of-memory.

Symbols (top-level; keep in sync; no ghosts):
- `TaskErrorCode` (enum): Canonical public-safe task terminal error codes used by async task channels.
- `PublicTaskError` (dataclass): Public terminal task error envelope stored by task workers and serialized by task routes.
- `build_cancelled_task_error` (function): Return the canonical public terminal envelope for cancellation.
- `build_missing_result_task_error` (function): Return the canonical public terminal envelope for invariant failures where no result payload was produced.
- `build_public_task_error` (function): Convert a raw exception/string into the canonical public-safe task error envelope.
- `public_task_error_message` (function): Convert a raw exception/string into a public-safe task error message.
- `public_task_error_code` (function): Convert a raw exception/string into the canonical public task error code.
- `public_task_error_id` (function): Return the public-safe error id for a raw exception/string when present.
- `public_http_error_detail` (function): Convert a raw exception/string into a public-safe HTTP detail string.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from apps.backend.runtime.diagnostics.error_summary import friendly_public_exception_message

_INTERNAL_ERROR_ID_RE = re.compile(r"^internal error \(error_id=[0-9a-f]{12}\)$")
_ENGINE_ERROR_RE = re.compile(r"^engine error:\s*", flags=re.IGNORECASE)
_OOM_WORD_RE = re.compile(r"\boom\b", flags=re.IGNORECASE)
_OOM_HINTS = (
    "out of memory",
    "cuda oom",
    "not enough memory",
    "alloc_failed",
    "allocation failed",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)
_OOM_CONTEXT_HINTS = (
    "cuda",
    "vram",
    "memory",
    "alloc",
    "allocation",
    "load",
    "loading",
    "model",
    "construction",
    "tiled",
    "upscal",
    "gguf",
    "wan",
    "spandrel",
)
_INTEGRITY_HINTS = ("sha256 mismatch",)
_SAFE_PUBLIC_ENGINE_ERROR_MESSAGES = frozenset(
    {
        "engine error: task completed without result payload",
    }
)


class TaskErrorCode(StrEnum):
    CANCELLED = "cancelled"
    OUT_OF_MEMORY = "out_of_memory"
    INTEGRITY_MISMATCH = "integrity_mismatch"
    ENGINE_ERROR = "engine_error"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class PublicTaskError:
    message: str
    code: TaskErrorCode
    error_id: str | None = None


def _normalize_error_text(err: Any) -> str:
    text = str(err or "").strip()
    if not text:
        return "internal error"
    return text


def _is_oom_error(*, err: Any, lowered_message: str) -> bool:
    if isinstance(err, MemoryError):
        return True
    if any(marker in lowered_message for marker in _OOM_HINTS):
        return True
    if _OOM_WORD_RE.fullmatch(lowered_message.strip()):
        return True
    if _OOM_WORD_RE.search(lowered_message):
        if any(context in lowered_message for context in _OOM_CONTEXT_HINTS):
            return True
    return False


def _extract_internal_error_id(raw: str) -> str | None:
    if _INTERNAL_ERROR_ID_RE.fullmatch(raw) is None:
        return None
    return raw.removeprefix("internal error (error_id=").removesuffix(")") or None


def _stable_error_id(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]


def _safe_public_engine_error(raw: str) -> PublicTaskError:
    normalized_raw = raw.strip()
    if normalized_raw.lower() in _SAFE_PUBLIC_ENGINE_ERROR_MESSAGES:
        return PublicTaskError(message=normalized_raw, code=TaskErrorCode.ENGINE_ERROR)
    error_id = _stable_error_id(normalized_raw)
    return PublicTaskError(
        message=f"engine error (error_id={error_id})",
        code=TaskErrorCode.ENGINE_ERROR,
        error_id=error_id,
    )


def build_cancelled_task_error() -> PublicTaskError:
    return PublicTaskError(message="cancelled", code=TaskErrorCode.CANCELLED)


def build_missing_result_task_error(task_label: str = "task") -> PublicTaskError:
    normalized_label = str(task_label or "task").strip() or "task"
    return PublicTaskError(
        message=f"engine error: {normalized_label} completed without result payload",
        code=TaskErrorCode.ENGINE_ERROR,
    )


def build_public_task_error(err: Any) -> PublicTaskError:
    if isinstance(err, PublicTaskError):
        return err

    raw = _normalize_error_text(err)
    lowered = raw.lower()
    friendly_message = friendly_public_exception_message(raw)

    if lowered == "cancelled":
        return build_cancelled_task_error()

    existing_error_id = _extract_internal_error_id(raw)
    if existing_error_id is not None:
        return PublicTaskError(message=raw, code=TaskErrorCode.INTERNAL_ERROR, error_id=existing_error_id)

    if any(marker in lowered for marker in _INTEGRITY_HINTS):
        return PublicTaskError(message="sha256 mismatch", code=TaskErrorCode.INTEGRITY_MISMATCH)

    if _is_oom_error(err=err, lowered_message=lowered):
        return PublicTaskError(message="out of memory", code=TaskErrorCode.OUT_OF_MEMORY)

    if friendly_message is not None:
        return PublicTaskError(message=friendly_message, code=TaskErrorCode.ENGINE_ERROR)

    if isinstance(err, str) and _ENGINE_ERROR_RE.match(raw):
        return _safe_public_engine_error(raw)

    try:
        from apps.backend.core.exceptions import EngineExecutionError, EngineLoadError

        if isinstance(err, (EngineExecutionError, EngineLoadError)):
            return _safe_public_engine_error(f"engine error: {raw}")
    except Exception:
        pass

    error_id = _stable_error_id(raw)
    return PublicTaskError(
        message=f"internal error (error_id={error_id})",
        code=TaskErrorCode.INTERNAL_ERROR,
        error_id=error_id,
    )


def public_task_error_message(err: Any) -> str:
    return build_public_task_error(err).message


def public_task_error_code(err: Any) -> TaskErrorCode:
    return build_public_task_error(err).code


def public_task_error_id(err: Any) -> str | None:
    return build_public_task_error(err).error_id


def public_http_error_detail(err: Any, *, fallback: str) -> str:
    public = build_public_task_error(err)
    if public.code is TaskErrorCode.OUT_OF_MEMORY:
        return public.message
    return str(fallback or "invalid request")


__all__ = [
    "TaskErrorCode",
    "PublicTaskError",
    "build_cancelled_task_error",
    "build_missing_result_task_error",
    "build_public_task_error",
    "public_http_error_detail",
    "public_task_error_message",
    "public_task_error_code",
    "public_task_error_id",
]
