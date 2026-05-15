"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime contract-trace sink for generation pipelines.
Writes opt-in JSONL events under `logs/contract-trace/` with prompt hashing only (never raw prompt text), so runtime stage/action/device
contracts can be audited without leaking prompt content.

Symbols (top-level; keep in sync; no ghosts):
- `is_enabled` (function): Returns whether contract tracing is enabled (`CODEX_TRACE_CONTRACT=1`).
- `prompt_hash` (function): Returns a SHA-256 hex hash for prompt strings (empty-string hash for missing/non-string values).
- `hash_request_prompt` (function): Extracts a prompt-like field from a request object and returns its hash.
- `error_meta` (function): Returns redacted error metadata for trace events (type + hash, never raw message).
- `emit_event` (function): Appends one contract-trace JSONL event (fail-loud logging; non-fatal on sink failure).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import datetime as _datetime
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from apps.backend.infra.config.env_flags import env_flag
from apps.backend.infra.config.repo_root import get_repo_root

_log = get_backend_logger("backend.contract_trace")
_lock = threading.Lock()
_trace_path: Path | None = None


def is_enabled() -> bool:
    return env_flag("CODEX_TRACE_CONTRACT", default=False)


def _get_trace_path() -> Path:
    global _trace_path
    with _lock:
        if _trace_path is not None:
            return _trace_path
        logs_dir = get_repo_root() / "logs" / "contract-trace"
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _datetime.datetime.now().strftime("%Y%m%d")
        _trace_path = logs_dir / f"{timestamp}-pid{os.getpid()}.jsonl"
        return _trace_path


def prompt_hash(prompt: object) -> str:
    if not isinstance(prompt, str):
        prompt = ""
    normalized = prompt.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def hash_request_prompt(request: object) -> str:
    prompt_fields = (
        "prompt",
        "img2img_prompt",
        "txt2vid_prompt",
        "img2vid_prompt",
        "vid2vid_prompt",
    )
    for field_name in prompt_fields:
        value = getattr(request, field_name, None)
        if isinstance(value, str) and value.strip():
            return prompt_hash(value)
    return prompt_hash("")


def error_meta(err: BaseException | Exception) -> dict[str, str]:
    error_type = type(err).__name__
    digest = hashlib.sha256(str(err).encode("utf-8")).hexdigest()
    return {
        "error_type": error_type,
        "error_hash": digest,
    }


def _json_safe(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def emit_event(
    *,
    task_id: str,
    mode: str,
    stage: str,
    action: str,
    component: str = "pipeline",
    device: str = "",
    storage_dtype: str | None = None,
    compute_dtype: str | None = None,
    strict: bool = True,
    fallback_enabled: bool = False,
    fallback_used: bool = False,
    prompt_hash_value: str = "",
    meta: Mapping[str, Any] | None = None,
) -> None:
    if not is_enabled():
        return

    event: dict[str, object] = {
        "request_id": str(task_id),
        "task_id": str(task_id),
        "mode": str(mode),
        "stage": str(stage),
        "action": str(action),
        "component": str(component),
        "device": str(device or ""),
        "storage_dtype": str(storage_dtype or ""),
        "compute_dtype": str(compute_dtype or ""),
        "timestamp_ms": int(time.time() * 1000),
        "strict": bool(strict),
        "fallback_enabled": bool(fallback_enabled),
        "fallback_used": bool(fallback_used),
        "prompt_hash": str(prompt_hash_value or prompt_hash("")),
    }
    if meta:
        event["meta"] = _json_safe(dict(meta))

    try:
        trace_path = _get_trace_path()
        line = json.dumps(event, ensure_ascii=True, separators=(",", ":"))
        with _lock:
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        _log.exception("contract-trace: failed to write event (task_id=%s mode=%s stage=%s)", task_id, mode, stage)


__all__ = [
    "error_meta",
    "emit_event",
    "hash_request_prompt",
    "is_enabled",
    "prompt_hash",
]
