"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Thread-local overrides + option-backed helpers for smart offload/fallback/cache flags, including cross-worker override snapshot propagation and canonical smart-offload telemetry enrichment.

Symbols (top-level; keep in sync; no ghosts):
- `smart_runtime_overrides` (function): Context manager to override smart flags for the current thread/request.
- `current_smart_runtime_overrides` (function): Snapshot the current thread smart override tri-state values for cross-worker propagation.
- `smart_offload_enabled` (function): True when smart offload is enabled (options + thread overrides).
- `smart_fallback_enabled` (function): True when smart CPU fallback on OOM is enabled (options + thread overrides).
- `smart_cache_enabled` (function): True when Smart Cache is enabled (options + thread overrides).
- `SmartOffloadAction` (enum): Canonical smart-offload action catalog for structured event emission.
- `_bytes_to_mib` (function): Convert a byte count to MiB for structured telemetry payloads.
- `_read_memory_snapshot` (function): Best-effort runtime memory snapshot reader used to enrich smart-offload events.
- `_memory_fields_from_snapshot` (function): Build structured memory fields from a snapshot using the requested prefix.
- `log_smart_offload_action` (function): Emits canonical smart-offload INFO events via the global event emitter.
- `_emit_smart_cache_counter_failure` (function): Emit classified smart-cache counter failures with request/task context.
- `_normalize_smart_cache_bucket_name` (function): Enforce strict smart-cache bucket key contract (non-empty string).
- `record_smart_cache_hit` (function): Increment Smart Cache hit counter for a named bucket.
- `record_smart_cache_miss` (function): Increment Smart Cache miss counter for a named bucket.
- `get_smart_cache_stats` (function): Return hit/miss counters for all Smart Cache buckets.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
import logging
import math
import threading
from typing import Dict, Iterator, Mapping

from apps.backend.core.strict_values import parse_bool_value
from apps.backend.runtime.logging import emit_backend_event


def _snapshot():
    from apps.backend.services import options_store

    return options_store.get_snapshot()


_THREAD_OVERRIDES = threading.local()


class SmartOffloadAction(str, Enum):
    LOAD = "load"
    UNLOAD = "unload"
    UNLOAD_NOOP = "unload_noop"
    STAGE_LOAD = "stage_load"
    DIRECT_OFFLOAD = "direct_offload"
    PIN_HOST_MEMORY = "pin_host_memory"


def _get_override(name: str) -> bool | None:
    return getattr(_THREAD_OVERRIDES, name, None)


def current_smart_runtime_overrides() -> Dict[str, bool | None]:
    """Return the current thread smart-override tri-state snapshot."""
    return {
        "smart_offload": _get_override("smart_offload"),
        "smart_fallback": _get_override("smart_fallback"),
        "smart_cache": _get_override("smart_cache"),
    }


@contextmanager
def smart_runtime_overrides(
    *,
    smart_offload: bool | None = None,
    smart_fallback: bool | None = None,
    smart_cache: bool | None = None,
) -> Iterator[None]:
    """Temporarily override smart flags for the current thread.

    This is intended for per-request overrides inside worker threads, so runtime
    code that consults these helpers (sampling, memory manager, patchers) can
    honor request-level flags without relying on persisted `/api/options`.
    """
    prev_offload = _get_override("smart_offload")
    prev_fallback = _get_override("smart_fallback")
    prev_cache = _get_override("smart_cache")
    _THREAD_OVERRIDES.smart_offload = smart_offload
    _THREAD_OVERRIDES.smart_fallback = smart_fallback
    _THREAD_OVERRIDES.smart_cache = smart_cache
    try:
        yield
    finally:
        _THREAD_OVERRIDES.smart_offload = prev_offload
        _THREAD_OVERRIDES.smart_fallback = prev_fallback
        _THREAD_OVERRIDES.smart_cache = prev_cache


def smart_offload_enabled() -> bool:
    """Return True when smart offload is enabled (Codex options only)."""
    override = _get_override("smart_offload")
    if override is not None:
        if not isinstance(override, bool):
            raise RuntimeError(f"Invalid smart override 'smart_offload': expected bool, got {type(override).__name__}.")
        return override
    snap = _snapshot()
    return parse_bool_value(
        getattr(snap, "codex_smart_offload", None),
        field="options.codex_smart_offload",
        default=False,
    )


def smart_fallback_enabled() -> bool:
    """Return True when smart CPU fallback on OOM is enabled (Codex options only)."""
    override = _get_override("smart_fallback")
    if override is not None:
        if not isinstance(override, bool):
            raise RuntimeError(f"Invalid smart override 'smart_fallback': expected bool, got {type(override).__name__}.")
        return override
    snap = _snapshot()
    return parse_bool_value(
        getattr(snap, "codex_smart_fallback", None),
        field="options.codex_smart_fallback",
        default=False,
    )


def smart_cache_enabled() -> bool:
    """Return True when SDXL smart caching (TE + embed_values) is enabled."""
    override = _get_override("smart_cache")
    if override is not None:
        if not isinstance(override, bool):
            raise RuntimeError(f"Invalid smart override 'smart_cache': expected bool, got {type(override).__name__}.")
        return override
    snap = _snapshot()
    return parse_bool_value(
        getattr(snap, "codex_smart_cache", None),
        field="options.codex_smart_cache",
        default=False,
    )


def _bytes_to_mib(value: int) -> float:
    return round(float(max(0, int(value))) / (1024.0 * 1024.0), 2)


def _read_memory_snapshot() -> dict[str, object] | None:
    """Best-effort runtime memory snapshot for smart-offload telemetry fields."""

    try:
        from apps.backend.runtime.memory import memory_management

        snapshot = memory_management.memory_snapshot()
    except Exception:
        return None
    if not isinstance(snapshot, dict):
        return None
    return snapshot


def _memory_fields_from_snapshot(snapshot: Mapping[str, object], *, prefix: str) -> Dict[str, object]:
    fields: Dict[str, object] = {}

    primary_device = snapshot.get("primary_device")
    if primary_device is not None:
        fields[f"{prefix}_device"] = str(primary_device)

    torch_stats = snapshot.get("torch")
    if not isinstance(torch_stats, Mapping):
        return fields

    mapping = (
        ("allocated_bytes", "alloc"),
        ("reserved_bytes", "reserved"),
        ("free_bytes", "free"),
        ("total_bytes", "total"),
    )
    for source_key, target_key in mapping:
        raw_value = torch_stats.get(source_key)
        if not isinstance(raw_value, (int, float)):
            continue
        if isinstance(raw_value, float) and not math.isfinite(raw_value):
            continue
        try:
            bytes_value = max(0, int(raw_value))
        except (TypeError, ValueError, OverflowError):
            continue
        fields[f"{prefix}_{target_key}_mb"] = _bytes_to_mib(bytes_value)

    return fields


def log_smart_offload_action(action: SmartOffloadAction, /, **fields: object) -> None:
    """Emit the canonical INFO log event for a smart-offload action."""

    if not isinstance(action, SmartOffloadAction):
        raise TypeError(
            "smart_offload action must be SmartOffloadAction "
            f"(received {type(action).__name__})."
        )
    action_name = action.value.strip().replace(" ", "_")
    if not action_name:
        action_name = "unknown"

    event_fields: Dict[str, object] = dict(fields)
    has_window_fields = any(
        key.startswith("memory_before_") or key.startswith("memory_after_")
        for key in event_fields
    )
    if not has_window_fields:
        snapshot = _read_memory_snapshot()
        if snapshot is not None:
            try:
                snapshot_fields = _memory_fields_from_snapshot(snapshot, prefix="memory_current")
            except Exception:
                snapshot_fields = {}
            for key, value in snapshot_fields.items():
                event_fields.setdefault(key, value)

    emit_backend_event(
        f"smart_offload.{action_name}",
        logger="smart_offload",
        **event_fields,
    )


_SMART_CACHE_COUNTERS: Dict[str, Dict[str, int]] = {}
_SMART_CACHE_COUNTERS_LOCK = threading.Lock()


def _emit_smart_cache_counter_failure(
    *,
    category: str,
    operation: str,
    bucket: object,
    error: BaseException,
    transient: bool,
) -> None:
    emit_backend_event(
        "smart_cache.counter_failure",
        logger="smart_offload",
        level=(logging.WARNING if transient else logging.ERROR),
        category=str(category),
        operation=str(operation),
        bucket=(str(bucket) if bucket is not None else None),
        task_context="smart_cache",
        request_context=threading.current_thread().name,
        error_type=type(error).__name__,
        error=str(error),
    )


def _normalize_smart_cache_bucket_name(name: object, *, operation: str) -> str:
    if not isinstance(name, str):
        raise RuntimeError(
            f"Smart Cache bucket name contract violation in {operation}: "
            f"expected non-empty str, got {type(name).__name__}."
        )
    if not name:
        raise RuntimeError(
            f"Smart Cache bucket name contract violation in {operation}: "
            "expected non-empty str, got empty value."
        )
    if name.strip() != name:
        raise RuntimeError(
            f"Smart Cache bucket name contract violation in {operation}: "
            "leading/trailing whitespace is not allowed."
        )
    return name


def _bucket(name: str) -> Dict[str, int]:
    bucket = _SMART_CACHE_COUNTERS.get(name)
    if bucket is None:
        bucket = {"hits": 0, "misses": 0}
        _SMART_CACHE_COUNTERS[name] = bucket
    return bucket


def record_smart_cache_hit(name: str) -> None:
    """Increment Smart Cache hit counter for the given bucket name."""
    operation = "record_hit"
    try:
        bucket_name = _normalize_smart_cache_bucket_name(name, operation=operation)
    except RuntimeError as exc:
        _emit_smart_cache_counter_failure(
            category="contract",
            operation=operation,
            bucket=name,
            error=exc,
            transient=False,
        )
        raise
    try:
        with _SMART_CACHE_COUNTERS_LOCK:
            bucket = _bucket(bucket_name)
            bucket["hits"] += 1
    except Exception as exc:  # noqa: BLE001 - explicit transient classification/logging
        _emit_smart_cache_counter_failure(
            category="transient",
            operation=operation,
            bucket=bucket_name,
            error=exc,
            transient=True,
        )
        raise RuntimeError(
            f"Smart Cache counter update failed during {operation} (bucket={bucket_name!r})."
        ) from exc


def record_smart_cache_miss(name: str) -> None:
    """Increment Smart Cache miss counter for the given bucket name."""
    operation = "record_miss"
    try:
        bucket_name = _normalize_smart_cache_bucket_name(name, operation=operation)
    except RuntimeError as exc:
        _emit_smart_cache_counter_failure(
            category="contract",
            operation=operation,
            bucket=name,
            error=exc,
            transient=False,
        )
        raise
    try:
        with _SMART_CACHE_COUNTERS_LOCK:
            bucket = _bucket(bucket_name)
            bucket["misses"] += 1
    except Exception as exc:  # noqa: BLE001 - explicit transient classification/logging
        _emit_smart_cache_counter_failure(
            category="transient",
            operation=operation,
            bucket=bucket_name,
            error=exc,
            transient=True,
        )
        raise RuntimeError(
            f"Smart Cache counter update failed during {operation} (bucket={bucket_name!r})."
        ) from exc


def get_smart_cache_stats() -> Dict[str, Dict[str, int]]:
    """Return a shallow copy of Smart Cache hit/miss counters."""
    try:
        with _SMART_CACHE_COUNTERS_LOCK:
            return {name: dict(counts) for name, counts in _SMART_CACHE_COUNTERS.items()}
    except Exception as exc:  # noqa: BLE001 - explicit transient classification/logging
        _emit_smart_cache_counter_failure(
            category="transient",
            operation="snapshot",
            bucket=None,
            error=exc,
            transient=True,
        )
        raise RuntimeError("Smart Cache counter snapshot failed (transient runtime failure).") from exc


__all__ = [
    "SmartOffloadAction",
    "current_smart_runtime_overrides",
    "log_smart_offload_action",
    "smart_offload_enabled",
    "smart_fallback_enabled",
    "smart_cache_enabled",
    "smart_runtime_overrides",
    "record_smart_cache_hit",
    "record_smart_cache_miss",
    "get_smart_cache_stats",
]
