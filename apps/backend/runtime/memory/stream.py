"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Best-effort CUDA/XPU stream helpers for async swap operations.
Resolves swap method from runtime memory config (source of truth) with args fallback, and enables stream workers for
`swap_method=async|block_swap_experimental`.

Symbols (top-level; keep in sync; no ghosts):
- `stream_context` (function): Return the torch stream context manager for the active backend, or None.
- `get_current_stream` (function): Return the current device stream when safe/available.
- `get_new_stream` (function): Create and validate a new device stream when safe/available.
- `should_use_stream` (function): True when streams are activated and both streams are available.
- `stream_activated` (constant): Whether stream-based swapping is currently activated by config.
- `current_stream` (constant): Best-effort current compute stream object (or None).
- `mover_stream` (constant): Best-effort mover stream object (or None).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging

import torch
from apps.backend.infra.config.args import args
from apps.backend.runtime.memory.config import SwapMethod


logger = get_backend_logger("backend.runtime.memory.stream")

_STREAM_SWAP_METHODS = {
    SwapMethod.ASYNC.value,
    SwapMethod.BLOCK_SWAP_EXPERIMENTAL.value,
}

_cached_swap_method: str | None = None
stream_activated = False
current_stream = None
mover_stream = None


def _normalize_swap_method(raw: object) -> str:
    if isinstance(raw, SwapMethod):
        return raw.value
    return str(raw or "").strip().lower()


def _resolve_swap_method() -> str:
    """Resolve swap method from runtime memory config (source of truth), with args fallback."""

    try:
        from apps.backend.runtime.memory import memory_management

        manager = getattr(memory_management, "manager", None)
        config = getattr(manager, "config", None)
        swap_cfg = getattr(config, "swap", None)
        method = getattr(swap_cfg, "method", None)
        if method is not None:
            return _normalize_swap_method(method)
    except Exception:  # noqa: BLE001
        pass
    return _normalize_swap_method(getattr(args, "swap_method", None))


def _stream_method_active(method: str) -> bool:
    return method in _STREAM_SWAP_METHODS


def stream_context():
    if torch.cuda.is_available():
        return torch.cuda.stream

    if torch.xpu.is_available():
        return torch.xpu.stream

    return None


def get_current_stream():
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
            current = torch.cuda.current_stream(device)
            with torch.cuda.stream(current):
                torch.zeros((1, 1)).to(device, torch.float32)
            current.synchronize()
            return current
        if torch.xpu.is_available():
            device = torch.device("xpu")
            current = torch.xpu.current_stream(device)
            with torch.xpu.stream(current):
                torch.zeros((1, 1)).to(device, torch.float32)
            current.synchronize()
            return current
    except Exception:  # noqa: BLE001
        return None
    return None


def get_new_stream():
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
            mover = torch.cuda.Stream(device)
            with torch.cuda.stream(mover):
                torch.zeros((1, 1)).to(device, torch.float32)
            mover.synchronize()
            return mover
        if torch.xpu.is_available():
            device = torch.device("xpu")
            mover = torch.xpu.Stream(device)
            with torch.xpu.stream(mover):
                torch.zeros((1, 1)).to(device, torch.float32)
            mover.synchronize()
            return mover
    except Exception:  # noqa: BLE001
        return None
    return None


def _refresh_stream_state() -> None:
    global _cached_swap_method, stream_activated, current_stream, mover_stream

    method = _resolve_swap_method()
    method_active = _stream_method_active(method)

    if _cached_swap_method == method and (
        (not method_active and current_stream is None and mover_stream is None)
        or (method_active and current_stream is not None and mover_stream is not None)
    ):
        stream_activated = method_active
        return

    _cached_swap_method = method
    if not method_active:
        stream_activated = False
        current_stream = None
        mover_stream = None
        return

    current_stream = get_current_stream()
    mover_stream = get_new_stream()
    stream_activated = current_stream is not None and mover_stream is not None
    if not stream_activated:
        cuda_available = torch.cuda.is_available()
        log_fn = logger.warning if cuda_available else logger.debug
        log_fn(
            "Swap stream method requested but streams are unavailable. method=%s cuda=%s xpu=%s "
            "current_stream=%s mover_stream=%s",
            method,
            cuda_available,
            torch.xpu.is_available(),
            current_stream is not None,
            mover_stream is not None,
        )


def should_use_stream():
    _refresh_stream_state()
    return stream_activated and current_stream is not None and mover_stream is not None


_refresh_stream_state()
