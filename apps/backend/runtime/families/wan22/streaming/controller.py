"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Memory controller for WAN22 core streaming (shared core wrapper).
WAN22 and Flux share the same controller implementation to keep streaming semantics identical and avoid drift. This module keeps the WAN22
public import path stable while delegating the implementation to `apps.backend.runtime.streaming.controller`.

Symbols (top-level; keep in sync; no ghosts):
- `WanStreamingPolicy` (enum): Streaming policy (`naive`/`window`/`aggressive`) controlling segment residency.
- `TransferStats` (dataclass): Tracks CPU↔GPU transfer bytes/counts/time for streaming telemetry.
- `WanCoreController` (class): Core streaming controller; loads/unloads segments around forward passes and manages LRU/window policies.
- `_resolve_default_streaming_devices` (function): Resolves default storage/compute devices from memory-manager lifecycle policy.
- `create_controller` (function): Factory that builds a `WanCoreController` from policy and blocks-per-segment inputs.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Optional

import torch

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.streaming.controller import (
    StreamingController as _StreamingController,
    StreamingPolicy,
    TransferStats,
)

from .specs import WanSegment

logger = get_backend_logger("backend.runtime.wan22.streaming.controller")

WanStreamingPolicy = StreamingPolicy


class WanCoreController(_StreamingController[WanSegment]):
    """WAN22 streaming controller (shared implementation)."""


def _resolve_default_streaming_devices() -> tuple[torch.device, torch.device]:
    manager = getattr(memory_management, "manager", None)
    if manager is None:
        raise RuntimeError("WAN22 streaming controller requires an active memory manager instance.")
    storage_device = manager.offload_device()
    compute_device = manager.mount_device()
    if not isinstance(storage_device, torch.device):
        raise RuntimeError(
            "memory manager offload_device() must return torch.device "
            f"(got {type(storage_device).__name__})."
        )
    if not isinstance(compute_device, torch.device):
        raise RuntimeError(
            "memory manager mount_device() must return torch.device "
            f"(got {type(compute_device).__name__})."
        )
    return storage_device, compute_device


def create_controller(
    policy: str | WanStreamingPolicy = "naive",
    window_size: int = 2,
    storage_device: str | torch.device | None = None,
    compute_device: str | torch.device | None = None,
) -> WanCoreController:
    if isinstance(policy, str):
        policy = WanStreamingPolicy(policy.lower())

    default_storage_device, default_compute_device = _resolve_default_streaming_devices()
    resolved_storage_device = default_storage_device if storage_device is None else torch.device(storage_device)
    resolved_compute_device = default_compute_device if compute_device is None else torch.device(compute_device)

    return WanCoreController(
        storage_device=resolved_storage_device,
        compute_device=resolved_compute_device,
        policy=policy,
        window_size=window_size,
        non_blocking=True,
        logger=logger,
    )
