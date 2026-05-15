"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Memory controller for LTX2 core streaming (shared core wrapper).
Keeps the LTX2-family public import path stable while delegating actual controller behavior to
`apps.backend.runtime.streaming.controller`.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2StreamingPolicy` (constant): Streaming policy enum controlling segment residency.
- `TransferStats` (class): CPU↔GPU transfer statistics from the shared controller.
- `Ltx2CoreController` (class): LTX2 streaming controller wrapper over the shared implementation.
- `_resolve_default_streaming_devices` (function): Resolves storage/compute devices from the active memory manager.
- `create_controller` (function): Factory that builds an `Ltx2CoreController`.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging

import torch

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.streaming.controller import (
    StreamingController as _StreamingController,
    StreamingPolicy,
    TransferStats,
)

from .specs import Ltx2Segment

logger = get_backend_logger("backend.runtime.ltx2.streaming.controller")

Ltx2StreamingPolicy = StreamingPolicy


class Ltx2CoreController(_StreamingController[Ltx2Segment]):
    """LTX2 streaming controller (shared implementation)."""


def _resolve_default_streaming_devices() -> tuple[torch.device, torch.device]:
    manager = getattr(memory_management, "manager", None)
    if manager is None:
        raise RuntimeError("LTX2 streaming controller requires an active memory manager instance.")
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
    policy: str | Ltx2StreamingPolicy = "naive",
    window_size: int = 1,
    storage_device: str | torch.device | None = None,
    compute_device: str | torch.device | None = None,
) -> Ltx2CoreController:
    if isinstance(policy, str):
        policy = Ltx2StreamingPolicy(policy.lower())

    default_storage_device, default_compute_device = _resolve_default_streaming_devices()
    resolved_storage_device = default_storage_device if storage_device is None else torch.device(storage_device)
    resolved_compute_device = default_compute_device if compute_device is None else torch.device(compute_device)

    return Ltx2CoreController(
        storage_device=resolved_storage_device,
        compute_device=resolved_compute_device,
        policy=policy,
        window_size=window_size,
        non_blocking=True,
        logger=logger,
    )
