"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF stage selection and model mounting.
Validates stage GGUF paths and mounts stage weights into `WanTransformer2DModel` via Codex GGUF operations (`using_codex_operations(weight_format="gguf")`) and WAN keyspace resolution, with GGUF state loading/materialization wired to the memory-manager mount device (`dequantize=False`, `computation_dtype=dtype`) so placement policy remains centralized. Also triggers generic SRAM-attention warmup at stage-load time so extension load/JIT compile can happen before denoise.
Optionally applies an ordered per-stage LoRA sequence (merge/online) for LightX2V-style stage patches.

Symbols (top-level; keep in sync; no ghosts):
- `pick_stage_gguf` (function): Validates and returns the stage GGUF file path (strict: must be an explicit `.gguf` file).
- `_resolve_stage_mount_device` (function): Resolves the mount device from memory manager policy.
- `mount_stage_model_from_gguf` (function): Mounts a stage GGUF into a runtime transformer (mount-device GGUF load + keyspace resolution + mount-device ops wrapper); final lifecycle ownership remains delegated to memory manager.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import torch

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.logging import BackendLoggerProxy, emit_backend_message
from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.checkpoint.io import load_gguf_state_dict

from .diagnostics import get_logger, log_cuda_mem
from .model import load_wan_transformer_from_state_dict, resolve_wan22_gguf_keyspace
from .paths import normalize_win_path
from .stage_lora import apply_wan22_stage_lora


def pick_stage_gguf(dir_path: Optional[str], *, stage: str) -> Optional[str]:
    if not dir_path:
        return None

    raw = normalize_win_path(dir_path)
    abspath = raw if os.path.isabs(raw) else os.path.abspath(raw)
    if os.path.isfile(abspath) and abspath.lower().endswith(".gguf"):
        return abspath
    if os.path.isdir(abspath):
        raise RuntimeError(
            f"WAN22 GGUF stage '{stage}' requires an explicit .gguf file path (sha-selected); got directory: {abspath}"
        )
    return None


def _resolve_stage_mount_device() -> torch.device:
    manager = getattr(memory_management, "manager", None)
    if manager is None or not hasattr(manager, "mount_device"):
        raise RuntimeError("WAN22 GGUF stage mount requires an active memory manager with mount_device().")
    mount_device = manager.mount_device()
    if not isinstance(mount_device, torch.device):
        raise RuntimeError(
            "WAN22 GGUF stage mount requires memory manager mount_device() to return torch.device "
            f"(got {type(mount_device).__name__})."
        )
    return mount_device


def mount_stage_model_from_gguf(
    gguf_path: str,
    *,
    stage: str,
    dtype: torch.dtype,
    loras: Optional[Sequence[tuple[str, float]]] = None,
    logger: BackendLoggerProxy | None,
):
    log = get_logger(logger)
    logger_name = log.name
    dequantize = False
    mount_device = _resolve_stage_mount_device()
    log_cuda_mem(log, label=f"{stage}:before-mount-load")
    state = load_gguf_state_dict(
        gguf_path,
        dequantize=dequantize,
        computation_dtype=dtype,
        device=mount_device,
    )
    log_cuda_mem(log, label=f"{stage}:after-mount-load")
    state = resolve_wan22_gguf_keyspace(state)
    with using_codex_operations(device=mount_device, dtype=dtype, weight_format="gguf"):
        model = load_wan_transformer_from_state_dict(state, config=None)
    del state
    log_cuda_mem(log, label=f"{stage}:after-mount-materialize")
    model.eval()
    apply_wan22_stage_lora(
        model,
        stage=stage,
        loras=loras,
        logger=logger,
    )
    sram_mode_raw = str(os.environ.get("CODEX_ATTENTION_SRAM_MODE", "off")).strip().lower()
    try:
        from apps.backend.runtime.attention.sram import warmup_extension_for_load
    except Exception as exc:
        if sram_mode_raw in {"force", "required"}:
            raise RuntimeError(
                "SRAM attention force mode requested but warmup import failed during stage load."
            ) from exc
        emit_backend_message(
            "[wan22.gguf] attention_sram warmup skipped",
            logger=logger_name,
            level="WARNING",
            stage=stage,
            import_failed=type(exc).__name__,
            error=str(exc),
        )
    else:
        warmup = warmup_extension_for_load(mode=None)
        emit_backend_message(
            "[wan22.gguf] attention_sram warmup",
            logger=logger_name,
            stage=stage,
            mode=warmup.mode.value,
            attempted=warmup.attempted,
            loaded=warmup.loaded,
            ready=warmup.ready,
            jit_build=warmup.build_enabled,
            detail=warmup.detail,
        )
    emit_backend_message(
        "[wan22.gguf] mounted stage model",
        logger=logger_name,
        file=os.path.basename(gguf_path),
        dequantize=dequantize,
        mount_device=mount_device,
    )
    return model
