"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend runtime memory manager singleton (bind config -> live manager instance).
Exposes the active `CodexMemoryManager` as `manager` and provides a small set of reconfiguration helpers for API/runtime code.

Symbols (top-level; keep in sync; no ghosts):
- `manager` (constant): Active `CodexMemoryManager` instance (replaced on `reinitialize`).
- `_bind_config` (function): Internal initializer; unloads any previous manager, creates a new `CodexMemoryManager`, and wires globals.
- `reinitialize` (function): Replaces the active memory manager with a new `RuntimeMemoryConfig`.
- `memory_snapshot` (function): Returns a JSON-serializable snapshot of current memory/config state.
- `switch_primary_device` (function): Changes the primary device backend (cpu/cuda/mps/xpu/directml) at runtime (returns success bool).
- `set_attention_backend` (function): Sets the global attention backend (pytorch/xformers/split/quad) at runtime (returns success bool).
- `set_component_backend` (function): Sets preferred backend for a role (core/vae/tenc/vision/intermediate) at runtime (returns success bool).
- `set_component_dtype` (function): Sets forced dtype for a role (core/vae/tenc/vision/intermediate) at runtime (returns success bool).
- `set_component_compute_dtype` (function): Sets forced compute dtype for a role (core/vae/tenc/vision/intermediate) at runtime (returns success bool).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from copy import deepcopy

from apps.backend.infra.config import args as config_args
from apps.backend.runtime.load_authority import (
    LoadAuthorityStage,
    coordinator_load_permit,
)

from .config import AttentionBackend, DeviceBackend, DeviceRole, RuntimeMemoryConfig
from .manager import CodexMemoryManager


logger = get_backend_logger("backend.memory.facade")


_CONFIG: RuntimeMemoryConfig | None = None
manager: CodexMemoryManager


def _bind_config(config: RuntimeMemoryConfig) -> None:
    global _CONFIG, manager

    old_manager: CodexMemoryManager | None = globals().get("manager")
    if old_manager is not None:
        with coordinator_load_permit(
            owner="runtime.memory.memory_management._bind_config",
            stage=LoadAuthorityStage.CLEANUP,
        ):
            old_manager.unload_all_models()

    _CONFIG = config
    manager = CodexMemoryManager.create(config)


def reinitialize(config: RuntimeMemoryConfig) -> None:
    """Replace the active memory manager with a new configuration."""
    logger.info(
        "Reinitializing memory manager (device_backend=%s, gpu_prefer_construct=%s)",
        getattr(config.device_backend, "value", config.device_backend),
        getattr(config, "gpu_prefer_construct", False),
    )
    _bind_config(config)


_bind_config(config_args.memory_config)


def memory_snapshot() -> dict[str, object]:
    """Expose a JSON-friendly view of the current runtime memory state."""
    return manager.memory_snapshot()


def switch_primary_device(backend: str) -> bool:
    """Switch the primary device backend explicitly.

    Accepts one of: 'cpu' | 'cuda' | 'mps' | 'xpu' | 'directml'.
    Returns True if the configuration changed and a reinitialization occurred.
    """

    normalized = (backend or "").strip().lower()
    mapping = {
        "cpu": DeviceBackend.CPU,
        "cuda": DeviceBackend.CUDA,
        "mps": DeviceBackend.MPS,
        "xpu": DeviceBackend.XPU,
        "directml": DeviceBackend.DIRECTML,
    }
    if normalized not in mapping:
        raise ValueError(f"Invalid device backend '{backend}'. Allowed: cpu, cuda, mps, xpu, directml")

    current_cfg = manager.config
    target = mapping[normalized]
    if current_cfg.device_backend == target:
        return False

    new_cfg = deepcopy(current_cfg)
    new_cfg.device_backend = target
    reinitialize(new_cfg)
    logger.info("[memory] primary device switched to %s", normalized)
    return True


def set_attention_backend(backend: str) -> bool:
    """Set the global attention backend and reinitialize the memory manager.

    This is a runtime change (no backend restart) but it triggers a full unload/reload
    cycle via `reinitialize(...)`.

    Accepted values:
    - pytorch
    - xformers
    - split
    - quad
    """

    normalized = (backend or "").strip().lower()
    mapping = {
        "pytorch": AttentionBackend.PYTORCH,
        "xformers": AttentionBackend.XFORMERS,
        "split": AttentionBackend.SPLIT,
        "quad": AttentionBackend.QUAD,
    }
    if normalized not in mapping:
        allowed = ", ".join(sorted(mapping))
        raise ValueError(f"Invalid attention backend '{backend}'. Allowed: {allowed}")

    target = mapping[normalized]
    current_cfg = manager.config
    if current_cfg.attention.backend == target:
        return False

    if target == AttentionBackend.XFORMERS:
        if current_cfg.disable_xformers:
            raise ValueError("xformers attention backend requested, but xformers is disabled via --disable-xformers.")
        try:
            import xformers  # type: ignore # noqa: F401
        except Exception as exc:
            raise ValueError(f"xformers attention backend requested, but xformers is not available: {exc}") from exc

    new_cfg = deepcopy(current_cfg)
    new_cfg.attention.backend = target
    # Preserve SDPA policy flags when backend changes. They are only consumed when
    # backend==PYTORCH, so mutating them on xformers/split/quad switches causes
    # silent policy drift when users switch back to pytorch.

    reinitialize(new_cfg)
    logger.info("[memory] attention backend set to %s", target.value)
    return True


def set_component_backend(role: str, backend: str) -> bool:
    """Update the preferred backend for a component role and reinitialize.

    role: one of core|text_encoder|vae|clip_vision|intermediate
    backend: one of auto|cpu|cuda|mps|xpu|directml
    """

    role_norm = str(role).strip().lower()
    mapping_role = {
        "core": DeviceRole.CORE,
        "text_encoder": DeviceRole.TEXT_ENCODER,
        "te": DeviceRole.TEXT_ENCODER,
        "vae": DeviceRole.VAE,
        "clip_vision": DeviceRole.CLIP_VISION,
        "vision": DeviceRole.CLIP_VISION,
        "intermediate": DeviceRole.INTERMEDIATE,
    }
    if role_norm not in mapping_role:
        raise ValueError(f"Invalid role '{role}'")
    role_enum = mapping_role[role_norm]

    backend_norm = (backend or "").strip().lower()
    mapping_backend = {
        "auto": DeviceBackend.AUTO,
        "cpu": DeviceBackend.CPU,
        "cuda": DeviceBackend.CUDA,
        "mps": DeviceBackend.MPS,
        "xpu": DeviceBackend.XPU,
        "directml": DeviceBackend.DIRECTML,
    }
    if backend_norm not in mapping_backend:
        raise ValueError(f"Invalid backend '{backend}'")

    current_cfg = manager.config
    new_cfg = deepcopy(current_cfg)
    if role_enum == DeviceRole.CORE:
        new_cfg.device_backend = mapping_backend[backend_norm]

    policy = new_cfg.component_policy(role_enum)
    policy.preferred_backend = mapping_backend[backend_norm]
    reinitialize(new_cfg)
    logger.info("[memory] backend for %s set to %s", role_enum.value, backend_norm)
    return True


def set_component_dtype(role: str, dtype: str) -> bool:
    """Force dtype for a component role. Use 'auto' to clear overrides."""

    role_norm = str(role).strip().lower()
    mapping_role = {
        "core": DeviceRole.CORE,
        "text_encoder": DeviceRole.TEXT_ENCODER,
        "te": DeviceRole.TEXT_ENCODER,
        "vae": DeviceRole.VAE,
        "clip_vision": DeviceRole.CLIP_VISION,
        "vision": DeviceRole.CLIP_VISION,
        "intermediate": DeviceRole.INTERMEDIATE,
    }
    if role_norm not in mapping_role:
        raise ValueError(f"Invalid role '{role}'")
    role_enum = mapping_role[role_norm]

    dtype_norm = (dtype or "").strip().lower()
    if dtype_norm not in {"auto", "fp16", "bf16", "fp32"}:
        raise ValueError("dtype must be one of auto|fp16|bf16|fp32")

    current_cfg = manager.config
    new_cfg = deepcopy(current_cfg)

    # Reset precision flags for the target role
    flags = new_cfg.precision
    if role_enum == DeviceRole.CORE:
        flags.core_fp16 = flags.core_bf16 = flags.core_fp8_e4m3fn = flags.core_fp8_e5m2 = False
    elif role_enum == DeviceRole.TEXT_ENCODER:
        flags.clip_fp16 = flags.clip_fp32 = flags.clip_bf16 = flags.clip_fp8_e4m3fn = flags.clip_fp8_e5m2 = False
    elif role_enum == DeviceRole.VAE:
        flags.vae_fp16 = flags.vae_fp32 = flags.vae_bf16 = False

    policy = new_cfg.component_policy(role_enum)
    policy.forced_dtype = None

    if dtype_norm != "auto":
        torch_name = {
            "fp16": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
        }[dtype_norm]
        policy.forced_dtype = torch_name
        if role_enum == DeviceRole.CORE:
            flags.core_fp16 = dtype_norm == "fp16"
            flags.core_bf16 = dtype_norm == "bf16"
        elif role_enum == DeviceRole.TEXT_ENCODER:
            flags.clip_fp16 = dtype_norm == "fp16"
            flags.clip_bf16 = dtype_norm == "bf16"
            flags.clip_fp32 = dtype_norm == "fp32"
        elif role_enum == DeviceRole.VAE:
            flags.vae_fp16 = dtype_norm == "fp16"
            flags.vae_bf16 = dtype_norm == "bf16"
            flags.vae_fp32 = dtype_norm == "fp32"

    reinitialize(new_cfg)
    logger.info("[memory] dtype for %s set to %s", role_enum.value, dtype_norm)
    return True


def set_component_compute_dtype(role: str, dtype: str) -> bool:
    """Force compute dtype for a component role. Use 'auto' to clear overrides."""

    role_norm = str(role).strip().lower()
    mapping_role = {
        "core": DeviceRole.CORE,
        "text_encoder": DeviceRole.TEXT_ENCODER,
        "te": DeviceRole.TEXT_ENCODER,
        "vae": DeviceRole.VAE,
        "clip_vision": DeviceRole.CLIP_VISION,
        "vision": DeviceRole.CLIP_VISION,
        "intermediate": DeviceRole.INTERMEDIATE,
    }
    if role_norm not in mapping_role:
        raise ValueError(f"Invalid role '{role}'")
    role_enum = mapping_role[role_norm]

    dtype_norm = (dtype or "").strip().lower()
    if dtype_norm not in {"auto", "fp16", "bf16", "fp32"}:
        raise ValueError("dtype must be one of auto|fp16|bf16|fp32")

    current_cfg = manager.config
    new_cfg = deepcopy(current_cfg)
    policy = new_cfg.component_policy(role_enum)
    policy.forced_compute_dtype = None

    if dtype_norm != "auto":
        torch_name = {
            "fp16": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
        }[dtype_norm]
        policy.forced_compute_dtype = torch_name

    reinitialize(new_cfg)
    logger.info("[memory] compute dtype for %s set to %s", role_enum.value, dtype_norm)
    return True


__all__ = [
    "manager",
    "reinitialize",
    "memory_snapshot",
    "switch_primary_device",
    "set_attention_backend",
    "set_component_backend",
    "set_component_dtype",
    "set_component_compute_dtype",
]
