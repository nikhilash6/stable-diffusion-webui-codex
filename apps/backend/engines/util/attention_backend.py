"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Engine-side attention backend selection for diffusers pipelines.
Applies an attention backend (PyTorch SDPA / xFormers) based on explicit input or runtime memory config, failing loud on invalid runtime config.

Symbols (top-level; keep in sync; no ghosts):
- `_resolve_attention_config` (function): Resolves runtime attention config and fails loud when required fields are missing/invalid.
- `_get_selected_backend` (function): Reads effective attention backend from runtime memory config.
- `_selected_sdpa_flags` (function): Reads effective SDPA enable flags (`flash`, `mem_efficient`) from runtime memory config.
- `apply_to_diffusers_pipeline` (function): Applies the chosen attention backend to a diffusers pipeline or raises with cause.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Any, Optional

from apps.backend.runtime.memory import memory_management


_LOGGER = get_backend_logger("backend.engines.util.attention")
_FLASH_DIFFUSERS_FALLBACK_LOGGED: set[str] = set()


def _resolve_attention_config() -> Any:
    manager = getattr(memory_management, "manager", None)
    if manager is None:
        raise RuntimeError("memory_management.manager is not initialized")
    config = getattr(manager, "config", None)
    if config is None:
        raise RuntimeError("memory_management.manager.config is not available")
    attention_cfg = getattr(config, "attention", None)
    if attention_cfg is None:
        raise RuntimeError("memory_management.manager.config.attention is not available")
    return attention_cfg


def _get_selected_backend() -> str:
    attention_cfg = _resolve_attention_config()
    backend = getattr(attention_cfg, "backend", None)
    backend_value = getattr(backend, "value", backend)
    if not isinstance(backend_value, str) or not backend_value.strip():
        raise RuntimeError(f"Invalid runtime attention backend value: {backend!r}")
    return backend_value.strip()


def _selected_sdpa_flags() -> tuple[bool, bool]:
    attention_cfg = _resolve_attention_config()
    enable_flash = getattr(attention_cfg, "enable_flash", None)
    enable_mem_efficient = getattr(attention_cfg, "enable_mem_efficient", None)
    if not isinstance(enable_flash, bool):
        raise RuntimeError(f"Invalid runtime SDPA flag 'enable_flash': {enable_flash!r}")
    if not isinstance(enable_mem_efficient, bool):
        raise RuntimeError(f"Invalid runtime SDPA flag 'enable_mem_efficient': {enable_mem_efficient!r}")
    return enable_flash, enable_mem_efficient


def apply_to_diffusers_pipeline(pipe: Any, *, backend: Optional[str] = None, logger=None) -> str:
    """Apply the chosen attention backend to a diffusers pipeline (if supported).

    Returns the effective backend string applied or attempted.
    """
    choice = (backend or _get_selected_backend()).lower().strip()
    if choice not in ("pytorch", "xformers", "split", "quad"):
        raise ValueError(f"Invalid attention backend '{choice}'. Allowed: pytorch, xformers, split, quad")

    # Torch SDPA (Flash/Math/Mem) — default in PyTorch 2.x
    if choice == "pytorch":
        # If xformers was previously enabled, disable it when possible (failure is an error now)
        if hasattr(pipe, "disable_xformers_memory_efficient_attention"):
            pipe.disable_xformers_memory_efficient_attention()
        enable_flash, enable_mem_efficient = _selected_sdpa_flags()

        if enable_flash and not enable_mem_efficient:
            flash_unavailable_reason: str | None = None
            import torch  # type: ignore

            if not torch.cuda.is_available():
                flash_unavailable_reason = "cuda_unavailable"
            else:
                try:
                    major, _minor = torch.cuda.get_device_capability()
                    if major < 8:
                        flash_unavailable_reason = f"compute_capability_sm{major}x"
                except Exception:
                    flash_unavailable_reason = "capability_probe_failed"
            if flash_unavailable_reason is not None:
                if flash_unavailable_reason not in _FLASH_DIFFUSERS_FALLBACK_LOGGED:
                    _FLASH_DIFFUSERS_FALLBACK_LOGGED.add(flash_unavailable_reason)
                    (logger or _LOGGER).warning(
                        "[attention] requested SDPA flash for diffusers, but flash appears unavailable (%s); "
                        "expect internal fallback to non-flash kernels.",
                        flash_unavailable_reason,
                    )

        if logger:
            logger.info(
                "attention backend: pytorch (sdpa flash=%s mem_efficient=%s)",
                enable_flash,
                enable_mem_efficient,
            )
        return "pytorch"

    # xFormers memory-efficient attention
    if choice == "xformers":
        if not hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            raise RuntimeError("Pipeline does not expose xformers enable hook")
        pipe.enable_xformers_memory_efficient_attention()
        if logger:
            logger.info("attention backend: xformers")
        return "xformers"

    if choice in {"split", "quad"}:
        raise NotImplementedError(
            "Attention backend "
            f"'{choice}' is not supported for diffusers pipelines. Use 'pytorch' or 'xformers'.",
        )

    # Should not reach here
    return choice
