"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GGUF runtime operations backed by `apps.backend.quantization` (CodexQuantization).
Provides direct dequantization helpers for runtime GGUF parameters and a structural guard for unsupported packed artifacts.

Symbols (top-level; keep in sync; no ghosts):
- `CodexParameter` (class): Packed GGUF tensor wrapper (imported from `apps.backend.quantization.tensor`).
- `is_packed_gguf_artifact` (function): Returns True when a tensor/parameter still carries removed packed-artifact markers.
- `dequantize_tensor` (function): Dequantize a `CodexParameter` to a float tensor (pass-through for non-GGUF tensors).
- `__all__` (constant): Public export list for GGUF runtime operations.
"""

from __future__ import annotations

from apps.backend.quantization.api import dequantize as codex_dequantize
from apps.backend.quantization.tensor import CodexParameter

__all__ = [
    "CodexParameter",
    "dequantize_tensor",
    "is_packed_gguf_artifact",
]


def is_packed_gguf_artifact(tensor) -> bool:
    """Return True when a tensor/parameter still carries removed packed-artifact markers."""
    if tensor is None:
        return False
    return hasattr(tensor, "keymap_id") or hasattr(tensor, "kernel_id")


def dequantize_tensor(tensor):
    """Return a dequantized float tensor (or pass-through for non-quant tensors)."""
    if tensor is None:
        return None
    if not isinstance(tensor, CodexParameter) or tensor.qtype is None:
        return tensor
    return codex_dequantize(tensor)
