"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: State-dict diagnostics and helper utilities for runtime codepaths.
Provides prefix-based parameter counting, post-quant state dict snapshots, and GGUF static-entry summaries.

Symbols (top-level; keep in sync; no ghosts):
- `calculate_parameters` (function): Computes parameter count for a state dict subtree (by prefix).
- `get_state_dict_after_quant` (function): Extracts a post-quant state dict view for a model (prefix-aware; fails loud on NF4/FP4 weights).
- `beautiful_print_gguf_state_dict_statics` (function): Prints/returns a compact summary of GGUF state-dict “static” tensor entries.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging

import torch

_log = get_backend_logger("backend.runtime.utils")


def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params


def get_state_dict_after_quant(model, prefix=""):
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if hasattr(weight, "bnb_quantized"):
            raise NotImplementedError(
                "NF4/FP4 is not supported. "
                "Convert the model to GGUF or use a safetensors fp16/bf16/fp32 checkpoint."
            )

    sd = model.state_dict()
    sd = {(prefix + k): v.clone() for k, v in sd.items()}
    return sd


def beautiful_print_gguf_state_dict_statics(state_dict):
    try:
        from apps.backend.quantization.tensor import CodexParameter
    except Exception:
        return
    type_counts = {}
    for k, v in state_dict.items():
        if isinstance(v, CodexParameter) and v.qtype is not None:
            type_name = v.qtype.name
            if type_name in type_counts:
                type_counts[type_name] += 1
            else:
                type_counts[type_name] = 1
    _log.info("GGUF state dict: %s", type_counts)
    return


__all__ = [
    "beautiful_print_gguf_state_dict_statics",
    "calculate_parameters",
    "get_state_dict_after_quant",
]
