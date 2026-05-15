"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime utility facade (state_dict views, SafeTensors loading, nested device moves, and checkpoint IO).
This module keeps a stable import surface for callers while delegating implementations to focused submodules:
`state_dict/views.py`, `checkpoint/io.py`, `attr_access.py`, `nested.py`, and `state_dict/tools.py`.

Symbols (top-level; keep in sync; no ghosts):
- `KeyPrefixView` (class): Mapping view that exposes `base` keys under a fixed prefix without materializing values.
- `FilterPrefixView` (class): Mapping view that filters keys by prefix and optionally re-prefixes them lazily.
- `KeyspaceLookupView` (class): Mapping view that exposes one keyspace through source-key lookup without mutating the underlying state dict.
- `CastOnGetView` (class): Mapping view that casts tensors/values on access (`__getitem__`) to a target dtype/device (no eager conversion).
- `read_arbitrary_config` (function): Reads a best-effort config from a directory (supports JSON/YAML-like inputs where present).
- `load_torch_file` (function): Loads a torch checkpoint with safe-load options (prefers safe loaders, falls back to pickle loader when allowed).
- `_load_gguf_state_dict` (function): Loads a GGUF state dict from a `.gguf` file path (used by runtime helpers without importing heavy ops).
- `load_gguf_state_dict` (function): Public GGUF state-dict loader with explicit dequant policy control.
- `LazySafetensorsDict` (class): Lazy mapping over a SafeTensors file; keeps a single handle and loads tensors on demand.
- `_load_pickled_checkpoint` (function): Loads a pickled checkpoint using the restricted/guarded unpickler (`checkpoint_pickle`).
- `set_attr` (function): Sets a nested attribute on an object by dotted path (type-aware).
- `set_attr_raw` (function): Sets a nested attribute by dotted path without conversions.
- `copy_to_param` (function): Copies a tensor/value into an existing `nn.Parameter` or tensor attribute.
- `get_attr` (function): Reads a nested attribute by dotted path.
- `get_attr_with_parent` (function): Reads a nested attribute and returns `(parent, attr_name, value)` for patching.
- `calculate_parameters` (function): Computes parameter count for a state dict subtree (by prefix).
- `tensor2parameter` (function): Converts a tensor-like to an `nn.Parameter`.
- `fp16_fix` (function): Applies fp16 compatibility fixes for legacy checkpoints (best-effort).
- `dtype_to_element_size` (function): Returns element size in bytes for a dtype name/torch dtype.
- `nested_compute_size` (function): Computes total size of a nested object tree (dict/list/tuples) given element size.
- `nested_move_to_device` (function): Recursively moves tensors/parameters in a nested structure to a device/dtype.
- `get_state_dict_after_quant` (function): Extracts a post-quant state dict view for a model (prefix-aware).
- `beautiful_print_gguf_state_dict_statics` (function): Prints/returns a compact summary of GGUF state-dict “static” tensor entries.
"""

from __future__ import annotations

from .attr_access import (
    copy_to_param,
    get_attr,
    get_attr_with_parent,
    set_attr,
    set_attr_raw,
    tensor2parameter,
)
from .checkpoint.io import (
    _load_gguf_state_dict,
    _load_pickled_checkpoint,
    load_gguf_state_dict,
    load_torch_file,
    read_arbitrary_config,
)
from .nested import dtype_to_element_size, fp16_fix, nested_compute_size, nested_move_to_device
from .state_dict.tools import calculate_parameters, beautiful_print_gguf_state_dict_statics, get_state_dict_after_quant
from .state_dict.views import CastOnGetView, FilterPrefixView, KeyPrefixView, KeyspaceLookupView, LazySafetensorsDict

__all__ = [
    "CastOnGetView",
    "FilterPrefixView",
    "KeyPrefixView",
    "LazySafetensorsDict",
    "KeyspaceLookupView",
    "_load_gguf_state_dict",
    "_load_pickled_checkpoint",
    "beautiful_print_gguf_state_dict_statics",
    "calculate_parameters",
    "copy_to_param",
    "dtype_to_element_size",
    "fp16_fix",
    "get_attr",
    "get_attr_with_parent",
    "get_state_dict_after_quant",
    "load_gguf_state_dict",
    "load_torch_file",
    "nested_compute_size",
    "nested_move_to_device",
    "read_arbitrary_config",
    "set_attr",
    "set_attr_raw",
    "tensor2parameter",
]
