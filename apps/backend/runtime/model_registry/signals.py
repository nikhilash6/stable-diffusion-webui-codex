"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Lazy state-dict signal extraction helpers for model detection.
Wraps a checkpoint mapping in a `SignalBundle` that exposes keys and lazily computed shapes, plus small helpers used across detectors.

Symbols (top-level; keep in sync; no ghosts):
- `SignalBundle` (dataclass): State-dict wrapper exposing keys and lazy/cached shape lookup.
- `_resolve_source_format` (function): Resolves source format hints (`safetensors|gguf`) from a mapping/view chain.
- `build_bundle` (function): Builds a `SignalBundle` without materializing all tensors.
- `count_blocks` (function): Counts sequential blocks matching a template prefix pattern.
- `has_all_keys` (function): Returns True if all required keys exist in a bundle.
- `get_tensor_dtype` (function): Best-effort dtype name extraction for a tensor-like object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Tuple


@dataclass
class SignalBundle:
    state_dict: Mapping[str, Any]
    keys: Tuple[str, ...]
    shapes: MutableMapping[str, Tuple[int, ...]]
    source_format: str | None = None

    def shape(self, key: str) -> Tuple[int, ...] | None:
        # Fast path: cached
        cached = self.shapes.get(key)
        if cached is not None:
            return cached
        # Header/index-backed shape lookup (no tensor materialization) when available.
        shape_getter = getattr(self.state_dict, "shape_of", None)
        if callable(shape_getter):
            try:
                shape_from_mapping = shape_getter(key)
            except Exception:
                shape_from_mapping = None
            if shape_from_mapping is not None:
                shape_tuple = tuple(int(x) for x in shape_from_mapping)
                try:
                    self.shapes[key] = shape_tuple
                except Exception:
                    pass
                return shape_tuple
        # Lazy path: load a single tensor from the mapping (avoids materializing thousands of tensors)
        try:
            v = self.state_dict[key]
        except Exception:
            return None
        shape = getattr(v, "shape", None)
        if shape is None and hasattr(v, "size"):
            try:
                shape = tuple(int(x) for x in v.size())  # type: ignore[arg-type]
            except Exception:
                shape = None
        if shape is None:
            return None
        try:
            # Cache for subsequent calls when shapes is mutable
            self.shapes[key] = tuple(int(x) for x in shape)  # type: ignore[index]
        except Exception:
            pass
        return tuple(int(x) for x in shape)

    def has_prefix(self, prefix: str) -> bool:
        return any(k.startswith(prefix) for k in self.keys)

    def is_gguf_quantized(self) -> bool:
        fmt = str(self.source_format or "").strip().lower()
        if fmt == "gguf":
            return True
        if fmt in {"safetensor", "safetensors"}:
            return False
        values = getattr(self.state_dict, "values", None)
        if not callable(values):
            return False
        try:
            from apps.backend.quantization.tensor import CodexParameter
        except Exception:
            return False
        for idx, value in enumerate(values()):
            if isinstance(value, CodexParameter) and value.qtype is not None:
                return True
            if idx >= 63:
                break
        return False


def _resolve_source_format(state_dict: Mapping[str, Any]) -> str | None:
    current: object | None = state_dict
    seen_ids: set[int] = set()
    while current is not None:
        marker = id(current)
        if marker in seen_ids:
            break
        seen_ids.add(marker)

        source_format = getattr(current, "source_format", None)
        if isinstance(source_format, str) and source_format.strip():
            return source_format.strip().lower()

        filepath = getattr(current, "filepath", None)
        if isinstance(filepath, str) and filepath:
            lower = filepath.lower()
            if lower.endswith(".safetensors") or lower.endswith(".safetensor"):
                return "safetensors"
            if lower.endswith(".gguf"):
                return "gguf"

        current = getattr(current, "_base", None)
    return None


def build_bundle(state_dict: Mapping[str, Any]) -> SignalBundle:
    # Do not materialize all tensors; only list keys and compute shapes lazily via SignalBundle.shape()
    keys = tuple(state_dict.keys())
    shapes: dict[str, Tuple[int, ...]] = {}
    source_format = _resolve_source_format(state_dict)
    return SignalBundle(state_dict=state_dict, keys=keys, shapes=shapes, source_format=source_format)


def count_blocks(keys: Iterable[str], template: str) -> int:
    """Count sequential blocks in keys following a zero-based template.

    Example template: ``"model.diffusion_model.input_blocks.{}."``
    """
    count = 0
    while True:
        prefix = template.format(count)
        if not any(k.startswith(prefix) for k in keys):
            break
        count += 1
    return count


def has_all_keys(bundle: SignalBundle, *required: str) -> bool:
    return all(k in bundle.state_dict for k in required)


def get_tensor_dtype(tensor: Any) -> str | None:
    return getattr(getattr(tensor, "dtype", None), "name", None)
