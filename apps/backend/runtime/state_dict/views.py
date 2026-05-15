"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Lightweight mapping views for state_dict handling.
Provides prefix/filter/keyspace-lookup/computed-keyspace/cast views plus a SafeTensors-backed lazy dict used to stream state_dict
preprocessing with configurable tensor device placement.

Symbols (top-level; keep in sync; no ghosts):
- `KeyPrefixView` (class): Mapping view that exposes `base` keys under a fixed prefix without materializing values.
- `FilterPrefixView` (class): Mapping view that filters keys by prefix and optionally re-prefixes them lazily.
- `KeyspaceLookupView` (class): Mapping view that exposes one keyspace through source-key lookup without mutating the underlying state dict.
- `ComputedKeyspaceView` (class): Mapping view that mixes direct source-key lookups with computed tensor transforms (e.g. fused/unfused QKV).
- `CastOnGetView` (class): Mapping view that casts tensors/values on access (`__getitem__`) to a target dtype/device (no eager conversion).
- `LazySafetensorsDict` (class): Lazy SafeTensors-backed state_dict; keeps a single handle and loads tensors on demand to the configured device (Windows: materializes on first access to avoid repeated opens).
"""

from __future__ import annotations

import sys
import threading
from collections.abc import MutableMapping
from pathlib import Path
from typing import Callable, Dict

import torch
from safetensors.torch import safe_open


def _inherit_source_metadata(view: object, base: object) -> None:
    source_format = getattr(base, "source_format", None)
    source_path = getattr(base, "source_path", None)
    primary_dtype_hint = getattr(base, "primary_dtype_hint", None)

    if isinstance(source_format, str) and source_format:
        setattr(view, "source_format", source_format)
    if isinstance(source_path, str) and source_path:
        setattr(view, "source_path", source_path)
    if isinstance(primary_dtype_hint, str) and primary_dtype_hint:
        setattr(view, "primary_dtype_hint", primary_dtype_hint)


class KeyPrefixView(MutableMapping):
    """Lightweight mapping view that exposes `base` keys with a fixed prefix.

    - Does not materialize tensor values; delegates to `base[key_without_prefix]` on access.
    - Deletions and sets propagate to the underlying mapping.
    - Useful to avoid rebuilding huge state_dicts on CPU.
    """

    def __init__(self, base: MutableMapping, prefix: str):
        self._base = base
        self._prefix = prefix
        _inherit_source_metadata(self, base)
        header_shapes = getattr(base, "header_shapes", None)
        if isinstance(header_shapes, dict):
            self.header_shapes = {f"{prefix}{str(key)}": shape for key, shape in header_shapes.items()}

    def _strip(self, k: str) -> str:
        if not k.startswith(self._prefix):
            raise KeyError(k)
        return k[len(self._prefix) :]

    def __getitem__(self, k: str):
        return self._base[self._strip(k)]

    def __setitem__(self, k: str, v):
        self._base[self._strip(k)] = v

    def __delitem__(self, k: str):
        del self._base[self._strip(k)]

    def __iter__(self):
        for k in self._base.keys():
            yield f"{self._prefix}{k}"

    def __len__(self) -> int:
        try:
            return len(self._base.keys())
        except Exception:
            # Fallback: iterate
            return sum(1 for _ in self.__iter__())

    def shape_of(self, k: str):
        shape_getter = getattr(self._base, "shape_of", None)
        if not callable(shape_getter):
            return None
        try:
            return shape_getter(self._strip(k))
        except Exception:
            return None


class FilterPrefixView(MutableMapping):
    """View over keys under a given prefix, optionally re-prefixed lazily.

    - base: mapping with original keys (e.g., LazySafetensorsDict or KeyPrefixView)
    - prefix: filter only keys that start with this
    - new_prefix: keys presented by this view will start with new_prefix instead
    """

    def __init__(self, base: MutableMapping, prefix: str, new_prefix: str = ""):
        self._base = base
        self._prefix = prefix
        self._new_prefix = new_prefix
        _inherit_source_metadata(self, base)
        header_shapes = getattr(base, "header_shapes", None)
        if isinstance(header_shapes, dict):
            mapped_shapes: Dict[str, tuple[int, ...]] = {}
            for key, shape in header_shapes.items():
                key_str = str(key)
                if not key_str.startswith(prefix):
                    continue
                presented = self._present_key(key_str)
                mapped_shapes[presented] = shape
            self.header_shapes = mapped_shapes

    def _to_base_key(self, k: str) -> str:
        # Map presented key 'k' back to the underlying base mapping key.
        if self._new_prefix:
            if k.startswith(self._new_prefix):
                return self._prefix + k[len(self._new_prefix) :]
            # If caller already uses base prefix, pass-through
            if k.startswith(self._prefix):
                return k
            # Fallback: assume k is suffix; prepend prefix
            return self._prefix + k
        else:
            # Presented keys are suffix-only; base keys use prefix
            if k.startswith(self._prefix):
                return k
            return self._prefix + k

    def _present_key(self, base_key: str) -> str:
        if not self._prefix:
            if self._new_prefix:
                return f"{self._new_prefix}{base_key}"
            return base_key
        suffix = base_key[len(self._prefix) :]
        if self._new_prefix:
            return f"{self._new_prefix}{suffix}"
        return suffix

    def __getitem__(self, k: str):
        return self._base[self._to_base_key(k)]

    def __setitem__(self, k: str, v):
        self._base[self._to_base_key(k)] = v

    def __delitem__(self, k: str):
        del self._base[self._to_base_key(k)]

    def __iter__(self):
        for k in self._base.keys():
            if k.startswith(self._prefix):
                out = self._new_prefix + k[len(self._prefix) :]
                yield out

    def __len__(self) -> int:
        c = 0
        for _ in self.__iter__():
            c += 1
        return c

    def materialize(self, *, return_mapping: bool = False):
        """Realise all tensors matching the prefix into a concrete dict.

        Prefers calling the underlying mapping's `materialize` helper when
        available so SafeTensors files are streamed with a single handle open.
        """

        materializer = getattr(self._base, "materialize", None)
        if callable(materializer):
            try:
                return materializer(prefix=self._prefix, new_prefix=self._new_prefix, return_mapping=return_mapping)
            except TypeError:
                result = materializer(prefix=self._prefix, new_prefix=self._new_prefix)
                if return_mapping:
                    raise
                return result

        out: Dict[str, object] = {}
        mapping: Dict[str, str] = {}
        for key in self._base.keys():
            if not key.startswith(self._prefix):
                continue
            presented = self._present_key(key)
            out[presented] = self._base[key]
            mapping[presented] = key
        if return_mapping:
            return out, mapping
        return out

    def shape_of(self, k: str):
        shape_getter = getattr(self._base, "shape_of", None)
        if not callable(shape_getter):
            return None
        try:
            return shape_getter(self._to_base_key(k))
        except Exception:
            return None


class KeyspaceLookupView(MutableMapping):
    """Present a resolved lookup keyspace over an underlying mapping lazily.

    - base: original mapping (e.g., LazySafetensorsDict)
    - mapping: dict[presented_key] -> source_key in the base mapping
    - Does not materialize any tensor unless __getitem__ is called.
    """

    def __init__(self, base: MutableMapping, mapping: dict[str, str]):
        self._base = base
        self._map = dict(mapping)
        _inherit_source_metadata(self, base)
        header_shapes = getattr(base, "header_shapes", None)
        if isinstance(header_shapes, dict):
            mapped_shapes: Dict[str, tuple[int, ...]] = {}
            for canonical_key, source_key in self._map.items():
                shape = header_shapes.get(source_key)
                if shape is not None:
                    mapped_shapes[canonical_key] = shape
            self.header_shapes = mapped_shapes

    def __getitem__(self, k: str):
        return self._base[self._map[k]]

    def __setitem__(self, k: str, v):
        self._map[k] = k
        self._base[k] = v

    def __delitem__(self, k: str):
        old = self._map.pop(k, None)
        if old is not None and old in self._base:
            del self._base[old]

    def __iter__(self):
        return iter(self._map.keys())

    def __len__(self):
        return len(self._map)

    def __contains__(self, k: object) -> bool:
        return k in self._map

    def keys(self):
        return list(self._map.keys())

    def items(self):
        for k in self._map.keys():
            yield k, self._base[self._map[k]]

    def shape_of(self, key: str):
        shape_getter = getattr(self._base, "shape_of", None)
        if not callable(shape_getter):
            return None
        source_key = self._map.get(key)
        if source_key is None:
            return None
        try:
            return shape_getter(source_key)
        except Exception:
            return None


def _copy_parameter_like(reference: object, raw_data: torch.Tensor, *, logical_shape: tuple[int, ...]):
    copy_with_data = getattr(reference, "copy_with_data", None)
    if not callable(copy_with_data):
        return raw_data
    copied = copy_with_data(raw_data)
    if hasattr(copied, "real_shape"):
        try:
            copied.real_shape = torch.Size(tuple(int(v) for v in logical_shape))
        except Exception:
            pass
    return copied


def _concat_tensor_rows(values: tuple[object, ...], *, logical_shape: tuple[int, ...]):
    if not values:
        raise RuntimeError("concat_dim0 requires at least one tensor")
    if all(hasattr(value, "qtype") and getattr(value, "qtype", None) is not None for value in values):
        qtypes = {getattr(value, "qtype", None) for value in values}
        if len(qtypes) != 1:
            raise RuntimeError("concat_dim0 cannot mix different GGUF quantization types")
        raw = torch.cat(tuple(value.data for value in values), dim=0)
        return _copy_parameter_like(values[0], raw, logical_shape=logical_shape)
    tensors = []
    for value in values:
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"concat_dim0 expects tensor values, got {type(value).__name__}")
        tensors.append(value)
    return torch.cat(tuple(tensors), dim=0)


def _split_tensor_rows(value: object, *, chunks: int, index: int, logical_shape: tuple[int, ...]):
    if hasattr(value, "qtype") and getattr(value, "qtype", None) is not None:
        parts = torch.chunk(value.data, chunks, dim=0)
        return _copy_parameter_like(value, parts[index], logical_shape=logical_shape)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"split_dim0 expects a tensor value, got {type(value).__name__}")
    return torch.chunk(value, chunks, dim=0)[index]


def _swap_tensor_row_halves(value: object, *, logical_shape: tuple[int, ...]):
    if hasattr(value, "qtype") and getattr(value, "qtype", None) is not None:
        rows = int(value.data.shape[0])
        if rows % 2 != 0:
            raise RuntimeError(f"swap_dim0_halves requires an even row count, got {rows}")
        half = rows // 2
        swapped = torch.cat((value.data[half:], value.data[:half]), dim=0)
        return _copy_parameter_like(value, swapped, logical_shape=logical_shape)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"swap_dim0_halves expects a tensor value, got {type(value).__name__}")
    rows = int(value.shape[0])
    if rows % 2 != 0:
        raise RuntimeError(f"swap_dim0_halves requires an even row count, got {rows}")
    half = rows // 2
    return torch.cat((value[half:], value[:half]), dim=0)


class ComputedKeyspaceView(MutableMapping):
    """Present a mixed direct/computed lookup keyspace over an underlying mapping lazily.

    - `mapping` exposes canonical keys that map directly to source keys in `base`.
    - `computed` exposes canonical keys whose values are derived on access from one or
      more source tensors (for example fused/unfused GGUF conventions).
    - Computed values are not cached or materialized up-front.
    """

    def __init__(
        self,
        base: MutableMapping,
        mapping: dict[str, str],
        computed: dict[str, Callable[[], object]],
        *,
        computed_shapes: dict[str, tuple[int, ...]] | None = None,
        computed_sources: dict[str, str] | None = None,
    ):
        self._base = base
        self._map = dict(mapping)
        self._computed = dict(computed)
        self._computed_shapes = {
            str(key): tuple(int(v) for v in shape)
            for key, shape in (computed_shapes or {}).items()
        }
        self._computed_sources = dict(computed_sources or {})
        self._keys = tuple(list(self._map.keys()) + [k for k in self._computed.keys() if k not in self._map])
        _inherit_source_metadata(self, base)
        header_shapes = getattr(base, "header_shapes", None)
        if isinstance(header_shapes, dict):
            mapped_shapes: Dict[str, tuple[int, ...]] = {}
            for canonical_key, source_key in self._map.items():
                shape = header_shapes.get(source_key)
                if shape is not None:
                    mapped_shapes[canonical_key] = shape
            for canonical_key, shape in self._computed_shapes.items():
                mapped_shapes[canonical_key] = shape
            self.header_shapes = mapped_shapes

    def __getitem__(self, k: str):
        source_key = self._map.get(k)
        if source_key is not None:
            return self._base[source_key]
        compute = self._computed.get(k)
        if compute is None:
            raise KeyError(k)
        return compute()

    def __setitem__(self, k: str, v):
        source_key = self._map.get(k)
        if source_key is None:
            self._computed.pop(k, None)
            self._computed_shapes.pop(k, None)
            self._computed_sources.pop(k, None)
            self._map[k] = k
            if k not in self._keys:
                self._keys = tuple(list(self._keys) + [k])
            self._base[k] = v
            return
        self._base[source_key] = v

    def __delitem__(self, k: str):
        if k in self._computed:
            del self._computed[k]
            self._computed_shapes.pop(k, None)
            self._computed_sources.pop(k, None)
            self._keys = tuple(key for key in self._keys if key != k)
            return
        source_key = self._map.pop(k, None)
        if source_key is None:
            raise KeyError(k)
        self._keys = tuple(key for key in self._keys if key != k)
        if source_key in self._base:
            del self._base[source_key]

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, k: object) -> bool:
        return k in self._map or k in self._computed

    def keys(self):
        return list(self._keys)

    def items(self):
        for key in self._keys:
            yield key, self[key]

    def shape_of(self, key: str):
        if key in self._computed_shapes:
            return self._computed_shapes[key]
        shape_getter = getattr(self._base, "shape_of", None)
        if not callable(shape_getter):
            return None
        source_key = self._map.get(key)
        if source_key is None:
            return None
        try:
            return shape_getter(source_key)
        except Exception:
            return None


class CastOnGetView(MutableMapping):
    """Mapping view that casts tensor values on CPU to a target dtype on access.

    Useful to avoid fragile CPU bf16/fp16 ops during preprocessing. Only casts
    floating tensors matching `from_dtypes` and `device_type`.
    """

    def __init__(self, base: MutableMapping, *, device_type: str = "cpu", from_dtypes=None, to_dtype=None):
        import torch as _torch

        self._base = base
        self._device_type = device_type
        self._from = tuple(from_dtypes) if from_dtypes is not None else (_torch.bfloat16, _torch.float16)
        self._to = to_dtype or _torch.float32

    def __getitem__(self, k: str):
        import torch as _torch

        v = self._base[k]
        if isinstance(v, _torch.Tensor):
            try:
                if v.device.type == self._device_type and v.dtype in self._from:
                    return v.to(self._to)
            except Exception:
                return v
        return v

    def __setitem__(self, k: str, v):
        self._base[k] = v

    def __delitem__(self, k: str):
        del self._base[k]

    def __iter__(self):
        return iter(self._base.keys())

    def __len__(self) -> int:
        try:
            return len(self._base.keys())
        except Exception:
            return sum(1 for _ in self.__iter__())

    def shape_of(self, key: str):
        shape_getter = getattr(self._base, "shape_of", None)
        if not callable(shape_getter):
            return None
        try:
            return shape_getter(key)
        except Exception:
            return None


class LazySafetensorsDict(MutableMapping):
    """Lazy, mutable mapping backed by a .safetensors file.

    - Keys come from the file; values are loaded on demand with safe_open.get_tensor.
    - Supports overlay writes and deletions without touching the underlying file.
    - Device: tensors are produced on the configured `device` (cpu/cuda/...).

    Windows crash prevention: Once any tensor is accessed, the entire file is
    materialized into memory to avoid reopening the file repeatedly (which causes
    torch_cpu.dll crashes on Windows). On non-Windows platforms, keep a single
    SafeTensors handle open and load tensors on demand to stay truly lazy.
    """

    def __init__(self, filepath: str, device: str = "cpu"):
        self.filepath = filepath
        self.device = device or "cpu"
        self.source_format = "safetensors"
        self.source_path = str(filepath)
        self.primary_dtype_hint: str | None = None
        self.header_shapes: Dict[str, tuple[int, ...]] = {}
        self._platform_windows = sys.platform.startswith("win")
        self._overlay = {}  # in-memory writes/overrides
        self._deleted = set()  # keys logically removed
        self._keys_cache = None  # cached set of underlying keys
        self._shape_cache = None  # cached tensor shapes from safetensors header
        self._materialized = None  # holds all tensors after first access
        self._materialized_triggered = False
        self._handle = None  # persistent SafeTensors handle (non-Windows only)
        self._handle_lock = threading.Lock()

    def _base_keys(self):
        if self._keys_cache is None:
            if self._platform_windows:
                with safe_open(self.filepath, framework="pt", device=self.device) as f:
                    self._keys_cache = set(f.keys())
            else:
                with self._handle_lock:
                    if self._handle is None:
                        self._handle = safe_open(self.filepath, framework="pt", device=self.device)
                    self._keys_cache = set(self._handle.keys())
        return self._keys_cache

    def _base_shapes(self):
        if self._shape_cache is None:
            self._shape_cache = {}
            try:
                from apps.backend.runtime.checkpoint.safetensors_header import (
                    detect_safetensors_primary_dtype_from_header,
                    read_safetensors_header,
                )

                header = read_safetensors_header(Path(self.filepath))
                self.primary_dtype_hint = detect_safetensors_primary_dtype_from_header(header)
                for raw_key, metadata in header.items():
                    if raw_key == "__metadata__":
                        continue
                    if not isinstance(metadata, dict):
                        continue
                    shape = metadata.get("shape")
                    if not isinstance(shape, (list, tuple)):
                        continue
                    try:
                        self._shape_cache[str(raw_key)] = tuple(int(v) for v in shape)
                    except Exception:
                        continue
            except Exception:
                self.primary_dtype_hint = None
                self._shape_cache = {}
            self.header_shapes = dict(self._shape_cache)
        return self._shape_cache

    def _ensure_materialized(self):
        """Load all tensors from file once to avoid reopening repeatedly."""

        if not self._platform_windows:
            return
        if self._materialized is None and not self._materialized_triggered:
            self._materialized_triggered = True
            self._materialized = {}
            try:
                with safe_open(self.filepath, framework="pt", device=self.device) as f:
                    self._keys_cache = set(f.keys())
                    for key in f.keys():
                        self._materialized[key] = f.get_tensor(key)
            except Exception:
                # If materialization fails, clear and fall back to per-key loading
                self._materialized = None
                self._materialized_triggered = False

    # Mapping protocol
    def __getitem__(self, key):
        if key in self._overlay:
            return self._overlay[key]
        if key in self._deleted:
            raise KeyError(key)

        if self._platform_windows:
            # Materialize all tensors on first access to avoid repeated file opens.
            self._ensure_materialized()

            if self._materialized is not None:
                if key in self._materialized:
                    return self._materialized[key]
                raise KeyError(key)

            # Fallback for edge cases (should rarely happen)
            if key not in self._base_keys():
                raise KeyError(key)
            with safe_open(self.filepath, framework="pt", device=self.device) as f:
                return f.get_tensor(key)

        if key not in self._base_keys():
            raise KeyError(key)
        with self._handle_lock:
            if self._handle is None:
                self._handle = safe_open(self.filepath, framework="pt", device=self.device)
            return self._handle.get_tensor(key)

    def __setitem__(self, key, value):
        self._overlay[key] = value
        if self._keys_cache is None and key not in self._deleted:
            # do not expand base key set; overlay keys are separate
            pass
        if key in self._deleted:
            self._deleted.remove(key)

    def __delitem__(self, key):
        if key in self._overlay:
            del self._overlay[key]
        else:
            # mark as deleted logically
            self._deleted.add(key)

    def __iter__(self):
        base = (k for k in self._base_keys() if k not in self._deleted)
        # overlay can shadow base
        for k in base:
            if k not in self._overlay:
                yield k
        for k in self._overlay.keys():
            yield k

    def __len__(self):
        base_keys = self._base_keys()
        overlay_keys = set(self._overlay.keys())
        return len(base_keys - self._deleted - overlay_keys) + len(self._overlay)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._overlay:
            return True
        if key in self._deleted:
            return False
        try:
            return key in self._base_keys()
        except Exception:
            return False

    def shape_of(self, key: str):
        if not isinstance(key, str):
            return None
        if key in self._deleted:
            return None
        if key in self._overlay:
            value = self._overlay[key]
            shape = getattr(value, "shape", None)
            if shape is None:
                return None
            try:
                return tuple(int(v) for v in shape)
            except Exception:
                return None
        return self._base_shapes().get(key)

    # Convenience helpers
    def keys(self):
        return list(iter(self))

    def items(self):
        # Use materialization to avoid per-item file opens (Windows only)
        if self._platform_windows:
            self._ensure_materialized()
        for k in self:
            yield k, self[k]

    def materialize(
        self,
        *,
        prefix: str = "",
        new_prefix: str = "",
        return_mapping: bool = False,
    ):
        """Eagerly load tensors matching `prefix`, optionally re-prefixing keys."""

        def _translate(key: str) -> str:
            suffix = key[len(prefix) :] if prefix and key.startswith(prefix) else key
            if new_prefix:
                return f"{new_prefix}{suffix}"
            if prefix:
                return suffix
            return key

        result: Dict[str, object] = {}
        mapping: Dict[str, str] = {}
        if self._platform_windows:
            with safe_open(self.filepath, framework="pt", device=self.device) as handle:
                for key in handle.keys():
                    if prefix and not key.startswith(prefix):
                        continue
                    if key in self._deleted:
                        continue
                    if key in self._overlay:
                        continue
                    presented = _translate(key)
                    result[presented] = handle.get_tensor(key)
                    mapping[presented] = key
        else:
            with self._handle_lock:
                if self._handle is None:
                    self._handle = safe_open(self.filepath, framework="pt", device=self.device)
                handle = self._handle
                for key in handle.keys():
                    if prefix and not key.startswith(prefix):
                        continue
                    if key in self._deleted:
                        continue
                    if key in self._overlay:
                        continue
                    presented = _translate(key)
                    result[presented] = handle.get_tensor(key)
                    mapping[presented] = key

        for key, value in self._overlay.items():
            if prefix and not key.startswith(prefix):
                continue
            if key in self._deleted:
                continue
            presented = _translate(key)
            result[presented] = value
            mapping[presented] = key

        if return_mapping:
            return result, mapping
        return result

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass


__all__ = [
    "CastOnGetView",
    "ComputedKeyspaceView",
    "FilterPrefixView",
    "KeyPrefixView",
    "LazySafetensorsDict",
    "KeyspaceLookupView",
]
