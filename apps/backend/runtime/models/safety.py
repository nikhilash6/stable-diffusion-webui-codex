"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Secure checkpoint validation and native-safe `torch.load` wrapper for pickle-backed checkpoints.
Uses native `torch.load(..., weights_only=True)` when the active torch build supports it, adds `mmap=True` only for zip/new-serialization
checkpoints when supported, and keeps restricted pre-validation plus restricted pickle fallback for older torch builds that lack native
weights-only loading. Used by the backend when `safe_load=True`.

Symbols (top-level; keep in sync; no ghosts):
- `UnsafeCheckpointError` (class): Raised when checkpoint validation fails.
- `RestrictedUnpickler` (class): Restricted unpickler that blocks globals outside an allowlist.
- `_validate_zip_checkpoint` (function): Validates zip-format checkpoints and their `data.pkl` payload.
- `_validate_legacy_pickle` (function): Validates legacy multi-pickle checkpoint format.
- `validate_checkpoint` (function): Validates a checkpoint path and raises `UnsafeCheckpointError` on failure.
- `_is_zip_checkpoint` (function): Detects whether a checkpoint uses torch's zip/new-serialization container.
- `_torch_supports_weights_only` (function): Detects whether this torch build supports `torch.load(..., weights_only=...)`.
- `_torch_supports_mmap` (function): Detects whether this torch build supports `torch.load(..., mmap=...)`.
- `_native_safe_torch_load` (function): Loads a checkpoint through torch's native weights-only path when available.
- `_restricted_pickle_module` (function): Builds a pickle module wrapper using the restricted unpickler.
- `safe_torch_load` (function): Safer `torch.load` wrapper with validation and conservative defaults.
- `extra_globals` (contextmanager): Temporarily allows additional globals during restricted unpickling.
"""

from __future__ import annotations

import collections
import io
import pickle
import re
import types
import zipfile
from contextlib import contextmanager
from typing import Any, Callable

import torch

if not hasattr(torch, "UntypedStorage"):  # pragma: no cover - unsupported torch build
    raise RuntimeError("Unsupported torch build: torch.UntypedStorage unavailable; checkpoint safety cannot run.")

from apps.backend.runtime import errors as runtime_errors

_ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("torch._utils", "_rebuild_device_tensor_from_numpy"): torch._utils._rebuild_device_tensor_from_numpy,
    ("torch", "UntypedStorage"): torch.UntypedStorage,
    ("torch", "FloatStorage"): getattr(torch, "FloatStorage", torch.UntypedStorage),
    ("torch", "HalfStorage"): getattr(torch, "HalfStorage", torch.UntypedStorage),
    ("torch", "DoubleStorage"): getattr(torch, "DoubleStorage", torch.UntypedStorage),
    ("torch", "LongStorage"): getattr(torch, "LongStorage", torch.UntypedStorage),
    ("torch", "IntStorage"): getattr(torch, "IntStorage", torch.UntypedStorage),
    ("torch", "ByteStorage"): getattr(torch, "ByteStorage", torch.UntypedStorage),
    ("torch", "BFloat16Storage"): getattr(torch, "BFloat16Storage", torch.UntypedStorage),
    ("torch", "float32"): getattr(torch, "float32"),
    ("torch", "float16"): getattr(torch, "float16"),
    ("torch", "bfloat16"): getattr(torch, "bfloat16"),
    ("torch.nn.modules.container", "ParameterDict"): torch.nn.modules.container.ParameterDict,
    ("numpy", "dtype"): __import__("numpy").dtype,  # type: ignore[attr-defined]
    ("numpy", "ndarray"): __import__("numpy").ndarray,  # type: ignore[attr-defined]
    ("numpy.core.multiarray", "scalar"): __import__("numpy").core.multiarray.scalar,  # type: ignore[attr-defined]
    ("numpy.core.multiarray", "_reconstruct"): __import__("numpy").core.multiarray._reconstruct,  # type: ignore[attr-defined]
}

_ALLOWED_ZIP_ENTRIES = re.compile(r"^([^/]+)/((data/\d+)|version|byteorder|\.data/serialization_id|data\.pkl)$")
_DATA_PKL_RE = re.compile(r"^([^/]+)/data\.pkl$")


class UnsafeCheckpointError(RuntimeError):
    """Raised when checkpoint validation fails."""


class RestrictedUnpickler(pickle.Unpickler):
    """Restricted unpickler used to validate torch checkpoints."""

    def __init__(self, file_obj: io.BufferedIOBase, extra_handler: Callable[[str, str], Any] | None = None):
        super().__init__(file_obj)
        self._extra_handler = extra_handler

    def find_class(self, module: str, name: str):  # type: ignore[override]
        if self._extra_handler is not None:
            candidate = self._extra_handler(module, name)
            if candidate is not None:
                return candidate
        key = (module, name)
        if key in _ALLOWED_GLOBALS:
            return _ALLOWED_GLOBALS[key]
        raise UnsafeCheckpointError(f"Disallowed global '{module}/{name}' during pickle load")


def _validate_zip_checkpoint(path: str, *, extra_handler: Callable[[str, str], Any] | None = None) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for entry in names:
            if not _ALLOWED_ZIP_ENTRIES.match(entry):
                raise UnsafeCheckpointError(f"disallowed file inside checkpoint: {entry}")
        data_files = [e for e in names if _DATA_PKL_RE.match(e)]
        if len(data_files) != 1:
            raise UnsafeCheckpointError("checkpoint archive must contain exactly one data.pkl")
        with archive.open(data_files[0]) as data_fp:
            RestrictedUnpickler(data_fp, extra_handler=extra_handler).load()


def _validate_legacy_pickle(path: str, *, extra_handler: Callable[[str, str], Any] | None = None) -> None:
    with open(path, "rb") as handle:
        unpickler = RestrictedUnpickler(handle, extra_handler=extra_handler)
        # Legacy format writes five pickles sequentially.
        for _ in range(5):
            try:
                unpickler.load()
            except EOFError:
                break


def validate_checkpoint(path: str, *, extra_handler: Callable[[str, str], Any] | None = None) -> None:
    """Raise :class:`UnsafeCheckpointError` if *path* fails validation."""

    try:
        with open(path, "rb") as fp:
            signature = fp.read(4)
    except FileNotFoundError as exc:  # pragma: no cover - caller handles
        raise UnsafeCheckpointError(f"checkpoint not found: {path}") from exc

    if signature == b"PK\x03\x04":  # zipfile
        _validate_zip_checkpoint(path, extra_handler=extra_handler)
        return
    # Legacy pickled checkpoints (no zip header)
    _validate_legacy_pickle(path, extra_handler=extra_handler)


def _is_zip_checkpoint(path: str) -> bool:
    with open(path, "rb") as fp:
        return fp.read(4) == b"PK\x03\x04"


def _torch_supports_weights_only() -> bool:
    return "weights_only" in torch.load.__code__.co_varnames  # type: ignore[attr-defined]


def _torch_supports_mmap() -> bool:
    return "mmap" in torch.load.__code__.co_varnames  # type: ignore[attr-defined]


def _native_safe_torch_load(path: str, *, map_location: torch.device | str | None = None) -> Any:
    load_kwargs: dict[str, Any] = {
        "map_location": map_location or "cpu",
        "weights_only": True,
    }
    if _torch_supports_mmap() and _is_zip_checkpoint(path):
        load_kwargs["mmap"] = True
    return torch.load(path, **load_kwargs)


def _restricted_pickle_module(extra_handler: Callable[[str, str], Any] | None = None):
    namespace = types.SimpleNamespace()

    def _load(file_obj: io.BufferedIOBase, *args: Any, **kwargs: Any) -> Any:
        return RestrictedUnpickler(file_obj, extra_handler=extra_handler).load()

    class _Restricted(RestrictedUnpickler):
        def __init__(self, file_obj: io.BufferedIOBase):
            super().__init__(file_obj, extra_handler=extra_handler)

    namespace.load = _load
    namespace.Unpickler = _Restricted
    return namespace


def safe_torch_load(
    path: str,
    *,
    map_location: torch.device | str | None = None,
    extra_handler: Callable[[str, str], Any] | None = None,
    strict: bool = True,
) -> Any:
    """Safely load a torch checkpoint.

    When native weights-only loading is available, it is the authoritative safe
    path because it understands real torch archive metadata and persistent
    storage references. Native mmap is only enabled for zip/new-serialization
    checkpoints. ``strict`` only applies to the legacy restricted-pickle
    fallback path used on older torch builds without ``weights_only`` support.
    """

    map_location = map_location or "cpu"
    if _torch_supports_weights_only():
        return _native_safe_torch_load(path, map_location=map_location)

    if strict:
        try:
            validate_checkpoint(path, extra_handler=extra_handler)
        except UnsafeCheckpointError:
            runtime_errors.report_error(f"Unsafe checkpoint rejected: {path}", exc_info=False, context="checkpoint_safety")
            raise

    # Fallback to restricted pickle module if weights_only is unsupported.
    pickle_module = _restricted_pickle_module(extra_handler)
    return torch.load(path, map_location=map_location, pickle_module=pickle_module)


@contextmanager
def extra_globals(handler: Callable[[str, str], Any]):
    """Temporarily allow additional globals during checkpoint validation."""

    yield handler


__all__ = [
    "RestrictedUnpickler",
    "UnsafeCheckpointError",
    "extra_globals",
    "safe_torch_load",
    "validate_checkpoint",
]
