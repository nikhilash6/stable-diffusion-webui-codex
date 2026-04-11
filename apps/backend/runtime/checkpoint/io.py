"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Checkpoint IO helpers for runtime codepaths.
Loads safetensors/GGUF/pickle checkpoints and reads lightweight model configs from directories.

Symbols (top-level; keep in sync; no ghosts):
- `read_arbitrary_config` (function): Reads a best-effort config from a directory (supports JSON/YAML-like inputs where present).
- `load_torch_file` (function): Loads a torch checkpoint with safe-load options and explicit device targeting (prefers safe loaders, falls back to pickle loader when allowed).
- `read_safetensors_metadata` (function): Reads the raw SafeTensors string metadata map from a `.safetensors` file header.
- `read_gguf_metadata` (function): Reads GGUF key/value metadata from a `.gguf` file header (scoped here to keep quantization imports out of engines).
- `_load_gguf_state_dict` (function): Loads a GGUF state dict from a `.gguf` file path (used by runtime helpers without importing heavy ops).
- `load_gguf_state_dict` (function): Public GGUF state-dict loader with explicit dequant policy (`dequantize=False` default when omitted) and target-device tensor exposure.
- `_load_pickled_checkpoint` (function): Loads a pickled checkpoint using the restricted/guarded unpickler (`checkpoint_pickle`).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import json
import logging
import os
from collections.abc import Mapping
from typing import Any

import torch
from safetensors.torch import safe_open

from apps.backend.runtime.misc import checkpoint_pickle

from ..state_dict.views import LazySafetensorsDict

_log = get_backend_logger("backend.runtime.utils")
_CONFIG_FILENAMES = ("config.json", "scheduler_config.json")


def read_arbitrary_config(directory):
    for filename in _CONFIG_FILENAMES:
        config_path = os.path.join(directory, filename)
        if not os.path.exists(config_path):
            continue
        with open(config_path, "rt", encoding="utf-8") as file:
            config_data = json.load(file)
        return config_data

    expected = ", ".join(_CONFIG_FILENAMES)
    raise FileNotFoundError(f"No supported config file found in the directory: {directory} (expected one of: {expected})")


def load_torch_file(ckpt, safe_load=True, device=None):
    """Load a checkpoint (safetensors/gguf/pickle) honoring an explicit device.

    - When ``device`` is None, use the current core execution device from the
      memory manager to avoid accidental CPU pinning.
    - For safetensors, the returned mapping lazily loads tensors using
      ``safe_open(..., device=<device>)`` so values are produced directly on the
      requested device when possible.
    """

    from apps.backend.runtime.memory import memory_management as _mm  # local import avoids cycles

    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        from apps.backend.runtime.memory.config import DeviceRole

        device = _mm.manager.get_device(DeviceRole.CORE)

    checkpoint_path = str(ckpt)
    suffix = os.path.splitext(checkpoint_path)[1].lower()

    if suffix in {".safetensor", ".safetensors"}:
        return LazySafetensorsDict(checkpoint_path, device=str(device))
    if suffix == ".gguf":
        return _load_gguf_state_dict(checkpoint_path, device=device)

    pl_sd = _load_pickled_checkpoint(checkpoint_path, device, safe_load)

    if "global_step" in pl_sd:
        _log.info("Global Step: %s", pl_sd["global_step"])

    if "state_dict" in pl_sd:
        return pl_sd["state_dict"]
    return pl_sd


def load_gguf_state_dict(
    path: str,
    *,
    dequantize: bool | None = None,
    computation_dtype: torch.dtype = torch.float16,
    device: torch.device | str | None = None,
):
    """Load a GGUF state dict, with optional explicit dequantization policy.

    - When `dequantize` is None, this loader defaults to forward dequantization
      (`dequantize=False`) for runtime GGUF paths.
    - Callers with an explicit policy (e.g. "VAE GGUFs always dequantize") should pass
      `dequantize=True` to make the intent unambiguous and avoid drift.
    """

    from apps.backend.quantization.gguf_loader import load_gguf_state_dict as _load

    if dequantize is None:
        dequantize = False
    return _load(path, dequantize=bool(dequantize), computation_dtype=computation_dtype, device=device)


def read_safetensors_metadata(path: str) -> dict[str, str]:
    """Read SafeTensors header metadata as a strict string map."""

    checkpoint_path = str(path)
    suffix = os.path.splitext(checkpoint_path)[1].lower()
    if suffix not in {".safetensor", ".safetensors"}:
        raise RuntimeError(
            "SafeTensors metadata can only be read from `.safetensor` / `.safetensors` files; "
            f"got path={checkpoint_path!r}."
        )

    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        raw_metadata = handle.metadata() or {}

    if not isinstance(raw_metadata, dict):
        raise RuntimeError(
            "SafeTensors metadata must be a dict[str, str]. "
            f"Got {type(raw_metadata).__name__} from {checkpoint_path!r}."
        )

    metadata: dict[str, str] = {}
    for raw_key, raw_value in raw_metadata.items():
        if not isinstance(raw_key, str):
            raise RuntimeError(
                "SafeTensors metadata keys must be strings; "
                f"got {type(raw_key).__name__} in {checkpoint_path!r}."
            )
        if not isinstance(raw_value, str):
            raise RuntimeError(
                "SafeTensors metadata values must be strings; "
                f"got key={raw_key!r} type={type(raw_value).__name__} in {checkpoint_path!r}."
            )
        metadata[raw_key] = raw_value
    return metadata


def read_gguf_metadata(path: str) -> Mapping[str, Any]:
    """Read GGUF metadata (key/value table) from the file header."""

    from apps.backend.quantization.gguf_loader import get_gguf_metadata as _get

    metadata = _get(path)
    if not isinstance(metadata, Mapping):
        raise RuntimeError(
            "GGUF metadata reader must return a mapping. "
            f"Got {type(metadata).__name__} from {path!r}."
        )
    for raw_key in metadata.keys():
        if not isinstance(raw_key, str):
            raise RuntimeError(
                "GGUF metadata keys must be strings. "
                f"Got {type(raw_key).__name__} in {path!r}."
            )
    return metadata


def _load_gguf_state_dict(path: str, *, device: torch.device | str | None = None):
    # Back-compat internal alias; prefer calling `load_gguf_state_dict(...)` directly.
    return load_gguf_state_dict(path, device=device)


def _load_pickled_checkpoint(path, device, safe_load):
    if safe_load:
        from apps.backend.runtime.models import safety as model_safety

        try:
            return model_safety.safe_torch_load(path, map_location=device)
        except model_safety.UnsafeCheckpointError:
            raise
    return torch.load(path, map_location=device, pickle_module=checkpoint_pickle)


__all__ = [
    "_load_gguf_state_dict",
    "load_gguf_state_dict",
    "read_safetensors_metadata",
    "read_gguf_metadata",
    "load_torch_file",
    "read_arbitrary_config",
]
