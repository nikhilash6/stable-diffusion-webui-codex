"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SafeTensors header readers for lightweight runtime tooling.
Provides header-only helpers to read the SafeTensors JSON header and derive small metadata hints (e.g. primary dtype, tensor shapes)
without importing torch or loading tensor payloads.

Symbols (top-level; keep in sync; no ghosts):
- `read_safetensors_header` (function): Reads and parses the SafeTensors JSON header (no tensor payload reads).
- `extract_safetensors_tensor_shapes_from_header` (function): Extracts a tensor-name -> shape map from an already-read SafeTensors header.
- `read_safetensors_tensor_shapes` (function): Reads a SafeTensors header and returns a tensor-name -> shape map.
- `detect_safetensors_primary_dtype_from_header` (function): Best-effort primary dtype hint from an already-read SafeTensors header mapping.
- `detect_safetensors_primary_dtype` (function): Best-effort dtype hint for `.safetensors` (header-only parse; whole-file).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Mapping


def read_safetensors_header(path: Path) -> dict[str, object]:
    """Read and parse the SafeTensors JSON header (no tensor payload reads)."""

    suffix = path.suffix.lower()
    if suffix not in {".safetensor", ".safetensors"}:
        raise ValueError(f"Not a safetensors file: {path}")

    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise EOFError("Unexpected EOF reading safetensors header length.")
        (header_len,) = struct.unpack("<Q", raw_len)
        # Defensive cap: corrupted files can claim absurd header sizes.
        if header_len <= 0 or header_len > 64 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header length: {header_len}")
        raw = handle.read(int(header_len))
        if len(raw) != int(header_len):
            raise EOFError("Unexpected EOF reading safetensors header payload.")

    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid safetensors header (expected a JSON object).")
    return data  # type: ignore[return-value]


def detect_safetensors_primary_dtype_from_header(header: Mapping[str, object]) -> str | None:
    """Best-effort primary dtype hint from an already-read SafeTensors header."""

    float_types = {"F16", "BF16", "F32", "F64", "F8_E4M3FN", "F8_E5M2"}
    totals: Dict[str, int] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, Mapping):
            continue
        dtype = meta.get("dtype")
        if not isinstance(dtype, str) or dtype not in float_types:
            continue
        offsets = meta.get("data_offsets")
        if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
            continue
        try:
            start = int(offsets[0])
            end = int(offsets[1])
        except Exception:
            continue
        if end < start:
            continue
        totals[dtype] = totals.get(dtype, 0) + (end - start)

    if not totals:
        return None

    best = max(totals.items(), key=lambda kv: kv[1])[0]
    mapping = {
        "F16": "fp16",
        "BF16": "bf16",
        "F32": "fp32",
        "F64": "fp64",
        "F8_E4M3FN": "fp8_e4m3fn",
        "F8_E5M2": "fp8_e5m2",
    }
    return mapping.get(best)


def extract_safetensors_tensor_shapes_from_header(header: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
    """Extract tensor-name -> shape tuples from an already-read SafeTensors header."""

    shapes: dict[str, tuple[int, ...]] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, Mapping):
            continue
        raw_shape = meta.get("shape")
        if not isinstance(raw_shape, (list, tuple)):
            continue
        try:
            shape = tuple(int(dim) for dim in raw_shape)
        except Exception:
            continue
        shapes[str(name)] = shape
    return shapes


def read_safetensors_tensor_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Read a SafeTensors header and return a tensor-name -> shape map."""

    header = read_safetensors_header(path)
    if not isinstance(header, Mapping):
        raise ValueError("Invalid safetensors header (expected a JSON object).")
    return extract_safetensors_tensor_shapes_from_header(header)


def detect_safetensors_primary_dtype(path: Path) -> str | None:
    """Best-effort dtype hint for `.safetensors` (header-only parse).

    Computes the dominant float dtype by summing tensor payload sizes per dtype
    using header offsets. Returns a normalized dtype label suitable for UI/debug
    (e.g. `fp16`, `bf16`, `fp32`).
    """

    suffix = path.suffix.lower()
    if suffix not in {".safetensor", ".safetensors"}:
        return None

    try:
        data = read_safetensors_header(path)
    except Exception:
        return None

    if not isinstance(data, Mapping):
        return None

    return detect_safetensors_primary_dtype_from_header(data)


__all__ = [
    "detect_safetensors_primary_dtype",
    "detect_safetensors_primary_dtype_from_header",
    "extract_safetensors_tensor_shapes_from_header",
    "read_safetensors_header",
    "read_safetensors_tensor_shapes",
]
