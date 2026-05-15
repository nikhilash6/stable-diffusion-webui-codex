"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SafeTensors merge tool for collapsing a source layout into one `.safetensors` file.
Accepts the same source formats as the GGUF converter source helper (single file, sharded index, or directory) and reports typed merge progress.

Symbols (top-level; keep in sync; no ghosts):
- `SafetensorsMergeConfig` (dataclass): Merge input/output configuration (`source_path`, `output_path`, `overwrite`).
- `SafetensorsMergeProgress` (dataclass): Progress payload emitted while loading/merging tensors.
- `validate_safetensors_merge_config` (function): Validate merge paths and resolve the concrete safetensors source layout.
- `merge_safetensors_source` (function): Merge a safetensors source into one output `.safetensors` file.
- `__all__` (constant): Explicit export list for merge tool public API.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Mapping

from apps.backend.runtime.checkpoint.safetensors_header import read_safetensors_header
from apps.backend.runtime.tools.gguf_converter_safetensors_source import ResolvedSafetensorsSource, resolve_safetensors_source

_COPY_CHUNK_SIZE = 8 * 1024 * 1024
_MAX_SAFE_TENSORS_HEADER_BYTES = 64 * 1024 * 1024
_DTYPE_SIZES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "C64": 8,
}


@dataclass(slots=True)
class SafetensorsMergeConfig:
    """Configuration for a safetensors merge job."""

    source_path: str
    output_path: str
    overwrite: bool = False


@dataclass(slots=True)
class SafetensorsMergeProgress:
    """Progress tracking payload for safetensors merge."""

    current_step: int = 0
    total_steps: int = 0
    current_tensor: str = ""
    status: str = "pending"
    error: str | None = None

    @property
    def progress_percent(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return (float(self.current_step) / float(self.total_steps)) * 100.0


@dataclass(frozen=True, slots=True)
class _ValidatedTensorEntry:
    source_path: Path
    source_data_start: int
    dtype: str
    shape: tuple[int, ...]
    source_offsets: tuple[int, int]

    @property
    def byte_length(self) -> int:
        return self.source_offsets[1] - self.source_offsets[0]


@dataclass(frozen=True, slots=True)
class _ParsedSafetensorsFile:
    metadata: dict[str, str]
    tensors: dict[str, _ValidatedTensorEntry]


ProgressCallback = Callable[[SafetensorsMergeProgress], None]


def _emit_progress(progress_callback: ProgressCallback | None, progress: SafetensorsMergeProgress) -> None:
    if progress_callback is None:
        return
    progress_callback(progress)


def _extract_header_metadata(path: Path, raw_metadata: object) -> dict[str, str]:
    if raw_metadata is None:
        return {}
    if not isinstance(raw_metadata, Mapping):
        raise TypeError(f"Expected safetensors metadata to be dict[str, str], got: {type(raw_metadata).__name__}")

    metadata: dict[str, str] = {}
    for key, value in raw_metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError(
                "Safetensors metadata must contain string keys/values; "
                f"got key={type(key).__name__}, value={type(value).__name__}"
            )
        metadata[key] = value
    return metadata


def _validate_header_shape(path: Path, tensor_name: str, raw_shape: object) -> tuple[int, ...]:
    if not isinstance(raw_shape, (list, tuple)):
        raise TypeError(f"Invalid safetensors shape for tensor {tensor_name!r} in {path}: expected list[int]")

    dims: list[int] = []
    for dim in raw_shape:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError(
                f"Invalid safetensors shape for tensor {tensor_name!r} in {path}: "
                f"expected integer dimensions, got {type(dim).__name__}"
            )
        if dim < 0:
            raise ValueError(
                f"Invalid safetensors shape for tensor {tensor_name!r} in {path}: negative dimension {dim}"
            )
        dims.append(dim)
    return tuple(dims)


def _validate_header_offsets(path: Path, tensor_name: str, raw_offsets: object) -> tuple[int, int]:
    if not isinstance(raw_offsets, (list, tuple)) or len(raw_offsets) != 2:
        raise TypeError(
            f"Invalid safetensors offsets for tensor {tensor_name!r} in {path}: expected [start, end]"
        )

    start_raw, end_raw = raw_offsets
    if not isinstance(start_raw, int) or isinstance(start_raw, bool):
        raise TypeError(
            f"Invalid safetensors offsets for tensor {tensor_name!r} in {path}: start must be int"
        )
    if not isinstance(end_raw, int) or isinstance(end_raw, bool):
        raise TypeError(
            f"Invalid safetensors offsets for tensor {tensor_name!r} in {path}: end must be int"
        )
    if start_raw < 0:
        raise ValueError(
            f"Invalid safetensors offsets for tensor {tensor_name!r} in {path}: start must be >= 0"
        )
    if end_raw < start_raw:
        raise ValueError(
            f"Invalid safetensors offsets for tensor {tensor_name!r} in {path}: end must be >= start"
        )
    return int(start_raw), int(end_raw)


def _validate_tensor_entry(
    path: Path,
    *,
    data_start: int,
    payload_size: int,
    tensor_name: str,
    raw_entry: object,
) -> _ValidatedTensorEntry:
    if not isinstance(raw_entry, Mapping):
        raise TypeError(f"Invalid safetensors header entry for tensor {tensor_name!r} in {path}: expected object")

    dtype_raw = raw_entry.get("dtype")
    if not isinstance(dtype_raw, str):
        raise TypeError(f"Invalid safetensors dtype for tensor {tensor_name!r} in {path}: expected string")
    if dtype_raw not in _DTYPE_SIZES:
        raise ValueError(f"Unsupported safetensors dtype for tensor {tensor_name!r} in {path}: {dtype_raw!r}")

    shape = _validate_header_shape(path, tensor_name, raw_entry.get("shape"))
    source_offsets = _validate_header_offsets(path, tensor_name, raw_entry.get("data_offsets"))

    expected_size = math.prod(shape) * _DTYPE_SIZES[dtype_raw]
    actual_size = source_offsets[1] - source_offsets[0]
    if actual_size != expected_size:
        raise ValueError(
            f"Invalid safetensors payload size for tensor {tensor_name!r} in {path}: "
            f"dtype={dtype_raw} shape={list(shape)} expects {expected_size} bytes, got {actual_size}"
        )
    if source_offsets[1] > payload_size:
        raise ValueError(
            f"Tensor {tensor_name!r} in {path} exceeds payload bounds: "
            f"end={source_offsets[1]} payload={payload_size}"
        )

    return _ValidatedTensorEntry(
        source_path=path,
        source_data_start=data_start,
        dtype=dtype_raw,
        shape=shape,
        source_offsets=source_offsets,
    )


def _read_safetensors_data_start(path: Path) -> int:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
    if len(raw_len) != 8:
        raise EOFError("Unexpected EOF reading safetensors header length.")
    (header_len,) = struct.unpack("<Q", raw_len)
    if header_len <= 0 or header_len > _MAX_SAFE_TENSORS_HEADER_BYTES:
        raise ValueError(f"Invalid safetensors header length: {header_len}")
    return 8 + int(header_len)


def _validate_payload_layout(
    path: Path,
    *,
    payload_size: int,
    tensors: Mapping[str, _ValidatedTensorEntry],
) -> None:
    if not tensors:
        if payload_size != 0:
            raise ValueError(f"Invalid safetensors payload layout for {path}: empty header leaves {payload_size} payload bytes unindexed")
        return

    ordered_entries = sorted(tensors.items(), key=lambda item: item[1].source_offsets)
    previous_end = 0
    for tensor_name, tensor in ordered_entries:
        start, end = tensor.source_offsets
        if start != previous_end:
            raise ValueError(
                f"Invalid safetensors payload layout for {path}: tensor {tensor_name!r} starts at {start}, expected {previous_end}"
            )
        previous_end = end

    if previous_end != payload_size:
        raise ValueError(
            f"Invalid safetensors payload layout for {path}: indexed payload ends at {previous_end}, payload size is {payload_size}"
        )


def _parse_safetensors_file(path: Path) -> _ParsedSafetensorsFile:
    header = read_safetensors_header(path)
    data_start = _read_safetensors_data_start(path)
    file_size = path.stat().st_size
    payload_size = file_size - data_start
    if payload_size < 0:
        raise ValueError(f"Invalid safetensors payload bounds for {path}: header exceeds file size")

    metadata = _extract_header_metadata(path, header.get("__metadata__"))
    tensors: dict[str, _ValidatedTensorEntry] = {}
    for tensor_name, raw_entry in header.items():
        if tensor_name == "__metadata__":
            continue
        tensors[tensor_name] = _validate_tensor_entry(
            path,
            data_start=data_start,
            payload_size=payload_size,
            tensor_name=tensor_name,
            raw_entry=raw_entry,
        )

    _validate_payload_layout(path, payload_size=payload_size, tensors=tensors)
    return _ParsedSafetensorsFile(metadata=metadata, tensors=tensors)


def _build_tensor_copy_plan(
    resolved: ResolvedSafetensorsSource,
) -> tuple[list[str], dict[str, _ValidatedTensorEntry], dict[str, str]]:
    parsed_files = {file_path: _parse_safetensors_file(file_path) for file_path in resolved.data_files}

    if resolved.single_file_path is not None:
        parsed = parsed_files[resolved.single_file_path]
        tensor_names = sorted(parsed.tensors)
        tensor_map = {name: parsed.tensors[name] for name in tensor_names}
        return tensor_names, tensor_map, dict(parsed.metadata)

    tensor_names = sorted(resolved.tensor_to_shard)
    tensor_map: dict[str, _ValidatedTensorEntry] = {}
    for tensor_name in tensor_names:
        shard_path = resolved.tensor_to_shard[tensor_name]
        parsed = parsed_files[shard_path]
        tensor_entry = parsed.tensors.get(tensor_name)
        if tensor_entry is None:
            raise KeyError(f"Tensor referenced by safetensors index missing from shard header: {tensor_name} ({shard_path})")
        tensor_map[tensor_name] = tensor_entry
    return tensor_names, tensor_map, {}


def _encode_safetensors_header(header: Mapping[str, object]) -> bytes:
    raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    padding = (-len(raw)) % 8
    header_bytes = raw + (b" " * padding)
    if len(header_bytes) > _MAX_SAFE_TENSORS_HEADER_BYTES:
        raise ValueError(
            f"Merged safetensors header is too large: {len(header_bytes)} bytes exceeds {_MAX_SAFE_TENSORS_HEADER_BYTES}"
        )
    return header_bytes


def _write_exact_copy(source: BinaryIO, destination: BinaryIO, byte_count: int) -> None:
    if byte_count <= 0:
        return

    buffer = bytearray(min(_COPY_CHUNK_SIZE, byte_count))
    view = memoryview(buffer)
    remaining = byte_count
    while remaining > 0:
        chunk_size = min(len(buffer), remaining)
        read_count = source.readinto(view[:chunk_size])
        if read_count is None or read_count <= 0:
            raise EOFError(f"Unexpected EOF while copying {byte_count} bytes from safetensors source payload.")
        destination.write(view[:read_count])
        remaining -= read_count


def _write_merged_output(
    output_path: Path,
    *,
    tensor_names: list[str],
    tensor_map: Mapping[str, _ValidatedTensorEntry],
    metadata: Mapping[str, str],
    progress: SafetensorsMergeProgress,
    progress_callback: ProgressCallback | None,
) -> None:
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = dict(metadata)

    payload_offset = 0
    for tensor_name in tensor_names:
        tensor = tensor_map[tensor_name]
        next_offset = payload_offset + tensor.byte_length
        header[tensor_name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [payload_offset, next_offset],
        }
        payload_offset = next_offset

    header_bytes = _encode_safetensors_header(header)
    temp_fd, temp_name = tempfile.mkstemp(
        dir=str(output_path.parent),
        prefix=f".{output_path.stem}.merge-",
        suffix=output_path.suffix,
    )
    temp_path = Path(temp_name)

    try:
        with os.fdopen(temp_fd, "wb") as destination:
            destination.write(struct.pack("<Q", len(header_bytes)))
            destination.write(header_bytes)

            with contextlib.ExitStack() as stack:
                source_handles = {
                    file_path: stack.enter_context(file_path.open("rb"))
                    for file_path in sorted({tensor.source_path for tensor in tensor_map.values()}, key=lambda p: str(p))
                }

                for index, tensor_name in enumerate(tensor_names, start=1):
                    tensor = tensor_map[tensor_name]
                    source_handle = source_handles[tensor.source_path]
                    source_handle.seek(tensor.source_data_start + tensor.source_offsets[0])
                    _write_exact_copy(source_handle, destination, tensor.byte_length)
                    progress.current_step = index
                    progress.current_tensor = tensor_name
                    _emit_progress(progress_callback, progress)

        os.replace(temp_path, output_path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        raise


def _validate_merge_paths(config: SafetensorsMergeConfig) -> tuple[Path, Path]:
    source_raw = str(config.source_path or "").strip()
    output_raw = str(config.output_path or "").strip()

    if not source_raw:
        raise ValueError("source_path is required")
    if not output_raw:
        raise ValueError("output_path is required")

    source_path = Path(source_raw).expanduser().resolve()
    output_path = Path(output_raw).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"source_path not found: {source_path}")

    if not output_path.name.lower().endswith(".safetensors"):
        raise ValueError("output_path must end with `.safetensors`.")

    parent = output_path.parent
    if not parent.exists():
        raise FileNotFoundError(f"output_path parent directory does not exist: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"output_path parent is not a directory: {parent}")

    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"output_path is a directory: {output_path}")
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"output file already exists: {output_path}")

    return source_path, output_path


def _validate_output_aliases(
    *,
    resolved: ResolvedSafetensorsSource,
    output_path: Path,
) -> None:
    source_files = set(resolved.data_files)
    if output_path in source_files:
        source_kind = "shard" if resolved.is_sharded else "source file"
        raise ValueError(f"output_path must not overwrite a safetensors {source_kind}: {output_path}")


def validate_safetensors_merge_config(
    config: SafetensorsMergeConfig,
) -> tuple[Path, Path, ResolvedSafetensorsSource]:
    """Validate merge paths and resolve the concrete safetensors source layout."""

    source_path, output_path = _validate_merge_paths(config)
    try:
        resolved = resolve_safetensors_source(str(source_path))
    except ValueError as exc:
        if str(exc).startswith("Expected a .safetensors file/dir/index.json, got:"):
            raise ValueError(
                "Expected a .safetensors file, '*.safetensors.index.json', or directory, "
                f"got: {source_path}"
            ) from exc
        raise
    _validate_output_aliases(resolved=resolved, output_path=output_path)
    return source_path, output_path, resolved


def merge_safetensors_source(
    config: SafetensorsMergeConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Merge a safetensors source into a single `.safetensors` output file."""

    progress = SafetensorsMergeProgress(status="pending")
    try:
        _source_path, output_path, resolved = validate_safetensors_merge_config(config)

        progress.status = "loading_weights"
        _emit_progress(progress_callback, progress)

        tensor_names, tensor_map, metadata = _build_tensor_copy_plan(resolved)

        progress.total_steps = len(tensor_names)
        progress.current_step = 0
        progress.current_tensor = ""
        progress.status = "merging_safetensors"
        _emit_progress(progress_callback, progress)

        _write_merged_output(
            output_path,
            tensor_names=tensor_names,
            tensor_map=tensor_map,
            metadata=metadata,
            progress=progress,
            progress_callback=progress_callback,
        )

        progress.current_step = progress.total_steps
        progress.status = "complete"
        progress.error = None
        _emit_progress(progress_callback, progress)
    except Exception as exc:
        progress.status = "error"
        progress.error = str(exc)
        _emit_progress(progress_callback, progress)
        raise


__all__ = [
    "SafetensorsMergeConfig",
    "SafetensorsMergeProgress",
    "validate_safetensors_merge_config",
    "merge_safetensors_source",
]
