"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SafeTensors source helpers for the GGUF converter (single-file and sharded weights).
Supports opening `.safetensors` files, `*.safetensors.index.json` sharded indexes, and directories containing either layout.

Symbols (top-level; keep in sync; no ghosts):
- `ResolvedSafetensorsSource` (dataclass): Concrete safetensors source layout selected from a file/dir/index path.
- `resolve_config_json_path` (function): Resolve a config path (file/dir/HF layout) to a concrete `config.json` path.
- `resolve_safetensors_source` (function): Resolve a safetensors source path into a concrete single-file or sharded layout.
- `open_safetensors_source` (function): Context manager that opens a safetensors source (single or sharded) for reading.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from safetensors import safe_open


def resolve_config_json_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"config.json not found at: {path}")
    return path


@dataclass(frozen=True, slots=True)
class _ShardedSafetensorsIndex:
    index_path: Path
    tensor_to_shard: dict[str, Path]


@dataclass(frozen=True, slots=True)
class ResolvedSafetensorsSource:
    """Concrete safetensors source layout selected from a file/dir/index path."""

    path: Path
    single_file_path: Path | None
    index_path: Path | None
    tensor_to_shard: Mapping[str, Path]

    @property
    def is_sharded(self) -> bool:
        return self.index_path is not None

    @property
    def data_files(self) -> tuple[Path, ...]:
        if self.single_file_path is not None:
            return (self.single_file_path,)
        return tuple(sorted(set(self.tensor_to_shard.values()), key=lambda shard: str(shard)))


def _load_sharded_safetensors_index(index_path: Path) -> _ShardedSafetensorsIndex:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid safetensors index (missing weight_map): {index_path}")

    base = index_path.parent
    out: dict[str, Path] = {}
    for k, v in weight_map.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"Invalid safetensors index (non-string weight_map entry): {index_path}")
        shard = (base / v).resolve()
        if shard.suffix.lower() != ".safetensors":
            raise ValueError(f"Unsupported shard type in {index_path}: {v!r} (expected .safetensors)")
        if not shard.is_file():
            raise FileNotFoundError(f"Shard referenced by {index_path} is missing: {shard}")
        out[k] = shard

    return _ShardedSafetensorsIndex(index_path=index_path.resolve(), tensor_to_shard=out)


def _pick_safetensors_index_path(weights_dir: Path) -> Path | None:
    preferred = weights_dir / "model.safetensors.index.json"
    if preferred.is_file():
        return preferred

    candidates = sorted(weights_dir.glob("*.safetensors.index.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    # Common HF naming conventions; prefer the most specific.
    for fname in (
        "diffusion_pytorch_model.safetensors.index.json",
        "pytorch_model.safetensors.index.json",
    ):
        p = weights_dir / fname
        if p.is_file():
            return p

    names = ", ".join(p.name for p in candidates[:6])
    more = "" if len(candidates) <= 6 else f" (+{len(candidates) - 6} more)"
    raise ValueError(
        f"Multiple safetensors index files found under {weights_dir}: {names}{more}. "
        "Pass the desired '*.safetensors.index.json' path explicitly."
    )


def resolve_safetensors_source(path: str) -> ResolvedSafetensorsSource:
    """Resolve a user-provided safetensors path into a concrete single-file or sharded source layout."""

    p = Path(path).expanduser()
    if p.is_dir():
        index_path = _pick_safetensors_index_path(p)
        if index_path is not None:
            index = _load_sharded_safetensors_index(index_path)
            return ResolvedSafetensorsSource(
                path=index.index_path,
                single_file_path=None,
                index_path=index.index_path,
                tensor_to_shard=dict(index.tensor_to_shard),
            )

        candidates = sorted(p.glob("*.safetensors"))
        if len(candidates) == 1:
            return ResolvedSafetensorsSource(
                path=candidates[0].resolve(),
                single_file_path=candidates[0].resolve(),
                index_path=None,
                tensor_to_shard={},
            )
        if not candidates:
            raise FileNotFoundError(f"No .safetensors files found under: {p}")
        names = ", ".join(c.name for c in candidates[:6])
        more = "" if len(candidates) <= 6 else f" (+{len(candidates) - 6} more)"
        raise ValueError(
            f"Multiple .safetensors files found under {p}: {names}{more}. "
            "Pass a single file path or the '*.safetensors.index.json' path explicitly."
        )

    # Explicit index file path.
    if p.is_file() and p.name.endswith(".safetensors.index.json"):
        index = _load_sharded_safetensors_index(p)
        return ResolvedSafetensorsSource(
            path=index.index_path,
            single_file_path=None,
            index_path=index.index_path,
            tensor_to_shard=dict(index.tensor_to_shard),
        )

    if p.suffix.lower() != ".safetensors":
        raise ValueError(f"Expected a .safetensors file, '*.safetensors.index.json', or directory, got: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"Safetensors file not found: {p}")
    resolved = p.resolve()
    return ResolvedSafetensorsSource(
        path=resolved,
        single_file_path=resolved,
        index_path=None,
        tensor_to_shard={},
    )


class _ShardedSafetensors:
    def __init__(self, index: _ShardedSafetensorsIndex) -> None:
        self._index = index
        self._handles: dict[Path, Any] = {}
        self._stack: contextlib.ExitStack | None = None

    def __enter__(self) -> "_ShardedSafetensors":
        self._stack = contextlib.ExitStack()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._handles.clear()

    def keys(self):
        return self._index.tensor_to_shard.keys()

    def _handle_for(self, shard: Path):
        if shard in self._handles:
            return self._handles[shard]
        if self._stack is None:
            raise RuntimeError("Sharded safetensors handle is not open (missing context manager).")
        handle = self._stack.enter_context(safe_open(str(shard), framework="pt", device="cpu"))
        self._handles[shard] = handle
        return handle

    def get_slice(self, name: str):
        shard = self._index.tensor_to_shard.get(name)
        if shard is None:
            raise KeyError(f"Tensor not found in sharded safetensors index: {name}")
        return self._handle_for(shard).get_slice(name)

    def get_tensor(self, name: str):
        shard = self._index.tensor_to_shard.get(name)
        if shard is None:
            raise KeyError(f"Tensor not found in sharded safetensors index: {name}")
        return self._handle_for(shard).get_tensor(name)


@contextlib.contextmanager
def open_safetensors_source(path: str) -> Iterator[Any]:
    """Open a safetensors source from either:
    - a single `.safetensors` file,
    - a `.safetensors.index.json` file (sharded),
    - or a directory containing either a single `.safetensors` or an index file.
    """

    resolved = resolve_safetensors_source(path)
    if resolved.single_file_path is not None:
        with safe_open(str(resolved.single_file_path), framework="pt", device="cpu") as source:
            yield source
        return

    index = _ShardedSafetensorsIndex(
        index_path=resolved.index_path if resolved.index_path is not None else resolved.path,
        tensor_to_shard=dict(resolved.tensor_to_shard),
    )
    with _ShardedSafetensors(index) as source:
        yield source


__all__ = [
    "ResolvedSafetensorsSource",
    "open_safetensors_source",
    "resolve_config_json_path",
    "resolve_safetensors_source",
]
