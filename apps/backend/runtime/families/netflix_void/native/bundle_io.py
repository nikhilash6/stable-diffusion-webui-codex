"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Base-bundle component path/config helpers for the native Netflix VOID runtime.
Resolves required component directories and their local config/weights files from the family-owned base bundle without
calling generic diffusers loaders or adding component-local guessing outside the explicit family runtime seam.

Symbols (top-level; keep in sync; no ghosts):
- `NETFLIX_VOID_COMPONENT_WEIGHT_FILENAMES` (constant): Preferred local weight filenames searched inside base-bundle components.
- `resolve_netflix_void_component_dir` (function): Resolve one required component directory from the typed base bundle contract.
- `read_netflix_void_component_config` (function): Read the canonical local config for one component directory.
- `resolve_netflix_void_component_weights_path` (function): Resolve exactly one local weights file for a required component directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from apps.backend.runtime.checkpoint.io import read_arbitrary_config

from ..model import NetflixVoidBaseBundle

NETFLIX_VOID_COMPONENT_WEIGHT_FILENAMES: tuple[str, ...] = (
    "model.safetensors",
    "model.safetensor",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "pytorch_model.bin",
    "model.bin",
)


def resolve_netflix_void_component_dir(
    base_bundle: NetflixVoidBaseBundle,
    *,
    component_name: str,
) -> Path:
    attribute_name = f"{str(component_name).strip()}_dir"
    raw_dir = getattr(base_bundle, attribute_name, None)
    resolved = Path(str(raw_dir or "")).expanduser()
    if not str(raw_dir or "").strip() or not resolved.is_dir():
        raise RuntimeError(
            "Netflix VOID base bundle is missing required component directory "
            f"{component_name!r}: {resolved!s}."
        )
    return resolved


def read_netflix_void_component_config(
    base_bundle: NetflixVoidBaseBundle,
    *,
    component_name: str,
) -> Mapping[str, Any]:
    component_dir = resolve_netflix_void_component_dir(base_bundle, component_name=component_name)
    config = read_arbitrary_config(str(component_dir))
    if not isinstance(config, Mapping):
        raise RuntimeError(
            f"Netflix VOID component config for {component_name!r} must be a mapping, got {type(config).__name__}."
        )
    return config


def resolve_netflix_void_component_weights_path(
    base_bundle: NetflixVoidBaseBundle,
    *,
    component_name: str,
) -> Path:
    component_dir = resolve_netflix_void_component_dir(base_bundle, component_name=component_name)
    for filename in NETFLIX_VOID_COMPONENT_WEIGHT_FILENAMES:
        candidate = component_dir / filename
        if candidate.is_file():
            return candidate

    wildcard_candidates = sorted(
        path
        for pattern in ("*.safetensors", "*.safetensor", "*.bin")
        for path in component_dir.glob(pattern)
        if path.is_file()
    )
    if len(wildcard_candidates) == 1:
        return wildcard_candidates[0]
    if not wildcard_candidates:
        raise RuntimeError(
            "Netflix VOID component directory does not contain a supported weights file: "
            f"component={component_name!r} dir={component_dir!s}."
        )
    raise RuntimeError(
        "Netflix VOID component directory has ambiguous weights candidates; keep one owner path only: "
        f"component={component_name!r} dir={component_dir!s} candidates={[str(path.name) for path in wildcard_candidates]!r}."
    )


__all__ = [
    "NETFLIX_VOID_COMPONENT_WEIGHT_FILENAMES",
    "read_netflix_void_component_config",
    "resolve_netflix_void_component_dir",
    "resolve_netflix_void_component_weights_path",
]
