"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Static config helpers for the native LTX2 runtime seam.
Defines the parser-owned component names, the required external text-encoder slot, and strict local vendored metadata
path resolution for `Lightricks/LTX-2`, including profile-scoped `latent_upsampler` config enforcement for the
explicit `two_stage` lane without inventing alternate repo layouts.

Symbols (top-level; keep in sync; no ghosts):
- `LTX2_VENDOR_REPO_ID` (constant): Canonical vendored Hugging Face repo id for LTX2 metadata.
- `LTX2_REQUIRED_TEXT_ENCODER_SLOT` (constant): Required external text-encoder alias for the current backend-only slice.
- `LTX2_COMPONENT_NAMES` (constant): Immutable parser-owned component names consumed by loader/runtime.
- `resolve_ltx2_vendor_paths` (function): Resolve fail-loud local vendor metadata paths for the LTX2 repo, with optional `two_stage` config enforcement.
"""

from __future__ import annotations

from pathlib import Path

from apps.backend.runtime.checkpoint.io import read_arbitrary_config

from .model import Ltx2VendorPaths

LTX2_VENDOR_REPO_ID = "Lightricks/LTX-2"
LTX2_REQUIRED_TEXT_ENCODER_SLOT = "gemma3_12b"
LTX2_COMPONENT_NAMES = ("transformer", "connectors", "vae", "audio_vae", "vocoder")
_LTX2_REQUIRED_RUNTIME_CONFIG_DIRS = (
    "text_encoder",
    "scheduler",
    "connectors",
    "transformer",
    "vae",
    "audio_vae",
    "vocoder",
)
_LTX2_TWO_STAGE_RUNTIME_CONFIG_DIRS = ("latent_upsampler",)
_LTX2_REQUIRED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def resolve_ltx2_vendor_paths(
    *,
    backend_root: Path,
    repo_id: str = LTX2_VENDOR_REPO_ID,
    require_latent_upsampler: bool = False,
) -> Ltx2VendorPaths:
    repo_id = str(repo_id or "").strip()
    if not repo_id:
        raise RuntimeError("LTX2 vendored repo id is empty.")
    repo_dir = backend_root / "huggingface" / Path(repo_id)
    model_index_path = repo_dir / "model_index.json"
    tokenizer_dir = repo_dir / "tokenizer"
    connectors_config_path = repo_dir / "connectors" / "config.json"

    if not repo_dir.is_dir():
        raise RuntimeError(f"LTX2 vendored repo metadata directory not found: {repo_dir}")
    if not model_index_path.is_file():
        raise RuntimeError(f"LTX2 model_index.json not found: {model_index_path}")
    if not tokenizer_dir.is_dir():
        raise RuntimeError(f"LTX2 tokenizer directory not found: {tokenizer_dir}")
    for filename in _LTX2_REQUIRED_TOKENIZER_FILES:
        tokenizer_file = tokenizer_dir / filename
        if not tokenizer_file.is_file():
            raise RuntimeError(f"LTX2 tokenizer file not found: {tokenizer_file}")
    if not connectors_config_path.is_file():
        raise RuntimeError(f"LTX2 connectors config not found: {connectors_config_path}")
    required_runtime_dirs = list(_LTX2_REQUIRED_RUNTIME_CONFIG_DIRS)
    if require_latent_upsampler:
        required_runtime_dirs.extend(_LTX2_TWO_STAGE_RUNTIME_CONFIG_DIRS)
    for component_name in required_runtime_dirs:
        component_dir = repo_dir / component_name
        if not component_dir.is_dir():
            raise RuntimeError(f"LTX2 vendored component directory not found: {component_dir}")
        try:
            read_arbitrary_config(str(component_dir))
        except Exception as exc:
            raise RuntimeError(
                f"LTX2 vendored component config load failed for {component_dir}: {exc}"
            ) from exc

    return Ltx2VendorPaths(
        repo_dir=str(repo_dir),
        model_index_path=str(model_index_path),
        tokenizer_dir=str(tokenizer_dir),
        connectors_config_path=str(connectors_config_path),
    )
