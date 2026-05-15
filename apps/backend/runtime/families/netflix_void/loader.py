"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Netflix VOID bundle-planning helpers for loader/runtime handoff.
Resolves exactly one valid local CogVideoX-Fun base bundle from `paths.json["netflix_void_base"]`, resolves the
selected Pass 1 overlay checkpoint plus its literal Pass 2 sibling through the metadata-only registry seam, and emits
one typed bundle envelope for the native engine/runtime scaffold.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_netflix_void_base_dirs` (function): Return all valid local base-bundle directories discovered under `netflix_void_base`.
- `resolve_netflix_void_base_bundle` (function): Resolve exactly one valid local base-bundle contract.
- `resolve_netflix_void_checkpoint_record` (function): Resolve and validate the selected Pass 1 overlay checkpoint record.
- `prepare_netflix_void_bundle_inputs` (function): Build the typed Netflix VOID bundle-planning envelope from local inventory.
- `build_netflix_void_bundle_metadata` (function): Serialize stable bundle metadata for runtime/engine assembly.
"""

from __future__ import annotations

import os
from pathlib import Path

from apps.backend.infra.config.paths import get_paths_for
from apps.backend.runtime.model_registry.netflix_void_execution import (
    NETFLIX_VOID_KIND_PASS1,
    NETFLIX_VOID_KIND_PASS2,
    NETFLIX_VOID_KIND_UNKNOWN,
    netflix_void_checkpoint_kind,
    resolve_netflix_void_pass2_partner,
)
from apps.backend.runtime.models import api as model_api
from apps.backend.runtime.models.types import CheckpointRecord

from .config import NETFLIX_VOID_ENGINE_ID, netflix_void_base_dir_is_valid
from .model import NetflixVoidBaseBundle, NetflixVoidBundleInputs, NetflixVoidOverlayPair


def resolve_netflix_void_base_dirs() -> tuple[str, ...]:
    candidates: list[str] = []
    seen: set[str] = set()

    for raw_root in get_paths_for("netflix_void_base"):
        root = Path(os.path.expanduser(str(raw_root).strip()))
        if not root.exists():
            continue
        if root.is_dir() and netflix_void_base_dir_is_valid(root):
            resolved = str(root.resolve(strict=False))
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)
            continue
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir(), key=lambda item: str(item).lower()):
            if not netflix_void_base_dir_is_valid(child):
                continue
            resolved = str(child.resolve(strict=False))
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)

    return tuple(candidates)


def resolve_netflix_void_base_bundle() -> NetflixVoidBaseBundle:
    candidates = resolve_netflix_void_base_dirs()
    if not candidates:
        raise RuntimeError(
            "Netflix VOID base bundle resolution failed: no valid directory was found under paths.json['netflix_void_base']."
        )
    if len(candidates) != 1:
        raise RuntimeError(
            "Netflix VOID base bundle resolution failed: expected exactly one valid directory under "
            f"paths.json['netflix_void_base'], got {len(candidates)} candidates: {list(candidates)!r}"
        )
    root = Path(candidates[0])
    return NetflixVoidBaseBundle(
        root_dir=str(root),
        model_index_path=str(root / "model_index.json"),
        scheduler_dir=str(root / "scheduler"),
        text_encoder_dir=str(root / "text_encoder"),
        tokenizer_dir=str(root / "tokenizer"),
        transformer_dir=str(root / "transformer"),
        vae_dir=str(root / "vae"),
    )


def resolve_netflix_void_checkpoint_record(model_ref: str) -> CheckpointRecord:
    record = model_api.find_checkpoint(model_ref)
    if record is None:
        raise RuntimeError(f"Netflix VOID checkpoint not found: {model_ref!r}")
    family_hint = str(getattr(record, "family_hint", "") or "").strip().lower()
    if family_hint != NETFLIX_VOID_ENGINE_ID:
        raise RuntimeError(
            "Netflix VOID bundle planning requires a checkpoint discovered with family_hint='netflix_void'; "
            f"got {family_hint or '<empty>'!r} for {getattr(record, 'filename', model_ref)!r}."
        )
    checkpoint_kind = netflix_void_checkpoint_kind(record)
    if checkpoint_kind == NETFLIX_VOID_KIND_UNKNOWN:
        raise RuntimeError(
            "Selected Netflix VOID checkpoint is unsupported by the current literal pairing contract: "
            f"{getattr(record, 'filename', model_ref)!r}."
        )
    if checkpoint_kind == NETFLIX_VOID_KIND_PASS2:
        raise RuntimeError(
            "Selected Netflix VOID checkpoint is a Pass 2 refinement overlay. "
            "Use the literal Pass 1 selector `void_pass1.safetensors` instead."
        )
    if checkpoint_kind != NETFLIX_VOID_KIND_PASS1:
        raise RuntimeError(
            f"Netflix VOID bundle planning expected checkpoint_kind='pass1', got {checkpoint_kind!r}."
        )
    return record


def prepare_netflix_void_bundle_inputs(model_ref: str) -> NetflixVoidBundleInputs:
    pass1_record = resolve_netflix_void_checkpoint_record(model_ref)
    pass2_path = resolve_netflix_void_pass2_partner(pass1_record)
    pass2_record = model_api.find_checkpoint(pass2_path)
    if pass2_record is None:
        raise RuntimeError(
            "Netflix VOID Pass 2 partner resolved from the public selector is missing from model inventory: "
            f"{pass2_path!r}. Refresh `/api/models` after fixing the checkpoint roots."
        )
    pass2_family_hint = str(getattr(pass2_record, "family_hint", "") or "").strip().lower()
    if pass2_family_hint != NETFLIX_VOID_ENGINE_ID:
        raise RuntimeError(
            "Netflix VOID Pass 2 partner resolved to a checkpoint outside the netflix_void family hint: "
            f"{pass2_family_hint or '<empty>'!r} for {getattr(pass2_record, 'filename', pass2_path)!r}."
        )
    if netflix_void_checkpoint_kind(pass2_record) != NETFLIX_VOID_KIND_PASS2:
        raise RuntimeError(
            "Netflix VOID Pass 2 partner resolution requires literal sibling "
            f"`void_pass2.safetensors`, got {getattr(pass2_record, 'filename', pass2_path)!r}."
        )

    return NetflixVoidBundleInputs(
        model_ref=str(model_ref),
        base_bundle=resolve_netflix_void_base_bundle(),
        overlays=NetflixVoidOverlayPair(pass1_record=pass1_record, pass2_record=pass2_record),
        checkpoint_kind=NETFLIX_VOID_KIND_PASS1,
    )


def build_netflix_void_bundle_metadata(inputs: NetflixVoidBundleInputs) -> dict[str, object]:
    return {
        "engine_key": NETFLIX_VOID_ENGINE_ID,
        "base_bundle_root": inputs.base_bundle.root_dir,
        "pass1_overlay": str(getattr(inputs.overlays.pass1_record, "filename", "") or ""),
        "pass2_overlay": str(getattr(inputs.overlays.pass2_record, "filename", "") or ""),
        "checkpoint_kind": inputs.checkpoint_kind,
    }
