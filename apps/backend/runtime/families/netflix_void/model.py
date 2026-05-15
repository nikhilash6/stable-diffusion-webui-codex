"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed loader/runtime handoff contracts for the Netflix VOID family scaffold.
Defines immutable dataclasses for the resolved base CogVideoX-Fun bundle, the literal Pass 1/Pass 2 overlay pair, and
one typed bundle-input envelope consumed by the engine/runtime assembly seam.

Symbols (top-level; keep in sync; no ghosts):
- `NetflixVoidBaseBundle` (dataclass): Resolved local base-bundle directory contract.
- `NetflixVoidOverlayPair` (dataclass): Resolved literal Pass 1 + Pass 2 overlay checkpoint pair.
- `NetflixVoidBundleInputs` (dataclass): Typed bundle-planning envelope consumed by engine/runtime assembly.
"""

from __future__ import annotations

from dataclasses import dataclass

from apps.backend.runtime.models.types import CheckpointRecord


@dataclass(frozen=True, slots=True)
class NetflixVoidBaseBundle:
    root_dir: str
    model_index_path: str
    scheduler_dir: str
    text_encoder_dir: str
    tokenizer_dir: str
    transformer_dir: str
    vae_dir: str


@dataclass(frozen=True, slots=True)
class NetflixVoidOverlayPair:
    pass1_record: CheckpointRecord
    pass2_record: CheckpointRecord


@dataclass(frozen=True, slots=True)
class NetflixVoidBundleInputs:
    model_ref: str
    base_bundle: NetflixVoidBaseBundle
    overlays: NetflixVoidOverlayPair
    checkpoint_kind: str
