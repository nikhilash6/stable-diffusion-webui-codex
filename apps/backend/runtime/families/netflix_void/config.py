"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Immutable Netflix VOID family constants and base-bundle validation helpers.
Defines the canonical engine id, literal Pass 1/Pass 2 overlay filenames, required base-bundle entries, and upstream-
aligned default geometry/config values used by loader/runtime scaffolding and route/runtime request defaults.

Symbols (top-level; keep in sync; no ghosts):
- `NETFLIX_VOID_ENGINE_ID` (constant): Canonical internal engine/family id.
- `NETFLIX_VOID_PUBLIC_LABEL` (constant): Human-readable family label.
- `NETFLIX_VOID_PASS1_FILENAME` (constant): Literal Pass 1 overlay filename.
- `NETFLIX_VOID_PASS2_FILENAME` (constant): Literal Pass 2 overlay filename.
- `NETFLIX_VOID_BASE_REQUIRED_ENTRIES` (constant): Required base-bundle files/directories.
- `NETFLIX_VOID_DEFAULT_HEIGHT` (constant): Upstream-aligned default output height.
- `NETFLIX_VOID_DEFAULT_WIDTH` (constant): Upstream-aligned default output width.
- `NETFLIX_VOID_DEFAULT_FPS` (constant): Upstream-aligned default export fps.
- `NETFLIX_VOID_DEFAULT_TEMPORAL_WINDOW` (constant): Upstream-aligned default temporal window size.
- `NETFLIX_VOID_DEFAULT_MAX_VIDEO_LENGTH` (constant): Upstream-aligned default max video length.
- `NETFLIX_VOID_DEFAULT_PASS1_STEPS` (constant): Upstream-aligned Pass 1 default denoise steps.
- `NETFLIX_VOID_DEFAULT_PASS2_STEPS` (constant): Upstream-aligned Pass 2 default denoise steps.
- `NETFLIX_VOID_DEFAULT_PASS1_CFG` (constant): Upstream-aligned Pass 1 CFG default.
- `NETFLIX_VOID_DEFAULT_PASS2_CFG` (constant): Upstream-aligned Pass 2 CFG default.
- `netflix_void_base_dir_is_valid` (function): Return whether a candidate base-bundle directory exposes the required entries.
"""

from __future__ import annotations

from pathlib import Path

NETFLIX_VOID_ENGINE_ID = "netflix_void"
NETFLIX_VOID_PUBLIC_LABEL = "Netflix VOID"

NETFLIX_VOID_PASS1_FILENAME = "void_pass1.safetensors"
NETFLIX_VOID_PASS2_FILENAME = "void_pass2.safetensors"

NETFLIX_VOID_BASE_REQUIRED_ENTRIES: tuple[str, ...] = (
    "model_index.json",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
)

NETFLIX_VOID_DEFAULT_HEIGHT = 384
NETFLIX_VOID_DEFAULT_WIDTH = 672
NETFLIX_VOID_DEFAULT_FPS = 12
NETFLIX_VOID_DEFAULT_TEMPORAL_WINDOW = 85
NETFLIX_VOID_DEFAULT_MAX_VIDEO_LENGTH = 197
NETFLIX_VOID_DEFAULT_PASS1_STEPS = 30
NETFLIX_VOID_DEFAULT_PASS2_STEPS = 50
NETFLIX_VOID_DEFAULT_PASS1_CFG = 1.0
NETFLIX_VOID_DEFAULT_PASS2_CFG = 6.0


def netflix_void_base_dir_is_valid(path: Path) -> bool:
    return path.is_dir() and all((path / entry).exists() for entry in NETFLIX_VOID_BASE_REQUIRED_ENTRIES)


__all__ = [
    "NETFLIX_VOID_ENGINE_ID",
    "NETFLIX_VOID_PUBLIC_LABEL",
    "NETFLIX_VOID_PASS1_FILENAME",
    "NETFLIX_VOID_PASS2_FILENAME",
    "NETFLIX_VOID_BASE_REQUIRED_ENTRIES",
    "NETFLIX_VOID_DEFAULT_HEIGHT",
    "NETFLIX_VOID_DEFAULT_WIDTH",
    "NETFLIX_VOID_DEFAULT_FPS",
    "NETFLIX_VOID_DEFAULT_TEMPORAL_WINDOW",
    "NETFLIX_VOID_DEFAULT_MAX_VIDEO_LENGTH",
    "NETFLIX_VOID_DEFAULT_PASS1_STEPS",
    "NETFLIX_VOID_DEFAULT_PASS2_STEPS",
    "NETFLIX_VOID_DEFAULT_PASS1_CFG",
    "NETFLIX_VOID_DEFAULT_PASS2_CFG",
    "netflix_void_base_dir_is_valid",
]
