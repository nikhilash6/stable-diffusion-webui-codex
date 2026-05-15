"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Video-VAE bundle validation helpers for the native LTX2 seam.
Rechecks the parser-owned LTX2 video VAE component contract at loader/runtime handoff so future assembly never starts
from a partially-corrupted bundle.

Symbols (top-level; keep in sync; no ghosts):
- `validate_ltx2_video_vae_contract` (function): Validate the required LTX2 video-VAE sentinel keys.
"""

from __future__ import annotations

from typing import Any, Mapping


def validate_ltx2_video_vae_contract(state_dict: Mapping[str, Any]) -> None:
    required = ("per_channel_statistics.mean-of-means", "per_channel_statistics.std-of-means")
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise RuntimeError(f"LTX2 video VAE bundle is missing required keys: {missing!r}")
