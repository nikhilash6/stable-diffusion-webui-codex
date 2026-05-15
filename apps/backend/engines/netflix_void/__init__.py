"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Netflix VOID engine package surface.
Exports the `NetflixVoidEngine` facade and keeps the package layout aligned with the existing family-specific
spec/factory pattern used by other native engines.

Symbols (top-level; keep in sync; no ghosts):
- `NetflixVoidEngine` (class): Backend Netflix VOID video engine facade.
"""

from __future__ import annotations

from .netflix_void import NetflixVoidEngine

__all__ = ["NetflixVoidEngine"]
