"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LTX2 engine package surface.
Exports the `Ltx2Engine` facade and keeps the package layout aligned with the existing family-specific spec/factory
pattern used by FLUX.2 and Anima.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2Engine` (class): Backend LTX2 video engine facade.
"""

from __future__ import annotations

from .ltx2 import Ltx2Engine

__all__ = ["Ltx2Engine"]

