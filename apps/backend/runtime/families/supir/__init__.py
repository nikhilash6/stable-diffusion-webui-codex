"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR model-family runtime package.
Provides SUPIR-specific config parsing, weight resolution, and guardrails for the native SDXL img2img/inpaint SUPIR mode.

Symbols (top-level; keep in sync; no ghosts):
- `__all__` (constant): Explicit export list for SUPIR runtime helpers (kept intentionally small).
"""

from __future__ import annotations

__all__ = [
    "config",
    "loader",
    "runtime",
    "sdxl_guard",
    "weights",
]
