"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z-Image L2P engine package exports.
Exports the exact `zimage_l2p` engine facade without aliases.

Symbols (top-level; keep in sync; no ghosts):
- `ZImageL2PEngine` (class): Public engine facade for L2P txt2img.
"""

from __future__ import annotations

from .zimage_l2p import ZImageL2PEngine

__all__ = ["ZImageL2PEngine"]
