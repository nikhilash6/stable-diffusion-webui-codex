"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared primitives for lightweight backend infra registries.
Defines the base `AssetEntry` record used by the active registry modules.

Symbols (top-level; keep in sync; no ghosts):
- `AssetEntry` (dataclass): Generic asset record (name/path/kind/tags/meta) used by active registry entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class AssetEntry:
    name: str
    path: str
    kind: str
    tags: List[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


__all__ = ["AssetEntry"]
