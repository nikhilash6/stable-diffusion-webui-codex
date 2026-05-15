"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Process-wide LoRA selection state for runtime workflows.
Stores the currently selected LoRAs (path + text-encoder weight + optional UNet weight) so API endpoints and workflow builders can
apply them during generation without depending on legacy/compat selection surfaces.

Symbols (top-level; keep in sync; no ghosts):
- `LoraSelection` (dataclass): Selected LoRA record (path/text-encoder weight/optional UNet weight/online flag).
- `set_selections` (function): Replaces the global selection list (tolerates dict-like inputs from API plumbing).
- `get_selections` (function): Returns a copy of the current selection list.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Iterable, List


@dataclass(frozen=True)
class LoraSelection:
    path: str
    weight: float = 1.0
    unet_weight: float | None = None
    online: bool = False


_LOCK = threading.Lock()
_SELECTIONS: List[LoraSelection] = []


def set_selections(selections: Iterable[Any]) -> None:
    out: List[LoraSelection] = []
    for s in list(selections or []):
        if isinstance(s, LoraSelection):
            out.append(s)
            continue
        if isinstance(s, dict):
            path = str(s.get("path") or "")
            if not path:
                continue
            raw_unet_weight = s.get("unet_weight", None)
            out.append(
                LoraSelection(
                    path=path,
                    weight=float(s.get("weight", 1.0)),
                    unet_weight=(
                        None if raw_unet_weight in (None, "") else float(raw_unet_weight)
                    ),
                    online=bool(s.get("online", False)),
                )
            )
            continue
        path = str(getattr(s, "path", "") or "")
        if not path:
            continue
        raw_unet_weight = getattr(s, "unet_weight", None)
        out.append(
            LoraSelection(
                path=path,
                weight=float(getattr(s, "weight", 1.0)),
                unet_weight=(
                    None if raw_unet_weight in (None, "") else float(raw_unet_weight)
                ),
                online=bool(getattr(s, "online", False)),
            )
        )

    with _LOCK:
        global _SELECTIONS
        _SELECTIONS = out


def get_selections() -> List[LoraSelection]:
    with _LOCK:
        return list(_SELECTIONS)


__all__ = ["LoraSelection", "get_selections", "set_selections"]
