"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA file discovery policy used by backend inventories and registries.
Defines the default roots and per-family overrides (paths.json keys, including Flux.2, LTX2, and Anima) and yields LoRA weight files in
stable order.

Symbols (top-level; keep in sync; no ghosts):
- `LORA_EXTS` (constant): Recognized LoRA weight file extensions.
- `list_lora_roots` (function): Resolves LoRA search roots from `apps/paths.json` per-family keys (`*_loras`).
- `iter_lora_files` (function): Yields LoRA file paths under the resolved roots (recursive, stable order).
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

from apps.backend.infra.config.paths import get_paths_for

from .base import dedupe_keep_order, iter_files

LORA_EXTS: tuple[str, ...] = (".safetensors",)


def list_lora_roots(models_root: str | None = None) -> list[str]:
    roots: list[str] = []

    # Per-family roots from apps/paths.json.
    for key in ("sd15_loras", "sdxl_loras", "flux1_loras", "flux2_loras", "ltx2_loras", "anima_loras", "wan22_loras", "zimage_loras"):
        for p in get_paths_for(key):
            if os.path.isdir(p) or (os.path.isfile(p) and p.lower().endswith(LORA_EXTS)):
                roots.append(p)

    return dedupe_keep_order(roots)


def iter_lora_files(models_root: str | None = None, *, roots: Sequence[str] | None = None) -> Iterable[str]:
    use_roots = list(roots) if roots is not None else list_lora_roots(models_root=models_root)
    out: list[str] = []
    for root in use_roots:
        if os.path.isfile(root) and root.lower().endswith(LORA_EXTS):
            out.append(root)
        elif os.path.isdir(root):
            out.extend(list(iter_files([root], exts=LORA_EXTS)))
    return dedupe_keep_order(out)


__all__ = ["LORA_EXTS", "iter_lora_files", "list_lora_roots"]
