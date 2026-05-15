"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Text encoder weight-file discovery policy used by backend inventories.
Defines text encoder locations from per-family `apps/paths.json` keys (`*_tenc`, including Flux.2, LTX2, and Anima) and yields weight
file paths in stable order while excluding non-text-encoder GGUF sidecars such as `mmproj`.

Symbols (top-level; keep in sync; no ghosts):
- `TEXT_ENCODER_EXTS` (constant): Recognized text encoder weight file extensions.
- `_files_in_dir` (function): Lists weight files in a directory (non-recursive, stable order).
- `list_text_encoder_roots` (function): Resolves text encoder search roots from `apps/paths.json` per-family keys (`*_tenc`).
- `iter_text_encoder_files` (function): Yields weight-file paths under the resolved roots (non-recursive, stable order).
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

from apps.backend.infra.config.paths import get_paths_for

from .base import dedupe_keep_order

TEXT_ENCODER_EXTS: tuple[str, ...] = (".safetensors", ".pt", ".bin", ".gguf")


def _files_in_dir(dir_path: str, *, exts: Sequence[str]) -> list[str]:
    if not dir_path or not os.path.isdir(dir_path):
        return []
    out: list[str] = []
    try:
        for name in sorted(os.listdir(dir_path), key=lambda s: s.lower()):
            full = os.path.join(dir_path, name)
            lower_name = name.lower()
            if os.path.isfile(full) and lower_name.endswith(tuple(exts)) and "mmproj" not in lower_name:
                out.append(full)
    except Exception:
        return []
    return out


def list_text_encoder_roots(models_root: str | None = None) -> list[str]:
    roots: list[str] = []

    # Per-family roots from apps/paths.json.
    for key in ("sd15_tenc", "sdxl_tenc", "flux1_tenc", "flux2_tenc", "ltx2_tenc", "anima_tenc", "wan22_tenc", "zimage_tenc"):
        for p in get_paths_for(key):
            if os.path.isdir(p):
                roots.append(p)
            elif os.path.isfile(p) and p.lower().endswith(TEXT_ENCODER_EXTS) and "mmproj" not in os.path.basename(p).lower():
                roots.append(p)

    return dedupe_keep_order(roots)


def iter_text_encoder_files(models_root: str | None = None, *, roots: Sequence[str] | None = None) -> Iterable[str]:
    use_roots = list(roots) if roots is not None else list_text_encoder_roots(models_root=models_root)
    out: list[str] = []
    for root in use_roots:
        if os.path.isfile(root) and root.lower().endswith(TEXT_ENCODER_EXTS) and "mmproj" not in os.path.basename(root).lower():
            out.append(root)
        elif os.path.isdir(root):
            out.extend(_files_in_dir(root, exts=TEXT_ENCODER_EXTS))
    return dedupe_keep_order(out)


__all__ = ["TEXT_ENCODER_EXTS", "iter_text_encoder_files", "list_text_encoder_roots"]
