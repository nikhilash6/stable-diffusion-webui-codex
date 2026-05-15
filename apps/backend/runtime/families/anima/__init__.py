"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima (Cosmos Predict2) runtime package.
Contains the Codex-native DiT runtime and strict loader seams used by the Anima engine.

Symbols (top-level; keep in sync; no ghosts):
- `AnimaConfig` (dataclass): Anima runtime config (Cosmos Predict2 + LLMAdapter).
- `AnimaDiT` (class): Core Anima diffusion model (DiT + adapter) used by the sampler adapter.
- `load_anima_dit_from_state_dict` (function): Strict loader helper for raw `net.*` or already-canonical Anima transformer state dicts.
"""

from __future__ import annotations

from .config import AnimaConfig
from .model import AnimaDiT
from .loader import load_anima_dit_from_state_dict

__all__ = [
    "AnimaConfig",
    "AnimaDiT",
    "load_anima_dit_from_state_dict",
]
