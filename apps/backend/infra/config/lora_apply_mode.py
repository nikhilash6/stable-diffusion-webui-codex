"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Global LoRA application mode selection (merge vs online).
Centralizes the meaning and parsing of the LoRA apply-mode used by patchers/runtimes.

Symbols (top-level; keep in sync; no ghosts):
- `ENV_LORA_APPLY_MODE` (constant): Environment variable name controlling LoRA apply mode.
- `LoraApplyMode` (enum): Supported apply modes (`merge`, `online`).
- `DEFAULT_LORA_APPLY_MODE` (constant): Default apply mode (`online`) when unset.
- `parse_lora_apply_mode` (function): Parses a string into `LoraApplyMode` (strict; raises on invalid).
- `read_lora_apply_mode` (function): Reads apply mode from an env mapping (strict; raises on invalid).
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Mapping, Optional

from .bootstrap_env import get_bootstrap_env

ENV_LORA_APPLY_MODE = "CODEX_LORA_APPLY_MODE"


class LoraApplyMode(Enum):
    """How LoRA patches are applied to runtime weights."""

    MERGE = "merge"
    ONLINE = "online"


DEFAULT_LORA_APPLY_MODE = LoraApplyMode.ONLINE


def parse_lora_apply_mode(raw: str) -> LoraApplyMode:
    value = str(raw).strip().lower()
    for mode in LoraApplyMode:
        if mode.value == value:
            return mode
    allowed = ", ".join(m.value for m in LoraApplyMode)
    raise ValueError(f"{ENV_LORA_APPLY_MODE} must be one of: {allowed}; got: {raw!r}")


def read_lora_apply_mode(env: Optional[Mapping[str, str]] = None) -> LoraApplyMode:
    """Return the configured LoRA apply mode.

    Precedence:
    - bootstrap override (resolved CLI) when env is None
    - env[CODEX_LORA_APPLY_MODE] if set (strict)
    - DEFAULT_LORA_APPLY_MODE
    """

    env_map = os.environ if env is None else env
    raw = None if env is not None else get_bootstrap_env(ENV_LORA_APPLY_MODE)
    if raw is None:
        raw = env_map.get(ENV_LORA_APPLY_MODE)
    if raw is None:
        return DEFAULT_LORA_APPLY_MODE
    text = str(raw).strip()
    if not text:
        return DEFAULT_LORA_APPLY_MODE
    return parse_lora_apply_mode(text)


__all__ = [
    "DEFAULT_LORA_APPLY_MODE",
    "ENV_LORA_APPLY_MODE",
    "LoraApplyMode",
    "parse_lora_apply_mode",
    "read_lora_apply_mode",
]
