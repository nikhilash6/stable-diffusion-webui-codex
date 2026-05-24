"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared CFG batch-mode contract resolver for repo-owned sampling loops.
Keeps `CODEX_CFG_BATCH_MODE=fused|split` parsing fail-loud and consistent across generic and family-local samplers.

Symbols (top-level; keep in sync; no ghosts):
- `CFG_BATCH_MODE_ENV` (constant): Canonical environment/bootstrap key for CFG batch mode.
- `CfgBatchMode` (type alias): Accepted CFG batch-mode literal values.
- `resolve_cfg_batch_mode` (function): Returns the requested CFG batch mode or raises on unsupported values.
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

import os
from typing import Literal, cast

from apps.backend.infra.config.bootstrap_env import get_bootstrap_env
from apps.backend.infra.config.env_flags import env_str

CFG_BATCH_MODE_ENV = "CODEX_CFG_BATCH_MODE"
_CFG_BATCH_MODE_ALLOWED = frozenset({"fused", "split"})
CfgBatchMode = Literal["fused", "split"]


def resolve_cfg_batch_mode(*, env_name: str = CFG_BATCH_MODE_ENV) -> CfgBatchMode:
    """Resolve the global CFG batch mode without laundering invalid explicit values."""

    raw_value = get_bootstrap_env(env_name)
    if raw_value is None:
        raw_value = os.getenv(env_name)
    if raw_value is not None and str(raw_value).strip():
        normalized_raw = str(raw_value).strip().lower()
        if normalized_raw not in _CFG_BATCH_MODE_ALLOWED:
            allowed_text = ", ".join(sorted(_CFG_BATCH_MODE_ALLOWED))
            raise RuntimeError(f"Unsupported {env_name}={raw_value!r}; allowed values: {allowed_text}.")

    resolved_value = env_str(env_name, default="fused", allowed=set(_CFG_BATCH_MODE_ALLOWED))
    if resolved_value not in _CFG_BATCH_MODE_ALLOWED:
        allowed_text = ", ".join(sorted(_CFG_BATCH_MODE_ALLOWED))
        raise RuntimeError(f"Resolved unsupported {env_name}={resolved_value!r}; allowed values: {allowed_text}.")
    return cast(CfgBatchMode, resolved_value)


__all__ = ["CFG_BATCH_MODE_ENV", "CfgBatchMode", "resolve_cfg_batch_mode"]
