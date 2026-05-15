"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Strict loader utilities for the Anima core transformer (`MiniTrainDiT` + `LLMAdapter`).
Resolves raw `net.*` checkpoints through the explicit Anima transformer keyspace owner, then configures and strict-loads the canonical runtime
lookup view; any keyspace/load mismatch is fatal and reported with actionable samples.

Symbols (top-level; keep in sync; no ghosts):
- `load_anima_dit_from_state_dict` (function): Instantiate + strict-load `AnimaDiT` from a transformer state dict.
"""

from __future__ import annotations

from collections.abc import Mapping
import torch

from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime.state_dict.keymap_anima_transformer import resolve_anima_transformer_keyspace

from .config import AnimaConfig, infer_anima_config_from_state_dict
from .model import AnimaDiT


def load_anima_dit_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> AnimaDiT:
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"state_dict must be a mapping; got {type(state_dict).__name__}")

    try:
        resolved = resolve_anima_transformer_keyspace(state_dict)
    except Exception as exc:  # noqa: BLE001 - surfaced as a load-time error with context
        raise RuntimeError(f"Anima transformer keyspace resolution failed: {exc}") from exc
    canonical_state_dict = resolved.view

    try:
        config: AnimaConfig = infer_anima_config_from_state_dict(canonical_state_dict)
    except Exception as exc:  # noqa: BLE001 - surfaced as a load-time error with context
        raise RuntimeError(f"Anima config inference failed: {exc}") from exc
    model = AnimaDiT(config=config, device=device, dtype=dtype).eval()

    missing, unexpected = safe_load_state_dict(model, canonical_state_dict, log_name="anima.transformer")
    if missing or unexpected:
        sample_missing = ", ".join(missing[:10])
        sample_unexpected = ", ".join(unexpected[:10])
        raise RuntimeError(
            "Anima core transformer strict load failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample=[{sample_missing}] unexpected_sample=[{sample_unexpected}]"
        )

    return model
