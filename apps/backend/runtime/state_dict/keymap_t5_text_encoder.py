"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical key-style detection + keyspace resolver for T5 text-encoder state_dict keys.
Provides strict, fail-loud mapping from HF-style T5 keys (`encoder.*`, `shared.weight`, `embed_tokens*`)
into Codex IntegratedT5 layout (`transformer.*`) so loader paths do not perform ad-hoc generic key preprocessing.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_t5_text_encoder_keyspace` (function): Resolves a T5 encoder state_dict into canonical IntegratedT5 keyspace.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TypeVar

from apps.backend.runtime.state_dict.key_mapping import (
    KeySentinel,
    KeyStyle,
    KeyStyleDetector,
    KeyStyleSpec,
    ResolvedKeyspace,
    SentinelKind,
    resolve_state_dict_keyspace,
)

_T = TypeVar("_T")


_DETECTOR = KeyStyleDetector(
    name="t5_text_encoder_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "transformer."),
            ),
            min_sentinel_hits=1,
        ),
        KeyStyleSpec(
            style=KeyStyle.HF,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "encoder."),
                KeySentinel(SentinelKind.EXACT, "shared.weight"),
                KeySentinel(SentinelKind.PREFIX, "embed_tokens"),
            ),
            min_sentinel_hits=1,
        ),
    ),
)


def resolve_t5_text_encoder_keyspace(
    state_dict: MutableMapping[str, _T],
) -> ResolvedKeyspace[_T]:
    """Resolve T5 text-encoder keys into canonical IntegratedT5 keys.

    - CODEX style (`transformer.*`) is a no-op.
    - HF style (`encoder.*`, `shared.weight`, `embed_tokens*`) is mapped to `transformer.*`.
    - Unknown/ambiguous styles fail loud via key-style detection.
    """

    def _map_hf(key: str) -> str:
        if key.startswith("encoder.") or key == "shared.weight" or key.startswith("embed_tokens"):
            return f"transformer.{key}"
        return key

    resolved = resolve_state_dict_keyspace(
        state_dict,
        detector=_DETECTOR,
        mappers={
            KeyStyle.CODEX: lambda k: k,
            KeyStyle.HF: _map_hf,
        },
    )
    resolved.metadata.setdefault("resolver", "t5_text_encoder")
    return resolved


__all__ = ["resolve_t5_text_encoder_keyspace"]
