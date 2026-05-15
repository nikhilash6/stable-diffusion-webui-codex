"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN2.1 VAE key-style detection + strict canonical keyspace resolver.
Validates canonical key ownership fail-loud before model load and rejects any attempt to rewrite incoming wrapper/prefix chains outside explicit keyspace mapping.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_wan21_vae_keyspace` (function): Resolves WAN2.1 VAE keys into canonical keyspace.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import TypeVar

from apps.backend.runtime.state_dict.key_mapping import (
    fail_on_key_name_rewrite,
    KeyMappingError,
    KeySentinel,
    KeyStyle,
    KeyStyleDetector,
    KeyStyleSpec,
    ResolvedKeyspace,
    SentinelKind,
    resolve_state_dict_keyspace,
)

_T = TypeVar("_T")

_WAN21_VAE_PREFIXES = (
    "module.",
    "vae.",
    "first_stage_model.",
)
_WAN21_VAE_REQUIRED = (
    "decoder.head.0.gamma",
    "encoder.conv1.weight",
    "decoder.conv1.weight",
    "conv1.weight",
    "conv2.weight",
)

_WAN21_VAE_DETECTOR = KeyStyleDetector(
    name="wan21_vae_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "encoder."),
                KeySentinel(SentinelKind.PREFIX, "decoder."),
                KeySentinel(SentinelKind.EXACT, "decoder.head.0.gamma"),
                KeySentinel(SentinelKind.EXACT, "conv1.weight"),
                KeySentinel(SentinelKind.EXACT, "conv2.weight"),
            ),
            min_sentinel_hits=2,
        ),
    ),
)


def _validate_required_keys(*, keys: Sequence[str], required: Sequence[str], detector_name: str) -> None:
    keys_set = frozenset(keys)
    missing = [key for key in required if key not in keys_set]
    if missing:
        raise KeyMappingError(
            f"{detector_name}: resolver output is missing required canonical keys. "
            f"missing_sample={missing[:10]}"
        )


def resolve_wan21_vae_keyspace(state_dict: MutableMapping[str, _T]) -> ResolvedKeyspace[_T]:
    def _validate_output(keys: Sequence[str]) -> None:
        _validate_required_keys(
            keys=keys,
            required=_WAN21_VAE_REQUIRED,
            detector_name=_WAN21_VAE_DETECTOR.name,
        )

    resolved = resolve_state_dict_keyspace(
        state_dict,
        detector=_WAN21_VAE_DETECTOR,
        source_key_guard=lambda key: fail_on_key_name_rewrite(key, _WAN21_VAE_PREFIXES),
        mappers={KeyStyle.CODEX: lambda key: key},
        output_validator=_validate_output,
    )
    resolved.metadata.setdefault("resolver", "wan21_vae")
    return resolved


__all__ = ["resolve_wan21_vae_keyspace"]
