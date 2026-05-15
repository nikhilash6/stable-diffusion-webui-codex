"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical key-style detection + explicit source-style mapping for Qwen text-encoder/backbone state_dict keys.
Provides strict, fail-loud keyspace resolution for Qwen text checkpoints that arrive as native HF keys (`model.*`) or the known wrapper/container
surfaces around that backbone, while explicitly allowing known auxiliary heads (`lm_head.*`, optional `visual.*`), ignoring safetensors metadata
sentinels, and refusing unknown keyspaces.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_qwen_text_encoder_keyspace` (function): Resolves Qwen text-checkpoint source styles into canonical backbone keys (`model.*`) and drops known auxiliary heads.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import TypeVar

from apps.backend.runtime.state_dict.key_mapping import (
    KeyMappingError,
    KeySentinel,
    KeyStyle,
    KeyStyleDetector,
    KeyStyleSpec,
    ResolvedKeyspace,
    SentinelKind,
)
from apps.backend.runtime.state_dict.views import KeyspaceLookupView

_T = TypeVar("_T")


_WRAPPER_PREFIXES = (
    "module.",
    "text_encoder.",
    "language_model.",
    "text_model.",
)

_REQUIRED_BACKBONE_KEYS = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.norm.weight",
)

_IGNORED_META_KEYS = frozenset(
    {
        "__metadata__",
    }
)

_DETECTOR = KeyStyleDetector(
    name="qwen_text_encoder_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.HF,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "model."),
                KeySentinel(SentinelKind.PREFIX, "lm_head."),
                KeySentinel(SentinelKind.PREFIX, "visual."),
            ),
            min_sentinel_hits=1,
        ),
    ),
)


def _is_supported_qwen_root_key(key: str) -> bool:
    return key in _IGNORED_META_KEYS or key.startswith(("model.", "lm_head.", "visual."))


def _map_source_key_to_backbone_key(key: str) -> str:
    source_key = str(key)
    if _is_supported_qwen_root_key(source_key):
        return source_key
    for wrapper_prefix in _WRAPPER_PREFIXES:
        if source_key.startswith(wrapper_prefix):
            candidate_key = source_key[len(wrapper_prefix) :]
            if _is_supported_qwen_root_key(candidate_key):
                return candidate_key
            break
    return source_key


def _validate_required_backbone_keys(keys: Sequence[str], *, context: str) -> None:
    missing = [key for key in _REQUIRED_BACKBONE_KEYS if key not in keys]
    if missing:
        preview = ", ".join(sorted(keys)[:10])
        raise KeyMappingError(
            f"{context}: missing required Qwen backbone keys: {missing}. sample_keys=[{preview}]"
        )


def resolve_qwen_text_encoder_keyspace(
    state_dict: MutableMapping[str, _T],
    *,
    allow_lm_head_aux: bool = True,
    allow_visual_aux: bool = True,
    require_backbone_keys: bool = True,
) -> ResolvedKeyspace[_T]:
    """Resolve Qwen text-encoder source styles into canonical backbone keys.

    Supported upstream styles:
    - HF: `model.*` (plus optional `lm_head.*`, `visual.*`)
    - Wrapped HF: `module.*`, `text_encoder.*`, `language_model.*`, `text_model.*`

    Resolver behavior:
    - Keeps canonical backbone weights under `model.*`.
    - Drops known auxiliary heads (`lm_head.*`, optional `visual.*`).
    - Ignores known metadata-only sentinel keys (currently `__metadata__`).
    - Raises on unknown keys, ambiguous style detection, key collisions, or missing required backbone keys.
    """

    keys = [str(key) for key in state_dict.keys()]
    if not keys:
        raise KeyMappingError("qwen_text_encoder_key_style: empty key list; cannot detect key style")

    lookup_keys_for_detection: list[str] = []
    for key in keys:
        lookup_key = _map_source_key_to_backbone_key(key)
        if lookup_key in _IGNORED_META_KEYS:
            continue
        lookup_keys_for_detection.append(lookup_key)
    if not lookup_keys_for_detection:
        raise KeyMappingError(
            "qwen_text_encoder_key_style: no tensor keys remained after metadata filtering; cannot detect key style"
        )
    style = _DETECTOR.detect(lookup_keys_for_detection)

    canonical_to_source: dict[str, str] = {}
    unsupported_keys: list[str] = []
    for source_key in keys:
        lookup_key = _map_source_key_to_backbone_key(source_key)
        if lookup_key in _IGNORED_META_KEYS:
            continue
        if lookup_key.startswith("model."):
            previous = canonical_to_source.get(lookup_key)
            if previous is not None and previous != source_key:
                raise KeyMappingError(
                    "qwen_text_encoder_key_style: multiple source keys map to the same destination key: "
                    f"dst={lookup_key!r} srcs={previous!r},{source_key!r}"
                )
            canonical_to_source[lookup_key] = source_key
            continue

        if lookup_key.startswith("lm_head."):
            if not allow_lm_head_aux:
                unsupported_keys.append(lookup_key)
            continue

        if lookup_key.startswith("visual."):
            if not allow_visual_aux:
                unsupported_keys.append(lookup_key)
            continue

        unsupported_keys.append(lookup_key)

    if unsupported_keys:
        sample = ", ".join(unsupported_keys[:10])
        raise KeyMappingError(
            "qwen_text_encoder_key_style: unsupported source keys after explicit source-style mapping. "
            "Allowed destinations are `model.*` plus optional aux branches "
            f"`lm_head.*`/`visual.*`. offenders_sample=[{sample}]"
        )

    canonical_keys = list(canonical_to_source.keys())
    if not canonical_keys:
        raise KeyMappingError(
            "qwen_text_encoder_key_style: no canonical backbone keys (`model.*`) were produced after explicit source-style mapping"
        )

    if require_backbone_keys:
        _validate_required_backbone_keys(
            canonical_keys,
            context="qwen_text_encoder_key_style",
        )

    return ResolvedKeyspace(
        style=style,
        canonical_to_source=canonical_to_source,
        metadata={
            "resolver": "qwen_text_encoder",
            "allow_lm_head_aux": bool(allow_lm_head_aux),
            "allow_visual_aux": bool(allow_visual_aux),
            "require_backbone_keys": bool(require_backbone_keys),
        },
        view=KeyspaceLookupView(state_dict, canonical_to_source),
    )


__all__ = ["resolve_qwen_text_encoder_keyspace"]
