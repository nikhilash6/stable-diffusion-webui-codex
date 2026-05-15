"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical keyspace resolver for Anima transformer checkpoints.
Resolves raw Anima core checkpoints stored under `net.*` (the native checkpoint keyspace) or already-canonical runtime keys into the
canonical lookup space used by the Codex Anima runtime, without mutating source layer names or materializing a remapped tensor dict;
mixed keyspaces, wrapper-prefix rewrite attempts, and unsupported owners fail loud.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_anima_transformer_keyspace` (function): Resolves raw `net.*` or canonical Anima transformer keys into the canonical runtime lookup keyspace.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TypeVar, cast

from apps.backend.runtime.state_dict.key_mapping import (
    KeyMappingError,
    KeyStyle,
    ResolvedKeyspace,
    fail_on_key_name_rewrite,
)
from apps.backend.runtime.state_dict.views import KeyspaceLookupView

_T = TypeVar("_T")


_RAW_PREFIX = "net."
_IGNORED_META_KEYS = frozenset({"__metadata__"})
_CANONICAL_PREFIXES = (
    "x_embedder.",
    "t_embedder.",
    "fps_embedder.",
    "blocks.",
    "final_layer.",
    "t_embedding_norm.",
    "pos_embedder.",
    "extra_pos_embedder.",
    "llm_adapter.",
)
_FORBIDDEN_WRAPPER_PREFIXES = (
    "module.",
    "model.",
    "transformer.",
    "diffusion_model.",
    "model.diffusion_model.",
)
_REQUIRED_CANONICAL_KEYS = (
    "x_embedder.proj.1.weight",
    "t_embedder.1.linear_1.weight",
    "blocks.0.self_attn.q_proj.weight",
    "blocks.0.cross_attn.k_proj.weight",
    "final_layer.linear.weight",
    "llm_adapter.embed.weight",
)


def _is_ignored_key(key: str) -> bool:
    return key in _IGNORED_META_KEYS


def _is_canonical_anima_key(key: str) -> bool:
    return key.startswith(_CANONICAL_PREFIXES)


def _remember_mapping(mapping: dict[str, str], *, canonical_key: str, source_key: str) -> None:
    previous = mapping.get(canonical_key)
    if previous is not None and previous != source_key:
        raise KeyMappingError(
            "anima_transformer_keyspace: multiple source keys map to the same canonical key. "
            f"dst={canonical_key!r} srcs={previous!r},{source_key!r}"
        )
    mapping[canonical_key] = source_key


def _validate_required_keys(keys: Sequence[str]) -> None:
    missing = [key for key in _REQUIRED_CANONICAL_KEYS if key not in keys]
    if missing:
        raise KeyMappingError(
            "anima_transformer_keyspace: resolver output is missing required canonical keys. "
            f"missing_sample={missing[:10]}"
        )


def resolve_anima_transformer_keyspace(state_dict: Mapping[str, _T]) -> ResolvedKeyspace[_T]:
    """Resolve Anima transformer source keys into the canonical runtime lookup space.

    Supported source styles:
    - raw checkpoint-native keys under `net.*`
    - already-canonical runtime keys (`x_embedder.*`, `blocks.*`, `llm_adapter.*`, ...)

    The resolver is strict:
    - mixed raw + canonical inputs are rejected;
    - wrapper-prefix rewrite attempts (`module.*`, `model.*`, `transformer.*`, ...) are rejected;
    - unsupported top-level owners fail loud instead of being silently dropped.
    """

    keys: list[str] = []
    for key in state_dict.keys():
        if not isinstance(key, str):
            raise KeyMappingError(
                "anima_transformer_keyspace: checkpoint keys must be strings; "
                f"got {type(key).__name__}."
            )
        keys.append(key)
    if not keys:
        raise KeyMappingError("anima_transformer_keyspace: empty key list; cannot resolve keyspace")

    canonical_to_source: dict[str, str] = {}
    has_raw_keys = False
    has_canonical_keys = False
    unsupported_keys: list[str] = []

    for source_key in keys:
        if _is_ignored_key(source_key):
            continue

        fail_on_key_name_rewrite(source_key, _FORBIDDEN_WRAPPER_PREFIXES)

        if source_key.startswith(_RAW_PREFIX):
            has_raw_keys = True
            canonical_key = source_key[len(_RAW_PREFIX):]
            if not canonical_key:
                raise KeyMappingError("anima_transformer_keyspace: encountered empty canonical key after 'net.' prefix")
            if not _is_canonical_anima_key(canonical_key):
                unsupported_keys.append(source_key)
                continue
            _remember_mapping(canonical_to_source, canonical_key=canonical_key, source_key=source_key)
            continue

        if _is_canonical_anima_key(source_key):
            has_canonical_keys = True
            _remember_mapping(canonical_to_source, canonical_key=source_key, source_key=source_key)
            continue

        unsupported_keys.append(source_key)

    if has_raw_keys and has_canonical_keys:
        raise KeyMappingError(
            "anima_transformer_keyspace: mixed raw `net.*` and canonical Anima keys are unsupported. "
            "Pass one source style only."
        )

    if unsupported_keys:
        raise KeyMappingError(
            "anima_transformer_keyspace: unsupported keys for Anima transformer resolution. "
            f"offenders_sample={unsupported_keys[:10]}"
        )

    if not canonical_to_source:
        raise KeyMappingError(
            "anima_transformer_keyspace: no tensor keys remained after metadata filtering and source-style validation"
        )

    _validate_required_keys(tuple(canonical_to_source.keys()))
    style: KeyStyle | str = "anima_net" if has_raw_keys else KeyStyle.CODEX

    return ResolvedKeyspace(
        style=style,
        canonical_to_source=canonical_to_source,
        metadata={
            "detector": "anima_transformer_keyspace",
            "source_keys": len(keys),
            "canonical_keys": len(canonical_to_source),
        },
        view=KeyspaceLookupView(cast(MutableMapping[str, _T], state_dict), canonical_to_source),
    )


__all__ = ["resolve_anima_transformer_keyspace"]
