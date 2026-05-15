"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDXL checkpoint key-style detection + explicit source-style mapping for original-layout checkpoints.
Provides a strict, fail-loud resolver that maps known original-checkpoint wrapper/container source styles into the canonical parser lookup
keyspace without mutating stored layer names or inventing unsupported layouts.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_sdxl_checkpoint_keyspace` (function): Resolves original-format SDXL checkpoint source styles into the canonical parser lookup keyspace.
"""

from __future__ import annotations

from collections.abc import MutableMapping
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


_DETECTOR = KeyStyleDetector(
    name="sdxl_checkpoint_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "model.diffusion_model."),
                KeySentinel(SentinelKind.PREFIX, "conditioner."),
                KeySentinel(SentinelKind.PREFIX, "first_stage_model."),
            ),
            min_sentinel_hits=1,
        ),
    ),
)


def _map_label_embedding_alias(lookup_key: str) -> str:
    if not lookup_key.startswith("model.diffusion_model.label_emb.0."):
        return lookup_key
    suffix = lookup_key[len("model.diffusion_model.label_emb.0.") :]
    suffix_parts = suffix.split(".")
    if len(suffix_parts) >= 2 and suffix_parts[0].isdigit():
        return "model.diffusion_model.label_emb." + ".".join([suffix_parts[0], *suffix_parts[1:]])
    return lookup_key


def _map_source_key_to_checkpoint_lookup_key(source_key: str) -> str:
    candidate_key = str(source_key)
    if candidate_key.startswith("module."):
        module_inner_key = candidate_key[len("module.") :]
        if not module_inner_key.startswith("module."):
            candidate_key = module_inner_key

    if candidate_key.startswith("model.diffusion_model."):
        return _map_label_embedding_alias(candidate_key)
    if candidate_key.startswith("diffusion_model."):
        return _map_label_embedding_alias("model.diffusion_model." + candidate_key[len("diffusion_model.") :])
    if candidate_key.startswith("model.model.diffusion_model."):
        return _map_label_embedding_alias(
            "model.diffusion_model." + candidate_key[len("model.model.diffusion_model.") :]
        )
    if candidate_key.startswith("conditioner."):
        return candidate_key
    if candidate_key.startswith("model.conditioner."):
        return "conditioner." + candidate_key[len("model.conditioner.") :]
    if candidate_key.startswith("model.model.conditioner."):
        return "conditioner." + candidate_key[len("model.model.conditioner.") :]
    if candidate_key.startswith("first_stage_model."):
        return candidate_key
    if candidate_key.startswith("model.first_stage_model."):
        return "first_stage_model." + candidate_key[len("model.first_stage_model.") :]
    if candidate_key.startswith("model.model.first_stage_model."):
        return "first_stage_model." + candidate_key[len("model.model.first_stage_model.") :]
    if candidate_key.startswith("vae."):
        return "first_stage_model." + candidate_key[len("vae.") :]
    if candidate_key.startswith("model.vae."):
        return "first_stage_model." + candidate_key[len("model.vae.") :]
    if candidate_key.startswith("model.model.vae."):
        return "first_stage_model." + candidate_key[len("model.model.vae.") :]
    return candidate_key


def resolve_sdxl_checkpoint_keyspace(state_dict: MutableMapping[str, _T]) -> ResolvedKeyspace[_T]:
    """Resolve original-format SDXL checkpoint source styles into canonical parser lookup keys."""

    source_keys = [str(key) for key in state_dict.keys()]
    if not source_keys:
        raise KeyMappingError("sdxl_checkpoint_key_style: empty key list; cannot detect key style")

    lookup_keys = [_map_source_key_to_checkpoint_lookup_key(source_key) for source_key in source_keys]
    style = _DETECTOR.detect(lookup_keys)

    canonical_to_source: dict[str, str] = {}
    for source_key, lookup_key in zip(source_keys, lookup_keys, strict=True):
        previous_source = canonical_to_source.get(lookup_key)
        if previous_source is not None and previous_source != source_key:
            raise KeyMappingError(
                "sdxl_checkpoint_key_style: multiple source keys map to the same lookup key: "
                f"dst={lookup_key!r} srcs={previous_source!r},{source_key!r}"
            )
        canonical_to_source[lookup_key] = source_key

    return ResolvedKeyspace(
        style=style,
        canonical_to_source=canonical_to_source,
        metadata={
            "resolver": "sdxl_checkpoint",
            "detector": _DETECTOR.name,
            "source_keys": len(source_keys),
            "canonical_keys": len(canonical_to_source),
        },
        view=KeyspaceLookupView(state_dict, canonical_to_source),
    )


__all__ = ["resolve_sdxl_checkpoint_keyspace"]
