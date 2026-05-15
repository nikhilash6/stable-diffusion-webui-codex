"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Variant detection and validation for canonical CLIP vision state dicts.
Infers supported `ClipVisionVariant` values from the normalized HF `CLIPVisionModelWithProjection` keyspace and performs lightweight
consistency checks without assuming raw OpenCLIP or wrapper-prefixed source layouts.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger for detection/validation diagnostics.
- `detect_variant_from_state_dict` (function): Detects the supported CLIP vision variant from canonical HF keys.
- `get_spec_for_state_dict` (function): Returns the variant spec corresponding to a detected state dict.
- `validate_state_dict` (function): Validates a canonical CLIP vision state dict matches the provided variant spec.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Mapping, Sequence

from .errors import ClipVisionConfigError, ClipVisionLoadError
from .specs import ClipVisionVariant, ClipVisionVariantSpec, get_variant_spec

logger = get_backend_logger("backend.runtime.vision.clip.registry")


def _shape_of(state_dict: Mapping[str, object], key: str) -> tuple[int, ...] | None:
    shape_getter = getattr(state_dict, "shape_of", None)
    if callable(shape_getter):
        try:
            shape = shape_getter(key)
        except Exception:
            shape = None
        if shape is not None:
            try:
                return tuple(int(v) for v in shape)
            except Exception:
                return None
    if key not in state_dict:
        return None
    value = state_dict[key]
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(v) for v in shape)
    except Exception:
        return None


def detect_variant_from_state_dict(state_dict: Mapping[str, object]) -> ClipVisionVariant:
    """Infer the clip vision variant from canonical HF CLIP vision keys."""
    keys = state_dict.keys()
    if "vision_model.encoder.layers.47.layer_norm1.weight" in keys:
        return ClipVisionVariant.G
    if "vision_model.encoder.layers.30.layer_norm1.weight" in keys:
        return ClipVisionVariant.H
    if "vision_model.encoder.layers.22.layer_norm1.weight" in keys:
        embed_key = "vision_model.embeddings.position_embedding.weight"
        if embed_key not in keys:
            raise ClipVisionLoadError(
                "Unable to detect clip vision variant: missing position embedding for VIT-L family."
            )
        embed_shape = _shape_of(state_dict, embed_key)
        if embed_shape is None or not embed_shape:
            raise ClipVisionLoadError(
                "Unable to detect clip vision variant: position embedding shape is unavailable."
            )
        embed_rows = embed_shape[0]
        if embed_rows == 577:
            return ClipVisionVariant.VIT_L
        if embed_rows in (729, 1024):
            raise ClipVisionConfigError(
                "SigLIP-style clip vision checkpoints are not yet supported; "
                "please convert or downsample to a supported Codex variant."
            )
        raise ClipVisionLoadError(
            f"Unrecognised position embedding shape ({embed_rows}) for clip vision encoder."
        )
    raise ClipVisionLoadError(
        "Unable to detect clip vision variant: expected canonical `vision_model.encoder.layers.*.layer_norm1.weight` markers."
    )


def get_spec_for_state_dict(state_dict: Mapping[str, object]) -> ClipVisionVariantSpec:
    variant = detect_variant_from_state_dict(state_dict)
    return get_variant_spec(variant)


def validate_state_dict(state_dict: Mapping[str, object], spec: ClipVisionVariantSpec) -> None:
    """Perform cheap validations to surface mismatches early."""
    expected_prefix = "vision_model.encoder.layers."
    candidate_layers: Sequence[int] = []
    for key in state_dict.keys():
        if key.startswith(expected_prefix) and key.endswith(".layer_norm1.weight"):
            try:
                layer_index = int(key[len(expected_prefix) :].split(".", 1)[0])
            except ValueError:
                continue
            candidate_layers.append(layer_index)
    if not candidate_layers:
        raise ClipVisionLoadError("State dict does not contain encoder layer norm weights.")
    max_layer_index = max(candidate_layers)
    if max_layer_index + 1 != spec.num_hidden_layers:
        raise ClipVisionLoadError(
            f"State dict encoder layer count mismatch: expected {spec.num_hidden_layers}, "
            f"detected {max_layer_index + 1}."
        )
    if "vision_model.post_layernorm.weight" not in state_dict:
        raise ClipVisionLoadError("State dict missing post_layernorm weights required for projection head.")
    if "visual_projection.weight" not in state_dict:
        raise ClipVisionLoadError("State dict missing `visual_projection.weight` required for CLIP vision output projection.")
    logger.debug(
        "Validated clip vision state dict against variant %s (%d layers).",
        spec.variant.value,
        spec.num_hidden_layers,
    )
