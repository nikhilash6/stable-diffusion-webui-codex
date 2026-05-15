"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical CLIP vision keyspace detection + explicit source-style mapping into HF `CLIPVisionModelWithProjection` keys.
Supports native HF `vision_model.*` checkpoints (including explicit full-CLIP extra text/logit keys), explicit wrapped
`image_encoder.vision_model.*` image-encoder checkpoints, and OpenCLIP `visual.*` checkpoints (including explicit full-CLIP extras) via
lazy lookup/computed views. Fails loud on mixed layouts, unsupported extra keys, and structural conversions blocked by
`CODEX_WEIGHT_STRUCTURAL_CONVERSION`.

Symbols (top-level; keep in sync; no ghosts):
- `ClipVisionLayoutMetadata` (class): Source-layout metadata for CLIP vision state-dict resolution.
- `resolve_clip_vision_keyspace_with_layout` (function): Resolves supported CLIP vision source styles into canonical HF lookup keys.
- `clip_vision_layout_metadata_from_resolved` (function): Extracts typed CLIP vision layout metadata from a resolved keyspace.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Literal, TypeVar

import torch

from apps.backend.infra.config.weight_structural_conversion import (
    ENV_WEIGHT_STRUCTURAL_CONVERSION,
    is_structural_weight_conversion_enabled,
)
from apps.backend.runtime.state_dict.key_mapping import KeyMappingError, KeyStyle, ResolvedKeyspace
from apps.backend.runtime.state_dict.views import ComputedKeyspaceView, KeyspaceLookupView

_T = TypeVar("_T")
_ClipVisionSourceStyle = Literal["hf", "hf_wrapped", "openclip"]
_QKVLayout = Literal["split", "fused"]
_ProjectionOrientation = Literal["linear", "matmul"]

_HF_EXACT_KEYS: tuple[str, ...] = (
    "vision_model.embeddings.class_embedding",
    "vision_model.embeddings.patch_embedding.weight",
    "vision_model.embeddings.position_embedding.weight",
    "vision_model.pre_layrnorm.weight",
    "vision_model.pre_layrnorm.bias",
    "vision_model.post_layernorm.weight",
    "vision_model.post_layernorm.bias",
    "visual_projection.weight",
)
_HF_ALLOWED_DROPS: tuple[str, ...] = ("vision_model.embeddings.position_ids",)
_HF_ALLOWED_EXTRA_PREFIXES: tuple[str, ...] = ("text_model.",)
_HF_ALLOWED_EXTRA_KEYS: tuple[str, ...] = ("text_projection.weight", "logit_scale")
_HF_LAYER_KEY_RE = re.compile(
    r"^vision_model\.encoder\.layers\.(\d+)\."
    r"(?:(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj))|(?:layer_norm[12])|(?:mlp\.(?:fc1|fc2)))"
    r"\.(?:weight|bias)$"
)

_OPENCLIP_DIRECT_KEYS: dict[str, str] = {
    "visual.class_embedding": "vision_model.embeddings.class_embedding",
    "visual.conv1.weight": "vision_model.embeddings.patch_embedding.weight",
    "visual.positional_embedding": "vision_model.embeddings.position_embedding.weight",
    "visual.ln_pre.weight": "vision_model.pre_layrnorm.weight",
    "visual.ln_pre.bias": "vision_model.pre_layrnorm.bias",
    "visual.ln_post.weight": "vision_model.post_layernorm.weight",
    "visual.ln_post.bias": "vision_model.post_layernorm.bias",
}
_OPENCLIP_ALLOWED_EXTRA_PREFIXES: tuple[str, ...] = ("transformer.resblocks.",)
_OPENCLIP_ALLOWED_EXTRA_KEYS: tuple[str, ...] = (
    "token_embedding.weight",
    "positional_embedding",
    "ln_final.weight",
    "ln_final.bias",
    "text_projection",
    "text_projection.weight",
    "logit_scale",
)
_OPENCLIP_LAYER_RE = re.compile(r"^visual\.transformer\.resblocks\.(\d+)\.(.+)$")
_OPENCLIP_LAYER_DIRECT_SUFFIXES: dict[str, str] = {
    "attn.out_proj.weight": "self_attn.out_proj.weight",
    "attn.out_proj.bias": "self_attn.out_proj.bias",
    "ln_1.weight": "layer_norm1.weight",
    "ln_1.bias": "layer_norm1.bias",
    "ln_2.weight": "layer_norm2.weight",
    "ln_2.bias": "layer_norm2.bias",
    "mlp.c_fc.weight": "mlp.fc1.weight",
    "mlp.c_fc.bias": "mlp.fc1.bias",
    "mlp.c_proj.weight": "mlp.fc2.weight",
    "mlp.c_proj.bias": "mlp.fc2.bias",
}
_REQUIRED_CANONICAL_KEYS: tuple[str, ...] = (
    "vision_model.embeddings.class_embedding",
    "vision_model.embeddings.patch_embedding.weight",
    "vision_model.embeddings.position_embedding.weight",
    "vision_model.pre_layrnorm.weight",
    "vision_model.post_layernorm.weight",
    "visual_projection.weight",
    "vision_model.encoder.layers.0.self_attn.q_proj.weight",
    "vision_model.encoder.layers.0.self_attn.k_proj.weight",
    "vision_model.encoder.layers.0.self_attn.v_proj.weight",
    "vision_model.encoder.layers.0.self_attn.out_proj.weight",
    "vision_model.encoder.layers.0.layer_norm1.weight",
    "vision_model.encoder.layers.0.layer_norm2.weight",
    "vision_model.encoder.layers.0.mlp.fc1.weight",
    "vision_model.encoder.layers.0.mlp.fc2.weight",
)


@dataclass(frozen=True, slots=True)
class ClipVisionLayoutMetadata:
    source_style: _ClipVisionSourceStyle
    qkv_layout: _QKVLayout
    projection_orientation: _ProjectionOrientation


def _source_keys(state_dict: MutableMapping[str, object]) -> list[str]:
    keys: list[str] = []
    for raw_key in state_dict.keys():
        if not isinstance(raw_key, str):
            raise KeyMappingError(
                f"clip_vision: checkpoint keys must be strings; got {type(raw_key).__name__}"
            )
        keys.append(raw_key)
    return keys


def _shape_of(mapping: MutableMapping[str, object], key: str) -> tuple[int, ...] | None:
    shape_getter = getattr(mapping, "shape_of", None)
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
    if key not in mapping:
        return None
    value = mapping[key]
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(v) for v in shape)
    except Exception:
        return None


def _is_supported_hf_key(key: str) -> bool:
    return key in _HF_EXACT_KEYS or key in _HF_ALLOWED_DROPS or _HF_LAYER_KEY_RE.match(key) is not None


def _has_hf_native_keys(keys: list[str]) -> bool:
    return any(
        key.startswith("vision_model.")
        or key == "visual_projection.weight"
        for key in keys
    )


def _has_hf_wrapped_keys(keys: list[str]) -> bool:
    return any(
        key.startswith("image_encoder.vision_model.")
        or key == "image_encoder.visual_projection.weight"
        for key in keys
    )


def _is_openclip_key(key: str) -> bool:
    if key in _OPENCLIP_DIRECT_KEYS or key == "visual.proj":
        return True
    return _OPENCLIP_LAYER_RE.match(key) is not None


def _has_openclip_keys(keys: list[str]) -> bool:
    return any(_is_openclip_key(key) for key in keys)


def _detect_layout_metadata(state_dict: MutableMapping[str, object]) -> ClipVisionLayoutMetadata:
    keys = _source_keys(state_dict)
    has_native_hf = _has_hf_native_keys(keys)
    has_wrapped_hf = _has_hf_wrapped_keys(keys)
    has_openclip = _has_openclip_keys(keys)
    matched_styles = [
        style
        for style, present in (
            ("hf", has_native_hf),
            ("hf_wrapped", has_wrapped_hf),
            ("openclip", has_openclip),
        )
        if present
    ]
    if not matched_styles:
        preview = ", ".join(sorted(keys)[:8])
        raise KeyMappingError(
            "clip_vision: unsupported image-encoder keyspace. "
            f"Expected HF `vision_model.*`, wrapped `image_encoder.vision_model.*`, or OpenCLIP `visual.*`. "
            f"sample_keys=[{preview}]"
        )
    if len(matched_styles) > 1:
        raise KeyMappingError(
            "clip_vision: ambiguous image-encoder keyspace (mixed supported source styles). "
            f"matched_styles={matched_styles}"
        )
    style = matched_styles[0]
    if style == "openclip":
        return ClipVisionLayoutMetadata(
            source_style="openclip",
            qkv_layout="fused",
            projection_orientation="matmul",
        )
    if style == "hf_wrapped":
        return ClipVisionLayoutMetadata(
            source_style="hf_wrapped",
            qkv_layout="split",
            projection_orientation="linear",
        )
    return ClipVisionLayoutMetadata(
        source_style="hf",
        qkv_layout="split",
        projection_orientation="linear",
    )


def _validate_layout_metadata(layout_metadata: ClipVisionLayoutMetadata) -> None:
    if layout_metadata.source_style not in {"hf", "hf_wrapped", "openclip"}:
        raise KeyMappingError(
            "clip_vision: invalid cached source_style=%r (expected one of: hf, hf_wrapped, openclip)"
            % (layout_metadata.source_style,)
        )
    if layout_metadata.qkv_layout not in {"split", "fused"}:
        raise KeyMappingError(
            "clip_vision: invalid cached qkv_layout=%r (expected one of: split, fused)"
            % (layout_metadata.qkv_layout,)
        )
    if layout_metadata.projection_orientation not in {"linear", "matmul"}:
        raise KeyMappingError(
            "clip_vision: invalid cached projection_orientation=%r (expected one of: linear, matmul)"
            % (layout_metadata.projection_orientation,)
        )


def _is_hf_allowed_extra_key(key: str) -> bool:
    return key in _HF_ALLOWED_EXTRA_KEYS or key.startswith(_HF_ALLOWED_EXTRA_PREFIXES)


def _is_openclip_allowed_extra_key(key: str) -> bool:
    return key in _OPENCLIP_ALLOWED_EXTRA_KEYS or key.startswith(_OPENCLIP_ALLOWED_EXTRA_PREFIXES)


def _ensure_required_keys(mapping: dict[str, object]) -> None:
    missing = [key for key in _REQUIRED_CANONICAL_KEYS if key not in mapping]
    if missing:
        raise KeyMappingError(
            "clip_vision: canonical keyspace is missing essential tensors. "
            f"missing_sample={missing[:8]}"
        )


def _source_style_to_key_style(source_style: _ClipVisionSourceStyle) -> KeyStyle:
    if source_style in {"hf", "hf_wrapped"}:
        return KeyStyle.HF
    return KeyStyle.OPENCLIP


def _transpose_projection_tensor(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise KeyMappingError(
            f"clip_vision: expected projection tensor to be torch.Tensor, got {type(value).__name__}"
        )
    if value.ndim != 2:
        raise KeyMappingError(
            "clip_vision: OpenCLIP `visual.proj` must be 2-D before transpose. "
            f"shape={tuple(value.shape)}"
        )
    return value.transpose(0, 1).contiguous()


def _split_qkv_tensor(value: object, *, index: int, source_key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise KeyMappingError(
            f"clip_vision: expected fused QKV tensor for {source_key!r}, got {type(value).__name__}"
        )
    if value.ndim < 1:
        raise KeyMappingError(
            f"clip_vision: fused QKV tensor for {source_key!r} must have at least one dimension."
        )
    total = int(value.shape[0])
    if total % 3 != 0:
        raise KeyMappingError(
            f"clip_vision: fused QKV tensor for {source_key!r} has invalid first dim {total}; expected divisible by 3."
        )
    chunk = total // 3
    return value[(chunk * index):(chunk * (index + 1))].contiguous()


def _split_qkv_shape(shape: tuple[int, ...], *, source_key: str) -> tuple[int, ...]:
    if len(shape) < 1:
        raise KeyMappingError(
            f"clip_vision: fused QKV tensor for {source_key!r} is missing a row dimension."
        )
    total = int(shape[0])
    if total % 3 != 0:
        raise KeyMappingError(
            f"clip_vision: fused QKV tensor for {source_key!r} has invalid first dim {total}; expected divisible by 3."
        )
    return (total // 3, *shape[1:])


def _transpose_shape(shape: tuple[int, ...], *, source_key: str) -> tuple[int, ...]:
    if len(shape) != 2:
        raise KeyMappingError(
            f"clip_vision: projection tensor for {source_key!r} must be 2-D before transpose; shape={shape}"
        )
    return (shape[1], shape[0])


def _resolve_hf_keyspace(
    state_dict: MutableMapping[str, _T],
    *,
    wrapper_prefix: str,
    layout_metadata: ClipVisionLayoutMetadata,
) -> ResolvedKeyspace[_T]:
    direct_mapping: dict[str, str] = {}
    for source_key in _source_keys(state_dict):
        if not source_key.startswith(wrapper_prefix):
            raise KeyMappingError(
                "clip_vision: mixed or unsupported HF image-encoder keys. "
                f"expected_prefix={wrapper_prefix!r} offending_key={source_key!r}"
            )
        canonical_key = source_key[len(wrapper_prefix):]
        if canonical_key in _HF_ALLOWED_DROPS:
            continue
        if _is_hf_allowed_extra_key(canonical_key):
            continue
        if not _is_supported_hf_key(canonical_key):
            raise KeyMappingError(
                "clip_vision: unsupported HF image-encoder key. "
                f"source_key={source_key!r} canonical_key={canonical_key!r}"
            )
        if canonical_key in direct_mapping:
            raise KeyMappingError(
                "clip_vision: duplicate canonical HF key after wrapper resolution. "
                f"canonical_key={canonical_key!r} sources={direct_mapping[canonical_key]!r},{source_key!r}"
            )
        direct_mapping[canonical_key] = source_key
    _ensure_required_keys({key: source for key, source in direct_mapping.items()})
    return ResolvedKeyspace(
        style=_source_style_to_key_style(layout_metadata.source_style),
        canonical_to_source=dict(direct_mapping),
        metadata={
            "resolver": "clip_vision",
            "source_style": layout_metadata.source_style,
            "qkv_layout": layout_metadata.qkv_layout,
            "projection_orientation": layout_metadata.projection_orientation,
        },
        view=KeyspaceLookupView(state_dict, direct_mapping),
    )


def _resolve_openclip_keyspace(
    state_dict: MutableMapping[str, _T],
    *,
    layout_metadata: ClipVisionLayoutMetadata,
) -> ResolvedKeyspace[_T]:
    if not is_structural_weight_conversion_enabled():
        raise KeyMappingError(
            "clip_vision: OpenCLIP image-encoder weights require structural conversion into "
            "HF `CLIPVisionModelWithProjection` keys (split q/k/v + transpose `visual.proj`), "
            f"but {ENV_WEIGHT_STRUCTURAL_CONVERSION}=auto forbids it. "
            f"Set {ENV_WEIGHT_STRUCTURAL_CONVERSION}=convert to allow."
        )

    direct_mapping: dict[str, str] = {}
    computed_mapping: dict[str, object] = {}
    computed_shapes: dict[str, tuple[int, ...]] = {}
    canonical_to_source: dict[str, str] = {}

    def _put_direct(destination_key: str, source_key: str) -> None:
        if destination_key in direct_mapping or destination_key in computed_mapping:
            raise KeyMappingError(
                "clip_vision: duplicate canonical OpenCLIP destination key. "
                f"destination_key={destination_key!r} source_key={source_key!r}"
            )
        direct_mapping[destination_key] = source_key
        canonical_to_source[destination_key] = source_key

    def _put_computed(
        destination_key: str,
        *,
        source_key: str,
        compute: object,
        shape: tuple[int, ...] | None,
    ) -> None:
        if destination_key in direct_mapping or destination_key in computed_mapping:
            raise KeyMappingError(
                "clip_vision: duplicate computed OpenCLIP destination key. "
                f"destination_key={destination_key!r} source_key={source_key!r}"
            )
        computed_mapping[destination_key] = compute
        canonical_to_source[destination_key] = source_key
        if shape is not None:
            computed_shapes[destination_key] = shape

    for source_key in _source_keys(state_dict):
        if _is_openclip_allowed_extra_key(source_key):
            continue
        destination_key = _OPENCLIP_DIRECT_KEYS.get(source_key)
        if destination_key is not None:
            _put_direct(destination_key, source_key)
            continue
        if source_key == "visual.proj":
            projection_shape = _shape_of(state_dict, source_key)
            _put_computed(
                "visual_projection.weight",
                source_key=source_key,
                compute=lambda source_key=source_key: _transpose_projection_tensor(state_dict[source_key]),
                shape=None if projection_shape is None else _transpose_shape(projection_shape, source_key=source_key),
            )
            continue

        layer_match = _OPENCLIP_LAYER_RE.match(source_key)
        if layer_match is None:
            raise KeyMappingError(
                "clip_vision: unsupported OpenCLIP image-encoder key. "
                f"source_key={source_key!r}"
            )
        layer_index = int(layer_match.group(1))
        suffix = layer_match.group(2)
        layer_prefix = f"vision_model.encoder.layers.{layer_index}."
        mapped_suffix = _OPENCLIP_LAYER_DIRECT_SUFFIXES.get(suffix)
        if mapped_suffix is not None:
            _put_direct(layer_prefix + mapped_suffix, source_key)
            continue
        if suffix in {"attn.in_proj_weight", "attn.in_proj_bias"}:
            source_shape = _shape_of(state_dict, source_key)
            for qkv_index, projection_name in enumerate(("q_proj", "k_proj", "v_proj")):
                destination_key = layer_prefix + f"self_attn.{projection_name}.{suffix.rsplit('_', 1)[1]}"
                _put_computed(
                    destination_key,
                    source_key=source_key,
                    compute=lambda source_key=source_key, qkv_index=qkv_index: _split_qkv_tensor(
                        state_dict[source_key],
                        index=qkv_index,
                        source_key=source_key,
                    ),
                    shape=None if source_shape is None else _split_qkv_shape(source_shape, source_key=source_key),
                )
            continue
        raise KeyMappingError(
            "clip_vision: unsupported OpenCLIP transformer suffix for image encoder. "
            f"source_key={source_key!r} suffix={suffix!r}"
        )

    _ensure_required_keys({**direct_mapping, **computed_mapping})
    return ResolvedKeyspace(
        style=_source_style_to_key_style(layout_metadata.source_style),
        canonical_to_source=canonical_to_source,
        metadata={
            "resolver": "clip_vision",
            "source_style": layout_metadata.source_style,
            "qkv_layout": layout_metadata.qkv_layout,
            "projection_orientation": layout_metadata.projection_orientation,
        },
        view=ComputedKeyspaceView(
            state_dict,
            direct_mapping,
            computed_mapping,
            computed_shapes=computed_shapes,
        ),
    )


def resolve_clip_vision_keyspace_with_layout(
    state_dict: MutableMapping[str, _T],
) -> ResolvedKeyspace[_T]:
    detected_layout = _detect_layout_metadata(state_dict)
    _validate_layout_metadata(detected_layout)
    if detected_layout.source_style == "openclip":
        return _resolve_openclip_keyspace(state_dict, layout_metadata=detected_layout)
    wrapper_prefix = "" if detected_layout.source_style == "hf" else "image_encoder."
    return _resolve_hf_keyspace(
        state_dict,
        wrapper_prefix=wrapper_prefix,
        layout_metadata=detected_layout,
    )


def clip_vision_layout_metadata_from_resolved(resolved: ResolvedKeyspace[object]) -> ClipVisionLayoutMetadata:
    metadata = dict(getattr(resolved, "metadata", {}) or {})
    source_style = str(metadata.get("source_style", "")).strip().lower()
    qkv_layout = str(metadata.get("qkv_layout", "")).strip().lower()
    projection_orientation = str(metadata.get("projection_orientation", "")).strip().lower()
    layout = ClipVisionLayoutMetadata(
        source_style=source_style,  # type: ignore[arg-type]
        qkv_layout=qkv_layout,  # type: ignore[arg-type]
        projection_orientation=projection_orientation,  # type: ignore[arg-type]
    )
    _validate_layout_metadata(layout)
    return layout


__all__ = [
    "ClipVisionLayoutMetadata",
    "clip_vision_layout_metadata_from_resolved",
    "resolve_clip_vision_keyspace_with_layout",
]
