"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z-Image transformer GGUF/runtime key-style resolver.
Resolves Z-Image core checkpoints between runtime-export keys (`x_embedder.*`, fused `attention.qkv.*`) and native/source keys
(`all_x_embedder.*`, split `attention.to_{q,k,v}.*`) into the canonical runtime lookup keyspace used by the Codex Z-Image runtime,
without mutating or materializing a remapped state dict; wrapper/prefix rewrite attempts fail loud.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_zimage_transformer_keyspace` (function): Resolves runtime-export or native/source Z-Image transformer keys into canonical runtime lookup keys.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import Any, TypeVar

from apps.backend.runtime.state_dict.key_mapping import (
    fail_on_key_name_rewrite,
    KeyMappingError,
    KeySentinel,
    KeyStyle,
    KeyStyleDetector,
    KeyStyleSpec,
    ResolvedKeyspace,
    SentinelKind,
)
from apps.backend.runtime.state_dict.views import (
    ComputedKeyspaceView,
    KeyspaceLookupView,
    _concat_tensor_rows,
)

_T = TypeVar("_T")

_WRAPPER_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
)

_IGNORED_KEYS = frozenset({"__metadata__"})
_UNSUPPORTED_PREFIXES = (
    "text_encoder.",
    "text_encoders.",
    "vae.",
)

_RX_ALL_X_EMBEDDER = re.compile(r"^all_x_embedder\.[^.]+\.(?P<suffix>weight|bias)$")
_RX_ALL_FINAL_LAYER = re.compile(r"^all_final_layer\.[^.]+\.(?P<rest>.+)$")
_RX_ATTN_QKV = re.compile(r"^(?P<prefix>.+\.attention)\.to_(?P<which>[qkv])\.(?P<param>weight|bias)$")
_RX_ATTN_OUT = re.compile(r"^(?P<prefix>.+\.attention)\.to_out\.0\.(?P<param>weight|bias)$")
_RX_ATTN_NORM = re.compile(r"^(?P<prefix>.+\.attention)\.norm_(?P<which>[qk])\.weight$")

_DETECTOR = KeyStyleDetector(
    name="zimage_transformer_runtime_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.EXACT, "x_embedder.weight"),
                KeySentinel(SentinelKind.EXACT, "final_layer.linear.weight"),
                KeySentinel(SentinelKind.PREFIX, "layers.0.attention.qkv."),
                KeySentinel(SentinelKind.PREFIX, "layers.0.attention.out."),
            ),
            min_sentinel_hits=2,
        ),
        KeyStyleSpec(
            style=KeyStyle.DIFFUSERS,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "all_x_embedder."),
                KeySentinel(SentinelKind.PREFIX, "all_final_layer."),
                KeySentinel(SentinelKind.PREFIX, "layers.0.attention.to_q."),
                KeySentinel(SentinelKind.PREFIX, "layers.0.attention.to_out.0."),
            ),
            min_sentinel_hits=2,
        ),
    ),
)


def _validated_source_key(key: str) -> str:
    return fail_on_key_name_rewrite(str(key), _WRAPPER_PREFIXES)


def _source_keys_to_source(state_dict: MutableMapping[str, _T]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_key in state_dict.keys():
        source_key = str(raw_key)
        validated_source_key = _validated_source_key(source_key)
        if validated_source_key in _IGNORED_KEYS:
            continue
        previous = out.get(validated_source_key)
        if previous is not None and previous != source_key:
            raise KeyMappingError(
                "zimage_transformer_runtime_key_style: source-key collision "
                f"for key={validated_source_key!r} srcs={previous!r},{source_key!r}"
            )
        out[validated_source_key] = source_key
    if not out:
        raise KeyMappingError("zimage_transformer_runtime_key_style: no tensor keys remained after source-key validation")
    return out


def _reject_unsupported_prefixes(keys: tuple[str, ...]) -> None:
    offenders = [key for key in keys if key.startswith(_UNSUPPORTED_PREFIXES)]
    if offenders:
        raise KeyMappingError(
            "zimage_transformer_runtime_key_style: Z-Image GGUF core checkpoints must not embed text encoder/VAE keys. "
            f"offenders_sample={offenders[:10]}"
        )


def _shape_of(state_dict: MutableMapping[str, _T], source_key: str) -> tuple[int, ...]:
    shape_getter = getattr(state_dict, "shape_of", None)
    if callable(shape_getter):
        shape = shape_getter(source_key)
        if shape is not None:
            return tuple(int(v) for v in shape)
    tensor = state_dict[source_key]
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise KeyMappingError(
            "zimage_transformer_runtime_key_style: source tensor is missing shape metadata "
            f"for key={source_key!r}"
        )
    return tuple(int(v) for v in shape)


def _validate_concat_shapes(
    state_dict: MutableMapping[str, _T],
    *,
    source_keys: tuple[str, ...],
    destination_key: str,
) -> tuple[int, ...]:
    shapes = tuple(_shape_of(state_dict, source_key) for source_key in source_keys)
    base = shapes[0]
    if len(base) not in (1, 2):
        raise KeyMappingError(
            "zimage_transformer_runtime_key_style: concat_dim0 expects rank-1 or rank-2 tensors. "
            f"destination={destination_key!r} shape={base!r}"
        )
    trailing = base[1:]
    total_rows = 0
    for source_key, shape in zip(source_keys, shapes, strict=True):
        if len(shape) != len(base) or shape[1:] != trailing:
            raise KeyMappingError(
                "zimage_transformer_runtime_key_style: concat_dim0 source shape mismatch. "
                f"destination={destination_key!r} source={source_key!r} shape={shape!r} expected_trailing={trailing!r}"
            )
        total_rows += int(shape[0])
    return (total_rows, *trailing)


def resolve_zimage_transformer_keyspace(state_dict: MutableMapping[str, _T]) -> ResolvedKeyspace[_T]:
    source_keys_to_source = _source_keys_to_source(state_dict)
    source_keys = tuple(source_keys_to_source.keys())
    _reject_unsupported_prefixes(source_keys)
    style = _DETECTOR.detect(source_keys)

    if style is KeyStyle.CODEX:
        return ResolvedKeyspace(
            style=style,
            canonical_to_source=dict(source_keys_to_source),
            metadata={
                "resolver": "zimage_transformer",
                "source_style": style.value,
            },
            view=KeyspaceLookupView(state_dict, source_keys_to_source),
        )

    canonical_to_source: dict[str, str] = {}
    computed_shapes: dict[str, tuple[int, ...]] = {}
    computed_sources: dict[str, str] = {}
    computed: dict[str, Any] = {}

    def _register_direct(canonical_key: str, source_key: str) -> None:
        previous = canonical_to_source.get(canonical_key)
        if previous is not None and previous != source_key:
            raise KeyMappingError(
                "zimage_transformer_runtime_key_style: multiple source keys map to the same runtime key. "
                f"dst={canonical_key!r} srcs={previous!r},{source_key!r}"
            )
        canonical_to_source[canonical_key] = source_key

    qkv_groups: dict[tuple[str, str], dict[str, str]] = {}

    for source_key_name, source_key in source_keys_to_source.items():
        match = _RX_ALL_X_EMBEDDER.match(source_key_name)
        if match is not None:
            _register_direct(f"x_embedder.{match.group('suffix')}", source_key)
            continue

        match = _RX_ALL_FINAL_LAYER.match(source_key_name)
        if match is not None:
            _register_direct(f"final_layer.{match.group('rest')}", source_key)
            continue

        match = _RX_ATTN_OUT.match(source_key_name)
        if match is not None:
            _register_direct(f"{match.group('prefix')}.out.{match.group('param')}", source_key)
            continue

        match = _RX_ATTN_NORM.match(source_key_name)
        if match is not None:
            suffix = "q_norm" if match.group("which") == "q" else "k_norm"
            _register_direct(f"{match.group('prefix')}.{suffix}.weight", source_key)
            continue

        match = _RX_ATTN_QKV.match(source_key_name)
        if match is not None:
            group_key = (match.group("prefix"), match.group("param"))
            qkv_groups.setdefault(group_key, {})[match.group("which")] = source_key
            continue

        _register_direct(source_key_name, source_key)

    for (prefix, param), sources in sorted(qkv_groups.items()):
        missing = [which for which in ("q", "k", "v") if which not in sources]
        if missing:
            raise KeyMappingError(
                "zimage_transformer_runtime_key_style: missing split attention projections for fused runtime key. "
                f"prefix={prefix!r} param={param!r} missing={missing!r}"
            )
        source_keys = (sources["q"], sources["k"], sources["v"])
        destination_key = f"{prefix}.qkv.{param}"
        logical_shape = _validate_concat_shapes(state_dict, source_keys=source_keys, destination_key=destination_key)
        computed_shapes[destination_key] = logical_shape
        computed_sources[destination_key] = f"concat_dim0({', '.join(source_keys)})"
        computed[destination_key] = lambda source_keys=source_keys, logical_shape=logical_shape: _concat_tensor_rows(
            tuple(state_dict[source_key] for source_key in source_keys),
            logical_shape=logical_shape,
        )

    return ResolvedKeyspace(
        style=style,
        canonical_to_source=dict(canonical_to_source),
        metadata={
            "resolver": "zimage_transformer",
            "source_style": style.value,
        },
        view=ComputedKeyspaceView(
            state_dict,
            canonical_to_source,
            computed,
            computed_shapes=computed_shapes,
            computed_sources=computed_sources,
        ),
    )


__all__ = ["resolve_zimage_transformer_keyspace"]
