"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: FLUX.2 transformer runtime key-style resolver.
Resolves FLUX.2 core checkpoints between runtime-export keys (`double_blocks.*`, `single_blocks.*`, `img_in.*`) and native/source
Diffusers keys (`transformer_blocks.*`, `single_transformer_blocks.*`, `x_embedder.*`) into the canonical Diffusers lookup keyspace used by
`Flux2Transformer2DModel`, without mutating or materializing a remapped state dict; wrapper/prefix rewrite attempts fail loud.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_flux2_transformer_keyspace` (function): Resolves runtime-export or native/source FLUX.2 transformer keys into canonical Diffusers lookup keys.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
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
    _split_tensor_rows,
    _swap_tensor_row_halves,
)

_T = TypeVar("_T")

_WRAPPER_PREFIXES = (
    "transformer.",
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
)

_IGNORED_KEYS = frozenset({"__metadata__"})
_UNSUPPORTED_PREFIXES = (
    "text_encoder.",
    "text_encoders.",
    "vae.",
    "guidance_in.",
    "vector_in.",
    "time_guidance_embed.guidance_embedder.",
    "time_guidance_embed.text_embedder.",
)

_DETECTOR = KeyStyleDetector(
    name="flux2_transformer_runtime_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.EXACT, "img_in.weight"),
                KeySentinel(SentinelKind.EXACT, "double_blocks.0.img_attn.qkv.weight"),
                KeySentinel(SentinelKind.EXACT, "single_blocks.0.linear1.weight"),
                KeySentinel(SentinelKind.EXACT, "final_layer.adaLN_modulation.1.weight"),
            ),
            min_sentinel_hits=2,
        ),
        KeyStyleSpec(
            style=KeyStyle.DIFFUSERS,
            sentinels=(
                KeySentinel(SentinelKind.EXACT, "x_embedder.weight"),
                KeySentinel(SentinelKind.PREFIX, "transformer_blocks.0.attn.to_q."),
                KeySentinel(SentinelKind.PREFIX, "single_transformer_blocks.0.attn.to_qkv_mlp_proj."),
                KeySentinel(SentinelKind.EXACT, "proj_out.weight"),
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
                "flux2_transformer_runtime_key_style: source-key collision "
                f"for key={validated_source_key!r} srcs={previous!r},{source_key!r}"
            )
        out[validated_source_key] = source_key
    if not out:
        raise KeyMappingError("flux2_transformer_runtime_key_style: no tensor keys remained after source-key validation")
    return out


def _reject_unsupported_prefixes(keys: Sequence[str]) -> None:
    offenders = [key for key in keys if key.startswith(_UNSUPPORTED_PREFIXES)]
    if offenders:
        raise KeyMappingError(
            "flux2_transformer_runtime_key_style: FLUX.2 core checkpoints must not embed text encoder/VAE/guidance extras. "
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
            "flux2_transformer_runtime_key_style: source tensor is missing shape metadata "
            f"for key={source_key!r}"
        )
    return tuple(int(v) for v in shape)


def _validate_split_shape(
    state_dict: MutableMapping[str, _T],
    *,
    source_key: str,
    destination_key: str,
    chunks: int,
) -> tuple[int, ...]:
    shape = _shape_of(state_dict, source_key)
    if len(shape) not in (1, 2):
        raise KeyMappingError(
            "flux2_transformer_runtime_key_style: split_dim0 expects rank-1 or rank-2 tensors. "
            f"destination={destination_key!r} shape={shape!r}"
        )
    if shape[0] % chunks != 0:
        raise KeyMappingError(
            "flux2_transformer_runtime_key_style: split_dim0 expects the leading dimension to be divisible by chunks. "
            f"destination={destination_key!r} shape={shape!r} chunks={chunks}"
        )
    return (shape[0] // chunks, *shape[1:])


def _validate_swap_shape(
    state_dict: MutableMapping[str, _T],
    *,
    source_key: str,
    destination_key: str,
) -> tuple[int, ...]:
    shape = _shape_of(state_dict, source_key)
    if len(shape) not in (1, 2):
        raise KeyMappingError(
            "flux2_transformer_runtime_key_style: swap_dim0_halves expects rank-1 or rank-2 tensors. "
            f"destination={destination_key!r} shape={shape!r}"
        )
    if shape[0] % 2 != 0:
        raise KeyMappingError(
            "flux2_transformer_runtime_key_style: swap_dim0_halves expects an even leading dimension. "
            f"destination={destination_key!r} shape={shape!r}"
        )
    return shape


def _extract_indices(keys: Sequence[str], prefix: str) -> list[int]:
    indices: set[int] = set()
    for key in keys:
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        head = remainder.split(".", 1)[0]
        if head.isdigit():
            indices.add(int(head))
    return sorted(indices)


def resolve_flux2_transformer_keyspace(state_dict: MutableMapping[str, _T]) -> ResolvedKeyspace[_T]:
    source_keys_to_source = _source_keys_to_source(state_dict)
    source_keys = tuple(source_keys_to_source.keys())
    _reject_unsupported_prefixes(source_keys)
    style = _DETECTOR.detect(source_keys)

    if style is KeyStyle.DIFFUSERS:
        return ResolvedKeyspace(
            style=style,
            canonical_to_source=dict(source_keys_to_source),
            metadata={
                "resolver": "flux2_transformer",
                "source_style": style.value,
            },
            view=KeyspaceLookupView(state_dict, source_keys_to_source),
        )

    canonical_to_source: dict[str, str] = {}
    computed_shapes: dict[str, tuple[int, ...]] = {}
    computed_sources: dict[str, str] = {}
    computed: dict[str, Any] = {}
    consumed: set[str] = set()

    def _register_direct(canonical_key: str, source_key_name: str) -> None:
        source_key = source_keys_to_source.get(source_key_name)
        if source_key is None:
            raise KeyMappingError(
                "flux2_transformer_runtime_key_style: missing runtime-export key for Diffusers lookup. "
                f"dst={canonical_key!r} src={source_key_name!r}"
            )
        previous = canonical_to_source.get(canonical_key)
        if previous is not None and previous != source_key:
            raise KeyMappingError(
                "flux2_transformer_runtime_key_style: multiple source keys map to the same Diffusers key. "
                f"dst={canonical_key!r} srcs={previous!r},{source_key!r}"
        )
        canonical_to_source[canonical_key] = source_key
        consumed.add(source_key_name)

    def _register_split(
        canonical_key: str,
        source_key_name: str,
        *,
        chunks: int,
        index: int,
    ) -> None:
        source_key = source_keys_to_source.get(source_key_name)
        if source_key is None:
            raise KeyMappingError(
                "flux2_transformer_runtime_key_style: missing fused runtime-export tensor for Diffusers lookup. "
                f"dst={canonical_key!r} src={source_key_name!r}"
            )
        logical_shape = _validate_split_shape(
            state_dict,
            source_key=source_key,
            destination_key=canonical_key,
            chunks=chunks,
        )
        computed_shapes[canonical_key] = logical_shape
        computed_sources[canonical_key] = f"split_dim0[{index}]({source_key})"
        computed[canonical_key] = lambda source_key=source_key, chunks=chunks, index=index, logical_shape=logical_shape: _split_tensor_rows(
            state_dict[source_key],
            chunks=chunks,
            index=index,
            logical_shape=logical_shape,
        )
        consumed.add(source_key_name)

    def _register_swap(canonical_key: str, source_key_name: str) -> None:
        source_key = source_keys_to_source.get(source_key_name)
        if source_key is None:
            raise KeyMappingError(
                "flux2_transformer_runtime_key_style: missing runtime-export tensor for swapped Diffusers key. "
                f"dst={canonical_key!r} src={source_key_name!r}"
            )
        logical_shape = _validate_swap_shape(state_dict, source_key=source_key, destination_key=canonical_key)
        computed_shapes[canonical_key] = logical_shape
        computed_sources[canonical_key] = f"swap_dim0_halves({source_key})"
        computed[canonical_key] = lambda source_key=source_key, logical_shape=logical_shape: _swap_tensor_row_halves(
            state_dict[source_key],
            logical_shape=logical_shape,
        )
        consumed.add(source_key_name)

    direct_pairs = (
        ("x_embedder.weight", "img_in.weight"),
        ("x_embedder.bias", "img_in.bias"),
        ("context_embedder.weight", "txt_in.weight"),
        ("context_embedder.bias", "txt_in.bias"),
        ("time_guidance_embed.timestep_embedder.linear_1.weight", "time_in.in_layer.weight"),
        ("time_guidance_embed.timestep_embedder.linear_1.bias", "time_in.in_layer.bias"),
        ("time_guidance_embed.timestep_embedder.linear_2.weight", "time_in.out_layer.weight"),
        ("time_guidance_embed.timestep_embedder.linear_2.bias", "time_in.out_layer.bias"),
        ("double_stream_modulation_img.linear.weight", "double_stream_modulation_img.lin.weight"),
        ("double_stream_modulation_img.linear.bias", "double_stream_modulation_img.lin.bias"),
        ("double_stream_modulation_txt.linear.weight", "double_stream_modulation_txt.lin.weight"),
        ("double_stream_modulation_txt.linear.bias", "double_stream_modulation_txt.lin.bias"),
        ("single_stream_modulation.linear.weight", "single_stream_modulation.lin.weight"),
        ("single_stream_modulation.linear.bias", "single_stream_modulation.lin.bias"),
        ("proj_out.weight", "final_layer.linear.weight"),
        ("proj_out.bias", "final_layer.linear.bias"),
    )
    for canonical_key, source_key_name in direct_pairs:
        _register_direct(canonical_key, source_key_name)

    _register_swap("norm_out.linear.weight", "final_layer.adaLN_modulation.1.weight")

    double_indices = _extract_indices(source_keys, "double_blocks.")
    for index in double_indices:
        runtime = f"double_blocks.{index}."
        native = f"transformer_blocks.{index}."
        _register_direct(f"{native}attn.norm_q.weight", runtime + "img_attn.norm.query_norm.scale")
        _register_direct(f"{native}attn.norm_k.weight", runtime + "img_attn.norm.key_norm.scale")
        _register_direct(f"{native}attn.norm_added_q.weight", runtime + "txt_attn.norm.query_norm.scale")
        _register_direct(f"{native}attn.norm_added_k.weight", runtime + "txt_attn.norm.key_norm.scale")
        _register_split(f"{native}attn.to_q.weight", runtime + "img_attn.qkv.weight", chunks=3, index=0)
        _register_split(f"{native}attn.to_k.weight", runtime + "img_attn.qkv.weight", chunks=3, index=1)
        _register_split(f"{native}attn.to_v.weight", runtime + "img_attn.qkv.weight", chunks=3, index=2)
        _register_split(f"{native}attn.to_q.bias", runtime + "img_attn.qkv.bias", chunks=3, index=0)
        _register_split(f"{native}attn.to_k.bias", runtime + "img_attn.qkv.bias", chunks=3, index=1)
        _register_split(f"{native}attn.to_v.bias", runtime + "img_attn.qkv.bias", chunks=3, index=2)
        _register_split(f"{native}attn.add_q_proj.weight", runtime + "txt_attn.qkv.weight", chunks=3, index=0)
        _register_split(f"{native}attn.add_k_proj.weight", runtime + "txt_attn.qkv.weight", chunks=3, index=1)
        _register_split(f"{native}attn.add_v_proj.weight", runtime + "txt_attn.qkv.weight", chunks=3, index=2)
        _register_split(f"{native}attn.add_q_proj.bias", runtime + "txt_attn.qkv.bias", chunks=3, index=0)
        _register_split(f"{native}attn.add_k_proj.bias", runtime + "txt_attn.qkv.bias", chunks=3, index=1)
        _register_split(f"{native}attn.add_v_proj.bias", runtime + "txt_attn.qkv.bias", chunks=3, index=2)
        _register_direct(f"{native}attn.to_out.0.weight", runtime + "img_attn.proj.weight")
        _register_direct(f"{native}attn.to_out.0.bias", runtime + "img_attn.proj.bias")
        _register_direct(f"{native}attn.to_add_out.weight", runtime + "txt_attn.proj.weight")
        _register_direct(f"{native}attn.to_add_out.bias", runtime + "txt_attn.proj.bias")
        _register_direct(f"{native}ff.linear_in.weight", runtime + "img_mlp.0.weight")
        _register_direct(f"{native}ff.linear_in.bias", runtime + "img_mlp.0.bias")
        _register_direct(f"{native}ff.linear_out.weight", runtime + "img_mlp.2.weight")
        _register_direct(f"{native}ff.linear_out.bias", runtime + "img_mlp.2.bias")
        _register_direct(f"{native}ff_context.linear_in.weight", runtime + "txt_mlp.0.weight")
        _register_direct(f"{native}ff_context.linear_in.bias", runtime + "txt_mlp.0.bias")
        _register_direct(f"{native}ff_context.linear_out.weight", runtime + "txt_mlp.2.weight")
        _register_direct(f"{native}ff_context.linear_out.bias", runtime + "txt_mlp.2.bias")

    single_indices = _extract_indices(source_keys, "single_blocks.")
    for index in single_indices:
        runtime = f"single_blocks.{index}."
        native = f"single_transformer_blocks.{index}."
        _register_direct(f"{native}attn.to_qkv_mlp_proj.weight", runtime + "linear1.weight")
        _register_direct(f"{native}attn.to_qkv_mlp_proj.bias", runtime + "linear1.bias")
        _register_direct(f"{native}attn.to_out.weight", runtime + "linear2.weight")
        _register_direct(f"{native}attn.to_out.bias", runtime + "linear2.bias")
        _register_direct(f"{native}attn.norm_q.weight", runtime + "norm.query_norm.scale")
        _register_direct(f"{native}attn.norm_k.weight", runtime + "norm.key_norm.scale")

    leftovers = sorted(set(source_keys).difference(consumed))
    if leftovers:
        raise KeyMappingError(
            "flux2_transformer_runtime_key_style: unsupported runtime-export FLUX.2 tensors remain after resolution. "
            f"leftovers_sample={leftovers[:12]}"
        )

    return ResolvedKeyspace(
        style=style,
        canonical_to_source=dict(canonical_to_source),
        metadata={
            "resolver": "flux2_transformer",
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


__all__ = ["resolve_flux2_transformer_keyspace"]
