"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux transformer GGUF/runtime key-style resolver.
Resolves Flux core checkpoints between runtime-export keys (`double_blocks.*`, `single_blocks.*`) and native/source Diffusers keys
(`transformer_blocks.*`, `single_transformer_blocks.*`) into the canonical runtime lookup keyspace used by the Codex Flux runtime,
without mutating or materializing a remapped state dict; wrapper/prefix rewrite attempts fail loud.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_flux_transformer_keyspace` (function): Resolves runtime-export or native/source Flux transformer keys into canonical runtime lookup keys.
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
    _concat_tensor_rows,
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
    "text_encoder_2.",
    "text_encoders.",
    "vae.",
)

_DETECTOR = KeyStyleDetector(
    name="flux_transformer_runtime_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.EXACT, "img_in.weight"),
                KeySentinel(SentinelKind.EXACT, "double_blocks.0.img_attn.qkv.weight"),
                KeySentinel(SentinelKind.EXACT, "single_blocks.0.linear1.weight"),
                KeySentinel(SentinelKind.EXACT, "final_layer.linear.weight"),
            ),
            min_sentinel_hits=2,
        ),
        KeyStyleSpec(
            style=KeyStyle.DIFFUSERS,
            sentinels=(
                KeySentinel(SentinelKind.EXACT, "x_embedder.weight"),
                KeySentinel(SentinelKind.EXACT, "context_embedder.weight"),
                KeySentinel(SentinelKind.PREFIX, "transformer_blocks.0."),
                KeySentinel(SentinelKind.PREFIX, "single_transformer_blocks.0."),
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
                "flux_transformer_runtime_key_style: source-key collision "
                f"for key={validated_source_key!r} srcs={previous!r},{source_key!r}"
            )
        out[validated_source_key] = source_key
    if not out:
        raise KeyMappingError("flux_transformer_runtime_key_style: no tensor keys remained after source-key validation")
    return out


def _reject_unsupported_prefixes(keys: Sequence[str]) -> None:
    offenders = [key for key in keys if key.startswith(_UNSUPPORTED_PREFIXES)]
    if offenders:
        raise KeyMappingError(
            "flux_transformer_runtime_key_style: Flux GGUF core checkpoints must not embed text encoder/VAE keys. "
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
            "flux_transformer_runtime_key_style: source tensor is missing shape metadata "
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
            "flux_transformer_runtime_key_style: concat_dim0 expects rank-1 or rank-2 tensors. "
            f"destination={destination_key!r} shape={base!r}"
        )
    trailing = base[1:]
    total_rows = 0
    for source_key, shape in zip(source_keys, shapes, strict=True):
        if len(shape) != len(base) or shape[1:] != trailing:
            raise KeyMappingError(
                "flux_transformer_runtime_key_style: concat_dim0 source shape mismatch. "
                f"destination={destination_key!r} source={source_key!r} shape={shape!r} expected_trailing={trailing!r}"
            )
        total_rows += int(shape[0])
    return (total_rows, *trailing)


def _validate_swap_shape(
    state_dict: MutableMapping[str, _T],
    *,
    source_key: str,
    destination_key: str,
) -> tuple[int, ...]:
    shape = _shape_of(state_dict, source_key)
    if len(shape) not in (1, 2):
        raise KeyMappingError(
            "flux_transformer_runtime_key_style: swap_dim0_halves expects rank-1 or rank-2 tensors. "
            f"destination={destination_key!r} shape={shape!r}"
        )
    if shape[0] % 2 != 0:
        raise KeyMappingError(
            "flux_transformer_runtime_key_style: swap_dim0_halves expects an even leading dimension. "
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


def resolve_flux_transformer_keyspace(state_dict: MutableMapping[str, _T]) -> ResolvedKeyspace[_T]:
    source_keys_to_source = _source_keys_to_source(state_dict)
    source_keys = tuple(source_keys_to_source.keys())
    _reject_unsupported_prefixes(source_keys)
    style = _DETECTOR.detect(source_keys)

    if style is KeyStyle.CODEX:
        return ResolvedKeyspace(
            style=style,
            canonical_to_source=dict(source_keys_to_source),
            metadata={
                "resolver": "flux_transformer",
                "source_style": style.value,
            },
            view=KeyspaceLookupView(state_dict, source_keys_to_source),
        )

    canonical_to_source: dict[str, str] = {}
    computed_shapes: dict[str, tuple[int, ...]] = {}
    computed_sources: dict[str, str] = {}
    computed: dict[str, Any] = {}
    consumed: set[str] = set()

    def _register_direct(canonical_key: str, source_key_name: str, *, required: bool = True) -> None:
        source_key = source_keys_to_source.get(source_key_name)
        if source_key is None:
            if required:
                raise KeyMappingError(
                    "flux_transformer_runtime_key_style: missing native/source key for runtime lookup. "
                    f"dst={canonical_key!r} src={source_key_name!r}"
                )
            return
        previous = canonical_to_source.get(canonical_key)
        if previous is not None and previous != source_key:
            raise KeyMappingError(
                "flux_transformer_runtime_key_style: multiple source keys map to the same runtime key. "
                f"dst={canonical_key!r} srcs={previous!r},{source_key!r}"
        )
        canonical_to_source[canonical_key] = source_key
        consumed.add(source_key_name)

    def _register_concat(canonical_key: str, source_key_names: tuple[str, ...]) -> None:
        missing = [key for key in source_key_names if key not in source_keys_to_source]
        if missing:
            raise KeyMappingError(
                "flux_transformer_runtime_key_style: missing native/source tensors for fused runtime key. "
                f"dst={canonical_key!r} missing={missing!r}"
            )
        source_keys = tuple(source_keys_to_source[key] for key in source_key_names)
        logical_shape = _validate_concat_shapes(state_dict, source_keys=source_keys, destination_key=canonical_key)
        computed_shapes[canonical_key] = logical_shape
        computed_sources[canonical_key] = f"concat_dim0({', '.join(source_keys)})"
        computed[canonical_key] = lambda source_keys=source_keys, logical_shape=logical_shape: _concat_tensor_rows(
            tuple(state_dict[source_key] for source_key in source_keys),
            logical_shape=logical_shape,
        )
        consumed.update(source_key_names)

    def _register_swap(canonical_key: str, source_key_name: str) -> None:
        source_key = source_keys_to_source.get(source_key_name)
        if source_key is None:
            raise KeyMappingError(
                "flux_transformer_runtime_key_style: missing native/source tensor for swapped runtime key. "
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
        ("img_in.weight", "x_embedder.weight", True),
        ("img_in.bias", "x_embedder.bias", True),
        ("txt_in.weight", "context_embedder.weight", True),
        ("txt_in.bias", "context_embedder.bias", True),
        ("time_in.in_layer.weight", "time_text_embed.timestep_embedder.linear_1.weight", True),
        ("time_in.in_layer.bias", "time_text_embed.timestep_embedder.linear_1.bias", True),
        ("time_in.out_layer.weight", "time_text_embed.timestep_embedder.linear_2.weight", True),
        ("time_in.out_layer.bias", "time_text_embed.timestep_embedder.linear_2.bias", True),
        ("vector_in.in_layer.weight", "time_text_embed.text_embedder.linear_1.weight", True),
        ("vector_in.in_layer.bias", "time_text_embed.text_embedder.linear_1.bias", True),
        ("vector_in.out_layer.weight", "time_text_embed.text_embedder.linear_2.weight", True),
        ("vector_in.out_layer.bias", "time_text_embed.text_embedder.linear_2.bias", True),
        ("guidance_in.in_layer.weight", "time_text_embed.guidance_embedder.linear_1.weight", False),
        ("guidance_in.in_layer.bias", "time_text_embed.guidance_embedder.linear_1.bias", False),
        ("guidance_in.out_layer.weight", "time_text_embed.guidance_embedder.linear_2.weight", False),
        ("guidance_in.out_layer.bias", "time_text_embed.guidance_embedder.linear_2.bias", False),
        ("final_layer.linear.weight", "proj_out.weight", True),
        ("final_layer.linear.bias", "proj_out.bias", True),
    )
    for canonical_key, source_key_name, required in direct_pairs:
        _register_direct(canonical_key, source_key_name, required=required)

    _register_swap("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight")
    _register_swap("final_layer.adaLN_modulation.1.bias", "norm_out.linear.bias")

    double_indices = _extract_indices(source_keys, "transformer_blocks.")
    for index in double_indices:
        base = f"transformer_blocks.{index}."
        runtime = f"double_blocks.{index}."
        _register_direct(f"{runtime}img_attn.norm.query_norm.scale", base + "attn.norm_q.weight")
        _register_direct(f"{runtime}img_attn.norm.key_norm.scale", base + "attn.norm_k.weight")
        _register_direct(f"{runtime}txt_attn.norm.query_norm.scale", base + "attn.norm_added_q.weight")
        _register_direct(f"{runtime}txt_attn.norm.key_norm.scale", base + "attn.norm_added_k.weight")
        _register_concat(
            f"{runtime}img_attn.qkv.weight",
            (
                base + "attn.to_q.weight",
                base + "attn.to_k.weight",
                base + "attn.to_v.weight",
            ),
        )
        _register_concat(
            f"{runtime}img_attn.qkv.bias",
            (
                base + "attn.to_q.bias",
                base + "attn.to_k.bias",
                base + "attn.to_v.bias",
            ),
        )
        _register_concat(
            f"{runtime}txt_attn.qkv.weight",
            (
                base + "attn.add_q_proj.weight",
                base + "attn.add_k_proj.weight",
                base + "attn.add_v_proj.weight",
            ),
        )
        _register_concat(
            f"{runtime}txt_attn.qkv.bias",
            (
                base + "attn.add_q_proj.bias",
                base + "attn.add_k_proj.bias",
                base + "attn.add_v_proj.bias",
            ),
        )
        _register_direct(f"{runtime}img_attn.proj.weight", base + "attn.to_out.0.weight")
        _register_direct(f"{runtime}img_attn.proj.bias", base + "attn.to_out.0.bias")
        _register_direct(f"{runtime}txt_attn.proj.weight", base + "attn.to_add_out.weight")
        _register_direct(f"{runtime}txt_attn.proj.bias", base + "attn.to_add_out.bias")
        _register_direct(f"{runtime}img_mlp.0.weight", base + "ff.net.0.proj.weight")
        _register_direct(f"{runtime}img_mlp.0.bias", base + "ff.net.0.proj.bias")
        _register_direct(f"{runtime}img_mlp.2.weight", base + "ff.net.2.weight")
        _register_direct(f"{runtime}img_mlp.2.bias", base + "ff.net.2.bias")
        _register_direct(f"{runtime}txt_mlp.0.weight", base + "ff_context.net.0.proj.weight")
        _register_direct(f"{runtime}txt_mlp.0.bias", base + "ff_context.net.0.proj.bias")
        _register_direct(f"{runtime}txt_mlp.2.weight", base + "ff_context.net.2.weight")
        _register_direct(f"{runtime}txt_mlp.2.bias", base + "ff_context.net.2.bias")
        _register_direct(f"{runtime}img_mod.lin.weight", base + "norm1.linear.weight")
        _register_direct(f"{runtime}img_mod.lin.bias", base + "norm1.linear.bias")
        _register_direct(f"{runtime}txt_mod.lin.weight", base + "norm1_context.linear.weight")
        _register_direct(f"{runtime}txt_mod.lin.bias", base + "norm1_context.linear.bias")

    single_indices = _extract_indices(source_keys, "single_transformer_blocks.")
    for index in single_indices:
        base = f"single_transformer_blocks.{index}."
        runtime = f"single_blocks.{index}."
        _register_direct(f"{runtime}norm.query_norm.scale", base + "attn.norm_q.weight")
        _register_direct(f"{runtime}norm.key_norm.scale", base + "attn.norm_k.weight")
        _register_concat(
            f"{runtime}linear1.weight",
            (
                base + "attn.to_q.weight",
                base + "attn.to_k.weight",
                base + "attn.to_v.weight",
                base + "proj_mlp.weight",
            ),
        )
        _register_concat(
            f"{runtime}linear1.bias",
            (
                base + "attn.to_q.bias",
                base + "attn.to_k.bias",
                base + "attn.to_v.bias",
                base + "proj_mlp.bias",
            ),
        )
        _register_direct(f"{runtime}linear2.weight", base + "proj_out.weight")
        _register_direct(f"{runtime}linear2.bias", base + "proj_out.bias")
        _register_direct(f"{runtime}modulation.lin.weight", base + "norm.linear.weight")
        _register_direct(f"{runtime}modulation.lin.bias", base + "norm.linear.bias")

    leftovers = sorted(set(source_keys).difference(consumed))
    if leftovers:
        raise KeyMappingError(
            "flux_transformer_runtime_key_style: unsupported native/source Flux transformer tensors remain after resolution. "
            f"leftovers_sample={leftovers[:12]}"
        )

    return ResolvedKeyspace(
        style=style,
        canonical_to_source=dict(canonical_to_source),
        metadata={
            "resolver": "flux_transformer",
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


__all__ = ["resolve_flux_transformer_keyspace"]
