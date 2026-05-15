"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 VAE key-style detection + strict canonical keyspace resolvers for 2D/3D lanes.
Owns WAN22 VAE source-key validation, lane detection, and canonical keyspace resolution without stored-key rewrites, keeping resolver logic
out of router/payload seams. Supports 2D native LDM VAE keys and 3D Codex/Diffusers keyspaces with fail-loud mixed-style/collision guards.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_wan22_vae_keyspace` (function): Detects the WAN22 VAE lane from validated raw source keys and resolves the canonical lookup view.
- `resolve_wan22_vae_2d_keyspace` (function): Validates WAN22 2D VAE source keys and resolves canonical lookup ownership.
- `resolve_wan22_vae_3d_keyspace` (function): Validates WAN22 3D VAE source keys and resolves canonical codex keyspace (`diffusers|codex` → codex).
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import Any

from apps.backend.runtime.state_dict.key_mapping import KeyMappingError, ResolvedKeyspace, fail_on_key_name_rewrite
from apps.backend.runtime.state_dict.keymap_wan21_vae import resolve_wan21_vae_keyspace
from apps.backend.runtime.state_dict.views import KeyspaceLookupView

_WRAPPER_PREFIXES = (
    "module.",
    "vae.",
    "first_stage_model.",
)

_WAN22_2D_REQUIRED_KEYS = (
    "encoder.conv_in.weight",
    "decoder.conv_in.weight",
)
_WAN22_2D_OPTIONAL_QUANT_KEYS = (
    "quant_conv.weight",
    "post_quant_conv.weight",
)

_FIXED_DIFFUSERS_TO_CODEX_KEYS: dict[str, str] = {
    "encoder.conv_in.weight": "encoder.conv1.weight",
    "encoder.conv_in.bias": "encoder.conv1.bias",
    "decoder.conv_in.weight": "decoder.conv1.weight",
    "decoder.conv_in.bias": "decoder.conv1.bias",
    "encoder.mid_block.resnets.0.norm1.gamma": "encoder.middle.0.residual.0.gamma",
    "encoder.mid_block.resnets.0.conv1.weight": "encoder.middle.0.residual.2.weight",
    "encoder.mid_block.resnets.0.conv1.bias": "encoder.middle.0.residual.2.bias",
    "encoder.mid_block.resnets.0.norm2.gamma": "encoder.middle.0.residual.3.gamma",
    "encoder.mid_block.resnets.0.conv2.weight": "encoder.middle.0.residual.6.weight",
    "encoder.mid_block.resnets.0.conv2.bias": "encoder.middle.0.residual.6.bias",
    "encoder.mid_block.resnets.1.norm1.gamma": "encoder.middle.2.residual.0.gamma",
    "encoder.mid_block.resnets.1.conv1.weight": "encoder.middle.2.residual.2.weight",
    "encoder.mid_block.resnets.1.conv1.bias": "encoder.middle.2.residual.2.bias",
    "encoder.mid_block.resnets.1.norm2.gamma": "encoder.middle.2.residual.3.gamma",
    "encoder.mid_block.resnets.1.conv2.weight": "encoder.middle.2.residual.6.weight",
    "encoder.mid_block.resnets.1.conv2.bias": "encoder.middle.2.residual.6.bias",
    "decoder.mid_block.resnets.0.norm1.gamma": "decoder.middle.0.residual.0.gamma",
    "decoder.mid_block.resnets.0.conv1.weight": "decoder.middle.0.residual.2.weight",
    "decoder.mid_block.resnets.0.conv1.bias": "decoder.middle.0.residual.2.bias",
    "decoder.mid_block.resnets.0.norm2.gamma": "decoder.middle.0.residual.3.gamma",
    "decoder.mid_block.resnets.0.conv2.weight": "decoder.middle.0.residual.6.weight",
    "decoder.mid_block.resnets.0.conv2.bias": "decoder.middle.0.residual.6.bias",
    "decoder.mid_block.resnets.1.norm1.gamma": "decoder.middle.2.residual.0.gamma",
    "decoder.mid_block.resnets.1.conv1.weight": "decoder.middle.2.residual.2.weight",
    "decoder.mid_block.resnets.1.conv1.bias": "decoder.middle.2.residual.2.bias",
    "decoder.mid_block.resnets.1.norm2.gamma": "decoder.middle.2.residual.3.gamma",
    "decoder.mid_block.resnets.1.conv2.weight": "decoder.middle.2.residual.6.weight",
    "decoder.mid_block.resnets.1.conv2.bias": "decoder.middle.2.residual.6.bias",
    "encoder.mid_block.attentions.0.norm.gamma": "encoder.middle.1.norm.gamma",
    "encoder.mid_block.attentions.0.to_qkv.weight": "encoder.middle.1.to_qkv.weight",
    "encoder.mid_block.attentions.0.to_qkv.bias": "encoder.middle.1.to_qkv.bias",
    "encoder.mid_block.attentions.0.proj.weight": "encoder.middle.1.proj.weight",
    "encoder.mid_block.attentions.0.proj.bias": "encoder.middle.1.proj.bias",
    "decoder.mid_block.attentions.0.norm.gamma": "decoder.middle.1.norm.gamma",
    "decoder.mid_block.attentions.0.to_qkv.weight": "decoder.middle.1.to_qkv.weight",
    "decoder.mid_block.attentions.0.to_qkv.bias": "decoder.middle.1.to_qkv.bias",
    "decoder.mid_block.attentions.0.proj.weight": "decoder.middle.1.proj.weight",
    "decoder.mid_block.attentions.0.proj.bias": "decoder.middle.1.proj.bias",
    "encoder.norm_out.gamma": "encoder.head.0.gamma",
    "encoder.conv_out.weight": "encoder.head.2.weight",
    "encoder.conv_out.bias": "encoder.head.2.bias",
    "decoder.norm_out.gamma": "decoder.head.0.gamma",
    "decoder.conv_out.weight": "decoder.head.2.weight",
    "decoder.conv_out.bias": "decoder.head.2.bias",
    "quant_conv.weight": "conv1.weight",
    "quant_conv.bias": "conv1.bias",
    "post_quant_conv.weight": "conv2.weight",
    "post_quant_conv.bias": "conv2.bias",
}

_DECODER_RESNET_TO_UPSAMPLE_INDEX: dict[tuple[int, int], int] = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (1, 0): 4,
    (1, 1): 5,
    (1, 2): 6,
    (2, 0): 8,
    (2, 1): 9,
    (2, 2): 10,
    (3, 0): 12,
    (3, 1): 13,
    (3, 2): 14,
}

_DECODER_UPSAMPLER_TO_UPSAMPLE_INDEX: dict[int, int] = {
    0: 3,
    1: 7,
    2: 11,
}


def _source_keys_to_source(state_dict: MutableMapping[str, Any], *, detector_name: str) -> dict[str, str]:
    source_key_map: dict[str, str] = {}
    for key in state_dict.keys():
        source_key = str(key)
        validated_source_key = fail_on_key_name_rewrite(source_key, _WRAPPER_PREFIXES)
        if validated_source_key in source_key_map:
            raise KeyMappingError(
                f"{detector_name}: source-key collision for key={validated_source_key!r}."
            )
        source_key_map[validated_source_key] = source_key
    return source_key_map


def _shape_of(state_dict: MutableMapping[str, Any], source_key: str) -> tuple[int, ...] | None:
    shape_getter = getattr(state_dict, "shape_of", None)
    if callable(shape_getter):
        shape = shape_getter(source_key)
        if shape is not None:
            try:
                return tuple(int(dim) for dim in shape)
            except Exception:
                return None
    value = state_dict[source_key]
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except Exception:
        return None


def _lane_from_shape(*, key: str, shape: tuple[int, ...]) -> str:
    ndim = len(shape)
    if ndim == 4:
        return "2d_native"
    if ndim == 5:
        return "3d_native"
    raise KeyMappingError(
        "wan22_vae_key_style: unsupported core VAE kernel rank "
        f"key={key!r} shape={shape} (expected rank 4 or 5)."
    )


def _detect_wan22_vae_lane_from_source_keys(
    state_dict: MutableMapping[str, Any],
    *,
    source_key_map: dict[str, str],
) -> str:
    evidence_keys = (
        "encoder.conv_in.weight",
        "decoder.conv_in.weight",
        "encoder.conv1.weight",
        "decoder.conv1.weight",
    )
    observed: set[str] = set()
    seen: list[tuple[str, tuple[int, ...]]] = []

    for key in evidence_keys:
        source_key = source_key_map.get(key)
        if source_key is None:
            continue
        shape = _shape_of(state_dict, source_key)
        if shape is None:
            continue
        seen.append((key, shape))
        observed.add(_lane_from_shape(key=key, shape=shape))

    if not seen:
        fallback_seen: list[tuple[str, tuple[int, ...]]] = []
        for key, source_key in source_key_map.items():
            if not key.endswith(".weight"):
                continue
            if not (
                key.startswith("encoder.conv")
                or key.startswith("decoder.conv")
                or key.startswith("encoder.head.2")
                or key.startswith("decoder.head.2")
            ):
                continue
            shape = _shape_of(state_dict, source_key)
            if shape is None:
                continue
            fallback_seen.append((key, shape))
            observed.add(_lane_from_shape(key=key, shape=shape))
        if not fallback_seen:
            raise KeyMappingError(
                "wan22_vae_key_style: cannot detect lane (missing canonical core convolution weights)."
            )
        seen = fallback_seen

    if len(observed) != 1:
        raise KeyMappingError(
            "wan22_vae_key_style: mixed VAE lane evidence in core kernels "
            f"(lanes={sorted(observed)} evidence={seen[:8]})."
        )
    return next(iter(observed))


def _map_encoder_down_block_key(key: str) -> str:
    if ".resnets." in key:
        raise KeyMappingError(
            "wan22_vae_3d_key_style: unsupported encoder down_block layout "
            f"for key={key!r} (expected canonical encoder.down_blocks.<idx>.<field>)."
        )
    mapped = key.replace("encoder.down_blocks.", "encoder.downsamples.", 1)
    mapped = mapped.replace(".norm1.gamma", ".residual.0.gamma")
    mapped = mapped.replace(".conv1.weight", ".residual.2.weight")
    mapped = mapped.replace(".conv1.bias", ".residual.2.bias")
    mapped = mapped.replace(".norm2.gamma", ".residual.3.gamma")
    mapped = mapped.replace(".conv2.weight", ".residual.6.weight")
    mapped = mapped.replace(".conv2.bias", ".residual.6.bias")
    mapped = mapped.replace(".conv_shortcut.weight", ".shortcut.weight")
    mapped = mapped.replace(".conv_shortcut.bias", ".shortcut.bias")
    return mapped


def _map_decoder_up_block_key(key: str) -> str:
    resnet_match = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if resnet_match is not None:
        block_index = int(resnet_match.group(1))
        resnet_index = int(resnet_match.group(2))
        tail = str(resnet_match.group(3))
        if (block_index, resnet_index) not in _DECODER_RESNET_TO_UPSAMPLE_INDEX:
            raise KeyMappingError(
                "wan22_vae_3d_key_style: unsupported decoder up_block residual index "
                f"(block={block_index} resnet={resnet_index}) for key={key!r}."
            )
        upsample_index = _DECODER_RESNET_TO_UPSAMPLE_INDEX[(block_index, resnet_index)]
        if tail == "norm1.gamma":
            mapped_tail = "residual.0.gamma"
        elif tail == "conv1.weight":
            mapped_tail = "residual.2.weight"
        elif tail == "conv1.bias":
            mapped_tail = "residual.2.bias"
        elif tail == "norm2.gamma":
            mapped_tail = "residual.3.gamma"
        elif tail == "conv2.weight":
            mapped_tail = "residual.6.weight"
        elif tail == "conv2.bias":
            mapped_tail = "residual.6.bias"
        elif tail.startswith("conv_shortcut."):
            mapped_tail = "shortcut." + tail[len("conv_shortcut.") :]
        else:
            raise KeyMappingError(
                "wan22_vae_3d_key_style: unsupported decoder up_block residual field "
                f"tail={tail!r} key={key!r}."
            )
        return f"decoder.upsamples.{upsample_index}.{mapped_tail}"

    upsample_match = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.+)$", key)
    if upsample_match is not None:
        block_index = int(upsample_match.group(1))
        tail = str(upsample_match.group(2))
        if block_index not in _DECODER_UPSAMPLER_TO_UPSAMPLE_INDEX:
            raise KeyMappingError(
                "wan22_vae_3d_key_style: unsupported decoder up_block upsampler index "
                f"(block={block_index}) for key={key!r}."
            )
        upsample_index = _DECODER_UPSAMPLER_TO_UPSAMPLE_INDEX[block_index]
        return f"decoder.upsamples.{upsample_index}.{tail}"

    raise KeyMappingError(
        "wan22_vae_3d_key_style: unsupported decoder up_block key layout "
        f"for key={key!r}."
    )


def _resolve_wan22_vae_2d_keyspace_from_source_keys(
    state_dict: MutableMapping[str, Any],
    *,
    source_key_map: dict[str, str],
) -> ResolvedKeyspace[Any]:
    keys = tuple(source_key_map.keys())
    keys_set = frozenset(keys)

    missing_required = [key for key in _WAN22_2D_REQUIRED_KEYS if key not in keys_set]
    if missing_required:
        raise KeyMappingError(
            "wan22_vae_2d_key_style: resolver output is missing required canonical keys. "
            f"missing_sample={missing_required[:10]}"
        )
    if not any(key in keys_set for key in _WAN22_2D_OPTIONAL_QUANT_KEYS):
        raise KeyMappingError(
            "wan22_vae_2d_key_style: resolver output is missing quantization convolution keys "
            f"(requires one of {sorted(_WAN22_2D_OPTIONAL_QUANT_KEYS)})."
        )
    if any(key.startswith("encoder.downsamples.") or key.startswith("decoder.upsamples.") for key in keys):
        raise KeyMappingError(
            "wan22_vae_2d_key_style: received 3d codex keyspace in 2d lane (encoder.downsamples/decoder.upsamples)."
        )
    if any(key.startswith("encoder.down_blocks.") or key.startswith("decoder.up_blocks.") for key in keys):
        raise KeyMappingError(
            "wan22_vae_2d_key_style: received 3d diffusers keyspace in 2d lane (encoder.down_blocks/decoder.up_blocks)."
        )

    return ResolvedKeyspace(
        style="ldm_2d",
        canonical_to_source=dict(source_key_map),
        metadata={
            "resolver": "wan22_vae_2d",
            "lane": "2d_native",
            "source_style": "ldm_2d",
        },
        view=KeyspaceLookupView(state_dict, source_key_map),
    )


def resolve_wan22_vae_2d_keyspace(state_dict: MutableMapping[str, Any]) -> ResolvedKeyspace[Any]:
    source_key_map = _source_keys_to_source(state_dict, detector_name="wan22_vae_2d_key_style")
    return _resolve_wan22_vae_2d_keyspace_from_source_keys(state_dict, source_key_map=source_key_map)


def _resolve_wan22_vae_3d_keyspace_from_source_keys(
    state_dict: MutableMapping[str, Any],
    *,
    source_key_map: dict[str, str],
) -> ResolvedKeyspace[Any]:
    source_key_view = KeyspaceLookupView(state_dict, source_key_map)

    has_codex = any(
        key.startswith("encoder.downsamples.")
        or key.startswith("decoder.upsamples.")
        or key in {"conv1.weight", "conv2.weight"}
        for key in source_key_map.keys()
    )
    has_diffusers = any(
        key.startswith("encoder.down_blocks.")
        or key.startswith("decoder.up_blocks.")
        or key.startswith("quant_conv.")
        or key.startswith("post_quant_conv.")
        for key in source_key_map.keys()
    )

    if has_codex and has_diffusers:
        raise KeyMappingError(
            "wan22_vae_3d_key_style: mixed codex/diffusers VAE keyspace detected "
            "(cannot resolve a single canonical lane)."
        )

    if has_diffusers:
        mapped_to_source: dict[str, str] = {}
        for key in source_key_map.keys():
            if key in _FIXED_DIFFUSERS_TO_CODEX_KEYS:
                mapped_key = _FIXED_DIFFUSERS_TO_CODEX_KEYS[key]
            elif key.startswith("encoder.down_blocks."):
                mapped_key = _map_encoder_down_block_key(key)
            elif key.startswith("decoder.up_blocks."):
                mapped_key = _map_decoder_up_block_key(key)
            else:
                mapped_key = key
            if mapped_key in mapped_to_source:
                raise KeyMappingError(
                    "wan22_vae_3d_key_style: resolver produced output collision "
                    f"for mapped key={mapped_key!r} (source key={key!r})."
                )
            mapped_to_source[mapped_key] = source_key_map[key]

        validated = resolve_wan21_vae_keyspace(KeyspaceLookupView(state_dict, mapped_to_source))
        canonical_to_source = {
            canonical: mapped_to_source.get(source, source)
            for canonical, source in validated.canonical_to_source.items()
        }
        return ResolvedKeyspace(
            style="diffusers",
            canonical_to_source=canonical_to_source,
            metadata={
                "resolver": "wan22_vae_3d",
                "lane": "3d_native",
                "source_style": "diffusers",
                "validator_style": (
                    validated.style.value if hasattr(validated.style, "value") else str(validated.style)
                ),
            },
            view=KeyspaceLookupView(state_dict, canonical_to_source),
        )

    validated = resolve_wan21_vae_keyspace(source_key_view)
    canonical_to_source = {
        canonical: source_key_map.get(source, source)
        for canonical, source in validated.canonical_to_source.items()
    }
    return ResolvedKeyspace(
        style="codex",
        canonical_to_source=canonical_to_source,
        metadata={
            "resolver": "wan22_vae_3d",
            "lane": "3d_native",
            "source_style": "codex",
            "validator_style": (
                validated.style.value if hasattr(validated.style, "value") else str(validated.style)
            ),
        },
        view=KeyspaceLookupView(state_dict, canonical_to_source),
    )


def resolve_wan22_vae_3d_keyspace(state_dict: MutableMapping[str, Any]) -> ResolvedKeyspace[Any]:
    source_key_map = _source_keys_to_source(state_dict, detector_name="wan22_vae_3d_key_style")
    return _resolve_wan22_vae_3d_keyspace_from_source_keys(state_dict, source_key_map=source_key_map)


def resolve_wan22_vae_keyspace(state_dict: MutableMapping[str, Any]) -> ResolvedKeyspace[Any]:
    source_key_map = _source_keys_to_source(state_dict, detector_name="wan22_vae_key_style")
    lane = _detect_wan22_vae_lane_from_source_keys(state_dict, source_key_map=source_key_map)
    if lane == "2d_native":
        return _resolve_wan22_vae_2d_keyspace_from_source_keys(state_dict, source_key_map=source_key_map)
    if lane == "3d_native":
        return _resolve_wan22_vae_3d_keyspace_from_source_keys(state_dict, source_key_map=source_key_map)
    raise KeyMappingError(f"wan22_vae_key_style: unsupported lane={lane!r}.")


__all__ = [
    "resolve_wan22_vae_keyspace",
    "resolve_wan22_vae_2d_keyspace",
    "resolve_wan22_vae_3d_keyspace",
]
