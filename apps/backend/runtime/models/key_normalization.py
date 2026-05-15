"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: State-dict keyspace resolution helpers for loader assembly.
Provides UNet diffusers→LDM keyspace lookup assembly and shared prefix stripping for transformer state dicts.

Symbols (top-level; keep in sync; no ghosts):
- `_strip_unet_prefixes_mapping` (function): Builds a UNet key lookup by stripping known prefixes in a checkpoint mapping.
- `_normalize_depth_list` (function): Normalizes “depth list” inputs (pad/trim) to a fixed length used by model configs.
- `_build_diffusers_to_ldm_map` (function): Builds a diffusers→LDM key mapping for UNet state dict conversion based on config.
- `_normalize_unet_state_dict` (function): Resolves a UNet keyspace lookup view for the expected internal layout.
- `_strip_transformer_prefixes` (function): Strips common wrapper prefixes from transformer state dict keys.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Any, Dict, Mapping

from apps.backend.runtime.state_dict.views import KeyspaceLookupView

UNET_LOG = get_backend_logger("apps.backend.runtime.models.loader.unet")

_UNET_PREFIXES: tuple[str, ...] = (
    "model.diffusion_model.",
    "model.model.",
    "diffusion_model.",
    "model.",
)

_UNET_MAP_ATTENTIONS: tuple[str, ...] = (
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
)

_TRANSFORMER_BLOCK_FIELDS: tuple[str, ...] = (
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
)

_UNET_MAP_RESNET: dict[str, str] = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}

_UNET_MAP_BASIC: tuple[tuple[str, str], ...] = (
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
)

_ESSENTIAL_UNET_KEYS: tuple[str, ...] = (
    "input_blocks.0.0.weight",
    "time_embed.0.weight",
    "out.2.weight",
)

_TRANSFORMER_PREFIXES: tuple[str, ...] = (
    "model.diffusion_model.",
    "model.model.",
    "diffusion_model.",
    "model.",
)


def _strip_unet_prefixes_mapping(sd: Mapping[str, Any]) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for key in list(sd.keys()):
        name = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in _UNET_PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    changed = True
                    break
        previous = mapping.get(name)
        if previous is not None and previous != key:
            raise RuntimeError(
                "UNet prefix stripping collision: destination key "
                f"{name!r} maps to multiple source keys ({previous!r}, {key!r})."
            )
        mapping[name] = key
    return mapping


def _normalize_depth_list(values: Any, total: int, default: int = 0) -> list[int]:
    if values is None:
        return [default] * total
    if isinstance(values, int):
        base = [values] * total
    else:
        base = list(values)
    if len(base) < total:
        pad = base[-1] if base else default
        base.extend([pad] * (total - len(base)))
    return base[:total]


def _build_diffusers_to_ldm_map(unet_config: Mapping[str, Any]) -> Dict[str, str]:
    channel_mult = list(unet_config.get("channel_mult", []))
    if not channel_mult:
        return {}

    num_blocks = len(channel_mult)
    num_res_blocks_cfg = unet_config.get("num_res_blocks", [])
    if isinstance(num_res_blocks_cfg, int):
        num_res_blocks = [num_res_blocks_cfg] * num_blocks
    else:
        num_res_blocks = list(num_res_blocks_cfg)
        if len(num_res_blocks) < num_blocks:
            pad = num_res_blocks[-1] if num_res_blocks else 0
            num_res_blocks.extend([pad] * (num_blocks - len(num_res_blocks)))
        num_res_blocks = num_res_blocks[:num_blocks]

    total_down_transformers = sum(num_res_blocks)
    transformer_depth = _normalize_depth_list(unet_config.get("transformer_depth"), total_down_transformers, default=0)
    transformer_depth_output = _normalize_depth_list(
        unet_config.get("transformer_depth_output"),
        sum(res + 1 for res in num_res_blocks),
        default=0,
    )
    raw_mid = unet_config.get("transformer_depth_middle")
    if isinstance(raw_mid, int):
        transformers_mid = raw_mid
    elif isinstance(raw_mid, (list, tuple)) and raw_mid:
        transformers_mid = int(raw_mid[-1])
    elif transformer_depth:
        transformers_mid = transformer_depth[-1]
    else:
        transformers_mid = 0

    mapping: Dict[str, str] = {}
    for dest, src in _UNET_MAP_BASIC:
        mapping[src] = dest

    depth_iter = iter(transformer_depth)
    for block_idx in range(num_blocks):
        base_index = 1 + (num_res_blocks[block_idx] + 1) * block_idx
        for res_idx in range(num_res_blocks[block_idx]):
            for dest, src in _UNET_MAP_RESNET.items():
                mapping[f"down_blocks.{block_idx}.resnets.{res_idx}.{src}"] = f"input_blocks.{base_index}.0.{dest}"
            num_transformers = next(depth_iter, 0)
            if num_transformers > 0:
                for field in _UNET_MAP_ATTENTIONS:
                    mapping[f"down_blocks.{block_idx}.attentions.{res_idx}.{field}"] = f"input_blocks.{base_index}.1.{field}"
                for t in range(num_transformers):
                    for field in _TRANSFORMER_BLOCK_FIELDS:
                        mapping[
                            f"down_blocks.{block_idx}.attentions.{res_idx}.transformer_blocks.{t}.{field}"
                        ] = f"input_blocks.{base_index}.1.transformer_blocks.{t}.{field}"
            base_index += 1
        for suffix in ("weight", "bias"):
            mapping[f"down_blocks.{block_idx}.downsamplers.0.conv.{suffix}"] = f"input_blocks.{base_index}.0.op.{suffix}"

    # Mid block
    for idx, target in enumerate((0, 2)):
        for dest, src in _UNET_MAP_RESNET.items():
            mapping[f"mid_block.resnets.{idx}.{src}"] = f"middle_block.{target}.{dest}"
    for field in _UNET_MAP_ATTENTIONS:
        mapping[f"mid_block.attentions.0.{field}"] = f"middle_block.1.{field}"
    for t in range(max(int(transformers_mid), 0)):
        for field in _TRANSFORMER_BLOCK_FIELDS:
            mapping[f"mid_block.attentions.0.transformer_blocks.{t}.{field}"] = f"middle_block.1.transformer_blocks.{t}.{field}"

    # Up blocks (reverse order)
    up_res_counts = list(reversed(num_res_blocks))
    depth_output = list(transformer_depth_output)
    for block_idx in range(num_blocks):
        base_index = (up_res_counts[block_idx] + 1) * block_idx
        block_len = up_res_counts[block_idx] + 1
        for res_idx in range(block_len):
            stage_conv_index = 0
            for dest, src in _UNET_MAP_RESNET.items():
                mapping[f"up_blocks.{block_idx}.resnets.{res_idx}.{src}"] = f"output_blocks.{base_index}.0.{dest}"
            stage_conv_index += 1
            num_transformers = depth_output.pop() if depth_output else 0
            if num_transformers > 0:
                stage_conv_index += 1
                for field in _UNET_MAP_ATTENTIONS:
                    mapping[f"up_blocks.{block_idx}.attentions.{res_idx}.{field}"] = f"output_blocks.{base_index}.1.{field}"
                for t in range(num_transformers):
                    for field in _TRANSFORMER_BLOCK_FIELDS:
                        mapping[
                            f"up_blocks.{block_idx}.attentions.{res_idx}.transformer_blocks.{t}.{field}"
                        ] = f"output_blocks.{base_index}.1.transformer_blocks.{t}.{field}"
            if res_idx == block_len - 1:
                for suffix in ("weight", "bias"):
                    mapping[f"up_blocks.{block_idx}.upsamplers.0.conv.{suffix}"] = f"output_blocks.{base_index}.{stage_conv_index}.conv.{suffix}"
            base_index += 1

    return mapping


def _normalize_unet_state_dict(state_dict: Mapping[str, Any], config: Mapping[str, Any]) -> Mapping[str, Any]:
    stripped_map = _strip_unet_prefixes_mapping(state_dict)

    diff_to_ldm = _build_diffusers_to_ldm_map(config)
    key_lookup: Dict[str, Any] = {}
    leftovers: list[str] = []
    for key in stripped_map.keys():
        if key.startswith((
            "input_blocks.",
            "output_blocks.",
            "middle_block.",
            "out.",
            "time_embed.",
            "label_emb.",
            "add_embedding.",
        )):
            source_key = stripped_map[key]
            previous = key_lookup.get(key)
            if previous is not None and previous != source_key:
                raise RuntimeError(
                    "UNet state dict normalisation collision: destination key "
                    f"{key!r} maps to multiple source keys ({previous!r}, {source_key!r})."
                )
            key_lookup[key] = source_key
            continue
        target = diff_to_ldm.get(key)
        if target is not None:
            source_key = stripped_map[key]
            previous = key_lookup.get(target)
            if previous is not None and previous != source_key:
                raise RuntimeError(
                    "UNet state dict normalisation collision: destination key "
                    f"{target!r} maps to multiple source keys ({previous!r}, {source_key!r})."
                )
            key_lookup[target] = source_key
        else:
            leftovers.append(key)

    missing = [k for k in _ESSENTIAL_UNET_KEYS if k not in key_lookup]
    if missing:
        sample = list(sorted(leftovers))[:10]
        raise RuntimeError(
            "UNet state dict normalisation failed; missing essentials %s. Sample diffusers keys: %s"
            % (missing, sample)
        )

    if leftovers:
        UNET_LOG.debug("UNet leftover keys (diffusers layout) count=%d sample=%s", len(leftovers), leftovers[:5])

    return KeyspaceLookupView(state_dict, key_lookup)


def _strip_transformer_prefixes(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    mapping: Dict[str, Any] = {}
    for raw_key in state_dict.keys():
        key = str(raw_key)
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in _TRANSFORMER_PREFIXES:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
                    break
        previous = mapping.get(new_key)
        if previous is not None and previous != key:
            raise RuntimeError(
                "Transformer prefix stripping collision: destination key "
                f"{new_key!r} maps to multiple source keys ({previous!r}, {key!r})."
            )
        mapping[new_key] = key
    return KeyspaceLookupView(state_dict, mapping)
