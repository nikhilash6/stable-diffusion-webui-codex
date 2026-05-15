"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical IP-Adapter slot-to-UNet coordinate layouts.
Owns the explicit per-family attn2 slot order used to bind IP-Adapter checkpoint slots onto the active denoiser and validates the
binding against the actual UNet `attn2.to_k` widths. This is a checkpoint/runtime contract, not generic UNet discovery order.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_ip_adapter_transformer_coordinates` (function): Return the canonical IP-Adapter attn2 slot order for the active denoiser and validate it against the actual UNet transformer inventory plus slot/projector widths.
"""

from __future__ import annotations

from collections.abc import Iterable

from apps.backend.runtime.common.nn.unet.layers import SpatialTransformer
from apps.backend.runtime.model_parser.builders import _CORE_CONFIG_PRESETS
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.models.key_normalization import _build_diffusers_to_ldm_map


def resolve_ip_adapter_transformer_coordinates(
    *,
    patched_denoiser,
    semantic_engine: str,
    ip_layers,
) -> tuple[tuple[str, int, int], ...]:
    normalized_engine = str(semantic_engine).strip().lower()
    if normalized_engine == "sd15":
        expected = _sd15_ip_adapter_layout()
    elif normalized_engine == "sdxl":
        expected = _sdxl_ip_adapter_layout()
    else:
        raise RuntimeError(
            f"Unsupported IP-Adapter semantic engine '{semantic_engine}' for transformer slot resolution."
        )
    available = tuple(patched_denoiser._iter_transformer_coordinates())
    _assert_layout_matches_expected(
        semantic_engine=normalized_engine,
        expected=expected,
        available=available,
    )
    _assert_layout_matches_translated_parameter_order(
        semantic_engine=normalized_engine,
        coordinates=expected,
    )
    _assert_slot_projection_widths_match(
        patched_denoiser=patched_denoiser,
        ip_layers=ip_layers,
        coordinates=expected,
    )
    return expected


def _sd15_ip_adapter_layout() -> tuple[tuple[str, int, int], ...]:
    coordinates: list[tuple[str, int, int]] = []
    for block_index in (1, 2, 4, 5, 7, 8):
        coordinates.append(("input", block_index, 0))
    for block_index in (3, 4, 5, 6, 7, 8, 9, 10, 11):
        coordinates.append(("output", block_index, 0))
    coordinates.append(("middle", 0, 0))
    return tuple(coordinates)


def _sdxl_ip_adapter_layout() -> tuple[tuple[str, int, int], ...]:
    coordinates: list[tuple[str, int, int]] = []
    for block_index in (4, 5, 7, 8):
        transformer_depth = 2 if block_index in (4, 5) else 10
        for transformer_index in range(transformer_depth):
            coordinates.append(("input", block_index, transformer_index))
    for block_index in range(6):
        transformer_depth = 10 if block_index in (0, 1, 2) else 2
        for transformer_index in range(transformer_depth):
            coordinates.append(("output", block_index, transformer_index))
    for transformer_index in range(10):
        coordinates.append(("middle", 0, transformer_index))
    return tuple(coordinates)


def _assert_layout_matches_expected(
    *,
    semantic_engine: str,
    expected: Iterable[tuple[str, int, int]],
    available: Iterable[tuple[str, int, int]],
) -> None:
    expected_tuple = tuple(expected)
    available_tuple = tuple(available)
    expected_set = set(expected_tuple)
    available_set = set(available_tuple)
    if expected_set == available_set and len(expected_tuple) == len(available_tuple):
        return
    missing = sorted(expected_set - available_set)
    extra = sorted(available_set - expected_set)
    mismatch_message = (
        "IP-Adapter transformer layout mismatch: "
        f"semantic_engine={semantic_engine} expected_slots={len(expected_tuple)} available_slots={len(available_tuple)}"
    )
    details: list[str] = []
    if missing:
        details.append("missing=" + ", ".join(str(item) for item in missing[:8]))
    if extra:
        details.append("extra=" + ", ".join(str(item) for item in extra[:8]))
    if details:
        mismatch_message = mismatch_message + " (" + "; ".join(details) + ")"
    raise RuntimeError(mismatch_message)


def _assert_layout_matches_translated_parameter_order(
    *,
    semantic_engine: str,
    coordinates: tuple[tuple[str, int, int], ...],
) -> None:
    if semantic_engine != "sdxl":
        return
    translated = tuple(_coordinate_attn2_to_k_key(coordinate) for coordinate in coordinates)
    expected = _sdxl_translated_attn2_to_k_order()
    if translated == expected:
        return
    if len(translated) != len(expected):
        raise RuntimeError(
            "IP-Adapter translated slot-order mismatch: "
            f"semantic_engine={semantic_engine} expected_count={len(expected)} actual_count={len(translated)}."
        )
    mismatch_index = next(
        index
        for index, (expected_key, actual_key) in enumerate(zip(expected, translated))
        if expected_key != actual_key
    )
    raise RuntimeError(
        "IP-Adapter translated slot-order mismatch: "
        f"semantic_engine={semantic_engine} slot={mismatch_index} "
        f"expected='{expected[mismatch_index]}' actual='{translated[mismatch_index]}'."
    )


def _sdxl_translated_attn2_to_k_order() -> tuple[str, ...]:
    config = _CORE_CONFIG_PRESETS[ModelFamily.SDXL]
    diffusers_to_ldm = _build_diffusers_to_ldm_map(config)
    num_res_blocks = list(config["num_res_blocks"])
    transformer_depth = list(config["transformer_depth"])
    transformer_depth_output = list(config["transformer_depth_output"])
    translated: list[str] = []

    depth_index = 0
    for block_index in range(len(config["channel_mult"])):
        for res_index in range(num_res_blocks[block_index]):
            transformer_count = transformer_depth[depth_index]
            depth_index += 1
            for transformer_index in range(transformer_count):
                translated.append(
                    diffusers_to_ldm[
                        f"down_blocks.{block_index}.attentions.{res_index}.transformer_blocks.{transformer_index}.attn2.to_k.weight"
                    ]
                )

    up_res_counts = list(reversed(num_res_blocks))
    for block_index in range(len(config["channel_mult"])):
        block_length = up_res_counts[block_index] + 1
        for res_index in range(block_length):
            transformer_count = transformer_depth_output.pop() if transformer_depth_output else 0
            for transformer_index in range(transformer_count):
                translated.append(
                    diffusers_to_ldm[
                        f"up_blocks.{block_index}.attentions.{res_index}.transformer_blocks.{transformer_index}.attn2.to_k.weight"
                    ]
                )

    transformer_depth_middle = int(config["transformer_depth_middle"])
    for transformer_index in range(transformer_depth_middle):
        translated.append(
            diffusers_to_ldm[
                f"mid_block.attentions.0.transformer_blocks.{transformer_index}.attn2.to_k.weight"
            ]
        )
    return tuple(translated)


def _coordinate_attn2_to_k_key(coordinate: tuple[str, int, int]) -> str:
    block_name, block_index, transformer_index = coordinate
    if block_name == "input":
        return f"input_blocks.{block_index}.1.transformer_blocks.{transformer_index}.attn2.to_k.weight"
    if block_name == "middle":
        return f"middle_block.1.transformer_blocks.{transformer_index}.attn2.to_k.weight"
    if block_name == "output":
        return f"output_blocks.{block_index}.1.transformer_blocks.{transformer_index}.attn2.to_k.weight"
    raise RuntimeError(f"Unsupported IP-Adapter block name '{block_name}'.")


def _assert_slot_projection_widths_match(*, patched_denoiser, ip_layers, coordinates: tuple[tuple[str, int, int], ...]) -> None:
    slot_specs = tuple(ip_layers.slot_specs)
    if len(slot_specs) != len(coordinates):
        raise RuntimeError(
            "IP-Adapter slot/source-key mismatch during width validation: "
            f"slot_specs={len(slot_specs)} coordinates={len(coordinates)}."
        )
    for slot_index, (coordinate, slot_spec) in enumerate(zip(coordinates, slot_specs, strict=True)):
        coordinate_width = _coordinate_attn2_to_k_width(patched_denoiser=patched_denoiser, coordinate=coordinate)
        source_key = slot_spec.k_source_key
        projection_width = int(ip_layers.projection(source_key).weight.shape[0])
        if projection_width != coordinate_width:
            raise RuntimeError(
                "IP-Adapter slot-width mismatch: "
                f"slot={slot_index} source_key='{source_key}' projection_width={projection_width} "
                f"coordinate={coordinate} attn2_to_k_width={coordinate_width}"
            )


def _coordinate_attn2_to_k_width(*, patched_denoiser, coordinate: tuple[str, int, int]) -> int:
    block_name, block_index, transformer_index = coordinate
    diffusion_model = getattr(getattr(patched_denoiser, "model", None), "diffusion_model", None)
    if diffusion_model is None:
        raise RuntimeError("IP-Adapter slot-width validation requires patched_denoiser.model.diffusion_model.")
    if block_name == "input":
        block = getattr(diffusion_model, "input_blocks", [])[block_index]
    elif block_name == "middle":
        if int(block_index) != 0:
            raise RuntimeError(f"Unexpected middle-block coordinate index {block_index}.")
        block = getattr(diffusion_model, "middle_block", None)
    elif block_name == "output":
        block = getattr(diffusion_model, "output_blocks", [])[block_index]
    else:
        raise RuntimeError(f"Unsupported IP-Adapter block name '{block_name}'.")
    if block is None:
        raise RuntimeError(f"Missing diffusion-model block for coordinate {coordinate}.")
    spatial_transformers = [module for module in block if isinstance(module, SpatialTransformer)]
    if len(spatial_transformers) != 1:
        raise RuntimeError(
            "IP-Adapter slot-width validation requires exactly one SpatialTransformer host per block; "
            f"got {len(spatial_transformers)} in coordinate {coordinate}."
        )
    transformer_blocks = spatial_transformers[0].transformer_blocks
    if transformer_index >= len(transformer_blocks):
        raise RuntimeError(
            f"IP-Adapter coordinate {coordinate} exceeds transformer depth {len(transformer_blocks)}."
        )
    attn2_to_k = transformer_blocks[transformer_index].attn2.to_k
    weight = getattr(attn2_to_k, "weight", None)
    if weight is None:
        raise RuntimeError(f"IP-Adapter coordinate {coordinate} is missing attn2.to_k.weight.")
    return int(weight.shape[0])
