"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux parser plan builder (GGUF core-only vs full checkpoints).
Defines a `ParserPlan` for Flux-family models, handling GGUF core-only checkpoints (transformer-only with external CLIP/T5/VAE) and full
checkpoints that embed both text encoders. Registers stable text-encoder aliases for override resolution and validates core presence.

Symbols (top-level; keep in sync; no ghosts):
- `_FLUX_CORE_PREFIXES` (constant): Accepted prefixes for Flux core weights in non-GGUF checkpoints.
- `_register_flux_text_encoders` (function): Registers `clip_l`/`t5xxl` aliases in the parser context for core-only checkpoints.
- `build_plan` (function): Builds and returns the Flux `ParserPlanBundle` (splits + converters + validations).
- `_convert_clip_l` (function): Converts CLIP-L tensors and registers the text encoder alias mapping.
- `_convert_t5` (function): Converts T5-XXL tensors and registers the text encoder alias mapping.
- `_validate_transformer_core` (function): Validates presence of required Flux core keys.
- `_validate_clip_l` (function): Validates CLIP-L conversion output keys.
- `_validate_t5` (function): Validates T5 conversion output keys.
"""

from __future__ import annotations

from typing import Dict

import torch

from apps.backend.runtime.model_registry.specs import ModelSignature, QuantizationKind

from ..builders import build_estimated_config, register_text_encoder
from ..converters import convert_clip, convert_t5xxl_encoder
from ..errors import ValidationError
from ..specs import (
    ParserPlan,
    ParserPlanBundle,
    SplitSpec,
    ConverterSpec,
    ValidationSpec,
)
from ..quantization import validate_component_dtypes


_FLUX_CORE_PREFIXES = ("transformer.", "model.diffusion_model.")


def _register_flux_text_encoders(context) -> None:
    """Register expected Flux text encoder aliases even when weights are external.

    For GGUF core-only checkpoints, CLIP-L/T5/VAE do not live in the primary
    state_dict. We still need a stable alias → component map so that the loader
    can apply text encoder overrides (via `TextEncoderOverrideConfig`) for
    `clip_l` and `t5xxl`.
    """
    register_text_encoder(context, "clip_l", "text_encoder")
    register_text_encoder(context, "t5xxl", "text_encoder_2")


def build_plan(signature: ModelSignature) -> ParserPlanBundle:
    # GGUF core-only checkpoints: only the rectified-flow backbone lives in the
    # state_dict (double_blocks.+guidance), while CLIP/T5/VAE come from the
    # diffusers repo or external paths. For these, keep the plan minimal and
    # avoid text-encoder/vae validations, but still declare the logical text
    # encoders so overrides can be resolved.
    if signature.quantization.kind == QuantizationKind.GGUF:
        plan = ParserPlan(
            splits=[
                # Core-only: include all tensors under the single transformer component.
                SplitSpec(name="transformer", prefixes=("",)),
            ],
            converters=(),
            validations=(
                ValidationSpec(name="register_flux_text_encoders", function=_register_flux_text_encoders),
                ValidationSpec(name="core_presence", function=_validate_transformer_core),
                ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
            ),
        )
        return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))

    # Full Flux checkpoints: expect transformer + VAE + both text encoders.
    plan = ParserPlan(
        splits=[
            SplitSpec(name="transformer", prefixes=_FLUX_CORE_PREFIXES),
            SplitSpec(name="vae", prefixes=("vae.",), required=False),
            SplitSpec(name="text_encoder", prefixes=("text_encoders.clip_l.",)),
            SplitSpec(name="text_encoder_2", prefixes=("text_encoders.t5xxl.",)),
        ],
        converters=(
            ConverterSpec(component="text_encoder", function=_convert_clip_l),
            ConverterSpec(component="text_encoder_2", function=_convert_t5),
        ),
        validations=(
            ValidationSpec(name="core_presence", function=_validate_transformer_core),
            ValidationSpec(name="clip_l_presence", function=_validate_clip_l),
            ValidationSpec(name="t5_presence", function=_validate_t5),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))


def _convert_clip_l(tensors: Dict[str, torch.Tensor], context):
    converted = convert_clip(
        tensors,
        alias="clip_l",
        layers=32,
        ensure_position_ids=True,
        drop_logit_scale=True,
    )
    register_text_encoder(context, "clip_l", "text_encoder")
    return converted


def _convert_t5(tensors: Dict[str, torch.Tensor], context):
    converted = convert_t5xxl_encoder(tensors)
    register_text_encoder(context, "t5xxl", "text_encoder_2")
    return converted


def _validate_transformer_core(context):
    unet = context.require("transformer").tensors
    key = "double_blocks.0.img_attn.norm.key_norm.scale"
    if key not in unet:
        if any(k.startswith(("transformer_blocks.", "single_transformer_blocks.")) for k in unet):
            raise ValidationError(
                "Flux transformer reached the parser with source/native Diffusers block keys "
                "(`transformer_blocks.*`, `single_transformer_blocks.*`) instead of the resolved Flux runtime keyspace (`double_blocks.*`). "
                "Resolve the checkpoint through the expected-family keyspace path before parser execution.",
                component="transformer",
            )
        raise ValidationError("Flux transformer missing double block attn scale", component="transformer")


def _validate_clip_l(context):
    clip = context.require("text_encoder").tensors
    key = "transformer.text_model.encoder.layers.0.layer_norm1.weight"
    if key not in clip:
        raise ValidationError("Flux CLIP-L conversion failed", component="text_encoder")


def _validate_t5(context):
    t5 = context.require("text_encoder_2").tensors
    key = "transformer.encoder.final_layer_norm.weight"
    if key not in t5:
        raise ValidationError("Flux T5 conversion missing final layer norm", component="text_encoder_2")
