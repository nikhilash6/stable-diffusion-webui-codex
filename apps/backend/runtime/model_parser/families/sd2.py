"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SD2.x parser plan builder (UNet + optional VAE + CLIP-H).
Defines split/conversion/validation steps for SD2.x checkpoints, converting CLIP-H (`clip_h`) and validating UNet channel count and
required CLIP layer-norm keys.

Symbols (top-level; keep in sync; no ghosts):
- `build_plan` (function): Builds and returns the SD2.x `ParserPlanBundle`.
- `_convert_clip` (function): Converts SD2.x CLIP tensors and registers the `clip_h` alias mapping.
- `_validate_unet_channels` (function): Validates UNet `channels_in` vs the `ModelSignature` expectation.
- `_validate_clip_keys` (function): Validates that required CLIP keys exist after conversion.
"""

from __future__ import annotations

from typing import Dict

import torch

from apps.backend.runtime.model_registry.specs import ModelSignature

from ..builders import build_estimated_config, register_text_encoder
from ..converters.clip import convert_sd20_clip
from ..errors import ValidationError
from ..specs import (
    ParserPlan,
    ParserPlanBundle,
    SplitSpec,
    ConverterSpec,
    ValidationSpec,
)
from ..quantization import validate_component_dtypes


def build_plan(signature: ModelSignature) -> ParserPlanBundle:
    plan = ParserPlan(
        splits=[
            SplitSpec(name="unet", prefixes=("model.diffusion_model.",)),
            SplitSpec(name="vae", prefixes=("first_stage_model.",), required=False),
            SplitSpec(name="text_encoder", prefixes=("conditioner.embedders.0.model.", "cond_stage_model.model.")),
        ],
        converters=(
            ConverterSpec(component="text_encoder", function=_convert_clip),
        ),
        validations=(
            ValidationSpec(name="unet_channels", function=_validate_unet_channels),
            ValidationSpec(name="clip_presence", function=_validate_clip_keys),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))


def _convert_clip(tensors: Dict[str, torch.Tensor], context):
    converted = convert_sd20_clip(tensors)
    register_text_encoder(context, "clip_h", "text_encoder")
    return converted


def _validate_unet_channels(context):
    unet = context.require("unet").tensors
    key = "input_blocks.0.0.weight"
    shape_getter = getattr(unet, "shape_of", None)
    weight_shape: tuple[int, ...] | None = None
    if callable(shape_getter):
        try:
            shape_raw = shape_getter(key)
            if shape_raw is not None:
                weight_shape = tuple(int(v) for v in shape_raw)
        except Exception:
            weight_shape = None
    if weight_shape is None:
        weight = unet.get(key)
        if not isinstance(weight, torch.Tensor):
            raise ValidationError(f"Expected '{key}' in UNet state dict", component="unet")
        weight_shape = tuple(int(v) for v in weight.shape)
    if len(weight_shape) < 2:
        raise ValidationError(
            f"UNet weight shape for '{key}' is invalid: {weight_shape}",
            component="unet",
        )
    expected = context.signature.core.channels_in
    if int(weight_shape[1]) != expected:
        raise ValidationError(
            f"UNet channels_in mismatch: expected {expected}, found {int(weight_shape[1])}",
            component="unet",
        )


def _validate_clip_keys(context):
    clip = context.require("text_encoder").tensors
    required = "transformer.text_model.encoder.layers.0.layer_norm1.weight"
    if required not in clip:
        raise ValidationError(f"Missing key '{required}' after conversion", component="text_encoder")
