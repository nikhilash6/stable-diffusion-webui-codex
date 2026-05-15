"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDXL parser plan builder (UNet + optional VAE + CLIP-L/CLIP-G).
Builds split/validation steps and registers text-encoder aliases; checkpoint/CLIP key normalization is deferred to loader keymaps to keep parsing lazy.

Symbols (top-level; keep in sync; no ghosts):
- `build_plan` (function): Builds and returns the SDXL `ParserPlanBundle` (also used for SDXL refiner).
- `_validate_unet_channels` (function): Validates UNet `channels_in` vs the `ModelSignature` expectation.
- `_register_base_text_encoders` (function): Registers SDXL base text-encoder aliases in the parser context.
- `_register_refiner_text_encoders` (function): Registers SDXL refiner text-encoder aliases in the parser context.
"""

from __future__ import annotations

import torch

from apps.backend.runtime.model_registry.specs import ModelFamily, ModelSignature

from ..builders import build_estimated_config, register_text_encoder
from ..errors import ValidationError
from ..specs import (
    ParserPlan,
    ParserPlanBundle,
    SplitSpec,
    ValidationSpec,
)
from ..quantization import validate_component_dtypes


def build_plan(signature: ModelSignature) -> ParserPlanBundle:
    is_refiner = signature.family is ModelFamily.SDXL_REFINER
    is_core_only = bool(signature.extras.get("core_only"))
    if is_refiner:
        plan = ParserPlan(
            splits=[
                SplitSpec(name="unet", prefixes=("model.diffusion_model.",)),
                SplitSpec(name="vae", prefixes=("first_stage_model.", "vae."), required=False),
                SplitSpec(name="text_encoder", prefixes=("conditioner.embedders.0.model.",), required=not is_core_only),
            ],
            converters=(),
            validations=(
                ValidationSpec(name="unet_channels", function=_validate_unet_channels),
                ValidationSpec(name="register_text_encoders", function=_register_refiner_text_encoders),
                ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
            ),
        )
        return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))

    plan = ParserPlan(
        splits=[
            SplitSpec(name="unet", prefixes=("model.diffusion_model.",)),
            SplitSpec(name="vae", prefixes=("first_stage_model.", "vae."), required=False),
            SplitSpec(
                name="text_encoder",
                prefixes=("conditioner.embedders.0.model.", "conditioner.embedders.0."),
                required=not is_core_only,
            ),
            SplitSpec(
                name="text_encoder_2",
                prefixes=("conditioner.embedders.1.model.", "conditioner.embedders.1."),
                required=not is_core_only,
            ),
        ],
        converters=(),
        validations=(
            ValidationSpec(name="unet_channels", function=_validate_unet_channels),
            ValidationSpec(name="register_text_encoders", function=_register_base_text_encoders),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))


def _register_base_text_encoders(context) -> None:
    register_text_encoder(context, "clip_l", "text_encoder")
    register_text_encoder(context, "clip_g", "text_encoder_2")


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


def _register_refiner_text_encoders(context) -> None:
    register_text_encoder(context, "clip_g", "text_encoder")
