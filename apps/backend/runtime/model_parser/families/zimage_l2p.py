"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z-Image L2P parser plan builder (core-only pixel DiT checkpoints).
Builds a single transformer component from native L2P tensor names and registers the exact `qwen3_4b` external text encoder slot for
loader/runtime handoff without accepting Z-Image latent aliases.

Symbols (top-level; keep in sync; no ghosts):
- `_register_zimage_l2p_text_encoder` (function): Registers the exact `qwen3_4b` parser slot/component mapping.
- `build_plan` (function): Builds and returns the Z-Image L2P `ParserPlanBundle`.
- `_validate_transformer_core` (function): Validates required native L2P transformer tensors.
"""

from __future__ import annotations

from apps.backend.runtime.model_registry.specs import ModelSignature

from ..builders import build_estimated_config, register_text_encoder
from ..errors import ValidationError
from ..quantization import validate_component_dtypes
from ..specs import ParserPlan, ParserPlanBundle, SplitSpec, ValidationSpec


_REQUIRED_L2P_KEYS: tuple[str, ...] = (
    "all_x_embedder.16-1.weight",
    "local_decoder.out_conv.weight",
    "layers.0.adaLN_modulation.0.weight",
    "noise_refiner.0.adaLN_modulation.0.weight",
    "cap_embedder.1.weight",
)


def _register_zimage_l2p_text_encoder(context) -> None:
    register_text_encoder(context, "qwen3_4b", "qwen3_4b")


def build_plan(signature: ModelSignature) -> ParserPlanBundle:
    plan = ParserPlan(
        splits=[
            SplitSpec(name="transformer", prefixes=("",)),
        ],
        converters=(),
        validations=(
            ValidationSpec(name="register_zimage_l2p_text_encoder", function=_register_zimage_l2p_text_encoder),
            ValidationSpec(name="core_presence", function=_validate_transformer_core),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))


def _validate_transformer_core(context) -> None:
    transformer = context.require("transformer").tensors
    missing = [key for key in _REQUIRED_L2P_KEYS if key not in transformer]
    if missing:
        raise ValidationError(
            "Z-Image L2P transformer missing required native tensors: " + ", ".join(missing),
            component="transformer",
        )
    forbidden = [key for key in ("final_layer.linear.weight",) if key in transformer]
    if forbidden:
        raise ValidationError(
            "Z-Image L2P transformer contains latent Z-Image final-layer tensors: " + ", ".join(forbidden),
            component="transformer",
        )
