"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: FLUX.2 parser plan builder for core-only Klein 4B/base-4B checkpoints.
Builds a strict parser plan for the supported FLUX.2 core-only slice after the loader has resolved the source/native checkpoint keys into the
Diffusers `Flux2Transformer2DModel` lookup space. Registers a single external Qwen3-4B text-encoder alias for override resolution and
fails loud on unsupported layouts/configs.

Symbols (top-level; keep in sync; no ghosts):
- `_FLUX2_REQUIRED_NATIVE_KEYS` (constant): Required Diffusers-native FLUX.2 transformer keys.
- `_FLUX2_SUPPORTED_PREFIXES` (constant): Accepted wrapper prefixes before the loader keyspace resolver runs.
- `_register_flux2_text_encoders` (function): Registers the `qwen3_4b` alias mapping in the parser context.
- `build_plan` (function): Builds and returns the FLUX.2 `ParserPlanBundle`.
- `_validate_flux2_transformer_component` (function): Validates the resolved FLUX.2 transformer component.
- `_assert_supported_flux2_native_layout` (function): Validates the supported FLUX.2 4B/base-4B Diffusers-native layout contract.
- `_shape_2d` (function): Returns the 2D tensor shape for a required key.
"""

from __future__ import annotations

from typing import Any, Mapping

from apps.backend.runtime.model_registry.specs import ModelSignature

from ..builders import build_estimated_config, register_text_encoder
from ..errors import ValidationError
from ..quantization import validate_component_dtypes
from ..specs import ParserPlan, ParserPlanBundle, SplitSpec, ValidationSpec

_FLUX2_REQUIRED_NATIVE_KEYS = (
    "x_embedder.weight",
    "context_embedder.weight",
    "time_guidance_embed.timestep_embedder.linear_1.weight",
    "time_guidance_embed.timestep_embedder.linear_2.weight",
    "double_stream_modulation_img.linear.weight",
    "double_stream_modulation_txt.linear.weight",
    "single_stream_modulation.linear.weight",
    "transformer_blocks.0.attn.to_q.weight",
    "transformer_blocks.0.attn.add_q_proj.weight",
    "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
    "proj_out.weight",
    "norm_out.linear.weight",
)

_FLUX2_SUPPORTED_PREFIXES = (
    "transformer.",
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
    "",
)

_UNSUPPORTED_PREFIXES = (
    "text_encoder.",
    "text_encoders.",
    "vae.",
    "guidance_in.",
    "vector_in.",
    "time_guidance_embed.guidance_embedder.",
    "time_guidance_embed.text_embedder.",
)


def _register_flux2_text_encoders(context) -> None:
    register_text_encoder(context, "qwen3_4b", "text_encoder")


def build_plan(signature: ModelSignature) -> ParserPlanBundle:
    plan = ParserPlan(
        splits=[
            SplitSpec(name="transformer", prefixes=_FLUX2_SUPPORTED_PREFIXES),
        ],
        validations=(
            ValidationSpec(name="register_flux2_text_encoders", function=_register_flux2_text_encoders),
            ValidationSpec(name="flux2_transformer", function=_validate_flux2_transformer_component),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))


def _validate_flux2_transformer_component(context) -> None:
    transformer = context.require("transformer").tensors
    missing = [key for key in _FLUX2_REQUIRED_NATIVE_KEYS if key not in transformer]
    if missing:
        if any(
            key.startswith(("img_in.", "double_blocks.", "single_blocks.", "final_layer."))
            for key in transformer
        ):
            raise ValidationError(
                "FLUX.2 expected-family runtime load reached the parser without native keyspace resolution. "
                "Got the legacy Flux runtime keyspace (`img_in.*`, `double_blocks.*`, `single_blocks.*`) where Diffusers-native keys were required.",
                component="transformer",
            )
        raise ValidationError(
            "FLUX.2 transformer keyspace is incomplete after resolution; missing Diffusers-native keys. "
            f"missing_sample={missing[:10]}",
            component="transformer",
        )
    _assert_supported_flux2_native_layout(transformer, signature=context.signature)


def _assert_supported_flux2_native_layout(tensors: Mapping[str, Any], *, signature: ModelSignature) -> None:
    embedded = [key for key in tensors if key.startswith(_UNSUPPORTED_PREFIXES)]
    if embedded:
        raise ValidationError(
            "FLUX.2 core-only slice does not support embedded text encoder/VAE/guidance assets. "
            f"embedded_sample={embedded[:10]}",
            component="transformer",
        )

    expected_in_channels = int(signature.core.channels_in)
    expected_context_dim = int(signature.core.context_dim or 0)
    expected_double = int((signature.extras or {}).get("flow_double_layers", 0))
    expected_single = int((signature.extras or {}).get("flow_single_layers", 0))

    img_in_shape = _shape_2d(tensors, "x_embedder.weight", component="transformer")
    txt_in_shape = _shape_2d(tensors, "context_embedder.weight", component="transformer")
    final_shape = _shape_2d(tensors, "proj_out.weight", component="transformer")

    hidden_dim = int(img_in_shape[0])
    if int(img_in_shape[1]) != expected_in_channels:
        raise ValidationError(
            "FLUX.2 image input projection channel mismatch. "
            f"got={img_in_shape[1]} expected={expected_in_channels}",
            component="transformer",
        )
    if int(txt_in_shape[0]) != hidden_dim:
        raise ValidationError(
            "FLUX.2 hidden-dim mismatch between x_embedder and context_embedder projections. "
            f"img_hidden={hidden_dim} txt_hidden={txt_in_shape[0]}",
            component="transformer",
        )
    if int(txt_in_shape[1]) != expected_context_dim:
        raise ValidationError(
            "Unsupported FLUX.2 context dimension. Only Klein 4B/base-4B is supported. "
            f"got={txt_in_shape[1]} expected={expected_context_dim}",
            component="transformer",
        )
    if int(final_shape[0]) != expected_in_channels or int(final_shape[1]) != hidden_dim:
        raise ValidationError(
            "FLUX.2 final projection shape mismatch. "
            f"got={final_shape} expected=({expected_in_channels}, {hidden_dim})",
            component="transformer",
        )

    double_layers = sum(1 for idx in range(expected_double) if any(k.startswith(f"transformer_blocks.{idx}.") for k in tensors))
    if double_layers != expected_double or any(k.startswith(f"transformer_blocks.{expected_double}.") for k in tensors):
        raise ValidationError(
            "Unsupported FLUX.2 double-block depth. Only Klein 4B/base-4B is supported. "
            f"got={double_layers} expected={expected_double}",
            component="transformer",
        )

    single_layers = sum(
        1 for idx in range(expected_single) if any(k.startswith(f"single_transformer_blocks.{idx}.") for k in tensors)
    )
    if single_layers != expected_single or any(k.startswith(f"single_transformer_blocks.{expected_single}.") for k in tensors):
        raise ValidationError(
            "Unsupported FLUX.2 single-block depth. Only Klein 4B/base-4B is supported. "
            f"got={single_layers} expected={expected_single}",
            component="transformer",
        )


def _shape_2d(tensors: Mapping[str, Any], key: str, *, component: str) -> tuple[int, int]:
    tensor = tensors.get(key)
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValidationError(
            f"FLUX.2 tensor {key!r} must be rank-2, got shape={shape!r}",
            component=component,
        )
    return int(shape[0]), int(shape[1])
