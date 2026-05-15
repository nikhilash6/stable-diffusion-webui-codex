"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux-family model detectors for the model registry.
Detects full Flux checkpoints (dev/schnell) and core-only GGUF Flux checkpoints, building `ModelSignature` metadata for inventory/UI and
registering detectors into the shared `REGISTRY` at import time.

Symbols (top-level; keep in sync; no ghosts):
- `FLUX_CORE_KEYS` (constant): Key set used to identify full Flux core checkpoints.
- `_FluxBaseDetector` (class): Shared detector logic for full Flux variants (guidance vs schnell gating).
- `FluxDetector` (class): Detector for FLUX.1-dev (guidance expected).
- `FluxSchnellDetector` (class): Detector for FLUX.1-schnell (no guidance).
- `FluxCoreGGUFDetector` (class): Detector for core-only GGUF Flux checkpoints (transformer + guidance only).
- `_shape_at` (function): Helper to read a shape dimension from bundle metadata/state dict.
"""

from __future__ import annotations

from typing import Optional

import torch

from apps.backend.runtime.model_registry.detectors.base import ModelDetector, REGISTRY
from apps.backend.runtime.model_registry.signals import SignalBundle, count_blocks, has_all_keys
from apps.backend.runtime.model_registry.specs import (
    CodexCoreArchitecture,
    CodexCoreSignature,
    LatentFormat,
    ModelFamily,
    ModelSignature,
    PredictionKind,
    QuantizationHint,
    QuantizationKind,
    TextEncoderSignature,
    VAESignature,
)

FLUX_CORE_KEYS = (
    "x_embedder.weight",
    "proj_out.weight",
    "context_embedder.weight",
    "double_blocks.0.img_attn.norm.key_norm.scale",
    "single_transformer_blocks.0.attn.norm_k.weight",
)


class _FluxBaseDetector:
    repo_hint: str
    guidance_expected: bool
    family: ModelFamily
    priority = 150

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        if not has_all_keys(bundle, *FLUX_CORE_KEYS):
            return False
        guidance_key = "guidance_in.in_layer.weight"
        has_guidance = guidance_key in bundle.state_dict
        return has_guidance if self.guidance_expected else not has_guidance

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        channels_in = _shape_at(bundle, "x_embedder.weight", dim=1) or 64
        channels_out = _shape_at(bundle, "proj_out.weight", dim=0) or channels_in
        context_dim = _shape_at(bundle, "context_embedder.weight", dim=0) or 4096
        guidance_key = "guidance_in.in_layer.weight"
        adm_in_channels = _shape_at(bundle, guidance_key, dim=1)
        double_layers = count_blocks(bundle.keys, "double_blocks.{}.")
        single_layers = count_blocks(bundle.keys, "single_transformer_blocks.{}.")

        text_encoders = [
            TextEncoderSignature(
                name="clip_l",
                key_prefix="text_encoders.clip_l.",
                expected_dim=768,
                tokenizer_hint=f"{self.repo_hint}/tokenizer",
            ),
            TextEncoderSignature(
                name="t5xxl",
                key_prefix="text_encoders.t5xxl.",
                expected_dim=4096,
                tokenizer_hint=f"{self.repo_hint}/tokenizer_2",
            ),
        ]

        vae = None
        if any(key.startswith("vae.") for key in bundle.keys):
            latent = _shape_at(bundle, "vae.decoder.conv_out.weight", dim=0) or 3
            vae = VAESignature(key_prefix="vae.", latent_channels=16 if latent == 3 else 16)

        extras = {
            "flow_double_layers": double_layers,
            "flow_single_layers": single_layers,
            "guidance_embed": adm_in_channels is not None,
        }

        return ModelSignature(
            family=self.family,
            repo_hint=self.repo_hint,
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.FLOW16,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.FLOW_TRANSFORMER,
                channels_in=channels_in,
                channels_out=channels_out,
                context_dim=context_dim,
                temporal=False,
                depth=double_layers + single_layers,
                key_prefixes=["double_blocks.", "single_transformer_blocks."],
            ),
            text_encoders=text_encoders,
            vae=vae,
            extras=extras,
        )


class FluxDetector(_FluxBaseDetector):
    repo_hint = "black-forest-labs/FLUX.1-dev"
    guidance_expected = True
    family = ModelFamily.FLUX
    priority = 140


class FluxSchnellDetector(_FluxBaseDetector):
    repo_hint = "black-forest-labs/FLUX.1-schnell"
    guidance_expected = False
    family = ModelFamily.FLUX
    priority = 145


class FluxCoreGGUFDetector(ModelDetector):
    """Detector for Flux core-only GGUF checkpoints (transformer + guidance only).

    These files typically contain only the rectified-flow backbone (double_blocks.*)
    plus guidance layers, with no VAE or text encoders embedded. They are expected
    to be consumed together with external CLIP-L/T5 encoders and VAE weights.
    """

    priority = 135  # run before full Flux detectors

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        # Require core Flux blocks + guidance layers
        if "guidance_in.in_layer.weight" not in bundle.state_dict:
            return False
        if not any(k.startswith("double_blocks.") for k in bundle.keys):
            return False

        # Core-only: must NOT contain the higher-level embedder/projection keys used
        # by full Flux checkpoints.
        for key in ("x_embedder.weight", "proj_out.weight", "context_embedder.weight"):
            if key in bundle.state_dict:
                return False

        # Ensure this is actually a GGUF quantized checkpoint.
        if not bundle.is_gguf_quantized():
            return False

        return True

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        # Derive basic channel configuration from attention weights when possible.
        channels_in = 64
        channels_out = 64
        context_dim = 4096

        qkv_key = "double_blocks.0.img_attn.qkv.weight"
        qkv_shape = bundle.shape(qkv_key)
        if qkv_shape and len(qkv_shape) == 2:
            channels_out = int(qkv_shape[0])

        guidance_key = "guidance_in.in_layer.weight"
        adm_in_channels = _shape_at(bundle, guidance_key, dim=1)
        double_layers = count_blocks(bundle.keys, "double_blocks.{}.")
        single_layers = count_blocks(bundle.keys, "single_transformer_blocks.{}.")

        # We still describe the expected text encoders so downstream code knows
        # which components to load from the diffusers repo / paths, even though
        # they are not embedded in this GGUF checkpoint.
        text_encoders = [
            TextEncoderSignature(
                name="clip_l",
                key_prefix="text_encoders.clip_l.",
                expected_dim=768,
                tokenizer_hint=None,
            ),
            TextEncoderSignature(
                name="t5xxl",
                key_prefix="text_encoders.t5xxl.",
                expected_dim=4096,
                tokenizer_hint=None,
            ),
        ]

        extras = {
            "flow_double_layers": double_layers,
            "flow_single_layers": single_layers,
            "guidance_embed": adm_in_channels is not None,
            "gguf_core_only": True,
        }

        return ModelSignature(
            family=ModelFamily.FLUX,
            repo_hint="",
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.FLOW16,
            quantization=QuantizationHint(kind=QuantizationKind.GGUF, detail="parameter_gguf"),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.FLOW_TRANSFORMER,
                channels_in=channels_in,
                channels_out=channels_out,
                context_dim=context_dim,
                temporal=False,
                depth=double_layers + single_layers,
                key_prefixes=["double_blocks."],
            ),
            text_encoders=text_encoders,
            vae=None,
            extras=extras,
        )


def _shape_at(bundle: SignalBundle, key: str, dim: int) -> Optional[int]:
    shape = bundle.shape(key)
    if shape and len(shape) > dim:
        return int(shape[dim])
    tensor = bundle.state_dict.get(key)
    if isinstance(tensor, torch.Tensor) and tensor.ndim > dim:
        return int(tensor.shape[dim])
    return None


REGISTRY.register(FluxDetector())
REGISTRY.register(FluxSchnellDetector())
REGISTRY.register(FluxCoreGGUFDetector())
