"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z Image model detector for the Codex model registry.
Identifies Z Image core transformer checkpoints (GGUF or prefixed SafeTensors exports), infers key architecture dimensions, and builds a
`ModelSignature` describing the core-only transformer plus external text encoder/VAE expectations.

Symbols (top-level; keep in sync; no ghosts):
- `ZIMAGE_CORE_KEYS` (constant): Minimal key set used to identify Z Image core weights (with optional prefix).
- `ZIMAGE_CORE_PREFIXES` (constant): Accepted prefixes for Z Image core keys (GGUF + common wrapper prefixes).
- `_has_zimage_keys` (function): Checks presence of Z Image core keys in a signal bundle (prefixed or unprefixed).
- `ZImageDetector` (class): Detector that matches Z Image bundles and builds a `ModelSignature` (dims + quantization hint).
- `_shape` (function): Shape helper reading from bundle metadata or tensor objects.
"""

from __future__ import annotations

from typing import Optional

import torch

from apps.backend.runtime.model_registry.detectors.base import ModelDetector, REGISTRY
from apps.backend.runtime.model_registry.signals import SignalBundle
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
from apps.backend.runtime.families.zimage.inference import infer_zimage_dims


# Core keys for Z Image Turbo (NextDiT/Lumina2 format)
# Some checkpoints use these directly, others include wrapper prefixes.
ZIMAGE_CORE_KEYS = (
    "x_embedder.weight",
    "cap_embedder.0.weight",
    "t_embedder.mlp.0.weight",
    "layers.0.adaLN_modulation.0.weight",
    "final_layer.linear.weight",
)

ZIMAGE_CORE_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
    "",
)


def _has_zimage_keys(bundle: SignalBundle) -> bool:
    """Check if bundle has Z Image core keys, with or without prefix."""
    keys_set = set(bundle.keys)

    for prefix in ZIMAGE_CORE_PREFIXES:
        prefixed_keys = tuple(prefix + k for k in ZIMAGE_CORE_KEYS)
        if all(k in keys_set for k in prefixed_keys):
            return True

    return False


class ZImageDetector(ModelDetector):
    priority = 160

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        if not _has_zimage_keys(bundle):
            return False
        # Skip Wan family (temporal) by checking absence of modulation head.
        if any(key.endswith("head.modulation") for key in bundle.keys):
            return False
        return True

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        # Detect if this is a GGUF quantized checkpoint.
        is_gguf = bundle.is_gguf_quantized()
        
        # Detect which key prefix is used (prefixed for some exports, unprefixed for GGUF).
        keys_set = set(bundle.keys)
        prefix = ""
        for candidate in ZIMAGE_CORE_PREFIXES:
            if candidate and (candidate + "x_embedder.weight") in keys_set:
                prefix = candidate
                break
        
        stripped_keys = [k[len(prefix):] for k in bundle.keys if prefix and k.startswith(prefix)] if prefix else list(bundle.keys)

        def _shape_prefixed(name: str) -> Optional[tuple[int, ...]]:
            return _shape(bundle, prefix + name) if prefix else _shape(bundle, name)

        dims = infer_zimage_dims(stripped_keys, _shape_prefixed, patch_size=2)

        channels_out = dims.latent_channels
        channels_in = dims.latent_channels * 4  # patch_size=2 => 2*2=4

        extras = {
            "hidden_dim": dims.hidden_dim,
            "context_dim": dims.context_dim,
            "num_layers": dims.num_layers,
            "num_refiner_layers": dims.num_refiner_layers,
            "num_heads": dims.num_heads,
            "latent_channels": dims.latent_channels,
            "guidance_embeds": False,
            # Both GGUF and prefixed safetensors (FP8/BF16) are core-only
            # They don't have embedded VAE or text encoder
            "gguf_core_only": is_gguf or bool(prefix),
        }

        text_encoders = [
            TextEncoderSignature(
                name="qwen3_4b",
                key_prefix="text_encoder.",
                expected_dim=dims.context_dim,
                tokenizer_hint="Qwen/Qwen3-4B",
            )
        ]

        # Core-only models (GGUF or prefixed safetensors) don't have embedded VAE
        is_core_only = is_gguf or bool(prefix)
        vae = VAESignature(key_prefix="vae.", latent_channels=dims.latent_channels) if not is_core_only else None
        
        # Set quantization hint based on detection
        if is_gguf:
            quantization = QuantizationHint(kind=QuantizationKind.GGUF, detail="parameter_gguf")
        else:
            quantization = QuantizationHint()

        return ModelSignature(
            family=ModelFamily.ZIMAGE,
            repo_hint="Tongyi-MAI/Z-Image-Turbo",
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.ZIMAGE,
            quantization=quantization,
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.DIT,
                channels_in=channels_in,
                channels_out=channels_out,
                context_dim=dims.context_dim,
                temporal=False,
                depth=dims.num_layers,
                key_prefixes=[prefix + "layers."] if prefix else ["layers."],
            ),
            text_encoders=text_encoders,
            vae=vae,
            extras=extras,
        )


def _shape(bundle: SignalBundle, key: str) -> Optional[tuple[int, ...]]:
    shape = bundle.shape(key)
    if shape is not None:
        return tuple(int(v) for v in shape)
    tensor = bundle.state_dict.get(key)
    if isinstance(tensor, torch.Tensor):
        return tuple(int(v) for v in tensor.shape)
    return None


REGISTRY.register(ZImageDetector())
