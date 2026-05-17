"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen Image model detector for the model registry.
Detects Qwen Image checkpoints (flow transformer + Qwen2.5-VL-7B text encoder + VAE) and builds a `ModelSignature` used by the loader/UI inventory.

Symbols (top-level; keep in sync; no ghosts):
- `QWEN_IMAGE_REQUIRED_KEYS` (constant): Key set used to identify Qwen Image checkpoints.
- `QwenImageDetector` (class): Detector that builds a `ModelSignature` for Qwen Image checkpoints.
- `_shape` (function): Helper to fetch a tensor shape from a bundle/state dict (best-effort).
"""

from __future__ import annotations

from typing import Optional, Tuple

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
    TextEncoderSignature,
    VAESignature,
)


QWEN_IMAGE_REQUIRED_KEYS = (
    "img_in.weight",
    "proj_out.weight",
    "txt_norm.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "transformer_blocks.0.attn.add_q_proj.weight",
    "transformer_blocks.0.attn.add_k_proj.weight",
    "transformer_blocks.0.txt_mod.1.weight",
    "text_encoder.transformer.layers.0.self_attn.q_proj.weight",
    "vae.decoder.conv_out.weight",
)


class QwenImageDetector(ModelDetector):
    priority = 162

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        if not has_all_keys(bundle, *QWEN_IMAGE_REQUIRED_KEYS):
            return False
        # Avoid matching WAN family shards accidentally (some exports mix checkpoints).
        if any(key.endswith("head.modulation") for key in bundle.keys):
            return False
        return True

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        img_in = _shape(bundle, "img_in.weight")
        channels_in = int(img_in[1]) if img_in and len(img_in) == 2 else 64

        vae_shape = _shape(bundle, "vae.decoder.conv_out.weight")
        latent_channels = int(vae_shape[1]) if vae_shape and len(vae_shape) >= 2 else 16

        txt_norm = _shape(bundle, "txt_norm.weight")
        context_dim = int(txt_norm[0]) if txt_norm and len(txt_norm) == 1 else None

        depth = count_blocks(bundle.keys, "transformer_blocks.{}.") or 1

        return ModelSignature(
            family=ModelFamily.QWEN_IMAGE,
            repo_hint=None,
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.QWEN_IMAGE,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.FLOW_TRANSFORMER,
                channels_in=channels_in,
                channels_out=latent_channels,
                context_dim=context_dim,
                temporal=False,
                depth=depth,
                key_prefixes=["transformer_blocks."],
            ),
            text_encoders=[
                TextEncoderSignature(
                    name="qwen2_5_vl_7b",
                    key_prefix="text_encoder.",
                    expected_dim=context_dim,
                )
            ],
            vae=VAESignature(key_prefix="vae.", latent_channels=latent_channels),
            extras={},
        )


def _shape(bundle: SignalBundle, key: str) -> Optional[Tuple[int, ...]]:
    shape = bundle.shape(key)
    if shape is not None:
        return tuple(int(v) for v in shape)
    tensor = bundle.state_dict.get(key)
    if isinstance(tensor, torch.Tensor):
        return tuple(int(v) for v in tensor.shape)
    return None


REGISTRY.register(QwenImageDetector())
