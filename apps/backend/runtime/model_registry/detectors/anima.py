"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima (Cosmos Predict2) model detector for the Codex model registry.
Detects Anima core checkpoints exported with a `net.*` prefix (Cosmos Predict2 / MiniTrainDiT format) and builds a
`ModelSignature` describing the core transformer plus required external assets (Qwen3-0.6B text encoder + WAN-style VAE).

Symbols (top-level; keep in sync; no ghosts):
- `ANIMA_REQUIRED_KEYS` (constant): Key set used to identify Anima core checkpoints.
- `AnimaDetector` (class): Detector that matches Anima bundles and builds a `ModelSignature` (dims + contract extras).
- `_shape` (function): Helper to fetch a tensor shape from a bundle/state dict (best-effort).
- `_shape_2d` (function): Helper to fetch a 2D tensor shape.
- `_infer_patch_config` (function): Infer PatchEmbed/final-layer patch config from tensor shapes.
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
)


ANIMA_REQUIRED_KEYS = (
    "net.x_embedder.proj.1.weight",
    "net.t_embedder.1.linear_1.weight",
    "net.blocks.0.self_attn.q_proj.weight",
    "net.blocks.0.self_attn.output_proj.weight",
    "net.blocks.0.cross_attn.k_proj.weight",
    "net.blocks.0.cross_attn.output_proj.weight",
    "net.final_layer.linear.weight",
)


class AnimaDetector(ModelDetector):
    priority = 158

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        if not has_all_keys(bundle, *ANIMA_REQUIRED_KEYS):
            return False
        # Avoid matching WAN family shards accidentally (some exports mix checkpoints).
        if any(".head.modulation" in key for key in bundle.keys):
            return False
        return True

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        x_embedder_shape = _shape_2d(bundle, "net.x_embedder.proj.1.weight")
        if x_embedder_shape is None:
            raise RuntimeError("Anima detector missing x_embedder shape for 'net.x_embedder.proj.1.weight'")
        hidden_dim, x_in_dim = x_embedder_shape

        final_layer_shape = _shape_2d(bundle, "net.final_layer.linear.weight")
        if final_layer_shape is None:
            raise RuntimeError("Anima detector missing final_layer shape for 'net.final_layer.linear.weight'")
        final_out, hidden_dim_final = final_layer_shape
        if int(hidden_dim_final) != int(hidden_dim):
            raise RuntimeError(
                "Anima detector hidden_dim mismatch between x_embedder and final_layer. "
                f"x_embedder.hidden_dim={hidden_dim} final_layer.hidden_dim={hidden_dim_final}"
            )

        latent_channels, out_channels, patch_spatial, patch_temporal, concat_padding_mask = _infer_patch_config(
            x_in_dim=int(x_in_dim),
            final_out=int(final_out),
        )
        context_dim = _shape(bundle, "net.blocks.0.cross_attn.k_proj.weight", dim=1, default=None)
        if context_dim is None:
            raise RuntimeError("Anima detector missing context_dim shape for 'net.blocks.0.cross_attn.k_proj.weight'")
        if int(context_dim) <= 0:
            raise RuntimeError(f"Anima detector inferred invalid context_dim={context_dim}")
        depth = count_blocks(bundle.keys, "net.blocks.{}.")

        extras = {
            # Fixed Anima flow defaults for the supported image slice.
            "flow_shift": 3.0,
            "flow_multiplier": 1.0,
            # Dual tokenization: Qwen embeddings + T5XXL tokenizer ids/weights forwarded to the adapter (no T5 text encoder).
            "requires_t5xxl_ids": True,
            # PatchEmbed config inferred from core weights.
            "patch_spatial": int(patch_spatial),
            "patch_temporal": int(patch_temporal),
            "concat_padding_mask": bool(concat_padding_mask),
            "signature_source": "detector",
        }

        text_encoders = [
            TextEncoderSignature(
                name="qwen3_06b",
                key_prefix="text_encoder.",
                expected_dim=int(context_dim),
                tokenizer_hint="Qwen/Qwen3-0.6B",
            )
        ]

        return ModelSignature(
            family=ModelFamily.ANIMA,
            repo_hint="circlestone-labs/Anima",
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.FLOW16,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.DIT,
                channels_in=int(latent_channels),
                channels_out=int(out_channels),
                context_dim=int(context_dim),
                temporal=bool(int(patch_temporal) > 1),
                depth=depth,
                key_prefixes=["net.blocks."],
            ),
            text_encoders=text_encoders,
            vae=None,
            extras=extras,
        )


def _shape(bundle: SignalBundle, key: str, *, dim: int, default: int | None = None) -> Optional[int]:
    shape = bundle.shape(key)
    if shape is not None and len(shape) > dim:
        try:
            return int(shape[dim])
        except Exception:  # noqa: BLE001 - defensive
            return default
    tensor = bundle.state_dict.get(key)
    if isinstance(tensor, torch.Tensor) and tensor.ndim > dim:
        try:
            return int(tensor.shape[dim])
        except Exception:  # noqa: BLE001 - defensive
            return default
    return default


def _shape_2d(bundle: SignalBundle, key: str) -> Optional[Tuple[int, int]]:
    shape = bundle.shape(key)
    if shape is not None and len(shape) == 2:
        try:
            return int(shape[0]), int(shape[1])
        except Exception:  # noqa: BLE001 - defensive
            return None
    tensor = bundle.state_dict.get(key)
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
        try:
            return int(tensor.shape[0]), int(tensor.shape[1])
        except Exception:  # noqa: BLE001 - defensive
            return None
    return None


def _infer_patch_config(
    *,
    x_in_dim: int,
    final_out: int,
) -> Tuple[int, int, int, int, bool]:
    """Infer `(latent_channels, out_channels, patch_spatial, patch_temporal, concat_padding_mask)` from core shapes.

    Cosmos Predict2 PatchEmbed flattens `(C, T, H, W)` patches into a single dimension:
    `x_in_dim = (latent_channels + int(concat_padding_mask)) * patch_spatial^2 * patch_temporal`.

    The final projection mirrors this as:
    `final_out = out_channels * patch_spatial^2 * patch_temporal`.
    """
    if x_in_dim <= 0:
        raise RuntimeError(f"Invalid Anima x_embedder input dim: {x_in_dim}")
    if final_out <= 0:
        raise RuntimeError(f"Invalid Anima final_layer output dim: {final_out}")

    candidates: list[tuple[int, int, int, int, int, bool]] = []
    for patch_spatial in (1, 2, 4):
        for patch_temporal in (1, 2, 4):
            patch_area = patch_spatial * patch_spatial * patch_temporal
            if patch_area <= 0:
                continue
            if x_in_dim % patch_area != 0 or final_out % patch_area != 0:
                continue

            out_channels = final_out // patch_area
            latent_plus_mask = x_in_dim // patch_area
            for concat_padding_mask in (False, True):
                latent_channels = latent_plus_mask - (1 if concat_padding_mask else 0)
                if latent_channels <= 0:
                    continue
                if out_channels != latent_channels:
                    continue

                score = 0
                if patch_spatial == 2:
                    score += 10
                if patch_temporal == 1:
                    score += 5
                if concat_padding_mask:
                    score += 2
                if latent_channels == 16:
                    score += 3

                candidates.append(
                    (score, latent_channels, out_channels, patch_spatial, patch_temporal, concat_padding_mask)
                )

    if not candidates:
        raise RuntimeError(
            "Unable to infer Anima patch config from core shapes. "
            f"x_in_dim={x_in_dim} final_out={final_out}"
        )

    candidates.sort(reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == best[0] and candidates[1][1:] != best[1:]:
        top = ", ".join(
            f"(latent={c[1]} out={c[2]} patch={c[3]}x{c[3]}x{c[4]} mask={c[5]})" for c in candidates[:3]
        )
        raise RuntimeError(
            "Ambiguous Anima patch config inference. "
            f"x_in_dim={x_in_dim} final_out={final_out} candidates={top}"
        )

    _score, latent_channels, out_channels, patch_spatial, patch_temporal, concat_padding_mask = best
    return int(latent_channels), int(out_channels), int(patch_spatial), int(patch_temporal), bool(concat_padding_mask)


REGISTRY.register(AnimaDetector())
