"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: FLUX.2 family helpers used by the truthful backend runtime seam.
Exports the older centered-latent bridge (`model.py`) plus the single-Qwen FLUX.2 text encoder wrapper (`text_encoder.py`).
The live FLUX.2 engine/img2img seam now imports `runtime.py` directly for normalized external latents + `image_latents`
conditioning, so this package surface is no longer the authoritative adapter entry point for img2img.

Symbols (top-level; keep in sync; no ghosts):
- `FLUX2_LATENT_CHANNELS` (constant): Raw FLUX.2 VAE latent channels.
- `FLUX2_PATCH_CHANNELS` (constant): Patchified FLUX.2 transformer channels.
- `Flux2KleinTransformerAdapter` (class): Sampler-facing FLUX.2 transformer adapter on centered raw 32-channel latents.
- `Flux2NoisePrediction` (class): FlowMatch predictor with FLUX.2-specific raw-space noise initialization.
- `decode_flux2_external_latents` (function): Decode centered raw FLUX.2 sampler latents into pixels.
- `encode_flux2_external_latents` (function): Encode pixels into centered raw FLUX.2 sampler latents.
- `flux2_decode_offset_raw` (function): Convert the FLUX.2 patch-space batch-norm mean into raw latent offset.
- `flux2_patch_bn_stats` (function): Read/validate FLUX.2 VAE patch-space batch-norm stats.
- `pack_flux2_latents` (function): Convert BCHW patch-space latents into `(B, HW, C)` token layout.
- `patchify_flux2_latents` (function): Convert raw 32-channel BCHW latents into 128-channel patch space.
- `prepare_flux2_latent_ids` (function): Build FLUX.2 image position ids `(B, HW, 4)`.
- `prepare_flux2_text_ids` (function): Build FLUX.2 text position ids `(B, S, 4)`.
- `unpack_flux2_latents` (function): Convert `(B, HW, C)` tokens back into BCHW patch-space latents.
- `unpatchify_flux2_latents` (function): Convert 128-channel patch-space latents back into raw 32-channel space.
- `FLUX2_QWEN_HIDDEN_LAYERS` (constant): Intermediate Qwen hidden-state layers concatenated for FLUX.2 conditioning.
- `FLUX2_QWEN_HIDDEN_SIZE` (constant): Supported Qwen hidden size for the truthful FLUX.2 slice.
- `Flux2TextEncoder` (class): Qwen3-4B wrapper for FLUX.2 text encoding.
- `Flux2TextProcessingEngine` (class): Thin callable prompt-embedding adapter around `Flux2TextEncoder`.
"""

from __future__ import annotations

from .model import (
    FLUX2_LATENT_CHANNELS,
    FLUX2_PATCH_CHANNELS,
    Flux2KleinTransformerAdapter,
    Flux2NoisePrediction,
    decode_flux2_external_latents,
    encode_flux2_external_latents,
    flux2_decode_offset_raw,
    flux2_patch_bn_stats,
    pack_flux2_latents,
    patchify_flux2_latents,
    prepare_flux2_latent_ids,
    prepare_flux2_text_ids,
    unpack_flux2_latents,
    unpatchify_flux2_latents,
)
from .text_encoder import (
    FLUX2_QWEN_HIDDEN_LAYERS,
    FLUX2_QWEN_HIDDEN_SIZE,
    Flux2TextEncoder,
    Flux2TextProcessingEngine,
)

__all__ = [
    "FLUX2_LATENT_CHANNELS",
    "FLUX2_PATCH_CHANNELS",
    "FLUX2_QWEN_HIDDEN_LAYERS",
    "FLUX2_QWEN_HIDDEN_SIZE",
    "Flux2KleinTransformerAdapter",
    "Flux2NoisePrediction",
    "Flux2TextEncoder",
    "Flux2TextProcessingEngine",
    "decode_flux2_external_latents",
    "encode_flux2_external_latents",
    "flux2_decode_offset_raw",
    "flux2_patch_bn_stats",
    "pack_flux2_latents",
    "patchify_flux2_latents",
    "prepare_flux2_latent_ids",
    "prepare_flux2_text_ids",
    "unpack_flux2_latents",
    "unpatchify_flux2_latents",
]
