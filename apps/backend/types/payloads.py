"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Payload validation key set definitions.
Defines frozen key groups for SHA selection, txt2img, and extras payloads (including ER-SDE/guidance/IP-Adapter option envelopes and the generation `settings_revision` contract key) and exposes singleton instances used by request validators.

Symbols (top-level; keep in sync; no ghosts):
- `ShaKeys` (dataclass): Frozen key groups for SHA256-based asset selection payload fields.
- `Txt2ImgKeys` (dataclass): Frozen key groups for txt2img payload fields (CORE/DIFFUSION/FLOW/HIRES, SMART flags, and `settings_revision` contract key).
- `ExtrasKeys` (dataclass): Frozen key groups for `payload.extras` fields (includes Z-Image Turbo/Base `zimage_variant` plus optional shared `er_sde`/`guidance`/`ip_adapter` options).
- `SHA_KEYS` (constant): Singleton instance of `ShaKeys`.
- `TXT2IMG_KEYS` (constant): Singleton instance of `Txt2ImgKeys`.
- `EXTRAS_KEYS` (constant): Singleton instance of `ExtrasKeys`.
- `__all__` (constant): Explicit export list for this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class ShaKeys:
    """SHA256 keys for asset selection."""
    MODEL: FrozenSet[str] = frozenset({"model_sha"})
    TENC: FrozenSet[str] = frozenset({"tenc_sha", "tenc1_sha", "tenc2_sha"})
    VAE: FrozenSet[str] = frozenset({"vae_sha"})
    LORA: FrozenSet[str] = frozenset({"lora_sha"})
    
    @property
    def ALL(self) -> FrozenSet[str]:
        return self.MODEL | self.TENC | self.VAE | self.LORA


@dataclass(frozen=True)
class Txt2ImgKeys:
    """Keys for txt2img payload."""
    
    # Generation params (all models)
    CORE: FrozenSet[str] = frozenset({
        "prompt",
        "width",
        "height",
        "steps",
        "sampler",
        "scheduler",
        "seed",
        "clip_skip",
        "styles",
        "metadata",
    })
    
    # Diffusion (SD, SDXL)
    DIFFUSION: FrozenSet[str] = frozenset({
        "negative_prompt",
        "cfg",
    })
    
    # Flow (Flux, Z Image)
    FLOW: FrozenSet[str] = frozenset({
        "distilled_cfg",
    })
    
    # Nested extras.hires request keys
    HIRES: FrozenSet[str] = frozenset({
        "enable",
        "denoise",
        "scale",
        "resize_x",
        "resize_y",
        "steps",
        "upscaler",
        "tile",
        "swap_model",
        "sampler",
        "scheduler",
        "prompt",
        "negative_prompt",
        "cfg",
        "refiner",
        "distilled_cfg",
    })
    
    # Infra
    DEVICE: FrozenSet[str] = frozenset({"device"})
    MODEL: FrozenSet[str] = frozenset({"engine", "model"})
    SMART: FrozenSet[str] = frozenset({"smart_offload", "smart_fallback", "smart_cache"})
    REVISION: FrozenSet[str] = frozenset({"settings_revision"})
    
    # Extras container (passed through to engine)
    EXTRAS: FrozenSet[str] = frozenset({"extras"})
    
    @property
    def ALL(self) -> FrozenSet[str]:
        return self.CORE | self.DIFFUSION | self.FLOW | self.DEVICE | self.MODEL | self.SMART | self.REVISION | self.EXTRAS
    

@dataclass(frozen=True)
class ExtrasKeys:
    """Keys for payload.extras."""
    
    COMMON: FrozenSet[str] = frozenset({
        "hires",
        "swap_model",
        "refiner",
        "ip_adapter",
        "text_encoder_override",
        "batch_size",
        "batch_count",
        "eta_noise_seed_delta",
        "zimage_variant",
        "er_sde",
        "guidance",
    })
    
    @property
    def ALL(self) -> FrozenSet[str]:
        return self.COMMON | SHA_KEYS.ALL


# Singletons
SHA_KEYS = ShaKeys()
TXT2IMG_KEYS = Txt2ImgKeys()
EXTRAS_KEYS = ExtrasKeys()

__all__ = ["ShaKeys", "Txt2ImgKeys", "ExtrasKeys", "SHA_KEYS", "TXT2IMG_KEYS", "EXTRAS_KEYS"]
