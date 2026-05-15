"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime state-dict views and helpers.
Provides mapping views and small utilities used during checkpoint loading/normalization.

Symbols (top-level; keep in sync; no ghosts):
- `keymap_llama_gguf` (module): Keyspace resolver helpers for llama.cpp-style GGUF tensor names.
- `keymap_flux2_transformer` (module): FLUX.2 transformer runtime/native key-style resolver (`runtime-export`/native → Diffusers lookup keys).
- `keymap_flux_transformer` (module): Flux transformer runtime/native key-style resolver (`runtime-export`/native → Codex runtime lookup keys).
- `keymap_qwen_text_encoder` (module): Qwen text-encoder key-style detection + strict keyspace resolver to canonical `model.*` backbone keys.
- `keymap_sdxl_checkpoint` (module): SDXL checkpoint wrapper/prefix key normalization (checkpoint-wrapper / original SDXL layout).
- `keymap_sdxl_clip` (module): SDXL base text-encoder key mapping (CLIP-L/CLIP-G → Codex IntegratedCLIP layout).
- `keymap_sdxl_vae` (module): SDXL/Flow16 VAE key-style resolver (LDM-style → diffusers AutoencoderKL).
- `keymap_t5_text_encoder` (module): T5 text-encoder key-style resolver (HF `encoder.*`/`shared.weight` → IntegratedT5 `transformer.*`).
- `keymap_wan21_vae` (module): WAN2.1 VAE key-style detection + strict canonical resolver.
- `keymap_wan22_vae` (module): WAN22 VAE key-style detection + strict keyspace resolvers for 2D/3D lanes.
- `keymap_wan22_transformer` (module): WAN22 transformer key-style detection + canonical keyspace resolver (Diffusers/WAN-export/Codex).
- `keymap_zimage_transformer` (module): Z-Image transformer runtime/native key-style resolver (`runtime-export`/native → Codex runtime lookup keys).
- `key_mapping` (module): Strict key-style detection + canonical keyspace resolver primitives.
- `tools` (module): State-dict diagnostics and helper utilities.
- `views` (module): Lightweight mapping views for state_dict handling.
"""

__all__ = [
    "keymap_flux2_transformer",
    "keymap_flux_transformer",
    "keymap_llama_gguf",
    "keymap_qwen_text_encoder",
    "keymap_sdxl_checkpoint",
    "keymap_sdxl_clip",
    "keymap_sdxl_vae",
    "keymap_t5_text_encoder",
    "keymap_wan21_vae",
    "keymap_wan22_vae",
    "keymap_wan22_transformer",
    "keymap_zimage_transformer",
    "key_mapping",
    "tools",
    "views",
]
