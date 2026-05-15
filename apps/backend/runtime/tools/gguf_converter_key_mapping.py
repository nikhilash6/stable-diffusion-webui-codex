"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Hugging Face → GGUF key mapping helpers for the GGUF converter.
Provides deterministic tensor-name translation for known architectures (layer-indexed mappings).

Symbols (top-level; keep in sync; no ghosts):
- `build_key_mapping` (function): Build the full HuggingFace → GGUF key mapping for a model given number of layers.
"""

from __future__ import annotations

from typing import Dict


# Key mappings: HuggingFace → GGUF
HF_TO_GGUF_KEYS = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}


def _get_layer_key_mapping(layer_idx: int) -> Dict[str, str]:
    """Get key mappings for a specific layer."""

    prefix_hf = f"model.layers.{layer_idx}"
    prefix_gguf = f"blk.{layer_idx}"

    return {
        f"{prefix_hf}.self_attn.q_proj.weight": f"{prefix_gguf}.attn_q.weight",
        f"{prefix_hf}.self_attn.q_proj.bias": f"{prefix_gguf}.attn_q.bias",
        f"{prefix_hf}.self_attn.k_proj.weight": f"{prefix_gguf}.attn_k.weight",
        f"{prefix_hf}.self_attn.k_proj.bias": f"{prefix_gguf}.attn_k.bias",
        f"{prefix_hf}.self_attn.v_proj.weight": f"{prefix_gguf}.attn_v.weight",
        f"{prefix_hf}.self_attn.v_proj.bias": f"{prefix_gguf}.attn_v.bias",
        f"{prefix_hf}.self_attn.o_proj.weight": f"{prefix_gguf}.attn_output.weight",
        f"{prefix_hf}.self_attn.o_proj.bias": f"{prefix_gguf}.attn_output.bias",
        f"{prefix_hf}.self_attn.q_norm.weight": f"{prefix_gguf}.attn_q_norm.weight",
        f"{prefix_hf}.self_attn.q_norm.bias": f"{prefix_gguf}.attn_q_norm.bias",
        f"{prefix_hf}.self_attn.k_norm.weight": f"{prefix_gguf}.attn_k_norm.weight",
        f"{prefix_hf}.self_attn.k_norm.bias": f"{prefix_gguf}.attn_k_norm.bias",
        f"{prefix_hf}.mlp.gate_proj.weight": f"{prefix_gguf}.ffn_gate.weight",
        f"{prefix_hf}.mlp.gate_proj.bias": f"{prefix_gguf}.ffn_gate.bias",
        f"{prefix_hf}.mlp.up_proj.weight": f"{prefix_gguf}.ffn_up.weight",
        f"{prefix_hf}.mlp.up_proj.bias": f"{prefix_gguf}.ffn_up.bias",
        f"{prefix_hf}.mlp.down_proj.weight": f"{prefix_gguf}.ffn_down.weight",
        f"{prefix_hf}.mlp.down_proj.bias": f"{prefix_gguf}.ffn_down.bias",
        f"{prefix_hf}.input_layernorm.weight": f"{prefix_gguf}.attn_norm.weight",
        f"{prefix_hf}.input_layernorm.bias": f"{prefix_gguf}.attn_norm.bias",
        f"{prefix_hf}.post_attention_layernorm.weight": f"{prefix_gguf}.ffn_norm.weight",
        f"{prefix_hf}.post_attention_layernorm.bias": f"{prefix_gguf}.ffn_norm.bias",
    }


def build_key_mapping(num_layers: int) -> Dict[str, str]:
    """Build complete HuggingFace → GGUF key mapping."""

    mapping = dict(HF_TO_GGUF_KEYS)
    for i in range(num_layers):
        mapping.update(_get_layer_key_mapping(i))
    return mapping


__all__ = [
    "build_key_mapping",
]
