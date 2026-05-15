"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native Qwen3 text encoder implementation used by the ZImage runtime.
This is a standalone implementation that avoids `transformers`, enabling GGUF support through the quantization system. Supports an optional
`compute_dtype` attribute to cast embeddings for stable fp32 compute while keeping storage dtype separate.

Symbols (top-level; keep in sync; no ghosts):
- `Qwen3Config` (dataclass): Configuration defaults for Qwen3 variants (dims/layers/heads/RoPE/norm eps).
- `RMSNorm` (class): RMSNorm implementation used throughout the encoder (fp32 compute, dtype-preserving output).
- `rotate_half` (function): Helper for RoPE rotation (splits and rotates half-dims).
- `apply_rotary_pos_emb` (function): Applies rotary positional embeddings in fp32 and preserves `(q, k)` dtypes.
- `RotaryEmbedding` (class): Builds/caches fp32 rotary embedding frequency tensors for a given head dim and max positions.
- `Attention` (class): Attention module (GQA support + optional Q/K norm + SDPA).
- `MLP` (class): SwiGLU feed-forward block.
- `TransformerBlock` (class): One transformer layer (attn + MLP + residual/norm plumbing).
- `Qwen3Model` (class): Core Qwen3 transformer stack (embeddings + blocks + forward).
- `Qwen3_4B` (class): Convenience wrapper for the 4B variant (loads config, provides encode-style forward usage, strict load contract in `load_sd`).
- `Qwen3_06B` (class): Convenience wrapper for the 0.6B variant (Anima text encoder; 1024-dim, 28 layers, strict load contract in `load_sd`).
- `resolve_qwen3_gguf_keyspace` (function): Resolves GGUF tensor keys into this implementation’s lookup keyspace without materializing a renamed state dict.
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

# Architecture notes (Qwen3-4B):
# - hidden_size: 2560
# - intermediate_size: 9728
# - num_hidden_layers: 36
# - num_attention_heads: 32
# - num_key_value_heads: 8 (GQA)
# - RoPE theta: 1_000_000
# - Q/K normalization (Gemma3 style)
# - SwiGLU activation

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from apps.backend.runtime.attention import attention_function_pre_shaped
from apps.backend.runtime.memory.config import AttentionBackend
from apps.backend.runtime.misc.autocast import autocast_disabled

logger = get_backend_logger("backend.runtime.zimage.qwen3")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Qwen3Config:
    """Configuration for Qwen3-4B model."""
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    head_dim: int = 128  # hidden_size // num_attention_heads = 80, but Qwen uses 128
    qkv_bias: bool = False
    use_qk_norm: bool = True  # Qwen3 uses Q/K normalization


# =============================================================================
# Core Layers
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(f"Qwen3 RMSNorm expects a floating-point input tensor; got dtype={x.dtype}.")
        dtype = x.dtype
        with autocast_disabled(x.device.type):
            x = x.float()
            norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            out = norm * self.weight.float()
            return out.to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K."""
    if not q.is_floating_point() or not k.is_floating_point():
        raise TypeError(f"RoPE expects floating-point Q/K tensors; got q.dtype={q.dtype} k.dtype={k.dtype}.")
    if not cos.is_floating_point() or not sin.is_floating_point():
        raise TypeError(f"RoPE expects floating-point cos/sin tensors; got cos.dtype={cos.dtype} sin.dtype={sin.dtype}.")
    q_dtype = q.dtype
    k_dtype = k.dtype
    with autocast_disabled(q.device.type):
        q_float = q.float()
        k_float = k.float()
        cos_float = cos.float()
        sin_float = sin.float()
        q_embed = (q_float * cos_float) + (rotate_half(q_float) * sin_float)
        k_embed = (k_float * cos_float) + (rotate_half(k_float) * sin_float)
        return q_embed.to(dtype=q_dtype), k_embed.to(dtype=k_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 40960, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache cos/sin
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len].to(dtype=torch.float32),
            self.sin_cached[:seq_len].to(dtype=torch.float32),
        )


class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA)."""
    
    def __init__(self, config: Qwen3Config, layer_idx: int = 0, ops=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        ops = ops or nn

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        # Projections
        self.q_proj = ops.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = ops.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = ops.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = ops.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Q/K normalization (Qwen3 style)
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply Q/K normalization
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(v, seq_len)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Repeat K, V for GQA
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Scaled dot-product attention
        # IMPORTANT: Always use is_causal=False! The causal mask is already in attention_mask.
        # Construct causal mask manually (required by this implementation).
        attn_output = attention_function_pre_shaped(
            q,
            k,
            v,
            mask=attention_mask,
            is_causal=False,  # Causal mask is in attention_mask, not is_causal flag
            backend=AttentionBackend.PYTORCH,
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class MLP(nn.Module):
    """SwiGLU MLP."""
    
    def __init__(self, config: Qwen3Config, ops=None):
        super().__init__()
        ops = ops or nn
        self.gate_proj = ops.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = ops.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = ops.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization."""
    
    def __init__(self, config: Qwen3Config, layer_idx: int, ops=None):
        super().__init__()
        self.self_attn = Attention(config, layer_idx, ops=ops)
        self.mlp = MLP(config, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return hidden_states


# =============================================================================
# Main Model
# =============================================================================

class Qwen3Model(nn.Module):
    """Qwen3 transformer model (without LM head)."""
    
    def __init__(self, config: Qwen3Config, ops=None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        ops = ops or nn
        
        self.embed_tokens = ops.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i, ops=ops)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def _build_attention_mask(
        self,
        *,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # NOTE: We intentionally use a *finite* sentinel value (instead of `-inf`) so the combined
        # causal+padding mask stays numerically stable and avoids NaN-producing `0 * -inf` paths.
        mask_value = torch.finfo(dtype).min / 4
        causal_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype, device=device)
        causal_mask = causal_mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1),
            mask_value,
        )
        if attention_mask is None:
            return causal_mask
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch_size, seq_len):
            raise ValueError(f"attention_mask must be [B, L]; got shape={tuple(attention_mask.shape)}")
        is_padding = attention_mask.to(device=device).view(batch_size, 1, 1, seq_len) == 0
        return causal_mask.expand(batch_size, 1, seq_len, seq_len).masked_fill(is_padding, mask_value)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        all_hidden_states = [] if output_hidden_states else None
        
        batch_size = int(hidden_states.shape[0])
        seq_len = int(hidden_states.shape[1])
        causal_mask = self._build_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            attention_mask=attention_mask,
        )
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, causal_mask)
        
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        return hidden_states, all_hidden_states


class Qwen3_4B(nn.Module):
    """Qwen3-4B for text encoding.
    
    This is a wrapper that provides the same interface as transformers models
    but uses our native implementation.
    """
    
    def __init__(self, config: Optional[Qwen3Config] = None, dtype=None, device=None, ops=None):
        super().__init__()
        config = config or Qwen3Config()
        self.config = config
        self.model = Qwen3Model(config, ops=ops)
        self.num_layers = config.num_hidden_layers
        self.dtype = dtype
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value: nn.Embedding):
        self.model.set_input_embeddings(value)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        intermediate_output: Optional[int] = None,
        final_layer_norm_intermediate: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional intermediate output.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Pre-computed embeddings (alternative to input_ids)
            intermediate_output: Layer index to extract intermediate output from
            final_layer_norm_intermediate: Whether to apply final norm to intermediate
        
        Returns:
            Tuple of (final_hidden_states, intermediate_hidden_states)
        """
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        if inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        hidden_states = inputs_embeds
        compute_dtype = getattr(self, "compute_dtype", None)
        if isinstance(compute_dtype, torch.dtype) and compute_dtype != hidden_states.dtype:
            hidden_states = hidden_states.to(dtype=compute_dtype)
        intermediate = None

        batch_size = int(hidden_states.shape[0])
        seq_len = int(hidden_states.shape[1])
        causal_mask = self.model._build_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            attention_mask=attention_mask,
        )

        if intermediate_output is not None and intermediate_output < 0:
            intermediate_output = len(self.model.layers) + int(intermediate_output)

        for i, layer in enumerate(self.model.layers):
            hidden_states = layer(hidden_states, causal_mask)
            if intermediate_output is not None and i == intermediate_output:
                intermediate = hidden_states.clone()

        hidden_states = self.model.norm(hidden_states)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.model.norm(intermediate)
        return hidden_states, intermediate
    
    def load_sd(self, state_dict: Mapping[str, object]) -> Tuple[List[str], List[str]]:
        """Load state dict with keyspace resolution for GGUF compatibility.
        
        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Qwen3 strict load failed: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
            )
        return missing, unexpected

class Qwen3_06B(nn.Module):
    """Qwen3-0.6B for text encoding (Anima).

    Uses the same core implementation as Qwen3_4B with a different config.
    """

    def __init__(self, config: Optional[Qwen3Config] = None, dtype=None, device=None, ops=None):
        super().__init__()
        config = config or Qwen3Config(
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            max_position_embeddings=32768,
        )
        self.config = config
        self.model = Qwen3Model(config, ops=ops)
        self.num_layers = config.num_hidden_layers
        self.dtype = dtype

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        intermediate_output: Optional[int] = None,
        final_layer_norm_intermediate: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        if inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        hidden_states = inputs_embeds
        compute_dtype = getattr(self, "compute_dtype", None)
        if isinstance(compute_dtype, torch.dtype) and compute_dtype != hidden_states.dtype:
            hidden_states = hidden_states.to(dtype=compute_dtype)
        intermediate = None

        batch_size = int(hidden_states.shape[0])
        seq_len = int(hidden_states.shape[1])
        causal_mask = self.model._build_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            attention_mask=attention_mask,
        )

        if intermediate_output is not None and intermediate_output < 0:
            intermediate_output = len(self.model.layers) + int(intermediate_output)

        for i, layer in enumerate(self.model.layers):
            hidden_states = layer(hidden_states, causal_mask)
            if intermediate_output is not None and i == intermediate_output:
                intermediate = hidden_states.clone()

        hidden_states = self.model.norm(hidden_states)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.model.norm(intermediate)
        return hidden_states, intermediate

    def load_sd(self, state_dict: Mapping[str, object]) -> Tuple[List[str], List[str]]:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Qwen3-0.6B strict load failed: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
            )
        return missing, unexpected


# =============================================================================
# GGUF Key Mapping
# =============================================================================

def resolve_qwen3_gguf_keyspace(
    gguf_state_dict: Mapping[str, object],
    num_layers: int = 36,
) -> Mapping[str, object]:
    """Resolve llama.cpp-style GGUF tensor keys into this Qwen3 implementation’s lookup keyspace.

    This is strict by default: if the input looks like llama.cpp GGUF keys (`token_embd.weight`, `blk.N.*`), all keys must be understood.
    """

    from apps.backend.runtime.state_dict.keymap_llama_gguf import (
        QWEN3_LLAMA_GGUF_LAYER_SUFFIX_TO_HF_PREFIX,
        resolve_llama_gguf_text_model_keyspace,
    )

    resolved = resolve_llama_gguf_text_model_keyspace(
        gguf_state_dict,
        num_layers=num_layers,
        layer_suffix_to_hf_prefix=QWEN3_LLAMA_GGUF_LAYER_SUFFIX_TO_HF_PREFIX,
    )
    style = resolved.style
    style_label = style.value if hasattr(style, "value") else str(style)
    emit_backend_message(
        "Qwen3 keyspace: detected style",
        logger=logger.name,
        level=logging.DEBUG,
        style=style_label,
    )
    return resolved.view


__all__ = [
    "Qwen3Config",
    "Qwen3_4B",
    "Qwen3_06B",
    "Qwen3Model",
    "resolve_qwen3_gguf_keyspace",
]
