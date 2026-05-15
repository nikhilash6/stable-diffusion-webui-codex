"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima LLMAdapter runtime (dual-tokenization adapter for text embeddings).
The adapter consumes:
- source embeddings (Qwen3 hidden states), and
- target token ids (T5XXL tokenizer ids),
and produces adapted cross-attention embeddings for the DiT core.

Symbols (top-level; keep in sync; no ghosts):
- `LLMAdapter` (class): Adapter module used by Anima to map `(text_embeds, text_ids)` → adapted embeddings.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LLMAdapterConfig
from .nn import RMSNorm


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q_or_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 1) -> torch.Tensor:
    original_dtype = q_or_k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    compute_dtype = torch.float32 if original_dtype in {torch.float16, torch.bfloat16} else original_dtype
    if q_or_k.dtype != compute_dtype:
        q_or_k = q_or_k.to(dtype=compute_dtype)
    if cos.dtype != compute_dtype:
        cos = cos.to(dtype=compute_dtype)
    if sin.dtype != compute_dtype:
        sin = sin.to(dtype=compute_dtype)
    out = (q_or_k * cos) + (_rotate_half(q_or_k) * sin)
    if out.dtype != original_dtype:
        out = out.to(dtype=original_dtype)
    return out


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, *, rope_theta: float = 10000.0) -> None:
        super().__init__()
        dim = int(head_dim)
        if dim <= 0:
            raise ValueError("head_dim must be > 0")
        self.rope_theta = float(rope_theta)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / float(dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError(f"position_ids must be 2D (B,S); got shape={tuple(position_ids.shape)}")
        inv_freq = self.inv_freq[None, :, None].to(device=x.device, dtype=torch.float32)
        pos = position_ids[:, None, :].to(device=x.device, dtype=torch.float32)
        device_type = str(x.device.type) if isinstance(x.device.type, str) else "cpu"
        if device_type not in {"cpu", "cuda"}:
            autocast_ctx = nullcontext()
        else:
            try:
                autocast_ctx = torch.autocast(device_type=device_type, enabled=False)
            except Exception:
                autocast_ctx = nullcontext()
        with autocast_ctx:
            freqs = (inv_freq @ pos).transpose(1, 2)  # (B,S,dim/2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos, sin


class Attention(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        layer_norm: bool,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        inner_dim = int(head_dim) * int(num_heads)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.query_dim = int(query_dim)
        self.context_dim = int(context_dim)

        self.q_proj = nn.Linear(self.query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = (
            nn.LayerNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)
            if layer_norm
            else RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        )

        self.k_proj = nn.Linear(self.context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = (
            nn.LayerNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)
            if layer_norm
            else RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        )

        self.v_proj = nn.Linear(self.context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(inner_dim, self.query_dim, bias=False, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor,
        attn_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        position_embeddings_context: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x as (B,S,C); got shape={tuple(x.shape)}")
        if context.ndim != 3:
            raise ValueError(f"Expected context as (B,S,C); got shape={tuple(context.shape)}")

        b, s_q, _ = x.shape
        b2, s_kv, _ = context.shape
        if b2 != b:
            raise ValueError(f"Batch mismatch: x.B={b} context.B={b2}")

        q = self.q_proj(x).view(b, s_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(b, s_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(b, s_kv, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if position_embeddings is not None:
            if position_embeddings_context is None:
                raise ValueError("position_embeddings_context is required when position_embeddings is provided")
            cos, sin = position_embeddings
            q = _apply_rotary(q, cos, sin, unsqueeze_dim=1)
            cos_ctx, sin_ctx = position_embeddings_context
            k = _apply_rotary(k, cos_ctx, sin_ctx, unsqueeze_dim=1)

        if attn_mask is not None:
            attn_mask = attn_mask.to(torch.bool)
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(b, s_q, self.num_heads * self.head_dim).contiguous()
        return self.o_proj(out)


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        source_dim: int,
        model_dim: int,
        num_heads: int,
        use_self_attn: bool,
        layer_norm: bool,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.use_self_attn = bool(use_self_attn)
        if int(num_heads) <= 0:
            raise ValueError("num_heads must be > 0")
        if (int(model_dim) % int(num_heads)) != 0:
            raise ValueError(f"model_dim must be divisible by num_heads (model_dim={model_dim} num_heads={num_heads})")

        if self.use_self_attn:
            self.norm_self_attn = (
                nn.LayerNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
                if layer_norm
                else RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
            )
            self.self_attn = Attention(
                query_dim=model_dim,
                context_dim=model_dim,
                num_heads=num_heads,
                head_dim=model_dim // num_heads,
                layer_norm=layer_norm,
                device=device,
                dtype=dtype,
            )

        self.norm_cross_attn = (
            nn.LayerNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
            if layer_norm
            else RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        )
        self.cross_attn = Attention(
            query_dim=model_dim,
            context_dim=source_dim,
            num_heads=num_heads,
            head_dim=model_dim // num_heads,
            layer_norm=layer_norm,
            device=device,
            dtype=dtype,
        )

        self.norm_mlp = (
            nn.LayerNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
            if layer_norm
            else RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        )
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * 4.0), device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(int(model_dim * 4.0), model_dim, device=device, dtype=dtype),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor,
        target_attn_mask: torch.Tensor | None,
        source_attn_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_context: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if self.use_self_attn:
            normed = self.norm_self_attn(x)
            x = x + self.self_attn(
                normed,
                context=normed,
                attn_mask=target_attn_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings,
            )

        normed = self.norm_cross_attn(x)
        x = x + self.cross_attn(
            normed,
            context=context,
            attn_mask=source_attn_mask,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )

        x = x + self.mlp(self.norm_mlp(x))
        return x


class LLMAdapter(nn.Module):
    def __init__(
        self,
        *,
        config: LLMAdapterConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if int(config.vocab_size) <= 0:
            raise ValueError("vocab_size must be > 0")
        if int(config.source_dim) <= 0:
            raise ValueError("source_dim must be > 0")
        if int(config.target_dim) <= 0:
            raise ValueError("target_dim must be > 0")
        if int(config.model_dim) <= 0:
            raise ValueError("model_dim must be > 0")
        if int(config.num_layers) <= 0:
            raise ValueError("num_layers must be >= 1")
        if int(config.num_heads) <= 0:
            raise ValueError("num_heads must be > 0")
        if (int(config.model_dim) % int(config.num_heads)) != 0:
            raise ValueError(
                f"model_dim must be divisible by num_heads (model_dim={int(config.model_dim)} num_heads={int(config.num_heads)})"
            )
        head_dim = int(config.model_dim) // int(config.num_heads)
        if (head_dim % 2) != 0:
            raise ValueError(f"RoPE head_dim must be even; got head_dim={head_dim} (model_dim/num_heads).")

        self.embed = nn.Embedding(int(config.vocab_size), int(config.target_dim), device=device, dtype=dtype)
        if int(config.model_dim) != int(config.target_dim):
            self.in_proj = nn.Linear(int(config.target_dim), int(config.model_dim), device=device, dtype=dtype)
        else:
            self.in_proj = nn.Identity()

        self.rotary_emb = RotaryEmbedding(int(config.model_dim) // int(config.num_heads))
        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    source_dim=int(config.source_dim),
                    model_dim=int(config.model_dim),
                    num_heads=int(config.num_heads),
                    use_self_attn=bool(config.use_self_attn),
                    layer_norm=bool(config.layer_norm),
                    device=device,
                    dtype=dtype,
                )
                for _ in range(int(config.num_layers))
            ]
        )
        self.out_proj = nn.Linear(int(config.model_dim), int(config.target_dim), device=device, dtype=dtype)
        self.norm = RMSNorm(int(config.target_dim), eps=1e-6, device=device, dtype=dtype)

    def forward(
        self,
        source_hidden_states: torch.Tensor,
        target_input_ids: torch.Tensor,
        *,
        target_attention_mask: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_input_ids.ndim != 2:
            raise ValueError(f"target_input_ids must be 2D (B,S); got shape={tuple(target_input_ids.shape)}")

        def _prep_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
            if mask is None:
                return None
            if mask.ndim != 2:
                raise ValueError(f"attention_mask must be 2D (B,S); got shape={tuple(mask.shape)}")
            return mask.to(torch.bool).unsqueeze(1).unsqueeze(1)

        target_mask = _prep_mask(target_attention_mask)
        source_mask = _prep_mask(source_attention_mask)

        x = self.in_proj(self.embed(target_input_ids))
        context = source_hidden_states

        pos_tgt = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        pos_ctx = torch.arange(context.shape[1], device=context.device).unsqueeze(0).expand(context.shape[0], -1)
        pe_tgt = self.rotary_emb(x, pos_tgt)
        pe_ctx = self.rotary_emb(x, pos_ctx)

        for block in self.blocks:
            x = block(
                x,
                context=context,
                target_attn_mask=target_mask,
                source_attn_mask=source_mask,
                position_embeddings=pe_tgt,
                position_embeddings_context=pe_ctx,
            )

        return self.norm(self.out_proj(x))
