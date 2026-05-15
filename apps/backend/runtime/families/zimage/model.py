"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z Image Turbo (Alibaba) transformer architecture (NextDiT/Lumina2-style) and state-dict loader.
Implements core blocks (norm/RoPE/attention/transformers/refiners) and loads checkpoints into `ZImageTransformer2DModel`.

Symbols (top-level; keep in sync; no ghosts):
- `ZImageConfig` (dataclass): Architecture/config defaults for ZImage transformer (dims, layers, RoPE axes, eps, etc).
- `RMSNorm` (class): RMSNorm layer used across transformer blocks (fp32 compute, dtype-preserving output).
- `TimestepEmbedder` (class): Timestep embedding module for diffusion timestep conditioning.
- `SwiGLU` (class): SwiGLU MLP block used inside transformer blocks.
- `RoPEEmbedding` (class): Rotary positional embedding builder for multi-axis RoPE.
- `apply_rotary_emb` (function): Applies rotary embeddings in fp32 using precomputed complex frequencies (dtype-preserving output).
- `apply_rope_pair` (function): Applies RoPE to a `(q, k)` pair and returns the rotated tensors.
- `Attention` (class): Attention module (QKV + optional norms + RoPE application; used by transformer blocks).
- `TransformerBlock` (class): Main transformer block (attn + MLP + residuals; composes the core model depth).
- `RefinerBlock` (class): Refiner-stage block used for late refinement layers.
- `NoiseRefinerBlock` (class): Noise-refiner variant block used for specific refinement passes.
- `FinalLayer` (class): Final projection layer mapping hidden states back to output channels/patches.
- `ZImageTransformer2DModel` (class): Full ZImage transformer model (owns embeddings, blocks, refiners, and forward pass).
- `load_zimage_from_state_dict` (function): Loads `ZImageTransformer2DModel` weights from a checkpoint state dict (strict fail-loud on missing/unexpected keys).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

# Key dimensions (from `z_image_turbo_bf16.safetensors`):
# - hidden_dim = 3840
# - context_dim = 2560
# - t_dim = 256 (timestep embedding intermediate)
# - head_dim = 128, num_heads = 30 (3840/128)
# - mlp_hidden = 10240
# - latent_channels = 16, patch_size = 2
# - num_layers = 30, num_refiner_layers = 2

import logging
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from apps.backend.runtime.attention import attention_function_pre_shaped
from apps.backend.runtime.memory.config import AttentionBackend
from apps.backend.runtime.misc.autocast import autocast_disabled
from .debug import env_flag, env_int, tensor_stats

logger = get_backend_logger("backend.runtime.zimage.model")

_DEFAULT_ZIMAGE_AXES_DIMS: tuple[int, int, int] = (32, 48, 48)
_SEQ_MULTI_OF = 32

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ZImageConfig:
    """Configuration for Z Image Transformer."""
    hidden_dim: int = 3840
    context_dim: int = 2560
    latent_channels: int = 16
    patch_size: int = 2
    num_layers: int = 30
    num_refiner_layers: int = 2
    num_heads: int = 30  # 3840 / 128
    head_dim: int = 128
    qk_norm: bool = True
    qkv_bias: bool = False
    out_bias: bool = False
    t_dim: int = 256  # Timestep embedding dimension
    mlp_hidden: int = 10240
    eps: float = 1e-5
    # HF config: apps/backend/huggingface/Tongyi-MAI/Z-Image-Turbo/transformer/config.json
    rope_theta: float = 256.0
    axes_dims: tuple[int, int, int] = (32, 48, 48)  # Must sum to head_dim
    axes_lens: tuple[int, int, int] = (1536, 512, 512)  # Max positions per axis
    t_scale: float = 1000.0
    
    @property
    def in_channels(self) -> int:
        """Input channels after patchification."""
        return self.latent_channels * self.patch_size * self.patch_size


# =============================================================================
# Core Layers
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(f"ZImage RMSNorm expects a floating-point input tensor; got dtype={x.dtype}.")
        dtype = x.dtype
        with autocast_disabled(x.device.type):
            x = x.float()
            norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return (norm * self.weight.float()).to(dtype)


class TimestepEmbedder(nn.Module):
    """Timestep embedding: sinusoidal -> MLP -> t_dim.
    
    Architecture from checkpoint:
    - Sinusoidal encoding: 256 dims
    - mlp.0: Linear(256, 1024)
    - SiLU
    - mlp.2: Linear(1024, 256) -> t_dim output
    """
    
    def __init__(self, t_dim: int = 256, frequency_dim: int = 256, mlp_hidden: int = 1024):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, t_dim),
        )
    
    def forward(self, t: torch.Tensor, dtype: torch.dtype = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.mlp[0].weight.dtype
        
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return self.mlp(emb.to(dtype))


class SwiGLU(nn.Module):
    """SwiGLU feedforward: w1(x) * silu(w3(x)) -> w2."""
    
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RoPEEmbedding(nn.Module):
    """3D Rotary Position Embedding using diffusers' complex polar format.
    
    This matches diffusers' RopeEmbedder from transformer_z_image.py exactly:
    - Uses torch.polar to create complex frequency embedding
    - Precomputes freqs_cis for axes_lens positions
    - Returns complex64 tensor for multiplication with Q/K
    
    Key parameters from HF config:
    - axes_dims: [32, 48, 48] (must sum to head_dim=128)
    - axes_lens: [1536, 512, 512] (max positions per axis)
    - theta: 256.0 (base frequency)
    """
    
    def __init__(
        self,
        head_dim: int,
        theta: float = 256.0,
        axes_dims: tuple[int, int, int] | None = None,
        axes_lens: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        
        # Default axes_dims from HF config
        if axes_dims is None:
            axes_dims = (32, 48, 48)  # Z Image Turbo default
        if sum(int(v) for v in axes_dims) != int(head_dim):
            raise ValueError(f"axes_dims must sum to head_dim={head_dim}; got {axes_dims}")
        self.axes_dims = tuple(int(v) for v in axes_dims)
        
        # Default axes_lens from HF config
        if axes_lens is None:
            axes_lens = (1536, 512, 512)  # Z Image Turbo default
        self.axes_lens = tuple(int(v) for v in axes_lens)
        
        # Precomputed freqs (will be populated on first forward)
        self.freqs_cis: list[torch.Tensor] | None = None
    
    def _precompute_freqs_cis(self, device: torch.device) -> list[torch.Tensor]:
        """Precompute complex freq embeddings for each axis.
        
        Matches diffusers RopeEmbedder.precompute_freqs_cis exactly.
        """
        freqs_cis = []
        for d, e in zip(self.axes_dims, self.axes_lens):
            # freqs = 1.0 / (theta ** (arange(0, d, 2) / d))
            freqs = 1.0 / (self.theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
            timestep = torch.arange(e, device="cpu", dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            # Convert to complex via polar form: e^(i*theta) = cos(theta) + i*sin(theta)
            freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
            freqs_cis.append(freqs_cis_i.to(device))
        return freqs_cis
    
    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        """Compute RoPE complex embeddings for 3D positions.
        
        Args:
            pos_ids: [B, N, num_axes] position IDs (Frame, Height, Width)
                     or [N, num_axes] for unbatched
        
        Returns:
            freqs: [B, N, head_dim//2] complex64 tensor 
                   (or [N, head_dim//2] for unbatched)
        """
        # Handle both batched [B, N, 3] and unbatched [N, 3] inputs
        was_batched = pos_ids.ndim == 3
        if not was_batched:
            pos_ids = pos_ids.unsqueeze(0)  # [1, N, 3]
        
        B, N, num_axes = pos_ids.shape
        device = pos_ids.device
        
        # Lazy precomputation
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_freqs_cis(device)
        else:
            # Ensure on correct device
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [fc.to(device) for fc in self.freqs_cis]
        
        # Gather freqs for each axis and concatenate
        result_list = []
        for i in range(min(num_axes, len(self.axes_dims))):
            idx = pos_ids[..., i].long()  # [B, N]
            # Clamp to valid range
            idx = idx.clamp(0, self.axes_lens[i] - 1)
            # Gather: freqs_cis[i] is [axes_lens[i], axes_dims[i]//2]
            # idx is [B, N], result is [B, N, axes_dims[i]//2]
            gathered = self.freqs_cis[i][idx.view(-1)].view(B, N, -1)
            result_list.append(gathered)
        
        # Concatenate along last dim: [B, N, sum(axes_dims)//2] = [B, N, head_dim//2]
        result = torch.cat(result_list, dim=-1)
        
        if not was_batched:
            result = result.squeeze(0)
        
        return result


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding using complex multiplication.
    
    Matches diffusers' ZSingleStreamAttnProcessor.apply_rotary_emb exactly.
    
    Args:
        x_in: [B, N, H, D] query or key tensor where D = head_dim
        freqs_cis: [B, N, D//2] complex64 freq embeddings
    
    Returns:
        Rotated tensor [B, N, H, D]
    """
    if not x_in.is_floating_point():
        raise TypeError(f"ZImage RoPE expects a floating-point input tensor; got dtype={x_in.dtype}.")
    with autocast_disabled(x_in.device.type):
        # Reshape x to complex: [B, N, H, D] -> [B, N, H, D//2] complex
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        # freqs_cis is [B, N, D//2], needs unsqueeze for head broadcast
        freqs_cis = freqs_cis.unsqueeze(2)  # [B, N, 1, D//2]
        # Complex multiply and convert back
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


def apply_rope_pair(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> tuple:
    """Apply rotary position embedding to query and key.
    
    Args:
        q: [B, N, H, D] query tensor
        k: [B, N, H, D] key tensor
        freqs: [B, N, D//2] complex freq embeddings
    
    Returns:
        Tuple of rotated (q, k) tensors [B, N, H, D]
    """
    return apply_rotary_emb(q, freqs), apply_rotary_emb(k, freqs)


class Attention(nn.Module):
    """Self-attention with combined QKV, QK normalization, and RoPE.
    
    Matches the JointAttention dimension ordering for RoPE.
    Q/K/V are [B, N, H, D] during RoPE, then [B, H, N, D] for SDPA.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        *,
        qk_norm: bool = True,
        qkv_bias: bool = False,
        out_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        
        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.out = nn.Linear(self.inner_dim, dim, bias=out_bias)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape

        debug_dtype = env_flag("CODEX_PIPELINE_DEBUG", False) and logger.isEnabledFor(logging.DEBUG)
        debug_stack = debug_dtype
        if debug_dtype:
            try:
                w = getattr(self.qkv, "weight", None)
                if isinstance(w, torch.Tensor) and getattr(w, "qtype", None) is not None:
                    if not getattr(self, "_zimage_dtype_logged_attn_qkv_gguf", False):
                        qtype = getattr(w, "qtype", None)
                        qtype_name = getattr(qtype, "name", str(qtype))
                        real_shape = getattr(w, "real_shape", None)
                        logger.debug(
                            "[zimage-dtype] Attention.qkv weight is GGUF packed: mat1=x dtype=%s; "
                            "mat2=qkv.weight storage_dtype=%s computation_dtype=%s qtype=%s storage_shape=%s real_shape=%s",
                            str(x.dtype),
                            str(w.dtype),
                            str(getattr(w, "computation_dtype", None)),
                            qtype_name,
                            tuple(w.shape),
                            tuple(real_shape) if real_shape is not None else None,
                        )
                        setattr(self, "_zimage_dtype_logged_attn_qkv_gguf", True)
                elif isinstance(w, torch.Tensor) and x.dtype != w.dtype:
                    stack = ""
                    if debug_stack:
                        import traceback as _traceback

                        stack = "\n" + "".join(_traceback.format_stack(limit=10))
                    logger.debug(
                        "[zimage-dtype] Attention.qkv matmul will mismatch: mat1=x dtype=%s shape=%s; mat2=qkv.weight dtype=%s shape=%s%s",
                        str(x.dtype),
                        tuple(x.shape),
                        str(w.dtype),
                        tuple(w.shape),
                        stack,
                    )
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("[zimage-dtype] Attention.qkv dtype debug failed", exc_info=True)
        
        # QKV projection and reshape to [B, N, 3, H, D]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # Split into [B, N, H, D] each (JointAttention layout)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # QK normalization (operates on last dim = head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply RoPE in [B, N, H, D] format
        if freqs is not None:
            q, k = apply_rope_pair(q, k, freqs)
        
        # Transpose to [B, H, N, D] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Match diffusers ZSingleStreamAttnProcessor: accept [B, seq_len] boolean masks
        # (True = keep, False = mask) and broadcast them across heads/queries.
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        out = attention_function_pre_shaped(
            q,
            k,
            v,
            mask=attention_mask,
            is_causal=False,
            backend=AttentionBackend.PYTORCH,
        )
        out = out.transpose(1, 2).reshape(B, N, self.inner_dim)
        if debug_dtype:
            try:
                w_out = getattr(self.out, "weight", None)
                if isinstance(w_out, torch.Tensor) and getattr(w_out, "qtype", None) is not None:
                    if not getattr(self, "_zimage_dtype_logged_attn_out_gguf", False):
                        qtype = getattr(w_out, "qtype", None)
                        qtype_name = getattr(qtype, "name", str(qtype))
                        real_shape = getattr(w_out, "real_shape", None)
                        logger.debug(
                            "[zimage-dtype] Attention.out weight is GGUF packed: mat1=sdpa_out dtype=%s; "
                            "mat2=out.weight storage_dtype=%s computation_dtype=%s qtype=%s storage_shape=%s real_shape=%s",
                            str(out.dtype),
                            str(w_out.dtype),
                            str(getattr(w_out, "computation_dtype", None)),
                            qtype_name,
                            tuple(w_out.shape),
                            tuple(real_shape) if real_shape is not None else None,
                        )
                        setattr(self, "_zimage_dtype_logged_attn_out_gguf", True)
                elif isinstance(w_out, torch.Tensor) and out.dtype != w_out.dtype:
                    stack = ""
                    if debug_stack:
                        import traceback as _traceback

                        stack = "\n" + "".join(_traceback.format_stack(limit=10))
                    logger.debug(
                        "[zimage-dtype] Attention.out matmul will mismatch: mat1=sdpa_out dtype=%s shape=%s; mat2=out.weight dtype=%s shape=%s%s",
                        str(out.dtype),
                        tuple(out.shape),
                        str(w_out.dtype),
                        tuple(w_out.shape),
                        stack,
                    )
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("[zimage-dtype] Attention.out dtype debug failed", exc_info=True)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Transformer block with adaLN modulation.
    
    Architecture matching NextDiT for z_image_modulation:
    - adaLN_modulation.0: Linear(min(dim, 256), 4 * dim) = Linear(256, 4*3840)
    - Outputs 4 modulation values: scale_msa, gate_msa, scale_mlp, gate_mlp
    - Uses tanh gating like NextDiT
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_hidden: int,
        t_dim: int = 256,
        eps: float = 1e-5,
        **kwargs,  # Ignore extra args like modulation_dim
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # z_image_modulation: Linear(min(dim, 256), 4 * dim)
        # For dim=3840, min(3840, 256) = 256, output = 4*3840 = 15360
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(t_dim, 4 * hidden_dim, bias=True),
        )
        
        # Attention norms
        self.attention_norm1 = RMSNorm(hidden_dim, eps=eps)
        self.attention_norm2 = RMSNorm(hidden_dim, eps=eps)
        
        # Attention
        self.attention = Attention(
            hidden_dim,
            num_heads,
            head_dim,
            eps=eps,
            qk_norm=bool(kwargs.get("qk_norm", True)),
            qkv_bias=kwargs.get("qkv_bias", False),
            out_bias=kwargs.get("out_bias", False),
        )
        
        # FFN norms
        self.ffn_norm1 = RMSNorm(hidden_dim, eps=eps)
        self.ffn_norm2 = RMSNorm(hidden_dim, eps=eps)
        
        # FFN
        self.feed_forward = SwiGLU(hidden_dim, mlp_hidden, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
        t_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t_emb is not None:
            # Get modulation values (4 * hidden_dim total)
            mod = self.adaLN_modulation(t_emb)  # [B, 4 * hidden_dim]
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
            
            # Attention with modulation (tanh gating like NextDiT)
            normed = self.attention_norm1(x)
            normed = normed * (1 + scale_msa.unsqueeze(1))
            attn_out = self.attention(normed, attention_mask, freqs)
            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(attn_out)
            
            # FFN with modulation
            normed = self.ffn_norm1(x)
            normed = normed * (1 + scale_mlp.unsqueeze(1))
            ffn_out = self.feed_forward(normed)
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(ffn_out)
        else:
            # Diffusers parity (ZImageTransformerBlock modulation=False):
            # - attn_out = attention(attention_norm1(x))
            # - x = x + attention_norm2(attn_out)
            attn_out = self.attention(self.attention_norm1(x), attention_mask, freqs)
            x = x + self.attention_norm2(attn_out)
            # - x = x + ffn_norm2(feed_forward(ffn_norm1(x)))
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        
        return x


class RefinerBlock(nn.Module):
    """Refiner block for context (no adaLN modulation)."""
    
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_hidden: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.attention_norm1 = RMSNorm(dim, eps=eps)
        self.attention_norm2 = RMSNorm(dim, eps=eps)
        self.attention = Attention(
            dim,
            num_heads,
            head_dim,
            eps=eps,
            qk_norm=bool(kwargs.get("qk_norm", True)),
            qkv_bias=kwargs.get("qkv_bias", False),
            out_bias=kwargs.get("out_bias", False),
        )
        self.ffn_norm1 = RMSNorm(dim, eps=eps)
        self.ffn_norm2 = RMSNorm(dim, eps=eps)
        self.feed_forward = SwiGLU(dim, mlp_hidden, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
        t_emb: Optional[torch.Tensor] = None,  # Ignored for context_refiner
    ) -> torch.Tensor:
        attn_out = self.attention(self.attention_norm1(x), attention_mask, freqs)
        x = x + self.attention_norm2(attn_out)
        x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class NoiseRefinerBlock(nn.Module):
    """Noise refiner block with adaLN modulation for Z Image.
    
    Unlike context_refiner, noise_refiner uses timestep-conditioned modulation
    similar to the main transformer blocks but with only 4 modulation values
    (scale_msa, gate_msa, scale_mlp, gate_mlp).
    """
    
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_hidden: int, 
                 t_dim: int = 256, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.dim = dim
        
        # adaLN modulation: t_dim -> 4 * dim (scale_msa, gate_msa, scale_mlp, gate_mlp)
        # For z_image_modulation, input is min(dim, 256) = 256
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(t_dim, 4 * dim, bias=True),
        )
        
        self.attention_norm1 = RMSNorm(dim, eps=eps)
        self.attention_norm2 = RMSNorm(dim, eps=eps)
        self.attention = Attention(
            dim,
            num_heads,
            head_dim,
            eps=eps,
            qk_norm=bool(kwargs.get("qk_norm", True)),
            qkv_bias=kwargs.get("qkv_bias", False),
            out_bias=kwargs.get("out_bias", False),
        )
        self.ffn_norm1 = RMSNorm(dim, eps=eps)
        self.ffn_norm2 = RMSNorm(dim, eps=eps)
        self.feed_forward = SwiGLU(dim, mlp_hidden, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
        t_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t_emb is not None:
            # Get modulation values
            mod = self.adaLN_modulation(t_emb)  # [B, 4 * dim]
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
            
            # Attention with modulation (tanh gating like NextDiT)
            normed = self.attention_norm1(x)
            normed = normed * (1 + scale_msa.unsqueeze(1))
            attn_out = self.attention(normed, attention_mask, freqs)
            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(attn_out)
            
            # FFN with modulation
            normed = self.ffn_norm1(x)
            normed = normed * (1 + scale_mlp.unsqueeze(1))
            ffn_out = self.feed_forward(normed)
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(ffn_out)
        else:
            attn_out = self.attention(self.attention_norm1(x), attention_mask, freqs)
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        
        return x


class FinalLayer(nn.Module):
    """Final layer with adaLN modulation.
    
    Uses LayerNorm with elementwise_affine=False (no learnable params) to match
    the GGUF checkpoint structure from NextDiT. The GGUF doesn't have weights for this norm
    layer because the reference implementation uses non-affine LayerNorm.
    """
    
    def __init__(self, hidden_dim: int, t_dim: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        # Non-affine LayerNorm: no weight/bias in the checkpoint for this norm
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)
        # Checkpoint: adaLN_modulation.1.weight is [hidden_dim, t_dim]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, hidden_dim, bias=True),  # Only scale, not shift+scale
        )
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale = self.adaLN_modulation(t_emb)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1))  # Only scale modulation
        return self.linear(x)



# =============================================================================
# Main Model
# =============================================================================

class ZImageTransformer2DModel(nn.Module):
    """Z Image Turbo Diffusion Transformer (NextDiT/Lumina2 style)."""
    
    def __init__(
        self,
        hidden_dim: int = 3840,
        context_dim: int = 2560,
        latent_channels: int = 16,
        patch_size: int = 2,
        num_layers: int = 30,
        num_refiner_layers: int = 2,
        num_heads: int = 30,
        head_dim: int = 128,
        qkv_bias: bool = False,  # Lumina uses bias=False for QKV
        out_bias: bool = False,
        t_dim: int = 256,
        mlp_hidden: int = 10240,
        eps: float = 1e-5,
        qk_norm: bool = True,
        rope_theta: float = 256.0,
        axes_dims: tuple[int, int, int] | None = None,
        axes_lens: tuple[int, int, int] | None = None,
        time_scale: float | None = None,
        config: Optional[ZImageConfig] = None,
        **kwargs,  # Ignore unknown HuggingFace config parameters
    ):
        super().__init__()

        # HuggingFace config compatibility: map common keys when present.
        # We keep canonical names (hidden_dim/context_dim/...) internally.
        if config is None:
            hidden_dim = int(kwargs.pop("dim", hidden_dim))
            context_dim = int(kwargs.pop("cap_feat_dim", context_dim))
            latent_channels = int(kwargs.pop("in_channels", latent_channels))
            if "all_patch_size" in kwargs and patch_size == 2:
                raw_patch = kwargs.pop("all_patch_size")
                if isinstance(raw_patch, (list, tuple)) and raw_patch:
                    patch_size = int(raw_patch[0])
            num_layers = int(kwargs.pop("n_layers", num_layers))
            num_refiner_layers = int(kwargs.pop("n_refiner_layers", num_refiner_layers))
            num_heads = int(kwargs.pop("n_heads", num_heads))
            eps = float(kwargs.pop("norm_eps", eps))
            qk_norm = bool(kwargs.pop("qk_norm", qk_norm))
            if axes_dims is None and "axes_dims" in kwargs:
                raw_axes = kwargs.pop("axes_dims")
                if isinstance(raw_axes, (list, tuple)) and len(raw_axes) == 3:
                    axes_dims = (int(raw_axes[0]), int(raw_axes[1]), int(raw_axes[2]))
            if axes_lens is None and "axes_lens" in kwargs:
                raw_lens = kwargs.pop("axes_lens")
                if isinstance(raw_lens, (list, tuple)) and len(raw_lens) == 3:
                    axes_lens = (int(raw_lens[0]), int(raw_lens[1]), int(raw_lens[2]))
            if time_scale is None and "t_scale" in kwargs:
                time_scale = float(kwargs.pop("t_scale"))

        # HF config key is "t_scale"; default to 1000.0 for Z Image Turbo.
        if time_scale is None:
            time_scale = 1000.0

        # Create config from resolved values if not provided
        if config is None:
            config = ZImageConfig(
                hidden_dim=hidden_dim,
                context_dim=context_dim,
                latent_channels=latent_channels,
                patch_size=patch_size,
                num_layers=num_layers,
                num_refiner_layers=num_refiner_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                t_dim=t_dim,
                mlp_hidden=mlp_hidden,
                eps=eps,
                rope_theta=rope_theta,
                axes_dims=axes_dims or _DEFAULT_ZIMAGE_AXES_DIMS,
                axes_lens=axes_lens or (1536, 512, 512),
                t_scale=float(time_scale),
                qk_norm=qk_norm,
                qkv_bias=qkv_bias,
                out_bias=out_bias,
            )
        self.config = config
        self.codex_config = SimpleNamespace(
            in_channels=int(config.latent_channels),
            context_dim=int(config.context_dim),
            adm_in_channels=None,
        )
        self.time_scale = float(getattr(config, "t_scale", float(time_scale)))
        
        self.patch_size = config.patch_size
        self.hidden_dim = config.hidden_dim
        self.latent_channels = config.latent_channels
        
        # Patch embedding
        self.x_embedder = nn.Linear(config.in_channels, config.hidden_dim, bias=True)
        
        # Caption embedding: RMSNorm(context_dim) + Linear(context_dim, hidden_dim)
        self.cap_embedder = nn.Sequential(
            RMSNorm(config.context_dim, eps=config.eps),
            nn.Linear(config.context_dim, config.hidden_dim, bias=True),
        )
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(t_dim=config.t_dim)
        
        # Padding tokens (2D in checkpoint: [1, hidden_dim])
        self.x_pad_token = nn.Parameter(torch.zeros(1, config.hidden_dim))
        self.cap_pad_token = nn.Parameter(torch.zeros(1, config.hidden_dim))
        
        # RoPE (axes_dims must sum to head_dim, axes_lens from HF config)
        self.rope = RoPEEmbedding(
            config.head_dim, 
            config.rope_theta, 
            axes_dims=getattr(config, "axes_dims", None),
            axes_lens=getattr(config, "axes_lens", None),
        )
        
        # Refiners use different hidden_dim (context_dim for context_refiner)
        # Actually, they share the same hidden_dim after embedding
        self.context_refiner = nn.ModuleList([
            RefinerBlock(config.hidden_dim, config.num_heads, config.head_dim, config.mlp_hidden, config.eps,
                         qk_norm=config.qk_norm, qkv_bias=config.qkv_bias, out_bias=config.out_bias)
            for _ in range(config.num_refiner_layers)
        ])
        
        self.noise_refiner = nn.ModuleList([
            NoiseRefinerBlock(config.hidden_dim, config.num_heads, config.head_dim, 
                              config.mlp_hidden, config.t_dim, config.eps,
                              qk_norm=config.qk_norm, qkv_bias=config.qkv_bias, out_bias=config.out_bias)
            for _ in range(config.num_refiner_layers)
        ])
        
        # Main transformer
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                mlp_hidden=config.mlp_hidden,
                t_dim=config.t_dim,
                eps=config.eps,
                qkv_bias=config.qkv_bias,
                out_bias=config.out_bias,
                qk_norm=config.qk_norm,
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer
        out_dim = config.patch_size * config.patch_size * config.latent_channels
        self.final_layer = FinalLayer(config.hidden_dim, config.t_dim, out_dim, config.eps)
        
        self.cnt = 0

        if env_flag("CODEX_ZIMAGE_DEBUG_CONFIG", False):
            logger.info(
                "[zimage-debug] core_config dim=%d context_dim=%d latent_channels=%d patch=%d layers=%d refiners=%d heads=%d head_dim=%d qk_norm=%s rope_theta=%s axes_dims=%s t_scale=%s",
                int(config.hidden_dim),
                int(config.context_dim),
                int(config.latent_channels),
                int(config.patch_size),
                int(config.num_layers),
                int(config.num_refiner_layers),
                int(config.num_heads),
                int(config.head_dim),
                bool(config.qk_norm),
                str(config.rope_theta),
                str(getattr(config, "axes_dims", None)),
                str(getattr(config, "t_scale", None)),
            )
    
    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype for Diffusers CPU offload compatibility."""
        # Return dtype of first parameter, or default to bfloat16
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.bfloat16
    
    def _patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        # Flatten x: [B, C, H, W] -> [B, H*W, C] (wrong)
        # Patchify: split into patches of size p
        # [B, C, H, W] -> [B, (H/p)*(W/p), C*p*p]
        
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f"Image size ({H}, {W}) must be divisible by patch size {p}"
        
        h_tokens = H // p
        w_tokens = W // p
        
        # Unfold/Reshape approach
        # [B, C, H, W] -> [B, C, h, p, w, p]
        x = x.reshape(B, C, h_tokens, p, w_tokens, p)
        # Permute to [B, h, w, p, p, C]
        x = x.permute(0, 2, 4, 3, 5, 1)
        # Flatten to [B, h*w, p*p*C]
        x = x.reshape(B, h_tokens * w_tokens, p * p * C)
        
        return x, (H, W)

    def _unpatchify(self, x: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
        # Diffusers unpatchify uses the FIRST `ori_len` tokens as image tokens, and the tail
        # contains caption tokens (and/or padded image tokens). Mirror that behavior here.
        H, W = original_size
        p = self.patch_size
        h_tokens = H // p
        w_tokens = W // p
        ori_len = h_tokens * w_tokens
        x = x[:, :ori_len, :]

        B = x.shape[0]
        C = self.latent_channels
        
        # [B, h*w, p*p*C] -> [B, h, w, p, p, C]
        x = x.reshape(B, h_tokens, w_tokens, p, p, C)
        # Permute to [B, C, h, p, w, p]
        x = x.permute(0, 5, 1, 3, 2, 4)
        # Reshape to [B, C, H, W]
        x = x.reshape(B, C, H, W)
        
        return x
    
    @staticmethod
    def _create_coordinate_grid(
        size: tuple[int, int, int],
        *,
        start: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Diffusers-parity coordinate grid builder (transformer_z_image.create_coordinate_grid)."""
        axes = [
            torch.arange(x0, x0 + span, dtype=torch.int32, device=device)
            for x0, span in zip(start, size)
        ]
        grids = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def _build_pos_ids(
        self,
        *,
        cap_total_len: int,
        h_tokens: int,
        w_tokens: int,
        image_total_len: int,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build unified position IDs matching diffusers Z-Image tokenization semantics.

        Notes
        - Token order is: image tokens first, caption tokens after.
        - Caption positions start at (1,0,0) and include multi-of-32 padding.
        - Image time coordinate starts at cap_total_len+1 (even though tokens come first).
        - Image padding positions (to multi-of-32) use (0,0,0).
        """
        cap_pos = self._create_coordinate_grid(
            (int(cap_total_len), 1, 1),
            start=(1, 0, 0),
            device=device,
        ).flatten(0, 2)

        # Image positions are (F,H,W) with F=1 for Z-Image Turbo.
        image_start_t = int(cap_total_len) + 1
        image_pos = self._create_coordinate_grid(
            (1, int(h_tokens), int(w_tokens)),
            start=(image_start_t, 0, 0),
            device=device,
        ).flatten(0, 2)
        image_ori_len = int(image_pos.shape[0])
        if int(image_total_len) > image_ori_len:
            pad = torch.zeros((int(image_total_len) - image_ori_len, 3), device=device, dtype=torch.int32)
            image_pos = torch.cat([image_pos, pad], dim=0)

        pos_ids = torch.cat([image_pos, cap_pos], dim=0)  # [N_total, 3]
        pos_ids = pos_ids.unsqueeze(0).expand(int(B), -1, -1).contiguous()
        return pos_ids
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        debug_dtype = env_flag("CODEX_PIPELINE_DEBUG", False) and logger.isEnabledFor(logging.DEBUG)
        debug_stack = debug_dtype

        # Handle 5D input
        if x.dim() == 5:
            x = x.squeeze(2)
            was_5d = True
        else:
            was_5d = False
        
        # Patchify image input
        img_patches, img_size = self._patchify(x)
        image_ori_len = int(img_patches.shape[1])
        image_padding_len = (-image_ori_len) % _SEQ_MULTI_OF
        image_total_len = image_ori_len + int(image_padding_len)
        if image_padding_len:
            img_patches = torch.cat(
                [img_patches, img_patches[:, -1:, :].repeat(1, int(image_padding_len), 1)],
                dim=1,
            )
        if debug_dtype:
            try:
                w = getattr(self.x_embedder, "weight", None)
                if isinstance(w, torch.Tensor) and getattr(w, "qtype", None) is not None:
                    if not getattr(self, "_zimage_dtype_logged_x_embedder_gguf", False):
                        qtype = getattr(w, "qtype", None)
                        qtype_name = getattr(qtype, "name", str(qtype))
                        real_shape = getattr(w, "real_shape", None)
                        logger.debug(
                            "[zimage-dtype] x_embedder weight is GGUF packed: mat1=img_patches dtype=%s; "
                            "mat2=x_embedder.weight storage_dtype=%s computation_dtype=%s qtype=%s storage_shape=%s "
                            "real_shape=%s",
                            str(img_patches.dtype),
                            str(w.dtype),
                            str(getattr(w, "computation_dtype", None)),
                            qtype_name,
                            tuple(w.shape),
                            tuple(real_shape) if real_shape is not None else None,
                        )
                        setattr(self, "_zimage_dtype_logged_x_embedder_gguf", True)
                elif isinstance(w, torch.Tensor) and img_patches.dtype != w.dtype:
                    stack = ""
                    if debug_stack:
                        import traceback as _traceback

                        stack = "\n" + "".join(_traceback.format_stack(limit=10))
                    logger.debug(
                        "[zimage-dtype] x_embedder matmul will mismatch: mat1=img_patches dtype=%s shape=%s; mat2=x_embedder.weight dtype=%s shape=%s%s",
                        str(img_patches.dtype),
                        tuple(img_patches.shape),
                        str(w.dtype),
                        tuple(w.shape),
                        stack,
                    )
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("[zimage-dtype] x_embedder dtype debug failed", exc_info=True)
        img_patches = self.x_embedder(img_patches)
        if image_padding_len:
            image_pad_mask = torch.zeros((int(img_patches.shape[0]), image_total_len), device=x.device, dtype=torch.bool)
            image_pad_mask[:, image_ori_len:] = True
            img_patches[image_pad_mask] = self.x_pad_token.type_as(img_patches)
        
        # Timestep embedding
        # Diffusers ZImagePipeline passes `t_norm = (1000 - timestep) / 1000`.
        # Our runtime uses a sigma-style timestep in [1→0]. For parity we invert:
        #   t_inv = 1 - sigma  (0 at start, 1 at end), then apply `t_scale` (default 1000.0).
        t_inv = 1.0 - timestep
        t_scaled = t_inv * self.time_scale
        t_emb = self.t_embedder(t_scaled, dtype=x.dtype)
        
        # Caption embedding
        cap_ori_len = int(context.shape[1])
        cap_padding_len = (-cap_ori_len) % _SEQ_MULTI_OF
        cap_total_len = cap_ori_len + int(cap_padding_len)
        if cap_padding_len:
            context = torch.cat(
                [context, context[:, -1:, :].repeat(1, int(cap_padding_len), 1)],
                dim=1,
            )
        cap_norm = self.cap_embedder[0](context)
        if debug_dtype:
            try:
                w = getattr(self.cap_embedder[1], "weight", None)
                if isinstance(w, torch.Tensor) and getattr(w, "qtype", None) is not None:
                    if not getattr(self, "_zimage_dtype_logged_cap_embedder_gguf", False):
                        qtype = getattr(w, "qtype", None)
                        qtype_name = getattr(qtype, "name", str(qtype))
                        real_shape = getattr(w, "real_shape", None)
                        logger.debug(
                            "[zimage-dtype] cap_embedder weight is GGUF packed: mat1=cap_norm dtype=%s; "
                            "mat2=cap_linear.weight storage_dtype=%s computation_dtype=%s qtype=%s "
                            "storage_shape=%s real_shape=%s",
                            str(cap_norm.dtype),
                            str(w.dtype),
                            str(getattr(w, "computation_dtype", None)),
                            qtype_name,
                            tuple(w.shape),
                            tuple(real_shape) if real_shape is not None else None,
                        )
                        setattr(self, "_zimage_dtype_logged_cap_embedder_gguf", True)
                elif isinstance(w, torch.Tensor) and cap_norm.dtype != w.dtype:
                    stack = ""
                    if debug_stack:
                        import traceback as _traceback

                        stack = "\n" + "".join(_traceback.format_stack(limit=10))
                    logger.debug(
                        "[zimage-dtype] cap_embedder matmul will mismatch: mat1=cap_norm dtype=%s shape=%s; mat2=cap_linear.weight dtype=%s shape=%s%s",
                        str(cap_norm.dtype),
                        tuple(cap_norm.shape),
                        str(w.dtype),
                        tuple(w.shape),
                        stack,
                    )
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("[zimage-dtype] cap_embedder dtype debug failed", exc_info=True)
        cap_feats = self.cap_embedder[1](cap_norm)
        if cap_padding_len:
            cap_pad_mask = torch.zeros((int(cap_feats.shape[0]), cap_total_len), device=x.device, dtype=torch.bool)
            cap_pad_mask[:, cap_ori_len:] = True
            cap_feats[cap_pad_mask] = self.cap_pad_token.type_as(cap_feats)
        
        debug_verbose = env_flag("CODEX_ZIMAGE_DEBUG_VERBOSE", False)
        debug_layers = env_flag("CODEX_ZIMAGE_DEBUG_LAYERS", False)
        debug_config = env_flag("CODEX_ZIMAGE_DEBUG_CONFIG", False)
        debug_enabled = debug_verbose or debug_layers or debug_config
        debug_limit = env_int("CODEX_ZIMAGE_DEBUG_STEPS", 3) if debug_enabled else 0
        debug_step = int(self.cnt)

        # DEBUG LOGS
        if debug_step < debug_limit:  # Only log first few steps
            logger.info(f"[zimage-debug] sigma: {timestep[0]:.4f}")
            logger.info(f"[zimage-debug] t_inv: {t_inv[0]:.4f}")
            logger.info(f"[zimage-debug] t_scaled: {t_scaled[0]:.4f}")
            logger.info(f"[zimage-debug] t_emb={t_emb[0, :8]}... range=[{t_emb.min():.2f}, {t_emb.max():.2f}]")
            logger.info(f"[zimage-debug] img_patches range=[{img_patches.min():.2f}, {img_patches.max():.2f}] mean={img_patches.mean():.4f}")
            logger.info(f"[zimage-debug] cap_feats range=[{cap_feats.min():.2f}, {cap_feats.max():.2f}] mean={cap_feats.mean():.4f}")
            if debug_verbose:
                logger.info("[zimage-debug] forward.kwargs keys=%s", sorted(str(k) for k in kwargs.keys()))
            
        # Position IDs
        B = int(img_patches.shape[0])
        h_tokens = (int(img_size[0]) + self.patch_size - 1) // self.patch_size
        w_tokens = (int(img_size[1]) + self.patch_size - 1) // self.patch_size
        pos_ids = self._build_pos_ids(
            cap_total_len=int(cap_total_len),
            h_tokens=int(h_tokens),
            w_tokens=int(w_tokens),
            image_total_len=int(image_total_len),
            B=B,
            device=x.device,
        )
        freqs = self.rope(pos_ids)
        if debug_layers and debug_step < debug_limit:
            tensor_stats(logger.name, "rope.pos_ids", pos_ids)
            # freqs is large but bounded; stats help spot dtype/device mismatches.
            tensor_stats(logger.name, "rope.freqs", freqs)
        
        # Diffusers order: refine image first, then caption. Token order in unified stream:
        # image tokens first, caption tokens after.
        attn_mask_img = torch.ones((B, int(image_total_len)), device=x.device, dtype=torch.bool)
        attn_mask_cap = torch.ones((B, int(cap_total_len)), device=x.device, dtype=torch.bool)

        for layer in self.noise_refiner:
            img_patches = layer(img_patches, attn_mask_img, freqs[:, : int(image_total_len)], t_emb)
        if debug_layers and debug_step < debug_limit:
            tensor_stats(logger.name, "after.noise_refiner", img_patches)

        for layer in self.context_refiner:
            cap_start = int(image_total_len)
            cap_end = cap_start + int(cap_total_len)
            cap_feats = layer(cap_feats, attn_mask_cap, freqs[:, cap_start:cap_end])
        if debug_layers and debug_step < debug_limit:
            tensor_stats(logger.name, "after.context_refiner", cap_feats)
        
        # Concatenate
        full_seq = torch.cat([img_patches, cap_feats], dim=1)
        if debug_layers and debug_step < debug_limit:
            tensor_stats(logger.name, "full_seq", full_seq)
        
        # Main transformer
        full_mask = torch.ones((B, int(full_seq.shape[1])), device=x.device, dtype=torch.bool)
        layer_every = max(1, env_int("CODEX_ZIMAGE_DEBUG_LAYER_EVERY", 10))
        for idx, layer in enumerate(self.layers):
            full_seq = layer(full_seq, full_mask, freqs, t_emb)
            if debug_layers and debug_step < debug_limit and ((idx == 0) or ((idx + 1) % layer_every == 0) or ((idx + 1) == len(self.layers))):
                tensor_stats(logger.name, f"layer.{idx+1:02d}.full_seq", full_seq)
        
        # Final projection
        output = self.final_layer(full_seq, t_emb)
        output = self._unpatchify(output, img_size)
        
        if was_5d:
            output = output.unsqueeze(2)
        
        # DEBUG: Log output statistics
        if debug_step < debug_limit:
            logger.info(f"[zimage-debug] output: range=[{output.min():.4f}, {output.max():.4f}] mean={output.mean():.4f} norm={output.norm():.2f}")

        self.cnt = debug_step + 1
        
        # CRITICAL: Negate output for sigma-space sampler compatibility.
        # 
        # Flow-matching model predicts velocity v = dx/dt = noise - x_0
        # For denoising to move toward x_0, the sampler needs to step in direction of -v.
        #
        # Euler update rule (sigma-space, "const" prediction):
        #   denoised = x - model_output * sigma  (from 'const' prediction type)
        #   eps = (x - denoised) / sigma = model_output
        #   x_new = x - (sigma - sigma_next) * eps = x - dt * model_output
        #
        # If model returns +v:
        #   x_new = x - dt * v  → moves AWAY from x_0 (wrong!)
        # 
        # If model returns -v:
        #   x_new = x - dt * (-v) = x + dt * v → moves TOWARD x_0 (correct!)
        #
        # Log 6 confirmed: without negation, norm(x) increased 566→1305 (diverging).
        return -output


# =============================================================================
# Model Loading
# =============================================================================

def load_zimage_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    config: Optional[ZImageConfig] = None,
) -> ZImageTransformer2DModel:
    """Load Z Image model from state dict with automatic config detection."""

    if config is None:
        from .inference import infer_zimage_dims_from_state_dict

        dims = infer_zimage_dims_from_state_dict(state_dict, patch_size=2)
        logger.info(
            "Detected: hidden=%d context=%d t_dim=%d layers=%d refiner=%d heads=%d mlp=%d",
            dims.hidden_dim,
            dims.context_dim,
            dims.t_dim,
            dims.num_layers,
            dims.num_refiner_layers,
            dims.num_heads,
            dims.mlp_hidden,
        )
        config = ZImageConfig(
            hidden_dim=dims.hidden_dim,
            context_dim=dims.context_dim,
            t_dim=dims.t_dim,
            num_layers=dims.num_layers,
            num_refiner_layers=dims.num_refiner_layers,
            num_heads=dims.num_heads,
            mlp_hidden=dims.mlp_hidden,
        )
    
    model = ZImageTransformer2DModel(config=config)
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "ZImage transformer strict load failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
        )
    
    return model


QwenImageTransformer2DModel = ZImageTransformer2DModel
