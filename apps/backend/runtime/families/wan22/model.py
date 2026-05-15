"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 Transformer model as nn.Module (format-agnostic).
Provides `WanTransformer2DModel`, a native PyTorch implementation of the WAN diffusion transformer that can load weights from
multiple formats (GGUF, safetensors, etc.) via the operations registry; GGUF handling is transparent through `CodexOperationsGGUF`.

Symbols (top-level; keep in sync; no ghosts):
- `WanArchitectureConfig` (dataclass): Architecture hyperparameters for WAN (dims/heads/blocks/patch size/etc) used for construction/inference.
- `_wan22_fp32_compute_mode` (function): Returns cached `CODEX_WAN22_FP32_COMPUTE` mode (`auto|on|off`) from bootstrap-aware env parsing.
- `_wan22_use_fp32_compute` (function): Resolves whether WAN norms/residual hot paths should run with full-tensor fp32 casts for a given input dtype.
- `_wan22_norm_compute_label` (function): Emits trace label (`float32` or `native`) for the effective WAN norm compute mode.
- `_WAN22_FFN_ACTIVATION_BUDGET_BYTES` (constant): Default byte budget used to bound FFN fc1 activation footprint via sequence chunking.
- `_wan22_resolve_ffn_chunk_tokens` (function): Determines token chunk size for FFN to reduce peak VRAM on long sequences.
- `WanRMSNorm` (class): RMSNorm with optional GGUF dequantization and env-controlled compute policy (`fp32` vs native dtype).
- `WanFP32LayerNorm` (class): LayerNorm with env-controlled compute policy (`fp32` vs native dtype).
- `WanRotaryPosEmbed` (class): Rotary positional embedding (RoPE) cache + per-input embedding builder (fp32 caches).
- `WanSelfAttention` (class): Self-attention block for WAN (QKV projection + SDPA implementation).
- `WanCrossAttention` (class): Cross-attention block for WAN (text context attention path).
- `WanFFN` (class): Feed-forward (MLP) block used in WAN transformer blocks.
- `WanTransformerBlock` (class): One transformer block combining attention + FFN + norms/residuals.
- `WanTransformer2DModel` (class): Full WAN transformer stack (embeddings/blocks/forward); used by `runtime/wan22/wan22.py`.
- `resolve_wan22_gguf_keyspace` (function): Resolves WAN22 transformer checkpoint keys into this module’s expected parameter lookup space (Diffusers/WAN-export/Codex).
- `infer_wan_architecture_from_state_dict` (function): Infers `WanArchitectureConfig` from a loaded state dict (dims/layers/heads).
- `load_wan_transformer_from_state_dict` (function): Constructs `WanTransformer2DModel` and loads weights from a state dict (with strict fail-loud key mismatch handling).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from apps.backend.infra.config.env_flags import env_flag, env_str
from apps.backend.runtime.attention.sram import (
    SramAttentionContractError,
    is_rope_helper_available,
    try_attention_pre_shaped,
)
from apps.backend.runtime.misc.autocast import autocast_disabled
from apps.backend.runtime.ops.operations import get_operation_context
from apps.backend.runtime.ops.operations_gguf import CodexParameter, dequantize_tensor as gguf_dequantize_tensor
from apps.backend.runtime.sampling.block_progress import resolve_block_progress_callback

from .inference import infer_wan22_latent_channels, infer_wan22_patch_embedding, infer_wan22_patch_size_and_in_channels
from .sdpa import get_sram_attention_mode, sdpa as wan_sdpa

logger = get_backend_logger("backend.runtime.wan22.model")


def _wan_trace_verbose_enabled() -> bool:
    return env_flag("CODEX_TRACE_INFERENCE_DEBUG", default=False)


@lru_cache(maxsize=1)
def _wan22_fp32_compute_mode() -> str:
    return env_str(
        "CODEX_WAN22_FP32_COMPUTE",
        default="auto",
        allowed={"auto", "on", "off"},
    )


def _wan22_use_fp32_compute(input_dtype: torch.dtype) -> bool:
    mode = _wan22_fp32_compute_mode()
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode == "auto":
        return input_dtype == torch.float16
    raise RuntimeError(f"Unsupported CODEX_WAN22_FP32_COMPUTE mode: {mode!r}")


def _wan22_norm_compute_label(input_dtype: torch.dtype) -> str:
    return "float32" if _wan22_use_fp32_compute(input_dtype) else "native"


_WAN22_FFN_ACTIVATION_BUDGET_BYTES = 128 * 1024 * 1024


def _wan22_resolve_ffn_chunk_tokens(*, x_blc: torch.Tensor, hidden_dim: int) -> int:
    """Returns a token chunk size for WAN22 FFN that bounds the fc1 activation footprint.

    The FFN is token-independent (operates on the last dim), so chunking over sequence
    preserves outputs while reducing peak VRAM. This is a perf/memory tradeoff tuned
    for 12GB-class GPUs.
    """
    if not x_blc.is_cuda:
        return 0
    if x_blc.ndim != 3:
        return 0
    batch, seq_len, _channels = (int(x_blc.shape[0]), int(x_blc.shape[1]), int(x_blc.shape[2]))
    if seq_len <= 0 or batch <= 0 or hidden_dim <= 0:
        return 0

    bytes_per_token = int(hidden_dim) * int(x_blc.element_size())
    denom = batch * bytes_per_token
    if denom <= 0:
        return 0
    max_tokens = int(_WAN22_FFN_ACTIVATION_BUDGET_BYTES // denom)
    if max_tokens <= 0:
        return 1
    if seq_len <= max_tokens:
        return 0

    for candidate in (4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
        if candidate <= max_tokens:
            return candidate
    return 1


def _module_parameter_dtype(module: nn.Module) -> str:
    for param in module.parameters(recurse=True):
        return str(param.dtype)
    return "<no-params>"


def _module_parameter_device(module: nn.Module) -> str:
    for param in module.parameters(recurse=True):
        return str(param.device)
    return "<no-params>"


def _cuda_mem_snapshot_str(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cuda_mem=n/a"
    try:
        alloc_mb = float(torch.cuda.memory_allocated(device)) / (1024**2)
        reserved_mb = float(torch.cuda.memory_reserved(device)) / (1024**2)
        max_alloc_mb = float(torch.cuda.max_memory_allocated(device)) / (1024**2)
        max_reserved_mb = float(torch.cuda.max_memory_reserved(device)) / (1024**2)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_mb = float(free_bytes) / (1024**2)
        total_mb = float(total_bytes) / (1024**2)
        return (
            f"alloc={alloc_mb:.0f}MB reserved={reserved_mb:.0f}MB free={free_mb:.0f}MB total={total_mb:.0f}MB "
            f"max_alloc={max_alloc_mb:.0f}MB max_reserved={max_reserved_mb:.0f}MB"
        )
    except Exception:
        return "cuda_mem=unavailable"


# Configuration
@dataclass(frozen=True)
class WanArchitectureConfig:
    """Configuration for WAN transformer architecture."""

    d_model: int = 5120
    # WAN2.2 commonly uses head_dim=128 => n_heads=d_model//128 (ex.: 5120 -> 40).
    n_heads: int = 40
    n_blocks: int = 40
    mlp_ratio: float = 4.0
    context_dim: int = 4096
    time_embed_dim: int = 256
    rope_max_seq_len: int = 1024
    patch_size: Tuple[int, int, int] = (1, 2, 2)  # T, H, W
    in_channels: int = 16
    latent_channels: int = 16
    qkv_bias: bool = True
    use_text_projection: bool = True
    use_guidance: bool = True


# Submodules
class WanRMSNorm(nn.Module):
    """RMS normalization with learned weight (WAN GGUF uses affine RMSNorm weights)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(f"WanRMSNorm expects a floating-point input tensor; got dtype={x.dtype}.")
        original_dtype = x.dtype
        use_fp32_compute = _wan22_use_fp32_compute(original_dtype)

        w = self.weight
        if isinstance(w, CodexParameter) and w.qtype is not None:
            w = gguf_dequantize_tensor(w)
        if not torch.is_tensor(w):
            w = torch.as_tensor(w)
        w = w.to(device=x.device)

        if use_fp32_compute:
            with autocast_disabled(x.device.type):
                x_fp32 = x.float()
                w_fp32 = w.float()
                out = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                out = out * w_fp32
                return out.to(dtype=original_dtype)

        weight_native = w.to(device=x.device, dtype=original_dtype)
        return F.rms_norm(x, (self.dim,), weight_native, self.eps)


class WanFP32LayerNorm(nn.LayerNorm):
    """LayerNorm with env-controlled fp32-vs-native compute policy."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        use_fp32_compute = _wan22_use_fp32_compute(inputs.dtype)
        if not use_fp32_compute:
            weight_native = self.weight.to(device=inputs.device, dtype=inputs.dtype) if self.weight is not None else None
            bias_native = self.bias.to(device=inputs.device, dtype=inputs.dtype) if self.bias is not None else None
            return F.layer_norm(
                inputs,
                self.normalized_shape,
                weight_native,
                bias_native,
                self.eps,
            )

        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


def _wan_1d_rope_cos_sin(
    dim: int,
    max_seq_len: int,
    *,
    theta: float = 10000.0,
    freqs_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(dim) % 2 != 0:
        raise ValueError(f"WAN RoPE: dim must be even; got dim={dim}")
    if int(max_seq_len) <= 0:
        raise ValueError(f"WAN RoPE: max_seq_len must be > 0; got max_seq_len={max_seq_len}")

    pos = torch.arange(int(max_seq_len), device=device, dtype=freqs_dtype)
    freqs = 1.0 / (float(theta) ** (torch.arange(0, int(dim), 2, device=device, dtype=freqs_dtype) / float(dim)))
    freqs = torch.outer(pos, freqs)  # [S, dim/2]
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1).to(torch.float32)  # [S, dim]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1).to(torch.float32)  # [S, dim]
    return freqs_cos, freqs_sin


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        *,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if attention_head_dim <= 0:
            raise ValueError(f"WAN RoPE: attention_head_dim must be > 0; got {attention_head_dim}")
        if attention_head_dim % 2 != 0:
            raise ValueError(
                f"WAN RoPE: attention_head_dim must be even (pairs for complex rotation); got {attention_head_dim}"
            )
        self.attention_head_dim = int(attention_head_dim)
        self.patch_size = tuple(int(x) for x in patch_size)
        self.max_seq_len = int(max_seq_len)

        h_dim = w_dim = 2 * (self.attention_head_dim // 6)
        t_dim = self.attention_head_dim - h_dim - w_dim
        if t_dim <= 0 or t_dim % 2 != 0:
            raise ValueError(
                "WAN RoPE: invalid head_dim split "
                f"(head_dim={self.attention_head_dim}, t_dim={t_dim}, h_dim={h_dim}, w_dim={w_dim})"
            )

        self.t_dim = int(t_dim)
        self.h_dim = int(h_dim)
        self.w_dim = int(w_dim)

        freqs_dtype = torch.float32

        freqs_cos = []
        freqs_sin = []
        for dim in (self.t_dim, self.h_dim, self.w_dim):
            cos, sin = _wan_1d_rope_cos_sin(
                dim,
                self.max_seq_len,
                theta=theta,
                freqs_dtype=freqs_dtype,
                device=torch.device("cpu"),
            )
            freqs_cos.append(cos)
            freqs_sin.append(sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 5:
            raise ValueError(f"WAN RoPE: expected hidden_states [B,C,T,H,W], got shape={tuple(hidden_states.shape)}")
        _b, _c, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size

        if int(num_frames) % int(p_t) != 0:
            raise ValueError(f"WAN RoPE: num_frames={num_frames} not divisible by patch_t={p_t}")
        if int(height) % int(p_h) != 0:
            raise ValueError(f"WAN RoPE: height={height} not divisible by patch_h={p_h}")
        if int(width) % int(p_w) != 0:
            raise ValueError(f"WAN RoPE: width={width} not divisible by patch_w={p_w}")

        ppf, pph, ppw = int(num_frames) // int(p_t), int(height) // int(p_h), int(width) // int(p_w)
        if ppf > self.max_seq_len or pph > self.max_seq_len or ppw > self.max_seq_len:
            raise ValueError(
                "WAN RoPE: token grid exceeds rope cache "
                f"(ppf={ppf}, pph={pph}, ppw={ppw}, rope_max_seq_len={self.max_seq_len})"
            )

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]
        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos_out = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin_out = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos_out.to(device=hidden_states.device), freqs_sin_out.to(device=hidden_states.device)


class WanSelfAttention(nn.Module):
    """Self-attention layer for WAN transformer."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.o = nn.Linear(dim, dim, bias=True)

        # q/k RMSNorm is applied across the full packed head dimension (dim = head_dim * heads).
        self.norm_q = WanRMSNorm(dim)
        self.norm_k = WanRMSNorm(dim)

    def _apply_rope(
        self,
        hidden_states: torch.Tensor,  # [B, L, H, D]
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        *,
        sram_mode: str,
    ) -> torch.Tensor:
        if not hidden_states.is_floating_point():
            raise TypeError(
                "WAN RoPE expects floating-point hidden states; "
                f"got dtype={hidden_states.dtype} shape={tuple(hidden_states.shape)}."
            )
        if (
            hidden_states.is_cuda
            and hidden_states.is_contiguous()
            and hidden_states.dtype in {torch.float16, torch.bfloat16}
            and freqs_cos.is_cuda
            and freqs_sin.is_cuda
            and freqs_cos.dtype == torch.float32
            and freqs_sin.dtype == torch.float32
            and freqs_cos.is_contiguous()
            and freqs_sin.is_contiguous()
            and freqs_cos.ndim == 4
            and freqs_sin.ndim == 4
            and freqs_cos.shape == freqs_sin.shape
            and freqs_cos.shape[0] == 1
            and freqs_cos.shape[2] == 1
            and freqs_cos.shape[1] == hidden_states.shape[1]
            and freqs_cos.shape[3] == hidden_states.shape[3]
        ):
            # Prefer the fused CUDA RoPE op when available. This is an in-place update
            # of [B,L,H,D] hidden_states and avoids fp32 full-tensor materialization.
            if is_rope_helper_available(mode=sram_mode):
                torch.ops.attention_sram_v1.rope_blhd_(hidden_states, freqs_cos, freqs_sin)
                return hidden_states

        with autocast_disabled(hidden_states.device.type):
            # Keep RoPE math in fp32 for parity, but avoid materializing fp32 copies of the
            # full Q/K tensors (`.float()` here can be ~GB-scale for WAN22 long sequences).
            x1 = hidden_states[..., 0::2]
            x2 = hidden_states[..., 1::2]
            cos = freqs_cos[..., 0::2]
            sin = freqs_sin[..., 1::2]

            tmp = torch.empty_like(x1, dtype=torch.float32)
            torch.mul(x1, cos, out=tmp)
            tmp.addcmul_(x2, sin, value=-1.0)
        out = torch.empty_like(hidden_states)
        out[..., 0::2].copy_(tmp)
        torch.mul(x1, sin, out=tmp)
        tmp.addcmul_(x2, cos, value=1.0)
        out[..., 1::2].copy_(tmp)
        return out

    def forward(
        self,
        x: torch.Tensor,
        *,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        trace_debug: bool = False,
        trace_block_idx: int | None = None,
    ) -> torch.Tensor:
        B, L, C = x.shape
        trace_enabled = bool(trace_debug and logger.isEnabledFor(logging.DEBUG))
        trace_mem_detail = bool(trace_enabled and trace_block_idx is not None and int(trace_block_idx) == 0)
        block_tag = "?" if trace_block_idx is None else str(int(trace_block_idx) + 1)

        sram_mode = get_sram_attention_mode()

        # QKV projections
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        if trace_enabled:
            q_norm_compute = _wan22_norm_compute_label(q.dtype)
            k_norm_compute = _wan22_norm_compute_label(k.dtype)
            logger.debug(
                "[wan22.trace] block[%s] self_attn qkv: x_dtype=%s x_device=%s q_dtype=%s q_device=%s k_dtype=%s k_device=%s "
                "v_dtype=%s v_device=%s "
                "norm_q=WanRMSNorm(compute=%s) norm_k=WanRMSNorm(compute=%s)",
                block_tag,
                str(x.dtype),
                str(x.device),
                str(q.dtype),
                str(q.device),
                str(k.dtype),
                str(k.device),
                str(v.dtype),
                str(v.device),
                q_norm_compute,
                k_norm_compute,
            )

        # Reshape to heads (keep token-major layout for RoPE parity with Diffusers)
        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim)

        if rotary_emb is not None:
            if trace_mem_detail:
                logger.debug(
                    "[wan22.trace] block[%s] self_attn pre_rope: %s",
                    block_tag,
                    _cuda_mem_snapshot_str(x.device),
                )
            freqs_cos, freqs_sin = rotary_emb
            q = self._apply_rope(q, freqs_cos, freqs_sin, sram_mode=sram_mode)
            if trace_mem_detail:
                logger.debug(
                    "[wan22.trace] block[%s] self_attn post_rope_q: %s",
                    block_tag,
                    _cuda_mem_snapshot_str(x.device),
                )
            k = self._apply_rope(k, freqs_cos, freqs_sin, sram_mode=sram_mode)
            if trace_mem_detail:
                logger.debug(
                    "[wan22.trace] block[%s] self_attn post_rope_k: %s",
                    block_tag,
                    _cuda_mem_snapshot_str(x.device),
                )

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if sram_mode != "off":
            try:
                sram_result = try_attention_pre_shaped(
                    mode=sram_mode,
                    q=q,
                    k=k,
                    v=v,
                    is_causal=False,
                )
                if sram_result.output is not None:
                    if trace_enabled:
                        logger.debug(
                            "[wan22.trace] block[%s] self_attn sram path used: mode=%s",
                            block_tag,
                            sram_mode,
                        )
                    attn_out = sram_result.output
                else:
                    if trace_enabled:
                        logger.debug(
                            "[wan22.trace] block[%s] self_attn sram fallback: mode=%s reason=%s detail=%s",
                            block_tag,
                            sram_mode,
                            sram_result.reason_code,
                            sram_result.reason_detail,
                        )
                    attn_out = wan_sdpa(q, k, v, causal=False)
            except SramAttentionContractError as ex:
                if sram_mode == "force":
                    raise
                if trace_enabled:
                    logger.debug(
                        "[wan22.trace] block[%s] self_attn sram contract fallback: mode=%s detail=%s",
                        block_tag,
                        sram_mode,
                        str(ex),
                    )
                attn_out = wan_sdpa(q, k, v, causal=False)
        else:
            attn_out = wan_sdpa(q, k, v, causal=False)
        if trace_mem_detail:
            logger.debug(
                "[wan22.trace] block[%s] self_attn post_sdpa: %s",
                block_tag,
                _cuda_mem_snapshot_str(x.device),
            )

        # Merge heads
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        if trace_mem_detail:
            logger.debug(
                "[wan22.trace] block[%s] self_attn post_merge_heads: %s",
                block_tag,
                _cuda_mem_snapshot_str(x.device),
            )

        out = self.o(attn_out)
        return out


class WanCrossAttention(nn.Module):
    """Cross-attention layer for WAN transformer."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.o = nn.Linear(dim, dim, bias=True)

        # q/k RMSNorm is applied across the full packed head dimension (dim = head_dim * heads).
        self.norm_q = WanRMSNorm(dim)
        self.norm_k = WanRMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        trace_debug: bool = False,
        trace_block_idx: int | None = None,
    ) -> torch.Tensor:
        B, L, C = x.shape
        _, S, _ = context.shape
        trace_enabled = bool(trace_debug and logger.isEnabledFor(logging.DEBUG))
        block_tag = "?" if trace_block_idx is None else str(int(trace_block_idx) + 1)

        sram_mode = get_sram_attention_mode()

        # QKV projections
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)

        if trace_enabled:
            q_norm_compute = _wan22_norm_compute_label(q.dtype)
            k_norm_compute = _wan22_norm_compute_label(k.dtype)
            logger.debug(
                "[wan22.trace] block[%s] cross_attn qkv: x_dtype=%s x_device=%s ctx_dtype=%s ctx_device=%s "
                "q_dtype=%s q_device=%s k_dtype=%s k_device=%s v_dtype=%s v_device=%s "
                "norm_q=WanRMSNorm(compute=%s) norm_k=WanRMSNorm(compute=%s)",
                block_tag,
                str(x.dtype),
                str(x.device),
                str(context.dtype),
                str(context.device),
                str(q.dtype),
                str(q.device),
                str(k.dtype),
                str(k.device),
                str(v.dtype),
                str(v.device),
                q_norm_compute,
                k_norm_compute,
            )

        # Reshape to heads
        q = q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if trace_enabled and sram_mode != "off":
            logger.debug(
                "[wan22.trace] block[%s] cross_attn sram bypass: mode=%s detail=phase1_self_only",
                block_tag,
                sram_mode,
            )

        # Scaled dot-product attention
        attn_out = wan_sdpa(q, k, v, causal=False)

        # Merge heads
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, C)

        out = self.o(attn_out)
        return out


class WanFFN(nn.Module):
    """Feed-forward network with SiLU activation."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Apply modulation if provided
        if scale is not None or shift is not None:
            if shift is not None:
                x = x + shift[:, None, :]
            if scale is not None:
                x = x * (1 + scale[:, None, :])

        x = self.fc1(x)
        x = x * torch.sigmoid(x)  # SiLU
        x = self.fc2(x)
        return x


class WanTransformerBlock(nn.Module):
    """Single transformer block for WAN model.

    Each block contains:
    - Self-attention with modulation
    - Cross-attention with context
    - Feed-forward with modulation
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        # WAN GGUF semantics:
        # - norm1/norm2: LayerNorm without affine (pre-norm for SA/FFN)
        # - norm3: LayerNorm with affine (pre-norm for CA)
        self.norm1 = WanFP32LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm2 = WanFP32LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm3 = WanFP32LayerNorm(dim, eps=1e-6, elementwise_affine=True)

        self.self_attn = WanSelfAttention(dim, num_heads, qkv_bias)
        self.cross_attn = WanCrossAttention(dim, context_dim, num_heads, qkv_bias)

        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, dim, bias=True),
        )

        # Per-block modulation: [1, 6, dim] for [sa_shift, sa_scale, sa_gate, ffn_shift, ffn_scale, ffn_gate]
        # Matches Diffusers `WanTransformerBlock.scale_shift_table` and upstream WAN exports.
        op_ctx = get_operation_context()
        modulation_kwargs = {}
        if op_ctx.device is not None:
            modulation_kwargs["device"] = op_ctx.device
        if op_ctx.dtype is not None:
            modulation_kwargs["dtype"] = op_ctx.dtype
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim, **modulation_kwargs))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time_emb: torch.Tensor,  # [B, 6, dim]
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        trace_block_idx: int | None = None,
        trace_debug: bool = False,
    ) -> torch.Tensor:
        trace_enabled = bool(trace_debug and logger.isEnabledFor(logging.DEBUG))
        block_tag = "?" if trace_block_idx is None else str(int(trace_block_idx) + 1)
        x_dtype = x.dtype
        use_fp32_compute = _wan22_use_fp32_compute(x_dtype)
        fp32_mode = _wan22_fp32_compute_mode()
        norm_compute_label = _wan22_norm_compute_label(x_dtype)
        # Combine time embedding with per-block modulation in the effective compute dtype.
        if use_fp32_compute:
            mod = time_emb.float() + self.modulation.float()  # [B, 6, dim]
        else:
            mod = time_emb.to(device=x.device, dtype=x_dtype) + self.modulation.to(device=x.device, dtype=x_dtype)

        sa_shift, sa_scale, sa_gate = mod[:, 0], mod[:, 1], mod[:, 2]
        ffn_shift, ffn_scale, ffn_gate = mod[:, 3], mod[:, 4], mod[:, 5]

        if trace_enabled:
            logger.debug(
                "[wan22.trace] block[%s] pre: x_dtype=%s x_device=%s ctx_dtype=%s ctx_device=%s "
                "time_emb_dtype=%s time_emb_device=%s mod_device=%s "
                "norm1/2/3=WanFP32LayerNorm(compute=%s) qk_norm=WanRMSNorm(compute=%s) fp32_policy=%s "
                "ffn=Linear->GELU(tanh)->Linear %s",
                block_tag,
                str(x.dtype),
                str(x.device),
                str(context.dtype),
                str(context.device),
                str(time_emb.dtype),
                str(time_emb.device),
                str(mod.device),
                norm_compute_label,
                norm_compute_label,
                fp32_mode,
                _cuda_mem_snapshot_str(x.device),
            )

        # Self-attention: pre-norm (no affine) + time modulation + gated residual.
        x_norm_input = x.float() if use_fp32_compute else x
        x_sa = self.norm1(x_norm_input) * (1.0 + sa_scale[:, None, :]) + sa_shift[:, None, :]
        if use_fp32_compute:
            x_sa = x_sa.to(dtype=x_dtype)
        sa_out = self.self_attn(
            x_sa,
            rotary_emb=rotary_emb,
            trace_debug=trace_enabled,
            trace_block_idx=trace_block_idx,
        )
        if use_fp32_compute:
            x = (x_norm_input + sa_out.float() * sa_gate[:, None, :]).to(dtype=x_dtype)
        else:
            x.addcmul_(sa_out, sa_gate[:, None, :])
        if trace_enabled:
            logger.debug(
                "[wan22.trace] block[%s] self_attn: x_sa_dtype=%s x_sa_device=%s sa_out_dtype=%s sa_out_device=%s "
                "gate_dtype=%s residual_dtype=%s residual_device=%s %s",
                block_tag,
                str(x_sa.dtype),
                str(x_sa.device),
                str(sa_out.dtype),
                str(sa_out.device),
                str(sa_gate.dtype),
                str(x.dtype),
                str(x.device),
                _cuda_mem_snapshot_str(x.device),
            )

        # Cross-attention: pre-norm3 (affine) + residual (no time modulation).
        x_ca_input = x.float() if use_fp32_compute else x
        x_ca = self.norm3(x_ca_input)
        if use_fp32_compute:
            x_ca = x_ca.to(dtype=x_dtype)
        ca_out = self.cross_attn(
            x_ca,
            context,
            trace_debug=trace_enabled,
            trace_block_idx=trace_block_idx,
        )
        x.add_(ca_out)
        if trace_enabled:
            logger.debug(
                "[wan22.trace] block[%s] cross_attn: x_ca_dtype=%s x_ca_device=%s ca_out_dtype=%s ca_out_device=%s "
                "residual_dtype=%s residual_device=%s %s",
                block_tag,
                str(x_ca.dtype),
                str(x_ca.device),
                str(ca_out.dtype),
                str(ca_out.device),
                str(x.dtype),
                str(x.device),
                _cuda_mem_snapshot_str(x.device),
            )

        # FFN: pre-norm (no affine) + time modulation + gated residual.
        x_ffn_input = x.float() if use_fp32_compute else x
        x_ffn = self.norm2(x_ffn_input) * (1.0 + ffn_scale[:, None, :]) + ffn_shift[:, None, :]
        if use_fp32_compute:
            x_ffn = x_ffn.to(dtype=x_dtype)

        ffn_hidden_dim = None
        if isinstance(self.ffn, nn.Sequential) and len(self.ffn) > 0 and isinstance(self.ffn[0], nn.Linear):
            ffn_hidden_dim = int(self.ffn[0].out_features)
        ffn_chunk_tokens = 0 if ffn_hidden_dim is None else _wan22_resolve_ffn_chunk_tokens(x_blc=x_ffn, hidden_dim=ffn_hidden_dim)
        if trace_enabled and ffn_chunk_tokens > 0:
            logger.debug(
                "[wan22.trace] block[%s] ffn chunking active: tokens=%d hidden_dim=%d budget_mb=%d",
                block_tag,
                int(ffn_chunk_tokens),
                int(ffn_hidden_dim) if ffn_hidden_dim is not None else -1,
                int(_WAN22_FFN_ACTIVATION_BUDGET_BYTES // (1024 * 1024)),
            )

        ffn_out_for_trace: torch.Tensor | None = None
        if ffn_chunk_tokens > 0 and not use_fp32_compute:
            # Chunked FFN path: avoid materializing the full fc1 activation and also
            # avoid allocating a full-sized ffn_out buffer by updating the residual
            # in-place per token chunk.
            seq_len = int(x_ffn.shape[1])
            for start in range(0, seq_len, int(ffn_chunk_tokens)):
                end = min(seq_len, int(start + ffn_chunk_tokens))
                out_chunk = self.ffn(x_ffn[:, start:end, :])
                x[:, start:end, :].addcmul_(out_chunk, ffn_gate[:, None, :])
                ffn_out_for_trace = out_chunk
        else:
            if ffn_chunk_tokens > 0:
                ffn_out = torch.empty_like(x_ffn)
                seq_len = int(x_ffn.shape[1])
                for start in range(0, seq_len, int(ffn_chunk_tokens)):
                    end = min(seq_len, int(start + ffn_chunk_tokens))
                    ffn_out[:, start:end, :].copy_(self.ffn(x_ffn[:, start:end, :]))
            else:
                ffn_out = self.ffn(x_ffn)
            ffn_out_for_trace = ffn_out

            if use_fp32_compute:
                x = (x_ffn_input + ffn_out.float() * ffn_gate[:, None, :]).to(dtype=x_dtype)
            else:
                x.addcmul_(ffn_out, ffn_gate[:, None, :])
        if trace_enabled:
            logger.debug(
                "[wan22.trace] block[%s] ffn: x_ffn_dtype=%s x_ffn_device=%s ffn_out_dtype=%s ffn_out_device=%s "
                "gate_dtype=%s residual_dtype=%s residual_device=%s %s",
                block_tag,
                str(x_ffn.dtype),
                str(x_ffn.device),
                str(ffn_out_for_trace.dtype if ffn_out_for_trace is not None else "<none>"),
                str(ffn_out_for_trace.device if ffn_out_for_trace is not None else "<none>"),
                str(ffn_gate.dtype),
                str(x.dtype),
                str(x.device),
                _cuda_mem_snapshot_str(x.device),
            )

        return x


class WanTransformer2DModel(nn.Module):
    """WAN Diffusion Transformer as nn.Module.

    This is a format-agnostic implementation that works with any weight
    format (GGUF, safetensors, etc.) via the operations registry.
    """

    def __init__(self, config: WanArchitectureConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_blocks = config.n_blocks

        # Patch embedding (3D conv for video)
        kT, kH, kW = config.patch_size
        patch_dim = config.latent_channels * kT * kH * kW
        self.patch_embed = nn.Conv3d(
            config.in_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=(1, kH, kW),
            padding=0,
        )

        # Time embedding
        time_dim = config.d_model
        self.time_embed = nn.Sequential(
            nn.Linear(config.time_embed_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Time projection to modulation
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * config.d_model),  # [6, d_model] per block
        )

        # Text embedding projection (optional; some checkpoints already output d_model)
        if config.use_text_projection:
            self.text_embed = nn.Sequential(
                nn.Linear(config.context_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
        else:
            self.text_embed = nn.Identity()

        # Rotary positional embedding (RoPE) used by WAN self-attention.
        self.rope = WanRotaryPosEmbed(
            attention_head_dim=(config.d_model // config.n_heads),
            patch_size=config.patch_size,
            max_seq_len=config.rope_max_seq_len,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            WanTransformerBlock(
                dim=config.d_model,
                num_heads=config.n_heads,
                context_dim=config.d_model,  # After text projection
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
            )
            for _ in range(config.n_blocks)
        ])

        # Output head
        self.norm_out = WanFP32LayerNorm(config.d_model, eps=1e-6, elementwise_affine=False)
        # Head modulation: [1, 2, dim] (shift/scale). Matches Diffusers `WanTransformer3DModel.scale_shift_table`.
        op_ctx = get_operation_context()
        modulation_kwargs = {}
        if op_ctx.device is not None:
            modulation_kwargs["device"] = op_ctx.device
        if op_ctx.dtype is not None:
            modulation_kwargs["dtype"] = op_ctx.dtype
        self.head_modulation = nn.Parameter(torch.zeros(1, 2, config.d_model, **modulation_kwargs))
        self.head = nn.Linear(config.d_model, patch_dim)

        logger.info(
            "WanTransformer2DModel created: d_model=%d, n_heads=%d, n_blocks=%d",
            config.d_model,
            config.n_heads,
            config.n_blocks,
        )

    def _timestep_embedding(
        self,
        t: torch.Tensor,
        dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Create sinusoidal timestep embedding."""
        base_dim = int(dim if dim is not None else self.config.time_embed_dim)
        half = max(base_dim // 2, 1)
        freq = torch.arange(half, device=t.device, dtype=torch.float32)
        # Diffusers `Timesteps(..., downscale_freq_shift=0)` => denominator is `half_dim`.
        div_term = torch.exp(-math.log(10000.0) * freq / float(half))
        angles = t.to(dtype=torch.float32)[:, None] * div_term[None, :]
        # Use cosine-first sinusoidal ordering on the timestep embedding.
        emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
        if emb.shape[1] != base_dim:
            emb = torch.nn.functional.pad(emb, (0, base_dim - emb.shape[1]))
        return emb

    def forward(
        self,
        x: torch.Tensor,  # [B, C, T, H, W] latent video
        timestep: torch.Tensor,  # [B,] or scalar
        context: torch.Tensor,  # [B, L, context_dim] text embeddings
        transformer_options: dict | None = None,
    ) -> torch.Tensor:
        """Forward pass of WAN transformer.

        Args:
            x: Input latent video [B, C, T, H, W]
            timestep: Diffusion timestep
            context: Text conditioning embeddings

        Returns:
            Output latent video [B, C, T, H, W]
        """
        device = x.device
        dtype = x.dtype
        B, C, T, H, W = x.shape
        trace_debug = _wan_trace_verbose_enabled() and logger.isEnabledFor(logging.DEBUG)

        # Timestep to scalar tensor
        if isinstance(timestep, (int, float)):
            timestep = torch.tensor([timestep], device=device, dtype=torch.float32)
        timestep = timestep.to(device=device, dtype=torch.float32).view(-1)
        if timestep.numel() == 1 and B > 1:
            timestep = timestep.expand(B)

        rotary_emb = self.rope(x)

        # Time embedding
        t_emb = self._timestep_embedding(timestep)
        t_emb = self.time_embed(t_emb.to(dtype))  # [B, d_model]

        # Time projection to modulation [B, 6, d_model]
        t_proj = self.time_proj(t_emb)
        t_proj = t_proj.view(B, 6, self.d_model)

        # Text embedding projection
        ctx = self.text_embed(context.to(dtype))  # [B, L, d_model]
        # Patch embed: [B, C, T, H, W] -> [B, d_model, T', H', W'] -> [B, T'*H'*W', d_model]
        tokens = self.patch_embed(x)
        _, _, t_grid, h_grid, w_grid = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, L, d_model]

        if trace_debug:
            logger.debug(
                "[wan22.trace] model pre-blocks: x_shape=%s x_dtype=%s x_device=%s tokens_shape=%s tokens_dtype=%s tokens_device=%s "
                "ctx_shape=%s ctx_dtype=%s ctx_device=%s t_proj_shape=%s t_proj_dtype=%s t_proj_device=%s "
                "rotary=(cos=%s@%s sin=%s@%s) %s",
                tuple(x.shape),
                str(x.dtype),
                str(x.device),
                tuple(tokens.shape),
                str(tokens.dtype),
                str(tokens.device),
                tuple(ctx.shape),
                str(ctx.dtype),
                str(ctx.device),
                tuple(t_proj.shape),
                str(t_proj.dtype),
                str(t_proj.device),
                str(rotary_emb[0].dtype),
                str(rotary_emb[0].device),
                str(rotary_emb[1].dtype),
                str(rotary_emb[1].device),
                _cuda_mem_snapshot_str(device),
            )

        # Apply transformer blocks
        block_progress_callback = resolve_block_progress_callback(transformer_options)
        total_blocks = int(len(self.blocks))
        for block_idx, block in enumerate(self.blocks):
            if block_progress_callback is not None:
                block_progress_callback(int(block_idx + 1), total_blocks)
            if trace_debug:
                logger.debug(
                    "[wan22.trace] block[%d/%d] dispatch: block_dtype=%s block_device=%s tokens_dtype=%s tokens_device=%s "
                    "ctx_dtype=%s ctx_device=%s t_proj_dtype=%s t_proj_device=%s %s",
                    int(block_idx + 1),
                    int(self.n_blocks),
                    _module_parameter_dtype(block),
                    _module_parameter_device(block),
                    str(tokens.dtype),
                    str(tokens.device),
                    str(ctx.dtype),
                    str(ctx.device),
                    str(t_proj.dtype),
                    str(t_proj.device),
                    _cuda_mem_snapshot_str(device),
                )
            tokens = block(
                tokens,
                ctx,
                t_proj,
                rotary_emb=rotary_emb,
                trace_block_idx=block_idx,
                trace_debug=trace_debug,
            )
            if trace_debug:
                logger.debug(
                    "[wan22.trace] block[%d/%d] done: tokens_shape=%s tokens_dtype=%s tokens_device=%s %s",
                    int(block_idx + 1),
                    int(self.n_blocks),
                    tuple(tokens.shape),
                    str(tokens.dtype),
                    str(tokens.device),
                    _cuda_mem_snapshot_str(device),
                )

        # Output head uses the effective fp32/native compute policy for modulation + norm.
        tokens_dtype = tokens.dtype
        use_fp32_compute = _wan22_use_fp32_compute(tokens_dtype)
        if use_fp32_compute:
            shift, scale = (self.head_modulation.float() + t_emb.float()[:, None, :]).chunk(2, dim=1)  # [B, 1, C] each
            tokens_norm_input = tokens.float()
        else:
            shift, scale = (
                self.head_modulation.to(device=tokens.device, dtype=tokens_dtype)
                + t_emb.to(device=tokens.device, dtype=tokens_dtype)[:, None, :]
            ).chunk(2, dim=1)
            tokens_norm_input = tokens
        tokens = self.norm_out(tokens_norm_input)
        fused = tokens * (1.0 + scale) + shift
        if use_fp32_compute:
            fused = fused.to(dtype=tokens_dtype)
        patches = self.head(fused)

        if trace_debug:
            logger.debug(
                "[wan22.trace] model post-blocks: tokens_dtype=%s tokens_device=%s fused_dtype=%s fused_device=%s "
                "patches_dtype=%s patches_device=%s %s",
                str(tokens_dtype),
                str(tokens.device),
                str(fused.dtype),
                str(fused.device),
                str(patches.dtype),
                str(patches.device),
                _cuda_mem_snapshot_str(device),
            )

        # Unpatchify: [B, L, patch_dim] -> [B, C, T, H, W]
        kT, kH, kW = self.config.patch_size
        out = patches.view(B, t_grid, h_grid, w_grid, kT, kH, kW, self.config.latent_channels)
        out = out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        out = out.view(B, self.config.latent_channels, t_grid * kT, h_grid * kH, w_grid * kW)

        return out


# Weight loading helper
def resolve_wan22_gguf_keyspace(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve WAN transformer checkpoint keys to WanTransformer2DModel lookup keys.

    Supported input styles:
    - Diffusers-style keys (e.g. `condition_embedder.*`, `blocks.N.attn1/attn2.*`, `ffn.net.*`, `proj_out.*`).
    - WAN export-style keys (e.g. `patch_embedding.*`, `time_embedding.*`, `head.head.*`, `head.modulation`).
    - Codex-native keys (e.g. `patch_embed.*`, `time_embed.*`, `head.*`, `head_modulation`).

    This function is strict and fails loud on unknown/ambiguous layouts.
    """

    from apps.backend.runtime.state_dict.keymap_wan22_transformer import resolve_wan22_transformer_keyspace

    resolved = resolve_wan22_transformer_keyspace(state_dict)
    style = resolved.style
    style_label = style.value if hasattr(style, "value") else str(style)
    logger.debug("WAN22 keyspace: detected style=%s", style_label)
    return resolved.view


def infer_wan_architecture_from_state_dict(state_dict: dict) -> WanArchitectureConfig:
    """Infer WAN architecture parameters from a resolved WAN keyspace."""

    def _shape(key: str) -> tuple[int, ...] | None:
        value = state_dict.get(key)
        if value is None:
            return None
        try:
            shape = tuple(int(s) for s in getattr(value, "shape", ()) or ())
        except Exception:
            return None
        return shape or None

    d_model = 5120
    patch_embed_shape = _shape("patch_embed.weight")
    if patch_embed_shape and len(patch_embed_shape) == 5:
        _in, model_dim, _patch = infer_wan22_patch_embedding(patch_embed_shape, default_model_dim=d_model)
        d_model = int(model_dim)
    else:
        for key in ("time_embed.0.weight", "blocks.0.self_attn.q.weight"):
            shape = _shape(key)
            if shape and len(shape) >= 1:
                d_model = int(shape[0])
                break

    n_blocks = 0
    for key in state_dict.keys():
        ks = str(key)
        if not ks.startswith("blocks."):
            continue
        parts = ks.split(".", 2)
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        n_blocks = max(n_blocks, idx + 1)

    # Default head_dim heuristic (matches the legacy GGUF runner).
    n_heads = 32
    for head_dim in (128, 64):
        if d_model % head_dim == 0:
            candidate = d_model // head_dim
            if 8 <= candidate <= 64:
                n_heads = int(candidate)
                break

    patch_shape = _shape("patch_embed.weight")
    head_shape = _shape("head.weight")
    patch_size, in_channels = infer_wan22_patch_size_and_in_channels(
        patch_shape,
        default_patch_size=(1, 2, 2),
        default_in_channels=16,
    )

    time_embed_dim = 256
    te0_shape = _shape("time_embed.0.weight")
    if te0_shape and len(te0_shape) == 2:
        time_embed_dim = int(te0_shape[1])

    use_text_projection = "text_embed.0.weight" in state_dict and "text_embed.2.weight" in state_dict
    context_dim = d_model
    if use_text_projection:
        t0_shape = _shape("text_embed.0.weight")
        if t0_shape and len(t0_shape) == 2:
            context_dim = int(t0_shape[1])

    mlp_ratio = 4.0
    ffn_shape = _shape("blocks.0.ffn.0.weight")
    if ffn_shape and len(ffn_shape) == 2 and ffn_shape[1] == d_model and d_model > 0:
        mlp_ratio = float(ffn_shape[0]) / float(d_model)

    qkv_bias = "blocks.0.self_attn.q.bias" in state_dict or "blocks.0.self_attn.k.bias" in state_dict

    latent_channels = infer_wan22_latent_channels(
        head_shape,
        patch_size=patch_size,
        default_latent_channels=in_channels,
    )

    return WanArchitectureConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_blocks=n_blocks or 1,
        mlp_ratio=mlp_ratio,
        context_dim=context_dim,
        time_embed_dim=time_embed_dim,
        patch_size=patch_size,
        in_channels=in_channels,
        latent_channels=latent_channels,
        qkv_bias=qkv_bias,
        use_text_projection=use_text_projection,
    )


def load_wan_transformer_from_state_dict(
    state_dict: dict,
    config: Optional[WanArchitectureConfig] = None,
) -> WanTransformer2DModel:
    """Load WanTransformer2DModel from a state dict.

    Can handle both native format and converted GGUF weights.

    Args:
        state_dict: Model weights (may contain CodexParameter)
        config: Model configuration (derived from state if not provided)

    Returns:
        Loaded WanTransformer2DModel
    """
    if config is None:
        config = infer_wan_architecture_from_state_dict(state_dict)

    model = WanTransformer2DModel(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "WAN transformer strict load failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
        )

    logger.info(
        "Loaded WanTransformer2DModel: %d blocks, d_model=%d",
        config.n_blocks,
        config.d_model,
    )

    return model
