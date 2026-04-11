"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native LTX2 text-connector stack for packed video/audio prompt embeddings.
Implements the legacy parser-owned connector surface and the real LTX 2.3 split-pack connector surface without relying
on LTX2-specific Diffusers classes, while keeping the stored state-dict layout intact.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2TextConnectors` (class): Native connector stack that projects packed text embeddings into video/audio streams.
- `load_ltx2_connectors` (function): Strict config/state-driven connector loader used by the LTX2 runtime.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from apps.backend.runtime.ops.operations_gguf import dequantize_tensor
from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime.state_dict.views import KeyspaceLookupView

_CONNECTORS_WRAPPER_PREFIX = "connectors."
_REAL_23_CONNECTOR_PREFIXES = (
    "audio_embeddings_connector.",
    "text_embedding_projection.",
    "video_embeddings_connector.",
)


class _DequantizedConnectorStateDictView(MappingABC[str, Any]):
    def __init__(self, base: Mapping[str, Any]) -> None:
        self._base = base

    def __getitem__(self, key: str) -> Any:
        if not isinstance(key, str):
            raise RuntimeError(
                "LTX2 connectors dequantized state_dict expects string keys. "
                f"Got {type(key).__name__}."
            )
        return dequantize_tensor(self._base[key])

    def __iter__(self):
        return iter(self._base.keys())

    def __len__(self) -> int:
        return len(self._base)

    def keys(self):
        return self._base.keys()

    def shape_of(self, key: str):
        shape_getter = getattr(self._base, "shape_of", None)
        if not callable(shape_getter):
            return None
        try:
            return shape_getter(key)
        except Exception:
            return None

    def materialize(self):
        materializer = getattr(self._base, "materialize", None)
        if callable(materializer):
            materialized = materializer()
            if isinstance(materialized, tuple):
                materialized = materialized[0]
            if isinstance(materialized, Mapping):
                return {key: dequantize_tensor(value) for key, value in materialized.items()}
        return {key: dequantize_tensor(self._base[key]) for key in self._base.keys()}


def _require_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise RuntimeError(f"LTX2 connectors config requires integer `{key}`, got {value!r}.")
    return int(value)


def _require_float(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"LTX2 connectors config requires float `{key}`, got {value!r}.")
    return float(value)


def _require_bool(config: Mapping[str, Any], key: str) -> bool:
    value = config.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"LTX2 connectors config requires bool `{key}`, got {value!r}.")
    return bool(value)


def _require_str(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"LTX2 connectors config requires non-empty string `{key}`, got {value!r}.")
    return value


def apply_interleaved_rotary_emb(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return (x.float() * cos + x_rotated.float() * sin).to(dtype=x.dtype)


def apply_split_rotary_emb(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs
    original_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        batch_size = x.shape[0]
        _, heads, tokens, _ = cos.shape
        x = x.reshape(batch_size, tokens, heads, -1).swapaxes(1, 2)
        needs_reshape = True

    if x.shape[-1] % 2 != 0:
        raise RuntimeError(f"LTX2 split rotary expects even last-dim width, got {x.shape[-1]}.")

    half = x.shape[-1] // 2
    split_x = x.reshape(*x.shape[:-1], 2, half).float()
    first = split_x[..., :1, :]
    second = split_x[..., 1:, :]
    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    out[..., :1, :].addcmul_(-sin_u, second)
    out[..., 1:, :].addcmul_(sin_u, first)
    out = out.reshape(*out.shape[:-2], x.shape[-1])

    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(batch_size, tokens, -1)
    return out.to(dtype=original_dtype)


def _rms_norm(hidden_states: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.rms_norm(hidden_states, (hidden_states.shape[-1],), eps=eps)


class _Ltx2RotaryPosEmbed1d(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        base_seq_len: int,
        theta: float,
        double_precision: bool,
        rope_type: str,
        num_attention_heads: int,
    ) -> None:
        super().__init__()
        if rope_type not in {"interleaved", "split"}:
            raise RuntimeError(f"LTX2 connectors rope_type must be 'interleaved' or 'split', got {rope_type!r}.")
        self.dim = int(dim)
        self.base_seq_len = int(base_seq_len)
        self.theta = float(theta)
        self.double_precision = bool(double_precision)
        self.rope_type = rope_type
        self.num_attention_heads = int(num_attention_heads)

    def forward(
        self,
        batch_size: int,
        positions: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = torch.arange(positions, dtype=torch.float32, device=device) / float(self.base_seq_len)
        grid = grid.unsqueeze(0).repeat(batch_size, 1)
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        base = torch.pow(
            torch.tensor(self.theta, dtype=freqs_dtype, device=device),
            torch.linspace(0.0, 1.0, steps=self.dim // 2, dtype=freqs_dtype, device=device),
        )
        freqs = ((grid.unsqueeze(-1) * 2.0) - 1.0) * (base.to(dtype=torch.float32) * (torch.pi / 2.0))

        if self.rope_type == "interleaved":
            cos = freqs.cos().repeat_interleave(2, dim=-1)
            sin = freqs.sin().repeat_interleave(2, dim=-1)
            return cos, sin

        expected_freqs = self.dim // 2
        current_freqs = freqs.shape[-1]
        if current_freqs > expected_freqs:
            raise RuntimeError(
                f"LTX2 connectors split-rope expected <= {expected_freqs} frequencies, got {current_freqs}."
            )
        if current_freqs < expected_freqs:
            pad_size = expected_freqs - current_freqs
            cos = torch.cat([torch.ones_like(freqs[:, :, :pad_size]), freqs.cos()], dim=-1)
            sin = torch.cat([torch.zeros_like(freqs[:, :, :pad_size]), freqs.sin()], dim=-1)
        else:
            cos = freqs.cos()
            sin = freqs.sin()
        cos = cos.reshape(batch_size, positions, self.num_attention_heads, -1).swapaxes(1, 2)
        sin = sin.reshape(batch_size, positions, self.num_attention_heads, -1).swapaxes(1, 2)
        return cos, sin


class _GELUProjector(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, approximate: str = "none", bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate=self.approximate)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, *, activation_fn: str = "gelu-approximate", bias: bool = True) -> None:
        super().__init__()
        if activation_fn != "gelu-approximate":
            raise RuntimeError(
                "LTX2 connectors feed-forward only supports activation_fn='gelu-approximate' in the native slice."
            )
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList(
            [
                _GELUProjector(dim, inner_dim, approximate="tanh", bias=bias),
                nn.Dropout(0.0),
                nn.Linear(inner_dim, dim, bias=bias),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class _Ltx2Attention(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        heads: int,
        kv_heads: int,
        dim_head: int,
        bias: bool = True,
        out_bias: bool = True,
        rope_type: str,
    ) -> None:
        super().__init__()
        self.head_dim = int(dim_head)
        self.heads = int(heads)
        self.kv_heads = int(kv_heads)
        self.inner_dim = self.head_dim * self.heads
        self.inner_kv_dim = self.head_dim * self.kv_heads
        self.query_dim = int(query_dim)
        self.rope_type = rope_type
        self.norm_q = torch.nn.RMSNorm(self.inner_dim, eps=1e-6, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(self.inner_kv_dim, eps=1e-6, elementwise_affine=True)
        self.to_q = nn.Linear(self.query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.query_dim, self.inner_kv_dim, bias=bias)
        self.to_v = nn.Linear(self.query_dim, self.inner_kv_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, self.query_dim, bias=out_bias), nn.Dropout(0.0)])

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        *,
        batch_size: int,
        query_len: int,
        key_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        mask = attention_mask
        if mask.ndim == 2:
            mask = mask[:, None, None, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
        elif mask.ndim != 4:
            raise RuntimeError(f"LTX2 connectors attention mask must be 2D/3D/4D, got {mask.ndim}D.")
        if mask.shape[0] != batch_size:
            raise RuntimeError(
                f"LTX2 connectors attention mask batch mismatch: expected {batch_size}, got {mask.shape[0]}."
            )
        if mask.shape[-1] != key_len:
            raise RuntimeError(
                f"LTX2 connectors attention mask key length mismatch: expected {key_len}, got {mask.shape[-1]}."
            )
        if mask.shape[-2] == 1:
            mask = mask.expand(batch_size, 1, query_len, key_len)
        elif mask.shape[-2] != query_len:
            raise RuntimeError(
                f"LTX2 connectors attention mask query length mismatch: expected 1 or {query_len}, got {mask.shape[-2]}."
            )
        return mask.to(device=device, dtype=dtype).expand(batch_size, self.heads, query_len, key_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise RuntimeError(
                f"LTX2 connectors attention expects [batch,tokens,channels], got shape={tuple(hidden_states.shape)!r}."
            )
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch_size, query_len, _ = hidden_states.shape
        key_len = int(context.shape[1])

        query = self.norm_q(self.to_q(hidden_states))
        key = self.norm_k(self.to_k(context))
        value = self.to_v(context)

        if query_rotary_emb is not None:
            if self.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(key, key_rotary_emb or query_rotary_emb)
            else:
                query = apply_split_rotary_emb(query, query_rotary_emb)
                key = apply_split_rotary_emb(key, key_rotary_emb or query_rotary_emb)

        query = query.unflatten(2, (self.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.kv_heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (self.kv_heads, -1)).transpose(1, 2)

        attention_bias = self._prepare_attention_mask(
            attention_mask,
            batch_size=batch_size,
            query_len=query_len,
            key_len=key_len,
            device=query.device,
            dtype=query.dtype,
        )
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class _Ltx2TransformerBlock1d(nn.Module):
    def __init__(self, *, dim: int, num_attention_heads: int, attention_head_dim: int, rope_type: str) -> None:
        super().__init__()
        self.norm1 = torch.nn.RMSNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn1 = _Ltx2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            rope_type=rope_type,
        )
        self.norm2 = torch.nn.RMSNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff = _FeedForward(dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn1(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            query_rotary_emb=rotary_emb,
        )
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class _Ltx2ConnectorTransformer1d(nn.Module):
    def __init__(
        self,
        *,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        num_learnable_registers: int | None,
        rope_base_seq_len: int,
        rope_theta: float,
        rope_double_precision: bool,
        causal_temporal_positioning: bool,
        rope_type: str,
    ) -> None:
        super().__init__()
        self.num_attention_heads = int(num_attention_heads)
        self.inner_dim = self.num_attention_heads * int(attention_head_dim)
        self.causal_temporal_positioning = bool(causal_temporal_positioning)
        self.num_learnable_registers = num_learnable_registers
        self.learnable_registers = None
        if num_learnable_registers is not None:
            init_registers = (torch.rand(num_learnable_registers, self.inner_dim) * 2.0) - 1.0
            self.learnable_registers = nn.Parameter(init_registers)
        self.rope = _Ltx2RotaryPosEmbed1d(
            self.inner_dim,
            base_seq_len=rope_base_seq_len,
            theta=rope_theta,
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.num_attention_heads,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                _Ltx2TransformerBlock1d(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    rope_type=rope_type,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm_out = torch.nn.RMSNorm(self.inner_dim, eps=1e-6, elementwise_affine=False)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attn_mask_binarize_threshold: float = -9000.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hidden_states.ndim != 3:
            raise RuntimeError(
                f"LTX2 connector transformer expects [batch,tokens,channels], got shape={tuple(hidden_states.shape)!r}."
            )
        batch_size, seq_len, _ = hidden_states.shape

        if self.learnable_registers is not None:
            if attention_mask is None:
                raise RuntimeError("LTX2 connector transformer with learnable registers requires an attention_mask.")
            if seq_len % int(self.num_learnable_registers) != 0:
                raise RuntimeError(
                    "LTX2 connector transformer sequence length must be divisible by the number of learnable registers; "
                    f"got seq_len={seq_len} registers={self.num_learnable_registers}."
                )
            binary_mask = (attention_mask >= float(attn_mask_binarize_threshold)).int()
            if binary_mask.ndim == 4:
                binary_mask = binary_mask.squeeze(1).squeeze(1)
            if binary_mask.ndim != 2:
                raise RuntimeError(
                    f"LTX2 connector transformer binary mask must reduce to [batch,tokens], got {binary_mask.ndim}D."
                )
            if binary_mask.shape != (batch_size, seq_len):
                raise RuntimeError(
                    "LTX2 connector transformer binary mask shape mismatch: "
                    f"expected {(batch_size, seq_len)!r}, got {tuple(binary_mask.shape)!r}."
                )
            register_repeats = seq_len // int(self.num_learnable_registers)
            registers = torch.tile(self.learnable_registers, (register_repeats, 1))
            non_padded = [hidden_states[index, binary_mask[index].bool(), :] for index in range(batch_size)]
            pad_lengths = [seq_len - item.shape[0] for item in non_padded]
            padded = [F.pad(item, pad=(0, 0, 0, pad), value=0.0) for item, pad in zip(non_padded, pad_lengths)]
            hidden_states = torch.cat([item.unsqueeze(0) for item in padded], dim=0)
            flipped = torch.flip(binary_mask, dims=[1]).unsqueeze(-1)
            hidden_states = flipped * hidden_states + (1 - flipped) * registers
            attention_mask = torch.zeros_like(attention_mask)

        rotary_emb = self.rope(batch_size, seq_len, device=hidden_states.device)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, rotary_emb=rotary_emb)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states, attention_mask


class Ltx2TextConnectors(nn.Module):
    def __init__(
        self,
        *,
        caption_channels: int,
        text_proj_in_factor: int,
        video_connector_num_attention_heads: int,
        video_connector_attention_head_dim: int,
        video_connector_num_layers: int,
        video_connector_num_learnable_registers: int | None,
        audio_connector_num_attention_heads: int,
        audio_connector_attention_head_dim: int,
        audio_connector_num_layers: int,
        audio_connector_num_learnable_registers: int | None,
        connector_rope_base_seq_len: int,
        rope_theta: float,
        rope_double_precision: bool,
        causal_temporal_positioning: bool,
        rope_type: str = "interleaved",
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            caption_channels=int(caption_channels),
            text_proj_in_factor=int(text_proj_in_factor),
            video_connector_num_attention_heads=int(video_connector_num_attention_heads),
            video_connector_attention_head_dim=int(video_connector_attention_head_dim),
            video_connector_num_layers=int(video_connector_num_layers),
            video_connector_num_learnable_registers=video_connector_num_learnable_registers,
            audio_connector_num_attention_heads=int(audio_connector_num_attention_heads),
            audio_connector_attention_head_dim=int(audio_connector_attention_head_dim),
            audio_connector_num_layers=int(audio_connector_num_layers),
            audio_connector_num_learnable_registers=audio_connector_num_learnable_registers,
            connector_rope_base_seq_len=int(connector_rope_base_seq_len),
            rope_theta=float(rope_theta),
            rope_double_precision=bool(rope_double_precision),
            causal_temporal_positioning=bool(causal_temporal_positioning),
            rope_type=rope_type,
        )
        self.text_proj_in = nn.Linear(
            int(caption_channels) * int(text_proj_in_factor),
            int(caption_channels),
            bias=False,
        )
        self.video_connector = _Ltx2ConnectorTransformer1d(
            num_attention_heads=video_connector_num_attention_heads,
            attention_head_dim=video_connector_attention_head_dim,
            num_layers=video_connector_num_layers,
            num_learnable_registers=video_connector_num_learnable_registers,
            rope_base_seq_len=connector_rope_base_seq_len,
            rope_theta=rope_theta,
            rope_double_precision=rope_double_precision,
            causal_temporal_positioning=causal_temporal_positioning,
            rope_type=rope_type,
        )
        self.audio_connector = _Ltx2ConnectorTransformer1d(
            num_attention_heads=audio_connector_num_attention_heads,
            attention_head_dim=audio_connector_attention_head_dim,
            num_layers=audio_connector_num_layers,
            num_learnable_registers=audio_connector_num_learnable_registers,
            rope_base_seq_len=connector_rope_base_seq_len,
            rope_theta=rope_theta,
            rope_double_precision=rope_double_precision,
            causal_temporal_positioning=causal_temporal_positioning,
            rope_type=rope_type,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Ltx2TextConnectors":
        if not isinstance(config, Mapping):
            raise RuntimeError(f"LTX2 connectors from_config requires a mapping, got {type(config).__name__}.")
        rope_type = _require_str(config, "rope_type")
        if rope_type not in {"interleaved", "split"}:
            raise RuntimeError(f"LTX2 connectors rope_type must be 'interleaved' or 'split', got {rope_type!r}.")
        return cls(
            caption_channels=_require_int(config, "caption_channels"),
            text_proj_in_factor=_require_int(config, "text_proj_in_factor"),
            video_connector_num_attention_heads=_require_int(config, "video_connector_num_attention_heads"),
            video_connector_attention_head_dim=_require_int(config, "video_connector_attention_head_dim"),
            video_connector_num_layers=_require_int(config, "video_connector_num_layers"),
            video_connector_num_learnable_registers=_require_int(config, "video_connector_num_learnable_registers"),
            audio_connector_num_attention_heads=_require_int(config, "audio_connector_num_attention_heads"),
            audio_connector_attention_head_dim=_require_int(config, "audio_connector_attention_head_dim"),
            audio_connector_num_layers=_require_int(config, "audio_connector_num_layers"),
            audio_connector_num_learnable_registers=_require_int(config, "audio_connector_num_learnable_registers"),
            connector_rope_base_seq_len=_require_int(config, "connector_rope_base_seq_len"),
            rope_theta=_require_float(config, "rope_theta"),
            rope_double_precision=_require_bool(config, "rope_double_precision"),
            causal_temporal_positioning=_require_bool(config, "causal_temporal_positioning"),
            rope_type=rope_type,
        )

    def forward(
        self,
        text_encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        additive_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if text_encoder_hidden_states.ndim != 3:
            raise RuntimeError(
                "LTX2 connectors expect packed text encoder hidden states with shape [batch,tokens,channels]; "
                f"got {tuple(text_encoder_hidden_states.shape)!r}."
            )
        if attention_mask is None:
            raise RuntimeError("LTX2 connectors require an attention_mask.")
        if not additive_mask:
            text_dtype = text_encoder_hidden_states.dtype
            attention_mask = (attention_mask - 1).reshape(
                attention_mask.shape[0],
                1,
                -1,
                attention_mask.shape[-1],
            )
            attention_mask = attention_mask.to(text_dtype) * torch.finfo(text_dtype).max

        text_encoder_hidden_states = self.text_proj_in(text_encoder_hidden_states)
        video_text_embedding, new_attn_mask = self.video_connector(text_encoder_hidden_states, attention_mask)
        attn_mask = (new_attn_mask < 1e-6).to(torch.int64)
        attn_mask = attn_mask.reshape(video_text_embedding.shape[0], video_text_embedding.shape[1], 1)
        video_text_embedding = video_text_embedding * attn_mask
        new_attn_mask = attn_mask.squeeze(-1)
        audio_text_embedding, _ = self.audio_connector(text_encoder_hidden_states, attention_mask)
        return video_text_embedding, audio_text_embedding, new_attn_mask


@dataclass(frozen=True)
class _Ltx23ConnectorSpec:
    prefix: str
    inner_dim: int
    num_attention_heads: int
    attention_head_dim: int
    num_layers: int
    num_learnable_registers: int | None


def _infer_ltx23_connector_spec(*, prefix: str, state_dict: Mapping[str, Any]) -> _Ltx23ConnectorSpec:
    layer_prefix = f"{prefix}.transformer_1d_blocks."
    layer_indexes = sorted(
        {
            int(str(key)[len(layer_prefix) :].split(".", 1)[0])
            for key in state_dict
            if str(key).startswith(layer_prefix)
            and str(key)[len(layer_prefix) :].split(".", 1)[0].isdigit()
        }
    )
    if not layer_indexes:
        raise RuntimeError(f"LTX2 2.3 connector state is missing `{prefix}.transformer_1d_blocks.*` tensors.")
    expected_indexes = list(range(layer_indexes[-1] + 1))
    if layer_indexes != expected_indexes:
        raise RuntimeError(
            f"LTX2 2.3 connector state has non-contiguous layer indexes for {prefix!r}: {layer_indexes!r}."
        )

    q_weight = state_dict.get(f"{prefix}.transformer_1d_blocks.0.attn1.to_q.weight")
    gate_weight = state_dict.get(f"{prefix}.transformer_1d_blocks.0.attn1.to_gate_logits.weight")
    if not isinstance(q_weight, torch.Tensor) or q_weight.ndim != 2:
        raise RuntimeError(f"LTX2 2.3 connector state is missing a rank-2 `{prefix}.transformer_1d_blocks.0.attn1.to_q.weight` tensor.")
    if not isinstance(gate_weight, torch.Tensor) or gate_weight.ndim != 2:
        raise RuntimeError(
            f"LTX2 2.3 connector state is missing a rank-2 `{prefix}.transformer_1d_blocks.0.attn1.to_gate_logits.weight` tensor."
        )

    inner_dim = int(q_weight.shape[0])
    num_attention_heads = int(gate_weight.shape[0])
    if inner_dim <= 0 or num_attention_heads <= 0 or inner_dim % num_attention_heads != 0:
        raise RuntimeError(
            "LTX2 2.3 connector state has incompatible attention dimensions: "
            f"inner_dim={inner_dim} heads={num_attention_heads}."
        )

    learnable_registers = state_dict.get(f"{prefix}.learnable_registers")
    num_learnable_registers: int | None = None
    if learnable_registers is not None:
        if not isinstance(learnable_registers, torch.Tensor) or learnable_registers.ndim != 2:
            raise RuntimeError(
                f"LTX2 2.3 connector state `{prefix}.learnable_registers` must be a rank-2 tensor."
            )
        if int(learnable_registers.shape[1]) != inner_dim:
            raise RuntimeError(
                "LTX2 2.3 connector learnable registers width mismatch: "
                f"expected {inner_dim}, got {int(learnable_registers.shape[1])}."
            )
        num_learnable_registers = int(learnable_registers.shape[0])

    return _Ltx23ConnectorSpec(
        prefix=prefix,
        inner_dim=inner_dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=inner_dim // num_attention_heads,
        num_layers=len(layer_indexes),
        num_learnable_registers=num_learnable_registers,
    )


class _Ltx23Attention(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        heads: int,
        dim_head: int,
        rope_type: str,
    ) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.heads = int(heads)
        self.head_dim = int(dim_head)
        self.inner_dim = self.heads * self.head_dim
        self.rope_type = rope_type

        self.q_norm = torch.nn.RMSNorm(self.inner_dim, eps=1e-6, elementwise_affine=True)
        self.k_norm = torch.nn.RMSNorm(self.inner_dim, eps=1e-6, elementwise_affine=True)
        self.to_q = nn.Linear(self.query_dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(self.query_dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(self.query_dim, self.inner_dim, bias=True)
        self.to_gate_logits = nn.Linear(self.query_dim, self.heads, bias=True)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, self.query_dim, bias=True), nn.Identity())

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise RuntimeError(
                f"LTX2 2.3 connector attention expects [batch,tokens,channels], got {tuple(hidden_states.shape)!r}."
            )

        source_states = hidden_states
        batch_size, query_len, _ = hidden_states.shape
        query = self.q_norm(self.to_q(hidden_states))
        key = self.k_norm(self.to_k(hidden_states))
        value = self.to_v(hidden_states)

        if query_rotary_emb is not None:
            if self.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(key, key_rotary_emb or query_rotary_emb)
            else:
                query = apply_split_rotary_emb(query, query_rotary_emb)
                key = apply_split_rotary_emb(key, key_rotary_emb or query_rotary_emb)

        query = query.unflatten(2, (self.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (self.heads, -1)).transpose(1, 2)
        attention_bias = None
        if attention_mask is not None:
            attention_bias = attention_mask
            if attention_bias.ndim == 2:
                attention_bias = attention_bias[:, None, None, :]
            elif attention_bias.ndim == 3:
                attention_bias = attention_bias[:, None, :, :]
            elif attention_bias.ndim != 4:
                raise RuntimeError(
                    f"LTX2 2.3 connector attention mask must be 2D/3D/4D, got {attention_bias.ndim}D."
                )
            attention_bias = attention_bias.to(device=query.device, dtype=query.dtype).expand(
                batch_size,
                self.heads,
                query_len,
                int(key.shape[-2]),
            )

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, query_len, self.inner_dim)
        gate_logits = self.to_gate_logits(source_states)
        gates = 2.0 * torch.sigmoid(gate_logits)
        hidden_states = hidden_states.view(batch_size, query_len, self.heads, self.head_dim)
        hidden_states = hidden_states * gates.unsqueeze(-1)
        hidden_states = hidden_states.view(batch_size, query_len, self.inner_dim)
        return self.to_out(hidden_states)


class _Ltx23TransformerBlock1d(nn.Module):
    def __init__(self, *, dim: int, num_attention_heads: int, attention_head_dim: int, rope_type: str) -> None:
        super().__init__()
        self.attn1 = _Ltx23Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            rope_type=rope_type,
        )
        self.ff = _FeedForward(dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        norm_hidden_states = _rms_norm(hidden_states)
        hidden_states = self.attn1(
            norm_hidden_states,
            attention_mask=attention_mask,
            query_rotary_emb=rotary_emb,
        ) + hidden_states
        hidden_states = self.ff(_rms_norm(hidden_states)) + hidden_states
        return hidden_states


class _Ltx23Embeddings1DConnector(nn.Module):
    def __init__(
        self,
        *,
        inner_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        num_learnable_registers: int | None,
        rope_base_seq_len: int,
        rope_theta: float,
        rope_double_precision: bool,
        causal_temporal_positioning: bool,
        rope_type: str,
    ) -> None:
        super().__init__()
        self.inner_dim = int(inner_dim)
        self.num_attention_heads = int(num_attention_heads)
        self.causal_temporal_positioning = bool(causal_temporal_positioning)
        self.num_learnable_registers = num_learnable_registers
        self.learnable_registers = None
        if num_learnable_registers is not None:
            init_registers = (torch.rand(num_learnable_registers, self.inner_dim) * 2.0) - 1.0
            self.learnable_registers = nn.Parameter(init_registers)
        self.rope = _Ltx2RotaryPosEmbed1d(
            self.inner_dim,
            base_seq_len=rope_base_seq_len,
            theta=rope_theta,
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.num_attention_heads,
        )
        self.transformer_1d_blocks = nn.ModuleList(
            [
                _Ltx23TransformerBlock1d(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    rope_type=rope_type,
                )
                for _ in range(int(num_layers))
            ]
        )

    def _replace_padded_with_learnable_registers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        attn_mask_binarize_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len % int(self.num_learnable_registers) != 0:
            raise RuntimeError(
                "LTX2 2.3 connector sequence length must be divisible by the number of learnable registers; "
                f"got seq_len={seq_len} registers={self.num_learnable_registers}."
            )
        binary_mask = (attention_mask >= float(attn_mask_binarize_threshold)).int()
        if binary_mask.ndim == 4:
            binary_mask = binary_mask.squeeze(1).squeeze(1)
        if binary_mask.ndim != 2:
            raise RuntimeError(
                f"LTX2 2.3 connector binary mask must reduce to [batch,tokens], got {binary_mask.ndim}D."
            )
        register_repeats = seq_len // int(self.num_learnable_registers)
        registers = torch.tile(self.learnable_registers, (register_repeats, 1))
        non_padded = [hidden_states[index, binary_mask[index].bool(), :] for index in range(batch_size)]
        pad_lengths = [seq_len - item.shape[0] for item in non_padded]
        padded = [F.pad(item, pad=(0, 0, 0, pad), value=0.0) for item, pad in zip(non_padded, pad_lengths)]
        hidden_states = torch.cat([item.unsqueeze(0) for item in padded], dim=0)
        flipped = torch.flip(binary_mask.unsqueeze(-1), dims=[1])
        hidden_states = flipped * hidden_states + (1 - flipped) * registers
        attention_mask = torch.zeros_like(attention_mask)
        return hidden_states, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attn_mask_binarize_threshold: float = -9000.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hidden_states.ndim != 3:
            raise RuntimeError(
                f"LTX2 2.3 connector expects [batch,tokens,channels], got {tuple(hidden_states.shape)!r}."
            )
        if self.learnable_registers is not None:
            if attention_mask is None:
                raise RuntimeError("LTX2 2.3 connector with learnable registers requires an attention_mask.")
            hidden_states, attention_mask = self._replace_padded_with_learnable_registers(
                hidden_states,
                attention_mask,
                attn_mask_binarize_threshold=attn_mask_binarize_threshold,
            )

        rotary_emb = self.rope(hidden_states.shape[0], hidden_states.shape[1], device=hidden_states.device)
        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, rotary_emb=rotary_emb)
        return _rms_norm(hidden_states), attention_mask


class _Ltx23TextEmbeddingProjection(nn.Module):
    def __init__(self, *, packed_input_dim: int, video_dim: int, audio_dim: int) -> None:
        super().__init__()
        self.video_aggregate_embed = nn.Linear(int(packed_input_dim), int(video_dim), bias=True)
        self.audio_aggregate_embed = nn.Linear(int(packed_input_dim), int(audio_dim), bias=True)


class _Ltx23TextConnectors(nn.Module):
    def __init__(
        self,
        *,
        packed_input_dim: int,
        video_spec: _Ltx23ConnectorSpec,
        audio_spec: _Ltx23ConnectorSpec,
        connector_rope_base_seq_len: int,
        rope_theta: float,
        rope_double_precision: bool,
        causal_temporal_positioning: bool,
        rope_type: str,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            packed_input_dim=int(packed_input_dim),
            connector_rope_base_seq_len=int(connector_rope_base_seq_len),
            rope_theta=float(rope_theta),
            rope_double_precision=bool(rope_double_precision),
            causal_temporal_positioning=bool(causal_temporal_positioning),
            rope_type=rope_type,
        )
        self.text_embedding_projection = _Ltx23TextEmbeddingProjection(
            packed_input_dim=packed_input_dim,
            video_dim=video_spec.inner_dim,
            audio_dim=audio_spec.inner_dim,
        )
        self.video_embeddings_connector = _Ltx23Embeddings1DConnector(
            inner_dim=video_spec.inner_dim,
            num_attention_heads=video_spec.num_attention_heads,
            attention_head_dim=video_spec.attention_head_dim,
            num_layers=video_spec.num_layers,
            num_learnable_registers=video_spec.num_learnable_registers,
            rope_base_seq_len=connector_rope_base_seq_len,
            rope_theta=rope_theta,
            rope_double_precision=rope_double_precision,
            causal_temporal_positioning=causal_temporal_positioning,
            rope_type=rope_type,
        )
        self.audio_embeddings_connector = _Ltx23Embeddings1DConnector(
            inner_dim=audio_spec.inner_dim,
            num_attention_heads=audio_spec.num_attention_heads,
            attention_head_dim=audio_spec.attention_head_dim,
            num_layers=audio_spec.num_layers,
            num_learnable_registers=audio_spec.num_learnable_registers,
            rope_base_seq_len=connector_rope_base_seq_len,
            rope_theta=rope_theta,
            rope_double_precision=rope_double_precision,
            causal_temporal_positioning=causal_temporal_positioning,
            rope_type=rope_type,
        )

    @classmethod
    def from_state_dict_and_config(
        cls,
        *,
        config: Mapping[str, Any],
        state_dict: Mapping[str, Any],
    ) -> "_Ltx23TextConnectors":
        rope_type = _require_str(config, "rope_type")
        if rope_type not in {"interleaved", "split"}:
            raise RuntimeError(f"LTX2 connectors rope_type must be 'interleaved' or 'split', got {rope_type!r}.")

        video_spec = _infer_ltx23_connector_spec(prefix="video_embeddings_connector", state_dict=state_dict)
        audio_spec = _infer_ltx23_connector_spec(prefix="audio_embeddings_connector", state_dict=state_dict)

        video_projection = state_dict.get("text_embedding_projection.video_aggregate_embed.weight")
        audio_projection = state_dict.get("text_embedding_projection.audio_aggregate_embed.weight")
        if not isinstance(video_projection, torch.Tensor) or video_projection.ndim != 2:
            raise RuntimeError("LTX2 2.3 connector state is missing `text_embedding_projection.video_aggregate_embed.weight`.")
        if not isinstance(audio_projection, torch.Tensor) or audio_projection.ndim != 2:
            raise RuntimeError("LTX2 2.3 connector state is missing `text_embedding_projection.audio_aggregate_embed.weight`.")
        if int(video_projection.shape[0]) != video_spec.inner_dim:
            raise RuntimeError(
                "LTX2 2.3 video aggregate projection output dim mismatch: "
                f"expected {video_spec.inner_dim}, got {int(video_projection.shape[0])}."
            )
        if int(audio_projection.shape[0]) != audio_spec.inner_dim:
            raise RuntimeError(
                "LTX2 2.3 audio aggregate projection output dim mismatch: "
                f"expected {audio_spec.inner_dim}, got {int(audio_projection.shape[0])}."
            )
        if int(video_projection.shape[1]) != int(audio_projection.shape[1]):
            raise RuntimeError(
                "LTX2 2.3 text aggregate projections must share the same packed input width; "
                f"got video={int(video_projection.shape[1])} audio={int(audio_projection.shape[1])}."
            )

        return cls(
            packed_input_dim=int(video_projection.shape[1]),
            video_spec=video_spec,
            audio_spec=audio_spec,
            connector_rope_base_seq_len=_require_int(config, "connector_rope_base_seq_len"),
            rope_theta=_require_float(config, "rope_theta"),
            rope_double_precision=_require_bool(config, "rope_double_precision"),
            causal_temporal_positioning=_require_bool(config, "causal_temporal_positioning"),
            rope_type=rope_type,
        )

    def forward(
        self,
        text_encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        additive_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if text_encoder_hidden_states.ndim != 3:
            raise RuntimeError(
                "LTX2 2.3 connectors expect packed text encoder hidden states with shape [batch,tokens,channels]; "
                f"got {tuple(text_encoder_hidden_states.shape)!r}."
            )
        if attention_mask is None:
            raise RuntimeError("LTX2 2.3 connectors require an attention_mask.")
        if not additive_mask:
            text_dtype = text_encoder_hidden_states.dtype
            attention_mask = (attention_mask - 1).reshape(
                attention_mask.shape[0],
                1,
                -1,
                attention_mask.shape[-1],
            )
            attention_mask = attention_mask.to(text_dtype) * torch.finfo(text_dtype).max

        video_hidden_states = self.text_embedding_projection.video_aggregate_embed(text_encoder_hidden_states)
        audio_hidden_states = self.text_embedding_projection.audio_aggregate_embed(text_encoder_hidden_states)
        video_text_embedding, new_attn_mask = self.video_embeddings_connector(video_hidden_states, attention_mask)
        attn_mask = (new_attn_mask < 1e-6).to(torch.int64)
        attn_mask = attn_mask.reshape(video_text_embedding.shape[0], video_text_embedding.shape[1], 1)
        video_text_embedding = video_text_embedding * attn_mask
        new_attn_mask = attn_mask.squeeze(-1)
        audio_text_embedding, _ = self.audio_embeddings_connector(audio_hidden_states, attention_mask)
        return video_text_embedding, audio_text_embedding, new_attn_mask


def _resolve_connector_state_dict_view(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(
            "LTX2 connectors state_dict must be a mapping. "
            f"Got {type(state_dict).__name__}."
        )
    wrapped_keys = 0
    direct_keys = 0
    wrapped_lookup: dict[str, str] = {}
    for raw_key in state_dict.keys():
        if not isinstance(raw_key, str):
            raise RuntimeError(
                "LTX2 connectors state_dict keys must be strings. "
                f"Got {type(raw_key).__name__}."
            )
        if raw_key.startswith(_CONNECTORS_WRAPPER_PREFIX):
            suffix = raw_key[len(_CONNECTORS_WRAPPER_PREFIX) :]
            if not suffix:
                raise RuntimeError("LTX2 connectors state contains an empty `connectors.` wrapper key.")
            previous = wrapped_lookup.get(suffix)
            if previous is not None and previous != raw_key:
                raise RuntimeError(
                    "LTX2 connectors wrapped layout collides after explicit keyspace mapping: "
                    f"dst={suffix!r} srcs={previous!r},{raw_key!r}."
                )
            wrapped_lookup[suffix] = raw_key
            wrapped_keys += 1
            continue
        direct_keys += 1

    if wrapped_keys and direct_keys:
        raise RuntimeError(
            "LTX2 connectors state mixes wrapped `connectors.*` keys with direct keys; "
            "supported layouts are all-direct or all-wrapped only."
        )
    if wrapped_keys:
        return KeyspaceLookupView(state_dict, wrapped_lookup)
    return state_dict


def load_ltx2_connectors(
    *,
    config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    device: torch.device,
    torch_dtype: torch.dtype,
) -> Ltx2TextConnectors:
    resolved_state_dict = _resolve_connector_state_dict_view(state_dict)
    dequantized_state_dict = _DequantizedConnectorStateDictView(resolved_state_dict)
    if any(key.startswith(_REAL_23_CONNECTOR_PREFIXES) for key in resolved_state_dict.keys()):
        module = _Ltx23TextConnectors.from_state_dict_and_config(
            config=config,
            state_dict=dequantized_state_dict,
        )
    else:
        module = Ltx2TextConnectors.from_config(config)
    try:
        missing, unexpected = safe_load_state_dict(
            module,
            dequantized_state_dict,
            log_name="LTX2TextConnectors",
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 connectors state load failed: {exc}") from exc
    if missing or unexpected:
        raise RuntimeError(
            "LTX2 connectors strict load failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
        )
    try:
        module = module.to(device=device, dtype=torch_dtype)
    except Exception:
        module = module.to(device=device)
    module.eval()
    return module
