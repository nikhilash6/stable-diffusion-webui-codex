"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native LTX2 audiovisual transformer with parser-state-compatible parameter names.
Implements the LTX2 transformer under `apps/**` without importing the official Diffusers LTX2 model class while
preserving the parser-produced raw/original state-dict surface (`patchify_proj`, `adaln_single`, `q_norm`,
`scale_shift_table_a2v_ca_*`, etc.).

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2VideoTransformer3DModel` (class): Config/state-driven native LTX2 audiovisual transformer.
- `__all__` (constant): Explicit public export list for runtime imports.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["Ltx2VideoTransformer3DModel"]


_CONFIG_META_KEYS = frozenset({"_class_name", "_diffusers_version", "_name_or_path", "architectures"})
_ALLOWED_CLASS_NAMES = frozenset({"LTX2VideoTransformer3DModel"})


def _require_mapping(config: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise RuntimeError(f"LTX2 {label} config must be a mapping, got {type(config).__name__}.")
    return dict(config)


def _validate_class_name(raw: dict[str, Any], *, allowed: Sequence[str], label: str) -> None:
    class_name = raw.get("_class_name")
    if class_name is None:
        return
    if str(class_name) not in allowed:
        raise RuntimeError(
            f"LTX2 {label} config `_class_name` must be one of {tuple(allowed)!r}, got {class_name!r}."
        )


def _reject_unexpected_keys(raw: dict[str, Any], *, allowed: set[str], label: str) -> None:
    unexpected = sorted(set(raw) - allowed - _CONFIG_META_KEYS)
    if unexpected:
        raise RuntimeError(f"LTX2 {label} config has unsupported keys: {unexpected!r}.")


def _as_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"LTX2 config field {name!r} must be an int, got bool.")
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 config field {name!r} must be an int, got {value!r}.") from exc


def _as_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 config field {name!r} must be a float, got {value!r}.") from exc


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise RuntimeError(f"LTX2 config field {name!r} must be a bool, got {value!r}.")


def _as_str(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"LTX2 config field {name!r} must be a string, got {type(value).__name__}.")
    return value


def _as_tuple(value: Any, *, name: str, item_type: type[int] | type[bool] | type[float]) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"LTX2 config field {name!r} must be a list/tuple, got {type(value).__name__}.")
    caster = {int: _as_int, bool: _as_bool, float: _as_float}[item_type]
    return tuple(caster(v, name=f"{name}[]") for v in value)


def _get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise RuntimeError(f"LTX2 timestep embedding expects a 1D tensor, got shape={tuple(timesteps.shape)!r}.")
    half_dim = embedding_dim // 2
    if half_dim <= 0:
        raise RuntimeError(f"LTX2 timestep embedding dimension must be >= 2, got {embedding_dim!r}.")

    exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class _Timesteps(nn.Module):
    def __init__(self, num_channels: int, *, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: float = 1.0):
        super().__init__()
        self.num_channels = int(num_channels)
        self.flip_sin_to_cos = bool(flip_sin_to_cos)
        self.downscale_freq_shift = float(downscale_freq_shift)
        self.scale = float(scale)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return _get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class _TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, *, act_fn: str = "silu") -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        if act_fn != "silu":
            raise RuntimeError(f"LTX2 timestep embedding only supports act_fn='silu', got {act_fn!r}.")
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class _PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, size_emb_dim: int, *, use_additional_conditions: bool = False) -> None:
        super().__init__()
        self.outdim = int(size_emb_dim)
        self.time_proj = _Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = _TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.use_additional_conditions = bool(use_additional_conditions)
        if self.use_additional_conditions:
            raise RuntimeError("LTX2 native transformer does not support additional PixArt size conditions.")

    def forward(
        self,
        timestep: torch.Tensor,
        *,
        resolution: torch.Tensor | None,
        aspect_ratio: torch.Tensor | None,
        batch_size: int | None,
        hidden_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        del resolution, aspect_ratio, batch_size
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype or timesteps_proj.dtype))


class _PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, *, out_features: int | None = None, act_fn: str = "gelu_tanh"):
        super().__init__()
        out_features = hidden_size if out_features is None else out_features
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        if act_fn != "gelu_tanh":
            raise RuntimeError(f"LTX2 text projection only supports act_fn='gelu_tanh', got {act_fn!r}.")
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(hidden_size, out_features, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(dtype=hidden_dtype)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        return hidden_states


class _GELUProj(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, approximate: str = "none", bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(hidden_states), approximate=self.approximate)


class _FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        bias: bool = True,
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        if activation_fn == "gelu-approximate":
            act_fn: nn.Module = _GELUProj(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "gelu":
            act_fn = _GELUProj(dim, inner_dim, approximate="none", bias=bias)
        else:
            raise RuntimeError(f"LTX2 feed-forward activation {activation_fn!r} is unsupported.")
        self.net = nn.ModuleList([act_fn, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out, bias=bias)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


@dataclass(slots=True)
class _AudioVisualModelOutput:
    sample: torch.Tensor
    audio_sample: torch.Tensor


class _LTX2AdaLayerNormSingle(nn.Module):
    def __init__(self, embedding_dim: int, *, num_mod_params: int = 6, use_additional_conditions: bool = False) -> None:
        super().__init__()
        self.num_mod_params = int(num_mod_params)
        self.emb = _PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, self.num_mod_params * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        *,
        batch_size: int | None,
        hidden_dtype: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(
            timestep,
            resolution=None,
            aspect_ratio=None,
            batch_size=batch_size,
            hidden_dtype=hidden_dtype,
        )
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


def _apply_interleaved_rotary_emb(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return (x.float() * cos + x_rotated.float() * sin).to(dtype=x.dtype)


def _apply_split_rotary_emb(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs
    x_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        batch = x.shape[0]
        _, heads, tokens, _ = cos.shape
        x = x.reshape(batch, tokens, heads, -1).swapaxes(1, 2)
        needs_reshape = True

    last_dim = x.shape[-1]
    if last_dim % 2 != 0:
        raise RuntimeError(f"LTX2 split rotary embedding expects an even channel dim, got {last_dim!r}.")
    split_x = x.reshape(*x.shape[:-1], 2, last_dim // 2).float()
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]

    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)
    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]
    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)
    out = out.reshape(*out.shape[:-2], last_dim)

    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(batch, tokens, -1)
    return out.to(dtype=x_dtype)


class _LTX2Attention(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        heads: int,
        kv_heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = True,
        cross_attention_dim: int | None = None,
        out_bias: bool = True,
        qk_norm: str = "rms_norm_across_heads",
        norm_eps: float = 1e-6,
        norm_elementwise_affine: bool = True,
        rope_type: str = "interleaved",
    ) -> None:
        super().__init__()
        if qk_norm != "rms_norm_across_heads":
            raise RuntimeError(f"LTX2 attention only supports qk_norm='rms_norm_across_heads', got {qk_norm!r}.")
        if rope_type not in {"interleaved", "split"}:
            raise RuntimeError(f"LTX2 attention rope_type must be 'interleaved' or 'split', got {rope_type!r}.")

        self.head_dim = int(dim_head)
        self.inner_dim = int(dim_head * heads)
        self.inner_kv_dim = int(dim_head * (heads if kv_heads is None else kv_heads))
        self.query_dim = int(query_dim)
        self.cross_attention_dim = int(query_dim if cross_attention_dim is None else cross_attention_dim)
        self.dropout = float(dropout)
        self.out_dim = int(query_dim)
        self.heads = int(heads)
        self.kv_heads = int(heads if kv_heads is None else kv_heads)
        self.rope_type = rope_type

        self.q_norm = _RMSNorm(dim_head * heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.k_norm = _RMSNorm(dim_head * self.kv_heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.to_q = nn.Linear(self.query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, self.out_dim, bias=out_bias),
            nn.Dropout(dropout),
        ])

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        *,
        batch_size: int,
        query_length: int,
        key_length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.ndim == 3:
            if attention_mask.shape[1] == 1:
                attention_mask = attention_mask[:, None, :, :]
            else:
                attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim != 4:
            raise RuntimeError(
                "LTX2 attention mask must have rank 2, 3, or 4; "
                f"got shape={tuple(attention_mask.shape)!r}."
            )

        if attention_mask.shape[0] != batch_size:
            raise RuntimeError(
                f"LTX2 attention mask batch mismatch: expected {batch_size}, got {attention_mask.shape[0]}."
            )
        if attention_mask.shape[-1] != key_length:
            raise RuntimeError(
                f"LTX2 attention mask key length mismatch: expected {key_length}, got {attention_mask.shape[-1]}."
            )
        if attention_mask.shape[-2] not in {1, query_length}:
            raise RuntimeError(
                "LTX2 attention mask query length mismatch: "
                f"expected 1 or {query_length}, got {attention_mask.shape[-2]}."
            )
        return attention_mask.to(dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch_size, query_length, _ = hidden_states.shape
        _, key_length, _ = encoder_hidden_states.shape

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = self.q_norm(query)
        key = self.k_norm(key)

        if query_rotary_emb is not None:
            if self.rope_type == "interleaved":
                query = _apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = _apply_interleaved_rotary_emb(key, key_rotary_emb or query_rotary_emb)
            else:
                query = _apply_split_rotary_emb(query, query_rotary_emb)
                key = _apply_split_rotary_emb(key, key_rotary_emb or query_rotary_emb)

        query = query.view(batch_size, query_length, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, key_length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, key_length, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.kv_heads != self.heads:
            if self.heads % self.kv_heads != 0:
                raise RuntimeError(
                    f"LTX2 attention requires heads % kv_heads == 0, got heads={self.heads} kv_heads={self.kv_heads}."
                )
            repeat = self.heads // self.kv_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        attn_mask = self._prepare_attention_mask(
            attention_mask,
            batch_size=batch_size,
            query_length=query_length,
            key_length=key_length,
            dtype=query.dtype,
        )
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, query_length, self.inner_dim)
        hidden_states = hidden_states.to(dtype=query.dtype)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class _LTX2VideoTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_dim: int,
        audio_num_attention_heads: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        rope_type: str = "interleaved",
    ) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = _LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        self.audio_norm1 = _RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_attn1 = _LTX2Attention(
            query_dim=audio_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        self.norm2 = _RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = _LTX2Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        self.audio_norm2 = _RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_attn2 = _LTX2Attention(
            query_dim=audio_dim,
            cross_attention_dim=audio_cross_attention_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        self.audio_to_video_norm = _RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_to_video_attn = _LTX2Attention(
            query_dim=dim,
            cross_attention_dim=audio_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        self.video_to_audio_norm = _RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.video_to_audio_attn = _LTX2Attention(
            query_dim=audio_dim,
            cross_attention_dim=dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        self.norm3 = _RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.ff = _FeedForward(dim, activation_fn=activation_fn)

        self.audio_norm3 = _RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_ff = _FeedForward(audio_dim, activation_fn=activation_fn)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)
        self.audio_scale_shift_table = nn.Parameter(torch.randn(6, audio_dim) / audio_dim**0.5)
        self.scale_shift_table_a2v_ca_video = nn.Parameter(torch.randn(5, dim))
        self.scale_shift_table_a2v_ca_audio = nn.Parameter(torch.randn(5, audio_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        temb_audio: torch.Tensor,
        temb_ca_scale_shift: torch.Tensor,
        temb_ca_audio_scale_shift: torch.Tensor,
        temb_ca_gate: torch.Tensor,
        temb_ca_audio_gate: torch.Tensor,
        video_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        ca_video_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        ca_audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        a2v_cross_attention_mask: torch.Tensor | None = None,
        v2a_cross_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)

        norm_hidden_states = self.norm1(hidden_states)
        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None].to(dtype=temb.dtype, device=temb.device) + temb.reshape(
            batch_size, temb.size(1), num_ada_params, -1
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            query_rotary_emb=video_rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        norm_audio_hidden_states = self.audio_norm1(audio_hidden_states)
        num_audio_ada_params = self.audio_scale_shift_table.shape[0]
        audio_ada_values = self.audio_scale_shift_table[None, None].to(
            dtype=temb_audio.dtype,
            device=temb_audio.device,
        ) + temb_audio.reshape(batch_size, temb_audio.size(1), num_audio_ada_params, -1)
        (
            audio_shift_msa,
            audio_scale_msa,
            audio_gate_msa,
            audio_shift_mlp,
            audio_scale_mlp,
            audio_gate_mlp,
        ) = audio_ada_values.unbind(dim=2)
        norm_audio_hidden_states = norm_audio_hidden_states * (1 + audio_scale_msa) + audio_shift_msa
        attn_audio_hidden_states = self.audio_attn1(
            hidden_states=norm_audio_hidden_states,
            encoder_hidden_states=None,
            query_rotary_emb=audio_rotary_emb,
        )
        audio_hidden_states = audio_hidden_states + attn_audio_hidden_states * audio_gate_msa

        norm_hidden_states = self.norm2(hidden_states)
        attn_hidden_states = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states

        norm_audio_hidden_states = self.audio_norm2(audio_hidden_states)
        attn_audio_hidden_states = self.audio_attn2(
            norm_audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=audio_encoder_attention_mask,
        )
        audio_hidden_states = audio_hidden_states + attn_audio_hidden_states

        norm_hidden_states = self.audio_to_video_norm(hidden_states)
        norm_audio_hidden_states = self.video_to_audio_norm(audio_hidden_states)

        video_per_layer_ca_scale_shift = self.scale_shift_table_a2v_ca_video[:4, :]
        video_per_layer_ca_gate = self.scale_shift_table_a2v_ca_video[4:, :]
        video_ca_scale_shift_table = (
            video_per_layer_ca_scale_shift.to(dtype=temb_ca_scale_shift.dtype, device=temb_ca_scale_shift.device)
            + temb_ca_scale_shift.reshape(batch_size, temb_ca_scale_shift.shape[1], 4, -1)
        ).unbind(dim=2)
        video_ca_gate = (
            video_per_layer_ca_gate.to(dtype=temb_ca_gate.dtype, device=temb_ca_gate.device)
            + temb_ca_gate.reshape(batch_size, temb_ca_gate.shape[1], 1, -1)
        ).unbind(dim=2)
        video_a2v_ca_scale, video_a2v_ca_shift, video_v2a_ca_scale, video_v2a_ca_shift = video_ca_scale_shift_table
        a2v_gate = video_ca_gate[0].squeeze(2)

        audio_per_layer_ca_scale_shift = self.scale_shift_table_a2v_ca_audio[:4, :]
        audio_per_layer_ca_gate = self.scale_shift_table_a2v_ca_audio[4:, :]
        audio_ca_scale_shift_table = (
            audio_per_layer_ca_scale_shift.to(
                dtype=temb_ca_audio_scale_shift.dtype,
                device=temb_ca_audio_scale_shift.device,
            )
            + temb_ca_audio_scale_shift.reshape(batch_size, temb_ca_audio_scale_shift.shape[1], 4, -1)
        ).unbind(dim=2)
        audio_ca_gate = (
            audio_per_layer_ca_gate.to(dtype=temb_ca_audio_gate.dtype, device=temb_ca_audio_gate.device)
            + temb_ca_audio_gate.reshape(batch_size, temb_ca_audio_gate.shape[1], 1, -1)
        ).unbind(dim=2)
        audio_a2v_ca_scale, audio_a2v_ca_shift, audio_v2a_ca_scale, audio_v2a_ca_shift = audio_ca_scale_shift_table
        v2a_gate = audio_ca_gate[0].squeeze(2)

        mod_norm_hidden_states = norm_hidden_states * (1 + video_a2v_ca_scale.squeeze(2)) + video_a2v_ca_shift.squeeze(2)
        mod_norm_audio_hidden_states = (
            norm_audio_hidden_states * (1 + audio_a2v_ca_scale.squeeze(2)) + audio_a2v_ca_shift.squeeze(2)
        )
        a2v_attn_hidden_states = self.audio_to_video_attn(
            mod_norm_hidden_states,
            encoder_hidden_states=mod_norm_audio_hidden_states,
            query_rotary_emb=ca_video_rotary_emb,
            key_rotary_emb=ca_audio_rotary_emb,
            attention_mask=a2v_cross_attention_mask,
        )
        hidden_states = hidden_states + a2v_gate * a2v_attn_hidden_states

        mod_norm_hidden_states = norm_hidden_states * (1 + video_v2a_ca_scale.squeeze(2)) + video_v2a_ca_shift.squeeze(2)
        mod_norm_audio_hidden_states = (
            norm_audio_hidden_states * (1 + audio_v2a_ca_scale.squeeze(2)) + audio_v2a_ca_shift.squeeze(2)
        )
        v2a_attn_hidden_states = self.video_to_audio_attn(
            mod_norm_audio_hidden_states,
            encoder_hidden_states=mod_norm_hidden_states,
            query_rotary_emb=ca_audio_rotary_emb,
            key_rotary_emb=ca_video_rotary_emb,
            attention_mask=v2a_cross_attention_mask,
        )
        audio_hidden_states = audio_hidden_states + v2a_gate * v2a_attn_hidden_states

        norm_hidden_states = self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + self.ff(norm_hidden_states) * gate_mlp

        norm_audio_hidden_states = self.audio_norm3(audio_hidden_states) * (1 + audio_scale_mlp) + audio_shift_mlp
        audio_hidden_states = audio_hidden_states + self.audio_ff(norm_audio_hidden_states) * audio_gate_mlp

        return hidden_states, audio_hidden_states


class _LTX2AudioVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        scale_factors: Sequence[int] = (8, 32, 32),
        theta: float = 10000.0,
        causal_offset: int = 1,
        modality: str = "video",
        double_precision: bool = True,
        rope_type: str = "interleaved",
        num_attention_heads: int = 32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.patch_size = int(patch_size)
        self.patch_size_t = int(patch_size_t)
        self.base_num_frames = int(base_num_frames)
        self.base_height = int(base_height)
        self.base_width = int(base_width)
        self.sampling_rate = int(sampling_rate)
        self.hop_length = int(hop_length)
        self.scale_factors = tuple(int(v) for v in scale_factors)
        self.theta = float(theta)
        self.causal_offset = int(causal_offset)
        self.modality = modality
        self.double_precision = bool(double_precision)
        self.rope_type = rope_type
        self.num_attention_heads = int(num_attention_heads)
        if self.modality not in {"video", "audio"}:
            raise RuntimeError(f"LTX2 RoPE modality must be 'video' or 'audio', got {modality!r}.")
        if self.rope_type not in {"interleaved", "split"}:
            raise RuntimeError(f"LTX2 RoPE type must be 'interleaved' or 'split', got {rope_type!r}.")

    def prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        *,
        fps: float = 24.0,
    ) -> torch.Tensor:
        if num_frames is None or height is None or width is None:
            raise RuntimeError("LTX2 video RoPE coordinate generation requires num_frames, height, and width.")
        grid_f = torch.arange(0, num_frames, self.patch_size_t, dtype=torch.float32, device=device)
        grid_h = torch.arange(0, height, self.patch_size, dtype=torch.float32, device=device)
        grid_w = torch.arange(0, width, self.patch_size, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)
        patch_size = (self.patch_size_t, self.patch_size, self.patch_size)
        patch_size_delta = torch.tensor(patch_size, dtype=grid.dtype, device=grid.device)
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)
        latent_coords = torch.stack([grid, patch_ends], dim=-1).flatten(1, 3)
        latent_coords = latent_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        scale_tensor = torch.tensor(self.scale_factors, dtype=latent_coords.dtype, device=device)
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)
        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + self.causal_offset - self.scale_factors[0]).clamp(min=0)
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / float(fps)
        return pixel_coords

    def prepare_audio_coords(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        *,
        shift: int = 0,
    ) -> torch.Tensor:
        if num_frames is None:
            raise RuntimeError("LTX2 audio RoPE coordinate generation requires audio_num_frames.")
        grid_f = torch.arange(shift, num_frames + shift, self.patch_size_t, dtype=torch.float32, device=device)
        audio_scale_factor = self.scale_factors[0]
        grid_start_mel = (grid_f * audio_scale_factor + self.causal_offset - audio_scale_factor).clamp(min=0)
        grid_start_s = grid_start_mel * self.hop_length / self.sampling_rate
        grid_end_mel = ((grid_f + self.patch_size_t) * audio_scale_factor + self.causal_offset - audio_scale_factor).clamp(
            min=0
        )
        grid_end_s = grid_end_mel * self.hop_length / self.sampling_rate
        audio_coords = torch.stack([grid_start_s, grid_end_s], dim=-1)
        audio_coords = audio_coords.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1)
        return audio_coords

    def forward(
        self,
        coords: torch.Tensor,
        *,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = coords.device if device is None else torch.device(device)
        num_pos_dims = coords.shape[1]
        if coords.ndim == 4:
            coords_start, coords_end = coords.chunk(2, dim=-1)
            coords = ((coords_start + coords_end) / 2.0).squeeze(-1)

        if self.modality == "video":
            max_positions = (self.base_num_frames, self.base_height, self.base_width)
        else:
            max_positions = (self.base_num_frames,)
        grid = torch.stack([coords[:, i] / max_positions[i] for i in range(num_pos_dims)], dim=-1).to(device=device)
        num_rope_elems = num_pos_dims * 2
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        pow_indices = torch.pow(
            self.theta,
            torch.linspace(
                0.0,
                1.0,
                steps=self.dim // num_rope_elems,
                dtype=freqs_dtype,
                device=device,
            ),
        )
        freqs = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)
        freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs
        freqs = freqs.transpose(-1, -2).flatten(2)

        if self.rope_type == "interleaved":
            cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
            sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)
            if self.dim % num_rope_elems != 0:
                pad = self.dim % num_rope_elems
                cos_padding = torch.ones_like(cos_freqs[:, :, :pad])
                sin_padding = torch.zeros_like(cos_freqs[:, :, :pad])
                cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
                sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)
            return cos_freqs, sin_freqs

        expected_freqs = self.dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq = freqs.cos()
        sin_freq = freqs.sin()
        if pad_size != 0:
            cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
            sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])
            cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
            sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
        batch, tokens = cos_freq.shape[:2]
        cos_freq = cos_freq.reshape(batch, tokens, self.num_attention_heads, -1).swapaxes(1, 2)
        sin_freq = sin_freq.reshape(batch, tokens, self.num_attention_heads, -1).swapaxes(1, 2)
        return cos_freq, sin_freq


class Ltx2VideoTransformer3DModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 128,
        out_channels: int | None = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        cross_attention_dim: int = 4096,
        vae_scale_factors: Sequence[int] = (8, 32, 32),
        pos_embed_max_pos: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        audio_in_channels: int = 128,
        audio_out_channels: int | None = 128,
        audio_patch_size: int = 1,
        audio_patch_size_t: int = 1,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_cross_attention_dim: int = 2048,
        audio_scale_factor: int = 4,
        audio_pos_embed_max_pos: int = 20,
        audio_sampling_rate: int = 16000,
        audio_hop_length: int = 160,
        num_layers: int = 48,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 3840,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_offset: int = 1,
        timestep_scale_multiplier: int = 1000,
        cross_attn_timestep_scale_multiplier: int = 1000,
        rope_type: str = "interleaved",
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        audio_out_channels = audio_in_channels if audio_out_channels is None else audio_out_channels
        vae_scale_factors = tuple(int(v) for v in vae_scale_factors)
        if len(vae_scale_factors) != 3:
            raise RuntimeError(f"LTX2 transformer vae_scale_factors must have length 3, got {vae_scale_factors!r}.")
        if rope_type not in {"interleaved", "split"}:
            raise RuntimeError(f"LTX2 transformer rope_type must be 'interleaved' or 'split', got {rope_type!r}.")
        inner_dim = int(num_attention_heads * attention_head_dim)
        audio_inner_dim = int(audio_num_attention_heads * audio_attention_head_dim)
        self.config = SimpleNamespace(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            patch_size=int(patch_size),
            patch_size_t=int(patch_size_t),
            num_attention_heads=int(num_attention_heads),
            attention_head_dim=int(attention_head_dim),
            cross_attention_dim=int(cross_attention_dim),
            vae_scale_factors=vae_scale_factors,
            pos_embed_max_pos=int(pos_embed_max_pos),
            base_height=int(base_height),
            base_width=int(base_width),
            audio_in_channels=int(audio_in_channels),
            audio_out_channels=int(audio_out_channels),
            audio_patch_size=int(audio_patch_size),
            audio_patch_size_t=int(audio_patch_size_t),
            audio_num_attention_heads=int(audio_num_attention_heads),
            audio_attention_head_dim=int(audio_attention_head_dim),
            audio_cross_attention_dim=int(audio_cross_attention_dim),
            audio_scale_factor=int(audio_scale_factor),
            audio_pos_embed_max_pos=int(audio_pos_embed_max_pos),
            audio_sampling_rate=int(audio_sampling_rate),
            audio_hop_length=int(audio_hop_length),
            num_layers=int(num_layers),
            activation_fn=str(activation_fn),
            qk_norm=str(qk_norm),
            norm_elementwise_affine=bool(norm_elementwise_affine),
            norm_eps=float(norm_eps),
            caption_channels=int(caption_channels),
            attention_bias=bool(attention_bias),
            attention_out_bias=bool(attention_out_bias),
            rope_theta=float(rope_theta),
            rope_double_precision=bool(rope_double_precision),
            causal_offset=int(causal_offset),
            timestep_scale_multiplier=int(timestep_scale_multiplier),
            cross_attn_timestep_scale_multiplier=int(cross_attn_timestep_scale_multiplier),
            rope_type=str(rope_type),
        )

        self.patchify_proj = nn.Linear(in_channels, inner_dim)
        self.audio_patchify_proj = nn.Linear(audio_in_channels, audio_inner_dim)
        self.caption_projection = _PixArtAlphaTextProjection(caption_channels, inner_dim)
        self.audio_caption_projection = _PixArtAlphaTextProjection(caption_channels, audio_inner_dim)

        self.adaln_single = _LTX2AdaLayerNormSingle(inner_dim, num_mod_params=6, use_additional_conditions=False)
        self.audio_adaln_single = _LTX2AdaLayerNormSingle(
            audio_inner_dim,
            num_mod_params=6,
            use_additional_conditions=False,
        )
        self.av_ca_video_scale_shift_adaln_single = _LTX2AdaLayerNormSingle(
            inner_dim,
            num_mod_params=4,
            use_additional_conditions=False,
        )
        self.av_ca_audio_scale_shift_adaln_single = _LTX2AdaLayerNormSingle(
            audio_inner_dim,
            num_mod_params=4,
            use_additional_conditions=False,
        )
        self.av_ca_a2v_gate_adaln_single = _LTX2AdaLayerNormSingle(
            inner_dim,
            num_mod_params=1,
            use_additional_conditions=False,
        )
        self.av_ca_v2a_gate_adaln_single = _LTX2AdaLayerNormSingle(
            audio_inner_dim,
            num_mod_params=1,
            use_additional_conditions=False,
        )

        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.audio_scale_shift_table = nn.Parameter(torch.randn(2, audio_inner_dim) / audio_inner_dim**0.5)

        self.rope = _LTX2AudioVideoRotaryPosEmbed(
            dim=inner_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            scale_factors=vae_scale_factors,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        self.audio_rope = _LTX2AudioVideoRotaryPosEmbed(
            dim=audio_inner_dim,
            patch_size=audio_patch_size,
            patch_size_t=audio_patch_size_t,
            base_num_frames=audio_pos_embed_max_pos,
            sampling_rate=audio_sampling_rate,
            hop_length=audio_hop_length,
            scale_factors=(audio_scale_factor,),
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=audio_num_attention_heads,
        )
        cross_attn_pos_embed_max_pos = max(pos_embed_max_pos, audio_pos_embed_max_pos)
        self.cross_attn_rope = _LTX2AudioVideoRotaryPosEmbed(
            dim=audio_cross_attention_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        self.cross_attn_audio_rope = _LTX2AudioVideoRotaryPosEmbed(
            dim=audio_cross_attention_dim,
            patch_size=audio_patch_size,
            patch_size_t=audio_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            sampling_rate=audio_sampling_rate,
            hop_length=audio_hop_length,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=audio_num_attention_heads,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                _LTX2VideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    audio_dim=audio_inner_dim,
                    audio_num_attention_heads=audio_num_attention_heads,
                    audio_attention_head_dim=audio_attention_head_dim,
                    audio_cross_attention_dim=audio_cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    rope_type=rope_type,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)
        self.audio_norm_out = nn.LayerNorm(audio_inner_dim, eps=1e-6, elementwise_affine=False)
        self.audio_proj_out = nn.Linear(audio_inner_dim, audio_out_channels)
        self.gradient_checkpointing = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Ltx2VideoTransformer3DModel":
        raw = _require_mapping(config, label="transformer")
        _validate_class_name(raw, allowed=_ALLOWED_CLASS_NAMES, label="transformer")
        expected = {
            "in_channels",
            "out_channels",
            "patch_size",
            "patch_size_t",
            "num_attention_heads",
            "attention_head_dim",
            "cross_attention_dim",
            "vae_scale_factors",
            "pos_embed_max_pos",
            "base_height",
            "base_width",
            "audio_in_channels",
            "audio_out_channels",
            "audio_patch_size",
            "audio_patch_size_t",
            "audio_num_attention_heads",
            "audio_attention_head_dim",
            "audio_cross_attention_dim",
            "audio_scale_factor",
            "audio_pos_embed_max_pos",
            "audio_sampling_rate",
            "audio_hop_length",
            "num_layers",
            "activation_fn",
            "qk_norm",
            "norm_elementwise_affine",
            "norm_eps",
            "caption_channels",
            "attention_bias",
            "attention_out_bias",
            "rope_theta",
            "rope_double_precision",
            "causal_offset",
            "timestep_scale_multiplier",
            "cross_attn_timestep_scale_multiplier",
            "rope_type",
        }
        _reject_unexpected_keys(raw, allowed=expected, label="transformer")
        return cls(
            in_channels=_as_int(raw.get("in_channels", 128), name="in_channels"),
            out_channels=(None if raw.get("out_channels", 128) is None else _as_int(raw.get("out_channels", 128), name="out_channels")),
            patch_size=_as_int(raw.get("patch_size", 1), name="patch_size"),
            patch_size_t=_as_int(raw.get("patch_size_t", 1), name="patch_size_t"),
            num_attention_heads=_as_int(raw.get("num_attention_heads", 32), name="num_attention_heads"),
            attention_head_dim=_as_int(raw.get("attention_head_dim", 128), name="attention_head_dim"),
            cross_attention_dim=_as_int(raw.get("cross_attention_dim", 4096), name="cross_attention_dim"),
            vae_scale_factors=_as_tuple(raw.get("vae_scale_factors", (8, 32, 32)), name="vae_scale_factors", item_type=int),
            pos_embed_max_pos=_as_int(raw.get("pos_embed_max_pos", 20), name="pos_embed_max_pos"),
            base_height=_as_int(raw.get("base_height", 2048), name="base_height"),
            base_width=_as_int(raw.get("base_width", 2048), name="base_width"),
            audio_in_channels=_as_int(raw.get("audio_in_channels", 128), name="audio_in_channels"),
            audio_out_channels=(
                None
                if raw.get("audio_out_channels", 128) is None
                else _as_int(raw.get("audio_out_channels", 128), name="audio_out_channels")
            ),
            audio_patch_size=_as_int(raw.get("audio_patch_size", 1), name="audio_patch_size"),
            audio_patch_size_t=_as_int(raw.get("audio_patch_size_t", 1), name="audio_patch_size_t"),
            audio_num_attention_heads=_as_int(
                raw.get("audio_num_attention_heads", 32),
                name="audio_num_attention_heads",
            ),
            audio_attention_head_dim=_as_int(
                raw.get("audio_attention_head_dim", 64),
                name="audio_attention_head_dim",
            ),
            audio_cross_attention_dim=_as_int(
                raw.get("audio_cross_attention_dim", 2048),
                name="audio_cross_attention_dim",
            ),
            audio_scale_factor=_as_int(raw.get("audio_scale_factor", 4), name="audio_scale_factor"),
            audio_pos_embed_max_pos=_as_int(raw.get("audio_pos_embed_max_pos", 20), name="audio_pos_embed_max_pos"),
            audio_sampling_rate=_as_int(raw.get("audio_sampling_rate", 16000), name="audio_sampling_rate"),
            audio_hop_length=_as_int(raw.get("audio_hop_length", 160), name="audio_hop_length"),
            num_layers=_as_int(raw.get("num_layers", 48), name="num_layers"),
            activation_fn=_as_str(raw.get("activation_fn", "gelu-approximate"), name="activation_fn"),
            qk_norm=_as_str(raw.get("qk_norm", "rms_norm_across_heads"), name="qk_norm"),
            norm_elementwise_affine=_as_bool(raw.get("norm_elementwise_affine", False), name="norm_elementwise_affine"),
            norm_eps=_as_float(raw.get("norm_eps", 1e-6), name="norm_eps"),
            caption_channels=_as_int(raw.get("caption_channels", 3840), name="caption_channels"),
            attention_bias=_as_bool(raw.get("attention_bias", True), name="attention_bias"),
            attention_out_bias=_as_bool(raw.get("attention_out_bias", True), name="attention_out_bias"),
            rope_theta=_as_float(raw.get("rope_theta", 10000.0), name="rope_theta"),
            rope_double_precision=_as_bool(raw.get("rope_double_precision", True), name="rope_double_precision"),
            causal_offset=_as_int(raw.get("causal_offset", 1), name="causal_offset"),
            timestep_scale_multiplier=_as_int(
                raw.get("timestep_scale_multiplier", 1000),
                name="timestep_scale_multiplier",
            ),
            cross_attn_timestep_scale_multiplier=_as_int(
                raw.get("cross_attn_timestep_scale_multiplier", 1000),
                name="cross_attn_timestep_scale_multiplier",
            ),
            rope_type=_as_str(raw.get("rope_type", "interleaved"), name="rope_type"),
        )

    def load_strict_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise RuntimeError(
                f"LTX2 transformer strict load expects a mapping state_dict, got {type(state_dict).__name__}."
            )
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "LTX2 transformer strict load failed: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        audio_timestep: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float = 24.0,
        audio_num_frames: int | None = None,
        video_coords: torch.Tensor | None = None,
        audio_coords: torch.Tensor | None = None,
        attention_kwargs: Mapping[str, Any] | None = None,
        return_dict: bool = True,
    ) -> _AudioVisualModelOutput | tuple[torch.Tensor, torch.Tensor]:
        if attention_kwargs:
            unused = sorted(attention_kwargs)
            raise RuntimeError(f"LTX2 native transformer does not support attention_kwargs, got {unused!r}.")

        audio_timestep = timestep if audio_timestep is None else audio_timestep
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
            audio_encoder_attention_mask = (1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)) * -10000.0
            audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)
        if video_coords is None:
            video_coords = self.rope.prepare_video_coords(
                batch_size,
                num_frames,
                height,
                width,
                hidden_states.device,
                fps=fps,
            )
        if audio_coords is None:
            audio_coords = self.audio_rope.prepare_audio_coords(batch_size, audio_num_frames, audio_hidden_states.device)

        video_rotary_emb = self.rope(video_coords, device=hidden_states.device)
        audio_rotary_emb = self.audio_rope(audio_coords, device=audio_hidden_states.device)
        video_cross_attn_rotary_emb = self.cross_attn_rope(video_coords[:, 0:1, :], device=hidden_states.device)
        audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(
            audio_coords[:, 0:1, :],
            device=audio_hidden_states.device,
        )

        hidden_states = self.patchify_proj(hidden_states)
        audio_hidden_states = self.audio_patchify_proj(audio_hidden_states)

        timestep_cross_attn_gate_scale_factor = (
            self.config.cross_attn_timestep_scale_multiplier / self.config.timestep_scale_multiplier
        )
        temb, embedded_timestep = self.adaln_single(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        temb_audio, audio_embedded_timestep = self.audio_adaln_single(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
        audio_embedded_timestep = audio_embedded_timestep.view(batch_size, -1, audio_embedded_timestep.size(-1))

        video_cross_attn_scale_shift, _ = self.av_ca_video_scale_shift_adaln_single(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_a2v_gate, _ = self.av_ca_a2v_gate_adaln_single(
            timestep.flatten() * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(batch_size, -1, video_cross_attn_scale_shift.size(-1))
        video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(batch_size, -1, video_cross_attn_a2v_gate.size(-1))

        audio_cross_attn_scale_shift, _ = self.av_ca_audio_scale_shift_adaln_single(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_v2a_gate, _ = self.av_ca_v2a_gate_adaln_single(
            audio_timestep.flatten() * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(batch_size, -1, audio_cross_attn_scale_shift.size(-1))
        audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(batch_size, -1, audio_cross_attn_v2a_gate.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states).view(batch_size, -1, hidden_states.size(-1))
        audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states).view(
            batch_size,
            -1,
            audio_hidden_states.size(-1),
        )

        for block in self.transformer_blocks:
            hidden_states, audio_hidden_states = block(
                hidden_states,
                audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                temb=temb,
                temb_audio=temb_audio,
                temb_ca_scale_shift=video_cross_attn_scale_shift,
                temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                temb_ca_gate=video_cross_attn_a2v_gate,
                temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                video_rotary_emb=video_rotary_emb,
                audio_rotary_emb=audio_rotary_emb,
                ca_video_rotary_emb=video_cross_attn_rotary_emb,
                ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
            )

        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        audio_scale_shift_values = self.audio_scale_shift_table[None, None] + audio_embedded_timestep[:, :, None]
        audio_shift, audio_scale = audio_scale_shift_values[:, :, 0], audio_scale_shift_values[:, :, 1]
        audio_hidden_states = self.audio_norm_out(audio_hidden_states)
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        audio_output = self.audio_proj_out(audio_hidden_states)

        if not return_dict:
            return output, audio_output
        return _AudioVisualModelOutput(sample=output, audio_sample=audio_output)
