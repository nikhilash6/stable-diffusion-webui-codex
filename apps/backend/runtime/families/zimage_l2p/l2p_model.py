"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native Z-Image L2P pixel-space DiT runtime.
Implements the native checkpoint keyspace exactly without VAE/layer-name rewriting and loads SafeTensors/GGUF through strict views.
Keeps GGUF timestep activations floating, masks cross-sample batch padding for variable-length prompt fusion, returns native [-1, 1]
pixel tensors, and logs first forced-PyTorch SDPA attention calls.

Symbols (top-level; keep in sync; no ghosts):
- `_is_floating_dtype` (function): Returns true only for floating `torch.dtype` values.
- `_resolve_timestep_activation_dtype` (function): Resolves the floating dtype used for L2P timestep MLP activations.
- `_l2p_sdpa_policy_label` (function): Reports the PyTorch SDPA policy L2P forced attention will request.
- `_l2p_attention_mask_label` (function): Formats attention masks for one-shot backend logs.
- `_batch_padding_attention_mask` (function): Builds bool SDPA masks for cross-sample `pad_sequence` padding.
- `ZImageL2PConfig` (dataclass): Architecture constants for the first supported L2P 1K checkpoint.
- `RMSNorm` (class): RMS normalization layer matching checkpoint `*.weight` names.
- `TimestepEmbedder` (class): Sinusoidal timestep embedder feeding adaLN modulation.
- `FeedForward` (class): SwiGLU feed-forward block with native `w1/w2/w3` parameter names.
- `L2PAttention` (class): Separate Q/K/V self-attention block with native `to_q/to_k/to_v/to_out.0` parameter names.
- `ZImageL2PTransformerBlock` (class): Shared transformer block used by noise/context refiners and main layers.
- `RopeEmbedder` (class): 3-axis complex RoPE lookup used by L2P token streams.
- `MicroDiffusionModel` (class): Pixel-space local decoder consuming noisy RGB plus DiT feature maps.
- `ZImageL2PDiT` (class): Full L2P DiT model; forward returns negated flow update in pixel space.
- `load_zimage_l2p_from_state_dict` (function): Instantiate and strict-load the L2P DiT from a checkpoint mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from apps.backend.runtime.attention import attention_function_pre_shaped
from apps.backend.runtime.logging import emit_backend_message
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import AttentionBackend
from apps.backend.runtime.misc.autocast import autocast_disabled
from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.models.state_dict import safe_load_state_dict

_SEQ_MULTI_OF = 32
_ADALN_EMBED_DIM = 256
_L2P_PATCH_KEY = "16-1"
_L2P_ATTENTION_SHAPE_LOGGED: set[str] = set()
_L2P_REQUIRED_KEYS: tuple[str, ...] = (
    "all_x_embedder.16-1.weight",
    "local_decoder.out_conv.weight",
    "layers.0.adaLN_modulation.0.weight",
    "noise_refiner.0.adaLN_modulation.0.weight",
    "cap_embedder.1.weight",
)


def _l2p_sdpa_policy_label() -> str:
    try:
        attention_cfg = memory_management.manager.config.attention
    except Exception:
        return "auto"
    if getattr(attention_cfg, "backend", None) != AttentionBackend.PYTORCH:
        return "auto"
    enable_flash = bool(getattr(attention_cfg, "enable_flash", False))
    enable_mem_efficient = bool(getattr(attention_cfg, "enable_mem_efficient", False))
    if enable_flash and enable_mem_efficient:
        return "auto"
    if enable_flash:
        return "flash"
    if enable_mem_efficient:
        return "mem_efficient"
    return "math"


def _l2p_attention_mask_label(mask: torch.Tensor | None) -> str:
    if mask is None:
        return "none"
    return f"{tuple(mask.shape)} {mask.dtype} {mask.device}"


def _batch_padding_attention_mask(lengths: Sequence[int], *, device: torch.device) -> torch.Tensor | None:
    token_lengths = [int(length) for length in lengths]
    if not token_lengths:
        raise RuntimeError("L2P attention mask requires at least one sequence length.")
    if any(length <= 0 for length in token_lengths):
        raise RuntimeError(f"L2P attention mask requires positive sequence lengths; got {token_lengths}.")
    max_tokens = max(token_lengths)
    if all(length == max_tokens for length in token_lengths):
        return None
    positions = torch.arange(max_tokens, device=device)
    length_tensor = torch.tensor(token_lengths, device=device)
    key_keep_mask = positions.unsqueeze(0) < length_tensor.unsqueeze(1)
    return key_keep_mask.view(len(token_lengths), 1, 1, max_tokens)


def _is_floating_dtype(dtype: object) -> bool:
    return isinstance(dtype, torch.dtype) and bool(dtype.is_floating_point)


def _resolve_timestep_activation_dtype(
    *,
    requested_dtype: torch.dtype | None,
    weight: object,
) -> torch.dtype:
    if requested_dtype is not None:
        if not _is_floating_dtype(requested_dtype):
            raise RuntimeError(
                "L2P timestep embedder requires a floating caller dtype; "
                f"got {requested_dtype!r}."
            )
        return requested_dtype

    computation_dtype = getattr(weight, "computation_dtype", None)
    if _is_floating_dtype(computation_dtype):
        return computation_dtype

    weight_dtype = getattr(weight, "dtype", None)
    if _is_floating_dtype(weight_dtype):
        return weight_dtype

    raise RuntimeError(
        "L2P timestep embedder could not resolve a floating activation dtype "
        f"(weight_dtype={weight_dtype!r}, computation_dtype={computation_dtype!r})."
    )


@dataclass(frozen=True, slots=True)
class ZImageL2PConfig:
    in_channels: int = 3
    hidden_dim: int = 3840
    context_dim: int = 2560
    patch_size: int = 16
    frame_patch_size: int = 1
    num_layers: int = 30
    num_refiner_layers: int = 2
    num_heads: int = 30
    norm_eps: float = 1e-5
    rope_theta: float = 256.0
    t_scale: float = 1000.0
    axes_dims: tuple[int, int, int] = (32, 48, 48)
    axes_lens: tuple[int, int, int] = (1024, 512, 512)

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def patch_embed_dim(self) -> int:
        return self.frame_patch_size * self.patch_size * self.patch_size * self.in_channels

    @property
    def mlp_hidden_dim(self) -> int:
        return int(self.hidden_dim / 3 * 8)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(f"L2P RMSNorm expects floating-point input; got {x.dtype}.")
        dtype = x.dtype
        with autocast_disabled(x.device.type):
            x_float = x.float()
            normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (normed * self.weight.float()).to(dtype=dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size: int, *, mid_size: int = 1024, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, int(mid_size), bias=True),
            nn.SiLU(),
            nn.Linear(int(mid_size), int(out_size), bias=True),
        )

    @staticmethod
    def timestep_embedding(timestep: torch.Tensor, dim: int, *, max_period: int = 10000) -> torch.Tensor:
        with autocast_disabled(timestep.device.type):
            half = dim // 2
            freqs = torch.exp(
                -math.log(float(max_period))
                * torch.arange(start=0, end=half, dtype=torch.float32, device=timestep.device)
                / half
            )
            args = timestep[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        activation_dtype = _resolve_timestep_activation_dtype(requested_dtype=dtype, weight=self.mlp[0].weight)
        freq = self.timestep_embedding(timestep, self.frequency_embedding_size)
        return self.mlp(freq.to(dtype=activation_dtype))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(int(dim), int(hidden_dim), bias=False)
        self.w2 = nn.Linear(int(hidden_dim), int(dim), bias=False)
        self.w3 = nn.Linear(int(dim), int(hidden_dim), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class L2PAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        head_dim: int,
        norm_eps: float,
        bias_q: bool = False,
        bias_kv: bool = False,
        bias_out: bool = False,
    ) -> None:
        super().__init__()
        inner_dim = int(num_heads) * int(head_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.to_q = nn.Linear(int(dim), inner_dim, bias=bool(bias_q))
        self.to_k = nn.Linear(int(dim), inner_dim, bias=bool(bias_kv))
        self.to_v = nn.Linear(int(dim), inner_dim, bias=bool(bias_kv))
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, int(dim), bias=bool(bias_out))])
        self.norm_q = RMSNorm(int(head_dim), eps=norm_eps)
        self.norm_k = RMSNorm(int(head_dim), eps=norm_eps)

    @staticmethod
    def _apply_rotary(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        with autocast_disabled(x_in.device.type):
            x_complex = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs = freqs_cis.unsqueeze(2)
            rotated = torch.view_as_real(x_complex * freqs).flatten(3)
            return rotated.to(dtype=x_in.dtype)

    def _log_attention_shape_once(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
    ) -> None:
        mask_label = _l2p_attention_mask_label(attention_mask)
        mask_key = "none" if attention_mask is None else "masked"
        if mask_key in _L2P_ATTENTION_SHAPE_LOGGED:
            return
        _L2P_ATTENTION_SHAPE_LOGGED.add(mask_key)
        emit_backend_message(
            "[zimage_l2p] first attention call",
            logger=__name__,
            backend=AttentionBackend.PYTORCH.value,
            sdpa_policy=_l2p_sdpa_policy_label(),
            mask=mask_label,
            is_causal=False,
            q_shape=tuple(q.shape),
            k_shape=tuple(k.shape),
            v_shape=tuple(v.shape),
            dtype=str(q.dtype),
            device=str(q.device),
            heads=self.num_heads,
            head_dim=self.head_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        freqs_cis: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        query = self.to_q(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)
        if freqs_cis is not None:
            query = self._apply_rotary(query, freqs_cis)
            key = self._apply_rotary(key, freqs_cis)

        query_shaped = query.transpose(1, 2)
        key_shaped = key.transpose(1, 2)
        value_shaped = value.transpose(1, 2)
        self._log_attention_shape_once(query_shaped, key_shaped, value_shaped, attention_mask=attention_mask)
        out = attention_function_pre_shaped(
            query_shaped,
            key_shaped,
            value_shaped,
            mask=attention_mask,
            is_causal=False,
            backend=AttentionBackend.PYTORCH,
        )
        out = out.transpose(1, 2).reshape(batch, tokens, self.num_heads * self.head_dim)
        output = self.to_out[0](out)
        if len(self.to_out) > 1:
            output = self.to_out[1](output)
        return output


class ZImageL2PTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        norm_eps: float,
        modulation: bool,
    ) -> None:
        super().__init__()
        head_dim = int(dim) // int(num_heads)
        self.modulation = bool(modulation)
        self.attention = L2PAttention(
            dim=int(dim),
            num_heads=int(num_heads),
            head_dim=head_dim,
            norm_eps=float(norm_eps),
        )
        self.feed_forward = FeedForward(dim=int(dim), hidden_dim=int(int(dim) / 3 * 8))
        self.attention_norm1 = RMSNorm(int(dim), eps=float(norm_eps))
        self.ffn_norm1 = RMSNorm(int(dim), eps=float(norm_eps))
        self.attention_norm2 = RMSNorm(int(dim), eps=float(norm_eps))
        self.ffn_norm2 = RMSNorm(int(dim), eps=float(norm_eps))
        if self.modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(int(dim), _ADALN_EMBED_DIM), 4 * int(dim), bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            if adaln_input is None:
                raise RuntimeError("L2P modulated transformer block requires adaln_input.")
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            scale_msa = 1.0 + scale_msa
            scale_mlp = 1.0 + scale_mlp
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
            )
            x = x + gate_msa.tanh() * self.attention_norm2(attn_out)
            x = x + gate_mlp.tanh() * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
            return x

        attn_out = self.attention(self.attention_norm1(x), freqs_cis=freqs_cis, attention_mask=attention_mask)
        x = x + self.attention_norm2(attn_out)
        x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class RopeEmbedder:
    def __init__(self, *, theta: float, axes_dims: Sequence[int], axes_lens: Sequence[int]) -> None:
        if len(axes_dims) != len(axes_lens):
            raise ValueError("axes_dims and axes_lens must have matching lengths.")
        self.theta = float(theta)
        self.axes_dims = tuple(int(value) for value in axes_dims)
        self.axes_lens = tuple(int(value) for value in axes_lens)
        self.freqs_cis: list[torch.Tensor] | None = None

    @staticmethod
    def _precompute_freqs_cis(dimensions: Sequence[int], ends: Sequence[int], *, theta: float) -> list[torch.Tensor]:
        result: list[torch.Tensor] = []
        for dim, end in zip(dimensions, ends):
            freqs = 1.0 / (theta ** (torch.arange(0, int(dim), 2, dtype=torch.float64, device="cpu") / int(dim)))
            timestep = torch.arange(int(end), device="cpu", dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            result.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return result

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2:
            raise RuntimeError(f"L2P RoPE ids must be [N, axes], got {tuple(ids.shape)}.")
        if int(ids.shape[-1]) != len(self.axes_dims):
            raise RuntimeError(
                f"L2P RoPE expected {len(self.axes_dims)} axes, got shape {tuple(ids.shape)}."
            )
        device = ids.device
        if self.freqs_cis is None or any(item.device != device for item in self.freqs_cis):
            self.freqs_cis = [item.to(device) for item in self._precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)]
        chunks = []
        for axis, axis_len in enumerate(self.axes_lens):
            index = ids[:, axis].long().clamp(0, int(axis_len) - 1)
            chunks.append(self.freqs_cis[axis][index])
        return torch.cat(chunks, dim=-1)


class MicroDiffusionModel(nn.Module):
    def __init__(self, *, in_channels: int, si_t_hidden_size: int) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.SiLU())
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.SiLU())
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.SiLU())
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.SiLU())
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.bottleneck = nn.Sequential(nn.Conv2d(512 + int(si_t_hidden_size), 512, kernel_size=1), nn.SiLU())
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.dec4 = nn.Sequential(nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1), nn.SiLU())
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.dec3 = nn.Sequential(nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1), nn.SiLU())
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.dec2 = nn.Sequential(nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1), nn.SiLU())
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.dec1 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.SiLU())
        self.out_conv = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        pooled1 = self.pool1(enc1)
        enc2 = self.enc2(pooled1)
        pooled2 = self.pool2(enc2)
        enc3 = self.enc3(pooled2)
        pooled3 = self.pool3(enc3)
        enc4 = self.enc4(pooled3)
        pooled4 = self.pool4(enc4)
        if conditioning.shape[-2:] != pooled4.shape[-2:]:
            conditioning = F.interpolate(conditioning, size=pooled4.shape[-2:], mode="nearest")
        bottleneck = self.bottleneck(torch.cat([pooled4, conditioning], dim=1))
        dec4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))
        dec3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        return self.out_conv(dec1)


class ZImageL2PDiT(nn.Module):
    _supports_gradient_checkpointing = False
    _no_split_modules = ["ZImageL2PTransformerBlock"]

    def __init__(self, config: ZImageL2PConfig | None = None) -> None:
        super().__init__()
        self.config = config or ZImageL2PConfig()
        cfg = self.config
        if cfg.head_dim != sum(cfg.axes_dims):
            raise RuntimeError(f"L2P axes_dims must sum to head_dim={cfg.head_dim}; got {cfg.axes_dims}.")

        self.codex_config = SimpleNamespace(in_channels=cfg.in_channels, context_dim=cfg.context_dim, adm_in_channels=None)
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.in_channels
        self.all_patch_size = (cfg.patch_size,)
        self.all_f_patch_size = (cfg.frame_patch_size,)
        self.dim = cfg.hidden_dim
        self.n_heads = cfg.num_heads
        self.rope_theta = cfg.rope_theta
        self.t_scale = cfg.t_scale

        self.all_x_embedder = nn.ModuleDict({
            _L2P_PATCH_KEY: nn.Linear(cfg.patch_embed_dim, cfg.hidden_dim, bias=True),
        })
        self.local_decoder = MicroDiffusionModel(in_channels=cfg.in_channels, si_t_hidden_size=cfg.hidden_dim)
        self.noise_refiner = nn.ModuleList([
            ZImageL2PTransformerBlock(
                dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                norm_eps=cfg.norm_eps,
                modulation=True,
            )
            for _ in range(cfg.num_refiner_layers)
        ])
        self.context_refiner = nn.ModuleList([
            ZImageL2PTransformerBlock(
                dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                norm_eps=cfg.norm_eps,
                modulation=False,
            )
            for _ in range(cfg.num_refiner_layers)
        ])
        self.t_embedder = TimestepEmbedder(min(cfg.hidden_dim, _ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cfg.context_dim, eps=cfg.norm_eps),
            nn.Linear(cfg.context_dim, cfg.hidden_dim, bias=True),
        )
        self.x_pad_token = nn.Parameter(torch.empty((1, cfg.hidden_dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, cfg.hidden_dim)))
        self.layers = nn.ModuleList([
            ZImageL2PTransformerBlock(
                dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                norm_eps=cfg.norm_eps,
                modulation=True,
            )
            for _ in range(cfg.num_layers)
        ])
        self.rope_embedder = RopeEmbedder(theta=cfg.rope_theta, axes_dims=cfg.axes_dims, axes_lens=cfg.axes_lens)

    @staticmethod
    def create_coordinate_grid(
        size: tuple[int, int, int],
        *,
        start: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        axes = [
            torch.arange(origin, origin + span, dtype=torch.int32, device=device)
            for origin, span in zip(start, size)
        ]
        grids = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def _patchify_and_embed(
        self,
        all_images: Sequence[torch.Tensor],
        all_caption_features: Sequence[torch.Tensor],
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[tuple[int, int, int]],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        cfg = self.config
        device = all_images[0].device
        image_features: list[torch.Tensor] = []
        caption_features: list[torch.Tensor] = []
        image_sizes: list[tuple[int, int, int]] = []
        image_pos_ids: list[torch.Tensor] = []
        caption_pos_ids: list[torch.Tensor] = []
        image_pad_masks: list[torch.Tensor] = []
        caption_pad_masks: list[torch.Tensor] = []

        for image, caption_feature in zip(all_images, all_caption_features):
            if caption_feature.ndim != 2 or int(caption_feature.shape[-1]) != cfg.context_dim:
                raise RuntimeError(
                    "L2P prompt embedding must be [tokens, 2560]; "
                    f"got shape={tuple(caption_feature.shape)}."
                )
            caption_len = int(caption_feature.shape[0])
            if caption_len <= 0:
                raise RuntimeError("L2P prompt embedding list contains an empty prompt tensor.")
            caption_padding = (-caption_len) % _SEQ_MULTI_OF
            caption_pos = self.create_coordinate_grid(
                (caption_len + caption_padding, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            caption_pos_ids.append(caption_pos)
            caption_pad_masks.append(torch.cat([
                torch.zeros((caption_len,), dtype=torch.bool, device=device),
                torch.ones((caption_padding,), dtype=torch.bool, device=device),
            ], dim=0))
            if caption_padding:
                caption_feature = torch.cat([caption_feature, caption_feature[-1:].repeat(caption_padding, 1)], dim=0)
            caption_features.append(caption_feature)

            if image.ndim != 4:
                raise RuntimeError(f"L2P image item must be [C,F,H,W]; got {tuple(image.shape)}.")
            channels, frames, height, width = (int(value) for value in image.shape)
            if channels != cfg.in_channels or frames != 1:
                raise RuntimeError(f"L2P image item must be [3,1,H,W]; got {tuple(image.shape)}.")
            if height % cfg.patch_size != 0 or width % cfg.patch_size != 0:
                raise RuntimeError(
                    f"L2P image dimensions must be divisible by {cfg.patch_size}; got {width}x{height}."
                )
            image_sizes.append((frames, height, width))
            frame_tokens = frames // cfg.frame_patch_size
            height_tokens = height // cfg.patch_size
            width_tokens = width // cfg.patch_size
            patches = image.view(
                channels,
                frame_tokens,
                cfg.frame_patch_size,
                height_tokens,
                cfg.patch_size,
                width_tokens,
                cfg.patch_size,
            )
            patches = patches.permute(1, 3, 5, 2, 4, 6, 0).reshape(
                frame_tokens * height_tokens * width_tokens,
                cfg.patch_embed_dim,
            )
            image_len = int(patches.shape[0])
            image_padding = (-image_len) % _SEQ_MULTI_OF
            image_pos = self.create_coordinate_grid(
                (frame_tokens, height_tokens, width_tokens),
                start=(caption_len + caption_padding + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            if image_padding:
                image_pos = torch.cat(
                    [image_pos, torch.zeros((image_padding, 3), dtype=torch.int32, device=device)],
                    dim=0,
                )
                patches = torch.cat([patches, patches[-1:].repeat(image_padding, 1)], dim=0)
            image_pos_ids.append(image_pos)
            image_pad_masks.append(torch.cat([
                torch.zeros((image_len,), dtype=torch.bool, device=device),
                torch.ones((image_padding,), dtype=torch.bool, device=device),
            ], dim=0))
            image_features.append(patches)

        return (
            image_features,
            caption_features,
            image_sizes,
            image_pos_ids,
            caption_pos_ids,
            image_pad_masks,
            caption_pad_masks,
        )

    @staticmethod
    def _split_padded_sequence(values: torch.Tensor, lengths: Sequence[int]) -> list[torch.Tensor]:
        return list(values.split([int(length) for length in lengths], dim=0))

    def _forward_list(
        self,
        images: Sequence[torch.Tensor],
        sigma: torch.Tensor,
        prompt_embeds: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(images) != len(prompt_embeds):
            raise RuntimeError(
                f"L2P batch mismatch: images={len(images)} prompt_embeds={len(prompt_embeds)}."
            )
        batch_size = len(images)
        if batch_size <= 0:
            raise RuntimeError("L2P forward requires at least one image.")
        device = images[0].device
        sigma = sigma.to(device=device, dtype=torch.float32).reshape(-1)
        if int(sigma.shape[0]) != batch_size:
            raise RuntimeError(f"L2P sigma batch mismatch: sigma={tuple(sigma.shape)} batch={batch_size}.")
        adaln_input = self.t_embedder((1.0 - sigma) * self.t_scale, dtype=images[0].dtype)

        (
            image_items,
            caption_items,
            image_sizes,
            image_pos_ids,
            caption_pos_ids,
            image_pad_masks,
            caption_pad_masks,
        ) = self._patchify_and_embed(images, prompt_embeds)

        image_lengths = [int(item.shape[0]) for item in image_items]
        if any(length % _SEQ_MULTI_OF != 0 for length in image_lengths):
            raise RuntimeError(f"L2P image sequence lengths must be multiples of {_SEQ_MULTI_OF}; got {image_lengths}.")
        image_token_len = image_lengths[0]
        if any(length != image_token_len for length in image_lengths):
            raise RuntimeError("L2P first tranche supports uniform image token lengths only.")
        image_embed = torch.cat(image_items, dim=0)
        image_embed = self.all_x_embedder[_L2P_PATCH_KEY](image_embed)
        image_pad_mask = torch.cat(image_pad_masks, dim=0)
        image_embed[image_pad_mask] = self.x_pad_token.to(dtype=image_embed.dtype, device=image_embed.device)
        image_embed_items = self._split_padded_sequence(image_embed, image_lengths)
        image_freqs_items = self._split_padded_sequence(self.rope_embedder(torch.cat(image_pos_ids, dim=0)), image_lengths)
        image_embed_batch = pad_sequence(image_embed_items, batch_first=True, padding_value=0.0)
        image_freqs_batch = pad_sequence(image_freqs_items, batch_first=True, padding_value=0.0)

        for layer in self.noise_refiner:
            image_embed_batch = layer(image_embed_batch, freqs_cis=image_freqs_batch, adaln_input=adaln_input)

        caption_lengths = [int(item.shape[0]) for item in caption_items]
        if any(length % _SEQ_MULTI_OF != 0 for length in caption_lengths):
            raise RuntimeError(f"L2P caption sequence lengths must be multiples of {_SEQ_MULTI_OF}; got {caption_lengths}.")
        captions = torch.cat(caption_items, dim=0)
        captions = self.cap_embedder(captions)
        caption_pad_mask = torch.cat(caption_pad_masks, dim=0)
        captions[caption_pad_mask] = self.cap_pad_token.to(dtype=captions.dtype, device=captions.device)
        caption_items = self._split_padded_sequence(captions, caption_lengths)
        caption_freqs_items = self._split_padded_sequence(self.rope_embedder(torch.cat(caption_pos_ids, dim=0)), caption_lengths)
        caption_batch = pad_sequence(caption_items, batch_first=True, padding_value=0.0)
        caption_freqs_batch = pad_sequence(caption_freqs_items, batch_first=True, padding_value=0.0)
        caption_attention_mask = _batch_padding_attention_mask(caption_lengths, device=caption_batch.device)

        for layer in self.context_refiner:
            caption_batch = layer(caption_batch, freqs_cis=caption_freqs_batch, attention_mask=caption_attention_mask)

        unified_items: list[torch.Tensor] = []
        unified_freqs_items: list[torch.Tensor] = []
        for index in range(batch_size):
            image_len = image_lengths[index]
            caption_len = caption_lengths[index]
            unified_items.append(torch.cat([image_embed_batch[index, :image_len], caption_batch[index, :caption_len]], dim=0))
            unified_freqs_items.append(torch.cat([image_freqs_batch[index, :image_len], caption_freqs_batch[index, :caption_len]], dim=0))
        unified_lengths = [int(item.shape[0]) for item in unified_items]
        unified = pad_sequence(unified_items, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(unified_freqs_items, batch_first=True, padding_value=0.0)
        unified_attention_mask = _batch_padding_attention_mask(unified_lengths, device=unified.device)

        for layer in self.layers:
            unified = layer(
                unified,
                freqs_cis=unified_freqs,
                adaln_input=adaln_input,
                attention_mask=unified_attention_mask,
            )

        image_features = unified[:, :image_token_len, :]
        _, height, width = image_sizes[0]
        feature_height = height // self.config.patch_size
        feature_width = width // self.config.patch_size
        feature_map = image_features.view(batch_size, feature_height, feature_width, self.dim).permute(0, 3, 1, 2)
        noisy_images = torch.stack(list(images), dim=0)
        if noisy_images.dim() == 5:
            noisy_images = noisy_images.squeeze(2)
        decoded = self.local_decoder(noisy_images, feature_map)
        return -decoded

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        *,
        context: torch.Tensor | Sequence[torch.Tensor] | None = None,
        prompt_embeds: Sequence[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if prompt_embeds is None:
            if context is None:
                raise RuntimeError("L2P forward requires prompt_embeds or context.")
            if isinstance(context, torch.Tensor):
                if context.ndim != 3:
                    raise RuntimeError(f"L2P tensor context must be [B,S,C]; got {tuple(context.shape)}.")
                prompt_embeds = [context[index] for index in range(int(context.shape[0]))]
            else:
                prompt_embeds = list(context)
        if x.ndim != 4:
            raise RuntimeError(f"L2P input must be [B,3,H,W]; got {tuple(x.shape)}.")
        images = [x[index].unsqueeze(1) for index in range(int(x.shape[0]))]
        return self._forward_list(images, sigma, list(prompt_embeds))


def _validate_l2p_state_dict(state_dict: Mapping[str, Any]) -> None:
    non_string = [repr(key) for key in state_dict.keys() if not isinstance(key, str)]
    if non_string:
        raise RuntimeError(f"Z-Image L2P state_dict keys must be strings; sample={non_string[:10]}.")
    missing = [key for key in _L2P_REQUIRED_KEYS if key not in state_dict]
    if missing:
        raise RuntimeError(f"Z-Image L2P checkpoint missing required keys: {missing}.")
    if "final_layer.linear.weight" in state_dict:
        raise RuntimeError("Z-Image L2P checkpoint must not contain latent Z-Image final_layer.linear.weight.")


def load_zimage_l2p_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    config: ZImageL2PConfig | None = None,
    weight_format: str | None = None,
    storage_dtype: torch.dtype | str | None = None,
    computation_dtype: torch.dtype | None = None,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    initial_device: torch.device | str | None = None,
) -> ZImageL2PDiT:
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"state_dict must be a mapping; got {type(state_dict).__name__}.")
    _validate_l2p_state_dict(state_dict)
    normalized_weight_format = str(weight_format or "").strip().lower() or None
    if normalized_weight_format not in {None, "gguf"}:
        raise RuntimeError(f"Unsupported Z-Image L2P weight_format={weight_format!r}; expected None or 'gguf'.")

    construct_device = torch.device(device) if device is not None else None
    construct_dtype = dtype
    if normalized_weight_format == "gguf":
        with using_codex_operations(
            device=construct_device,
            dtype=construct_dtype,
            manual_cast_enabled=False,
            weight_format="gguf",
        ):
            model = ZImageL2PDiT(config=config).eval()
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:  # noqa: BLE001 - preserve strict GGUF context
            raise RuntimeError(
                "Z-Image L2P GGUF transformer strict load failed; "
                "update the converter profile/keyspace rather than rewriting checkpoint keys."
            ) from exc
    else:
        with using_codex_operations(device=construct_device, dtype=construct_dtype, manual_cast_enabled=True):
            model = ZImageL2PDiT(config=config).eval()
            if construct_device is not None or construct_dtype is not None:
                to_kwargs: dict[str, Any] = {}
                if construct_device is not None:
                    to_kwargs["device"] = construct_device
                if construct_dtype is not None:
                    to_kwargs["dtype"] = construct_dtype
                model = model.to(**to_kwargs)
        missing, unexpected = safe_load_state_dict(model, state_dict, log_name="zimage_l2p.transformer")
        if missing or unexpected:
            raise RuntimeError(
                "Z-Image L2P transformer strict load failed: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
            )

    first_parameter = next(model.parameters())
    resolved_load_device = torch.device(load_device) if load_device is not None else (
        construct_device if construct_device is not None else first_parameter.device
    )
    resolved_initial_device = torch.device(initial_device) if initial_device is not None else first_parameter.device
    resolved_offload_device = torch.device(offload_device) if offload_device is not None else resolved_initial_device
    model.storage_dtype = storage_dtype if storage_dtype is not None else (construct_dtype or first_parameter.dtype)
    model.computation_dtype = computation_dtype if computation_dtype is not None else (construct_dtype or torch.float32)
    model.load_device = resolved_load_device
    model.initial_device = resolved_initial_device
    model.offload_device = resolved_offload_device
    return model


__all__ = [
    "ZImageL2PConfig",
    "ZImageL2PDiT",
    "load_zimage_l2p_from_state_dict",
]
