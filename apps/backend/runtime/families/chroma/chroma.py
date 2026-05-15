"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Chroma transformer runtime built on the Codex Flux components.
Implements the Chroma core transformer blocks and model module (Flux-like double/single stream blocks + modulation), assembling the
transformer used by the Chroma runtime/engine.

Symbols (top-level; keep in sync; no ghosts):
- `Approximator` (class): Auxiliary nn.Module used in modulation/conditioning paths (contains nested helpers for forward computation).
- `ModulationOut` (dataclass): Modulation output container for per-block modulation parameters.
- `DoubleStreamBlock` (class): Dual-stream transformer block (separate streams + cross interactions) used in early Chroma layers.
- `SingleStreamBlock` (class): Single-stream transformer block used in later Chroma layers.
- `LastLayer` (class): Final projection/output layer for Chroma transformer outputs.
- `ChromaTransformer2DModel` (class): Main Chroma transformer module; builds blocks from config and performs forward pass with rotary embeddings
  and Flux-shared attention primitives.
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
from einops import rearrange, repeat

from apps.backend.runtime import utils
from apps.backend.runtime.attention import attention_function
from apps.backend.runtime.families.flux.components import QKNorm, RMSNorm, SelfAttention
from apps.backend.runtime.families.flux.config import FluxPositionalConfig
from apps.backend.runtime.families.flux.embed import EmbedND, MLPEmbedder
from apps.backend.runtime.families.flux.geometry import apply_rotary_embeddings, timestep_embedding
from .config import ChromaArchitectureConfig, ChromaGuidanceConfig

logger = get_backend_logger("backend.runtime.chroma")


class Approximator(nn.Module):
    """Guidance modulation approximator used by Chroma variants."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim, bias=True)
        self.blocks = nn.ModuleList(MLPEmbedder(hidden_dim, hidden_dim) for _ in range(layers))
        self.norms = nn.ModuleList(RMSNorm(hidden_dim) for _ in range(layers))
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for block, norm in zip(self.blocks, self.norms):
            x = x + block(norm(x))
        return self.output(x)


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class DoubleStreamBlock(nn.Module):
    """Chroma double-stream block using pre-baked modulation vectors."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, *, qkv_bias: bool) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        mod: Tuple[Tuple[ModulationOut, ModulationOut], Tuple[ModulationOut, ModulationOut]],
        rotary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = mod

        img_pre = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
        txt_pre = (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift

        img_qkv = self.img_attn.qkv(img_pre)
        txt_qkv = self.txt_attn.qkv(txt_pre)

        b, seq, _ = img_qkv.shape
        head_dim = img_qkv.shape[-1] // (3 * self.num_heads)

        img_q, img_k, img_v = img_qkv.view(b, seq, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        txt_q, txt_k, txt_v = txt_qkv.view(b, seq, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)

        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        img_q, img_k = apply_rotary_embeddings(img_q, img_k, rotary)
        txt_q, txt_k = apply_rotary_embeddings(txt_q, txt_k, rotary)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention_function(q, k, v, q.shape[1], skip_reshape=True)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]

        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        txt = utils.fp16_fix(txt)
        return img, txt


class SingleStreamBlock(nn.Module):
    """Chroma single-stream block using pre-baked modulation vectors."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.norm = QKNorm(hidden_size // num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_hidden = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden, hidden_size)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor, mod: ModulationOut, rotary: torch.Tensor) -> torch.Tensor:
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden], dim=-1)
        qkv = qkv.view(qkv.size(0), qkv.size(1), 3, self.num_heads, self.hidden_size // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)
        q, k = apply_rotary_embeddings(q, k, rotary)
        attn = attention_function(q, k, v, q.shape[1], skip_reshape=True)
        output = self.linear2(torch.cat((attn, self.act(mlp)), dim=2))
        x = x + mod.gate * output
        return utils.fp16_fix(x)


class LastLayer(nn.Module):
    """Chroma final layer with modulation shift/scale."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x: torch.Tensor, mod: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        shift, scale = mod
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.linear((1 + scale[:, None, :]) * self.norm(x) + shift[:, None, :])


class ChromaTransformer2DModel(nn.Module):
    """Codex-native Chroma transformer using shared Flux utilities."""

    def __init__(self, config: ChromaArchitectureConfig | None = None, **raw_config) -> None:
        super().__init__()
        if config is None:
            if not raw_config:
                raise ValueError("ChromaTransformer2DModel requires configuration parameters")
            axes_dim = tuple(raw_config.pop("axes_dim"))
            theta = raw_config.pop("theta")
            positional = FluxPositionalConfig(patch_size=2, axes_dim=axes_dim, theta=theta)
            guidance = ChromaGuidanceConfig(
                out_dim=raw_config.pop("guidance_out_dim"),
                hidden_dim=raw_config.pop("guidance_hidden_dim"),
                layers=raw_config.pop("guidance_n_layers"),
            )
            config = ChromaArchitectureConfig(
                positional=positional,
                guidance=guidance,
                double_blocks=raw_config.pop("depth"),
                single_blocks=raw_config.pop("depth_single_blocks"),
                **raw_config,
            )

        self.config = config
        self.in_channels = config.in_channels * config.patch_area
        self.out_channels = self.in_channels

        pe_dim = config.hidden_size // config.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=config.positional.theta, axes_dim=tuple(config.positional.axes_dim))
        self.img_in = nn.Linear(self.in_channels, config.hidden_size, bias=True)
        self.txt_in = nn.Linear(config.context_in_dim, config.hidden_size)
        self.guidance = Approximator(
            in_dim=64 + 64 + 32,
            out_dim=config.guidance.out_dim,
            hidden_dim=config.guidance.hidden_dim,
            layers=config.guidance.layers,
        )

        self.double_blocks = nn.ModuleList(
            DoubleStreamBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
            )
            for _ in range(config.double_blocks)
        )
        self.single_blocks = nn.ModuleList(
            SingleStreamBlock(config.hidden_size, config.num_heads, mlp_ratio=config.mlp_ratio)
            for _ in range(config.single_blocks)
        )
        self.final_layer = LastLayer(config.hidden_size, config.positional.patch_size, self.out_channels)

    @property
    def patch_size(self) -> int:
        return self.config.positional.patch_size

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("expected latent tensor (B, C, H, W)")
        batch, _, height, width = x.shape
        patch = self.patch_size
        pad_h = (-height) % patch
        pad_w = (-width) % patch
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        h_len = (height + pad_h) // patch
        w_len = (width + pad_w) // patch

        img_ids = self._build_spatial_ids(batch, h_len, w_len, x.device, x.dtype)
        txt_ids = torch.zeros((batch, context.shape[1], 3), device=x.device, dtype=x.dtype)

        inner = self._inner_forward(img, img_ids, context, txt_ids, timestep)
        out = rearrange(inner, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=patch, pw=patch)
        return out[:, :, :height, :width]

    def _inner_forward(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        img = self.img_in(img)
        txt = self.txt_in(txt)

        rotary = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))
        modulation_vectors = self._build_modulations(img, timesteps)
        mod_dict = self._distribute_modulations(modulation_vectors)

        for i, block in enumerate(self.double_blocks):
            img_mod = mod_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod = mod_dict[f"double_blocks.{i}.txt_mod.lin"]
            img, txt = block(img, txt, (img_mod, txt_mod), rotary)

        tokens = torch.cat((txt, img), dim=1)
        for i, block in enumerate(self.single_blocks):
            single_mod = mod_dict[f"single_blocks.{i}.modulation.lin"]
            tokens = block(tokens, single_mod, rotary)

        tokens = tokens[:, txt.shape[1]:]
        final_mod = mod_dict["final_layer.adaLN_modulation.1"]
        return self.final_layer(tokens, final_mod)

    def _build_spatial_ids(self, batch: int, h_len: int, w_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base = torch.zeros((h_len, w_len, 3), device=device, dtype=dtype)
        base[..., 1] = torch.linspace(0, h_len - 1, steps=h_len, device=device, dtype=dtype)[:, None]
        base[..., 2] = torch.linspace(0, w_len - 1, steps=w_len, device=device, dtype=dtype)[None, :]
        return repeat(base, "h w c -> b (h w) c", b=batch)

    def _build_modulations(self, img: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # Chroma distillation uses fixed zero guidance and index embeddings.
        device = img.device
        dtype = img.dtype
        num_double = len(self.double_blocks)
        num_single = len(self.single_blocks)
        modulation_count = num_double * 12 + num_single * 3 + 2

        timestep_embed = timestep_embedding(timesteps.detach().clone(), 16).to(device=device, dtype=dtype)
        zero_guidance = timestep_embedding(torch.zeros_like(timesteps), 16).to(device=device, dtype=dtype)
        modulation_index = timestep_embedding(torch.arange(modulation_count, device=device), 32).to(dtype=dtype)
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)

        conditioning = torch.cat((timestep_embed, zero_guidance), dim=1).unsqueeze(1)
        conditioning = conditioning.repeat(1, modulation_count, 1)
        approximator_input = torch.cat((conditioning, modulation_index), dim=-1)
        emit_backend_message(
            "Chroma modulation input built",
            logger=logger.name,
            level=logging.DEBUG,
            batch=img.shape[0],
            vectors=modulation_count,
            dim=approximator_input.shape[-1],
        )
        return self.guidance(approximator_input)

    def _distribute_modulations(self, tensor: torch.Tensor) -> Dict[str, object]:
        num_single = len(self.single_blocks)
        num_double = len(self.double_blocks)
        block_dict: Dict[str, object] = {}
        idx = 0
        for i in range(num_single):
            block_dict[f"single_blocks.{i}.modulation.lin"] = ModulationOut(
                shift=tensor[:, idx : idx + 1, :],
                scale=tensor[:, idx + 1 : idx + 2, :],
                gate=tensor[:, idx + 2 : idx + 3, :],
            )
            idx += 3
        for category in ("img_mod", "txt_mod"):
            for i in range(num_double):
                block_mods: List[ModulationOut] = []
                for _ in range(2):
                    block_mods.append(
                        ModulationOut(
                            shift=tensor[:, idx : idx + 1, :],
                            scale=tensor[:, idx + 1 : idx + 2, :],
                            gate=tensor[:, idx + 2 : idx + 3, :],
                        )
                    )
                    idx += 3
                block_dict[f"double_blocks.{i}.{category}.lin"] = tuple(block_mods)  # type: ignore[assignment]
        block_dict["final_layer.adaLN_modulation.1"] = [
            tensor[:, idx : idx + 1, :],
            tensor[:, idx + 1 : idx + 2, :],
        ]
        return block_dict
