"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WanVAE-style (3D conv) VAE used by Anima (`qwen_image_vae.safetensors`).
Implements the WAN 2.1 VAE architecture used by ComfyUI for image inference, with strict keymap normalization from `runtime/state_dict/keymap_wan21_vae.py`, explicit Wan21 latent stats exposure, strict loader diagnostics, and owner-device birth/load semantics for the external VAE asset.
This implementation is scoped to **images**: inputs are treated as `T=1` (no video caching or temporal chunking).

Symbols (top-level; keep in sync; no ghosts):
- `WanVaeConfig` (dataclass): Minimal config surface required by the shared VAE patcher (`apps/backend/patchers/vae.py`), including optional per-channel latent stats and explicit no-shift (`shift_factor=None`) semantics.
- `WanVAE` (class): WAN-style VAE module with `encode`/`decode` supporting 4D tensors (image mode; `T=1`).
- `detect_wan_vae_variant_from_header` (function): Detect WAN VAE variant (`2.1`/`2.2`) from safetensors header keys.
- `infer_wan_vae_config_from_safetensors_header` (function): Infer WAN VAE config values from header-only metadata.
- `load_wan_vae_from_safetensors` (function): Strict owner-device loader for WAN VAE weights (fail-loud missing/unexpected keys).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from apps.backend.runtime.attention import attention_function_single_head_spatial
from apps.backend.runtime.checkpoint.io import load_torch_file
from apps.backend.runtime.checkpoint.safetensors_header import read_safetensors_header
from apps.backend.runtime.families.wan22.wan_latent_norms import WAN21_LATENTS_MEAN, WAN21_LATENTS_STD
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.state_dict.keymap_wan21_vae import resolve_wan21_vae_keyspace

logger = get_backend_logger("backend.runtime.anima.wan_vae")
WAN_VAE_BASE_MARKER_KEY = "decoder.middle.0.residual.0.gamma"
WAN_VAE_22_MARKER_KEY = "decoder.upsamples.0.upsamples.0.residual.2.weight"
WanVaeVariant = Literal["2.1", "2.2"]


@dataclass(frozen=True, slots=True)
class WanVaeConfig:
    # Required by `apps/backend/patchers/vae.py`.
    down_block_types: Sequence[str]
    latent_channels: int
    scaling_factor: float = 1.0
    shift_factor: float | None = None
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None


class CausalConv3d(nn.Conv3d):
    """Causal Conv3d along time (zero padding on the left)."""

    def __init__(self, *args, **kwargs):  # noqa: D401
        padding = kwargs.get("padding", 0)
        if isinstance(padding, int):
            pad_t, pad_h, pad_w = padding, padding, padding
        else:
            pad_t, pad_h, pad_w = padding
        self._pad_t = int(pad_t) * 2
        # Spatial padding handled by conv; time padding is explicit.
        kwargs["padding"] = (0, int(pad_h), int(pad_w))
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"CausalConv3d expects 5D input (B,C,T,H,W), got shape={tuple(x.shape)}")
        if self._pad_t > 0:
            x = F.pad(x, (0, 0, 0, 0, int(self._pad_t), 0))
        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim: int, *, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1) if images else (1, 1, 1)
        shape = (int(dim), *broadcastable_dims) if channel_first else (int(dim),)
        self.channel_first = bool(channel_first)
        self.scale = float(dim) ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        out = F.normalize(x, dim=dim) * self.scale * self.gamma.to(dtype=x.dtype, device=x.device)
        if self.bias is not None:
            out = out + self.bias.to(dtype=x.dtype, device=x.device)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.residual = nn.Sequential(
            RMS_norm(self.in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(self.in_dim, self.out_dim, 3, padding=1),
            RMS_norm(self.out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            CausalConv3d(self.out_dim, self.out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(self.in_dim, self.out_dim, 1) if self.in_dim != self.out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Single-head self-attention over spatial tokens (per-frame)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = RMS_norm(self.dim, images=True)
        self.to_qkv = nn.Conv2d(self.dim, self.dim * 3, 1)
        self.proj = nn.Conv2d(self.dim, self.dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"AttentionBlock expects 5D input (B,C,T,H,W), got shape={tuple(x.shape)}")
        identity = x
        b, c, t, h, w = x.shape
        x2 = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x2 = self.norm(x2)
        q, k, v = self.to_qkv(x2).chunk(3, dim=1)
        attn = attention_function_single_head_spatial(q, k, v)
        out = self.proj(attn)
        out = out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return out + identity


class Resample(nn.Module):
    def __init__(self, dim: int, *, mode: str) -> None:
        super().__init__()
        self.dim = int(dim)
        self.mode = str(mode)
        if self.mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest"),
                nn.Conv2d(self.dim, self.dim // 2, 3, padding=1),
            )
        elif self.mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(self.dim, self.dim, 3, stride=(2, 2)),
            )
        elif self.mode in {"upsample3d", "downsample3d"}:
            # Image-mode only: temporal resample is not supported in v1 (T must be 1).
            # Keep the same spatial ops so the state dict matches.
            if self.mode == "upsample3d":
                self.resample = nn.Sequential(
                    nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest"),
                    nn.Conv2d(self.dim, self.dim // 2, 3, padding=1),
                )
            else:
                self.resample = nn.Sequential(
                    nn.ZeroPad2d((0, 1, 0, 1)),
                    nn.Conv2d(self.dim, self.dim, 3, stride=(2, 2)),
                )
            # Define time_conv parameters so weights load, but do not execute them in v1.
            if self.mode == "upsample3d":
                self.time_conv = CausalConv3d(self.dim, self.dim * 2, (3, 1, 1), padding=(1, 0, 0))
            else:
                self.time_conv = CausalConv3d(self.dim, self.dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        elif self.mode == "none":
            self.resample = nn.Identity()
        else:
            raise ValueError(f"Unsupported Resample mode: {self.mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Resample expects 5D input (B,C,T,H,W), got shape={tuple(x.shape)}")
        b, c, t, h, w = x.shape
        if self.mode.endswith("3d") and int(t) != 1:
            raise NotImplementedError("WanVAE image-mode resample only supports T=1 (video not yet ported).")
        x2 = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x2 = self.resample(x2)
        c2 = int(x2.shape[1])
        h2, w2 = int(x2.shape[2]), int(x2.shape[3])
        out = x2.reshape(b, t, c2, h2, w2).permute(0, 2, 1, 3, 4).contiguous()
        return out


class Encoder3d(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        z_dim: int,
        input_channels: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        attn_scales: Sequence[float],
        temperal_downsample: Sequence[bool],
        dropout: float,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.z_dim = int(z_dim)
        self.dim_mult = tuple(int(x) for x in dim_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.attn_scales = tuple(float(x) for x in attn_scales)
        self.temperal_downsample = tuple(bool(x) for x in temperal_downsample)

        dims = [self.dim * u for u in (1, *self.dim_mult)]
        scale = 1.0

        self.conv1 = CausalConv3d(int(input_channels), int(dims[0]), 3, padding=1)

        downsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            for _ in range(self.num_res_blocks):
                downsamples.append(ResidualBlock(int(in_dim), int(out_dim), dropout=dropout))
                if scale in self.attn_scales:
                    downsamples.append(AttentionBlock(int(out_dim)))
                in_dim = out_dim
            if i != len(self.dim_mult) - 1:
                mode = "downsample3d" if self.temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(int(out_dim), mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(int(out_dim), int(out_dim), dropout=dropout),
            AttentionBlock(int(out_dim)),
            ResidualBlock(int(out_dim), int(out_dim), dropout=dropout),
        )

        self.head = nn.Sequential(
            RMS_norm(int(out_dim), images=False),
            nn.SiLU(),
            CausalConv3d(int(out_dim), int(self.z_dim), 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.downsamples(x)
        x = self.middle(x)
        x = self.head(x)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        z_dim: int,
        output_channels: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        attn_scales: Sequence[float],
        temperal_upsample: Sequence[bool],
        dropout: float,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.z_dim = int(z_dim)
        self.dim_mult = tuple(int(x) for x in dim_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.attn_scales = tuple(float(x) for x in attn_scales)
        self.temperal_upsample = tuple(bool(x) for x in temperal_upsample)

        dims = [self.dim * u for u in (self.dim_mult[-1], *reversed(self.dim_mult))]
        scale = 1.0 / (2 ** (len(self.dim_mult) - 2))

        self.conv1 = CausalConv3d(int(self.z_dim), int(dims[0]), 3, padding=1)

        self.middle = nn.Sequential(
            ResidualBlock(int(dims[0]), int(dims[0]), dropout=dropout),
            AttentionBlock(int(dims[0])),
            ResidualBlock(int(dims[0]), int(dims[0]), dropout=dropout),
        )

        upsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            in_dim_i = int(in_dim)
            if i in (1, 2, 3):
                in_dim_i = in_dim_i // 2
            for _ in range(self.num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim_i, int(out_dim), dropout=dropout))
                if scale in self.attn_scales:
                    upsamples.append(AttentionBlock(int(out_dim)))
                in_dim_i = int(out_dim)
            if i != len(self.dim_mult) - 1:
                mode = "upsample3d" if self.temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(int(out_dim), mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMS_norm(int(out_dim), images=False),
            nn.SiLU(),
            CausalConv3d(int(out_dim), int(output_channels), 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.middle(x)
        x = self.upsamples(x)
        x = self.head(x)
        return x


class WanVAE(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        z_dim: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        attn_scales: Sequence[float],
        temperal_downsample: Sequence[bool],
        image_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.z_dim = int(z_dim)
        self.dim_mult = tuple(int(x) for x in dim_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.attn_scales = tuple(float(x) for x in attn_scales)
        self.temperal_downsample = tuple(bool(x) for x in temperal_downsample)
        self.temperal_upsample = tuple(reversed(self.temperal_downsample))
        self.image_channels = int(image_channels)
        expected_latent_channels = len(WAN21_LATENTS_MEAN)
        if self.z_dim != expected_latent_channels:
            raise RuntimeError(
                "Anima WanVAE requires z_dim=16 for Wan21 latent normalization stats; "
                f"got z_dim={self.z_dim}."
            )

        # Minimal config surface consumed by `apps/backend/patchers/vae.py`.
        self.config = WanVaeConfig(
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            latent_channels=int(self.z_dim),
            scaling_factor=1.0,
            shift_factor=None,
            latents_mean=WAN21_LATENTS_MEAN,
            latents_std=WAN21_LATENTS_STD,
        )

        self.encoder = Encoder3d(
            dim=self.dim,
            z_dim=self.z_dim * 2,
            input_channels=self.image_channels,
            dim_mult=self.dim_mult,
            num_res_blocks=self.num_res_blocks,
            attn_scales=self.attn_scales,
            temperal_downsample=self.temperal_downsample,
            dropout=float(dropout),
        )
        self.conv1 = CausalConv3d(self.z_dim * 2, self.z_dim * 2, 1)
        self.conv2 = CausalConv3d(self.z_dim, self.z_dim, 1)
        self.decoder = Decoder3d(
            dim=self.dim,
            z_dim=self.z_dim,
            output_channels=self.image_channels,
            dim_mult=self.dim_mult,
            num_res_blocks=self.num_res_blocks,
            attn_scales=self.attn_scales,
            temperal_upsample=self.temperal_upsample,
            dropout=float(dropout),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_t = False
        if x.ndim == 4:
            squeeze_t = True
            x = x.unsqueeze(2)
        if x.ndim != 5:
            raise ValueError(f"WanVAE.encode expects 4D/5D input, got shape={tuple(x.shape)}")
        if int(x.shape[2]) != 1:
            raise NotImplementedError("WanVAE encode only supports T=1 (video not yet ported).")
        out = self.encoder(x)
        mu, _log_var = self.conv1(out).chunk(2, dim=1)
        if squeeze_t:
            mu = mu.squeeze(2)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_t = False
        if z.ndim == 4:
            squeeze_t = True
            z = z.unsqueeze(2)
        if z.ndim != 5:
            raise ValueError(f"WanVAE.decode expects 4D/5D input, got shape={tuple(z.shape)}")
        if int(z.shape[2]) != 1:
            raise NotImplementedError("WanVAE decode only supports T=1 (video not yet ported).")
        x = self.conv2(z)
        out = self.decoder(x)
        if squeeze_t:
            out = out.squeeze(2)
        return out


@dataclass(frozen=True, slots=True)
class WanVaeHeaderConfig:
    variant: WanVaeVariant
    dim: int
    latent_channels: int
    image_channels: int


def detect_wan_vae_variant_from_header(header: Mapping[str, object]) -> WanVaeVariant:
    has_base_marker = WAN_VAE_BASE_MARKER_KEY in header
    has_22_marker = WAN_VAE_22_MARKER_KEY in header
    if has_22_marker and not has_base_marker:
        raise RuntimeError(
            "WAN VAE header is ambiguous: found WAN 2.2 marker "
            f"'{WAN_VAE_22_MARKER_KEY}' without required base marker '{WAN_VAE_BASE_MARKER_KEY}'."
        )
    if has_22_marker:
        return "2.2"
    if has_base_marker:
        return "2.1"
    raise RuntimeError(f"Not a WAN VAE weights file (missing {WAN_VAE_BASE_MARKER_KEY}).")


def infer_wan_vae_config_from_safetensors_header(header: Mapping[str, object]) -> WanVaeHeaderConfig:
    def _shape(key: str) -> tuple[int, ...] | None:
        meta = header.get(key)
        if isinstance(meta, dict):
            shape = meta.get("shape")
            if isinstance(shape, (list, tuple)) and all(isinstance(x, (int, float)) for x in shape):
                return tuple(int(x) for x in shape)
        return None

    variant = detect_wan_vae_variant_from_header(header)

    dim_shape = _shape("decoder.head.0.gamma")
    if dim_shape is None:
        raise RuntimeError("WAN VAE header missing decoder.head.0.gamma shape.")
    if len(dim_shape) != 4 or tuple(dim_shape[1:]) != (1, 1, 1):
        raise RuntimeError(
            "WAN VAE invalid decoder.head.0.gamma shape: "
            f"got {tuple(dim_shape)}, expected (dim, 1, 1, 1)."
        )
    dim = int(dim_shape[0])
    if dim <= 0:
        raise RuntimeError(f"WAN VAE invalid dim inferred from decoder.head.0.gamma: {dim}")

    conv1_shape = _shape("encoder.conv1.weight")
    if conv1_shape is None or len(conv1_shape) != 5:
        raise RuntimeError("WAN VAE header missing encoder.conv1.weight 5D shape.")
    if int(conv1_shape[0]) != dim:
        raise RuntimeError(
            "WAN VAE header dim mismatch: "
            f"decoder.head.0.gamma implies dim={dim}, "
            f"but encoder.conv1.weight has out_channels={int(conv1_shape[0])}."
        )
    image_channels = int(conv1_shape[1])
    if image_channels <= 0:
        raise RuntimeError(f"WAN VAE invalid image_channels inferred from encoder.conv1.weight: {image_channels}")

    z_shape = _shape("decoder.conv1.weight")
    if z_shape is None or len(z_shape) != 5:
        raise RuntimeError("WAN VAE header missing decoder.conv1.weight 5D shape.")
    latent_channels = int(z_shape[1])
    if latent_channels <= 0:
        raise RuntimeError(f"WAN VAE invalid latent_channels inferred from decoder.conv1.weight: {latent_channels}")

    return WanVaeHeaderConfig(
        variant=variant,
        dim=dim,
        latent_channels=latent_channels,
        image_channels=image_channels,
    )


def load_wan_vae_from_safetensors(
    vae_path: str,
    *,
    torch_dtype: torch.dtype,
    device: torch.device | str,
) -> WanVAE:
    raw = str(vae_path or "").strip()
    if not raw:
        raise ValueError("Anima VAE path is required.")
    p = Path(os.path.expanduser(raw))
    try:
        p = p.resolve()
    except Exception:
        p = p.absolute()
    if device is None:
        raise ValueError("Anima VAE loader requires an explicit owner device.")
    if not p.exists() or not p.is_file():
        raise RuntimeError(f"Anima VAE path not found: {p}")
    if p.suffix.lower() not in {".safetensor", ".safetensors"}:
        raise ValueError(f"Anima VAE must be a .safetensors file, got: {p}")

    header = read_safetensors_header(p)
    variant = detect_wan_vae_variant_from_header(header)
    if variant == "2.2":
        raise NotImplementedError(
            "WAN VAE 2.2 detected by safetensors header keys; "
            "Anima v1 ports WAN 2.1 image-mode only (latent_channels=16)."
        )
    cfg = infer_wan_vae_config_from_safetensors_header(header)
    if int(cfg.latent_channels) != 16:
        raise RuntimeError(f"WAN VAE latent_channels mismatch: got {cfg.latent_channels}, expected 16 for Anima.")

    sd = load_torch_file(str(p), device="cpu")
    if not isinstance(sd, Mapping):
        raise RuntimeError(f"WAN VAE checkpoint loader returned non-mapping state_dict: {type(sd).__name__}")
    non_string_keys = [repr(key) for key in sd.keys() if not isinstance(key, str)]
    if non_string_keys:
        raise RuntimeError(
            "WAN VAE state_dict keys must be strings. "
            f"non_string_keys_sample={non_string_keys[:10]}"
        )
    try:
        sd = resolve_wan21_vae_keyspace(sd).view
    except Exception as exc:  # noqa: BLE001 - surfaced as a load-time error with context
        raise RuntimeError(f"WAN VAE keyspace resolution failed: {exc}") from exc

    load_device = torch.device(device)
    to_args = dict(device=load_device, dtype=torch_dtype)
    with using_codex_operations(**to_args, manual_cast_enabled=True):
        model = WanVAE(
            dim=int(cfg.dim),
            z_dim=int(cfg.latent_channels),
            dim_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            attn_scales=(),
            temperal_downsample=(False, True, True),
            image_channels=int(cfg.image_channels),
            dropout=0.0,
        ).to(**to_args)
    missing, unexpected = safe_load_state_dict(model, sd, log_name="anima.wanvae")
    if missing or unexpected:
        raise RuntimeError(
            "WAN VAE strict load failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
        )
    model.eval()
    return model


__all__ = [
    "WanVAE",
    "WanVaeConfig",
    "WanVaeHeaderConfig",
    "detect_wan_vae_variant_from_header",
    "infer_wan_vae_config_from_safetensors_header",
    "load_wan_vae_from_safetensors",
]
