"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native LTX2 audio VAE with parser-state-compatible parameter names.
Implements the LTX2 audio autoencoder under `apps/**` without importing the official Diffusers LTX2 class while
preserving the parser-produced raw/original state-dict surface (`per_channel_statistics.*`).

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2AudioAutoencoder` (class): Config/state-driven native LTX2 audio KL autoencoder.
- `__all__` (constant): Explicit public export list for runtime imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["Ltx2AudioAutoencoder"]


_CONFIG_META_KEYS = frozenset({"_class_name", "_diffusers_version", "_name_or_path", "architectures"})
_ALLOWED_CLASS_NAMES = frozenset({"AutoencoderKLLTX2Audio", "Ltx2AudioAutoencoder"})
_LATENT_DOWNSAMPLE_FACTOR = 4


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


def _as_tuple_or_none(value: Any, *, name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"LTX2 config field {name!r} must be a list/tuple or null, got {type(value).__name__}.")
    return tuple(_as_int(v, name=f"{name}[]") for v in value)


def _as_tuple(value: Any, *, name: str, item_type: type[int]) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"LTX2 config field {name!r} must be a list/tuple, got {type(value).__name__}.")
    return tuple(_as_int(v, name=f"{name}[]") for v in value)


@dataclass(slots=True)
class _AutoencoderKLOutput:
    latent_dist: "_DiagonalGaussianDistribution"


@dataclass(slots=True)
class _DecoderOutput:
    sample: torch.Tensor


class _DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = bool(deterministic)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean


class _PerChannelStatistics(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("mean-of-means", torch.zeros(channels, dtype=torch.float32), persistent=True)
        self.register_buffer("std-of-means", torch.ones(channels, dtype=torch.float32), persistent=True)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean-of-means").to(device=x.device, dtype=x.dtype)
        std = self.get_buffer("std-of-means").to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean-of-means").to(device=x.device, dtype=x.dtype)
        std = self.get_buffer("std-of-means").to(device=x.device, dtype=x.dtype)
        return (x * std) + mean


class _AudioCausalConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        *,
        stride: int = 1,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: str = "height",
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis
        kernel_size = (
            (int(kernel_size), int(kernel_size))
            if isinstance(kernel_size, int)
            else tuple(int(v) for v in kernel_size)
        )
        dilation = (
            (int(dilation), int(dilation)) if isinstance(dilation, int) else tuple(int(v) for v in dilation)
        )
        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]
        if self.causality_axis == "none":
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis in {"width", "width-compatibility"}:
            self.padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis == "height":
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
        else:
            raise RuntimeError(f"LTX2 audio VAE invalid causality_axis: {causality_axis!r}.")
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.padding))


class _AudioPixelNorm(nn.Module):
    def __init__(self, *, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)
        return x / torch.sqrt(mean_sq + self.eps)


class _AudioAttnBlock(nn.Module):
    def __init__(self, in_channels: int, *, norm_type: str = "group") -> None:
        super().__init__()
        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        elif norm_type == "pixel":
            self.norm = _AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise RuntimeError(f"LTX2 audio VAE invalid norm_type: {norm_type!r}.")
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(x)
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        batch, channels, height, width = q.shape
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1).contiguous()
        k = k.reshape(batch, channels, height * width).contiguous()
        attn = torch.bmm(q, k) * (int(channels) ** (-0.5))
        attn = torch.softmax(attn, dim=2)
        v = v.reshape(batch, channels, height * width)
        attn = attn.permute(0, 2, 1).contiguous()
        hidden_states = torch.bmm(v, attn).reshape(batch, channels, height, width)
        hidden_states = self.proj_out(hidden_states)
        return x + hidden_states


class _AudioResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: str = "group",
        causality_axis: str | None = "height",
    ) -> None:
        super().__init__()
        if causality_axis is not None and causality_axis != "none" and norm_type == "group":
            raise RuntimeError("LTX2 audio VAE does not support causal ResNet blocks with GroupNorm.")
        self.in_channels = int(in_channels)
        self.out_channels = self.in_channels if out_channels is None else int(out_channels)
        self.use_conv_shortcut = bool(conv_shortcut)
        self.causality_axis = causality_axis
        if norm_type == "group":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=self.in_channels, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels, eps=1e-6, affine=True)
        elif norm_type == "pixel":
            self.norm1 = _AudioPixelNorm(dim=1, eps=1e-6)
            self.norm2 = _AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise RuntimeError(f"LTX2 audio VAE invalid norm_type: {norm_type!r}.")
        self.non_linearity = nn.SiLU()
        if causality_axis is not None:
            self.conv1 = _AudioCausalConv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                stride=1,
                causality_axis=causality_axis,
            )
            self.conv2 = _AudioCausalConv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                stride=1,
                causality_axis=causality_axis,
            )
        else:
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        else:
            self.temb_proj = None
        self.dropout = nn.Dropout(dropout)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = (
                    _AudioCausalConv2d(
                        self.in_channels,
                        self.out_channels,
                        kernel_size=3,
                        stride=1,
                        causality_axis=causality_axis,
                    )
                    if causality_axis is not None
                    else nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
                )
                self.nin_shortcut = None
            else:
                self.nin_shortcut = (
                    _AudioCausalConv2d(
                        self.in_channels,
                        self.out_channels,
                        kernel_size=1,
                        stride=1,
                        causality_axis=causality_axis,
                    )
                    if causality_axis is not None
                    else nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
                )
                self.conv_shortcut = None
        else:
            self.conv_shortcut = None
            self.nin_shortcut = None

    def forward(self, x: torch.Tensor, *, temb: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.norm1(x)
        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        if temb is not None and self.temb_proj is not None:
            hidden_states = hidden_states + self.temb_proj(self.non_linearity(temb))[:, :, None, None]
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.in_channels != self.out_channels:
            if self.conv_shortcut is not None:
                x = self.conv_shortcut(x)
            elif self.nin_shortcut is not None:
                x = self.nin_shortcut(x)
        return x + hidden_states


class _AudioDownsample(nn.Module):
    def __init__(self, in_channels: int, *, with_conv: bool, causality_axis: str | None = "height") -> None:
        super().__init__()
        self.with_conv = bool(with_conv)
        self.causality_axis = causality_axis
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            if self.causality_axis == "none":
                pad = (0, 1, 0, 1)
            elif self.causality_axis == "width":
                pad = (2, 0, 0, 1)
            elif self.causality_axis == "height":
                pad = (0, 1, 2, 0)
            elif self.causality_axis == "width-compatibility":
                pad = (1, 0, 0, 1)
            else:
                raise RuntimeError(f"LTX2 audio VAE invalid downsample causality_axis: {self.causality_axis!r}.")
            x = F.pad(x, pad, mode="constant", value=0)
            return self.conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class _AudioUpsample(nn.Module):
    def __init__(self, in_channels: int, *, with_conv: bool, causality_axis: str | None = "height") -> None:
        super().__init__()
        self.with_conv = bool(with_conv)
        self.causality_axis = causality_axis
        if self.with_conv:
            if causality_axis is not None:
                self.conv = _AudioCausalConv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    causality_axis=causality_axis,
                )
            else:
                self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if not self.with_conv:
            return x
        x = self.conv(x)
        if self.causality_axis is None or self.causality_axis == "none":
            return x
        if self.causality_axis == "height":
            return x[:, :, 1:, :]
        if self.causality_axis == "width":
            return x[:, :, :, 1:]
        if self.causality_axis == "width-compatibility":
            return x
        raise RuntimeError(f"LTX2 audio VAE invalid upsample causality_axis: {self.causality_axis!r}.")


class _AudioPatchifier:
    def __init__(
        self,
        *,
        patch_size: int,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = _LATENT_DOWNSAMPLE_FACTOR,
        is_causal: bool = True,
    ) -> None:
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        self.audio_latent_downsample_factor = int(audio_latent_downsample_factor)
        self.is_causal = bool(is_causal)
        self._patch_size = (1, int(patch_size), int(patch_size))

    def patchify(self, audio_latents: torch.Tensor) -> torch.Tensor:
        batch, channels, time, freq = audio_latents.shape
        return audio_latents.permute(0, 2, 1, 3).reshape(batch, time, channels * freq)

    def unpatchify(self, audio_latents: torch.Tensor, *, channels: int, mel_bins: int) -> torch.Tensor:
        batch, time, _ = audio_latents.shape
        return audio_latents.view(batch, time, channels, mel_bins).permute(0, 2, 1, 3)

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self._patch_size


class _AudioEncoder(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 128,
        output_channels: int = 1,
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] | None = None,
        in_channels: int = 2,
        resolution: int = 256,
        latent_channels: int = 8,
        ch_mult: Sequence[int] = (1, 2, 4),
        norm_type: str = "group",
        causality_axis: str | None = "width",
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: int | None = 64,
        double_z: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.mel_hop_length = int(mel_hop_length)
        self.is_causal = bool(is_causal)
        self.mel_bins = None if mel_bins is None else int(mel_bins)
        self.base_channels = int(base_channels)
        self.temb_ch = 0
        self.num_resolutions = len(tuple(ch_mult))
        self.num_res_blocks = int(num_res_blocks)
        self.resolution = int(resolution)
        self.in_channels = int(in_channels)
        self.out_ch = int(output_channels)
        self.give_pre_end = False
        self.tanh_out = False
        self.norm_type = norm_type
        self.latent_channels = int(latent_channels)
        self.channel_multipliers = tuple(int(v) for v in ch_mult)
        self.attn_resolutions = set(attn_resolutions) if attn_resolutions else None
        self.causality_axis = causality_axis

        base_block_channels = self.base_channels
        self.z_shape = (1, self.latent_channels, self.resolution, self.resolution)
        if self.causality_axis is not None:
            self.conv_in = _AudioCausalConv2d(
                self.in_channels,
                base_block_channels,
                kernel_size=3,
                stride=1,
                causality_axis=self.causality_axis,
            )
        else:
            self.conv_in = nn.Conv2d(self.in_channels, base_block_channels, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        block_in = base_block_channels
        curr_res = self.resolution
        for level in range(self.num_resolutions):
            stage = nn.Module()
            stage.block = nn.ModuleList()
            stage.attn = nn.ModuleList()
            block_out = self.base_channels * self.channel_multipliers[level]
            for _ in range(self.num_res_blocks):
                stage.block.append(
                    _AudioResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        norm_type=self.norm_type,
                        causality_axis=self.causality_axis,
                    )
                )
                block_in = block_out
                if self.attn_resolutions and curr_res in self.attn_resolutions:
                    stage.attn.append(_AudioAttnBlock(block_in, norm_type=self.norm_type))
            if level != self.num_resolutions - 1:
                stage.downsample = _AudioDownsample(block_in, with_conv=True, causality_axis=self.causality_axis)
                curr_res //= 2
            self.down.append(stage)

        self.mid = nn.Module()
        self.mid.block_1 = _AudioResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )
        self.mid.attn_1 = _AudioAttnBlock(block_in, norm_type=self.norm_type) if mid_block_add_attention else nn.Identity()
        self.mid.block_2 = _AudioResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )

        z_channels = 2 * self.latent_channels if double_z else self.latent_channels
        if self.norm_type == "group":
            self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        elif self.norm_type == "pixel":
            self.norm_out = _AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise RuntimeError(f"LTX2 audio VAE invalid norm_type: {self.norm_type!r}.")
        self.non_linearity = nn.SiLU()
        if self.causality_axis is not None:
            self.conv_out = _AudioCausalConv2d(
                block_in,
                z_channels,
                kernel_size=3,
                stride=1,
                causality_axis=self.causality_axis,
            )
        else:
            self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for level in range(self.num_resolutions):
            stage = self.down[level]
            for block_index, block in enumerate(stage.block):
                hidden_states = block(hidden_states, temb=None)
                if len(stage.attn) > block_index:
                    hidden_states = stage.attn[block_index](hidden_states)
            if level != self.num_resolutions - 1 and hasattr(stage, "downsample"):
                hidden_states = stage.downsample(hidden_states)
        hidden_states = self.mid.block_1(hidden_states, temb=None)
        hidden_states = self.mid.attn_1(hidden_states)
        hidden_states = self.mid.block_2(hidden_states, temb=None)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.non_linearity(hidden_states)
        return self.conv_out(hidden_states)


class _AudioDecoder(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 128,
        output_channels: int = 1,
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] | None = None,
        in_channels: int = 2,
        resolution: int = 256,
        latent_channels: int = 8,
        ch_mult: Sequence[int] = (1, 2, 4),
        norm_type: str = "group",
        causality_axis: str | None = "width",
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: int | None = 64,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.mel_hop_length = int(mel_hop_length)
        self.is_causal = bool(is_causal)
        self.mel_bins = None if mel_bins is None else int(mel_bins)
        self.patchifier = _AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=_LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )
        self.base_channels = int(base_channels)
        self.temb_ch = 0
        self.num_resolutions = len(tuple(ch_mult))
        self.num_res_blocks = int(num_res_blocks)
        self.resolution = int(resolution)
        self.in_channels = int(in_channels)
        self.out_ch = int(output_channels)
        self.give_pre_end = False
        self.tanh_out = False
        self.norm_type = norm_type
        self.latent_channels = int(latent_channels)
        self.channel_multipliers = tuple(int(v) for v in ch_mult)
        self.attn_resolutions = set(attn_resolutions) if attn_resolutions else None
        self.causality_axis = causality_axis

        base_block_channels = self.base_channels * self.channel_multipliers[-1]
        base_resolution = self.resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, self.latent_channels, base_resolution, base_resolution)
        if self.causality_axis is not None:
            self.conv_in = _AudioCausalConv2d(
                self.latent_channels,
                base_block_channels,
                kernel_size=3,
                stride=1,
                causality_axis=self.causality_axis,
            )
        else:
            self.conv_in = nn.Conv2d(self.latent_channels, base_block_channels, kernel_size=3, stride=1, padding=1)
        self.non_linearity = nn.SiLU()
        self.mid = nn.Module()
        self.mid.block_1 = _AudioResnetBlock(
            in_channels=base_block_channels,
            out_channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )
        self.mid.attn_1 = _AudioAttnBlock(base_block_channels, norm_type=self.norm_type) if mid_block_add_attention else nn.Identity()
        self.mid.block_2 = _AudioResnetBlock(
            in_channels=base_block_channels,
            out_channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )

        self.up = nn.ModuleList()
        block_in = base_block_channels
        curr_res = self.resolution // (2 ** (self.num_resolutions - 1))
        for level in reversed(range(self.num_resolutions)):
            stage = nn.Module()
            stage.block = nn.ModuleList()
            stage.attn = nn.ModuleList()
            block_out = self.base_channels * self.channel_multipliers[level]
            for _ in range(self.num_res_blocks + 1):
                stage.block.append(
                    _AudioResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        norm_type=self.norm_type,
                        causality_axis=self.causality_axis,
                    )
                )
                block_in = block_out
                if self.attn_resolutions and curr_res in self.attn_resolutions:
                    stage.attn.append(_AudioAttnBlock(block_in, norm_type=self.norm_type))
            if level != 0:
                stage.upsample = _AudioUpsample(block_in, with_conv=True, causality_axis=self.causality_axis)
                curr_res *= 2
            self.up.insert(0, stage)

        if self.norm_type == "group":
            self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        elif self.norm_type == "pixel":
            self.norm_out = _AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise RuntimeError(f"LTX2 audio VAE invalid norm_type: {self.norm_type!r}.")
        if self.causality_axis is not None:
            self.conv_out = _AudioCausalConv2d(
                block_in,
                self.out_ch,
                kernel_size=3,
                stride=1,
                causality_axis=self.causality_axis,
            )
        else:
            self.conv_out = nn.Conv2d(block_in, self.out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        _, _, frames, mel_bins = sample.shape
        target_frames = frames * _LATENT_DOWNSAMPLE_FACTOR
        if self.causality_axis is not None:
            target_frames = max(target_frames - (_LATENT_DOWNSAMPLE_FACTOR - 1), 1)
        target_channels = self.out_ch
        target_mel_bins = self.mel_bins if self.mel_bins is not None else mel_bins

        hidden_features = self.conv_in(sample)
        hidden_features = self.mid.block_1(hidden_features, temb=None)
        hidden_features = self.mid.attn_1(hidden_features)
        hidden_features = self.mid.block_2(hidden_features, temb=None)
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_index, block in enumerate(stage.block):
                hidden_features = block(hidden_features, temb=None)
                if len(stage.attn) > block_index:
                    hidden_features = stage.attn[block_index](hidden_features)
            if level != 0 and hasattr(stage, "upsample"):
                hidden_features = stage.upsample(hidden_features)
        if self.give_pre_end:
            return hidden_features
        hidden = self.norm_out(hidden_features)
        hidden = self.non_linearity(hidden)
        decoded_output = self.conv_out(hidden)
        decoded_output = torch.tanh(decoded_output) if self.tanh_out else decoded_output
        _, _, current_time, current_freq = decoded_output.shape
        decoded_output = decoded_output[:, :target_channels, : min(current_time, target_frames), : min(current_freq, target_mel_bins)]
        time_padding = target_frames - decoded_output.shape[2]
        freq_padding = target_mel_bins - decoded_output.shape[3]
        if time_padding > 0 or freq_padding > 0:
            decoded_output = F.pad(decoded_output, (0, max(freq_padding, 0), 0, max(time_padding, 0)))
        return decoded_output[:, :target_channels, :target_frames, :target_mel_bins]


class Ltx2AudioAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 128,
        output_channels: int = 2,
        ch_mult: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] | None = None,
        in_channels: int = 2,
        resolution: int = 256,
        latent_channels: int = 8,
        norm_type: str = "pixel",
        causality_axis: str | None = "height",
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: int | None = 64,
        double_z: bool = True,
    ) -> None:
        super().__init__()
        supported_causality_axes = {"none", "width", "height", "width-compatibility"}
        if causality_axis not in supported_causality_axes:
            raise RuntimeError(
                f"LTX2 audio VAE causality_axis must be one of {sorted(supported_causality_axes)!r}, got {causality_axis!r}."
            )
        ch_mult = tuple(int(v) for v in ch_mult)
        attn_resolutions_tuple = None if attn_resolutions is None else tuple(int(v) for v in attn_resolutions)
        self.config = SimpleNamespace(
            base_channels=int(base_channels),
            output_channels=int(output_channels),
            ch_mult=ch_mult,
            num_res_blocks=int(num_res_blocks),
            attn_resolutions=attn_resolutions_tuple,
            in_channels=int(in_channels),
            resolution=int(resolution),
            latent_channels=int(latent_channels),
            norm_type=str(norm_type),
            causality_axis=causality_axis,
            dropout=float(dropout),
            mid_block_add_attention=bool(mid_block_add_attention),
            sample_rate=int(sample_rate),
            mel_hop_length=int(mel_hop_length),
            is_causal=bool(is_causal),
            mel_bins=(None if mel_bins is None else int(mel_bins)),
            double_z=bool(double_z),
        )
        self.encoder = _AudioEncoder(
            base_channels=base_channels,
            output_channels=output_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions_tuple,
            in_channels=in_channels,
            resolution=resolution,
            latent_channels=latent_channels,
            norm_type=norm_type,
            causality_axis=causality_axis,
            dropout=dropout,
            mid_block_add_attention=mid_block_add_attention,
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            is_causal=is_causal,
            mel_bins=mel_bins,
            double_z=double_z,
        )
        self.decoder = _AudioDecoder(
            base_channels=base_channels,
            output_channels=output_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions_tuple,
            in_channels=in_channels,
            resolution=resolution,
            latent_channels=latent_channels,
            norm_type=norm_type,
            causality_axis=causality_axis,
            dropout=dropout,
            mid_block_add_attention=mid_block_add_attention,
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            is_causal=is_causal,
            mel_bins=mel_bins,
        )
        self.per_channel_statistics = _PerChannelStatistics(channels=int(base_channels))
        self.temporal_compression_ratio = _LATENT_DOWNSAMPLE_FACTOR
        self.mel_compression_ratio = _LATENT_DOWNSAMPLE_FACTOR
        self.use_slicing = False

    @property
    def latents_mean(self) -> torch.Tensor:
        return self.per_channel_statistics.get_buffer("mean-of-means")

    @property
    def latents_std(self) -> torch.Tensor:
        return self.per_channel_statistics.get_buffer("std-of-means")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Ltx2AudioAutoencoder":
        raw = _require_mapping(config, label="audio VAE")
        _validate_class_name(raw, allowed=_ALLOWED_CLASS_NAMES, label="audio VAE")
        expected = {
            "base_channels",
            "output_channels",
            "ch_mult",
            "num_res_blocks",
            "attn_resolutions",
            "in_channels",
            "resolution",
            "latent_channels",
            "norm_type",
            "causality_axis",
            "dropout",
            "mid_block_add_attention",
            "sample_rate",
            "mel_hop_length",
            "is_causal",
            "mel_bins",
            "double_z",
        }
        _reject_unexpected_keys(raw, allowed=expected, label="audio VAE")
        return cls(
            base_channels=_as_int(raw.get("base_channels", 128), name="base_channels"),
            output_channels=_as_int(raw.get("output_channels", 2), name="output_channels"),
            ch_mult=_as_tuple(raw.get("ch_mult", (1, 2, 4)), name="ch_mult", item_type=int),
            num_res_blocks=_as_int(raw.get("num_res_blocks", 2), name="num_res_blocks"),
            attn_resolutions=_as_tuple_or_none(raw.get("attn_resolutions"), name="attn_resolutions"),
            in_channels=_as_int(raw.get("in_channels", 2), name="in_channels"),
            resolution=_as_int(raw.get("resolution", 256), name="resolution"),
            latent_channels=_as_int(raw.get("latent_channels", 8), name="latent_channels"),
            norm_type=_as_str(raw.get("norm_type", "pixel"), name="norm_type"),
            causality_axis=(None if raw.get("causality_axis") is None else _as_str(raw.get("causality_axis"), name="causality_axis")),
            dropout=_as_float(raw.get("dropout", 0.0), name="dropout"),
            mid_block_add_attention=_as_bool(raw.get("mid_block_add_attention", False), name="mid_block_add_attention"),
            sample_rate=_as_int(raw.get("sample_rate", 16000), name="sample_rate"),
            mel_hop_length=_as_int(raw.get("mel_hop_length", 160), name="mel_hop_length"),
            is_causal=_as_bool(raw.get("is_causal", True), name="is_causal"),
            mel_bins=(None if raw.get("mel_bins") is None else _as_int(raw.get("mel_bins"), name="mel_bins")),
            double_z=_as_bool(raw.get("double_z", True), name="double_z"),
        )

    def load_strict_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise RuntimeError(f"LTX2 audio VAE strict load expects a mapping, got {type(state_dict).__name__}.")
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "LTX2 audio VAE strict load failed: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
            )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode(self, x: torch.Tensor, *, return_dict: bool = True) -> _AutoencoderKLOutput | tuple[_DiagonalGaussianDistribution]:
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            hidden_states = torch.cat(encoded_slices)
        else:
            hidden_states = self._encode(x)
        posterior = _DiagonalGaussianDistribution(hidden_states)
        if not return_dict:
            return (posterior,)
        return _AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def decode(self, z: torch.Tensor, *, return_dict: bool = True) -> _DecoderOutput | tuple[torch.Tensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)
        if not return_dict:
            return (decoded,)
        return _DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        *,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> _DecoderOutput | tuple[torch.Tensor]:
        posterior = self.encode(sample).latent_dist
        latent = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        decoded = self.decode(latent)
        if not return_dict:
            return (decoded.sample,)
        return decoded
