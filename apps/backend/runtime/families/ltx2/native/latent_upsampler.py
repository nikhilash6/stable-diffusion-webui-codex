"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native LTX2 latent x2 upsampler used by the explicit two-stage execution profile.
Rebuilds the LTX2 spatial latent-upsample model under `apps/**` from exact config owners carried by the vendored metadata
or the side-asset SafeTensors header, keeps the implementation native-only, and exposes strict config/state loading
without importing Diffusers runtime/model classes.

Symbols (top-level; keep in sync; no ghosts):
- `LTX2_RATIONAL_RESAMPLER_SCALE_MAPPING` (constant): Supported rational spatial resample factors.
- `Ltx2LatentUpsamplerModel` (class): Native LTX2 latent spatial upsampler with strict config/state loading.
"""

from __future__ import annotations

from math import comb
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

LTX2_RATIONAL_RESAMPLER_SCALE_MAPPING: dict[float, tuple[int, int]] = {
    0.75: (3, 4),
    1.5: (3, 2),
    2.0: (2, 1),
    4.0: (4, 1),
}
_LTX2_DIFFUSERS_UPSAMPLER_CLASS = "LTX2LatentUpsamplerModel"
_LTX2_LEGACY_UPSAMPLER_CLASS = "LatentUpsampler"


def _as_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"LTX2 latent upsampler config '{name}' must be an integer, got {type(value).__name__}.")
    return int(value)


def _as_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"LTX2 latent upsampler config '{name}' must be a bool, got {type(value).__name__}.")
    return bool(value)


def _as_optional_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(
            f"LTX2 latent upsampler config '{name}' must be a float when provided, got {type(value).__name__}."
        )
    return float(value)


def _require_mapping(config: Mapping[str, Any] | object) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise RuntimeError(
            f"LTX2 latent upsampler config must be a mapping, got {type(config).__name__}."
        )
    return config


def _reject_unexpected_keys(raw: Mapping[str, Any], *, allowed: set[str]) -> None:
    unexpected = sorted(str(key) for key in raw.keys() if str(key) not in allowed)
    if unexpected:
        raise RuntimeError(
            "LTX2 latent upsampler config includes unsupported keys: "
            f"{unexpected!r}."
        )


def _resolve_rational_spatial_scale(raw: Mapping[str, Any], *, class_name: str) -> float | None:
    if class_name == _LTX2_DIFFUSERS_UPSAMPLER_CLASS:
        legacy_keys = [key for key in ("spatial_scale", "rational_resampler") if key in raw]
        if legacy_keys:
            raise RuntimeError(
                "LTX2 latent upsampler diffusers-style config must not mix legacy keys "
                f"{legacy_keys!r}."
            )
        return _as_optional_float(raw.get("rational_spatial_scale", 2.0), name="rational_spatial_scale")

    if class_name == _LTX2_LEGACY_UPSAMPLER_CLASS:
        if "rational_spatial_scale" in raw:
            raise RuntimeError(
                "LTX2 latent upsampler legacy config must not declare `rational_spatial_scale`; "
                "use `spatial_scale` plus `rational_resampler` instead."
            )
        rational_resampler = _as_bool(raw.get("rational_resampler", False), name="rational_resampler")
        spatial_scale = _as_optional_float(raw.get("spatial_scale", 2.0), name="spatial_scale")
        if spatial_scale is None:
            spatial_scale = 2.0
        return float(spatial_scale) if rational_resampler else None

    raise RuntimeError(
        "LTX2 latent upsampler config requires `_class_name` in "
        f"{[_LTX2_DIFFUSERS_UPSAMPLER_CLASS, _LTX2_LEGACY_UPSAMPLER_CLASS]!r}; got {class_name!r}."
    )


class _ResBlock(nn.Module):
    def __init__(self, channels: int, *, mid_channels: int | None = None, dims: int = 3) -> None:
        super().__init__()
        hidden_channels = channels if mid_channels is None else int(mid_channels)
        conv_cls = nn.Conv2d if dims == 2 else nn.Conv3d
        self.conv1 = conv_cls(channels, hidden_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, hidden_channels)
        self.conv2 = conv_cls(hidden_channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return self.activation(hidden_states + residual)


class _PixelShuffleND(nn.Module):
    def __init__(self, dims: int, *, upscale_factors: tuple[int, ...] = (2, 2, 2)) -> None:
        super().__init__()
        if dims not in {1, 2, 3}:
            raise RuntimeError(f"LTX2 latent upsampler pixel shuffle dims must be 1, 2, or 3; got {dims}.")
        self.dims = int(dims)
        self.upscale_factors = tuple(int(factor) for factor in upscale_factors)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.dims == 3:
            return (
                hidden_states.unflatten(1, (-1, *self.upscale_factors[:3]))
                .permute(0, 1, 5, 2, 6, 3, 7, 4)
                .flatten(6, 7)
                .flatten(4, 5)
                .flatten(2, 3)
            )
        if self.dims == 2:
            return (
                hidden_states.unflatten(1, (-1, *self.upscale_factors[:2]))
                .permute(0, 1, 4, 2, 5, 3)
                .flatten(4, 5)
                .flatten(2, 3)
            )
        return hidden_states.unflatten(1, (-1, *self.upscale_factors[:1])).permute(0, 1, 3, 2, 4, 5).flatten(2, 3)


class _BlurDownsample(nn.Module):
    def __init__(self, *, dims: int, stride: int, kernel_size: int = 5) -> None:
        super().__init__()
        if dims not in {2, 3}:
            raise RuntimeError(f"LTX2 latent upsampler blur dims must be 2 or 3; got {dims}.")
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise RuntimeError(f"LTX2 latent upsampler blur kernel_size must be odd and >= 3; got {kernel_size}.")
        self.dims = int(dims)
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)
        kernel_1d = torch.tensor([comb(kernel_size - 1, index) for index in range(kernel_size)], dtype=torch.float32)
        kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.register_buffer("kernel", kernel_2d[None, None, :, :], persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return hidden_states
        if self.dims == 2:
            channels = int(hidden_states.shape[1])
            weight = self.kernel.expand(channels, 1, self.kernel_size, self.kernel_size)
            return F.conv2d(
                hidden_states,
                weight=weight,
                bias=None,
                stride=self.stride,
                padding=self.kernel_size // 2,
                groups=channels,
            )
        batch_size, channels, num_frames, _, _ = hidden_states.shape
        flattened = hidden_states.transpose(1, 2).flatten(0, 1)
        weight = self.kernel.expand(channels, 1, self.kernel_size, self.kernel_size)
        flattened = F.conv2d(
            flattened,
            weight=weight,
            bias=None,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=channels,
        )
        height_out, width_out = flattened.shape[-2:]
        return flattened.unflatten(0, (batch_size, num_frames)).reshape(batch_size, channels, num_frames, height_out, width_out)


class _SpatialRationalResampler(nn.Module):
    def __init__(self, *, mid_channels: int, scale: float) -> None:
        super().__init__()
        normalized_scale = float(scale)
        ratio = LTX2_RATIONAL_RESAMPLER_SCALE_MAPPING.get(normalized_scale)
        if ratio is None:
            raise RuntimeError(
                "LTX2 latent upsampler rational_spatial_scale is unsupported; "
                f"got {scale!r} supported={sorted(LTX2_RATIONAL_RESAMPLER_SCALE_MAPPING)!r}."
            )
        numerator, denominator = ratio
        self.scale = normalized_scale
        self.num = int(numerator)
        self.den = int(denominator)
        self.conv = nn.Conv2d(mid_channels, (self.num**2) * mid_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = _PixelShuffleND(2, upscale_factors=(self.num, self.num))
        self.blur_down = _BlurDownsample(dims=2, stride=self.den)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.pixel_shuffle(hidden_states)
        return self.blur_down(hidden_states)


class Ltx2LatentUpsamplerModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 128,
        mid_channels: int = 1024,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        rational_spatial_scale: float | None = 2.0,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            in_channels=int(in_channels),
            mid_channels=int(mid_channels),
            num_blocks_per_stage=int(num_blocks_per_stage),
            dims=int(dims),
            spatial_upsample=bool(spatial_upsample),
            temporal_upsample=bool(temporal_upsample),
            rational_spatial_scale=rational_spatial_scale,
        )

        conv_cls = nn.Conv2d if int(dims) == 2 else nn.Conv3d
        self.initial_conv = conv_cls(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(32, mid_channels)
        self.initial_activation = nn.SiLU()
        self.res_blocks = nn.ModuleList([_ResBlock(mid_channels, dims=int(dims)) for _ in range(int(num_blocks_per_stage))])

        if spatial_upsample and temporal_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
                _PixelShuffleND(3),
            )
        elif spatial_upsample:
            if rational_spatial_scale is None:
                self.upsampler = nn.Sequential(
                    nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
                    _PixelShuffleND(2),
                )
            else:
                self.upsampler = _SpatialRationalResampler(mid_channels=mid_channels, scale=float(rational_spatial_scale))
        elif temporal_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
                _PixelShuffleND(1),
            )
        else:
            raise RuntimeError("LTX2 latent upsampler requires spatial_upsample or temporal_upsample to be enabled.")

        self.post_upsample_res_blocks = nn.ModuleList(
            [_ResBlock(mid_channels, dims=int(dims)) for _ in range(int(num_blocks_per_stage))]
        )
        self.final_conv = conv_cls(mid_channels, in_channels, kernel_size=3, padding=1)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Ltx2LatentUpsamplerModel":
        raw = _require_mapping(config)
        class_name = str(raw.get("_class_name", _LTX2_DIFFUSERS_UPSAMPLER_CLASS) or _LTX2_DIFFUSERS_UPSAMPLER_CLASS).strip()
        allowed = {
            "_class_name",
            "_diffusers_version",
            "dims",
            "in_channels",
            "mid_channels",
            "num_blocks_per_stage",
            "rational_spatial_scale",
            "spatial_scale",
            "rational_resampler",
            "spatial_upsample",
            "temporal_upsample",
        }
        _reject_unexpected_keys(raw, allowed=allowed)
        return cls(
            in_channels=_as_int(raw.get("in_channels", 128), name="in_channels"),
            mid_channels=_as_int(raw.get("mid_channels", 1024), name="mid_channels"),
            num_blocks_per_stage=_as_int(raw.get("num_blocks_per_stage", 4), name="num_blocks_per_stage"),
            dims=_as_int(raw.get("dims", 3), name="dims"),
            spatial_upsample=_as_bool(raw.get("spatial_upsample", True), name="spatial_upsample"),
            temporal_upsample=_as_bool(raw.get("temporal_upsample", False), name="temporal_upsample"),
            rational_spatial_scale=_resolve_rational_spatial_scale(raw, class_name=class_name),
        )

    def load_strict_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise RuntimeError(
                f"LTX2 latent upsampler strict load expects a mapping state_dict, got {type(state_dict).__name__}."
            )
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "LTX2 latent upsampler strict load failed: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 5:
            raise RuntimeError(
                "LTX2 latent upsampler expects [batch, channels, frames, height, width] latents; "
                f"got shape={tuple(int(dim) for dim in hidden_states.shape)!r}."
            )
        batch_size = int(hidden_states.shape[0])
        if int(self.config.dims) == 2:
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
            hidden_states = self.initial_conv(hidden_states)
            hidden_states = self.initial_norm(hidden_states)
            hidden_states = self.initial_activation(hidden_states)
            for block in self.res_blocks:
                hidden_states = block(hidden_states)
            hidden_states = self.upsampler(hidden_states)
            for block in self.post_upsample_res_blocks:
                hidden_states = block(hidden_states)
            hidden_states = self.final_conv(hidden_states)
            return hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

        hidden_states = self.initial_conv(hidden_states)
        hidden_states = self.initial_norm(hidden_states)
        hidden_states = self.initial_activation(hidden_states)
        for block in self.res_blocks:
            hidden_states = block(hidden_states)
        if bool(self.config.temporal_upsample):
            hidden_states = self.upsampler(hidden_states)
            hidden_states = hidden_states[:, :, 1:, :, :]
        else:
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
            hidden_states = self.upsampler(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
        for block in self.post_upsample_res_blocks:
            hidden_states = block(hidden_states)
        return self.final_conv(hidden_states)


__all__ = [
    "LTX2_RATIONAL_RESAMPLER_SCALE_MAPPING",
    "Ltx2LatentUpsamplerModel",
]
