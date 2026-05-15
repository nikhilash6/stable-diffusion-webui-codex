"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native raw-key LTX2 vocoder owners for legacy and real LTX 2.3 wrapped layouts.
Implements both the legacy flat raw LTX2 vocoder and the real LTX 2.3 wrapped `VocoderWithBWE` owner under `apps/**`
with strict config parsing and strict state-dict loading.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2VocoderConfig` (dataclass): Strict parsed config contract for the legacy flat native LTX2 vocoder.
- `Ltx2Vocoder` (class): Native legacy flat raw-key LTX2 vocoder module.
- `load_ltx2_vocoder` (function): Strict loader for parser-owned LTX2 vocoder state dicts across supported raw layouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from apps.backend.runtime.models.state_dict import safe_load_state_dict

_ALLOWED_CONFIG_METADATA_KEYS = frozenset({"_class_name", "_diffusers_version"})
_RAW_VOCODER_PREFIXES = ("conv_pre.", "ups.", "resblocks.", "conv_post.")
_RAW_VOCODER_TOP_LEVEL_PREFIXES = frozenset({"conv_pre", "ups", "resblocks", "conv_post"})
_WRAPPED_23_VOCODER_PREFIXES = ("bwe_generator.", "mel_stft.", "vocoder.")
_WRAPPED_23_VOCODER_TOP_LEVEL_PREFIXES = frozenset({"bwe_generator", "mel_stft", "vocoder"})
_REJECTED_VOCODER_PREFIXES = ("conv_in.", "upsamplers.", "resnets.", "conv_out.")
_LRELU_SLOPE = 0.1
_WRAPPED_GENERATOR_CONFIG_KEYS = frozenset(
    {
        "resblock_kernel_sizes",
        "upsample_rates",
        "upsample_kernel_sizes",
        "resblock_dilation_sizes",
        "upsample_initial_channel",
        "stereo",
        "activation",
        "use_bias_at_final",
        "use_tanh_at_final",
        "apply_final_activation",
        "resblock",
        "output_sampling_rate",
    }
)


@dataclass(frozen=True)
class Ltx2VocoderConfig:
    in_channels: int
    hidden_channels: int
    out_channels: int
    upsample_kernel_sizes: tuple[int, ...]
    upsample_factors: tuple[int, ...]
    resnet_kernel_sizes: tuple[int, ...]
    resnet_dilations: tuple[tuple[int, ...], ...]
    leaky_relu_negative_slope: float
    output_sampling_rate: int


@dataclass(frozen=True)
class _GeneratorConfig:
    resblock_kernel_sizes: tuple[int, ...]
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    upsample_initial_channel: int
    stereo: bool
    activation: str
    use_bias_at_final: bool
    use_tanh_at_final: bool
    apply_final_activation: bool
    resblock: str
    output_sampling_rate: int | None = None


@dataclass(frozen=True)
class _Wrapped23VocoderConfig:
    vocoder: _GeneratorConfig
    bwe: _GeneratorConfig
    input_sampling_rate: int
    output_sampling_rate: int
    hop_length: int
    n_fft: int
    num_mels: int


class _ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilations: tuple[int, ...] = (1, 3, 5),
        leaky_relu_negative_slope: float = 0.1,
        padding_mode: str = "same",
    ) -> None:
        super().__init__()
        self.dilations = tuple(int(value) for value in dilations)
        self.negative_slope = float(leaky_relu_negative_slope)
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    int(channels),
                    int(channels),
                    int(kernel_size),
                    stride=int(stride),
                    dilation=int(dilation),
                    padding=padding_mode,
                )
                for dilation in self.dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    int(channels),
                    int(channels),
                    int(kernel_size),
                    stride=int(stride),
                    dilation=1,
                    padding=padding_mode,
                )
                for _ in self.dilations
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = F.leaky_relu(hidden_states, negative_slope=self.negative_slope)
            residual = conv1(residual)
            residual = F.leaky_relu(residual, negative_slope=self.negative_slope)
            residual = conv2(residual)
            hidden_states = hidden_states + residual
        return hidden_states


class Ltx2Vocoder(nn.Module):
    def __init__(self, config: Ltx2VocoderConfig) -> None:
        super().__init__()
        self.config = config
        self.num_upsample_layers = len(config.upsample_kernel_sizes)
        self.resblocks_per_upsample = len(config.resnet_kernel_sizes)
        self.out_channels = int(config.out_channels)
        self.total_upsample_factor = math.prod(config.upsample_factors)
        self.negative_slope = float(config.leaky_relu_negative_slope)

        if self.num_upsample_layers != len(config.upsample_factors):
            raise RuntimeError(
                "Unsupported LTX2 vocoder config: `upsample_kernel_sizes` and `upsample_factors` must be the same length."
            )
        if self.resblocks_per_upsample != len(config.resnet_dilations):
            raise RuntimeError(
                "Unsupported LTX2 vocoder config: `resnet_kernel_sizes` and `resnet_dilations` must be the same length."
            )

        self.conv_pre = nn.Conv1d(int(config.in_channels), int(config.hidden_channels), kernel_size=7, stride=1, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        input_channels = int(config.hidden_channels)
        output_channels = input_channels
        for stride, kernel_size in zip(config.upsample_factors, config.upsample_kernel_sizes):
            output_channels = input_channels // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    input_channels,
                    output_channels,
                    int(kernel_size),
                    stride=int(stride),
                    padding=(int(kernel_size) - int(stride)) // 2,
                )
            )
            for resnet_kernel_size, dilations in zip(config.resnet_kernel_sizes, config.resnet_dilations):
                self.resblocks.append(
                    _ResBlock(
                        output_channels,
                        kernel_size=int(resnet_kernel_size),
                        dilations=tuple(int(value) for value in dilations),
                        leaky_relu_negative_slope=self.negative_slope,
                    )
                )
            input_channels = output_channels

        self.conv_post = nn.Conv1d(output_channels, int(config.out_channels), kernel_size=7, stride=1, padding=3)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Ltx2Vocoder":
        return cls(_parse_vocoder_config(config))

    def forward(self, hidden_states: torch.Tensor, time_last: bool = False) -> torch.Tensor:
        if hidden_states.ndim != 4:
            raise RuntimeError(
                "LTX2 vocoder expects a rank-4 mel tensor. "
                f"Got shape={tuple(int(dim) for dim in hidden_states.shape)!r}."
            )
        if not bool(time_last):
            hidden_states = hidden_states.transpose(2, 3)
        hidden_states = hidden_states.flatten(1, 2)
        hidden_states = self.conv_pre(hidden_states)

        for upsample_index in range(self.num_upsample_layers):
            hidden_states = F.leaky_relu(hidden_states, negative_slope=self.negative_slope)
            hidden_states = self.ups[upsample_index](hidden_states)
            start = upsample_index * self.resblocks_per_upsample
            end = (upsample_index + 1) * self.resblocks_per_upsample
            resblock_outputs = torch.stack(
                [self.resblocks[index](hidden_states) for index in range(start, end)],
                dim=0,
            )
            hidden_states = torch.mean(resblock_outputs, dim=0)

        hidden_states = F.leaky_relu(hidden_states, negative_slope=0.01)
        hidden_states = self.conv_post(hidden_states)
        return torch.tanh(hidden_states)


class _LowPassFilter1d(nn.Module):
    def __init__(
        self,
        *,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ) -> None:
        super().__init__()
        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be >= 0.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        if kernel_size <= 0:
            raise ValueError("LowPassFilter1d kernel_size must be > 0.")
        self.kernel_size = int(kernel_size)
        self.even = self.kernel_size % 2 == 0
        self.pad_left = self.kernel_size // 2 - int(self.even)
        self.pad_right = self.kernel_size // 2
        self.stride = int(stride)
        self.padding = bool(padding)
        self.padding_mode = str(padding_mode)
        filter_tensor = _kaiser_sinc_filter1d(cutoff=float(cutoff), half_width=float(half_width), kernel_size=self.kernel_size)
        self.register_buffer("filter", filter_tensor)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, channels, _ = hidden_states.shape
        if self.padding:
            hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode=self.padding_mode)
        filter_tensor = _cast_like(self.filter.expand(channels, -1, -1), hidden_states)
        return F.conv1d(hidden_states, filter_tensor, stride=self.stride, groups=channels)


class _UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None, *, persistent: bool = True, window_type: str = "kaiser") -> None:
        super().__init__()
        if ratio <= 0:
            raise ValueError(f"Upsample ratio must be > 0; got {ratio!r}.")
        self.ratio = int(ratio)
        self.stride = int(ratio)

        if str(window_type) == "hann":
            rolloff = 0.99
            lowpass_filter_width = 6
            width = math.ceil(lowpass_filter_width / rolloff)
            self.kernel_size = 2 * width * self.ratio + 1
            self.pad = width
            self.pad_left = 2 * width * self.ratio
            self.pad_right = self.kernel_size - self.ratio
            time = (torch.arange(self.kernel_size, dtype=torch.float32) / self.ratio - width) * rolloff
            time_clamped = time.clamp(-lowpass_filter_width, lowpass_filter_width)
            window = torch.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
            filter_tensor = (torch.sinc(time) * window * rolloff / self.ratio).view(1, 1, -1)
        else:
            self.kernel_size = int(6 * self.ratio // 2) * 2 if kernel_size is None else int(kernel_size)
            self.pad = self.kernel_size // self.ratio - 1
            self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
            self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            filter_tensor = _kaiser_sinc_filter1d(
                cutoff=0.5 / self.ratio,
                half_width=0.6 / self.ratio,
                kernel_size=self.kernel_size,
            )

        self.register_buffer("filter", filter_tensor, persistent=persistent)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, channels, _ = hidden_states.shape
        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        filter_tensor = _cast_like(self.filter.expand(channels, -1, -1), hidden_states)
        hidden_states = self.ratio * F.conv_transpose1d(hidden_states, filter_tensor, stride=self.stride, groups=channels)
        return hidden_states[..., self.pad_left : -self.pad_right]


class _DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        if ratio <= 0:
            raise ValueError(f"Downsample ratio must be > 0; got {ratio!r}.")
        resolved_kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else int(kernel_size)
        self.lowpass = _LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=resolved_kernel_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lowpass(hidden_states)


class _Activation1d(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        *,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.act = activation
        self.upsample = _UpSample1d(up_ratio, up_kernel_size)
        self.downsample = _DownSample1d(down_ratio, down_kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.downsample(hidden_states)
        return hidden_states


class _Snake(nn.Module):
    def __init__(self, in_features: int, *, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = True) -> None:
        super().__init__()
        self.alpha_logscale = bool(alpha_logscale)
        initial_alpha = torch.zeros(int(in_features)) if self.alpha_logscale else torch.ones(int(in_features)) * float(alpha)
        self.alpha = nn.Parameter(initial_alpha)
        self.alpha.requires_grad = bool(alpha_trainable)
        self.eps = 1e-9

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = _cast_like(self.alpha.unsqueeze(0).unsqueeze(-1), hidden_states)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return hidden_states + (1.0 / (alpha + self.eps)) * torch.sin(hidden_states * alpha).pow(2)


class _SnakeBeta(nn.Module):
    def __init__(self, in_features: int, *, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = True) -> None:
        super().__init__()
        self.alpha_logscale = bool(alpha_logscale)
        initial = torch.zeros(int(in_features)) if self.alpha_logscale else torch.ones(int(in_features)) * float(alpha)
        self.alpha = nn.Parameter(initial.clone())
        self.beta = nn.Parameter(initial.clone())
        self.alpha.requires_grad = bool(alpha_trainable)
        self.beta.requires_grad = bool(alpha_trainable)
        self.eps = 1e-9

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = _cast_like(self.alpha.unsqueeze(0).unsqueeze(-1), hidden_states)
        beta = _cast_like(self.beta.unsqueeze(0).unsqueeze(-1), hidden_states)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return hidden_states + (1.0 / (beta + self.eps)) * torch.sin(hidden_states * alpha).pow(2)


class _AmpBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Sequence[int] = (1, 3, 5), *, activation: str = "snake") -> None:
        super().__init__()
        dilation_values = tuple(int(value) for value in dilation)
        act_cls = _SnakeBeta if str(activation).strip().lower() == "snakebeta" else _Snake
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=dilation_values[0], padding=_get_padding(int(kernel_size), dilation_values[0])),
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=dilation_values[1], padding=_get_padding(int(kernel_size), dilation_values[1])),
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=dilation_values[2], padding=_get_padding(int(kernel_size), dilation_values[2])),
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=1, padding=_get_padding(int(kernel_size), 1)),
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=1, padding=_get_padding(int(kernel_size), 1)),
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=1, padding=_get_padding(int(kernel_size), 1)),
            ]
        )
        self.acts1 = nn.ModuleList([_Activation1d(act_cls(int(channels))) for _ in range(len(self.convs1))])
        self.acts2 = nn.ModuleList([_Activation1d(act_cls(int(channels))) for _ in range(len(self.convs2))])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            residual = act1(hidden_states)
            residual = conv1(residual)
            residual = act2(residual)
            residual = conv2(residual)
            hidden_states = hidden_states + residual
        return hidden_states


class _ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Sequence[int] = (1, 3, 5)) -> None:
        super().__init__()
        dilation_values = tuple(int(value) for value in dilation)
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=dilation_values[0], padding=_get_padding(int(kernel_size), dilation_values[0])),
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=dilation_values[1], padding=_get_padding(int(kernel_size), dilation_values[1])),
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=dilation_values[2], padding=_get_padding(int(kernel_size), dilation_values[2])),
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=1, padding=_get_padding(int(kernel_size), 1)),
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=1, padding=_get_padding(int(kernel_size), 1)),
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=1, padding=_get_padding(int(kernel_size), 1)),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = F.leaky_relu(hidden_states, _LRELU_SLOPE)
            residual = conv1(residual)
            residual = F.leaky_relu(residual, _LRELU_SLOPE)
            residual = conv2(residual)
            hidden_states = hidden_states + residual
        return hidden_states


class _ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Sequence[int] = (1, 3)) -> None:
        super().__init__()
        dilation_values = tuple(int(value) for value in dilation)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=dilation_values[0], padding=_get_padding(int(kernel_size), dilation_values[0])),
                nn.Conv1d(int(channels), int(channels), int(kernel_size), 1, dilation=dilation_values[1], padding=_get_padding(int(kernel_size), dilation_values[1])),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            residual = F.leaky_relu(hidden_states, _LRELU_SLOPE)
            residual = conv(residual)
            hidden_states = hidden_states + residual
        return hidden_states


class _GeneratorVocoder(nn.Module):
    def __init__(self, config: _GeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        if len(config.upsample_rates) != len(config.upsample_kernel_sizes):
            raise RuntimeError(
                "Unsupported wrapped LTX2 vocoder config: `upsample_rates` and `upsample_kernel_sizes` must have equal length."
            )
        if len(config.resblock_kernel_sizes) != len(config.resblock_dilation_sizes):
            raise RuntimeError(
                "Unsupported wrapped LTX2 vocoder config: `resblock_kernel_sizes` and `resblock_dilation_sizes` must have equal length."
            )

        in_channels = 128 if config.stereo else 64
        self.conv_pre = nn.Conv1d(in_channels, int(config.upsample_initial_channel), 7, 1, padding=3)

        if config.resblock == "1":
            resblock_cls = _ResBlock1
        elif config.resblock == "2":
            resblock_cls = _ResBlock2
        elif config.resblock == "AMP1":
            resblock_cls = _AmpBlock1
        else:
            raise RuntimeError(f"Unsupported wrapped LTX2 vocoder resblock type: {config.resblock!r}.")

        self.ups = nn.ModuleList()
        for index, (stride, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    int(config.upsample_initial_channel) // (2**index),
                    int(config.upsample_initial_channel) // (2 ** (index + 1)),
                    int(kernel_size),
                    int(stride),
                    padding=(int(kernel_size) - int(stride)) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for index in range(len(self.ups)):
            channels = int(config.upsample_initial_channel) // (2 ** (index + 1))
            for kernel_size, dilations in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                if config.resblock == "AMP1":
                    self.resblocks.append(_AmpBlock1(channels, int(kernel_size), dilations, activation=config.activation))
                else:
                    self.resblocks.append(resblock_cls(channels, int(kernel_size), dilations))

        output_channels = 2 if config.stereo else 1
        if config.resblock == "AMP1":
            act_cls = _SnakeBeta if config.activation == "snakebeta" else _Snake
            self.act_post = _Activation1d(act_cls(channels))
        else:
            self.act_post = nn.LeakyReLU()
        self.conv_post = nn.Conv1d(channels, output_channels, 7, 1, padding=3, bias=config.use_bias_at_final)
        self.upsample_factor = math.prod(config.upsample_rates)

    def forward(self, mel_spectrograms: torch.Tensor) -> torch.Tensor:
        hidden_states = mel_spectrograms
        if hidden_states.dim() == 4:
            if hidden_states.shape[1] != 2:
                raise RuntimeError(
                    "Wrapped LTX2 vocoder expects stereo mel input with shape (batch, 2, mel_bins, frames). "
                    f"Got shape={tuple(int(dim) for dim in hidden_states.shape)!r}."
                )
            hidden_states = torch.cat((hidden_states[:, 0, :, :], hidden_states[:, 1, :, :]), dim=1)
        elif hidden_states.dim() != 3:
            raise RuntimeError(
                "Wrapped LTX2 vocoder expects a rank-3 or rank-4 mel tensor. "
                f"Got shape={tuple(int(dim) for dim in hidden_states.shape)!r}."
            )

        hidden_states = self.conv_pre(hidden_states)
        for upsample_index in range(self.num_upsamples):
            if self.config.resblock != "AMP1":
                hidden_states = F.leaky_relu(hidden_states, _LRELU_SLOPE)
            hidden_states = self.ups[upsample_index](hidden_states)
            resblock_sum = None
            for kernel_index in range(self.num_kernels):
                block = self.resblocks[upsample_index * self.num_kernels + kernel_index]
                block_output = block(hidden_states)
                resblock_sum = block_output if resblock_sum is None else resblock_sum + block_output
            hidden_states = resblock_sum / self.num_kernels

        hidden_states = self.act_post(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        if self.config.apply_final_activation:
            if self.config.use_tanh_at_final:
                hidden_states = torch.tanh(hidden_states)
            else:
                hidden_states = torch.clamp(hidden_states, -1, 1)
        return hidden_states


class _STFTFn(nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        n_freqs = int(filter_length) // 2 + 1
        self.register_buffer("forward_basis", torch.zeros(n_freqs * 2, 1, int(filter_length)))
        self.register_buffer("inverse_basis", torch.zeros(n_freqs * 2, 1, int(filter_length)))

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        left_pad = max(0, self.win_length - self.hop_length)
        waveform = F.pad(waveform, (left_pad, 0))
        forward_basis = _cast_like(self.forward_basis, waveform)
        spectrum = F.conv1d(waveform, forward_basis, stride=self.hop_length, padding=0)
        n_freqs = spectrum.shape[1] // 2
        real = spectrum[:, :n_freqs]
        imag = spectrum[:, n_freqs:]
        magnitude = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag.float(), real.float()).to(real.dtype)
        return magnitude, phase


class _MelSTFT(nn.Module):
    def __init__(
        self,
        *,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
    ) -> None:
        super().__init__()
        self.stft_fn = _STFTFn(filter_length=filter_length, hop_length=hop_length, win_length=win_length)
        n_freqs = int(filter_length) // 2 + 1
        self.register_buffer("mel_basis", torch.zeros(int(n_mel_channels), n_freqs))

    def mel_spectrogram(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        magnitude, phase = self.stft_fn(waveform)
        energy = torch.norm(magnitude, dim=1)
        mel_basis = _cast_like(self.mel_basis, magnitude)
        mel = torch.matmul(mel_basis, magnitude)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel, magnitude, phase, energy


class _Ltx23VocoderWithBwe(nn.Module):
    def __init__(self, config: _Wrapped23VocoderConfig) -> None:
        super().__init__()
        self.config = config
        self.vocoder = _GeneratorVocoder(config.vocoder)
        self.bwe_generator = _GeneratorVocoder(config.bwe)
        self.input_sample_rate = int(config.input_sampling_rate)
        self.output_sample_rate = int(config.output_sampling_rate)
        self.hop_length = int(config.hop_length)
        self.mel_stft = _MelSTFT(
            filter_length=int(config.n_fft),
            hop_length=int(config.hop_length),
            win_length=int(config.n_fft),
            n_mel_channels=int(config.num_mels),
        )
        ratio = int(config.output_sampling_rate) // int(config.input_sampling_rate)
        if ratio <= 1 or int(config.output_sampling_rate) % int(config.input_sampling_rate) != 0:
            raise RuntimeError(
                "Unsupported wrapped LTX2 vocoder config: output/input sampling rates must form an integer upsample ratio > 1. "
                f"Got input={config.input_sampling_rate!r} output={config.output_sampling_rate!r}."
            )
        self.resampler = _UpSample1d(ratio=ratio, persistent=False, window_type="hann")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "_Ltx23VocoderWithBwe":
        return cls(_parse_wrapped_23_vocoder_config(config))

    def _compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = audio.shape
        flattened = audio.reshape(batch * channels, -1)
        mel, _, _, _ = self.mel_stft.mel_spectrogram(flattened)
        return mel.reshape(batch, channels, mel.shape[1], mel.shape[2])

    def forward(self, mel_spectrograms: torch.Tensor) -> torch.Tensor:
        audio = self.vocoder(mel_spectrograms)
        _, _, low_rate_length = audio.shape
        output_length = low_rate_length * self.output_sample_rate // self.input_sample_rate

        remainder = low_rate_length % self.hop_length
        if remainder != 0:
            audio = F.pad(audio, (0, self.hop_length - remainder))

        mel = self._compute_mel(audio)
        residual = self.bwe_generator(mel)
        skip = self.resampler(audio)
        if residual.shape != skip.shape:
            raise RuntimeError(
                "Wrapped LTX2 vocoder residual/skip shapes diverged. "
                f"residual={tuple(int(dim) for dim in residual.shape)!r} skip={tuple(int(dim) for dim in skip.shape)!r}."
            )
        return torch.clamp(residual + skip, -1, 1)[..., :output_length]



def load_ltx2_vocoder(
    config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    device: torch.device,
    torch_dtype: torch.dtype,
) -> nn.Module:
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"LTX2 vocoder state_dict must be a mapping; got {type(state_dict).__name__}.")

    if _is_wrapped_23_vocoder_state_dict(state_dict):
        _validate_wrapped_23_vocoder_state_dict(state_dict)
        if not isinstance(config, Mapping):
            raise RuntimeError(
                "Wrapped LTX2 2.3 vocoder loading requires a mapping config with `vocoder` and `bwe` sections. "
                f"Got {type(config).__name__}."
            )
        try:
            module = _Ltx23VocoderWithBwe.from_config(config)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Wrapped LTX2 2.3 vocoder config instantiation failed: {exc}") from exc
    else:
        _validate_raw_vocoder_state_dict(state_dict)
        try:
            module = Ltx2Vocoder.from_config(config)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LTX2 vocoder config instantiation failed: {exc}") from exc

    missing, unexpected = safe_load_state_dict(module, state_dict, log_name="ltx2.vocoder")
    if missing or unexpected:
        raise RuntimeError(
            "LTX2 vocoder strict load failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
        )

    try:
        module = module.to(device=device, dtype=torch_dtype)
    except Exception:
        module = module.to(device=device)
    module.eval()
    return module



def _parse_vocoder_config(config: Mapping[str, Any]) -> Ltx2VocoderConfig:
    if not isinstance(config, Mapping):
        raise TypeError(f"LTX2 vocoder config must be a mapping; got {type(config).__name__}.")

    required_keys = {
        "in_channels",
        "hidden_channels",
        "out_channels",
        "upsample_kernel_sizes",
        "upsample_factors",
        "resnet_kernel_sizes",
        "resnet_dilations",
        "leaky_relu_negative_slope",
        "output_sampling_rate",
    }
    allowed_keys = required_keys | set(_ALLOWED_CONFIG_METADATA_KEYS)
    raw_keys = {str(key) for key in config.keys()}
    missing = sorted(required_keys - raw_keys)
    unexpected = sorted(raw_keys - allowed_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Unsupported LTX2 vocoder config keys. "
            f"missing={missing!r} unexpected={unexpected!r}"
        )

    upsample_kernel_sizes = _require_positive_int_sequence(config, "upsample_kernel_sizes")
    upsample_factors = _require_positive_int_sequence(config, "upsample_factors")
    resnet_kernel_sizes = _require_positive_int_sequence(config, "resnet_kernel_sizes")
    raw_resnet_dilations = config.get("resnet_dilations")
    if not isinstance(raw_resnet_dilations, Sequence) or isinstance(raw_resnet_dilations, (str, bytes)):
        raise RuntimeError("LTX2 vocoder config key 'resnet_dilations' must be a sequence of int sequences.")
    resnet_dilations = tuple(_require_positive_int_tuple(entry, key="resnet_dilations") for entry in raw_resnet_dilations)

    parsed = Ltx2VocoderConfig(
        in_channels=_require_positive_int(config, "in_channels"),
        hidden_channels=_require_positive_int(config, "hidden_channels"),
        out_channels=_require_positive_int(config, "out_channels"),
        upsample_kernel_sizes=upsample_kernel_sizes,
        upsample_factors=upsample_factors,
        resnet_kernel_sizes=resnet_kernel_sizes,
        resnet_dilations=resnet_dilations,
        leaky_relu_negative_slope=_require_nonnegative_float(config, "leaky_relu_negative_slope"),
        output_sampling_rate=_require_positive_int(config, "output_sampling_rate"),
    )

    if len(parsed.upsample_kernel_sizes) != len(parsed.upsample_factors):
        raise RuntimeError(
            "Unsupported LTX2 vocoder config: `upsample_kernel_sizes` and `upsample_factors` must have equal length."
        )
    if len(parsed.resnet_kernel_sizes) != len(parsed.resnet_dilations):
        raise RuntimeError(
            "Unsupported LTX2 vocoder config: `resnet_kernel_sizes` and `resnet_dilations` must have equal length."
        )
    return parsed



def _parse_wrapped_23_vocoder_config(config: Mapping[str, Any]) -> _Wrapped23VocoderConfig:
    if not isinstance(config, Mapping):
        raise TypeError(f"Wrapped LTX2 2.3 vocoder config must be a mapping; got {type(config).__name__}.")

    required_keys = {"vocoder", "bwe"}
    raw_keys = {str(key) for key in config.keys()}
    missing = sorted(required_keys - raw_keys)
    unexpected = sorted(raw_keys - required_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Unsupported wrapped LTX2 2.3 vocoder config keys. "
            f"missing={missing!r} unexpected={unexpected!r}"
        )

    raw_vocoder_config = config.get("vocoder")
    raw_bwe_config = config.get("bwe")
    if not isinstance(raw_vocoder_config, Mapping):
        raise RuntimeError(
            "Wrapped LTX2 2.3 vocoder config field `vocoder` must be a mapping; "
            f"got {type(raw_vocoder_config).__name__}."
        )
    if not isinstance(raw_bwe_config, Mapping):
        raise RuntimeError(
            "Wrapped LTX2 2.3 vocoder config field `bwe` must be a mapping; "
            f"got {type(raw_bwe_config).__name__}."
        )

    base_generator = _parse_generator_config(
        {
            **_extract_wrapped_generator_config(raw_vocoder_config),
            "apply_final_activation": raw_vocoder_config.get("apply_final_activation", True),
        }
    )
    bwe_generator = _parse_generator_config(
        {
            **_extract_wrapped_generator_config(raw_bwe_config),
            "apply_final_activation": raw_bwe_config.get("apply_final_activation", False),
        }
    )

    return _Wrapped23VocoderConfig(
        vocoder=base_generator,
        bwe=bwe_generator,
        input_sampling_rate=_require_positive_int(raw_bwe_config, "input_sampling_rate"),
        output_sampling_rate=_require_positive_int(raw_bwe_config, "output_sampling_rate"),
        hop_length=_require_positive_int(raw_bwe_config, "hop_length"),
        n_fft=_require_positive_int(raw_bwe_config, "n_fft"),
        num_mels=_require_positive_int(raw_bwe_config, "num_mels"),
    )



def _parse_generator_config(config: Mapping[str, Any]) -> _GeneratorConfig:
    required_keys = {
        "resblock_kernel_sizes",
        "upsample_rates",
        "upsample_kernel_sizes",
        "resblock_dilation_sizes",
        "upsample_initial_channel",
        "stereo",
        "activation",
        "use_bias_at_final",
        "use_tanh_at_final",
        "apply_final_activation",
        "resblock",
    }
    optional_keys = {"output_sampling_rate"} | set(_ALLOWED_CONFIG_METADATA_KEYS)
    raw_keys = {str(key) for key in config.keys()}
    missing = sorted(required_keys - raw_keys)
    unexpected = sorted(raw_keys - (required_keys | optional_keys))
    if missing or unexpected:
        raise RuntimeError(
            "Unsupported wrapped LTX2 generator config keys. "
            f"missing={missing!r} unexpected={unexpected!r}"
        )

    resblock_kernel_sizes = _require_positive_int_sequence(config, "resblock_kernel_sizes")
    upsample_rates = _require_positive_int_sequence(config, "upsample_rates")
    upsample_kernel_sizes = _require_positive_int_sequence(config, "upsample_kernel_sizes")
    raw_dilations = config.get("resblock_dilation_sizes")
    if not isinstance(raw_dilations, Sequence) or isinstance(raw_dilations, (str, bytes)):
        raise RuntimeError(
            "Wrapped LTX2 generator config key 'resblock_dilation_sizes' must be a sequence of int sequences."
        )
    resblock_dilation_sizes = tuple(_require_positive_int_tuple(entry, key="resblock_dilation_sizes") for entry in raw_dilations)

    output_sampling_rate: int | None = None
    if "output_sampling_rate" in config and config.get("output_sampling_rate") is not None:
        output_sampling_rate = _require_positive_int(config, "output_sampling_rate")

    parsed = _GeneratorConfig(
        resblock_kernel_sizes=resblock_kernel_sizes,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_initial_channel=_require_positive_int(config, "upsample_initial_channel"),
        stereo=_require_bool(config, "stereo"),
        activation=_require_allowed_string(config, "activation", allowed={"snake", "snakebeta"}),
        use_bias_at_final=_require_bool(config, "use_bias_at_final"),
        use_tanh_at_final=_require_bool(config, "use_tanh_at_final"),
        apply_final_activation=_require_bool(config, "apply_final_activation"),
        resblock=_require_allowed_string(config, "resblock", allowed={"1", "2", "AMP1"}),
        output_sampling_rate=output_sampling_rate,
    )

    if len(parsed.upsample_rates) != len(parsed.upsample_kernel_sizes):
        raise RuntimeError(
            "Unsupported wrapped LTX2 generator config: `upsample_rates` and `upsample_kernel_sizes` must have equal length."
        )
    if len(parsed.resblock_kernel_sizes) != len(parsed.resblock_dilation_sizes):
        raise RuntimeError(
            "Unsupported wrapped LTX2 generator config: `resblock_kernel_sizes` and `resblock_dilation_sizes` must have equal length."
        )
    return parsed


def _extract_wrapped_generator_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in config.items() if str(key) in _WRAPPED_GENERATOR_CONFIG_KEYS}



def _validate_raw_vocoder_state_dict(state_dict: Mapping[str, Any]) -> None:
    raw_keys = tuple(str(key) for key in state_dict.keys())
    if not raw_keys:
        raise RuntimeError("LTX2 vocoder state_dict is empty.")
    if any(key.startswith(("model.diffusion_model.", "vae.", "audio_vae.")) for key in raw_keys):
        raise RuntimeError("LTX2 vocoder loader received non-vocoder component keys.")
    if any(key.startswith(_REJECTED_VOCODER_PREFIXES) for key in raw_keys):
        raise RuntimeError(
            "Unsupported Diffusers-remapped LTX2 vocoder keyspace. "
            "The native loader expects raw `conv_pre`/`ups`/`resblocks`/`conv_post` keys only."
        )
    if any(key.startswith(_WRAPPED_23_VOCODER_PREFIXES) for key in raw_keys):
        raise RuntimeError(
            "Mixed LTX2 vocoder layout detected. Wrapped 2.3 keys reached the legacy flat loader path."
        )

    required = ("conv_pre.weight", "conv_post.weight")
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise RuntimeError(f"LTX2 vocoder state_dict is missing required raw keys: {missing!r}.")

    unexpected_prefixes = sorted({key.split(".", 1)[0] for key in raw_keys if key.split(".", 1)[0] not in _RAW_VOCODER_TOP_LEVEL_PREFIXES})
    if unexpected_prefixes:
        raise RuntimeError(
            "Unsupported LTX2 vocoder layout. "
            f"Unexpected top-level prefixes: {unexpected_prefixes!r}."
        )



def _validate_wrapped_23_vocoder_state_dict(state_dict: Mapping[str, Any]) -> None:
    raw_keys = tuple(str(key) for key in state_dict.keys())
    if not raw_keys:
        raise RuntimeError("Wrapped LTX2 2.3 vocoder state_dict is empty.")
    if any(key.startswith(("model.diffusion_model.", "vae.", "audio_vae.")) for key in raw_keys):
        raise RuntimeError("Wrapped LTX2 2.3 vocoder loader received non-vocoder component keys.")
    if any(key.startswith(_REJECTED_VOCODER_PREFIXES) for key in raw_keys):
        raise RuntimeError(
            "Unsupported Diffusers-remapped LTX2 vocoder keyspace. "
            "The native loader expects raw stored keys only."
        )
    if any(key.startswith(_RAW_VOCODER_PREFIXES) for key in raw_keys):
        raise RuntimeError(
            "Mixed LTX2 vocoder layout detected. Flat legacy keys reached the wrapped 2.3 loader path."
        )

    missing_groups = [
        prefix for prefix in _WRAPPED_23_VOCODER_TOP_LEVEL_PREFIXES if not any(key.startswith(f"{prefix}.") for key in raw_keys)
    ]
    if missing_groups:
        raise RuntimeError(
            "Wrapped LTX2 2.3 vocoder state_dict is missing required top-level groups: "
            f"{missing_groups!r}."
        )

    required_keys = (
        "vocoder.conv_pre.weight",
        "vocoder.conv_post.weight",
        "bwe_generator.conv_pre.weight",
        "bwe_generator.conv_post.weight",
        "mel_stft.mel_basis",
        "mel_stft.stft_fn.forward_basis",
        "mel_stft.stft_fn.inverse_basis",
    )
    missing_required = [key for key in required_keys if key not in state_dict]
    if missing_required:
        raise RuntimeError(
            "Wrapped LTX2 2.3 vocoder state_dict is missing required keys: "
            f"{missing_required!r}."
        )

    unexpected_prefixes = sorted(
        {
            key.split(".", 1)[0]
            for key in raw_keys
            if key.split(".", 1)[0] not in _WRAPPED_23_VOCODER_TOP_LEVEL_PREFIXES
        }
    )
    if unexpected_prefixes:
        raise RuntimeError(
            "Unsupported wrapped LTX2 2.3 vocoder layout. "
            f"Unexpected top-level prefixes: {unexpected_prefixes!r}."
        )



def _is_wrapped_23_vocoder_state_dict(state_dict: Mapping[str, Any]) -> bool:
    return any(str(key).startswith(_WRAPPED_23_VOCODER_PREFIXES) for key in state_dict.keys())



def _cast_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return tensor.to(device=reference.device, dtype=reference.dtype)



def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((int(kernel_size) * int(dilation) - int(dilation)) / 2)



def _sinc(values: torch.Tensor) -> torch.Tensor:
    return torch.where(values == 0, torch.ones_like(values), torch.sin(math.pi * values) / (math.pi * values))



def _kaiser_sinc_filter1d(*, cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    even = int(kernel_size) % 2 == 0
    half_size = int(kernel_size) // 2
    delta_f = 4 * float(half_width)
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21.0) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(int(kernel_size), beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size, dtype=torch.float32) + 0.5
    else:
        time = torch.arange(int(kernel_size), dtype=torch.float32) - half_size
    if float(cutoff) == 0.0:
        filter_tensor = torch.zeros_like(time)
    else:
        filter_tensor = 2 * float(cutoff) * window * _sinc(2 * float(cutoff) * time)
        filter_tensor = filter_tensor / filter_tensor.sum()
    return filter_tensor.view(1, 1, int(kernel_size))



def _require_positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must be an int; got {type(value).__name__}.")
    if value <= 0:
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must be > 0; got {value!r}.")
    return int(value)



def _require_nonnegative_float(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(
            f"LTX2 vocoder config key {key!r} must be a float-compatible number; got {type(value).__name__}."
        )
    value = float(value)
    if value < 0.0:
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must be >= 0; got {value!r}.")
    return value



def _require_positive_int_sequence(config: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = config.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must be a sequence of ints.")
    parsed = tuple(_coerce_positive_int(entry, key=key) for entry in value)
    if not parsed:
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must not be empty.")
    return parsed



def _require_positive_int_tuple(value: object, *, key: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(f"LTX2 vocoder config key {key!r} entries must be int sequences.")
    parsed = tuple(_coerce_positive_int(entry, key=key) for entry in value)
    if not parsed:
        raise RuntimeError(f"LTX2 vocoder config key {key!r} entries must not be empty.")
    return parsed



def _coerce_positive_int(value: object, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must contain ints; got {type(value).__name__}.")
    if value <= 0:
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must contain values > 0; got {value!r}.")
    return int(value)



def _require_bool(config: Mapping[str, Any], key: str) -> bool:
    value = config.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must be a bool; got {type(value).__name__}.")
    return bool(value)



def _require_allowed_string(config: Mapping[str, Any], key: str, *, allowed: set[str]) -> str:
    value = config.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must be a string; got {type(value).__name__}.")
    if value not in allowed:
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must be one of {sorted(allowed)!r}; got {value!r}.")
    return value


__all__ = [
    "Ltx2Vocoder",
    "Ltx2VocoderConfig",
    "load_ltx2_vocoder",
]
