"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public types for the GGUF converter tool.
Defines the conversion config, quantization selector enum, precision controls, progress tracking, and verification error type.

Symbols (top-level; keep in sync; no ghosts):
- `QuantizationType` (enum): Supported “human” quantization selectors for conversion (maps to `GGMLQuantizationType`).
- `PrecisionMode` (enum): Mixed-quant float precision policy selector (`full_*` and `*_plus_fp32` modes).
- `MIXED_FLOAT_OVERRIDE_VALUES` (tuple): Canonical allowed literals for mixed float override values.
- `PRECISION_MODE_VALUES` (tuple): Canonical allowed literals for precision-mode selectors.
- `normalize_mixed_float_override` (function): Normalizes and validates a mixed float override literal (`auto|F16|BF16|F32`).
- `normalize_precision_mode` (function): Normalizes and validates a precision-mode selector.
- `ConversionConfig` (dataclass): Conversion configuration (paths, profile selection, quantization, and dtype override knobs).
- `ConversionProgress` (dataclass): Progress/report structure for long conversions (stage counters, timings, and status fields).
- `GGUFVerificationError` (exception): Raised when a written GGUF file fails validation/verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence


class QuantizationType(str, Enum):
    """Supported GGUF quantization types."""

    F16 = "F16"
    F32 = "F32"
    Q8_0 = "Q8_0"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q5_K = "Q5_K"
    Q5_1 = "Q5_1"
    Q5_0 = "Q5_0"
    Q4_K_M = "Q4_K_M"
    Q4_K = "Q4_K"
    Q4_1 = "Q4_1"
    Q4_0 = "Q4_0"
    Q3_K = "Q3_K"
    Q2_K = "Q2_K"
    IQ4_NL = "IQ4_NL"


class PrecisionMode(str, Enum):
    """Mixed-quant float precision mode selector."""

    FULL_BF16 = "FULL_BF16"
    FULL_FP16 = "FULL_FP16"
    FULL_FP32 = "FULL_FP32"
    FP16_PLUS_FP32 = "FP16_PLUS_FP32"
    BF16_PLUS_FP32 = "BF16_PLUS_FP32"


MIXED_FLOAT_OVERRIDE_VALUES: tuple[str, str, str, str] = ("auto", "F16", "BF16", "F32")
PRECISION_MODE_VALUES: tuple[str, str, str, str, str] = (
    PrecisionMode.FULL_BF16.value,
    PrecisionMode.FULL_FP16.value,
    PrecisionMode.FULL_FP32.value,
    PrecisionMode.FP16_PLUS_FP32.value,
    PrecisionMode.BF16_PLUS_FP32.value,
)


def normalize_mixed_float_override(value: object) -> str:
    if value is None:
        return "auto"
    if not isinstance(value, str):
        raise ValueError(
            f"Invalid float dtype selection: {value!r} (expected {'|'.join(MIXED_FLOAT_OVERRIDE_VALUES)})"
        )
    raw = value.strip().upper()
    if raw in {"", "AUTO"}:
        return "auto"
    if raw in {"F16", "BF16", "F32"}:
        return raw
    raise ValueError(
        f"Invalid float dtype selection: {value!r} (expected {'|'.join(MIXED_FLOAT_OVERRIDE_VALUES)})"
    )


def normalize_precision_mode(value: object) -> PrecisionMode | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"Invalid precision_mode: {value!r} (expected {'|'.join(PRECISION_MODE_VALUES)})"
        )
    raw = value.strip().upper()
    if not raw:
        return None
    aliases = {
        "FULL BF16": PrecisionMode.FULL_BF16,
        "FULL BFLOAT16": PrecisionMode.FULL_BF16,
        "FULL_FP16": PrecisionMode.FULL_FP16,
        "FULL FP16": PrecisionMode.FULL_FP16,
        "FULL_FP32": PrecisionMode.FULL_FP32,
        "FULL FP32": PrecisionMode.FULL_FP32,
        "FP16+FP32": PrecisionMode.FP16_PLUS_FP32,
        "BF16+FP32": PrecisionMode.BF16_PLUS_FP32,
    }
    if raw in aliases:
        return aliases[raw]
    try:
        return PrecisionMode(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid precision_mode: {value!r} (expected {'|'.join(PRECISION_MODE_VALUES)})"
        ) from exc


@dataclass(slots=True)
class ConversionConfig:
    """Configuration for GGUF conversion."""

    config_path: str  # Path to config.json or folder containing it
    safetensors_path: str  # Path to .safetensors file
    output_path: str  # Output .gguf path
    profile_id: Optional[str] = None
    quantization: QuantizationType = QuantizationType.F16
    tensor_type_overrides: Sequence[str] = ()
    float_group_overrides: dict[str, str] = field(default_factory=dict)
    precision_mode: PrecisionMode | None = None


@dataclass(slots=True)
class ConversionProgress:
    """Progress tracking for conversion."""

    current_step: int = 0
    total_steps: int = 0
    current_tensor: str = ""
    status: str = "idle"
    error: Optional[str] = None

    @property
    def progress_percent(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100.0


class GGUFVerificationError(Exception):
    """Raised when GGUF file verification fails."""


__all__ = [
    "ConversionConfig",
    "ConversionProgress",
    "GGUFVerificationError",
    "MIXED_FLOAT_OVERRIDE_VALUES",
    "PRECISION_MODE_VALUES",
    "PrecisionMode",
    "QuantizationType",
    "normalize_mixed_float_override",
    "normalize_precision_mode",
]
