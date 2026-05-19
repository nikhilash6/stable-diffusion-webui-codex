"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public types for the GGUF converter tool.
Defines the conversion config, quantization selector enum, progress tracking, and verification error type.

Symbols (top-level; keep in sync; no ghosts):
- `QuantizationType` (enum): Supported “human” quantization selectors for conversion (maps to `GGMLQuantizationType`).
- `ConversionConfig` (dataclass): Conversion configuration (paths, profile selection, quantization, and regex dtype overrides).
- `ConversionProgress` (dataclass): Progress/report structure for long conversions (stage counters, timings, and status fields).
- `GGUFVerificationError` (exception): Raised when a written GGUF file fails validation/verification.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(slots=True)
class ConversionConfig:
    """Configuration for GGUF conversion."""

    config_path: str  # Path to config.json or folder containing it
    safetensors_path: str  # Path to .safetensors file
    output_path: str  # Output .gguf path
    profile_id: Optional[str] = None
    quantization: QuantizationType = QuantizationType.F16
    tensor_type_overrides: Sequence[str] = ()


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
    "QuantizationType",
]
