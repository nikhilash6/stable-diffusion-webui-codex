"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public types for the GGUF converter tool.
Defines the conversion config, public file recipe enum, physical tensor target enum, quant-policy preset enum, progress tracking, and verification error type.

Symbols (top-level; keep in sync; no ghosts):
- `QuantizationRecipe` (enum): Public file-level GGUF recipe selector for conversion outputs.
- `TensorQuantizationType` (enum): Physical per-tensor GGML target type used by rules and advanced overrides.
- `QuantPolicyPreset` (enum): Policy preset controlling optional profile quality rules (`HQ|MQ|LQ`).
- `normalize_quantization_recipe` (function): Normalizes and validates a public recipe selector.
- `normalize_tensor_quantization_type` (function): Normalizes and validates a physical tensor target selector.
- `normalize_quant_policy_preset` (function): Normalizes and validates a required quant-policy preset selector.
- `normalize_optional_quant_policy_preset` (function): Normalizes an optional quant-policy preset selector.
- `ConversionConfig` (dataclass): Conversion configuration (paths, profile selection, recipe, optional policy preset, and regex tensor overrides).
- `ConversionProgress` (dataclass): Progress/report structure for long conversions (stage counters, timings, and status fields).
- `GGUFVerificationError` (exception): Raised when a written GGUF file fails validation/verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence


class QuantizationRecipe(str, Enum):
    """Public file-level GGUF conversion recipes."""

    F16 = "F16"
    F32 = "F32"
    Q8_0 = "Q8_0"
    Q6_K = "Q6_K"
    Q5_K_M = "Q5_K_M"
    Q5_K_S = "Q5_K_S"
    Q4_K_M = "Q4_K_M"
    Q4_K_S = "Q4_K_S"
    Q3_K_L = "Q3_K_L"
    Q3_K_M = "Q3_K_M"
    Q3_K_S = "Q3_K_S"
    Q2_K = "Q2_K"
    Q2_K_S = "Q2_K_S"
    Q5_1 = "Q5_1"
    Q5_0 = "Q5_0"
    Q4_1 = "Q4_1"
    Q4_0 = "Q4_0"
    IQ4_NL = "IQ4_NL"


class TensorQuantizationType(str, Enum):
    """Physical per-tensor GGML target types for policy rules and user overrides."""

    F16 = "F16"
    F32 = "F32"
    Q8_0 = "Q8_0"
    Q6_K = "Q6_K"
    Q5_K = "Q5_K"
    Q5_1 = "Q5_1"
    Q5_0 = "Q5_0"
    Q4_K = "Q4_K"
    Q4_1 = "Q4_1"
    Q4_0 = "Q4_0"
    Q3_K = "Q3_K"
    Q2_K = "Q2_K"
    IQ4_NL = "IQ4_NL"


class QuantPolicyPreset(str, Enum):
    """Optional per-profile quality policy preset."""

    HQ = "HQ"
    MQ = "MQ"
    LQ = "LQ"


def _normalize_enum_value(value: object, enum_type: type[Enum], *, field: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    raw = str(value or "").strip().upper()
    try:
        return enum_type(raw)
    except ValueError as exc:
        allowed = ", ".join(str(member.value) for member in enum_type)
        raise ValueError(f"Invalid {field} {value!r}; expected one of: {allowed}") from exc


def normalize_quantization_recipe(value: object) -> QuantizationRecipe:
    """Normalize a public file-level recipe selector."""

    return _normalize_enum_value(value, QuantizationRecipe, field="quantization")  # type: ignore[return-value]


def normalize_tensor_quantization_type(value: object) -> TensorQuantizationType:
    """Normalize a physical per-tensor target selector."""

    return _normalize_enum_value(value, TensorQuantizationType, field="tensor quantization type")  # type: ignore[return-value]


def normalize_quant_policy_preset(value: object) -> QuantPolicyPreset:
    """Normalize a required quant-policy preset selector."""

    if value is None:
        allowed = ", ".join(preset.value for preset in QuantPolicyPreset)
        raise ValueError(f"Invalid quant_policy_preset {value!r}; expected one of: {allowed}")
    return _normalize_enum_value(value, QuantPolicyPreset, field="quant_policy_preset")  # type: ignore[return-value]


def normalize_optional_quant_policy_preset(value: object) -> Optional[QuantPolicyPreset]:
    """Normalize an optional quant-policy preset selector."""

    if value is None:
        return None
    if isinstance(value, QuantPolicyPreset):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    return normalize_quant_policy_preset(raw)


@dataclass(slots=True)
class ConversionConfig:
    """Configuration for GGUF conversion."""

    config_path: str  # Path to config.json or folder containing it
    safetensors_path: str  # Path to .safetensors file, index, or directory
    output_path: str  # Output .gguf path
    profile_id: Optional[str] = None
    quantization: QuantizationRecipe = QuantizationRecipe.F16
    quant_policy_preset: Optional[QuantPolicyPreset] = None
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
    "QuantPolicyPreset",
    "QuantizationRecipe",
    "TensorQuantizationType",
    "normalize_optional_quant_policy_preset",
    "normalize_quant_policy_preset",
    "normalize_quantization_recipe",
    "normalize_tensor_quantization_type",
]
