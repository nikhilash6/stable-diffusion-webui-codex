"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime tools facade exposing heavyweight offline-style utilities (GGUF conversion and SafeTensors merge).
Re-exports public runtime tool APIs used by `/api/tools/*` and CLI-like tooling.

Symbols (top-level; keep in sync; no ghosts):
- `ConversionConfig` (class): Conversion configuration for SafeTensors → GGUF (inputs, outputs, profile/policy, quantization).
- `ConversionProgress` (class): Progress callback payload emitted by the converter.
- `ConversionPreflight` (class): Resolved GGUF conversion contract produced by preflight validation.
- `QuantPolicyPreset` (class): Quant policy preset enum used by the converter.
- `QuantizationRecipe` (class): Public GGUF file-recipe enum used by the converter.
- `TensorQuantizationType` (class): Physical tensor target enum used by advanced overrides.
- `convert_safetensors_to_gguf` (function): Convert SafeTensors weights (including sharded indexes) to GGUF.
- `preflight_conversion_contract` (function): Validate profile/recipe/policy before heavy tensor IO.
- `SafetensorsMergeConfig` (class): Merge configuration for collapsing safetensors sources into one file.
- `SafetensorsMergeProgress` (class): Progress callback payload emitted by the safetensors merge tool.
- `merge_safetensors_source` (function): Merge a safetensors source (file/index/dir) into one `.safetensors` file.
- `__all__` (constant): Export list for the tools facade.
"""

from .gguf_converter import (
    ConversionConfig,
    ConversionPreflight,
    ConversionProgress,
    QuantPolicyPreset,
    QuantizationRecipe,
    convert_safetensors_to_gguf,
    preflight_conversion_contract,
)
from .gguf_converter_types import TensorQuantizationType
from .safetensors_merge import SafetensorsMergeConfig, SafetensorsMergeProgress, merge_safetensors_source

__all__ = [
    "ConversionConfig",
    "ConversionPreflight",
    "ConversionProgress",
    "QuantPolicyPreset",
    "QuantizationRecipe",
    "SafetensorsMergeConfig",
    "SafetensorsMergeProgress",
    "TensorQuantizationType",
    "convert_safetensors_to_gguf",
    "merge_safetensors_source",
    "preflight_conversion_contract",
]
