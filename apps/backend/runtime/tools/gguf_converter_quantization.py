"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Quantization recipe and tensor-target helpers for the GGUF converter.
Maps public file-level recipes to GGUF file metadata/default physical GGML types, maps physical tensor targets for policy rules, and enforces generic per-tensor shape/block-size compatibility rules.

Symbols (top-level; keep in sync; no ghosts):
- `QuantizationRecipeSpec` (dataclass): Canonical descriptor for one public file-level recipe.
- `all_quantization_recipe_specs` (function): Returns descriptors for every public recipe.
- `all_tensor_quantization_type_specs` (function): Returns descriptors for physical override target names.
- `recipe_spec` (function): Resolves a public recipe descriptor.
- `recipe_default_ggml_type` (function): Returns the default physical GGML type for a file recipe.
- `tensor_quantization_type_to_ggml_type` (function): Maps a physical tensor target to `GGMLQuantizationType`.
- `generated_rule_is_downgrade` (function): Checks generated recipe/profile rule targets against the selected recipe baseline.
- `select_tensor_ggml_type` (function): Selects the effective GGML type for a tensor given shape and requested type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from apps.backend.quantization.gguf import GGML_QUANT_SIZES, GGMLQuantizationType, LlamaFileType
from apps.backend.quantization.gguf.constants import GGML_QUANT_VERSION
from apps.backend.runtime.tools.gguf_converter_types import QuantizationRecipe, TensorQuantizationType


@dataclass(frozen=True, slots=True)
class QuantizationRecipeSpec:
    recipe: QuantizationRecipe
    label: str
    group: str
    description: str
    default_tensor_type: TensorQuantizationType
    ggml_type: GGMLQuantizationType
    llama_file_type: LlamaFileType
    quantization_version: int
    family: str
    tier: str
    requires_recipe_intrinsics: bool = False

    @property
    def is_float(self) -> bool:
        return self.recipe in {QuantizationRecipe.F16, QuantizationRecipe.F32}


_TENSOR_TYPE_TO_GGML: dict[TensorQuantizationType, GGMLQuantizationType] = {
    TensorQuantizationType.F16: GGMLQuantizationType.F16,
    TensorQuantizationType.F32: GGMLQuantizationType.F32,
    TensorQuantizationType.Q8_0: GGMLQuantizationType.Q8_0,
    TensorQuantizationType.Q6_K: GGMLQuantizationType.Q6_K,
    TensorQuantizationType.Q5_K: GGMLQuantizationType.Q5_K,
    TensorQuantizationType.Q5_1: GGMLQuantizationType.Q5_1,
    TensorQuantizationType.Q5_0: GGMLQuantizationType.Q5_0,
    TensorQuantizationType.Q4_K: GGMLQuantizationType.Q4_K,
    TensorQuantizationType.Q4_1: GGMLQuantizationType.Q4_1,
    TensorQuantizationType.Q4_0: GGMLQuantizationType.Q4_0,
    TensorQuantizationType.Q3_K: GGMLQuantizationType.Q3_K,
    TensorQuantizationType.Q2_K: GGMLQuantizationType.Q2_K,
    TensorQuantizationType.IQ4_NL: GGMLQuantizationType.IQ4_NL,
}


def _spec(
    recipe: QuantizationRecipe,
    *,
    label: str,
    group: str,
    description: str,
    default_tensor_type: TensorQuantizationType,
    llama_file_type: LlamaFileType,
    family: str,
    tier: str,
    requires_recipe_intrinsics: bool = False,
) -> QuantizationRecipeSpec:
    return QuantizationRecipeSpec(
        recipe=recipe,
        label=label,
        group=group,
        description=description,
        default_tensor_type=default_tensor_type,
        ggml_type=_TENSOR_TYPE_TO_GGML[default_tensor_type],
        llama_file_type=llama_file_type,
        quantization_version=GGML_QUANT_VERSION,
        family=family,
        tier=tier,
        requires_recipe_intrinsics=requires_recipe_intrinsics,
    )


_RECIPE_SPECS: dict[QuantizationRecipe, QuantizationRecipeSpec] = {
    QuantizationRecipe.F32: _spec(
        QuantizationRecipe.F32,
        label="F32",
        group="Float",
        description="Full float32 output; no profile policy.",
        default_tensor_type=TensorQuantizationType.F32,
        llama_file_type=LlamaFileType.ALL_F32,
        family="float",
        tier="float",
    ),
    QuantizationRecipe.F16: _spec(
        QuantizationRecipe.F16,
        label="F16",
        group="Float",
        description="Mostly float16 output; no profile policy.",
        default_tensor_type=TensorQuantizationType.F16,
        llama_file_type=LlamaFileType.MOSTLY_F16,
        family="float",
        tier="float",
    ),
    QuantizationRecipe.Q8_0: _spec(
        QuantizationRecipe.Q8_0,
        label="Q8_0",
        group="K-quants",
        description="8-bit single physical recipe.",
        default_tensor_type=TensorQuantizationType.Q8_0,
        llama_file_type=LlamaFileType.MOSTLY_Q8_0,
        family="k",
        tier="8",
    ),
    QuantizationRecipe.Q6_K: _spec(
        QuantizationRecipe.Q6_K,
        label="Q6_K",
        group="K-quants",
        description="6-bit K recipe.",
        default_tensor_type=TensorQuantizationType.Q6_K,
        llama_file_type=LlamaFileType.MOSTLY_Q6_K,
        family="k",
        tier="6",
    ),
    QuantizationRecipe.Q5_K_M: _spec(
        QuantizationRecipe.Q5_K_M,
        label="Q5_K_M",
        group="K-quants",
        description="5-bit K medium file recipe.",
        default_tensor_type=TensorQuantizationType.Q5_K,
        llama_file_type=LlamaFileType.MOSTLY_Q5_K_M,
        family="k",
        tier="5_m",
        requires_recipe_intrinsics=True,
    ),
    QuantizationRecipe.Q5_K_S: _spec(
        QuantizationRecipe.Q5_K_S,
        label="Q5_K_S",
        group="K-quants",
        description="5-bit K small file recipe.",
        default_tensor_type=TensorQuantizationType.Q5_K,
        llama_file_type=LlamaFileType.MOSTLY_Q5_K_S,
        family="k",
        tier="5_s",
    ),
    QuantizationRecipe.Q4_K_M: _spec(
        QuantizationRecipe.Q4_K_M,
        label="Q4_K_M",
        group="K-quants",
        description="4-bit K medium file recipe.",
        default_tensor_type=TensorQuantizationType.Q4_K,
        llama_file_type=LlamaFileType.MOSTLY_Q4_K_M,
        family="k",
        tier="4_m",
        requires_recipe_intrinsics=True,
    ),
    QuantizationRecipe.Q4_K_S: _spec(
        QuantizationRecipe.Q4_K_S,
        label="Q4_K_S",
        group="K-quants",
        description="4-bit K small file recipe.",
        default_tensor_type=TensorQuantizationType.Q4_K,
        llama_file_type=LlamaFileType.MOSTLY_Q4_K_S,
        family="k",
        tier="4_s",
    ),
    QuantizationRecipe.Q3_K_L: _spec(
        QuantizationRecipe.Q3_K_L,
        label="Q3_K_L",
        group="K-quants",
        description="3-bit K large file recipe.",
        default_tensor_type=TensorQuantizationType.Q3_K,
        llama_file_type=LlamaFileType.MOSTLY_Q3_K_L,
        family="k",
        tier="3_l",
        requires_recipe_intrinsics=True,
    ),
    QuantizationRecipe.Q3_K_M: _spec(
        QuantizationRecipe.Q3_K_M,
        label="Q3_K_M",
        group="K-quants",
        description="3-bit K medium file recipe.",
        default_tensor_type=TensorQuantizationType.Q3_K,
        llama_file_type=LlamaFileType.MOSTLY_Q3_K_M,
        family="k",
        tier="3_m",
        requires_recipe_intrinsics=True,
    ),
    QuantizationRecipe.Q3_K_S: _spec(
        QuantizationRecipe.Q3_K_S,
        label="Q3_K_S",
        group="K-quants",
        description="3-bit K small file recipe.",
        default_tensor_type=TensorQuantizationType.Q3_K,
        llama_file_type=LlamaFileType.MOSTLY_Q3_K_S,
        family="k",
        tier="3_s",
    ),
    QuantizationRecipe.Q2_K: _spec(
        QuantizationRecipe.Q2_K,
        label="Q2_K",
        group="K-quants",
        description="2-bit K recipe.",
        default_tensor_type=TensorQuantizationType.Q2_K,
        llama_file_type=LlamaFileType.MOSTLY_Q2_K,
        family="k",
        tier="2",
        requires_recipe_intrinsics=True,
    ),
    QuantizationRecipe.Q2_K_S: _spec(
        QuantizationRecipe.Q2_K_S,
        label="Q2_K_S",
        group="K-quants",
        description="2-bit K small file recipe.",
        default_tensor_type=TensorQuantizationType.Q2_K,
        llama_file_type=LlamaFileType.MOSTLY_Q2_K_S,
        family="k",
        tier="2_s",
    ),
    QuantizationRecipe.Q5_1: _spec(
        QuantizationRecipe.Q5_1,
        label="Q5_1",
        group="Legacy",
        description="Legacy 5-bit type-1 physical recipe.",
        default_tensor_type=TensorQuantizationType.Q5_1,
        llama_file_type=LlamaFileType.MOSTLY_Q5_1,
        family="legacy",
        tier="5_1",
    ),
    QuantizationRecipe.Q5_0: _spec(
        QuantizationRecipe.Q5_0,
        label="Q5_0",
        group="Legacy",
        description="Legacy 5-bit type-0 physical recipe.",
        default_tensor_type=TensorQuantizationType.Q5_0,
        llama_file_type=LlamaFileType.MOSTLY_Q5_0,
        family="legacy",
        tier="5_0",
    ),
    QuantizationRecipe.Q4_1: _spec(
        QuantizationRecipe.Q4_1,
        label="Q4_1",
        group="Legacy",
        description="Legacy 4-bit type-1 physical recipe.",
        default_tensor_type=TensorQuantizationType.Q4_1,
        llama_file_type=LlamaFileType.MOSTLY_Q4_1,
        family="legacy",
        tier="4_1",
    ),
    QuantizationRecipe.Q4_0: _spec(
        QuantizationRecipe.Q4_0,
        label="Q4_0",
        group="Legacy",
        description="Legacy 4-bit type-0 physical recipe.",
        default_tensor_type=TensorQuantizationType.Q4_0,
        llama_file_type=LlamaFileType.MOSTLY_Q4_0,
        family="legacy",
        tier="4_0",
    ),
    QuantizationRecipe.IQ4_NL: _spec(
        QuantizationRecipe.IQ4_NL,
        label="IQ4_NL",
        group="Experimental",
        description="Experimental 4-bit non-linear IQ recipe.",
        default_tensor_type=TensorQuantizationType.IQ4_NL,
        llama_file_type=LlamaFileType.MOSTLY_IQ4_NL,
        family="iq",
        tier="4_nl",
    ),
}

_K_RANKS: dict[GGMLQuantizationType, int] = {
    GGMLQuantizationType.Q2_K: 2,
    GGMLQuantizationType.Q3_K: 3,
    GGMLQuantizationType.Q4_K: 4,
    GGMLQuantizationType.Q5_K: 5,
    GGMLQuantizationType.Q6_K: 6,
    GGMLQuantizationType.Q8_0: 8,
}

_FLOAT_TYPES = {GGMLQuantizationType.F16, GGMLQuantizationType.BF16, GGMLQuantizationType.F32}


def all_quantization_recipe_specs() -> tuple[QuantizationRecipeSpec, ...]:
    return tuple(_RECIPE_SPECS[recipe] for recipe in QuantizationRecipe)


def all_tensor_quantization_type_specs() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "id": tensor_type.value,
            "label": tensor_type.value,
            "ggml_type": tensor_quantization_type_to_ggml_type(tensor_type).name,
        }
        for tensor_type in TensorQuantizationType
    )


def recipe_spec(recipe: QuantizationRecipe) -> QuantizationRecipeSpec:
    return _RECIPE_SPECS[recipe]


def recipe_default_ggml_type(recipe: QuantizationRecipe) -> GGMLQuantizationType:
    return recipe_spec(recipe).ggml_type


def tensor_quantization_type_to_ggml_type(tensor_type: TensorQuantizationType) -> GGMLQuantizationType:
    try:
        return _TENSOR_TYPE_TO_GGML[tensor_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor quantization type: {tensor_type}") from exc


def ggml_type_label(ggml_type: GGMLQuantizationType) -> str:
    return ggml_type.name


def generated_rule_is_downgrade(recipe: QuantizationRecipe, target: GGMLQuantizationType) -> bool:
    """Return True when a generated rule lowers a comparable K-family recipe baseline."""

    baseline = recipe_default_ggml_type(recipe)
    if target in _FLOAT_TYPES:
        return False
    if baseline not in _K_RANKS or target not in _K_RANKS:
        return False
    return _K_RANKS[target] < _K_RANKS[baseline]


def select_tensor_ggml_type(shape: Sequence[int], requested: GGMLQuantizationType) -> GGMLQuantizationType:
    """Select the per-tensor GGML type.

    Behavior:
    - If requested is F16/BF16/F32: apply to all tensors.
    - Otherwise: keep 1D tensors in F16 and only quantize tensors whose last dim
      is divisible by the block size.
    """

    if requested in _FLOAT_TYPES:
        return requested

    # Common GGUF convention: keep 1D tensors in F16.
    if len(shape) <= 1:
        return GGMLQuantizationType.F16

    block_size, _ = GGML_QUANT_SIZES[requested]
    if shape[-1] % block_size != 0:
        return GGMLQuantizationType.F16

    return requested


__all__ = [
    "QuantizationRecipeSpec",
    "all_quantization_recipe_specs",
    "all_tensor_quantization_type_specs",
    "generated_rule_is_downgrade",
    "ggml_type_label",
    "recipe_default_ggml_type",
    "recipe_spec",
    "select_tensor_ggml_type",
    "tensor_quantization_type_to_ggml_type",
]
