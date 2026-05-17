"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen Image Qwen2.5-VL text-encoder metadata and prompt-template helpers.
Validates the lightweight Qwen2.5-VL config contract and builds prompt-template plans for internal Qwen Image variants
without importing `transformers` or loading text-encoder weights.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImagePromptPlan` (dataclass): Rendered prompt plus template-drop/max-sequence metadata for text encoding.
- `QwenImageTextEncoderConfig` (dataclass): Strict Qwen2.5-VL text-encoder metadata contract.
- `QwenImageVisionConfig` (dataclass): Strict Qwen2.5-VL visual-tower metadata contract.
- `qwen_image_prompt_plan` (function): Render a prompt through the variant-owned Qwen Image template.
- `qwen_image_text_encoder_config_from_mapping` (function): Validate and convert a text-encoder config mapping.
- `qwen_image_validate_max_sequence_length` (function): Enforce Qwen Image tokenizer max sequence length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .config import QWEN_IMAGE_CONTEXT_DIM, QWEN_IMAGE_TOKENIZER_MAX_LENGTH, qwen_image_variant_spec


@dataclass(frozen=True, slots=True)
class QwenImageVisionConfig:
    model_type: str
    out_hidden_size: int
    patch_size: int

    def __post_init__(self) -> None:
        if self.model_type != "qwen2_5_vl":
            raise ValueError("Qwen Image vision config model_type must be qwen2_5_vl")
        if self.out_hidden_size != QWEN_IMAGE_CONTEXT_DIM:
            raise ValueError(f"Qwen Image vision out_hidden_size must be {QWEN_IMAGE_CONTEXT_DIM}")
        if self.patch_size != 14:
            raise ValueError("Qwen Image vision patch_size must be 14")


@dataclass(frozen=True, slots=True)
class QwenImageTextEncoderConfig:
    model_type: str
    hidden_size: int
    vision: QwenImageVisionConfig

    def __post_init__(self) -> None:
        if self.model_type != "qwen2_5_vl":
            raise ValueError("Qwen Image text encoder model_type must be qwen2_5_vl")
        if self.hidden_size != QWEN_IMAGE_CONTEXT_DIM:
            raise ValueError(f"Qwen Image text encoder hidden_size must be {QWEN_IMAGE_CONTEXT_DIM}")
        if self.vision.out_hidden_size != self.hidden_size:
            raise ValueError("Qwen Image vision projection must match text encoder hidden_size")


@dataclass(frozen=True, slots=True)
class QwenImagePromptPlan:
    prompt: str
    rendered_prompt: str
    template_start_idx: int
    max_sequence_length: int

    def __post_init__(self) -> None:
        if self.template_start_idx < 0:
            raise ValueError("Qwen Image prompt template_start_idx must be non-negative")
        qwen_image_validate_max_sequence_length(self.max_sequence_length)


def _require_mapping(value: object, *, field: str, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context}: {field} must be an object.")
    return value


def qwen_image_text_encoder_config_from_mapping(
    config: Mapping[str, object],
    *,
    context: str = "Qwen Image text encoder metadata",
) -> QwenImageTextEncoderConfig:
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{context}: text encoder config must be a mapping.")
    vision_config = _require_mapping(config.get("vision_config"), field="vision_config", context=context)
    try:
        return QwenImageTextEncoderConfig(
            model_type=str(config.get("model_type") or "").strip(),
            hidden_size=int(config.get("hidden_size") or 0),
            vision=QwenImageVisionConfig(
                model_type=str(vision_config.get("model_type") or "").strip(),
                out_hidden_size=int(vision_config.get("out_hidden_size") or 0),
                patch_size=int(vision_config.get("patch_size") or 0),
            ),
        )
    except ValueError as exc:
        raise RuntimeError(f"{context}: {exc}") from exc


def qwen_image_validate_max_sequence_length(value: object) -> int:
    try:
        length = int(value)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - strict runtime validation
        raise RuntimeError("Qwen Image max_sequence_length must be an integer.") from exc
    if length <= 0:
        raise RuntimeError(f"Qwen Image max_sequence_length must be positive; got {length}.")
    if length > QWEN_IMAGE_TOKENIZER_MAX_LENGTH:
        raise RuntimeError(
            f"Qwen Image max_sequence_length cannot exceed {QWEN_IMAGE_TOKENIZER_MAX_LENGTH}; got {length}."
        )
    return length


def qwen_image_prompt_plan(
    prompt: object,
    *,
    variant: object,
    max_sequence_length: object = QWEN_IMAGE_TOKENIZER_MAX_LENGTH,
) -> QwenImagePromptPlan:
    spec = qwen_image_variant_spec(variant)
    prompt_text = "" if prompt is None else str(prompt)
    sequence_length = qwen_image_validate_max_sequence_length(max_sequence_length)
    return QwenImagePromptPlan(
        prompt=prompt_text,
        rendered_prompt=spec.prompt_template.format(prompt_text),
        template_start_idx=spec.prompt_template_start_idx,
        max_sequence_length=sequence_length,
    )


__all__ = [
    "QwenImagePromptPlan",
    "QwenImageTextEncoderConfig",
    "QwenImageVisionConfig",
    "qwen_image_prompt_plan",
    "qwen_image_text_encoder_config_from_mapping",
    "qwen_image_validate_max_sequence_length",
]
