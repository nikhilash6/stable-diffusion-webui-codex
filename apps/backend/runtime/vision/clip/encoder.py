"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime wrapper for Codex-native CLIP vision encoders.
Constructs and loads HF `CLIPVisionModelWithProjection`, normalizes supported source keyspaces through the canonical state-dict resolver,
applies memory-management policies, and returns structured outputs.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger for clip vision encoder lifecycle and timing logs.
- `ClipVisionEncoder` (class): Encapsulates model construction/loading and provides `encode(...)` returning `ClipVisionOutput`.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import time
from collections.abc import Mapping, MutableMapping
from typing import Optional

import torch
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection, modeling_utils

from apps.backend.patchers.base import ModelPatcher
from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime import ops as runtime_ops
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole

from .errors import ClipVisionInputError, ClipVisionLoadError
from .preprocess import preprocess_image
from .registry import get_spec_for_state_dict, validate_state_dict
from .specs import ClipVisionVariantSpec
from .state_dict import normalize_clip_vision_state_dict_with_layout
from .types import ClipVisionOutput

logger = get_backend_logger("backend.runtime.vision.clip.encoder")


class ClipVisionEncoder:
    """Codex-native runtime wrapper for CLIP vision encoders."""

    def __init__(self, spec: ClipVisionVariantSpec):
        self.spec = spec
        self.load_device = memory_management.manager.get_device(DeviceRole.CLIP_VISION)
        self.offload_device = memory_management.manager.get_offload_device(DeviceRole.CLIP_VISION)
        self.runtime_dtype = memory_management.manager.dtype_for_role(DeviceRole.CLIP_VISION)
        to_args = dict(device=self.load_device, dtype=self.runtime_dtype)
        logger.debug(
            "Initialising clip vision encoder variant=%s load_device=%s offload_device=%s dtype=%s",
            spec.variant.value,
            self.load_device,
            self.offload_device,
            self.runtime_dtype,
        )
        config = CLIPVisionConfig(**spec.to_huggingface_kwargs())
        with runtime_ops.using_codex_operations(**to_args, manual_cast_enabled=True):
            with modeling_utils.no_init_weights():
                self.model = CLIPVisionModelWithProjection(config).to(**to_args)
        self.model.eval()
        self.patcher = ModelPatcher(
            self.model,
            load_device=self.load_device,
            offload_device=self.offload_device,
        )

    @classmethod
    def from_state_dict(cls, state_dict: MutableMapping[str, object]) -> "ClipVisionEncoder":
        normalized_state_dict, layout = normalize_clip_vision_state_dict_with_layout(state_dict)
        spec = get_spec_for_state_dict(normalized_state_dict)
        encoder = cls(spec)
        encoder.load_state_dict(normalized_state_dict)
        logger.info(
            "Clip vision image encoder resolved to canonical keyspace: variant=%s source_style=%s qkv_layout=%s projection_orientation=%s",
            spec.variant.value,
            layout.source_style,
            layout.qkv_layout,
            layout.projection_orientation,
        )
        return encoder

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        validate_state_dict(state_dict, self.spec)
        missing, unexpected = safe_load_state_dict(self.model, state_dict, log_name="ClipVisionEncoder")
        if missing or unexpected:
            raise ClipVisionLoadError(
                "Clip vision state dict mismatch after canonical resolution: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
            )
        logger.info(
            "Loaded clip vision encoder variant=%s with %d parameters.",
            self.spec.variant.value,
            sum(p.numel() for p in self.model.parameters()),
        )

    def encode(
        self,
        image: torch.Tensor,
        *,
        crop: bool = True,
        return_all_hidden_states: bool = False,
    ) -> ClipVisionOutput:
        if not isinstance(image, torch.Tensor):
            raise ClipVisionInputError("ClipVisionEncoder.encode expects a torch.Tensor input.")
        processed = self.prepare_pixels(image, crop=crop)
        return self.encode_pixels(processed, return_all_hidden_states=return_all_hidden_states)

    def prepare_pixels(self, image: torch.Tensor, *, crop: bool = True) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise ClipVisionInputError("ClipVisionEncoder.prepare_pixels expects a torch.Tensor input.")
        return preprocess_image(image, self.spec.preprocess, crop=crop)

    def encode_pixels(
        self,
        pixel_values: torch.Tensor,
        *,
        return_all_hidden_states: bool = False,
    ) -> ClipVisionOutput:
        if not isinstance(pixel_values, torch.Tensor):
            raise ClipVisionInputError("ClipVisionEncoder.encode_pixels expects a torch.Tensor input.")
        if pixel_values.ndim != 4:
            raise ClipVisionInputError(
                "ClipVisionEncoder.encode_pixels expects a 4D tensor "
                f"(batch, channels, height, width); got {tuple(pixel_values.shape)}."
            )
        start = time.perf_counter()
        memory_management.manager.load_model(self.patcher)
        processed = pixel_values.to(device=self.load_device, dtype=self.runtime_dtype)
        outputs = self.model(
            pixel_values=processed,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states or ()
        if len(hidden_states) < 1:
            raise ClipVisionLoadError("Vision model did not return hidden states.")
        penultimate_index = -2 if len(hidden_states) >= 2 else -1
        intermediate_device = memory_management.manager.get_device(DeviceRole.INTERMEDIATE)
        last_hidden = outputs.last_hidden_state.to(intermediate_device)
        penultimate = hidden_states[penultimate_index].to(intermediate_device)
        embeds = outputs.image_embeds.to(intermediate_device)
        all_hidden: Optional[torch.Tensor] = None
        if return_all_hidden_states:
            try:
                all_hidden = torch.stack(
                    [state.to(intermediate_device) for state in hidden_states],
                    dim=1,
                )
            except RuntimeError as exc:  # pragma: no cover - defensive guard
                raise ClipVisionInputError("Failed to stack hidden states for return.") from exc
        runtime = time.perf_counter() - start
        logger.debug(
            "Encoded clip vision batch=%d seq_len=%d hidden_size=%d embeddings=%s runtime=%.3fs",
            processed.shape[0],
            last_hidden.shape[1],
            last_hidden.shape[2],
            tuple(embeds.shape),
            runtime,
        )
        return ClipVisionOutput(
            last_hidden_state=last_hidden,
            penultimate_hidden_states=penultimate,
            image_embeds=embeds,
            all_hidden_states=all_hidden,
            mm_projected=None,
        )
