"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Merge ControlNet tensor blocks with strength, pooling, schedules, and masks.
Implements advanced per-block/per-frame/per-sigma weighting and optional chaining of multiple ControlNet modules.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger used for control weighting diagnostics.
- `broadcast_image_to` (function): Broadcasts a conditioning tensor to the requested batch size.
- `merge_control_signals` (function): Merges control tensors with advanced weighting/masking and optional chaining.
- `_process_tensor` (function): Applies pooling/strength/dtype conversion to an individual tensor.
- `_merge_previous` (function): Merges a previous control chain into the current block dict.
- `_apply_weighting_and_mask` (function): Applies schedule weighting and an optional spatial mask.
- `_apply_weight_schedule` (function): Applies per-block weights derived from `ControlWeightSchedule`.
- `_apply_mask` (function): Applies a resized spatial mask to all control tensors.
- `_get_weight` (function): Returns a weight at an index with a default fallback.
- `_require_option` (function): Fetches a required `transformer_options` entry or raises loudly.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

from typing import Any, Dict, List, Mapping, Optional, Sequence

import logging
import torch
import torch.nn.functional as F

from apps.backend.runtime.controlnet.config import ControlMaskConfig, ControlWeightSchedule

logger = get_backend_logger("backend.patchers.controlnet.weighting")


def broadcast_image_to(tensor: torch.Tensor, target_batch_size: int, batched_number: int) -> torch.Tensor:
    """Broadcast a conditioning tensor to match the target batch size."""
    if tensor.shape[0] == target_batch_size:
        return tensor

    if tensor.shape[0] == 1:
        repeats = target_batch_size
        return tensor.repeat(repeats, 1, 1, 1)

    per_batch = target_batch_size // batched_number
    tensor = tensor[:per_batch]

    if per_batch > tensor.shape[0]:
        full_repeats, remainder = divmod(per_batch, tensor.shape[0])
        parts = [tensor] * full_repeats
        if remainder:
            parts.append(tensor[:remainder])
        tensor = torch.cat(parts, dim=0)

    if tensor.shape[0] == target_batch_size:
        return tensor

    return torch.cat([tensor] * batched_number, dim=0)


def merge_control_signals(
    *,
    control_input: Optional[List[Optional[torch.Tensor]]] = None,
    control_output: Optional[List[Optional[torch.Tensor]]] = None,
    previous: Optional[Dict[str, List[Optional[torch.Tensor]]]] = None,
    output_dtype: Optional[torch.dtype],
    context,
) -> Dict[str, List[Optional[torch.Tensor]]]:
    """Merge control signals with advanced weighting and optional chaining."""

    result: Dict[str, List[Optional[torch.Tensor]]] = {"input": [], "middle": [], "output": []}

    if control_input:
        for tensor in control_input:
            processed = _process_tensor(
                tensor,
                strength=context.strength,
                output_dtype=output_dtype,
                global_pooling=context.global_average_pooling,
            )
            result["input"].insert(0, processed)

    if control_output:
        for index, tensor in enumerate(control_output):
            block = "middle" if index == len(control_output) - 1 else "output"
            processed = _process_tensor(
                tensor,
                strength=context.strength,
                output_dtype=output_dtype,
                global_pooling=context.global_average_pooling,
            )
            result[block].append(processed)

    _apply_weighting_and_mask(result, context)

    if previous:
        _merge_previous(result, previous)

    return result


def _process_tensor(
    tensor: Optional[torch.Tensor],
    *,
    strength: float,
    output_dtype: Optional[torch.dtype],
    global_pooling: bool,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None

    out = tensor

    if global_pooling:
        out = torch.mean(out, dim=(2, 3), keepdim=True).repeat(1, 1, tensor.shape[2], tensor.shape[3])

    out = out * strength

    if output_dtype is not None and out.dtype != output_dtype:
        logger.debug("Casting control tensor from %s to %s", out.dtype, output_dtype)
        out = out.to(output_dtype)

    return out


def _merge_previous(
    current: Dict[str, List[Optional[torch.Tensor]]],
    previous: Dict[str, List[Optional[torch.Tensor]]],
) -> None:
    for key in ("input", "middle", "output"):
        current_list = current[key]
        previous_list = previous.get(key, [])

        for index, prev_value in enumerate(previous_list):
            if index >= len(current_list):
                current_list.append(prev_value)
                continue

            cur_value = current_list[index]
            if prev_value is None:
                continue

            if cur_value is None:
                current_list[index] = prev_value
                continue

            if cur_value.shape[0] < prev_value.shape[0]:
                current_list[index] = prev_value + cur_value
            else:
                current_list[index] = cur_value + prev_value


def _apply_weighting_and_mask(result: Dict[str, List[Optional[torch.Tensor]]], context) -> None:
    schedule: ControlWeightSchedule = context.schedule
    mask_config: ControlMaskConfig = context.mask_config
    transformer_options = context.transformer_options

    mask = mask_config.mask
    if mask is not None:
        mask_config.validate()

    if not schedule.is_noop():
        _apply_weight_schedule(result, schedule, transformer_options)

    if mask is not None:
        _apply_mask(result, mask)


def _apply_weight_schedule(
    result: Dict[str, List[Optional[torch.Tensor]]],
    schedule: ControlWeightSchedule,
    transformer_options: Mapping[str, Any],
) -> None:
    if schedule.is_noop():
        return

    cond_or_uncond = _require_option(transformer_options, "cond_or_uncond")
    sigmas = _require_option(transformer_options, "sigmas")
    cond_mark = _require_option(transformer_options, "cond_mark")

    if not isinstance(cond_mark, torch.Tensor):
        raise TypeError("transformer_options['cond_mark'] must be a torch.Tensor")

    block_lengths = {key: len(values) for key, values in result.items()}
    schedule.validate(block_lengths=block_lengths, batch_size=cond_mark.shape[0])

    frame_weight = None
    if schedule.frame is not None:
        frame_weight = torch.tensor(schedule.frame, device=sigmas.device, dtype=sigmas.dtype)
        frame_weight = frame_weight.repeat(len(cond_or_uncond))
        if frame_weight.shape[0] != cond_mark.shape[0]:
            raise ValueError("frame weighting size mismatch against cond_mark")

    sigma_weight = None
    if schedule.sigma is not None:
        sigma_values = schedule.sigma(sigmas)
        if not isinstance(sigma_values, torch.Tensor):
            sigma_values = torch.tensor(sigma_values, device=sigmas.device, dtype=sigmas.dtype)
        sigma_values = sigma_values.to(sigmas)
        if sigma_values.shape[0] != sigmas.shape[0]:
            raise ValueError("sigma weighting callable must return tensor matching sigmas length")
        sigma_weight = sigma_values.repeat(len(cond_or_uncond))

    cond_mark = cond_mark.to(sigmas)

    for block_name, tensors in result.items():
        positive_weights = (schedule.positive or {}).get(block_name, [])
        negative_weights = (schedule.negative or {}).get(block_name, [])
        for index, tensor in enumerate(tensors):
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue

            positive = _get_weight(positive_weights, index, default=1.0)
            negative = _get_weight(negative_weights, index, default=1.0)

            final_weight = positive * (1.0 - cond_mark) + negative * cond_mark

            if sigma_weight is not None:
                final_weight = final_weight * sigma_weight
            if frame_weight is not None:
                final_weight = final_weight * frame_weight

            tensors[index] = tensor * final_weight[:, None, None, None].to(tensor)
            logger.debug(
                "Applied advanced weighting block=%s index=%d positive=%s negative=%s",
                block_name,
                index,
                bool(schedule.positive),
                bool(schedule.negative),
            )


def _apply_mask(result: Dict[str, List[Optional[torch.Tensor]]], mask: torch.Tensor) -> None:
    mask = mask.detach()

    for tensors in result.values():
        for index, tensor in enumerate(tensors):
            if tensor is None:
                continue

            resized_mask = mask
            if resized_mask.shape[0] != tensor.shape[0]:
                repeats, remainder = divmod(tensor.shape[0], resized_mask.shape[0])
                if remainder != 0:
                    raise ValueError("control mask batch size must divide control tensor batch size")
                resized_mask = resized_mask.repeat(repeats, 1, 1, 1)

            resized_mask = resized_mask.to(dtype=tensor.dtype, device=tensor.device)
            resized_mask = F.interpolate(resized_mask, size=(tensor.shape[2], tensor.shape[3]), mode="bilinear")
            tensors[index] = tensor * resized_mask
            logger.debug("Applied control mask to tensor %s", tuple(tensor.shape))


def _get_weight(values: Sequence[float], index: int, *, default: float) -> float:
    return values[index] if 0 <= index < len(values) else default


def _require_option(options: Mapping[str, Any], key: str):
    if key not in options:
        raise KeyError(f"transformer_options missing required key '{key}'")
    return options[key]
