"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Attach ControlNet nodes to a UNet patcher (advanced weighting + masks).
Builds validated `ControlRequest` + `ControlNode` objects and wires them into the patcher graph.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger for ControlNet node attachment diagnostics.
- `apply_controlnet_advanced` (function): Clones a UNet patcher and appends an advanced ControlNet node.
- `_build_node_config` (function): Derives a stable `ControlNodeConfig` for a given control module.
- `_normalize_block_weights` (function): Normalizes per-block weighting mappings into validated lists.
- `_normalize_sequence` (function): Normalizes frame-weight sequences into a list of floats.
- `_validate_sigma_callable` (function): Validates/normalizes the sigma-weighting callable.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Mapping, Optional, Sequence

import torch

from apps.backend.runtime.controlnet.config import (
    ControlMaskConfig,
    ControlNode,
    ControlNodeConfig,
    ControlRequest,
    ControlWeightSchedule,
)
from .base import ControlModuleBase

logger = get_backend_logger("backend.patchers.controlnet.apply")


def apply_controlnet_advanced(
    unet,
    controlnet: ControlModuleBase,
    image_bchw: torch.Tensor,
    strength: float,
    start_percent: float,
    end_percent: float,
    positive_advanced_weighting: Optional[Mapping[str, Sequence[float]]] = None,
    negative_advanced_weighting: Optional[Mapping[str, Sequence[float]]] = None,
    advanced_frame_weighting: Optional[Sequence[float]] = None,
    advanced_sigma_weighting=None,
    advanced_mask_weighting: Optional[torch.Tensor] = None,
):
    """Clone the UNet patcher and append an advanced ControlNet node."""

    if not isinstance(image_bchw, torch.Tensor):
        raise TypeError("image_bchw must be a torch.Tensor")

    control_clone = controlnet.clone()
    control_clone.set_cond_hint(
        cond_hint=image_bchw,
        strength=strength,
        timestep_percent_range=(start_percent, end_percent),
    )

    schedule = ControlWeightSchedule(
        positive=_normalize_block_weights(positive_advanced_weighting, "positive weights"),
        negative=_normalize_block_weights(negative_advanced_weighting, "negative weights"),
        frame=_normalize_sequence(advanced_frame_weighting, "frame weights"),
        sigma=_validate_sigma_callable(advanced_sigma_weighting),
    )
    mask_config = ControlMaskConfig(mask=advanced_mask_weighting)

    control_clone.configure_weighting(schedule, advanced_mask_weighting)

    request = ControlRequest(
        image=image_bchw,
        strength=strength,
        start_percent=start_percent,
        end_percent=end_percent,
        weight_schedule=schedule,
        mask_config=mask_config,
    )
    request.validate()

    node = ControlNode(
        config=_build_node_config(control_clone),
        request=request,
        control=control_clone,
    )

    patched = unet.clone()
    patched.add_control_node(node)
    logger.debug(
        "Attached ControlNet node '%s' (strength=%.3f, range=(%.2f, %.2f))",
        node.config.name,
        strength,
        start_percent,
        end_percent,
    )
    return patched


def _build_node_config(control: ControlModuleBase) -> ControlNodeConfig:
    name = getattr(control, "name", control.__class__.__name__)
    model_type = control.__class__.__name__.lower()
    return ControlNodeConfig(name=name, model_type=model_type)


def _normalize_block_weights(
    weights: Optional[Mapping[str, Sequence[float]]],
    label: str,
) -> Optional[Mapping[str, Sequence[float]]]:
    if weights is None:
        return None
    normalized: dict[str, Sequence[float]] = {}
    for key, values in weights.items():
        if not isinstance(values, Sequence):
            raise TypeError(f"{label} for block '{key}' must be a sequence of floats")
        normalized[key] = [float(v) for v in values]
    return normalized


def _normalize_sequence(values: Optional[Sequence[float]], label: str) -> Optional[Sequence[float]]:
    if values is None:
        return None
    if not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of floats")
    return [float(v) for v in values]


def _validate_sigma_callable(callable_or_none):
    if callable_or_none is None:
        return None
    if not callable(callable_or_none):
        raise TypeError("advanced sigma weighting must be callable")
    return callable_or_none
