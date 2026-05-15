"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed ControlNet graph/config dataclasses with explicit validation.
Defines the request/schedule/mask configuration used by patchers and the runtime graph nodes that wrap ControlNet modules.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger for ControlNet config/validation diagnostics.
- `ControlWeightSchedule` (dataclass): Advanced weighting configuration for ControlNet contributions.
- `ControlMaskConfig` (dataclass): Mask guidance configuration for ControlNet.
- `ControlRequest` (dataclass): Input request for ControlNet execution (validation, weighting, mask).
- `ControlNodeConfig` (dataclass): Static configuration for a ControlNet node.
- `ControlNodeState` (dataclass): Mutable runtime state for a ControlNet node.
- `ControlNode` (dataclass): Executable ControlNet node embedded within a graph.
- `ControlGraph` (dataclass): Ordered collection of ControlNet nodes applied to a UNet.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import logging
import torch

from apps.backend.runtime.memory import memory_management

logger = get_backend_logger("backend.runtime.controlnet.config")


@dataclass(frozen=True, slots=True)
class ControlWeightSchedule:
    """Advanced weighting configuration for ControlNet contributions."""

    positive: Optional[Mapping[str, Sequence[float]]] = None
    negative: Optional[Mapping[str, Sequence[float]]] = None
    frame: Optional[Sequence[float]] = None
    sigma: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def validate(self, *, block_lengths: Mapping[str, int], batch_size: int) -> None:
        if self.positive is not None:
            _assert_block_lengths(self.positive, block_lengths, "positive weighting")
        if self.negative is not None:
            _assert_block_lengths(self.negative, block_lengths, "negative weighting")
        if self.frame is not None and len(self.frame) != batch_size:
            raise ValueError(f"frame weighting list length ({len(self.frame)}) must match batch size ({batch_size})")
        if self.sigma is not None and not callable(self.sigma):
            raise TypeError("sigma weighting must be callable")

    def is_noop(self) -> bool:
        return self.positive is None and self.negative is None and self.frame is None and self.sigma is None


def _assert_block_lengths(values: Mapping[str, Sequence[float]], block_lengths: Mapping[str, int], label: str) -> None:
    unknown_keys = set(values.keys()) - set(block_lengths.keys())
    if unknown_keys:
        raise ValueError(f"{label} includes unknown blocks: {', '.join(sorted(unknown_keys))}")
    for block, expected in block_lengths.items():
        block_values = values.get(block)
        if block_values is None:
            continue
        if len(block_values) != expected:
            raise ValueError(f"{label} length mismatch for block '{block}': expected {expected}, got {len(block_values)}")


@dataclass(frozen=True, slots=True)
class ControlMaskConfig:
    """Mask guidance configuration for ControlNet."""

    mask: Optional[torch.Tensor] = None

    def validate(self) -> None:
        if self.mask is None:
            return
        if self.mask.dim() != 4:
            raise ValueError("Control mask must be a 4D tensor (B, 1, H, W)")
        if self.mask.size(1) != 1:
            raise ValueError("Control mask channel dimension must be 1")
        if self.mask.size(0) <= 0 or self.mask.size(2) <= 0 or self.mask.size(3) <= 0:
            raise ValueError("Control mask must have positive batch/height/width dimensions")


@dataclass(frozen=True, slots=True)
class ControlRequest:
    """Input request for ControlNet execution."""

    image: torch.Tensor
    strength: float
    start_percent: float
    end_percent: float
    weight_schedule: ControlWeightSchedule = field(default_factory=ControlWeightSchedule)
    mask_config: ControlMaskConfig = field(default_factory=ControlMaskConfig)

    def validate(self) -> None:
        if not isinstance(self.image, torch.Tensor):
            raise TypeError("ControlRequest.image must be a torch.Tensor")
        if self.image.dim() != 4:
            raise ValueError("ControlRequest.image must be a 4D tensor (B, C, H, W)")
        if not (0.0 <= self.start_percent <= 1.0 and 0.0 <= self.end_percent <= 1.0):
            raise ValueError("start_percent and end_percent must be between 0 and 1")
        if self.start_percent > self.end_percent:
            raise ValueError("start_percent cannot exceed end_percent")
        if not torch.isfinite(self.image).all():
            raise ValueError("ControlRequest.image contains non-finite values")
        _validate_weight_schedule_structure(self.weight_schedule)
        self.mask_config.validate()


def _validate_weight_schedule_structure(schedule: ControlWeightSchedule) -> None:
    if schedule.positive is not None:
        for block, values in schedule.positive.items():
            _ensure_numeric_sequence(values, f"positive weighting for block '{block}'")
    if schedule.negative is not None:
        for block, values in schedule.negative.items():
            _ensure_numeric_sequence(values, f"negative weighting for block '{block}'")
    if schedule.frame is not None:
        _ensure_numeric_sequence(schedule.frame, "frame weighting")
    if schedule.sigma is not None and not callable(schedule.sigma):
        raise TypeError("sigma weighting must be callable")


def _ensure_numeric_sequence(values: Sequence[float], label: str) -> None:
    if not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    for index, value in enumerate(values):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{label} index {index} must be numeric, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ControlNodeConfig:
    """Static configuration for a ControlNet node."""

    name: str
    model_type: str
    supports_online_lora: bool = False


@dataclass
class ControlNodeState:
    """Mutable runtime state for a ControlNet node."""

    device: torch.device
    dtype: torch.dtype
    weight_schedule: ControlWeightSchedule
    mask_config: ControlMaskConfig


@dataclass
class ControlNode:
    """Executable ControlNet node embedded within a graph."""

    config: ControlNodeConfig
    request: ControlRequest
    control: Any
    state: Optional[ControlNodeState] = None

    def prepare(self, model, percent_to_sigma: Callable[[float], torch.Tensor]) -> None:
        self.request.validate()
        device = getattr(model, "device", None)
        dtype = getattr(model, "dtype", None)
        diffusion_model = getattr(model, "diffusion_model", None)
        if device is None and diffusion_model is not None:
            device = getattr(diffusion_model, "device", None)
        if dtype is None and diffusion_model is not None:
            dtype = getattr(diffusion_model, "dtype", None)
        if dtype is None and hasattr(model, "parameters"):
            try:
                dtype = next(model.parameters()).dtype
            except StopIteration:
                dtype = torch.float32
        if device is None and hasattr(model, "parameters"):
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = memory_management.manager.mount_device()
        if dtype is None:
            raise ValueError("Unable to determine dtype for control node preparation")
        if device is None:
            raise ValueError("Unable to determine device for control node preparation")
        self.state = ControlNodeState(
            device=device,
            dtype=dtype,
            weight_schedule=self.request.weight_schedule,
            mask_config=self.request.mask_config,
        )
        logger.debug(
            "Prepared ControlNode '%s' on device=%s dtype=%s",
            self.config.name,
            device,
            dtype,
        )


@dataclass
class ControlGraph:
    """Ordered collection of ControlNet nodes applied to a UNet."""

    nodes: list[ControlNode] = field(default_factory=list)

    def append(self, node: ControlNode) -> None:
        self.nodes.append(node)

    def validate(self) -> None:
        for node in self.nodes:
            node.request.validate()
