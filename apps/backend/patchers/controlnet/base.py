"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Core ControlNet patcher types and the `ControlModuleBase` contract.
Defines the runtime context used by advanced weighting/masking and shared lifecycle helpers used by architecture modules.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger used for ControlNet patcher diagnostics.
- `ControlRuntimeContext` (dataclass): Immutable context used when merging control tensor blocks.
- `ControlModuleBase` (class): Base class for ControlNet-compatible modules (lifecycle, cloning, merge helpers).
- `_summarize_block` (function): Formats control tensor blocks for DEBUG logging.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import logging
import torch

from apps.backend.runtime.controlnet.config import ControlMaskConfig, ControlWeightSchedule
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from .weighting import merge_control_signals

logger = get_backend_logger("backend.patchers.controlnet.base")


@dataclass(frozen=True, slots=True)
class ControlRuntimeContext:
    """Context describing how a control contribution should be merged."""

    strength: float
    global_average_pooling: bool
    transformer_options: Dict[str, Any]
    schedule: ControlWeightSchedule
    mask_config: ControlMaskConfig


class ControlModuleBase(ABC):
    """Base class for Codex ControlNet-compatible modules."""

    def __init__(self, *, device: Optional[torch.device] = None, global_average_pooling: bool = False) -> None:
        if device is None:
            device = memory_management.manager.get_device(DeviceRole.CORE)

        self.device = device
        self.global_average_pooling = global_average_pooling

        self.cond_hint_original: Optional[torch.Tensor] = None
        self.cond_hint: Optional[torch.Tensor] = None
        self.strength: float = 1.0
        self.timestep_percent_range: tuple[float, float] = (0.0, 1.0)
        self.timestep_range: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self.transformer_options: Dict[str, Any] = {}
        self.previous_control: Optional["ControlModuleBase"] = None

        self.weight_schedule: ControlWeightSchedule = ControlWeightSchedule()
        self.mask_config: ControlMaskConfig = ControlMaskConfig()

    # ------------------------------------------------------------------ #
    # Lifecycle helpers
    # ------------------------------------------------------------------ #

    def set_cond_hint(
        self,
        cond_hint: torch.Tensor,
        strength: float = 1.0,
        timestep_percent_range: tuple[float, float] = (0.0, 1.0),
    ) -> "ControlModuleBase":
        if not isinstance(cond_hint, torch.Tensor):
            raise TypeError("cond_hint must be a torch.Tensor")

        self.cond_hint_original = cond_hint
        self.strength = float(strength)
        self.timestep_percent_range = timestep_percent_range
        return self

    def configure_weighting(
        self, schedule: ControlWeightSchedule, mask: Optional[torch.Tensor] = None
    ) -> None:
        self.weight_schedule = schedule
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError("advanced mask must be a torch.Tensor")
            config = ControlMaskConfig(mask=mask)
            config.validate()
            self.mask_config = config
        else:
            self.mask_config = ControlMaskConfig()

    def set_previous_controlnet(self, control: Optional["ControlModuleBase"]) -> "ControlModuleBase":
        self.previous_control = control
        return self

    def pre_run(self, model, percent_to_timestep_function) -> None:
        start, end = self.timestep_percent_range
        if start > end:
            raise ValueError("timestep_percent_range start must not exceed end")
        self.timestep_range = (
            percent_to_timestep_function(start),
            percent_to_timestep_function(end),
        )
        if self.previous_control is not None:
            self.previous_control.pre_run(model, percent_to_timestep_function)

    def cleanup(self) -> None:
        if self.previous_control is not None:
            self.previous_control.cleanup()
        if self.cond_hint is not None:
            del self.cond_hint
        self.cond_hint = None
        self.timestep_range = None

    def get_models(self) -> list[object]:
        if self.previous_control is None:
            return []
        return self.previous_control.get_models()

    def inference_memory_requirements(self, dtype) -> int:
        if self.previous_control is None:
            return 0
        return self.previous_control.inference_memory_requirements(dtype)

    # ------------------------------------------------------------------ #
    # Cloning
    # ------------------------------------------------------------------ #

    def clone(self) -> "ControlModuleBase":
        clone = self._clone_impl()
        self._copy_runtime_state_to(clone)
        return clone

    def _copy_runtime_state_to(self, target: "ControlModuleBase") -> None:
        target.cond_hint_original = self.cond_hint_original
        target.cond_hint = None
        target.strength = self.strength
        target.timestep_percent_range = self.timestep_percent_range
        target.global_average_pooling = self.global_average_pooling
        target.mask_config = self.mask_config
        target.weight_schedule = self.weight_schedule
        target.transformer_options = self.transformer_options.copy()

    @abstractmethod
    def _clone_impl(self) -> "ControlModuleBase":
        """Create a new instance without copying runtime state."""

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def _should_skip_timestep(self, timestep: torch.Tensor) -> bool:
        if self.timestep_range is None:
            return False
        start, end = self.timestep_range
        if timestep[0] > start or timestep[0] < end:
            return True
        return False

    def _runtime_context(self) -> ControlRuntimeContext:
        return ControlRuntimeContext(
            strength=self.strength,
            global_average_pooling=self.global_average_pooling,
            transformer_options=self.transformer_options,
            schedule=self.weight_schedule,
            mask_config=self.mask_config,
        )

    def merge_control_outputs(
        self,
        *,
        control_input: Optional[list[Optional[torch.Tensor]]] = None,
        control_output: Optional[list[Optional[torch.Tensor]]] = None,
        control_prev: Optional[Dict[str, list[Optional[torch.Tensor]]]] = None,
        output_dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, list[Optional[torch.Tensor]]]:
        result = merge_control_signals(
            control_input=control_input,
            control_output=control_output,
            previous=control_prev,
            output_dtype=output_dtype,
            context=self._runtime_context(),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Control merge complete: strength=%.3f pooling=%s blocks=%s",
                self.strength,
                self.global_average_pooling,
                {
                    key: _summarize_block(values)
                    for key, values in result.items()
                },
            )
        return result


def _summarize_block(block: list[Optional[torch.Tensor]]) -> list[str]:
    summary: list[str] = []
    for tensor in block:
        if tensor is None:
            summary.append("∅")
        else:
            summary.append(f"{tuple(tensor.shape)}@{tensor.dtype}:{tensor.device}")
    return summary
