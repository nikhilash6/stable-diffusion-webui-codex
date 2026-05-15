"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Dataclasses for LTX2 transformer-core streaming segments and execution plans.
Builds fixed-order streaming segments from `Ltx2VideoTransformer3DModel.transformer_blocks` so the shared controller can
move consecutive transformer blocks between storage and compute devices.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2BlockInfo` (class): Metadata for a single LTX2 transformer block.
- `Ltx2Segment` (class): Streaming segment containing one or more consecutive LTX2 transformer blocks.
- `Ltx2ExecutionPlan` (class): Ordered execution plan for LTX2 transformer-core streaming.
- `calculate_module_bytes` (function): Computes parameter size in bytes for an nn.Module.
- `build_execution_plan` (function): Builds an execution plan by grouping `transformer_blocks` into fixed-size segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn


@dataclass
class Ltx2BlockInfo:
    index: int
    module: nn.Module
    param_bytes: int = 0

    def __hash__(self) -> int:
        return hash(self.index)


@dataclass
class Ltx2Segment:
    name: str
    blocks: List[Ltx2BlockInfo] = field(default_factory=list)
    param_bytes: int = 0

    @property
    def modules(self) -> List[nn.Module]:
        return [block.module for block in self.blocks]

    def to_device(self, device: torch.device, *, non_blocking: bool = False) -> None:
        for module in self.modules:
            module.to(device, non_blocking=non_blocking)

    def __len__(self) -> int:
        return len(self.blocks)


@dataclass
class Ltx2ExecutionPlan:
    segments: List[Ltx2Segment] = field(default_factory=list)
    block_count: int = 0

    @property
    def total_bytes(self) -> int:
        return sum(segment.param_bytes for segment in self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)


def calculate_module_bytes(module: nn.Module) -> int:
    total = 0
    for param in module.parameters():
        total += param.numel() * param.element_size()
    return total


def build_execution_plan(model: nn.Module, blocks_per_segment: int = 1) -> Ltx2ExecutionPlan:
    if isinstance(blocks_per_segment, bool) or not isinstance(blocks_per_segment, int) or blocks_per_segment < 1:
        raise RuntimeError(
            "LTX2 streaming blocks_per_segment must be an integer >= 1; "
            f"got {blocks_per_segment!r}."
        )
    if not hasattr(model, "transformer_blocks"):
        raise RuntimeError("LTX2 streaming requires a transformer with `transformer_blocks`.")

    raw_blocks = getattr(model, "transformer_blocks")
    if not isinstance(raw_blocks, nn.ModuleList):
        raise RuntimeError(
            "LTX2 streaming requires `transformer_blocks` to be an nn.ModuleList; "
            f"got {type(raw_blocks).__name__}."
        )
    if len(raw_blocks) == 0:
        raise RuntimeError("LTX2 streaming requires at least one transformer block.")

    blocks: List[Ltx2BlockInfo] = []
    for index, block in enumerate(raw_blocks):
        blocks.append(
            Ltx2BlockInfo(
                index=index,
                module=block,
                param_bytes=calculate_module_bytes(block),
            )
        )

    segments: List[Ltx2Segment] = []
    for offset in range(0, len(blocks), blocks_per_segment):
        chunk = blocks[offset : offset + blocks_per_segment]
        start_index = chunk[0].index
        end_index = chunk[-1].index
        segments.append(
            Ltx2Segment(
                name=f"ltx2_{start_index}_{end_index}",
                blocks=list(chunk),
                param_bytes=sum(block.param_bytes for block in chunk),
            )
        )

    return Ltx2ExecutionPlan(
        segments=segments,
        block_count=len(blocks),
    )
