"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux core streaming execution-plan tracer (block enumeration + hook validation).
Builds an `ExecutionPlan` by enumerating Flux transformer blocks and grouping them into segments, with an optional hook-based trace mode to
validate real forward execution order.

Symbols (top-level; keep in sync; no ghosts):
- `_enumerate_blocks` (function): Enumerates double/single transformer blocks from a Flux core model into `BlockInfo` lists.
- `_group_blocks_into_segments` (function): Groups consecutive blocks into `Segment` objects based on `blocks_per_segment`.
- `trace_execution_plan` (function): Builds an `ExecutionPlan` via static enumeration (fast/default).
- `trace_execution_plan_with_hooks` (function): Validates execution order by registering forward hooks and running a dummy forward pass.
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

import logging
from typing import List, Tuple, TYPE_CHECKING

import torch
from torch import nn

from .specs import BlockInfo, BlockType, ExecutionPlan, Segment, calculate_module_bytes

if TYPE_CHECKING:
    from apps.backend.runtime.families.flux.model import FluxTransformer2DModel

logger = get_backend_logger("backend.runtime.flux.streaming.tracer")


def _enumerate_blocks(core: FluxTransformer2DModel) -> Tuple[List[BlockInfo], List[BlockInfo]]:
    """Enumerate all transformer blocks in a Flux core model.

    Returns:
        Tuple of (double_blocks, single_blocks) as BlockInfo lists.
    """
    double_blocks: List[BlockInfo] = []
    single_blocks: List[BlockInfo] = []

    # Access double_blocks ModuleList
    if hasattr(core, "double_blocks"):
        for idx, module in enumerate(core.double_blocks):
            info = BlockInfo(
                index=idx,
                block_type=BlockType.DOUBLE,
                module=module,
                param_bytes=calculate_module_bytes(module),
            )
            double_blocks.append(info)
            emit_backend_message(
                "Enumerated double block",
                logger=logger.name,
                level=logging.DEBUG,
                index=idx,
                mb=info.param_bytes / (1024 * 1024),
            )

    # Access single_blocks ModuleList
    if hasattr(core, "single_blocks"):
        for idx, module in enumerate(core.single_blocks):
            info = BlockInfo(
                index=idx,
                block_type=BlockType.SINGLE,
                module=module,
                param_bytes=calculate_module_bytes(module),
            )
            single_blocks.append(info)
            emit_backend_message(
                "Enumerated single block",
                logger=logger.name,
                level=logging.DEBUG,
                index=idx,
                mb=info.param_bytes / (1024 * 1024),
            )

    return double_blocks, single_blocks


def _group_blocks_into_segments(
    blocks: List[BlockInfo],
    blocks_per_segment: int,
    prefix: str,
) -> List[Segment]:
    """Group consecutive blocks into segments.

    Args:
        blocks: List of BlockInfo to group.
        blocks_per_segment: Maximum blocks per segment.
        prefix: Name prefix for segments (e.g., "double", "single").

    Returns:
        List of Segment objects.
    """
    if not blocks:
        return []

    segments: List[Segment] = []
    for i in range(0, len(blocks), blocks_per_segment):
        chunk = blocks[i : i + blocks_per_segment]
        start_idx = chunk[0].index
        end_idx = chunk[-1].index
        name = f"{prefix}_{start_idx}_{end_idx}"
        total_bytes = sum(b.param_bytes for b in chunk)

        segment = Segment(
            name=name,
            blocks=list(chunk),
            param_bytes=total_bytes,
        )
        segments.append(segment)
        emit_backend_message(
            "Created streaming segment",
            logger=logger.name,
            level=logging.DEBUG,
            name=name,
            blocks=len(chunk),
            mb=total_bytes / (1024 * 1024),
        )

    return segments


def trace_execution_plan(
    core: FluxTransformer2DModel,
    blocks_per_segment: int = 4,
    *,
    validate: bool = True,
) -> ExecutionPlan:
    """Generate an execution plan for streaming a Flux transformer core.

    This function enumerates all DoubleStreamBlock and SingleStreamBlock
    modules in the core, groups them into segments, and returns an ordered
    execution plan matching the forward pass execution order.

    The Flux forward pass has a fixed order:
    1. All double_blocks in sequence (img, txt = block(img, txt, vec, rotary))
    2. All single_blocks in sequence (tokens = block(tokens, vec, rotary))

    Args:
        core: The FluxTransformer2DModel to trace.
        blocks_per_segment: Number of blocks to group per segment.
            Larger values reduce transfer overhead but increase peak VRAM.
            Default: 4 (balanced for ~8GB GPUs).
        validate: If True, verify the core has the expected structure.

    Returns:
        ExecutionPlan with ordered segments matching forward execution.

    Raises:
        ValueError: If validation fails or core structure is unexpected.
    """
    emit_backend_message(
        "Tracing execution plan for Flux core",
        logger=logger.name,
        blocks_per_segment=blocks_per_segment,
    )

    # Enumerate all blocks
    double_blocks, single_blocks = _enumerate_blocks(core)

    if validate:
        if not double_blocks:
            raise ValueError("FluxTransformer2DModel has no double_blocks")
        if not single_blocks:
            raise ValueError("FluxTransformer2DModel has no single_blocks")

    # Group into segments (execution order: double first, then single)
    double_segments = _group_blocks_into_segments(double_blocks, blocks_per_segment, "double")
    single_segments = _group_blocks_into_segments(single_blocks, blocks_per_segment, "single")

    # Build execution plan (double blocks execute before single blocks)
    all_segments = double_segments + single_segments

    plan = ExecutionPlan(
        segments=all_segments,
        double_block_count=len(double_blocks),
        single_block_count=len(single_blocks),
    )

    emit_backend_message(
        "Execution plan created",
        logger=logger.name,
        segments=len(plan),
        double_segments=len(double_segments),
        single_segments=len(single_segments),
        total_mb=plan.total_bytes / (1024 * 1024),
    )

    return plan


def trace_execution_plan_with_hooks(
    core: FluxTransformer2DModel,
    blocks_per_segment: int = 4,
    *,
    dummy_batch: int = 1,
    dummy_height: int = 64,
    dummy_width: int = 64,
    dummy_context_len: int = 256,
) -> ExecutionPlan:
    """Trace execution order via forward hooks (advanced/validation mode).

    This function registers forward_pre_hooks on all transformer blocks,
    runs a dummy forward pass, and captures the actual execution order.
    This is useful for validating that the static enumeration matches
    the runtime behavior.

    Note: For production use, prefer trace_execution_plan() which uses
    static enumeration and is faster/safer.

    Args:
        core: The FluxTransformer2DModel to trace.
        blocks_per_segment: Number of blocks to group per segment.
        dummy_batch: Batch size for dummy forward.
        dummy_height: Latent height for dummy forward.
        dummy_width: Latent width for dummy forward.
        dummy_context_len: Context sequence length for dummy forward.

    Returns:
        ExecutionPlan with segments ordered by actual forward execution.
    """
    emit_backend_message(
        "Tracing execution plan via hooks",
        logger=logger.name,
        blocks_per_segment=blocks_per_segment,
    )

    execution_order: List[Tuple[BlockType, int, nn.Module]] = []
    hooks = []

    def make_hook(block_type: BlockType, idx: int, module: nn.Module):
        def hook(mod, inputs):
            execution_order.append((block_type, idx, module))

        return hook

    # Register hooks on double_blocks
    if hasattr(core, "double_blocks"):
        for idx, module in enumerate(core.double_blocks):
            h = module.register_forward_pre_hook(make_hook(BlockType.DOUBLE, idx, module))
            hooks.append(h)

    # Register hooks on single_blocks
    if hasattr(core, "single_blocks"):
        for idx, module in enumerate(core.single_blocks):
            h = module.register_forward_pre_hook(make_hook(BlockType.SINGLE, idx, module))
            hooks.append(h)

    try:
        # Build dummy inputs matching FluxTransformer2DModel.forward signature
        device = next(core.parameters()).device
        dtype = next(core.parameters()).dtype

        # x: (B, C, H, W) latent
        in_channels = core.config.in_channels if hasattr(core, "config") else 64
        x = torch.zeros((dummy_batch, in_channels, dummy_height, dummy_width), device=device, dtype=dtype)

        # timestep: (B,) scalar
        timestep = torch.zeros((dummy_batch,), device=device, dtype=dtype)

        # context: (B, seq_len, context_dim)
        context_dim = core.config.context_in_dim if hasattr(core, "config") else 4096
        context = torch.zeros((dummy_batch, dummy_context_len, context_dim), device=device, dtype=dtype)

        # y: (B, vec_in_dim) pooled embedding
        vec_in_dim = core.config.vec_in_dim if hasattr(core, "config") else 768
        y = torch.zeros((dummy_batch, vec_in_dim), device=device, dtype=dtype)

        # guidance: (B,) optional
        guidance = None
        if hasattr(core, "config") and core.config.guidance.enabled:
            guidance = torch.ones((dummy_batch,), device=device, dtype=dtype) * 3.5

        # Run dummy forward (inference mode, no grad)
        with torch.inference_mode():
            _ = core(x, timestep, context, y, guidance)

    finally:
        # Always clean up hooks
        for h in hooks:
            h.remove()

    emit_backend_message(
        "Hook trace captured block executions",
        logger=logger.name,
        level=logging.DEBUG,
        count=len(execution_order),
    )

    # Build BlockInfo list from execution order
    all_blocks: List[BlockInfo] = []
    for block_type, idx, module in execution_order:
        info = BlockInfo(
            index=idx,
            block_type=block_type,
            module=module,
            param_bytes=calculate_module_bytes(module),
        )
        all_blocks.append(info)

    # Group into segments preserving execution order
    # Note: We don't separate by type here since we're respecting actual order
    segments: List[Segment] = []
    for i in range(0, len(all_blocks), blocks_per_segment):
        chunk = all_blocks[i : i + blocks_per_segment]
        block_type = chunk[0].block_type
        start_idx = chunk[0].index
        end_idx = chunk[-1].index
        prefix = "double" if block_type == BlockType.DOUBLE else "single"
        name = f"{prefix}_{start_idx}_{end_idx}"
        total_bytes = sum(b.param_bytes for b in chunk)

        segment = Segment(
            name=name,
            blocks=list(chunk),
            param_bytes=total_bytes,
        )
        segments.append(segment)

    double_count = sum(1 for b in all_blocks if b.block_type == BlockType.DOUBLE)
    single_count = sum(1 for b in all_blocks if b.block_type == BlockType.SINGLE)

    plan = ExecutionPlan(
        segments=segments,
        double_block_count=double_count,
        single_block_count=single_count,
    )

    emit_backend_message(
        "Hook-traced plan created",
        logger=logger.name,
        segments=len(plan),
        blocks=plan.total_blocks,
        total_mb=plan.total_bytes / (1024 * 1024),
    )

    return plan
