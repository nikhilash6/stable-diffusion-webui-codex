"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: StreamedWanTransformer wrapper for segment-based WAN22 core streaming (nn.Module pattern).
Mirrors Flux's `StreamedFluxCore` by intercepting the WAN transformer forward pass and executing blocks segment-by-segment using a
`WanExecutionPlan` and `WanCoreController` to manage GPU residency and transfer policy.

Symbols (top-level; keep in sync; no ghosts):
- `StreamedWanTransformer` (class): Wrapper around `WanTransformer2DModel` enabling segment streaming for the forward pass.
- `wrap_for_streaming` (function): Factory that builds a plan+controller and returns a `StreamedWanTransformer` wrapper.
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

import logging
from typing import Optional, TYPE_CHECKING

import torch
from torch import nn

from apps.backend.runtime.sampling.block_progress import resolve_block_progress_callback
from .specs import WanExecutionPlan, build_execution_plan
from .controller import WanCoreController

if TYPE_CHECKING:
    from apps.backend.runtime.families.wan22.model import WanTransformer2DModel

logger = get_backend_logger("backend.runtime.wan22.streaming.wrapper")


class StreamedWanTransformer(nn.Module):
    """Wrapper around WanTransformer2DModel enabling segment-based streaming.

    This wrapper intercepts the forward pass and executes transformer blocks
    segment-by-segment, using the WanCoreController to manage GPU memory.
    The original model is not modified.

    Example:
        plan = build_execution_plan(model, blocks_per_segment=4)
        controller = WanCoreController(storage="cpu", compute="cuda")
        streamed = StreamedWanTransformer(model, plan, controller)
        output = streamed(x, timestep, context)
    """

    def __init__(
        self,
        base_model: "WanTransformer2DModel",
        execution_plan: WanExecutionPlan,
        controller: WanCoreController,
    ) -> None:
        super().__init__()
        self._base = base_model
        self._plan = execution_plan
        self._controller = controller

        # Cache config from base
        self.config = base_model.config
        self.d_model = base_model.d_model
        self.n_heads = base_model.n_heads
        self.n_blocks = base_model.n_blocks

        emit_backend_message(
            "StreamedWanTransformer initialized",
            logger=logger.name,
            segments=len(execution_plan),
            blocks=execution_plan.block_count,
        )

    @property
    def base_model(self) -> "WanTransformer2DModel":
        """Access the underlying WanTransformer2DModel."""
        return self._base

    @property
    def controller(self) -> WanCoreController:
        """Access the streaming controller."""
        return self._controller

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        transformer_options: dict | None = None,
    ) -> torch.Tensor:
        """Forward pass with segment-based streaming.

        This replicates WanTransformer2DModel.forward() but executes
        blocks segment-by-segment with GPU memory management.
        """
        device = x.device
        dtype = x.dtype
        B, C, T, H, W = x.shape

        # Timestep to scalar tensor
        if isinstance(timestep, (int, float)):
            timestep = torch.tensor([timestep], device=device, dtype=torch.float32)
        if timestep.numel() == 1 and B > 1:
            timestep = timestep.expand(B)

        # Time embedding (non-streamed, small footprint)
        t_emb = self._base._timestep_embedding(timestep)
        t_emb = self._base.time_embed(t_emb.to(dtype))
        t_proj = self._base.time_proj(t_emb)
        t_proj = t_proj.view(B, 6, self.d_model)

        # Text embedding projection (non-streamed)
        ctx = self._base.text_embed(context.to(dtype))

        # Patch embed (non-streamed)
        tokens = self._base.patch_embed(x)
        _, _, T2, H2, W2 = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)

        # === Block-wise streaming ===
        segments = list(self._plan)
        block_progress_callback = resolve_block_progress_callback(transformer_options)
        total_blocks = int(self._plan.block_count)
        for seg_idx, segment in enumerate(segments):
            # Prefetch next segment (AGGRESSIVE policy)
            next_seg = segments[seg_idx + 1] if seg_idx + 1 < len(segments) else None
            self._controller.prefetch_next(next_seg)

            # Load segment to GPU
            self._controller.ensure_on_device(segment)

            # Execute all blocks in segment
            for block_info in segment.blocks:
                if block_progress_callback is not None:
                    block_progress_callback(int(block_info.index + 1), total_blocks)
                tokens = block_info.module(tokens, ctx, t_proj)

            # Maybe evict segment
            self._controller.maybe_evict(segment)

        # Output head (non-streamed)
        tokens = self._base.norm_out(tokens)
        mod = t_proj[:, :2] + self._base.head_modulation.unsqueeze(0)
        shift, scale = mod[:, 0], mod[:, 1]
        tokens = tokens * (1 + scale[:, None, :]) + shift[:, None, :]
        patches = self._base.head(tokens)

        # Unpatchify
        kT, kH, kW = self.config.patch_size
        out = patches.view(B, T2, H2, W2, kT, kH, kW, self.config.latent_channels)
        out = out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        out = out.view(B, self.config.latent_channels, T2 * kT, H2 * kH, W2 * kW)

        return out

    def reset_controller(self) -> None:
        """Reset controller state (call between generations)."""
        self._controller.reset()

    def get_transfer_stats(self) -> dict:
        """Get transfer statistics summary."""
        return self._controller.stats.summary()

    def move_all_to_storage(self) -> None:
        """Move all segments to storage device (cleanup)."""
        for segment in self._plan:
            segment.to_device(self._controller.storage_device)
        self._controller.evict_all()
        emit_backend_message("All segments moved to storage device", logger=logger.name)

    def move_all_to_compute(self) -> None:
        """Move all segments to compute device (disable streaming)."""
        for segment in self._plan:
            segment.to_device(self._controller.compute_device)
        emit_backend_message("All segments moved to compute device (streaming disabled)", logger=logger.name)


def wrap_for_streaming(
    model: "WanTransformer2DModel",
    policy: str = "naive",
    blocks_per_segment: int = 4,
    window_size: int = 2,
    compute_device: Optional[str] = None,
) -> StreamedWanTransformer:
    """Factory function to wrap a WanTransformer2DModel for streaming.

    Args:
        model: The WanTransformer2DModel to wrap.
        policy: Streaming policy ("naive", "window", "aggressive").
        blocks_per_segment: Blocks per segment.
        window_size: Window size for "window" policy.
        compute_device: Compute device (default: auto-detect).

    Returns:
        StreamedWanTransformer wrapper.
    """
    from .controller import create_controller

    plan = build_execution_plan(model, blocks_per_segment)
    controller = create_controller(
        policy=policy,
        window_size=window_size,
        compute_device=compute_device,
    )

    return StreamedWanTransformer(model, plan, controller)
