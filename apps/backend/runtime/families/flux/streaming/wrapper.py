"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Streamed Flux core wrapper (segment-by-segment execution with memory control).
Wraps `FluxTransformer2DModel` to execute transformer blocks in segments according to an `ExecutionPlan`, delegating device placement and
eviction decisions to a `CoreController` for reduced peak VRAM.

Symbols (top-level; keep in sync; no ghosts):
- `StreamedFluxCore` (class): Flux core wrapper enabling segment-based streaming (executes double/single blocks per segment and exposes
  controller reset/stats helpers).
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

import logging
from typing import Optional, TYPE_CHECKING

import torch
from einops import rearrange, repeat
from torch import nn

from apps.backend.runtime.sampling.block_progress import resolve_block_progress_callback
from .specs import BlockType, ExecutionPlan
from .controller import CoreController

if TYPE_CHECKING:
    from apps.backend.runtime.families.flux.model import FluxTransformer2DModel

logger = get_backend_logger("backend.runtime.flux.streaming.wrapper")


class StreamedFluxCore(nn.Module):
    """Wrapper around FluxTransformer2DModel enabling segment-based streaming.

    This wrapper intercepts the forward pass and executes transformer blocks
    segment-by-segment, using the CoreController to manage GPU memory.
    The original model is not modified.

    The forward signature matches FluxTransformer2DModel exactly:
        (x, timestep, context, y, guidance) -> output

    Example:
        plan = trace_execution_plan(flux_core)
        controller = CoreController(storage="cpu", compute="cuda")
        streamed = StreamedFluxCore(flux_core, plan, controller)
        output = streamed(x, timestep, context, y, guidance)
    """

    def __init__(
        self,
        base_core: FluxTransformer2DModel,
        execution_plan: ExecutionPlan,
        controller: CoreController,
    ) -> None:
        super().__init__()
        self._base = base_core
        self._plan = execution_plan
        self._controller = controller

        # Cache config from base
        self.config = base_core.config
        self.hidden_size = base_core.hidden_size
        self.num_heads = base_core.num_heads

        # Build segment lookup for fast access
        self._double_segments = [s for s in execution_plan if s.block_type == BlockType.DOUBLE]
        self._single_segments = [s for s in execution_plan if s.block_type == BlockType.SINGLE]

        emit_backend_message(
            "StreamedFluxCore initialized",
            logger=logger.name,
            double_segments=len(self._double_segments),
            single_segments=len(self._single_segments),
        )

    @property
    def patch_size(self) -> int:
        return self._base.patch_size

    @property
    def base_core(self) -> FluxTransformer2DModel:
        """Access the underlying FluxTransformer2DModel."""
        return self._base

    @property
    def controller(self) -> CoreController:
        """Access the streaming controller."""
        return self._controller

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        **kwargs,  # Accept extra args like 'control' from sampler
    ) -> torch.Tensor:
        """Forward pass with segment-based streaming.

        This replicates the FluxTransformer2DModel.forward logic but
        executes blocks segment-by-segment with GPU memory management.
        """
        # Validate inputs using base method
        self._base._validate_inputs(x, timestep, context, y, guidance)

        batch, _, height, width = x.shape
        if logger.isEnabledFor(logging.DEBUG):
            emit_backend_message(
                "StreamedFluxCore forward",
                logger=logger.name,
                level=logging.DEBUG,
                batch=batch,
                latent_height=height,
                latent_width=width,
                segments=len(self._plan),
            )

        # === Pre-block operations (always on GPU, small footprint) ===
        patch = self.patch_size
        pad_h = (-height) % patch
        pad_w = (-width) % patch
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

        img_ids = self._build_spatial_ids(
            batch,
            height=height + pad_h,
            width=width + pad_w,
            device=x.device,
            dtype=x.dtype,
        )
        txt_ids = torch.zeros((batch, context.shape[1], 3), device=x.device, dtype=x.dtype)

        # Input projections
        img = self._base.img_in(img)
        from apps.backend.runtime.families.flux.geometry import timestep_embedding

        vec = self._base.time_in(timestep_embedding(timestep, 256).to(img.dtype))
        vec = vec + self._base.vector_in(y)

        if self.config.guidance.enabled:
            if guidance is None:
                raise ValueError("guidance embedding required but not provided")
            vec = vec + self._base.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        txt = self._base.txt_in(context)
        rotary = self._build_rotary(img_ids, txt_ids)
        transformer_options = kwargs.get("transformer_options", None)
        block_progress_callback = resolve_block_progress_callback(transformer_options)
        total_blocks = int(
            sum(len(segment.blocks) for segment in self._double_segments)
            + sum(len(segment.blocks) for segment in self._single_segments)
        )
        if block_progress_callback is not None and total_blocks <= 0:
            raise RuntimeError("Flux streamed core block progress callback requires total_blocks >= 1.")
        global_block_index = 0

        # === Double-stream blocks (streamed) ===
        for seg_idx, segment in enumerate(self._double_segments):
            # Prefetch next segment (AGGRESSIVE policy)
            next_seg = self._double_segments[seg_idx + 1] if seg_idx + 1 < len(self._double_segments) else None
            if next_seg is None and self._single_segments:
                next_seg = self._single_segments[0]
            self._controller.prefetch_next(next_seg)

            # Load segment to GPU
            self._controller.ensure_on_device(segment)

            # Execute all blocks in segment
            for block_info in segment.blocks:
                global_block_index += 1
                if block_progress_callback is not None:
                    block_progress_callback(global_block_index, total_blocks)
                img, txt = block_info.module(img=img, txt=txt, vec=vec, rotary_freqs=rotary)

            # Maybe evict segment
            self._controller.maybe_evict(segment)

        # === Concatenate streams for single-stream processing ===
        tokens = torch.cat((txt, img), dim=1)

        # === Single-stream blocks (streamed) ===
        for seg_idx, segment in enumerate(self._single_segments):
            # Prefetch next segment
            next_seg = self._single_segments[seg_idx + 1] if seg_idx + 1 < len(self._single_segments) else None
            self._controller.prefetch_next(next_seg)

            # Load segment to GPU
            self._controller.ensure_on_device(segment)

            # Execute all blocks in segment
            for block_info in segment.blocks:
                global_block_index += 1
                if block_progress_callback is not None:
                    block_progress_callback(global_block_index, total_blocks)
                tokens = block_info.module(tokens, vec=vec, rotary_freqs=rotary)

            # Maybe evict segment
            self._controller.maybe_evict(segment)

        # === Final layer (always on GPU, small) ===
        tokens = tokens[:, txt.shape[1] :]
        out = self._base.final_layer(tokens, vec)

        # Reshape output
        out = rearrange(
            out,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            ph=patch,
            pw=patch,
            h=(height + pad_h) // patch,
            w=(width + pad_w) // patch,
        )
        return out[:, :, :height, :width]

    def _build_spatial_ids(
        self,
        batch: int,
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build spatial position IDs for image patches."""
        h_len = height // self.patch_size
        w_len = width // self.patch_size
        base = torch.zeros((h_len, w_len, 3), device=device, dtype=dtype)
        base[..., 1] = torch.linspace(0, h_len - 1, steps=h_len, device=device, dtype=dtype)[:, None]
        base[..., 2] = torch.linspace(0, w_len - 1, steps=w_len, device=device, dtype=dtype)[None, :]
        return repeat(base, "h w c -> b (h w) c", b=batch)

    def _build_rotary(self, img_ids: torch.Tensor, txt_ids: torch.Tensor) -> torch.Tensor:
        """Build rotary position embeddings."""
        ids = torch.cat((txt_ids, img_ids), dim=1)
        rotary = self._base.pe_embedder(ids)
        return rotary

    def reset_controller(self) -> None:
        """Reset controller state (call between images)."""
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
