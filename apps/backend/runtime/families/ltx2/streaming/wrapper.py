"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Hook-based wrapper for LTX2 transformer-core streaming.
Wraps the native `Ltx2VideoTransformer3DModel` without rewriting its giant forward: segment boundary hooks move
`transformer_blocks` between storage and compute devices around the existing internal loop while proxying the
transformer surfaces consumed by the native LTX2 pipeline.

Symbols (top-level; keep in sync; no ghosts):
- `StreamedLtx2Transformer` (class): Wrapper that streams `transformer_blocks` via pre/post-forward hooks.
- `wrap_for_streaming` (function): Factory that builds a plan+controller and returns a `StreamedLtx2Transformer`.
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

from contextlib import nullcontext
import logging
from typing import Any

from torch import nn

from .controller import Ltx2CoreController, create_controller
from .specs import Ltx2ExecutionPlan, Ltx2Segment, build_execution_plan

logger = get_backend_logger("backend.runtime.ltx2.streaming.wrapper")


class StreamedLtx2Transformer(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        execution_plan: Ltx2ExecutionPlan,
        controller: Ltx2CoreController,
    ) -> None:
        super().__init__()
        self._base = base_model
        self._plan = execution_plan
        self._controller = controller
        self.config = getattr(base_model, "config", None)
        self._hook_handles: list[Any] = []
        self._install_segment_hooks()
        self._initialize_storage_residency()

        emit_backend_message(
            "StreamedLtx2Transformer initialized",
            logger=logger.name,
            segments=len(execution_plan),
            blocks=execution_plan.block_count,
        )

    @property
    def base_model(self) -> nn.Module:
        return self._base

    @property
    def controller(self) -> Ltx2CoreController:
        return self._controller

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            base_model = object.__getattribute__(self, "_modules").get("_base")
            if base_model is not None:
                try:
                    return getattr(base_model, name)
                except AttributeError:
                    pass
            raise exc

    def forward(self, *args: Any, **kwargs: Any):
        return self._base(*args, **kwargs)

    def cache_context(self, *args: Any, **kwargs: Any):
        cache_context = getattr(self._base, "cache_context", None)
        if not callable(cache_context):
            return nullcontext()
        return cache_context(*args, **kwargs)

    def reset_controller(self) -> None:
        self._controller.reset()
        self._controller.reset_stats()

    def get_transfer_stats(self) -> dict[str, float]:
        return self._controller.stats.summary()

    def move_all_to_storage(self) -> None:
        self._controller.evict_all()
        self._controller.clear_residency()

    def move_all_to_compute(self) -> None:
        for segment in self._plan:
            segment.to_device(self._controller.compute_device)

    def _initialize_storage_residency(self) -> None:
        for segment in self._plan:
            segment.to_device(self._controller.storage_device)
        self._controller.clear_residency()

    def _install_segment_hooks(self) -> None:
        segments = list(self._plan)
        if not segments:
            raise RuntimeError("LTX2 streamed transformer requires at least one segment.")

        for index, segment in enumerate(segments):
            next_segment = segments[index + 1] if index + 1 < len(segments) else None
            first_block = segment.blocks[0].module
            last_block = segment.blocks[-1].module

            self._hook_handles.append(first_block.register_forward_pre_hook(self._make_pre_hook(segment, next_segment)))
            self._hook_handles.append(last_block.register_forward_hook(self._make_post_hook(segment)))

    def _make_pre_hook(self, segment: Ltx2Segment, next_segment: Ltx2Segment | None):
        def _hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            del module, inputs
            self._controller.prefetch_next(next_segment)
            self._controller.ensure_on_device(segment)

        return _hook

    def _make_post_hook(self, segment: Ltx2Segment):
        def _hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del module, inputs, output
            self._controller.maybe_evict(segment)

        return _hook


def wrap_for_streaming(
    model: nn.Module,
    *,
    policy: str = "naive",
    blocks_per_segment: int = 1,
    window_size: int = 1,
) -> StreamedLtx2Transformer:
    plan = build_execution_plan(model, blocks_per_segment=blocks_per_segment)
    controller = create_controller(
        policy=policy,
        window_size=window_size,
    )
    return StreamedLtx2Transformer(
        base_model=model,
        execution_plan=plan,
        controller=controller,
    )
