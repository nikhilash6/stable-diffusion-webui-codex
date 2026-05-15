"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: ControlNet runtime composite wrapper built on `ControlGraph`.
Links multiple control nodes into a chain, runs `pre_run` / `get_control` / `cleanup`, and exposes legacy-facing methods.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger for ControlNet runtime diagnostics.
- `ControlComposite` (class): Composite wrapper that exposes legacy ControlNet interfaces while using `ControlGraph`.
- `build_composite` (function): Builds a `ControlComposite` from an iterable of nodes (returns `None` if empty).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Iterable, List, Optional

import torch

from .config import ControlGraph, ControlNode

logger = get_backend_logger("backend.runtime.controlnet")


class ControlComposite:
    """Composite wrapper that exposes legacy ControlNet interfaces while using ControlGraph."""

    def __init__(self, graph: ControlGraph):
        self.graph = graph
        self.head = None
        self._update_links()

    def _update_links(self) -> None:
        previous = None
        for node in self.graph.nodes:
            control = node.control
            control.set_previous_controlnet(previous)
            previous = control
        self.head = previous
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Linked control chain: %s",
                [node.config.name for node in self.graph.nodes],
            )

    def prepare(self, model, percent_to_sigma) -> None:
        node_count = len(self.graph.nodes)
        logger.debug("Preparing %d control nodes", node_count)
        self._update_links()
        for node in self.graph.nodes:
            logger.debug(
                "Pre-run control '%s' (model_type=%s)",
                node.config.name,
                node.config.model_type,
            )
            node.prepare(model, percent_to_sigma)
            node.control.configure_weighting(
                node.request.weight_schedule,
                mask=node.request.mask_config.mask,
            )
            node.control.pre_run(model, percent_to_sigma)

    def cleanup(self) -> None:
        for node in self.graph.nodes:
            node.control.cleanup()
        self.head = None
        logger.debug("Control composite cleaned up")

    def inference_memory_requirements(self, dtype: torch.dtype) -> int:
        total = 0
        for node in self.graph.nodes:
            total += node.control.inference_memory_requirements(dtype)
        return total

    def get_models(self) -> List[object]:
        models: List[object] = []
        for node in self.graph.nodes:
            models.extend(node.control.get_models())
        return models

    def set_transformer_options(self, transformer_options):
        for node in self.graph.nodes:
            node.control.transformer_options = transformer_options

    def get_control(self, x_noisy, t, cond, batched_number):
        if self.head is None:
            logger.debug("Control composite inactive; returning None")
            return None
        logger.debug(
            "Executing control chain on batch=%d shape=%s dtype=%s device=%s",
            x_noisy.shape[0],
            tuple(x_noisy.shape),
            x_noisy.dtype,
            x_noisy.device,
        )
        return self.head.get_control(x_noisy, t, cond, batched_number)

    def __bool__(self):
        return bool(self.graph.nodes)


def build_composite(nodes: Iterable[ControlNode]) -> Optional[ControlComposite]:
    nodes_list = list(nodes)
    if not nodes_list:
        logger.debug("No control nodes provided; composite not created")
        return None
    graph = ControlGraph(nodes=nodes_list)
    logger.debug("Created control composite with %d nodes", len(nodes_list))
    return ControlComposite(graph)
