"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public API for LTX2 core streaming.
Exports the LTX2 family-local streaming config/spec/controller/wrapper so runtime assembly can enable transformer-core
streaming without importing shared-controller internals directly.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2StreamingConfig` (class): Fixed-default config for enabling LTX2 transformer-core streaming.
- `Ltx2StreamingPolicy` (constant): Alias of the shared streaming policy enum for LTX2 ownership.
- `Ltx2CoreController` (class): LTX2 wrapper over the shared streaming controller.
- `Ltx2ExecutionPlan` (class): Ordered segment plan for LTX2 transformer blocks.
- `Ltx2Segment` (class): Streaming segment for consecutive LTX2 transformer blocks.
- `build_execution_plan` (function): Builds an LTX2 execution plan from `transformer_blocks`.
- `create_controller` (function): Creates an LTX2 streaming controller with default memory-manager devices.
- `StreamedLtx2Transformer` (class): Hook-based wrapper around the native LTX2 transformer core.
- `wrap_for_streaming` (function): Builds a plan+controller and wraps the native LTX2 transformer.
"""

from .config import Ltx2StreamingConfig
from .controller import Ltx2CoreController, Ltx2StreamingPolicy, create_controller
from .specs import Ltx2ExecutionPlan, Ltx2Segment, build_execution_plan
from .wrapper import StreamedLtx2Transformer, wrap_for_streaming

__all__ = [
    "Ltx2CoreController",
    "Ltx2ExecutionPlan",
    "Ltx2Segment",
    "Ltx2StreamingConfig",
    "Ltx2StreamingPolicy",
    "StreamedLtx2Transformer",
    "build_execution_plan",
    "create_controller",
    "wrap_for_streaming",
]
