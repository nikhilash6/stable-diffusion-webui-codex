"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 GGUF runtime facade (entrypoints + config types).
Keeps the stable import path used by WAN engines by re-exporting the run functions and config dataclasses implemented in focused modules,
including the exact single-stage 5B entrypoints and the existing dual-stage 14B entrypoints.

Symbols (top-level; keep in sync; no ghosts):
- `StageConfig` (class): Stage-level configuration for a WAN run (stage GGUF path + sampler/scheduler/steps/cfg + optional LoRA).
- `RunConfig` (class): Full run configuration for txt2vid/img2vid (assets, devices/dtypes, and the truthful exact stage owner: `single` or `high` + `low`).
- `run_txt2vid_single` (function): Batch single-stage txt2vid runner for WAN 2.2 5B GGUF.
- `run_txt2vid` (function): Batch txt2vid runner (GGUF stages → sampling → VAE decode).
- `stream_txt2vid_single` (function): Streaming single-stage txt2vid generator yielding progress events and final frames.
- `stream_txt2vid` (function): Streaming txt2vid generator yielding progress events and final frames.
- `run_img2vid_single` (function): Batch single-stage img2vid runner for WAN 2.2 5B GGUF.
- `run_img2vid` (function): Batch img2vid runner (encode I2V conditioning video → seeded noise state → stages → VAE decode).
- `stream_img2vid_single` (function): Streaming single-stage img2vid generator yielding progress events and final frames.
- `stream_img2vid` (function): Streaming img2vid generator yielding progress events and final frames.
- `stream_img2vid_chunked` (function): Chunked img2vid streaming runner (single text-conditioning pass + phase-batched high/low over chunks).
- `stream_img2vid_sliding_window` (function): Sliding-window img2vid streaming runner (window/stride/commit controls over chunk runtime).
- `stream_img2vid_svi2` (function): SVI 2.0 img2vid streaming runner (anchor-padded conditioning profile).
- `stream_img2vid_svi2_pro` (function): SVI 2.0 Pro img2vid streaming runner (anchor+motion+zero latent profile).
- `__all__` (constant): Export list for the WAN22 GGUF runtime facade.
"""

from __future__ import annotations

from .config import RunConfig, StageConfig
from .run import (
    run_img2vid_single,
    run_img2vid,
    run_txt2vid_single,
    run_txt2vid,
    stream_img2vid_single,
    stream_img2vid,
    stream_img2vid_chunked,
    stream_img2vid_sliding_window,
    stream_img2vid_svi2,
    stream_img2vid_svi2_pro,
    stream_txt2vid_single,
    stream_txt2vid,
)

__all__ = [
    "RunConfig",
    "StageConfig",
    "run_txt2vid_single",
    "run_txt2vid",
    "run_img2vid_single",
    "run_img2vid",
    "stream_txt2vid_single",
    "stream_txt2vid",
    "stream_img2vid_single",
    "stream_img2vid",
    "stream_img2vid_chunked",
    "stream_img2vid_sliding_window",
    "stream_img2vid_svi2",
    "stream_img2vid_svi2_pro",
]
