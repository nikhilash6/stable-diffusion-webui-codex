"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared IP-Adapter sampling stage wrapper.
Exposes the side-effect-only request-scoped context manager that applies the validated global IP-Adapter module to the active denoiser
for one sampling pass and restores the baseline Codex objects afterwards.

Symbols (top-level; keep in sync; no ghosts):
- `apply_processing_ip_adapter` (function): Sampling-stage context manager for the active processing object.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib

from apps.backend.runtime.adapters.ip_adapter import apply_ip_adapter_for_sampling


@contextlib.contextmanager
def apply_processing_ip_adapter(processing) -> Iterator[None]:
    with apply_ip_adapter_for_sampling(processing):
        yield
