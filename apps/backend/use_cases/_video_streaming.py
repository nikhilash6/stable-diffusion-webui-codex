"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared video progress helpers for canonical video use-cases.
Provides narrow use-case-local helpers for converting runtime video progress payloads into backend `ProgressEvent`s without moving task/SSE ownership out of the canonical video use-cases.

Symbols (top-level; keep in sync; no ghosts):
- `_yield_wan22_gguf_progress` (function): Map WAN22 GGUF stream dict events into backend `ProgressEvent`s while preserving additive runtime diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from apps.backend.core.requests import ProgressEvent


def _yield_wan22_gguf_progress(ev: dict[str, Any]) -> ProgressEvent | None:
    if ev.get("type") != "progress":
        return None
    stage = str(ev.get("stage", "") or "")
    step = int(ev.get("step", 0))
    total = int(ev.get("total", 0))
    pct = float(ev.get("percent", 0.0))
    pct_out = (pct * 100.0) if (0.0 <= pct <= 1.0) else pct
    eta_raw = ev.get("eta_seconds", None)
    eta = float(eta_raw) if eta_raw is not None else None
    message_raw = ev.get("message", None)
    message = str(message_raw) if message_raw is not None else None
    raw_data = ev.get("data", None)
    data_payload: dict[str, Any] = dict(raw_data) if isinstance(raw_data, Mapping) else {}
    for key in ("progress_adapter", "progress_granularity", "coarse_reason"):
        if key in ev and ev.get(key) is not None:
            data_payload[key] = ev.get(key)
    return ProgressEvent(
        stage=stage,
        percent=pct_out,
        step=step,
        total_steps=total,
        eta_seconds=eta,
        message=message,
        data=data_payload,
    )
