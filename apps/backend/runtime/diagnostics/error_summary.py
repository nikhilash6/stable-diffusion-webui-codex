"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Known runtime-error text projections shared by backend console logging and public task error shaping.
Classifies specific high-signal runtime failures into a concise backend console summary plus an optional friendly public-safe message,
without changing the originating exception or swallowing full exception-log dumps.

Symbols (top-level; keep in sync; no ghosts):
- `RuntimeErrorProjection` (dataclass): Shared projection for one classified runtime failure.
- `diagnose_runtime_error` (function): Returns a known projection for a runtime failure string when a bounded classifier matches.
- `friendly_public_exception_message` (function): Returns the friendly public-safe message for a classified runtime failure, if any.
- `summarize_exception_for_console` (function): Returns the concise backend-console summary for a classified runtime failure, falling back to the normalized raw text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_WAN22_TARGET_MISMATCH_RE = re.compile(
    r"WAN22 GGUF stage '(?P<stage>[^']+)': LoRA target-shape mismatch after key resolution\. "
    r"shape_compatible_standard_targets=(?P<matched>\d+)/(?P<total>\d+)\. "
    r"samples=(?P<samples>.*?)(?: file=(?P<file>.+?))?"
    r"(?:\. Additional cleanup failure\(s\): .*)?$"
)
_SHAPE_PAIR_RE = re.compile(
    r"expected=\((?P<expected_rows>\d+)\s*,\s*(?P<expected_cols>\d+)\) "
    r"actual=\((?P<actual_rows>\d+)\s*,\s*(?P<actual_cols>\d+)\)"
)


@dataclass(frozen=True, slots=True)
class RuntimeErrorProjection:
    public_message: str
    console_summary: str


def _normalize_exception_text(err: Any) -> str:
    return str(err or "").strip()


def _wan22_target_mismatch_projection(raw: str) -> RuntimeErrorProjection | None:
    match = _WAN22_TARGET_MISMATCH_RE.search(raw)
    if match is None:
        return None
    matched = int(match.group("matched"))
    total = int(match.group("total"))
    if matched != 0 or total <= 0:
        return None
    stage = match.group("stage")
    samples = match.group("samples")
    file_path = (match.group("file") or "").strip() or None
    shape_match = _SHAPE_PAIR_RE.search(samples)
    if shape_match is None:
        public_message = (
            f"WAN22 stage '{stage}': this LoRA is structurally incompatible with the mounted stage model. "
            f"No target shapes matched ({matched}/{total}), so the LoRA was not applied."
        )
        console_summary = (
            f"WAN22 GGUF stage '{stage}': LoRA target-shape mismatch after key resolution "
            f"({matched}/{total} compatible targets)."
        )
        if file_path is not None:
            console_summary = f"{console_summary} file={file_path}"
        return RuntimeErrorProjection(public_message=public_message, console_summary=console_summary)

    expected_rows = int(shape_match.group("expected_rows"))
    expected_cols = int(shape_match.group("expected_cols"))
    actual_rows = int(shape_match.group("actual_rows"))
    actual_cols = int(shape_match.group("actual_cols"))
    public_message = (
        f"WAN22 stage '{stage}': this LoRA is structurally incompatible with the mounted stage model. "
        f"No target shapes matched ({matched}/{total}), so the LoRA was not applied. "
        f"Example mismatch: expected target shape ({expected_rows}, {expected_cols}) but the mounted stage exposes "
        f"({actual_rows}, {actual_cols})."
    )
    console_summary = (
        f"WAN22 GGUF stage '{stage}': LoRA target-shape mismatch after key resolution "
        f"({matched}/{total} compatible targets; example expected=({expected_rows}, {expected_cols}) "
        f"actual=({actual_rows}, {actual_cols}))."
    )
    if file_path is not None:
        console_summary = f"{console_summary} file={file_path}"
    return RuntimeErrorProjection(public_message=public_message, console_summary=console_summary)


def diagnose_runtime_error(err: Any) -> RuntimeErrorProjection | None:
    raw = _normalize_exception_text(err)
    if not raw:
        return None
    projection = _wan22_target_mismatch_projection(raw)
    if projection is not None:
        return projection
    return None


def friendly_public_exception_message(err: Any) -> str | None:
    projection = diagnose_runtime_error(err)
    if projection is None:
        return None
    return projection.public_message


def summarize_exception_for_console(err: Any) -> str:
    raw = _normalize_exception_text(err)
    projection = diagnose_runtime_error(raw)
    if projection is not None:
        return projection.console_summary
    return raw or "internal error"


__all__ = [
    "RuntimeErrorProjection",
    "diagnose_runtime_error",
    "friendly_public_exception_message",
    "summarize_exception_for_console",
]
