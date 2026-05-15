"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Block-level progress callback contract for sampling runtimes.
Defines the canonical transformer-options key used to propagate per-block progress callbacks from the sampling driver into model block loops.
Also provides a shared Rich-based console progress controller for block callbacks, lazily materializing the first visible task so output starts with truthful block totals (optional/no-op when Rich is unavailable).

Symbols (top-level; keep in sync; no ghosts):
- `BlockProgressCallback` (type alias): Callable contract receiving `(block_index, total_blocks)` as 1-based progress.
- `BLOCK_PROGRESS_CALLBACK_KEY` (constant): Canonical transformer-options key for block progress callback injection.
- `BLOCK_PROGRESS_TOTAL_KEY` (constant): Transformer-options key for per-forward global block totals.
- `BLOCK_PROGRESS_INDEX_KEY` (constant): Transformer-options key for per-forward running global block index.
- `validate_block_progress_payload` (function): Strict payload validator for `(block_index, total_blocks)` callback updates.
- `RichBlockProgressController` (class): Optional Rich progress bar controller for block callbacks (safe no-op without Rich).
- `resolve_block_progress_callback` (function): Validate and resolve optional block progress callback from transformer options.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Callable

try:
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
except Exception:  # pragma: no cover - runtime optional dependency
    BarColumn = None  # type: ignore[assignment]
    Progress = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]


BlockProgressCallback = Callable[[int, int], None]

BLOCK_PROGRESS_CALLBACK_KEY = "codex_sampling_block_progress_callback"
BLOCK_PROGRESS_TOTAL_KEY = "codex_sampling_block_progress_total"
BLOCK_PROGRESS_INDEX_KEY = "codex_sampling_block_progress_index"


def validate_block_progress_payload(block_index: int, total_blocks: int) -> tuple[int, int]:
    if isinstance(block_index, bool) or not isinstance(block_index, int):
        raise RuntimeError(
            "Block progress callback payload block_index must be an integer "
            f"(got {type(block_index).__name__})."
        )
    if isinstance(total_blocks, bool) or not isinstance(total_blocks, int):
        raise RuntimeError(
            "Block progress callback payload total_blocks must be an integer "
            f"(got {type(total_blocks).__name__})."
        )
    if total_blocks <= 0:
        raise RuntimeError(
            "Block progress callback payload total_blocks must be >= 1 "
            f"(got {total_blocks})."
        )
    if block_index < 1 or block_index > total_blocks:
        raise RuntimeError(
            "Block progress callback payload block_index must be in [1, total_blocks] "
            f"(got block_index={block_index}, total_blocks={total_blocks})."
        )
    return int(block_index), int(total_blocks)


class RichBlockProgressController:
    def __init__(self, *, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self._progress: Progress | None = None
        self._task_id: Any | None = None
        self._cycle_total = 0
        self._cycle_last_index = 0
        self._cycle_started_at: float | None = None

        if not self._enabled:
            return
        if Progress is None or TextColumn is None or BarColumn is None or TimeElapsedColumn is None:
            return
        try:
            progress = Progress(
                TextColumn("  {task.percentage:>3.0f}% |"),
                BarColumn(bar_width=32),
                TextColumn("|"),
                TimeElapsedColumn(),
                TextColumn("| {task.completed:.0f}/{task.total:.0f} {task.fields[label]} [{task.fields[blocks_per_second]}blocks/s]"),
                transient=True,
                auto_refresh=True,
                refresh_per_second=12,
            )
            progress.start()
        except Exception:
            return
        self._progress = progress
        self._task_id = None

    @property
    def is_active(self) -> bool:
        return self._progress is not None

    def update(self, block_index: int, total_blocks: int, *, label: str | None = None) -> None:
        normalized_index, normalized_total = validate_block_progress_payload(block_index, total_blocks)
        if not self.is_active:
            return
        assert self._progress is not None

        now = time.perf_counter()
        normalized_label = str(label).strip() if isinstance(label, str) and label.strip() else "layer"
        if self._task_id is None:
            self._task_id = self._progress.add_task(
                "",
                total=normalized_total,
                completed=normalized_index,
                blocks_per_second="0.00",
                label=normalized_label,
            )
            self._cycle_started_at = now
            self._cycle_last_index = normalized_index
            self._cycle_total = normalized_total
            return

        assert self._task_id is not None
        if (
            self._cycle_started_at is None
            or normalized_total != self._cycle_total
            or normalized_index <= self._cycle_last_index
        ):
            self._cycle_started_at = now
            self._cycle_last_index = 0
            self._cycle_total = normalized_total

        self._cycle_last_index = normalized_index
        elapsed = max(now - self._cycle_started_at, 1e-6)
        blocks_per_second = 0.0 if normalized_index <= 1 else float(normalized_index) / elapsed
        self._progress.update(
            self._task_id,
            total=normalized_total,
            completed=normalized_index,
            blocks_per_second=f"{blocks_per_second:.2f}",
            label=normalized_label,
        )

    def close(self) -> None:
        progress = self._progress
        self._progress = None
        self._task_id = None
        self._cycle_total = 0
        self._cycle_last_index = 0
        self._cycle_started_at = None
        if progress is None:
            return
        try:
            progress.stop()
        except Exception:
            pass


def resolve_block_progress_callback(
    transformer_options: Mapping[str, Any] | None,
) -> BlockProgressCallback | None:
    if transformer_options is None:
        return None
    if not isinstance(transformer_options, Mapping):
        raise RuntimeError(
            "transformer_options must be Mapping[str, Any] when resolving block progress callback "
            f"(got {type(transformer_options).__name__})."
        )
    raw_callback = transformer_options.get(BLOCK_PROGRESS_CALLBACK_KEY, None)
    if raw_callback is None:
        return None
    if not callable(raw_callback):
        raise RuntimeError(
            f"transformer_options['{BLOCK_PROGRESS_CALLBACK_KEY}'] must be callable when provided "
            f"(got {type(raw_callback).__name__})."
        )
    return raw_callback


__all__ = [
    "BlockProgressCallback",
    "BLOCK_PROGRESS_CALLBACK_KEY",
    "BLOCK_PROGRESS_TOTAL_KEY",
    "BLOCK_PROGRESS_INDEX_KEY",
    "validate_block_progress_payload",
    "RichBlockProgressController",
    "resolve_block_progress_callback",
]
