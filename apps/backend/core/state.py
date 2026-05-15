"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend state snapshot for runtime progress reporting.
Provides a small thread-safe state object (`BackendState`) used by runtimes/services to report sampling progress and VAE phase block progress (encode/decode), including a bounded per-run owner token so concurrent tasks can ignore foreign snapshots without relying on legacy globals.

Symbols (top-level; keep in sync; no ghosts):
- `BackendState` (dataclass): Backend progress state (job/task counters + current image/latent pointers).
- `state` (constant): Global `BackendState` singleton used by services.
"""

from __future__ import annotations

import datetime
import threading
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Optional


_thread_progress_owner_token: ContextVar[str] = ContextVar(
    "backend_state_progress_owner_token",
    default="",
)
_thread_sampling_context: ContextVar[tuple[int, int | None, int, int]] = ContextVar(
    "backend_state_sampling_context",
    default=(0, None, 0, 0),
)


@dataclass
class BackendState:
    """Lightweight, explicit backend state used for progress reporting.

    This replaces any dependence on legacy `modules.shared.state`.
    """

    job_count: int = 0
    job_no: int = 0
    sampling_steps: int | None = 0
    sampling_step: int = 0
    sampling_block_total: int = 0
    sampling_block_index: int = 0
    sampling_owner_token: str = ""
    vae_phase: str = ""
    vae_block_total: int = 0
    vae_block_index: int = 0
    vae_owner_token: str = ""
    vae_sampling_step: int = 0
    vae_sampling_total: int | None = None
    progress_owner_token: str = ""
    time_start: float = 0.0
    textinfo: str = ""
    current_image: Optional[Any] = None
    current_image_owner_token: str = ""
    current_latent: Optional[Any] = None
    id_live_preview: int = 0
    current_image_sampling_step: int = 0
    job: str = ""
    job_timestamp: str = ""
    processing_has_refined_job_count: bool = False
    skipped: bool = False
    interrupted: bool = False
    stopping_generation: bool = False

    _lock: threading.Lock = threading.Lock()

    def start(self, job_count: int, sampling_steps: int | None, progress_owner_token: str | None = None) -> None:
        normalized_total = None if sampling_steps is None else int(sampling_steps)
        normalized_owner = str(progress_owner_token or "")
        with self._lock:
            self.job_count = int(job_count)
            self.job_no = 0
            self.sampling_steps = normalized_total
            self.sampling_step = 0
            self.sampling_block_total = 0
            self.sampling_block_index = 0
            self.sampling_owner_token = normalized_owner
            self.vae_phase = ""
            self.vae_block_total = 0
            self.vae_block_index = 0
            self.vae_sampling_step = 0
            self.vae_sampling_total = None
            self.vae_owner_token = ""
            self.progress_owner_token = normalized_owner
            self.time_start = time.time()
            self.textinfo = ""
            self.current_image = None
            self.current_image_owner_token = ""
            self.current_latent = None
            self.id_live_preview = 0
            self.current_image_sampling_step = 0
            self.job = ""
            self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.processing_has_refined_job_count = False
            self.skipped = False
            self.interrupted = False
            self.stopping_generation = False
        _thread_progress_owner_token.set(normalized_owner)
        _thread_sampling_context.set((0, normalized_total, 0, 0))

    def begin(self, job: str = "(unknown)") -> None:
        with self._lock:
            self.job = job
            self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.current_latent = None
            self.current_image = None
            self.current_image_owner_token = ""
            self.current_image_sampling_step = 0
            self.id_live_preview = 0
            self.sampling_block_total = 0
            self.sampling_block_index = 0
            self.sampling_owner_token = ""
            self.vae_phase = ""
            self.vae_block_total = 0
            self.vae_block_index = 0
            self.vae_sampling_step = 0
            self.vae_sampling_total = None
            self.vae_owner_token = ""
            self.skipped = False
            self.interrupted = False
            self.stopping_generation = False
            self.time_start = time.time()
        _thread_sampling_context.set((0, None, 0, 0))

    def end(self) -> None:
        with self._lock:
            # Preserve final sampling/progress snapshots until the next `start()`
            # so task-stream polling can emit the terminal progress update.
            self.job = ""
            self.job_no = 0
            self.job_count = 0
            self.textinfo = ""
            self.current_latent = None
            self.processing_has_refined_job_count = False

    def clear_progress_snapshot(self) -> None:
        with self._lock:
            self.sampling_step = 0
            self.sampling_steps = 0
            self.sampling_block_total = 0
            self.sampling_block_index = 0
            self.sampling_owner_token = ""
            self.vae_phase = ""
            self.vae_block_total = 0
            self.vae_block_index = 0
            self.vae_sampling_step = 0
            self.vae_sampling_total = None
            self.vae_owner_token = ""
            self.progress_owner_token = ""
            self.current_image = None
            self.current_image_owner_token = ""
            self.current_latent = None
            self.current_image_sampling_step = 0
            self.id_live_preview = 0
        _thread_progress_owner_token.set("")
        _thread_sampling_context.set((0, None, 0, 0))

    def set_progress_owner_token(self, token: str | None) -> None:
        normalized = str(token or "")
        with self._lock:
            self.progress_owner_token = normalized
        _thread_progress_owner_token.set(normalized)
        _thread_sampling_context.set((0, None, 0, 0))

    def current_thread_progress_owner_token(self) -> str:
        return str(_thread_progress_owner_token.get("") or "")

    def current_thread_sampling_context(self) -> tuple[int, int | None, int, int]:
        step, total, block_index, block_total = _thread_sampling_context.get((0, None, 0, 0))
        normalized_total = None if total is None else int(total)
        return int(step), normalized_total, int(block_index), int(block_total)

    def next_job(self) -> None:
        with self._lock:
            if self.job_count > 0:
                self.job_no = min(self.job_no + 1, self.job_count)
            self.sampling_step = 0
            self.sampling_block_total = 0
            self.sampling_block_index = 0
            self.sampling_owner_token = ""
            self.vae_phase = ""
            self.vae_block_total = 0
            self.vae_block_index = 0
            self.vae_sampling_step = 0
            self.vae_sampling_total = None
            self.vae_owner_token = ""
            self.current_image_owner_token = ""
            self.current_image_sampling_step = 0
        _thread_sampling_context.set((0, self.sampling_steps, 0, 0))

    def set_current_image(
        self,
        image: Optional[Any] = None,
        *,
        sampling_step: Optional[int] = None,
        owner_token: str,
    ) -> None:
        with self._lock:
            if image is not None:
                self.current_image = image
                self.id_live_preview += 1
                self.current_image_owner_token = str(owner_token or "")
            if sampling_step is not None:
                self.current_image_sampling_step = int(sampling_step)
                self.current_image_owner_token = str(owner_token or "")

    def set_current_latent(self, latent: Optional[Any]) -> None:
        with self._lock:
            self.current_latent = latent

    def update_sampling(
        self,
        *,
        step: Optional[int] = None,
        total: Optional[int] = None,
        owner_token: str,
    ) -> None:
        normalized_total: int | None
        with self._lock:
            if step is not None:
                self.sampling_step = int(step)
            if total is not None:
                self.sampling_steps = int(total)
            self.sampling_owner_token = str(owner_token or "")
            normalized_total = None if self.sampling_steps is None else int(self.sampling_steps)
            thread_context = (
                int(self.sampling_step),
                normalized_total,
                int(self.sampling_block_index),
                int(self.sampling_block_total),
            )
        _thread_sampling_context.set(thread_context)

    def reset_sampling_blocks(self, *, owner_token: str) -> None:
        with self._lock:
            self.sampling_block_index = 0
            self.sampling_block_total = 0
            self.sampling_owner_token = str(owner_token or "")
            normalized_total = None if self.sampling_steps is None else int(self.sampling_steps)
            thread_context = (
                int(self.sampling_step),
                normalized_total,
                int(self.sampling_block_index),
                int(self.sampling_block_total),
            )
        _thread_sampling_context.set(thread_context)

    def update_sampling_block(
        self,
        *,
        block_index: Optional[int] = None,
        total_blocks: Optional[int] = None,
        owner_token: str,
    ) -> None:
        with self._lock:
            if total_blocks is not None:
                normalized_total = max(0, int(total_blocks))
                self.sampling_block_total = normalized_total
                if normalized_total == 0:
                    self.sampling_block_index = 0
                elif self.sampling_block_index > normalized_total:
                    self.sampling_block_index = normalized_total
            if block_index is not None:
                normalized_index = max(0, int(block_index))
                if self.sampling_block_total > 0:
                    normalized_index = min(normalized_index, self.sampling_block_total)
                self.sampling_block_index = max(self.sampling_block_index, normalized_index)
            self.sampling_owner_token = str(owner_token or "")

    def update_vae_progress(
        self,
        *,
        phase: str,
        block_index: Optional[int] = None,
        total_blocks: Optional[int] = None,
        owner_token: str,
        sampling_step: Optional[int] = None,
        sampling_total: int | None = None,
    ) -> None:
        normalized_phase = str(phase or "").strip().lower()
        if normalized_phase not in {"encode", "decode"}:
            raise ValueError("phase must be 'encode' or 'decode'")
        with self._lock:
            if self.vae_phase != normalized_phase:
                self.vae_phase = normalized_phase
                self.vae_block_index = 0
                self.vae_block_total = 0
                self.vae_sampling_step = 0
                self.vae_sampling_total = None
            if total_blocks is not None:
                normalized_total = max(0, int(total_blocks))
                self.vae_block_total = normalized_total
                if normalized_total == 0:
                    self.vae_block_index = 0
                elif self.vae_block_index > normalized_total:
                    self.vae_block_index = normalized_total
            if block_index is not None:
                normalized_index = max(0, int(block_index))
                if self.vae_block_total > 0:
                    normalized_index = min(normalized_index, self.vae_block_total)
                self.vae_block_index = normalized_index
            if sampling_step is not None:
                self.vae_sampling_step = max(0, int(sampling_step))
            self.vae_sampling_total = None if sampling_total is None else max(0, int(sampling_total))
            self.vae_owner_token = str(owner_token or "")

    def reset_vae_progress(self) -> None:
        with self._lock:
            self.vae_phase = ""
            self.vae_block_index = 0
            self.vae_block_total = 0
            self.vae_sampling_step = 0
            self.vae_sampling_total = None
            self.vae_owner_token = ""

    def tick(
        self,
        *,
        job_no: Optional[int] = None,
        sampling_step: Optional[int] = None,
        owner_token: str,
    ) -> None:
        with self._lock:
            if job_no is not None:
                self.job_no = int(job_no)
            if sampling_step is not None:
                self.sampling_step = int(sampling_step)
            self.sampling_owner_token = str(owner_token or "")
            normalized_total = None if self.sampling_steps is None else int(self.sampling_steps)
            thread_context = (
                int(self.sampling_step),
                normalized_total,
                int(self.sampling_block_index),
                int(self.sampling_block_total),
            )
        _thread_sampling_context.set(thread_context)

    def sampling_snapshot(self) -> tuple[str, int, int | None, int, int]:
        with self._lock:
            return (
                str(self.sampling_owner_token or ""),
                int(self.sampling_step),
                None if self.sampling_steps is None else int(self.sampling_steps),
                int(self.sampling_block_index),
                int(self.sampling_block_total),
            )

    def vae_progress_snapshot(self) -> tuple[str, str, int, int, int, int | None]:
        with self._lock:
            return (
                str(self.vae_owner_token or ""),
                str(self.vae_phase or ""),
                int(self.vae_block_index),
                int(self.vae_block_total),
                int(self.vae_sampling_step),
                None if self.vae_sampling_total is None else int(self.vae_sampling_total),
            )

    def live_preview_snapshot(self) -> tuple[str, int, Any | None, int]:
        with self._lock:
            return (
                str(self.current_image_owner_token or ""),
                int(self.id_live_preview),
                self.current_image,
                int(self.current_image_sampling_step),
            )

    def set_textinfo(self, message: str) -> None:
        with self._lock:
            self.textinfo = message

    def skip(self) -> None:
        with self._lock:
            self.skipped = True

    def interrupt(self) -> None:
        with self._lock:
            self.interrupted = True

    def stop_generating(self) -> None:
        with self._lock:
            self.stopping_generation = True

    def clear_flags(self) -> None:
        with self._lock:
            self.skipped = False
            self.interrupted = False
            self.stopping_generation = False

    @property
    def should_stop(self) -> bool:
        return self.interrupted or self.stopping_generation or self.skipped

    def dict(self) -> dict[str, Any]:
        # drop lock and non-serializable fields
        d = asdict(self)
        d.pop("_lock", None)
        return d


# Global singleton used by services
state = BackendState()

__all__ = ["BackendState", "state"]
