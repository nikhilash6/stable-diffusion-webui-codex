"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared streaming controller core (segment device placement + transfer stats).
Implements a generic controller used by multiple runtime families (e.g., Flux and WAN22) to keep streaming semantics identical and avoid
copy/paste drift, including deterministic state reset between generations and stats-preserving residency clear helpers for wrapper cleanup.

Symbols (top-level; keep in sync; no ghosts):
- `StreamingPolicy` (enum): Streaming policy (`naive`/`window`/`aggressive`) controlling segment residency.
- `TransferStats` (dataclass): Tracks CPU↔GPU transfer bytes/counts/time for streaming telemetry.
- `StreamingSegment` (protocol): Minimal runtime protocol for a streamable segment (name/bytes/to_device).
- `StreamingController` (dataclass): Streaming controller operating on `StreamingSegment` objects (reset clears residency/access/segment maps).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Generic, List, Optional, Protocol, TypeVar

import torch


class StreamingPolicy(Enum):
    """Streaming policy determining how segments are moved between devices."""

    NAIVE = "naive"
    """Load segment → forward → unload. Simplest, lowest VRAM, slowest."""

    WINDOW = "window"
    """Keep K most recent segments pinned on compute device, LRU eviction."""

    AGGRESSIVE = "aggressive"
    """Naive with async prefetch of next segment during current forward."""


@dataclass
class TransferStats:
    """Statistics for CPU↔GPU transfers during streaming."""

    bytes_to_gpu: int = 0
    bytes_to_cpu: int = 0
    transfers_to_gpu: int = 0
    transfers_to_cpu: int = 0
    total_time_ms: float = 0.0

    def record_to_gpu(self, bytes_count: int, time_ms: float) -> None:
        self.bytes_to_gpu += bytes_count
        self.transfers_to_gpu += 1
        self.total_time_ms += time_ms

    def record_to_cpu(self, bytes_count: int, time_ms: float) -> None:
        self.bytes_to_cpu += bytes_count
        self.transfers_to_cpu += 1
        self.total_time_ms += time_ms

    def summary(self) -> Dict[str, float]:
        return {
            "to_gpu_mb": self.bytes_to_gpu / (1024 * 1024),
            "to_cpu_mb": self.bytes_to_cpu / (1024 * 1024),
            "transfers_to_gpu": float(self.transfers_to_gpu),
            "transfers_to_cpu": float(self.transfers_to_cpu),
            "total_time_ms": self.total_time_ms,
        }


class StreamingSegment(Protocol):
    """Minimal interface required by the shared streaming controller."""

    name: str
    param_bytes: int

    def to_device(self, device: torch.device, *, non_blocking: bool = False) -> None: ...


SegmentT = TypeVar("SegmentT", bound=StreamingSegment)


@dataclass
class StreamingController(Generic[SegmentT]):
    """Memory controller managing segment placement for streaming."""

    storage_device: torch.device
    compute_device: torch.device
    policy: StreamingPolicy = StreamingPolicy.NAIVE
    window_size: int = 2
    non_blocking: bool = True
    logger: logging.Logger = field(
        default_factory=lambda: get_backend_logger("backend.runtime.streaming.controller"),
        repr=False,
    )

    _on_gpu: set[str] = field(default_factory=set, repr=False)
    _access_order: List[str] = field(default_factory=list, repr=False)
    _stats: TransferStats = field(default_factory=TransferStats, repr=False)
    _prefetch_segment: Optional[SegmentT] = field(default=None, repr=False)
    _segments_by_name: Dict[str, SegmentT] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.storage_device, str):
            self.storage_device = torch.device(self.storage_device)
        if isinstance(self.compute_device, str):
            self.compute_device = torch.device(self.compute_device)

    @property
    def stats(self) -> TransferStats:
        return self._stats

    def reset(self) -> None:
        """Reset controller state between generations."""
        self.clear_residency()
        self.logger.debug("Controller state reset")

    def clear_residency(self) -> None:
        """Clear tracked residency/access state without touching transfer stats."""
        self._on_gpu.clear()
        self._access_order.clear()
        self._segments_by_name.clear()
        self._prefetch_segment = None
        self.logger.debug("Controller residency cleared")

    def reset_stats(self) -> None:
        self._stats = TransferStats()

    def is_on_gpu(self, segment: SegmentT) -> bool:
        return segment.name in self._on_gpu

    def ensure_on_device(self, segment: SegmentT) -> None:
        """Ensure segment is on compute device (GPU)."""
        self._segments_by_name[segment.name] = segment

        if self.is_on_gpu(segment):
            if segment.name in self._access_order:
                self._access_order.remove(segment.name)
            self._access_order.append(segment.name)
            return

        start = time.perf_counter()
        segment.to_device(self.compute_device, non_blocking=self.non_blocking)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._on_gpu.add(segment.name)
        self._access_order.append(segment.name)
        self._stats.record_to_gpu(segment.param_bytes, elapsed_ms)

        self.logger.debug(
            "Loaded segment '%s' to compute device (%.2f MB, %.1f ms)",
            segment.name,
            segment.param_bytes / (1024 * 1024),
            elapsed_ms,
        )

    def maybe_evict(self, segment: SegmentT, *, force: bool = False) -> None:
        """Potentially evict segment back to storage device."""
        if not self.is_on_gpu(segment) and not force:
            return

        if self.policy == StreamingPolicy.NAIVE or force:
            self._evict_segment(segment)
        elif self.policy == StreamingPolicy.WINDOW:
            if len(self._on_gpu) > self.window_size:
                self._evict_lru()
        elif self.policy == StreamingPolicy.AGGRESSIVE:
            self._evict_segment(segment)

    def _evict_segment(self, segment: SegmentT) -> None:
        if not self.is_on_gpu(segment):
            return

        start = time.perf_counter()
        segment.to_device(self.storage_device, non_blocking=self.non_blocking)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._on_gpu.discard(segment.name)
        if segment.name in self._access_order:
            self._access_order.remove(segment.name)
        self._stats.record_to_cpu(segment.param_bytes, elapsed_ms)

        self.logger.debug(
            "Evicted segment '%s' to storage device (%.2f MB, %.1f ms)",
            segment.name,
            segment.param_bytes / (1024 * 1024),
            elapsed_ms,
        )

    def _evict_lru(self) -> None:
        if not self._access_order:
            return

        for name in list(self._access_order):
            if name not in self._on_gpu:
                continue
            segment = self._segments_by_name.get(name)
            if segment is None:
                self._on_gpu.discard(name)
                self._access_order.remove(name)
                self.logger.debug("LRU evicted segment '%s' (no segment object cached)", name)
                return
            self._evict_segment(segment)
            return

    def prefetch_next(self, next_segment: Optional[SegmentT]) -> None:
        """Hint to prefetch next segment (AGGRESSIVE policy)."""
        if self.policy != StreamingPolicy.AGGRESSIVE:
            return
        if next_segment is None or self.is_on_gpu(next_segment):
            return

        self._segments_by_name[next_segment.name] = next_segment
        self._prefetch_segment = next_segment
        start = time.perf_counter()
        next_segment.to_device(self.compute_device, non_blocking=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._on_gpu.add(next_segment.name)
        if next_segment.name in self._access_order:
            self._access_order.remove(next_segment.name)
        self._access_order.append(next_segment.name)
        self._stats.record_to_gpu(next_segment.param_bytes, elapsed_ms)
        self.logger.debug(
            "Prefetched segment '%s' to compute device (%.2f MB, %.1f ms)",
            next_segment.name,
            next_segment.param_bytes / (1024 * 1024),
            elapsed_ms,
        )

    def evict_all(self) -> None:
        """Evict all known segments from compute device."""
        for name in list(self._on_gpu):
            segment = self._segments_by_name.get(name)
            if segment is None:
                continue
            self._evict_segment(segment)
        self._prefetch_segment = None
        self.logger.debug("All segments evicted")
