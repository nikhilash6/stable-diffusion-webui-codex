"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Global, opt-in runtime profiler (CPU/CUDA + transfer attribution) with trace export.
Provides an env-driven wrapper around `torch.profiler` that can be installed at shared runtime seams (sampling loop, ops, workflows) to
produce Chrome-trace output (Perfetto) and a compact summary with totals for CPU↔GPU transfers (`Memcpy HtoD/DtoH`) and cast/move ops.
Profiler activation accepts either `CODEX_PROFILE=1` (legacy) or `CODEX_TRACE_PROFILER=1` (launcher trace toggle).

Symbols (top-level; keep in sync; no ghosts):
- `ProfilerConfig` (dataclass): Env-driven config for the global profiler.
- `GlobalProfiler` (class): Profiler controller with `profile_run(...)` and `section(...)` context managers.
- `profiler` (instance): Global profiler instance (thread-local active state; process-wide exclusivity when enabled).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import datetime as _datetime
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

from apps.backend.infra.config.env_flags import env_flag, env_int
from apps.backend.infra.config.repo_root import get_repo_root

_log = get_backend_logger("backend.profiler")


@dataclass(frozen=True, slots=True)
class ProfilerConfig:
    enabled: bool
    trace: bool
    record_shapes: bool
    profile_memory: bool
    with_stack: bool
    top_n: int
    max_steps: int

    @classmethod
    def from_env(cls) -> "ProfilerConfig":
        enabled = env_flag("CODEX_TRACE_PROFILER", False) or env_flag("CODEX_PROFILE", False)
        trace = env_flag("CODEX_PROFILE_TRACE", True)
        record_shapes = env_flag("CODEX_PROFILE_RECORD_SHAPES", False)
        profile_memory = env_flag("CODEX_PROFILE_PROFILE_MEMORY", True)
        with_stack = env_flag("CODEX_PROFILE_WITH_STACK", False)
        top_n = env_int("CODEX_PROFILE_TOP_N", 25, min_value=1, max_value=500)
        max_steps = env_int("CODEX_PROFILE_MAX_STEPS", 0, min_value=0, max_value=10_000)
        return cls(
            enabled=enabled,
            trace=trace,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            top_n=top_n,
            max_steps=max_steps,
        )


def _safe_name(name: str) -> str:
    raw = str(name or "profile").strip()
    if not raw:
        raw = "profile"
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in raw)


def _get_logs_dir() -> Path:
    base = get_repo_root()
    out = base / "logs" / "profiler"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _ms(us: float) -> float:
    try:
        return float(us) / 1000.0
    except Exception:
        return 0.0


def _iter_key_averages(averages: Iterable[object]) -> Iterator[object]:
    # Defensive: torch profiler returns an iterable of FunctionEventAvg.
    for item in averages:
        yield item


def _compute_transfer_totals_ms(key_averages: Iterable[object]) -> dict[str, float]:
    totals_us = {
        "memcpy_htod_us": 0.0,
        "memcpy_dtoh_us": 0.0,
        "memcpy_dtod_us": 0.0,
        "memcpy_other_us": 0.0,
        "aten_to_us": 0.0,
    }

    for item in _iter_key_averages(key_averages):
        key = str(getattr(item, "key", "") or "")
        key_l = key.lower()
        cuda_us = float(getattr(item, "self_cuda_time_total", 0.0) or 0.0)
        cpu_us = float(getattr(item, "self_cpu_time_total", 0.0) or 0.0)

        if key_l in {"aten::to", "aten::_to_copy"}:
            totals_us["aten_to_us"] += cpu_us

        if "memcpy" in key_l:
            if "htod" in key_l:
                totals_us["memcpy_htod_us"] += cuda_us
            elif "dtoh" in key_l:
                totals_us["memcpy_dtoh_us"] += cuda_us
            elif "dtod" in key_l:
                totals_us["memcpy_dtod_us"] += cuda_us
            else:
                totals_us["memcpy_other_us"] += cuda_us

    return {k.replace("_us", "_ms"): _ms(v) for k, v in totals_us.items()}


class _ActiveProfilerState:
    def __init__(self, *, config: ProfilerConfig, name: str, meta: Mapping[str, Any] | None) -> None:
        self.config = config
        self.name = name
        self.meta = dict(meta or {})
        self.profile = None
        self.trace_path: Path | None = None
        self.summary_path: Path | None = None


class GlobalProfiler:
    """Global, opt-in torch profiler wrapper.

    Notes:
    - Thread-local active state: `section(...)` is a no-op outside an active `profile_run(...)`.
    - Process-wide exclusivity when enabled: we fail loud if two runs attempt to profile concurrently.
    """

    def __init__(self) -> None:
        self._local = threading.local()
        self._active_lock = threading.Lock()

    def config(self) -> ProfilerConfig:
        state: _ActiveProfilerState | None = getattr(self._local, "state", None)
        if state is not None:
            return state.config
        return ProfilerConfig.from_env()

    @property
    def enabled(self) -> bool:
        return self.config().enabled

    def should_profile_step(self, step_index: int) -> bool:
        cfg = self.config()
        if not cfg.enabled:
            return False
        if cfg.max_steps <= 0:
            return True
        return int(step_index) < int(cfg.max_steps)

    @contextmanager
    def profile_run(self, name: str, *, meta: Mapping[str, Any] | None = None):
        cfg = ProfilerConfig.from_env()
        if not cfg.enabled:
            yield
            return

        if not self._active_lock.acquire(blocking=False):
            raise RuntimeError("CODEX_PROFILE=1 only supports one concurrent profiled run per process.")

        state = _ActiveProfilerState(config=cfg, name=str(name or "profile"), meta=meta)
        self._local.state = state
        run_exc: BaseException | None = None

        try:
            self._start_profiler(state)
            yield
        except BaseException as exc:
            run_exc = exc
            raise
        finally:
            try:
                self._stop_and_report(state)
            except BaseException as exc:
                if run_exc is not None:
                    _log.error(
                        "[profile] Failed to finalize profiler output; preserving the original exception. Error: %s",
                        exc,
                        exc_info=True,
                    )
                else:
                    raise
            finally:
                self._local.state = None
                self._active_lock.release()

    @contextmanager
    def section(self, name: str):
        state: _ActiveProfilerState | None = getattr(self._local, "state", None)
        if state is None or state.profile is None:
            yield
            return
        try:
            import torch
        except Exception as exc:  # pragma: no cover - torch is required at runtime
            raise RuntimeError(f"CODEX_PROFILE requires torch, but torch import failed: {exc}") from exc
        with torch.profiler.record_function(str(name)):
            yield

    def step(self) -> None:
        state: _ActiveProfilerState | None = getattr(self._local, "state", None)
        if state is None or state.profile is None:
            return
        try:
            state.profile.step()
        except Exception:
            # Best-effort only; the profiler is still useful without explicit step marks.
            pass

    # ------------------------------------------------------------------ internals

    def _start_profiler(self, state: _ActiveProfilerState) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - torch is required at runtime
            raise RuntimeError(f"CODEX_PROFILE requires torch, but torch import failed: {exc}") from exc

        try:
            from torch.profiler import ProfilerActivity
        except Exception as exc:
            raise RuntimeError(f"CODEX_PROFILE requires torch.profiler, but import failed: {exc}") from exc

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        logs_dir = _get_logs_dir()
        timestamp = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = _safe_name(state.name)
        base = logs_dir / f"{timestamp}-{safe}"
        state.trace_path = base.with_suffix(".json")
        state.summary_path = base.with_suffix(".txt")

        # Avoid leaking huge traces accidentally: log the config loudly.
        _log.info(
            "[profile] enabled=1 trace=%s record_shapes=%s profile_memory=%s with_stack=%s top_n=%d max_steps=%d name=%s",
            int(state.config.trace),
            int(state.config.record_shapes),
            int(state.config.profile_memory),
            int(state.config.with_stack),
            state.config.top_n,
            state.config.max_steps,
            state.name,
        )

        # Lazy: only instantiate profiler when enabled.
        schedule = None
        if state.config.max_steps > 0:
            # Requires callers to call `profiler.step()` per iteration (sampling driver does this).
            schedule = torch.profiler.schedule(wait=0, warmup=0, active=state.config.max_steps, repeat=1)
        state.profile = torch.profiler.profile(
            activities=activities,
            record_shapes=state.config.record_shapes,
            profile_memory=state.config.profile_memory,
            with_stack=state.config.with_stack,
            schedule=schedule,
        )
        state.profile.__enter__()

    def _stop_and_report(self, state: _ActiveProfilerState) -> None:
        prof = state.profile
        if prof is None:
            return

        try:
            prof.__exit__(None, None, None)
        finally:
            state.profile = None

        trace_path = state.trace_path
        summary_path = state.summary_path
        if trace_path is None or summary_path is None:
            return

        try:
            key_avg = prof.key_averages()
        except Exception as exc:
            raise RuntimeError(f"CODEX_PROFILE failed to read profiler results: {exc}") from exc

        transfers = _compute_transfer_totals_ms(key_avg)

        # Write summary first (helps even when trace export fails).
        try:
            cpu_table = key_avg.table(sort_by="self_cpu_time_total", row_limit=state.config.top_n)
            cuda_table = key_avg.table(sort_by="self_cuda_time_total", row_limit=state.config.top_n)
            lines = []
            lines.append(f"name: {state.name}")
            if state.meta:
                lines.append(f"meta: {dict(state.meta)}")
            lines.append("")
            lines.append("transfer_totals_ms:")
            for k, v in sorted(transfers.items()):
                lines.append(f"  {k}: {v:.3f}")
            lines.append("")
            lines.append("top_by_self_cuda_time:")
            lines.append(cuda_table)
            lines.append("")
            lines.append("top_by_self_cpu_time:")
            lines.append(cpu_table)
            lines.append("")
            summary_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(f"CODEX_PROFILE failed to write summary {summary_path}: {exc}") from exc

        if state.config.trace:
            try:
                prof.export_chrome_trace(str(trace_path))
            except Exception as exc:
                raise RuntimeError(f"CODEX_PROFILE failed to export chrome trace {trace_path}: {exc}") from exc

        _log.info("[profile] summary: %s", summary_path)
        if state.config.trace:
            _log.info("[profile] chrome trace: %s", trace_path)
            _log.info("[profile] view: https://ui.perfetto.dev/ (drag & drop the JSON)")
        _log.info(
            "[profile] transfers ms: htod=%.3f dtoh=%.3f dtod=%.3f other=%.3f aten_to_cpu=%.3f",
            float(transfers.get("memcpy_htod_ms", 0.0)),
            float(transfers.get("memcpy_dtoh_ms", 0.0)),
            float(transfers.get("memcpy_dtod_ms", 0.0)),
            float(transfers.get("memcpy_other_ms", 0.0)),
            float(transfers.get("aten_to_ms", 0.0)),
        )


profiler = GlobalProfiler()

__all__ = [
    "GlobalProfiler",
    "ProfilerConfig",
    "profiler",
]
