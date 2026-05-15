"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF runtime diagnostics helpers.
Centralizes opt-in debug logging (sigma/timestep parity, CUDA memory snapshots) without scattering ad-hoc logger wrappers.

Symbols (top-level; keep in sync; no ghosts):
- `log_sigmas_enabled` (function): Enables/disables sigma/timestep parity logs via env toggles.
- `log_numerics_enabled` (function): Enables/disables numeric debug logs (NaN/Inf checks + stats) via env toggles.
- `summarize_tensor` (function): Debug helper summarizing tensor-ish objects (shape/dtype/range sample).
- `summarize_numerics` (function): Debug helper summarizing tensor numerics (finite counts + min/max/mean/std).
- `get_logger` (function): Resolves an optional `BackendLoggerProxy` to the canonical WAN22 diagnostics logger.
- `cuda_empty_cache` (function): Best-effort CUDA cache emptying with optional logging.
- `log_cuda_mem` (function): Logs CUDA memory stats for debugging long video runs.
- `log_t_mapping` (function): Logs a coarse mapping of scheduler index → normalized timestep for parity debugging.
- `warn_fallback` (function): Emits an emphatic warning for any runtime fallback (dtype/device/precision/etc).
"""

from __future__ import annotations
from apps.backend.runtime.logging import BackendLoggerProxy, get_backend_logger

import math

import torch

from apps.backend.infra.config.env_flags import env_flag
from apps.backend.runtime.diagnostics.fallback_state import mark_fallback_used
from apps.backend.runtime.memory import memory_management


def log_sigmas_enabled() -> bool:
    return env_flag("CODEX_LOG_SIGMAS", default=False)


def log_numerics_enabled() -> bool:
    return env_flag("CODEX_WAN22_DEBUG_NUMERICS", default=False)


def warn_fallback(logger: BackendLoggerProxy | None, *, component: str, detail: str, reason: str) -> None:
    log = get_logger(logger)
    mark_fallback_used()
    # Intentionally loud: fallbacks must be visible in logs.
    log.warning("!!! [WAN22][FALLBACK] %s: %s (reason=%s) !!!", component, detail, reason)


def summarize_numerics(t: object, *, name: str = "tensor") -> str:
    if not isinstance(t, torch.Tensor):
        return f"{name}: <not-a-tensor>"

    try:
        shape = tuple(int(x) for x in t.shape)
        n = int(t.numel())
        if n == 0:
            return f"{name}: empty shape={shape} dtype={t.dtype} device={t.device}"

        x = t.detach().to(device="cpu", dtype=torch.float32)
        finite = torch.isfinite(x)
        bad = int((~finite).sum().item())
        if bad >= n:
            return f"{name}: ALL_NONFINITE shape={shape} dtype={t.dtype} device={t.device}"

        x0 = x.masked_fill(~finite, 0.0)
        count = float(n - bad)
        sumv = float(x0.sum().item())
        sumsq = float((x0 * x0).sum().item())
        mean = sumv / count
        var = max(0.0, (sumsq / count) - (mean * mean))
        std = math.sqrt(var)

        x_min = float(x.masked_fill(~finite, float("inf")).min().item())
        x_max = float(x.masked_fill(~finite, float("-inf")).max().item())
        return (
            f"{name}: shape={shape} dtype={t.dtype} device={t.device} "
            f"finite={int(count)}/{n} min={x_min:.6g} max={x_max:.6g} mean={mean:.6g} std={std:.6g}"
        )
    except Exception:
        return f"{name}: <stats unavailable>"

def summarize_tensor(t: object, *, window: int = 6) -> str:
    if not isinstance(t, torch.Tensor):
        return "<not-a-tensor>"
    try:
        flat = t.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        n = int(flat.numel())
        if n == 0:
            return "<empty>"
        if n <= window * 2:
            values = flat.tolist()
            return ",".join(f"{float(v):.6g}" for v in values)
        head = [float(v) for v in flat[:window].tolist()]
        tail = [float(v) for v in flat[-window:].tolist()]
        return f"{','.join(f'{v:.6g}' for v in head)},...,{','.join(f'{v:.6g}' for v in tail)}"
    except Exception:
        return "<unavailable>"


def get_logger(logger: BackendLoggerProxy | None) -> BackendLoggerProxy:
    return logger or get_backend_logger("backend.runtime.wan22.gguf")


def cuda_empty_cache(logger: BackendLoggerProxy | None, *, label: str) -> None:
    if not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
        return
    manager = getattr(memory_management, "manager", None)
    if manager is None or not hasattr(manager, "soft_empty_cache"):
        return
    log = get_logger(logger)
    try:
        alloc_before = int(torch.cuda.memory_allocated() // (1024 * 1024))
        reserved_before = int(torch.cuda.memory_reserved() // (1024 * 1024))
        manager.soft_empty_cache(force=True)
        alloc_after = int(torch.cuda.memory_allocated() // (1024 * 1024))
        reserved_after = int(torch.cuda.memory_reserved() // (1024 * 1024))
        log.info(
            "[wan22.gguf] cuda.gc(%s): alloc %d→%d MB reserved %d→%d MB",
            label,
            alloc_before,
            alloc_after,
            reserved_before,
            reserved_after,
        )
    except Exception:
        # Diagnostics only; keep permissive.
        return


def log_cuda_mem(logger: BackendLoggerProxy | None, *, label: str) -> None:
    log = get_logger(logger)
    if not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
        return
    try:
        alloc = float(torch.cuda.memory_allocated()) / (1024**2)
        reserv = float(torch.cuda.memory_reserved()) / (1024**2)
        total = float(torch.cuda.get_device_properties(0).total_memory) / (1024**2)
        log.info(
            "[wan22.gguf] %s: cuda mem alloc=%.0fMB reserved=%.0fMB total=%.0fMB",
            label,
            alloc,
            reserv,
            total,
        )
    except Exception:
        log.debug("[wan22.gguf] failed to read cuda memory stats", exc_info=True)


def log_t_mapping(
    scheduler: object,
    timesteps: object,
    *,
    label: str,
    logger: BackendLoggerProxy | None,
) -> None:
    log = get_logger(logger)
    try:
        n = len(timesteps)
        idxs = [0, max(0, n // 2 - 1), n - 1]
        vals: list[float] = []
        sigmas = getattr(scheduler, "sigmas", None)
        for i in idxs:
            sig_ok = bool(sigmas is not None and len(sigmas) in (n, n + 1))
            if sig_ok:
                s = float(sigmas[i])
                s_min = float(sigmas[-1])
                s_max = float(sigmas[0])
                t = max(0.0, min(1.0, (s - s_min) / (s_max - s_min))) if (s_max - s_min) > 0 else 0.0
            else:
                t = 1.0 - (float(i) / float(max(1, n - 1)))
            vals.append(float(t))
        log.info(
            "[wan22.gguf] t-map(%s): t0=%.4f tmid=%.4f tend=%.4f (sigmas=%s)",
            label,
            vals[0],
            vals[1],
            vals[2],
            bool(sigmas is not None and len(sigmas) in (n, n + 1)),
        )
    except Exception:
        log.debug("[wan22.gguf] failed to log timestep mapping", exc_info=True)
