"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Generic SRAM/shared-memory attention runtime bridge.
Provides fail-loud mode parsing, retired-contract rejection, extension load/build warmup,
pre-shaped attention attempt dispatch over supported non-overlapping dense `[B,H,S,D]`
layouts with contiguous head-dim lanes, layout-preserving output for the returned tensor,
and generic runtime metrics for the versioned `attention_sram_v1` CUDA backend.

Symbols (top-level; keep in sync; no ghosts):
- `SramAttentionMode` (enum): Runtime mode for generic SRAM attention dispatch (`off|auto|force`).
- `SramAttentionContractError` (class): Fail-loud contract error with stable `code` field.
- `SramAttentionAttemptResult` (dataclass): Result envelope for non-forced attempts (`output` or `reason_code`).
- `SramAttentionWarmupStatus` (dataclass): Load-time warmup result with truthful `loaded` vs `ready` fields.
- `parse_sram_attention_mode` (function): Parses and validates a SRAM attention mode string.
- `resolve_effective_sram_attention_mode` (function): Resolves SRAM attention mode from override/env with retired-env rejection.
- `is_extension_available` (function): Returns whether generic SRAM attention ops are loaded and registered.
- `is_rope_helper_available` (function): Returns whether the optional generic in-place RoPE helper op is available for the current SRAM mode.
- `last_extension_error` (function): Returns the last extension load/build error details.
- `warmup_extension_for_load` (function): Triggers extension load/build during model-load seam and performs a narrow readiness smoke call.
- `warmup_extension_for_diagnostics` (function): Triggers a diagnostics-scoped build-enabled extension load retry before the same readiness smoke call.
- `try_attention_pre_shaped` (function): Attempts generic SRAM attention dispatch for supported pre-shaped `[B,H,S,D]` Q/K/V tensors and returns output with `q` layout preserved.
- `sram_attention_runtime_metrics_reset` (function): Resets per-run SRAM runtime metrics state.
- `sram_attention_runtime_metrics_is_active` (function): Returns whether a SRAM runtime metrics context is active.
- `sram_attention_runtime_metrics_set_stage` (function): Sets the active sampling stage label for SRAM counter attribution.
- `sram_attention_runtime_metrics_log_summary` (function): Emits end-of-run SRAM dispatch summary and optionally resets run state.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import importlib
import logging
import os
import subprocess
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch

logger = get_backend_logger("backend.runtime.attention.sram")

_MODE_ENV_KEY = "CODEX_ATTENTION_SRAM_MODE"
_JIT_ENV_KEY = "CODEX_ATTENTION_SRAM_JIT"
_EXTENSION_MODULE_NAME = "attention_sram_v1_cuda"
_REQUIRED_EXTENSION_ABI = 1
_RETIRED_ENV_KEYS = (
    "CODEX_WAN22_FUSED_ATTN_V1_MODE",
    "CODEX_WAN_FUSED_V1_ATTN_CORE",
)

E_SRAM_ATTENTION_DISABLED = "E_SRAM_ATTENTION_DISABLED"
E_SRAM_ATTENTION_EXTENSION_UNAVAILABLE = "E_SRAM_ATTENTION_EXTENSION_UNAVAILABLE"
E_SRAM_ATTENTION_KERNEL_RUNTIME_ERROR = "E_SRAM_ATTENTION_KERNEL_RUNTIME_ERROR"
E_SRAM_ATTENTION_INVALID_MODE = "E_SRAM_ATTENTION_INVALID_MODE"
E_SRAM_ATTENTION_DEVICE_UNSUPPORTED = "E_SRAM_ATTENTION_DEVICE_UNSUPPORTED"
E_SRAM_ATTENTION_DTYPE_UNSUPPORTED = "E_SRAM_ATTENTION_DTYPE_UNSUPPORTED"
E_SRAM_ATTENTION_HEAD_DIM_UNSUPPORTED = "E_SRAM_ATTENTION_HEAD_DIM_UNSUPPORTED"
E_SRAM_ATTENTION_INVALID_SHAPE = "E_SRAM_ATTENTION_INVALID_SHAPE"
E_SRAM_ATTENTION_LAYOUT_UNSUPPORTED = "E_SRAM_ATTENTION_LAYOUT_UNSUPPORTED"
E_SRAM_ATTENTION_INVALID_ENV = "E_SRAM_ATTENTION_INVALID_ENV"


@dataclass(frozen=True, slots=True)
class _AttemptError:
    stage: str
    message: str


@dataclass(frozen=True, slots=True)
class SramAttentionAttemptResult:
    output: torch.Tensor | None
    reason_code: str | None
    reason_detail: str | None = None


@dataclass(frozen=True, slots=True)
class SramAttentionWarmupStatus:
    mode: "SramAttentionMode"
    build_enabled: bool
    attempted: bool
    loaded: bool
    ready: bool
    detail: str | None = None


class SramAttentionMode(str, Enum):
    OFF = "off"
    AUTO = "auto"
    FORCE = "force"


class SramAttentionContractError(RuntimeError):
    def __init__(self, *, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


@dataclass(slots=True)
class _SramAttentionRuntimeMetrics:
    run_label: str
    mode: str
    attempts: int = 0
    hits: int = 0
    fallback_by_reason: dict[str, int] = field(default_factory=dict)
    attempts_by_stage: dict[str, int] = field(default_factory=dict)
    hits_by_stage: dict[str, int] = field(default_factory=dict)
    fallback_by_stage_reason: dict[str, int] = field(default_factory=dict)


_ext: Any | None = None
_last_error: str | None = None
_attempt_errors: list[_AttemptError] = []
_max_load_attempt_level = -1
_rope_helper_available: bool | None = None
_rope_helper_cache_attempt_level = -1
_runtime_metrics_var: ContextVar[_SramAttentionRuntimeMetrics | None] = ContextVar(
    "sram_attention_runtime_metrics",
    default=None,
)
_runtime_stage_var: ContextVar[str] = ContextVar(
    "sram_attention_runtime_stage",
    default="unset",
)


def _kernel_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "kernels" / "attention_sram_v1"


def _setup_path() -> Path:
    return _kernel_dir() / "setup.py"


def _fail(*, code: str, message: str) -> None:
    raise SramAttentionContractError(code=code, message=message)


def _retired_env_values() -> dict[str, str]:
    hits: dict[str, str] = {}
    for key in _RETIRED_ENV_KEYS:
        value = os.environ.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            hits[key] = normalized
    return hits


def _raise_on_retired_env_keys() -> None:
    hits = _retired_env_values()
    if not hits:
        return
    details = ", ".join(f"{key}={value!r}" for key, value in sorted(hits.items()))
    _fail(
        code=E_SRAM_ATTENTION_INVALID_ENV,
        message=(
            "Retired WAN-only SRAM attention env keys are no longer accepted. "
            f"Remove: {details}. Use {_MODE_ENV_KEY}=off|auto|force instead."
        ),
    )


def _normalize_mode_alias(raw: str, *, field_name: str) -> SramAttentionMode:
    normalized = str(raw).strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "off": "off",
        "1": "auto",
        "true": "auto",
        "yes": "auto",
        "on": "auto",
        "auto": "auto",
        "force": "force",
        "required": "force",
    }
    mapped = aliases.get(normalized)
    if mapped is None:
        _fail(
            code=E_SRAM_ATTENTION_INVALID_MODE,
            message=f"{field_name} must be one of off|auto|force; got {raw!r}.",
        )
    return SramAttentionMode(mapped)


def parse_sram_attention_mode(value: str, *, field_name: str) -> SramAttentionMode:
    return _normalize_mode_alias(value, field_name=field_name)


def resolve_effective_sram_attention_mode(mode_override: str | SramAttentionMode | None) -> SramAttentionMode:
    _raise_on_retired_env_keys()
    if isinstance(mode_override, SramAttentionMode):
        return mode_override
    raw = mode_override if mode_override is not None else os.environ.get(_MODE_ENV_KEY, SramAttentionMode.OFF.value)
    return _normalize_mode_alias(str(raw), field_name=_MODE_ENV_KEY if mode_override is None else "mode")


def _jit_build_enabled() -> bool:
    return str(os.environ.get(_JIT_ENV_KEY, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _format_counter_map(counter: dict[str, int]) -> str:
    if not counter:
        return "-"
    return ",".join(f"{key}:{int(counter[key])}" for key in sorted(counter))


def _set_attempt_error(stage: str, ex: Exception) -> None:
    global _last_error
    message = f"{type(ex).__name__}: {ex}"
    _attempt_errors.append(_AttemptError(stage=stage, message=message))
    _last_error = "\n".join(f"{entry.stage}: {entry.message}" for entry in _attempt_errors)


def _reset_attempt_errors() -> None:
    global _last_error
    _attempt_errors.clear()
    _last_error = None


def _purge_extension_module_cache() -> None:
    sys.modules.pop(_EXTENSION_MODULE_NAME, None)
    importlib.invalidate_caches()


def _ops_namespace() -> Any | None:
    ops = getattr(torch, "ops", None)
    if ops is None or not hasattr(ops, "attention_sram_v1"):
        return None
    return getattr(ops, "attention_sram_v1")


def _has_ops() -> bool:
    namespace = _ops_namespace()
    return namespace is not None and hasattr(namespace, "attn_fwd")


def _has_rope_helper() -> bool:
    namespace = _ops_namespace()
    return namespace is not None and hasattr(namespace, "rope_blhd_")


def _ensure_extension_abi(module: Any) -> None:
    abi = getattr(module, "ATTENTION_SRAM_V1_ABI", None)
    try:
        abi_value = int(abi)
    except Exception as ex:
        raise RuntimeError(
            "incompatible attention_sram_v1 extension ABI: "
            f"required={_REQUIRED_EXTENSION_ABI} got={abi!r}"
        ) from ex
    if abi_value != int(_REQUIRED_EXTENSION_ABI):
        raise RuntimeError(
            "incompatible attention_sram_v1 extension ABI: "
            f"required={_REQUIRED_EXTENSION_ABI} got={abi_value}"
        )


def _import_extension_from(dir_path: Path) -> Any:
    sys.path.insert(0, str(dir_path))
    try:
        _purge_extension_module_cache()
        return importlib.import_module(_EXTENSION_MODULE_NAME)
    finally:
        try:
            sys.path.remove(str(dir_path))
        except ValueError:
            pass


def _load_attempt_level(*, build: bool) -> int:
    return 1 if build else 0


def _try_load_ext(*, build: bool, force_retry: bool = False) -> None:
    global _ext, _max_load_attempt_level, _rope_helper_available, _rope_helper_cache_attempt_level
    if _ext is not None and _has_ops():
        return
    attempt_level = _load_attempt_level(build=build)
    if not force_retry and _max_load_attempt_level >= attempt_level:
        return
    _raise_on_retired_env_keys()
    _max_load_attempt_level = attempt_level
    _reset_attempt_errors()
    _rope_helper_available = None
    _rope_helper_cache_attempt_level = -1

    try:
        _purge_extension_module_cache()
        loaded = importlib.import_module(_EXTENSION_MODULE_NAME)
        _ensure_extension_abi(loaded)
        if not _has_ops():
            raise RuntimeError("prebuilt module loaded but torch.ops.attention_sram_v1.attn_fwd is missing")
        _ext = loaded
        return
    except Exception as ex:
        _ext = None
        _set_attempt_error("prebuilt", ex)

    ext_dir = _kernel_dir()
    if ext_dir.is_dir():
        try:
            loaded = _import_extension_from(ext_dir)
            _ensure_extension_abi(loaded)
            if not _has_ops():
                raise RuntimeError("in-place module loaded but torch.ops.attention_sram_v1.attn_fwd is missing")
            _ext = loaded
            return
        except Exception as ex:
            _ext = None
            _set_attempt_error("in_place", ex)

    if not build:
        return

    setup_path = _setup_path()
    if not setup_path.is_file():
        _set_attempt_error(
            "jit",
            RuntimeError(
                f"cannot build {_EXTENSION_MODULE_NAME}: missing setup.py at {setup_path}"
            ),
        )
        return

    try:
        subprocess.run(
            [sys.executable, str(setup_path), "build_ext", "--inplace"],
            cwd=str(ext_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        loaded = _import_extension_from(ext_dir)
        _ensure_extension_abi(loaded)
        if not _has_ops():
            raise RuntimeError("jit-built module loaded but torch.ops.attention_sram_v1.attn_fwd is missing")
        _ext = loaded
    except Exception as ex:
        _ext = None
        _set_attempt_error("jit", ex)


def is_extension_available() -> bool:
    _try_load_ext(build=_jit_build_enabled())
    return _ext is not None and _has_ops()


def is_rope_helper_available(*, mode: str | SramAttentionMode | None = None) -> bool:
    global _rope_helper_available, _rope_helper_cache_attempt_level
    sram_mode = resolve_effective_sram_attention_mode(mode)
    if sram_mode is SramAttentionMode.OFF:
        return False
    attempt_level = _load_attempt_level(build=_jit_build_enabled())
    if _rope_helper_available is not None and _rope_helper_cache_attempt_level >= attempt_level:
        return _rope_helper_available
    _try_load_ext(build=bool(attempt_level))
    _rope_helper_available = _has_rope_helper()
    _rope_helper_cache_attempt_level = max(_max_load_attempt_level, attempt_level)
    return _rope_helper_available


def last_extension_error() -> str | None:
    return _last_error


def _smoke_ready_call() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA not available for readiness smoke call."
    namespace = _ops_namespace()
    if namespace is None or not hasattr(namespace, "attn_fwd"):
        return "torch.ops.attention_sram_v1.attn_fwd is missing."
    device = torch.device("cuda", torch.cuda.current_device())
    q = torch.zeros((1, 1, 1, 128), device=device, dtype=torch.float16)
    k = torch.zeros((1, 1, 1, 128), device=device, dtype=torch.float16)
    v = torch.zeros((1, 1, 1, 128), device=device, dtype=torch.float16)
    try:
        out = namespace.attn_fwd(q, k, v, False)
    except Exception as ex:
        return f"{type(ex).__name__}: {ex}"
    if not isinstance(out, torch.Tensor):
        return f"unexpected readiness output type {type(out).__name__}"
    if tuple(out.shape) != (1, 1, 1, 128):
        return f"unexpected readiness output shape {tuple(out.shape)}"
    return None


def _warmup_extension(
    *,
    mode: str | SramAttentionMode | None,
    build_enabled: bool,
    force_retry: bool,
    reason_label: str,
) -> SramAttentionWarmupStatus:
    sram_mode = resolve_effective_sram_attention_mode(mode)
    if sram_mode is SramAttentionMode.OFF:
        logger.info("attention_sram warmup skipped at %s (mode=off)", reason_label)
        return SramAttentionWarmupStatus(
            mode=sram_mode,
            build_enabled=build_enabled,
            attempted=False,
            loaded=False,
            ready=False,
            detail=None,
        )

    logger.info(
        "attention_sram warmup start (%s mode=%s jit_build=%s force_retry=%s)",
        reason_label,
        sram_mode.value,
        build_enabled,
        force_retry,
    )
    _try_load_ext(build=build_enabled, force_retry=force_retry)
    loaded = _ext is not None and _has_ops()
    detail = last_extension_error()
    ready = False
    if loaded:
        smoke_error = _smoke_ready_call()
        if smoke_error is None:
            ready = True
            detail = None
        else:
            detail = smoke_error

    if ready:
        logger.info(
            "attention_sram warmup ready (%s mode=%s jit_build=%s)",
            reason_label,
            sram_mode.value,
            build_enabled,
        )
        return SramAttentionWarmupStatus(
            mode=sram_mode,
            build_enabled=build_enabled,
            attempted=True,
            loaded=True,
            ready=True,
            detail=None,
        )

    logger.warning(
        "attention_sram warmup unavailable (mode=%s jit_build=%s loaded=%s ready=%s detail=%r)",
        sram_mode.value,
        build_enabled,
        loaded,
        ready,
        detail,
    )
    if sram_mode is SramAttentionMode.FORCE:
        _fail(
            code=E_SRAM_ATTENTION_EXTENSION_UNAVAILABLE,
            message=(
                f"SRAM attention force mode requested but extension warmup failed during {reason_label}. "
                f"loaded={loaded} ready={ready} details={detail!r}"
            ),
        )
    return SramAttentionWarmupStatus(
        mode=sram_mode,
        build_enabled=build_enabled,
        attempted=True,
        loaded=loaded,
        ready=ready,
        detail=detail,
    )


def warmup_extension_for_load(mode: str | SramAttentionMode | None = None) -> SramAttentionWarmupStatus:
    return _warmup_extension(
        mode=mode,
        build_enabled=_jit_build_enabled(),
        force_retry=False,
        reason_label="load",
    )


def warmup_extension_for_diagnostics(mode: str | SramAttentionMode | None = None) -> SramAttentionWarmupStatus:
    return _warmup_extension(
        mode=mode,
        build_enabled=True,
        force_retry=True,
        reason_label="diagnostics",
    )


def _ensure_runtime_metrics() -> _SramAttentionRuntimeMetrics:
    metrics = _runtime_metrics_var.get()
    if metrics is None:
        metrics = _SramAttentionRuntimeMetrics(run_label="implicit", mode=SramAttentionMode.OFF.value)
        _runtime_metrics_var.set(metrics)
    return metrics


def sram_attention_runtime_metrics_reset(*, run_label: str, mode: str) -> None:
    _runtime_metrics_var.set(_SramAttentionRuntimeMetrics(run_label=str(run_label), mode=str(mode)))
    _runtime_stage_var.set("unset")


def sram_attention_runtime_metrics_is_active() -> bool:
    return _runtime_metrics_var.get() is not None


def sram_attention_runtime_metrics_set_stage(stage_name: str) -> None:
    _runtime_stage_var.set(str(stage_name or "unset"))


def _record_attempt(*, used: bool, reason_code: str | None) -> None:
    metrics = _runtime_metrics_var.get()
    if metrics is None:
        return
    stage_name = _runtime_stage_var.get() or "unset"
    metrics.attempts += 1
    metrics.attempts_by_stage[stage_name] = int(metrics.attempts_by_stage.get(stage_name, 0)) + 1
    if used:
        metrics.hits += 1
        metrics.hits_by_stage[stage_name] = int(metrics.hits_by_stage.get(stage_name, 0)) + 1
        return
    reason = str(reason_code or "unknown")
    metrics.fallback_by_reason[reason] = int(metrics.fallback_by_reason.get(reason, 0)) + 1
    stage_key = f"{stage_name}:{reason}"
    metrics.fallback_by_stage_reason[stage_key] = int(metrics.fallback_by_stage_reason.get(stage_key, 0)) + 1


def sram_attention_runtime_metrics_log_summary(
    *,
    logger_obj: logging.Logger | None = None,
    reset: bool = False,
    emit_when_idle: bool = True,
) -> None:
    metrics = _runtime_metrics_var.get()
    if metrics is None:
        if reset:
            _runtime_stage_var.set("unset")
        return
    if bool(emit_when_idle) or int(metrics.attempts) > 0:
        log = logger_obj if logger_obj is not None else logger
        log.info(
            "attention_sram.summary run=%s mode=%s attempts=%d hits=%d fallback_by_reason=%s attempts_by_stage=%s hits_by_stage=%s fallback_by_stage_reason=%s",
            str(metrics.run_label),
            str(metrics.mode),
            int(metrics.attempts),
            int(metrics.hits),
            _format_counter_map(metrics.fallback_by_reason),
            _format_counter_map(metrics.attempts_by_stage),
            _format_counter_map(metrics.hits_by_stage),
            _format_counter_map(metrics.fallback_by_stage_reason),
        )
    if reset:
        _runtime_metrics_var.set(None)
        _runtime_stage_var.set("unset")


def _unsupported_result(*, mode: SramAttentionMode, code: str, detail: str) -> SramAttentionAttemptResult:
    _record_attempt(used=False, reason_code=code)
    if mode is SramAttentionMode.FORCE:
        _fail(code=code, message=detail)
    return SramAttentionAttemptResult(output=None, reason_code=code, reason_detail=detail)


def _is_non_overlapping_and_dense_pre_shaped_layout(tensor: torch.Tensor) -> bool:
    if tensor.ndim != 4:
        return False
    sizes = tuple(int(value) for value in tensor.shape)
    strides = tuple(int(value) for value in tensor.stride())
    for size, stride in zip(sizes, strides):
        if stride < 0:
            return False
        if stride == 0 and size != 1:
            return False
    expected_stride = 1
    for stride, size in sorted(zip(strides, sizes)):
        if size <= 1:
            continue
        if stride != expected_stride:
            return False
        expected_stride *= size
    return True


def _supports_pre_shaped_layout(tensor: torch.Tensor) -> bool:
    if tensor.ndim != 4:
        return False
    strides = tuple(int(value) for value in tensor.stride())
    return strides[3] == 1 and _is_non_overlapping_and_dense_pre_shaped_layout(tensor)


def _validate_pre_shaped_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
) -> tuple[str, str] | None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return E_SRAM_ATTENTION_INVALID_SHAPE, (
            "SRAM attention requires pre-shaped Q/K/V tensors with rank 4 `[B,H,S,D]`; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    if q.device != k.device or q.device != v.device:
        return E_SRAM_ATTENTION_INVALID_SHAPE, (
            "SRAM attention requires Q/K/V on the same device; "
            f"got q={q.device} k={k.device} v={v.device}."
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return E_SRAM_ATTENTION_INVALID_SHAPE, (
            "SRAM attention requires Q/K/V with the same dtype; "
            f"got q={q.dtype} k={k.dtype} v={v.dtype}."
        )
    if tuple(q.shape[:2]) != tuple(k.shape[:2]) or tuple(q.shape[:2]) != tuple(v.shape[:2]):
        return E_SRAM_ATTENTION_INVALID_SHAPE, (
            "SRAM attention requires matching batch/head dimensions; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    if int(q.shape[0]) <= 0 or int(q.shape[1]) <= 0:
        return E_SRAM_ATTENTION_INVALID_SHAPE, (
            "SRAM attention requires batch > 0 and heads > 0; "
            f"got q={tuple(q.shape)}."
        )
    if int(k.shape[2]) != int(v.shape[2]):
        return E_SRAM_ATTENTION_INVALID_SHAPE, (
            "SRAM attention requires matching K/V sequence length; "
            f"got k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    if int(k.shape[2]) <= 0:
        return E_SRAM_ATTENTION_INVALID_SHAPE, (
            "SRAM attention requires K/V sequence length > 0; "
            f"got k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    if int(q.shape[3]) != int(k.shape[3]) or int(q.shape[3]) != int(v.shape[3]):
        return E_SRAM_ATTENTION_INVALID_SHAPE, (
            "SRAM attention requires matching head_dim across Q/K/V; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    if q.device.type != "cuda":
        return E_SRAM_ATTENTION_DEVICE_UNSUPPORTED, (
            f"SRAM attention supports CUDA tensors only; got device={q.device}."
        )
    if q.dtype != torch.float16:
        return E_SRAM_ATTENTION_DTYPE_UNSUPPORTED, (
            f"SRAM attention supports float16 only in the current slice; got dtype={q.dtype}."
        )
    if int(q.shape[3]) != 128:
        return E_SRAM_ATTENTION_HEAD_DIM_UNSUPPORTED, (
            f"SRAM attention supports head_dim=128 only in the current slice; got head_dim={int(q.shape[3])}."
        )
    if not _supports_pre_shaped_layout(q) or not _supports_pre_shaped_layout(k) or not _supports_pre_shaped_layout(v):
        return E_SRAM_ATTENTION_LAYOUT_UNSUPPORTED, (
            "SRAM attention currently supports non-overlapping dense `[B,H,S,D]` layouts with contiguous head_dim lanes "
            "(stride[-1] == 1); "
            f"got q_stride={tuple(int(value) for value in q.stride())} "
            f"k_stride={tuple(int(value) for value in k.stride())} "
            f"v_stride={tuple(int(value) for value in v.stride())}."
        )
    if is_causal and int(q.shape[2]) > ((2**31) - 1):
        return E_SRAM_ATTENTION_INVALID_SHAPE, (
            "SRAM attention causal path requires q sequence length to fit int32; "
            f"got q={tuple(q.shape)}."
        )
    return None


def try_attention_pre_shaped(
    *,
    mode: str | SramAttentionMode | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> SramAttentionAttemptResult:
    sram_mode = resolve_effective_sram_attention_mode(mode)
    if sram_mode is SramAttentionMode.OFF:
        _record_attempt(used=False, reason_code=E_SRAM_ATTENTION_DISABLED)
        return SramAttentionAttemptResult(
            output=None,
            reason_code=E_SRAM_ATTENTION_DISABLED,
            reason_detail="SRAM attention mode is off.",
        )

    validation_error = _validate_pre_shaped_inputs(q, k, v, is_causal=bool(is_causal))
    if validation_error is not None:
        code, detail = validation_error
        return _unsupported_result(mode=sram_mode, code=code, detail=detail)

    _try_load_ext(build=_jit_build_enabled())
    if _ext is None or not _has_ops():
        return _unsupported_result(
            mode=sram_mode,
            code=E_SRAM_ATTENTION_EXTENSION_UNAVAILABLE,
            detail=(
                "SRAM attention extension is not available for pre-shaped dispatch. "
                f"details={last_extension_error()!r}"
            ),
        )

    attn_fwd = getattr(_ops_namespace(), "attn_fwd")

    try:
        out = attn_fwd(q, k, v, bool(is_causal))
    except Exception as ex:
        _record_attempt(used=False, reason_code=E_SRAM_ATTENTION_KERNEL_RUNTIME_ERROR)
        if sram_mode is SramAttentionMode.FORCE:
            _fail(
                code=E_SRAM_ATTENTION_KERNEL_RUNTIME_ERROR,
                message=f"SRAM attention kernel runtime failure: {type(ex).__name__}: {ex}",
            )
        return SramAttentionAttemptResult(
            output=None,
            reason_code=E_SRAM_ATTENTION_KERNEL_RUNTIME_ERROR,
            reason_detail=f"{type(ex).__name__}: {ex}",
        )

    _record_attempt(used=True, reason_code=None)
    return SramAttentionAttemptResult(output=out, reason_code=None, reason_detail=None)
