"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN fused-attention V1 contract and extension bridge.
Provides strict fail-loud validators, mode resolution, and optional CUDA extension dispatch for WAN fused
self/cross attention (`QKV + RoPE + attention + out-proj`) in inference mode. Resolves projection/norm
weights through runtime ops dequantization helpers so GGUF-backed modules feed dense floating tensors
into fused-kernel contracts.

Symbols (top-level; keep in sync; no ghosts):
- `WanFusedMode` (enum): Runtime mode for fused attention dispatch (`off|auto|force`).
- `WanFusedContractError` (class): Fail-loud contract error with stable `code` field.
- `WanFusedAttemptResult` (dataclass): Result envelope for non-forced attempts (`output` or `reason_code`).
- `WanFusedWarmupStatus` (dataclass): Load-time warmup result for fused extension readiness.
- `parse_wan_fused_mode` (function): Parses and validates a fused mode string.
- `resolve_effective_wan_fused_mode` (function): Resolves fused mode from override/env.
- `resolve_effective_wan_fused_attn_core` (function): Resolves effective fused attention core telemetry tuple (`core`, `source`, `raw`).
- `is_extension_available` (function): Returns whether WAN fused CUDA ops are available.
- `last_extension_error` (function): Returns the last extension load/build error details.
- `warmup_extension_for_load` (function): Triggers extension load/build during model-load seam and emits explicit readiness logs.
- `wan_fused_runtime_metrics_reset` (function): Resets per-run fused dispatch counters/lifecycle state.
- `wan_fused_runtime_metrics_is_active` (function): Returns whether a fused runtime metrics context is active for this execution flow.
- `wan_fused_runtime_metrics_set_stage` (function): Sets the active sampling stage label for fused counter attribution.
- `wan_fused_runtime_metrics_log_summary` (function): Emits end-of-run fused dispatch summary and optionally resets run state.
- `try_fused_self_attention` (function): Attempts fused self-attention dispatch and returns output or reason.
- `try_fused_cross_attention` (function): Attempts fused cross-attention dispatch and returns output or reason.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import importlib
import logging
import os
import re
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch

from apps.backend.runtime.ops.operations import get_weight_and_bias
from apps.backend.runtime.ops.operations_gguf import dequantize_tensor

logger = get_backend_logger("backend.runtime.attention.wan_fused_v1")


_MODE_ENV_KEY = "CODEX_WAN22_FUSED_ATTN_V1_MODE"
_JIT_ENV_KEY = "CODEX_WAN_FUSED_V1_JIT"
_ATTN_CORE_ENV_KEY = "CODEX_WAN_FUSED_V1_ATTN_CORE"
_FORCE_DEFAULT_ATTN_CORE_VALUE = "aten"
_REQUIRED_EXTENSION_ABI = 4


E_WAN_FUSED_DISABLED = "E_WAN_FUSED_DISABLED"
E_WAN_FUSED_EXTENSION_UNAVAILABLE = "E_WAN_FUSED_EXTENSION_UNAVAILABLE"
E_WAN_FUSED_KERNEL_RUNTIME_ERROR = "E_WAN_FUSED_KERNEL_RUNTIME_ERROR"
E_WAN_FUSED_INVALID_MODE = "E_WAN_FUSED_INVALID_MODE"
E_WAN_FUSED_DROPOUT_UNSUPPORTED = "E_WAN_FUSED_DROPOUT_UNSUPPORTED"
E_WAN_FUSED_DEVICE_UNSUPPORTED = "E_WAN_FUSED_DEVICE_UNSUPPORTED"
E_WAN_FUSED_DTYPE_UNSUPPORTED = "E_WAN_FUSED_DTYPE_UNSUPPORTED"
E_WAN_FUSED_HEAD_DIM_UNSUPPORTED = "E_WAN_FUSED_HEAD_DIM_UNSUPPORTED"
E_WAN_FUSED_UNSUPPORTED_ARCH = "E_WAN_FUSED_UNSUPPORTED_ARCH"
E_WAN_FUSED_INVALID_SHAPE = "E_WAN_FUSED_INVALID_SHAPE"
E_WAN_FUSED_MISSING_ROPE = "E_WAN_FUSED_MISSING_ROPE"
E_WAN_FUSED_NONCONTIGUOUS = "E_WAN_FUSED_NONCONTIGUOUS"
E_WAN_FUSED_INVALID_ENV = "E_WAN_FUSED_INVALID_ENV"
E_WAN_FUSED_STREAMING_INVARIANT_VIOLATION = "E_WAN_FUSED_STREAMING_INVARIANT_VIOLATION"


@dataclass(frozen=True, slots=True)
class _AttemptError:
    stage: str
    message: str


@dataclass(frozen=True, slots=True)
class WanFusedAttemptResult:
    output: torch.Tensor | None
    reason_code: str | None
    reason_detail: str | None = None


@dataclass(slots=True)
class _WanFusedRuntimeMetrics:
    run_label: str
    attempts: int = 0
    hits: int = 0
    fallback_by_reason: dict[str, int] = field(default_factory=dict)
    attempts_by_stage: dict[str, int] = field(default_factory=dict)
    hits_by_stage: dict[str, int] = field(default_factory=dict)
    fallback_by_stage_reason: dict[str, int] = field(default_factory=dict)
    warned_auto_reasons: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class WanFusedWarmupStatus:
    mode: WanFusedMode
    build_enabled: bool
    attempted: bool
    available: bool
    detail: str | None = None


class WanFusedMode(str, Enum):
    OFF = "off"
    AUTO = "auto"
    FORCE = "force"


class WanFusedContractError(RuntimeError):
    def __init__(self, *, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


_ext = None
_last_error: str | None = None
_attempt_errors: list[_AttemptError] = []
_load_attempted = False
_last_attempt_with_build = False
_runtime_metrics_var: ContextVar[_WanFusedRuntimeMetrics | None] = ContextVar(
    "wan_fused_v1_runtime_metrics",
    default=None,
)
_runtime_stage_var: ContextVar[str] = ContextVar(
    "wan_fused_v1_runtime_stage",
    default="unset",
)


def _format_counter_map(values: dict[str, int]) -> str:
    if not values:
        return "none"
    return ",".join(f"{key}={int(values[key])}" for key in sorted(values.keys()))


def _stage_for_metrics() -> str:
    stage = str(_runtime_stage_var.get() or "").strip()
    return stage if stage else "unset"


def _ensure_runtime_metrics() -> _WanFusedRuntimeMetrics:
    metrics = _runtime_metrics_var.get()
    if metrics is None:
        metrics = _WanFusedRuntimeMetrics(run_label="implicit")
        _runtime_metrics_var.set(metrics)
    return metrics


def wan_fused_runtime_metrics_reset(*, run_label: str) -> None:
    _runtime_metrics_var.set(_WanFusedRuntimeMetrics(run_label=str(run_label)))
    _runtime_stage_var.set("unset")


def wan_fused_runtime_metrics_is_active() -> bool:
    return _runtime_metrics_var.get() is not None


def wan_fused_runtime_metrics_set_stage(stage_name: str) -> None:
    _runtime_stage_var.set(str(stage_name))


def _record_attempt(
    *,
    op_kind: str,
    mode: WanFusedMode,
    success: bool,
    reason_code: str | None,
    reason_detail: str | None,
) -> None:
    metrics = _ensure_runtime_metrics()
    stage = _stage_for_metrics()
    metrics.attempts += 1
    metrics.attempts_by_stage[stage] = int(metrics.attempts_by_stage.get(stage, 0)) + 1
    if success:
        metrics.hits += 1
        metrics.hits_by_stage[stage] = int(metrics.hits_by_stage.get(stage, 0)) + 1
        return

    code = str(reason_code or E_WAN_FUSED_KERNEL_RUNTIME_ERROR)
    metrics.fallback_by_reason[code] = int(metrics.fallback_by_reason.get(code, 0)) + 1
    stage_reason_key = f"{stage}:{code}"
    metrics.fallback_by_stage_reason[stage_reason_key] = int(metrics.fallback_by_stage_reason.get(stage_reason_key, 0)) + 1
    if mode is not WanFusedMode.AUTO:
        return
    if code in metrics.warned_auto_reasons:
        return
    metrics.warned_auto_reasons.add(code)
    logger.warning(
        "wan_fused_v1 auto fallback: reason=%s stage=%s op=%s detail=%s (logged once per reason per run)",
        code,
        stage,
        str(op_kind),
        str(reason_detail),
    )


def wan_fused_runtime_metrics_log_summary(
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
            "wan_fused_v1.summary run=%s attempts=%d hits=%d fallback_by_reason=%s attempts_by_stage=%s hits_by_stage=%s "
            "fallback_by_stage_reason=%s",
            str(metrics.run_label),
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


def _set_attempt_error(stage: str, ex: Exception) -> None:
    global _last_error
    message = f"{type(ex).__name__}: {ex}"
    _attempt_errors.append(_AttemptError(stage=stage, message=message))
    _last_error = "\n".join(f"{entry.stage}: {entry.message}" for entry in _attempt_errors)


def _has_ops() -> bool:
    ops = getattr(torch, "ops", None)
    if ops is None or not hasattr(ops, "wan_fused_v1"):
        return False
    wan_ops = ops.wan_fused_v1
    return hasattr(wan_ops, "self_fwd") and hasattr(wan_ops, "cross_fwd")


def _ensure_extension_abi(module: Any) -> None:
    abi = getattr(module, "WAN_FUSED_V1_ABI", None)
    try:
        abi_value = int(abi)
    except Exception as ex:
        raise RuntimeError(
            "incompatible wan_fused_v1 extension ABI: "
            f"required={_REQUIRED_EXTENSION_ABI} got={abi!r}"
        ) from ex
    if abi_value != int(_REQUIRED_EXTENSION_ABI):
        raise RuntimeError(
            "incompatible wan_fused_v1 extension ABI: "
            f"required={_REQUIRED_EXTENSION_ABI} got={abi!r}"
        )


def _purge_extension_module_cache(*, module_name: str) -> None:
    sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _try_load_ext(*, build: bool) -> None:
    global _ext
    global _last_error
    global _load_attempted
    global _last_attempt_with_build

    if _ext is not None and _has_ops():
        return
    if _load_attempted and _ext is None and (_last_attempt_with_build or build == _last_attempt_with_build):
        return

    _load_attempted = True
    _last_attempt_with_build = bool(build)
    _attempt_errors.clear()
    _last_error = None

    if build:
        try:
            _purge_extension_module_cache(module_name="wan_fused_v1_cuda_jit")
            from torch.utils.cpp_extension import load

            this_dir = os.path.dirname(__file__)
            src_dir = os.path.normpath(os.path.join(this_dir, "..", "..", "kernels", "wan_fused_v1"))

            def _src(path: str) -> str:
                return os.path.join(src_dir, path)

            sources = [
                _src("wan_fused_v1_binding.cpp"),
                _src("wan_fused_v1_kernels.cu"),
            ]

            loaded = load(
                name="wan_fused_v1_cuda_jit",
                sources=sources,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-DUSE_CUDA"],
            )
            _ensure_extension_abi(loaded)
            _ext = loaded
            if not _has_ops():
                raise RuntimeError("JIT module loaded but torch.ops.wan_fused_v1.{self_fwd,cross_fwd} is missing")
            logger.info("built wan_fused_v1_cuda extension via JIT")
            return
        except Exception as ex:
            _set_attempt_error("jit", ex)
            _ext = None
            logger.error("failed to build wan_fused_v1_cuda via JIT: %s", ex)

    try:
        _purge_extension_module_cache(module_name="wan_fused_v1_cuda")
        import wan_fused_v1_cuda as loaded

        _ensure_extension_abi(loaded)
        _ext = loaded
        if not _has_ops():
            raise RuntimeError("module loaded but torch.ops.wan_fused_v1.{self_fwd,cross_fwd} is missing")
        logger.info("loaded wan_fused_v1_cuda extension (prebuilt)")
        return
    except Exception as ex:
        _ext = None
        _set_attempt_error("prebuilt", ex)
        logger.info("wan_fused_v1_cuda prebuilt not available: %s", ex)

    try:
        this_dir = os.path.dirname(__file__)
        ext_dir = os.path.normpath(os.path.join(this_dir, "..", "..", "kernels", "wan_fused_v1"))
        if os.path.isdir(ext_dir) and ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)

        _purge_extension_module_cache(module_name="wan_fused_v1_cuda")
        import wan_fused_v1_cuda as loaded

        _ensure_extension_abi(loaded)
        _ext = loaded
        if not _has_ops():
            raise RuntimeError("in-place module loaded but torch.ops.wan_fused_v1.{self_fwd,cross_fwd} is missing")
        logger.info("loaded wan_fused_v1_cuda extension from in-place build (%s)", ext_dir)
        return
    except Exception as ex:
        _ext = None
        _set_attempt_error("in_place", ex)
        logger.info("wan_fused_v1_cuda in-place module not available: %s", ex)

    if not build:
        return


def _jit_build_enabled() -> bool:
    return str(os.environ.get(_JIT_ENV_KEY, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def is_extension_available() -> bool:
    build = _jit_build_enabled()
    _try_load_ext(build=build)
    return _ext is not None and _has_ops()


def last_extension_error() -> str | None:
    return _last_error


def warmup_extension_for_load(mode: str | WanFusedMode | None = None) -> WanFusedWarmupStatus:
    fused_mode = resolve_effective_wan_fused_mode(mode)
    effective_core, effective_core_source, effective_core_raw = resolve_effective_wan_fused_attn_core(fused_mode)
    build_enabled = _jit_build_enabled()
    if fused_mode is WanFusedMode.OFF:
        logger.info(
            "wan_fused_v1 warmup skipped at load (mode=off effective_core=%s effective_core_source=%s "
            "effective_core_raw=%s)",
            effective_core,
            effective_core_source,
            effective_core_raw,
        )
        return WanFusedWarmupStatus(
            mode=fused_mode,
            build_enabled=build_enabled,
            attempted=False,
            available=False,
            detail=None,
        )

    logger.info(
        "wan_fused_v1 warmup start (mode=%s jit_build=%s effective_core=%s effective_core_source=%s "
        "effective_core_raw=%s)",
        fused_mode.value,
        build_enabled,
        effective_core,
        effective_core_source,
        effective_core_raw,
    )
    _try_load_ext(build=build_enabled)
    available = _ext is not None and _has_ops()
    detail = last_extension_error()
    if available:
        logger.info(
            "wan_fused_v1 warmup ready (mode=%s jit_build=%s effective_core=%s effective_core_source=%s "
            "effective_core_raw=%s)",
            fused_mode.value,
            build_enabled,
            effective_core,
            effective_core_source,
            effective_core_raw,
        )
        return WanFusedWarmupStatus(
            mode=fused_mode,
            build_enabled=build_enabled,
            attempted=True,
            available=True,
            detail=None,
        )

    logger.warning(
        "wan_fused_v1 warmup unavailable (mode=%s jit_build=%s effective_core=%s effective_core_source=%s "
        "effective_core_raw=%s detail=%r)",
        fused_mode.value,
        build_enabled,
        effective_core,
        effective_core_source,
        effective_core_raw,
        detail,
    )
    if fused_mode is WanFusedMode.FORCE:
        _fail(
            code=E_WAN_FUSED_EXTENSION_UNAVAILABLE,
            message=(
                "WAN fused V1 force mode requested but extension warmup failed during model load. "
                f"details={detail!r}"
            ),
        )
    return WanFusedWarmupStatus(
        mode=fused_mode,
        build_enabled=build_enabled,
        attempted=True,
        available=False,
        detail=detail,
    )


def parse_wan_fused_mode(value: str, *, field_name: str) -> WanFusedMode:
    normalized = str(value).strip().lower()
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
        allowed = "off|auto|force"
        raise WanFusedContractError(
            code=E_WAN_FUSED_INVALID_MODE,
            message=f"{field_name} must be one of {allowed}; got {value!r}.",
        )
    return WanFusedMode(mapped)


def resolve_effective_wan_fused_mode(mode_override: str | WanFusedMode | None) -> WanFusedMode:
    if isinstance(mode_override, WanFusedMode):
        return mode_override
    if mode_override is not None:
        return parse_wan_fused_mode(mode_override, field_name="wan_fused_mode")
    env_value = os.environ.get(_MODE_ENV_KEY, "off")
    return parse_wan_fused_mode(env_value, field_name=_MODE_ENV_KEY)


def _format_attn_core_raw(raw_value: str | None) -> str:
    if raw_value is None:
        return "<unset>"
    raw_str = str(raw_value).strip()
    raw_token = re.sub(r"[^0-9A-Za-z._:+-]+", "_", raw_str)
    if not raw_token:
        return "<empty>"
    return raw_token


def resolve_effective_wan_fused_attn_core(mode_override: str | WanFusedMode | None) -> tuple[str, str, str]:
    fused_mode = resolve_effective_wan_fused_mode(mode_override)
    raw_value = os.environ.get(_ATTN_CORE_ENV_KEY)
    raw_token = _format_attn_core_raw(raw_value)
    if raw_value is None:
        if fused_mode is WanFusedMode.FORCE:
            return _FORCE_DEFAULT_ATTN_CORE_VALUE, "force_default", raw_token
        return "aten", "kernel_default", raw_token

    normalized = str(raw_value).lower()
    if normalized in {"aten", "default", "off"}:
        return "aten", "env", raw_token
    if normalized in {"cuda", "cuda_experimental"}:
        return "cuda_experimental", "env", raw_token
    return "invalid", "env", raw_token


def _fail(*, code: str, message: str) -> None:
    raise WanFusedContractError(code=code, message=message)


def _resolve_head_count(*, channels: int, head_dim: int, field_name: str) -> int:
    if head_dim <= 0:
        _fail(code=E_WAN_FUSED_HEAD_DIM_UNSUPPORTED, message=f"{field_name} must be > 0; got {head_dim}.")
    if channels % head_dim != 0:
        _fail(
            code=E_WAN_FUSED_HEAD_DIM_UNSUPPORTED,
            message=(
                f"channels/head_dim mismatch for {field_name}: channels={channels} head_dim={head_dim} "
                "(channels must be divisible by head_dim)."
            ),
        )
    return channels // head_dim


def _validate_arch_dtype_head_dim(*, device: torch.device, dtype: torch.dtype, head_dim: int) -> None:
    if device.type != "cuda":
        _fail(code=E_WAN_FUSED_DEVICE_UNSUPPORTED, message=f"WAN fused V1 requires CUDA tensors; got device={device}.")

    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        _fail(code=E_WAN_FUSED_DTYPE_UNSUPPORTED, message=f"WAN fused V1 supports fp16/bf16/fp32; got dtype={dtype}.")

    if head_dim % 8 != 0:
        _fail(code=E_WAN_FUSED_HEAD_DIM_UNSUPPORTED, message=f"head_dim must be a multiple of 8; got head_dim={head_dim}.")

    major, minor = torch.cuda.get_device_capability(device)
    capability = (int(major), int(minor))

    if capability < (7, 5):
        _fail(
            code=E_WAN_FUSED_UNSUPPORTED_ARCH,
            message=f"WAN fused V1 requires SM75+; got compute_capability={capability}.",
        )

    if capability == (7, 5):
        if dtype == torch.bfloat16:
            _fail(
                code=E_WAN_FUSED_DTYPE_UNSUPPORTED,
                message="WAN fused V1 does not support bf16 on SM75.",
            )
        if head_dim > 256:
            _fail(
                code=E_WAN_FUSED_HEAD_DIM_UNSUPPORTED,
                message=f"WAN fused V1 supports head_dim<=256 on SM75; got head_dim={head_dim}.",
            )
        return

    if capability in {(8, 0), (8, 6), (8, 9), (9, 0)}:
        if dtype in {torch.float16, torch.bfloat16} and head_dim > 512:
            _fail(
                code=E_WAN_FUSED_HEAD_DIM_UNSUPPORTED,
                message=(
                    "WAN fused V1 supports fp16/bf16 head_dim<=512 on SM80/86/89/90; "
                    f"got head_dim={head_dim}."
                ),
            )
        if dtype == torch.float32 and head_dim > 256:
            _fail(
                code=E_WAN_FUSED_HEAD_DIM_UNSUPPORTED,
                message=(
                    "WAN fused V1 supports fp32 head_dim<=256 on SM80/86/89/90; "
                    f"got head_dim={head_dim}."
                ),
            )
        return

    _fail(
        code=E_WAN_FUSED_UNSUPPORTED_ARCH,
        message=(
            "WAN fused V1 has no declared architecture contract for this GPU; "
            f"got compute_capability={capability}."
        ),
    )


def _resolve_linear_weight_bias(
    linear: Any,
    *,
    expected_out: int,
    expected_in: int,
    label: str,
    target_device: torch.device,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    non_blocking = target_device.type != "mps"
    weight, bias = get_weight_and_bias(
        linear,
        weight_args={"device": target_device, "dtype": target_dtype, "non_blocking": non_blocking},
        bias_args={"device": target_device, "dtype": target_dtype, "non_blocking": non_blocking},
        weight_fn=dequantize_tensor,
        bias_fn=dequantize_tensor,
    )
    if not torch.is_tensor(weight):
        _fail(code=E_WAN_FUSED_INVALID_SHAPE, message=f"{label}.weight is missing or invalid.")
    if tuple(weight.shape) != (expected_out, expected_in):
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=f"{label}.weight shape mismatch: got {tuple(weight.shape)} expected {(expected_out, expected_in)}.",
        )
    if int(weight.numel()) != int(expected_out * expected_in):
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=(
                f"{label}.weight storage mismatch after dequantize: numel={int(weight.numel())} "
                f"expected={int(expected_out * expected_in)} shape={tuple(weight.shape)}."
            ),
        )
    if not weight.is_floating_point():
        _fail(
            code=E_WAN_FUSED_DTYPE_UNSUPPORTED,
            message=f"{label}.weight must be floating for fused path; got dtype={weight.dtype}.",
        )
    if bias is not None:
        if not torch.is_tensor(bias):
            _fail(code=E_WAN_FUSED_INVALID_SHAPE, message=f"{label}.bias is not a tensor.")
        if tuple(bias.shape) != (expected_out,):
            _fail(
                code=E_WAN_FUSED_INVALID_SHAPE,
                message=f"{label}.bias shape mismatch: got {tuple(bias.shape)} expected {(expected_out,)}.",
            )
        if not bias.is_floating_point():
            _fail(
                code=E_WAN_FUSED_DTYPE_UNSUPPORTED,
                message=f"{label}.bias must be floating for fused path; got dtype={bias.dtype}.",
            )
    return weight, bias


def _validate_common_inputs(*, x: torch.Tensor, dropout_p: float) -> None:
    if x.ndim != 3:
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=f"WAN fused V1 expects x with shape [B,L,C]; got shape={tuple(x.shape)}.",
        )
    if not x.is_floating_point():
        _fail(code=E_WAN_FUSED_DTYPE_UNSUPPORTED, message=f"WAN fused V1 expects floating x; got dtype={x.dtype}.")
    if abs(float(dropout_p)) > 0.0:
        _fail(
            code=E_WAN_FUSED_DROPOUT_UNSUPPORTED,
            message=f"WAN fused V1 only supports dropout=0.0; got dropout_p={float(dropout_p)}.",
        )
    if not x.is_contiguous():
        _fail(
            code=E_WAN_FUSED_NONCONTIGUOUS,
            message=f"WAN fused V1 requires contiguous x tensor; got stride={tuple(x.stride())}.",
        )


def _validate_rope_tensor(*, tensor: torch.Tensor | None, expected_len: int, label: str) -> None:
    if tensor is None:
        _fail(code=E_WAN_FUSED_MISSING_ROPE, message=f"missing required RoPE tensor: {label}.")
    if tensor.ndim != 4:
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=f"{label} must be [1,S,1,D]; got shape={tuple(tensor.shape)}.",
        )
    if int(tensor.shape[0]) != 1 or int(tensor.shape[2]) != 1:
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=f"{label} must have shape prefix [1,*,1,*]; got shape={tuple(tensor.shape)}.",
        )
    if int(tensor.shape[1]) != int(expected_len):
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=f"{label} sequence mismatch: got S={int(tensor.shape[1])} expected={int(expected_len)}.",
        )
    if not tensor.is_contiguous():
        _fail(
            code=E_WAN_FUSED_NONCONTIGUOUS,
            message=f"{label} must be contiguous; got stride={tuple(tensor.stride())}.",
        )


def _resolve_norm_weight(
    *,
    weight: torch.Tensor,
    expected_channels: int,
    label: str,
    target_device: torch.device,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    resolved = dequantize_tensor(weight)
    if not torch.is_tensor(resolved):
        _fail(code=E_WAN_FUSED_INVALID_SHAPE, message=f"{label} is missing or invalid.")
    resolved = resolved.to(device=target_device, dtype=target_dtype, non_blocking=(target_device.type != "mps"))
    if tuple(resolved.shape) != (expected_channels,):
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=f"{label} shape mismatch: got {tuple(resolved.shape)} expected {(expected_channels,)}.",
        )
    if not resolved.is_floating_point():
        _fail(
            code=E_WAN_FUSED_DTYPE_UNSUPPORTED,
            message=f"{label} must be floating for fused path; got dtype={resolved.dtype}.",
        )
    return resolved.contiguous()


def _maybe_return_unavailable(
    *,
    mode: WanFusedMode,
    effective_core: str,
    effective_core_source: str,
    effective_core_raw: str,
) -> WanFusedAttemptResult | None:
    if is_extension_available():
        return None
    detail = last_extension_error()
    detail_with_core = (
        f"details={detail!r} effective_core={effective_core} "
        f"effective_core_source={effective_core_source} effective_core_raw={effective_core_raw}"
    )
    if mode is WanFusedMode.FORCE:
        _fail(
            code=E_WAN_FUSED_EXTENSION_UNAVAILABLE,
            message=(
                "WAN fused V1 forced mode requested but extension/ops are unavailable. "
                f"{detail_with_core}"
            ),
        )
    return WanFusedAttemptResult(
        output=None,
        reason_code=E_WAN_FUSED_EXTENSION_UNAVAILABLE,
        reason_detail=detail_with_core,
    )


def _maybe_return_invalid_attn_core(
    *,
    mode: WanFusedMode,
    effective_core: str,
    effective_core_source: str,
    effective_core_raw: str,
) -> WanFusedAttemptResult | None:
    if effective_core != "invalid":
        return None
    detail = (
        f"invalid attention core selector: env={_ATTN_CORE_ENV_KEY} effective_core={effective_core} "
        f"effective_core_source={effective_core_source} effective_core_raw={effective_core_raw} "
        "allowed=aten|cuda_experimental"
    )
    if mode is WanFusedMode.FORCE:
        _fail(code=E_WAN_FUSED_INVALID_ENV, message=detail)
    return WanFusedAttemptResult(
        output=None,
        reason_code=E_WAN_FUSED_INVALID_ENV,
        reason_detail=detail,
    )


def _map_kernel_runtime_error_code(message: str) -> str:
    normalized = str(message)
    lowered = normalized.lower()
    if "streaming invariant violated" in lowered:
        return E_WAN_FUSED_STREAMING_INVARIANT_VIOLATION
    if "codex_wan_fused_v1_" in lowered and (
        "strict integer" in lowered
        or "positive integer" in lowered
        or "boolean flag" in lowered
        or "must be > 0" in lowered
        or "exceeds hard cap" in lowered
        or "must be one of: aten | cuda_experimental" in lowered
    ):
        return E_WAN_FUSED_INVALID_ENV
    return E_WAN_FUSED_KERNEL_RUNTIME_ERROR


def try_fused_self_attention(
    *,
    mode: str | WanFusedMode | None,
    x: torch.Tensor,
    q_proj: Any,
    k_proj: Any,
    v_proj: Any,
    o_proj: Any,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    rope_cos_qk: torch.Tensor | None,
    rope_sin_qk: torch.Tensor | None,
    dropout_p: float,
) -> WanFusedAttemptResult:
    fused_mode = resolve_effective_wan_fused_mode(mode)
    if fused_mode is WanFusedMode.OFF:
        return WanFusedAttemptResult(output=None, reason_code=E_WAN_FUSED_DISABLED)
    effective_core, effective_core_source, effective_core_raw = resolve_effective_wan_fused_attn_core(fused_mode)
    unavailable = _maybe_return_unavailable(
        mode=fused_mode,
        effective_core=effective_core,
        effective_core_source=effective_core_source,
        effective_core_raw=effective_core_raw,
    )
    if unavailable is not None:
        _record_attempt(
            op_kind="self",
            mode=fused_mode,
            success=False,
            reason_code=unavailable.reason_code,
            reason_detail=unavailable.reason_detail,
        )
        return unavailable
    invalid_attn_core = _maybe_return_invalid_attn_core(
        mode=fused_mode,
        effective_core=effective_core,
        effective_core_source=effective_core_source,
        effective_core_raw=effective_core_raw,
    )
    if invalid_attn_core is not None:
        _record_attempt(
            op_kind="self",
            mode=fused_mode,
            success=False,
            reason_code=invalid_attn_core.reason_code,
            reason_detail=invalid_attn_core.reason_detail,
        )
        return invalid_attn_core

    try:
        _validate_common_inputs(x=x, dropout_p=dropout_p)
        bsz, seq_len, channels = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))

        _validate_rope_tensor(tensor=rope_cos_qk, expected_len=seq_len, label="rope_cos_qk")
        _validate_rope_tensor(tensor=rope_sin_qk, expected_len=seq_len, label="rope_sin_qk")

        head_dim = int(rope_cos_qk.shape[-1])
        _resolve_head_count(channels=channels, head_dim=head_dim, field_name="self")
        _validate_arch_dtype_head_dim(device=x.device, dtype=x.dtype, head_dim=head_dim)

        w_q, b_q = _resolve_linear_weight_bias(
            q_proj,
            expected_out=channels,
            expected_in=channels,
            label="q_proj",
            target_device=x.device,
            target_dtype=x.dtype,
        )
        w_k, b_k = _resolve_linear_weight_bias(
            k_proj,
            expected_out=channels,
            expected_in=channels,
            label="k_proj",
            target_device=x.device,
            target_dtype=x.dtype,
        )
        w_v, b_v = _resolve_linear_weight_bias(
            v_proj,
            expected_out=channels,
            expected_in=channels,
            label="v_proj",
            target_device=x.device,
            target_dtype=x.dtype,
        )
        w_o, b_o = _resolve_linear_weight_bias(
            o_proj,
            expected_out=channels,
            expected_in=channels,
            label="o_proj",
            target_device=x.device,
            target_dtype=x.dtype,
        )

        norm_q_weight = _resolve_norm_weight(
            weight=norm_q_weight,
            expected_channels=channels,
            label="norm_q_weight",
            target_device=x.device,
            target_dtype=x.dtype,
        )
        norm_k_weight = _resolve_norm_weight(
            weight=norm_k_weight,
            expected_channels=channels,
            label="norm_k_weight",
            target_device=x.device,
            target_dtype=x.dtype,
        )

        has_q_bias = b_q is not None
        has_k_bias = b_k is not None
        has_v_bias = b_v is not None
        if has_q_bias != has_k_bias or has_q_bias != has_v_bias:
            _fail(
                code=E_WAN_FUSED_INVALID_SHAPE,
                message=(
                    "WAN fused self requires all-or-none Q/K/V biases; "
                    f"got q={has_q_bias} k={has_k_bias} v={has_v_bias}."
                ),
            )
        out = torch.ops.wan_fused_v1.self_fwd(
            x,
            w_q,
            b_q,
            w_k,
            b_k,
            w_v,
            b_v,
            norm_q_weight,
            norm_k_weight,
            rope_cos_qk,
            rope_sin_qk,
            w_o,
            b_o,
            effective_core,
        )
        if tuple(out.shape) != (bsz, seq_len, channels):
            _fail(
                code=E_WAN_FUSED_INVALID_SHAPE,
                message=(
                    "WAN fused self returned unexpected shape: "
                    f"got {tuple(out.shape)} expected {(bsz, seq_len, channels)}."
                ),
            )
        _record_attempt(op_kind="self", mode=fused_mode, success=True, reason_code=None, reason_detail=None)
        return WanFusedAttemptResult(output=out, reason_code=None)
    except WanFusedContractError as ex:
        if fused_mode is WanFusedMode.FORCE:
            _record_attempt(
                op_kind="self",
                mode=fused_mode,
                success=False,
                reason_code=ex.code,
                reason_detail=str(ex),
            )
            raise
        _record_attempt(
            op_kind="self",
            mode=fused_mode,
            success=False,
            reason_code=ex.code,
            reason_detail=str(ex),
        )
        return WanFusedAttemptResult(output=None, reason_code=ex.code, reason_detail=str(ex))
    except Exception as ex:
        code = _map_kernel_runtime_error_code(str(ex))
        if fused_mode is WanFusedMode.FORCE:
            _record_attempt(
                op_kind="self",
                mode=fused_mode,
                success=False,
                reason_code=code,
                reason_detail=f"{type(ex).__name__}: {ex}",
            )
            _fail(
                code=code,
                message=f"WAN fused self kernel failed: {type(ex).__name__}: {ex}",
            )
        _record_attempt(
            op_kind="self",
            mode=fused_mode,
            success=False,
            reason_code=code,
            reason_detail=f"{type(ex).__name__}: {ex}",
        )
        return WanFusedAttemptResult(
            output=None,
            reason_code=code,
            reason_detail=f"{type(ex).__name__}: {ex}",
        )


def try_fused_cross_attention(
    *,
    mode: str | WanFusedMode | None,
    x: torch.Tensor,
    context: torch.Tensor,
    q_proj: Any,
    k_proj: Any,
    v_proj: Any,
    o_proj: Any,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    rope_cos_q: torch.Tensor | None,
    rope_sin_q: torch.Tensor | None,
    rope_cos_k: torch.Tensor | None,
    rope_sin_k: torch.Tensor | None,
    dropout_p: float,
) -> WanFusedAttemptResult:
    fused_mode = resolve_effective_wan_fused_mode(mode)
    if fused_mode is WanFusedMode.OFF:
        return WanFusedAttemptResult(output=None, reason_code=E_WAN_FUSED_DISABLED)
    effective_core, effective_core_source, effective_core_raw = resolve_effective_wan_fused_attn_core(fused_mode)
    unavailable = _maybe_return_unavailable(
        mode=fused_mode,
        effective_core=effective_core,
        effective_core_source=effective_core_source,
        effective_core_raw=effective_core_raw,
    )
    if unavailable is not None:
        _record_attempt(
            op_kind="cross",
            mode=fused_mode,
            success=False,
            reason_code=unavailable.reason_code,
            reason_detail=unavailable.reason_detail,
        )
        return unavailable
    invalid_attn_core = _maybe_return_invalid_attn_core(
        mode=fused_mode,
        effective_core=effective_core,
        effective_core_source=effective_core_source,
        effective_core_raw=effective_core_raw,
    )
    if invalid_attn_core is not None:
        _record_attempt(
            op_kind="cross",
            mode=fused_mode,
            success=False,
            reason_code=invalid_attn_core.reason_code,
            reason_detail=invalid_attn_core.reason_detail,
        )
        return invalid_attn_core

    try:
        _validate_common_inputs(x=x, dropout_p=dropout_p)
        if context.ndim != 3:
            _fail(
                code=E_WAN_FUSED_INVALID_SHAPE,
                message=f"WAN fused cross expects context [B,S,Cctx]; got shape={tuple(context.shape)}.",
            )
        if int(context.shape[0]) != int(x.shape[0]):
            _fail(
                code=E_WAN_FUSED_INVALID_SHAPE,
                message=(
                    "WAN fused cross batch mismatch between x and context: "
                    f"x.B={int(x.shape[0])} context.B={int(context.shape[0])}."
                ),
            )

        bsz, q_len, channels = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
        kv_len = int(context.shape[1])
        ctx_dim = int(context.shape[2])

        _validate_rope_tensor(tensor=rope_cos_q, expected_len=q_len, label="rope_cos_q")
        _validate_rope_tensor(tensor=rope_sin_q, expected_len=q_len, label="rope_sin_q")
        _validate_rope_tensor(tensor=rope_cos_k, expected_len=kv_len, label="rope_cos_k")
        _validate_rope_tensor(tensor=rope_sin_k, expected_len=kv_len, label="rope_sin_k")

        head_dim = int(rope_cos_q.shape[-1])
        _resolve_head_count(channels=channels, head_dim=head_dim, field_name="cross")
        if int(rope_cos_k.shape[-1]) != head_dim or int(rope_sin_k.shape[-1]) != head_dim:
            _fail(
                code=E_WAN_FUSED_INVALID_SHAPE,
                message=(
                    "WAN fused cross requires matching RoPE head_dim between query/key tensors. "
                    f"got q={head_dim} k_cos={int(rope_cos_k.shape[-1])} k_sin={int(rope_sin_k.shape[-1])}."
                ),
            )

        _validate_arch_dtype_head_dim(device=x.device, dtype=x.dtype, head_dim=head_dim)

        w_q, b_q = _resolve_linear_weight_bias(
            q_proj,
            expected_out=channels,
            expected_in=channels,
            label="q_proj",
            target_device=x.device,
            target_dtype=x.dtype,
        )
        w_k, b_k = _resolve_linear_weight_bias(
            k_proj,
            expected_out=channels,
            expected_in=ctx_dim,
            label="k_proj",
            target_device=x.device,
            target_dtype=x.dtype,
        )
        w_v, b_v = _resolve_linear_weight_bias(
            v_proj,
            expected_out=channels,
            expected_in=ctx_dim,
            label="v_proj",
            target_device=x.device,
            target_dtype=x.dtype,
        )
        w_o, b_o = _resolve_linear_weight_bias(
            o_proj,
            expected_out=channels,
            expected_in=channels,
            label="o_proj",
            target_device=x.device,
            target_dtype=x.dtype,
        )

        norm_q_weight = _resolve_norm_weight(
            weight=norm_q_weight,
            expected_channels=channels,
            label="norm_q_weight",
            target_device=x.device,
            target_dtype=x.dtype,
        )
        norm_k_weight = _resolve_norm_weight(
            weight=norm_k_weight,
            expected_channels=channels,
            label="norm_k_weight",
            target_device=x.device,
            target_dtype=x.dtype,
        )

        out = torch.ops.wan_fused_v1.cross_fwd(
            x,
            context,
            w_q,
            b_q,
            norm_q_weight,
            rope_cos_q,
            rope_sin_q,
            w_k,
            b_k,
            norm_k_weight,
            rope_cos_k,
            rope_sin_k,
            w_v,
            b_v,
            w_o,
            b_o,
            effective_core,
        )
        if tuple(out.shape) != (bsz, q_len, channels):
            _fail(
                code=E_WAN_FUSED_INVALID_SHAPE,
                message=(
                    "WAN fused cross returned unexpected shape: "
                    f"got {tuple(out.shape)} expected {(bsz, q_len, channels)}."
                ),
            )
        _record_attempt(op_kind="cross", mode=fused_mode, success=True, reason_code=None, reason_detail=None)
        return WanFusedAttemptResult(output=out, reason_code=None)
    except WanFusedContractError as ex:
        if fused_mode is WanFusedMode.FORCE:
            _record_attempt(
                op_kind="cross",
                mode=fused_mode,
                success=False,
                reason_code=ex.code,
                reason_detail=str(ex),
            )
            raise
        _record_attempt(
            op_kind="cross",
            mode=fused_mode,
            success=False,
            reason_code=ex.code,
            reason_detail=str(ex),
        )
        return WanFusedAttemptResult(output=None, reason_code=ex.code, reason_detail=str(ex))
    except Exception as ex:
        code = _map_kernel_runtime_error_code(str(ex))
        if fused_mode is WanFusedMode.FORCE:
            _record_attempt(
                op_kind="cross",
                mode=fused_mode,
                success=False,
                reason_code=code,
                reason_detail=f"{type(ex).__name__}: {ex}",
            )
            _fail(
                code=code,
                message=f"WAN fused cross kernel failed: {type(ex).__name__}: {ex}",
            )
        _record_attempt(
            op_kind="cross",
            mode=fused_mode,
            success=False,
            reason_code=code,
            reason_detail=f"{type(ex).__name__}: {ex}",
        )
        return WanFusedAttemptResult(
            output=None,
            reason_code=code,
            reason_detail=f"{type(ex).__name__}: {ex}",
        )
