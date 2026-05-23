"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Q8_0 stored-scale underflow analysis for GGUF converter plans.
Scans planned Q8_0 tensors before GGUF header emission and promotes materially affected tensors to float storage.

Symbols (top-level; keep in sync; no ghosts):
- `Q8UnderflowReport` (dataclass): Per-tensor scan/promotion report for Q8_0 stored-scale underflow.
- `Q8UnderflowPromotionResult` (dataclass): Updated tensor plans plus underflow reports.
- `q8_stored_scale_underflow_mask` (function): Returns blocks whose Q8_0 stored float16 scale would become zero.
- `apply_q8_underflow_promotions` (function): Scans Q8_0 plans and returns plans with material tensors promoted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from apps.backend.quantization.gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from apps.backend.runtime.tools.gguf_converter_tensor_planner import TensorPlan, retype_tensor_plan

Q8_UNDERFLOW_BLOCK_SIZE = GGML_QUANT_SIZES[GGMLQuantizationType.Q8_0][0]
Q8_UNDERFLOW_SCALE_DIVISOR = np.float32(127.0)
Q8_UNDERFLOW_MATERIAL_RATIO = 0.001
Q8_UNDERFLOW_MATERIAL_BLOCKS = 1024
Q8_UNDERFLOW_SMALL_FLOAT_EXTRA_BYTES = 16 * 1024 * 1024

_FLOAT_GGML_TYPES = frozenset(
    {
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
        GGMLQuantizationType.F32,
    }
)


@dataclass(frozen=True, slots=True)
class Q8UnderflowReport:
    src_name: str
    gguf_name: str
    raw_shape: tuple[int, ...]
    source_ggml_type: GGMLQuantizationType | None
    total_blocks: int
    affected_blocks: int
    affected_ratio: float
    planned_q8_nbytes: int
    material: bool
    promoted: bool
    selected_ggml_type: GGMLQuantizationType | None
    selected_nbytes: int | None
    size_delta_bytes: int | None
    reason: str


@dataclass(frozen=True, slots=True)
class Q8UnderflowPromotionResult:
    plans: list[TensorPlan]
    reports: tuple[Q8UnderflowReport, ...]


def q8_stored_scale_underflow_mask(blocks: np.ndarray) -> np.ndarray:
    """Return Q8_0 blocks whose non-zero data cannot be represented by the stored FP16 scale."""

    if blocks.ndim != 2 or blocks.shape[1] != Q8_UNDERFLOW_BLOCK_SIZE:
        raise ValueError(
            f"Q8_0 underflow analysis expects (n_blocks, {Q8_UNDERFLOW_BLOCK_SIZE}), got {blocks.shape}"
        )
    values = blocks.astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("Q8_0 underflow analysis received non-finite values")

    max_abs = np.abs(values).max(axis=1)
    nonzero_blocks = max_abs > np.float32(0.0)
    stored_scale = (max_abs / Q8_UNDERFLOW_SCALE_DIVISOR).astype(np.float16)
    return nonzero_blocks & (stored_scale == np.float16(0.0))


def apply_q8_underflow_promotions(
    plans: Sequence[TensorPlan],
    safetensors_handle: Any,
    *,
    check_cancel: Callable[[], None],
    chunk_rows: int,
    on_scan_tensor: Callable[[TensorPlan], None] | None = None,
) -> Q8UnderflowPromotionResult:
    updated_plans = list(plans)
    reports: list[Q8UnderflowReport] = []

    for plan_index, plan in enumerate(plans):
        check_cancel()
        if plan.ggml_type != GGMLQuantizationType.Q8_0:
            continue
        if len(plan.raw_shape) < 2:
            continue
        if plan.op != "copy":
            raise RuntimeError(
                f"Q8_0 underflow analysis does not support tensor op {plan.op!r} for {plan.gguf_name!r}"
            )

        if on_scan_tensor is not None:
            on_scan_tensor(plan)
        report = _scan_q8_copy_plan(plan, safetensors_handle, check_cancel=check_cancel, chunk_rows=chunk_rows)
        if report.affected_blocks == 0:
            continue

        selected_ggml_type, selected_nbytes, reason = _select_promotion_target(plan, report)
        if selected_ggml_type is not None:
            updated_plans[plan_index] = retype_tensor_plan(plan, selected_ggml_type)
            report = replace(
                report,
                promoted=True,
                selected_ggml_type=selected_ggml_type,
                selected_nbytes=selected_nbytes,
                size_delta_bytes=selected_nbytes - plan.stored_nbytes,
                reason=reason,
            )
        else:
            report = replace(report, reason=reason)
        reports.append(report)

    return Q8UnderflowPromotionResult(plans=updated_plans, reports=tuple(reports))


def _scan_q8_copy_plan(
    plan: TensorPlan,
    safetensors_handle: Any,
    *,
    check_cancel: Callable[[], None],
    chunk_rows: int,
) -> Q8UnderflowReport:
    if plan.raw_shape[-1] % Q8_UNDERFLOW_BLOCK_SIZE != 0:
        raise RuntimeError(
            f"Q8_0 planned tensor {plan.gguf_name!r} has incompatible last dimension {plan.raw_shape[-1]}"
        )

    total_blocks = 0
    affected_blocks = 0
    safetensors_slice = safetensors_handle.get_slice(plan.src_name)
    shape = tuple(int(dimension) for dimension in safetensors_slice.get_shape())
    if shape != plan.raw_shape:
        raise RuntimeError(f"Tensor shape changed during Q8_0 underflow analysis for {plan.src_name}: {shape} vs {plan.raw_shape}")

    if len(shape) == 2:
        rows = shape[0]
        for start in range(0, rows, chunk_rows):
            check_cancel()
            stop = min(rows, start + chunk_rows)
            chunk = safetensors_slice[start:stop].to(torch.float32).contiguous()
            chunk_total, chunk_affected = _count_q8_underflow_blocks(
                chunk.numpy(),
                tensor_name=plan.gguf_name,
            )
            total_blocks += chunk_total
            affected_blocks += chunk_affected
    else:
        check_cancel()
        tensor = safetensors_handle.get_tensor(plan.src_name).to(torch.float32).contiguous()
        total_blocks, affected_blocks = _count_q8_underflow_blocks(tensor.numpy(), tensor_name=plan.gguf_name)

    affected_ratio = (affected_blocks / total_blocks) if total_blocks else 0.0
    minimal_float_nbytes = _minimal_float_promotion_nbytes(plan)
    small_float_delta = minimal_float_nbytes - plan.stored_nbytes
    material = affected_blocks > 0 and (
        affected_ratio >= Q8_UNDERFLOW_MATERIAL_RATIO
        or affected_blocks >= Q8_UNDERFLOW_MATERIAL_BLOCKS
        or small_float_delta <= Q8_UNDERFLOW_SMALL_FLOAT_EXTRA_BYTES
    )

    return Q8UnderflowReport(
        src_name=plan.src_name,
        gguf_name=plan.gguf_name,
        raw_shape=plan.raw_shape,
        source_ggml_type=plan.source_ggml_type,
        total_blocks=total_blocks,
        affected_blocks=affected_blocks,
        affected_ratio=affected_ratio,
        planned_q8_nbytes=plan.stored_nbytes,
        material=material,
        promoted=False,
        selected_ggml_type=None,
        selected_nbytes=None,
        size_delta_bytes=None,
        reason="below-material-thresholds" if not material else "material",
    )


def _count_q8_underflow_blocks(data: np.ndarray, *, tensor_name: str) -> tuple[int, int]:
    if data.ndim < 2:
        raise ValueError(f"Q8_0 underflow analysis expected rank >= 2 for {tensor_name!r}, got {data.shape}")
    last_dim = int(data.shape[-1])
    if last_dim % Q8_UNDERFLOW_BLOCK_SIZE != 0:
        raise ValueError(
            f"Q8_0 underflow analysis tensor {tensor_name!r} last dim {last_dim} is not divisible by {Q8_UNDERFLOW_BLOCK_SIZE}"
        )

    flat = data.astype(np.float32, copy=False).reshape((-1, last_dim))
    total_blocks = int(flat.size // Q8_UNDERFLOW_BLOCK_SIZE)
    if total_blocks == 0:
        return 0, 0
    blocks = flat.reshape((total_blocks, Q8_UNDERFLOW_BLOCK_SIZE))
    try:
        affected_mask = q8_stored_scale_underflow_mask(blocks)
    except ValueError as exc:
        raise ValueError(f"Failed Q8_0 underflow analysis for tensor {tensor_name!r}: {exc}") from exc
    return total_blocks, int(np.count_nonzero(affected_mask))


def _select_promotion_target(
    plan: TensorPlan,
    report: Q8UnderflowReport,
) -> tuple[GGMLQuantizationType | None, int | None, str]:
    if not report.material:
        return None, None, "below-material-thresholds"

    source_type = plan.source_ggml_type
    f32_nbytes = _float_storage_nbytes(plan.raw_shape, GGMLQuantizationType.F32)
    if source_type == GGMLQuantizationType.F32:
        return GGMLQuantizationType.F32, f32_nbytes, "source-f32"

    if source_type in {GGMLQuantizationType.F16, GGMLQuantizationType.BF16}:
        source_nbytes = _float_storage_nbytes(plan.raw_shape, source_type)
        if f32_nbytes - source_nbytes <= Q8_UNDERFLOW_SMALL_FLOAT_EXTRA_BYTES:
            return GGMLQuantizationType.F32, f32_nbytes, "small-f32-delta"
        return source_type, source_nbytes, "source-dtype"

    f32_extra_over_q8 = f32_nbytes - plan.stored_nbytes
    if f32_extra_over_q8 <= Q8_UNDERFLOW_SMALL_FLOAT_EXTRA_BYTES:
        return GGMLQuantizationType.F32, f32_nbytes, "unknown-source-small-f32"

    raise RuntimeError(
        f"Q8_0 stored-scale underflow is material for tensor {plan.gguf_name!r} "
        f"({report.affected_blocks}/{report.total_blocks} blocks, ratio={report.affected_ratio:.6g}), "
        "but source floating dtype metadata is unavailable or unsupported and F32 promotion would add "
        f"{f32_extra_over_q8} bytes over Q8_0."
    )


def _minimal_float_promotion_nbytes(plan: TensorPlan) -> int:
    if plan.source_ggml_type in _FLOAT_GGML_TYPES:
        return _float_storage_nbytes(plan.raw_shape, plan.source_ggml_type)
    return _float_storage_nbytes(plan.raw_shape, GGMLQuantizationType.F32)


def _float_storage_nbytes(raw_shape: tuple[int, ...], ggml_type: GGMLQuantizationType) -> int:
    element_count = int(np.prod(raw_shape, dtype=np.int64))
    if ggml_type in {GGMLQuantizationType.F16, GGMLQuantizationType.BF16}:
        return element_count * 2
    if ggml_type == GGMLQuantizationType.F32:
        return element_count * 4
    raise ValueError(f"Expected F16/BF16/F32, got {ggml_type.name}")


__all__ = [
    "Q8UnderflowPromotionResult",
    "Q8UnderflowReport",
    "apply_q8_underflow_promotions",
    "q8_stored_scale_underflow_mask",
]
