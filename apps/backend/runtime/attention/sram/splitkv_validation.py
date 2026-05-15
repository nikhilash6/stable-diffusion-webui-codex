"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Bounded live-validation helpers for the generic SRAM split-KV diagnostics route.
Provides strict request parsing, a CPU-safe mirror of the current split-KV branch gate
(split-count selection + temp-budget clamp), diagnostics-scoped extension auto-build retry, explicit
bottom-right causal oracle math, and operator-useful live receipts that report exact execution
failures instead of hiding expected precondition/runtime outcomes behind HTTP status gymnastics.

Symbols (top-level; keep in sync; no ghosts):
- `SplitKvValidationInvalidRequest` (exception): Raised for invalid diagnostics request payload/config.
- `SplitKvValidationPreconditionFailure` (exception): Raised when environment/capability prerequisites are missing.
- `SplitKvValidationInternalInvariant` (exception): Raised when the live bridge/kernel violates expected invariants.
- `SplitKvValidationRequest` (dataclass): Normalized request payload for the bounded split-KV diagnostics route.
- `SplitKvCaseDefinition` (dataclass): Locked tensor-shape case definition for control/split diagnostics.
- `SplitKvDispatchPrediction` (dataclass): CPU-safe prediction of the internal split-KV branch result.
- `SplitKvCaseResult` (dataclass): One live case execution receipt.
- `SplitKvValidationReport` (dataclass): Structured live diagnostics response for the split-KV route.
- `parse_splitkv_validation_request` (function): Validates and normalizes the bounded request payload.
- `predict_splitkv_dispatch` (function): Mirrors the current CUDA split-count and temp-budget branch gates.
- `run_splitkv_validation` (function): Executes warmup + one control case + one split-taking case and returns a structured report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Final

import torch

from apps.backend.runtime.attention.sram import (
    SramAttentionContractError,
    try_attention_pre_shaped,
    warmup_extension_for_diagnostics,
)

_HEAD_DIM: Final[int] = 128
_WARPS_PER_BLOCK: Final[int] = 4
_KV_TILE_TOKENS: Final[int] = 32
_SPLITKV_MAX_SPLITS: Final[int] = 8
_SPLITKV_TEMP_BUDGET_BYTES: Final[int] = 256 * 1024 * 1024
_DEFAULT_MODE: Final[str] = "force"
_DEFAULT_SEED: Final[int] = 1234
_DEFAULT_ATOL: Final[float] = 3e-2
_DEFAULT_RTOL: Final[float] = 3e-2


class SplitKvValidationInvalidRequest(RuntimeError):
    pass


class SplitKvValidationPreconditionFailure(RuntimeError):
    def __init__(self, *, code: str, message: str):
        super().__init__(str(message))
        self.code = str(code)


class SplitKvValidationInternalInvariant(RuntimeError):
    def __init__(self, *, code: str, message: str):
        super().__init__(str(message))
        self.code = str(code)


@dataclass(frozen=True, slots=True)
class SplitKvValidationRequest:
    mode: str
    seed: int
    atol: float
    rtol: float


@dataclass(frozen=True, slots=True)
class SplitKvCaseDefinition:
    name: str
    batch: int
    heads: int
    q_len: int
    kv_len: int
    is_causal: bool


@dataclass(frozen=True, slots=True)
class SplitKvDispatchPrediction:
    requested_num_splits: int
    effective_num_splits: int
    split_temp_bytes: int | None
    temp_budget_clamped: bool


@dataclass(frozen=True, slots=True)
class SplitKvCaseResult:
    name: str
    batch: int
    heads: int
    q_len: int
    kv_len: int
    is_causal: bool
    requested_num_splits: int
    expected_num_splits: int
    split_temp_bytes: int | None
    temp_budget_clamped: bool
    q_stride: tuple[int, ...]
    out_stride: tuple[int, ...]
    same_stride: bool
    max_abs_diff: float
    allclose: bool


@dataclass(frozen=True, slots=True)
class SplitKvValidationReport:
    ok: bool
    phase: str
    reason_code: str | None
    reason_detail: str | None
    mode: str
    seed: int
    atol: float
    rtol: float
    device_name: str | None
    device_index: int | None
    num_sms: int | None
    control_case: SplitKvCaseResult | None
    split_case: SplitKvCaseResult | None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


_CONTROL_CASE_CANDIDATES: Final[tuple[SplitKvCaseDefinition, ...]] = (
    SplitKvCaseDefinition(name="control-square-causal", batch=1, heads=4, q_len=256, kv_len=256, is_causal=True),
    SplitKvCaseDefinition(name="control-square-noncausal", batch=1, heads=4, q_len=256, kv_len=256, is_causal=False),
    SplitKvCaseDefinition(name="control-longer-q-noncausal", batch=1, heads=8, q_len=128, kv_len=128, is_causal=False),
)

_SPLIT_CASE_CANDIDATES: Final[tuple[SplitKvCaseDefinition, ...]] = (
    SplitKvCaseDefinition(name="split-small-q-long-kv", batch=1, heads=1, q_len=2, kv_len=2048, is_causal=True),
    SplitKvCaseDefinition(name="split-rect-causal", batch=1, heads=1, q_len=1, kv_len=4096, is_causal=True),
    SplitKvCaseDefinition(name="split-rect-noncausal", batch=1, heads=1, q_len=1, kv_len=4096, is_causal=False),
    SplitKvCaseDefinition(name="split-medium-q-long-kv", batch=1, heads=1, q_len=8, kv_len=4096, is_causal=False),
)


def parse_splitkv_validation_request(payload: Any) -> SplitKvValidationRequest:
    if payload is None:
        raise SplitKvValidationInvalidRequest("payload must be a JSON object")
    if not isinstance(payload, dict):
        raise SplitKvValidationInvalidRequest("payload must be a JSON object")
    allowed_keys = {"mode", "seed", "atol", "rtol"}
    unknown_keys = sorted(str(key) for key in payload.keys() if key not in allowed_keys)
    if unknown_keys:
        raise SplitKvValidationInvalidRequest(f"unknown payload keys: {', '.join(unknown_keys)}")
    if "mode" not in payload:
        raise SplitKvValidationInvalidRequest("mode is required and must be 'force'")
    mode = str(payload.get("mode") or "").strip().lower()
    if mode != _DEFAULT_MODE:
        raise SplitKvValidationInvalidRequest("mode must be 'force'")
    try:
        seed = int(payload.get("seed", _DEFAULT_SEED))
    except Exception as exc:
        raise SplitKvValidationInvalidRequest("seed must be an integer") from exc
    atol = _parse_non_negative_float(payload.get("atol", _DEFAULT_ATOL), field_name="atol")
    rtol = _parse_non_negative_float(payload.get("rtol", _DEFAULT_RTOL), field_name="rtol")
    return SplitKvValidationRequest(mode=mode, seed=seed, atol=atol, rtol=rtol)


def predict_splitkv_dispatch(
    case: SplitKvCaseDefinition,
    *,
    num_sms: int,
    temp_budget_bytes: int = _SPLITKV_TEMP_BUDGET_BYTES,
) -> SplitKvDispatchPrediction:
    requested_num_splits = _choose_splitkv_num_splits(case=case, num_sms=num_sms)
    effective_num_splits = requested_num_splits
    split_temp_bytes: int | None = None
    temp_budget_clamped = False
    if requested_num_splits > 1:
        split_temp_bytes = _compute_splitkv_temp_bytes(
            num_splits=requested_num_splits,
            batch=case.batch,
            heads=case.heads,
            q_len=case.q_len,
        )
        if split_temp_bytes is None or split_temp_bytes > int(temp_budget_bytes):
            effective_num_splits = 1
            temp_budget_clamped = True
    return SplitKvDispatchPrediction(
        requested_num_splits=int(requested_num_splits),
        effective_num_splits=int(effective_num_splits),
        split_temp_bytes=split_temp_bytes,
        temp_budget_clamped=bool(temp_budget_clamped),
    )


def run_splitkv_validation(payload: SplitKvValidationRequest) -> SplitKvValidationReport:
    if str(payload.mode).strip().lower() != _DEFAULT_MODE:
        raise SplitKvValidationInvalidRequest("mode must be 'force'")
    if not torch.cuda.is_available():
        return _build_failure_report(
            request=payload,
            phase="preflight",
            reason_code="E_SRAM_SPLITKV_CUDA_UNAVAILABLE",
            reason_detail="CUDA is unavailable for SRAM split-KV validation",
        )
    try:
        warmup = warmup_extension_for_diagnostics(mode=payload.mode)
    except SramAttentionContractError as exc:
        return _build_failure_report(
            request=payload,
            phase="warmup",
            reason_code=str(exc.code),
            reason_detail=str(exc),
        )
    if not bool(warmup.loaded) or not bool(warmup.ready):
        return _build_failure_report(
            request=payload,
            phase="warmup",
            reason_code="E_SRAM_SPLITKV_WARMUP_NOT_READY",
            reason_detail=(
                f"SRAM extension warmup failed: loaded={warmup.loaded} "
                f"ready={warmup.ready} detail={warmup.detail!r}"
            ),
        )
    device_index = int(torch.cuda.current_device())
    device_properties = torch.cuda.get_device_properties(device_index)
    device_name = str(torch.cuda.get_device_name(device_index))
    num_sms = int(device_properties.multi_processor_count)
    control_case = _select_case(_CONTROL_CASE_CANDIDATES, num_sms=num_sms, want_split=False)
    split_case = _select_case(_SPLIT_CASE_CANDIDATES, num_sms=num_sms, want_split=True)
    if control_case is None:
        return _build_failure_report(
            request=payload,
            phase="case_selection",
            reason_code="E_SRAM_SPLITKV_CONTROL_CASE_UNAVAILABLE",
            reason_detail="No non-split control tuple matched the current device split-KV gate",
            device_name=device_name,
            device_index=device_index,
            num_sms=num_sms,
        )
    if split_case is None:
        return _build_failure_report(
            request=payload,
            phase="case_selection",
            reason_code="E_SRAM_SPLITKV_SPLIT_CASE_UNAVAILABLE",
            reason_detail="No split-taking tuple matched the current device split-KV gate",
            device_name=device_name,
            device_index=device_index,
            num_sms=num_sms,
        )
    torch.manual_seed(int(payload.seed))
    try:
        control_result = _run_live_case(
            case=control_case,
            request=payload,
            device_index=device_index,
            num_sms=num_sms,
        )
    except SramAttentionContractError as exc:
        return _build_failure_report(
            request=payload,
            phase="control_case",
            reason_code=str(exc.code),
            reason_detail=str(exc),
            device_name=device_name,
            device_index=device_index,
            num_sms=num_sms,
        )
    except SplitKvValidationPreconditionFailure as exc:
        return _build_failure_report(
            request=payload,
            phase="control_case",
            reason_code=str(exc.code),
            reason_detail=str(exc),
            device_name=device_name,
            device_index=device_index,
            num_sms=num_sms,
        )
    except SplitKvValidationInternalInvariant as exc:
        return _build_failure_report(
            request=payload,
            phase="control_case",
            reason_code=str(exc.code),
            reason_detail=str(exc),
            device_name=device_name,
            device_index=device_index,
            num_sms=num_sms,
        )
    try:
        split_result = _run_live_case(
            case=split_case,
            request=payload,
            device_index=device_index,
            num_sms=num_sms,
        )
    except SramAttentionContractError as exc:
        return _build_failure_report(
            request=payload,
            phase="split_case",
            reason_code=str(exc.code),
            reason_detail=str(exc),
            device_name=device_name,
            device_index=device_index,
            num_sms=num_sms,
            control_case=control_result,
        )
    except SplitKvValidationPreconditionFailure as exc:
        return _build_failure_report(
            request=payload,
            phase="split_case",
            reason_code=str(exc.code),
            reason_detail=str(exc),
            device_name=device_name,
            device_index=device_index,
            num_sms=num_sms,
            control_case=control_result,
        )
    except SplitKvValidationInternalInvariant as exc:
        return _build_failure_report(
            request=payload,
            phase="split_case",
            reason_code=str(exc.code),
            reason_detail=str(exc),
            device_name=device_name,
            device_index=device_index,
            num_sms=num_sms,
            control_case=control_result,
        )
    return SplitKvValidationReport(
        ok=True,
        phase="completed",
        reason_code=None,
        reason_detail=None,
        mode=payload.mode,
        seed=int(payload.seed),
        atol=float(payload.atol),
        rtol=float(payload.rtol),
        device_name=device_name,
        device_index=device_index,
        num_sms=num_sms,
        control_case=control_result,
        split_case=split_result,
    )


def _build_failure_report(
    *,
    request: SplitKvValidationRequest,
    phase: str,
    reason_code: str,
    reason_detail: str,
    device_name: str | None = None,
    device_index: int | None = None,
    num_sms: int | None = None,
    control_case: SplitKvCaseResult | None = None,
    split_case: SplitKvCaseResult | None = None,
) -> SplitKvValidationReport:
    return SplitKvValidationReport(
        ok=False,
        phase=str(phase),
        reason_code=str(reason_code),
        reason_detail=str(reason_detail),
        mode=request.mode,
        seed=int(request.seed),
        atol=float(request.atol),
        rtol=float(request.rtol),
        device_name=device_name,
        device_index=device_index,
        num_sms=num_sms,
        control_case=control_case,
        split_case=split_case,
    )


def _parse_non_negative_float(raw_value: Any, *, field_name: str) -> float:
    try:
        numeric = float(raw_value)
    except Exception as exc:
        raise SplitKvValidationInvalidRequest(f"{field_name} must be a finite number >= 0") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise SplitKvValidationInvalidRequest(f"{field_name} must be a finite number >= 0")
    return float(numeric)


def _ceil_div(value: int, divisor: int) -> int:
    if value == 0:
        return 0
    return 1 + ((int(value) - 1) // int(divisor))


def _choose_splitkv_num_splits(*, case: SplitKvCaseDefinition, num_sms: int) -> int:
    if int(num_sms) <= 0:
        return 1
    num_k_tiles = _ceil_div(case.kv_len, _KV_TILE_TOKENS)
    if num_k_tiles <= 1:
        return 1
    num_q_blocks = _ceil_div(case.q_len, _WARPS_PER_BLOCK)
    work_tiles = int(case.batch) * int(case.heads) * int(num_q_blocks)
    occupancy_target = max(1, _ceil_div(int(num_sms) * 4, 5))
    requested_splits = _ceil_div(occupancy_target, max(1, work_tiles))
    num_splits = min(requested_splits, num_k_tiles, _SPLITKV_MAX_SPLITS)
    if num_splits <= 1:
        return 1
    while num_splits > 1 and _ceil_div(num_k_tiles, num_splits) == _ceil_div(num_k_tiles, num_splits - 1):
        num_splits -= 1
    return max(1, int(num_splits))


def _compute_splitkv_temp_bytes(*, num_splits: int, batch: int, heads: int, q_len: int) -> int | None:
    try:
        stats_elements = int(num_splits) * int(batch) * int(heads) * int(q_len)
        acc_elements = stats_elements * _HEAD_DIM
        total_elements = (stats_elements * 2) + acc_elements
        return int(total_elements * 4)
    except Exception:
        return None


def _select_case(
    cases: tuple[SplitKvCaseDefinition, ...],
    *,
    num_sms: int,
    want_split: bool,
) -> SplitKvCaseDefinition | None:
    for case in cases:
        prediction = predict_splitkv_dispatch(case, num_sms=num_sms)
        is_split = prediction.effective_num_splits > 1
        if is_split == bool(want_split):
            return case
    return None


def _allowed(q_index: int, kv_index: int, *, q_len: int, kv_len: int, is_causal: bool) -> bool:
    if not is_causal:
        return True
    return int(kv_index) <= int(q_index) + (int(kv_len) - int(q_len))


def _direct_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    batch, heads, q_len, head_dim = q.shape
    kv_len = k.shape[2]
    out = torch.zeros((batch, heads, q_len, head_dim), dtype=torch.float32)
    scale = 1.0 / math.sqrt(float(head_dim))
    for batch_index in range(batch):
        for head_index in range(heads):
            for q_index in range(q_len):
                scores: list[float] = []
                values: list[torch.Tensor] = []
                for kv_index in range(kv_len):
                    if not _allowed(q_index, kv_index, q_len=q_len, kv_len=kv_len, is_causal=is_causal):
                        continue
                    score = torch.dot(q[batch_index, head_index, q_index], k[batch_index, head_index, kv_index]).item() * scale
                    scores.append(float(score))
                    values.append(v[batch_index, head_index, kv_index])
                if not scores:
                    continue
                weights = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0)
                out[batch_index, head_index, q_index] = (
                    weights[:, None] * torch.stack(values).to(torch.float32)
                ).sum(dim=0)
    return out


def _run_live_case(
    *,
    case: SplitKvCaseDefinition,
    request: SplitKvValidationRequest,
    device_index: int,
    num_sms: int,
) -> SplitKvCaseResult:
    prediction = predict_splitkv_dispatch(case, num_sms=num_sms)
    device = torch.device(f"cuda:{device_index}")
    q = torch.randn((case.batch, case.q_len, case.heads, _HEAD_DIM), device=device, dtype=torch.float16).permute(0, 2, 1, 3)
    k = torch.randn((case.batch, case.kv_len, case.heads, _HEAD_DIM), device=device, dtype=torch.float16).permute(0, 2, 1, 3)
    v = torch.randn((case.batch, case.kv_len, case.heads, _HEAD_DIM), device=device, dtype=torch.float16).permute(0, 2, 1, 3)
    ref = _direct_attention(q=q.float().cpu(), k=k.float().cpu(), v=v.float().cpu(), is_causal=case.is_causal)
    ref = ref.to(torch.float16).to(torch.float32)
    result = try_attention_pre_shaped(mode=request.mode, q=q, k=k, v=v, is_causal=case.is_causal)
    if result.output is None:
        raise SplitKvValidationPreconditionFailure(
            code=str(result.reason_code or "E_SRAM_SPLITKV_ATTENTION_FALLBACK"),
            message=(
                "SRAM bridge returned fallback on a locked diagnostics tuple: "
                f"reason_code={result.reason_code!r} reason_detail={result.reason_detail!r}"
            ),
        )
    same_stride = tuple(int(value) for value in result.output.stride()) == tuple(int(value) for value in q.stride())
    out = result.output.float().cpu()
    max_abs_diff = float((out - ref).abs().max().item())
    allclose = bool(torch.allclose(out, ref, atol=request.atol, rtol=request.rtol))
    if not same_stride:
        raise SplitKvValidationInternalInvariant(
            code="E_SRAM_SPLITKV_OUTPUT_STRIDE_DRIFT",
            message=f"{case.name}: output stride no longer matches q layout",
        )
    if not allclose:
        raise SplitKvValidationInternalInvariant(
            code="E_SRAM_SPLITKV_PARITY_FAILED",
            message=f"{case.name}: parity failed (max_abs_diff={max_abs_diff:.6f}, atol={request.atol}, rtol={request.rtol})",
        )
    return SplitKvCaseResult(
        name=case.name,
        batch=case.batch,
        heads=case.heads,
        q_len=case.q_len,
        kv_len=case.kv_len,
        is_causal=case.is_causal,
        requested_num_splits=prediction.requested_num_splits,
        expected_num_splits=prediction.effective_num_splits,
        split_temp_bytes=prediction.split_temp_bytes,
        temp_budget_clamped=prediction.temp_budget_clamped,
        q_stride=tuple(int(value) for value in q.stride()),
        out_stride=tuple(int(value) for value in result.output.stride()),
        same_stride=same_stride,
        max_abs_diff=max_abs_diff,
        allclose=allclose,
    )
