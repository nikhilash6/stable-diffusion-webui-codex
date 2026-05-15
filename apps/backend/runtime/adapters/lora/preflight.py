"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared structural LoRA preflight helpers for repo-owned runtime seams.
Provides target-shape resolution for plain and sliced patch targets, structural validation for parsed LoRA patch dictionaries,
and a cheap SafeTensors header-only fast path for standard LoRA/DIFF/SET layouts before any runtime mutation occurs.

Symbols (top-level; keep in sync; no ghosts):
- `HeaderPreflightResult` (dataclass): Result bundle for the cheap header-only fast path.
- `ShapeCompatibilitySummary` (dataclass): Counts + mismatch samples for structural patch validation.
- `collect_parameter_shapes` (function): Collects live parameter shapes from a module by runtime parameter name.
- `resolve_patch_target_shape` (function): Resolves the effective target shape for a plain or sliced `PatchTarget`.
- `shapeify_patch_dict` (function): Converts a parsed patch dict into a shape-only representation for structural validation.
- `build_standard_shape_patch_dict_from_shape_map` (function): Builds a cheap shape patch dict from an already-available tensor-name -> shape map.
- `build_standard_shape_patch_dict_from_safetensors` (function): Builds a cheap header-only shape patch dict for standard LoRA/DIFF/SET tensors.
- `validate_shape_patch_dict` (function): Validates structural compatibility between a shape patch dict and live target shapes.
- `format_shape_compatibility_samples` (function): Formats bounded mismatch samples for fail-loud diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from apps.backend.runtime.adapters.base import PatchTarget
from apps.backend.runtime.adapters.lora.loader import (
    STANDARD_LORA_TENSOR_CANDIDATES,
    _bias_target_for,
    _maybe_convert_bfl_control,
    _modulation_tensor_name_candidates,
)
from apps.backend.runtime.checkpoint.safetensors_header import read_safetensors_tensor_shapes


Shape = tuple[int, ...]
ShapePatchPayload = tuple[Any, ...]
ShapePatchDict = dict[PatchTarget, tuple[str, ShapePatchPayload]]

_NONSTANDARD_SUFFIX_GROUPS: tuple[tuple[str, ...], ...] = (
    ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "hada_t1", "hada_t2"),
    ("lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b", "lokr_t2"),
    ("a1.weight", "a2.weight", "b1.weight", "b2.weight"),
)
_HEADER_ONLY_SUFFIXES: tuple[str, ...] = (
    ".alpha",
    ".dora_scale",
    ".diff",
    ".diff_b",
    ".set_weight",
    ".diff_m",
)


@dataclass(frozen=True, slots=True)
class HeaderPreflightResult:
    shape_patch_dict: ShapePatchDict
    requires_materialized_preflight: bool
    matched_targets: int


@dataclass(frozen=True, slots=True)
class ShapeMismatchRecord:
    target_key: str
    kind: str
    target_shape: Shape
    detail: str


@dataclass(frozen=True, slots=True)
class ShapeCompatibilitySummary:
    total_targets: int
    compatible_targets: int
    mismatches: tuple[ShapeMismatchRecord, ...]


def _shape_tuple(value: Any) -> Shape:
    shape = getattr(value, "shape", value)
    if not isinstance(shape, (tuple, list)):
        raise RuntimeError(f"Expected shape-like value, got {type(value).__name__}.")
    return tuple(int(dim) for dim in shape)


def _shape_numel(shape: Shape) -> int:
    if not shape:
        return 1
    return math.prod(int(dim) for dim in shape)


def _flatten_start_dim_one(shape: Shape) -> Shape:
    if not shape:
        raise RuntimeError("LoRA tensor shape must have rank >= 1.")
    return (int(shape[0]), int(_shape_numel(shape[1:])))


def _matmul_shape(left: Shape, right: Shape, *, label: str) -> Shape:
    if len(left) != 2 or len(right) != 2:
        raise RuntimeError(f"{label} requires rank-2 inputs, got left={left!r} right={right!r}.")
    if int(left[1]) != int(right[0]):
        raise RuntimeError(f"{label} inner-dim mismatch: left={left!r} right={right!r}.")
    return (int(left[0]), int(right[1]))


def _tucker_pair_shape(core: Shape, right: Shape, left: Shape, *, label: str) -> Shape:
    if len(core) != 4 or len(right) != 2 or len(left) != 2:
        raise RuntimeError(
            f"{label} Tucker path requires core rank-4 and factor rank-2 tensors; "
            f"got core={core!r} right={right!r} left={left!r}."
        )
    if int(core[1]) != int(right[0]) or int(core[0]) != int(left[0]):
        raise RuntimeError(
            f"{label} Tucker factor mismatch: core={core!r} right={right!r} left={left!r}."
        )
    return (int(left[1]), int(right[1]), int(core[2]), int(core[3]))


def _kron_shape(left: Shape, right: Shape) -> Shape:
    max_rank = max(len(left), len(right))
    padded_left = (1,) * (max_rank - len(left)) + tuple(left)
    padded_right = (1,) * (max_rank - len(right)) + tuple(right)
    return tuple(int(a) * int(b) for a, b in zip(padded_left, padded_right))


def _register_shape_patch(
    patch_dict: ShapePatchDict,
    *,
    parameter: PatchTarget,
    kind: str,
    payload: ShapePatchPayload,
) -> None:
    existing = patch_dict.get(parameter)
    if existing is not None:
        existing_kind, _existing_payload = existing
        raise RuntimeError(
            "LoRA structural preflight patch collision: multiple variants target the same parameter "
            f"{parameter!r} (existing={existing_kind!r}, incoming={kind!r})."
        )
    patch_dict[parameter] = (kind, payload)


def _has_nonstandard_family(shape_map: Mapping[str, Shape], logical_key: str) -> bool:
    base = f"{logical_key}."
    for suffix_group in _NONSTANDARD_SUFFIX_GROUPS:
        if any(f"{base}{suffix}" in shape_map for suffix in suffix_group):
            return True
    return False


def collect_parameter_shapes(model: torch.nn.Module) -> dict[str, Shape]:
    return {str(name): _shape_tuple(parameter) for name, parameter in model.named_parameters()}


def resolve_patch_target_shape(target_shape_by_key: Mapping[str, Shape], target: PatchTarget) -> tuple[str, Shape]:
    base_key, offset = (target if isinstance(target, tuple) else (target, None))
    base_key = str(base_key)
    full_shape = target_shape_by_key.get(base_key)
    if full_shape is None:
        raise RuntimeError(f"LoRA structural preflight target is missing on the active model: {base_key!r}.")
    if offset is None:
        return base_key, full_shape
    dim, start, length = (int(offset[0]), int(offset[1]), int(offset[2]))
    rank = len(full_shape)
    if dim < 0 or dim >= rank:
        raise RuntimeError(
            f"LoRA structural preflight slice target {base_key!r} has invalid dim={dim} for shape={full_shape!r}."
        )
    if start < 0 or length <= 0 or start + length > int(full_shape[dim]):
        raise RuntimeError(
            "LoRA structural preflight slice target {key!r} has invalid narrow range "
            "(dim={dim}, start={start}, length={length}) for shape={shape!r}.".format(
                key=base_key,
                dim=dim,
                start=start,
                length=length,
                shape=full_shape,
            )
        )
    narrowed = list(full_shape)
    narrowed[dim] = int(length)
    return base_key, tuple(narrowed)


def shapeify_patch_dict(patch_dict: Mapping[PatchTarget, tuple[str, tuple[Any, ...]]]) -> ShapePatchDict:
    shape_patch_dict: ShapePatchDict = {}
    for parameter, (kind, payload) in patch_dict.items():
        shape_payload: list[Any] = []
        for item in payload:
            if item is None or isinstance(item, (int, float)):
                shape_payload.append(item)
                continue
            shape_payload.append(_shape_tuple(item))
        shape_patch_dict[parameter] = (str(kind), tuple(shape_payload))
    return shape_patch_dict


def _extract_standard_shape_patch(
    logical_key: str,
    target_param: PatchTarget,
    shape_map: Mapping[str, Shape],
) -> tuple[str, ShapePatchPayload] | None:
    for up_suffix, down_suffix, mid_suffix in STANDARD_LORA_TENSOR_CANDIDATES:
        up_key = f"{logical_key}{up_suffix}"
        down_key = f"{logical_key}{down_suffix}"
        mid_key = f"{logical_key}{mid_suffix}" if mid_suffix is not None else None
        if up_key not in shape_map or down_key not in shape_map:
            continue
        return (
            "lora",
            (
                shape_map[up_key],
                shape_map[down_key],
                None,
                shape_map[mid_key] if mid_key is not None and mid_key in shape_map else None,
                shape_map.get(f"{logical_key}.dora_scale"),
            ),
        )
    return None


def build_standard_shape_patch_dict_from_shape_map(
    shape_map: Mapping[str, Shape],
    *,
    to_load: Mapping[str, PatchTarget],
) -> HeaderPreflightResult:
    shape_map = _maybe_convert_bfl_control(shape_map)
    patch_dict: ShapePatchDict = {}
    requires_materialized_preflight = False

    for logical_key, target_param in to_load.items():
        if _has_nonstandard_family(shape_map, logical_key):
            requires_materialized_preflight = True
            continue

        standard = _extract_standard_shape_patch(logical_key, target_param, shape_map)
        if standard is not None:
            kind, payload = standard
            _register_shape_patch(patch_dict, parameter=target_param, kind=kind, payload=payload)

        diff_key = f"{logical_key}.diff"
        if diff_key in shape_map:
            _register_shape_patch(
                patch_dict,
                parameter=target_param,
                kind="diff",
                payload=(shape_map[diff_key],),
            )
        modulation_key = next(
            (name for name in _modulation_tensor_name_candidates(logical_key) if name in shape_map),
            None,
        )
        if modulation_key is not None:
            _register_shape_patch(
                patch_dict,
                parameter=target_param,
                kind="diff",
                payload=(shape_map[modulation_key],),
            )
        bias_key = f"{logical_key}.diff_b"
        if bias_key in shape_map:
            _register_shape_patch(
                patch_dict,
                parameter=_bias_target_for(target_param),
                kind="diff",
                payload=(shape_map[bias_key],),
            )
        set_key = f"{logical_key}.set_weight"
        if set_key in shape_map:
            _register_shape_patch(
                patch_dict,
                parameter=target_param,
                kind="set",
                payload=(shape_map[set_key],),
            )

    return HeaderPreflightResult(
        shape_patch_dict=patch_dict,
        requires_materialized_preflight=requires_materialized_preflight,
        matched_targets=len(patch_dict),
    )


def build_standard_shape_patch_dict_from_safetensors(
    path: str | Path,
    *,
    to_load: Mapping[str, PatchTarget],
) -> HeaderPreflightResult:
    return build_standard_shape_patch_dict_from_shape_map(
        read_safetensors_tensor_shapes(Path(path)),
        to_load=to_load,
    )


def _validate_diff_payload(
    *,
    diff_shape: Shape,
    target_shape: Shape,
    is_slice: bool,
    target_key: str,
) -> ShapeMismatchRecord | None:
    if diff_shape == target_shape:
        return None
    if not is_slice and len(diff_shape) == len(target_shape) == 4:
        return None
    return ShapeMismatchRecord(
        target_key=target_key,
        kind="diff",
        target_shape=target_shape,
        detail=f"diff_shape={diff_shape!r}",
    )


def _validate_set_payload(
    *,
    set_shape: Shape,
    target_shape: Shape,
    target_key: str,
) -> ShapeMismatchRecord | None:
    if set_shape == target_shape:
        return None
    return ShapeMismatchRecord(
        target_key=target_key,
        kind="set",
        target_shape=target_shape,
        detail=f"set_shape={set_shape!r}",
    )


def _validate_lora_payload(
    *,
    payload: ShapePatchPayload,
    target_shape: Shape,
    target_key: str,
) -> ShapeMismatchRecord | None:
    up_shape = _shape_tuple(payload[0])
    down_shape = _shape_tuple(payload[1])
    mid_shape = None if payload[3] is None else _shape_tuple(payload[3])
    up_flat = _flatten_start_dim_one(up_shape)
    down_flat = _flatten_start_dim_one(down_shape)
    if up_flat[1] != down_flat[0]:
        return ShapeMismatchRecord(
            target_key=target_key,
            kind="lora",
            target_shape=target_shape,
            detail=f"rank_mismatch up={up_shape!r} down={down_shape!r}",
        )
    if mid_shape is None:
        diff_numel = int(up_flat[0]) * int(down_flat[1])
        if diff_numel == _shape_numel(target_shape):
            return None
        return ShapeMismatchRecord(
            target_key=target_key,
            kind="lora",
            target_shape=target_shape,
            detail=f"expected_numel={diff_numel} up={up_shape!r} down={down_shape!r}",
        )
    if len(mid_shape) != 4 or len(down_shape) < 2:
        return ShapeMismatchRecord(
            target_key=target_key,
            kind="lora",
            target_shape=target_shape,
            detail=f"unsupported_mid_shapes down={down_shape!r} mid={mid_shape!r}",
        )
    if int(down_shape[0]) != int(mid_shape[0]) or int(down_shape[0]) != int(mid_shape[1]):
        return ShapeMismatchRecord(
            target_key=target_key,
            kind="lora",
            target_shape=target_shape,
            detail=f"mid_rank_mismatch down={down_shape!r} mid={mid_shape!r}",
        )
    diff_shape = (int(up_flat[0]), int(down_shape[1]), int(mid_shape[2]), int(mid_shape[3]))
    if _shape_numel(diff_shape) == _shape_numel(target_shape):
        return None
    return ShapeMismatchRecord(
        target_key=target_key,
        kind="lora",
        target_shape=target_shape,
        detail=f"expected_numel={_shape_numel(diff_shape)} up={up_shape!r} down={down_shape!r} mid={mid_shape!r}",
    )


def _validate_loha_payload(
    *,
    payload: ShapePatchPayload,
    target_shape: Shape,
    target_key: str,
) -> ShapeMismatchRecord | None:
    w1_a = _shape_tuple(payload[0])
    w1_b = _shape_tuple(payload[1])
    w2_a = _shape_tuple(payload[3])
    w2_b = _shape_tuple(payload[4])
    t1 = None if payload[5] is None else _shape_tuple(payload[5])
    t2 = None if payload[6] is None else _shape_tuple(payload[6])
    try:
        if t1 is None and t2 is None:
            m1 = _matmul_shape(w1_a, w1_b, label="LoHa.w1")
            m2 = _matmul_shape(w2_a, w2_b, label="LoHa.w2")
        elif t1 is not None and t2 is not None:
            m1 = _tucker_pair_shape(t1, w1_b, w1_a, label="LoHa.t1")
            m2 = _tucker_pair_shape(t2, w2_b, w2_a, label="LoHa.t2")
        else:
            raise RuntimeError("LoHa Tucker tensors must be present as a complete pair.")
    except RuntimeError as exc:
        return ShapeMismatchRecord(
            target_key=target_key,
            kind="loha",
            target_shape=target_shape,
            detail=str(exc),
        )
    if m1 != m2:
        return ShapeMismatchRecord(
            target_key=target_key,
            kind="loha",
            target_shape=target_shape,
            detail=f"diff_shape_mismatch m1={m1!r} m2={m2!r}",
        )
    if _shape_numel(m1) == _shape_numel(target_shape):
        return None
    return ShapeMismatchRecord(
        target_key=target_key,
        kind="loha",
        target_shape=target_shape,
        detail=f"expected_numel={_shape_numel(m1)} m={m1!r}",
    )


def _validate_lokr_payload(
    *,
    payload: ShapePatchPayload,
    target_shape: Shape,
    target_key: str,
) -> ShapeMismatchRecord | None:
    w1 = None if payload[0] is None else _shape_tuple(payload[0])
    w2 = None if payload[1] is None else _shape_tuple(payload[1])
    w1_a = None if payload[3] is None else _shape_tuple(payload[3])
    w1_b = None if payload[4] is None else _shape_tuple(payload[4])
    w2_a = None if payload[5] is None else _shape_tuple(payload[5])
    w2_b = None if payload[6] is None else _shape_tuple(payload[6])
    t2 = None if payload[7] is None else _shape_tuple(payload[7])
    try:
        if w1 is None:
            if w1_a is None or w1_b is None:
                raise RuntimeError("LoKr is missing both `w1` and the `w1_a/w1_b` factor pair.")
            w1 = _matmul_shape(w1_a, w1_b, label="LoKr.w1")
        if w2 is None:
            if w2_a is None or w2_b is None:
                raise RuntimeError("LoKr is missing both `w2` and the `w2_a/w2_b` factor pair.")
            if t2 is None:
                w2 = _matmul_shape(w2_a, w2_b, label="LoKr.w2")
            else:
                w2 = _tucker_pair_shape(t2, w2_b, w2_a, label="LoKr.t2")
    except RuntimeError as exc:
        return ShapeMismatchRecord(
            target_key=target_key,
            kind="lokr",
            target_shape=target_shape,
            detail=str(exc),
        )
    assert w1 is not None and w2 is not None
    if len(w2) == 4 and len(w1) == 2:
        w1 = (int(w1[0]), int(w1[1]), 1, 1)
    diff_shape = _kron_shape(w1, w2)
    if _shape_numel(diff_shape) == _shape_numel(target_shape):
        return None
    return ShapeMismatchRecord(
        target_key=target_key,
        kind="lokr",
        target_shape=target_shape,
        detail=f"expected_numel={_shape_numel(diff_shape)} kron_shape={diff_shape!r}",
    )


def _validate_glora_payload(
    *,
    payload: ShapePatchPayload,
    target_shape: Shape,
    target_key: str,
) -> ShapeMismatchRecord | None:
    a1_raw = _shape_tuple(payload[0])
    a2_raw = _shape_tuple(payload[1])
    b1_raw = _shape_tuple(payload[2])
    b2_raw = _shape_tuple(payload[3])
    if len(target_shape) < 2:
        return ShapeMismatchRecord(
            target_key=target_key,
            kind="glora",
            target_shape=target_shape,
            detail="GLORA requires target rank >= 2.",
        )
    a1 = _flatten_start_dim_one(a1_raw)
    a2 = _flatten_start_dim_one(a2_raw)
    b1 = _flatten_start_dim_one(b1_raw)
    b2 = _flatten_start_dim_one(b2_raw)
    old_glora = False
    if len(b2_raw) >= 2 and len(b1_raw) >= 2 and len(a1_raw) >= 2 and len(a2_raw) >= 2:
        if int(b2_raw[1]) == int(b1_raw[0]) == int(a1_raw[0]) == int(a2_raw[1]):
            old_glora = True
        if int(b2_raw[0]) == int(b1_raw[1]) == int(a1_raw[1]) == int(a2_raw[0]):
            if not (old_glora and int(a2_raw[0]) == int(target_shape[0]) == int(target_shape[1])):
                old_glora = False
    try:
        mm_b = _matmul_shape(b2, b1, label="GLoRA.b")
    except RuntimeError as exc:
        return ShapeMismatchRecord(
            target_key=target_key,
            kind="glora",
            target_shape=target_shape,
            detail=str(exc),
        )
    target_flat = (int(target_shape[0]), int(_shape_numel(target_shape[1:])))
    if old_glora:
        try:
            left = _matmul_shape(target_flat, a2, label="GLoRA.target_a2")
            right = _matmul_shape(left, a1, label="GLoRA.target_a1")
        except RuntimeError as exc:
            return ShapeMismatchRecord(
                target_key=target_key,
                kind="glora",
                target_shape=target_shape,
                detail=str(exc),
            )
        if right != mm_b:
            return ShapeMismatchRecord(
                target_key=target_key,
                kind="glora",
                target_shape=target_shape,
                detail=f"old_glora_add_shape_mismatch diff={right!r} bias={mm_b!r}",
            )
        if _shape_numel(right) == _shape_numel(target_shape):
            return None
        return ShapeMismatchRecord(
            target_key=target_key,
            kind="glora",
            target_shape=target_shape,
            detail=f"expected_numel={_shape_numel(right)} diff_shape={right!r}",
        )
    if len(target_shape) > 2:
        if int(target_shape[1]) != int(a1[0]):
            return ShapeMismatchRecord(
                target_key=target_key,
                kind="glora",
                target_shape=target_shape,
                detail=f"target_second_dim={target_shape[1]} a1_in={a1[0]}",
            )
        if int(a1[1]) != int(a2[0]):
            return ShapeMismatchRecord(
                target_key=target_key,
                kind="glora",
                target_shape=target_shape,
                detail=f"a1_out={a1[1]} a2_in={a2[0]}",
            )
        diff_shape = (int(target_shape[0]), int(a2[1]), *tuple(int(dim) for dim in target_shape[2:]))
        if int(a2[1]) != int(target_shape[1]):
            return ShapeMismatchRecord(
                target_key=target_key,
                kind="glora",
                target_shape=target_shape,
                detail=f"target_second_dim={target_shape[1]} a2_out={a2[1]}",
            )
    else:
        try:
            left = _matmul_shape(target_shape, a1, label="GLoRA.target_a1")
            diff_shape = _matmul_shape(left, a2, label="GLoRA.target_a2")
        except RuntimeError as exc:
            return ShapeMismatchRecord(
                target_key=target_key,
                kind="glora",
                target_shape=target_shape,
                detail=str(exc),
            )
        if _shape_numel(diff_shape) != _shape_numel(target_shape):
            return ShapeMismatchRecord(
                target_key=target_key,
                kind="glora",
                target_shape=target_shape,
                detail=f"expected_numel={_shape_numel(diff_shape)} diff_shape={diff_shape!r}",
            )
    if _shape_numel(mm_b) != _shape_numel(target_shape):
        return ShapeMismatchRecord(
            target_key=target_key,
            kind="glora",
            target_shape=target_shape,
            detail=f"bias_numel={_shape_numel(mm_b)} bias_shape={mm_b!r}",
        )
    return None


def validate_shape_patch_dict(
    shape_patch_dict: Mapping[PatchTarget, tuple[str, ShapePatchPayload]],
    *,
    target_shape_by_key: Mapping[str, Shape],
) -> ShapeCompatibilitySummary:
    mismatches: list[ShapeMismatchRecord] = []
    compatible_targets = 0

    for target, (kind, payload) in shape_patch_dict.items():
        target_key, target_shape = resolve_patch_target_shape(target_shape_by_key, target)
        is_slice = isinstance(target, tuple)
        mismatch: ShapeMismatchRecord | None
        if kind == "diff":
            mismatch = _validate_diff_payload(
                diff_shape=_shape_tuple(payload[0]),
                target_shape=target_shape,
                is_slice=is_slice,
                target_key=target_key,
            )
        elif kind == "set":
            mismatch = _validate_set_payload(
                set_shape=_shape_tuple(payload[0]),
                target_shape=target_shape,
                target_key=target_key,
            )
        elif kind == "lora":
            mismatch = _validate_lora_payload(payload=payload, target_shape=target_shape, target_key=target_key)
        elif kind == "loha":
            mismatch = _validate_loha_payload(payload=payload, target_shape=target_shape, target_key=target_key)
        elif kind == "lokr":
            mismatch = _validate_lokr_payload(payload=payload, target_shape=target_shape, target_key=target_key)
        elif kind == "glora":
            mismatch = _validate_glora_payload(payload=payload, target_shape=target_shape, target_key=target_key)
        else:
            mismatch = ShapeMismatchRecord(
                target_key=target_key,
                kind=str(kind),
                target_shape=target_shape,
                detail="unsupported patch kind for structural preflight",
            )
        if mismatch is None:
            compatible_targets += 1
            continue
        mismatches.append(mismatch)

    return ShapeCompatibilitySummary(
        total_targets=len(shape_patch_dict),
        compatible_targets=compatible_targets,
        mismatches=tuple(mismatches),
    )


def format_shape_compatibility_samples(
    summary: ShapeCompatibilitySummary,
    *,
    limit: int = 5,
) -> str:
    samples: list[str] = []
    for record in summary.mismatches[: max(1, int(limit))]:
        samples.append(
            "target={target} kind={kind} target_shape={shape} detail={detail}".format(
                target=record.target_key,
                kind=record.kind,
                shape=record.target_shape,
                detail=record.detail,
            )
        )
    return "; ".join(samples)


__all__ = [
    "HeaderPreflightResult",
    "ShapeCompatibilitySummary",
    "build_standard_shape_patch_dict_from_shape_map",
    "build_standard_shape_patch_dict_from_safetensors",
    "collect_parameter_shapes",
    "format_shape_compatibility_samples",
    "resolve_patch_target_shape",
    "shapeify_patch_dict",
    "validate_shape_patch_dict",
]
