"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA weight merge helpers (variant-dispatched).
Normalizes raw LoRA patch entries into typed segments and applies them onto weight tensors with dtype/device management.

Symbols (top-level; keep in sync; no ghosts):
- `weight_decompose` (function): Decomposes weights to an apply-friendly form for certain variants/quant paths.
- `merge_lora_to_weight` (function): Merges a LoRA patch into a target weight tensor (variant-dispatched).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Callable, Optional, Sequence

import torch

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.adapters.lora.types import (
    DiffWeights,
    GloraWeights,
    LohaWeights,
    LokrWeights,
    LoraWeights,
    SetWeights,
)

from .lora_registry import extra_weight_calculators
from .lora_types import LoraPatchEntry, LoraPatchSegment, LoraVariant, OffsetSpec

logger = get_backend_logger("backend.patchers.lora")


def _to_offset(offset: Optional[Sequence[int]]) -> Optional[OffsetSpec]:
    if offset is None:
        return None
    if len(offset) != 3:
        raise ValueError(f"Invalid offset specification: {offset}")
    return OffsetSpec(dim=int(offset[0]), start=int(offset[1]), length=int(offset[2]))


def _compose_segment(entry: LoraPatchEntry, key: str, computation_dtype: torch.dtype) -> LoraPatchSegment:
    strength_patch, payload, strength_model, offset, transform = entry
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"Composite LoRA payload for {key} is empty.")
        base_weight = payload[0]
        nested_entries = payload[1:]
        if not torch.is_tensor(base_weight):
            raise TypeError(f"Composite LoRA payload for {key} must begin with a tensor.")
        merged = merge_lora_to_weight(
            nested_entries,
            base_weight.clone(),
            key=key,
            computation_dtype=computation_dtype,
        )
        payload = ("diff", (merged,))

    segment_variant: Optional[LoraVariant] = None
    custom_kind: Optional[str] = None
    typed_payload: object = payload

    if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], str):
        tag, data = payload
        if tag in extra_weight_calculators:
            custom_kind = tag
            typed_payload = data
        else:
            segment_variant = LoraVariant.from_tag(tag)
            typed_payload = _normalize_payload(segment_variant, data, key)
    elif isinstance(payload, tuple) and len(payload) == 1 and torch.is_tensor(payload[0]):
        segment_variant = LoraVariant.DIFF
        typed_payload = DiffWeights(weight=payload[0])
    elif torch.is_tensor(payload):
        segment_variant = LoraVariant.DIFF
        typed_payload = DiffWeights(weight=payload)
    else:
        raise TypeError(f"Unsupported LoRA payload structure for {key}: {type(payload)}")

    return LoraPatchSegment(
        strength_patch=float(strength_patch),
        strength_model=float(strength_model),
        payload=typed_payload,
        offset=_to_offset(offset),
        transform=transform,
        variant=segment_variant,
        custom_kind=custom_kind,
    )


def _normalize_payload(variant: LoraVariant, data: object, key: str) -> object:
    if variant is LoraVariant.LORA:
        up, down, alpha, mid, dora_scale = data
        return LoraWeights(
            up=_expect_tensor(up, key, "lora.up"),
            down=_expect_tensor(down, key, "lora.down"),
            alpha=None if alpha is None else float(alpha),
            mid=None if mid is None else _expect_tensor(mid, key, "lora.mid"),
            dora_scale=None if dora_scale is None else _expect_tensor(dora_scale, key, "lora.dora_scale"),
        )
    if variant is LoraVariant.LOHA:
        (
            w1_a,
            w1_b,
            alpha,
            w2_a,
            w2_b,
            t1,
            t2,
            dora_scale,
        ) = data
        return LohaWeights(
            w1_a=_expect_tensor(w1_a, key, "loha.w1_a"),
            w1_b=_expect_tensor(w1_b, key, "loha.w1_b"),
            alpha=None if alpha is None else float(alpha),
            w2_a=_expect_tensor(w2_a, key, "loha.w2_a"),
            w2_b=_expect_tensor(w2_b, key, "loha.w2_b"),
            t1=None if t1 is None else _expect_tensor(t1, key, "loha.t1"),
            t2=None if t2 is None else _expect_tensor(t2, key, "loha.t2"),
            dora_scale=None if dora_scale is None else _expect_tensor(dora_scale, key, "loha.dora_scale"),
        )
    if variant is LoraVariant.LOKR:
        (
            w1,
            w2,
            alpha,
            w1_a,
            w1_b,
            w2_a,
            w2_b,
            t2,
            dora_scale,
        ) = data
        return LokrWeights(
            w1=None if w1 is None else _expect_tensor(w1, key, "lokr.w1"),
            w2=None if w2 is None else _expect_tensor(w2, key, "lokr.w2"),
            alpha=None if alpha is None else float(alpha),
            w1_a=None if w1_a is None else _expect_tensor(w1_a, key, "lokr.w1_a"),
            w1_b=None if w1_b is None else _expect_tensor(w1_b, key, "lokr.w1_b"),
            w2_a=None if w2_a is None else _expect_tensor(w2_a, key, "lokr.w2_a"),
            w2_b=None if w2_b is None else _expect_tensor(w2_b, key, "lokr.w2_b"),
            t2=None if t2 is None else _expect_tensor(t2, key, "lokr.t2"),
            dora_scale=None if dora_scale is None else _expect_tensor(dora_scale, key, "lokr.dora_scale"),
        )
    if variant is LoraVariant.GLORA:
        a1, a2, b1, b2, alpha, dora_scale = data
        return GloraWeights(
            a1=_expect_tensor(a1, key, "glora.a1"),
            a2=_expect_tensor(a2, key, "glora.a2"),
            b1=_expect_tensor(b1, key, "glora.b1"),
            b2=_expect_tensor(b2, key, "glora.b2"),
            alpha=None if alpha is None else float(alpha),
            dora_scale=None if dora_scale is None else _expect_tensor(dora_scale, key, "glora.dora_scale"),
        )
    if variant is LoraVariant.DIFF:
        (weight,) = data
        return DiffWeights(weight=_expect_tensor(weight, key, "diff"))
    if variant is LoraVariant.SET:
        (weight,) = data
        return SetWeights(weight=_expect_tensor(weight, key, "set"))
    raise ValueError(f"Variant {variant} is not supported for normalization")


def _expect_tensor(obj: object, key: str, label: str) -> torch.Tensor:
    if not torch.is_tensor(obj):
        raise TypeError(f"Expected tensor for {label} in {key}, received {type(obj)}")
    return obj


def _cast_tensor(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return memory_management.manager.cast_to_device(tensor, device, dtype)


def _reshape_like(tensor: torch.Tensor, reference: torch.Tensor, *, key: str) -> torch.Tensor:
    try:
        return tensor.reshape(reference.shape)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to reshape LoRA diff for {key}: {exc}") from exc


def _pad_mismatch(diff: torch.Tensor, current: torch.Tensor, *, key: str) -> torch.Tensor:
    if diff.ndim != current.ndim or diff.ndim != 4:
        raise RuntimeError(
            f"LoRA diff shape mismatch for {key}: expected {tuple(current.shape)} but received {tuple(diff.shape)}",
        )
    new_shape = tuple(max(a, b) for a, b in zip(current.shape, diff.shape))
    logger.info("LoRA diff for %s requires channel expansion %s -> %s", key, tuple(current.shape), new_shape)
    expanded = torch.zeros(new_shape, device=current.device, dtype=current.dtype)
    slices_current = tuple(slice(0, dim) for dim in current.shape)
    slices_diff = tuple(slice(0, dim) for dim in diff.shape)
    expanded[slices_current] = current
    expanded[slices_diff] += diff
    return expanded


@torch.no_grad()
def weight_decompose(
    dora_scale: torch.Tensor,
    weight: torch.Tensor,
    lora_diff: torch.Tensor,
    alpha: float,
    strength: float,
    computation_dtype: torch.dtype,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    dora_scale = _cast_tensor(dora_scale, weight.device, computation_dtype)
    lora_diff = lora_diff * alpha
    contribution = lora_diff if transform is None else transform(lora_diff)
    weight_calc = weight + contribution.to(weight.dtype)

    wd_on_output_axis = dora_scale.shape[0] == weight_calc.shape[0]
    if wd_on_output_axis:
        weight_norm = (
            weight.reshape(weight.shape[0], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight.shape[0], *[1] * (weight.dim() - 1))
        )
    else:
        weight_norm = (
            weight_calc.transpose(0, 1)
            .reshape(weight_calc.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1))
            .transpose(0, 1)
        )
    weight_norm = weight_norm + torch.finfo(weight.dtype).eps

    scaled = (dora_scale / weight_norm).to(weight.dtype)
    weight_calc *= scaled
    if strength != 1.0:
        delta = weight_calc - weight
        weight = weight + strength * delta
    else:
        weight.copy_(weight_calc)
    return weight


@torch.no_grad()
def merge_lora_to_weight(
    patches: Sequence[LoraPatchEntry],
    weight: torch.Tensor,
    *,
    key: str = "lora",
    computation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not patches:
        return weight

    weight_dtype_backup: Optional[torch.dtype] = None
    if computation_dtype == weight.dtype:
        merged = weight.clone()
    else:
        weight_dtype_backup = weight.dtype
        merged = weight.to(dtype=computation_dtype)

    for entry in patches:
        segment = _compose_segment(entry, key, computation_dtype)
        merged = _apply_segment(merged, segment, key, computation_dtype)

    if weight_dtype_backup is not None:
        merged = merged.to(dtype=weight_dtype_backup)
    return merged


def _apply_segment(weight: torch.Tensor, segment: LoraPatchSegment, key: str, computation_dtype: torch.dtype) -> torch.Tensor:
    target = weight if segment.offset is None else segment.offset.apply(weight)

    if segment.strength_model != 1.0:
        target.mul_(segment.strength_model)

    if segment.custom_kind is not None:
        calculator = extra_weight_calculators.get(segment.custom_kind)
        if calculator is None:
            raise KeyError(f"Custom LoRA patch type '{segment.custom_kind}' not registered.")
        updated = calculator(target, segment.strength_patch, segment.payload)
        if not torch.is_tensor(updated):
            raise TypeError(f"Custom LoRA calculator for {segment.custom_kind} must return a tensor.")
        if updated.shape != target.shape:
            raise RuntimeError(
                f"Custom LoRA calculator for {segment.custom_kind} changed tensor shape {target.shape} -> {updated.shape}.",
            )
        target.copy_(updated)
        return weight

    if segment.variant is None:
        raise RuntimeError(f"LoRA patch segment for {key} is missing variant information.")

    if segment.variant is LoraVariant.DIFF:
        return _apply_diff_patch(weight, target, segment, key)
    if segment.variant is LoraVariant.SET:
        return _apply_set_patch(weight, target, segment, key)
    if segment.variant is LoraVariant.LORA:
        return _apply_lora_patch(weight, target, segment, key, computation_dtype)
    if segment.variant is LoraVariant.LOHA:
        return _apply_loha_patch(weight, target, segment, key, computation_dtype)
    if segment.variant is LoraVariant.LOKR:
        return _apply_lokr_patch(weight, target, segment, key, computation_dtype)
    if segment.variant is LoraVariant.GLORA:
        return _apply_glora_patch(weight, target, segment, key, computation_dtype)
    raise NotImplementedError(f"LoRA variant {segment.variant} not yet supported.")


def _apply_diff_patch(weight: torch.Tensor, target: torch.Tensor, segment: LoraPatchSegment, key: str) -> torch.Tensor:
    payload = segment.payload
    assert isinstance(payload, DiffWeights)
    diff = _cast_tensor(payload.weight, target.device, target.dtype)
    scaled = diff * segment.strength_patch
    if segment.transform is not None:
        scaled = segment.transform(scaled)
    if scaled.shape != target.shape:
        if segment.offset is not None:
            raise RuntimeError(f"Cannot resize narrowed tensor for {key}: {scaled.shape} vs {target.shape}")
        replacement = _pad_mismatch(scaled, target, key=key)
        return replacement
    target.add_(scaled.to(dtype=target.dtype))
    return weight


def _apply_set_patch(weight: torch.Tensor, target: torch.Tensor, segment: LoraPatchSegment, key: str) -> torch.Tensor:
    payload = segment.payload
    assert isinstance(payload, SetWeights)
    tensor = _cast_tensor(payload.weight, target.device, target.dtype)
    if tensor.shape != target.shape:
        raise RuntimeError(f"Set patch for {key} expects shape {tuple(target.shape)}, received {tuple(tensor.shape)}")
    target.copy_(tensor)
    return weight


def _apply_lora_patch(
    weight: torch.Tensor,
    target: torch.Tensor,
    segment: LoraPatchSegment,
    key: str,
    computation_dtype: torch.dtype,
) -> torch.Tensor:
    payload = segment.payload
    assert isinstance(payload, LoraWeights)

    up = _cast_tensor(payload.up, target.device, computation_dtype)
    down = _cast_tensor(payload.down, target.device, computation_dtype)
    mid = None if payload.mid is None else _cast_tensor(payload.mid, target.device, computation_dtype)
    dora_scale = payload.dora_scale

    if payload.alpha is not None:
        alpha = payload.alpha / down.shape[0]
    else:
        alpha = 1.0

    if mid is not None:
        reshaped = torch.mm(
            down.transpose(0, 1).flatten(start_dim=1),
            mid.transpose(0, 1).flatten(start_dim=1),
        )
        final_shape = (down.shape[1], down.shape[0], mid.shape[2], mid.shape[3])
        down = reshaped.reshape(final_shape).transpose(0, 1)

    diff = torch.mm(up.flatten(start_dim=1), down.flatten(start_dim=1))
    if dora_scale is not None:
        if diff.numel() != target.numel():
            raise RuntimeError(f"DoRA LoRA diff mismatch for {key}: expected {target.numel()} elements, got {diff.numel()}.")
        diff = diff.reshape(target.shape)
        return weight_decompose(
            dora_scale=dora_scale,
            weight=target,
            lora_diff=diff,
            alpha=alpha,
            strength=segment.strength_patch,
            computation_dtype=computation_dtype,
            transform=segment.transform,
        )

    try:
        diff = diff.reshape(target.shape)
        shape_matches = True
    except RuntimeError:
        shape_matches = False

    scaled = ((segment.strength_patch * alpha) * diff).to(dtype=target.dtype)
    if segment.transform is not None:
        scaled = segment.transform(scaled)
    if shape_matches:
        target.add_(scaled)
        return weight
    if segment.offset is not None:
        raise RuntimeError(f"Cannot reshape narrowed tensor for {key}")
    return _pad_mismatch(scaled, target, key=key)


def _apply_lokr_patch(
    weight: torch.Tensor,
    target: torch.Tensor,
    segment: LoraPatchSegment,
    key: str,
    computation_dtype: torch.dtype,
) -> torch.Tensor:
    payload = segment.payload
    assert isinstance(payload, LokrWeights)

    dim: Optional[int] = None

    if payload.w1 is None:
        if payload.w1_a is None or payload.w1_b is None:
            raise RuntimeError(f"LOKR patch missing w1 components for {key}")
        dim = payload.w1_b.shape[0]
        w1 = torch.mm(
            _cast_tensor(payload.w1_a, target.device, computation_dtype),
            _cast_tensor(payload.w1_b, target.device, computation_dtype),
        )
    else:
        dim = payload.w1.shape[0]
        w1 = _cast_tensor(payload.w1, target.device, computation_dtype)

    if payload.w2 is None:
        if payload.w2_a is None or payload.w2_b is None:
            raise RuntimeError(f"LOKR patch missing w2 components for {key}")
        dim = payload.w2_b.shape[0]
        if payload.t2 is None:
            w2 = torch.mm(
                _cast_tensor(payload.w2_a, target.device, computation_dtype),
                _cast_tensor(payload.w2_b, target.device, computation_dtype),
            )
        else:
            w2 = torch.einsum(
                "i j k l, j r, i p -> p r k l",
                _cast_tensor(payload.t2, target.device, computation_dtype),
                _cast_tensor(payload.w2_b, target.device, computation_dtype),
                _cast_tensor(payload.w2_a, target.device, computation_dtype),
            )
    else:
        dim = payload.w2.shape[0]
        w2 = _cast_tensor(payload.w2, target.device, computation_dtype)

    if w2.ndim == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)

    alpha = 1.0
    if payload.alpha is not None and dim is not None:
        alpha = payload.alpha / dim

    diff = torch.kron(w1, w2)
    try:
        diff = _reshape_like(diff, target, key=key)
    except RuntimeError as exc:
        raise RuntimeError(f"LOKR reshape failed for {key}: {exc}") from exc

    if payload.dora_scale is not None:
        return weight_decompose(
            dora_scale=payload.dora_scale,
            weight=target,
            lora_diff=diff,
            alpha=alpha,
            strength=segment.strength_patch,
            computation_dtype=computation_dtype,
            transform=segment.transform,
        )

    scaled = ((segment.strength_patch * alpha) * diff).to(dtype=target.dtype)
    if segment.transform is not None:
        scaled = segment.transform(scaled)
    target.add_(scaled)
    return weight


def _apply_loha_patch(
    weight: torch.Tensor,
    target: torch.Tensor,
    segment: LoraPatchSegment,
    key: str,
    computation_dtype: torch.dtype,
) -> torch.Tensor:
    payload = segment.payload
    assert isinstance(payload, LohaWeights)

    w1a = _cast_tensor(payload.w1_a, target.device, computation_dtype)
    w1b = _cast_tensor(payload.w1_b, target.device, computation_dtype)
    w2a = _cast_tensor(payload.w2_a, target.device, computation_dtype)
    w2b = _cast_tensor(payload.w2_b, target.device, computation_dtype)
    if payload.alpha is not None:
        alpha = payload.alpha / w1b.shape[0]
    else:
        alpha = 1.0

    if payload.t1 is not None and payload.t2 is not None:
        t1 = _cast_tensor(payload.t1, target.device, computation_dtype)
        t2 = _cast_tensor(payload.t2, target.device, computation_dtype)
        m1 = torch.einsum("i j k l, j r, i p -> p r k l", t1, w1b, w1a)
        m2 = torch.einsum("i j k l, j r, i p -> p r k l", t2, w2b, w2a)
    else:
        m1 = torch.mm(w1a, w1b)
        m2 = torch.mm(w2a, w2b)

    diff = (m1 * m2)
    try:
        diff = _reshape_like(diff, target, key=key)
    except RuntimeError as exc:
        raise RuntimeError(f"LOHA reshape failed for {key}: {exc}") from exc

    if payload.dora_scale is not None:
        return weight_decompose(
            dora_scale=payload.dora_scale,
            weight=target,
            lora_diff=diff,
            alpha=alpha,
            strength=segment.strength_patch,
            computation_dtype=computation_dtype,
            transform=segment.transform,
        )

    scaled = ((segment.strength_patch * alpha) * diff).to(dtype=target.dtype)
    if segment.transform is not None:
        scaled = segment.transform(scaled)
    target.add_(scaled)
    return weight


def _apply_glora_patch(
    weight: torch.Tensor,
    target: torch.Tensor,
    segment: LoraPatchSegment,
    key: str,
    computation_dtype: torch.dtype,
) -> torch.Tensor:
    payload = segment.payload
    assert isinstance(payload, GloraWeights)

    a1_raw = payload.a1
    a2_raw = payload.a2
    b1_raw = payload.b1
    b2_raw = payload.b2

    old_glora = False
    if (
        b2_raw.shape[1] == b1_raw.shape[0] == a1_raw.shape[0] == a2_raw.shape[1]
    ):
        old_glora = True

    if (
        b2_raw.shape[0] == b1_raw.shape[1] == a1_raw.shape[1] == a2_raw.shape[0]
    ):
        if old_glora and a2_raw.shape[0] == target.shape[0] and target.shape[0] == target.shape[1]:
            pass
        else:
            old_glora = False

    a1 = _cast_tensor(a1_raw.flatten(start_dim=1), target.device, computation_dtype)
    a2 = _cast_tensor(a2_raw.flatten(start_dim=1), target.device, computation_dtype)
    b1 = _cast_tensor(b1_raw.flatten(start_dim=1), target.device, computation_dtype)
    b2 = _cast_tensor(b2_raw.flatten(start_dim=1), target.device, computation_dtype)

    if payload.alpha is None:
        alpha = 1.0
    else:
        alpha = payload.alpha / (a1_raw.shape[0] if old_glora else a2_raw.shape[0])

    if old_glora:
        diff = (
            torch.mm(b2, b1)
            + torch.mm(
                torch.mm(target.flatten(start_dim=1).to(dtype=computation_dtype), a2),
                a1,
            )
        ).reshape(target.shape)
    else:
        if target.dim() > 2:
            diff = torch.einsum(
                "o i ..., i j -> o j ...",
                torch.einsum("o i ..., i j -> o j ...", target.to(dtype=computation_dtype), a1),
                a2,
            ).reshape(target.shape)
        else:
            diff = torch.mm(torch.mm(target.to(dtype=computation_dtype), a1), a2).reshape(target.shape)
        diff = diff + torch.mm(b1, b2).reshape(target.shape)

    if payload.dora_scale is not None:
        return weight_decompose(
            dora_scale=payload.dora_scale,
            weight=target,
            lora_diff=diff,
            alpha=alpha,
            strength=segment.strength_patch,
            computation_dtype=computation_dtype,
            transform=segment.transform,
        )

    scaled = ((segment.strength_patch * alpha) * diff).to(dtype=target.dtype)
    if segment.transform is not None:
        scaled = segment.transform(scaled)
    target.add_(scaled)
    return weight


__all__ = [
    "merge_lora_to_weight",
    "weight_decompose",
]
