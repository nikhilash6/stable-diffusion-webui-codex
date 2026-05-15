"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Apply per-stage LoRA patches to WAN22 GGUF stage models (merge or online).
Controlled by `CODEX_LORA_APPLY_MODE` and maps LoRA keys to Codex WAN transformer keys via
`resolve_wan22_lora_logical_key` from `keymap_wan22_transformer.py` (canonical keymap authority),
with optional strict logical-key coverage gating via `CODEX_WAN22_STAGE_LORA_MIN_MATCH_RATIO`,
structured no-remap diagnostics for logical misses plus unsupported tensor suffix families, a cheap SafeTensors
header-only preflight before tensor materialization, and fail-loud structural validation against the mounted WAN22
stage model before patch construction and loader refresh.

Symbols (top-level; keep in sync; no ghosts):
- `_resolve_stage_lora_offload_device` (function): Resolves stage-LoRA offload device from memory-manager policy.
- `_collect_target_shape_by_key` (function): Collects live runtime parameter shapes for the mounted WAN22 stage model.
- `_validate_standard_lora_shapes` (function): Verifies mapped standard LoRA pair tensor shapes against mounted WAN22 target parameter shapes and raises explicit structural incompatibility errors.
- `apply_wan22_stage_lora` (function): Applies an ordered LoRA sequence to a loaded stage model (merge or online).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Set

import safetensors.torch as sf
import torch

from apps.backend.infra.config.bootstrap_env import get_bootstrap_env
from apps.backend.infra.config.lora_apply_mode import LoraApplyMode, read_lora_apply_mode
from apps.backend.patchers.lora_loader import CodexLoraLoader
from apps.backend.runtime.adapters.lora.pipeline import build_patch_dicts
from apps.backend.runtime.adapters.lora.preflight import (
    collect_parameter_shapes,
    format_shape_compatibility_samples,
    shapeify_patch_dict,
    validate_shape_patch_dict,
)
from apps.backend.runtime.adapters.lora.loader import STANDARD_LORA_TENSOR_CANDIDATES
from apps.backend.runtime.checkpoint.safetensors_header import read_safetensors_tensor_shapes
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.state_dict.keymap_wan22_transformer import resolve_wan22_lora_logical_key

from .diagnostics import get_logger
from .paths import normalize_win_path

_WAN22_LORA_PREFIXES = (
    "transformer_2.",
    "transformer.",
    "model.model.diffusion_model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
)
_WAN22_LORA_WRAPPER_PREFIXES = ("lora_unet_", "lycoris_")

_LORA_LOGICAL_SUFFIXES: tuple[str, ...] = (
    # Standard LoRA (multiple conventions)
    ".lora_up.weight",
    ".lora_down.weight",
    ".lora_mid.weight",
    "_lora.up.weight",
    "_lora.down.weight",
    ".lora_B.weight",
    ".lora_A.weight",
    ".lora.up.weight",
    ".lora.down.weight",
    ".lora_linear_layer.up.weight",
    ".lora_linear_layer.down.weight",
    # Optional metadata
    ".alpha",
    ".dora_scale",
    # DIFF / SET
    ".diff",
    ".diff_b",
    ".set_weight",
    # LoHa
    ".hada_w1_a",
    ".hada_w1_b",
    ".hada_w2_a",
    ".hada_w2_b",
    ".hada_t1",
    ".hada_t2",
    # LoKr
    ".lokr_w1",
    ".lokr_w2",
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_t2",
    # GLoRA
    ".a1.weight",
    ".a2.weight",
    ".b1.weight",
    ".b2.weight",
)

_MODULATION_TENSOR_SUFFIX = ".diff_m"
_RECOGNIZED_TENSOR_SUFFIXES = _LORA_LOGICAL_SUFFIXES + (_MODULATION_TENSOR_SUFFIX,)
_DIAGNOSTIC_EXAMPLE_LIMIT = 5
_RX_BLOCK_MODULATION_ROOT = re.compile(r"^blocks\.(?P<idx>\d+)$")

_ENV_WAN22_STAGE_LORA_MIN_MATCH_RATIO = "CODEX_WAN22_STAGE_LORA_MIN_MATCH_RATIO"
_WAN21_480P_ATTN_SHAPE = (1536, 1536)
_WAN21_480P_FFN0_SHAPE = (8960, 1536)
_WAN21_480P_FFN2_SHAPE = (1536, 8960)


@dataclass
class _StageLoraInspection:
    logical_key_count: int = 0
    matched_count: int = 0
    logical_to_load: dict[str, str] = field(default_factory=dict)
    extra_to_load: dict[str, str] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    class_examples: dict[str, list[str]] = field(default_factory=dict)
    unsupported_tensor_suffix_counts: dict[str, int] = field(default_factory=dict)
    unsupported_tensor_suffix_examples: dict[str, list[str]] = field(default_factory=dict)
    extra_matched_examples: list[str] = field(default_factory=list)

    @property
    def extra_matched_count(self) -> int:
        return len(self.extra_to_load)

    @property
    def to_load(self) -> dict[str, str]:
        combined = dict(self.logical_to_load)
        combined.update(self.extra_to_load)
        return combined


@dataclass(frozen=True, slots=True)
class _StandardLoraShapeRecord:
    logical_key: str
    target_key: str
    up_tensor_shape: tuple[int, ...]
    down_tensor_shape: tuple[int, ...]
    expected_target_shape: tuple[int, ...]
    actual_target_shape: tuple[int, ...]


def _read_min_match_ratio() -> float:
    raw = get_bootstrap_env(_ENV_WAN22_STAGE_LORA_MIN_MATCH_RATIO)
    if raw is None:
        raw = os.getenv(_ENV_WAN22_STAGE_LORA_MIN_MATCH_RATIO)
    if raw is None:
        return 0.0
    text = str(raw).strip()
    if not text:
        return 0.0
    try:
        ratio = float(text)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"{_ENV_WAN22_STAGE_LORA_MIN_MATCH_RATIO} must be a float in [0, 1], got: {raw!r}"
        ) from exc
    if not math.isfinite(ratio):
        raise RuntimeError(
            f"{_ENV_WAN22_STAGE_LORA_MIN_MATCH_RATIO} must be finite, got: {raw!r}"
        )
    if ratio < 0.0 or ratio > 1.0:
        raise RuntimeError(
            f"{_ENV_WAN22_STAGE_LORA_MIN_MATCH_RATIO} must be in [0, 1], got: {raw!r}"
        )
    return ratio


def _resolve_stage_lora_offload_device() -> torch.device:
    manager = getattr(memory_management, "manager", None)
    if manager is None or not hasattr(manager, "offload_device"):
        raise RuntimeError("WAN22 GGUF stage LoRA requires an active memory manager with offload_device().")
    offload_device = manager.offload_device()
    if not isinstance(offload_device, torch.device):
        raise RuntimeError(
            "WAN22 GGUF stage LoRA requires memory manager offload_device() to return torch.device "
            f"(got {type(offload_device).__name__})."
        )
    return offload_device


def _strip_known_prefixes(name: str) -> str:
    k = str(name)
    changed = True
    while changed:
        changed = False
        for prefix in _WAN22_LORA_PREFIXES:
            if k.startswith(prefix):
                k = k[len(prefix) :]
                changed = True
                break
    return k


def _display_logical_key(logical_key: str) -> str:
    key = str(logical_key)
    for prefix in _WAN22_LORA_WRAPPER_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    return _strip_known_prefixes(key)


def _record_example(store: dict[str, list[str]], bucket: str, value: str) -> None:
    examples = store.setdefault(bucket, [])
    if value not in examples and len(examples) < _DIAGNOSTIC_EXAMPLE_LIMIT:
        examples.append(value)


def _record_classification(inspection: _StageLoraInspection, bucket: str, value: str) -> None:
    inspection.class_counts[bucket] = inspection.class_counts.get(bucket, 0) + 1
    _record_example(inspection.class_examples, bucket, value)


def _record_unsupported_tensor_suffix(inspection: _StageLoraInspection, suffix: str, value: str) -> None:
    inspection.unsupported_tensor_suffix_counts[suffix] = inspection.unsupported_tensor_suffix_counts.get(suffix, 0) + 1
    _record_example(inspection.unsupported_tensor_suffix_examples, suffix, value)


def _extract_logical_keys(tensors: Mapping[str, torch.Tensor]) -> Set[str]:
    logical: set[str] = set()
    for key in tensors.keys():
        s = str(key)
        for suffix in _LORA_LOGICAL_SUFFIXES:
            if s.endswith(suffix):
                logical.add(s[: -len(suffix)])
                break
    return logical


def _extract_modulation_roots(tensors: Mapping[str, torch.Tensor]) -> Set[str]:
    roots: set[str] = set()
    for key in tensors.keys():
        s = str(key)
        if s.endswith(_MODULATION_TENSOR_SUFFIX):
            roots.add(s[: -len(_MODULATION_TENSOR_SUFFIX)])
    return roots


def _extract_unrecognized_tensor_keys(tensors: Mapping[str, torch.Tensor]) -> Set[str]:
    unknown: set[str] = set()
    for key in tensors.keys():
        s = str(key)
        if any(s.endswith(suffix) for suffix in _RECOGNIZED_TENSOR_SUFFIXES):
            continue
        unknown.add(s)
    return unknown


def _is_unsupported_i2v_branch(logical_key: str) -> bool:
    stripped = _display_logical_key(logical_key)
    return (
        stripped.endswith(".cross_attn.k_img")
        or stripped.endswith(".cross_attn.v_img")
        or stripped.endswith(".cross_attn.norm_k_img")
        or stripped.startswith("img_emb.proj.")
    )


def _resolve_candidate_targets(logical_key: str) -> tuple[str, list[tuple[str, str]]]:
    stripped = _display_logical_key(logical_key)
    resolved: list[tuple[str, str]] = []
    target = resolve_wan22_lora_logical_key(logical_key)
    if target is not None:
        resolved.append((logical_key, target))
    return stripped, resolved


def _modulation_logical_key(root: str) -> tuple[str | None, str]:
    stripped = _display_logical_key(root)
    tensor_label = f"{stripped}{_MODULATION_TENSOR_SUFFIX}"
    if stripped == "head":
        return f"{root}.modulation", tensor_label
    match = _RX_BLOCK_MODULATION_ROOT.match(stripped)
    if match:
        return f"{root}.modulation", tensor_label
    return None, tensor_label


def _register_target(
    target_owner: dict[str, str],
    target: str,
    logical_key: str,
    inspection: _StageLoraInspection,
) -> None:
    previous_owner = target_owner.get(target)
    if previous_owner is not None and previous_owner != logical_key:
        _record_classification(
            inspection,
            "alias_collision",
            f"{_display_logical_key(previous_owner)} -> {target} <- {_display_logical_key(logical_key)}",
        )
        raise RuntimeError(
            "WAN22 GGUF stage LoRA maps multiple logical keys to the same target weight. "
            f"target={target!r} keys={previous_owner!r},{logical_key!r}"
        )
    target_owner[target] = logical_key


def _inspect_stage_lora_mapping(model_keys: Set[str], tensors: Mapping[str, torch.Tensor]) -> _StageLoraInspection:
    """Inspect WAN22 stage-LoRA logical coverage and build the parser target map.

    Logical-key coverage (`matched/total/ratio`) counts only standard logical roots extracted
    from `_LORA_LOGICAL_SUFFIXES`. Non-logical tensor suffix families (for example `.diff_m`)
    are inspected separately so they never disappear from diagnostics.
    """

    inspection = _StageLoraInspection()
    logical_keys = sorted(_extract_logical_keys(tensors))
    inspection.logical_key_count = len(logical_keys)
    target_owner: dict[str, str] = {}

    for logical_key in logical_keys:
        display_key, resolved_targets = _resolve_candidate_targets(logical_key)
        if _is_unsupported_i2v_branch(display_key):
            _record_classification(inspection, "unsupported_i2v_branch", display_key)
            continue
        if not resolved_targets:
            _record_classification(inspection, "resolver_none", display_key)
            continue

        target = next((mapped for _candidate, mapped in resolved_targets if mapped in model_keys), None)
        if target is None:
            _record_classification(
                inspection,
                "resolved_target_missing",
                f"{display_key} -> {resolved_targets[0][1]}",
            )
            continue

        _register_target(target_owner, target, logical_key, inspection)
        inspection.logical_to_load[logical_key] = target
        inspection.matched_count += 1
        _record_classification(inspection, "matched", display_key)

    for modulation_root in sorted(_extract_modulation_roots(tensors)):
        logical_key, tensor_label = _modulation_logical_key(modulation_root)
        if logical_key is None:
            _record_unsupported_tensor_suffix(inspection, _MODULATION_TENSOR_SUFFIX, tensor_label)
            continue
        target = resolve_wan22_lora_logical_key(logical_key)
        if target is None:
            _record_unsupported_tensor_suffix(inspection, _MODULATION_TENSOR_SUFFIX, tensor_label)
            continue
        if target not in model_keys:
            _record_classification(inspection, "resolved_target_missing", f"{tensor_label} -> {target}")
            continue

        _register_target(target_owner, target, logical_key, inspection)
        inspection.extra_to_load[logical_key] = target
        if tensor_label not in inspection.extra_matched_examples and len(inspection.extra_matched_examples) < _DIAGNOSTIC_EXAMPLE_LIMIT:
            inspection.extra_matched_examples.append(tensor_label)

    for tensor_key in sorted(_extract_unrecognized_tensor_keys(tensors)):
        _record_unsupported_tensor_suffix(inspection, "<unknown>", _display_logical_key(tensor_key))

    return inspection


def _format_unsupported_tensor_suffix_counts(inspection: _StageLoraInspection) -> str:
    if not inspection.unsupported_tensor_suffix_counts:
        return "none"
    return ",".join(
        f"{suffix}={inspection.unsupported_tensor_suffix_counts[suffix]}"
        for suffix in sorted(inspection.unsupported_tensor_suffix_counts)
    )


def _format_stage_lora_diagnostics(inspection: _StageLoraInspection) -> str:
    class_order = (
        "matched",
        "resolver_none",
        "resolved_target_missing",
        "unsupported_i2v_branch",
        "alias_collision",
    )
    parts = [
        "class_counts[" + ", ".join(f"{name}={inspection.class_counts.get(name, 0)}" for name in class_order) + "]"
    ]
    if inspection.extra_matched_count > 0:
        parts.append(f"matched_modulation_tensor={inspection.extra_matched_count}")
    if inspection.unsupported_tensor_suffix_counts:
        parts.append(
            "unsupported_tensor_suffix["
            + ", ".join(
                f"{suffix}={inspection.unsupported_tensor_suffix_counts[suffix]}"
                for suffix in sorted(inspection.unsupported_tensor_suffix_counts)
            )
            + "]"
        )
    for name in class_order:
        examples = inspection.class_examples.get(name)
        if examples:
            parts.append(f"examples[{name}]={examples}")
    if inspection.extra_matched_examples:
        parts.append(f"examples[matched_modulation_tensor]={inspection.extra_matched_examples}")
    for suffix in sorted(inspection.unsupported_tensor_suffix_examples):
        examples = inspection.unsupported_tensor_suffix_examples[suffix]
        if examples:
            parts.append(f"examples[unsupported_tensor_suffix:{suffix}]={examples}")
    return "; ".join(parts)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", value)
    return tuple(int(dim) for dim in shape)


def _shape_numel(shape: Sequence[int]) -> int:
    if not shape:
        return 1
    return math.prod(int(dim) for dim in shape)


def _collect_target_shape_by_key(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return collect_parameter_shapes(model)


def _expected_target_shape_from_standard_lora(
    *, logical_key: str, up_tensor_shape: tuple[int, ...], down_tensor_shape: tuple[int, ...]
) -> tuple[int, ...]:
    if len(up_tensor_shape) != 2 or len(down_tensor_shape) != 2:
        raise RuntimeError(
            "WAN22 GGUF stage LoRA standard pair tensors must be rank-2. "
            f"logical_key={logical_key!r} up_shape={up_tensor_shape!r} down_shape={down_tensor_shape!r}"
        )
    rank = up_tensor_shape[1]
    if down_tensor_shape[0] != rank:
        raise RuntimeError(
            "WAN22 GGUF stage LoRA standard pair rank mismatch. "
            f"logical_key={logical_key!r} up_shape={up_tensor_shape!r} down_shape={down_tensor_shape!r}"
        )
    return (int(up_tensor_shape[0]), int(down_tensor_shape[1]))


def _iter_standard_lora_shape_records(
    *,
    tensors: Mapping[str, torch.Tensor],
    logical_to_target: Mapping[str, str],
    target_shape_by_key: Mapping[str, tuple[int, ...]],
) -> list[_StandardLoraShapeRecord]:
    records: list[_StandardLoraShapeRecord] = []
    for logical_key in sorted(logical_to_target):
        up_tensor_shape: tuple[int, ...] | None = None
        down_tensor_shape: tuple[int, ...] | None = None
        for up_suffix, down_suffix, _mid_suffix in STANDARD_LORA_TENSOR_CANDIDATES:
            up_key = f"{logical_key}{up_suffix}"
            down_key = f"{logical_key}{down_suffix}"
            if up_key in tensors and down_key in tensors:
                up_tensor_shape = _shape_tuple(tensors[up_key])
                down_tensor_shape = _shape_tuple(tensors[down_key])
                break
        if up_tensor_shape is None or down_tensor_shape is None:
            continue
        target_key = logical_to_target[logical_key]
        target_shape = target_shape_by_key.get(target_key)
        if target_shape is None:
            continue
        expected_target_shape = _expected_target_shape_from_standard_lora(
            logical_key=logical_key,
            up_tensor_shape=up_tensor_shape,
            down_tensor_shape=down_tensor_shape,
        )
        records.append(
            _StandardLoraShapeRecord(
                logical_key=logical_key,
                target_key=target_key,
                up_tensor_shape=up_tensor_shape,
                down_tensor_shape=down_tensor_shape,
                expected_target_shape=expected_target_shape,
                actual_target_shape=target_shape,
            )
        )
    return records


def _matches_wan21_480p_shape(record: _StandardLoraShapeRecord) -> bool:
    if record.target_key.endswith((".self_attn.q.weight", ".self_attn.k.weight", ".self_attn.v.weight", ".self_attn.o.weight")):
        return record.expected_target_shape == _WAN21_480P_ATTN_SHAPE
    if record.target_key.endswith((".cross_attn.q.weight", ".cross_attn.k.weight", ".cross_attn.v.weight", ".cross_attn.o.weight")):
        return record.expected_target_shape == _WAN21_480P_ATTN_SHAPE
    if record.target_key.endswith(".ffn.0.weight"):
        return record.expected_target_shape == _WAN21_480P_FFN0_SHAPE
    if record.target_key.endswith(".ffn.2.weight"):
        return record.expected_target_shape == _WAN21_480P_FFN2_SHAPE
    return False


def _format_shape_record_samples(records: Sequence[_StandardLoraShapeRecord]) -> str:
    samples = []
    for record in records[:_DIAGNOSTIC_EXAMPLE_LIMIT]:
        samples.append(
            "{logical}->{target} up={a} down={b} expected={expected} actual={actual}".format(
                logical=_display_logical_key(record.logical_key),
                target=record.target_key,
                a=record.up_tensor_shape,
                b=record.down_tensor_shape,
                expected=record.expected_target_shape,
                actual=record.actual_target_shape,
            )
        )
    return "; ".join(samples)


def _validate_standard_lora_shapes(
    *,
    tensors: Mapping[str, torch.Tensor],
    logical_to_target: Mapping[str, str],
    target_shape_by_key: Mapping[str, tuple[int, ...]],
    stage: str,
    resolved_path: str,
) -> None:
    records = _iter_standard_lora_shape_records(
        tensors=tensors,
        logical_to_target=logical_to_target,
        target_shape_by_key=target_shape_by_key,
    )
    if not records:
        return

    compatible_records = [record for record in records if record.expected_target_shape == record.actual_target_shape]
    if len(compatible_records) == len(records):
        return

    mismatched_records = [record for record in records if record.expected_target_shape != record.actual_target_shape]
    saw_attention = any(
        record.target_key.endswith(
            (
                ".self_attn.q.weight",
                ".self_attn.k.weight",
                ".self_attn.v.weight",
                ".self_attn.o.weight",
                ".cross_attn.q.weight",
                ".cross_attn.k.weight",
                ".cross_attn.v.weight",
                ".cross_attn.o.weight",
            )
        )
        and record.expected_target_shape == _WAN21_480P_ATTN_SHAPE
        for record in mismatched_records
    )
    saw_ffn0 = any(
        record.target_key.endswith(".ffn.0.weight") and record.expected_target_shape == _WAN21_480P_FFN0_SHAPE
        for record in mismatched_records
    )
    saw_ffn2 = any(
        record.target_key.endswith(".ffn.2.weight") and record.expected_target_shape == _WAN21_480P_FFN2_SHAPE
        for record in mismatched_records
    )
    if (
        not compatible_records
        and mismatched_records
        and saw_attention
        and saw_ffn0
        and saw_ffn2
        and all(_matches_wan21_480p_shape(record) for record in mismatched_records)
        and all(_shape_numel(record.actual_target_shape) > _shape_numel(record.expected_target_shape) for record in mismatched_records)
    ):
        raise RuntimeError(
            "WAN22 GGUF stage '{stage}': structural LoRA mismatch for wan22_14b. "
            "This adapter matches the Wan2.1 480p profile (hidden=1536, ffn=8960), but the mounted stage exposes larger runtime targets. "
            "wan22_14b does not support Wan2.1 480p LoRAs; use a 720p-style adapter instead. "
            "samples={samples} file={path}".format(
                stage=stage,
                samples=_format_shape_record_samples(mismatched_records),
                path=resolved_path,
            )
        )

    raise RuntimeError(
        "WAN22 GGUF stage '{stage}': LoRA target-shape mismatch after key resolution. "
        "shape_compatible_standard_targets={compatible}/{total}. samples={samples} file={path}".format(
            stage=stage,
            compatible=len(compatible_records),
            total=len(records),
            samples=_format_shape_record_samples(mismatched_records),
            path=resolved_path,
        )
    )


def apply_wan22_stage_lora(
    model: torch.nn.Module,
    *,
    stage: str,
    loras: Optional[Sequence[tuple[str, float]]],
    logger: Any,
) -> None:
    """Apply an ordered LoRA sequence to a loaded stage model (WAN22 GGUF runtime)."""

    if not loras:
        return

    log = get_logger(logger)
    min_match_ratio = _read_min_match_ratio()
    target_shape_by_key = _collect_target_shape_by_key(model)
    model_keys = set(target_shape_by_key.keys())
    parsed_loras: list[tuple[str, float, dict[str, list[tuple]], int]] = []
    for index, raw_spec in enumerate(loras):
        if not isinstance(raw_spec, (tuple, list)) or len(raw_spec) != 2:
            raise RuntimeError(
                f"WAN22 GGUF stage '{stage}': loras[{index}] must be a [path, weight] pair."
            )
        raw_path = raw_spec[0]
        raw_weight = raw_spec[1]
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise RuntimeError(
                f"WAN22 GGUF stage '{stage}': loras[{index}][0] must be a non-empty path string."
            )
        resolved_path = normalize_win_path(os.path.expanduser(raw_path.strip()))
        if not resolved_path.lower().endswith(".safetensors"):
            raise RuntimeError(
                f"WAN22 GGUF stage '{stage}': loras[{index}] path must be a .safetensors file, got: {resolved_path}"
            )
        if not os.path.isfile(resolved_path):
            raise RuntimeError(f"WAN22 GGUF stage '{stage}': loras[{index}] path not found: {resolved_path}")
        if raw_weight is None:
            strength = 1.0
        else:
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise RuntimeError(
                    f"WAN22 GGUF stage '{stage}': loras[{index}] weight must be numeric, got: {raw_weight!r}"
                )
            strength = float(raw_weight)
            if not math.isfinite(strength):
                raise RuntimeError(
                    f"WAN22 GGUF stage '{stage}': loras[{index}] weight must be finite, got: {raw_weight!r}"
                )

        header_shapes = read_safetensors_tensor_shapes(Path(resolved_path))
        header_inspection = _inspect_stage_lora_mapping(model_keys, header_shapes)
        header_logical_key_count = header_inspection.logical_key_count
        header_matched_count = header_inspection.matched_count
        header_coverage = (header_matched_count / header_logical_key_count) if header_logical_key_count > 0 else 0.0
        header_diagnostics = _format_stage_lora_diagnostics(header_inspection)
        _validate_standard_lora_shapes(
            tensors=header_shapes,
            logical_to_target=header_inspection.logical_to_load,
            target_shape_by_key=target_shape_by_key,
            stage=stage,
            resolved_path=resolved_path,
        )
        if not header_inspection.to_load:
            raise RuntimeError(
                "WAN22 GGUF stage '{stage}': LoRA file matched 0 targets; "
                "this LoRA key layout is not supported by the WAN transformer mapping. "
                "diagnostics={diagnostics} file={path}".format(
                    stage=stage,
                    diagnostics=header_diagnostics,
                    path=resolved_path,
                )
            )
        if min_match_ratio > 0.0 and header_coverage < min_match_ratio:
            raise RuntimeError(
                "WAN22 GGUF stage '{stage}': LoRA logical-key coverage below threshold "
                f"(matched={matched}/{total} ratio={ratio:.4f} required>={required:.4f}). "
                "diagnostics={diagnostics} "
                "Adjust CODEX_WAN22_STAGE_LORA_MIN_MATCH_RATIO or use a compatible adapter mapping. "
                "file={path}".format(
                    stage=stage,
                    matched=header_matched_count,
                    total=header_logical_key_count,
                    ratio=header_coverage,
                    required=min_match_ratio,
                    diagnostics=header_diagnostics,
                    path=resolved_path,
                )
            )

        try:
            tensors = sf.load_file(resolved_path)
        except Exception as exc:
            raise RuntimeError(
                f"WAN22 GGUF stage '{stage}': failed to load LoRA file at loras[{index}] ({resolved_path}): {exc}"
            ) from exc

        inspection = _inspect_stage_lora_mapping(model_keys, tensors)
        logical_key_count = inspection.logical_key_count
        matched_count = inspection.matched_count
        to_load = inspection.to_load
        coverage = (matched_count / logical_key_count) if logical_key_count > 0 else 0.0
        diagnostics = _format_stage_lora_diagnostics(inspection)
        unsupported_suffix_summary = _format_unsupported_tensor_suffix_counts(inspection)

        _validate_standard_lora_shapes(
            tensors=tensors,
            logical_to_target=inspection.logical_to_load,
            target_shape_by_key=target_shape_by_key,
            stage=stage,
            resolved_path=resolved_path,
        )

        if not to_load:
            raise RuntimeError(
                "WAN22 GGUF stage '{stage}': LoRA file matched 0 targets; "
                "this LoRA key layout is not supported by the WAN transformer mapping. "
                "diagnostics={diagnostics} file={path}".format(
                    stage=stage,
                    diagnostics=diagnostics,
                    path=resolved_path,
                )
            )
        if min_match_ratio > 0.0 and coverage < min_match_ratio:
            raise RuntimeError(
                "WAN22 GGUF stage '{stage}': LoRA logical-key coverage below threshold "
                f"(matched={matched_count}/{logical_key_count} ratio={coverage:.4f} required>={min_match_ratio:.4f}). "
                "diagnostics={diagnostics} "
                "Adjust CODEX_WAN22_STAGE_LORA_MIN_MATCH_RATIO or use a compatible adapter mapping. "
                "file={path}".format(
                    stage=stage,
                    diagnostics=diagnostics,
                    path=resolved_path,
                )
            )
        if logical_key_count > 0 and coverage < 1.0:
            log.warning(
                "[wan22.gguf] stage LoRA partial logical-key coverage: stage=%s index=%d matched=%d total=%d ratio=%.4f required_ratio=%.4f matched_modulation_tensor=%d unsupported_tensor_suffixes=%s diagnostics=%s",
                stage,
                index,
                matched_count,
                logical_key_count,
                coverage,
                min_match_ratio,
                inspection.extra_matched_count,
                unsupported_suffix_summary,
                diagnostics,
            )
        elif inspection.unsupported_tensor_suffix_counts:
            log.warning(
                "[wan22.gguf] stage LoRA has unsupported tensor suffix families outside logical-key coverage: stage=%s index=%d matched=%d total=%d ratio=%.4f unsupported_tensor_suffixes=%s diagnostics=%s",
                stage,
                index,
                matched_count,
                logical_key_count,
                coverage,
                unsupported_suffix_summary,
                diagnostics,
            )
        elif inspection.extra_matched_count > 0:
            log.info(
                "[wan22.gguf] stage LoRA parsed extra modulation tensors outside logical-key coverage: stage=%s index=%d matched=%d total=%d ratio=%.4f matched_modulation_tensor=%d diagnostics=%s",
                stage,
                index,
                matched_count,
                logical_key_count,
                coverage,
                inspection.extra_matched_count,
                diagnostics,
            )

        patch_dict = build_patch_dicts(tensors, to_load)
        if not patch_dict:
            raise RuntimeError(
                "WAN22 GGUF stage '{stage}': LoRA produced 0 patches after parsing; "
                "this usually indicates incomplete tensors for the mapped keys. "
                "diagnostics={diagnostics} file={path}".format(
                    stage=stage,
                    diagnostics=diagnostics,
                    path=resolved_path,
                )
            )
        patch_summary = validate_shape_patch_dict(
            shapeify_patch_dict(patch_dict),
            target_shape_by_key=target_shape_by_key,
        )
        if patch_summary.mismatches:
            raise RuntimeError(
                "WAN22 GGUF stage '{stage}': structural LoRA mismatch after patch parsing. "
                "shape_compatible_targets={compatible}/{total}. samples={samples} file={path}".format(
                    stage=stage,
                    compatible=patch_summary.compatible_targets,
                    total=patch_summary.total_targets,
                    samples=format_shape_compatibility_samples(patch_summary),
                    path=resolved_path,
                )
            )
        lora_patch_map = {
            key: [(strength, payload, 1.0, None, None)] for key, payload in patch_dict.items()
        }
        parsed_loras.append((resolved_path, strength, lora_patch_map, len(patch_dict)))

    if not parsed_loras:
        return
    apply_mode = read_lora_apply_mode()
    online_mode = apply_mode is LoraApplyMode.ONLINE

    loader = getattr(model, "lora_loader", None)
    if not isinstance(loader, CodexLoraLoader):
        loader = CodexLoraLoader(model)
        model.lora_loader = loader

    lora_patches: dict[tuple[str, float, float, bool], dict[str, list[tuple]]] = {}
    for index, (resolved_path, strength, lora_patch_map, _patch_count) in enumerate(parsed_loras):
        patch_source = f"{resolved_path}#stage_index={index}"
        lora_patches[(patch_source, strength, 1.0, online_mode)] = lora_patch_map

    offload_device = _resolve_stage_lora_offload_device()
    loader.refresh(lora_patches, offload_device=offload_device, force_refresh=False)

    total_loras = len(parsed_loras)
    for index, (resolved_path, strength, _lora_patch_map, patch_count) in enumerate(parsed_loras):
        log.info(
            "[wan22.gguf] stage LoRA applied: stage=%s index=%d/%d mode=%s file=%s params=%d weight=%s offload_device=%s",
            stage,
            index + 1,
            total_loras,
            apply_mode.value,
            os.path.basename(resolved_path),
            patch_count,
            strength,
            offload_device,
        )


__all__ = ["apply_wan22_stage_lora"]
