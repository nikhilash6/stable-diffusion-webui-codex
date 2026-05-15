"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA tensor parsing into runtime patch specs (LoRA/LoHa/LoKr/GLORA/DIFF/SET).
Parses adapter tensors into typed `PatchSpec` entries for the runtime adapter pipeline, supporting multiple LoRA conventions,
optional metadata keys (alpha/dora_scale), WAN modulation DIFF tensors (`*.diff_m` when the logical target is modulation),
and logs missing keys for diagnostics. Patch targets may be plain parameter names or `(parameter, offset)` tuples for slice
patches (e.g. fused-QKV text encoders).

Symbols (top-level; keep in sync; no ghosts):
- `STANDARD_LORA_TENSOR_CANDIDATES` (constant): Supported standard LoRA up/down[/mid] tensor suffix families, in parser priority order.
- `_bias_target_for` (function): Derives a bias patch target from a weight patch target (preserving offset slices when present).
- `_tensor_item` (function): Converts scalar tensors (alpha/dora_scale) into Python floats.
- `_maybe_convert_bfl_control` (function): Normalizes certain BFL/Control-style tensor key patterns into Codex naming.
- `_select_first_present` (function): Returns the first matching key + tensor from a list of candidate names.
- `_extract_lora` (function): Extracts standard LoRA weights (up/down[/mid]) into a `PatchSpec`.
- `_extract_loha` (function): Extracts LoHa weights into a `PatchSpec`.
- `_extract_lokr` (function): Extracts LoKr weights into a `PatchSpec`.
- `_extract_glora` (function): Extracts GLORA weights into a `PatchSpec`.
- `_extract_diff` (function): Extracts DIFF/SET weights (and optional diff bias) into `PatchSpec` entries.
- `parse_lora_tensors` (function): Main entrypoint; parses tensors and returns `(PatchSpec list, loaded-key set)`.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import re
from typing import Dict, Iterable, List, Mapping, Tuple, TypeVar

import torch

from apps.backend.runtime.adapters.base import PatchKind, PatchSpec, PatchTarget, log_missing_keys
from apps.backend.runtime.adapters.lora.types import (
    DiffWeights,
    GloraWeights,
    LohaWeights,
    LokrWeights,
    LoraWeights,
    SetWeights,
    make_spec,
)
from apps.backend.runtime.state_dict.views import KeyspaceLookupView

LOGGER = get_backend_logger(__name__)
_TensorValueT = TypeVar("_TensorValueT")
_RX_BLOCK_MODULATION_LOGICAL_KEY = re.compile(r"^blocks_(?P<idx>\d+)_modulation$")
STANDARD_LORA_TENSOR_CANDIDATES = (
    (".lora_up.weight", ".lora_down.weight", ".lora_mid.weight"),
    ("_lora.up.weight", "_lora.down.weight", None),
    (".lora_B.weight", ".lora_A.weight", None),
    (".lora.up.weight", ".lora.down.weight", None),
    (".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight", None),
)


def _bias_target_for(target: PatchTarget) -> PatchTarget:
    base, offset = (target if isinstance(target, tuple) else (target, None))
    if base.endswith(".weight"):
        bias_base = base[:-len(".weight")] + ".bias"
    else:
        bias_base = f"{base}.bias"
    if offset is None:
        return bias_base
    return (bias_base, offset)


def _tensor_item(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    if value.numel() != 1:
        raise RuntimeError("Expected scalar tensor for alpha/dora_scale")
    return float(value.item())


def _maybe_convert_bfl_control(tensors: Mapping[str, _TensorValueT]) -> Mapping[str, _TensorValueT]:
    if "img_in.lora_A.weight" not in tensors or "single_blocks.0.norm.key_norm.scale" not in tensors:
        return tensors
    converted_lookup: dict[str, str] = {}
    for key in tensors.keys():
        if not isinstance(key, str):
            raise RuntimeError(f"LoRA tensor map keys must be strings, got {type(key).__name__}.")
        new_key = key.replace(".lora_B.bias", ".diff_b").replace("_norm.scale", "_norm.scale.set_weight")
        mapped_key = f"diffusion_model.{new_key}"
        previous = converted_lookup.get(mapped_key)
        if previous is not None and previous != key:
            raise RuntimeError(
                "BFL/Control LoRA key conversion collided after explicit mapping: "
                f"dst={mapped_key!r} srcs={previous!r},{key!r}"
            )
        converted_lookup[mapped_key] = key
    return KeyspaceLookupView(tensors, converted_lookup)


def _select_first_present(tensors: Mapping[str, torch.Tensor], names: Iterable[str]) -> Tuple[str | None, torch.Tensor | None]:
    for name in names:
        if name in tensors:
            return name, tensors[name]
    return None, None


def _modulation_tensor_name_candidates(logical_key: str) -> Tuple[str, ...]:
    if logical_key in {"head.modulation", "head_modulation"}:
        return ("head.diff_m",)
    if logical_key.endswith(".modulation"):
        return (f"{logical_key[:-len('.modulation')]}.diff_m",)
    match = _RX_BLOCK_MODULATION_LOGICAL_KEY.match(logical_key)
    if match:
        return (f"blocks.{match.group('idx')}.diff_m",)
    return ()


def _extract_lora(
    logical_key: str,
    target_param: PatchTarget,
    tensors: Mapping[str, torch.Tensor],
    loaded: set[str],
) -> PatchSpec | None:
    A = B = mid = None
    a_name = b_name = m_name = None
    for up_suffix, down_suffix, mid_suffix in STANDARD_LORA_TENSOR_CANDIDATES:
        up_name = f"{logical_key}{up_suffix}"
        down_name = f"{logical_key}{down_suffix}"
        mid_name = f"{logical_key}{mid_suffix}" if mid_suffix is not None else None
        if up_name in tensors and down_name in tensors:
            a_name, b_name = up_name, down_name
            A, B = tensors[up_name], tensors[down_name]
            if mid_name and mid_name in tensors:
                m_name = mid_name
                mid = tensors[mid_name]
            break
    if A is None or B is None:
        return None

    alpha_tensor = tensors.get(f"{logical_key}.alpha")
    dora_scale = tensors.get(f"{logical_key}.dora_scale")

    if a_name:
        loaded.add(a_name)
    if b_name:
        loaded.add(b_name)
    if m_name:
        loaded.add(m_name)
    if alpha_tensor is not None:
        loaded.add(f"{logical_key}.alpha")
    if dora_scale is not None:
        loaded.add(f"{logical_key}.dora_scale")

    payload = LoraWeights(
        up=A,
        down=B,
        mid=mid,
        alpha=_tensor_item(alpha_tensor),
        dora_scale=dora_scale,
    )
    return make_spec(target_param, PatchKind.LORA, payload)


def _extract_loha(logical_key: str, target_param: PatchTarget, tensors: Mapping[str, torch.Tensor], loaded: set[str]) -> PatchSpec | None:
    base = f"{logical_key}."
    required = ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"]
    if not all(f"{base}{name}" in tensors for name in required):
        return None
    w1_a = tensors[f"{base}hada_w1_a"]
    w1_b = tensors[f"{base}hada_w1_b"]
    w2_a = tensors[f"{base}hada_w2_a"]
    w2_b = tensors[f"{base}hada_w2_b"]
    t1 = tensors.get(f"{base}hada_t1")
    t2 = tensors.get(f"{base}hada_t2")
    alpha_tensor = tensors.get(f"{logical_key}.alpha")
    dora_scale = tensors.get(f"{logical_key}.dora_scale")
    for suffix in required:
        loaded.add(f"{base}{suffix}")
    for optional in ("hada_t1", "hada_t2"):
        name = f"{base}{optional}"
        if name in tensors:
            loaded.add(name)
    if alpha_tensor is not None:
        loaded.add(f"{logical_key}.alpha")
    if dora_scale is not None:
        loaded.add(f"{logical_key}.dora_scale")
    payload = LohaWeights(
        w1_a=w1_a,
        w1_b=w1_b,
        alpha=_tensor_item(alpha_tensor),
        w2_a=w2_a,
        w2_b=w2_b,
        t1=t1,
        t2=t2,
        dora_scale=dora_scale,
    )
    return make_spec(target_param, PatchKind.LOHA, payload)


def _extract_lokr(logical_key: str, target_param: PatchTarget, tensors: Mapping[str, torch.Tensor], loaded: set[str]) -> PatchSpec | None:
    base = f"{logical_key}."
    keys = {
        "w1": tensors.get(f"{base}lokr_w1"),
        "w2": tensors.get(f"{base}lokr_w2"),
        "w1_a": tensors.get(f"{base}lokr_w1_a"),
        "w1_b": tensors.get(f"{base}lokr_w1_b"),
        "w2_a": tensors.get(f"{base}lokr_w2_a"),
        "w2_b": tensors.get(f"{base}lokr_w2_b"),
        "t2": tensors.get(f"{base}lokr_t2"),
    }
    if not any(value is not None for value in keys.values()):
        return None
    for suffix, value in keys.items():
        if value is not None:
            loaded.add(f"{base}lokr_{suffix}")
    alpha_tensor = tensors.get(f"{logical_key}.alpha")
    dora_scale = tensors.get(f"{logical_key}.dora_scale")
    if alpha_tensor is not None:
        loaded.add(f"{logical_key}.alpha")
    if dora_scale is not None:
        loaded.add(f"{logical_key}.dora_scale")
    payload = LokrWeights(
        w1=keys["w1"],
        w2=keys["w2"],
        alpha=_tensor_item(alpha_tensor),
        w1_a=keys["w1_a"],
        w1_b=keys["w1_b"],
        w2_a=keys["w2_a"],
        w2_b=keys["w2_b"],
        t2=keys["t2"],
        dora_scale=dora_scale,
    )
    return make_spec(target_param, PatchKind.LOKR, payload)


def _extract_glora(logical_key: str, target_param: PatchTarget, tensors: Mapping[str, torch.Tensor], loaded: set[str]) -> PatchSpec | None:
    base = f"{logical_key}."
    required = ["a1.weight", "a2.weight", "b1.weight", "b2.weight"]
    if not all(f"{base}{name}" in tensors for name in required):
        return None
    a1 = tensors[f"{base}a1.weight"]
    a2 = tensors[f"{base}a2.weight"]
    b1 = tensors[f"{base}b1.weight"]
    b2 = tensors[f"{base}b2.weight"]
    alpha_tensor = tensors.get(f"{logical_key}.alpha")
    dora_scale = tensors.get(f"{logical_key}.dora_scale")
    for suffix in required:
        loaded.add(f"{base}{suffix}")
    if alpha_tensor is not None:
        loaded.add(f"{logical_key}.alpha")
    if dora_scale is not None:
        loaded.add(f"{logical_key}.dora_scale")
    payload = GloraWeights(
        a1=a1,
        a2=a2,
        b1=b1,
        b2=b2,
        alpha=_tensor_item(alpha_tensor),
        dora_scale=dora_scale,
    )
    return make_spec(target_param, PatchKind.GLORA, payload)


def _extract_diff(logical_key: str, target_param: PatchTarget, tensors: Mapping[str, torch.Tensor], loaded: set[str]) -> List[PatchSpec]:
    specs: List[PatchSpec] = []
    diff = tensors.get(f"{logical_key}.diff")
    if diff is not None:
        loaded.add(f"{logical_key}.diff")
        specs.append(make_spec(target_param, PatchKind.DIFF, DiffWeights(weight=diff)))
    modulation_name, modulation_diff = _select_first_present(tensors, _modulation_tensor_name_candidates(logical_key))
    if modulation_diff is not None:
        if modulation_name is None:
            raise RuntimeError(f"Internal error: missing tensor name for modulation diff on {logical_key!r}")
        loaded.add(modulation_name)
        specs.append(make_spec(target_param, PatchKind.DIFF, DiffWeights(weight=modulation_diff)))
    bias_key = f"{logical_key}.diff_b"
    if bias_key in tensors:
        bias_target = _bias_target_for(target_param)
        specs.append(make_spec(bias_target, PatchKind.DIFF, DiffWeights(weight=tensors[bias_key])))
        loaded.add(bias_key)
    set_weight_key = f"{logical_key}.set_weight"
    if set_weight_key in tensors:
        specs.append(make_spec(target_param, PatchKind.SET, SetWeights(weight=tensors[set_weight_key])))
        loaded.add(set_weight_key)
    return specs


def parse_lora_tensors(tensors: Mapping[str, torch.Tensor], to_load: Dict[str, PatchTarget]) -> tuple[List[PatchSpec], set[str]]:
    tensor_map = _maybe_convert_bfl_control(tensors)
    loaded: set[str] = set()
    specs: List[PatchSpec] = []

    for logical_key, target_param in to_load.items():
        extractor_sequence = (
            _extract_lora,
            _extract_loha,
            _extract_lokr,
            _extract_glora,
        )
        for extractor in extractor_sequence:
            spec = extractor(logical_key, target_param, tensor_map, loaded)
            if spec:
                specs.append(spec)
        specs.extend(_extract_diff(logical_key, target_param, tensor_map, loaded))

    log_missing_keys(tensor_map.keys(), loaded, logger=LOGGER)
    return specs, loaded
