"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tensor planning helpers for the GGUF converter.
Plans per-tensor quantization/storage settings for the source/native keyspace emitted by the converter and exposes lightweight config
classifiers/metadata normalizers for supported component families.

Symbols (top-level; keep in sync; no ghosts):
- `TensorPlan` (dataclass): Planned tensor conversion entry (name/shape/type + storage strategy).
- `plan_tensors` (function): Plan per-tensor conversion settings for a safetensors source.
- `_select_ggml_type` (function): Selects an effective GGML type for a tensor (requested + per-name override rules).
- `is_zimage_transformer_config` (function): Returns True when a config.json represents a Z-Image transformer export.
- `normalize_zimage_transformer_metadata_config` (function): Adapts Z-Image transformer config fields to metadata helper inputs (variant-neutral; no Turbo defaulting).
- `is_flux_transformer_config` (function): Returns True when a config.json represents a Flux transformer export.
- `normalize_flux_transformer_metadata_config` (function): Adapts Flux transformer config fields to metadata helper inputs.
- `is_wan22_transformer_config` (function): Returns True when a config.json represents a WAN22 transformer export.
- `normalize_wan22_transformer_metadata_config` (function): Adapts WAN22 transformer config fields to metadata helper inputs.
- `is_ltx2_transformer_config` (function): Returns True when a config.json represents an LTX2 transformer export.
- `normalize_ltx2_transformer_metadata_config` (function): Adapts LTX2 transformer config fields to metadata helper inputs.
- `is_gemma3_text_encoder_config` (function): Returns True when a config.json represents a Gemma3 text encoder export.
- `normalize_gemma3_text_encoder_metadata_config` (function): Adapts Gemma3 config fields to metadata helper inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from apps.backend.quantization.gguf import GGMLQuantizationType
from apps.backend.quantization.gguf.quant_shapes import quant_shape_to_byte_shape
from apps.backend.runtime.tools.gguf_converter_quantization import select_tensor_ggml_type
from apps.backend.runtime.tools.gguf_converter_specs import CompiledTensorTypeRule

_ZIMAGE_PAD_TOKENS = {"x_pad_token", "cap_pad_token"}


@dataclass(frozen=True, slots=True)
class TensorPlan:
    src_name: str
    gguf_name: str
    raw_shape: tuple[int, ...]
    ggml_type: GGMLQuantizationType
    stored_shape: tuple[int, ...]
    stored_dtype: np.dtype
    stored_nbytes: int
    op: str
    src_names: tuple[str, ...]


def plan_tensors(
    tensor_names: list[str],
    safetensors_handle: Any,
    key_mapping: dict[str, str],
    requested: GGMLQuantizationType,
    overrides: list[CompiledTensorTypeRule],
) -> list[TensorPlan]:
    plans: list[TensorPlan] = []

    for src_name in tensor_names:
        sl = safetensors_handle.get_slice(src_name)
        raw_shape = tuple(int(x) for x in sl.get_shape())
        gguf_name = key_mapping.get(src_name, src_name)

        desired = requested
        for rule in overrides:
            if rule.apply_to.matches_src() and rule.pattern.search(src_name):
                desired = rule.ggml_type
            if rule.apply_to.matches_dst() and rule.pattern.search(gguf_name):
                desired = rule.ggml_type
        ggml_type = _select_ggml_type(raw_shape=raw_shape, gguf_name=gguf_name, desired=desired)

        if ggml_type == GGMLQuantizationType.F16:
            stored_dtype = np.dtype(np.float16)
            stored_shape = raw_shape
            stored_nbytes = int(np.prod(raw_shape, dtype=np.int64) * 2)
        elif ggml_type == GGMLQuantizationType.BF16:
            stored_dtype = np.dtype(np.uint16)
            stored_shape = raw_shape
            stored_nbytes = int(np.prod(raw_shape, dtype=np.int64) * 2)
        elif ggml_type == GGMLQuantizationType.F32:
            stored_dtype = np.dtype(np.float32)
            stored_shape = raw_shape
            stored_nbytes = int(np.prod(raw_shape, dtype=np.int64) * 4)
        else:
            stored_dtype = np.dtype(np.uint8)
            stored_shape = quant_shape_to_byte_shape(raw_shape, ggml_type)
            stored_nbytes = int(np.prod(stored_shape, dtype=np.int64))

        plans.append(
            TensorPlan(
                src_name=src_name,
                gguf_name=gguf_name,
                raw_shape=raw_shape,
                ggml_type=ggml_type,
                stored_shape=stored_shape,
                stored_dtype=stored_dtype,
                stored_nbytes=stored_nbytes,
                op="copy",
                src_names=(src_name,),
            )
        )

    return plans


def _select_ggml_type(
    *,
    raw_shape: tuple[int, ...],
    gguf_name: str,
    desired: GGMLQuantizationType,
) -> GGMLQuantizationType:
    ggml_type = select_tensor_ggml_type(raw_shape, desired)

    if gguf_name in _ZIMAGE_PAD_TOKENS:
        if ggml_type == GGMLQuantizationType.BF16:
            raise RuntimeError(
                f"BF16 override is not supported for Z-Image pad token {gguf_name!r}; use F16/F32 or auto."
            )
        if ggml_type not in {GGMLQuantizationType.F16, GGMLQuantizationType.F32}:
            return GGMLQuantizationType.F16

    return ggml_type


def is_zimage_transformer_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("_class_name") or "") == "ZImageTransformer2DModel"


def is_flux_transformer_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("_class_name") or "") == "FluxTransformer2DModel"


def is_wan22_transformer_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("_class_name") or "") in {"WanTransformer3DModel", "WanModel"}


def is_ltx2_transformer_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("_class_name") or "") == "LTX2VideoTransformer3DModel"


def is_gemma3_text_encoder_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("model_type") or "").strip() == "gemma3"


def normalize_flux_transformer_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Flux transformer config keys into the metadata helper's expected fields."""

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    num_heads = _as_int(config.get("num_attention_heads"), 24)
    head_dim = _as_int(config.get("attention_head_dim"), 128)
    hidden = max(1, num_heads) * max(1, head_dim)

    double_layers = _as_int(config.get("num_layers"), 19)
    single_layers = _as_int(config.get("num_single_layers"), 38)

    return {
        "model_type": "flux",
        "num_hidden_layers": max(1, double_layers + single_layers),
        "hidden_size": hidden,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "_name_or_path": str(
            config.get("_name_or_path") or config.get("name") or "black-forest-labs/FLUX.1-Kontext-dev"
        ),
    }


def normalize_wan22_transformer_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt WAN22 transformer config keys into the metadata helper's expected fields."""

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    class_name = str(config.get("_class_name") or "")
    if class_name == "WanModel":
        hidden = _as_int(config.get("dim"), 4096)
        num_layers = _as_int(config.get("num_layers"), 32)
        num_heads = _as_int(config.get("num_heads"), 32)
        max_pos = _as_int(config.get("text_len"), 512)
        eps = _as_float(config.get("eps"), 1e-6)
        model_type = str(config.get("model_type") or "").strip().lower()
        name = "Wan-AI/Wan2.2"
        if model_type == "i2v":
            name = "Wan-AI/Wan2.2-I2V-A14B"
        elif model_type == "t2v":
            name = "Wan-AI/Wan2.2-T2V-A14B"
        return {
            "model_type": "wan22",
            "num_hidden_layers": max(1, num_layers),
            "hidden_size": max(1, hidden),
            "num_attention_heads": max(1, num_heads),
            "num_key_value_heads": max(1, num_heads),
            "max_position_embeddings": max(1, max_pos),
            "rope_theta": 10000.0,
            "rms_norm_eps": eps,
            "_name_or_path": str(config.get("_name_or_path") or config.get("name") or name),
        }

    num_heads = _as_int(config.get("num_attention_heads"), 40)
    head_dim = _as_int(config.get("attention_head_dim"), 128)
    hidden = max(1, num_heads) * max(1, head_dim)

    num_layers = _as_int(config.get("num_layers"), 40)
    max_pos = _as_int(config.get("rope_max_seq_len"), 1024)
    eps = _as_float(config.get("eps"), 1e-6)

    name = "Wan-AI/Wan2.2"
    in_channels = _as_int(config.get("in_channels"), 16)
    if in_channels == 36:
        name = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    elif in_channels == 16:
        name = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    return {
        "model_type": "wan22",
        "num_hidden_layers": max(1, num_layers),
        "hidden_size": hidden,
        "num_attention_heads": max(1, num_heads),
        "num_key_value_heads": max(1, num_heads),
        "max_position_embeddings": max(1, max_pos),
        "rope_theta": 10000.0,
        "rms_norm_eps": eps,
        "_name_or_path": str(config.get("_name_or_path") or config.get("name") or name),
    }


def normalize_ltx2_transformer_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt LTX2 transformer config keys into the metadata helper's expected fields."""

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    num_heads = _as_int(config.get("num_attention_heads"), 32)
    head_dim = _as_int(config.get("attention_head_dim"), 128)
    hidden = max(1, num_heads) * max(1, head_dim)
    num_layers = _as_int(config.get("num_layers"), 48)
    rope_theta = _as_float(config.get("rope_theta"), 10000.0)
    norm_eps = _as_float(config.get("norm_eps"), 1e-6)

    return {
        "model_type": "ltx2",
        "num_hidden_layers": max(1, num_layers),
        "hidden_size": hidden,
        "num_attention_heads": max(1, num_heads),
        "num_key_value_heads": max(1, num_heads),
        "max_position_embeddings": 4096,
        "rope_theta": rope_theta,
        "rms_norm_eps": norm_eps,
        "_name_or_path": str(config.get("_name_or_path") or config.get("name") or "Lightricks/LTX-2"),
    }


def normalize_gemma3_text_encoder_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Gemma3 text encoder config keys into the metadata helper's expected fields.

    Gemma3 `config.json` used by Diffusers is a multimodal envelope that contains the LLM config under `text_config`.
    """

    text_cfg = config.get("text_config")
    if not isinstance(text_cfg, dict):
        raise ValueError("Gemma3 metadata normalize: expected `text_config` dict in config.json")

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    num_layers = _as_int(text_cfg.get("num_hidden_layers"), 48)
    hidden = _as_int(text_cfg.get("hidden_size"), 3840)
    num_heads = _as_int(text_cfg.get("num_attention_heads"), 16)
    num_kv = _as_int(text_cfg.get("num_key_value_heads"), 8)
    max_pos = _as_int(text_cfg.get("max_position_embeddings"), 131072)
    rope_theta = _as_float(text_cfg.get("rope_theta"), 1000000.0)
    eps = _as_float(text_cfg.get("rms_norm_eps"), 1e-6)

    return {
        "model_type": "gemma3",
        "num_hidden_layers": max(1, num_layers),
        "hidden_size": max(1, hidden),
        "num_attention_heads": max(1, num_heads),
        "num_key_value_heads": max(1, num_kv),
        "max_position_embeddings": max(1, max_pos),
        "rope_theta": rope_theta,
        "rms_norm_eps": eps,
        "_name_or_path": str(config.get("_name_or_path") or config.get("name") or "google/gemma-3"),
    }


def normalize_zimage_transformer_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Z-Image transformer config keys into the metadata helper's expected fields.

    The GGUF metadata helper is LLM-shaped (hidden_size, num_hidden_layers, etc.).
    Z-Image transformer configs use Diffusers keys (`dim`, `n_layers`, ...).
    """

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    patch_size = config.get("all_patch_size")
    if isinstance(patch_size, list) and patch_size:
        patch_size = patch_size[0]
    patch = _as_int(patch_size, 2)

    dim = _as_int(config.get("dim"), 3840)
    n_layers = _as_int(config.get("n_layers"), 30)
    n_heads = _as_int(config.get("n_heads"), 30)
    n_kv = _as_int(config.get("n_kv_heads"), n_heads)

    axes_lens = config.get("axes_lens")
    if isinstance(axes_lens, list) and len(axes_lens) == 3:
        max_pos = max(_as_int(v, 0) for v in axes_lens)
    else:
        max_pos = 4096

    return {
        "model_type": "zimage",
        "num_hidden_layers": n_layers,
        "hidden_size": dim,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "max_position_embeddings": max_pos * max(1, patch),
        "rope_theta": _as_float(config.get("rope_theta"), 256.0),
        "rms_norm_eps": _as_float(config.get("norm_eps"), 1e-5),
        "_name_or_path": str(config.get("_name_or_path") or config.get("name") or "zimage"),
    }


__all__ = [
    "TensorPlan",
    "is_flux_transformer_config",
    "is_gemma3_text_encoder_config",
    "is_ltx2_transformer_config",
    "is_wan22_transformer_config",
    "is_zimage_transformer_config",
    "normalize_flux_transformer_metadata_config",
    "normalize_gemma3_text_encoder_metadata_config",
    "normalize_ltx2_transformer_metadata_config",
    "normalize_wan22_transformer_metadata_config",
    "normalize_zimage_transformer_metadata_config",
    "plan_tensors",
]
