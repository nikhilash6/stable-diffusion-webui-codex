"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z-Image L2P runtime assembly.
Builds the pixel-space no-VAE L2P runtime from a core denoiser checkpoint plus exactly one external Qwen3-4B text encoder, supporting
SafeTensors and GGUF for both components while preserving native checkpoint key names.

Symbols (top-level; keep in sync; no ghosts):
- `ZImageL2PQwenPatcher` (class): Qwen text-encoder wrapper exposing a memory-manager `ModelPatcher`.
- `_torch_dtype_label` (function): Converts a `torch.dtype` into a supported dtype label (`fp16|bf16|fp32`).
- `_storage_dtype_label` (function): Converts a storage dtype (torch dtype or string label) into a runtime label.
- `_native_weights_dtype_for_path` (function): Best-effort native dtype inference for SafeTensors files.
- `ZImageL2PTextPipelines` (dataclass): Text processing pipeline bundle for L2P Qwen3-4B masked conditioning.
- `ZImageL2PEngineRuntime` (dataclass): Runtime container for L2P denoiser/text components and dtype metadata.
- `ZImageL2PEngineSpec` (dataclass): L2P engine defaults used by assembly and sampling.
- `_predictor` (function): Builds the flow-match predictor attached to the L2P denoiser patcher.
- `_load_external_text_encoder` (function): Loads the exact Qwen3-4B text encoder from SafeTensors or GGUF.
- `assemble_zimage_l2p_runtime` (function): Assembles the runtime and returns `ZImageL2PEngineRuntime`.
- `ZIMAGE_L2P_SPEC` (constant): Default L2P spec instance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from apps.backend.patchers.base import ModelPatcher
from apps.backend.patchers.denoiser import DenoiserPatcher
from apps.backend.runtime.checkpoint.io import load_torch_file
from apps.backend.runtime.families.zimage.text_encoder import ZImageTextEncoder, ZImageTextProcessingEngine
from apps.backend.runtime.families.zimage_l2p.l2p_model import load_zimage_l2p_from_state_dict
from apps.backend.runtime.logging import get_backend_logger
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.model_registry.specs import ModelFamily, QuantizationKind
from apps.backend.runtime.sampling_adapters.prediction import FlowMatchEulerPrediction


logger = get_backend_logger("backend.engines.zimage_l2p.spec")
_SUPPORTED_CORE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_SUPPORTED_TENC_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


class ZImageL2PQwenPatcher:
    """Qwen wrapper for L2P text encoder memory-management integration."""

    def __init__(self, text_encoder: ZImageTextEncoder) -> None:
        self.text_encoder = text_encoder
        load_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
        offload_device = memory_management.manager.get_offload_device(DeviceRole.TEXT_ENCODER)
        self.patcher = ModelPatcher(
            text_encoder.model,
            load_device=load_device,
            offload_device=offload_device,
        )


def _torch_dtype_label(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    raise ValueError(f"Unsupported Z-Image L2P torch dtype: {dtype!r}")


def _storage_dtype_label(dtype: torch.dtype | str) -> str:
    if isinstance(dtype, torch.dtype):
        return _torch_dtype_label(dtype)
    return str(dtype)


def _native_weights_dtype_for_path(path: str | None) -> torch.dtype | None:
    if not path:
        return None
    candidate = Path(str(path)).expanduser()
    try:
        candidate = candidate.resolve()
    except Exception:
        candidate = candidate.absolute()
    if not candidate.is_file() or candidate.suffix.lower() not in {".safetensor", ".safetensors"}:
        return None

    from apps.backend.runtime.checkpoint.safetensors_header import detect_safetensors_primary_dtype

    hint = detect_safetensors_primary_dtype(candidate)
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }.get(hint)


@dataclass(frozen=True)
class ZImageL2PTextPipelines:
    qwen3_text: ZImageTextProcessingEngine


@dataclass
class ZImageL2PEngineRuntime:
    denoiser: DenoiserPatcher
    text: ZImageL2PTextPipelines
    qwen: ZImageL2PQwenPatcher
    device: str
    core_storage_dtype: str = "bf16"
    core_compute_dtype: str = "fp32"
    te_storage_dtype: str = "bf16"
    te_compute_dtype: str = "fp32"


@dataclass(frozen=True)
class ZImageL2PEngineSpec:
    name: str = "zimage_l2p"
    family: ModelFamily = ModelFamily.ZIMAGE_L2P
    flow_shift: float = 3.0
    default_steps: int = 30
    default_cfg_scale: float = 2.0


def _predictor(spec: ZImageL2PEngineSpec) -> FlowMatchEulerPrediction:
    logger.debug("Using FlowMatch predictor for Z-Image L2P (shift=%.2f)", spec.flow_shift)
    return FlowMatchEulerPrediction(pseudo_timestep_range=1000, shift=spec.flow_shift)


def _require_text_encoder_path(tenc_path: str | None) -> str:
    if tenc_path is None or not str(tenc_path).strip():
        raise ValueError(
            "Z-Image L2P requires exactly one external Qwen3-4B text encoder. "
            "Select a qwen3_4b asset so the request can pass extras.tenc_sha."
        )
    resolved = str(Path(str(tenc_path).strip()).expanduser())
    if not Path(resolved).is_file():
        raise RuntimeError(f"Z-Image L2P text encoder path not found: {resolved}")
    suffix = Path(resolved).suffix.lower()
    if suffix not in {".safetensor", ".safetensors", ".gguf"}:
        raise RuntimeError(
            "Z-Image L2P text encoder must be SafeTensors or GGUF; "
            f"got path={resolved!r}."
        )
    return resolved


def _load_external_text_encoder(tenc_path: str | None, *, torch_dtype: torch.dtype) -> ZImageTextEncoder:
    resolved = _require_text_encoder_path(tenc_path)
    logger.debug("Loading Z-Image L2P Qwen3-4B text encoder: %s", resolved)
    if resolved.lower().endswith(".gguf"):
        return ZImageTextEncoder.from_gguf(resolved, torch_dtype=torch_dtype)

    state_dict = load_torch_file(resolved, device=memory_management.manager.cpu_device)
    if isinstance(state_dict, Mapping) and len(state_dict) == 1 and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(
            "Z-Image L2P external Qwen3-4B text encoder must resolve to a state_dict mapping; "
            f"got {type(state_dict).__name__}."
        )
    return ZImageTextEncoder.from_state_dict(state_dict, torch_dtype=torch_dtype)


def _load_l2p_denoiser(
    *,
    state_dict: Mapping[str, Any],
    model_path: str,
    model_format: str,
    quantization_kind: QuantizationKind,
) -> tuple[Any, torch.dtype | str, torch.dtype]:
    load_device = memory_management.manager.get_device(DeviceRole.CORE)
    offload_device = memory_management.manager.get_offload_device(DeviceRole.CORE)
    mem_config = memory_management.manager.config
    native_dtype = _native_weights_dtype_for_path(model_path)
    storage_dtype: torch.dtype | str
    weight_format: str | None = None
    if model_format == "gguf" or quantization_kind is QuantizationKind.GGUF:
        storage_dtype = "gguf"
        weight_format = "gguf"
    else:
        storage_dtype = memory_management.manager.dtype_for_role(
            DeviceRole.CORE,
            supported=_SUPPORTED_CORE_DTYPES,
            native_dtype=native_dtype,
        )

    computation_dtype = memory_management.manager.compute_dtype_for_role(
        DeviceRole.CORE,
        supported=_SUPPORTED_CORE_DTYPES,
        storage_dtype=storage_dtype if isinstance(storage_dtype, torch.dtype) else None,
    )

    if weight_format == "gguf":
        initial_device = offload_device if load_device.type != memory_management.manager.cpu_device.type else load_device
        construct_dtype = torch.bfloat16 if initial_device.type == "cpu" else computation_dtype
    else:
        prefer_gpu = bool(getattr(mem_config, "gpu_prefer_construct", False))
        initial_device = load_device if prefer_gpu else offload_device
        construct_dtype = storage_dtype if isinstance(storage_dtype, torch.dtype) else computation_dtype
        if initial_device.type == "cpu" and construct_dtype in (torch.bfloat16, torch.float16):
            construct_dtype = torch.float32

    model = load_zimage_l2p_from_state_dict(
        state_dict,
        device=initial_device,
        dtype=construct_dtype,
        weight_format=weight_format,
        storage_dtype=storage_dtype,
        computation_dtype=computation_dtype,
        load_device=load_device,
        offload_device=offload_device,
        initial_device=initial_device,
    )
    return model, storage_dtype, computation_dtype


def assemble_zimage_l2p_runtime(
    *,
    spec: ZImageL2PEngineSpec,
    codex_components: Mapping[str, object],
    estimated_config: Any,
    model_path: str,
    model_format: str,
    device: str,
    tenc_path: str | None = None,
) -> ZImageL2PEngineRuntime:
    """Assemble the no-VAE L2P runtime from parser components and external Qwen."""

    if model_format not in {"checkpoint", "gguf"}:
        raise RuntimeError(f"Z-Image L2P supports model_format='checkpoint'|'gguf'; got {model_format!r}.")
    transformer_state = codex_components.get("transformer")
    if not isinstance(transformer_state, Mapping):
        raise RuntimeError(
            "Z-Image L2P loader expected a transformer state-dict mapping from the parser; "
            f"got {type(transformer_state).__name__}."
        )

    quantization_kind = getattr(getattr(estimated_config, "quantization", None), "kind", QuantizationKind.NONE)
    denoiser_model, core_storage_dtype, core_compute_dtype = _load_l2p_denoiser(
        state_dict=transformer_state,
        model_path=model_path,
        model_format=model_format,
        quantization_kind=quantization_kind,
    )
    denoiser = DenoiserPatcher.from_model(
        model=denoiser_model,
        diffusers_scheduler=None,
        predictor=_predictor(spec),
        config=estimated_config,
    )

    tenc_native_dtype = _native_weights_dtype_for_path(tenc_path)
    tenc_storage_dtype = memory_management.manager.dtype_for_role(
        DeviceRole.TEXT_ENCODER,
        supported=_SUPPORTED_TENC_DTYPES,
        native_dtype=tenc_native_dtype,
    )
    text_encoder = _load_external_text_encoder(tenc_path, torch_dtype=tenc_storage_dtype)
    qwen = ZImageL2PQwenPatcher(text_encoder)
    text_engine = ZImageTextProcessingEngine(text_encoder)
    te_compute_dtype = memory_management.manager.compute_dtype_for_role(
        DeviceRole.TEXT_ENCODER,
        supported=_SUPPORTED_TENC_DTYPES,
        storage_dtype=tenc_storage_dtype,
    )
    setattr(text_encoder.model, "compute_dtype", te_compute_dtype)

    logger.debug(
        "Z-Image L2P runtime assembled: device=%s core_storage=%s core_compute=%s te_storage=%s te_compute=%s",
        device,
        _storage_dtype_label(core_storage_dtype),
        _torch_dtype_label(core_compute_dtype),
        _torch_dtype_label(tenc_storage_dtype),
        _torch_dtype_label(te_compute_dtype),
    )
    return ZImageL2PEngineRuntime(
        denoiser=denoiser,
        text=ZImageL2PTextPipelines(qwen3_text=text_engine),
        qwen=qwen,
        device=device,
        core_storage_dtype=_storage_dtype_label(core_storage_dtype),
        core_compute_dtype=_torch_dtype_label(core_compute_dtype),
        te_storage_dtype=_torch_dtype_label(tenc_storage_dtype),
        te_compute_dtype=_torch_dtype_label(te_compute_dtype),
    )


ZIMAGE_L2P_SPEC = ZImageL2PEngineSpec()

__all__ = [
    "ZImageL2PTextPipelines",
    "ZImageL2PEngineRuntime",
    "ZImageL2PEngineSpec",
    "ZImageL2PQwenPatcher",
    "assemble_zimage_l2p_runtime",
    "ZIMAGE_L2P_SPEC",
]
