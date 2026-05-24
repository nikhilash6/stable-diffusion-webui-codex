"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z Image engine specification and runtime assembly (analogous to Flux `spec.py`).
Builds a `ZImageEngineRuntime` from parsed components, loading external VAE/text-encoder assets when required (core-only checkpoints) while preserving role-owned storage/compute dtype selection.
Uses vendored diffusers scheduler metadata under `apps/backend/huggingface/Tongyi-MAI/**` for flow-shift parity when not overridden.

Symbols (top-level; keep in sync; no ghosts):
- `ZImageQwenPatcher` (class): Qwen text-encoder wrapper that exposes a `ModelPatcher` for memory management integration.
- `_torch_dtype_label` (function): Converts a `torch.dtype` into a supported dtype label (`fp16|bf16|fp32`) for logs/metadata.
- `_storage_dtype_label` (function): Converts a storage dtype (torch dtype or string label) into a runtime label.
- `_native_weights_dtype_for_path` (function): Best-effort native dtype inference for a weights file/directory (SafeTensors header).
- `ZImageTextPipelines` (dataclass): Text processing pipeline bundle for the Z Image engine (currently Qwen3 text).
- `ZImageEngineRuntime` (dataclass): Runtime container for Z Image engine components, including explicit storage vs compute dtypes per role.
- `ZImageEngineSpec` (dataclass): Engine specification delegating defaults to `FamilyRuntimeSpec` with optional overrides.
- `_predictor` (function): Builds the flow-match predictor used by the denoiser patcher for Z Image sampling (effective shift / alpha).
- `_load_external_vae` (function): Loads an external Flow16 VAE from a path (required for core-only; optional override for full checkpoints).
- `_load_external_text_encoder` (function): Loads an external Qwen3 text encoder from a path (required for core-only; optional override for full checkpoints).
- `assemble_zimage_runtime` (function): Assembles the runtime (including external assets and role-owned dtype selection when required) and returns a `ZImageEngineRuntime`.
- `ZIMAGE_SPEC` (constant): Default Z Image engine spec instance.
- `__all__` (constant): Public export list for this module.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from apps.backend.patchers.base import ModelPatcher
from apps.backend.patchers.denoiser import DenoiserPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.model_registry.family_runtime import get_family_spec, FamilyRuntimeSpec
from apps.backend.runtime.model_registry.flow_shift import flow_shift_spec_from_repo_dir
from apps.backend.runtime.ops.operations_gguf import is_packed_gguf_artifact
from apps.backend.runtime.sampling_adapters.prediction import FlowMatchEulerPrediction
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.checkpoint.io import load_torch_file

logger = get_backend_logger("backend.engines.zimage.spec")

def _torch_dtype_label(dtype):
    import torch

    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    raise ValueError(f"Unsupported Z Image torch dtype: {dtype!r}")


def _storage_dtype_label(dtype) -> str:
    import torch

    if isinstance(dtype, torch.dtype):
        return _torch_dtype_label(dtype)
    return str(dtype)


def _native_weights_dtype_for_path(path: str | None):
    """Best-effort native dtype inference for a weights file/directory.

    Uses SafeTensors header hints when possible; returns None when unknown (e.g., GGUF).
    """

    if not path:
        return None
    import torch
    from pathlib import Path

    p = Path(str(path)).expanduser()
    try:
        p = p.resolve()
    except Exception:
        p = p.absolute()

    candidates: list[Path] = []
    if p.is_file():
        candidates = [p]
    elif p.is_dir():
        # Prefer diffusers-style filename when present.
        preferred = p / "diffusion_pytorch_model.safetensors"
        if preferred.is_file():
            candidates = [preferred]
        else:
            candidates = sorted(p.glob("*.safetensors"))

    if not candidates:
        return None

    from apps.backend.runtime.checkpoint.safetensors_header import detect_safetensors_primary_dtype

    hint = detect_safetensors_primary_dtype(candidates[0])
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }
    fp8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    fp8_e5m2 = getattr(torch, "float8_e5m2", None)
    if isinstance(fp8_e4m3fn, torch.dtype):
        mapping["fp8_e4m3fn"] = fp8_e4m3fn
    if isinstance(fp8_e5m2, torch.dtype):
        mapping["fp8_e5m2"] = fp8_e5m2
    return mapping.get(hint)

class ZImageQwenPatcher:
    """Qwen wrapper for Z Image text encoder with memory management support.
    
    This wrapper provides a `patcher` attribute that integrates with the
    memory management system, allowing automatic GPU loading/offloading.
    """
    
    def __init__(self, text_encoder):
        """Initialize with a ZImageTextEncoder.
        
        Args:
        text_encoder: A ZImageTextEncoder instance with a `model` attribute.
        """
        self.text_encoder = text_encoder
        
        # Create ModelPatcher for memory management integration
        load_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
        offload_device = memory_management.manager.get_offload_device(DeviceRole.TEXT_ENCODER)
        
        # The patcher wraps the underlying model, enabling load_model_gpu/unload_model
        self.patcher = ModelPatcher(
            text_encoder.model,
            load_device=load_device,
            offload_device=offload_device,
        )


@dataclass(frozen=True)
class ZImageTextPipelines:
    """Text processing pipeline for Z Image."""
    qwen3_text: Any  # ZImageTextProcessingEngine


@dataclass
class ZImageEngineRuntime:
    """Runtime container for Z Image engine components."""
    vae: VAE
    denoiser: DenoiserPatcher  # wraps ZImageTransformer2DModel
    text: ZImageTextPipelines
    qwen: ZImageQwenPatcher  # wrapper with ModelPatcher for memory management
    device: str
    core_storage_dtype: str = "bf16"
    core_compute_dtype: str = "fp32"
    te_storage_dtype: str = "bf16"
    te_compute_dtype: str = "fp32"
    vae_storage_dtype: str = "bf16"
    vae_compute_dtype: str = "fp32"


@dataclass(frozen=True)
class ZImageEngineSpec:
    """Specification for Z Image engine.
    
    This spec delegates to FamilyRuntimeSpec for default values,
    with optional per-variant overrides.
    """
    name: str = "zimage"
    family: ModelFamily = ModelFamily.ZIMAGE
    
    # Optional overrides (if None, delegates to FamilyRuntimeSpec)
    _flow_shift_override: Optional[float] = field(default=None, repr=False)
    _default_steps_override: Optional[int] = field(default=None, repr=False)
    _default_cfg_override: Optional[float] = field(default=None, repr=False)
    
    def _get_family_spec(self) -> FamilyRuntimeSpec:
        """Get the FamilyRuntimeSpec for this engine."""
        return get_family_spec(self.family)
    
    @property
    def flow_shift(self) -> float:
        """Flow-match effective shift (alpha) for Z Image Turbo.

        Source of truth is the vendored diffusers scheduler config:
        `apps/backend/huggingface/Tongyi-MAI/Z-Image-Turbo/scheduler/scheduler_config.json`.
        """
        if self._flow_shift_override is not None:
            return self._flow_shift_override
        from apps.backend.infra.config.repo_root import get_repo_root

        repo_root = get_repo_root()
        vendor_dir = repo_root / "apps" / "backend" / "huggingface" / "Tongyi-MAI" / "Z-Image-Turbo"
        spec = flow_shift_spec_from_repo_dir(vendor_dir)
        return spec.resolve_effective_shift()
    
    @property
    def default_steps(self) -> int:
        """Default sampling steps, delegating to FamilyRuntimeSpec if not overridden."""
        if self._default_steps_override is not None:
            return self._default_steps_override
        return self._get_family_spec().default_steps
    
    @property
    def default_cfg_scale(self) -> float:
        """Default CFG scale, delegating to FamilyRuntimeSpec if not overridden."""
        if self._default_cfg_override is not None:
            return self._default_cfg_override
        return self._get_family_spec().default_cfg


def _predictor(spec: ZImageEngineSpec) -> FlowMatchEulerPrediction:
    """Create flow-match predictor for Z Image."""
    logger.debug("Using FlowMatch predictor for Z Image (shift=%.2f)", spec.flow_shift)
    return FlowMatchEulerPrediction(pseudo_timestep_range=1000, shift=spec.flow_shift)


def _load_external_vae(vae_path: str | None, *, torch_dtype) -> object:
    """Load VAE from external path for core-only checkpoints.
    
    Uses the shared Flow16 VAE loader since Z Image uses the same
    16-channel latent space as Flux.
    """
    from apps.backend.runtime.common.vae import load_flow16_vae
    from apps.backend.runtime.model_registry.specs import ModelFamily
    
    if vae_path is None or not str(vae_path).strip():
        raise ValueError(
            "Z Image core-only checkpoint requires an external VAE. "
            "Please select a VAE so the request can include 'vae_sha' (sha256)."
        )
    
    return load_flow16_vae(vae_path, dtype=torch_dtype, family=ModelFamily.ZIMAGE)


def _load_external_text_encoder(tenc_path: str | None, *, torch_dtype) -> object:
    """Load Qwen3 text encoder from external path for core-only checkpoints."""
    
    # Text encoder path is required - no automatic fallback
    if tenc_path is None:
        raise ValueError(
            "Z Image core-only checkpoint requires external text encoder (Qwen3-4B). "
            "Please select one in the UI or place it in models/zimage-tenc/"
        )
    
    logger.debug("Loading external text encoder from: %s", tenc_path)
    
    # Detect if GGUF or safetensors
    if tenc_path.lower().endswith(".gguf"):
        # Load GGUF text encoder
        from apps.backend.runtime.families.zimage.text_encoder import ZImageTextEncoder
        encoder = ZImageTextEncoder.from_gguf(tenc_path, torch_dtype=torch_dtype)
    else:
        # Load safetensors text encoder
        from apps.backend.runtime.families.zimage.text_encoder import ZImageTextEncoder
        state_dict = load_torch_file(tenc_path, device=memory_management.manager.cpu_device)
        if isinstance(state_dict, Mapping) and len(state_dict) == 1 and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise RuntimeError(
                "Z Image external text encoder must resolve to a state_dict mapping; "
                f"got {type(state_dict).__name__}."
            )
        encoder = ZImageTextEncoder.from_state_dict(state_dict, torch_dtype=torch_dtype)
    
    return encoder


def assemble_zimage_runtime(
    *,
    spec: ZImageEngineSpec,
    codex_components: Mapping[str, object],
    estimated_config: Any,
    device: str,
    vae_path: str | None = None,
    tenc_path: str | None = None,
) -> ZImageEngineRuntime:
    """Assemble Z Image runtime from components.
    
    Args:
        spec: Engine specification.
        codex_components: Dict with 'transformer', optionally 'vae', 'text_encoder'.
        estimated_config: Model config.
        device: Target device.
        vae_path: Optional external VAE path (required when the checkpoint is core-only).
        tenc_path: Optional external text encoder path (required when the checkpoint is core-only).
    
    Returns:
        Assembled ZImageEngineRuntime.
    """
    import torch

    def _log_vram(label: str) -> None:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            alloc = torch.cuda.memory_allocated()
            logger.debug("[zimage-assemble] %s: free=%.2f GB, alloc=%.2f GB", label, free / 1e9, alloc / 1e9)

    def _is_floating_dtype(dtype: object) -> bool:
        if not isinstance(dtype, torch.dtype):
            return False
        try:
            return bool(torch.empty((), dtype=dtype).is_floating_point())
        except (TypeError, RuntimeError):
            return False
    
    logger.debug("Assembling Z Image runtime")
    _log_vram("START")
    
    # Get transformer (always required)
    transformer = codex_components.get("transformer")
    if transformer is None:
        raise ValueError("Z Image requires 'transformer' component")

    core_storage_raw = getattr(transformer, "storage_dtype", None)
    if core_storage_raw is None:
        try:
            core_storage_raw = next(transformer.parameters()).dtype
        except Exception:  # noqa: BLE001
            core_storage_raw = torch.float32
    core_compute_raw = getattr(transformer, "computation_dtype", None)
    if not isinstance(core_compute_raw, torch.dtype):
        core_compute_raw = memory_management.manager.compute_dtype_for_role(DeviceRole.CORE)
    if any(is_packed_gguf_artifact(parameter) for parameter in transformer.parameters()):
        raise RuntimeError(
            "Packed GGUF artifacts are not supported on the root runtime path. "
            "Load the base `.gguf` artifact instead."
        )

    _log_vram("AFTER get transformer")
    
    # Detect core-only checkpoints (no VAE or text encoder embedded in components).
    base_vae = codex_components.get("vae")
    base_text_encoder = codex_components.get("text_encoder")
    is_core_only = base_vae is None or base_text_encoder is None

    vae_model = base_vae
    text_encoder = base_text_encoder

    external_vae = str(vae_path or "").strip() or None
    external_tenc = str(tenc_path or "").strip() or None
    external_tenc_storage_dtype: torch.dtype | None = None

    # External assets are opt-in for full checkpoints, and required for core-only
    # checkpoints (GGUF / transformer-only exports).
    if external_vae is not None:
        logger.debug("Z Image: loading external VAE (vae_path=%s)", external_vae)
        vae_native_dtype = _native_weights_dtype_for_path(external_vae)
        vae_storage_dtype = memory_management.manager.dtype_for_role(DeviceRole.VAE, native_dtype=vae_native_dtype) if vae_native_dtype is not None else memory_management.manager.dtype_for_role(DeviceRole.VAE)
        vae_model = _load_external_vae(external_vae, torch_dtype=vae_storage_dtype)
        _log_vram("AFTER load external VAE")
    if vae_model is None:
        if is_core_only:
            _load_external_vae(None, torch_dtype=torch.float32)  # raises with an actionable message
        raise ValueError("Z Image checkpoint did not include a VAE; provide an external VAE via 'vae_sha'.")

    if external_tenc is not None:
        logger.debug("Z Image: loading external text encoder (tenc_path=%s)", external_tenc)
        tenc_native_dtype = _native_weights_dtype_for_path(external_tenc)
        tenc_storage_dtype = memory_management.manager.dtype_for_role(
            DeviceRole.TEXT_ENCODER,
            native_dtype=tenc_native_dtype,
        ) if tenc_native_dtype is not None else memory_management.manager.dtype_for_role(DeviceRole.TEXT_ENCODER)
        external_tenc_storage_dtype = tenc_storage_dtype
        text_encoder = _load_external_text_encoder(external_tenc, torch_dtype=tenc_storage_dtype)
        _log_vram("AFTER load external TEnc")
    if text_encoder is None:
        if is_core_only:
            _load_external_text_encoder(None, torch_dtype=torch.float32)  # raises with an actionable message
        raise ValueError("Z Image checkpoint did not include a text encoder; provide one via 'tenc_sha'.")
    
    # Wrap VAE
    vae = VAE(model=vae_model, family=ModelFamily.ZIMAGE)
    _log_vram("AFTER VAE wrapper")
    
    # Wrap transformer in DenoiserPatcher
    predictor = _predictor(spec)
    denoiser = DenoiserPatcher.from_model(
        model=transformer,
        diffusers_scheduler=None,
        predictor=predictor,
        config=estimated_config,
    )
    _log_vram("AFTER DenoiserPatcher.from_model")
    
    # Wrap text encoder with ZImageQwenPatcher for memory management
    qwen = ZImageQwenPatcher(text_encoder)
    _log_vram("AFTER ZImageQwenPatcher wrapper")
    
    # Create text processing engine
    from apps.backend.runtime.families.zimage.text_encoder import ZImageTextProcessingEngine
    text_engine = ZImageTextProcessingEngine(text_encoder)

    _log_vram("FINAL")
    if external_tenc_storage_dtype is not None:
        te_storage = external_tenc_storage_dtype
    else:
        te_storage = getattr(text_encoder, "dtype", None)
        if not _is_floating_dtype(te_storage):
            raise RuntimeError(
                "Z Image text encoder storage dtype must resolve to a floating dtype before runtime assembly; "
                f"got {te_storage!r}."
            )
    te_storage = memory_management.manager.dtype_for_role(DeviceRole.TEXT_ENCODER, native_dtype=te_storage)
    te_compute = memory_management.manager.compute_dtype_for_role(DeviceRole.TEXT_ENCODER, storage_dtype=te_storage)
    setattr(text_encoder.model, "compute_dtype", te_compute)

    vae_storage = torch.float32
    try:
        vae_storage = next(vae_model.parameters()).dtype
    except Exception:  # noqa: BLE001
        vae_storage = torch.float32
    vae_storage = memory_management.manager.dtype_for_role(DeviceRole.VAE, native_dtype=vae_storage)
    vae_compute = memory_management.manager.compute_dtype_for_role(DeviceRole.VAE, storage_dtype=vae_storage)

    logger.debug(
        "Z Image runtime assembled: device=%s core_storage=%s core_compute=%s te_storage=%s te_compute=%s vae_storage=%s vae_compute=%s core_only=%s",
        device,
        _storage_dtype_label(core_storage_raw),
        _torch_dtype_label(core_compute_raw),
        _torch_dtype_label(te_storage),
        _torch_dtype_label(te_compute),
        _torch_dtype_label(vae_storage),
        _torch_dtype_label(vae_compute),
        is_core_only,
    )
    
    return ZImageEngineRuntime(
        vae=vae,
        denoiser=denoiser,
        text=ZImageTextPipelines(qwen3_text=text_engine),
        qwen=qwen,
        device=device,
        core_storage_dtype=_storage_dtype_label(core_storage_raw),
        core_compute_dtype=_torch_dtype_label(core_compute_raw),
        te_storage_dtype=_torch_dtype_label(te_storage),
        te_compute_dtype=_torch_dtype_label(te_compute),
        vae_storage_dtype=_torch_dtype_label(vae_storage),
        vae_compute_dtype=_torch_dtype_label(vae_compute),
    )


# Default spec
ZIMAGE_SPEC = ZImageEngineSpec()

__all__ = [
    "ZImageTextPipelines",
    "ZImageEngineRuntime",
    "ZImageEngineSpec",
    "assemble_zimage_runtime",
    "ZIMAGE_SPEC",
]
