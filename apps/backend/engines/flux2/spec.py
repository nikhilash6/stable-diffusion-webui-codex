"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: FLUX.2 engine spec + truthful runtime assembly for Klein 4B/base-4B.
Assembles a runtime from the loader-resolved FLUX.2 bundle without inventing parallel loading paths: Diffusers
`Flux2Transformer2DModel` + `AutoencoderKLFlux2` + one Qwen3 text encoder/tokenizer. The runtime keeps Codex sampler
state in normalized external 32-channel latent BCHW space, wires the dedicated FLUX.2 denoiser adapter/predictor
bridge, and supplies the FLUX.2 latent encode/decode contract used by txt2img plus image-conditioned img2img.

Symbols (top-level; keep in sync; no ghosts):
- `_torch_dtype_label` (function): Convert canonical torch dtypes into runtime metadata labels (`fp16`/`bf16`/`fp32`).
- `_parameter_dtype` (function): Best-effort parameter dtype reader for loaded modules.
- `_parameter_device` (function): Best-effort parameter device reader for loaded modules.
- `_single_path` (function): Normalize one external asset path from string/list metadata.
- `_metadata_path` (function): Read a normalized external asset path from bundle metadata.
- `_tenc_override_path` (function): Recover the single FLUX.2 text-encoder override path from loader metadata when present.
- `_apply_tokenizer_hint` (function): Bind a tokenizer object/path hint onto the FLUX.2 Qwen wrapper.
- `_load_external_text_encoder` (function): Load the required external FLUX.2 Qwen3-4B text encoder when the loader did not materialize one.
- `_resolve_variant` (function): Resolve supported FLUX.2 variant metadata (`klein` vs `base`) from bundle/config provenance.
- `Flux2TextPipelines` (dataclass): FLUX.2 text pipeline bundle (Qwen3 only).
- `Flux2EngineRuntime` (dataclass): Assembled runtime container (denoiser + VAE + text pipeline + patcher metadata).
- `Flux2EngineSpec` (dataclass): Spec/config holder for FLUX.2 runtime assembly.
- `assemble_flux2_runtime` (function): Assemble a truthful FLUX.2 runtime from loader-provided bundle components.
- `FLUX2_SPEC` (constant): Canonical FLUX.2 engine spec instance.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.patchers.base import ModelPatcher
from apps.backend.patchers.denoiser import DenoiserPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.runtime.checkpoint.io import load_torch_file
from apps.backend.runtime.common.vae import load_flux2_vae
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.families.flux2.runtime import Flux2CoreAdapter
from apps.backend.runtime.families.flux2.text_encoder import Flux2TextEncoder, Flux2TextProcessingEngine
from apps.backend.runtime.model_registry.family_runtime import FamilyRuntimeSpec, get_family_spec
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.sampling_adapters.prediction import FlowMatchEulerPrediction

logger = get_backend_logger("backend.engines.flux2.spec")


def _torch_dtype_label(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    raise ValueError(f"Unsupported torch dtype: {dtype!r}")


def _parameter_dtype(module: object, *, default: torch.dtype | None = None) -> torch.dtype | None:
    if isinstance(module, nn.Module):
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return default
        except Exception:
            return default
    return default


def _parameter_device(module: object, *, default: torch.device) -> torch.device:
    if isinstance(module, nn.Module):
        try:
            return next(module.parameters()).device
        except StopIteration:
            return default
        except Exception:
            return default
    return default


def _single_path(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return os.path.expanduser(trimmed) if trimmed else None
    if isinstance(value, (list, tuple)):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            return None
        if len(cleaned) != 1:
            raise RuntimeError(f"FLUX.2 expects exactly one {label} path; got {len(cleaned)} candidates.")
        return os.path.expanduser(cleaned[0])
    raise TypeError(f"FLUX.2 {label} must be a string or single-item list when provided.")


def _metadata_path(bundle, *, key: str) -> str | None:
    metadata = getattr(bundle, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    return _single_path(metadata.get(key), label=key)



def _tenc_override_path(bundle) -> str | None:
    metadata = getattr(bundle, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    override_paths = metadata.get("tenc_override_paths")
    if not isinstance(override_paths, Mapping):
        return None
    for key in ("text_encoder", "qwen3_4b"):
        resolved = _single_path(override_paths.get(key), label=f"tenc_override_paths[{key!r}]")
        if resolved:
            return resolved
    values = [str(value).strip() for value in override_paths.values() if str(value).strip()]
    if not values:
        return None
    if len(values) != 1:
        raise RuntimeError(
            "FLUX.2 text encoder override metadata is ambiguous; expected exactly one override path, "
            f"got {len(values)}."
        )
    return os.path.expanduser(values[0])


def _apply_tokenizer_hint(
    text_encoder: Flux2TextEncoder,
    *,
    tokenizer: object | None,
    repo_hint: str | None,
) -> None:
    if tokenizer is not None:
        text_encoder.set_tokenizer(tokenizer)

    hint: str | None = None
    if repo_hint:
        repo_root = get_repo_root()
        candidate = repo_root / "apps" / "backend" / "huggingface" / repo_hint / "tokenizer"
        if candidate.is_dir():
            hint = str(candidate)
    if hint is not None:
        text_encoder.set_tokenizer_path_hint(hint)



def _load_external_text_encoder(
    tenc_path: str,
    *,
    torch_dtype: torch.dtype,
) -> Flux2TextEncoder:
    resolved = os.path.expanduser(str(tenc_path).strip())
    if not resolved:
        raise RuntimeError("FLUX.2 text encoder path is empty.")
    if not os.path.exists(resolved):
        raise RuntimeError(f"FLUX.2 text encoder path not found: {resolved}")

    if os.path.isdir(resolved):
        try:
            from transformers import Qwen3ForCausalLM as HfQwen3ForCausalLM
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("transformers is required to load a directory-backed FLUX.2 text encoder.") from exc
        model = HfQwen3ForCausalLM.from_pretrained(
            resolved,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        wrapped = Flux2TextEncoder.from_pretrained_model(model)
        tokenizer_dir = Path(resolved) / "tokenizer"
        if tokenizer_dir.is_dir():
            wrapped.set_tokenizer_path_hint(str(tokenizer_dir))
        return wrapped

    if resolved.lower().endswith(".gguf"):
        return Flux2TextEncoder.from_gguf(resolved, torch_dtype=torch_dtype)

    state_dict = load_torch_file(resolved, device=memory_management.manager.cpu_device)
    if isinstance(state_dict, Mapping) and len(state_dict) == 1 and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(
            "FLUX.2 text encoder override must resolve to a state_dict mapping; "
            f"got {type(state_dict).__name__}."
        )
    return Flux2TextEncoder.from_state_dict(state_dict, torch_dtype=torch_dtype)


@dataclass(frozen=True, slots=True)
class Flux2TextPipelines:
    qwen3_text: Flux2TextProcessingEngine


@dataclass(slots=True)
class Flux2EngineRuntime:
    vae: VAE
    denoiser: DenoiserPatcher
    text: Flux2TextPipelines
    qwen: ModelPatcher
    use_distilled_cfg: bool
    variant: str
    device: torch.device = field(default_factory=lambda: memory_management.manager.mount_device())
    core_storage_dtype: str = "bf16"
    core_compute_dtype: str = "fp32"
    te_storage_dtype: str = "bf16"
    te_compute_dtype: str = "fp32"
    vae_storage_dtype: str = "bf16"
    vae_compute_dtype: str = "fp32"


@dataclass(frozen=True, slots=True)
class Flux2EngineSpec:
    name: str = "flux2"
    family: ModelFamily = ModelFamily.FLUX2
    max_length: int = 512

    def family_spec(self) -> FamilyRuntimeSpec:
        return get_family_spec(self.family)



def _resolve_variant(bundle, *, estimated_config: Any) -> tuple[str, bool, str | None]:
    signature = getattr(bundle, "signature", None)
    extras = getattr(signature, "extras", None)
    repo_hint = getattr(signature, "repo_hint", None)
    if not isinstance(repo_hint, str) or not repo_hint.strip():
        repo_hint = None

    if isinstance(extras, Mapping):
        variant = str(extras.get("flux2_variant") or "").strip().lower()
        is_distilled = extras.get("is_distilled")
        if variant in {"klein", "base"} and isinstance(is_distilled, bool):
            return variant, bool(is_distilled), repo_hint

    candidates: list[str] = []
    for candidate in (
        repo_hint,
        getattr(estimated_config, "repo_id", None),
        getattr(estimated_config, "huggingface_repo", None),
        getattr(bundle, "model_ref", None),
    ):
        if isinstance(candidate, str) and candidate.strip():
            candidates.append(candidate.strip())

    for candidate in candidates:
        lowered = candidate.lower()
        if any(marker in lowered for marker in ("flux.2-klein-base-9b", "flux2-klein-base-9b", "flux.2-klein-9b", "flux2-klein-9b")):
            raise RuntimeError("Unsupported FLUX.2 9B bundle reached the 4B/base-4B runtime lane.")
        if "flux.2-klein-base-4b" in lowered or "flux2-klein-base-4b" in lowered:
            return "base", False, repo_hint or candidate
        if "flux.2-klein-4b" in lowered or "flux2-klein-4b" in lowered:
            return "klein", True, repo_hint or candidate

    raise RuntimeError(
        "Unable to resolve FLUX.2 supported variant from bundle metadata. "
        f"Expected Klein 4B or Klein base-4B, got candidates={candidates!r}."
    )



def assemble_flux2_runtime(
    *,
    spec: Flux2EngineSpec,
    estimated_config: Any,
    codex_components: Mapping[str, object],
    bundle,
    engine_options: Mapping[str, Any] | None = None,
) -> Flux2EngineRuntime:
    options = dict(engine_options or {})
    variant, use_distilled_cfg, repo_hint = _resolve_variant(bundle, estimated_config=estimated_config)

    transformer = codex_components.get("transformer")
    if transformer is None:
        raise ValueError("FLUX.2 runtime assembly requires `transformer` component.")
    if not isinstance(transformer, nn.Module):
        raise TypeError(
            "FLUX.2 runtime assembly requires `transformer` to be a torch.nn.Module; "
            f"got {type(transformer).__name__}."
        )

    vae_model = codex_components.get("vae")
    if vae_model is None:
        vae_path = _single_path(options.get("vae_path"), label="vae") or _metadata_path(bundle, key="vae_path")
        if vae_path is None:
            raise RuntimeError(
                "FLUX.2 runtime assembly is missing a VAE component and no external `vae_path` fallback was provided."
            )
        vae_storage_dtype = memory_management.manager.dtype_for_role(DeviceRole.VAE)
        vae_model = load_flux2_vae(vae_path, dtype=vae_storage_dtype)

    text_encoder_obj = codex_components.get("text_encoder")
    tokenizer = codex_components.get("tokenizer")
    if text_encoder_obj is None:
        tenc_path = (
            _single_path(options.get("tenc_path"), label="text encoder")
            or _metadata_path(bundle, key="tenc_path")
            or _tenc_override_path(bundle)
        )
        if tenc_path is None:
            raise RuntimeError(
                "FLUX.2 runtime assembly is missing a text encoder component and no external `tenc_path` fallback was provided."
            )
        te_storage_dtype = memory_management.manager.dtype_for_role(DeviceRole.TEXT_ENCODER)
        text_encoder_obj = _load_external_text_encoder(tenc_path, torch_dtype=te_storage_dtype)

    if isinstance(text_encoder_obj, Flux2TextEncoder):
        qwen_encoder = text_encoder_obj
    elif isinstance(text_encoder_obj, nn.Module):
        qwen_encoder = Flux2TextEncoder.from_pretrained_model(text_encoder_obj, tokenizer=tokenizer)
    else:
        raise TypeError(
            "FLUX.2 runtime assembly expected a Flux2TextEncoder or Qwen3 module for `text_encoder`; "
            f"got {type(text_encoder_obj).__name__}."
        )
    _apply_tokenizer_hint(qwen_encoder, tokenizer=tokenizer, repo_hint=repo_hint)

    core_native_dtype = _parameter_dtype(transformer)
    core_storage_dtype = getattr(transformer, "storage_dtype", None)
    if not isinstance(core_storage_dtype, (torch.dtype, str)):
        core_storage_dtype = core_native_dtype or memory_management.manager.dtype_for_role(DeviceRole.CORE)
    core_compute_dtype = getattr(transformer, "computation_dtype", None)
    if not isinstance(core_compute_dtype, torch.dtype):
        core_compute_dtype = memory_management.manager.compute_dtype_for_role(
            DeviceRole.CORE,
            supported=(torch.bfloat16, torch.float16, torch.float32),
            storage_dtype=core_storage_dtype if isinstance(core_storage_dtype, torch.dtype) else None,
        )

    mount_device = memory_management.manager.mount_device()
    load_device = getattr(transformer, "load_device", None)
    if not isinstance(load_device, torch.device):
        load_device = _parameter_device(transformer, default=mount_device)

    core = Flux2CoreAdapter(
        transformer=transformer,
        context_dim=int(spec.family_spec().context_dim),
    )
    denoiser = DenoiserPatcher.from_model(
        model=core,
        diffusers_scheduler=None,
        predictor=FlowMatchEulerPrediction(
            seq_len=4096,
            base_seq_len=256,
            max_seq_len=4096,
            base_shift=0.5,
            max_shift=1.15,
            pseudo_timestep_range=1000,
            time_shift_type="exponential",
        ),
        config=estimated_config,
    )

    vae = VAE(model=vae_model, family=ModelFamily.FLUX2)

    qwen_model = qwen_encoder.model
    te_load_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
    te_offload_device = memory_management.manager.get_offload_device(DeviceRole.TEXT_ENCODER)
    qwen_patcher = ModelPatcher(
        qwen_model,
        load_device=te_load_device,
        offload_device=te_offload_device,
    )
    text_engine = Flux2TextProcessingEngine(qwen_encoder, max_length=int(spec.max_length))

    te_native_dtype = _parameter_dtype(qwen_model)
    te_storage_dtype = memory_management.manager.dtype_for_role(DeviceRole.TEXT_ENCODER, native_dtype=te_native_dtype)
    te_compute_dtype = memory_management.manager.compute_dtype_for_role(
        DeviceRole.TEXT_ENCODER,
        storage_dtype=te_storage_dtype,
    )
    vae_native_dtype = _parameter_dtype(vae_model)
    vae_storage_dtype = memory_management.manager.dtype_for_role(DeviceRole.VAE, native_dtype=vae_native_dtype)
    vae_compute_dtype = memory_management.manager.compute_dtype_for_role(
        DeviceRole.VAE,
        storage_dtype=vae_storage_dtype,
    )

    logger.debug(
        "FLUX.2 runtime assembled: variant=%s distilled_cfg=%s repo_hint=%s core_device=%s",
        variant,
        use_distilled_cfg,
        repo_hint,
        load_device,
    )

    return Flux2EngineRuntime(
        vae=vae,
        denoiser=denoiser,
        text=Flux2TextPipelines(qwen3_text=text_engine),
        qwen=qwen_patcher,
        use_distilled_cfg=bool(use_distilled_cfg),
        variant=variant,
        device=load_device,
        core_storage_dtype=(str(core_storage_dtype) if isinstance(core_storage_dtype, str) else _torch_dtype_label(core_storage_dtype)),
        core_compute_dtype=_torch_dtype_label(core_compute_dtype),
        te_storage_dtype=_torch_dtype_label(te_storage_dtype),
        te_compute_dtype=_torch_dtype_label(te_compute_dtype),
        vae_storage_dtype=_torch_dtype_label(vae_storage_dtype),
        vae_compute_dtype=_torch_dtype_label(vae_compute_dtype),
    )


FLUX2_SPEC = Flux2EngineSpec()


__all__ = [
    "Flux2TextPipelines",
    "Flux2EngineRuntime",
    "Flux2EngineSpec",
    "assemble_flux2_runtime",
    "FLUX2_SPEC",
]
