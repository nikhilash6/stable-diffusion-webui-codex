"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed bundle rehydration, native runtime assembly, and run-result contracts for the LTX2 seam.
Rebuilds the loader-produced LTX2 planning contract from a generic diffusion bundle, assembles the dedicated native
runtime from local `apps/**` modules (including optional wrapper-backed transformer-core streaming), enforces the
truthful `euler` / `simple` execution contract, exposes explicit latent-stage / upsample / refine primitives for the
`two_stage` profile, honors side-asset-carried SafeTensors config metadata for the x2 latent upsampler when present,
preflights the `two_stage` distilled-LoRA side asset before any transformer mutation (using the cheap SafeTensors
header path when available), and normalizes execution results into the family-local `frames + AudioExportAsset + metadata`
contract consumed by the canonical video use-cases.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2RunResult` (dataclass): Family-local execution result consumed by canonical video use-cases.
- `Ltx2NativeComponents` (dataclass): Loaded LTX2 runtime components reused across txt2vid/img2vid runs.
- `_parse_ltx2_transformer_lora_specs` (function): Parses the `two_stage` distilled-LoRA side asset into typed transformer patch specs.
- `_preflight_ltx2_transformer_lora_specs` (function): Verifies the `two_stage` distilled-LoRA shapes against the live native transformer before mutation.
- `build_ltx2_request_generator` (function): Builds the shared request-scoped RNG owner reused across LTX2 stage boundaries.
- `sample_ltx2_txt2vid_stage` (function): Execute a native LTX2 txt2vid latent stage and return the typed stage-result contract.
- `sample_ltx2_img2vid_stage` (function): Execute a native LTX2 img2vid latent stage and return the typed stage-result contract.
- `upsample_ltx2_two_stage_video_latents` (function): Run the native x2 latent upsampler for the explicit LTX2 `two_stage` profile.
- `refine_ltx2_txt2vid_two_stage` (function): Execute the stage-2 distilled refinement step for LTX2 txt2vid.
- `refine_ltx2_img2vid_two_stage` (function): Execute the stage-2 distilled refinement step for LTX2 img2vid.
- `decode_ltx2_stage_result` (function): Decode a native latent-stage result into the family-local `Ltx2RunResult`.
- `build_ltx2_run_result` (function): Normalize `frames + audio_asset + metadata` into an immutable LTX2 result object.
- `build_ltx2_native_components` (function): Assemble the loaded native LTX2 runtime components from a typed bundle.
- `run_ltx2_txt2vid` (function): Execute the native LTX2 txt2vid pipeline and normalize the result contract.
- `run_ltx2_img2vid` (function): Execute the native LTX2 img2vid pipeline and normalize the result contract.
- `require_ltx2_bundle_inputs` (function): Rehydrate and validate the loader-produced LTX2 planning contract from a bundle-like object.
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from apps.backend.runtime.adapters.lora.preflight import collect_parameter_shapes
from apps.backend.runtime.checkpoint.io import load_torch_file, read_arbitrary_config, read_safetensors_metadata
from apps.backend.runtime.checkpoint.safetensors_header import read_safetensors_tensor_shapes
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.pipeline_stages.video import (
    AudioExportAsset,
    GeneratedAudioExportPolicy,
    Ltx2TwoStageGeometry,
)

from .audio import is_ltx2_wrapped_vocoder_state, materialize_ltx2_generated_audio_asset
from .model import Ltx2BundleInputs, Ltx2ComponentStates, Ltx2TextEncoderAsset, Ltx2VendorPaths
from .text_encoder import Ltx2TextEncoderRuntime, load_ltx2_text_encoder_runtime

logger = get_backend_logger("backend.runtime.families.ltx2.runtime")
_LTX2_EFFECTIVE_SAMPLER = "euler"
_LTX2_EFFECTIVE_SCHEDULER = "FlowMatchEulerDiscreteScheduler"
_LTX2_ALLOWED_SAMPLERS = frozenset({"", "euler"})
_LTX2_ALLOWED_SCHEDULERS = frozenset({"", "simple"})


@dataclass(frozen=True, slots=True)
class Ltx2RunResult:
    frames: tuple[Any, ...]
    audio_asset: AudioExportAsset | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Ltx2NativeComponents:
    text_encoder: Any
    tokenizer: Any
    connectors: Any
    transformer: Any
    vae: Any
    audio_vae: Any
    vocoder: Any
    scheduler_config: dict[str, Any]
    requested_device_label: str
    device_label: str
    torch_dtype: torch.dtype
    dtype_label: str
    audio_sample_rate_hz: int
    runtime_impl: str
    transformers_version: str
    core_streaming_enabled: bool
    transformer_execution_lock: threading.RLock


@dataclass(frozen=True, slots=True)
class _Ltx2TransformerLoraSpec:
    target_parameter_name: str
    lora_a_key: str
    lora_b_key: str
    alpha: float
    rank: int


def _resolve_ltx2_sampler_contract(request: Any) -> tuple[str | None, str | None, str, str]:
    sampler_requested = str(getattr(request, "sampler", "") or "").strip().lower()
    scheduler_requested = str(getattr(request, "scheduler", "") or "").strip().lower()

    if sampler_requested not in _LTX2_ALLOWED_SAMPLERS:
        raise RuntimeError(
            "LTX2 runtime currently executes on a fixed FlowMatchEulerDiscreteScheduler path. "
            f"Accepted sampler values on this slice are empty or 'euler'; got {getattr(request, 'sampler', None)!r}."
        )
    if scheduler_requested not in _LTX2_ALLOWED_SCHEDULERS:
        raise RuntimeError(
            "LTX2 runtime currently executes on a fixed FlowMatchEulerDiscreteScheduler path. "
            "Accepted scheduler values on this live slice are empty or 'simple'; "
            f"got {getattr(request, 'scheduler', None)!r}."
        )

    return (
        sampler_requested or None,
        scheduler_requested or None,
        _LTX2_EFFECTIVE_SAMPLER,
        _LTX2_EFFECTIVE_SCHEDULER,
    )


def _read_ltx2_execution_extra(source: Any, key: str) -> str | None:
    extras = getattr(source, "extras", None)
    if not isinstance(extras, Mapping):
        return None
    raw_value = extras.get(key)
    normalized = str(raw_value or "").strip()
    return normalized or None


def _denormalize_ltx2_packed_audio_latents(
    latents: torch.Tensor,
    *,
    native: Ltx2NativeComponents,
) -> torch.Tensor:
    return (
        latents * native.audio_vae.latents_std.to(device=latents.device, dtype=latents.dtype)
    ) + native.audio_vae.latents_mean.to(device=latents.device, dtype=latents.dtype)


@contextmanager
def _ltx2_transformer_execution_context(native: Ltx2NativeComponents):
    native.transformer_execution_lock.acquire()
    try:
        yield
    finally:
        native.transformer_execution_lock.release()


def build_ltx2_run_result(
    *,
    frames: Sequence[Any],
    audio_asset: AudioExportAsset | None,
    metadata: Mapping[str, Any] | None = None,
) -> Ltx2RunResult:
    frames_tuple = tuple(frames)
    if not frames_tuple:
        raise RuntimeError("LTX2 run result requires at least one frame.")
    return Ltx2RunResult(
        frames=frames_tuple,
        audio_asset=audio_asset,
        metadata=dict(metadata or {}),
    )


def _as_torch_dtype(dtype_label: str) -> torch.dtype:
    normalized = str(dtype_label or "").strip().lower()
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise RuntimeError(f"LTX2 runtime dtype must be one of fp16|bf16|fp32, got {dtype_label!r}.")


def _resolve_device_name(device_label: str) -> torch.device:
    normalized = str(device_label or "").strip().lower()
    manager = getattr(memory_management, "manager", None)
    if manager is None or not hasattr(manager, "mount_device"):
        raise RuntimeError("LTX2 runtime requires an active memory manager with mount_device().")
    mount_device = manager.mount_device()
    if not isinstance(mount_device, torch.device):
        raise RuntimeError(
            "LTX2 runtime expected memory manager mount_device() -> torch.device, "
            f"got {type(mount_device).__name__}."
        )
    if normalized in {"auto", ""}:
        if mount_device.type in {"cpu", "cuda"}:
            return mount_device
        raise RuntimeError(
            "LTX2 runtime auto device requires a cpu/cuda mount device; "
            f"got {mount_device!s}."
        )
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not bool(getattr(manager.hardware_probe, "cuda_available", False)):
            raise RuntimeError(
                "LTX2 runtime requested device='cuda', but CUDA is unavailable in the memory-manager probe."
            )
        return torch.device("cuda")
    raise RuntimeError(f"LTX2 runtime device must be one of auto|cpu|cuda, got {device_label!r}.")


def _import_native_ltx2_runtime_symbols() -> dict[str, Any]:
    try:
        from apps.backend.runtime.families.ltx2.native import (
            Ltx2AudioAutoencoder,
            Ltx2LatentUpsamplerModel,
            Ltx2NativeLatentStageResult,
            Ltx2TextConnectors,
            Ltx2VideoAutoencoder,
            Ltx2VideoTransformer3DModel,
            Ltx2Vocoder,
            decode_ltx2_native_stage_result,
            load_ltx2_connectors,
            load_ltx2_vocoder,
            run_ltx2_img2vid_native,
            run_ltx2_txt2vid_native,
            sample_ltx2_img2vid_native,
            sample_ltx2_txt2vid_native,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "LTX2 native runtime modules are unavailable. "
            "The active LTX2 slice requires local model/scheduler/pipeline execution under "
            "`apps/backend/runtime/families/ltx2/native/**`."
        ) from exc

    try:
        import transformers
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("LTX2 runtime requires `transformers==4.57.3` with Gemma3 support.") from exc

    return {
        "Ltx2AudioAutoencoder": Ltx2AudioAutoencoder,
        "Ltx2LatentUpsamplerModel": Ltx2LatentUpsamplerModel,
        "Ltx2NativeLatentStageResult": Ltx2NativeLatentStageResult,
        "Ltx2TextConnectors": Ltx2TextConnectors,
        "Ltx2VideoAutoencoder": Ltx2VideoAutoencoder,
        "Ltx2VideoTransformer3DModel": Ltx2VideoTransformer3DModel,
        "Ltx2Vocoder": Ltx2Vocoder,
        "decode_ltx2_native_stage_result": decode_ltx2_native_stage_result,
        "load_ltx2_connectors": load_ltx2_connectors,
        "load_ltx2_vocoder": load_ltx2_vocoder,
        "run_ltx2_img2vid_native": run_ltx2_img2vid_native,
        "run_ltx2_txt2vid_native": run_ltx2_txt2vid_native,
        "sample_ltx2_img2vid_native": sample_ltx2_img2vid_native,
        "sample_ltx2_txt2vid_native": sample_ltx2_txt2vid_native,
        "transformers_version": getattr(transformers, "__version__", "unknown"),
    }


def _read_component_config(repo_dir: Path, component_name: str) -> dict[str, Any]:
    component_dir = repo_dir / component_name
    if not component_dir.is_dir():
        raise RuntimeError(f"LTX2 vendored component directory not found: {component_dir}")
    return read_arbitrary_config(str(component_dir))


def _finalize_loaded_module(
    *,
    module: Any,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> Any:
    try:
        module = module.to(device=device, dtype=torch_dtype)
    except Exception:
        module = module.to(device=device)
    module.eval()
    return module


def _load_native_component_module(
    *,
    label: str,
    module_cls: Any,
    config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    device: torch.device,
    torch_dtype: torch.dtype,
) -> Any:
    try:
        module = module_cls.from_config(dict(config))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 {label} config instantiation failed: {exc}") from exc

    load_strict_state_dict = getattr(module, "load_strict_state_dict", None)
    if not callable(load_strict_state_dict):
        raise RuntimeError(
            f"LTX2 {label} native module {module_cls.__name__} must implement load_strict_state_dict()."
        )

    try:
        load_strict_state_dict(state_dict)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 {label} state load failed: {exc}") from exc

    return _finalize_loaded_module(module=module, device=device, torch_dtype=torch_dtype)


def _load_native_component_via_loader(
    *,
    label: str,
    loader_fn: Any,
    config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    device: torch.device,
    torch_dtype: torch.dtype,
) -> Any:
    try:
        return loader_fn(
            config=dict(config),
            state_dict=state_dict,
            device=device,
            torch_dtype=torch_dtype,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 {label} state load failed: {exc}") from exc


def build_ltx2_native_components(
    *,
    bundle_inputs: Ltx2BundleInputs,
    device: str,
    dtype: str,
    engine_options: Mapping[str, Any] | None = None,
) -> Ltx2NativeComponents:
    from .streaming import Ltx2StreamingConfig, wrap_for_streaming

    streaming_config = Ltx2StreamingConfig.from_options(engine_options)
    symbols = _import_native_ltx2_runtime_symbols()
    resolved_device = _resolve_device_name(device)
    torch_dtype = _as_torch_dtype(dtype)
    dtype_label = str(dtype).strip().lower()
    if resolved_device.type == "cpu" and torch_dtype != torch.float32:
        emit_backend_message(
            "[ltx2] forcing fp32 on CPU",
            logger=logger.name,
            requested_dtype=dtype_label,
            resolved_device=resolved_device,
        )
        torch_dtype = torch.float32
        dtype_label = "fp32"

    repo_dir = Path(bundle_inputs.vendor_paths.repo_dir)
    text_runtime: Ltx2TextEncoderRuntime = load_ltx2_text_encoder_runtime(
        asset=bundle_inputs.text_encoder,
        vendor_paths=bundle_inputs.vendor_paths,
        device=resolved_device,
        torch_dtype=torch_dtype,
    )

    scheduler_config = _read_component_config(repo_dir, "scheduler")
    connectors = _load_native_component_via_loader(
        label="connectors",
        loader_fn=symbols["load_ltx2_connectors"],
        config=_read_component_config(repo_dir, "connectors"),
        state_dict=bundle_inputs.components.connectors,
        device=resolved_device,
        torch_dtype=torch_dtype,
    )
    transformer = _load_native_component_module(
        label="transformer",
        module_cls=symbols["Ltx2VideoTransformer3DModel"],
        config=_read_component_config(repo_dir, "transformer"),
        state_dict=bundle_inputs.components.transformer,
        device=resolved_device,
        torch_dtype=torch_dtype,
    )
    if streaming_config.enabled:
        emit_backend_message(
            "[ltx2] enabling transformer-core streaming",
            logger=logger.name,
            policy=streaming_config.policy,
            blocks_per_segment=streaming_config.blocks_per_segment,
            window_size=streaming_config.window_size,
        )
        transformer = wrap_for_streaming(
            transformer,
            policy=streaming_config.policy,
            blocks_per_segment=streaming_config.blocks_per_segment,
            window_size=streaming_config.window_size,
        )
    vae = _load_native_component_module(
        label="video_vae",
        module_cls=symbols["Ltx2VideoAutoencoder"],
        config=_read_component_config(repo_dir, "vae"),
        state_dict=bundle_inputs.components.vae,
        device=resolved_device,
        torch_dtype=torch_dtype,
    )
    audio_vae = _load_native_component_module(
        label="audio_vae",
        module_cls=symbols["Ltx2AudioAutoencoder"],
        config=_read_component_config(repo_dir, "audio_vae"),
        state_dict=bundle_inputs.components.audio_vae,
        device=resolved_device,
        torch_dtype=torch_dtype,
    )
    vocoder_config = bundle_inputs.vocoder_config
    if is_ltx2_wrapped_vocoder_state(bundle_inputs.components.vocoder):
        if vocoder_config is None:
            raise RuntimeError(
                "LTX2 wrapped vocoder assembly requires metadata-carried `vocoder_config`. "
                "The bundle lost the real audio-bundle wrapper config."
            )
    else:
        if vocoder_config is None:
            vocoder_config = _read_component_config(repo_dir, "vocoder")
    vocoder = _load_native_component_via_loader(
        label="vocoder",
        loader_fn=symbols["load_ltx2_vocoder"],
        config=vocoder_config,
        state_dict=bundle_inputs.components.vocoder,
        device=resolved_device,
        torch_dtype=torch_dtype,
    )

    audio_sample_rate_hz = int(getattr(getattr(vocoder, "config", None), "output_sampling_rate", 24000) or 24000)
    return Ltx2NativeComponents(
        text_encoder=text_runtime.model,
        tokenizer=text_runtime.tokenizer,
        connectors=connectors,
        transformer=transformer,
        vae=vae,
        audio_vae=audio_vae,
        vocoder=vocoder,
        scheduler_config=dict(scheduler_config),
        requested_device_label=str(device).strip().lower() or "auto",
        device_label=str(resolved_device),
        torch_dtype=torch_dtype,
        dtype_label=dtype_label,
        audio_sample_rate_hz=audio_sample_rate_hz,
        runtime_impl="native",
        transformers_version=str(symbols["transformers_version"]),
        core_streaming_enabled=streaming_config.enabled,
        transformer_execution_lock=threading.RLock(),
    )


def _normalize_video_frames(video: Any) -> tuple[Image.Image, ...]:
    array: Any = video
    if isinstance(array, (list, tuple)):
        if len(array) != 1:
            raise RuntimeError(
                "LTX2 runtime expects single-batch video output for canonical use-cases; "
                f"got batch={len(array)!r}."
            )
        array = array[0]
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.ndim == 5:
        if int(array.shape[0]) != 1:
            raise RuntimeError(
                "LTX2 runtime expects single-batch video tensor output; "
                f"got shape={tuple(int(dim) for dim in array.shape)!r}."
            )
        array = array[0]
    if array.ndim != 4:
        raise RuntimeError(
            "LTX2 video output must be 4D after batch normalization "
            f"(frames,height,width,channels or frames,channels,height,width); got {array.ndim}D."
        )
    if array.shape[-1] in {1, 3, 4}:
        frames_array = array
    elif array.shape[1] in {1, 3, 4}:
        frames_array = np.transpose(array, (0, 2, 3, 1))
    else:
        raise RuntimeError(
            "LTX2 video output channel layout is unsupported; "
            f"got shape={tuple(int(dim) for dim in array.shape)!r}."
        )

    if np.issubdtype(frames_array.dtype, np.floating):
        frames_array = np.clip(frames_array, 0.0, 1.0)
        frames_array = (frames_array * 255.0).round().astype(np.uint8)
    elif frames_array.dtype != np.uint8:
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)

    frames: list[Image.Image] = []
    for index, frame in enumerate(frames_array):
        try:
            frames.append(Image.fromarray(frame))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LTX2 failed to normalize frame {index} into a PIL image: {exc}") from exc
    if not frames:
        raise RuntimeError("LTX2 runtime produced zero video frames.")
    return tuple(frames)


def _coerce_native_media_outputs(outputs: Any, *, mode_label: str) -> tuple[Any, Any]:
    if isinstance(outputs, tuple):
        if len(outputs) < 2:
            raise RuntimeError(
                f"LTX2 {mode_label} native pipeline must return `(video, audio)` or an object with `video` and `audio`; "
                f"got tuple length={len(outputs)}."
            )
        return outputs[0], outputs[1]

    video = getattr(outputs, "video", None)
    audio = getattr(outputs, "audio", None)
    if video is None and hasattr(outputs, "videos"):
        video = getattr(outputs, "videos")
    if video is None:
        raise RuntimeError(
            f"LTX2 {mode_label} native pipeline output is unsupported: "
            f"expected `(video, audio)` tuple or object with `video`/`audio`, got {type(outputs).__name__}."
        )
    return video, audio


def _build_pipeline_metadata(
    *,
    native: Ltx2NativeComponents,
    pipeline_name: str,
    request: Any,
    plan: Any,
    frame_count: int,
    has_audio: bool,
    sampler_requested: str | None,
    scheduler_requested: str | None,
    sampler_effective: str,
    scheduler_effective: str,
) -> dict[str, Any]:
    return {
        "pipeline": pipeline_name,
        "requested_device": native.requested_device_label,
        "effective_device": native.device_label,
        "dtype": native.dtype_label,
        "audio_sample_rate_hz": native.audio_sample_rate_hz,
        "frame_count": int(frame_count),
        "fps": int(getattr(plan, "fps", 0) or 0),
        "steps": int(getattr(plan, "steps", 0) or 0),
        "has_audio": bool(has_audio),
        "sampler_requested": sampler_requested,
        "scheduler_requested": scheduler_requested,
        "sampler": sampler_effective,
        "scheduler": scheduler_effective,
        "sampler_effective": sampler_effective,
        "scheduler_effective": scheduler_effective,
        "ltx_checkpoint_kind": _read_ltx2_execution_extra(request, "ltx_checkpoint_kind"),
        "ltx_execution_profile": _read_ltx2_execution_extra(request, "ltx_execution_profile"),
        "runtime_impl": native.runtime_impl,
        "transformers_version": native.transformers_version,
    }


def _coerce_native_stage_result(stage_result: Any, *, mode_label: str) -> Any:
    stage_result_type = _import_native_ltx2_runtime_symbols()["Ltx2NativeLatentStageResult"]
    if not isinstance(stage_result, stage_result_type):
        raise RuntimeError(
            f"LTX2 {mode_label} native sampler must return `Ltx2NativeLatentStageResult`; "
            f"got {type(stage_result).__name__}."
        )
    return stage_result


def build_ltx2_request_generator(*, native: Ltx2NativeComponents, request: Any) -> torch.Generator | None:
    seed = getattr(request, "seed", None)
    resolved_device = torch.device(native.device_label)
    generator_device = resolved_device.type if resolved_device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    if seed is None:
        generator.seed()
        return generator
    seed_value = int(seed)
    if seed_value < 0:
        raise RuntimeError(f"LTX2 request seed must be >= 0 or None at runtime, got {seed_value}.")
    generator.manual_seed(seed_value)
    return generator


def _require_ltx2_path_extra(request: Any, key: str, *, label: str) -> str:
    value = _read_ltx2_execution_extra(request, key)
    if value is None:
        raise RuntimeError(
            f"LTX2 runtime is missing request extras[{key!r}] for {label}. "
            "This path must be resolved by the router-owned checkpoint admissibility seam."
        )
    return value


def _parse_ltx2_default_lora_alpha(path: str) -> float | None:
    if not Path(path).is_file():
        return None
    suffix = Path(path).suffix.lower()
    if suffix not in {".safetensor", ".safetensors"}:
        return None
    metadata = read_safetensors_metadata(path)
    for key in ("lora_alpha", "ss_network_alpha"):
        raw_value = str(metadata.get(key) or "").strip()
        if not raw_value:
            continue
        try:
            return float(raw_value)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"LTX2 distilled LoRA metadata field {key!r} must be a finite float string; got {raw_value!r} in {path!r}."
            ) from exc
    return None


def _read_ltx2_side_asset_config_from_metadata(
    side_asset_path: str,
    *,
    label: str,
) -> Mapping[str, Any] | None:
    suffix = Path(side_asset_path).suffix.lower()
    if suffix not in {".safetensor", ".safetensors"}:
        return None
    metadata = read_safetensors_metadata(side_asset_path)
    raw_config_json = str(metadata.get("config") or "").strip()
    if not raw_config_json:
        return None
    try:
        payload = json.loads(raw_config_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LTX2 {label} metadata contains invalid JSON in the `config` field: "
            f"path={side_asset_path!r} error={exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"LTX2 {label} metadata `config` must decode to a mapping. "
            f"Got {type(payload).__name__} from {side_asset_path!r}."
        )
    return dict(payload)


def _strip_ltx2_lora_target_prefix(target: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if target.startswith(prefix):
            return target[len(prefix) :]
    return target


def _shape_tuple_like(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", value)
    if not isinstance(shape, (tuple, list)):
        raise RuntimeError(f"Expected shape-like value, got {type(value).__name__}.")
    return tuple(int(dim) for dim in shape)


def _parse_ltx2_transformer_lora_specs(
    state_dict: Mapping[str, object],
    *,
    source_path: str,
    shape_only: bool = False,
) -> tuple[_Ltx2TransformerLoraSpec, ...]:
    default_alpha = _parse_ltx2_default_lora_alpha(source_path)
    grouped: dict[str, dict[str, str]] = {}
    unexpected: list[str] = []
    alpha_keys: list[str] = []
    for raw_key in state_dict.keys():
        key = str(raw_key)
        if key.endswith(".lora_A.weight"):
            grouped.setdefault(key[: -len(".lora_A.weight")], {})["a"] = key
            continue
        if key.endswith(".lora_B.weight"):
            grouped.setdefault(key[: -len(".lora_B.weight")], {})["b"] = key
            continue
        if key.endswith(".alpha") or key.endswith(".lora_alpha"):
            alpha_keys.append(key)
            continue
        unexpected.append(key)

    if unexpected:
        raise RuntimeError(
            "LTX2 two_stage distilled LoRA includes unsupported tensor keys. "
            f"Expected only standard `.lora_A.weight` / `.lora_B.weight` pairs; got sample={unexpected[:20]!r}."
        )

    specs: list[_Ltx2TransformerLoraSpec] = []
    for prefix, pair in sorted(grouped.items()):
        lora_a_key = pair.get("a")
        lora_b_key = pair.get("b")
        if lora_a_key is None or lora_b_key is None:
            raise RuntimeError(
                "LTX2 two_stage distilled LoRA contains an incomplete A/B pair for "
                f"{prefix!r} in {source_path!r}."
            )
        lora_a = state_dict[lora_a_key]
        lora_b = state_dict[lora_b_key]
        if not shape_only and (not isinstance(lora_a, torch.Tensor) or not isinstance(lora_b, torch.Tensor)):
            raise RuntimeError(
                "LTX2 two_stage distilled LoRA expects tensor A/B weights; "
                f"got A={type(lora_a).__name__} B={type(lora_b).__name__} for {prefix!r}."
            )
        rank = int(_shape_tuple_like(lora_a)[0])
        if rank <= 0:
            raise RuntimeError(f"LTX2 two_stage distilled LoRA rank must be > 0 for {prefix!r}.")
        alpha_value: float | None = None
        for alpha_key in (f"{prefix}.alpha", f"{prefix}.lora_alpha"):
            if alpha_key not in state_dict:
                continue
            alpha_tensor = state_dict[alpha_key]
            if isinstance(alpha_tensor, torch.Tensor):
                if alpha_tensor.numel() != 1:
                    raise RuntimeError(
                        f"LTX2 two_stage distilled LoRA alpha tensor must be scalar for {prefix!r}; got shape={tuple(alpha_tensor.shape)!r}."
                    )
                alpha_value = float(alpha_tensor.item())
            elif shape_only:
                alpha_value = None
            else:
                alpha_value = float(alpha_tensor)  # type: ignore[arg-type]
            break
        if alpha_value is None:
            alpha_value = float(default_alpha if default_alpha is not None else rank)
        specs.append(
            _Ltx2TransformerLoraSpec(
                target_parameter_name=f"{_strip_ltx2_lora_target_prefix(prefix)}.weight",
                lora_a_key=lora_a_key,
                lora_b_key=lora_b_key,
                alpha=float(alpha_value),
                rank=rank,
            )
        )
    if not specs:
        raise RuntimeError(f"LTX2 two_stage distilled LoRA produced zero valid A/B pairs from {source_path!r}.")
    return tuple(specs)


def _preflight_ltx2_transformer_lora_specs(
    *,
    transformer: Any,
    weights: Mapping[str, object],
    specs: Sequence[_Ltx2TransformerLoraSpec],
    source_path: str,
) -> None:
    target_shapes = collect_parameter_shapes(_resolve_ltx2_transformer_lora_owner(transformer))
    for spec in specs:
        target_shape = target_shapes.get(spec.target_parameter_name)
        if target_shape is None:
            raise RuntimeError(
                f"LTX2 two_stage distilled LoRA target {spec.target_parameter_name!r} does not exist on the native transformer."
            )
        if len(target_shape) < 2:
            raise RuntimeError(
                f"LTX2 two_stage distilled LoRA target {spec.target_parameter_name!r} must have ndim >= 2; got {target_shape!r}."
            )
        a_shape = _shape_tuple_like(weights[spec.lora_a_key])
        b_shape = _shape_tuple_like(weights[spec.lora_b_key])
        a_matrix = (int(a_shape[0]), int(math.prod(a_shape[1:])))
        b_matrix = (int(math.prod(b_shape[:-1])), int(b_shape[-1]))
        if int(a_matrix[0]) != int(spec.rank) or int(b_matrix[1]) != int(spec.rank):
            raise RuntimeError(
                f"LTX2 two_stage distilled LoRA rank mismatch for target {spec.target_parameter_name!r}: "
                f"expected rank={spec.rank}, got A={a_shape!r} B={b_shape!r} in {source_path!r}."
            )
        target_matrix = (int(target_shape[0]), int(math.prod(target_shape[1:])))
        if int(a_matrix[1]) != int(target_matrix[1]) or int(b_matrix[0]) != int(target_matrix[0]):
            raise RuntimeError(
                f"LTX2 two_stage distilled LoRA target shape mismatch for {spec.target_parameter_name!r}: "
                f"target={target_shape!r} A={a_shape!r} B={b_shape!r} file={source_path!r}."
            )


def _materialize_ltx2_lora_delta(
    *,
    parameter: torch.nn.Parameter,
    state_dict: Mapping[str, object],
    spec: _Ltx2TransformerLoraSpec,
) -> torch.Tensor:
    if parameter.ndim < 2:
        raise RuntimeError(
            f"LTX2 two_stage distilled LoRA target {spec.target_parameter_name!r} must have ndim >= 2; got {parameter.ndim}."
        )
    lora_a = state_dict[spec.lora_a_key]
    lora_b = state_dict[spec.lora_b_key]
    if not isinstance(lora_a, torch.Tensor) or not isinstance(lora_b, torch.Tensor):
        raise RuntimeError(
            f"LTX2 two_stage distilled LoRA target {spec.target_parameter_name!r} resolved non-tensor A/B weights."
        )
    a_matrix = lora_a.to(device=parameter.device, dtype=torch.float32).reshape(int(lora_a.shape[0]), -1)
    b_matrix = lora_b.to(device=parameter.device, dtype=torch.float32).reshape(-1, int(lora_b.shape[-1]))
    if int(a_matrix.shape[0]) != int(spec.rank) or int(b_matrix.shape[1]) != int(spec.rank):
        raise RuntimeError(
            f"LTX2 two_stage distilled LoRA rank mismatch for target {spec.target_parameter_name!r}: "
            f"expected rank={spec.rank}, got A={tuple(int(dim) for dim in a_matrix.shape)!r} "
            f"B={tuple(int(dim) for dim in b_matrix.shape)!r}."
        )
    target_matrix = parameter.data.reshape(int(parameter.shape[0]), -1)
    if int(a_matrix.shape[1]) != int(target_matrix.shape[1]) or int(b_matrix.shape[0]) != int(target_matrix.shape[0]):
        raise RuntimeError(
            f"LTX2 two_stage distilled LoRA target shape mismatch for {spec.target_parameter_name!r}: "
            f"target={tuple(int(dim) for dim in parameter.shape)!r} "
            f"A={tuple(int(dim) for dim in lora_a.shape)!r} "
            f"B={tuple(int(dim) for dim in lora_b.shape)!r}."
        )
    scale = float(spec.alpha) / float(spec.rank)
    delta = torch.matmul(b_matrix, a_matrix) * scale
    return delta.reshape(parameter.shape)


def _resolve_ltx2_transformer_lora_owner(transformer: Any) -> Any:
    owner = transformer
    visited: set[int] = set()
    while True:
        current_id = id(owner)
        if current_id in visited:
            break
        visited.add(current_id)
        base_model = getattr(owner, "base_model", None)
        if base_model is None or base_model is owner:
            break
        owner = base_model
    return owner


def _apply_ltx2_transformer_lora_specs(
    *,
    transformer: Any,
    state_dict: Mapping[str, object],
    specs: Sequence[_Ltx2TransformerLoraSpec],
    sign: float,
) -> None:
    target_owner = _resolve_ltx2_transformer_lora_owner(transformer)
    parameters = dict(target_owner.named_parameters())
    for spec in specs:
        parameter = parameters.get(spec.target_parameter_name)
        if parameter is None:
            raise RuntimeError(
                f"LTX2 two_stage distilled LoRA target {spec.target_parameter_name!r} does not exist on the native transformer."
            )
        delta = _materialize_ltx2_lora_delta(parameter=parameter, state_dict=state_dict, spec=spec)
        parameter.data.add_(delta.to(dtype=parameter.dtype), alpha=float(sign))


@contextmanager
def _temporary_ltx2_two_stage_transformer(*, request: Any, native: Ltx2NativeComponents):
    lora_path = _require_ltx2_path_extra(
        request,
        "ltx_two_stage_distilled_lora_path",
        label="LTX2 two_stage distilled LoRA",
    )
    if Path(lora_path).suffix.lower() in {".safetensor", ".safetensors"}:
        header_shapes = read_safetensors_tensor_shapes(Path(lora_path))
        header_specs = _parse_ltx2_transformer_lora_specs(
            header_shapes,
            source_path=lora_path,
            shape_only=True,
        )
        _preflight_ltx2_transformer_lora_specs(
            transformer=native.transformer,
            weights=header_shapes,
            specs=header_specs,
            source_path=lora_path,
        )
    state_dict = load_torch_file(lora_path, device="cpu")
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(
            f"LTX2 two_stage distilled LoRA must resolve to a mapping state_dict, got {type(state_dict).__name__}: {lora_path!r}."
        )
    specs = _parse_ltx2_transformer_lora_specs(state_dict, source_path=lora_path)
    _preflight_ltx2_transformer_lora_specs(
        transformer=native.transformer,
        weights=state_dict,
        specs=specs,
        source_path=lora_path,
    )
    applied: list[_Ltx2TransformerLoraSpec] = []
    try:
        for spec in specs:
            _apply_ltx2_transformer_lora_specs(
                transformer=native.transformer,
                state_dict=state_dict,
                specs=(spec,),
                sign=1.0,
            )
            applied.append(spec)
    except Exception:
        for spec in reversed(applied):
            _apply_ltx2_transformer_lora_specs(
                transformer=native.transformer,
                state_dict=state_dict,
                specs=(spec,),
                sign=-1.0,
            )
        raise
    try:
        yield native.transformer
    finally:
        for spec in reversed(applied):
            _apply_ltx2_transformer_lora_specs(
                transformer=native.transformer,
                state_dict=state_dict,
                specs=(spec,),
                sign=-1.0,
            )


def _load_ltx2_two_stage_spatial_upsampler(
    *,
    bundle_inputs: Ltx2BundleInputs,
    native: Ltx2NativeComponents,
    request: Any,
) -> Any:
    repo_dir = Path(bundle_inputs.vendor_paths.repo_dir)
    side_asset_path = _require_ltx2_path_extra(
        request,
        "ltx_two_stage_spatial_upsampler_path",
        label="LTX2 two_stage spatial upsampler",
    )
    vendored_config = _read_component_config(repo_dir, "latent_upsampler")
    metadata_config = _read_ltx2_side_asset_config_from_metadata(
        side_asset_path,
        label="two_stage spatial upsampler",
    )
    config = metadata_config or vendored_config
    state_dict = load_torch_file(
        side_asset_path,
        device="cpu",
    )
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("LTX2 two_stage spatial upsampler must resolve to a mapping state_dict.")
    symbols = _import_native_ltx2_runtime_symbols()
    return _load_native_component_module(
        label="latent_upsampler",
        module_cls=symbols["Ltx2LatentUpsamplerModel"],
        config=config,
        state_dict=state_dict,
        device=torch.device(native.device_label),
        torch_dtype=native.torch_dtype,
    )


def sample_ltx2_txt2vid_stage(
    *,
    native: Ltx2NativeComponents,
    request: Any,
    plan: Any,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    noise_scale: float = 0.0,
    latents: torch.Tensor | None = None,
    audio_latents: torch.Tensor | None = None,
    sigmas: Sequence[float] | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    with _ltx2_transformer_execution_context(native):
        stage_result = _import_native_ltx2_runtime_symbols()["sample_ltx2_txt2vid_native"](
            native=native,
            request=request,
            plan=plan,
            width=int(width),
            height=int(height),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            noise_scale=float(noise_scale),
            latents=latents,
            audio_latents=audio_latents,
            sigmas=sigmas,
            generator=generator,
        )
    return _coerce_native_stage_result(stage_result, mode_label="txt2vid")


def sample_ltx2_img2vid_stage(
    *,
    native: Ltx2NativeComponents,
    request: Any,
    plan: Any,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    noise_scale: float = 0.0,
    latents: torch.Tensor | None = None,
    audio_latents: torch.Tensor | None = None,
    sigmas: Sequence[float] | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    with _ltx2_transformer_execution_context(native):
        stage_result = _import_native_ltx2_runtime_symbols()["sample_ltx2_img2vid_native"](
            native=native,
            request=request,
            plan=plan,
            width=int(width),
            height=int(height),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            noise_scale=float(noise_scale),
            latents=latents,
            audio_latents=audio_latents,
            sigmas=sigmas,
            generator=generator,
        )
    return _coerce_native_stage_result(stage_result, mode_label="img2vid")


def upsample_ltx2_two_stage_video_latents(
    *,
    bundle_inputs: Ltx2BundleInputs,
    native: Ltx2NativeComponents,
    request: Any,
    stage_result: Any,
    geometry: Ltx2TwoStageGeometry,
) -> torch.Tensor:
    upsampler = _load_ltx2_two_stage_spatial_upsampler(
        bundle_inputs=bundle_inputs,
        native=native,
        request=request,
    )
    try:
        upsampled = upsampler(stage_result.video_latents_unpacked_unnormalized.to(dtype=native.torch_dtype))
    finally:
        del upsampler
        memory_management.manager.soft_empty_cache()

    spatial_ratio = int(getattr(native.vae, "spatial_compression_ratio", 32) or 32)
    expected_height = int(geometry.final_height) // spatial_ratio
    expected_width = int(geometry.final_width) // spatial_ratio
    actual_shape = tuple(int(dim) for dim in upsampled.shape)
    if actual_shape[-2:] != (expected_height, expected_width):
        raise RuntimeError(
            "LTX2 two_stage latent upsample produced the wrong final latent geometry: "
            f"expected (*, {expected_height}, {expected_width}) got {actual_shape!r}."
        )
    return upsampled


def refine_ltx2_txt2vid_two_stage(
    *,
    native: Ltx2NativeComponents,
    request: Any,
    plan: Any,
    geometry: Ltx2TwoStageGeometry,
    upscaled_video_latents: torch.Tensor,
    stage1_result: Any,
    generator: torch.Generator | None = None,
) -> Any:
    with _ltx2_transformer_execution_context(native):
        with _temporary_ltx2_two_stage_transformer(request=request, native=native):
            return sample_ltx2_txt2vid_stage(
                native=native,
                request=request,
                plan=plan,
                width=geometry.final_width,
                height=geometry.final_height,
                num_inference_steps=len(geometry.stage2_sigmas),
                guidance_scale=geometry.stage2_guidance_scale,
                noise_scale=geometry.stage2_noise_scale,
                latents=upscaled_video_latents,
                audio_latents=_denormalize_ltx2_packed_audio_latents(
                    stage1_result.audio_latents_packed_normalized,
                    native=native,
                ),
                sigmas=geometry.stage2_sigmas,
                generator=generator,
            )


def refine_ltx2_img2vid_two_stage(
    *,
    native: Ltx2NativeComponents,
    request: Any,
    plan: Any,
    geometry: Ltx2TwoStageGeometry,
    upscaled_video_latents: torch.Tensor,
    stage1_result: Any,
    generator: torch.Generator | None = None,
) -> Any:
    with _ltx2_transformer_execution_context(native):
        with _temporary_ltx2_two_stage_transformer(request=request, native=native):
            return sample_ltx2_img2vid_stage(
                native=native,
                request=request,
                plan=plan,
                width=geometry.final_width,
                height=geometry.final_height,
                num_inference_steps=len(geometry.stage2_sigmas),
                guidance_scale=geometry.stage2_guidance_scale,
                noise_scale=geometry.stage2_noise_scale,
                latents=upscaled_video_latents,
                audio_latents=_denormalize_ltx2_packed_audio_latents(
                    stage1_result.audio_latents_packed_normalized,
                    native=native,
                ),
                sigmas=geometry.stage2_sigmas,
                generator=generator,
            )


def decode_ltx2_stage_result(
    *,
    native: Ltx2NativeComponents,
    request: Any,
    plan: Any,
    stage_result: Any,
    generated_audio_export_policy: GeneratedAudioExportPolicy,
    pipeline_name: str,
    metadata_extra: Mapping[str, Any] | None = None,
) -> Ltx2RunResult:
    sampler_requested, scheduler_requested, sampler_effective, scheduler_effective = _resolve_ltx2_sampler_contract(
        request
    )
    outputs = _import_native_ltx2_runtime_symbols()["decode_ltx2_native_stage_result"](
        native=native,
        stage_result=stage_result,
    )
    video, audio = _coerce_native_media_outputs(outputs, mode_label=pipeline_name)
    frames = _normalize_video_frames(video)
    audio_asset = None
    if generated_audio_export_policy.materialize_audio_asset:
        audio_asset = materialize_ltx2_generated_audio_asset(
            audio,
            sample_rate_hz=native.audio_sample_rate_hz,
        )
    metadata = _build_pipeline_metadata(
        native=native,
        pipeline_name=pipeline_name,
        request=request,
        plan=plan,
        frame_count=len(frames),
        has_audio=audio_asset is not None,
        sampler_requested=sampler_requested,
        scheduler_requested=scheduler_requested,
        sampler_effective=sampler_effective,
        scheduler_effective=scheduler_effective,
    )
    if metadata_extra:
        metadata.update(dict(metadata_extra))
    return build_ltx2_run_result(frames=frames, audio_asset=audio_asset, metadata=metadata)


def run_ltx2_txt2vid(
    *,
    native: Ltx2NativeComponents,
    request: Any,
    plan: Any,
    generated_audio_export_policy: GeneratedAudioExportPolicy,
) -> Ltx2RunResult:
    sampler_requested, scheduler_requested, sampler_effective, scheduler_effective = _resolve_ltx2_sampler_contract(
        request
    )
    generator = build_ltx2_request_generator(native=native, request=request)
    with _ltx2_transformer_execution_context(native):
        outputs = _import_native_ltx2_runtime_symbols()["run_ltx2_txt2vid_native"](
            native=native,
            request=request,
            plan=plan,
            generator=generator,
        )
    video, audio = _coerce_native_media_outputs(outputs, mode_label="txt2vid")
    frames = _normalize_video_frames(video)
    audio_asset = None
    if generated_audio_export_policy.materialize_audio_asset:
        audio_asset = materialize_ltx2_generated_audio_asset(
            audio,
            sample_rate_hz=native.audio_sample_rate_hz,
        )
    metadata = _build_pipeline_metadata(
        native=native,
        pipeline_name="ltx2_native_txt2vid",
        request=request,
        plan=plan,
        frame_count=len(frames),
        has_audio=audio_asset is not None,
        sampler_requested=sampler_requested,
        scheduler_requested=scheduler_requested,
        sampler_effective=sampler_effective,
        scheduler_effective=scheduler_effective,
    )
    return build_ltx2_run_result(frames=frames, audio_asset=audio_asset, metadata=metadata)


def run_ltx2_img2vid(
    *,
    native: Ltx2NativeComponents,
    request: Any,
    plan: Any,
    generated_audio_export_policy: GeneratedAudioExportPolicy,
) -> Ltx2RunResult:
    init_image = getattr(request, "init_image", None)
    if init_image is None:
        raise RuntimeError("LTX2 img2vid requires `request.init_image`.")

    sampler_requested, scheduler_requested, sampler_effective, scheduler_effective = _resolve_ltx2_sampler_contract(
        request
    )
    generator = build_ltx2_request_generator(native=native, request=request)
    with _ltx2_transformer_execution_context(native):
        outputs = _import_native_ltx2_runtime_symbols()["run_ltx2_img2vid_native"](
            native=native,
            request=request,
            plan=plan,
            generator=generator,
        )
    video, audio = _coerce_native_media_outputs(outputs, mode_label="img2vid")
    frames = _normalize_video_frames(video)
    audio_asset = None
    if generated_audio_export_policy.materialize_audio_asset:
        audio_asset = materialize_ltx2_generated_audio_asset(
            audio,
            sample_rate_hz=native.audio_sample_rate_hz,
        )
    metadata = _build_pipeline_metadata(
        native=native,
        pipeline_name="ltx2_native_img2vid",
        request=request,
        plan=plan,
        frame_count=len(frames),
        has_audio=audio_asset is not None,
        sampler_requested=sampler_requested,
        scheduler_requested=scheduler_requested,
        sampler_effective=sampler_effective,
        scheduler_effective=scheduler_effective,
    )
    return build_ltx2_run_result(frames=frames, audio_asset=audio_asset, metadata=metadata)


def require_ltx2_bundle_inputs(bundle: object) -> Ltx2BundleInputs:
    family = getattr(bundle, "family", None)
    if family is not ModelFamily.LTX2:
        raise RuntimeError(
            "LTX2 runtime bundle rehydration requires a `ModelFamily.LTX2` bundle; "
            f"got {getattr(family, 'value', family)!r}."
        )

    metadata = getattr(bundle, "metadata", None)
    if not isinstance(metadata, dict):
        raise RuntimeError("LTX2 runtime bundle rehydration requires bundle.metadata to be a dict.")

    components = getattr(bundle, "components", None)
    if not isinstance(components, dict):
        raise RuntimeError("LTX2 runtime bundle rehydration requires bundle.components to be a dict.")

    model_ref = str(getattr(bundle, "model_ref", "") or "").strip()
    if not model_ref:
        raise RuntimeError("LTX2 runtime bundle rehydration requires a non-empty bundle.model_ref.")

    estimated_config = getattr(bundle, "estimated_config", None)
    signature = getattr(bundle, "signature", None)
    if estimated_config is None or signature is None:
        raise RuntimeError("LTX2 runtime bundle rehydration requires estimated_config and signature.")

    text_encoder = Ltx2TextEncoderAsset(
        alias=str(metadata.get("tenc_alias") or "").strip(),
        path=str(metadata.get("tenc_path") or "").strip(),
        kind=str(metadata.get("tenc_kind") or "").strip(),
        tokenizer_dir=str(metadata.get("tokenizer_dir") or "").strip(),
    )
    vendor_paths = Ltx2VendorPaths(
        repo_dir=str(metadata.get("vendor_repo_dir") or "").strip(),
        model_index_path=str(metadata.get("model_index_path") or "").strip(),
        tokenizer_dir=str(metadata.get("tokenizer_dir") or "").strip(),
        connectors_config_path=str(metadata.get("connectors_config_path") or "").strip(),
    )

    if not text_encoder.alias or not text_encoder.path or not text_encoder.kind:
        raise RuntimeError("LTX2 runtime bundle metadata is missing text-encoder planning fields.")
    if not vendor_paths.repo_dir or not vendor_paths.model_index_path or not vendor_paths.connectors_config_path:
        raise RuntimeError("LTX2 runtime bundle metadata is missing vendored LTX2 metadata paths.")

    vocoder_config = metadata.get("vocoder_config")
    if vocoder_config is not None and not isinstance(vocoder_config, Mapping):
        raise RuntimeError(
            "LTX2 runtime bundle metadata field `vocoder_config` must be a mapping when present; "
            f"got {type(vocoder_config).__name__}."
        )

    bundle_inputs = Ltx2BundleInputs(
        model_ref=model_ref,
        signature=signature,
        estimated_config=estimated_config,
        components=Ltx2ComponentStates.from_component_map(components),
        text_encoder=text_encoder,
        vendor_paths=vendor_paths,
        vocoder_config=vocoder_config,
    )
    if is_ltx2_wrapped_vocoder_state(bundle_inputs.components.vocoder) and bundle_inputs.vocoder_config is None:
        raise RuntimeError(
            "LTX2 runtime bundle rehydration requires metadata-carried `vocoder_config` for wrapped 2.3 vocoder states."
        )
    return bundle_inputs
