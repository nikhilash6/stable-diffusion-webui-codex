"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: IP-Adapter asset loading and tranche-1 layout validation.
Loads image encoders and adapter weights, validates supported IP-Adapter checkpoint layouts, and returns prepared runtime assets for the
shared IP-Adapter stage. CLIP vision image-encoder keyspace resolution is delegated to the canonical vision runtime.

Symbols (top-level; keep in sync; no ghosts):
- `assert_ip_adapter_engine_supported` (function): Fail-loud guard for exact engine-id and semantic-engine IP-Adapter support.
- `invalidate_ip_adapter_asset_cache` (function): Drops the process-local prepared-asset cache for IP-Adapter runtime bundles.
- `prepare_ip_adapter_assets` (function): Loads and caches the validated IP-Adapter asset bundle for one model/image-encoder pair.
- `prepare_ip_adapter_assets_for_paths` (function): Loads and caches the validated IP-Adapter asset bundle for explicit model/image-encoder paths.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import threading
from collections.abc import Mapping, MutableMapping

import torch

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.adapters.ip_adapter.modules import (
    ImageProjectionModel,
    IpAdapterKvProjectionSet,
    MlpProjectionModel,
    Resampler,
)
from apps.backend.runtime.adapters.ip_adapter.types import IpAdapterConfig, IpAdapterLayout, PreparedIpAdapterAssets
from apps.backend.runtime.checkpoint.io import load_torch_file
from apps.backend.runtime.model_registry.capabilities import ip_adapter_support_error
from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime.state_dict.views import FilterPrefixView
from apps.backend.runtime.vision.clip.encoder import ClipVisionEncoder

logger = get_backend_logger("backend.runtime.adapters.ip_adapter.assets")

_ASSET_CACHE: dict[tuple[str, str, str, str, str], PreparedIpAdapterAssets] = {}
_ASSET_CACHE_LOCK = threading.Lock()

def assert_ip_adapter_engine_supported(engine_id: str) -> None:
    detail = ip_adapter_support_error(engine_id)
    if detail is not None:
        raise RuntimeError(detail)


def invalidate_ip_adapter_asset_cache() -> None:
    with _ASSET_CACHE_LOCK:
        _ASSET_CACHE.clear()


def prepare_ip_adapter_assets(config: IpAdapterConfig) -> PreparedIpAdapterAssets:
    return prepare_ip_adapter_assets_for_paths(
        model_path=str(config.model),
        image_encoder_path=str(config.image_encoder),
    )


def prepare_ip_adapter_assets_for_paths(*, model_path: str, image_encoder_path: str) -> PreparedIpAdapterAssets:
    cache_key = _asset_cache_key(model_path=model_path, image_encoder_path=image_encoder_path)
    with _ASSET_CACHE_LOCK:
        cached = _ASSET_CACHE.get(cache_key)
        if cached is not None:
            return cached
    image_encoder_runtime = _load_image_encoder(image_encoder_path)
    image_proj_state, ip_adapter_state = _load_ip_adapter_checkpoint(model_path)
    prepared = _prepare_assets(
        model_path=model_path,
        image_encoder_path=image_encoder_path,
        image_encoder_runtime=image_encoder_runtime,
        image_proj_state=image_proj_state,
        ip_adapter_state=ip_adapter_state,
    )
    with _ASSET_CACHE_LOCK:
        _ASSET_CACHE[cache_key] = prepared
    return prepared


def _asset_cache_key(*, model_path: str, image_encoder_path: str) -> tuple[str, str, str, str, str]:
    load_device = str(memory_management.manager.get_device(DeviceRole.CLIP_VISION))
    offload_device = str(memory_management.manager.get_offload_device(DeviceRole.CLIP_VISION))
    runtime_dtype = str(memory_management.manager.dtype_for_role(DeviceRole.CLIP_VISION))
    return (
        str(model_path),
        str(image_encoder_path),
        load_device,
        offload_device,
        runtime_dtype,
    )


def _prepare_assets(
    *,
    model_path: str,
    image_encoder_path: str,
    image_encoder_runtime: ClipVisionEncoder,
    image_proj_state: Mapping[str, torch.Tensor],
    ip_adapter_state: Mapping[str, torch.Tensor],
) -> PreparedIpAdapterAssets:
    if _is_face_or_instant_layout(image_proj_state=image_proj_state, ip_adapter_state=ip_adapter_state):
        raise RuntimeError(
            "Unsupported IP-Adapter layout. FaceID/portrait/full/Instant-ID variants are not implemented in tranche 1."
        )
    uses_hidden_states = _is_plus_layout(image_proj_state)
    slot_specs = IpAdapterKvProjectionSet.inspect_state_dict(ip_adapter_state)
    if not slot_specs:
        raise RuntimeError("IP-Adapter checkpoint is missing KV projection slots.")
    output_cross_attention_dim = int(slot_specs[0].input_dim)
    target_semantic_engine = _target_semantic_engine_for_cross_attention_dim(output_cross_attention_dim)
    is_sdxl = target_semantic_engine == "sdxl"
    token_count = 16 if uses_hidden_states else 4
    internal_cross_attention_dim = 1280 if uses_hidden_states and is_sdxl else output_cross_attention_dim
    image_projector = _build_image_projector(
        image_proj_state=image_proj_state,
        image_encoder_runtime=image_encoder_runtime,
        uses_hidden_states=uses_hidden_states,
        output_cross_attention_dim=output_cross_attention_dim,
        internal_cross_attention_dim=internal_cross_attention_dim,
        token_count=token_count,
        is_sdxl=is_sdxl,
    )
    image_projector.eval()
    ip_layers = IpAdapterKvProjectionSet(ip_adapter_state)
    return PreparedIpAdapterAssets(
        model_path=str(model_path),
        image_encoder_path=str(image_encoder_path),
        layout=IpAdapterLayout.PLUS if uses_hidden_states else IpAdapterLayout.BASE,
        target_semantic_engine=target_semantic_engine,
        slot_count=int(ip_layers.slot_count),
        token_count=int(token_count),
        output_cross_attention_dim=int(output_cross_attention_dim),
        internal_cross_attention_dim=int(internal_cross_attention_dim),
        uses_hidden_states=bool(uses_hidden_states),
        image_encoder_runtime=image_encoder_runtime,
        image_projector=image_projector,
        ip_layers=ip_layers,
    )


def _load_image_encoder(path: str) -> ClipVisionEncoder:
    raw_state = load_torch_file(
        path,
        safe_load=True,
        device=memory_management.manager.get_offload_device(DeviceRole.TEXT_ENCODER),
    )
    if not isinstance(raw_state, MutableMapping):
        raise RuntimeError(f"IP-Adapter image encoder '{path}' did not load as a mutable mapping.")
    return ClipVisionEncoder.from_state_dict(raw_state)


def _load_ip_adapter_checkpoint(path: str) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    raw_state = load_torch_file(path, safe_load=True)
    if not isinstance(raw_state, Mapping):
        raise RuntimeError(f"IP-Adapter checkpoint '{path}' did not load as a mapping.")
    image_proj_state, ip_adapter_state = _split_ip_adapter_state(raw_state)
    if not image_proj_state:
        raise RuntimeError(f"IP-Adapter checkpoint '{path}' is missing 'image_proj' weights.")
    if not ip_adapter_state:
        raise RuntimeError(f"IP-Adapter checkpoint '{path}' is missing 'ip_adapter' weights.")
    return image_proj_state, ip_adapter_state


def _split_ip_adapter_state(raw_state: Mapping[str, object]) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    nested_image_proj = raw_state.get("image_proj")
    nested_ip_adapter = raw_state.get("ip_adapter")
    if isinstance(nested_image_proj, Mapping) and isinstance(nested_ip_adapter, Mapping):
        return _validated_tensor_mapping_view(
            nested_image_proj,
            label="image_proj",
        ), _validated_tensor_mapping_view(
            nested_ip_adapter,
            label="ip_adapter",
        )
    image_proj_keys: list[str] = []
    ip_adapter_keys: list[str] = []
    other_tensor_keys: list[str] = []
    for source_key, value in raw_state.items():
        if not isinstance(source_key, str):
            raise RuntimeError("IP-Adapter checkpoint keys must be strings.")
        if not isinstance(value, torch.Tensor):
            continue
        if source_key.startswith("image_proj."):
            image_proj_keys.append(source_key)
            continue
        if source_key.startswith("ip_adapter."):
            ip_adapter_keys.append(source_key)
            continue
        other_tensor_keys.append(source_key)
    if other_tensor_keys:
        raise RuntimeError(
            "Unsupported IP-Adapter checkpoint layout; unexpected tensor keys outside explicit 'image_proj.' / 'ip_adapter.' buckets: "
            + ", ".join(sorted(other_tensor_keys)[:8])
        )
    if not image_proj_keys or not ip_adapter_keys:
        raise RuntimeError(
            "Unsupported IP-Adapter checkpoint layout; expected nested 'image_proj'/'ip_adapter' mappings "
            "or explicit flat 'image_proj.'/'ip_adapter.' tensor buckets."
        )
    image_proj_state = FilterPrefixView(raw_state, "image_proj.")
    ip_adapter_state = FilterPrefixView(raw_state, "ip_adapter.")
    return _validated_tensor_mapping_view(image_proj_state, label="image_proj"), _validated_tensor_mapping_view(
        ip_adapter_state,
        label="ip_adapter",
    )


def _validated_tensor_mapping_view(mapping: Mapping[str, object], *, label: str) -> Mapping[str, torch.Tensor]:
    for source_key in mapping.keys():
        if not isinstance(source_key, str):
            raise RuntimeError(f"IP-Adapter {label} keys must be strings.")
    return mapping


def _is_plus_layout(image_proj_state: Mapping[str, torch.Tensor]) -> bool:
    return "latents" in image_proj_state and "proj_in.weight" in image_proj_state


def _is_face_or_instant_layout(
    *,
    image_proj_state: Mapping[str, torch.Tensor],
    ip_adapter_state: Mapping[str, torch.Tensor],
) -> bool:
    if "proj.3.weight" in image_proj_state:
        return True
    if "proj.2.weight" in image_proj_state:
        return True
    return any("to_q_lora" in key or "to_k_lora" in key or "to_v_lora" in key for key in ip_adapter_state.keys())


def _target_semantic_engine_for_cross_attention_dim(output_cross_attention_dim: int) -> str:
    if int(output_cross_attention_dim) == 768:
        return "sd15"
    if int(output_cross_attention_dim) == 2048:
        return "sdxl"
    raise RuntimeError(
        "Unsupported IP-Adapter family in tranche 1: "
        f"expected cross-attention dim 768 (SD15) or 2048 (SDXL), got {int(output_cross_attention_dim)}."
    )


def _build_image_projector(
    *,
    image_proj_state: Mapping[str, torch.Tensor],
    image_encoder_runtime: ClipVisionEncoder,
    uses_hidden_states: bool,
    output_cross_attention_dim: int,
    internal_cross_attention_dim: int,
    token_count: int,
    is_sdxl: bool,
) -> torch.nn.Module:
    if uses_hidden_states:
        embedding_dim = int(image_proj_state.get("proj_in.weight", torch.empty(0, 0)).shape[1])
        expected_embedding_dim = int(image_encoder_runtime.spec.hidden_size)
        if embedding_dim != expected_embedding_dim:
            raise RuntimeError(
                f"IP-Adapter image encoder hidden size mismatch: expected {expected_embedding_dim}, got {embedding_dim}."
            )
        image_projector = Resampler(
            dim=int(internal_cross_attention_dim),
            depth=4,
            dim_head=64,
            heads=20 if is_sdxl else 12,
            num_queries=int(token_count),
            embedding_dim=int(embedding_dim),
            output_dim=int(output_cross_attention_dim),
            ff_mult=4,
        )
        missing, unexpected = safe_load_state_dict(image_projector, image_proj_state, log_name="IPAdapterPlusProjector")
        if missing or unexpected:
            raise RuntimeError(
                f"IP-Adapter Plus image projector mismatch (missing={len(missing)}, unexpected={len(unexpected)})."
            )
        return image_projector
    embedding_dim = int(image_proj_state.get("proj.weight", torch.empty(0, 0)).shape[1])
    expected_embedding_dim = int(image_encoder_runtime.spec.projection_dim)
    if embedding_dim != expected_embedding_dim:
        raise RuntimeError(
            f"IP-Adapter image encoder projection dim mismatch: expected {expected_embedding_dim}, got {embedding_dim}."
        )
    image_projector = ImageProjectionModel(
        cross_attention_dim=int(output_cross_attention_dim),
        clip_embeddings_dim=int(embedding_dim),
        clip_extra_context_tokens=int(token_count),
    )
    missing, unexpected = safe_load_state_dict(image_projector, image_proj_state, log_name="IPAdapterBaseProjector")
    if missing or unexpected:
        raise RuntimeError(
            f"IP-Adapter base image projector mismatch (missing={len(missing)}, unexpected={len(unexpected)})."
        )
    return image_projector
