"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Request-scoped IP-Adapter apply/restore context management.
Builds conditional image tokens, patches the active denoiser clone for the current sampling pass, yields no payload, and restores the
baseline Codex objects in a `finally`-friendly context manager.

Symbols (top-level; keep in sync; no ghosts):
- `apply_ip_adapter_for_sampling` (function): Context manager that patches the active sampling denoiser for one sampling pass and restores it afterwards.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import copy
import contextlib
import logging
from collections.abc import Iterator

import torch

from apps.backend.infra.config.env_flags import env_flag, env_int
from apps.backend.runtime.adapters.ip_adapter.assets import assert_ip_adapter_engine_supported, prepare_ip_adapter_assets
from apps.backend.runtime.adapters.ip_adapter.layout import resolve_ip_adapter_transformer_coordinates
from apps.backend.runtime.adapters.ip_adapter.modules import IpAdapterCrossAttentionPatch
from apps.backend.runtime.adapters.ip_adapter.preprocess import prepare_ip_adapter_embeddings
from apps.backend.runtime.adapters.ip_adapter.types import IpAdapterConfig
from apps.backend.runtime.model_registry.capabilities import semantic_engine_for_engine_id

logger = get_backend_logger("backend.runtime.adapters.ip_adapter.session")


@contextlib.contextmanager
def apply_ip_adapter_for_sampling(processing) -> Iterator[None]:
    config = getattr(processing, "ip_adapter", None)
    if not isinstance(config, IpAdapterConfig) or not config.enabled:
        yield None
        return
    engine = getattr(processing, "sd_model", None)
    if engine is None:
        raise RuntimeError("IP-Adapter requires processing.sd_model before sampling begins.")
    engine_id = str(getattr(engine, "engine_id", "") or "").strip().lower()
    assert_ip_adapter_engine_supported(engine_id)
    assets = prepare_ip_adapter_assets(config)
    _assert_ip_adapter_family_compatible(engine_id=engine_id, target_semantic_engine=assets.target_semantic_engine)
    embeddings = prepare_ip_adapter_embeddings(processing=processing, config=config, assets=assets)
    previous_codex_objects = engine.codex_objects
    patched_codex_objects = previous_codex_objects.shallow_copy()
    patched_denoiser = previous_codex_objects.denoiser.clone()
    runtime_device, runtime_dtype = _resolve_runtime_device_and_dtype(patched_denoiser=patched_denoiser)
    sigma_start, sigma_end = _sigma_window(patched_denoiser=patched_denoiser, config=config)
    coordinates = resolve_ip_adapter_transformer_coordinates(
        patched_denoiser=patched_denoiser,
        semantic_engine=assets.target_semantic_engine,
        ip_layers=assets.ip_layers,
    )
    slot_specs = tuple(assets.ip_layers.slot_specs)
    if len(coordinates) != int(assets.slot_count):
        raise RuntimeError(
            f"IP-Adapter slot/layout mismatch: denoiser exposes {len(coordinates)} attn2 coordinates but adapter provides {assets.slot_count} slot(s)."
        )
    if len(slot_specs) != len(coordinates):
        raise RuntimeError(
            "IP-Adapter slot/source-key mismatch: "
            f"parsed slot specs={len(slot_specs)} coordinates={len(coordinates)}."
        )
    IpAdapterCrossAttentionPatch.reset_debug_counter()
    if env_flag("CODEX_IP_ADAPTER_DEBUG") or env_flag("CODEX_IP_ADAPTER_DEBUG_PATCH"):
        debug_limit = env_int("CODEX_IP_ADAPTER_DEBUG_PATCH_MAP_N", 8, min_value=0)
        slot_preview = [
            {
                "slot_index": slot_index,
                "coordinate": [block_name, int(block_index), int(transformer_index)],
                "k_source_key": slot_spec.k_source_key,
                "v_source_key": slot_spec.v_source_key,
            }
            for slot_index, ((block_name, block_index, transformer_index), slot_spec) in enumerate(
                zip(coordinates, slot_specs, strict=True)
            )
            if slot_index < debug_limit
        ]
        logger.info(
            "[ip-adapter-debug] session map | layout=%s slots=%d weight=%.3f start=%.3f end=%.3f sigma_start=%.6f sigma_end=%.6f token_shapes=(cond=%s uncond=%s) preview=%s",
            assets.layout.value,
            len(slot_specs),
            float(config.weight),
            float(config.start_at),
            float(config.end_at),
            float(sigma_start),
            float(sigma_end),
            tuple(int(dim) for dim in embeddings.condition.shape),
            tuple(int(dim) for dim in embeddings.uncondition.shape),
            slot_preview,
        )
    session_ip_layers = copy.deepcopy(assets.ip_layers).to(device=runtime_device, dtype=runtime_dtype)
    condition_tokens = embeddings.condition.to(device=runtime_device, dtype=runtime_dtype)
    uncondition_tokens = embeddings.uncondition.to(device=runtime_device, dtype=runtime_dtype)
    attn2_patches: dict[tuple[str, int, int], IpAdapterCrossAttentionPatch] = {}
    for slot_index, ((block_name, block_index, transformer_index), slot_spec) in enumerate(zip(coordinates, slot_specs, strict=True)):
        attn2_patches[(block_name, block_index, transformer_index)] = IpAdapterCrossAttentionPatch(
            slot_index=slot_index,
            k_source_key=slot_spec.k_source_key,
            v_source_key=slot_spec.v_source_key,
            weight=float(config.weight),
            sigma_start=float(sigma_start),
            sigma_end=float(sigma_end),
            ip_layers=session_ip_layers,
            condition=condition_tokens,
            uncondition=uncondition_tokens,
        )
    patched_denoiser.set_model_attn2_replace_many(attn2_patches)
    del attn2_patches
    del session_ip_layers
    del condition_tokens
    del uncondition_tokens
    patched_codex_objects.denoiser = patched_denoiser
    logger.debug(
        "Applying IP-Adapter for engine=%s layout=%s slots=%d weight=%.3f start=%.3f end=%.3f",
        engine_id,
        assets.layout.value,
        assets.slot_count,
        config.weight,
        config.start_at,
        config.end_at,
    )
    engine.codex_objects = patched_codex_objects
    try:
        yield
    finally:
        engine.codex_objects = previous_codex_objects


def _sigma_window(*, patched_denoiser, config: IpAdapterConfig) -> tuple[float, float]:
    predictor = getattr(getattr(patched_denoiser, "model", None), "predictor", None)
    if predictor is None or not hasattr(predictor, "percent_to_sigma"):
        raise RuntimeError("IP-Adapter requires a denoiser predictor exposing percent_to_sigma(...).")
    sigma_start = float(predictor.percent_to_sigma(float(config.start_at)))
    sigma_end = float(predictor.percent_to_sigma(float(config.end_at)))
    return sigma_start, sigma_end


def _resolve_runtime_device_and_dtype(*, patched_denoiser) -> tuple[torch.device, torch.dtype]:
    diffusion_model = getattr(getattr(patched_denoiser, "model", None), "diffusion_model", None)
    if diffusion_model is None:
        raise RuntimeError("IP-Adapter requires patched_denoiser.model.diffusion_model to resolve runtime device/dtype.")
    for parameter in diffusion_model.parameters():
        if isinstance(parameter, torch.Tensor):
            return parameter.device, parameter.dtype
    raise RuntimeError("IP-Adapter could not resolve an active diffusion-model parameter for runtime device/dtype.")


def _assert_ip_adapter_family_compatible(*, engine_id: str, target_semantic_engine: str) -> None:
    active_semantic_engine = semantic_engine_for_engine_id(engine_id).value
    if active_semantic_engine == str(target_semantic_engine):
        return
    raise RuntimeError(
        "Unsupported IP-Adapter family pairing in tranche 1: "
        f"engine '{engine_id}' resolves to '{active_semantic_engine}', "
        f"but the selected adapter targets '{target_semantic_engine}'."
    )
