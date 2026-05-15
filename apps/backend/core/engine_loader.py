"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Bundle-aware engine loader for backend use cases.
Resolves a `DiffusionModelBundle`, instantiates the matching engine via the registry, loads the model, and applies runtime options
(attention/accelerator) for engines backed by diffusers pipelines with explicit failures on invalid attention backend configuration.

Symbols (top-level; keep in sync; no ghosts):
- `EngineLoadOptions` (dataclass): Optional engine load overrides (runtime device/dtype/attention + explicit model/asset selectors).
- `_ensure_registry_ready` (function): Ensures the engine registry has the default engines registered (idempotent).
- `_instantiate_engine` (function): Creates an engine instance for a resolved diffusion bundle (family → engine key).
- `_options_to_kwargs` (function): Converts `EngineLoadOptions` into `engine.load(...)` keyword arguments.
- `_apply_runtime_options` (function): Applies runtime options (attention backend from explicit load options or runtime memory config, plus accelerator) to diffusers-backed engines.
- `load_engine` (function): Loads and initializes a diffusion engine for direct use (cleanup is fail-loud and residency-verified on failures).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from apps.backend.core.registry import create_engine
from apps.backend.engines import register_default_engines
from apps.backend.engines.util.accelerator import apply_to_diffusers_pipeline as _apply_accel
from apps.backend.engines.util.attention_backend import apply_to_diffusers_pipeline as _apply_attn
from apps.backend.runtime.load_authority import (
    LoadAuthorityStage,
    coordinator_load_permit,
    require_load_authority,
)
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models.loader import (
    DiffusionModelBundle,
    FAMILY_TO_ENGINE_KEY,
    resolve_diffusion_bundle,
)

_LOG = get_backend_logger("backend.core.engine_loader")


@dataclass
class EngineLoadOptions:
    device: Optional[str] = None  # 'cuda'|'cpu'|None → auto
    dtype: Optional[str] = None  # 'fp16'|'bf16'|'fp32'|None → default
    attention_backend: Optional[str] = None  # 'pytorch'|'xformers'|'split'|'quad'
    accelerator: Optional[str] = None  # 'tensorrt'|'none'
    vae_path: Optional[str] = None  # optional override
    vae_source: Optional[str] = None
    tenc_source: Optional[str] = None
    tenc_path: Any = None
    text_encoder_override: Optional[Mapping[str, Any]] = None
    checkpoint_core_only: Optional[bool] = None
    model_format: Optional[str] = None
    zimage_variant: Optional[str] = None


def _ensure_registry_ready() -> None:
    register_default_engines(replace=False)


def _instantiate_engine(bundle: DiffusionModelBundle):
    engine_key = FAMILY_TO_ENGINE_KEY.get(bundle.family)
    if engine_key is None:
        raise NotImplementedError(f"Model family {bundle.family.value} is not registered with Codex engines.")
    _ensure_registry_ready()
    return create_engine(engine_key)


def _options_to_kwargs(opts: EngineLoadOptions | None) -> Dict[str, Any]:
    if opts is None:
        return {}
    payload: Dict[str, Any] = {}
    if opts.device is not None:
        payload["device"] = str(opts.device)
    if opts.dtype is not None:
        payload["dtype"] = str(opts.dtype)
    if opts.vae_path is not None:
        payload["vae_path"] = str(opts.vae_path)
    if opts.vae_source is not None:
        payload["vae_source"] = str(opts.vae_source)
    if opts.tenc_source is not None:
        payload["tenc_source"] = str(opts.tenc_source)
    if opts.tenc_path is not None:
        payload["tenc_path"] = opts.tenc_path
    if opts.text_encoder_override is not None:
        payload["text_encoder_override"] = dict(opts.text_encoder_override)
    if opts.checkpoint_core_only is not None:
        payload["checkpoint_core_only"] = bool(opts.checkpoint_core_only)
    if opts.model_format is not None:
        payload["model_format"] = str(opts.model_format)
    if opts.zimage_variant is not None:
        payload["zimage_variant"] = str(opts.zimage_variant)
    if opts.attention_backend is not None:
        payload["attention_backend"] = str(opts.attention_backend)
    if opts.accelerator is not None:
        payload["accelerator"] = str(opts.accelerator)
    return payload


def _apply_runtime_options(engine: Any, opts: EngineLoadOptions | None) -> Any:
    pipe = getattr(getattr(engine, "_comp", None), "pipeline", None)
    if pipe is None:
        return engine

    attention_backend = None
    if opts and opts.attention_backend is not None:
        attention_backend = opts.attention_backend
    else:
        try:
            attention_backend = str(memory_management.manager.config.attention.backend.value)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Failed to resolve attention_backend from runtime memory config.") from exc

    if attention_backend is not None:
        _apply_attn(pipe, backend=attention_backend)

    if opts and opts.accelerator is not None:
        try:
            _apply_accel(pipe, accelerator=opts.accelerator)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Failed to apply accelerator %s: %s", opts.accelerator, exc)

    return engine


def _guarded_engine_load(engine: Any, model_ref: str, load_kwargs: Dict[str, Any]) -> None:
    require_load_authority(
        "core.engine_loader.engine.load",
        allowed_stages=(LoadAuthorityStage.LOAD, LoadAuthorityStage.RELOAD),
    )
    engine.load(model_ref, **load_kwargs)


def _guarded_engine_unload(engine: Any) -> None:
    require_load_authority(
        "core.engine_loader.engine.unload",
        allowed_stages=(
            LoadAuthorityStage.LOAD,
            LoadAuthorityStage.UNLOAD,
            LoadAuthorityStage.RELOAD,
            LoadAuthorityStage.CLEANUP,
        ),
    )
    engine.unload()


def _engine_residency_targets(engine: Any) -> list[tuple[str, object]]:
    codex_objects = getattr(engine, "codex_objects", None)
    if codex_objects is None:
        return []

    targets: list[tuple[str, object]] = []
    for label in ("denoiser", "unet", "vae", "clipvision"):
        candidate = getattr(codex_objects, label, None)
        if candidate is not None:
            targets.append((label, candidate))

    text_encoders = getattr(codex_objects, "text_encoders", None)
    if isinstance(text_encoders, Mapping):
        for name, candidate in text_encoders.items():
            if candidate is None:
                continue
            targets.append((f"text_encoder:{name}", candidate))

    return targets


def _verify_engine_unloaded(engine: Any, *, source: str) -> None:
    loaded_flag = getattr(engine, "_is_loaded", None)
    if not isinstance(loaded_flag, bool):
        raise RuntimeError(
            f"Post-unload residency verification failed at {source}: invalid engine._is_loaded={loaded_flag!r}."
        )

    status_loaded = loaded_flag
    status_payload = engine.status()
    if isinstance(status_payload, Mapping):
        raw_loaded = status_payload.get("loaded", loaded_flag)
        if not isinstance(raw_loaded, bool):
            raise RuntimeError(
                f"Post-unload residency verification failed at {source}: "
                f"engine.status()['loaded'] is not bool ({raw_loaded!r})."
            )
        status_loaded = raw_loaded

    if loaded_flag or status_loaded:
        raise RuntimeError(
            f"Post-unload residency verification failed at {source}: "
            f"engine still reports loaded (_is_loaded={loaded_flag}, status.loaded={status_loaded})."
        )


def _unload_engine_with_residency_verification(engine: Any, *, source: str) -> None:
    targets = _engine_residency_targets(engine)
    tracked_targets: list[tuple[str, object]] = []
    for label, target in targets:
        try:
            was_loaded = memory_management.manager.is_model_loaded(target)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to evaluate pre-unload residency target '{label}' at {source}: {exc}"
            ) from exc
        if was_loaded:
            tracked_targets.append((label, target))

    _guarded_engine_unload(engine)
    mark_unloaded = getattr(engine, "mark_unloaded", None)
    if not callable(mark_unloaded):
        raise RuntimeError(
            f"Post-unload residency verification failed at {source}: engine does not expose callable mark_unloaded()."
        )
    mark_unloaded()
    _verify_engine_unloaded(engine, source=source)

    lingering: list[str] = []
    for label, target in tracked_targets:
        try:
            if memory_management.manager.is_model_loaded(target):
                lingering.append(label)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to evaluate post-unload residency target '{label}' at {source}: {exc}"
            ) from exc

    if lingering:
        raise RuntimeError(
            f"Post-unload residency verification failed at {source}: "
            f"lingering memory-manager residency for {', '.join(lingering)}."
        )


def load_engine(name_or_path: str, options: EngineLoadOptions | None = None):
    """Load and initialize a Codex diffusion engine for direct use."""

    bundle = resolve_diffusion_bundle(name_or_path)
    engine = _instantiate_engine(bundle)

    load_kwargs = _options_to_kwargs(options)
    load_kwargs["_bundle"] = bundle

    try:
        with coordinator_load_permit(
            owner="core.engine_loader.load_engine",
            stage=LoadAuthorityStage.LOAD,
        ):
            _guarded_engine_load(engine, name_or_path, load_kwargs)
    except Exception as exc:
        cleanup_failures: list[str] = []
        with coordinator_load_permit(
            owner="core.engine_loader.load_engine",
            stage=LoadAuthorityStage.CLEANUP,
        ):
            try:
                _unload_engine_with_residency_verification(
                    engine,
                    source="core.engine_loader.load_engine.load_failure_cleanup",
                )
            except Exception as cleanup_exc:  # noqa: BLE001
                cleanup_failures.append(f"engine_unload:{cleanup_exc}")
        if cleanup_failures:
            detail = "; ".join(cleanup_failures)
            raise RuntimeError(
                f"Failed to load engine for '{name_or_path}': {exc}. "
                f"Additional cleanup failure(s): {detail}"
            ) from exc
        raise

    try:
        return _apply_runtime_options(engine, options)
    except Exception as exc:
        cleanup_failures: list[str] = []
        with coordinator_load_permit(
            owner="core.engine_loader.load_engine",
            stage=LoadAuthorityStage.UNLOAD,
        ):
            try:
                _unload_engine_with_residency_verification(
                    engine,
                    source="core.engine_loader.load_engine.apply_failure_cleanup",
                )
            except Exception as cleanup_exc:  # noqa: BLE001
                cleanup_failures.append(f"engine_unload:{cleanup_exc}")
        if cleanup_failures:
            detail = "; ".join(cleanup_failures)
            raise RuntimeError(
                f"Failed to apply runtime options for '{name_or_path}': {exc}. "
                f"Additional cleanup failure(s): {detail}"
            ) from exc
        raise


__all__ = ["EngineLoadOptions", "load_engine"]
