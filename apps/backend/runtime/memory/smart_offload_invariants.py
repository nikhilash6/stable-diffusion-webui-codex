"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Smart-offload GPU residency invariants (auto-unload forbidden components between stages).
Delegates canonical smart-offload generic load/unload event emission to the memory manager.

Symbols (top-level; keep in sync; no ghosts):
- `enforce_smart_offload_pre_conditioning_residency` (function): Ensures denoiser/VAE are not resident on the accelerator
  before TE conditioning begins (auto-unload when smart offload is enabled).
- `enforce_smart_offload_text_encoders_off` (function): Unloads any resident text encoders on the accelerator when they are
  no longer needed (e.g., Smart Cache hit provides embeddings without TE execution).
- `enforce_smart_offload_pre_sampling_residency` (function): Ensures text encoders are not resident on the accelerator at
  sampling start, and optionally enforces VAE residency rules based on live-preview needs.
- `enforce_smart_offload_pre_vae_residency` (function): Ensures denoiser/text-encoders are not resident on the accelerator
  before explicit VAE encode/decode stages outside the sampling loop.
- `enforce_smart_offload_post_decode_residency` (function): Enforces post-decode residency policy (VAE off accelerator;
  denoiser unloaded from accelerator residency).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Any, Iterable

import torch

from . import memory_management
from apps.backend.runtime.memory.smart_offload import smart_offload_enabled


_LOGGER = get_backend_logger(__name__)


def _as_device(value: object | None) -> torch.device | None:
    if value is None:
        return None
    if isinstance(value, torch.device):
        return value
    try:
        return torch.device(str(value))
    except Exception:
        return None


def _is_accelerator_device(device: torch.device | None) -> bool:
    return device is not None and device.type != "cpu"


def _is_model_loaded_on_accelerator(model: object) -> bool:
    if not memory_management.manager.is_model_loaded(model):
        return False

    load_device = _as_device(getattr(model, "load_device", None))
    if _is_accelerator_device(load_device):
        return True

    current_device = _as_device(getattr(model, "current_device", None))
    return _is_accelerator_device(current_device)


def _require_patcher_target(entry: object, *, component_name: str) -> object:
    """Return canonical patcher target or raise when contract is violated."""

    patcher = getattr(entry, "patcher", entry)
    if patcher is None:
        raise RuntimeError(
            "smart_offload invariant requires canonical patcher targets "
            f"(got null patcher for '{component_name}')."
        )
    if not hasattr(patcher, "codex_patch_model") or not hasattr(patcher, "codex_unpatch_model"):
        raise RuntimeError(
            "smart_offload invariant requires patcher targets exposing "
            f"codex_patch_model/codex_unpatch_model for '{component_name}'."
        )
    return patcher


def _iter_text_encoder_patchers(sd_model: Any) -> Iterable[tuple[str, object]]:
    codex_objects = getattr(sd_model, "codex_objects", None)
    if codex_objects is None:
        return
    mapping = getattr(codex_objects, "text_encoders", None)
    if not isinstance(mapping, dict):
        return
    for name, entry in mapping.items():
        if entry is None:
            continue
        try:
            patcher = entry.patcher
        except AttributeError as exc:
            raise RuntimeError(
                "smart_offload invariant requires TextEncoderHandle entries "
                f"(missing .patcher for text_encoders['{name}'])."
            ) from exc
        if patcher is None:
            raise RuntimeError(
                "smart_offload invariant requires TextEncoderHandle with non-null patcher "
                f"for text_encoders['{name}']."
            )
        if not hasattr(patcher, "codex_patch_model") or not hasattr(patcher, "codex_unpatch_model"):
            raise RuntimeError(
                "smart_offload invariant requires TextEncoderHandle patchers exposing "
                f"codex_patch_model/codex_unpatch_model for text_encoders['{name}']."
            )
        yield str(name), patcher


def _resolve_vae_patcher(sd_model: Any) -> object | None:
    codex_objects = getattr(sd_model, "codex_objects", None)
    if codex_objects is None:
        return None
    vae = getattr(codex_objects, "vae", None)
    if vae is None:
        return None
    return _require_patcher_target(vae, component_name="vae")


def _resolve_denoiser_target(sd_model: Any) -> object | None:
    codex_objects = getattr(sd_model, "codex_objects", None)
    if codex_objects is None:
        return None
    denoiser = getattr(codex_objects, "denoiser", None)
    if denoiser is None:
        return None
    return _require_patcher_target(denoiser, component_name="denoiser")


def enforce_smart_offload_pre_conditioning_residency(sd_model: Any, *, stage: str) -> None:
    """Ensure non-conditioning components are not resident on the accelerator.

    This is intentionally non-fatal: when smart offload is enabled, we prefer to
    auto-unload the forbidden residents to restore the intended stage order.
    """

    if not smart_offload_enabled():
        return

    codex_objects = getattr(sd_model, "codex_objects", None)
    if codex_objects is None:
        return

    denoiser_target = _resolve_denoiser_target(sd_model)
    if denoiser_target is not None and _is_model_loaded_on_accelerator(denoiser_target):
        _LOGGER.warning(
            "[smart-offload] stage=%s: denoiser was still resident on accelerator; unloading before conditioning.",
            stage,
        )
        memory_management.manager.unload_model(
            denoiser_target,
            source="runtime.memory.smart_offload_invariants.pre_conditioning",
            stage=stage,
            component_hint="denoiser",
        )

    vae_patcher = _resolve_vae_patcher(sd_model)
    if vae_patcher is not None and _is_model_loaded_on_accelerator(vae_patcher):
        _LOGGER.debug(
            "[smart-offload] stage=%s: VAE was resident on accelerator; unloading before conditioning.",
            stage,
        )
        memory_management.manager.unload_model(
            vae_patcher,
            source="runtime.memory.smart_offload_invariants.pre_conditioning",
            stage=stage,
            component_hint="vae",
        )


def enforce_smart_offload_text_encoders_off(sd_model: Any, *, stage: str) -> None:
    """Unload any resident text encoders on the accelerator.

    This should be called only when it is safe to do so (e.g., embeddings are
    already computed or served from Smart Cache). It is intentionally non-fatal.
    """

    if not smart_offload_enabled():
        return

    for name, patcher in _iter_text_encoder_patchers(sd_model):
        if _is_model_loaded_on_accelerator(patcher):
            _LOGGER.debug(
                "[smart-offload] stage=%s: unloading resident text encoder '%s' (embeddings already available).",
                stage,
                name,
            )
            memory_management.manager.unload_model(
                patcher,
                source="runtime.memory.smart_offload_invariants.text_encoders_off",
                stage=stage,
                component_hint=f"text_encoder:{name}",
            )


def enforce_smart_offload_pre_sampling_residency(
    sd_model: Any,
    *,
    stage: str,
    allow_vae_resident: bool,
) -> None:
    """Ensure text encoders are not resident on the accelerator when sampling begins.

    If `allow_vae_resident` is False, also unload any resident VAE patcher. This
    supports a strict stage order (TE -> denoiser -> VAE), with an explicit
    exception for live preview FULL mode (which uses VAE decode during sampling).
    """

    if not smart_offload_enabled():
        return

    for name, patcher in _iter_text_encoder_patchers(sd_model):
        if _is_model_loaded_on_accelerator(patcher):
            _LOGGER.warning(
                "[smart-offload] stage=%s: text encoder '%s' was still resident on accelerator; unloading before sampling.",
                stage,
                name,
            )
            memory_management.manager.unload_model(
                patcher,
                source="runtime.memory.smart_offload_invariants.pre_sampling",
                stage=stage,
                component_hint=f"text_encoder:{name}",
            )

    if not allow_vae_resident:
        vae_patcher = _resolve_vae_patcher(sd_model)
        if vae_patcher is not None and _is_model_loaded_on_accelerator(vae_patcher):
            _LOGGER.debug(
                "[smart-offload] stage=%s: VAE was resident on accelerator; unloading before sampling.",
                stage,
            )
            memory_management.manager.unload_model(
                vae_patcher,
                source="runtime.memory.smart_offload_invariants.pre_sampling",
                stage=stage,
                component_hint="vae",
            )


def enforce_smart_offload_pre_vae_residency(
    sd_model: Any,
    *,
    stage: str,
) -> None:
    """Ensure denoiser/text-encoders are not resident before explicit VAE stages.

    This guard is intended for VAE encode/decode phases that run outside sampling
    (for example hires init preparation and final output decode). It must not be
    used by live preview FULL inside the sampler loop.
    """

    if not smart_offload_enabled():
        return

    codex_objects = getattr(sd_model, "codex_objects", None)
    if codex_objects is None:
        return

    denoiser_target = _resolve_denoiser_target(sd_model)
    if denoiser_target is not None and _is_model_loaded_on_accelerator(denoiser_target):
        _LOGGER.warning(
            "[smart-offload] stage=%s: denoiser was still resident on accelerator; unloading before VAE stage.",
            stage,
        )
        memory_management.manager.unload_model(
            denoiser_target,
            source="runtime.memory.smart_offload_invariants.pre_vae",
            stage=stage,
            component_hint="denoiser",
        )

    for name, patcher in _iter_text_encoder_patchers(sd_model):
        if _is_model_loaded_on_accelerator(patcher):
            _LOGGER.debug(
                "[smart-offload] stage=%s: text encoder '%s' remained resident before VAE stage; unloading.",
                stage,
                name,
            )
            memory_management.manager.unload_model(
                patcher,
                source="runtime.memory.smart_offload_invariants.pre_vae",
                stage=stage,
                component_hint=f"text_encoder:{name}",
            )


def enforce_smart_offload_post_decode_residency(
    sd_model: Any,
    *,
    stage: str,
) -> None:
    """Enforce post-decode residency policy for VAE/denoiser.

    Policy:
    - VAE must not remain resident on accelerator after decode.
    - Denoiser is unloaded after decode if it remains resident on accelerator.
    """

    if not smart_offload_enabled():
        return

    codex_objects = getattr(sd_model, "codex_objects", None)
    if codex_objects is None:
        return

    vae_patcher = _resolve_vae_patcher(sd_model)
    if vae_patcher is not None and _is_model_loaded_on_accelerator(vae_patcher):
        _LOGGER.warning(
            "[smart-offload] stage=%s: VAE remained resident after decode; unloading.",
            stage,
        )
        memory_management.manager.unload_model(
            vae_patcher,
            source="runtime.memory.smart_offload_invariants.post_decode",
            stage=stage,
            component_hint="vae",
        )

    denoiser_target = _resolve_denoiser_target(sd_model)
    if denoiser_target is None:
        return

    denoiser_target_device = _as_device(getattr(denoiser_target, "load_device", None))
    if denoiser_target_device is not None and denoiser_target_device.type == "cpu":
        return

    if _is_model_loaded_on_accelerator(denoiser_target):
        _LOGGER.debug(
            "[smart-offload] stage=%s: unloading denoiser after decode.",
            stage,
        )
        memory_management.manager.unload_model(
            denoiser_target,
            source="runtime.memory.smart_offload_invariants.post_decode",
            stage=stage,
            component_hint="denoiser",
            event_reason="post_decode_unload",
        )


__all__ = [
    "enforce_smart_offload_pre_conditioning_residency",
    "enforce_smart_offload_text_encoders_off",
    "enforce_smart_offload_pre_sampling_residency",
    "enforce_smart_offload_pre_vae_residency",
    "enforce_smart_offload_post_decode_residency",
]
