"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native LoRA application pipeline (no legacy modules).
Preflights selected LoRAs against the active denoiser and text-encoder targets (cheap SafeTensors header fast path plus materialized structural validation),
converts compatible files into patch dictionaries and applies them to the engine's denoiser and text encoders via the `ModelPatcher` system before
refreshing LoRA state on the patchers (unset mode is `online`; explicit `merge` rewrites weights once at apply-time).
Fails loud when selected LoRAs do not match runtime parameters or fail structural compatibility checks.
Patch dictionary keys may be plain parameter names or `(parameter, offset)` tuples for slice patches (e.g. fused-QKV text encoders).

Symbols (top-level; keep in sync; no ghosts):
- `AppliedStats` (dataclass): Counters for applied LoRA files and matched parameters.
- `_unwrap_patcher` (function): Returns a `ModelPatcher` from canonical text-encoder handles (`.patcher` required).
- `_resolve_text_encoder_model` (function): Resolves the canonical text-encoder model from a `TextEncoderHandle` patcher.
- `_collect_text_encoder_patchers` (function): Collects resettable text-encoder patchers keyed by encoder name.
- `_clear_lora_state` (function): Clears `lora_patches` on a patcher with fail-loud contract checks.
- `_clear_and_refresh_lora_state` (function): Clears `lora_patches` and refreshes a patcher with fail-loud contract checks.
- `_refresh_lora_state` (function): Refreshes LoRA-merged weights on a patcher with fail-loud contract checks.
- `_normalize_selection` (function): Validates and normalizes a LoRA selection into `(path, text_encoder_weight, unet_weight)`.
- `_serialize_selection_hash` (function): Serializes deterministic LoRA selection identity for cache keys.
- `_set_engine_lora_hash` (function): Updates `engine.current_lora_hash` with fail-loud checks.
- `_reset_engine_lora_state` (function): Clears+refreshes all patchers and persists empty LoRA hash.
- `_raise_apply_failure` (function): Raises fail-loud errors after best-effort reset recovery.
- `_build_to_load_maps` (function): Builds LoRA-key → model patch-target maps for UNet and one text encoder.
- `_collect_target_shape_by_key` (function): Collects live target shapes from a runtime model state dict for structural preflight.
- `_run_header_only_preflight` (function): Runs the cheap SafeTensors header-only structural preflight for native UNet/text targets.
- `_validate_materialized_patch_dict` (function): Verifies parsed native patch dictionaries against live target shapes before mutation.
- `_apply_patches` (function): Adds patches to a patcher and returns the number of matched parameters.
- `apply_loras_to_engine` (function): Applies selected LoRAs to the engine's patchers and refreshes LoRA application (merge or online).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Tuple, Any, Iterable

import safetensors.torch as sf

from apps.backend.infra.config.lora_apply_mode import LoraApplyMode, read_lora_apply_mode
from apps.backend.runtime.adapters.base import PatchTarget
from apps.backend.runtime.adapters.lora.preflight import (
    build_standard_shape_patch_dict_from_shape_map,
    format_shape_compatibility_samples,
    shapeify_patch_dict,
    validate_shape_patch_dict,
)
from apps.backend.runtime.checkpoint.safetensors_header import read_safetensors_tensor_shapes

from .lora import model_lora_keys_unet, model_lora_keys_clip, load_lora


@dataclass
class AppliedStats:
    files: int = 0
    params_touched: int = 0


def _unwrap_patcher(entry: Any, *, label: str) -> Any:
    """Return a patcher from canonical text-encoder handles."""

    try:
        patcher = entry.patcher
    except AttributeError as exc:
        raise RuntimeError(
            "LoRA application requires canonical TextEncoderHandle entries with `.patcher` "
            f"(missing for {label})."
        ) from exc
    if patcher is None:
        raise RuntimeError(
            "LoRA application requires canonical TextEncoderHandle entries with non-null patcher "
            f"(missing for {label})."
        )
    if not hasattr(patcher, "add_patches") or not hasattr(patcher, "refresh_loras"):
        raise RuntimeError(
            f"LoRA application requires a patcher with add_patches/refresh_loras for {label}."
        )
    return patcher


def _resolve_text_encoder_model(entry: Any, *, label: str) -> Any:
    """Resolve the canonical text-encoder model from a `TextEncoderHandle` patcher."""

    patcher = _unwrap_patcher(entry, label=label)
    text_model = getattr(patcher, "model", None)
    if text_model is None:
        raise RuntimeError(
            f"LoRA key mapping requires {label} patcher exposing `.model`."
        )
    if not callable(getattr(text_model, "state_dict", None)):
        raise RuntimeError(
            f"LoRA key mapping requires {label} patcher model with `state_dict()`."
        )
    return text_model


def _collect_text_encoder_patchers(text_encoders: Any) -> Dict[str, Any]:
    """Collect text-encoder patchers keyed by encoder id."""

    if not isinstance(text_encoders, Mapping):
        return {}
    patchers: Dict[str, Any] = {}
    for key, entry in text_encoders.items():
        patchers[str(key)] = _unwrap_patcher(entry, label=f"text_encoders[{key!r}]")
    return patchers


def _resolve_text_encoder_entry(text_encoders: Any) -> tuple[str, Any]:
    """Resolve the primary text-encoder handle for LoRA key mapping."""

    if not isinstance(text_encoders, Mapping):
        raise RuntimeError("Engine does not expose text_encoders mapping required for LoRA key mapping.")

    for key in ("clip", "t5"):
        entry = text_encoders.get(key)
        if entry is not None:
            return key, entry

    for key, entry in text_encoders.items():
        if entry is not None:
            return str(key), entry

    raise RuntimeError("Engine does not expose any text encoder entry required for LoRA key mapping.")


def _resolve_primary_text_patcher(text_patchers: Mapping[str, Any]) -> tuple[str, Any]:
    """Resolve the primary text-encoder patcher for LoRA application."""

    for key in ("clip", "t5"):
        patcher = text_patchers.get(key)
        if patcher is not None:
            return key, patcher

    for key, patcher in text_patchers.items():
        if patcher is not None:
            return str(key), patcher

    raise RuntimeError("Engine does not expose any text encoder patcher required for LoRA application.")


def _clear_lora_state(patcher: Any, *, label: str) -> None:
    """Clear in-memory LoRA patch definitions without materializing refresh."""

    if not hasattr(patcher, "lora_patches"):
        raise RuntimeError(f"Engine exposes non-resettable LoRA patcher for {label}.")
    patcher.lora_patches = {}


def _clear_and_refresh_lora_state(patcher: Any, *, label: str) -> None:
    """Clear in-memory LoRA patch state and re-materialize merged weights."""

    if not hasattr(patcher, "refresh_loras"):
        raise RuntimeError(f"Engine exposes non-refreshable LoRA patcher for {label}.")
    _clear_lora_state(patcher, label=label)
    patcher.refresh_loras()


def _refresh_lora_state(patcher: Any, *, label: str) -> None:
    """Refresh merged LoRA weights without clearing patch definitions."""

    if not hasattr(patcher, "refresh_loras"):
        raise RuntimeError(f"Engine exposes non-refreshable LoRA patcher for {label}.")
    patcher.refresh_loras()


def _normalize_selection(selection: dict | Any) -> tuple[str, float, float] | None:
    """Return `(path, text_encoder_weight, unet_weight)` or `None` when path is empty."""

    path: str
    weight_raw: Any
    unet_weight_raw: Any
    if isinstance(selection, Mapping):
        path = str(selection.get("path") or "").strip()
        weight_raw = selection.get("weight", 1.0)
        unet_weight_raw = selection.get("unet_weight", None)
    else:
        path = str(getattr(selection, "path", "") or "").strip()
        weight_raw = getattr(selection, "weight", 1.0)
        unet_weight_raw = getattr(selection, "unet_weight", None)
    if not path:
        return None
    try:
        text_encoder_weight = float(weight_raw)
    except Exception as exc:  # noqa: BLE001 - strict fail-loud selection contract
        raise RuntimeError(
            f"LoRA selection for '{path}' has non-numeric text-encoder weight: {weight_raw!r}."
        ) from exc
    if unet_weight_raw in (None, ""):
        unet_weight = text_encoder_weight
    else:
        try:
            unet_weight = float(unet_weight_raw)
        except Exception as exc:  # noqa: BLE001 - strict fail-loud selection contract
            raise RuntimeError(
                f"LoRA selection for '{path}' has non-numeric UNet weight: {unet_weight_raw!r}."
            ) from exc
    return path, text_encoder_weight, unet_weight


def _serialize_selection_hash(
    normalized: Iterable[tuple[str, float, float]],
    *,
    apply_mode: LoraApplyMode,
) -> str:
    """Build deterministic LoRA selection identity for cache keys."""

    rows = [
        {
            "path": path,
            "weight": float(text_encoder_weight),
            "unet_weight": float(unet_weight),
        }
        for path, text_encoder_weight, unet_weight in normalized
    ]
    if not rows:
        return "[]"
    rows.sort(key=lambda item: (item["path"], item["weight"], item["unet_weight"]))
    payload = {"apply_mode": apply_mode.value, "selections": rows}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_unique_selections(selections: Iterable[dict | Any]) -> list[tuple[str, float, float]]:
    """Normalize selections and de-duplicate by path while preserving first-seen order."""

    normalized_unique: list[tuple[str, float, float]] = []
    seen_paths: set[str] = set()
    for sel in selections:
        normalized = _normalize_selection(sel)
        if normalized is None:
            continue
        path, text_encoder_weight, unet_weight = normalized
        if path in seen_paths:
            continue
        seen_paths.add(path)
        normalized_unique.append((path, text_encoder_weight, unet_weight))
    return normalized_unique


def selection_hash_for_request(
    selections: Iterable[dict | Any],
    *,
    apply_mode: LoraApplyMode | None = None,
) -> str:
    """Return deterministic LoRA selection hash without mutating patchers."""

    mode = apply_mode if apply_mode is not None else read_lora_apply_mode()
    normalized = _normalize_unique_selections(selections or [])
    return _serialize_selection_hash(normalized, apply_mode=mode)


def _set_engine_lora_hash(engine: Any, *, hash_value: str) -> None:
    """Update `engine.current_lora_hash` with fail-loud contract checks."""

    try:
        setattr(engine, "current_lora_hash", hash_value)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Engine must expose writable `current_lora_hash` for deterministic conditioning identity."
        ) from exc


def _reset_engine_lora_state(engine: Any, *, unet_patcher: Any, text_patchers: Mapping[str, Any]) -> None:
    """Clear+refresh all LoRA patchers and persist empty state hash."""

    _clear_and_refresh_lora_state(unet_patcher, label="denoiser")
    for encoder_name, patcher in text_patchers.items():
        _clear_and_refresh_lora_state(patcher, label=f"text_encoders[{encoder_name!r}]")
    _set_engine_lora_hash(engine, hash_value="[]")


def _raise_apply_failure(
    *,
    operation: str,
    original_error: Exception,
    engine: Any,
    unet_patcher: Any,
    text_patchers: Mapping[str, Any],
) -> None:
    """Raise fail-loud failure with best-effort reset to empty state."""

    try:
        _reset_engine_lora_state(engine, unet_patcher=unet_patcher, text_patchers=text_patchers)
    except Exception as reset_error:  # noqa: BLE001
        raise RuntimeError(
            f"{operation} failed ({original_error}) and reset recovery failed; engine state is unreliable."
        ) from reset_error
    raise RuntimeError(
        f"{operation} failed ({original_error}); engine was reset to empty LoRA state and request was rejected."
    ) from original_error


def _build_to_load_maps(
    engine,
    *,
    text_encoder_key: str | None = None,
) -> Tuple[Dict[str, PatchTarget], Dict[str, PatchTarget]]:
    """Return LoRA-key → model-param maps for UNet and one text encoder."""
    unet_model = engine.codex_objects_after_applying_lora.denoiser.model
    text_encoders = engine.codex_objects_after_applying_lora.text_encoders
    if text_encoder_key is None:
        text_encoder_key, text_entry = _resolve_text_encoder_entry(text_encoders)
    else:
        text_entry = text_encoders.get(text_encoder_key) if isinstance(text_encoders, Mapping) else None
        if text_entry is None:
            raise RuntimeError(
                f"Engine does not expose required text_encoders[{text_encoder_key!r}] entry for LoRA key mapping."
            )
    text_model = _resolve_text_encoder_model(
        text_entry,
        label=f"text_encoders[{text_encoder_key!r}]",
    )
    unet_map = model_lora_keys_unet(unet_model)
    text_map = model_lora_keys_clip(text_model)
    return unet_map, text_map


def _collect_target_shape_by_key(model: Any) -> Dict[str, tuple[int, ...]]:
    state_dict = model.state_dict()
    return {str(key): tuple(int(dim) for dim in tensor.shape) for key, tensor in state_dict.items()}


def _run_header_only_preflight(
    *,
    path: str,
    unet_map: Mapping[str, PatchTarget],
    text_map: Mapping[str, PatchTarget],
    unet_target_shapes: Mapping[str, tuple[int, ...]],
    text_target_shapes: Mapping[str, tuple[int, ...]],
) -> None:
    suffix = Path(path).suffix.lower()
    if suffix not in {".safetensor", ".safetensors"}:
        return

    shape_map = read_safetensors_tensor_shapes(Path(path))
    unet_header = build_standard_shape_patch_dict_from_shape_map(shape_map, to_load=unet_map)
    text_header = build_standard_shape_patch_dict_from_shape_map(shape_map, to_load=text_map)

    if (
        not unet_header.requires_materialized_preflight
        and not text_header.requires_materialized_preflight
        and not unet_header.shape_patch_dict
        and not text_header.shape_patch_dict
    ):
        raise RuntimeError(
            "LoRA key layout mismatch: no compatible layers were found for "
            f"'{path}' on the active model keymap."
        )

    if unet_header.shape_patch_dict and not unet_header.requires_materialized_preflight:
        summary = validate_shape_patch_dict(
            unet_header.shape_patch_dict,
            target_shape_by_key=unet_target_shapes,
        )
        if summary.mismatches:
            raise RuntimeError(
                "LoRA structural preflight failed for '{path}' on the active denoiser. "
                "shape_compatible_targets={compatible}/{total}. samples={samples}".format(
                    path=path,
                    compatible=summary.compatible_targets,
                    total=summary.total_targets,
                    samples=format_shape_compatibility_samples(summary),
                )
            )

    if text_header.shape_patch_dict and not text_header.requires_materialized_preflight:
        summary = validate_shape_patch_dict(
            text_header.shape_patch_dict,
            target_shape_by_key=text_target_shapes,
        )
        if summary.mismatches:
            raise RuntimeError(
                "LoRA structural preflight failed for '{path}' on the active text encoder. "
                "shape_compatible_targets={compatible}/{total}. samples={samples}".format(
                    path=path,
                    compatible=summary.compatible_targets,
                    total=summary.total_targets,
                    samples=format_shape_compatibility_samples(summary),
                )
            )


def _validate_materialized_patch_dict(
    *,
    path: str,
    patch_dict: Dict[PatchTarget, tuple],
    target_shapes: Mapping[str, tuple[int, ...]],
    label: str,
) -> None:
    if not patch_dict:
        return
    summary = validate_shape_patch_dict(
        shapeify_patch_dict(patch_dict),
        target_shape_by_key=target_shapes,
    )
    if not summary.mismatches:
        return
    raise RuntimeError(
        "LoRA structural preflight failed for '{path}' on the active {label}. "
        "shape_compatible_targets={compatible}/{total}. samples={samples}".format(
            path=path,
            label=label,
            compatible=summary.compatible_targets,
            total=summary.total_targets,
            samples=format_shape_compatibility_samples(summary),
        )
    )


def _apply_patches(patcher, filename: str, patch_dict: Dict[PatchTarget, Any], strength: float, *, online_mode: bool) -> int:
    """Add patches to a ModelPatcher and return number of matched parameters."""
    # The patchers expect a flattened dict of model_key -> patch tuple(s)
    touched = 0
    if not patch_dict:
        return 0
    matched = patcher.add_patches(
        filename=filename,
        patches=patch_dict,
        strength_patch=float(strength),
        strength_model=1.0,
        online_mode=online_mode,
    )
    touched += len(matched)
    return touched


def apply_loras_to_engine(engine, selections: Iterable[dict | Any]) -> AppliedStats:
    """Apply a list of LoRA selections to the engine (UNet + primary text encoder).

    Each selection item must carry `path`, optional `weight` (text encoder/default), and optional `unet_weight`.
    """
    stats = AppliedStats()
    selected = list(selections or [])

    codex_objects = getattr(engine, "codex_objects_after_applying_lora", None)
    if codex_objects is None:
        raise RuntimeError("Engine is missing codex_objects_after_applying_lora required for LoRA application.")

    unet_patcher = getattr(codex_objects, "denoiser", None)
    text_encoders = getattr(codex_objects, "text_encoders", None)
    text_patchers = _collect_text_encoder_patchers(text_encoders)

    if not selected:
        if unet_patcher is None and not text_patchers:
            _set_engine_lora_hash(engine, hash_value="[]")
            return stats
        if unet_patcher is None or not text_patchers:
            raise RuntimeError(
                "Engine exposes partial LoRA patcher state for empty selection reset "
                "(expected denoiser and at least one text encoder patcher)."
            )
        try:
            _reset_engine_lora_state(engine, unet_patcher=unet_patcher, text_patchers=text_patchers)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LoRA empty-selection reset failed ({exc}); engine state is unreliable.") from exc
        return stats

    if unet_patcher is None:
        raise RuntimeError(
            "LoRA selections were provided, but the active engine does not expose required LoRA patchers "
            "(missing denoiser patcher)."
        )
    text_encoder_key, text_patcher = _resolve_primary_text_patcher(text_patchers)

    apply_mode = read_lora_apply_mode()
    online_mode = apply_mode == LoraApplyMode.ONLINE

    normalized_selected = _normalize_unique_selections(selected)
    mutation_started = False
    try:
        unet_map, text_map = _build_to_load_maps(engine, text_encoder_key=text_encoder_key)
        unet_target_shapes = _collect_target_shape_by_key(
            engine.codex_objects_after_applying_lora.denoiser.model
        )
        text_target_shapes = _collect_target_shape_by_key(
            _resolve_text_encoder_model(
                text_encoders[text_encoder_key],
                label=f"text_encoders[{text_encoder_key!r}]",
            )
        )
        prepared: list[tuple[str, float, float, Dict[PatchTarget, tuple], Dict[PatchTarget, tuple]]] = []
        for path, text_encoder_weight, unet_weight in normalized_selected:
            _run_header_only_preflight(
                path=path,
                unet_map=unet_map,
                text_map=text_map,
                unet_target_shapes=unet_target_shapes,
                text_target_shapes=text_target_shapes,
            )

            # Load weights once
            try:
                tensor_map = sf.load_file(path)
            except Exception as e:
                raise RuntimeError(f"Failed to load LoRA '{path}': {e}")

            # Build per-model patch dictionaries
            unet_patch, _ = load_lora(tensor_map, to_load=unet_map)
            text_patch, _ = load_lora(tensor_map, to_load=text_map)
            if not unet_patch and not text_patch:
                raise RuntimeError(
                    "LoRA key layout mismatch: no compatible layers were found for "
                    f"'{path}' on the active model keymap."
                )
            _validate_materialized_patch_dict(
                path=path,
                patch_dict=unet_patch,
                target_shapes=unet_target_shapes,
                label="denoiser",
            )
            _validate_materialized_patch_dict(
                path=path,
                patch_dict=text_patch,
                target_shapes=text_target_shapes,
                label="text encoder",
            )
            prepared.append((path, text_encoder_weight, unet_weight, unet_patch, text_patch))

        mutation_started = True
        # Single-owner semantics: each non-empty apply starts from a clean patch state.
        _clear_lora_state(unet_patcher, label="denoiser")
        for encoder_name, patcher in text_patchers.items():
            _clear_lora_state(patcher, label=f"text_encoders[{encoder_name!r}]")

        for path, text_encoder_weight, unet_weight, unet_patch, text_patch in prepared:
            unet_touched = _apply_patches(
                unet_patcher,
                filename=path,
                patch_dict=unet_patch,
                strength=unet_weight,
                online_mode=online_mode,
            )
            text_touched = _apply_patches(
                text_patcher,
                filename=path,
                patch_dict=text_patch,
                strength=text_encoder_weight,
                online_mode=online_mode,
            )
            touched_total = unet_touched + text_touched
            if touched_total <= 0:
                raise RuntimeError(
                    "LoRA apply mismatch: zero parameters were touched for "
                    f"'{path}'. Verify LoRA/base-model compatibility and key layout."
                )
            stats.params_touched += touched_total
            stats.files += 1

        # Materialize merges onto actual model parameters.
        _refresh_lora_state(unet_patcher, label="denoiser")
        for encoder_name, patcher in text_patchers.items():
            _refresh_lora_state(patcher, label=f"text_encoders[{encoder_name!r}]")
    except Exception as exc:  # noqa: BLE001
        if mutation_started:
            # Never leak partial/stale LoRA state across requests on failure after mutation starts.
            _raise_apply_failure(
                operation="LoRA apply execution",
                original_error=exc,
                engine=engine,
                unet_patcher=unet_patcher,
                text_patchers=text_patchers,
            )
        raise

    try:
        _set_engine_lora_hash(
            engine,
            hash_value=_serialize_selection_hash(normalized_selected, apply_mode=apply_mode),
        )
    except Exception as exc:  # noqa: BLE001
        _raise_apply_failure(
            operation="LoRA apply hash persistence",
            original_error=exc,
            engine=engine,
            unet_patcher=unet_patcher,
            text_patchers=text_patchers,
        )

    return stats


__all__ = ["apply_loras_to_engine", "AppliedStats", "selection_hash_for_request"]
