"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Common engine base helpers for diffusion runtimes (component bundles, loading hooks, smart offload/cache integration).
Defines `CodexObjects` and the shared engine load/unload path, including fail-fast core-only checkpoint validation and explicit
`vae_source`/`tenc_source` selection. Also provides default first-stage VAE encode/decode for image engines and canonical task wrappers that
delegate to mode use-cases (Option A) so engines stay adapters.
External-asset-first families (e.g., Z-Image and Anima) treat `vae_path` as selection rather than a state-dict override.
State-dict VAE overrides are normalized once before lane resolution so known SDXL/Flow metadata is stripped without reintroducing model-class heuristics.
Engine lifecycle logs (`load.start` / `load.complete` / `unload.start`) are emitted through the global runtime event emitter.
Patcher-backed component memory lifecycle (for example denoiser/VAE/CLIP-vision wrappers) uses canonical patcher-first targets so smart-offload bookkeeping does not fork records across wrapper and patcher identities.

Symbols (top-level; keep in sync; no ghosts):
- `CodexObjects` (dataclass): Container for core diffusion components (denoiser/VAE/text encoders + optional clipvision) with validate/describe helpers.
- `TextEncoderHandle` (dataclass): Canonical text-encoder contract (`patcher` required, optional runtime wrapper reference).
- `LoadOptions` (dataclass): Typed engine load-option contract with normalized selectors and passthrough extras.
- `EngineStatus` (TypedDict): Explicit status mapping returned by `CodexDiffusionEngine.status()`.
- `_ComponentTracker` (class): Internal tracker for loaded components/paths (used to decide reload/unload behavior).
- `CodexDiffusionEngine` (class): Abstract base class for diffusion engines; provides shared load/unload orchestration, canonical task wrappers
  (e.g. `txt2img` delegates to `apps/backend/use_cases/txt2img.py`), and runtime helpers including explicit asset-source selection
  (`vae_source`/`tenc_source`), default image VAE stage helpers (`encode_first_stage`/`decode_first_stage`), and fail-fast validation for
  core-only checkpoints (subclasses implement required component sets).
"""

from __future__ import annotations
from apps.backend.runtime.logging import BackendLoggerProxy, get_backend_logger

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence, TypedDict, cast

import safetensors.torch as sf
import torch

from apps.backend.core.engine_interface import BaseInferenceEngine
from apps.backend.runtime.memory.smart_offload import (
    smart_offload_enabled,
    smart_fallback_enabled,
    smart_cache_enabled,
    record_smart_cache_hit,
    record_smart_cache_miss,
)
from apps.backend.runtime.logging import emit_backend_event
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.common.vae_lane_policy import (
    detect_vae_layout,
    resolve_vae_layout_lane,
    uses_ldm_native_lane,
)
from apps.backend.runtime.common.vae import prepare_external_vae_override_state_dict
from apps.backend.runtime.common.vae_ldm import is_ldm_native_vae_instance
from apps.backend.runtime.models.loader import DiffusionModelBundle, resolve_diffusion_bundle
from apps.backend.runtime.models.text_encoder_overrides import TextEncoderOverrideConfig
from apps.backend.runtime.state_dict.tools import get_state_dict_after_quant
from apps.backend.runtime.checkpoint.io import load_torch_file
from apps.backend.runtime.models.state_dict import safe_load_state_dict


logger = get_backend_logger(__name__)


class EngineStatus(TypedDict, total=False):
    """Typed status mapping exposed by Codex diffusion engines."""

    engine_id: str
    loaded: bool
    model_ref: str
    bundle_source: str
    families: tuple[str, ...]


@dataclass(slots=True)
class LoadOptions:
    """Normalized engine load options with explicit typed fields."""

    tenc_source: Literal["built_in", "external"] = "built_in"
    tenc_path: str | list[str] | None = None
    text_encoder_override: TextEncoderOverrideConfig | None = None
    vae_path: str | None = None
    vae_source: Literal["built_in", "external"] | None = None
    checkpoint_core_only: bool | None = None
    model_format: Literal["checkpoint", "diffusers", "gguf"] | None = None
    core_streaming_enabled: bool | None = None
    extras: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw_options: Mapping[str, Any]) -> "LoadOptions":
        working: dict[str, object] = dict(raw_options)

        tenc_source_raw = working.pop("tenc_source", "built_in")
        if not isinstance(tenc_source_raw, str) or not tenc_source_raw.strip():
            raise TypeError("tenc_source must be a non-empty string when provided.")
        tenc_source = tenc_source_raw.strip().lower()
        if tenc_source not in {"built_in", "external"}:
            raise RuntimeError("tenc_source must be 'built_in' or 'external' when provided.")

        tenc_path_raw = working.pop("tenc_path", None)
        tenc_path: str | list[str] | None = None
        if isinstance(tenc_path_raw, str):
            tenc_path = tenc_path_raw.strip() or None
        elif isinstance(tenc_path_raw, list):
            cleaned: list[str] = []
            for entry in tenc_path_raw:
                if not isinstance(entry, str):
                    raise TypeError("tenc_path must be a string or array of strings when provided.")
                item = entry.strip()
                if item:
                    cleaned.append(item)
            tenc_path = cleaned or None
        elif tenc_path_raw is not None:
            raise TypeError("tenc_path must be a string or array of strings when provided.")

        text_encoder_override_raw = working.pop("text_encoder_override", None)
        text_encoder_override: TextEncoderOverrideConfig | None
        if text_encoder_override_raw is None:
            text_encoder_override = None
        elif isinstance(text_encoder_override_raw, TextEncoderOverrideConfig):
            text_encoder_override = text_encoder_override_raw
        else:
            raise TypeError("text_encoder_override must be TextEncoderOverrideConfig after normalization.")

        vae_path_raw = working.pop("vae_path", None)
        vae_path: str | None
        if isinstance(vae_path_raw, str):
            vae_path = vae_path_raw.strip() or None
        elif vae_path_raw is not None:
            raise TypeError("vae_path must be a non-empty string when provided.")
        else:
            vae_path = None

        vae_source_raw = working.pop("vae_source", None)
        vae_source: Literal["built_in", "external"] | None = None
        if isinstance(vae_source_raw, str) and vae_source_raw.strip():
            normalized_vae_source = vae_source_raw.strip().lower()
            if normalized_vae_source not in {"built_in", "external"}:
                raise RuntimeError("vae_source must be 'built_in' or 'external' when provided.")
            vae_source = normalized_vae_source
        elif vae_source_raw is not None and not isinstance(vae_source_raw, str):
            raise TypeError("vae_source must be a non-empty string when provided.")

        checkpoint_core_only_raw = working.pop("checkpoint_core_only", None)
        checkpoint_core_only: bool | None = None
        if checkpoint_core_only_raw is not None:
            if not isinstance(checkpoint_core_only_raw, bool):
                raise TypeError("checkpoint_core_only must be a boolean when provided.")
            checkpoint_core_only = checkpoint_core_only_raw

        model_format_raw = working.pop("model_format", None)
        model_format: Literal["checkpoint", "diffusers", "gguf"] | None = None
        if isinstance(model_format_raw, str) and model_format_raw.strip():
            normalized_model_format = model_format_raw.strip().lower()
            if normalized_model_format not in {"checkpoint", "diffusers", "gguf"}:
                raise RuntimeError("model_format must be one of: checkpoint, diffusers, gguf.")
            model_format = cast(Literal["checkpoint", "diffusers", "gguf"], normalized_model_format)
        elif model_format_raw is not None and not isinstance(model_format_raw, str):
            raise TypeError("model_format must be a non-empty string when provided.")

        if "core_streaming_enabled" in working:
            explicit_streaming_raw = working.pop("core_streaming_enabled")
            if not isinstance(explicit_streaming_raw, bool):
                raise TypeError("core_streaming_enabled must be a boolean when provided.")
            core_streaming_enabled: bool | None = explicit_streaming_raw
        else:
            core_streaming_enabled = None

        return cls(
            tenc_source=tenc_source,
            tenc_path=tenc_path,
            text_encoder_override=text_encoder_override,
            vae_path=vae_path,
            vae_source=vae_source,
            checkpoint_core_only=checkpoint_core_only,
            model_format=model_format,
            core_streaming_enabled=core_streaming_enabled,
            extras=working,
        )

    def to_component_options(self) -> dict[str, object]:
        options: dict[str, object] = dict(self.extras)
        options["tenc_source"] = self.tenc_source
        if self.tenc_path is not None:
            options["tenc_path"] = self.tenc_path
        if self.text_encoder_override is not None:
            options["text_encoder_override"] = self.text_encoder_override
        if self.vae_path is not None:
            options["vae_path"] = self.vae_path
        if self.vae_source is not None:
            options["vae_source"] = self.vae_source
        if self.checkpoint_core_only is not None:
            options["checkpoint_core_only"] = self.checkpoint_core_only
        if self.model_format is not None:
            options["model_format"] = self.model_format
        if self.core_streaming_enabled is not None:
            options["core_streaming_enabled"] = self.core_streaming_enabled
        return options


@dataclass(slots=True)
class TextEncoderHandle:
    """Canonical text-encoder contract bound in `CodexObjects`.

    `patcher` is the canonical memory-manager target and is mandatory.
    `runtime` optionally stores the engine-specific wrapper object.
    """

    patcher: Any
    runtime: Any | None = None

    def validate(self, *, context: str, encoder_name: str) -> None:
        if self.patcher is None:
            raise ValueError(f"{context}: text encoder '{encoder_name}' is missing required patcher.")
        if not hasattr(self.patcher, "codex_patch_model") or not hasattr(self.patcher, "codex_unpatch_model"):
            raise TypeError(
                f"{context}: text encoder '{encoder_name}' patcher must expose codex_patch_model/codex_unpatch_model."
            )


@dataclass(slots=True)
class CodexObjects:
    """Container for core diffusion components attached to an engine.
    
    text_encoders is a canonical mapping of logical encoder names to
    `TextEncoderHandle` values.
    """

    denoiser: Any
    vae: Any
    text_encoders: dict[str, TextEncoderHandle]
    clipvision: Any | None = None

    def shallow_copy(self) -> "CodexObjects":
        """Return a shallow copy preserving component references."""
        return CodexObjects(
            denoiser=self.denoiser,
            vae=self.vae,
            text_encoders=dict(self.text_encoders),  # Shallow copy of dict
            clipvision=self.clipvision,
        )

    def validate(self, context: str, *, required_text_encoders: tuple[str, ...] = ("clip",)) -> None:
        """Ensure all mandatory components are present.
        
        Args:
            context: Error message context.
            required_text_encoders: Tuple of required text encoder names.
        """
        if self.denoiser is None:
            raise ValueError(f"{context}: denoiser component is required.")
        if self.vae is None:
            raise ValueError(f"{context}: VAE component is required.")
        if not isinstance(self.text_encoders, dict):
            raise TypeError(f"{context}: text_encoders must be a dict[str, TextEncoderHandle].")
        for te_name, te_entry in self.text_encoders.items():
            if not isinstance(te_entry, TextEncoderHandle):
                raise TypeError(
                    f"{context}: text_encoders['{te_name}'] must be TextEncoderHandle "
                    f"(got {type(te_entry).__name__})."
                )
            te_entry.validate(context=context, encoder_name=str(te_name))
        for te_name in required_text_encoders:
            if te_name not in self.text_encoders or self.text_encoders[te_name] is None:
                raise ValueError(f"{context}: '{te_name}' text encoder is required.")

    def describe(self) -> dict[str, str]:
        """Return human-readable component metadata for logging."""
        def _name(component: Any) -> str:
            return component.__class__.__name__ if component is not None else "None"

        result = {
            "denoiser": _name(self.denoiser),
            "vae": _name(self.vae),
            "clipvision": _name(self.clipvision),
        }
        # Add text encoders to description
        for te_name, te_obj in self.text_encoders.items():
            result[f"text_encoder.{te_name}"] = _name(te_obj)
        return result


class _ComponentTracker:
    """Tracks active/original/LoRA component snapshots for an engine."""

    def __init__(self, *, logger: BackendLoggerProxy) -> None:
        self._logger = logger
        self._active: CodexObjects | None = None
        self._original: CodexObjects | None = None
        self._after_lora: CodexObjects | None = None

    @staticmethod
    def _ensure_codex_objects(value: Any, context: str) -> CodexObjects:
        if not isinstance(value, CodexObjects):
            raise TypeError(f"{context}: expected CodexObjects, received {type(value).__name__}.")
        return value

    def initialize(self, components: CodexObjects, *, context: str, required_text_encoders: tuple[str, ...] = ("clip",)) -> None:
        components = self._ensure_codex_objects(components, context)
        components.validate(context, required_text_encoders=required_text_encoders)
        self._active = components
        self._original = components.shallow_copy()
        self._after_lora = components.shallow_copy()
        snapshot = components.describe()
        self._logger.debug(
            "Engine components bound (%s): denoiser=%s vae=%s clipvision=%s text_encoders=%s",
            context,
            snapshot["denoiser"],
            snapshot["vae"],
            snapshot["clipvision"],
            list(components.text_encoders.keys()),
        )

    def replace_active(self, components: CodexObjects, *, context: str, required_text_encoders: tuple[str, ...] = ("clip",)) -> None:
        components = self._ensure_codex_objects(components, context)
        components.validate(context, required_text_encoders=required_text_encoders)
        self._active = components
        self._logger.debug(
            "Engine components replaced (%s): %s", context, components.describe()
        )

    def snapshot_after_lora(self) -> None:
        active = self.require_active()
        self._after_lora = active.shallow_copy()
        snapshot = self._after_lora.describe()
        self._logger.debug(
            "Stored post-LoRA snapshot: denoiser=%s vae=%s clipvision=%s text_encoders=%s",
            snapshot["denoiser"],
            snapshot["vae"],
            snapshot["clipvision"],
            list(active.text_encoders.keys()),
        )

    def set_after_lora(self, components: CodexObjects, *, context: str, required_text_encoders: tuple[str, ...] = ("clip",)) -> None:
        components = self._ensure_codex_objects(components, context)
        components.validate(context, required_text_encoders=required_text_encoders)
        self._after_lora = components
        self._logger.debug(
            "External post-LoRA snapshot registered (%s): %s",
            context,
            components.describe(),
        )

    def set_original(self, components: CodexObjects, *, context: str, required_text_encoders: tuple[str, ...] = ("clip",)) -> None:
        components = self._ensure_codex_objects(components, context)
        components.validate(context, required_text_encoders=required_text_encoders)
        self._original = components
        self._logger.debug(
            "Original component snapshot replaced (%s): %s",
            context,
            components.describe(),
        )

    def require_active(self) -> CodexObjects:
        if self._active is None:
            raise RuntimeError("Diffusion engine components have not been bound.")
        return self._active

    def require_original(self) -> CodexObjects:
        if self._original is None:
            raise RuntimeError("Original engine components are unavailable.")
        return self._original

    def require_after_lora(self) -> CodexObjects:
        if self._after_lora is None:
            raise RuntimeError("LoRA-applied engine components are unavailable.")
        return self._after_lora

    def peek_active(self) -> CodexObjects | None:
        return self._active


class CodexDiffusionEngine(BaseInferenceEngine, ABC):
    """Common foundation for Codex diffusion engines."""

    engine_id = "codex.diffusion"
    matched_guesses: tuple[str, ...] = ()
    expected_family: ModelFamily | None = None

    _MODEL_FAMILY_FLAGS: dict[str, str] = {
        "sd1": "is_sd1",
        "sd2": "is_sd2",
        "sd3": "is_sd3",
        "sdxl": "is_sdxl",
    }

    def __init__(self, *, logger: BackendLoggerProxy | None = None) -> None:  # noqa: D401
        super().__init__()
        self._logger = logger or get_backend_logger(__name__)
        self.model_config: object | None = None
        self.current_lora_hash = "[]"
        self._component_tracker = _ComponentTracker(logger=self._logger)
        self._model_families: set[str] = set()
        self._tiling_enabled = False
        self._use_distilled_cfg_scale = False
        self._component_source: Mapping[str, object] | None = None
        self._current_bundle: DiffusionModelBundle | None = None
        self._current_model_ref: str | None = None
        self._load_options: LoadOptions = LoadOptions()
        # Conditioning cache: keyed by a tuple -> arbitrary cache payload (typically tensors stored on CPU).
        # Subclasses can use this for caching conditioning outputs (CLIP/T5/Qwen3/etc).
        # Keep payload tensors on CPU to avoid pinning VRAM between jobs.
        self._cond_cache: dict[tuple, Any] = {}

    # ------------------------------------------------------------------ Components
    @property
    def required_text_encoders(self) -> tuple[str, ...]:
        """Text encoders required by this engine. Override in subclasses.
        
        Default is ("clip",) for SD/SDXL compatibility.
        Other engines can override, e.g., ("qwen3",) for Z Image.
        """
        return ("clip",)

    def bind_components(self, components: CodexObjects, *, label: str | None = None) -> None:
        """Bind engine components and seed original/LoRA snapshots."""
        context = label or self.__class__.__name__
        self._component_tracker.initialize(
            components, context=context, required_text_encoders=self.required_text_encoders
        )

    @property
    def codex_objects(self) -> CodexObjects:
        return self._component_tracker.require_active()

    @codex_objects.setter
    def codex_objects(self, value: CodexObjects) -> None:
        self._component_tracker.replace_active(
            value, context="codex_objects setter", required_text_encoders=self.required_text_encoders
        )

    @property
    def codex_objects_original(self) -> CodexObjects:
        return self._component_tracker.require_original()

    @codex_objects_original.setter
    def codex_objects_original(self, value: CodexObjects) -> None:
        self._component_tracker.set_original(
            value, context="codex_objects_original", required_text_encoders=self.required_text_encoders
        )

    @property
    def codex_objects_after_applying_lora(self) -> CodexObjects:
        return self._component_tracker.require_after_lora()

    @codex_objects_after_applying_lora.setter
    def codex_objects_after_applying_lora(self, value: CodexObjects) -> None:
        self._component_tracker.set_after_lora(
            value, context="codex_objects_after_applying_lora", required_text_encoders=self.required_text_encoders
        )

    @property
    def smart_offload_enabled(self) -> bool:
        # Resolve dynamically so UI toggles and per-request overrides take effect.
        return smart_offload_enabled()

    @property
    def smart_fallback_enabled(self) -> bool:
        return smart_fallback_enabled()

    @property
    def smart_cache_enabled(self) -> bool:
        return smart_cache_enabled()

    # ------------------------------------------------------------------ Conditioning Cache
    def _get_cached_cond(
        self,
        cache_key: tuple,
        *,
        bucket_name: str,
        enabled: bool | None = None,
    ) -> Any | None:
        """Retrieve cached conditioning if enabled and key exists (records cache hit/miss metrics)."""
        enabled = self.smart_cache_enabled if enabled is None else bool(enabled)
        if not enabled:
            return None
        cached = self._cond_cache.get(cache_key)
        if cached is not None:
            record_smart_cache_hit(bucket_name)
            return cached
        record_smart_cache_miss(bucket_name)
        return None

    def _set_cached_cond(self, cache_key: tuple, value: Any, *, enabled: bool | None = None) -> None:
        """Store conditioning in cache (payload tensors should be on CPU to avoid pinning VRAM)."""
        enabled = self.smart_cache_enabled if enabled is None else bool(enabled)
        if not enabled:
            return
        # Clear old entries to keep cache bounded
        self._cond_cache.clear()
        self._cond_cache[cache_key] = value

    def _clear_cond_cache(self) -> None:
        """Clear conditioning cache (called on model reload)."""
        self._cond_cache.clear()

    def snapshot_after_lora(self) -> None:
        """Capture the current components as the LoRA-applied snapshot."""
        self._component_tracker.snapshot_after_lora()

    # ------------------------------------------------------------------ Lifecycle
    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        if self._is_loaded:
            self.unload()
        raw_options: dict[str, Any] = dict(options)
        te_override_raw = raw_options.pop("text_encoder_override", None)
        bundle_obj = raw_options.pop("_bundle", None)

        # Text encoder selection/override is explicit via `tenc_source` + `tenc_path` (engine-side) or
        # `text_encoder_override` (loader-side). Core-only checkpoints require external text encoder
        # weights; full checkpoints default to built-in unless overridden.
        raw_tenc_path = raw_options.get("tenc_path")
        tenc_path: str | list[str] | None = None
        if isinstance(raw_tenc_path, str):
            tenc_path = raw_tenc_path.strip() or None
        elif isinstance(raw_tenc_path, (list, tuple)):
            cleaned: list[str] = []
            for entry in raw_tenc_path:
                if not isinstance(entry, str):
                    raise TypeError("tenc_path must be a string or array of strings when provided.")
                item = entry.strip()
                if item:
                    cleaned.append(item)
            tenc_path = cleaned or None
        elif raw_tenc_path is not None:
            raise TypeError("tenc_path must be a string or array of strings when provided.")

        if tenc_path is None:
            raw_options.pop("tenc_path", None)
        else:
            raw_options["tenc_path"] = tenc_path

        raw_tenc_source = raw_options.get("tenc_source")
        tenc_source = raw_tenc_source.strip().lower() if isinstance(raw_tenc_source, str) and raw_tenc_source.strip() else None
        external_tenc_config_present = (te_override_raw is not None) or (tenc_path is not None)
        if tenc_source is None:
            tenc_source = "external" if external_tenc_config_present else "built_in"
        if tenc_source not in {"built_in", "external"}:
            raise RuntimeError("tenc_source must be 'built_in' or 'external' when provided.")
        raw_options["tenc_source"] = tenc_source
        if tenc_source == "built_in":
            if external_tenc_config_present:
                raise RuntimeError(
                    "tenc_source='built_in' does not allow tenc_path/text_encoder_override; "
                    "remove them or set tenc_source='external'."
                )
        else:
            if not external_tenc_config_present:
                raise RuntimeError("tenc_source='external' requires tenc_path or text_encoder_override.")
        te_override_cfg: TextEncoderOverrideConfig | None = None
        if te_override_raw is not None:
            if not isinstance(te_override_raw, dict):
                raise TypeError("text_encoder_override must be a mapping when provided.")
            family_raw = str(te_override_raw.get("family") or "").strip()
            label_raw = str(te_override_raw.get("label") or "").strip()
            if not family_raw or not label_raw:
                raise RuntimeError("text_encoder_override requires non-empty 'family' and 'label' fields.")
            try:
                family = ModelFamily(family_raw)
            except ValueError as exc:
                raise RuntimeError(f"Unsupported text encoder override family='{family_raw}'") from exc
            components_val = te_override_raw.get("components")
            components: tuple[str, ...] | None
            explicit_paths: dict[str, str] | None = None
            if components_val is None:
                components = None
            elif isinstance(components_val, (list, tuple)):
                aliases: list[str] = []
                paths: dict[str, str] = {}
                for raw in components_val:
                    s = str(raw).strip()
                    if not s:
                        continue
                    # Support \"alias=path\" entries for explicit path overrides (e.g., Flux).
                    if "=" in s:
                        alias, path = s.split("=", 1)
                        alias = alias.strip()
                        path = path.strip()
                        if alias and path:
                            aliases.append(alias)
                            paths[alias] = path
                    else:
                        aliases.append(s)
                # De-duplicate aliases while preserving order.
                seen: set[str] = set()
                ordered_aliases: list[str] = []
                for alias in aliases:
                    if alias not in seen:
                        seen.add(alias)
                        ordered_aliases.append(alias)
                components = tuple(ordered_aliases) if ordered_aliases else None
                explicit_paths = paths or None
            else:
                raise RuntimeError("text_encoder_override.components must be an array of strings when provided.")
            te_override_cfg = TextEncoderOverrideConfig(
                family=family,
                root_label=label_raw,
                components=components,
                explicit_paths=explicit_paths,
            )
        if bundle_obj is None:
            vae_path_for_bundle = raw_options.get("vae_path")
            if not isinstance(vae_path_for_bundle, str) or not vae_path_for_bundle.strip():
                vae_path_for_bundle = None
            else:
                vae_path_for_bundle = vae_path_for_bundle.strip()
            expected_family = self.expected_family
            bundle = resolve_diffusion_bundle(
                model_ref,
                text_encoder_override=te_override_cfg,
                vae_path=vae_path_for_bundle,
                tenc_path=tenc_path,
                expected_family=expected_family,
            )
        elif isinstance(bundle_obj, DiffusionModelBundle):
            bundle = bundle_obj
        else:
            raise TypeError("_bundle must be a DiffusionModelBundle when provided.")

        # Per-family invariants for text encoder override payloads.
        if bundle.family is ModelFamily.ZIMAGE and isinstance(tenc_path, list):
            raise RuntimeError("Z Image supports exactly 1 text encoder; tenc_path must be a string.")
        if bundle.family is ModelFamily.ANIMA and isinstance(tenc_path, list):
            raise RuntimeError("Anima supports exactly 1 text encoder; tenc_path must be a string.")

        load_options_payload: dict[str, Any] = dict(raw_options)
        if te_override_cfg is not None:
            load_options_payload["text_encoder_override"] = te_override_cfg
        load_options = LoadOptions.from_mapping(load_options_payload)

        explicit_model_format = load_options.model_format
        if explicit_model_format is None:
            raise RuntimeError(
                "Image load requires explicit model_format in load options; "
                "router/task must forward the request-authoritative selector."
            )
        checkpoint_is_core_only = load_options.checkpoint_core_only
        if checkpoint_is_core_only is None:
            raise RuntimeError(
                "Image load requires explicit checkpoint_core_only in load options; "
                "router/task must forward the request-authoritative selector."
            )
        if explicit_model_format == "gguf" and checkpoint_is_core_only is False:
            raise RuntimeError("model_format='gguf' is incompatible with checkpoint_core_only=False.")
        if explicit_model_format == "diffusers" and checkpoint_is_core_only is True:
            raise RuntimeError("model_format='diffusers' is incompatible with checkpoint_core_only=True.")

        api_tenc_selector_hint = (
            "'extras.tenc1_sha'/'extras.tenc2_sha'"
            if bundle.family in {ModelFamily.SDXL, ModelFamily.SDXL_REFINER} and checkpoint_is_core_only
            else "'extras.tenc_sha'"
        )
        if checkpoint_is_core_only and self.required_text_encoders and tenc_source != "external":
            raise RuntimeError(
                "Core-only checkpoint requires external text encoder(s). "
                "Provide them via engine option 'tenc_path' or 'text_encoder_override' "
                f"(or via the API {api_tenc_selector_hint} selector)."
            )

        emit_backend_event(
            "engine.load.start",
            logger=self._logger.name,
            engine=self.engine_id,
            model_ref=model_ref,
            source=bundle.source,
        )
        self._reset_state()

        self._current_bundle = bundle
        self._current_model_ref = model_ref
        self._component_source = bundle.components
        self._load_options = load_options

        self.model_config = bundle.estimated_config

        components = self._build_components(bundle, options=self._load_options.to_component_options())

        # For core-only checkpoints, text encoders are never embedded; fail fast with a clear message.
        if checkpoint_is_core_only:
            te_map = getattr(components, "text_encoders", None)
            if not isinstance(te_map, dict):
                te_map = {}
            missing_tenc = [name for name in self.required_text_encoders if not te_map.get(name)]
            if missing_tenc:
                raise RuntimeError(
                    "Core-only checkpoint requires external text encoder(s). "
                    f"Missing: {', '.join(missing_tenc)}. "
                    "Provide them via engine option 'tenc_path' or 'text_encoder_override' "
                    f"(or via the API {api_tenc_selector_hint} selector)."
                )

        # VAE selection/override
        #
        # Rule of thumb:
        # - For core-only checkpoints, `vae_path` is an *external asset selection*,
        #   not a state-dict override. It must be handled during bundle resolution or
        #   engine-specific assembly.
        # - For full checkpoints, engines may treat `vae_path` as an optional state-dict
        #   override (SD/SDXL/Flux/etc). ZImage always treats it as external selection.
        raw_vae_path = self._load_options.vae_path
        vae_path = raw_vae_path.strip() if isinstance(raw_vae_path, str) and raw_vae_path.strip() else None

        vae_source = self._load_options.vae_source
        if vae_source is None:
            raise RuntimeError(
                "Image load requires explicit vae_source in load options; "
                "router/task must forward the request-authoritative selector."
            )
        if vae_source not in {"built_in", "external"}:
            raise RuntimeError("vae_source must be 'built_in' or 'external' when provided.")
        if vae_source == "built_in":
            if vae_path is not None:
                raise RuntimeError("vae_source='built_in' does not allow vae_path; remove vae_path or set vae_source='external'.")
        else:
            if vae_path is None:
                raise RuntimeError("vae_source='external' requires vae_path.")

        # For core-only checkpoints, VAE is never embedded; fail fast if the assembly stage didn't supply one.
        if checkpoint_is_core_only and getattr(components, "vae", None) is None:
            raise RuntimeError(
                "Core-only checkpoint requires an external VAE. "
                "Provide one via engine option 'vae_path' (or via the API 'extras.vae_sha' selector)."
            )

        # ZImage and Anima treat `vae_path` as external selection; do not apply a state-dict override here.
        if vae_source == "external" and vae_path and (bundle.family not in (ModelFamily.ZIMAGE, ModelFamily.ANIMA)) and (not checkpoint_is_core_only):
            if not os.path.isfile(vae_path):
                raise FileNotFoundError(f"vae_path '{vae_path}' does not exist.")
            if getattr(components, "vae", None) is None:
                raise RuntimeError(
                    "vae_path was provided, but no VAE component was built from the checkpoint; "
                    "cannot apply override."
                )
            vae_device = getattr(components.vae, "device", None) or getattr(components.vae, "load_device", None)
            state_dict = prepare_external_vae_override_state_dict(
                state_dict=load_torch_file(vae_path, device=vae_device),
                family=bundle.family,
            )
            vae_layout = detect_vae_layout(state_dict)
            vae_lane = resolve_vae_layout_lane(family=bundle.family, layout=vae_layout)
            vae_target = getattr(components.vae, "first_stage_model", components.vae)
            vae_base = getattr(vae_target, "_base", vae_target)
            if uses_ldm_native_lane(vae_lane):
                if not is_ldm_native_vae_instance(vae_base):
                    raise RuntimeError(
                        "VAE override selected native LDM lane, but current VAE component is diffusers-native. "
                        "Reload the checkpoint with matching lane policy or set CODEX_VAE_LAYOUT_LANE=diffusers_native "
                        "when using diffusers-style VAE overrides."
                    )
            else:
                if bundle.family in (
                    ModelFamily.SDXL,
                    ModelFamily.SDXL_REFINER,
                    ModelFamily.FLUX,
                    ModelFamily.FLUX_KONTEXT,
                    ModelFamily.ZIMAGE,
                ):
                    from apps.backend.runtime.state_dict.keymap_sdxl_vae import resolve_sdxl_vae_keyspace

                    state_dict = resolve_sdxl_vae_keyspace(state_dict).view
            if not hasattr(vae_target, "state_dict"):
                raise TypeError(
                    "VAE override target does not expose state_dict(); "
                    f"got {type(vae_target).__name__} (from {type(components.vae).__name__})."
                )
            missing, unexpected = safe_load_state_dict(vae_target, state_dict, log_name="VAE override")
            if missing:
                sample = missing[:10]
                family = getattr(bundle, "family", None)
                family_hint = None
                try:
                    family_hint = family.value  # type: ignore[attr-defined]
                except Exception:
                    family_hint = str(family) if family is not None else None
                raise RuntimeError(
                    f"VAE override is missing {len(missing)} keys; sample={sample}. "
                    f"Ensure the override matches the expected VAE architecture for {family_hint or 'this checkpoint'}."
                )
            if unexpected:
                if bundle.family in (ModelFamily.SDXL, ModelFamily.SDXL_REFINER):
                    raise RuntimeError(
                        "VAE override produced unexpected keys for SDXL. "
                        "This indicates a keymap/conversion mismatch; refusing to continue. "
                        f"unexpected_count={len(unexpected)} sample={unexpected[:10]}"
                    )
                logger.warning("VAE override: unexpected %d keys (sample=%s)", len(unexpected), unexpected[:10])

        # Optional: best-effort probe for internal diagnostics (no console output)
        def _probe_device_dtype(obj):
            try:
                # Prefer nested .model/.diffusion_model when available
                candidate = getattr(obj, 'model', obj)
                candidate = getattr(candidate, 'diffusion_model', candidate)
                params = getattr(candidate, 'parameters', None)
                if callable(params):
                    it = params()
                    t = next(it)
                    return getattr(t, 'device', None), getattr(t, 'dtype', None)
            except Exception:
                return None, None
            return getattr(candidate, 'device', None), getattr(candidate, 'dtype', None)
        _ = _probe_device_dtype(getattr(components, 'denoiser', None))
        _ = _probe_device_dtype(getattr(components, 'clip', None))
        _ = _probe_device_dtype(getattr(components, 'vae', None))
        self.bind_components(components, label=self.engine_id)
        self.snapshot_after_lora()

        self.mark_loaded()
        emit_backend_event(
            "engine.load.complete",
            logger=self._logger.name,
            engine=self.engine_id,
            families=",".join(sorted(self._model_families)) or "unknown",
        )

    def unload(self) -> None:  # type: ignore[override]
        if not self._is_loaded:
            return
        emit_backend_event(
            "engine.unload.start",
            logger=self._logger.name,
            engine=self.engine_id,
        )
        try:
            self._unload_bound_models_from_memory_manager()
        except Exception:
            self._logger.debug("Failed to unload bound models from memory manager", exc_info=True)
        try:
            self._on_unload()
        finally:
            self._reset_state()
            self.model_config = None
            self._component_source = None
            self._current_bundle = None
            self._current_model_ref = None
            self._load_options = LoadOptions()
            self.mark_unloaded()

    def _unload_bound_models_from_memory_manager(self) -> None:
        """Best-effort unload of currently bound heavy components.

        The orchestrator caches engine instances, and engines may be reloaded
        when model refs or load-affecting options change. When that happens we
        must drop references held by the memory manager (loaded-model records),
        otherwise repeated reloads can accumulate duplicate model instances.
        """

        from apps.backend.runtime.memory import memory_management

        components = self._component_tracker.peek_active()
        if components is None:
            return

        targets: list[object] = []
        # Core components may be wrappers around patchers; use canonical patcher-first targets.
        targets.append(self._canonical_patcher_target(getattr(components, "denoiser", None)))
        targets.append(self._canonical_patcher_target(getattr(components, "vae", None)))
        targets.append(self._canonical_patcher_target(getattr(components, "clipvision", None)))

        # Text encoders vary by engine; prefer `.patcher` when present.
        try:
            for obj in (getattr(components, "text_encoders", {}) or {}).values():
                if obj is None:
                    continue
                targets.append(self._canonical_patcher_target(obj))
        except Exception:
            pass

        for target in targets:
            if target is None:
                continue
            try:
                memory_management.manager.unload_model_clones(target)
            except Exception:  # noqa: BLE001
                self._logger.debug("Failed to unload model clones for %s", type(target).__name__, exc_info=True)
            try:
                memory_management.manager.unload_model(target)
            except Exception:  # noqa: BLE001
                self._logger.debug("Failed to unload model for %s", type(target).__name__, exc_info=True)

    def _reset_state(self) -> None:
        self._component_tracker = _ComponentTracker(logger=self._logger)
        self._model_families.clear()
        self._tiling_enabled = False
        self._use_distilled_cfg_scale = False
        self.current_lora_hash = "[]"
        self._cond_cache.clear()

    @abstractmethod
    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, object],
    ) -> CodexObjects:
        """Construct CodexObjects for the provided bundle."""

    def _on_unload(self) -> None:
        """Subclass hook to release additional state on unload."""
        return None

    @property
    def model_ref(self) -> Optional[str]:
        return self._current_model_ref

    def status(self) -> EngineStatus:  # type: ignore[override]
        data = dict(super().status())
        raw_engine_id = data.get("engine_id", self.engine_id)
        if not isinstance(raw_engine_id, str):
            raise RuntimeError("Base engine status must provide 'engine_id' as a string.")
        engine_id = raw_engine_id.strip()
        if not engine_id:
            raise RuntimeError("Base engine status must provide a non-empty 'engine_id'.")
        data["engine_id"] = engine_id

        raw_loaded = data.get("loaded", self._is_loaded)
        if not isinstance(raw_loaded, bool):
            raise RuntimeError("Base engine status must provide 'loaded' as a boolean.")
        data["loaded"] = raw_loaded
        if self._current_model_ref is not None:
            data["model_ref"] = self._current_model_ref
        if self._current_bundle is not None:
            data["bundle_source"] = self._current_bundle.source
        if self._model_families:
            data["families"] = tuple(sorted(self._model_families))
        return cast(EngineStatus, data)

    # ------------------------------------------------------------------ Model families
    def register_model_family(self, family: str) -> None:
        """Tag the engine with a known model family (sd1/sd2/sd3/sdxl)."""
        if family not in self._MODEL_FAMILY_FLAGS:
            raise ValueError(f"Unsupported model family '{family}'.")
        self._model_families.add(family)
        self._logger.debug(
            "Model family registered: %s (now %s)",
            family,
            sorted(self._model_families),
        )

    @property
    def model_families(self) -> Sequence[str]:
        return tuple(sorted(self._model_families))

    @property
    def is_sd1(self) -> bool:
        return "sd1" in self._model_families

    @property
    def is_sd2(self) -> bool:
        return "sd2" in self._model_families

    @property
    def is_sd3(self) -> bool:
        return "sd3" in self._model_families

    @property
    def is_sdxl(self) -> bool:
        return "sdxl" in self._model_families

    def is_webui_legacy_model(self) -> bool:
        legacy = {"sd1", "sd2", "sd3", "sdxl"}
        return any(family in legacy for family in self._model_families)

    # ------------------------------------------------------------------ Engine flags
    @property
    def use_distilled_cfg_scale(self) -> bool:
        return self._use_distilled_cfg_scale

    @use_distilled_cfg_scale.setter
    def use_distilled_cfg_scale(self, enabled: bool) -> None:
        self._use_distilled_cfg_scale = bool(enabled)
        self._logger.debug("Distilled CFG scale toggled to %s", self._use_distilled_cfg_scale)

    # ------------------------------------------------------------------ Engine hooks
    def set_clip_skip(self, clip_skip: int) -> None:
        """Set CLIP skip layers for engines that use CLIP.

        Default behavior:
        - Engines that do *not* include `"clip"` in `required_text_encoders` treat this as a no-op.
        - Engines that *do* use CLIP must override and implement clip-skip (fail loud by default).
        """
        try:
            requested = int(clip_skip)
        except Exception as exc:  # noqa: BLE001
            raise TypeError("clip_skip must be an integer") from exc
        if requested < 0:
            raise ValueError("clip_skip must be >= 0")

        if "clip" not in self.required_text_encoders:
            if requested != 0:
                self._logger.debug(
                    "Ignoring clip_skip=%d for engine_id=%s (no CLIP text encoder).",
                    requested,
                    self.engine_id,
                )
            return

        raise NotImplementedError(f"{self.__class__.__name__}.set_clip_skip must be implemented.")

    def get_first_stage_encoding(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_learned_conditioning(self, prompt: Iterable[str]) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__}.get_learned_conditioning must be implemented.")

    def _decode_debug_stats_enabled(self) -> bool:
        return False

    def _log_decode_stats(self, stage: str, tensor: torch.Tensor) -> None:
        return None

    @staticmethod
    def _canonical_patcher_target(component: object | None) -> object | None:
        """Return canonical memory target (prefer patcher identity when available)."""
        if component is None:
            return None
        patcher = getattr(component, "patcher", None)
        return patcher if patcher is not None else component

    def _vae_memory_target(self) -> object:
        """Return the canonical memory-manager target for first-stage VAE lifecycle."""
        return self._canonical_patcher_target(self.codex_objects.vae)

    def _denoiser_memory_target(self) -> object | None:
        """Return the canonical memory-manager target for denoiser lifecycle."""
        return self._canonical_patcher_target(getattr(self.codex_objects, "denoiser", None))

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor, *, encode_seed: int | None = None) -> torch.Tensor:
        from apps.backend.runtime.memory import memory_management

        vae_target = self._vae_memory_target()
        memory_management.manager.load_model(vae_target)
        unload_vae = self.smart_offload_enabled
        try:
            sample = self.codex_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5, encode_seed=encode_seed)
            sample = self.codex_objects.vae.first_stage_model.process_in(sample)
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.manager.unload_model(vae_target)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        from apps.backend.runtime.memory import memory_management

        vae_target = self._vae_memory_target()
        denoiser_target = self._denoiser_memory_target()
        preserve_denoiser_residency = (
            denoiser_target is not None
            and denoiser_target is not vae_target
            and memory_management.manager.is_model_loaded(denoiser_target)
        )
        if preserve_denoiser_residency:
            memory_management.manager.load_models(
                [vae_target, denoiser_target],
                source="engines.common.base.decode_first_stage",
                stage="decode_first_stage",
                component_hint="vae",
                event_reason="preserve_denoiser_residency",
            )
        else:
            memory_management.manager.load_model(
                vae_target,
                source="engines.common.base.decode_first_stage",
                stage="decode_first_stage",
                component_hint="vae",
            )
        unload_vae = self.smart_offload_enabled
        debug_stats = self._decode_debug_stats_enabled()
        try:
            if debug_stats:
                self._log_decode_stats("latents", x)
            sample = self.codex_objects.vae.first_stage_model.process_out(x)
            if debug_stats:
                self._log_decode_stats("after_process_out", sample)
            sample = self.codex_objects.vae.decode(sample)
            if debug_stats:
                self._log_decode_stats("decoded", sample)
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.manager.unload_model(
                    vae_target,
                    source="engines.common.base.decode_first_stage",
                    stage="decode_first_stage",
                    component_hint="vae",
                )

    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        raise NotImplementedError(f"{self.__class__.__name__}.get_prompt_lengths_on_ui must be implemented.")

    # ------------------------------------------------------------------ Tasks
    def txt2img(self, request: Any, **kwargs: Any) -> Iterable[Any]:
        """Canonical txt2img wrapper (delegates to the use-case)."""

        from apps.backend.use_cases.txt2img import run_txt2img as _run_txt2img

        yield from _run_txt2img(engine=self, request=request)

    def img2img(self, request: Any, **kwargs: Any) -> Iterable[Any]:
        """Canonical img2img wrapper (delegates to the use-case)."""

        from apps.backend.use_cases.img2img import run_img2img as _run_img2img

        yield from _run_img2img(engine=self, request=request)

    def _post_txt2img_cleanup(self) -> None:
        """Post-job cleanup when smart offload is enabled.

        Keeps the denoiser resident but nudges CUDA to release unused cached memory so the
        next job starts from a clean allocator state without paying reload cost.
        """
        if not self.smart_offload_enabled:
            return
        try:
            from apps.backend.runtime.memory import memory_management
            memory_management.manager.soft_empty_cache(force=True)
        except Exception:  # pragma: no cover - diagnostics only
            self._logger.debug("Post-job cleanup failed", exc_info=True)


    # ------------------------------------------------------------------ Persistence helpers
    def save_unet(self, filename: str) -> str:
        """Persist the current UNet weights to a safetensors file."""
        components = self.codex_objects
        unet = getattr(components.denoiser, "model", None)
        if unet is None:
            raise RuntimeError("UNet patcher is unavailable; cannot export weights.")
        diffusion = getattr(unet, "diffusion_model", None)
        if diffusion is None:
            raise RuntimeError("UNet diffusion model missing; export aborted.")

        state_dict = get_state_dict_after_quant(diffusion)
        sf.save_file(state_dict, filename)
        self._logger.info("UNet weights saved to %s (%d tensors)", filename, len(state_dict))
        return filename


__all__ = ["CodexObjects", "CodexDiffusionEngine"]
