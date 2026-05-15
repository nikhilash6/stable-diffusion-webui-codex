"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux engine spec + runtime assembly (components + text pipelines + optional streaming core).
Defines the Flux engine runtime containers (denoiser/CLIP/T5/VAE + streaming policy) and assembles a runnable runtime from selected models,
with strict role validation (no implicit class-name fallbacks) and optional streamed core execution.

Symbols (top-level; keep in sync; no ghosts):
- `FluxTextPipelines` (dataclass): Holds the text processing engines used by Flux (optional CLIP classic + required T5).
- `FluxEngineRuntime` (dataclass): Fully assembled runtime components for Flux (CLIP, VAE, denoiser patcher, text pipelines, distilled CFG flag).
- `FluxEngineSpec` (dataclass): Spec/config holder for a Flux runtime build (repo/model selection + streaming policy/config).
- `_predictor` (function): Builds the FlowMatchEuler predictor (effective shift parameterization) for the selected Flux variant (Schnell vs dev).
- `_maybe_enable_streaming_core` (function): Wraps a core transformer with streaming support based on policy/config and runtime flags.
- `_is_clip_encoder` (function): Type guard for identifying CLIP text encoder models in a mixed component set.
- `_is_t5_encoder` (function): Type guard for identifying T5 text encoder models in a mixed component set.
- `assemble_flux_runtime` (function): Assembles a validated `FluxEngineRuntime` from selected components, applying device/memory policies
  and streaming options (contains nested helpers for controller setup and trace planning).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from apps.backend.runtime.families.flux.model import FluxTransformer2DModel
from apps.backend.runtime.families.flux.streaming import (
    StreamingConfig,
    StreamedFluxCore,
)
from apps.backend.patchers.clip import CLIP
from apps.backend.patchers.denoiser import DenoiserPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.sampling_adapters.prediction import FlowMatchEulerPrediction
from apps.backend.runtime.text_processing.classic_engine import ClassicTextProcessingEngine
from apps.backend.runtime.text_processing.t5_engine import T5TextProcessingEngine
from apps.backend.infra.config.args import dynamic_args

logger = get_backend_logger("backend.engines.flux.spec")


@dataclass(frozen=True)
class FluxTextPipelines:
    clip_text: Optional[ClassicTextProcessingEngine]
    t5_text: T5TextProcessingEngine


@dataclass(frozen=True)
class FluxEngineRuntime:
    clip: CLIP
    vae: VAE
    denoiser: DenoiserPatcher
    text: FluxTextPipelines
    use_distilled_cfg: bool

    def set_clip_skip(self, clip_skip: int) -> None:
        if self.text.clip_text is not None:
            if clip_skip == 0:
                clip_skip = 1
            if clip_skip < 1:
                raise ValueError("clip_skip must be >= 1 for Flux CLIP branch")
            self.text.clip_text.clip_skip = clip_skip


@dataclass(frozen=True)
class FluxEngineSpec:
    name: str
    uses_clip_branch: bool
    distilled_cfg_scale_default: float = 3.5
    schnell_threshold: Callable[[str], bool] | None = None

    def is_schnell(self, repo: str) -> bool:
        if self.schnell_threshold is None:
            return False
        return self.schnell_threshold(repo)


def _predictor(repo: str, is_schnell: bool) -> FlowMatchEulerPrediction:
    if is_schnell:
        logger.debug("Using FlowMatch predictor for schnell repo=%s", repo)
        return FlowMatchEulerPrediction(shift=1.0)
    logger.debug("Using FlowMatch predictor with seq_len scheduling for repo=%s", repo)
    return FlowMatchEulerPrediction(
        seq_len=4096,
        base_seq_len=256,
        max_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
    )


def _maybe_enable_streaming_core(
    transformer: object,
    *,
    spec: FluxEngineSpec,
    engine_options: Mapping[str, Any] | None,
) -> object:
    """Return the Flux core without streaming.

    Flux core streaming is temporarily disabled until the global, capability-driven
    streaming module contract is finalized.
    """
    if spec.name != "flux1":
        return transformer

    streamed: StreamedFluxCore | None = None
    if isinstance(transformer, StreamedFluxCore):
        streamed = transformer
    elif not isinstance(transformer, FluxTransformer2DModel):
        return transformer

    options = dict(engine_options or {})
    streaming_config = StreamingConfig.from_options(options)
    if streaming_config.enabled or streaming_config.auto_enable_threshold_mb > 0:
        raise NotImplementedError(
            "Flux core streaming is not implemented in the current capability contract. "
            "Disable streaming options and retry."
        )
    if streamed is not None:
        return streamed.base_core
    return transformer


def _is_clip_encoder(model: object) -> bool:
    """Detect if a text encoder is CLIP using structural shape only."""
    if model is None:
        return False
    if hasattr(model, "transformer"):
        transformer = model.transformer
        if hasattr(transformer, "text_model"):
            return True
    if hasattr(model, "text_model"):
        return True
    return False


def _is_t5_encoder(model: object) -> bool:
    """Detect if a text encoder is T5 using structural shape only."""
    if model is None:
        return False
    if hasattr(model, "transformer"):
        transformer = model.transformer
        if hasattr(transformer, "encoder") and hasattr(transformer.encoder, "block"):
            return True
    if hasattr(model, "encoder") and hasattr(model.encoder, "block"):
        return True
    return False


def assemble_flux_runtime(
    *,
    spec: FluxEngineSpec,
    estimated_config,
    codex_components: Mapping[str, object],
    engine_options: Mapping[str, Any] | None = None,
) -> FluxEngineRuntime:
    logger.debug("Assembling %s engine", spec.name)

    # Detect encoder types dynamically instead of assuming by slot position
    # This handles GGUF models that may have encoders in swapped positions
    te1 = codex_components.get("text_encoder")
    te2 = codex_components.get("text_encoder_2")
    tok1 = codex_components.get("tokenizer")
    tok2 = codex_components.get("tokenizer_2")
    
    if spec.uses_clip_branch:
        # Flux needs both CLIP and T5 - detect which is which by model structure
        encoder_slots = (te1, te2)
        tokenizer_slots = (tok1, tok2)
        clip_encoder_indices = [idx for idx, te in enumerate(encoder_slots) if _is_clip_encoder(te)]
        t5_encoder_indices = [idx for idx, te in enumerate(encoder_slots) if _is_t5_encoder(te)]

        if len(clip_encoder_indices) != 1 or len(t5_encoder_indices) != 1:
            raise RuntimeError(
                "Flux encoder role resolution failed: expected exactly one CLIP and one T5 encoder; "
                f"got clip_candidates={len(clip_encoder_indices)} t5_candidates={len(t5_encoder_indices)} "
                f"slot_types={[type(v).__name__ if v is not None else None for v in encoder_slots]}"
            )
        clip_slot = clip_encoder_indices[0]
        t5_slot = t5_encoder_indices[0]
        if clip_slot == t5_slot:
            raise RuntimeError("Flux encoder role resolution failed: CLIP and T5 resolved to the same slot.")

        clip_encoder = encoder_slots[clip_slot]
        t5_encoder = encoder_slots[t5_slot]
        if clip_encoder is t5_encoder:
            raise RuntimeError("Flux encoder role resolution failed: CLIP and T5 resolved to the same component.")

        clip_tokenizer = tokenizer_slots[clip_slot]
        t5_tokenizer = tokenizer_slots[t5_slot]
        if clip_tokenizer is None or t5_tokenizer is None:
            raise RuntimeError(
                "Flux tokenizer role resolution failed: expected tokenizer pair aligned with encoder slots; "
                f"resolved clip_slot={clip_slot} t5_slot={t5_slot} "
                f"slot_types={[type(v).__name__ if v is not None else None for v in tokenizer_slots]}"
            )
        if clip_tokenizer is t5_tokenizer:
            raise RuntimeError("Flux tokenizer role resolution failed: CLIP and T5 resolved to the same component.")

        logger.debug(
            "Encoder detection: CLIP=%s@slot%s (tok=%s) T5=%s@slot%s (tok=%s)",
            type(clip_encoder).__name__ if clip_encoder else None,
            clip_slot,
            type(clip_tokenizer).__name__ if clip_tokenizer else None,
            type(t5_encoder).__name__ if t5_encoder else None,
            t5_slot,
            type(t5_tokenizer).__name__ if t5_tokenizer else None,
        )
        
        model_dict = {"clip_l": clip_encoder, "t5xxl": t5_encoder}
        tokenizer_dict = {"clip_l": clip_tokenizer, "t5xxl": t5_tokenizer}
    else:
        # Chroma: only T5, no CLIP
        model_dict = {"t5xxl": te1}
        tokenizer_dict = {"t5xxl": tok1}

    clip = CLIP(model_dict=model_dict, tokenizer_dict=tokenizer_dict, model_config=estimated_config)
    vae_family = ModelFamily.FLUX if spec.name == "flux1" else ModelFamily.CHROMA
    vae = VAE(model=codex_components["vae"], family=vae_family)

    repo = getattr(estimated_config, "huggingface_repo", "" ) or ""
    schnell = spec.is_schnell(repo)
    predictor = _predictor(repo, schnell)
    if not schnell:
        logger.debug("Distilled CFG scale enabled for %s", spec.name)
    use_distilled_cfg = not schnell

    transformer = codex_components["transformer"]
    transformer = _maybe_enable_streaming_core(transformer, spec=spec, engine_options=engine_options)

    denoiser = DenoiserPatcher.from_model(
        model=transformer,
        diffusers_scheduler=None,
        predictor=predictor,
        config=estimated_config,
    )

    embedding_dir = dynamic_args["embedding_dir"]
    emphasis_name = dynamic_args["emphasis_name"]

    if spec.uses_clip_branch:
        clip_l = clip.cond_stage_model.clip_l
        tokenizer_l = clip.tokenizer.clip_l
        clip_engine = ClassicTextProcessingEngine(
            text_encoder=clip_l,
            tokenizer=tokenizer_l,
            embedding_dir=embedding_dir,
            embedding_key="clip_l",
            embedding_expected_shape=768,
            emphasis_name=emphasis_name,
            text_projection=False,
            minimal_clip_skip=1,
            clip_skip=1,
            return_pooled=True,
            final_layer_norm=True,
        )
    else:
        clip_engine = None

    t5_attr = "t5xxl"
    t5_encoder = getattr(clip.cond_stage_model, t5_attr)
    t5_tokenizer = getattr(clip.tokenizer, t5_attr)

    # Get t5_min_length from family spec
    from apps.backend.runtime.model_registry.family_runtime import FAMILY_RUNTIME_SPECS

    flux_family = ModelFamily.CHROMA if spec.name == "chroma" else ModelFamily.FLUX
    family_spec = FAMILY_RUNTIME_SPECS.get(flux_family)
    t5_min_len = family_spec.t5_min_length if family_spec and family_spec.t5_min_length else 256

    t5_engine = T5TextProcessingEngine(
        text_encoder=t5_encoder,
        tokenizer=t5_tokenizer,
        emphasis_name=emphasis_name,
        min_length=t5_min_len,
    )

    logger.debug("Flux runtime assembled (clip branch: %s, distilled cfg: %s)", spec.uses_clip_branch, use_distilled_cfg)

    return FluxEngineRuntime(
        clip=clip,
        vae=vae,
        denoiser=denoiser,
        text=FluxTextPipelines(clip_text=clip_engine, t5_text=t5_engine),
        use_distilled_cfg=use_distilled_cfg,
    )


FLUX_SPEC = FluxEngineSpec(
    name="flux1",
    uses_clip_branch=True,
    distilled_cfg_scale_default=3.5,
    schnell_threshold=lambda repo: "schnell" in repo.lower(),
)

CHROMA_SPEC = FluxEngineSpec(
    name="chroma",
    uses_clip_branch=False,
    distilled_cfg_scale_default=1.0,
)

__all__ = [
    "FluxTextPipelines",
    "FluxEngineRuntime",
    "FluxEngineSpec",
    "assemble_flux_runtime",
    "FLUX_SPEC",
    "CHROMA_SPEC",
]
