"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LTX2 checkpoint detector and GGUF contract inspector for the model registry.
Matches strict monolithic LTX 2.x combined checkpoints using explicit local evidence:
`model.diffusion_model.*` transformer+connector keys, plus `vae.*`, `audio_vae.*`, and `vocoder.*` component prefixes.
Also exposes header-safe inspection for LTX2 core-only GGUF checkpoints so inventory/preflight can reject connector-contaminated
transformer packs before the parser/load path. Grounds expected architecture dimensions in vendored
`apps/backend/huggingface/Lightricks/LTX-2/**` metadata and returns a structured backend-only `ModelSignature`
without exposing runtime engine/capability support.

Symbols (top-level; keep in sync; no ghosts):
- `_LTX2_CONFIG_REPO_HINT` (constant): Canonical vendored Hugging Face repo id used for LTX2 metadata/config hints.
- `_LTX23_CHECKPOINT_REPO_HINT` (constant): Official monolithic LTX 2.3 checkpoint repo id.
- `_MONOLITHIC_CORE_PREFIX` (constant): Combined-checkpoint prefix for the LTX2 transformer surface.
- `_MONOLITHIC_COMPONENT_MIN_KEYS` (constant): Minimum per-component key-count thresholds required for strict monolithic matching.
- `_LTX2_REQUIRED_CORE_KEYSETS` (constant): Accepted LTX2 transformer marker key sets under `model.diffusion_model.`.
- `_CONNECTOR_SUBPREFIXES` (constant): Monolithic-only connector key prefixes used for strict LTX2 checkpoint evidence.
- `Ltx2GgufCoreMetadata` (dataclass): Header-safe LTX2 GGUF core-only contract inspection result.
- `_GGUFHeaderTensorRef` (dataclass): Lightweight GGUF header tensor record used for key-only inspection.
- `_GGUFHeaderStateDict` (class): Mapping wrapper exposing GGUF tensor names without materializing weights.
- `Ltx2Detector` (class): Detector that matches strict monolithic LTX2 checkpoints and builds an LTX2 `ModelSignature`.
- `inspect_ltx2_gguf_path` (function): Returns header-safe LTX2 GGUF core-only contract metadata from a GGUF path.
- `_supported_ltx2_metadata` (function): Loads and validates required local LTX2 metadata JSON files.
- `_read_json` (function): Reads a required vendored JSON metadata file.
- `_repo_hint_from_bundle` (function): Infers the best source checkpoint repo hint from the checkpoint filepath when available.
- `_shape_dim` (function): Shape helper (bundle metadata first, tensor fallback).
- `_count_prefixed_keys` (function): Counts keys with a given prefix.
- `_has_connector_surface` (function): Returns True when connector evidence exists in the monolithic checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional
import json

import torch

from apps.backend.runtime.model_parser.families.ltx2 import split_ltx2_transformer_and_connectors_state
from apps.backend.runtime.model_registry.detectors.base import ModelDetector, REGISTRY
from apps.backend.runtime.model_registry.signals import SignalBundle, count_blocks, has_all_keys
from apps.backend.runtime.model_registry.specs import (
    CodexCoreArchitecture,
    CodexCoreSignature,
    LatentFormat,
    ModelFamily,
    ModelSignature,
    PredictionKind,
    QuantizationHint,
    TextEncoderSignature,
    VAESignature,
)

_LTX2_CONFIG_REPO_HINT = "Lightricks/LTX-2"
_LTX23_CHECKPOINT_REPO_HINT = "Lightricks/LTX-2.3"
_MONOLITHIC_CORE_PREFIX = "model.diffusion_model."
_VAE_PREFIX = "vae."
_AUDIO_VAE_PREFIX = "audio_vae."
_VOCODER_PREFIX = "vocoder."

_MONOLITHIC_COMPONENT_MIN_KEYS = {
    _MONOLITHIC_CORE_PREFIX: 256,
    _VAE_PREFIX: 128,
    _AUDIO_VAE_PREFIX: 32,
    _VOCODER_PREFIX: 32,
}

_LTX2_REQUIRED_CORE_KEYSETS = (
    (
        "adaln_single.emb.timestep_embedder.linear_1.bias",
        "patchify_proj.weight",
        "transformer_blocks.0.attn2.to_k.weight",
    ),
    (
        "av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1.weight",
        "patchify_proj.weight",
        "transformer_blocks.0.attn2.to_k.weight",
    ),
)

_CONNECTOR_SUBPREFIXES = (
    _MONOLITHIC_CORE_PREFIX + "video_embeddings_connector.",
    _MONOLITHIC_CORE_PREFIX + "audio_embeddings_connector.",
    _MONOLITHIC_CORE_PREFIX + "transformer_1d_blocks.",
    _MONOLITHIC_CORE_PREFIX + "text_embedding_projection.aggregate_embed.",
    _MONOLITHIC_CORE_PREFIX + "connectors.",
    _MONOLITHIC_CORE_PREFIX + "video_connector.",
    _MONOLITHIC_CORE_PREFIX + "audio_connector.",
    _MONOLITHIC_CORE_PREFIX + "text_proj_in.",
)


@dataclass(frozen=True)
class Ltx2GgufCoreMetadata:
    has_required_transformer_markers: bool
    transformer_key_count: int
    connector_key_count: int
    connector_key_sample: tuple[str, ...]


@dataclass(frozen=True)
class _GGUFHeaderTensorRef:
    shape: tuple[int, ...]


class _GGUFHeaderStateDict(Mapping[str, _GGUFHeaderTensorRef]):
    def __init__(self, gguf_path: str) -> None:
        from apps.backend.quantization.gguf import GGUFReader

        reader = GGUFReader(gguf_path)
        self._tensors = {
            tensor.name: _GGUFHeaderTensorRef(tuple(int(v) for v in reversed(tensor.shape.tolist())))
            for tensor in reader.tensors
        }
        self.filepath = gguf_path
        self.source_format = "gguf"

    def __getitem__(self, key: str) -> _GGUFHeaderTensorRef:
        return self._tensors[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    def shape_of(self, key: str) -> tuple[int, ...] | None:
        tensor = self._tensors.get(key)
        return None if tensor is None else tensor.shape


class Ltx2Detector(ModelDetector):
    priority = 168

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        if bundle.is_gguf_quantized():
            return False

        source_format = str(bundle.source_format or "").strip().lower()
        if source_format and source_format not in {"safetensors", "safetensor"}:
            return False

        if not any(
            has_all_keys(bundle, *(_MONOLITHIC_CORE_PREFIX + key for key in keyset))
            for keyset in _LTX2_REQUIRED_CORE_KEYSETS
        ):
            return False
        if not _has_connector_surface(bundle.keys):
            return False
        for component_prefix, min_keys in _MONOLITHIC_COMPONENT_MIN_KEYS.items():
            if _count_prefixed_keys(bundle.keys, component_prefix) < min_keys:
                return False

        metadata = _supported_ltx2_metadata()
        transformer_cfg = metadata["transformer"]

        to_k_key = _MONOLITHIC_CORE_PREFIX + "transformer_blocks.0.attn2.to_k.weight"
        patchify_key = _MONOLITHIC_CORE_PREFIX + "patchify_proj.weight"

        hidden_dim = _shape_dim(bundle, to_k_key, dim=0)
        context_dim = _shape_dim(bundle, to_k_key, dim=1)
        channels_in = _shape_dim(bundle, patchify_key, dim=1)
        layer_count = count_blocks(bundle.keys, _MONOLITHIC_CORE_PREFIX + "transformer_blocks.{}.")

        expected_hidden = int(transformer_cfg["attention_head_dim"]) * int(transformer_cfg["num_attention_heads"])
        expected_context = int(transformer_cfg["cross_attention_dim"])
        expected_in = int(transformer_cfg["in_channels"])
        expected_layers = int(transformer_cfg["num_layers"])

        return (
            hidden_dim == expected_hidden
            and context_dim == expected_context
            and channels_in == expected_in
            and layer_count == expected_layers
        )

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        metadata = _supported_ltx2_metadata()
        transformer_cfg = metadata["transformer"]
        vae_cfg = metadata["vae"]
        text_encoder_cfg = metadata["text_encoder"]
        repo_hint = _repo_hint_from_bundle(bundle)

        transformer_layers = int(transformer_cfg["num_layers"])
        in_channels = int(transformer_cfg["in_channels"])
        out_channels = int(transformer_cfg["out_channels"])
        cross_attention_dim = int(transformer_cfg["cross_attention_dim"])
        caption_channels = int(transformer_cfg["caption_channels"])
        vae_latent_channels = int(vae_cfg["latent_channels"])
        gemma_hidden = int(text_encoder_cfg["text_hidden_size"])
        has_audio_adaln = _MONOLITHIC_CORE_PREFIX + "audio_adaln_single.linear.weight" in bundle.state_dict

        return ModelSignature(
            family=ModelFamily.LTX2,
            repo_hint=repo_hint,
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.LTX2,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.FLOW_TRANSFORMER,
                channels_in=in_channels,
                channels_out=out_channels,
                context_dim=cross_attention_dim,
                temporal=True,
                depth=transformer_layers,
                key_prefixes=[_MONOLITHIC_CORE_PREFIX + "transformer_blocks."],
            ),
            text_encoders=[
                TextEncoderSignature(
                    name="gemma3_12b",
                    key_prefix="text_encoder.",
                    expected_dim=gemma_hidden,
                    tokenizer_hint=f"{_LTX2_CONFIG_REPO_HINT}/tokenizer",
                )
            ],
            vae=VAESignature(key_prefix=_VAE_PREFIX, latent_channels=vae_latent_channels),
            extras={
                "asset_repo_id": _LTX2_CONFIG_REPO_HINT,
                "monolithic_combined_checkpoint": True,
                "transformer_layers": transformer_layers,
                "caption_channels": caption_channels,
                "has_audio_transformer_path": has_audio_adaln,
                "source_checkpoint_repo_id": repo_hint or "",
                "component_prefixes": [
                    _MONOLITHIC_CORE_PREFIX,
                    _VAE_PREFIX,
                    _AUDIO_VAE_PREFIX,
                    _VOCODER_PREFIX,
                ],
                "connector_subprefixes": list(_CONNECTOR_SUBPREFIXES),
                "signature_source": "vendored_hf",
            },
        )


@lru_cache(maxsize=1)
def _supported_ltx2_metadata() -> dict[str, Mapping[str, Any]]:
    backend_root = Path(__file__).resolve().parents[3]
    hf_root = backend_root / "huggingface" / "Lightricks" / "LTX-2"

    model_index = _read_json(hf_root / "model_index.json")
    transformer_cfg = _read_json(hf_root / "transformer" / "config.json")
    connectors_cfg = _read_json(hf_root / "connectors" / "config.json")
    vae_cfg = _read_json(hf_root / "vae" / "config.json")
    audio_vae_cfg = _read_json(hf_root / "audio_vae" / "config.json")
    text_encoder_cfg = _read_json(hf_root / "text_encoder" / "config.json")
    vocoder_cfg = _read_json(hf_root / "vocoder" / "config.json")

    expected_components = {"transformer", "connectors", "vae", "audio_vae", "vocoder", "text_encoder"}
    missing_components = [name for name in expected_components if name not in model_index]
    if missing_components:
        raise RuntimeError(
            "Vendored LTX2 model_index is missing required component entries: "
            f"{sorted(missing_components)}."
        )

    if int(transformer_cfg.get("in_channels", 0)) != 128 or int(transformer_cfg.get("out_channels", 0)) != 128:
        raise RuntimeError(
            "Vendored LTX2 transformer config must declare in/out channels = 128 "
            f"(got in={transformer_cfg.get('in_channels')!r}, out={transformer_cfg.get('out_channels')!r})."
        )
    if int(transformer_cfg.get("num_layers", 0)) != 48:
        raise RuntimeError(
            "Vendored LTX2 transformer layer count drifted (expected 48, "
            f"got {transformer_cfg.get('num_layers')!r})."
        )
    if int(transformer_cfg.get("cross_attention_dim", 0)) != 4096:
        raise RuntimeError(
            "Vendored LTX2 transformer cross_attention_dim drifted "
            f"(expected 4096, got {transformer_cfg.get('cross_attention_dim')!r})."
        )
    if int(vae_cfg.get("latent_channels", 0)) != 128:
        raise RuntimeError(
            "Vendored LTX2 VAE latent_channels drifted "
            f"(expected 128, got {vae_cfg.get('latent_channels')!r})."
        )
    if int(audio_vae_cfg.get("latent_channels", 0)) != 8:
        raise RuntimeError(
            "Vendored LTX2 audio VAE latent_channels drifted "
            f"(expected 8, got {audio_vae_cfg.get('latent_channels')!r})."
        )
    if int(connectors_cfg.get("caption_channels", 0)) != int(transformer_cfg.get("caption_channels", 0)):
        raise RuntimeError(
            "Vendored LTX2 connectors/transformer caption_channels mismatch "
            f"(connectors={connectors_cfg.get('caption_channels')!r}, "
            f"transformer={transformer_cfg.get('caption_channels')!r})."
        )

    text_cfg = text_encoder_cfg.get("text_config")
    if not isinstance(text_cfg, Mapping):
        raise RuntimeError("Vendored LTX2 text_encoder config must include a `text_config` object.")
    text_hidden = int(text_cfg.get("hidden_size", 0))
    if text_hidden != 3840:
        raise RuntimeError(
            "Vendored LTX2 text encoder hidden_size drifted "
            f"(expected 3840, got {text_hidden!r})."
        )

    if int(vocoder_cfg.get("in_channels", 0)) != 128:
        raise RuntimeError(
            "Vendored LTX2 vocoder in_channels drifted "
            f"(expected 128, got {vocoder_cfg.get('in_channels')!r})."
        )

    return {
        "transformer": transformer_cfg,
        "connectors": connectors_cfg,
        "vae": vae_cfg,
        "audio_vae": audio_vae_cfg,
        "text_encoder": {"text_hidden_size": text_hidden},
        "vocoder": vocoder_cfg,
    }


def _repo_hint_from_bundle(bundle: SignalBundle) -> str | None:
    current: object | None = bundle.state_dict
    seen_ids: set[int] = set()
    while current is not None:
        marker = id(current)
        if marker in seen_ids:
            break
        seen_ids.add(marker)
        filepath = getattr(current, "filepath", None)
        if isinstance(filepath, str) and filepath.strip():
            lowered = filepath.strip().lower().replace("\\", "/")
            if "ltx-2.3" in lowered or "ltx2.3" in lowered:
                return _LTX23_CHECKPOINT_REPO_HINT
            if "ltx-2" in lowered or "ltx2" in lowered:
                return _LTX2_CONFIG_REPO_HINT
        current = getattr(current, "_base", None)
    return None


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required vendored LTX2 metadata file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in vendored LTX2 metadata file: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Vendored LTX2 metadata must be a JSON object: {path}")
    return payload


def _shape_dim(bundle: SignalBundle, key: str, *, dim: int) -> Optional[int]:
    shape = bundle.shape(key)
    if shape is not None and len(shape) > dim:
        return int(shape[dim])
    tensor = bundle.state_dict.get(key)
    if isinstance(tensor, torch.Tensor) and tensor.ndim > dim:
        return int(tensor.shape[dim])
    return None


def _count_prefixed_keys(keys: Iterable[str], prefix: str) -> int:
    return sum(1 for key in keys if key.startswith(prefix))


def _has_connector_surface(keys: Iterable[str]) -> bool:
    for prefix in _CONNECTOR_SUBPREFIXES:
        if any(key.startswith(prefix) for key in keys):
            return True
    return False


def inspect_ltx2_gguf_path(gguf_path: str) -> Ltx2GgufCoreMetadata:
    header_state = _GGUFHeaderStateDict(gguf_path)
    transformer, connectors = split_ltx2_transformer_and_connectors_state(header_state)
    connector_key_sample = tuple(sorted(connectors.keys())[:5])
    has_required_transformer_markers = any(
        all(key in transformer for key in marker_group)
        for marker_group in _LTX2_REQUIRED_CORE_KEYSETS
    )
    return Ltx2GgufCoreMetadata(
        has_required_transformer_markers=has_required_transformer_markers,
        transformer_key_count=len(transformer),
        connector_key_count=len(connectors),
        connector_key_sample=connector_key_sample,
    )


REGISTRY.register(Ltx2Detector())
