"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: FLUX.2 Klein 4B detector for core-only SafeTensors checkpoints.
Matches the runtime-export FLUX.2 single-file transformer layout (`double_blocks.*`, `single_blocks.*`, `img_in.*`, `txt_in.*`,
`single_stream_modulation.*`, `final_layer.*`) and builds a strict `ModelSignature` from vendored
`black-forest-labs/FLUX.2-klein-4B` / `FLUX.2-klein-base-4B` metadata. Only the truthful 4B/base-4B slice is supported;
unsupported FLUX.2 variants (for example 9B) intentionally do not match.

Symbols (top-level; keep in sync; no ghosts):
- `FLUX2_CORE_KEYS` (constant): Minimal key set for supported FLUX.2 core-only checkpoints.
- `_SUPPORTED_PREFIXES` (constant): Supported wrapper prefixes for core-only FLUX.2 checkpoints (`"", "transformer.", "model.diffusion_model.", ...`).
- `_SUPPORTED_FLUX2_REPOS` (constant): Vendored FLUX.2 repo ids accepted for the 4B/base-4B slice.
- `_matching_prefix` (function): Resolve the supported wrapper prefix used by a FLUX.2 core-only checkpoint.
- `_repo_hint_from_bundle` (function): Resolve the truthful supported vendored repo hint from the checkpoint filepath when available.
- `_read_json` (function): Reads a required vendored JSON metadata file.
- `_supported_flux2_metadata` (function): Loads and validates the supported FLUX.2 4B/base-4B metadata contract.
- `Flux2CoreDetector` (class): Detector for core-only FLUX.2 Klein 4B/base-4B SafeTensors checkpoints.
- `_shape_at` (function): Reads a shape dimension from bundle metadata/state dict.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional
import json

import torch

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
)

FLUX2_CORE_KEYS = (
    "img_in.weight",
    "txt_in.weight",
    "time_in.in_layer.weight",
    "time_in.out_layer.weight",
    "double_blocks.0.img_attn.norm.key_norm.scale",
    "double_stream_modulation_img.lin.weight",
    "double_stream_modulation_txt.lin.weight",
    "single_blocks.0.linear1.weight",
    "single_stream_modulation.lin.weight",
    "final_layer.linear.weight",
)

_SUPPORTED_FLUX2_REPOS = (
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-base-4B",
)

_SUPPORTED_PREFIXES = (
    "",
    "transformer.",
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
)


class Flux2CoreDetector(ModelDetector):
    priority = 136

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        if bundle.is_gguf_quantized():
            return False
        source_format = str(bundle.source_format or "").strip().lower()
        if source_format and source_format not in {"safetensors", "safetensor"}:
            return False
        prefix = _matching_prefix(bundle)
        if prefix is None:
            return False

        # This detector is intentionally limited to core-only runtime-export checkpoints.
        unsupported_prefixes = (
            "x_embedder.",
            "context_embedder.",
            "transformer_blocks.",
            "single_transformer_blocks.",
            "text_encoder.",
            "text_encoders.",
            "vae.",
            "guidance_in.",
            "vector_in.",
        )
        if any(key.startswith(tuple(prefix + value for value in unsupported_prefixes)) for key in bundle.keys):
            return False

        metadata = _supported_flux2_metadata()
        transformer_cfg = metadata["transformer"]

        hidden_dim = _shape_at(bundle, prefix + "img_in.weight", dim=0)
        in_channels = _shape_at(bundle, prefix + "img_in.weight", dim=1)
        context_dim = _shape_at(bundle, prefix + "txt_in.weight", dim=1)
        out_channels = _shape_at(bundle, prefix + "final_layer.linear.weight", dim=0)
        double_layers = count_blocks(bundle.keys, prefix + "double_blocks.{}.")
        single_layers = count_blocks(bundle.keys, prefix + "single_blocks.{}.")

        expected_hidden = int(transformer_cfg["attention_head_dim"]) * int(transformer_cfg["num_attention_heads"])
        expected_in_channels = int(transformer_cfg["in_channels"])
        expected_context = int(transformer_cfg["joint_attention_dim"])
        expected_double = int(transformer_cfg["num_layers"])
        expected_single = int(transformer_cfg["num_single_layers"])

        return (
            hidden_dim == expected_hidden
            and in_channels == expected_in_channels
            and context_dim == expected_context
            and out_channels == expected_in_channels
            and double_layers == expected_double
            and single_layers == expected_single
        )

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        metadata = _supported_flux2_metadata()
        transformer_cfg = metadata["transformer"]
        text_encoder_cfg = metadata["text_encoder"]
        repo_hint = _repo_hint_from_bundle(bundle)

        latent_channels = int(transformer_cfg["in_channels"])
        context_dim = int(transformer_cfg["joint_attention_dim"])
        double_layers = int(transformer_cfg["num_layers"])
        single_layers = int(transformer_cfg["num_single_layers"])
        guidance_embed = bool(transformer_cfg["guidance_embeds"])
        qwen_hidden = int(text_encoder_cfg["hidden_size"])

        return ModelSignature(
            family=ModelFamily.FLUX2,
            repo_hint=repo_hint,
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.FLUX2,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.FLOW_TRANSFORMER,
                channels_in=latent_channels,
                channels_out=latent_channels,
                context_dim=context_dim,
                temporal=False,
                depth=double_layers + single_layers,
                key_prefixes=["double_blocks.", "single_blocks."],
            ),
            text_encoders=[
                TextEncoderSignature(
                    name="qwen3_4b",
                    key_prefix="text_encoder.",
                    expected_dim=qwen_hidden,
                    tokenizer_hint=f"{repo_hint}/tokenizer",
                )
            ],
            vae=None,
            extras={
                "core_only": True,
                "flux2_variant": "base" if repo_hint.endswith("base-4B") else "klein",
                "flow_double_layers": double_layers,
                "flow_single_layers": single_layers,
                "guidance_embed": guidance_embed,
                "is_distilled": not repo_hint.endswith("base-4B"),
                "signature_source": "vendored_hf",
                "supported_repo_variants": list(_SUPPORTED_FLUX2_REPOS),
            },
        )


@lru_cache(maxsize=1)
def _supported_flux2_metadata() -> dict[str, Mapping[str, Any]]:
    backend_root = Path(__file__).resolve().parents[3]
    hf_root = backend_root / "huggingface" / "black-forest-labs"

    transformer_cfgs: list[Mapping[str, Any]] = []
    for repo_id in _SUPPORTED_FLUX2_REPOS:
        repo_name = repo_id.split("/", 1)[1]
        transformer_cfgs.append(_read_json(hf_root / repo_name / "transformer" / "config.json"))

    canonical = transformer_cfgs[0]
    comparable_fields = (
        "attention_head_dim",
        "guidance_embeds",
        "in_channels",
        "joint_attention_dim",
        "num_attention_heads",
        "num_layers",
        "num_single_layers",
        "patch_size",
        "timestep_guidance_channels",
    )
    for idx, cfg in enumerate(transformer_cfgs[1:], start=1):
        for field in comparable_fields:
            if cfg.get(field) != canonical.get(field):
                repo_name = _SUPPORTED_FLUX2_REPOS[idx].split("/", 1)[1]
                raise RuntimeError(
                    "Vendored FLUX.2 4B/base-4B transformer configs drifted on field "
                    f"{field!r}: canonical={canonical.get(field)!r} repo={repo_name!r} value={cfg.get(field)!r}."
                )

    text_encoder_cfg = _read_json(hf_root / "FLUX.2-klein-4B" / "text_encoder" / "config.json")
    if int(text_encoder_cfg.get("hidden_size", 0)) != 2560:
        raise RuntimeError(
            "Vendored FLUX.2 text_encoder config is not the supported Qwen3-4B variant "
            f"(hidden_size={text_encoder_cfg.get('hidden_size')!r})."
        )

    return {
        "transformer": canonical,
        "text_encoder": text_encoder_cfg,
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required vendored FLUX.2 metadata file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in vendored FLUX.2 metadata file: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Vendored FLUX.2 metadata must be a JSON object: {path}")
    return payload


def _matching_prefix(bundle: SignalBundle) -> str | None:
    for prefix in _SUPPORTED_PREFIXES:
        if has_all_keys(bundle, *(prefix + key for key in FLUX2_CORE_KEYS)):
            return prefix
    return None


def _repo_hint_from_bundle(bundle: SignalBundle) -> str:
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
            if any(tag in lowered for tag in ("flux.2-klein-base-9b", "flux2-klein-base-9b", "flux.2-klein-9b", "flux2-klein-9b")):
                raise RuntimeError("Unsupported FLUX.2 9B checkpoint matched the 4B detector contract.")
            if "flux.2-klein-base-4b" in lowered or "flux2-klein-base-4b" in lowered:
                return _SUPPORTED_FLUX2_REPOS[1]
            if "flux.2-klein-4b" in lowered or "flux2-klein-4b" in lowered:
                return _SUPPORTED_FLUX2_REPOS[0]
        current = getattr(current, "_base", None)
    return _SUPPORTED_FLUX2_REPOS[0]


def _shape_at(bundle: SignalBundle, key: str, dim: int) -> Optional[int]:
    shape = bundle.shape(key)
    if shape is not None and len(shape) > dim:
        return int(shape[dim])
    tensor = bundle.state_dict.get(key)
    if isinstance(tensor, torch.Tensor) and tensor.ndim > dim:
        return int(tensor.shape[dim])
    return None


REGISTRY.register(Flux2CoreDetector())
