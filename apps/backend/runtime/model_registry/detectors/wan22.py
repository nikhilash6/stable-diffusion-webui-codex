"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 model detector for the Codex model registry.
Matches WAN22 checkpoints by key suffixes and tensor shapes, infers patch/latent dimensions, detects embedded VAE/text-encoder components,
and exposes reusable structural metadata inspection for header-safe WAN GGUF inventory detection before returning a `ModelSignature`.

Symbols (top-level; keep in sync; no ghosts):
- `WAN_HEAD_KEY` (constant): Suffix used to locate the WAN modulation head in a state dict.
- `Wan22StructuralMetadata` (dataclass): Reusable structural WAN22 inspection result (prefix/model_type/family/repo_hint/core dims).
- `Wan22Detector` (class): Detector that matches WAN22 bundles and builds a `ModelSignature` (core dims + TE/VAE signatures).
- `inspect_wan22_bundle` (function): Returns authoritative WAN22 structural metadata from a `SignalBundle` or fails loud.
- `inspect_wan22_gguf_path` (function): Returns authoritative WAN22 structural metadata from a GGUF path via header-only inspection.
- `_collect_text_encoders` (function): Collects embedded text encoder signatures (UMT5-XXL, CLIP-L) when present.
- `_tensor_last_dim` (function): Returns the last dimension of a tensor/shape (used for TE expected dims).
- `_find_key` (function): Finds the shortest matching key by suffix (optional prefix filtering).
- `_detect_model_type` (function): Heuristically classifies WAN variant (t2v/i2v/ti2v/vace/s2v/animate).
- `_repo_hint_for_model_type` (function): Resolves the authoritative WAN repo hint for a detected WAN22 model type.
- `_family_for_model_type` (function): Resolves explicit WAN22 model family (`WAN22_5B`/`WAN22_14B`/`WAN22_ANIMATE`) from detected model type.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Optional

from apps.backend.runtime.model_registry.detectors.base import ModelDetector, REGISTRY
from apps.backend.runtime.model_registry.signals import SignalBundle, count_blocks
from apps.backend.runtime.model_registry.specs import (
    CodexCoreArchitecture,
    CodexCoreSignature,
    LatentFormat,
    ModelFamily,
    ModelSignature,
    PredictionKind,
    QuantizationHint,
    QuantizationKind,
    TextEncoderSignature,
    VAESignature,
)
from apps.backend.runtime.families.wan22.inference import infer_wan22_latent_channels, infer_wan22_patch_embedding

WAN_HEAD_KEY = "head.modulation"
_WAN22_REPO_HINT_BY_MODEL_TYPE = {
    "t2v": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "i2v": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "ti2v": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "animate": "Wan-AI/Wan2.2-Animate-14B-Diffusers",
    "s2v": "Wan-AI/Wan2.2-S2V-14B",
}


@dataclass(frozen=True)
class Wan22StructuralMetadata:
    prefix: str
    model_type: str
    family: ModelFamily
    repo_hint: str
    in_channels: int
    model_dim: int
    patch_size: tuple[int, int, int]
    latent_channels: int
    num_layers: int


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


class Wan22Detector(ModelDetector):
    priority = 170

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        key = _find_key(bundle, WAN_HEAD_KEY)
        if not key:
            return False
        prefix = key[: -len(WAN_HEAD_KEY)]
        patch = bundle.shape(f"{prefix}patch_embedding.weight")
        return bool(patch and len(patch) == 5)

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        is_gguf = bundle.is_gguf_quantized()
        metadata = inspect_wan22_bundle(bundle)

        vae_sig: Optional[VAESignature] = None
        vae_key = _find_key(bundle, "decoder.conv_out.weight", search_prefix="vae.")
        if vae_key:
            vae_shape = bundle.shape(vae_key)
            if vae_shape and len(vae_shape) >= 2:
                vae_sig = VAESignature(key_prefix="vae.", latent_channels=int(vae_shape[1]))

        extras = {
            "model_type": metadata.model_type,
            "patch_size": metadata.patch_size,
            "blocks": metadata.num_layers,
            "channels_in": metadata.in_channels,
        }

        text_encoders = _collect_text_encoders(bundle, metadata.prefix)

        quantization = (
            QuantizationHint(kind=QuantizationKind.GGUF, detail="parameter_gguf") if is_gguf else QuantizationHint()
        )

        return ModelSignature(
            family=metadata.family,
            repo_hint=metadata.repo_hint,
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.WAN22,
            quantization=quantization,
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.DIT,
                channels_in=metadata.in_channels,
                channels_out=metadata.latent_channels,
                context_dim=metadata.model_dim,
                temporal=True,
                depth=metadata.num_layers,
                key_prefixes=[metadata.prefix],
            ),
            text_encoders=text_encoders,
            vae=vae_sig,
            extras=extras,
        )


def _collect_text_encoders(bundle: SignalBundle, prefix: str) -> list[TextEncoderSignature]:
    encoders: list[TextEncoderSignature] = []
    # UMT5 XXL
    if f"{prefix}text_encoders.umt5xxl.transformer.encoder.final_layer_norm.weight" in bundle.state_dict:
        encoders.append(
            TextEncoderSignature(
                name="umt5xxl",
                key_prefix=f"{prefix}text_encoders.umt5xxl.",
                expected_dim=_tensor_last_dim(bundle, f"{prefix}text_encoders.umt5xxl.transformer.encoder.final_layer_norm.weight"),
                tokenizer_hint="Wan-AI/umt5xxl",
            )
        )
    if f"{prefix}text_encoders.clip_l.transformer.final_layer_norm.weight" in bundle.state_dict:
        encoders.append(
            TextEncoderSignature(
                name="clip_l",
                key_prefix=f"{prefix}text_encoders.clip_l.",
                expected_dim=_tensor_last_dim(bundle, f"{prefix}text_encoders.clip_l.transformer.final_layer_norm.weight"),
                tokenizer_hint="Wan-AI/clip-fp16",
            )
        )
    return encoders


def _tensor_last_dim(bundle: SignalBundle, key: str) -> Optional[int]:
    shape = bundle.shape(key)
    if shape:
        return int(shape[-1])
    tensor = bundle.state_dict.get(key)
    tensor_shape = getattr(tensor, "shape", None)
    if tensor_shape:
        return int(tensor_shape[-1])
    return None


def _find_key(bundle: SignalBundle, suffix: str, *, search_prefix: str | None = None) -> Optional[str]:
    candidates = []
    for key in bundle.state_dict:
        if search_prefix and not key.startswith(search_prefix):
            continue
        if key.endswith(suffix):
            candidates.append(key)
    return min(candidates, key=len) if candidates else None


def _detect_model_type(bundle: SignalBundle, prefix: str, in_channels: int) -> str:
    checks = {
        "vace": f"{prefix}vace_patch_embedding.weight",
        "s2v": f"{prefix}casual_audio_encoder.encoder.final_linear.weight",
        "animate": f"{prefix}face_adapter.fuser_blocks.0.k_norm.weight",
    }
    for model_type, key in checks.items():
        if key in bundle.state_dict:
            return model_type
    if in_channels >= 48:
        return "ti2v"
    # Local WAN22 I2V base checkpoints use the 36-channel concat surface
    # (`lat16 + mask4 + img16 -> patch_embedding`) and do not store a separate
    # `img_emb` branch. Older/upstream i2v checkpoints may still expose
    # `img_emb.proj.*`, so keep that as an auxiliary positive signal.
    if in_channels == 36 or f"{prefix}img_emb.proj.0.bias" in bundle.state_dict:
        return "i2v"
    return "t2v"


def _repo_hint_for_model_type(model_type: str) -> str:
    normalized = str(model_type or "").strip().lower()
    repo_hint = _WAN22_REPO_HINT_BY_MODEL_TYPE.get(normalized)
    if repo_hint is None:
        raise NotImplementedError(f"wan22 repo_hint for model_type {normalized} not yet implemented")
    return repo_hint


def _family_for_model_type(model_type: str) -> ModelFamily:
    normalized = str(model_type or "").strip().lower()
    if normalized == "ti2v":
        return ModelFamily.WAN22_5B
    if normalized == "animate":
        return ModelFamily.WAN22_ANIMATE
    return ModelFamily.WAN22_14B


def inspect_wan22_bundle(bundle: SignalBundle) -> Wan22StructuralMetadata:
    key = _find_key(bundle, WAN_HEAD_KEY)
    if key is None:
        raise RuntimeError("WAN22 structural inspection requires head.modulation")
    prefix = key[: -len(WAN_HEAD_KEY)]

    patch_shape = bundle.shape(f"{prefix}patch_embedding.weight")
    if patch_shape is None or len(patch_shape) != 5:
        raise RuntimeError(
            "WAN22 structural inspection requires patch_embedding.weight with 5 dimensions "
            f"(got {patch_shape!r} for prefix {prefix!r})"
        )
    in_channels, model_dim, patch_size = infer_wan22_patch_embedding(patch_shape)

    head_shape = bundle.shape(f"{prefix}head.head.weight")
    if head_shape is None:
        raise RuntimeError(f"WAN22 structural inspection requires {prefix}head.head.weight")
    latent_channels = infer_wan22_latent_channels(
        head_shape,
        patch_size=patch_size,
        default_latent_channels=in_channels,
    )

    model_type = _detect_model_type(bundle, prefix, in_channels)
    family = _family_for_model_type(model_type)
    repo_hint = _repo_hint_for_model_type(model_type)
    num_layers = count_blocks(bundle.keys, f"{prefix}blocks.{{}}.")

    return Wan22StructuralMetadata(
        prefix=prefix,
        model_type=model_type,
        family=family,
        repo_hint=repo_hint,
        in_channels=in_channels,
        model_dim=model_dim,
        patch_size=patch_size,
        latent_channels=latent_channels,
        num_layers=num_layers,
    )


def inspect_wan22_gguf_path(gguf_path: str) -> Wan22StructuralMetadata:
    from apps.backend.runtime.model_registry.signals import build_bundle

    return inspect_wan22_bundle(build_bundle(_GGUFHeaderStateDict(gguf_path)))


REGISTRY.register(Wan22Detector())
