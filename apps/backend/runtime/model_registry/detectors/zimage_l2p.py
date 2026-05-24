"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z-Image L2P model detector for the Codex model registry.
Identifies the public L2P pixel-space DiT checkpoint by native tensor names, requires dedicated L2P GGUF profile metadata for GGUF
artifacts, rejects embedded VAE/text-encoder layouts, and emits a no-VAE `ModelSignature` with the required external Qwen3-4B text encoder slot.

Symbols (top-level; keep in sync; no ghosts):
- `ZIMAGE_L2P_REQUIRED_KEYS` (constant): Native tensor sentinels required for L2P checkpoint detection.
- `ZIMAGE_L2P_FORBIDDEN_PREFIXES` (constant): Embedded component prefixes rejected by the L2P core-only detector.
- `ZIMAGE_L2P_DENOISER_PROFILE_ID` (constant): Required converter profile id for L2P denoiser GGUF artifacts.
- `ZIMAGE_L2P_TENC_PROFILE_ID` (constant): Required converter profile id for L2P Qwen3-4B TEnc GGUF artifacts.
- `validate_zimage_l2p_gguf_component_metadata` (function): Validates dedicated L2P GGUF profile metadata for denoiser/TEnc artifacts.
- `zimage_l2p_gguf_component_metadata_valid` (function): Boolean wrapper around L2P GGUF metadata validation.
- `ZImageL2PDetector` (class): Detector that matches the L2P pixel-space checkpoint layout and builds its signature.
- `_shape` (function): Header/tensor shape helper for signature evidence.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from apps.backend.runtime.model_registry.detectors.base import ModelDetector, REGISTRY
from apps.backend.runtime.model_registry.signals import SignalBundle
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
)


ZIMAGE_L2P_REQUIRED_KEYS: tuple[str, ...] = (
    "all_x_embedder.16-1.weight",
    "local_decoder.out_conv.weight",
    "layers.0.adaLN_modulation.0.weight",
    "noise_refiner.0.adaLN_modulation.0.weight",
    "cap_embedder.1.weight",
)

ZIMAGE_L2P_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "vae.",
    "first_stage_model.",
    "text_encoder.",
)
ZIMAGE_L2P_DENOISER_PROFILE_ID = "zimage_l2p_denoiser"
ZIMAGE_L2P_TENC_PROFILE_ID = "zimage_l2p_tenc"


def _metadata_string(metadata: Mapping[str, Any], key: str) -> str:
    return str(metadata.get(key) or "").strip()


def _metadata_bool(metadata: Mapping[str, Any], key: str) -> bool | None:
    raw = metadata.get(key)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"true", "1", "yes"}:
            return True
        if value in {"false", "0", "no"}:
            return False
    return None


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int | None:
    raw = metadata.get(key)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, float) and float(raw).is_integer():
        return int(raw)
    if isinstance(raw, str):
        value = raw.strip()
        if value.isdigit():
            return int(value)
    return None


def validate_zimage_l2p_gguf_component_metadata(metadata: Mapping[str, Any], *, component: str) -> None:
    expected_component = str(component or "").strip()
    expected_profile = {
        "denoiser": ZIMAGE_L2P_DENOISER_PROFILE_ID,
        "tenc": ZIMAGE_L2P_TENC_PROFILE_ID,
    }.get(expected_component)
    if expected_profile is None:
        raise ValueError(f"Unsupported Z-Image L2P GGUF component metadata target: {component!r}")

    profile_id = _metadata_string(metadata, "codex.zimage_l2p.profile_id")
    family = _metadata_string(metadata, "codex.zimage_l2p.family")
    actual_component = _metadata_string(metadata, "codex.zimage_l2p.component")
    pixel_space = _metadata_bool(metadata, "codex.zimage_l2p.pixel_space")
    requires_vae = _metadata_bool(metadata, "codex.zimage_l2p.requires_vae")
    if profile_id != expected_profile:
        raise ValueError(
            f"Z-Image L2P GGUF metadata profile_id mismatch: got {profile_id or '<missing>'!r}, "
            f"expected {expected_profile!r}."
        )
    if family != "zimage_l2p":
        raise ValueError(
            f"Z-Image L2P GGUF metadata family mismatch: got {family or '<missing>'!r}, expected 'zimage_l2p'."
        )
    if actual_component != expected_component:
        raise ValueError(
            "Z-Image L2P GGUF metadata component mismatch: "
            f"got {actual_component or '<missing>'!r}, expected {expected_component!r}."
        )
    if pixel_space is not True:
        raise ValueError("Z-Image L2P GGUF metadata requires codex.zimage_l2p.pixel_space=true.")
    if requires_vae is not False:
        raise ValueError("Z-Image L2P GGUF metadata requires codex.zimage_l2p.requires_vae=false.")

    if expected_component == "denoiser":
        architecture = _metadata_string(metadata, "model.architecture")
        if architecture != "zimage_l2p":
            raise ValueError(
                "Z-Image L2P denoiser GGUF metadata requires model.architecture='zimage_l2p'; "
                f"got {architecture or '<missing>'!r}."
            )
        if _metadata_bool(metadata, "codex.zimage_l2p.local_decoder") is not True:
            raise ValueError("Z-Image L2P denoiser GGUF metadata requires codex.zimage_l2p.local_decoder=true.")
        if _metadata_int(metadata, "codex.zimage_l2p.context_dim") != 2560:
            raise ValueError("Z-Image L2P denoiser GGUF metadata requires context_dim=2560.")
        return

    tenc_slot = _metadata_string(metadata, "codex.zimage_l2p.tenc_slot")
    if tenc_slot != "qwen3_4b":
        raise ValueError(
            f"Z-Image L2P TEnc GGUF metadata requires tenc_slot='qwen3_4b'; got {tenc_slot or '<missing>'!r}."
        )
    if _metadata_int(metadata, "codex.zimage_l2p.qwen_hidden_size") != 2560:
        raise ValueError("Z-Image L2P TEnc GGUF metadata requires qwen_hidden_size=2560.")
    if _metadata_int(metadata, "codex.zimage_l2p.qwen_layers") != 36:
        raise ValueError("Z-Image L2P TEnc GGUF metadata requires qwen_layers=36.")


def zimage_l2p_gguf_component_metadata_valid(metadata: Mapping[str, Any] | None, *, component: str) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    try:
        validate_zimage_l2p_gguf_component_metadata(metadata, component=component)
    except Exception:
        return False
    return True


class ZImageL2PDetector(ModelDetector):
    priority = 180

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        keys = set(bundle.keys)
        if not all(key in keys for key in ZIMAGE_L2P_REQUIRED_KEYS):
            return False
        if "final_layer.linear.weight" in keys:
            return False
        if bundle.is_gguf_quantized() and not zimage_l2p_gguf_component_metadata_valid(
            bundle.metadata,
            component="denoiser",
        ):
            return False
        return not any(any(key.startswith(prefix) for prefix in ZIMAGE_L2P_FORBIDDEN_PREFIXES) for key in keys)

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        embed_shape = _shape(bundle, "all_x_embedder.16-1.weight")
        decoder_shape = _shape(bundle, "local_decoder.out_conv.weight")
        cap_shape = _shape(bundle, "cap_embedder.1.weight")

        if embed_shape != (3840, 768):
            raise ValueError(f"Z-Image L2P x-embedder shape mismatch: {embed_shape!r} != (3840, 768)")
        if decoder_shape != (3, 64, 1, 1):
            raise ValueError(f"Z-Image L2P local decoder output shape mismatch: {decoder_shape!r} != (3, 64, 1, 1)")
        if cap_shape is None or len(cap_shape) != 2 or int(cap_shape[0]) != 3840 or int(cap_shape[1]) != 2560:
            raise ValueError(f"Z-Image L2P caption embedder shape mismatch: {cap_shape!r} != (3840, 2560)")

        text_encoders = [
            TextEncoderSignature(
                name="qwen3_4b",
                key_prefix="qwen3_4b.",
                expected_dim=2560,
                tokenizer_hint="Qwen/Qwen3-4B",
            )
        ]

        quantization = (
            QuantizationHint(kind=QuantizationKind.GGUF, detail="parameter_gguf")
            if bundle.is_gguf_quantized()
            else QuantizationHint()
        )

        return ModelSignature(
            family=ModelFamily.ZIMAGE_L2P,
            repo_hint="zhen-nan/L2P",
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.PIXEL_RGB,
            quantization=quantization,
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.DIT,
                channels_in=3,
                channels_out=3,
                context_dim=2560,
                temporal=False,
                depth=30,
                key_prefixes=["layers.", "noise_refiner.", "context_refiner.", "local_decoder."],
            ),
            text_encoders=text_encoders,
            vae=None,
            extras={
                "patch_size": 16,
                "f_patch_size": 1,
                "hidden_dim": 3840,
                "num_layers": 30,
                "num_refiner_layers": 2,
                "num_heads": 30,
                "mlp_hidden": 10240,
                "requires_vae": False,
                "signature_source": (
                    "zimage_l2p_gguf_profile"
                    if bundle.is_gguf_quantized()
                    else "zimage_l2p_safetensors_header"
                ),
            },
        )


def _shape(bundle: SignalBundle, key: str) -> Optional[tuple[int, ...]]:
    shape = bundle.shape(key)
    if shape is not None:
        return tuple(int(v) for v in shape)
    tensor = bundle.state_dict.get(key)
    if isinstance(tensor, torch.Tensor):
        return tuple(int(v) for v in tensor.shape)
    return None


REGISTRY.register(ZImageL2PDetector())
