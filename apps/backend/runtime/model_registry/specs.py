"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed signature specs produced by model detectors and consumed by loader/runtime assembly.
Defines the core enums and dataclasses for checkpoint signatures (family, prediction kind, latent format, quantization hints, and component signatures).
Includes explicit WAN 2.2 families (`WAN22_5B`, `WAN22_14B`, `WAN22_ANIMATE`), `ModelFamily.FLUX2` for FLUX.2 Klein
checkpoints, `ModelFamily.LTX2` for LTX 2.x monolithic combined checkpoints, `ModelFamily.NETFLIX_VOID` for the
CogVideoX-Fun-backed VOID inpainting family scaffold, and `ModelFamily.ANIMA` for Cosmos Predict2 / MiniTrainDiT-style
flow checkpoints.

Symbols (top-level; keep in sync; no ghosts):
- `ModelFamily` (enum): Checkpoint family tags (SD/SDXL/Flux.1/WAN22/etc).
- `PredictionKind` (enum): Prediction parameterization tags (`eps`, `v_prediction`, `flow`, etc).
- `LatentFormat` (enum): Latent space format tags used by runtimes.
- `QuantizationKind` (enum): Quantization scheme identifiers (`none`, `gguf`, plus detected-but-unsupported `nf4`/`fp4`).
- `QuantizationHint` (dataclass): Structured quantization hint (kind + optional detail).
- `TextEncoderSignature` (dataclass): Text encoder metadata (name/prefix/expected dim/tokenizer hint).
- `VAESignature` (dataclass): VAE metadata (key prefix + latent channels).
- `CodexCoreArchitecture` (enum): Core architecture tags (UNet/DiT/Transformer/FlowTransformer).
- `CodexCoreSignature` (dataclass): Core network signature (channels, context dim, depth, key prefixes).
- `ModelSignature` (dataclass): Full checkpoint signature contract produced by detectors (`unet` is a legacy alias for `core`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ModelFamily(Enum):
    SD15 = "sd15"
    SD20 = "sd20"
    SDXL = "sdxl"
    SDXL_REFINER = "sdxl_refiner"
    SD3 = "sd3"
    SD35 = "sd35"
    FLUX = "flux1"
    FLUX_KONTEXT = "flux1_kontext"
    FLUX2 = "flux2"
    LTX2 = "ltx2"
    NETFLIX_VOID = "netflix_void"
    STABLE_CASCADE = "stable_cascade"
    CHROMA = "chroma"
    KOALA = "koala"
    ZERO123 = "zero123"
    ZIMAGE = "zimage"
    QWEN_IMAGE = "qwen_image"
    ANIMA = "anima"
    WAN22_5B = "wan22_5b"
    WAN22_14B = "wan22_14b"
    WAN22_ANIMATE = "wan22_14b_animate"
    AURA = "aura"
    HUNYUAN = "hunyuan"
    SVD = "svd"
    STABLE_AUDIO = "stable_audio"
    UPSCALER = "upscaler"
    OTHER = "other"


class PredictionKind(Enum):
    EPSILON = "eps"
    V_PREDICTION = "v_prediction"
    EDM = "edm"
    FLOW = "flow"
    V_CONTINUOUS = "v_continuous"


class LatentFormat(Enum):
    SD_V1 = "sd_v1"
    SD_V2 = "sd_v2"
    SD_XL = "sd_xl"
    SD_3 = "sd_3"
    FLOW16 = "flow16"
    FLUX2 = "flux2"
    LTX2 = "ltx2"
    CHROMA_RADIANCE = "chroma_radiance"
    CASCADE = "cascade"
    WAN22 = "wan22"
    ZIMAGE = "zimage"
    QWEN_IMAGE = "qwen_image"
    OTHER = "other"


class QuantizationKind(Enum):
    NONE = "none"
    NF4 = "nf4"
    FP4 = "fp4"
    GGUF = "gguf"


@dataclass
class QuantizationHint:
    kind: QuantizationKind = QuantizationKind.NONE
    detail: Optional[str] = None


@dataclass
class TextEncoderSignature:
    name: str
    key_prefix: str
    expected_dim: Optional[int] = None
    tokenizer_hint: Optional[str] = None


@dataclass
class VAESignature:
    key_prefix: str
    latent_channels: int


class CodexCoreArchitecture(Enum):
    UNET = "unet"
    DIT = "dit"
    TRANSFORMER = "transformer"
    FLOW_TRANSFORMER = "flow_transformer"


@dataclass
class CodexCoreSignature:
    architecture: CodexCoreArchitecture
    channels_in: int
    channels_out: int
    context_dim: Optional[int]
    temporal: bool
    depth: Optional[int]
    key_prefixes: List[str] = field(default_factory=list)


@dataclass
class ModelSignature:
    family: ModelFamily
    repo_hint: Optional[str]
    prediction: PredictionKind
    latent_format: LatentFormat
    quantization: QuantizationHint
    core: CodexCoreSignature
    text_encoders: List[TextEncoderSignature]
    vae: Optional[VAESignature]
    extras: Dict[str, object] = field(default_factory=dict)

    @property
    def unet(self) -> CodexCoreSignature:
        return self.core


__all__ = [
    "ModelFamily",
    "PredictionKind",
    "LatentFormat",
    "QuantizationKind",
    "QuantizationHint",
    "TextEncoderSignature",
    "VAESignature",
    "CodexCoreArchitecture",
    "CodexCoreSignature",
    "ModelSignature",
]
