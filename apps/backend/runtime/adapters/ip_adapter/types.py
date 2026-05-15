"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed IP-Adapter runtime contracts.
Defines the validated request-owned source/config carriers plus prepared runtime asset and embedding bundles used by the shared
IP-Adapter stage.

Symbols (top-level; keep in sync; no ghosts):
- `IpAdapterLayout` (enum): Supported IP-Adapter runtime layouts for tranche 1.
- `IpAdapterSourceConfig` (dataclass): Typed nested source owner for IP-Adapter reference images.
- `IpAdapterConfig` (dataclass): Typed processing/runtime owner for one IP-Adapter application.
- `PreparedIpAdapterAssets` (dataclass): Loaded image-encoder/projector/KV asset bundle ready for embedding prep and patch apply.
- `PreparedIpAdapterEmbeddings` (dataclass): Prepared conditional/unconditional IP-Adapter prompt tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class IpAdapterLayout(str, Enum):
    BASE = "base"
    PLUS = "plus"


@dataclass(frozen=True)
class IpAdapterSourceConfig:
    kind: str
    reference_image_data: str | None = None


@dataclass(frozen=True)
class IpAdapterConfig:
    enabled: bool
    model: str
    image_encoder: str
    weight: float
    start_at: float
    end_at: float
    source: IpAdapterSourceConfig


@dataclass(frozen=True)
class PreparedIpAdapterAssets:
    model_path: str
    image_encoder_path: str
    layout: IpAdapterLayout
    target_semantic_engine: str
    slot_count: int
    token_count: int
    output_cross_attention_dim: int
    internal_cross_attention_dim: int
    uses_hidden_states: bool
    image_encoder_runtime: Any
    image_projector: torch.nn.Module
    ip_layers: torch.nn.Module


@dataclass(frozen=True)
class PreparedIpAdapterEmbeddings:
    condition: torch.Tensor
    uncondition: torch.Tensor


__all__ = [
    "IpAdapterConfig",
    "IpAdapterLayout",
    "IpAdapterSourceConfig",
    "PreparedIpAdapterAssets",
    "PreparedIpAdapterEmbeddings",
]
