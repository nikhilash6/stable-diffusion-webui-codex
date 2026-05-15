"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Core quantization types and registry for GGUF kernels.
Defines `QuantSpec` and a global registry mapping quantization type IDs to dequant/quant/bake kernels (PyTorch + optional NumPy helpers).

Symbols (top-level; keep in sync; no ghosts):
- `QuantType` (enum): GGML/GGUF quantization type identifiers (alias of `GGMLQuantizationType`).
- `BLOCK_SIZES` (constant): Mapping `{QuantType: (block_size, type_size)}` for packed GGUF tensors.
- `QuantSpec` (dataclass): Kernel specification for a quant type (block/type sizes + dequant/quant/bake hooks).
- `QUANT_REGISTRY` (constant): Global quant registry `{QuantType: QuantSpec}`.
- `register_quant` (function): Registers a `QuantType` and associated kernels/spec into `QUANT_REGISTRY`.
- `get_quant_spec` (function): Retrieves the registered `QuantSpec` for a `QuantType` (or None).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any

import torch

from .gguf.constants import GGMLQuantizationType as QuantType
from .gguf.constants import GGML_QUANT_SIZES as BLOCK_SIZES

logger = get_backend_logger(__name__)

__all__ = [
    "QuantType",
    "BLOCK_SIZES",
    "QuantSpec",
    "register_quant",
    "get_quant_spec",
    "QUANT_REGISTRY",
]


# Type aliases for kernel functions
DequantizeFn = Callable[[torch.Tensor, torch.dtype], torch.Tensor]
QuantizeFn = Callable[[torch.Tensor, Any], torch.Tensor]
BakeFn = Callable[[Any], None]
DequantizeNumpyFn = Callable  # For numpy compatibility


@dataclass
class QuantSpec:
    """Specification for a quantization type."""
    qtype: QuantType
    block_size: int
    type_size: int
    
    # PyTorch kernels
    dequantize: Optional[DequantizeFn] = None
    quantize: Optional[QuantizeFn] = None
    bake: Optional[BakeFn] = None
    
    # NumPy kernels (for compatibility)
    dequantize_numpy: Optional[DequantizeNumpyFn] = None
    quantize_numpy: Optional[DequantizeNumpyFn] = None
    
    # Metadata
    description: str = ""
    requires_bake: bool = True


# Global registry
QUANT_REGISTRY: Dict[QuantType, QuantSpec] = {}


def register_quant(
    qtype: QuantType,
    *,
    dequantize: Optional[DequantizeFn] = None,
    quantize: Optional[QuantizeFn] = None,
    bake: Optional[BakeFn] = None,
    dequantize_numpy: Optional[DequantizeNumpyFn] = None,
    quantize_numpy: Optional[DequantizeNumpyFn] = None,
    description: str = "",
    requires_bake: bool = True,
) -> None:
    """Register a quantization type with its kernels."""
    if qtype not in BLOCK_SIZES:
        raise ValueError(f"Unknown quant type: {qtype}")
    
    block_size, type_size = BLOCK_SIZES[qtype]
    
    spec = QuantSpec(
        qtype=qtype,
        block_size=block_size,
        type_size=type_size,
        dequantize=dequantize,
        quantize=quantize,
        bake=bake,
        dequantize_numpy=dequantize_numpy,
        quantize_numpy=quantize_numpy,
        description=description or qtype.name,
        requires_bake=requires_bake,
    )
    
    QUANT_REGISTRY[qtype] = spec
    logger.debug("Registered quant type: %s (block=%d, bytes=%d)", qtype.name, block_size, type_size)


def get_quant_spec(qtype: QuantType) -> Optional[QuantSpec]:
    """Get the specification for a quantization type."""
    return QUANT_REGISTRY.get(qtype)
