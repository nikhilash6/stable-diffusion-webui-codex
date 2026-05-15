"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public re-exports for GGUF IO helpers (reader/writer/constants).
Provides a small facade over the GGUF schema, IO primitives, and quantized-shape helpers for use by runtime loaders and tools.

Symbols (top-level; keep in sync; no ghosts):
- `GGML_QUANT_SIZES` (constant): Mapping `{GGMLQuantizationType: (block_size, type_size)}`.
- `GGMLQuantizationType` (enum): GGML/GGUF quantization type identifiers.
- `LlamaFileType` (enum): GGUF file type identifiers (compat metadata).
- `GGUFReader` (class): Memmap-based GGUF parser.
- `GGUFWriter` (class): GGUF v3 writer (tensor info + KV store).
- `ReaderTensor` (class): Tensor descriptor returned by `GGUFReader` for each tensor.
- `quant_shape_from_byte_shape` (function): Converts packed byte shapes → logical tensor shapes for a quant type.
- `quant_shape_to_byte_shape` (function): Converts logical tensor shapes → packed byte shapes for a quant type.
"""

from .constants import GGML_QUANT_SIZES, GGMLQuantizationType, LlamaFileType
from .quant_shapes import quant_shape_from_byte_shape, quant_shape_to_byte_shape
from .reader import GGUFReader, ReaderTensor
from .writer import GGUFWriter

__all__ = [
    "GGML_QUANT_SIZES",
    "GGMLQuantizationType",
    "GGUFReader",
    "GGUFWriter",
    "LlamaFileType",
    "ReaderTensor",
    "quant_shape_from_byte_shape",
    "quant_shape_to_byte_shape",
]
