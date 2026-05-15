"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Quantization public API (bake/quantize/dequantize) for GGUF packed tensors.
Exposes torch and NumPy helpers for working with GGML/GGUF quantized weights, used by runtime ops and tooling (e.g. GGUF converter).
Imports `apps.backend.quantization.kernels` on module import to ensure the global quant registry is populated before API calls.

Symbols (top-level; keep in sync; no ghosts):
- `bake` (function): Pre-bakes a `CodexParameter` for faster repeated dequantization.
- `dequantize` (function): Dequantizes a GGUF `CodexParameter` to a torch tensor (auto-bake as needed).
- `quantize` (function): Quantizes a torch tensor into a GGUF `CodexParameter` (when supported by the quant spec).
- `quantize_numpy` (function): Quantizes a NumPy float array into packed GGUF bytes (tooling path).
- `dequantize_numpy` (function): Dequantizes packed GGUF bytes into float32 NumPy arrays.
- `dequantize_tensor` (function): Convenience wrapper for callsites that may receive a `CodexParameter` or a regular tensor.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from . import kernels  # noqa: F401 - triggers quant kernel registration
from .core import get_quant_spec, QuantType

if TYPE_CHECKING:
    from .tensor import CodexParameter

logger = get_backend_logger(__name__)

__all__ = ["dequantize", "bake", "quantize", "dequantize_numpy", "quantize_numpy"]


def bake(tensor: "CodexParameter") -> None:
    """
    Bake a GGUF tensor - pre-process for faster dequantization.
    
    This converts internal format to be optimal for repeated dequantization.
    Called automatically when moving tensor to a device.
    
    Args:
        tensor: CodexParameter to bake
    """
    if tensor.baked:
        return
    
    if tensor.qtype is None:
        tensor.baked = True
        return
    
    spec = get_quant_spec(tensor.qtype)
    if spec is None:
        logger.warning("No spec found for qtype %s, skipping bake", tensor.qtype)
        tensor.baked = True
        return
    
    if spec.bake is not None:
        try:
            spec.bake(tensor)
        except Exception as e:
            logger.error("Failed to bake tensor (qtype=%s): %s", tensor.qtype.name, e)
            raise
    
    tensor.baked = True


def dequantize(tensor: "CodexParameter") -> torch.Tensor:
    """
    Dequantize a GGUF tensor to float.
    
    Automatically bakes the tensor if not already baked.
    
    Args:
        tensor: CodexParameter to dequantize
        
    Returns:
        Dequantized tensor with shape tensor.real_shape and dtype tensor.computation_dtype
    """
    if tensor is None:
        return None
    
    # Handle non-quantized tensors
    if not hasattr(tensor, 'qtype') or tensor.qtype is None:
        return tensor
    
    # Lazy bake if needed
    if not tensor.baked:
        bake(tensor)
    
    spec = get_quant_spec(tensor.qtype)
    if spec is None:
        raise ValueError(f"Unknown quantization type: {tensor.qtype}")
    
    if spec.dequantize is None:
        raise ValueError(f"No dequantize kernel for type: {tensor.qtype.name}")
    
    # Dequantize
    result = spec.dequantize(tensor.data, tensor.computation_dtype)
    
    # Reshape to logical shape
    if tensor.real_shape:
        result = result.view(tensor.real_shape)
    
    return result


def quantize(
    tensor: torch.Tensor,
    qtype: QuantType,
    computation_dtype: torch.dtype = torch.float16,
) -> "CodexParameter":
    """
    Quantize a float tensor to GGUF format.
    
    Args:
        tensor: Float tensor to quantize
        qtype: Target quantization type
        computation_dtype: Dtype for dequantized computation
        
    Returns:
        CodexParameter with quantized data
        
    Raises:
        NotImplementedError: If quantization not implemented for this type
    """
    from .tensor import CodexParameter
    
    spec = get_quant_spec(qtype)
    if spec is None:
        raise ValueError(f"Unknown quantization type: {qtype}")
    
    if spec.quantize is None:
        raise NotImplementedError(
            f"Quantization not implemented for {qtype.name}. "
            f"This type only supports dequantization."
        )
    
    # Flatten and reshape for block processing
    original_shape = tensor.shape
    flat = tensor.flatten()
    
    # Ensure divisible by block size
    if flat.numel() % spec.block_size != 0:
        raise ValueError(
            f"Tensor size {flat.numel()} not divisible by block size {spec.block_size}"
        )
    
    n_blocks = flat.numel() // spec.block_size
    blocks = flat.view(n_blocks, spec.block_size)
    
    # Create temporary parameter for quantization context
    temp = CodexParameter.__new__(
        CodexParameter,
        torch.empty(1),
        qtype=qtype,
        shape=original_shape,
        computation_dtype=computation_dtype,
    )
    temp.computation_dtype = computation_dtype
    
    # Quantize
    quantized = spec.quantize(blocks, temp)
    
    # Create result parameter
    result = CodexParameter(
        quantized,
        qtype=qtype,
        shape=original_shape,
        computation_dtype=computation_dtype,
    )
    result.baked = False
    
    return result


def quantize_numpy(data: np.ndarray, qtype: QuantType) -> np.ndarray:
    """Quantize a NumPy float tensor into packed GGUF bytes.

    This is used by tooling (e.g. GGUF converter) to emit real quantized GGUF.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"quantize_numpy expects numpy.ndarray, got {type(data)!r}")

    if qtype == QuantType.F32:
        return data.astype(np.float32, copy=False)
    if qtype == QuantType.F16:
        return data.astype(np.float16, copy=False)

    spec = get_quant_spec(qtype)
    if spec is None or spec.quantize_numpy is None:
        raise NotImplementedError(f"NumPy quantization not implemented for {qtype.name}")

    if data.shape[-1] % spec.block_size != 0:
        raise ValueError(
            f"Tensor last dim {data.shape[-1]} not divisible by block size {spec.block_size} for {qtype.name}"
        )

    rows = int(np.prod(data.shape[:-1], dtype=np.int64)) if data.ndim > 1 else 1
    elems_per_row = int(data.shape[-1]) if data.ndim > 1 else int(data.shape[0])

    blocks_per_row = elems_per_row // spec.block_size
    bytes_per_row = blocks_per_row * spec.type_size

    flat = data.astype(np.float32, copy=False).reshape((rows, elems_per_row))
    blocks = flat.reshape((rows * blocks_per_row, spec.block_size))
    packed = spec.quantize_numpy(blocks)
    if packed.dtype != np.uint8:
        raise TypeError(f"{qtype.name} quantize_numpy returned {packed.dtype}, expected uint8")
    if packed.shape != (rows * blocks_per_row, spec.type_size):
        raise ValueError(f"{qtype.name} quantize_numpy returned shape {packed.shape}, expected {(rows * blocks_per_row, spec.type_size)}")

    out = packed.reshape((rows, bytes_per_row))
    if data.ndim == 1:
        return out.reshape((bytes_per_row,))
    return out.reshape((*data.shape[:-1], bytes_per_row))


def dequantize_numpy(data: np.ndarray, qtype: QuantType) -> np.ndarray:
    """Dequantize packed GGUF bytes into float32 NumPy arrays."""
    if not isinstance(data, np.ndarray):
        raise TypeError(f"dequantize_numpy expects numpy.ndarray, got {type(data)!r}")

    if qtype == QuantType.F32:
        if data.dtype == np.float32:
            return data
        return data.view(np.float32)
    if qtype == QuantType.F16:
        if data.dtype == np.float16:
            return data.astype(np.float32)
        return data.view(np.float16).astype(np.float32)

    spec = get_quant_spec(qtype)
    if spec is None:
        raise NotImplementedError(f"Unknown quantization type: {qtype}")

    use_torch = False
    if spec.dequantize_numpy is None:
        if spec.dequantize is None:
            raise NotImplementedError(f"NumPy dequantization not implemented for {qtype.name}")
        use_torch = True

    if data.ndim == 1:
        rows = 1
        bytes_per_row = int(data.shape[0])
    else:
        rows = int(np.prod(data.shape[:-1], dtype=np.int64))
        bytes_per_row = int(data.shape[-1])

    if bytes_per_row % spec.type_size != 0:
        raise ValueError(
            f"Tensor bytes/row {bytes_per_row} not divisible by type size {spec.type_size} for {qtype.name}"
        )

    blocks_per_row = bytes_per_row // spec.type_size
    elems_per_row = blocks_per_row * spec.block_size

    packed_rows = data.reshape((rows, bytes_per_row)).view(np.uint8)
    blocks = packed_rows.reshape((rows * blocks_per_row, spec.type_size))
    if use_torch:
        blocks_np = blocks
        if not blocks_np.flags.writeable:
            blocks_np = np.array(blocks_np, copy=True)
        out_blocks = spec.dequantize(torch.from_numpy(blocks_np), torch.float32).detach().cpu().numpy()
    else:
        out_blocks = spec.dequantize_numpy(blocks)
    if out_blocks.dtype != np.float32:
        out_blocks = out_blocks.astype(np.float32, copy=False)
    out = out_blocks.reshape((rows, elems_per_row))

    if data.ndim == 1:
        return out.reshape((elems_per_row,))
    return out.reshape((*data.shape[:-1], elems_per_row))


# Convenience function for code that calls dequantize_tensor(...)
def dequantize_tensor(tensor) -> torch.Tensor:
    """
    Convenience wrapper for callsites that may receive a CodexParameter.
    """
    if tensor is None:
        return None

    if hasattr(tensor, "qtype"):
        return dequantize(tensor)

    return tensor
