"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: NumPy quantization kernels for GGML block formats.
Implements pure-NumPy quantization of `(n_blocks, 32)` float blocks into packed GGML block layouts (Q8_0/Q4_0/Q4_1/Q5_*/Q*_K/IQ4_NL),
matching ggml rounding/packing conventions.

Symbols (top-level; keep in sync; no ghosts):
- `_np_roundf` (function): GGML-style round-half-up rounding used for quant packing.
- `_pack_nibbles_32x4` (function): Packs 32 4-bit values into 16 bytes using ggml nibble order.
- `_pack_bits_32` (function): Packs 32 1-bit flags into 4 bytes (little-endian bit order).
- `quantize_blocks_q8_0` (function): Quantizes blocks to GGML Q8_0 packed layout.
- `quantize_blocks_q4_0` (function): Quantizes blocks to GGML Q4_0 packed layout.
- `quantize_blocks_q4_1` (function): Quantizes blocks to GGML Q4_1 packed layout.
- `quantize_blocks_q5_0` (function): Quantizes blocks to GGML Q5_0 packed layout.
- `quantize_blocks_q5_1` (function): Quantizes blocks to GGML Q5_1 packed layout.
- `quantize_blocks_iq4_nl` (function): Quantizes blocks to GGML IQ4_NL packed layout.
- `_pack_k_scale_min` (function): Packs K-quant scale/min arrays into the ggml K-block header layout.
- `_k_quant_scale_components` (function): Computes Q4_K/Q5_K group scales and stored FP16 block-scale headers.
- `k_quant_stored_scale_underflow_mask` (function): Returns Q4_K/Q5_K blocks whose stored FP16 scale/min headers underflow.
- `quantize_blocks_q4_k` (function): Quantizes blocks to GGML Q4_K packed layout.
- `quantize_blocks_q5_k` (function): Quantizes blocks to GGML Q5_K packed layout.
- `quantize_blocks_q3_k` (function): Quantizes blocks to GGML Q3_K packed layout.
- `quantize_blocks_q2_k` (function): Quantizes blocks to GGML Q2_K packed layout.
- `quantize_blocks_q6_k` (function): Quantizes blocks to GGML Q6_K packed layout.
"""

from __future__ import annotations

import numpy as np

from ..core import QuantType


def _np_roundf(values: np.ndarray) -> np.ndarray:
    """Round like ggml (round-half-up) for quant packing."""
    abs_values = np.abs(values)
    floored = np.floor(abs_values)
    delta = floored + np.floor(2 * (abs_values - floored))
    return np.sign(values) * delta


def _pack_nibbles_32x4(values: np.ndarray) -> np.ndarray:
    """Pack 32 4-bit values into 16 bytes using the ggml unpack order.

    Dequant code unpacks as:
      low nibbles for first 16 values, then high nibbles for last 16 values.

    Args:
        values: (n_blocks, 32) uint8 in [0, 15]

    Returns:
        (n_blocks, 16) uint8 bytes
    """
    if values.ndim != 2 or values.shape[1] != 32:
        raise ValueError(f"expected (n_blocks, 32) 4-bit values, got {values.shape}")
    lo = values[:, :16].astype(np.uint8, copy=False)
    hi = values[:, 16:].astype(np.uint8, copy=False)
    if np.any(lo > 0x0F) or np.any(hi > 0x0F):
        raise ValueError("values contain entries outside 4-bit range (0..15)")
    return (lo & np.uint8(0x0F)) | (hi << np.uint8(4))


def _pack_bits_32(bits: np.ndarray) -> np.ndarray:
    """Pack 32 1-bit flags into 4 bytes (little-endian bit order)."""
    if bits.ndim != 2 or bits.shape[1] != 32:
        raise ValueError(f"expected (n_blocks, 32) bits, got {bits.shape}")
    b = bits.astype(np.uint32, copy=False) & np.uint32(1)
    weights = np.left_shift(np.uint32(1), np.arange(32, dtype=np.uint32)).reshape((1, 32))
    packed_u32 = (b * weights).sum(axis=1).astype("<u4", copy=False)
    return packed_u32.view(np.uint8).reshape((-1, 4))


def quantize_blocks_q8_0(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 32) to GGML Q8_0 packed blocks (n, 34)."""
    if blocks.ndim != 2 or blocks.shape[1] != 32:
        raise ValueError(f"Q8_0 quantize expects (n_blocks, 32), got {blocks.shape}")
    values = blocks.astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("Q8_0 quantize received non-finite values")

    scale = np.abs(values).max(axis=1, keepdims=True) / np.float32(127.0)
    stored_scale = scale.astype(np.float16)
    stored_scale_is_nonzero = stored_scale != np.float16(0.0)
    inv = np.zeros_like(scale, dtype=np.float32)
    np.divide(np.float32(1.0), scale, out=inv, where=stored_scale_is_nonzero)
    qs = _np_roundf(values * inv)
    header = stored_scale.view(np.uint8)
    payload = qs.astype(np.int8).view(np.uint8)
    return np.concatenate([header, payload], axis=1)


def quantize_blocks_q4_0(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 32) to GGML Q4_0 packed blocks (n, 18)."""
    if blocks.ndim != 2 or blocks.shape[1] != 32:
        raise ValueError(f"Q4_0 quantize expects (n_blocks, 32), got {blocks.shape}")

    x = blocks.astype(np.float32, copy=False)
    d = np.abs(x).max(axis=1, keepdims=True) / np.float32(7.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(d == 0, 0, 1.0 / d)

    q = _np_roundf(x * inv) + np.float32(8.0)
    q = np.clip(q, 0, 15).astype(np.uint8)

    header = d.astype(np.float16).view(np.uint8)
    qs = _pack_nibbles_32x4(q)
    return np.concatenate([header, qs], axis=1)


def quantize_blocks_q4_1(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 32) to GGML Q4_1 packed blocks (n, 20)."""
    if blocks.ndim != 2 or blocks.shape[1] != 32:
        raise ValueError(f"Q4_1 quantize expects (n_blocks, 32), got {blocks.shape}")

    x = blocks.astype(np.float32, copy=False)
    mn = x.min(axis=1, keepdims=True)
    mx = x.max(axis=1, keepdims=True)

    d = (mx - mn) / np.float32(15.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(d == 0, 0, 1.0 / d)

    q = _np_roundf((x - mn) * inv)
    q = np.clip(q, 0, 15).astype(np.uint8)

    header_d = d.astype(np.float16).view(np.uint8)
    header_m = mn.astype(np.float16).view(np.uint8)
    qs = _pack_nibbles_32x4(q)
    return np.concatenate([header_d, header_m, qs], axis=1)


def quantize_blocks_q5_0(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 32) to GGML Q5_0 packed blocks (n, 22)."""
    if blocks.ndim != 2 or blocks.shape[1] != 32:
        raise ValueError(f"Q5_0 quantize expects (n_blocks, 32), got {blocks.shape}")

    x = blocks.astype(np.float32, copy=False)
    d = np.abs(x).max(axis=1, keepdims=True) / np.float32(15.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(d == 0, 0, 1.0 / d)

    q = _np_roundf(x * inv) + np.float32(16.0)
    q = np.clip(q, 0, 31).astype(np.uint8)

    lo4 = q & np.uint8(0x0F)
    hi1 = (q >> np.uint8(4)) & np.uint8(1)

    header = d.astype(np.float16).view(np.uint8)
    qh = _pack_bits_32(hi1)
    qs = _pack_nibbles_32x4(lo4)
    return np.concatenate([header, qh, qs], axis=1)


def quantize_blocks_q5_1(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 32) to GGML Q5_1 packed blocks (n, 24)."""
    if blocks.ndim != 2 or blocks.shape[1] != 32:
        raise ValueError(f"Q5_1 quantize expects (n_blocks, 32), got {blocks.shape}")

    x = blocks.astype(np.float32, copy=False)
    mn = x.min(axis=1, keepdims=True)
    mx = x.max(axis=1, keepdims=True)

    d = (mx - mn) / np.float32(31.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(d == 0, 0, 1.0 / d)

    q = _np_roundf((x - mn) * inv)
    q = np.clip(q, 0, 31).astype(np.uint8)

    lo4 = q & np.uint8(0x0F)
    hi1 = (q >> np.uint8(4)) & np.uint8(1)

    header_d = d.astype(np.float16).view(np.uint8)
    header_m = mn.astype(np.float16).view(np.uint8)
    qh = _pack_bits_32(hi1)
    qs = _pack_nibbles_32x4(lo4)
    return np.concatenate([header_d, header_m, qh, qs], axis=1)


_IQ4_NL_LEVELS = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)


def quantize_blocks_iq4_nl(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 32) to GGML IQ4_NL packed blocks (n, 18).

    Format matches `dequantize_blocks_IQ4_NL`:
    - 2 bytes: float16 scale `d`
    - 16 bytes: 32 x 4-bit codes (packed as nibbles)

    Note: This is an approximate encoder (nearest-level) meant for tooling and
    runtime experiments.
    """
    if blocks.ndim != 2 or blocks.shape[1] != 32:
        raise ValueError(f"IQ4_NL quantize expects (n_blocks, 32), got {blocks.shape}")

    x = blocks.astype(np.float32, copy=False)
    d = np.max(np.abs(x), axis=1, keepdims=True) / np.float32(127.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(d == 0, 0, 1.0 / d)

    scaled = x * inv  # (n_blocks, 32)
    diff = np.abs(scaled[:, :, None] - _IQ4_NL_LEVELS[None, None, :])
    codes = diff.argmin(axis=-1).astype(np.uint8)  # (n_blocks, 32)

    header = d.astype(np.float16).view(np.uint8)
    qs = _pack_nibbles_32x4(codes)
    return np.concatenate([header, qs], axis=1)


def _pack_k_scale_min(scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    """Pack Q4_K/Q5_K scale+min 6-bit tables into the 12-byte GGML layout."""
    if scales.dtype != np.uint8 or mins.dtype != np.uint8:
        raise TypeError("scales/mins must be uint8 arrays")
    if scales.shape != mins.shape:
        raise ValueError(f"scales/mins shape mismatch: {scales.shape} vs {mins.shape}")
    if scales.ndim != 2 or scales.shape[1] != 8:
        raise ValueError(f"expected (n_blocks, 8) scales/mins, got {scales.shape}")

    sc0 = scales[:, :4]
    sc1 = scales[:, 4:]
    mn0 = mins[:, :4]
    mn1 = mins[:, 4:]

    if np.any(sc0 > 0x3F) or np.any(sc1 > 0x3F) or np.any(mn0 > 0x3F) or np.any(mn1 > 0x3F):
        raise ValueError("scales/mins contain values outside 6-bit range (0..63)")

    d = (sc0 & np.uint8(0x3F)) | ((sc1 & np.uint8(0x30)) << np.uint8(2))
    m = (mn0 & np.uint8(0x3F)) | ((mn1 & np.uint8(0x30)) << np.uint8(2))
    m_d = ((mn1 & np.uint8(0x0F)) << np.uint8(4)) | (sc1 & np.uint8(0x0F))

    packed = np.stack([d, m, m_d], axis=1)  # (n_blocks, 3, 4)
    return packed.reshape((scales.shape[0], 12))


def _k_quant_scale_components(
    blocks: np.ndarray,
    qtype: QuantType,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if qtype == QuantType.Q4_K:
        scale_divisor = np.float32(15.0)
    elif qtype == QuantType.Q5_K:
        scale_divisor = np.float32(31.0)
    else:
        raise ValueError(f"K stored-scale underflow analysis expects Q4_K or Q5_K, got {qtype.name}")
    if blocks.ndim != 2 or blocks.shape[1] != 256:
        raise ValueError(f"{qtype.name} K quantize expects (n_blocks, 256), got {blocks.shape}")

    values = blocks.astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{qtype.name} K quantize received non-finite values")

    n_blocks = values.shape[0]
    groups = values.reshape((n_blocks, 8, 32))
    group_min = groups.min(axis=-1)
    group_max = groups.max(axis=-1)
    group_min = np.minimum(group_min, np.float32(0.0))
    group_min_delta = -group_min
    scale = (group_max - group_min) / scale_divisor

    max_scale = scale.max(axis=-1, keepdims=True)
    d = (max_scale / np.float32(63.0)).astype(np.float32)
    stored_d = d.astype(np.float16)

    max_group_min_delta = group_min_delta.max(axis=-1, keepdims=True)
    dmin = (max_group_min_delta / np.float32(63.0)).astype(np.float32)
    stored_dmin = dmin.astype(np.float16)

    return groups, group_min_delta, scale, d, dmin, stored_d, stored_dmin


def k_quant_stored_scale_underflow_mask(blocks: np.ndarray, qtype: QuantType) -> np.ndarray:
    """Return Q4_K/Q5_K blocks whose non-zero scale/min cannot be represented by stored FP16 headers."""

    _, _, _, d, dmin, stored_d, stored_dmin = _k_quant_scale_components(blocks, qtype)
    scale_underflow = (d.reshape(-1) > np.float32(0.0)) & (stored_d.reshape(-1) == np.float16(0.0))
    min_underflow = (dmin.reshape(-1) > np.float32(0.0)) & (stored_dmin.reshape(-1) == np.float16(0.0))
    return scale_underflow | min_underflow


def quantize_blocks_q4_k(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 256) to GGML Q4_K packed blocks (n, 144)."""
    groups, dm, scale, d, dmin, stored_d, stored_dmin = _k_quant_scale_components(blocks, QuantType.Q4_K)
    n_blocks = groups.shape[0]

    scaled_sc = np.zeros_like(scale, dtype=np.float32)
    np.divide(scale, d, out=scaled_sc, where=stored_d != np.float16(0.0))
    sc = np.rint(scaled_sc).astype(np.int32)
    sc = np.clip(sc, 0, 63).astype(np.uint8)

    scaled_mn = np.zeros_like(dm, dtype=np.float32)
    np.divide(dm, dmin, out=scaled_mn, where=stored_dmin != np.float16(0.0))
    mn = np.rint(scaled_mn).astype(np.int32)
    mn = np.clip(mn, 0, 63).astype(np.uint8)

    scale_q = d * sc.astype(np.float32)
    dm_q = dmin * mn.astype(np.float32)

    q_scaled = np.zeros_like(groups, dtype=np.float32)
    np.divide(
        groups + dm_q[:, :, None],
        scale_q[:, :, None],
        out=q_scaled,
        where=scale_q[:, :, None] != np.float32(0.0),
    )
    q = np.rint(q_scaled)
    q = np.clip(q, 0, 15).astype(np.uint8)

    header_d = stored_d.view(np.uint8)
    header_dmin = stored_dmin.view(np.uint8)
    scales_packed = _pack_k_scale_min(sc, mn)

    even = q[:, 0:8:2, :]
    odd = q[:, 1:8:2, :]
    qs = (even & np.uint8(0x0F)) | (odd << np.uint8(4))
    qs = qs.reshape((n_blocks, 128))

    return np.concatenate([header_d, header_dmin, scales_packed, qs], axis=-1)


def quantize_blocks_q5_k(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 256) to GGML Q5_K packed blocks (n, 176)."""
    groups, dm, scale, d, dmin, stored_d, stored_dmin = _k_quant_scale_components(blocks, QuantType.Q5_K)
    n_blocks = groups.shape[0]

    scaled_sc = np.zeros_like(scale, dtype=np.float32)
    np.divide(scale, d, out=scaled_sc, where=stored_d != np.float16(0.0))
    sc = np.rint(scaled_sc).astype(np.int32)
    sc = np.clip(sc, 0, 63).astype(np.uint8)

    scaled_mn = np.zeros_like(dm, dtype=np.float32)
    np.divide(dm, dmin, out=scaled_mn, where=stored_dmin != np.float16(0.0))
    mn = np.rint(scaled_mn).astype(np.int32)
    mn = np.clip(mn, 0, 63).astype(np.uint8)

    scale_q = d * sc.astype(np.float32)
    dm_q = dmin * mn.astype(np.float32)

    q_scaled = np.zeros_like(groups, dtype=np.float32)
    np.divide(
        groups + dm_q[:, :, None],
        scale_q[:, :, None],
        out=q_scaled,
        where=scale_q[:, :, None] != np.float32(0.0),
    )
    q = np.rint(q_scaled)
    q = np.clip(q, 0, 31).astype(np.uint8)

    ql = q & np.uint8(0x0F)
    qh_bits = (q >> np.uint8(4)) & np.uint8(1)

    even = ql[:, 0:8:2, :]
    odd = ql[:, 1:8:2, :]
    qs = (even & np.uint8(0x0F)) | (odd << np.uint8(4))
    qs = qs.reshape((n_blocks, 128))

    weights = (np.uint8(1) << np.arange(8, dtype=np.uint8)).reshape((1, 8, 1))
    qh = (qh_bits.astype(np.uint8) * weights).sum(axis=1).astype(np.uint8)  # (n_blocks, 32)

    header_d = stored_d.view(np.uint8)
    header_dmin = stored_dmin.view(np.uint8)
    scales_packed = _pack_k_scale_min(sc, mn)

    return np.concatenate([header_d, header_dmin, scales_packed, qh, qs], axis=-1)


def quantize_blocks_q3_k(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 256) to GGML Q3_K packed blocks (n, 110).

    This is a pragmatic encoder focused on producing a valid GGML layout for
    tooling/runtime. It does not attempt to exactly match llama.cpp’s reference
    quantizer heuristics.
    """
    if blocks.ndim != 2 or blocks.shape[1] != 256:
        raise ValueError(f"Q3_K quantize expects (n_blocks, 256), got {blocks.shape}")

    x = blocks.astype(np.float32, copy=False)
    n_blocks = x.shape[0]

    groups = x.reshape((n_blocks, 16, 16))
    amax = np.max(np.abs(groups), axis=-1)  # (n_blocks, 16)
    amax_global = np.max(amax, axis=1, keepdims=True)  # (n_blocks, 1)

    # Effective per-group scale is (d * sc[g]); q values are in [-4, 3].
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(amax_global == 0, 0, amax_global / np.float32(31.0 * 4.0)).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(d == 0, 0, np.rint(amax / (np.float32(4.0) * d))).astype(np.int32)

    sc = np.clip(sc, 0, 31).astype(np.int8)  # (n_blocks, 16)

    eff = (d * sc.astype(np.float32)).reshape((n_blocks, 16, 1))  # (n_blocks, 16, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(eff == 0, 0, np.rint(groups / eff)).astype(np.int32)

    q = np.clip(q, -4, 3)

    pos = (q >= 0).astype(np.uint8)  # (n_blocks, 16, 16)
    ql = np.where(q >= 0, q, q + 4).astype(np.uint8)  # (n_blocks, 16, 16), 0..3

    # hmask: 32 bytes, each byte contains 8 bit-planes (groups of 32 values).
    pos_flat = pos.reshape((n_blocks, 256)).reshape((n_blocks, 8, 32))
    weights = (np.uint8(1) << np.arange(8, dtype=np.uint8)).reshape((1, 8, 1))
    hmask = (pos_flat * weights).sum(axis=1).astype(np.uint8)  # (n_blocks, 32)

    # qs: pack 2-bit values into 64 bytes.
    ql_flat = ql.reshape((n_blocks, 256))
    seg = ql_flat.reshape((n_blocks, 2, 128))
    chunks = seg.reshape((n_blocks, 2, 4, 32))
    qs = (
        (chunks[:, :, 0, :] & np.uint8(0x03))
        | ((chunks[:, :, 1, :] & np.uint8(0x03)) << np.uint8(2))
        | ((chunks[:, :, 2, :] & np.uint8(0x03)) << np.uint8(4))
        | ((chunks[:, :, 3, :] & np.uint8(0x03)) << np.uint8(6))
    ).astype(np.uint8)
    qs = qs.reshape((n_blocks, 64))

    # scales: 12 bytes packing 16 6-bit values into (8 low-nibble bytes + 4 high-bits bytes).
    l6 = (sc.astype(np.int32) + 32).astype(np.uint8)  # 0..63
    scales = np.zeros((n_blocks, 12), dtype=np.uint8)

    low = l6 & np.uint8(0x0F)
    scales[:, :8] = low[:, :8]
    scales[:, :8] |= (low[:, 8:] << np.uint8(4))

    high = (l6 >> np.uint8(4)) & np.uint8(0x03)
    for j in range(16):
        scales[:, 8 + (j % 4)] |= (high[:, j] << np.uint8(2 * (j // 4)))

    d_bytes = d.astype(np.float16).view(np.uint8)

    return np.concatenate([hmask, qs, scales, d_bytes], axis=1)


def quantize_blocks_q2_k(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 256) to GGML Q2_K packed blocks (n, 84).

    This is a pragmatic encoder focused on producing a valid GGML layout for
    tooling/runtime. It does not attempt to exactly match llama.cpp’s reference
    quantizer heuristics.
    """
    if blocks.ndim != 2 or blocks.shape[1] != 256:
        raise ValueError(f"Q2_K quantize expects (n_blocks, 256), got {blocks.shape}")

    x = blocks.astype(np.float32, copy=False)
    n_blocks = x.shape[0]

    groups = x.reshape((n_blocks, 16, 16))

    g_min = groups.min(axis=-1)  # (n_blocks, 16)
    g_min = np.minimum(g_min, 0.0)
    g_max = groups.max(axis=-1)
    g_range = g_max - g_min

    max_range = np.max(g_range, axis=1, keepdims=True)  # (n_blocks, 1)
    max_dm = np.max(-g_min, axis=1, keepdims=True)  # (n_blocks, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(max_range == 0, 0, max_range / np.float32(3.0 * 15.0)).astype(np.float32)
        dmin = np.where(max_dm == 0, 0, max_dm / np.float32(15.0)).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(d == 0, 0, np.rint(g_range / (np.float32(3.0) * d))).astype(np.int32)
        m = np.where(dmin == 0, 0, np.rint((-g_min) / dmin)).astype(np.int32)

    s = np.clip(s, 0, 15).astype(np.uint8)
    m = np.clip(m, 0, 15).astype(np.uint8)

    dl = d * s.astype(np.float32)  # (n_blocks, 16)
    ml = dmin * m.astype(np.float32)  # (n_blocks, 16)
    dl = dl.reshape((n_blocks, 16, 1))
    ml = ml.reshape((n_blocks, 16, 1))

    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(dl == 0, 0, np.rint((groups + ml) / dl)).astype(np.int32)
    q = np.clip(q, 0, 3).astype(np.uint8)

    q_flat = q.reshape((n_blocks, 256))
    seg = q_flat.reshape((n_blocks, 2, 128))
    chunks = seg.reshape((n_blocks, 2, 4, 32))
    qs = (
        (chunks[:, :, 0, :] & np.uint8(0x03))
        | ((chunks[:, :, 1, :] & np.uint8(0x03)) << np.uint8(2))
        | ((chunks[:, :, 2, :] & np.uint8(0x03)) << np.uint8(4))
        | ((chunks[:, :, 3, :] & np.uint8(0x03)) << np.uint8(6))
    ).astype(np.uint8)
    qs = qs.reshape((n_blocks, 64))

    scales = (s & np.uint8(0x0F)) | ((m & np.uint8(0x0F)) << np.uint8(4))

    d_bytes = d.astype(np.float16).view(np.uint8)
    dmin_bytes = dmin.astype(np.float16).view(np.uint8)

    return np.concatenate([scales, qs, d_bytes, dmin_bytes], axis=1)


def quantize_blocks_q6_k(blocks: np.ndarray) -> np.ndarray:
    """Quantize float32 blocks (n, 256) to GGML Q6_K packed blocks (n, 210)."""
    if blocks.ndim != 2 or blocks.shape[1] != 256:
        raise ValueError(f"Q6_K quantize expects (n_blocks, 256), got {blocks.shape}")

    x = blocks.astype(np.float32, copy=False)
    n_blocks = x.shape[0]

    # 16 groups of 16 elements; each group shares a scale factor (d * scales[g]).
    groups = x.reshape((n_blocks, 16, 16))
    amax = np.max(np.abs(groups), axis=-1)  # (n_blocks, 16)

    # Target q range is [-32, 31] => 31 is the max positive magnitude.
    target_scale = amax / np.float32(31.0)  # (n_blocks, 16)
    max_scale = np.max(target_scale, axis=1, keepdims=True)  # (n_blocks, 1)

    # Store d as a float16 scalar, and 16 int8 scales as multipliers in [0..127].
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(max_scale == 0, 0, max_scale / np.float32(127.0)).astype(np.float32)
        scales = np.where(d == 0, 0, np.rint(target_scale / d)).astype(np.int32)

    scales = np.clip(scales, 0, 127).astype(np.int8)  # (n_blocks, 16)
    eff = (d * scales.astype(np.float32)).reshape((n_blocks, 16, 1))

    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(eff == 0, 0, np.rint(groups / eff)).astype(np.int32)

    q = np.clip(q, -32, 31).astype(np.int32)
    q_enc = (q + 32).astype(np.uint8).reshape((n_blocks, 8, 32))  # 0..63

    ql_unpacked = q_enc & np.uint8(0x0F)  # 0..15, (n_blocks, 8, 32)
    qh_unpacked = (q_enc >> np.uint8(4)) & np.uint8(0x03)  # 0..3

    # Reverse the dequant unpack order.
    ql_shifted = ql_unpacked.reshape((n_blocks, 2, 2, 64))
    ql_bytes = (ql_shifted[:, :, 0, :] & np.uint8(0x0F)) | (ql_shifted[:, :, 1, :] << np.uint8(4))
    ql_bytes = ql_bytes.reshape((n_blocks, 128))

    qh_shifted = qh_unpacked.reshape((n_blocks, 2, 4, 32))
    qh_bytes = (
        (qh_shifted[:, :, 0, :] & np.uint8(0x03))
        | ((qh_shifted[:, :, 1, :] & np.uint8(0x03)) << np.uint8(2))
        | ((qh_shifted[:, :, 2, :] & np.uint8(0x03)) << np.uint8(4))
        | ((qh_shifted[:, :, 3, :] & np.uint8(0x03)) << np.uint8(6))
    )
    qh_bytes = qh_bytes.reshape((n_blocks, 64))

    scales_bytes = scales.astype(np.int8, copy=False).view(np.uint8)
    d_bytes = d.astype(np.float16).view(np.uint8)

    return np.concatenate([ql_bytes, qh_bytes, scales_bytes, d_bytes], axis=1)


QUANTIZE_NUMPY_BY_TYPE: dict[QuantType, callable] = {
    QuantType.Q8_0: quantize_blocks_q8_0,
    QuantType.Q4_0: quantize_blocks_q4_0,
    QuantType.Q4_1: quantize_blocks_q4_1,
    QuantType.Q5_0: quantize_blocks_q5_0,
    QuantType.Q5_1: quantize_blocks_q5_1,
    QuantType.IQ4_NL: quantize_blocks_iq4_nl,
    QuantType.Q2_K: quantize_blocks_q2_k,
    QuantType.Q3_K: quantize_blocks_q3_k,
    QuantType.Q4_K: quantize_blocks_q4_k,
    QuantType.Q5_K: quantize_blocks_q5_k,
    QuantType.Q6_K: quantize_blocks_q6_k,
}
