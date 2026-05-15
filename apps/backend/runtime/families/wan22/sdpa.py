"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDPA backend selection helpers for WAN runtimes.
Provides a configurable `sdpa(...)` wrapper with optional chunking and strict policy validation,
delegating per-call SDPA execution to the central attention dispatcher and carrying the generic
SRAM-attention mode context for WAN22 self-attention integration.

Symbols (top-level; keep in sync; no ghosts):
- `_SDPA_SETTINGS_CTX` (constant): Context-local SDPA settings tuple (`policy`, `mode`, `chunk`, `sram_mode`, `request_id`).
- `_normalize_sram_mode` (function): Validates and normalizes SRAM attention mode input via the generic runtime bridge.
- `_normalize_sdpa_settings` (function): Validates and normalizes SDPA policy/chunk/mode/sram_mode inputs.
- `set_sdpa_settings` (function): Applies policy/chunk/mode/SRAM settings (explicit args override env when provided).
- `_get_sdpa_settings` (function): Reads effective context-local SDPA settings tuple.
- `get_sram_attention_mode` (function): Returns the effective SRAM attention mode (`off|auto|force`) for the active context.
- `sdpa` (function): Calls PyTorch SDPA using the configured backend policy and optional chunking.
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

from contextvars import ContextVar
import logging
from typing import Optional
from uuid import uuid4

import torch

from apps.backend.runtime.attention import attention_function_pre_shaped, set_attention_request_id
from apps.backend.runtime.attention.sram import resolve_effective_sram_attention_mode
from apps.backend.runtime.memory.config import AttentionBackend

_LOGGER = get_backend_logger("backend.runtime.wan22.sdpa")
_SDPA_CALL_COUNT_CTX: ContextVar[int] = ContextVar("wan22_sdpa_call_count", default=0)
_CROSS_ATTN_SLIDING_FALLBACK_LOGGED_CTX: ContextVar[bool] = ContextVar(
    "wan22_cross_attn_sliding_fallback_logged",
    default=False,
)

_SRAM_MODE_ENV = "CODEX_ATTENTION_SRAM_MODE"

_SDPA_SETTINGS_CTX: ContextVar[tuple[str, str, int, str, str]] = ContextVar(
    "wan22_sdpa_settings",
    default=("auto", "global", 0, "off", "wan22-unknown"),
)


def _normalize_sram_mode(sram_mode: Optional[str]) -> str:
    return resolve_effective_sram_attention_mode(sram_mode).value


def _normalize_sdpa_settings(
    policy: Optional[str],
    chunk: Optional[int],
    attention_mode: Optional[str],
    sram_mode: Optional[str],
) -> tuple[str, str, int, str]:
    if policy is not None and not isinstance(policy, str):
        raise TypeError(f"WAN22 SDPA: policy must be a string when provided, got {type(policy).__name__}.")
    pol = str(policy if policy is not None else "auto").strip().lower()
    if pol not in ("auto", "mem_efficient", "flash", "math"):
        raise RuntimeError(
            "WAN22 SDPA: unsupported policy "
            f"{policy!r} (expected one of: 'auto', 'mem_efficient', 'flash', 'math')."
        )
    mode = str(attention_mode if attention_mode is not None else "global").strip().lower()
    if mode not in ("global", "sliding"):
        raise RuntimeError(f"WAN22 SDPA: unsupported attention mode {attention_mode!r} (expected 'global' or 'sliding').")
    if chunk is None:
        ch = 0
    else:
        try:
            chunk_value = int(chunk)
        except Exception as exc:
            raise RuntimeError(f"WAN22 SDPA: chunk must be an integer when provided, got {chunk!r}.") from exc
        ch = chunk_value if chunk_value > 0 else 0
    normalized_sram_mode = _normalize_sram_mode(sram_mode)
    return pol, mode, ch, normalized_sram_mode


def set_sdpa_settings(
    policy: Optional[str],
    chunk: Optional[int],
    attention_mode: Optional[str] = None,
    sram_mode: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    pol, mode, ch, normalized_sram_mode = _normalize_sdpa_settings(policy, chunk, attention_mode, sram_mode)
    rid = str(request_id or "").strip() or f"wan22-{uuid4().hex[:12]}"
    _SDPA_SETTINGS_CTX.set((pol, mode, ch, normalized_sram_mode, rid))
    _SDPA_CALL_COUNT_CTX.set(0)
    _CROSS_ATTN_SLIDING_FALLBACK_LOGGED_CTX.set(False)
    set_attention_request_id(rid)


def _get_sdpa_settings() -> tuple[str, str, int, str, str]:
    pol, mode, ch, sram_mode, rid = _SDPA_SETTINGS_CTX.get()
    return str(pol), str(mode), int(ch), str(sram_mode), str(rid)


def get_sram_attention_mode() -> str:
    _pol, _mode, _chunk, sram_mode, _request_id = _get_sdpa_settings()
    return sram_mode


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
    pol, mode, ch, sram_mode, request_id = _get_sdpa_settings()
    set_attention_request_id(request_id)
    call_count = int(_SDPA_CALL_COUNT_CTX.get()) + 1
    _SDPA_CALL_COUNT_CTX.set(call_count)
    if call_count == 1:
        emit_backend_message(
            "[wan22.sdpa] first call",
            logger=_LOGGER.name,
            request_id=request_id,
            policy=pol,
            mode=mode,
            chunk=ch,
            device=str(q.device),
            dtype=str(q.dtype),
            qkv=(tuple(q.shape), tuple(k.shape), tuple(v.shape)),
            sram_mode=sram_mode,
            env=_SRAM_MODE_ENV,
        )

    if mode == "sliding":
        if ch <= 0:
            raise RuntimeError("WAN22 SDPA: sliding attention mode requires gguf_attn_chunk > 0.")
        q_length = int(q.shape[2])
        kv_length = int(k.shape[2])
        if kv_length != q_length:
            if not bool(_CROSS_ATTN_SLIDING_FALLBACK_LOGGED_CTX.get()):
                _CROSS_ATTN_SLIDING_FALLBACK_LOGGED_CTX.set(True)
                emit_backend_message(
                    "[wan22.sdpa] sliding mode fallback",
                    logger=_LOGGER.name,
                    level=logging.WARNING,
                    request_id=request_id,
                    q_len=q_length,
                    kv_len=kv_length,
                    reason="using full K/V per query chunk",
                )
            out_accum: torch.Tensor | None = None
            for start in range(0, q_length, ch):
                end = min(q_length, start + ch)
                chunk_out = attention_function_pre_shaped(
                    q[:, :, start:end],
                    k,
                    v,
                    is_causal=causal,
                    backend=AttentionBackend.PYTORCH,
                    sdpa_policy=pol,
                )
                if out_accum is None:
                    out_accum = torch.empty(
                        (
                            int(chunk_out.shape[0]),
                            int(chunk_out.shape[1]),
                            q_length,
                            int(chunk_out.shape[3]),
                        ),
                        device=chunk_out.device,
                        dtype=chunk_out.dtype,
                    )
                out_accum[:, :, start:end, :] = chunk_out
            if out_accum is None:
                raise RuntimeError("WAN22 SDPA: sliding cross-attention fallback produced no output chunks.")
            return out_accum
        _, _, length, _ = q.shape
        out_accum: torch.Tensor | None = None
        for start in range(0, length, ch):
            end = min(length, start + ch)
            window_start = max(0, start - ch)
            window_end = min(length, end + ch)
            chunk_out = attention_function_pre_shaped(
                q[:, :, start:end],
                k[:, :, window_start:window_end],
                v[:, :, window_start:window_end],
                is_causal=causal,
                backend=AttentionBackend.PYTORCH,
                sdpa_policy=pol,
            )
            if out_accum is None:
                out_accum = torch.empty(
                    (
                        int(chunk_out.shape[0]),
                        int(chunk_out.shape[1]),
                        length,
                        int(chunk_out.shape[3]),
                    ),
                    device=chunk_out.device,
                    dtype=chunk_out.dtype,
                )
            out_accum[:, :, start:end, :] = chunk_out
        if out_accum is None:
            raise RuntimeError("WAN22 SDPA: sliding self-attention produced no output chunks.")
        return out_accum

    if mode == "global" and ch > 0:
        _, _, length, _ = q.shape
        out_accum: torch.Tensor | None = None
        for start in range(0, length, ch):
            end = min(length, start + ch)
            chunk_out = attention_function_pre_shaped(
                q[:, :, start:end],
                k,
                v,
                is_causal=causal,
                backend=AttentionBackend.PYTORCH,
                sdpa_policy=pol,
            )
            if out_accum is None:
                out_accum = torch.empty(
                    (
                        int(chunk_out.shape[0]),
                        int(chunk_out.shape[1]),
                        length,
                        int(chunk_out.shape[3]),
                    ),
                    device=chunk_out.device,
                    dtype=chunk_out.dtype,
                )
            out_accum[:, :, start:end, :] = chunk_out
        if out_accum is None:
            raise RuntimeError("WAN22 SDPA: global chunked attention produced no output chunks.")
        return out_accum
    return attention_function_pre_shaped(
        q,
        k,
        v,
        is_causal=causal,
        backend=AttentionBackend.PYTORCH,
        sdpa_policy=pol,
    )
