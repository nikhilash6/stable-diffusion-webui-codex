"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Gemma3 text-encoder keyspace resolver for llama.cpp-style GGUF tensor names.
Provides strict, fail-loud mapping from Gemma3 text-only GGUF tensor keys (for example `token_embd.weight`,
`blk.N.attn_q.weight`) into the native `Gemma3TextModel` lookup keyspace without materializing renamed tensors; wrapper-prefix rewrite attempts fail loud.

Symbols (top-level; keep in sync; no ghosts):
- `GEMMA3_LLAMA_GGUF_LAYER_SUFFIX_TO_HF_PREFIX` (constant): Gemma3 per-layer GGUF suffix → `Gemma3TextModel` prefix.
- `resolve_gemma3_text_encoder_keyspace` (function): Resolves llama.cpp GGUF or native Gemma3-text keys into canonical `Gemma3TextModel` lookup keys.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import TypeVar

from apps.backend.runtime.state_dict.key_mapping import (
    fail_on_key_name_rewrite,
    KeyMappingError,
    KeySentinel,
    KeyStyle,
    KeyStyleDetector,
    KeyStyleSpec,
    ResolvedKeyspace,
    SentinelKind,
    resolve_state_dict_keyspace,
)

_T = TypeVar("_T")


GEMMA3_LLAMA_GGUF_LAYER_SUFFIX_TO_HF_PREFIX: dict[str, str] = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_norm": "pre_feedforward_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
}

_DIRECT_TO_HF = {
    "token_embd.weight": "embed_tokens.weight",
    "output_norm.weight": "norm.weight",
}
_HF_WRAPPER_PREFIXES = ("base_text_encoder.", "language_model.", "model.")

_RX_LAYER_PARAM = re.compile(r"^blk\.(?P<idx>\d+)\.(?P<suffix>[a-z0-9_]+)\.(?P<param>weight|bias)$")

_DETECTOR = KeyStyleDetector(
    name="gemma3_text_encoder_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.LLAMA_GGUF,
            sentinels=(
                KeySentinel(SentinelKind.EXACT, "token_embd.weight"),
                KeySentinel(SentinelKind.PREFIX, "blk."),
                KeySentinel(SentinelKind.EXACT, "output_norm.weight"),
            ),
            min_sentinel_hits=1,
        ),
        KeyStyleSpec(
            style=KeyStyle.HF,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "layers."),
                KeySentinel(SentinelKind.PREFIX, "embed_tokens."),
                KeySentinel(SentinelKind.PREFIX, "norm."),
            ),
            min_sentinel_hits=1,
        ),
    ),
)


def resolve_gemma3_text_encoder_keyspace(
    state_dict: MutableMapping[str, _T],
    *,
    num_layers: int,
) -> ResolvedKeyspace[_T]:
    """Resolve Gemma3 text GGUF tensor keys into `Gemma3TextModel` lookup keys."""

    def _map_llama_gguf(key: str) -> str:
        direct = _DIRECT_TO_HF.get(key)
        if direct is not None:
            return direct

        match = _RX_LAYER_PARAM.match(key)
        if match is None:
            raise KeyMappingError(f"Gemma3 text GGUF resolver: unsupported key={key!r}")

        idx = int(match.group("idx"))
        if idx < 0 or idx >= int(num_layers):
            raise KeyMappingError(
                "Gemma3 text GGUF resolver: layer index out of range "
                f"(idx={idx}, num_layers={num_layers}) for key={key!r}"
            )

        suffix = match.group("suffix")
        prefix = GEMMA3_LLAMA_GGUF_LAYER_SUFFIX_TO_HF_PREFIX.get(suffix)
        if prefix is None:
            raise KeyMappingError(
                f"Gemma3 text GGUF resolver: unknown per-layer key suffix={suffix!r} for key={key!r}"
            )

        param = match.group("param")
        return f"layers.{idx}.{prefix}.{param}"

    resolved = resolve_state_dict_keyspace(
        state_dict,
        detector=_DETECTOR,
        source_key_guard=lambda key: fail_on_key_name_rewrite(key, _HF_WRAPPER_PREFIXES),
        mappers={
            KeyStyle.HF: lambda key: key,
            KeyStyle.LLAMA_GGUF: _map_llama_gguf,
        },
    )
    resolved.metadata.setdefault("resolver", "gemma3_text_encoder")
    resolved.metadata.setdefault("num_layers", int(num_layers))
    resolved.metadata.setdefault("hf_wrapper_prefixes", _HF_WRAPPER_PREFIXES)
    return resolved


__all__ = [
    "GEMMA3_LLAMA_GGUF_LAYER_SUFFIX_TO_HF_PREFIX",
    "resolve_gemma3_text_encoder_keyspace",
]
