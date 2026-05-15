"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Keyspace resolver helpers for llama.cpp-style GGUF tensor names.
Provides strict, fail-loud mapping from common llama.cpp GGUF tensor keys (e.g. `token_embd.weight`, `blk.N.attn_q.weight`)
into HuggingFace/Codex-native parameter names used by runtime modules without any generic source-key rewrite seam.

Symbols (top-level; keep in sync; no ghosts):
- `QWEN3_LLAMA_GGUF_LAYER_SUFFIX_TO_HF_PREFIX` (constant): Qwen3 per-layer GGUF suffix → HF prefix (without `.weight`/`.bias`).
- `resolve_llama_gguf_text_model_keyspace` (function): Resolves llama.cpp GGUF-ish state_dict keys to canonical HF keys.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import Mapping, TypeVar

from apps.backend.runtime.state_dict.key_mapping import (
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


QWEN3_LLAMA_GGUF_LAYER_SUFFIX_TO_HF_PREFIX: dict[str, str] = {
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
    "ffn_norm": "post_attention_layernorm",
}

_DIRECT_TO_HF = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
}

_RX_LAYER_PARAM = re.compile(r"^blk\.(?P<idx>\d+)\.(?P<suffix>[a-z0-9_]+)\.(?P<param>weight|bias)$")

_DETECTOR = KeyStyleDetector(
    name="llama_gguf_or_hf_text_model_key_style",
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
                KeySentinel(SentinelKind.PREFIX, "model.layers."),
                KeySentinel(SentinelKind.PREFIX, "model.embed_tokens."),
                KeySentinel(SentinelKind.PREFIX, "model.norm."),
            ),
            min_sentinel_hits=1,
        ),
    ),
)


def resolve_llama_gguf_text_model_keyspace(
    state_dict: MutableMapping[str, _T],
    *,
    num_layers: int,
    layer_suffix_to_hf_prefix: Mapping[str, str],
) -> ResolvedKeyspace[_T]:
    """Resolve llama.cpp GGUF tensor keys into HF-style keys.

    - If the input already uses HF keys, this is a no-op.
    - If the input looks like llama.cpp GGUF keys, this is strict and fails loud on unknown keys.
    """

    def _map_llama_gguf(key: str) -> str:
        direct = _DIRECT_TO_HF.get(key)
        if direct is not None:
            return direct

        m = _RX_LAYER_PARAM.match(key)
        if not m:
            raise KeyMappingError(f"llama.cpp GGUF resolver: unsupported key={key!r}")

        idx = int(m.group("idx"))
        if idx < 0 or idx >= int(num_layers):
            raise KeyMappingError(
                f"llama.cpp GGUF resolver: layer index out of range (idx={idx}, num_layers={num_layers}) for key={key!r}"
            )

        suffix = m.group("suffix")
        prefix = layer_suffix_to_hf_prefix.get(suffix)
        if prefix is None:
            raise KeyMappingError(
                f"llama.cpp GGUF resolver: unknown per-layer key suffix={suffix!r} for key={key!r}"
            )

        param = m.group("param")
        return f"model.layers.{idx}.{prefix}.{param}"

    resolved = resolve_state_dict_keyspace(
        state_dict,
        detector=_DETECTOR,
        mappers={
            KeyStyle.HF: lambda k: k,
            KeyStyle.LLAMA_GGUF: _map_llama_gguf,
        },
    )
    resolved.metadata.setdefault("resolver", "llama_gguf_text_model")
    resolved.metadata.setdefault("num_layers", int(num_layers))
    return resolved


__all__ = [
    "QWEN3_LLAMA_GGUF_LAYER_SUFFIX_TO_HF_PREFIX",
    "resolve_llama_gguf_text_model_keyspace",
]
