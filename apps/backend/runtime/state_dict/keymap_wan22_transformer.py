"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 transformer key-style detection + keyspace resolution (Diffusers/WAN-export/Codex) plus explicit WAN LoRA logical-key mapping.
Resolves multiple upstream key layouts into the canonical Codex WAN22 runtime keyspace via lookup views, models supported WAN LoRA source-key families explicitly, and fails loud on unknown/ambiguous transformer inputs or unsupported wrapper/prefix rewrite attempts.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_wan22_lora_logical_key` (function): Maps WAN22 LoRA logical keys to canonical WAN22 transformer target keys.
- `resolve_wan22_transformer_keyspace` (function): Resolves WAN22 transformer keys into canonical keyspace (`ResolvedKeyspace`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, Sequence
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

_PREFIXES = (
    "model.model.diffusion_model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
)

_EXPORT_TO_CODEX_PREFIX_ALIASES = (
    ("patch_embedding.", "patch_embed."),
    ("time_embedding.", "time_embed."),
    ("time_projection.", "time_proj."),
    ("text_embedding.", "text_embed."),
    ("head.head.", "head."),
)

_RX_BLOCK_ATTN = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>attn1|attn2)\.to_(?P<proj>[qkv])\.(?P<param>weight|bias)$"
)
_RX_BLOCK_ATTN_OUT = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>attn1|attn2)\.to_out\.0\.(?P<param>weight|bias)$"
)
_RX_BLOCK_ATTN_NORM = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>attn1|attn2)\.norm_(?P<norm>[qk])\.weight$"
)
_RX_BLOCK_FFN_PROJ = re.compile(
    r"^blocks\.(?P<idx>\d+)\.ffn\.net\.(?P<which>0\.proj|2)\.(?P<param>weight|bias)$"
)
_RX_BLOCK_NORM2 = re.compile(r"^blocks\.(?P<idx>\d+)\.norm2\.(?P<param>weight|bias)$")
_RX_BLOCK_NORM3 = re.compile(r"^blocks\.(?P<idx>\d+)\.norm3\.(?P<param>weight|bias)$")
_RX_BLOCK_SCALE_SHIFT = re.compile(r"^blocks\.(?P<idx>\d+)\.scale_shift_table$")
_RX_LORA_CANONICAL_ATTN_DOT = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>self_attn|cross_attn)\.(?P<proj>q|k|v|o)$"
)
_RX_LORA_CANONICAL_FFN_DOT = re.compile(r"^blocks\.(?P<idx>\d+)\.ffn\.(?P<which>0|2)$")
_RX_LORA_CANONICAL_ATTN_NORM_DOT = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>self_attn|cross_attn)\.norm_(?P<norm>q|k)$"
)
_RX_LORA_CANONICAL_NORM3_DOT = re.compile(r"^blocks\.(?P<idx>\d+)\.norm3$")
_RX_LORA_CANONICAL_MODULATION_DOT = re.compile(r"^blocks\.(?P<idx>\d+)\.modulation$")
_RX_LORA_CANONICAL_ATTN_UNDERSCORE = re.compile(
    r"^blocks_(?P<idx>\d+)_(?P<which>self_attn|cross_attn)_(?P<proj>q|k|v|o)$"
)
_RX_LORA_CANONICAL_FFN_UNDERSCORE = re.compile(r"^blocks_(?P<idx>\d+)_ffn_(?P<which>0|2)$")
_RX_LORA_CANONICAL_ATTN_NORM_UNDERSCORE = re.compile(
    r"^blocks_(?P<idx>\d+)_(?P<which>self_attn|cross_attn)_norm_(?P<norm>q|k)$"
)
_RX_LORA_CANONICAL_NORM3_UNDERSCORE = re.compile(r"^blocks_(?P<idx>\d+)_norm3$")
_RX_LORA_CANONICAL_MODULATION_UNDERSCORE = re.compile(r"^blocks_(?P<idx>\d+)_modulation$")
_RX_LORA_DIFFUSERS_ATTN_DOT = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>attn1|attn2)\.to_(?P<proj>q|k|v)$"
)
_RX_LORA_DIFFUSERS_ATTN_UNDERSCORE = re.compile(
    r"^blocks_(?P<idx>\d+)_(?P<which>attn1|attn2)_to_(?P<proj>q|k|v)$"
)
_RX_LORA_DIFFUSERS_OUT_DOT = re.compile(r"^blocks\.(?P<idx>\d+)\.(?P<which>attn1|attn2)\.to_out\.0$")
_RX_LORA_DIFFUSERS_OUT_UNDERSCORE = re.compile(r"^blocks_(?P<idx>\d+)_(?P<which>attn1|attn2)_to_out_0$")
_RX_LORA_DIFFUSERS_FFN_DOT = re.compile(r"^blocks\.(?P<idx>\d+)\.ffn\.net\.(?P<which>0\.proj|2)$")
_RX_LORA_DIFFUSERS_FFN_UNDERSCORE = re.compile(r"^blocks_(?P<idx>\d+)_ffn_net_(?P<which>0_proj|2)$")
_LEGACY_LORA_LOGICAL_WRAPPER_PREFIXES = ("lora_unet_", "lycoris_")
_LORA_SOURCE_PREFIXES = (
    "model.model.diffusion_model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer_2.",
    "transformer.",
    "model.",
)
_LORA_TOP_LEVEL_TARGETS = {
    "patch_embedding": "patch_embed.weight",
    "patch_embed": "patch_embed.weight",
    "time_embedding.0": "time_embed.0.weight",
    "time_embed.0": "time_embed.0.weight",
    "time_embedding.2": "time_embed.2.weight",
    "time_embed.2": "time_embed.2.weight",
    "time_projection.1": "time_proj.1.weight",
    "time_proj.1": "time_proj.1.weight",
    "text_embedding.0": "text_embed.0.weight",
    "text_embed.0": "text_embed.0.weight",
    "text_embedding.2": "text_embed.2.weight",
    "text_embed.2": "text_embed.2.weight",
    "head.head": "head.weight",
    "head": "head.weight",
    "head.modulation": "head_modulation",
    "head_modulation": "head_modulation",
}

_DETECTOR = KeyStyleDetector(
    name="wan22_transformer_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.DIFFUSERS,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "condition_embedder."),
                KeySentinel(SentinelKind.SUBSTRING, ".attn1."),
                KeySentinel(SentinelKind.SUBSTRING, ".attn2."),
                KeySentinel(SentinelKind.SUBSTRING, ".ffn.net."),
                KeySentinel(SentinelKind.PREFIX, "proj_out."),
                KeySentinel(SentinelKind.EXACT, "scale_shift_table"),
                KeySentinel(SentinelKind.SUBSTRING, ".scale_shift_table"),
            ),
            min_sentinel_hits=1,
        ),
        KeyStyleSpec(
            style=KeyStyle.WAN_EXPORT,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "patch_embedding."),
                KeySentinel(SentinelKind.PREFIX, "time_embedding."),
                KeySentinel(SentinelKind.PREFIX, "time_projection."),
                KeySentinel(SentinelKind.PREFIX, "text_embedding."),
                KeySentinel(SentinelKind.PREFIX, "head.head."),
                KeySentinel(SentinelKind.EXACT, "head.modulation"),
            ),
            min_sentinel_hits=1,
        ),
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "patch_embed."),
                KeySentinel(SentinelKind.PREFIX, "time_embed."),
                KeySentinel(SentinelKind.PREFIX, "time_proj."),
                KeySentinel(SentinelKind.PREFIX, "text_embed."),
                KeySentinel(SentinelKind.EXACT, "head_modulation"),
                # A subset of canonical block keys also counts as Codex/WAN-export layout.
                KeySentinel(SentinelKind.REGEX, r"^blocks\.\d+\.(?:self_attn|cross_attn|ffn)\."),
                KeySentinel(SentinelKind.REGEX, r"^blocks\.\d+\.norm[123]\.(?:weight|bias)$"),
                KeySentinel(SentinelKind.REGEX, r"^blocks\.\d+\.modulation$"),
            ),
            min_sentinel_hits=1,
        ),
    ),
)


def resolve_wan22_lora_logical_key(logical_key: str) -> str | None:
    """Map a WAN22 LoRA logical key to a canonical WAN22 transformer target key.

    Supported logical-key families:
    - Canonical Codex style (`blocks.N.self_attn.q`, `blocks_N_self_attn_q`, `blocks.N.ffn.0`, `blocks_N_ffn_0`)
    - Canonical norm families (`blocks.N.self_attn.norm_q`, `blocks_N_self_attn_norm_q`, `blocks.N.norm3`, `blocks_N_norm3`)
    - Canonical modulation families (`blocks.N.modulation`, `blocks_N_modulation`, `head.modulation`, `head_modulation`)
    - Top-level export/native aliases (`patch_embedding`, `patch_embed`, `time_embedding.0`, `time_embed.0`,
      `time_projection.1`, `time_proj.1`, `text_embedding.0`, `text_embed.0`, `head.head`, `head`)
    - Diffusers style (`blocks.N.attn1.to_q`, `blocks_N_attn1_to_q`, `blocks.N.attn1.to_out.0`, `blocks_N_attn1_to_out_0`,
      `blocks.N.ffn.net.0.proj`, `blocks_N_ffn_net_0_proj`)
    - Explicit wrapper source families (`diffusion_model.`, `model.diffusion_model.`, `model.model.diffusion_model.`,
      `transformer.`, `transformer_2.`, `model.`)
    - Legacy trainer wrapper tags (`lora_unet_`, `lycoris_`); these are source-key wrappers, not WAN architecture owners
    """

    key = str(logical_key)
    if key.endswith(".weight"):
        key = key[: -len(".weight")]
    for prefix in _LEGACY_LORA_LOGICAL_WRAPPER_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break

    target = _resolve_wan22_lora_core_key(key)
    if target is not None:
        return target

    for source_prefix in _LORA_SOURCE_PREFIXES:
        if not key.startswith(source_prefix):
            continue
        inner_key = key[len(source_prefix) :]
        target = _resolve_wan22_lora_core_key(inner_key)
        if target is not None:
            return target
        break

    return None


def _resolve_wan22_lora_core_key(key: str) -> str | None:
    target = _LORA_TOP_LEVEL_TARGETS.get(key)
    if target is not None:
        return target

    m = _RX_LORA_CANONICAL_ATTN_DOT.match(key)
    if m:
        return f"blocks.{m.group('idx')}.{m.group('which')}.{m.group('proj')}.weight"

    m = _RX_LORA_CANONICAL_FFN_DOT.match(key)
    if m:
        return f"blocks.{m.group('idx')}.ffn.{m.group('which')}.weight"

    m = _RX_LORA_CANONICAL_ATTN_UNDERSCORE.match(key)
    if m:
        return f"blocks.{m.group('idx')}.{m.group('which')}.{m.group('proj')}.weight"

    m = _RX_LORA_CANONICAL_FFN_UNDERSCORE.match(key)
    if m:
        return f"blocks.{m.group('idx')}.ffn.{m.group('which')}.weight"

    m = _RX_LORA_CANONICAL_ATTN_NORM_DOT.match(key)
    if m:
        return f"blocks.{m.group('idx')}.{m.group('which')}.norm_{m.group('norm')}.weight"

    m = _RX_LORA_CANONICAL_NORM3_DOT.match(key)
    if m:
        return f"blocks.{m.group('idx')}.norm3.weight"

    m = _RX_LORA_CANONICAL_MODULATION_DOT.match(key)
    if m:
        return f"blocks.{m.group('idx')}.modulation"

    m = _RX_LORA_CANONICAL_ATTN_NORM_UNDERSCORE.match(key)
    if m:
        return f"blocks.{m.group('idx')}.{m.group('which')}.norm_{m.group('norm')}.weight"

    m = _RX_LORA_CANONICAL_NORM3_UNDERSCORE.match(key)
    if m:
        return f"blocks.{m.group('idx')}.norm3.weight"

    m = _RX_LORA_CANONICAL_MODULATION_UNDERSCORE.match(key)
    if m:
        return f"blocks.{m.group('idx')}.modulation"

    m = _RX_LORA_DIFFUSERS_ATTN_DOT.match(key)
    if m:
        which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
        return f"blocks.{m.group('idx')}.{which}.{m.group('proj')}.weight"

    m = _RX_LORA_DIFFUSERS_ATTN_UNDERSCORE.match(key)
    if m:
        which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
        return f"blocks.{m.group('idx')}.{which}.{m.group('proj')}.weight"

    m = _RX_LORA_DIFFUSERS_OUT_DOT.match(key)
    if m:
        which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
        return f"blocks.{m.group('idx')}.{which}.o.weight"

    m = _RX_LORA_DIFFUSERS_OUT_UNDERSCORE.match(key)
    if m:
        which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
        return f"blocks.{m.group('idx')}.{which}.o.weight"

    m = _RX_LORA_DIFFUSERS_FFN_DOT.match(key)
    if m:
        which = "0" if m.group("which") == "0.proj" else "2"
        return f"blocks.{m.group('idx')}.ffn.{which}.weight"

    m = _RX_LORA_DIFFUSERS_FFN_UNDERSCORE.match(key)
    if m:
        which = "0" if m.group("which") == "0_proj" else "2"
        return f"blocks.{m.group('idx')}.ffn.{which}.weight"

    return None


def resolve_wan22_transformer_keyspace(state_dict: MutableMapping[str, _T]) -> ResolvedKeyspace[_T]:
    def _export_to_codex(key: str) -> str:
        for export_prefix, codex_prefix in _EXPORT_TO_CODEX_PREFIX_ALIASES:
            if key.startswith(export_prefix):
                return codex_prefix + key[len(export_prefix) :]
        if key == "head.modulation":
            return "head_modulation"
        return key

    def _diffusers_to_export(key: str) -> str:
        for before, after in (
            ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
            ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
            ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
            ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
            ("condition_embedder.time_proj.", "time_projection.1."),
        ):
            if key.startswith(before):
                return after + key[len(before) :]

        if key.startswith("proj_out."):
            return "head.head." + key[len("proj_out.") :]
        if key == "scale_shift_table":
            return "head.modulation"
        if key.endswith(".scale_shift_table"):
            return key[: -len(".scale_shift_table")] + ".modulation"

        m = _RX_BLOCK_ATTN.match(key)
        if m:
            idx = m.group("idx")
            which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
            proj = m.group("proj")
            param = m.group("param")
            return f"blocks.{idx}.{which}.{proj}.{param}"

        m = _RX_BLOCK_ATTN_OUT.match(key)
        if m:
            idx = m.group("idx")
            which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
            param = m.group("param")
            return f"blocks.{idx}.{which}.o.{param}"

        m = _RX_BLOCK_ATTN_NORM.match(key)
        if m:
            idx = m.group("idx")
            which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
            norm = m.group("norm")
            return f"blocks.{idx}.{which}.norm_{norm}.weight"

        m = _RX_BLOCK_FFN_PROJ.match(key)
        if m:
            idx = m.group("idx")
            which = "0" if m.group("which") == "0.proj" else "2"
            param = m.group("param")
            return f"blocks.{idx}.ffn.{which}.{param}"

        # Diffusers uses norm1/norm2/norm3 (SA/CA/FFN), while WAN exports swap 2↔3.
        m = _RX_BLOCK_NORM2.match(key)
        if m:
            idx = m.group("idx")
            param = m.group("param")
            return f"blocks.{idx}.norm3.{param}"

        m = _RX_BLOCK_NORM3.match(key)
        if m:
            idx = m.group("idx")
            param = m.group("param")
            return f"blocks.{idx}.norm2.{param}"

        m = _RX_BLOCK_SCALE_SHIFT.match(key)
        if m:
            idx = m.group("idx")
            return f"blocks.{idx}.modulation"

        return key

    def _validate_output(keys: Sequence[str]) -> None:
        offenders: list[str] = []

        def _is_forbidden(k: str) -> bool:
            return (
                k.startswith("condition_embedder.")
                or k.startswith("proj_out.")
                or k == "scale_shift_table"
                or ".attn1." in k
                or ".attn2." in k
                or ".ffn.net." in k
                or ".scale_shift_table" in k
                or k.startswith("patch_embedding.")
                or k.startswith("time_embedding.")
                or k.startswith("time_projection.")
                or k.startswith("text_embedding.")
                or k.startswith("head.head.")
                or k == "head.modulation"
            )

        for k in keys:
            if _is_forbidden(k):
                offenders.append(k)

        if offenders:
            sample = sorted(offenders)[:10]
            raise KeyMappingError(
                "WAN22 keyspace resolver produced non-canonical keys (mapping incomplete). "
                f"offenders_sample={sample}"
            )

        # When loading a full model state dict, patch_embed is required (LoRA keyspace resolution may not include it).
        if len(keys) > 64 and not any(k.startswith("patch_embed.") for k in keys):
            preview = ", ".join(sorted(keys)[:10])
            raise KeyMappingError(
                "WAN22 keyspace resolver output is missing required patch_embed.* keys. "
                f"sample_keys=[{preview}]"
            )

    mappers = {
        KeyStyle.CODEX: lambda k: k,
        KeyStyle.WAN_EXPORT: _export_to_codex,
        KeyStyle.DIFFUSERS: lambda k: _export_to_codex(_diffusers_to_export(k)),
    }

    resolved = resolve_state_dict_keyspace(
        state_dict,
        detector=_DETECTOR,
        source_key_guard=lambda key: fail_on_key_name_rewrite(key, _PREFIXES),
        mappers=mappers,
        output_validator=_validate_output,
    )
    resolved.metadata.setdefault("resolver", "wan22_transformer")
    return resolved


__all__ = [
    "resolve_wan22_lora_logical_key",
    "resolve_wan22_transformer_keyspace",
]
