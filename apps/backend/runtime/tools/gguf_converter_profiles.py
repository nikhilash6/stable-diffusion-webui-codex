"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Converter profile registry for GGUF conversion.
Selects source/native metadata normalizers, key mappings, and per-model dtype policies.

Symbols (top-level; keep in sync; no ghosts):
- `_is_flux` (function): Detect whether a config.json describes a Flux transformer.
- `_is_zimage` (function): Detect whether a config.json describes a ZImage transformer.
- `_is_wan22` (function): Detect whether a config.json describes a WAN22 transformer.
- `_is_ltx2` (function): Detect whether a config.json describes an LTX2 transformer.
- `_is_gemma3_tenc` (function): Detect whether a config.json describes a Gemma3 text encoder export.
- `_build_llama_mapping` (function): Build a Llama HF→GGUF key mapping from the model config.
- `_COND_QUANTIZED` (constant): Condition helper matching any quantized preset (non-F16/F32).
- `_COND_FLUX_MIXED` (constant): Condition helper matching Flux mixed presets (`Q5_K_M`/`Q4_K_M`).
- `_COND_WAN22_MIXED` (constant): Condition helper matching WAN22 mixed presets (`Q5_K_M`/`Q4_K_M`).
- `FLUX_QUANT_POLICY` (constant): Flux per-tensor dtype policy (mixed presets keep more IO weights in float32).
- `WAN22_QUANT_POLICY` (constant): WAN22 per-tensor dtype policy (mixed presets keep sensitive weights in float32).
- `LTX2_QUANT_POLICY` (constant): LTX2 per-tensor dtype policy (stability-sensitive tensors stay float).
- `ZIMAGE_QUANT_POLICY` (constant): ZImage per-tensor dtype policy (pad tokens must remain float).
- `LLAMA_QUANT_POLICY` (constant): Llama per-tensor dtype policy (mixed presets bump key weights to higher precision).
- `PROFILE_REGISTRY` (constant): Registry of built-in converter profiles (table-driven dispatch).
- `_PROFILES_BY_ID` (constant): Indexed lookup table for profile ids.
- `resolve_profile` (function): Resolve the effective `ConverterProfileSpec` from a config.json.
- `profile_by_id` (function): Resolve a profile by its stable id string (no heuristics).
"""

from __future__ import annotations

from typing import Any, Mapping

from apps.backend.quantization.gguf import GGMLQuantizationType
from apps.backend.runtime.tools import gguf_converter_key_mapping as _key_mapping
from apps.backend.runtime.tools import gguf_converter_tensor_planner as _tensor_planner
from apps.backend.runtime.tools.gguf_converter_specs import (
    ConverterProfileId,
    ConverterProfileSpec,
    GGUFArch,
    KeyMappingSpec,
    QuantizationCondition,
    QuantizationPolicySpec,
    TensorNameTarget,
    TensorTypeRule,
)
from apps.backend.runtime.tools.gguf_converter_types import QuantizationType


def _is_flux(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_flux_transformer_config(config)


def _is_zimage(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_zimage_transformer_config(config)


def _is_wan22(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_wan22_transformer_config(config)


def _is_ltx2(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_ltx2_transformer_config(config)


def _is_gemma3_tenc(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_gemma3_text_encoder_config(config)


def _build_llama_mapping(config: Mapping[str, Any]) -> dict[str, str]:
    num_layers = int(config.get("num_hidden_layers", 32))
    return _key_mapping.build_key_mapping(num_layers)


_COND_QUANTIZED = QuantizationCondition(exclude=frozenset({QuantizationType.F16, QuantizationType.F32}))
_COND_FLUX_MIXED = QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M, QuantizationType.Q4_K_M}))
_COND_WAN22_MIXED = QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M, QuantizationType.Q4_K_M}))


FLUX_QUANT_POLICY = QuantizationPolicySpec(
    id="flux",
    # Required model policy: do not allow user overrides to violate these.
    required_rules=(
        TensorTypeRule(
            pattern=r"^x_embedder\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Flux input embedder is quality-sensitive; keep float",
        ),
        TensorTypeRule(
            pattern=r"^time_text_embed\.(?:timestep_embedder|text_embedder|guidance_embedder)\.linear_1\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Flux in-projections are quality-sensitive; keep float",
        ),
        TensorTypeRule(
            pattern=r"^proj_out\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Flux output projection is quality-sensitive; keep float",
        ),
        TensorTypeRule(
            pattern=r"^time_text_embed\.(?:timestep_embedder|text_embedder|guidance_embedder)\.linear_2\.weight$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Flux out-projections can be F16 without visible regressions",
        ),
        TensorTypeRule(
            pattern=r"^context_embedder\.weight$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Flux txt_in can be F16 while preserving prompt semantics",
        ),
        TensorTypeRule(
            pattern=r"^norm_out\.linear\.weight$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Flux final_layer modulation can be F16",
        ),
        TensorTypeRule(
            pattern=r"^(?:transformer_blocks|single_transformer_blocks)\..*\.bias$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Flux 1D tensors (biases/scales) stay F32 for stability",
        ),
        TensorTypeRule(
            pattern=r"^(?:x_embedder|context_embedder)\.bias$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Flux biases stay F32 for stability",
        ),
        TensorTypeRule(
            pattern=r"^time_text_embed\.(?:timestep_embedder|text_embedder|guidance_embedder)\.linear_[12]\.bias$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Flux biases stay F32 for stability",
        ),
        TensorTypeRule(
            pattern=r"^(?:proj_out|norm_out\.linear)\.bias$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Flux biases stay F32 for stability",
        ),
        TensorTypeRule(
            pattern=(
                r"^(?:transformer_blocks\.\d+\.attn\.(?:norm_q|norm_k|norm_added_q|norm_added_k)"
                r"|single_transformer_blocks\.\d+\.attn\.norm_[qk])\.weight$"
            ),
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Flux q/k norm weights stay F32 for stability",
        ),
        # Mixed presets are explicitly allowed to trade size for quality.
        TensorTypeRule(
            pattern=r"^time_text_embed\.(?:timestep_embedder|text_embedder|guidance_embedder)\.linear_2\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_FLUX_MIXED,
            reason="Flux mixed preset: keep out-projections float32 for higher quality",
        ),
        TensorTypeRule(
            pattern=r"^context_embedder\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_FLUX_MIXED,
            reason="Flux mixed preset: keep txt_in float32 for higher quality",
        ),
        TensorTypeRule(
            pattern=r"^norm_out\.linear\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_FLUX_MIXED,
            reason="Flux mixed preset: keep final modulation float32 for higher quality",
        ),
    ),
)

WAN22_QUANT_POLICY = QuantizationPolicySpec(
    id="wan22",
    # Required model policy: do not allow user overrides to violate these.
    required_rules=(
        # IO projections + patch embed are quality-sensitive.
        TensorTypeRule(
            pattern=r"^patch_embedding\.(?:weight|bias)$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="WAN22 patch embedding is quality-sensitive; keep float",
        ),
        TensorTypeRule(
            pattern=r"^proj_out\.(?:weight|bias)$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="WAN22 output projection is quality-sensitive; keep float",
        ),
        TensorTypeRule(
            pattern=r"^condition_embedder\.time_embedder\.linear_1\.(?:weight|bias)$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="WAN22 time embedding in-projection is quality-sensitive; keep float",
        ),
        TensorTypeRule(
            pattern=r"^condition_embedder\.time_proj\.(?:weight|bias)$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="WAN22 time projection to modulation is quality-sensitive; keep float",
        ),
        # Allow non-mixed quantized presets to keep the big embedder weights in float16
        # (mixed presets can trade size for more float32 below).
        TensorTypeRule(
            pattern=r"^condition_embedder\.time_embedder\.linear_2\.(?:weight|bias)$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="WAN22 time embedding out-projection can be float16 without visible regressions",
        ),
        TensorTypeRule(
            pattern=r"^condition_embedder\.text_embedder\.linear_(?:1|2)\.(?:weight|bias)$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="WAN22 text embedder weights can be float16 while preserving prompt semantics",
        ),
        # Stability: keep small tensors in float32.
        TensorTypeRule(
            pattern=r"^(?:scale_shift_table|blocks\.\d+\.scale_shift_table)$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="WAN22 modulation tables stay float32 for stability",
        ),
        TensorTypeRule(
            pattern=r"^blocks\.\d+\.norm2\.(?:weight|bias)$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="WAN22 LayerNorm affine tensors stay float32 for stability",
        ),
        TensorTypeRule(
            pattern=r"^blocks\.\d+\.attn[12]\.norm_[qk]\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="WAN22 q/k norm scales stay float32 for stability",
        ),
        TensorTypeRule(
            pattern=r"^blocks\.\d+\.attn[12]\.(?:to_[qkv]|to_out\.0)\.bias$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="WAN22 attention biases stay float32 for stability",
        ),
        TensorTypeRule(
            pattern=r"^blocks\.\d+\.ffn\.net\.(?:0\.proj|2)\.bias$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="WAN22 MLP biases stay float32 for stability",
        ),
        # Mixed presets explicitly trade size for quality.
        TensorTypeRule(
            pattern=r"^condition_embedder\.time_embedder\.linear_2\.(?:weight|bias)$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_WAN22_MIXED,
            reason="WAN22 mixed preset: keep time embedder out-projection float32 for higher quality",
        ),
        TensorTypeRule(
            pattern=r"^condition_embedder\.text_embedder\.linear_(?:1|2)\.(?:weight|bias)$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_WAN22_MIXED,
            reason="WAN22 mixed preset: keep text embedder weights float32 for higher quality",
        ),
    ),
)


LTX2_QUANT_POLICY = QuantizationPolicySpec(
    id="ltx2",
    required_rules=(
        TensorTypeRule(
            pattern=r"(?:^|\.)bias$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="LTX2 biases stay float32 for stability",
        ),
        TensorTypeRule(
            pattern=r"^(?:proj_in|audio_proj_in|proj_out|audio_proj_out)\.(?:weight|bias)$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="LTX2 IO projections are quality-sensitive; keep float",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)scale_shift_table(?:$|\.)",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="LTX2 modulation tables stay float32 for stability",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)time_embed\.linear\.weight$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="LTX2 adaLN linear weights stay float16 for stability",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)audio_time_embed\.linear\.weight$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="LTX2 audio adaLN linear weights stay float16 for stability",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)av_cross_attn_(?:video_scale_shift|audio_scale_shift|video_a2v_gate|audio_v2a_gate)\.linear\.weight$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="LTX2 AV cross-attn adaLN weights stay float16 for stability",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)norm_q\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="LTX2 q_norm stays float32 for stability",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)norm_k\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="LTX2 k_norm stays float32 for stability",
        ),
    ),
)


ZIMAGE_QUANT_POLICY = QuantizationPolicySpec(
    id="zimage",
    required_rules=(
        TensorTypeRule(
            pattern=r"^(?:x_pad_token|cap_pad_token)$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="ZImage pad tokens must remain float (load_state_dict cannot load quantized tensors)",
        ),
    ),
)


LLAMA_QUANT_POLICY = QuantizationPolicySpec(
    id="llama",
    default_rules=(
        TensorTypeRule(
            pattern=r"(?:^|\.)token_embd\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            reason="LLM embeddings: keep higher precision to preserve semantics",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)output\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            reason="LLM output head: keep higher precision to preserve semantics",
        ),
        TensorTypeRule(
            pattern=r"model\.embed_tokens\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            reason="LLM embeddings: keep higher precision to preserve semantics",
        ),
        TensorTypeRule(
            pattern=r"lm_head\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            reason="LLM output head: keep higher precision to preserve semantics",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)attn_(?:q|k|v|output)\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            reason="LLM attention projections: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            reason="LLM attention projections: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)token_embd\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            reason="LLM embeddings: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)output\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            reason="LLM output head: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"model\.embed_tokens\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            reason="LLM embeddings: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"lm_head\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            reason="LLM output head: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)attn_(?:q|k|v|output)\.weight$",
            ggml_type=GGMLQuantizationType.Q5_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            reason="LLM attention projections: bump to 5-bit K",
        ),
        TensorTypeRule(
            pattern=r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$",
            ggml_type=GGMLQuantizationType.Q5_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            reason="LLM attention projections: bump to 5-bit K",
        ),
    ),
)


GENERIC_QUANT_POLICY = QuantizationPolicySpec(id="generic")


_LLAMA_KEY_MAPPING = KeyMappingSpec(id="llama_hf_to_gguf", build=_build_llama_mapping)


PROFILE_REGISTRY: tuple[ConverterProfileSpec, ...] = (
    ConverterProfileSpec(
        id=ConverterProfileId.FLUX_TRANSFORMER,
        arch=GGUFArch.FLUX,
        detect=_is_flux,
        quant_policy=FLUX_QUANT_POLICY,
        metadata_normalizer=_tensor_planner.normalize_flux_transformer_metadata_config,
    ),
    ConverterProfileSpec(
        id=ConverterProfileId.ZIMAGE_TRANSFORMER,
        arch=GGUFArch.ZIMAGE,
        detect=_is_zimage,
        quant_policy=ZIMAGE_QUANT_POLICY,
        metadata_normalizer=_tensor_planner.normalize_zimage_transformer_metadata_config,
    ),
    ConverterProfileSpec(
        id=ConverterProfileId.WAN22_TRANSFORMER,
        arch=GGUFArch.WAN22,
        detect=_is_wan22,
        quant_policy=WAN22_QUANT_POLICY,
        metadata_normalizer=_tensor_planner.normalize_wan22_transformer_metadata_config,
    ),
    ConverterProfileSpec(
        id=ConverterProfileId.LTX2_TRANSFORMER,
        arch=GGUFArch.LTX2,
        detect=_is_ltx2,
        quant_policy=LTX2_QUANT_POLICY,
        metadata_normalizer=_tensor_planner.normalize_ltx2_transformer_metadata_config,
    ),
    ConverterProfileSpec(
        id=ConverterProfileId.GEMMA3_TENC,
        arch=GGUFArch.GEMMA3,
        detect=_is_gemma3_tenc,
        quant_policy=LLAMA_QUANT_POLICY,
        metadata_normalizer=_tensor_planner.normalize_gemma3_text_encoder_metadata_config,
    ),
    ConverterProfileSpec(
        id=ConverterProfileId.LLAMA_HF_TO_GGUF,
        arch=GGUFArch.LLAMA,
        detect=lambda cfg: not _is_flux(cfg)
        and not _is_zimage(cfg)
        and not _is_wan22(cfg)
        and not _is_ltx2(cfg)
        and not _is_gemma3_tenc(cfg),
        quant_policy=LLAMA_QUANT_POLICY,
        key_mapping=_LLAMA_KEY_MAPPING,
    ),
)


_PROFILES_BY_ID: dict[ConverterProfileId, ConverterProfileSpec] = {profile.id: profile for profile in PROFILE_REGISTRY}


def resolve_profile(config_json: Mapping[str, Any]) -> ConverterProfileSpec:
    for profile in PROFILE_REGISTRY:
        if profile.detect(config_json):
            return profile
    raise ValueError("Unsupported GGUF converter config: no registered profile matched the provided config.json")


def profile_by_id(profile_id: str) -> ConverterProfileSpec:
    try:
        pid = ConverterProfileId(str(profile_id))
    except ValueError as exc:
        raise ValueError(f"Unknown GGUF converter profile_id: {profile_id!r}") from exc

    profile = _PROFILES_BY_ID.get(pid)
    if profile is None:
        raise ValueError(f"GGUF converter profile_id not registered: {profile_id!r}")
    return profile


__all__ = ["PROFILE_REGISTRY", "profile_by_id", "resolve_profile"]
