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
- `_is_qwen_image` (function): Detect whether a config.json describes a Qwen Image transformer.
- `_is_qwen_image_tenc` (function): Detect whether a config.json describes a Qwen Image Qwen2.5-VL text encoder.
- `_is_zimage` (function): Detect whether a config.json describes a ZImage transformer.
- `_is_wan22` (function): Detect whether a config.json describes a WAN22 transformer.
- `_is_ltx2` (function): Detect whether a config.json describes an LTX2 transformer.
- `_is_gemma3_tenc` (function): Detect whether a config.json describes a Gemma3 text encoder export.
- `_is_llama_hf_to_gguf` (function): Detect explicit Llama-family HF configs accepted by the Llama key mapping profile.
- `_build_llama_mapping` (function): Build a Llama HF→GGUF key mapping from the model config.
- `_COND_QUANTIZED` (constant): Condition helper matching any quantized preset (non-F16/F32).
- `_COND_QWEN_IMAGE_MIXED` (constant): Condition helper matching Qwen Image mixed quant selectors (`Q5_K_M`/`Q4_K_M`).
- `_POLICY_HQ` (constant): Policy preset set for conservative/high-quality optional rules.
- `_POLICY_MQ_LQ` (constant): Policy preset set for balanced/compact baseline optional rules.
- `_POLICY_HQ_MQ` (constant): Policy preset set for Llama-family optional mixed precision bumps.
- `_qwen_image_num_layers` (function): Reads and validates Qwen Image transformer block count for policy edge rules.
- `_qwen_image_mq_quality_rules` (function): Builds Qwen Image MQ core-quality rules from `num_layers`.
- `FLUX_QUANT_POLICY` (constant): Flux per-tensor dtype policy (HQ preserves optional IO weights in source float dtype).
- `QWEN_IMAGE_QUANT_POLICY` (constant): Qwen Image per-tensor dtype policy (stability tensors preserve source float dtype).
- `QWEN_IMAGE_TENC_QUANT_POLICY` (constant): Qwen Image Qwen2.5-VL text-encoder dtype policy (native names, no Llama aliases).
- `WAN22_QUANT_POLICY` (constant): WAN22 per-tensor dtype policy (HQ preserves optional embedder weights in source float dtype).
- `LTX2_QUANT_POLICY` (constant): LTX2 per-tensor dtype policy (stability-sensitive tensors stay float).
- `ZIMAGE_QUANT_POLICY` (constant): ZImage per-tensor dtype policy (pad tokens must remain float).
- `LLAMA_QUANT_POLICY` (constant): Llama per-tensor dtype policy (HQ/MQ mixed quant selectors bump key weights to higher precision).
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
from apps.backend.runtime.tools.gguf_converter_types import QuantPolicyPreset, QuantizationType


def _is_flux(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_flux_transformer_config(config)


def _is_qwen_image(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_qwen_image_transformer_config(config)


def _is_qwen_image_tenc(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_qwen_image_text_encoder_config(config)


def _is_zimage(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_zimage_transformer_config(config)


def _is_wan22(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_wan22_transformer_config(config)


def _is_ltx2(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_ltx2_transformer_config(config)


def _is_gemma3_tenc(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_gemma3_text_encoder_config(config)


_LLAMA_FAMILY_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "llama",
        "mistral",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
    }
)
_LLAMA_FAMILY_ARCHITECTURES: frozenset[str] = frozenset(
    {
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
    }
)


def _is_llama_hf_to_gguf(config: Mapping[str, Any]) -> bool:
    if str(config.get("_class_name") or "").strip():
        return False

    model_type = str(config.get("model_type") or "").strip().lower()
    if model_type in _LLAMA_FAMILY_MODEL_TYPES:
        return True

    raw_architectures = config.get("architectures")
    if isinstance(raw_architectures, str):
        architectures = (raw_architectures,)
    elif isinstance(raw_architectures, list):
        architectures = tuple(str(value) for value in raw_architectures)
    else:
        architectures = ()
    return any(architecture in _LLAMA_FAMILY_ARCHITECTURES for architecture in architectures)


def _build_llama_mapping(config: Mapping[str, Any]) -> dict[str, str]:
    num_layers = int(config.get("num_hidden_layers", 32))
    return _key_mapping.build_key_mapping(num_layers)


_COND_QUANTIZED = QuantizationCondition(exclude=frozenset({QuantizationType.F16, QuantizationType.F32}))
_COND_QWEN_IMAGE_MIXED = QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M, QuantizationType.Q4_K_M}))
_POLICY_HQ = frozenset({QuantPolicyPreset.HQ})
_POLICY_MQ_LQ = frozenset({QuantPolicyPreset.MQ, QuantPolicyPreset.LQ})
_POLICY_HQ_MQ = frozenset({QuantPolicyPreset.HQ, QuantPolicyPreset.MQ})


def _qwen_image_num_layers(config: Mapping[str, Any]) -> int:
    raw = config.get("num_layers")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RuntimeError("Qwen Image MQ policy requires integer transformer config num_layers >= 4")
    num_layers = raw
    if num_layers < 4:
        raise RuntimeError("Qwen Image MQ policy requires integer transformer config num_layers >= 4")
    return num_layers


def _qwen_image_mq_quality_rules(
    config: Mapping[str, Any],
    quant: QuantizationType,
    policy_preset: QuantPolicyPreset,
) -> tuple[TensorTypeRule, ...]:
    if policy_preset is not QuantPolicyPreset.MQ:
        return ()
    if quant in {QuantizationType.F16, QuantizationType.F32}:
        return ()
    num_layers = _qwen_image_num_layers(config)
    if quant not in {QuantizationType.Q4_K_M, QuantizationType.Q5_K_M}:
        return ()

    last = num_layers - 1
    first_last = f"(?:0|{last})"
    adjacent = f"(?:1|{last - 1})"

    if quant is QuantizationType.Q4_K_M:
        adjacent_type = GGMLQuantizationType.Q5_K
    else:
        adjacent_type = GGMLQuantizationType.Q6_K

    return (
        TensorTypeRule(
            pattern=r"^transformer_blocks\.\d+\.attn\.(?:to_v|add_v_proj)\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QWEN_IMAGE_MIXED,
            reason="Qwen Image MQ policy: keep value projections at Q6_K",
        ),
        TensorTypeRule(
            pattern=r"^transformer_blocks\.\d+\.(?:img_mlp|txt_mlp)\.net\.2\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QWEN_IMAGE_MIXED,
            reason="Qwen Image MQ policy: keep MLP down/out projections at Q6_K",
        ),
        TensorTypeRule(
            pattern=(
                rf"^transformer_blocks\.{first_last}\."
                r"(?:(?:attn\.(?:add_[qk]_proj|to_add_out|to_[qk]|to_out\.0))"
                r"|(?:(?:img_mlp|txt_mlp)\.net\.0\.proj))\.weight$"
            ),
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QWEN_IMAGE_MIXED,
            reason="Qwen Image MQ policy: keep first/last core block weights at Q6_K",
        ),
        TensorTypeRule(
            pattern=(
                rf"^transformer_blocks\.{adjacent}\."
                r"(?:(?:attn\.(?:add_[qk]_proj|to_add_out|to_[qk]|to_out\.0))"
                r"|(?:(?:img_mlp|txt_mlp)\.net\.0\.proj))\.weight$"
            ),
            ggml_type=adjacent_type,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QWEN_IMAGE_MIXED,
            reason=f"Qwen Image MQ policy: keep adjacent core block weights at {adjacent_type.name}",
        ),
        TensorTypeRule(
            pattern=rf"^transformer_blocks\.{first_last}\.img_mod\.1\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QWEN_IMAGE_MIXED,
            reason="Qwen Image MQ policy: keep first/last image modulation weights at Q8_0",
        ),
        TensorTypeRule(
            pattern=rf"^transformer_blocks\.{first_last}\.txt_mod\.1\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QWEN_IMAGE_MIXED,
            reason="Qwen Image MQ policy: keep first/last text modulation weights at Q6_K",
        ),
        TensorTypeRule(
            pattern=rf"^transformer_blocks\.{adjacent}\.(?:img_mod|txt_mod)\.1\.weight$",
            ggml_type=adjacent_type,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QWEN_IMAGE_MIXED,
            reason=f"Qwen Image MQ policy: keep adjacent edge modulation weights at {adjacent_type.name}",
        ),
    )


FLUX_QUANT_POLICY = QuantizationPolicySpec(
    id="flux",
    default_rules=(
        TensorTypeRule(
            pattern=r"^time_text_embed\.(?:timestep_embedder|text_embedder|guidance_embedder)\.linear_2\.weight$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_MQ_LQ,
            reason="Flux MQ/LQ policy: out-projections use F16 baseline",
        ),
        TensorTypeRule(
            pattern=r"^context_embedder\.weight$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_MQ_LQ,
            reason="Flux MQ/LQ policy: txt_in uses F16 baseline",
        ),
        TensorTypeRule(
            pattern=r"^norm_out\.linear\.weight$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_MQ_LQ,
            reason="Flux MQ/LQ policy: final modulation uses F16 baseline",
        ),
        TensorTypeRule(
            pattern=r"^time_text_embed\.(?:timestep_embedder|text_embedder|guidance_embedder)\.linear_2\.weight$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_HQ,
            reason="Flux HQ policy: preserve out-projection source float dtype",
        ),
        TensorTypeRule(
            pattern=r"^context_embedder\.weight$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_HQ,
            reason="Flux HQ policy: preserve txt_in source float dtype",
        ),
        TensorTypeRule(
            pattern=r"^norm_out\.linear\.weight$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_HQ,
            reason="Flux HQ policy: preserve final modulation source float dtype",
        ),
    ),
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
    ),
)


QWEN_IMAGE_QUANT_POLICY = QuantizationPolicySpec(
    id="qwen_image",
    version=2,
    default_rules=(
        TensorTypeRule(
            pattern=r"^transformer_blocks\.\d+\.(?:img_mod|txt_mod)\.1\.weight$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_HQ,
            reason="Qwen Image HQ policy: preserve modulation weight source float dtype",
        ),
        TensorTypeRule(
            pattern=r"^(?:img_in|txt_in)\.weight$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_HQ_MQ,
            reason="Qwen Image HQ/MQ policy: preserve input projection source float dtype",
        ),
        TensorTypeRule(
            pattern=r"^time_text_embed\.timestep_embedder\.linear_[12]\.weight$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_HQ_MQ,
            reason="Qwen Image HQ/MQ policy: preserve timestep embedder source float dtype",
        ),
    ),
    default_rule_factories=(_qwen_image_mq_quality_rules,),
    required_rules=(
        TensorTypeRule(
            pattern=r".*\.bias$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Qwen Image biases preserve source float dtype for stability",
        ),
        TensorTypeRule(
            pattern=r"^transformer_blocks\.\d+\.attn\.(?:norm_[qk]|norm_added_[qk])\.weight$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Qwen Image q/k norm scales preserve source float dtype for stability",
        ),
        TensorTypeRule(
            pattern=r"^(?:proj_out|norm_out\.linear)\.(?:weight|bias)$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Qwen Image final head tensors preserve source float dtype for stability",
        ),
    ),
)


QWEN_IMAGE_TENC_QUANT_POLICY = QuantizationPolicySpec(
    id="qwen_image_tenc",
    default_rules=(
        TensorTypeRule(
            pattern=r"^visual\.merger\.mlp\.\d+\.weight$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_HQ,
            reason="Qwen Image text-encoder HQ policy: preserve visual merger projection source float dtype",
        ),
        TensorTypeRule(
            pattern=r"^(?:model\.embed_tokens|lm_head)\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="Qwen Image text-encoder Q5_K_M policy: keep language embeddings/output at Q8_0",
        ),
        TensorTypeRule(
            pattern=r"^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="Qwen Image text-encoder Q5_K_M policy: keep language attention projections at Q6_K",
        ),
        TensorTypeRule(
            pattern=r"^visual\.blocks\.\d+\.attn\.(?:qkv|proj)\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="Qwen Image text-encoder Q5_K_M policy: keep visual attention projections at Q6_K",
        ),
        TensorTypeRule(
            pattern=r"^(?:model\.embed_tokens|lm_head)\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="Qwen Image text-encoder Q4_K_M policy: keep language embeddings/output at Q6_K",
        ),
        TensorTypeRule(
            pattern=r"^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$",
            ggml_type=GGMLQuantizationType.Q5_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="Qwen Image text-encoder Q4_K_M policy: keep language attention projections at Q5_K",
        ),
        TensorTypeRule(
            pattern=r"^visual\.blocks\.\d+\.attn\.(?:qkv|proj)\.weight$",
            ggml_type=GGMLQuantizationType.Q5_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="Qwen Image text-encoder Q4_K_M policy: keep visual attention projections at Q5_K",
        ),
    ),
    required_rules=(
        TensorTypeRule(
            pattern=r".*\.bias$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Qwen Image text-encoder biases preserve source float dtype for stability",
        ),
        TensorTypeRule(
            pattern=(
                r"^(?:model\.norm|model\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)"
                r"|visual\.blocks\.\d+\.norm[12]|visual\.merger\.ln_q)\.weight$"
            ),
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Qwen Image text-encoder norm scales preserve source float dtype for stability",
        ),
        TensorTypeRule(
            pattern=r"^visual\.patch_embed\.proj\.weight$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="Qwen Image text-encoder patch embedding preserves source float dtype for shape/stability",
        ),
    ),
)


WAN22_QUANT_POLICY = QuantizationPolicySpec(
    id="wan22",
    default_rules=(
        TensorTypeRule(
            pattern=r"^condition_embedder\.time_embedder\.linear_2\.(?:weight|bias)$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_MQ_LQ,
            reason="WAN22 MQ/LQ policy: time embedding out-projection uses F16 baseline",
        ),
        TensorTypeRule(
            pattern=r"^condition_embedder\.text_embedder\.linear_(?:1|2)\.(?:weight|bias)$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_MQ_LQ,
            reason="WAN22 MQ/LQ policy: text embedder weights use F16 baseline",
        ),
        TensorTypeRule(
            pattern=r"^condition_embedder\.time_embedder\.linear_2\.(?:weight|bias)$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_HQ,
            reason="WAN22 HQ policy: preserve time embedder out-projection source float dtype",
        ),
        TensorTypeRule(
            pattern=r"^condition_embedder\.text_embedder\.linear_(?:1|2)\.(?:weight|bias)$",
            preserve_source_dtype=True,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            policy_presets=_POLICY_HQ,
            reason="WAN22 HQ policy: preserve text embedder source float dtype",
        ),
    ),
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
    # Llama mixed rules are quantized precision bumps, not preserved source-float groups.
    default_rules=(
        TensorTypeRule(
            pattern=r"(?:^|\.)token_embd\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="LLM embeddings: keep higher precision to preserve semantics",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)output\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="LLM output head: keep higher precision to preserve semantics",
        ),
        TensorTypeRule(
            pattern=r"model\.embed_tokens\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="LLM embeddings: keep higher precision to preserve semantics",
        ),
        TensorTypeRule(
            pattern=r"lm_head\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="LLM output head: keep higher precision to preserve semantics",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)attn_(?:q|k|v|output)\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="LLM attention projections: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="LLM attention projections: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)token_embd\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="LLM embeddings: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)output\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="LLM output head: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"model\.embed_tokens\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="LLM embeddings: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"lm_head\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="LLM output head: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)attn_(?:q|k|v|output)\.weight$",
            ggml_type=GGMLQuantizationType.Q5_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            policy_presets=_POLICY_HQ_MQ,
            reason="LLM attention projections: bump to 5-bit K",
        ),
        TensorTypeRule(
            pattern=r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$",
            ggml_type=GGMLQuantizationType.Q5_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            policy_presets=_POLICY_HQ_MQ,
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
        id=ConverterProfileId.QWEN_IMAGE_TRANSFORMER,
        arch=GGUFArch.QWEN_IMAGE,
        detect=_is_qwen_image,
        quant_policy=QWEN_IMAGE_QUANT_POLICY,
        metadata_normalizer=_tensor_planner.normalize_qwen_image_transformer_metadata_config,
    ),
    ConverterProfileSpec(
        id=ConverterProfileId.QWEN_IMAGE_TENC,
        arch=GGUFArch.QWEN2_5_VL,
        detect=_is_qwen_image_tenc,
        quant_policy=QWEN_IMAGE_TENC_QUANT_POLICY,
        metadata_normalizer=_tensor_planner.normalize_qwen_image_text_encoder_metadata_config,
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
        detect=_is_llama_hf_to_gguf,
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
