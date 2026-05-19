"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed “profile + policy” specs for the GGUF converter.
Defines converter architecture buckets, profile ids, metadata normalizers, key mappings, and per-model tensor dtype rules for the source/native tooling surface.

Symbols (top-level; keep in sync; no ghosts):
- `GGUFArch` (enum): High-level GGUF architecture buckets used by conversion profiles.
- `TensorNameTarget` (enum): Whether a tensor-type rule matches source names, destination names, or both.
- `ConverterProfileId` (enum): Stable identifiers for converter profiles (one truthful id per supported component family).
- `QuantizationCondition` (dataclass): Declarative condition for when a rule applies (include/exclude quantization selectors).
- `TensorTypeRule` (dataclass): Declarative per-tensor dtype rule (regex + target + quant/policy conditions + dtype action + reason).
- `PolicyRuleFactory` (type alias): Builds config-aware default rules for a policy preset.
- `CompiledTensorTypeRule` (dataclass): Compiled rule used during planning (compiled regex + target + dtype action + reason).
- `QuantizationPolicySpec` (dataclass): Bundle of built-in dtype rules; compiles them with optional user overrides.
- `KeyMappingSpec` (dataclass): Typed wrapper around “key mapping builders” (e.g. Llama HF→GGUF mapping).
- `ConverterProfileSpec` (dataclass): Full conversion profile (detection + key mapping + metadata normalization + policies).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, Mapping, Sequence

from apps.backend.quantization.gguf import GGMLQuantizationType
from apps.backend.runtime.tools.gguf_converter_quantization import requested_ggml_type
from apps.backend.runtime.tools.gguf_converter_types import QuantPolicyPreset, QuantizationType

_TensorNameTargetLiteral = Literal["src", "dst", "both"]


class GGUFArch(str, Enum):
    LLAMA = "llama"
    GEMMA3 = "gemma3"
    FLUX = "flux"
    QWEN_IMAGE = "qwen_image"
    ZIMAGE = "zimage"
    WAN22 = "wan22"
    LTX2 = "ltx2"


class TensorNameTarget(str, Enum):
    SRC = "src"
    DST = "dst"
    BOTH = "both"

    @classmethod
    def from_literal(cls, value: _TensorNameTargetLiteral) -> TensorNameTarget:
        return cls(value)

    def matches_src(self) -> bool:
        return self in {TensorNameTarget.SRC, TensorNameTarget.BOTH}

    def matches_dst(self) -> bool:
        return self in {TensorNameTarget.DST, TensorNameTarget.BOTH}


class ConverterProfileId(str, Enum):
    FLUX_TRANSFORMER = "flux_transformer"
    QWEN_IMAGE_TRANSFORMER = "qwen_image_transformer"
    ZIMAGE_TRANSFORMER = "zimage_transformer"
    WAN22_TRANSFORMER = "wan22_transformer"
    LTX2_TRANSFORMER = "ltx2_transformer"
    GEMMA3_TENC = "gemma3_tenc"
    LLAMA_HF_TO_GGUF = "llama_hf_to_gguf"


@dataclass(frozen=True, slots=True)
class QuantizationCondition:
    include: frozenset[QuantizationType] | None = None
    exclude: frozenset[QuantizationType] = frozenset()

    def matches(self, quant: QuantizationType) -> bool:
        if quant in self.exclude:
            return False
        if self.include is None:
            return True
        return quant in self.include


@dataclass(frozen=True, slots=True)
class TensorTypeRule:
    pattern: str
    ggml_type: GGMLQuantizationType | None = None
    preserve_source_dtype: bool = False
    apply_to: TensorNameTarget = TensorNameTarget.BOTH
    when: QuantizationCondition = QuantizationCondition()
    policy_presets: frozenset[QuantPolicyPreset] | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.preserve_source_dtype and self.ggml_type is not None:
            raise ValueError("TensorTypeRule cannot set both ggml_type and preserve_source_dtype")
        if not self.preserve_source_dtype and self.ggml_type is None:
            raise ValueError("TensorTypeRule requires ggml_type unless preserve_source_dtype is true")


@dataclass(frozen=True, slots=True)
class CompiledTensorTypeRule:
    pattern: re.Pattern[str]
    ggml_type: GGMLQuantizationType | None
    apply_to: TensorNameTarget
    preserve_source_dtype: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.preserve_source_dtype and self.ggml_type is not None:
            raise ValueError("CompiledTensorTypeRule cannot set both ggml_type and preserve_source_dtype")
        if not self.preserve_source_dtype and self.ggml_type is None:
            raise ValueError("CompiledTensorTypeRule requires ggml_type unless preserve_source_dtype is true")


PolicyRuleFactory = Callable[[Mapping[str, Any], QuantizationType, QuantPolicyPreset], Sequence[TensorTypeRule]]


@dataclass(frozen=True, slots=True)
class QuantizationPolicySpec:
    id: str
    default_rules: tuple[TensorTypeRule, ...] = ()
    default_rule_factories: tuple[PolicyRuleFactory, ...] = ()
    required_rules: tuple[TensorTypeRule, ...] = ()

    def compile(
        self,
        *,
        quant: QuantizationType,
        policy_preset: QuantPolicyPreset,
        model_config: Mapping[str, Any],
        user_rules: Sequence[str],
    ) -> list[CompiledTensorTypeRule]:
        compiled: list[CompiledTensorTypeRule] = []

        for rule in self._iter_default_rules(model_config=model_config, quant=quant, policy_preset=policy_preset):
            if self._rule_matches_context(rule, quant=quant, policy_preset=policy_preset):
                compiled.append(self._compile_rule(rule))

        for entry in user_rules:
            raw = str(entry or "").strip()
            if not raw:
                continue
            if "=" not in raw:
                raise ValueError(f"Invalid tensor override (expected '<regex>=<quant>'): {raw!r}")
            pattern, qname = raw.split("=", 1)
            pattern = pattern.strip()
            qname = qname.strip()
            if not pattern or not qname:
                raise ValueError(f"Invalid tensor override (expected '<regex>=<quant>'): {raw!r}")

            try:
                q_enum = QuantizationType(qname.upper())
            except ValueError as exc:
                raise ValueError(f"Invalid quant type in override {raw!r}: {qname!r}") from exc

            compiled.append(
                CompiledTensorTypeRule(
                    pattern=re.compile(pattern),
                    ggml_type=requested_ggml_type(q_enum),
                    preserve_source_dtype=False,
                    apply_to=TensorNameTarget.BOTH,
                    reason="user override",
                )
            )

        for rule in self.required_rules:
            if self._rule_matches_context(rule, quant=quant, policy_preset=policy_preset):
                compiled.append(self._compile_rule(rule))

        return compiled

    def _iter_default_rules(
        self,
        *,
        model_config: Mapping[str, Any],
        quant: QuantizationType,
        policy_preset: QuantPolicyPreset,
    ) -> tuple[TensorTypeRule, ...]:
        generated: list[TensorTypeRule] = []
        for factory in self.default_rule_factories:
            generated.extend(factory(model_config, quant, policy_preset))
        return (*self.default_rules, *generated)

    def _rule_matches_context(
        self,
        rule: TensorTypeRule,
        *,
        quant: QuantizationType,
        policy_preset: QuantPolicyPreset,
    ) -> bool:
        if not rule.when.matches(quant):
            return False
        if rule.policy_presets is not None and policy_preset not in rule.policy_presets:
            return False
        return True

    def _compile_rule(self, rule: TensorTypeRule) -> CompiledTensorTypeRule:
        return CompiledTensorTypeRule(
            pattern=re.compile(rule.pattern),
            ggml_type=rule.ggml_type,
            preserve_source_dtype=rule.preserve_source_dtype,
            apply_to=rule.apply_to,
            reason=rule.reason,
        )


@dataclass(frozen=True, slots=True)
class KeyMappingSpec:
    id: str
    build: Callable[[Mapping[str, Any]], dict[str, str]]


@dataclass(frozen=True, slots=True)
class ConverterProfileSpec:
    id: ConverterProfileId
    arch: GGUFArch
    detect: Callable[[Mapping[str, Any]], bool]
    quant_policy: QuantizationPolicySpec
    key_mapping: KeyMappingSpec | None = None
    metadata_normalizer: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None


__all__ = [
    "CompiledTensorTypeRule",
    "ConverterProfileId",
    "ConverterProfileSpec",
    "GGUFArch",
    "KeyMappingSpec",
    "PolicyRuleFactory",
    "QuantizationCondition",
    "QuantizationPolicySpec",
    "TensorNameTarget",
    "TensorTypeRule",
]
