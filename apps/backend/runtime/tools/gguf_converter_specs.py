"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed “profile + policy” specs for the GGUF converter.
Defines converter architecture buckets, profile ids, metadata normalizers, key mappings, public recipe support, recipe-intrinsic rules, and per-model tensor policy rules for the source/native tooling surface.
Includes exact Z-Image L2P denoiser and Qwen3-4B text-encoder profile ids without aliasing them to latent Z-Image profiles.

Symbols (top-level; keep in sync; no ghosts):
- `GGUFArch` (enum): High-level GGUF architecture buckets used by conversion profiles.
- `TensorNameTarget` (enum): Whether a tensor-type rule matches source names, destination names, or both.
- `ConverterProfileId` (enum): Stable identifiers for converter profiles (one truthful id per supported component family).
- `RuleLane` (enum): Provenance lane for compiled tensor rules (`RECIPE_INTRINSIC|PROFILE_POLICY|USER_OVERRIDE|REQUIRED_INVARIANT`).
- `QuantizationCondition` (dataclass): Declarative condition for when a rule applies (include/exclude public recipes).
- `TensorTypeRule` (dataclass): Declarative per-tensor dtype rule (regex + target + recipe/policy conditions + dtype action + reason).
- `RecipeRuleFactory` (type alias): Builds config-aware recipe-intrinsic rules.
- `PolicyRuleFactory` (type alias): Builds config-aware profile-policy rules for a policy preset.
- `CompiledTensorTypeRule` (dataclass): Compiled rule used during planning (compiled regex + target + dtype action + provenance lane + reason).
- `QuantizationSurfaceDescriptor` (dataclass): Backend-owned recipe/policy support metadata for API/UI descriptors.
- `QuantizationPolicySpec` (dataclass): Bundle of supported recipes, recipe rules, dtype rules, and metadata version.
- `KeyMappingSpec` (dataclass): Typed wrapper around “key mapping builders” (e.g. Llama HF→GGUF mapping).
- `ConverterProfileSpec` (dataclass): Full conversion profile (detection + key mapping + metadata normalization + policies).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, Mapping, Sequence

from apps.backend.quantization.gguf import GGMLQuantizationType
from apps.backend.runtime.tools.gguf_converter_quantization import (
    generated_rule_is_downgrade,
    recipe_spec,
    tensor_quantization_type_to_ggml_type,
)
from apps.backend.runtime.tools.gguf_converter_types import (
    QuantPolicyPreset,
    QuantizationRecipe,
    TensorQuantizationType,
    normalize_tensor_quantization_type,
)

_TensorNameTargetLiteral = Literal["src", "dst", "both"]


class GGUFArch(str, Enum):
    LLAMA = "llama"
    GEMMA3 = "gemma3"
    FLUX = "flux"
    QWEN_IMAGE = "qwen_image"
    QWEN2_5_VL = "qwen2_5_vl"
    ZIMAGE = "zimage"
    ZIMAGE_L2P = "zimage_l2p"
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
    QWEN_IMAGE_TENC = "qwen_image_tenc"
    ZIMAGE_TRANSFORMER = "zimage_transformer"
    ZIMAGE_L2P_DENOISER = "zimage_l2p_denoiser"
    ZIMAGE_L2P_TENC = "zimage_l2p_tenc"
    WAN22_TRANSFORMER = "wan22_transformer"
    LTX2_TRANSFORMER = "ltx2_transformer"
    GEMMA3_TENC = "gemma3_tenc"
    LLAMA_HF_TO_GGUF = "llama_hf_to_gguf"


class RuleLane(str, Enum):
    RECIPE_INTRINSIC = "RECIPE_INTRINSIC"
    PROFILE_POLICY = "PROFILE_POLICY"
    USER_OVERRIDE = "USER_OVERRIDE"
    REQUIRED_INVARIANT = "REQUIRED_INVARIANT"


@dataclass(frozen=True, slots=True)
class QuantizationCondition:
    include: frozenset[QuantizationRecipe] | None = None
    exclude: frozenset[QuantizationRecipe] = frozenset()

    def matches(self, recipe: QuantizationRecipe) -> bool:
        if recipe in self.exclude:
            return False
        if self.include is None:
            return True
        return recipe in self.include


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
    lane: RuleLane
    preserve_source_dtype: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.preserve_source_dtype and self.ggml_type is not None:
            raise ValueError("CompiledTensorTypeRule cannot set both ggml_type and preserve_source_dtype")
        if not self.preserve_source_dtype and self.ggml_type is None:
            raise ValueError("CompiledTensorTypeRule requires ggml_type unless preserve_source_dtype is true")


RecipeRuleFactory = Callable[[Mapping[str, Any], QuantizationRecipe], Sequence[TensorTypeRule]]
PolicyRuleFactory = Callable[[Mapping[str, Any], QuantizationRecipe, QuantPolicyPreset], Sequence[TensorTypeRule]]


@dataclass(frozen=True, slots=True)
class QuantizationSurfaceDescriptor:
    profile_id: ConverterProfileId
    supported_recipes: tuple[QuantizationRecipe, ...]
    default_recipe: QuantizationRecipe
    policy_presets_by_recipe: dict[QuantizationRecipe, tuple[QuantPolicyPreset, ...]]
    default_policy_preset_by_recipe: dict[QuantizationRecipe, QuantPolicyPreset | None]

    def to_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id.value,
            "supported_recipes": [recipe.value for recipe in self.supported_recipes],
            "default_recipe": self.default_recipe.value,
            "policy_presets_by_recipe": {
                recipe.value: [preset.value for preset in presets]
                for recipe, presets in self.policy_presets_by_recipe.items()
            },
            "default_policy_preset_by_recipe": {
                recipe.value: (preset.value if preset is not None else None)
                for recipe, preset in self.default_policy_preset_by_recipe.items()
            },
        }


@dataclass(frozen=True, slots=True)
class QuantizationPolicySpec:
    id: str
    version: int = 1
    supported_recipes: frozenset[QuantizationRecipe] = frozenset({QuantizationRecipe.F16, QuantizationRecipe.F32})
    default_recipe: QuantizationRecipe = QuantizationRecipe.F16
    recipe_rule_factories: tuple[RecipeRuleFactory, ...] = ()
    default_rules: tuple[TensorTypeRule, ...] = ()
    default_rule_factories: tuple[PolicyRuleFactory, ...] = ()
    required_rules: tuple[TensorTypeRule, ...] = ()

    def compile(
        self,
        *,
        recipe: QuantizationRecipe,
        policy_preset: QuantPolicyPreset | None,
        model_config: Mapping[str, Any],
        user_rules: Sequence[str],
    ) -> list[CompiledTensorTypeRule]:
        self.require_recipe_supported(recipe)
        compiled: list[CompiledTensorTypeRule] = []
        generated: list[CompiledTensorTypeRule] = []

        for rule in self._iter_recipe_rules(model_config=model_config, recipe=recipe):
            if self._rule_matches_context(rule, recipe=recipe, policy_preset=policy_preset):
                generated.append(self._compile_rule(rule, lane=RuleLane.RECIPE_INTRINSIC))

        if policy_preset is not None:
            for rule in self._iter_default_rules(
                model_config=model_config,
                recipe=recipe,
                policy_preset=policy_preset,
            ):
                if self._rule_matches_context(rule, recipe=recipe, policy_preset=policy_preset):
                    generated.append(self._compile_rule(rule, lane=RuleLane.PROFILE_POLICY))

        self._validate_generated_rules(recipe=recipe, rules=generated)
        compiled.extend(generated)

        for entry in user_rules:
            raw = str(entry or "").strip()
            if not raw:
                continue
            if "=" not in raw:
                raise ValueError(f"Invalid tensor override (expected '<regex>=<tensor_type>'): {raw!r}")
            pattern, qname = raw.split("=", 1)
            pattern = pattern.strip()
            qname = qname.strip()
            if not pattern or not qname:
                raise ValueError(f"Invalid tensor override (expected '<regex>=<tensor_type>'): {raw!r}")

            try:
                target = normalize_tensor_quantization_type(qname)
            except ValueError as exc:
                raise ValueError(f"Invalid tensor type in override {raw!r}: {qname!r}") from exc

            compiled.append(
                CompiledTensorTypeRule(
                    pattern=re.compile(pattern),
                    ggml_type=tensor_quantization_type_to_ggml_type(target),
                    preserve_source_dtype=False,
                    apply_to=TensorNameTarget.BOTH,
                    lane=RuleLane.USER_OVERRIDE,
                    reason="user override",
                )
            )

        for rule in self.required_rules:
            if self._rule_matches_context(rule, recipe=recipe, policy_preset=policy_preset):
                compiled.append(self._compile_rule(rule, lane=RuleLane.REQUIRED_INVARIANT))

        return compiled

    def require_recipe_supported(self, recipe: QuantizationRecipe) -> None:
        if recipe not in self.supported_recipes:
            supported = ", ".join(item.value for item in self.supported_recipes)
            raise ValueError(f"Recipe {recipe.value!r} is not supported by quantization policy {self.id!r}; supported: {supported}")

    def supports_policy_preset(
        self,
        *,
        recipe: QuantizationRecipe,
        policy_preset: QuantPolicyPreset,
        model_config: Mapping[str, Any],
    ) -> bool:
        return policy_preset in self._policy_presets_for_recipe(recipe=recipe, model_config=model_config)

    def default_policy_preset(self, *, recipe: QuantizationRecipe, model_config: Mapping[str, Any]) -> QuantPolicyPreset | None:
        presets = self._policy_presets_for_recipe(recipe=recipe, model_config=model_config)
        if not presets:
            return None
        if QuantPolicyPreset.MQ in presets:
            return QuantPolicyPreset.MQ
        return presets[0]

    def surface_descriptor(self, *, profile_id: ConverterProfileId, model_config: Mapping[str, Any]) -> QuantizationSurfaceDescriptor:
        recipes = tuple(recipe for recipe in QuantizationRecipe if recipe in self.supported_recipes)
        default_recipe = self.default_recipe if self.default_recipe in self.supported_recipes else recipes[0]
        presets_by_recipe: dict[QuantizationRecipe, tuple[QuantPolicyPreset, ...]] = {}
        default_by_recipe: dict[QuantizationRecipe, QuantPolicyPreset | None] = {}
        for recipe in recipes:
            presets = self._policy_presets_for_recipe(recipe=recipe, model_config=model_config)
            presets_by_recipe[recipe] = presets
            default_by_recipe[recipe] = self.default_policy_preset(recipe=recipe, model_config=model_config)
        return QuantizationSurfaceDescriptor(
            profile_id=profile_id,
            supported_recipes=recipes,
            default_recipe=default_recipe,
            policy_presets_by_recipe=presets_by_recipe,
            default_policy_preset_by_recipe=default_by_recipe,
        )

    def _iter_recipe_rules(
        self,
        *,
        model_config: Mapping[str, Any],
        recipe: QuantizationRecipe,
    ) -> tuple[TensorTypeRule, ...]:
        generated: list[TensorTypeRule] = []
        for factory in self.recipe_rule_factories:
            generated.extend(factory(model_config, recipe))
        return tuple(generated)

    def _iter_default_rules(
        self,
        *,
        model_config: Mapping[str, Any],
        recipe: QuantizationRecipe,
        policy_preset: QuantPolicyPreset,
    ) -> tuple[TensorTypeRule, ...]:
        generated: list[TensorTypeRule] = []
        for factory in self.default_rule_factories:
            generated.extend(factory(model_config, recipe, policy_preset))
        return (*self.default_rules, *generated)

    def _rule_matches_context(
        self,
        rule: TensorTypeRule,
        *,
        recipe: QuantizationRecipe,
        policy_preset: QuantPolicyPreset | None,
    ) -> bool:
        if not rule.when.matches(recipe):
            return False
        if rule.policy_presets is not None:
            if policy_preset is None or policy_preset not in rule.policy_presets:
                return False
        return True

    def _compile_rule(self, rule: TensorTypeRule, *, lane: RuleLane) -> CompiledTensorTypeRule:
        return CompiledTensorTypeRule(
            pattern=re.compile(rule.pattern),
            ggml_type=rule.ggml_type,
            preserve_source_dtype=rule.preserve_source_dtype,
            apply_to=rule.apply_to,
            lane=lane,
            reason=rule.reason,
        )

    def _validate_generated_rules(self, *, recipe: QuantizationRecipe, rules: Sequence[CompiledTensorTypeRule]) -> None:
        for rule in rules:
            if rule.lane not in {RuleLane.RECIPE_INTRINSIC, RuleLane.PROFILE_POLICY}:
                continue
            if rule.preserve_source_dtype or rule.ggml_type is None:
                continue
            if generated_rule_is_downgrade(recipe, rule.ggml_type):
                raise ValueError(
                    f"{self.id} generated {rule.lane.value} rule downgrades {recipe.value}: "
                    f"{rule.pattern.pattern!r} -> {rule.ggml_type.name} ({rule.reason})"
                )

    def _policy_presets_for_recipe(
        self,
        *,
        recipe: QuantizationRecipe,
        model_config: Mapping[str, Any],
    ) -> tuple[QuantPolicyPreset, ...]:
        if recipe_spec(recipe).is_float or recipe not in self.supported_recipes:
            return ()

        signatures = {
            preset: self._policy_signature_for_preset(recipe=recipe, model_config=model_config, policy_preset=preset)
            for preset in QuantPolicyPreset
        }
        if all(not signature for signature in signatures.values()):
            return ()

        selected: list[QuantPolicyPreset] = []
        seen: set[tuple[tuple[str, str, str, str], ...]] = set()

        for preset in (QuantPolicyPreset.HQ, QuantPolicyPreset.MQ):
            signature = signatures[preset]
            if not signature or signature in seen:
                continue
            selected.append(preset)
            seen.add(signature)

        lq_signature = signatures[QuantPolicyPreset.LQ]
        if lq_signature:
            if lq_signature not in seen:
                selected.append(QuantPolicyPreset.LQ)
        elif not signatures[QuantPolicyPreset.MQ]:
            selected.append(QuantPolicyPreset.MQ)
        else:
            selected.append(QuantPolicyPreset.LQ)

        return tuple(selected)

    def _policy_signature_for_preset(
        self,
        *,
        recipe: QuantizationRecipe,
        model_config: Mapping[str, Any],
        policy_preset: QuantPolicyPreset,
    ) -> tuple[tuple[str, str, str, str], ...]:
        rules = self._iter_default_rules(model_config=model_config, recipe=recipe, policy_preset=policy_preset)
        matched = tuple(
            rule
            for rule in rules
            if self._rule_matches_context(rule, recipe=recipe, policy_preset=policy_preset)
        )
        return tuple(
            (
                rule.pattern,
                rule.ggml_type.name if rule.ggml_type is not None else "SOURCE",
                rule.apply_to.value,
                "source" if rule.preserve_source_dtype else "fixed",
            )
            for rule in matched
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
    "QuantizationSurfaceDescriptor",
    "RecipeRuleFactory",
    "RuleLane",
    "TensorNameTarget",
    "TensorTypeRule",
]
