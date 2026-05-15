"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Dataclasses for the Codex model parser (plans, context, and estimated config).
Defines the declarative parser plan model (splits/converters/validations) plus the `ParserContext` and `CodexEstimatedConfig` structures used
to return a normalized view of checkpoint components to loaders/adapters without synthetic inpaint-capability heuristics.

Symbols (top-level; keep in sync; no ghosts):
- `SplitSpec` (dataclass): Declares how to split a state dict into one component by key prefix.
- `ConverterSpec` (dataclass): Declares a component conversion step (function + target component name).
- `ValidationSpec` (dataclass): Declares a validation step over the `ParserContext`.
- `ParserPlan` (dataclass): Full plan (splits + converters + validations) with safety checks for converter references.
- `ParserPlanBundle` (dataclass): `ParserPlan` plus a `build_config(...)` function that returns `CodexEstimatedConfig`.
- `ComponentState` (dataclass): Mutable component tensor mapping extracted from the root state dict.
- `ParserArtifacts` (dataclass): Plan execution output (components + leftovers).
- `ParserContext` (dataclass): Execution context passed through converters/validators (root state + signature + component states + metadata).
- `CodexComponent` (dataclass): Immutable component mapping used in `CodexEstimatedConfig`.
- `CodexEstimatedConfig` (dataclass): Normalized parse result returned to loaders (signature + repo id + components + extras + core config).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence

from apps.backend.runtime.model_registry.specs import (
    LatentFormat,
    ModelFamily,
    ModelSignature,
    PredictionKind,
    QuantizationHint,
)


@dataclass(slots=True)
class SplitSpec:
    name: str
    prefixes: Sequence[str]
    strip_prefix: str | None = None
    required: bool = True


@dataclass(slots=True)
class ConverterSpec:
    component: str
    function: Callable[[Dict[str, Any], "ParserContext"], Dict[str, Any]]


@dataclass(slots=True)
class ValidationSpec:
    name: str
    function: Callable[["ParserContext"], None]


@dataclass(slots=True)
class ParserPlan:
    splits: Sequence[SplitSpec]
    converters: Sequence[ConverterSpec] = field(default_factory=tuple)
    validations: Sequence[ValidationSpec] = field(default_factory=tuple)

    def ensure_component(self, component: str) -> None:
        if component not in {spec.name for spec in self.splits}:
            raise ValueError(f"Converter references unknown component '{component}'")


@dataclass(slots=True)
class ParserPlanBundle:
    plan: ParserPlan
    build_config: Callable[[ParserContext], "CodexEstimatedConfig"]


@dataclass(slots=True)
class ComponentState:
    name: str
    tensors: MutableMapping[str, Any]


@dataclass(slots=True)
class ParserArtifacts:
    components: Dict[str, ComponentState]
    leftovers: Dict[str, Any]


@dataclass(slots=True)
class ParserContext:
    root_state: MutableMapping[str, Any]
    signature: ModelSignature
    plan: ParserPlan
    components: Dict[str, ComponentState] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def require(self, component: str) -> ComponentState:
        try:
            return self.components[component]
        except KeyError as exc:
            raise KeyError(f"Component '{component}' missing from context") from exc


@dataclass(slots=True)
class CodexComponent:
    name: str
    state_dict: Mapping[str, Any]


@dataclass(slots=True)
class CodexEstimatedConfig:
    signature: ModelSignature
    repo_id: str
    family: ModelFamily
    prediction: PredictionKind
    latent_format: LatentFormat
    quantization: QuantizationHint
    components: Dict[str, CodexComponent]
    text_encoder_map: Dict[str, str]
    extras: Dict[str, Any] = field(default_factory=dict)
    core_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def huggingface_repo(self) -> str:
        return self.repo_id

    def replace_components(self, updates: Dict[str, Mapping[str, Any]]) -> "CodexEstimatedConfig":
        new_components = dict(self.components)
        for name, state in updates.items():
            new_components[name] = CodexComponent(name=name, state_dict=state)
        return replace(self, components=new_components)
