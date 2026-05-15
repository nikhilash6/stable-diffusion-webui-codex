"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SD engine spec/runtime assembly (classic CLIP text encoding + UNet/VAE patchers + dynamic args).
Defines the schema for SD “classic” engine branches and provides the assembly function that validates required components and produces
an `SDEngineRuntime` used by SD engines to run inference.

Symbols (top-level; keep in sync; no ghosts):
- `SDEngineConfigurationError` (exception): Raised when required SD components or dynamic args are missing/None.
- `_require_component` (function): Validates a required component is present and non-None in a component mapping.
- `_require_dynamic_arg` (function): Fetches a required dynamic arg and raises on missing/None.
- `_resolve_attr` (function): Resolves a dotted attribute path on an object and raises on missing/None (used for branch accessors).
- `SDClassicBranchSpec` (dataclass): Spec for a classic CLIP branch (identifier, embedding shape, clip-skip defaults, pooled/text-proj flags).
- `SDT5BranchSpec` (dataclass): Spec for a T5 branch (identifier + min length) used by SD3-style configurations (when applicable).
- `SDEngineSpec` (dataclass): High-level SD engine spec (family + branches + clip/tenc expectations) used to drive runtime assembly.
- `SDEngineRuntime` (dataclass): Assembled runtime (CLIP/text engine + UNet/VAE patchers + resolved model family/dynamic flags).
- `assemble_engine_runtime` (function): Validates components and builds an `SDEngineRuntime` (resolves CLIP branches, sets clip-skip, and
  wires `ClassicTextProcessingEngine` + patchers based on `dynamic_args` and model family).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Tuple

from apps.backend.infra.config.args import dynamic_args
from apps.backend.patchers.clip import CLIP
from apps.backend.patchers.unet import UnetPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.text_processing.classic_engine import ClassicTextProcessingEngine
from apps.backend.runtime.text_processing.t5_engine import T5TextProcessingEngine

logger = get_backend_logger("backend.engines.sd.spec")


class SDEngineConfigurationError(RuntimeError):
    """Raised when required SD engine components or configuration are missing."""


def _require_component(components: Mapping[str, object], key: str, context: str) -> object:
    try:
        component = components[key]
    except KeyError as error:
        raise SDEngineConfigurationError(f"Missing component '{key}' required for {context}.") from error
    if component is None:
        raise SDEngineConfigurationError(f"Component '{key}' is None for {context}.")
    return component


def _require_dynamic_arg(name: str) -> object:
    try:
        value = dynamic_args[name]
    except KeyError as error:
        raise SDEngineConfigurationError(f"Dynamic argument '{name}' is required for SD engines.") from error
    if value is None:
        raise SDEngineConfigurationError(f"Dynamic argument '{name}' is None.")
    return value


def _resolve_attr(obj: object, attr_path: str, *, context: str) -> object:
    current = obj
    for fragment in attr_path.split("."):
        if not hasattr(current, fragment):
            raise SDEngineConfigurationError(f"Attribute '{fragment}' missing on {context}.")
        current = getattr(current, fragment)
        if current is None:
            raise SDEngineConfigurationError(f"Attribute '{fragment}' resolved to None on {context}.")
    return current


@dataclass(frozen=True, slots=True)
class SDClassicBranchSpec:
    identifier: str
    clip_attr: str
    embedding_expected_shape: int
    minimal_clip_skip: int = 1
    default_clip_skip: int = 1
    text_projection: bool = False
    return_pooled: bool = False
    final_layer_norm: bool = False
    embedding_key: str | None = None

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("identifier must be provided")
        if self.default_clip_skip < self.minimal_clip_skip:
            raise ValueError("default_clip_skip cannot be smaller than minimal_clip_skip")

    @property
    def embedding_identifier(self) -> str:
        return self.embedding_key or self.identifier


@dataclass(frozen=True, slots=True)
class SDT5BranchSpec:
    identifier: str
    clip_attr: str
    min_length: int = 75

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("identifier must be provided")
        if self.min_length <= 0:
            raise ValueError("min_length must be positive")


@dataclass(frozen=True, slots=True)
class SDEngineSpec:
    name: str
    clip_model_keys: Mapping[str, str]
    tokenizer_keys: Mapping[str, str]
    classic_branches: Tuple[SDClassicBranchSpec, ...]
    t5_branches: Tuple[SDT5BranchSpec, ...] = ()
    unet_key: str = "unet"
    vae_key: str = "vae"
    scheduler_key: str = "scheduler"
    embedding_dir_arg: str = "embedding_dir"
    emphasis_arg: str = "emphasis_name"

    def __post_init__(self) -> None:
        if not self.classic_branches and not self.t5_branches:
            raise ValueError("At least one text branch must be defined")
        if set(self.clip_model_keys.keys()) != set(self.tokenizer_keys.keys()):
            raise ValueError("clip_model_keys and tokenizer_keys must have identical keys")
        expected_aliases = {branch.identifier for branch in self.classic_branches} | {
            branch.identifier for branch in self.t5_branches
        }
        if expected_aliases != set(self.clip_model_keys.keys()):
            raise ValueError("Branch identifiers must match clip/tokenizer aliases")

    @property
    def classic_order(self) -> Tuple[str, ...]:
        return tuple(branch.identifier for branch in self.classic_branches)

    @property
    def t5_order(self) -> Tuple[str, ...]:
        return tuple(branch.identifier for branch in self.t5_branches)


@dataclass(slots=True)
class SDEngineRuntime:
    clip: CLIP
    vae: VAE
    unet: UnetPatcher
    scheduler: object | None
    classic_text: Dict[str, ClassicTextProcessingEngine]
    classic_specs: Dict[str, SDClassicBranchSpec]
    classic_order: Tuple[str, ...]
    t5_text: Dict[str, T5TextProcessingEngine]
    t5_specs: Dict[str, SDT5BranchSpec]

    def set_clip_skip(self, clip_skip: int) -> None:
        if not isinstance(clip_skip, int):
            raise TypeError("clip_skip must be an integer")
        for identifier, spec in self.classic_specs.items():
            if clip_skip < spec.minimal_clip_skip:
                raise ValueError(
                    f"Clip skip {clip_skip} is below minimal {spec.minimal_clip_skip} for branch '{identifier}'."
                )
            self.classic_text[identifier].clip_skip = clip_skip

    def reset_clip_skip(self) -> None:
        """Reset classic branch clip_skip to their spec defaults.

        Used for the `clip_skip=0` "use default" sentinel: callers can always
        apply a per-job clip skip without leaking state across jobs.
        """
        for identifier, spec in self.classic_specs.items():
            self.classic_text[identifier].clip_skip = spec.default_clip_skip

    def primary_classic(self) -> ClassicTextProcessingEngine:
        if not self.classic_order:
            raise RuntimeError("No classic branches available")
        return self.classic_text[self.classic_order[0]]

    def classic_engine(self, identifier: str) -> ClassicTextProcessingEngine:
        try:
            return self.classic_text[identifier]
        except KeyError as error:
            raise KeyError(f"Unknown classic branch '{identifier}'.") from error

    def t5_engine(self, identifier: str) -> T5TextProcessingEngine:
        try:
            return self.t5_text[identifier]
        except KeyError as error:
            raise KeyError(f"Unknown T5 branch '{identifier}'.") from error


# Map spec names to ModelFamily enum
_NAME_TO_FAMILY = {
    "sd15": ModelFamily.SD15,
    "sd20": ModelFamily.SD20,
    "sdxl": ModelFamily.SDXL,
    "sdxl_refiner": ModelFamily.SDXL_REFINER,
    "sd35": ModelFamily.SD35,
}


def assemble_engine_runtime(
    spec: SDEngineSpec,
    estimated_config,
    components: Mapping[str, object],
) -> SDEngineRuntime:
    logger.debug("Assembling SD engine '%s'", spec.name)

    clip_model_dict: MutableMapping[str, object] = {}
    tokenizer_dict: MutableMapping[str, object] = {}

    for identifier in spec.clip_model_keys:
        model_component_key = spec.clip_model_keys[identifier]
        tokenizer_component_key = spec.tokenizer_keys[identifier]
        clip_model_dict[identifier] = _require_component(components, model_component_key, f"{spec.name}.text_encoder")
        tokenizer_dict[identifier] = _require_component(components, tokenizer_component_key, f"{spec.name}.tokenizer")

    clip = CLIP(model_dict=clip_model_dict, tokenizer_dict=tokenizer_dict, model_config=estimated_config)
    logger.debug("CLIP patcher assembled for '%s'.", spec.name)

    vae_model = _require_component(components, spec.vae_key, f"{spec.name}.vae")
    family = _NAME_TO_FAMILY.get(spec.name)
    vae = VAE(model=vae_model, family=family)
    logger.debug("VAE wrapper instantiated for '%s' (family=%s).", spec.name, family)

    unet_model = _require_component(components, spec.unet_key, f"{spec.name}.unet")
    scheduler = components.get(spec.scheduler_key)
    if scheduler is None:
        logger.debug("No scheduler provided for '%s' (using None)", spec.name)
    unet = UnetPatcher.from_model(model=unet_model, diffusers_scheduler=scheduler, config=estimated_config)
    logger.debug("UNet patcher created for '%s'.", spec.name)

    # --- Assembly invariants (no silent fallbacks) ---------------------------------
    diffusion_model = getattr(unet, "model", None)
    if diffusion_model is None or getattr(diffusion_model, "diffusion_model", None) is None:
        raise SDEngineConfigurationError(
            f"UNet patcher for '{spec.name}' lacks an inner diffusion model; cannot assemble runtime."
        )
    inner = diffusion_model.diffusion_model
    codex_cfg = getattr(inner, "codex_config", None)
    if codex_cfg is None:
        raise SDEngineConfigurationError(
            f"UNet for '{spec.name}' is missing 'codex_config' (UNetConfig); ensure model was constructed via Codex UNet."
        )
    # context_dim must be declared; numeric checking occurs at runtime
    if getattr(codex_cfg, "context_dim", None) is None:
        raise SDEngineConfigurationError(
            f"UNet for '{spec.name}' has no context_dim; cross-attention conditioning is not configured."
        )
    # Engines that require ADM/y must advertise it through num_classes
    if spec.name in ("sdxl", "sdxl_refiner", "sd35"):
        if getattr(inner, "num_classes", None) is None:
            raise SDEngineConfigurationError(
                f"Engine '{spec.name}' expects ADM/y conditioning but UNet 'num_classes' is None. "
                f"Configure UNet with class/ADM support (e.g., num_classes='sequential' and adm_in_channels)."
            )
        if getattr(inner, "num_classes", None) == "sequential" and getattr(codex_cfg, "adm_in_channels", None) in (None, 0):
            raise SDEngineConfigurationError(
                f"Engine '{spec.name}' requires ADM channels but UNet 'adm_in_channels' is not set."
            )

    embedding_dir = _require_dynamic_arg(spec.embedding_dir_arg)
    emphasis_name = _require_dynamic_arg(spec.emphasis_arg)

    classic_text: Dict[str, ClassicTextProcessingEngine] = {}
    classic_specs: Dict[str, SDClassicBranchSpec] = {}

    for branch in spec.classic_branches:
        text_encoder = _resolve_attr(
            clip.cond_stage_model,
            branch.clip_attr,
            context=f"{spec.name}.{branch.identifier}.cond_stage_model",
        )
        tokenizer = _resolve_attr(
            clip.tokenizer,
            branch.clip_attr,
            context=f"{spec.name}.{branch.identifier}.tokenizer",
        )

        engine = ClassicTextProcessingEngine(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            embedding_dir=embedding_dir,
            embedding_key=branch.embedding_identifier,
            embedding_expected_shape=branch.embedding_expected_shape,
            emphasis_name=emphasis_name,
            text_projection=branch.text_projection,
            minimal_clip_skip=branch.minimal_clip_skip,
            clip_skip=branch.default_clip_skip,
            return_pooled=branch.return_pooled,
            final_layer_norm=branch.final_layer_norm,
        )

        classic_text[branch.identifier] = engine
        classic_specs[branch.identifier] = branch
        logger.debug(
            "Classic text branch '%s' initialised for '%s' (clip_skip=%d).",
            branch.identifier,
            spec.name,
            branch.default_clip_skip,
        )

    t5_text: Dict[str, T5TextProcessingEngine] = {}
    t5_specs: Dict[str, SDT5BranchSpec] = {}

    for branch in spec.t5_branches:
        text_encoder = _resolve_attr(
            clip.cond_stage_model,
            branch.clip_attr,
            context=f"{spec.name}.{branch.identifier}.cond_stage_model",
        )
        tokenizer = _resolve_attr(
            clip.tokenizer,
            branch.clip_attr,
            context=f"{spec.name}.{branch.identifier}.tokenizer",
        )

        engine = T5TextProcessingEngine(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            emphasis_name=emphasis_name,
            min_length=branch.min_length,
        )

        t5_text[branch.identifier] = engine
        t5_specs[branch.identifier] = branch
        logger.debug("T5 text branch '%s' initialised for '%s'", branch.identifier, spec.name)

    rt = SDEngineRuntime(
        clip=clip,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        classic_text=classic_text,
        classic_specs=classic_specs,
        classic_order=spec.classic_order,
        t5_text=t5_text,
        t5_specs=t5_specs,
    )
    return rt


SD15_SPEC = SDEngineSpec(
    name="sd15",
    clip_model_keys={"clip_l": "text_encoder"},
    tokenizer_keys={"clip_l": "tokenizer"},
    classic_branches=(
        SDClassicBranchSpec(
            identifier="clip_l",
            clip_attr="clip_l",
            embedding_expected_shape=768,
            minimal_clip_skip=1,
            default_clip_skip=1,
            text_projection=False,
            return_pooled=False,
            final_layer_norm=True,
        ),
    ),
)

SD20_SPEC = SDEngineSpec(
    name="sd20",
    clip_model_keys={"clip_h": "text_encoder"},
    tokenizer_keys={"clip_h": "tokenizer"},
    classic_branches=(
        SDClassicBranchSpec(
            identifier="clip_h",
            clip_attr="clip_h",
            embedding_expected_shape=1024,
            minimal_clip_skip=1,
            default_clip_skip=1,
            text_projection=False,
            return_pooled=False,
            final_layer_norm=True,
        ),
    ),
)

SDXL_SPEC = SDEngineSpec(
    name="sdxl",
    clip_model_keys={"clip_l": "text_encoder", "clip_g": "text_encoder_2"},
    tokenizer_keys={"clip_l": "tokenizer", "clip_g": "tokenizer_2"},
    classic_branches=(
        SDClassicBranchSpec(
            identifier="clip_l",
            clip_attr="clip_l",
            embedding_expected_shape=2048,
            minimal_clip_skip=1,
            default_clip_skip=2,
            text_projection=False,
            return_pooled=False,
            final_layer_norm=False,
        ),
        SDClassicBranchSpec(
            identifier="clip_g",
            clip_attr="clip_g",
            embedding_expected_shape=2048,
            minimal_clip_skip=1,
            default_clip_skip=2,
            text_projection=True,
            return_pooled=True,
            final_layer_norm=False,
        ),
    ),
)

SDXL_REFINER_SPEC = SDEngineSpec(
    name="sdxl_refiner",
    clip_model_keys={"clip_g": "text_encoder"},
    tokenizer_keys={"clip_g": "tokenizer"},
    classic_branches=(
        SDClassicBranchSpec(
            identifier="clip_g",
            clip_attr="clip_g",
            embedding_expected_shape=2048,
            minimal_clip_skip=1,
            default_clip_skip=2,
            text_projection=True,
            return_pooled=True,
            final_layer_norm=False,
        ),
    ),
)

SD35_SPEC = SDEngineSpec(
    name="sd35",
    clip_model_keys={
        "clip_l": "text_encoder",
        "clip_g": "text_encoder_2",
        "t5xxl": "text_encoder_3",
    },
    tokenizer_keys={
        "clip_l": "tokenizer",
        "clip_g": "tokenizer_2",
        "t5xxl": "tokenizer_3",
    },
    classic_branches=(
        SDClassicBranchSpec(
            identifier="clip_l",
            clip_attr="clip_l",
            embedding_expected_shape=768,
            minimal_clip_skip=1,
            default_clip_skip=1,
            text_projection=True,
            return_pooled=True,
            final_layer_norm=False,
        ),
        SDClassicBranchSpec(
            identifier="clip_g",
            clip_attr="clip_g",
            embedding_expected_shape=1280,
            minimal_clip_skip=1,
            default_clip_skip=1,
            text_projection=True,
            return_pooled=True,
            final_layer_norm=False,
        ),
    ),
    t5_branches=(
        SDT5BranchSpec(
            identifier="t5xxl",
            clip_attr="t5xxl",
            min_length=256,
        ),
    ),
)


__all__ = [
    "SDClassicBranchSpec",
    "SDT5BranchSpec",
    "SDEngineSpec",
    "SDEngineRuntime",
    "SDEngineConfigurationError",
    "assemble_engine_runtime",
    "SD15_SPEC",
    "SD20_SPEC",
    "SDXL_SPEC",
    "SDXL_REFINER_SPEC",
    "SD35_SPEC",
]
