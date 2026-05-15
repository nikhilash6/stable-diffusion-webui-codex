"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Codex-native UNet patcher (sampling reservations + ControlNet chaining + patch registration).
Wraps the diffusion UNet in a `ModelPatcher` with extra state for deterministic sampling reservations and a structured ControlNet graph,
building a composite runtime on activation (no fallbacks; invalid payloads raise).

Symbols (top-level; keep in sync; no ghosts):
- `SamplingReservation` (dataclass): Tracks reserved memory and auxiliary patchers required during sampling (clone/add_memory/add_patcher).
- `UnetPatcher` (class): Main UNet patcher; wraps `SamplerModel`, tracks `ControlNode` graph, builds/activates composite control runtime, and
  exposes validated helpers for cloning, patch registration, and sampling-time reservations (contains nested helper methods for node cloning,
  composite rebuild, and property accessors).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

import torch

from apps.backend.runtime.common.nn.unet.layers import SpatialTransformer
from apps.backend.runtime.controlnet.config import ControlNode, ControlNodeConfig, ControlRequest, ControlWeightSchedule
from apps.backend.runtime.controlnet.runtime import build_composite
from apps.backend.runtime.sampling_adapters.sampler_model import SamplerModel
from .base import ModelPatcher

logger = get_backend_logger("backend.patchers.unet")


@dataclass
class SamplingReservation:
    preserved_bytes: int = 0
    auxiliary_patchers: List[ModelPatcher] = field(default_factory=list)

    def clone(self) -> "SamplingReservation":
        return SamplingReservation(
            preserved_bytes=self.preserved_bytes,
            auxiliary_patchers=self.auxiliary_patchers.copy(),
        )

    def add_memory(self, amount: int) -> None:
        if not isinstance(amount, int):
            raise TypeError("memory_in_bytes must be provided as an int")
        if amount < 0:
            raise ValueError("memory_in_bytes must be non-negative")
        logger.debug("Reserving %s additional bytes for UNet sampling", amount)
        self.preserved_bytes += amount

    def add_patcher(self, patcher: ModelPatcher) -> None:
        if not isinstance(patcher, ModelPatcher):
            raise TypeError("model_patcher must be a ModelPatcher instance")
        logger.debug("Registering auxiliary model patcher %s for sampling", patcher)
        self.auxiliary_patchers.append(patcher)


class UnetPatcher(ModelPatcher):
    """Codex-specific UNet patcher with validated state helpers."""

    MAX_PATCH_SLOTS = 16

    @classmethod
    def from_model(cls, model, diffusers_scheduler, config, predictor=None):
        model = SamplerModel(model=model, diffusers_scheduler=diffusers_scheduler, predictor=predictor, config=config)
        return cls(
            model,
            load_device=model.diffusion_model.load_device,
            offload_device=model.diffusion_model.offload_device,
            current_device=model.diffusion_model.initial_device,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._control_nodes: List[ControlNode] = []
        self._sampling = SamplingReservation()
        self.extra_concat_condition: Optional[torch.Tensor] = None
        self._active_control = None

    def clone(self):
        clone = self.__class__(self.model, self.load_device, self.offload_device, self.size, self.current_device)
        clone.lora_patches = self.lora_patches.copy()
        clone.object_patches = self.object_patches.copy()
        clone.model_options = copy.deepcopy(self.model_options)
        clone._control_nodes = [self._clone_control_node(node) for node in self._control_nodes]
        clone._sampling = self._sampling.clone()
        clone.extra_concat_condition = self.extra_concat_condition
        clone._active_control = None
        logger.debug("Cloned UnetPatcher %s -> %s", id(self), id(clone))
        return clone

    @property
    def control_nodes(self) -> List[ControlNode]:
        return list(self._control_nodes)

    def activate_control(self):
        nodes = self.control_nodes
        self._active_control = build_composite(nodes)
        logger.debug(
            "Activated control composite with %d nodes",
            len(nodes),
        )
        return self._active_control

    def clear_control(self) -> None:
        self._active_control = None

    @property
    def controlnet_linked_list(self):
        return self._active_control or build_composite(self.control_nodes)

    def _clone_control_node(self, node: ControlNode) -> ControlNode:
        control = node.control
        control_copy = control.copy() if hasattr(control, "copy") else copy.deepcopy(control)
        request = copy.deepcopy(node.request)
        config = copy.deepcopy(node.config)
        return ControlNode(config=config, request=request, control=control_copy)

    def add_control_node(self, node: ControlNode) -> None:
        if not isinstance(node, ControlNode):
            raise TypeError("Expected ControlNode instance")
        self._control_nodes.append(node)
        logger.debug("Appended ControlNode %s", node.config.name)
        self.clear_control()


    @property
    def extra_preserved_memory_during_sampling(self) -> int:
        return self._sampling.preserved_bytes

    @extra_preserved_memory_during_sampling.setter
    def extra_preserved_memory_during_sampling(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("extra_preserved_memory_during_sampling must be an int")
        if value < 0:
            raise ValueError("extra_preserved_memory_during_sampling must be non-negative")
        self._sampling.preserved_bytes = value

    @property
    def extra_model_patchers_during_sampling(self) -> List[ModelPatcher]:
        return self._sampling.auxiliary_patchers

    @extra_model_patchers_during_sampling.setter
    def extra_model_patchers_during_sampling(self, value: Sequence[ModelPatcher]) -> None:
        if not isinstance(value, Sequence):
            raise TypeError("extra_model_patchers_during_sampling must be a sequence")
        for patcher in value:
            if not isinstance(patcher, ModelPatcher):
                raise TypeError("All entries must be ModelPatcher instances")
        self._sampling.auxiliary_patchers = list(value)

    def add_extra_preserved_memory_during_sampling(self, memory_in_bytes: int) -> None:
        # Use this to ask Codex to preserve a certain amount of memory during sampling.
        # If GPU VRAM is 8 GB, and memory_in_bytes is 2GB, i.e., memory_in_bytes = 2 * 1024 * 1024 * 1024
        # Then the sampling will always use less than 6GB memory by dynamically offload modules to CPU RAM.
        # You can estimate this using memory_management.module_size(any_pytorch_model) to get size of any pytorch models.
        self._sampling.add_memory(memory_in_bytes)

    def add_extra_model_patcher_during_sampling(self, model_patcher: ModelPatcher) -> None:
        # Use this to ask Codex to move extra model patchers to GPU during sampling.
        # This method will manage GPU memory perfectly.
        self._sampling.add_patcher(model_patcher)

    def add_extra_torch_module_during_sampling(self, module: torch.nn.Module, cast_to_unet_dtype: bool = True) -> ModelPatcher:
        # Use this method to bind an extra torch.nn.Module to this UNet during sampling.
        # This model `m` will be delegated to Codex memory management system.
        # `m` will be loaded to GPU everytime when sampling starts.
        # `m` will be unloaded if necessary.
        # `m` will influence Codex's judgement about use GPU memory or
        # capacity and decide whether to use module offload to make user's batch size larger.
        # Use cast_to_unet_dtype if you want `m` to have same dtype with unet during sampling.

        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be an instance of torch.nn.Module")

        if cast_to_unet_dtype:
            diffusion_model = getattr(self.model, "diffusion_model", None)
            if diffusion_model is None or not hasattr(diffusion_model, "dtype"):
                raise AttributeError("diffusion_model with dtype is required to cast extra modules")
            module.to(diffusion_model.dtype)
            logger.debug("Cast auxiliary module %s to dtype %s", module, diffusion_model.dtype)

        patcher = ModelPatcher(model=module, load_device=self.load_device, offload_device=self.offload_device)

        self.add_extra_model_patcher_during_sampling(patcher)
        return patcher

    def add_patched_controlnet(self, controlnet: Any) -> None:
        request = self._build_request_from_control(controlnet)
        config = ControlNodeConfig(
            name=getattr(controlnet, "name", controlnet.__class__.__name__),
            model_type=controlnet.__class__.__name__.lower(),
        )
        node = ControlNode(config=config, request=request, control=controlnet)
        self.add_control_node(node)

    def _build_request_from_control(self, controlnet: Any) -> ControlRequest:
        image = getattr(controlnet, "cond_hint_original", None)
        if image is None:
            raise ValueError("ControlNet cond_hint_original must be set before adding to UNet")
        strength = getattr(controlnet, "strength", 1.0)
        start_percent, end_percent = getattr(controlnet, "timestep_percent_range", (0.0, 1.0))
        schedule = ControlWeightSchedule(
            positive=getattr(controlnet, "positive_advanced_weighting", None),
            negative=getattr(controlnet, "negative_advanced_weighting", None),
            frame=getattr(controlnet, "advanced_frame_weighting", None),
            sigma=getattr(controlnet, "advanced_sigma_weighting", None),
        )
        request = ControlRequest(
            image=image,
            strength=strength,
            start_percent=start_percent,
            end_percent=end_percent,
            weight_schedule=schedule,
        )
        mask = getattr(controlnet, "advanced_mask_weighting", None)
        request.mask_config.mask = mask
        return request

    def list_controlnets(self) -> List[Any]:
        return [node.control for node in self._control_nodes]

    def append_model_option(self, key: str, value: Any, ensure_uniqueness: bool = False) -> None:
        bucket = self.model_options.setdefault(key, [])
        self._append_to_bucket(bucket, value, ensure_uniqueness)
        logger.debug("Appended model option '%s' with %s", key, self._describe_value(value))

    def append_transformer_option(self, key: str, value: Any, ensure_uniqueness: bool = False) -> None:
        transformer_options = self.model_options.setdefault("transformer_options", {})
        bucket = transformer_options.setdefault(key, [])
        self._append_to_bucket(bucket, value, ensure_uniqueness)
        logger.debug("Appended transformer option '%s' with %s", key, self._describe_value(value))

    def set_transformer_option(self, key: str, value: Any) -> None:
        transformer_options = self.model_options.setdefault("transformer_options", {})
        transformer_options[key] = value
        logger.debug("Set transformer option '%s' -> %s", key, self._describe_value(value))

    def add_conditioning_modifier(self, modifier, ensure_uniqueness: bool = False) -> None:
        self._ensure_callable(modifier, "conditioning_modifier")
        self.append_model_option("conditioning_modifiers", modifier, ensure_uniqueness)

    def add_sampler_pre_cfg_function(self, modifier, ensure_uniqueness: bool = False) -> None:
        self._ensure_callable(modifier, "sampler_pre_cfg_function")
        self.append_model_option("sampler_pre_cfg_function", modifier, ensure_uniqueness)

    def set_memory_peak_estimation_modifier(self, modifier) -> None:
        self._ensure_callable(modifier, "memory_peak_estimation_modifier")
        self.model_options["memory_peak_estimation_modifier"] = modifier
        logger.debug("Set memory peak estimation modifier %s", modifier)

    def add_alphas_cumprod_modifier(self, modifier, ensure_uniqueness: bool = False) -> None:
        """

        For some reasons, this function only works in Codex's Script.process_batch(self, p, *args, **kwargs)

        For example, below is a worked modification:

        class ExampleScript(scripts.Script):

            def process_batch(self, p, *args, **kwargs):
                unet = p.sd_model.codex_objects.denoiser.clone()

                def modifier(x):
                    return x ** 0.5

                unet.add_alphas_cumprod_modifier(modifier)
                p.sd_model.codex_objects.denoiser = unet

                return

        This add_alphas_cumprod_modifier is the only patch option that should be used in process_batch()
        All other patch options should be called in process_before_every_sampling()

        """
        self._ensure_callable(modifier, "alphas_cumprod_modifier")
        self.append_model_option("alphas_cumprod_modifiers", modifier, ensure_uniqueness)

    def add_block_modifier(self, modifier, ensure_uniqueness: bool = False) -> None:
        self._ensure_callable(modifier, "block_modifier")
        self.append_transformer_option("block_modifiers", modifier, ensure_uniqueness)

    def add_block_inner_modifier(self, modifier, ensure_uniqueness: bool = False) -> None:
        self._ensure_callable(modifier, "block_inner_modifier")
        self.append_transformer_option("block_inner_modifiers", modifier, ensure_uniqueness)

    def add_controlnet_conditioning_modifier(self, modifier, ensure_uniqueness: bool = False) -> None:
        self._ensure_callable(modifier, "controlnet_conditioning_modifier")
        self.append_transformer_option("controlnet_conditioning_modifiers", modifier, ensure_uniqueness)

    def set_group_norm_wrapper(self, wrapper) -> None:
        self._ensure_callable(wrapper, "group_norm_wrapper")
        self.set_transformer_option("group_norm_wrapper", wrapper)

    def set_controlnet_model_function_wrapper(self, wrapper) -> None:
        self._ensure_callable(wrapper, "controlnet_model_function_wrapper")
        self.set_transformer_option("controlnet_model_function_wrapper", wrapper)

    def set_model_replace_all(self, patch, target: str = "attn1") -> None:
        self._ensure_callable(patch, "patch")
        layout = list(self._iter_transformer_coordinates())
        if not layout:
            raise RuntimeError("Unable to locate transformer blocks for UNet patch replacement")
        for block_name, block_index, transformer_index in layout:
            self.set_model_patch_replace(patch, target, block_name, block_index, transformer_index)
        logger.debug(
            "Registered patch %s across %d transformer positions for target '%s'",
            patch,
            len(layout),
            target,
        )

    def load_frozen_patcher(self, filename: str, state_dict: dict[str, Any], strength: float) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a dict[str, Any]")
        patch_dict: dict[str, dict[str, List[Any]]] = {}
        for k, w in state_dict.items():
            try:
                model_key, patch_type, weight_index_str = k.split("::", 2)
            except ValueError as exc:
                raise ValueError(f"Invalid frozen patcher key '{k}'") from exc
            try:
                weight_index = int(weight_index_str)
            except ValueError as exc:
                raise ValueError(f"Weight index must be int in key '{k}'") from exc
            if not 0 <= weight_index < self.MAX_PATCH_SLOTS:
                raise ValueError(f"Weight index {weight_index} outside supported range [0, {self.MAX_PATCH_SLOTS})")
            model_entry = patch_dict.setdefault(model_key, {})
            patch_entry = model_entry.setdefault(patch_type, [None] * self.MAX_PATCH_SLOTS)
            if patch_entry[weight_index] is not None:
                raise ValueError(f"Duplicate weight index {weight_index} for model '{model_key}' patch '{patch_type}'")
            patch_entry[weight_index] = w

        patch_flat: dict[str, tuple[str, List[Any]]] = {}
        for model_key, v in patch_dict.items():
            for patch_type, weight_list in v.items():
                trimmed_weights = self._trim_patch_weights(weight_list)
                patch_flat[model_key] = (patch_type, trimmed_weights)
                logger.debug(
                    "Prepared frozen patcher entry %s/%s with %d weights",
                    model_key,
                    patch_type,
                    len(trimmed_weights),
                )

        self.add_patches(
            filename=filename,
            patches=patch_flat,
            strength_patch=float(strength),
            strength_model=1.0,
        )

    # --- helpers -----------------------------------------------------------------

    def _append_to_bucket(self, bucket: List[Any], value: Any, ensure_uniqueness: bool) -> None:
        if ensure_uniqueness and value in bucket:
            logger.debug("Skipping duplicate registration for %s", self._describe_value(value))
            return
        bucket.append(value)

    @staticmethod
    def _describe_value(value: Any) -> str:
        if callable(value):
            return getattr(value, "__qualname__", repr(value))
        return repr(value)

    @staticmethod
    def _ensure_callable(value: Any, label: str) -> None:
        if not callable(value):
            raise TypeError(f"{label} must be callable")

    def _iter_transformer_coordinates(self) -> Iterable[tuple[str, int, int]]:
        diffusion_model = getattr(self.model, "diffusion_model", None)
        if diffusion_model is None:
            raise AttributeError("SamplerModel is missing diffusion_model reference")

        def block_transformer_indices(block: torch.nn.Module) -> range:
            spatial_transformers = [module for module in block if isinstance(module, SpatialTransformer)]
            if not spatial_transformers:
                return range(0)
            if len(spatial_transformers) != 1:
                raise RuntimeError(
                    "UNet transformer coordinate enumeration requires at most one SpatialTransformer per block; "
                    f"got {len(spatial_transformers)} in {type(block).__name__}."
                )
            return range(int(len(spatial_transformers[0].transformer_blocks)))

        input_blocks = getattr(diffusion_model, "input_blocks", [])
        for block_index, block in enumerate(input_blocks):
            for transformer_index in block_transformer_indices(block):
                yield "input", block_index, transformer_index

        middle_block = getattr(diffusion_model, "middle_block", None)
        if middle_block is not None:
            for transformer_index in block_transformer_indices(middle_block):
                yield "middle", 0, transformer_index

        output_blocks = getattr(diffusion_model, "output_blocks", [])
        for block_index, block in enumerate(output_blocks):
            for transformer_index in block_transformer_indices(block):
                yield "output", block_index, transformer_index

    def _trim_patch_weights(self, weight_list: List[Any]) -> List[Any]:
        trimmed = list(weight_list)
        while trimmed and trimmed[-1] is None:
            trimmed.pop()
        return trimmed or [None]
