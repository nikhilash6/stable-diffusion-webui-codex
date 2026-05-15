"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Request-scoped SDXL BrushNet session for masked img2img.
Resolves the pinned `random_mask_brushnet_ckpt_sdxl_v0` asset bundle from the dedicated SDXL root, loads the
BrushNet side model with strict state-dict ownership, prepares masked-image conditioning latents from the canonical
masked-img2img bundle, patches a cloned SDXL denoiser for one sampling pass, and restores baseline Codex objects
after sampling completes.

Symbols (top-level; keep in sync; no ghosts):
- `BrushNetAssets` (dataclass): Exact pinned BrushNet asset paths and variant metadata for the active repo/workspace.
- `BrushNetOutput` (dataclass): Down/mid/up residual blocks produced by one BrushNet forward pass.
- `BrushNetModel` (class): Minimal local BrushNet inference module matching the pinned SDXL checkpoint layout.
- `_BrushNetSamplingSession` (class): Request-scoped session bundle that owns the patched denoiser hooks, loaded BrushNet model, and restoration bookkeeping for one sampling pass.
- `resolve_brushnet_assets` (function): Resolve the pinned BrushNet SDXL assets from `sdxl_brushnet`.
- `apply_brushnet_for_sampling` (function): Context manager that patches one SDXL denoiser clone for the current masked sampling pass.
- `_find_variant_directories` (function): Resolve all candidate BrushNet variant directories under the configured `sdxl_brushnet` roots.
- `_load_brushnet_model` (function): Build and strict-load the pinned BrushNet side model on the request-scoped runtime device.
- `_load_brushnet_config` (function): Read the pinned BrushNet `config.json` as a strict object payload.
- `_validate_brushnet_against_base_unet` (function): Fail loud when the pinned BrushNet checkpoint does not match the active SDXL base UNet contract.
- `_build_brushnet_conditioning_latents` (function): Build the canonical masked-image conditioning latents consumed by BrushNet.
- `_resolve_runtime_device_and_dtype` (function): Resolve the active sampling owner device/dtype from the cloned denoiser seam.
- `_resolve_module_device_and_dtype` (function): Inspect one loaded module for its canonical device/dtype ownership.
- `_resolve_brushnet_pooled_dim` (function): Resolve the pooled text-conditioning width required by the loaded BrushNet checkpoint.
- `_brushnet_time_ids` (function): Build SDXL time ids for the current batch on the active BrushNet runtime device/dtype.
- `_broadcast_batch` (function): Broadcast one batch-shaped conditioning tensor to the current denoiser batch size.
- `_zero_module` (function): Zero one module in place during local BrushNet model construction.
- `_as_mapping` (function): Enforce strict mapping payloads for BrushNet config/state fragments.
- `_set_brushnet_hooks` (function): Install temporary BrushNet hook state on the cloned SDXL denoiser owner path.
- `_clean_brushnet_hooks` (function): Remove temporary BrushNet hook state from the cloned SDXL denoiser owner path.
- `_patch_brushnet_layer` (function): Wrap one SDXL layer forward for BrushNet residual injection during sampling.
- `_restore_brushnet_layer` (function): Restore one SDXL layer after BrushNet sampling completes.
- `_forward_patched_by_brushnet` (function): Layer wrapper that injects BrushNet residual samples into the cloned SDXL denoiser pass.
- `_resolve_sdxl_brushnet_layer_targets` (function): Resolve the exact SDXL down/mid/up layer targets patched by the pinned BrushNet checkpoint.
- `_resolve_block_layer` (function): Resolve one typed layer from one SDXL block with fail-loud errors.
- `_resolve_layer_from_block` (function): Resolve one typed layer from one block sequence with fail-loud errors.
- `_assign_brushnet_samples` (function): Assign one batch of BrushNet residual samples onto the patched SDXL layer targets.
- `_forward_up_block_with_samples` (function): Execute one patched SDXL up block while consuming the assigned BrushNet residual samples.
"""

from __future__ import annotations

import contextlib
import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block

from apps.backend.infra.config.paths import get_paths_for
from apps.backend.runtime.checkpoint.io import load_torch_file
from apps.backend.runtime.logging import get_backend_logger
from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.pipeline_stages.masked_img2img import MaskedImg2ImgBundle
from apps.backend.runtime.processing.conditioners import encode_image_batch, resolve_processing_encode_seed
from apps.backend.runtime.common.nn.unet.layers import Downsample, ResBlock, SpatialTransformer, Upsample

logger = get_backend_logger("backend.runtime.families.sd.brushnet")

_BRUSHNET_ROOT = "sdxl_brushnet"
_BRUSHNET_VARIANT = "random_mask_brushnet_ckpt_sdxl_v0"
_BRUSHNET_CONFIG_FILENAME = "config.json"
_BRUSHNET_WEIGHTS_FILENAME = "diffusion_pytorch_model.safetensors"
_SDXL_BRUSHNET_ALLOWED_CHECKPOINT_DIM = 2816
_SDXL_BRUSHNET_TIME_ID_COUNT = 6


@dataclass(frozen=True)
class BrushNetAssets:
    variant: str
    variant_dir: str
    config_path: str
    weights_path: str


@dataclass(frozen=True)
class BrushNetOutput:
    down_block_res_samples: tuple[torch.Tensor, ...]
    mid_block_res_sample: torch.Tensor
    up_block_res_samples: tuple[torch.Tensor, ...]


class BrushNetModel(torch.nn.Module, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 5,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: tuple[str, ...] = ("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        mid_block_type: str | None = "MidBlock2D",
        up_block_types: tuple[str, ...] = ("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        only_cross_attention: bool | tuple[bool, ...] = False,
        block_out_channels: tuple[int, ...] = (320, 640, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1.0,
        act_fn: str = "silu",
        norm_num_groups: int | None = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 2048,
        transformer_layers_per_block: int | tuple[int, ...] = (1, 2, 10),
        encoder_hid_dim: int | None = None,
        encoder_hid_dim_type: str | None = None,
        attention_head_dim: int | tuple[int, ...] = (5, 10, 20),
        num_attention_heads: int | tuple[int, ...] | None = None,
        use_linear_projection: bool = True,
        class_embed_type: str | None = None,
        addition_embed_type: str | None = "text_time",
        addition_time_embed_dim: int | None = 256,
        num_class_embeds: int | None = None,
        upcast_attention: bool | None = None,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: int | None = _SDXL_BRUSHNET_ALLOWED_CHECKPOINT_DIM,
        brushnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: tuple[int, ...] | None = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
    ) -> None:
        super().__init__()
        del conditioning_embedding_out_channels, global_pool_conditions, addition_embed_type_num_heads

        if int(in_channels) != 4:
            raise RuntimeError(f"BrushNet requires in_channels=4; got {in_channels}.")
        if int(conditioning_channels) != 5:
            raise RuntimeError(f"BrushNet requires conditioning_channels=5; got {conditioning_channels}.")
        if tuple(str(value) for value in down_block_types) != ("DownBlock2D", "DownBlock2D", "DownBlock2D"):
            raise RuntimeError(f"BrushNet requires SDXL down_block_types; got {down_block_types!r}.")
        if tuple(str(value) for value in up_block_types) != ("UpBlock2D", "UpBlock2D", "UpBlock2D"):
            raise RuntimeError(f"BrushNet requires SDXL up_block_types; got {up_block_types!r}.")
        if str(mid_block_type or "") not in {"MidBlock2D", "UNetMidBlock2D"}:
            raise RuntimeError(
                "BrushNet requires mid_block_type='MidBlock2D'/'UNetMidBlock2D'; "
                f"got {mid_block_type!r}."
            )
        if int(cross_attention_dim) != 2048:
            raise RuntimeError(f"BrushNet requires cross_attention_dim=2048; got {cross_attention_dim}.")
        if projection_class_embeddings_input_dim != _SDXL_BRUSHNET_ALLOWED_CHECKPOINT_DIM:
            raise RuntimeError(
                "BrushNet requires projection_class_embeddings_input_dim=2816; "
                f"got {projection_class_embeddings_input_dim!r}."
            )
        if str(addition_embed_type or "") != "text_time":
            raise RuntimeError(f"BrushNet requires addition_embed_type='text_time'; got {addition_embed_type!r}.")
        if addition_time_embed_dim != 256:
            raise RuntimeError(f"BrushNet requires addition_time_embed_dim=256; got {addition_time_embed_dim!r}.")
        if encoder_hid_dim is not None or encoder_hid_dim_type is not None:
            raise RuntimeError(
                "BrushNet SDXL tranche does not support encoder_hid_dim / encoder_hid_dim_type overrides."
            )
        if class_embed_type is not None or num_class_embeds is not None:
            raise RuntimeError("BrushNet SDXL tranche does not support class embedding overrides.")

        if num_attention_heads is None:
            num_attention_heads = attention_head_dim
        if isinstance(only_cross_attention, bool):
            only_cross_attention = (only_cross_attention,) * len(down_block_types)
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = (transformer_layers_per_block,) * len(down_block_types)

        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in_condition = torch.nn.Conv2d(
            int(in_channels) + int(conditioning_channels),
            int(block_out_channels[0]),
            kernel_size=conv_in_kernel,
            padding=conv_in_padding,
        )

        time_embed_dim = int(block_out_channels[0]) * 4
        self.time_proj = Timesteps(int(block_out_channels[0]), flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(int(block_out_channels[0]), time_embed_dim, act_fn=act_fn)
        self.add_time_proj = Timesteps(int(addition_time_embed_dim), flip_sin_to_cos, freq_shift)
        self.add_embedding = TimestepEmbedding(int(projection_class_embeddings_input_dim), time_embed_dim)

        self.down_blocks = torch.nn.ModuleList([])
        self.brushnet_down_blocks = torch.nn.ModuleList([])
        output_channel = int(block_out_channels[0])

        first_down = torch.nn.Conv2d(output_channel, output_channel, kernel_size=1)
        self.brushnet_down_blocks.append(_zero_module(first_down))

        for index, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = int(block_out_channels[index])
            is_final_block = index == len(block_out_channels) - 1
            down_block = get_down_block(
                str(down_block_type),
                num_layers=int(layers_per_block),
                transformer_layers_per_block=int(transformer_layers_per_block[index]),
                in_channels=int(input_channel),
                out_channels=int(output_channel),
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=float(norm_eps),
                resnet_act_fn=str(act_fn),
                resnet_groups=norm_num_groups,
                cross_attention_dim=int(cross_attention_dim),
                num_attention_heads=int(num_attention_heads[index]) if num_attention_heads[index] is not None else None,
                attention_head_dim=int(attention_head_dim[index]) if attention_head_dim[index] is not None else output_channel,
                downsample_padding=int(downsample_padding),
                use_linear_projection=bool(use_linear_projection),
                only_cross_attention=bool(only_cross_attention[index]),
                upcast_attention=bool(upcast_attention) if upcast_attention is not None else False,
                resnet_time_scale_shift=str(resnet_time_scale_shift),
            )
            self.down_blocks.append(down_block)
            for _ in range(int(layers_per_block)):
                brushnet_block = torch.nn.Conv2d(output_channel, output_channel, kernel_size=1)
                self.brushnet_down_blocks.append(_zero_module(brushnet_block))
            if not is_final_block:
                brushnet_block = torch.nn.Conv2d(output_channel, output_channel, kernel_size=1)
                self.brushnet_down_blocks.append(_zero_module(brushnet_block))

        mid_block_channel = int(block_out_channels[-1])
        self.brushnet_mid_block = _zero_module(torch.nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1))
        self.mid_block = UNetMidBlock2D(
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            dropout=0.0,
            num_layers=1,
            resnet_eps=float(norm_eps),
            resnet_act_fn=str(act_fn),
            output_scale_factor=float(mid_block_scale_factor),
            resnet_groups=int(norm_num_groups) if norm_num_groups is not None else 32,
            resnet_time_scale_shift=str(resnet_time_scale_shift),
            add_attention=False,
            attention_head_dim=int(attention_head_dim[-1]) if attention_head_dim[-1] is not None else 1,
        )

        reversed_block_out_channels = list(reversed([int(value) for value in block_out_channels]))
        reversed_num_attention_heads = list(reversed([int(value) for value in num_attention_heads]))
        reversed_transformer_layers_per_block = list(reversed([int(value) for value in transformer_layers_per_block]))
        reversed_only_cross_attention = list(reversed([bool(value) for value in only_cross_attention]))

        output_channel = reversed_block_out_channels[0]
        self.up_blocks = torch.nn.ModuleList([])
        self.brushnet_up_blocks = torch.nn.ModuleList([])

        for index, up_block_type in enumerate(up_block_types):
            is_final_block = index == len(block_out_channels) - 1
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[index]
            input_channel = reversed_block_out_channels[min(index + 1, len(block_out_channels) - 1)]
            add_upsample = not is_final_block

            up_block = get_up_block(
                str(up_block_type),
                num_layers=int(layers_per_block) + 1,
                transformer_layers_per_block=int(reversed_transformer_layers_per_block[index]),
                in_channels=int(input_channel),
                out_channels=int(output_channel),
                prev_output_channel=int(prev_output_channel),
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=float(norm_eps),
                resnet_act_fn=str(act_fn),
                resolution_idx=index,
                resnet_groups=norm_num_groups,
                cross_attention_dim=int(cross_attention_dim),
                num_attention_heads=int(reversed_num_attention_heads[index]),
                use_linear_projection=bool(use_linear_projection),
                only_cross_attention=bool(reversed_only_cross_attention[index]),
                upcast_attention=bool(upcast_attention) if upcast_attention is not None else False,
                resnet_time_scale_shift=str(resnet_time_scale_shift),
                attention_head_dim=int(attention_head_dim[index]) if attention_head_dim[index] is not None else output_channel,
            )
            self.up_blocks.append(up_block)
            for _ in range(int(layers_per_block) + 1):
                brushnet_block = torch.nn.Conv2d(output_channel, output_channel, kernel_size=1)
                self.brushnet_up_blocks.append(_zero_module(brushnet_block))
            if not is_final_block:
                brushnet_block = torch.nn.Conv2d(output_channel, output_channel, kernel_size=1)
                self.brushnet_up_blocks.append(_zero_module(brushnet_block))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        *,
        encoder_hidden_states: torch.Tensor,
        brushnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        added_cond_kwargs: Mapping[str, torch.Tensor] | None = None,
    ) -> BrushNetOutput:
        if added_cond_kwargs is None:
            raise RuntimeError("BrushNet requires added_cond_kwargs with 'text_embeds' and 'time_ids'.")
        if "text_embeds" not in added_cond_kwargs or "time_ids" not in added_cond_kwargs:
            raise RuntimeError("BrushNet requires added_cond_kwargs['text_embeds'] and added_cond_kwargs['time_ids'].")

        if str(self.config.brushnet_conditioning_channel_order) == "bgr":
            brushnet_cond = torch.flip(brushnet_cond, dims=[1])
        elif str(self.config.brushnet_conditioning_channel_order) != "rgb":
            raise RuntimeError(
                "BrushNet conditioning channel order must be 'rgb' or 'bgr'; "
                f"got {self.config.brushnet_conditioning_channel_order!r}."
            )

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            if isinstance(timestep, float):
                timesteps = torch.tensor([timestep], dtype=torch.float32, device=sample.device)
            else:
                timesteps = torch.tensor([timestep], dtype=torch.int64, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        text_embeds = added_cond_kwargs["text_embeds"]
        time_ids = added_cond_kwargs["time_ids"]
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        aug_emb = self.add_embedding(torch.cat([text_embeds, time_embeds], dim=-1).to(dtype=emb.dtype))
        emb = emb + aug_emb

        brushnet_cond = torch.cat([sample, brushnet_cond], dim=1)
        sample = self.conv_in_condition(brushnet_cond)

        down_block_res_samples: tuple[torch.Tensor, ...] = (sample,)
        for downsample_block in self.down_blocks:
            if getattr(downsample_block, "has_cross_attention", False):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += tuple(res_samples)

        brushnet_down_block_res_samples: list[torch.Tensor] = []
        for down_sample, zero_conv in zip(down_block_res_samples, self.brushnet_down_blocks):
            brushnet_down_block_res_samples.append(zero_conv(down_sample))

        if getattr(self.mid_block, "has_cross_attention", False):
            sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
        else:
            sample = self.mid_block(sample, emb)
        brushnet_mid_block_res_sample = self.brushnet_mid_block(sample)

        up_block_res_samples: tuple[torch.Tensor, ...] = ()
        remaining_down_samples = down_block_res_samples
        for upsample_block in self.up_blocks:
            res_samples = remaining_down_samples[-len(upsample_block.resnets) :]
            remaining_down_samples = remaining_down_samples[: -len(upsample_block.resnets)]
            sample, up_res_samples = _forward_up_block_with_samples(
                upsample_block,
                hidden_states=sample,
                res_hidden_states_tuple=tuple(res_samples),
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                remaining_down_samples=remaining_down_samples,
            )
            up_block_res_samples += tuple(up_res_samples)

        brushnet_up_block_res_samples: list[torch.Tensor] = []
        for up_sample, zero_conv in zip(up_block_res_samples, self.brushnet_up_blocks):
            brushnet_up_block_res_samples.append(zero_conv(up_sample))

        if float(conditioning_scale) != 1.0:
            brushnet_down_block_res_samples = [tensor * float(conditioning_scale) for tensor in brushnet_down_block_res_samples]
            brushnet_mid_block_res_sample = brushnet_mid_block_res_sample * float(conditioning_scale)
            brushnet_up_block_res_samples = [tensor * float(conditioning_scale) for tensor in brushnet_up_block_res_samples]

        return BrushNetOutput(
            down_block_res_samples=tuple(brushnet_down_block_res_samples),
            mid_block_res_sample=brushnet_mid_block_res_sample,
            up_block_res_samples=tuple(brushnet_up_block_res_samples),
        )


class _BrushNetSamplingSession:
    def __init__(
        self,
        *,
        brushnet_model: BrushNetModel,
        conditioning_latents: torch.Tensor,
        processing: Any,
        patched_denoiser: Any,
        conditioning_scale: float = 1.0,
    ) -> None:
        self.brushnet_model = brushnet_model
        self.conditioning_latents = conditioning_latents.detach().to(device="cpu")
        self.conditioning_scale = float(conditioning_scale)
        self.processing = processing
        diffusion_model = getattr(getattr(patched_denoiser, "model", None), "diffusion_model", None)
        if diffusion_model is None:
            raise RuntimeError("BrushNet requires patched_denoiser.model.diffusion_model.")
        input_layers, middle_layer, output_layers = _resolve_sdxl_brushnet_layer_targets(diffusion_model)
        self.input_layers = input_layers
        self.middle_layer = middle_layer
        self.output_layers = output_layers
        self.pooled_dim = _resolve_brushnet_pooled_dim(self.brushnet_model)

    def wrap_model_function(self, previous_wrapper: Any | None):
        def wrapper(apply_model, data):
            c_dict = dict(data["c"])
            transformer_options = c_dict.get("transformer_options")
            if not isinstance(transformer_options, dict):
                raise RuntimeError("BrushNet requires c['transformer_options'] as a dict.")

            x = data["input"]
            timestep = data["timestep"]
            crossattn = c_dict.get("c_crossattn")
            if not isinstance(crossattn, torch.Tensor) or crossattn.ndim != 3:
                raise RuntimeError(
                    "BrushNet requires c['c_crossattn'] as a 3D tensor during SDXL sampling."
                )
            vector = c_dict.get("y")
            if not isinstance(vector, torch.Tensor) or vector.ndim != 2:
                raise RuntimeError("BrushNet requires c['y'] as the SDXL pooled/time vector tensor.")
            if int(vector.shape[1]) != int(self.brushnet_model.config.projection_class_embeddings_input_dim):
                raise RuntimeError(
                    "BrushNet requires the SDXL pooled/time vector width to match the pinned checkpoint config. "
                    f"Expected {self.brushnet_model.config.projection_class_embeddings_input_dim}, got {tuple(vector.shape)}."
                )
            if int(vector.shape[0]) != int(crossattn.shape[0]):
                raise RuntimeError(
                    "BrushNet requires matching batch sizes for c_crossattn and y. "
                    f"Got {tuple(crossattn.shape)} and {tuple(vector.shape)}."
                )

            sample_model = getattr(self.processing.sd_model.codex_objects, "denoiser", None)
            sampler_model = getattr(sample_model, "model", None)
            predictor = getattr(sampler_model, "predictor", None)
            if predictor is None or not hasattr(predictor, "calculate_input"):
                raise RuntimeError("BrushNet requires a denoiser predictor exposing calculate_input(...).")

            brushnet_device, brushnet_dtype = _resolve_module_device_and_dtype(self.brushnet_model)
            control_model_input = predictor.calculate_input(timestep, x).to(device=brushnet_device, dtype=brushnet_dtype)
            conditioning_latents = _broadcast_batch(
                self.conditioning_latents.to(device=brushnet_device, dtype=brushnet_dtype),
                target_batch_size=int(control_model_input.shape[0]),
            )
            pooled_text = vector[:, : self.pooled_dim].to(device=brushnet_device, dtype=brushnet_dtype)
            time_ids = _brushnet_time_ids(
                processing=self.processing,
                batch=int(control_model_input.shape[0]),
                device=brushnet_device,
                dtype=brushnet_dtype,
            )

            outputs = self.brushnet_model(
                control_model_input,
                timestep.to(device=brushnet_device),
                encoder_hidden_states=crossattn.to(device=brushnet_device, dtype=brushnet_dtype),
                brushnet_cond=conditioning_latents,
                conditioning_scale=self.conditioning_scale,
                added_cond_kwargs={
                    "text_embeds": pooled_text,
                    "time_ids": time_ids,
                },
            )
            if len(outputs.down_block_res_samples) != len(self.input_layers):
                raise RuntimeError(
                    "BrushNet down-block output count does not match the active SDXL UNet input block layout. "
                    f"Expected {len(self.input_layers)}, got {len(outputs.down_block_res_samples)}."
                )
            if len(outputs.up_block_res_samples) != len(self.output_layers):
                raise RuntimeError(
                    "BrushNet up-block output count does not match the active SDXL UNet output block layout. "
                    f"Expected {len(self.output_layers)}, got {len(outputs.up_block_res_samples)}."
                )
            _assign_brushnet_samples(
                input_layers=self.input_layers,
                middle_layer=self.middle_layer,
                output_layers=self.output_layers,
                outputs=outputs,
            )
            next_data = dict(data)
            next_data["c"] = c_dict
            if previous_wrapper is not None:
                return previous_wrapper(apply_model, next_data)
            return apply_model(next_data["input"], next_data["timestep"], **next_data["c"])

        return wrapper


@contextlib.contextmanager
def apply_brushnet_for_sampling(*, processing: Any, masked_bundle: MaskedImg2ImgBundle) -> Iterator[BrushNetAssets]:
    engine = getattr(processing, "sd_model", None)
    if engine is None:
        raise RuntimeError("BrushNet requires processing.sd_model before sampling begins.")
    engine_id = str(getattr(engine, "engine_id", "") or "").strip().lower()
    if engine_id != "sdxl":
        raise RuntimeError(f"BrushNet requires exact engine id 'sdxl'; got '{engine_id or '<empty>'}'.")

    assets = resolve_brushnet_assets()
    previous_codex_objects = engine.codex_objects
    patched_codex_objects = previous_codex_objects.shallow_copy()
    patched_denoiser = previous_codex_objects.denoiser.clone()
    runtime_device, runtime_dtype = _resolve_runtime_device_and_dtype(patched_denoiser=patched_denoiser)
    base_diffusion_model = getattr(getattr(patched_denoiser, "model", None), "diffusion_model", None)
    if base_diffusion_model is None:
        raise RuntimeError("BrushNet requires patched_denoiser.model.diffusion_model to resolve the SDXL base UNet.")

    brushnet_model = _load_brushnet_model(
        assets=assets,
        device=runtime_device,
        dtype=runtime_dtype,
        base_diffusion_model=base_diffusion_model,
    )
    patched_denoiser.add_extra_torch_module_during_sampling(brushnet_model, cast_to_unet_dtype=False)
    conditioning_latents = _build_brushnet_conditioning_latents(
        processing=processing,
        masked_bundle=masked_bundle,
    )
    session = _BrushNetSamplingSession(
        brushnet_model=brushnet_model,
        conditioning_latents=conditioning_latents,
        processing=processing,
        patched_denoiser=patched_denoiser,
    )
    previous_wrapper = patched_denoiser.model_options.get("model_function_wrapper")
    hooks_applied = False
    try:
        _set_brushnet_hooks(base_diffusion_model)
        hooks_applied = True
        patched_denoiser.set_model_unet_function_wrapper(session.wrap_model_function(previous_wrapper))
        patched_codex_objects.denoiser = patched_denoiser
        engine.codex_objects = patched_codex_objects
        logger.info(
            "Applying BrushNet for engine=%s variant=%s config=%s weights=%s",
            engine_id,
            assets.variant,
            assets.config_path,
            assets.weights_path,
        )
        yield assets
    finally:
        if hooks_applied:
            _clean_brushnet_hooks(base_diffusion_model)
        engine.codex_objects = previous_codex_objects


def resolve_brushnet_assets() -> BrushNetAssets:
    roots = get_paths_for(_BRUSHNET_ROOT)
    if not roots:
        raise RuntimeError(
            "BrushNet requires paths.json key 'sdxl_brushnet' pointing at a root that contains "
            f"'{_BRUSHNET_VARIANT}/{_BRUSHNET_CONFIG_FILENAME}' and '{_BRUSHNET_VARIANT}/{_BRUSHNET_WEIGHTS_FILENAME}'."
        )
    candidates = _find_variant_directories(roots=roots, variant=_BRUSHNET_VARIANT)
    if not candidates:
        raise RuntimeError(
            f"BrushNet variant '{_BRUSHNET_VARIANT}' was not found under any configured '{_BRUSHNET_ROOT}' root: "
            + ", ".join(roots)
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"BrushNet variant '{_BRUSHNET_VARIANT}' is ambiguous under '{_BRUSHNET_ROOT}': "
            + ", ".join(candidates)
        )
    variant_dir = Path(candidates[0])
    config_path = variant_dir / _BRUSHNET_CONFIG_FILENAME
    weights_path = variant_dir / _BRUSHNET_WEIGHTS_FILENAME
    if not config_path.is_file():
        raise RuntimeError(f"BrushNet config is missing at {config_path}.")
    if not weights_path.is_file():
        raise RuntimeError(f"BrushNet weights are missing at {weights_path}.")
    return BrushNetAssets(
        variant=_BRUSHNET_VARIANT,
        variant_dir=str(variant_dir.resolve(strict=False)),
        config_path=str(config_path.resolve(strict=False)),
        weights_path=str(weights_path.resolve(strict=False)),
    )


def _find_variant_directories(*, roots: Sequence[str], variant: str) -> list[str]:
    matches: list[str] = []
    for raw_root in roots:
        root = Path(str(raw_root)).expanduser()
        if not root.exists():
            continue
        candidates: list[Path] = []
        if root.is_dir() and root.name == variant:
            candidates.append(root)
        if root.is_dir():
            try:
                candidates.extend(path for path in root.rglob(variant) if path.is_dir())
            except Exception:
                continue
        for candidate in candidates:
            resolved = str(candidate.resolve(strict=False))
            if resolved not in matches:
                matches.append(resolved)
    return sorted(matches)


def _load_brushnet_model(
    *,
    assets: BrushNetAssets,
    device: torch.device,
    dtype: torch.dtype,
    base_diffusion_model: Any,
) -> BrushNetModel:
    config_payload = _load_brushnet_config(assets.config_path)
    with using_codex_operations(device=device, dtype=dtype, manual_cast_enabled=True):
        model = BrushNetModel(**config_payload)
    model.to(device=device, dtype=dtype)
    weights = _as_mapping(load_torch_file(assets.weights_path, safe_load=True, device="cpu"), label="brushnet_weights")
    missing, unexpected = safe_load_state_dict(model, weights, log_name="BrushNet")
    if missing or unexpected:
        raise RuntimeError(
            "BrushNet failed strict load. "
            f"missing={missing} unexpected={unexpected}"
        )
    _validate_brushnet_against_base_unet(model=model, base_diffusion_model=base_diffusion_model, variant=assets.variant)
    model.eval()
    return model


def _load_brushnet_config(path: str) -> dict[str, object]:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"BrushNet config could not be read from {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"BrushNet config at {path} must be an object.")
    raw.pop("_class_name", None)
    raw.pop("_diffusers_version", None)
    raw.pop("_name_or_path", None)
    return raw


def _validate_brushnet_against_base_unet(*, model: BrushNetModel, base_diffusion_model: Any, variant: str) -> None:
    codex_config = getattr(base_diffusion_model, "codex_config", None)
    if codex_config is None:
        raise RuntimeError("BrushNet requires the active SDXL diffusion model to expose codex_config.")
    context_dim = int(getattr(codex_config, "context_dim", 0) or 0)
    adm_in_channels = int(getattr(codex_config, "adm_in_channels", 0) or 0)
    if context_dim != int(model.config.cross_attention_dim):
        raise RuntimeError(
            f"BrushNet variant '{variant}' expects SDXL context_dim={model.config.cross_attention_dim}, "
            f"but the active base UNet exposes context_dim={context_dim}."
        )
    if adm_in_channels != int(model.config.projection_class_embeddings_input_dim):
        raise RuntimeError(
            f"BrushNet variant '{variant}' expects SDXL adm_in_channels={model.config.projection_class_embeddings_input_dim}, "
            f"but the active base UNet exposes adm_in_channels={adm_in_channels}."
        )
    if int(getattr(codex_config, "in_channels", 0) or 0) != int(model.config.in_channels):
        raise RuntimeError(
            f"BrushNet variant '{variant}' expects base in_channels={model.config.in_channels}, "
            f"but the active base UNet exposes in_channels={getattr(codex_config, 'in_channels', None)!r}."
        )


def _build_brushnet_conditioning_latents(*, processing: Any, masked_bundle: MaskedImg2ImgBundle) -> torch.Tensor:
    init_tensor = masked_bundle.init_tensor.to(dtype=torch.float32)
    pixel_mask = F.interpolate(
        masked_bundle.latent_masked.to(dtype=torch.float32),
        size=tuple(init_tensor.shape[-2:]),
        mode="nearest",
    )
    masked_tensor = init_tensor * (1.0 - pixel_mask)
    masked_latent = encode_image_batch(
        processing.sd_model,
        masked_tensor,
        encode_seed=resolve_processing_encode_seed(processing),
        stage="runtime.families.sd.brushnet.build_conditioning_latents.encode",
    )
    conditioning = torch.cat(
        [
            masked_latent.to(dtype=torch.float32),
            masked_bundle.latent_unmasked.to(device=masked_latent.device, dtype=torch.float32),
        ],
        dim=1,
    )
    return conditioning.detach().to(device="cpu")


def _resolve_runtime_device_and_dtype(*, patched_denoiser: Any) -> tuple[torch.device, torch.dtype]:
    diffusion_model = getattr(getattr(patched_denoiser, "model", None), "diffusion_model", None)
    if diffusion_model is None:
        raise RuntimeError("BrushNet requires patched_denoiser.model.diffusion_model to resolve runtime device/dtype.")
    for parameter in diffusion_model.parameters():
        if isinstance(parameter, torch.Tensor):
            return parameter.device, parameter.dtype
    raise RuntimeError("BrushNet could not resolve an active diffusion-model parameter for runtime device/dtype.")


def _resolve_module_device_and_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    for parameter in module.parameters():
        if isinstance(parameter, torch.Tensor):
            return parameter.device, parameter.dtype
    raise RuntimeError(f"{type(module).__name__} does not expose any parameters for device/dtype resolution.")


def _resolve_brushnet_pooled_dim(model: BrushNetModel) -> int:
    raw_total = int(model.config.projection_class_embeddings_input_dim)
    raw_time = int(model.config.addition_time_embed_dim) * _SDXL_BRUSHNET_TIME_ID_COUNT
    pooled = raw_total - raw_time
    if pooled <= 0:
        raise RuntimeError(
            f"BrushNet pooled dimension is invalid: projection={raw_total} addition_time_embed_dim={model.config.addition_time_embed_dim}."
        )
    return pooled


def _brushnet_time_ids(*, processing: Any, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    width = int(getattr(processing, "width", 1024) or 1024)
    height = int(getattr(processing, "height", 1024) or 1024)
    hires_cfg = getattr(processing, "hires", None)
    target_width = int(getattr(hires_cfg, "resize_x", 0) or width)
    target_height = int(getattr(hires_cfg, "resize_y", 0) or height)
    crop_left = int(getattr(processing, "sdxl_crop_left", 0) or 0)
    crop_top = int(getattr(processing, "sdxl_crop_top", 0) or 0)
    raw = torch.tensor(
        [[height, width, crop_top, crop_left, target_height, target_width]],
        device=device,
        dtype=dtype,
    )
    return raw.repeat(int(batch), 1)


def _broadcast_batch(tensor: torch.Tensor, *, target_batch_size: int) -> torch.Tensor:
    if int(tensor.shape[0]) == int(target_batch_size):
        return tensor
    if int(tensor.shape[0]) == 1:
        return tensor.repeat(int(target_batch_size), 1, 1, 1)
    raise RuntimeError(
        f"BrushNet batch broadcast requires batch=1 or batch={target_batch_size}; got {tuple(tensor.shape)}."
    )


def _zero_module(module: torch.nn.Module) -> torch.nn.Module:
    for parameter in module.parameters():
        torch.nn.init.zeros_(parameter)
    return module


def _as_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    raise RuntimeError(f"{label} must resolve to a mapping state dict; got {type(value).__name__}.")


def _set_brushnet_hooks(diffusion_model: Any) -> None:
    for block in getattr(diffusion_model, "input_blocks", []):
        for layer in block:
            _patch_brushnet_layer(layer)
    for layer in getattr(diffusion_model, "middle_block", []):
        _patch_brushnet_layer(layer)
    for block in getattr(diffusion_model, "output_blocks", []):
        for layer in block:
            _patch_brushnet_layer(layer)


def _clean_brushnet_hooks(diffusion_model: Any) -> None:
    for block in getattr(diffusion_model, "input_blocks", []):
        for layer in block:
            _restore_brushnet_layer(layer)
    for layer in getattr(diffusion_model, "middle_block", []):
        _restore_brushnet_layer(layer)
    for block in getattr(diffusion_model, "output_blocks", []):
        for layer in block:
            _restore_brushnet_layer(layer)


def _patch_brushnet_layer(layer: torch.nn.Module) -> None:
    if not hasattr(layer, "_brushnet_original_forward"):
        setattr(layer, "_brushnet_original_forward", layer.forward)
    layer.forward = types.MethodType(_forward_patched_by_brushnet, layer)
    setattr(layer, "add_sample_after", 0)


def _restore_brushnet_layer(layer: torch.nn.Module) -> None:
    if hasattr(layer, "_brushnet_original_forward"):
        layer.forward = getattr(layer, "_brushnet_original_forward")
        delattr(layer, "_brushnet_original_forward")
    if hasattr(layer, "add_sample_after"):
        delattr(layer, "add_sample_after")


def _forward_patched_by_brushnet(self, x, *args, **kwargs):
    original_forward = getattr(self, "_brushnet_original_forward", None)
    if original_forward is None:
        raise RuntimeError(f"BrushNet patched layer {type(self).__name__} lost its original forward reference.")
    hidden = original_forward(x, *args, **kwargs)
    to_add = getattr(self, "add_sample_after", 0)
    if torch.is_tensor(to_add):
        if tuple(hidden.shape[-2:]) != tuple(to_add.shape[-2:]):
            to_add = F.interpolate(to_add, size=tuple(hidden.shape[-2:]), mode="bicubic", align_corners=False)
        hidden = hidden + to_add.to(device=hidden.device, dtype=hidden.dtype)
    elif to_add != 0:
        hidden = hidden + to_add
    self.add_sample_after = 0
    return hidden


def _resolve_sdxl_brushnet_layer_targets(diffusion_model: Any) -> tuple[list[torch.nn.Module], torch.nn.Module, list[torch.nn.Module]]:
    input_specs = [
        (0, torch.nn.Conv2d),
        (1, ResBlock),
        (2, ResBlock),
        (3, Downsample),
        (4, SpatialTransformer),
        (5, SpatialTransformer),
        (6, Downsample),
        (7, SpatialTransformer),
        (8, SpatialTransformer),
    ]
    output_specs = [
        (0, SpatialTransformer),
        (1, SpatialTransformer),
        (2, SpatialTransformer),
        (2, Upsample),
        (3, SpatialTransformer),
        (4, SpatialTransformer),
        (5, SpatialTransformer),
        (5, Upsample),
        (6, ResBlock),
        (7, ResBlock),
        (8, ResBlock),
    ]
    input_layers = [
        _resolve_block_layer(blocks=diffusion_model.input_blocks, block_index=block_index, layer_type=layer_type, label="input")
        for block_index, layer_type in input_specs
    ]
    middle_layer = _resolve_layer_from_block(
        diffusion_model.middle_block,
        layer_type=ResBlock,
        label="middle",
    )
    output_layers = [
        _resolve_block_layer(blocks=diffusion_model.output_blocks, block_index=block_index, layer_type=layer_type, label="output")
        for block_index, layer_type in output_specs
    ]
    return input_layers, middle_layer, output_layers


def _resolve_block_layer(*, blocks: Sequence[Sequence[torch.nn.Module]], block_index: int, layer_type: type[torch.nn.Module], label: str) -> torch.nn.Module:
    if block_index < 0 or block_index >= len(blocks):
        raise RuntimeError(
            f"BrushNet {label} block index {block_index} is outside the active SDXL UNet layout (len={len(blocks)})."
        )
    return _resolve_layer_from_block(blocks[block_index], layer_type=layer_type, label=f"{label}[{block_index}]")


def _resolve_layer_from_block(block: Sequence[torch.nn.Module], *, layer_type: type[torch.nn.Module], label: str) -> torch.nn.Module:
    for layer in reversed(list(block)):
        if isinstance(layer, layer_type):
            return layer
    available = ", ".join(type(layer).__name__ for layer in block)
    raise RuntimeError(
        f"BrushNet could not find layer type {layer_type.__name__} inside {label}. "
        f"Available layers: {available or '<empty>'}."
    )


def _assign_brushnet_samples(
    *,
    input_layers: Sequence[torch.nn.Module],
    middle_layer: torch.nn.Module,
    output_layers: Sequence[torch.nn.Module],
    outputs: BrushNetOutput,
) -> None:
    for layer, sample in zip(input_layers, outputs.down_block_res_samples):
        setattr(layer, "add_sample_after", sample)
    setattr(middle_layer, "add_sample_after", outputs.mid_block_res_sample)
    for layer, sample in zip(output_layers, outputs.up_block_res_samples):
        setattr(layer, "add_sample_after", sample)


def _forward_up_block_with_samples(
    upsample_block: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    res_hidden_states_tuple: tuple[torch.Tensor, ...],
    temb: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    remaining_down_samples: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    samples: list[torch.Tensor] = []
    residuals = res_hidden_states_tuple
    has_cross_attention = bool(getattr(upsample_block, "has_cross_attention", False))

    for resnet_index, resnet in enumerate(upsample_block.resnets):
        res_hidden_states = residuals[-1]
        residuals = residuals[:-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        hidden_states = resnet(hidden_states, temb)
        if has_cross_attention:
            attentions = getattr(upsample_block, "attentions", None)
            if attentions is None or resnet_index >= len(attentions):
                raise RuntimeError(
                    f"BrushNet expected attention module {resnet_index} on {type(upsample_block).__name__}."
                )
            hidden_states = attentions[resnet_index](hidden_states, encoder_hidden_states=encoder_hidden_states).sample
        samples.append(hidden_states)

    upsample_size = remaining_down_samples[-1].shape[2:] if remaining_down_samples else None
    for upsampler in getattr(upsample_block, "upsamplers", []) or []:
        hidden_states = upsampler(hidden_states, upsample_size)
        samples.append(hidden_states)

    return hidden_states, tuple(samples)
