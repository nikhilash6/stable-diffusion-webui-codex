"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Codex-native `UNet2DConditionModel` implementation (SD-style UNet with optional SpatialTransformer blocks).
Builds the UNet graph from `UNetConfig` and the core layers in `layers.py`, supporting cross-attention conditioning via `context` and
optional ADM conditioning via `y` (strict invariants: if `num_classes` is None then `y` must be None; otherwise `y` must be a 2D tensor).

Symbols (top-level; keep in sync; no ghosts):
- `UNet2DConditionModel` (class): Main UNet module; constructs input/down/middle/up blocks, embeds timesteps (and ADM when enabled),
  routes `context` into `SpatialTransformer` blocks, and exposes a diffusers-compatible config interface via `ConfigMixin`.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config

from apps.backend.runtime.logging import get_backend_logger
from apps.backend.runtime.sampling.block_progress import (
    BLOCK_PROGRESS_INDEX_KEY,
    BLOCK_PROGRESS_TOTAL_KEY,
)
from .config import UNetConfig
from .layers import (
    Downsample,
    ResBlock,
    SpatialTransformer,
    TimestepEmbedSequential,
    Upsample,
)
from .utils import apply_control, exists, timestep_embedding


class UNet2DConditionModel(nn.Module, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=-1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        transformer_depth_middle=None,
        transformer_depth_output=None,
    ):
        super().__init__()

        config = UNetConfig(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            disable_self_attentions=disable_self_attentions,
            num_attention_blocks=num_attention_blocks,
            disable_middle_self_attn=disable_middle_self_attn,
            use_linear_in_transformer=use_linear_in_transformer,
            adm_in_channels=adm_in_channels,
            transformer_depth_middle=transformer_depth_middle,
            transformer_depth_output=transformer_depth_output,
        )
        self.codex_config = config

        if context_dim is not None:
            assert use_spatial_transformer
        if num_heads == -1:
            assert num_head_channels != -1
        if num_head_channels == -1:
            assert num_heads != -1

        channel_mult = tuple(channel_mult)
        num_res_blocks_seq = list(config.expanded_num_res_blocks())
        transformer_depth_seq = config.transformer_depth_list(sum(num_res_blocks_seq))
        total_output_blocks = sum(block + 1 for block in num_res_blocks_seq)
        transformer_depth_output_seq = config.transformer_depth_output_list(total_output_blocks)

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks_seq
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Linear(adm_in_channels, time_embed_dim),
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, time_embed_dim),
                )
            else:
                raise ValueError("Unsupported conditioning mode")

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans: List[int] = [model_channels]
        ch = model_channels

        disable_self_attentions_seq: Sequence[bool] | None = disable_self_attentions
        num_attention_blocks_seq: Sequence[int] | None = num_attention_blocks

        transformer_depth_queue = list(transformer_depth_seq)

        for level, mult in enumerate(channel_mult):
            for block_index in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                num_transformers = transformer_depth_queue.pop(0) if transformer_depth_queue else 0
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = (
                        disable_self_attentions_seq[level]
                        if exists(disable_self_attentions_seq)
                        else False
                    )
                    if (not exists(num_attention_blocks_seq)) or (
                        block_index < num_attention_blocks_seq[level]
                    ):
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=num_transformers,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_checkpoint=use_checkpoint,
                                use_linear=use_linear_in_transformer,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                down_layer = (
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                    )
                    if resblock_updown
                    else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                )
                self.input_blocks.append(TimestepEmbedSequential(down_layer))
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        mid_block = [
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=None,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        ]

        transformer_depth_middle_value = (
            transformer_depth_middle if transformer_depth_middle is not None else -1
        )
        if transformer_depth_middle_value >= 0:
            mid_block.extend(
                [
                    SpatialTransformer(
                        ch,
                        num_heads,
                        dim_head,
                        depth=transformer_depth_middle_value,
                        context_dim=context_dim,
                        disable_self_attn=disable_middle_self_attn,
                        use_checkpoint=use_checkpoint,
                        use_linear=use_linear_in_transformer,
                    ),
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=None,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ),
                ]
            )
        self.middle_block = TimestepEmbedSequential(*mid_block)
        self._feature_size += ch

        transformer_depth_output_queue = list(transformer_depth_output_seq)
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for block_index in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                num_transformers = (
                    transformer_depth_output_queue.pop() if transformer_depth_output_queue else 0
                )
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = (
                        disable_self_attentions_seq[level]
                        if exists(disable_self_attentions_seq)
                        else False
                    )
                    if (not exists(num_attention_blocks_seq)) or (
                        block_index < num_attention_blocks_seq[level]
                    ):
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=num_transformers,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_checkpoint=use_checkpoint,
                                use_linear=use_linear_in_transformer,
                            )
                        )
                if level and block_index == self.num_res_blocks[level]:
                    out_ch = ch
                    up_layer = (
                        ResBlock(
                            channels=ch,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    layers.append(up_layer)
                    ch = out_ch
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self._total_spatial_transformer_blocks = self._count_total_spatial_transformer_blocks()

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def _count_total_spatial_transformer_blocks(self) -> int:
        total = 0

        def _count_in_sequence(sequence: TimestepEmbedSequential) -> None:
            nonlocal total
            for layer in sequence:
                if isinstance(layer, SpatialTransformer):
                    total += int(len(layer.transformer_blocks))

        for module in self.input_blocks:
            _count_in_sequence(module)
        _count_in_sequence(self.middle_block)
        for module in self.output_blocks:
            _count_in_sequence(module)
        return int(total)

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ):
        if transformer_options is None:
            transformer_options = {}
        elif not isinstance(transformer_options, dict):
            raise RuntimeError(
                "UNet forward transformer_options must be a dict when provided "
                f"(got {type(transformer_options).__name__})."
            )
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_options[BLOCK_PROGRESS_TOTAL_KEY] = int(self._total_spatial_transformer_blocks)
        transformer_options[BLOCK_PROGRESS_INDEX_KEY] = 0
        transformer_patches = transformer_options.get("patches", {})
        block_modifiers = transformer_options.get("block_modifiers", [])
        # Explicit invariant: ADM/class conditioning must match `num_classes`.
        if (y is not None) != (self.num_classes is not None):
            raise ValueError(
                f"UNet forward ADM mismatch: num_classes={self.num_classes} but y_present={y is not None}."
            )
        # Optional debug of context feature dim
        try:
            _logger = get_backend_logger("backend.runtime.unet")
            if _logger.isEnabledFor(10):  # DEBUG
                _cd = getattr(self, 'codex_config', None)
                _ctx = getattr(_cd, 'context_dim', None) if _cd is not None else None
                _logger.debug("UNet.forward: context_dim=%s x=%s t=%s y=%s", _ctx, tuple(x.shape), getattr(timesteps,'shape',None), getattr(y,'shape',None))
        except Exception:
            pass
        hs: List[torch.Tensor] = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
        h = x
        for block_id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", block_id)
            for modifier in block_modifiers:
                h = modifier(h, "before", transformer_options)
            h = module(h, emb, context, transformer_options)
            h = apply_control(h, control, "input")
            for modifier in block_modifiers:
                h = modifier(h, "after", transformer_options)
            if "input_block_patch" in transformer_patches:
                for patch in transformer_patches["input_block_patch"]:
                    h = patch(h, transformer_options)
            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                for patch in transformer_patches["input_block_patch_after_skip"]:
                    h = patch(h, transformer_options)

        transformer_options["block"] = ("middle", 0)
        for modifier in block_modifiers:
            h = modifier(h, "before", transformer_options)
        h = self.middle_block(h, emb, context, transformer_options)
        h = apply_control(h, control, "middle")
        for modifier in block_modifiers:
            h = modifier(h, "after", transformer_options)

        for block_id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", block_id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, "output")
            if "output_block_patch" in transformer_patches:
                for patch in transformer_patches["output_block_patch"]:
                    h, hsp = patch(h, hsp, transformer_options)
            h = torch.cat([h, hsp], dim=1)
            del hsp
            output_shape = hs[-1].shape if hs else None
            for modifier in block_modifiers:
                h = modifier(h, "before", transformer_options)
            h = module(h, emb, context, transformer_options, output_shape)
            for modifier in block_modifiers:
                h = modifier(h, "after", transformer_options)

        transformer_options["block"] = ("last", 0)
        for modifier in block_modifiers:
            h = modifier(h, "before", transformer_options)
        if "group_norm_wrapper" in transformer_options:
            out_norm, out_rest = self.out[0], self.out[1:]
            h = transformer_options["group_norm_wrapper"](out_norm, h, transformer_options)
            h = out_rest(h)
        else:
            h = self.out(h)
        for modifier in block_modifiers:
            h = modifier(h, "after", transformer_options)
        return h


__all__ = ["UNet2DConditionModel", "UNetConfig"]
