"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR control network (GLVControl).
Produces a list of per-block control tensors that the SUPIR UNet consumes.
Keeps the official SUPIR control checkpoint owner shape, including the nested sequential `label_emb` path used by
`num_classes='sequential'`.

This is a SUPIR-specific module and is not a generic ControlNet implementation.

Symbols (top-level; keep in sync; no ghosts):
- `GLVControl` (class): Control network producing a list of control tensors.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn

from apps.backend.runtime.common.nn.unet.layers import ResBlock, SpatialTransformer, TimestepEmbedSequential, TimestepBlock
from apps.backend.runtime.common.nn.unet.utils import timestep_embedding

from .zero import zero_module


class GLVControl(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        attention_resolutions: Sequence[int] = (2, 4),
        num_res_blocks: Sequence[int] | int,
        dropout: float = 0.0,
        channel_mult: Sequence[int] = (1, 2, 4),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: int | str | None = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = 64,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_spatial_transformer: bool = True,
        transformer_depth: Sequence[int] | int = (0, 0, 2, 2, 10, 10),
        context_dim: int | None = 2048,
        disable_self_attentions: Sequence[bool] | None = None,
        num_attention_blocks: Sequence[int] | None = None,
        disable_middle_self_attn: bool = False,
        use_linear_in_transformer: bool = True,
        adm_in_channels: int | None = 2816,
        transformer_depth_middle: int = 10,
        input_upscale: int = 1,
    ):
        super().__init__()

        if not use_spatial_transformer:
            raise NotImplementedError("GLVControl currently requires use_spatial_transformer=True")
        if context_dim is None:
            raise ValueError("GLVControl requires context_dim when use_spatial_transformer=True")
        if num_heads != -1 and num_head_channels != -1:
            # Mirror the UNet invariant: only one should be set.
            raise ValueError("GLVControl: set either num_heads or num_head_channels (not both)")
        if num_heads == -1 and num_head_channels == -1:
            raise ValueError("GLVControl: either num_heads or num_head_channels must be set")

        self.input_upscale = int(input_upscale)

        channel_mult = tuple(int(x) for x in channel_mult)
        if isinstance(num_res_blocks, int):
            num_res_blocks_seq = tuple(int(num_res_blocks) for _ in channel_mult)
        else:
            seq = tuple(int(x) for x in num_res_blocks)
            if len(seq) == 1:
                num_res_blocks_seq = tuple(int(seq[0]) for _ in channel_mult)
            else:
                num_res_blocks_seq = seq
        self.model_channels = int(model_channels)
        self.num_classes = num_classes

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(int(self.num_classes), time_embed_dim)
            elif self.num_classes == "continuous":
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                if adm_in_channels is None:
                    raise ValueError("GLVControl: adm_in_channels is required when num_classes='sequential'")
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        nn.Linear(int(adm_in_channels), time_embed_dim),
                        nn.SiLU(),
                        nn.Linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError(f"GLVControl: unsupported num_classes={self.num_classes!r}")

        # Build input blocks mirroring the Codex UNet structure (encoder path only).
        from apps.backend.runtime.common.nn.unet.utils import conv_nd
        from apps.backend.runtime.common.nn.unet.layers import Downsample

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))])

        # Mirror upstream: transformer_depth is per-level (len == len(channel_mult)) and only applies
        # when the current downsample factor is in attention_resolutions.
        if isinstance(transformer_depth, int):
            transformer_depth_levels = [int(transformer_depth) for _ in channel_mult]
        else:
            transformer_depth_levels = [int(x) for x in transformer_depth]
        if len(transformer_depth_levels) != len(channel_mult):
            raise ValueError(
                "GLVControl.transformer_depth must be an int or a sequence with length == len(channel_mult)"
            )

        ch = model_channels
        ds = 1
        disable_sa_seq = disable_self_attentions
        num_attn_blocks_seq = num_attention_blocks

        for level, mult in enumerate(channel_mult):
            for block_index in range(num_res_blocks_seq[level]):
                layers: list[nn.Module] = [
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
                num_transformers = transformer_depth_levels[level]
                if (ds in set(int(r) for r in attention_resolutions)) and num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                        heads = int(num_heads)
                    else:
                        heads = ch // int(num_head_channels)
                        dim_head = int(num_head_channels)
                    disabled_sa = disable_sa_seq[level] if (disable_sa_seq is not None and level < len(disable_sa_seq)) else False
                    if (num_attn_blocks_seq is None) or (block_index < num_attn_blocks_seq[level]):
                        layers.append(
                            SpatialTransformer(
                                ch,
                                heads,
                                dim_head,
                                depth=int(num_transformers),
                                context_dim=int(context_dim),
                                disable_self_attn=bool(disabled_sa),
                                use_checkpoint=use_checkpoint,
                                use_linear=use_linear_in_transformer,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))

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
                ds *= 2

        # Middle block mirrors UNet middle structure (ResBlock → SpatialTransformer → ResBlock).
        if num_head_channels == -1:
            dim_head = ch // num_heads
            heads = int(num_heads)
        else:
            heads = ch // int(num_head_channels)
            dim_head = int(num_head_channels)

        mid_block: list[nn.Module] = [
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

        if transformer_depth_middle and int(transformer_depth_middle) > 0:
            mid_block.append(
                SpatialTransformer(
                    ch,
                    heads,
                    dim_head,
                    depth=int(transformer_depth_middle),
                    context_dim=int(context_dim),
                    disable_self_attn=bool(disable_middle_self_attn),
                    use_checkpoint=use_checkpoint,
                    use_linear=use_linear_in_transformer,
                )
            )
        mid_block.append(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=None,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        )
        self.middle_block = TimestepEmbedSequential(*mid_block)

        # Hint block: a zero-initialized projection from input latents to model channels.
        self.input_hint_block = TimestepEmbedSequential(
            zero_module(conv_nd(dims, in_channels, model_channels, 3, padding=1))
        )

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, xt: torch.Tensor, context: torch.Tensor | None = None, y: torch.Tensor | None = None, **kwargs) -> List[torch.Tensor]:
        if context is None:
            raise ValueError("GLVControl.forward requires context")
        if (y is not None) != (self.num_classes is not None):
            raise ValueError("GLVControl.forward: y must be provided iff num_classes is not None")

        # Match upstream: cast xt/context/y to x dtype.
        xt = xt.to(x.dtype)
        context = context.to(x.dtype)
        if y is not None:
            y = y.to(x.dtype)

        if self.input_upscale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.input_upscale, mode="bilinear", antialias=True)

        hs: List[torch.Tensor] = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)
        if self.num_classes is not None:
            assert y is not None
            emb = emb + self.label_emb(y)

        guided_hint = self.input_hint_block(x, emb, context)
        guided_hint = guided_hint.to(xt.dtype)

        h = xt
        for module in self.input_blocks:
            h = module(h, emb, context)
            if guided_hint is not None:
                h = h + guided_hint
                guided_hint = None
            hs.append(h)

        h = self.middle_block(h, emb, context)
        hs.append(h)
        return hs


__all__ = ["GLVControl"]
