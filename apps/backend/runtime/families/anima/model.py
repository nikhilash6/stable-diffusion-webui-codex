"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima core diffusion model (Cosmos Predict2 MiniTrainDiT + LLMAdapter glue).
This module wires the Cosmos Predict2 core DiT with Anima's adapter that consumes T5 token ids, weights, and attention masks.

Symbols (top-level; keep in sync; no ghosts):
- `AnimaDiT` (class): MiniTrainDiT + `llm_adapter` with `preprocess_text_embeds(...)` and sampling-friendly `forward`.
"""

from __future__ import annotations

import torch

from .config import AnimaConfig, CosmosPredict2Config
from .dit import MiniTrainDiT, MiniTrainDiTConfig
from .llm_adapter import LLMAdapter


def _to_dit_config(cfg: CosmosPredict2Config) -> MiniTrainDiTConfig:
    return MiniTrainDiTConfig(
        max_img_h=int(cfg.max_img_h),
        max_img_w=int(cfg.max_img_w),
        max_frames=int(cfg.max_frames),
        in_channels=int(cfg.in_channels),
        out_channels=int(cfg.out_channels),
        patch_spatial=int(cfg.patch_spatial),
        patch_temporal=int(cfg.patch_temporal),
        concat_padding_mask=bool(cfg.concat_padding_mask),
        model_channels=int(cfg.model_channels),
        num_blocks=int(cfg.num_blocks),
        num_heads=int(cfg.num_heads),
        mlp_ratio=float(cfg.mlp_ratio),
        crossattn_emb_channels=int(cfg.crossattn_emb_channels),
        pos_emb_cls=str(cfg.pos_emb_cls),
        pos_emb_learnable=bool(cfg.pos_emb_learnable),
        pos_emb_interpolation=str(cfg.pos_emb_interpolation),
        min_fps=int(cfg.min_fps),
        max_fps=int(cfg.max_fps),
        use_adaln_lora=bool(cfg.use_adaln_lora),
        adaln_lora_dim=int(cfg.adaln_lora_dim),
        rope_h_extrapolation_ratio=float(cfg.rope_h_extrapolation_ratio),
        rope_w_extrapolation_ratio=float(cfg.rope_w_extrapolation_ratio),
        rope_t_extrapolation_ratio=float(cfg.rope_t_extrapolation_ratio),
        extra_per_block_abs_pos_emb=bool(cfg.extra_per_block_abs_pos_emb),
        extra_h_extrapolation_ratio=float(cfg.extra_h_extrapolation_ratio),
        extra_w_extrapolation_ratio=float(cfg.extra_w_extrapolation_ratio),
        extra_t_extrapolation_ratio=float(cfg.extra_t_extrapolation_ratio),
        rope_enable_fps_modulation=bool(cfg.rope_enable_fps_modulation),
    )


def _normalize_t5_ids(ids: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if ids.ndim != 2:
        raise ValueError(f"t5xxl_ids must be 2D (B,S); got shape={tuple(ids.shape)}")
    if ids.shape[0] != batch_size:
        raise ValueError(f"t5xxl_ids batch mismatch: ids.B={int(ids.shape[0])} expected {batch_size}")
    return ids


def _normalize_t5_weights(weights: torch.Tensor, *, batch_size: int, seq_len: int) -> torch.Tensor:
    if weights.ndim != 2:
        raise ValueError(f"t5xxl_weights must be 2D (B,S); got shape={tuple(weights.shape)}")
    if weights.shape[0] != batch_size:
        raise ValueError(f"t5xxl_weights batch mismatch: weights.B={int(weights.shape[0])} expected {batch_size}")
    if weights.shape[1] != seq_len:
        raise ValueError(f"t5xxl_weights seq mismatch: weights.S={int(weights.shape[1])} expected {seq_len}")
    return weights


def _normalize_t5_attention_mask(mask: torch.Tensor, *, batch_size: int, seq_len: int) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"t5xxl_attention_mask must be 2D (B,S); got shape={tuple(mask.shape)}")
    if mask.shape[0] != batch_size:
        raise ValueError(f"t5xxl_attention_mask batch mismatch: mask.B={int(mask.shape[0])} expected {batch_size}")
    if mask.shape[1] != seq_len:
        raise ValueError(f"t5xxl_attention_mask seq mismatch: mask.S={int(mask.shape[1])} expected {seq_len}")
    return mask


def _has_complete_t5_conditioning_trio(
    *,
    t5xxl_ids: torch.Tensor | None,
    t5xxl_weights: torch.Tensor | None,
    t5xxl_attention_mask: torch.Tensor | None,
) -> bool:
    fields = {
        "t5xxl_ids": t5xxl_ids is not None,
        "t5xxl_weights": t5xxl_weights is not None,
        "t5xxl_attention_mask": t5xxl_attention_mask is not None,
    }
    if not any(fields.values()):
        return False
    missing = [name for name, is_present in fields.items() if not is_present]
    if missing:
        raise ValueError(
            "Anima T5 conditioning requires `t5xxl_ids`, `t5xxl_weights`, and "
            f"`t5xxl_attention_mask` together; missing={missing}."
        )
    return True


class AnimaDiT(MiniTrainDiT):
    def __init__(
        self,
        *,
        config: AnimaConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(config=_to_dit_config(config.dit), device=device, dtype=dtype)
        self.anima_config = config
        self.llm_adapter = LLMAdapter(config=config.adapter, device=device, dtype=dtype)
        self._cross_attn_cache_key: tuple[object, ...] | None = None
        self._cross_attn_cache_value: torch.Tensor | None = None

    def preprocess_text_embeds(
        self,
        text_embeds: torch.Tensor,
        text_ids: torch.Tensor | None,
        *,
        text_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if text_ids is None:
            if text_attention_mask is not None:
                raise ValueError("Anima text preprocessing received `text_attention_mask` without `text_ids`.")
            return text_embeds
        if text_attention_mask is None:
            raise ValueError("Anima text preprocessing requires `text_attention_mask` when `text_ids` is present.")
        ids = _normalize_t5_ids(text_ids, batch_size=int(text_embeds.shape[0]))
        normalized_mask = _normalize_t5_attention_mask(
            text_attention_mask,
            batch_size=int(text_embeds.shape[0]),
            seq_len=int(ids.shape[1]),
        )
        return self.llm_adapter(
            text_embeds,
            ids,
            target_attention_mask=normalized_mask,
        )

    @staticmethod
    def prepare_cross_attn(
        cross_attn: torch.Tensor,
        *,
        diffusion_model: "AnimaDiT",
        t5xxl_ids: torch.Tensor | None,
        t5xxl_weights: torch.Tensor | None,
        t5xxl_attention_mask: torch.Tensor | None,
        min_seq_len: int = 512,
    ) -> torch.Tensor:
        def _tensor_identity(t: torch.Tensor | None) -> tuple[object, ...] | None:
            if t is None:
                return None
            return (
                int(t.data_ptr()),
                tuple(int(dim) for dim in t.shape),
                str(t.dtype),
                str(t.device),
                int(getattr(t, "_version", 0)),
            )

        if not _has_complete_t5_conditioning_trio(
            t5xxl_ids=t5xxl_ids,
            t5xxl_weights=t5xxl_weights,
            t5xxl_attention_mask=t5xxl_attention_mask,
        ):
            diffusion_model._cross_attn_cache_key = None
            diffusion_model._cross_attn_cache_value = None
            return cross_attn

        ids = _normalize_t5_ids(t5xxl_ids, batch_size=int(cross_attn.shape[0]))
        weights = _normalize_t5_weights(t5xxl_weights, batch_size=int(cross_attn.shape[0]), seq_len=int(ids.shape[1]))
        attention_mask = _normalize_t5_attention_mask(
            t5xxl_attention_mask,
            batch_size=int(cross_attn.shape[0]),
            seq_len=int(ids.shape[1]),
        )
        cache_key = (
            _tensor_identity(cross_attn),
            _tensor_identity(ids),
            _tensor_identity(weights),
            _tensor_identity(attention_mask),
            int(min_seq_len),
        )
        if diffusion_model._cross_attn_cache_key == cache_key and isinstance(diffusion_model._cross_attn_cache_value, torch.Tensor):
            return diffusion_model._cross_attn_cache_value

        out = diffusion_model.preprocess_text_embeds(
            cross_attn,
            ids,
            text_attention_mask=attention_mask,
        )
        out = out * weights.unsqueeze(-1).to(dtype=out.dtype, device=out.device)

        if out.shape[1] < int(min_seq_len):
            pad = int(min_seq_len) - int(out.shape[1])
            out = torch.nn.functional.pad(out, (0, 0, 0, pad))
        diffusion_model._cross_attn_cache_key = cache_key
        diffusion_model._cross_attn_cache_value = out
        return out

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        context: torch.Tensor,
        t5xxl_ids: torch.Tensor | None = None,
        t5xxl_weights: torch.Tensor | None = None,
        t5xxl_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Important: conditioning should ideally be preprocessed once per run (engine/use-case),
        # but we accept ids/weights here to keep the sampling adapter surface explicit.
        context = self.prepare_cross_attn(
            context,
            diffusion_model=self,
            t5xxl_ids=t5xxl_ids,
            t5xxl_weights=t5xxl_weights,
            t5xxl_attention_mask=t5xxl_attention_mask,
        )
        return super().forward(x, timesteps, context=context, **kwargs)
