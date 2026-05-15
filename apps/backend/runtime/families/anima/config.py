"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima runtime configuration and state-dict inference helpers.
Defines strict dataclasses for Cosmos Predict2 (MiniTrainDiT) and Anima's LLMAdapter, plus best-effort inference
from the canonical Anima transformer lookup keyspace resolved from raw `net.*` checkpoints or an already-canonical view (fail-loud on ambiguity).

Symbols (top-level; keep in sync; no ghosts):
- `CosmosPredict2Config` (dataclass): MiniTrainDiT runtime config used to instantiate the core model.
- `LLMAdapterConfig` (dataclass): Anima LLMAdapter config (token-id adapter applied to text embeddings).
- `AnimaConfig` (dataclass): Combined config (core + adapter).
- `infer_anima_config_from_state_dict` (function): Infer Anima config from a transformer state dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True, slots=True)
class CosmosPredict2Config:
    # Positional embedding limits (latent-space units, not pixels).
    max_img_h: int = 240
    max_img_w: int = 240
    max_frames: int = 128

    # PatchEmbed/unpatchify contract.
    in_channels: int = 16
    out_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    concat_padding_mask: bool = True

    # Transformer dims.
    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    crossattn_emb_channels: int = 1024

    # Positional embedding strategy.
    pos_emb_cls: str = "rope3d"
    pos_emb_learnable: bool = False
    pos_emb_interpolation: str = "crop"
    min_fps: int = 1
    max_fps: int = 30
    rope_h_extrapolation_ratio: float = 4.0
    rope_w_extrapolation_ratio: float = 4.0
    rope_t_extrapolation_ratio: float = 1.0
    rope_enable_fps_modulation: bool = True
    extra_per_block_abs_pos_emb: bool = False
    extra_h_extrapolation_ratio: float = 1.0
    extra_w_extrapolation_ratio: float = 1.0
    extra_t_extrapolation_ratio: float = 1.0

    # AdaLN modulation.
    use_adaln_lora: bool = True
    adaln_lora_dim: int = 256


@dataclass(frozen=True, slots=True)
class LLMAdapterConfig:
    vocab_size: int = 32128
    source_dim: int = 1024
    target_dim: int = 1024
    model_dim: int = 1024
    num_layers: int = 6
    num_heads: int = 16
    use_self_attn: bool = True
    layer_norm: bool = False


@dataclass(frozen=True, slots=True)
class AnimaConfig:
    dit: CosmosPredict2Config
    adapter: LLMAdapterConfig


def _require_2d(state_dict: Mapping[str, torch.Tensor], key: str) -> tuple[int, int]:
    value = state_dict.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise RuntimeError(f"Expected a 2D tensor for key={key!r}; got {type(value).__name__} shape={getattr(value, 'shape', None)}")
    return int(value.shape[0]), int(value.shape[1])


def _require_1d(state_dict: Mapping[str, torch.Tensor], key: str) -> int:
    value = state_dict.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise RuntimeError(f"Expected a 1D tensor for key={key!r}; got {type(value).__name__} shape={getattr(value, 'shape', None)}")
    return int(value.shape[0])


def _max_index(keys: Sequence[str], *, prefix: str) -> int:
    best = -1
    for k in keys:
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):]
        idx_str = rest.split(".", 1)[0]
        if not idx_str.isdigit():
            continue
        best = max(best, int(idx_str))
    return best


def _infer_patch_config(*, x_in_dim: int, final_out: int) -> tuple[int, int, int, int, bool]:
    if x_in_dim <= 0 or final_out <= 0:
        raise RuntimeError(f"Invalid patch inference dims: x_in_dim={x_in_dim} final_out={final_out}")

    candidates: list[tuple[int, int, int, int, int, bool]] = []
    for patch_spatial in (1, 2, 4):
        for patch_temporal in (1, 2, 4):
            patch_area = patch_spatial * patch_spatial * patch_temporal
            if (x_in_dim % patch_area) != 0 or (final_out % patch_area) != 0:
                continue
            out_channels = final_out // patch_area
            latent_plus_mask = x_in_dim // patch_area
            for concat_padding_mask in (False, True):
                latent_channels = latent_plus_mask - (1 if concat_padding_mask else 0)
                if latent_channels <= 0:
                    continue
                if out_channels != latent_channels:
                    continue

                score = 0
                if patch_spatial == 2:
                    score += 10
                if patch_temporal == 1:
                    score += 5
                if concat_padding_mask:
                    score += 2
                if latent_channels == 16:
                    score += 3
                candidates.append((score, latent_channels, out_channels, patch_spatial, patch_temporal, concat_padding_mask))

    if not candidates:
        raise RuntimeError(f"Unable to infer patch config: x_in_dim={x_in_dim} final_out={final_out}")

    candidates.sort(reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == best[0] and candidates[1][1:] != best[1:]:
        top = ", ".join(
            f"(latent={c[1]} out={c[2]} patch={c[3]}x{c[3]}x{c[4]} mask={c[5]})" for c in candidates[:3]
        )
        raise RuntimeError(f"Ambiguous patch config inference: x_in_dim={x_in_dim} final_out={final_out} candidates={top}")

    _score, latent_channels, out_channels, patch_spatial, patch_temporal, concat_padding_mask = best
    return int(latent_channels), int(out_channels), int(patch_spatial), int(patch_temporal), bool(concat_padding_mask)


def _infer_rope_extrapolation_ratios(*, in_channels: int) -> tuple[float, float, float]:
    if int(in_channels) == 16:
        return 4.0, 4.0, 1.0
    return 1.0, 1.0, 1.0


def _infer_cosmos_predict2_config_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    max_img_h: int,
    max_img_w: int,
    max_frames: int,
) -> CosmosPredict2Config:
    hidden_dim, x_in_dim = _require_2d(state_dict, "x_embedder.proj.1.weight")
    final_out, hidden_dim_final = _require_2d(state_dict, "final_layer.linear.weight")
    if hidden_dim_final != hidden_dim:
        raise RuntimeError(
            "Anima core hidden dim mismatch between x_embedder and final_layer. "
            f"x_embedder.hidden_dim={hidden_dim} final_layer.hidden_dim={hidden_dim_final}"
        )

    in_channels, out_channels, patch_spatial, patch_temporal, concat_padding_mask = _infer_patch_config(
        x_in_dim=int(x_in_dim),
        final_out=int(final_out),
    )

    inner_dim, ctx_dim = _require_2d(state_dict, "blocks.0.cross_attn.k_proj.weight")
    _inner_q, q_dim = _require_2d(state_dict, "blocks.0.self_attn.q_proj.weight")
    if q_dim != hidden_dim:
        raise RuntimeError(f"Anima self_attn.q_proj expects query_dim={hidden_dim}, got {q_dim}")

    head_dim = _require_1d(state_dict, "blocks.0.self_attn.q_norm.weight")
    if inner_dim % head_dim != 0:
        raise RuntimeError(f"Invalid attention dims: inner_dim={inner_dim} head_dim={head_dim}")
    num_heads = inner_dim // head_dim

    ff_dim, _ff_in = _require_2d(state_dict, "blocks.0.mlp.layer1.weight")
    if _ff_in != hidden_dim:
        raise RuntimeError(f"Anima block MLP expects in={hidden_dim}, got {_ff_in}")
    mlp_ratio = float(ff_dim) / float(hidden_dim)

    max_block = _max_index(list(state_dict.keys()), prefix="blocks.")
    num_blocks = max_block + 1
    if num_blocks <= 0:
        raise RuntimeError("Unable to infer num_blocks from state dict (no blocks.* keys).")

    use_adaln_lora = "t_embedder.1.linear_1.bias" not in state_dict
    adaln_lora_dim = 256
    if use_adaln_lora:
        try:
            adaln_lora_dim, _ = _require_2d(state_dict, "blocks.0.adaln_modulation_self_attn.1.weight")
        except Exception:
            adaln_lora_dim = 256
    rope_h_extrapolation_ratio, rope_w_extrapolation_ratio, rope_t_extrapolation_ratio = _infer_rope_extrapolation_ratios(
        in_channels=int(in_channels)
    )

    return CosmosPredict2Config(
        max_img_h=int(max_img_h),
        max_img_w=int(max_img_w),
        max_frames=int(max_frames),
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        patch_spatial=int(patch_spatial),
        patch_temporal=int(patch_temporal),
        concat_padding_mask=bool(concat_padding_mask),
        model_channels=int(hidden_dim),
        num_blocks=int(num_blocks),
        num_heads=int(num_heads),
        mlp_ratio=float(mlp_ratio),
        crossattn_emb_channels=int(ctx_dim),
        rope_h_extrapolation_ratio=float(rope_h_extrapolation_ratio),
        rope_w_extrapolation_ratio=float(rope_w_extrapolation_ratio),
        rope_t_extrapolation_ratio=float(rope_t_extrapolation_ratio),
        use_adaln_lora=bool(use_adaln_lora),
        adaln_lora_dim=int(adaln_lora_dim),
    )


def _infer_llm_adapter_config_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> LLMAdapterConfig:
    vocab_size, target_dim = _require_2d(state_dict, "llm_adapter.embed.weight")
    out_proj_out, out_proj_in = _require_2d(state_dict, "llm_adapter.out_proj.weight")
    if out_proj_out != target_dim:
        raise RuntimeError(f"llm_adapter.out_proj weight mismatch: out={out_proj_out} expected target_dim={target_dim}")
    model_dim = out_proj_in

    # Optional in-proj exists only when model_dim != target_dim (otherwise Identity).
    if model_dim != target_dim:
        in_proj_out, in_proj_in = _require_2d(state_dict, "llm_adapter.in_proj.weight")
        if in_proj_in != target_dim or in_proj_out != model_dim:
            raise RuntimeError(
                f"llm_adapter.in_proj weight mismatch: got ({in_proj_out},{in_proj_in}) expected ({model_dim},{target_dim})"
            )

    max_block = _max_index(list(state_dict.keys()), prefix="llm_adapter.blocks.")
    num_layers = max_block + 1
    if num_layers <= 0:
        raise RuntimeError("Unable to infer llm_adapter.num_layers from state dict (no llm_adapter.blocks.* keys).")

    use_self_attn = "llm_adapter.blocks.0.self_attn.q_proj.weight" in state_dict

    q_norm_dim = _require_1d(state_dict, "llm_adapter.blocks.0.cross_attn.q_norm.weight")
    q_proj_out, q_proj_in = _require_2d(state_dict, "llm_adapter.blocks.0.cross_attn.q_proj.weight")
    if q_proj_in != model_dim:
        raise RuntimeError(f"llm_adapter.cross_attn.q_proj expects in={model_dim}, got {q_proj_in}")
    if q_proj_out % q_norm_dim != 0:
        raise RuntimeError(f"Invalid llm_adapter attention dims: inner_dim={q_proj_out} head_dim={q_norm_dim}")
    num_heads = q_proj_out // q_norm_dim

    layer_norm = "llm_adapter.blocks.0.norm_cross_attn.bias" in state_dict

    return LLMAdapterConfig(
        vocab_size=int(vocab_size),
        source_dim=int(target_dim),
        target_dim=int(target_dim),
        model_dim=int(model_dim),
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        use_self_attn=bool(use_self_attn),
        layer_norm=bool(layer_norm),
    )


def infer_anima_config_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    max_img_h: int = 240,
    max_img_w: int = 240,
    max_frames: int = 128,
) -> AnimaConfig:
    dit = _infer_cosmos_predict2_config_from_state_dict(
        state_dict,
        max_img_h=max_img_h,
        max_img_w=max_img_w,
        max_frames=max_frames,
    )
    if "llm_adapter.embed.weight" not in state_dict:
        raise RuntimeError("Anima core checkpoint is missing required llm_adapter weights (expected llm_adapter.embed.weight).")
    adapter = _infer_llm_adapter_config_from_state_dict(state_dict)
    return AnimaConfig(dit=dit, adapter=adapter)
