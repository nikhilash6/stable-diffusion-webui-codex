"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native LTX2 prompt encoding and text-packing helpers.
Encodes Gemma3 prompt pairs directly against the loaded native LTX2 components, packs hidden states with the official
LTX2 masking/normalization rule, and returns connector-ready video/audio prompt embeddings plus attention masks without
importing any LTX2 Diffusers pipeline wrappers.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2EncodedPromptPair` (dataclass): Connector-ready video/audio prompt embeddings plus attention mask.
- `pack_ltx2_text_hidden_states` (function): Masked LTX2 text hidden-state normalization + packing helper.
- `encode_ltx2_prompt_pair` (function): Encode prompt/negative_prompt into connector-ready video/audio prompt embeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True, slots=True)
class Ltx2EncodedPromptPair:
    video_prompt_embeds: torch.Tensor
    audio_prompt_embeds: torch.Tensor
    attention_mask: torch.Tensor


def _resolve_device(native: Any, device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device(str(getattr(native, "device_label", "cpu") or "cpu"))


def _resolve_dtype(native: Any, dtype: torch.dtype | None) -> torch.dtype:
    if dtype is not None:
        return dtype
    native_dtype = getattr(native, "torch_dtype", None)
    if isinstance(native_dtype, torch.dtype):
        return native_dtype
    return torch.float32


def _normalize_prompt_input(value: str | Sequence[str], *, field: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence):
        raise RuntimeError(f"LTX2 {field} must be a string or sequence of strings; got {type(value).__name__}.")
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise RuntimeError(
                f"LTX2 {field}[{index}] must be a string; got {type(item).__name__}."
            )
        normalized.append(item)
    if not normalized:
        raise RuntimeError(f"LTX2 {field} sequence must not be empty.")
    return normalized


def pack_ltx2_text_hidden_states(
    text_hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    device: torch.device | str,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pack and normalize per-layer text hidden states using the official LTX2 masking rule."""
    if text_hidden_states.ndim != 4:
        raise RuntimeError(
            "LTX2 text packing expects hidden states shaped [batch, seq, hidden, layers]; "
            f"got shape={tuple(int(dim) for dim in text_hidden_states.shape)!r}."
        )
    if sequence_lengths.ndim != 1:
        raise RuntimeError(
            "LTX2 text packing expects sequence_lengths shaped [batch]; "
            f"got shape={tuple(int(dim) for dim in sequence_lengths.shape)!r}."
        )

    batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape
    original_dtype = text_hidden_states.dtype
    target_device = torch.device(device)

    token_indices = torch.arange(seq_len, device=target_device).unsqueeze(0)
    sequence_lengths = sequence_lengths.to(device=target_device, dtype=torch.long)
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = seq_len - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise RuntimeError(f"LTX2 padding_side must be 'left' or 'right'; got {padding_side!r}.")
    mask = mask[:, :, None, None]

    masked_text_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
    num_valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
    masked_mean = masked_text_hidden_states.sum(dim=(1, 2), keepdim=True) / (num_valid_positions + eps)

    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

    normalized_hidden_states = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
    normalized_hidden_states = normalized_hidden_states * scale_factor
    normalized_hidden_states = normalized_hidden_states.flatten(2)

    mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
    normalized_hidden_states = normalized_hidden_states.masked_fill(~mask_flat, 0.0)
    return normalized_hidden_states.to(dtype=original_dtype)


@torch.no_grad()
def _encode_single_prompt_batch(
    *,
    native: Any,
    prompt: list[str],
    device: torch.device,
    dtype: torch.dtype,
    num_videos_per_prompt: int,
    max_sequence_length: int,
    scale_factor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = getattr(native, "tokenizer", None)
    text_encoder = getattr(native, "text_encoder", None)
    if tokenizer is None or text_encoder is None:
        raise RuntimeError("LTX2 prompt encoding requires loaded tokenizer and text_encoder components.")

    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = [item.strip() for item in prompt]
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=int(max_sequence_length),
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_attention_mask = text_inputs.attention_mask.to(device)

    text_encoder_outputs = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_attention_mask,
        output_hidden_states=True,
    )
    text_hidden_states = torch.stack(text_encoder_outputs.hidden_states, dim=-1)
    sequence_lengths = prompt_attention_mask.sum(dim=-1)

    prompt_embeds = pack_ltx2_text_hidden_states(
        text_hidden_states,
        sequence_lengths,
        device=device,
        padding_side=tokenizer.padding_side,
        scale_factor=scale_factor,
    ).to(dtype=dtype)

    if int(num_videos_per_prompt) > 1:
        prompt_embeds = prompt_embeds.repeat_interleave(int(num_videos_per_prompt), dim=0)
        prompt_attention_mask = prompt_attention_mask.repeat_interleave(int(num_videos_per_prompt), dim=0)

    return prompt_embeds, prompt_attention_mask


@torch.no_grad()
def encode_ltx2_prompt_pair(
    *,
    native: Any,
    prompt: str | Sequence[str],
    negative_prompt: str | Sequence[str] | None = None,
    guidance_scale: float = 4.0,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 1024,
    scale_factor: int = 8,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Ltx2EncodedPromptPair:
    """Encode prompt/negative_prompt into connector-ready video/audio prompt embeddings + mask."""
    resolved_device = _resolve_device(native, device)
    resolved_dtype = _resolve_dtype(native, dtype)
    do_classifier_free_guidance = float(guidance_scale) > 1.0

    prompt_batch = _normalize_prompt_input(prompt, field="prompt")
    batch_size = len(prompt_batch)

    prompt_embeds, prompt_attention_mask = _encode_single_prompt_batch(
        native=native,
        prompt=prompt_batch,
        device=resolved_device,
        dtype=resolved_dtype,
        num_videos_per_prompt=int(num_videos_per_prompt),
        max_sequence_length=int(max_sequence_length),
        scale_factor=int(scale_factor),
    )

    if do_classifier_free_guidance:
        if negative_prompt is None:
            negative_prompt_batch = [""] * batch_size
        else:
            negative_prompt_batch = _normalize_prompt_input(negative_prompt, field="negative_prompt")
            if len(negative_prompt_batch) != batch_size:
                raise RuntimeError(
                    "LTX2 negative_prompt batch must match prompt batch size; "
                    f"got prompt_batch={batch_size} negative_prompt_batch={len(negative_prompt_batch)}."
                )

        negative_prompt_embeds, negative_prompt_attention_mask = _encode_single_prompt_batch(
            native=native,
            prompt=negative_prompt_batch,
            device=resolved_device,
            dtype=resolved_dtype,
            num_videos_per_prompt=int(num_videos_per_prompt),
            max_sequence_length=int(max_sequence_length),
            scale_factor=int(scale_factor),
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

    additive_attention_mask = (1 - prompt_attention_mask.to(dtype=prompt_embeds.dtype)) * -1_000_000.0
    connectors = getattr(native, "connectors", None)
    if connectors is None:
        raise RuntimeError("LTX2 prompt encoding requires a loaded connectors component.")
    connector_video_embeds, connector_audio_embeds, connector_attention_mask = connectors(
        prompt_embeds,
        additive_attention_mask,
        additive_mask=True,
    )

    return Ltx2EncodedPromptPair(
        video_prompt_embeds=connector_video_embeds,
        audio_prompt_embeds=connector_audio_embeds,
        attention_mask=connector_attention_mask,
    )


__all__ = [
    "Ltx2EncodedPromptPair",
    "encode_ltx2_prompt_pair",
    "pack_ltx2_text_hidden_states",
]
