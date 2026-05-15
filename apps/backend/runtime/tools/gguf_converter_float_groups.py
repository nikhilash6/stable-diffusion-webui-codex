"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Float dtype override groups for the GGUF converter.
Defines stable, profile-scoped “groups” of tensor-name patterns that the UI can expose as simple FP16/FP32 knobs
without requiring users to type regex overrides. Groups match the source/native tensor names emitted by the converter.

Symbols (top-level; keep in sync; no ghosts):
- `FloatDtypeGroup` (dataclass): Named group of tensor-name regex patterns (applies to destination names).
- `float_groups_for_profile_id` (function): Returns the float dtype groups for a given converter profile id.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FloatDtypeGroup:
    id: str
    label: str
    patterns: tuple[str, ...]


_FLOAT_GROUPS: dict[str, tuple[FloatDtypeGroup, ...]] = {
    "flux_transformer": (
        FloatDtypeGroup(
            id="io_weights",
            label="IO weights (context_embedder + time_text_embed linear_2 + norm_out)",
            patterns=(
                r"^context_embedder\.weight$",
                r"^time_text_embed\.(?:timestep_embedder|text_embedder|guidance_embedder)\.linear_2\.weight$",
                r"^norm_out\.linear\.weight$",
            ),
        ),
    ),
    "zimage_transformer": (
        FloatDtypeGroup(
            id="pad_tokens",
            label="Pad tokens (x_pad_token + cap_pad_token)",
            patterns=(r"^(?:x_pad_token|cap_pad_token)$",),
        ),
    ),
    "wan22_transformer": (
        FloatDtypeGroup(
            id="sensitive_weights",
            label="Sensitive weights (patch embed + time/text embed + head)",
            patterns=(
                r"^patch_embedding\.weight$",
                r"^condition_embedder\.time_embedder\.linear_(?:1|2)\.weight$",
                r"^condition_embedder\.time_proj\.weight$",
                r"^condition_embedder\.text_embedder\.linear_(?:1|2)\.weight$",
                r"^proj_out\.weight$",
                # Some source checkpoints already use these normalized WAN tensor names.
                r"^time_embedding\.(?:0|2)\.weight$",
                r"^time_projection\.1\.weight$",
                r"^text_embedding\.(?:0|2)\.weight$",
                r"^head\.head\.weight$",
            ),
        ),
    ),
    # LLM (HF → GGUF mapping; destination keys are GGUF names).
    "llama_hf_to_gguf": (
        FloatDtypeGroup(
            id="embeddings_output",
            label="Embeddings + output head (token_embd.weight + output.weight)",
            patterns=(r"^(?:token_embd|output)\.weight$",),
        ),
    ),
}


def float_groups_for_profile_id(profile_id: str) -> tuple[FloatDtypeGroup, ...]:
    return _FLOAT_GROUPS.get(str(profile_id), ())


__all__ = ["FloatDtypeGroup", "float_groups_for_profile_id"]
