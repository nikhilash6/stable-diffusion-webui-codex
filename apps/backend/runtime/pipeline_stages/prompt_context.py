"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Prompt parsing and LoRA/clip-skip context helpers for pipeline processing objects.
Builds a `PromptContext` (cleaned prompts, cleaned negative prompts, LoRA tags, request-owned clip-skip), and applies it onto a processing object.

Symbols (top-level; keep in sync; no ghosts):
- `_resolve_lora_overrides` (function): Normalize `override_settings.lora_path` into deterministic `LoraSelection` entries.
- `_merge_prompt_loras` (function): Merge inherited/request LoRAs with prompt-local LoRA tags, with prompt-local same-path weights winning.
- `build_prompt_context` (function): Parse positive/negative prompts and return a `PromptContext` (LoRAs + request-owned clip skip).
- `build_hires_prompt_context` (function): Parse hires prompt text while inheriting base/request LoRAs unless prompt-local hires tags replace them.
- `apply_prompt_context` (function): Apply a `PromptContext` onto a processing object (prompts + clip skip).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from apps.backend.runtime.adapters.lora.selections import LoraSelection
from apps.backend.runtime.processing.datatypes import PromptContext
from apps.backend.runtime.text_processing.extra_nets import parse_prompts


def _resolve_lora_overrides(processing: Any) -> list[LoraSelection]:
    overrides = getattr(processing, "override_settings", None)
    if not isinstance(overrides, dict):
        return []

    raw_lora_path = overrides.get("lora_path")
    if raw_lora_path is None:
        return []

    resolved_paths: list[str] = []
    if isinstance(raw_lora_path, str):
        normalized = raw_lora_path.strip()
        if normalized:
            resolved_paths.append(Path(normalized).expanduser().resolve(strict=False).as_posix())
    elif isinstance(raw_lora_path, list):
        for index, value in enumerate(raw_lora_path):
            if not isinstance(value, str):
                raise ValueError(
                    "Invalid override_settings.lora_path: expected array of strings "
                    f"(entry {index} was {type(value).__name__})."
                )
            normalized = value.strip()
            if normalized:
                resolved_paths.append(Path(normalized).expanduser().resolve(strict=False).as_posix())
    else:
        raise ValueError(
            "Invalid override_settings.lora_path: expected string or array of strings, "
            f"got {type(raw_lora_path).__name__}."
        )

    return [LoraSelection(path=path, weight=1.0, online=False) for path in resolved_paths]


def _merge_prompt_loras(
    inherited_loras: Sequence[LoraSelection | Any],
    prompt_loras: Sequence[LoraSelection | Any],
) -> list[LoraSelection | Any]:
    ordered_paths: list[str] = []
    last_by_path: dict[str, LoraSelection | Any] = {}
    for selection in list(inherited_loras) + list(prompt_loras):
        path = str(getattr(selection, "path", "") or "")
        if not path:
            continue
        if path not in last_by_path:
            ordered_paths.append(path)
        last_by_path[path] = selection
    return [last_by_path[path] for path in ordered_paths]


def build_prompt_context(processing: Any, prompts: Sequence[str]) -> PromptContext:
    """Parse prompts, negative prompts, and LoRA descriptors."""
    positive_prompts = list(prompts)
    negative_prompt_source = getattr(processing, "negative_prompts", None)
    if negative_prompt_source is None:
        negative_prompt_source = [getattr(processing, "negative_prompt", "")]
    negative_prompts = list(negative_prompt_source)
    cleaned_combined, parsed_loras = parse_prompts(positive_prompts + negative_prompts)
    cleaned_prompts = cleaned_combined[: len(positive_prompts)]
    cleaned_negative_prompts = cleaned_combined[len(positive_prompts) :]
    prompt_loras = list(parsed_loras)
    lora_overrides = _resolve_lora_overrides(processing)
    if lora_overrides:
        seen_paths = {
            str(getattr(selection, "path", "") or "")
            for selection in prompt_loras
            if str(getattr(selection, "path", "") or "")
        }
        for selection in lora_overrides:
            if selection.path in seen_paths:
                continue
            prompt_loras.append(selection)
            seen_paths.add(selection.path)

    clip_skip: int | None = None
    meta = getattr(processing, "metadata", None)
    if isinstance(meta, dict) and meta.get("clip_skip") is not None:
        raw_clip_skip = meta.get("clip_skip")
        try:
            clip_skip = int(raw_clip_skip)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError("Invalid clip_skip in metadata: must be an integer") from exc
        if clip_skip < 0:
            raise ValueError("Invalid clip_skip in metadata: must be >= 0")
    return PromptContext(
        prompts=cleaned_prompts,
        negative_prompts=cleaned_negative_prompts,
        loras=prompt_loras,
        clip_skip=clip_skip,
    )


def build_hires_prompt_context(
    *,
    prompt_seed: str | Sequence[str] | None,
    negative_seed: str | Sequence[str] | None,
    base_context: PromptContext,
) -> PromptContext:
    if isinstance(prompt_seed, str):
        hires_prompts_source = [prompt_seed]
    elif prompt_seed is None:
        hires_prompts_source = []
    else:
        hires_prompts_source = list(prompt_seed)
    if not hires_prompts_source:
        hires_prompts_source = list(base_context.prompts)

    if isinstance(negative_seed, str):
        hires_negative_source = [negative_seed]
    elif negative_seed is None:
        hires_negative_source = []
    else:
        hires_negative_source = list(negative_seed)
    if not hires_negative_source:
        hires_negative_source = list(base_context.negative_prompts)

    cleaned_combined, hires_prompt_loras = parse_prompts(hires_prompts_source + hires_negative_source)
    hires_cleaned_prompts = cleaned_combined[: len(hires_prompts_source)]
    hires_negative_prompts = cleaned_combined[len(hires_prompts_source) :]
    merged_loras = _merge_prompt_loras(base_context.loras, hires_prompt_loras)
    return PromptContext(
        prompts=hires_cleaned_prompts,
        negative_prompts=hires_negative_prompts,
        loras=merged_loras,
        clip_skip=base_context.clip_skip,
    )


def apply_prompt_context(processing: Any, context: PromptContext) -> None:
    """Mutate processing object with normalized prompt data."""
    processing.prompts = context.prompts
    processing.negative_prompts = context.negative_prompts
    processing.cfg_scale = getattr(processing, "guidance_scale", 7.0)

    if context.clip_skip is not None:
        clip_skip = int(context.clip_skip)
        model = getattr(processing, "sd_model", None)
        if model is not None and hasattr(model, "set_clip_skip"):
            model.set_clip_skip(clip_skip)
