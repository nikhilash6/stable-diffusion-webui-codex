"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native LoRA prompt-tag parser.
Parses valid `<lora:name>`, `<lora:name:weight>`, and `<lora:name:text_encoder_weight:unet_weight>` tags only, strips them from
prompt text, resolves deterministic LoRA selections, fails loud on invalid LoRA-like tags, and leaves every other angle-bracket token
untouched as literal prompt text.

Symbols (top-level; keep in sync; no ghosts):
- `_TAG_RE` (constant): Strict regex matching well-formed LoRA tags with optional text-encoder/UNet weights.
- `_LORA_START_RE` (constant): Case-insensitive detector for the start of a LoRA-like prompt tag.
- `_iter_lora_like_segments` (function): Yield bounded LoRA-like raw segments for fail-loud validation.
- `_normalize_lora_alias` (function): Canonicalize LoRA token aliases for case/slash-insensitive lookup.
- `_build_lora_alias_index` (function): Build alias -> matching-paths index for LoRA resolution (stem/filename/path variants).
- `_resolve_lora_path` (function): Resolve one LoRA tag name and return explicit missing/ambiguity failure reasons.
- `ExtraNetsParseError` (class): Structured fail-loud parse error for invalid LoRA-like prompt tags.
- `_dedupe_lora_selections` (function): De-duplicate LoRA selections by path with last-selection-wins semantics.
- `ParsedExtras` (dataclass): Parsed prompt bundle (cleaned prompt + selected LoRAs).
- `_parse_prompt_for_extras` (function): Internal parser entrypoint that threads one lazy LoRA alias index across prompts.
- `parse_prompt_for_extras` (function): Parse a single prompt, resolving LoRAs via the registry and stripping supported tags.
- `parse_prompts` (function): Parse a list of prompts, returning cleaned prompts and deduplicated LoRA selections.
- `__all__` (constant): Export list for extra-net parsing helpers.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Tuple

from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.inventory.scanners.loras import iter_lora_files
from apps.backend.runtime.adapters.lora.selections import LoraSelection


_TAG_RE = re.compile(
    r"<\s*lora\s*:\s*(?P<name>[^:>\s][^:>]*?)\s*(?::\s*(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?::\s*(?P<unet_weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*)?)?\s*>",
    re.IGNORECASE,
)

_LORA_START_RE = re.compile(r"<\s*lora\s*:", re.IGNORECASE)


def _iter_lora_like_segments(prompt: str):
    text = str(prompt or "")
    cursor = 0
    while True:
        match = _LORA_START_RE.search(text, cursor)
        if match is None:
            break
        start = match.start()
        end = start
        while end < len(text) and text[end] != ">":
            end += 1
        if end < len(text):
            end += 1
            while end < len(text) and text[end] == ">":
                end += 1
        else:
            end = len(text)
        yield start, end, text[start:end]
        cursor = end


def _normalize_lora_alias(value: str) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def _build_lora_alias_index() -> dict[str, list[str]]:
    alias_index: dict[str, list[str]] = defaultdict(list)

    try:
        repo_root = get_repo_root().resolve(strict=False)
    except Exception:
        repo_root = None

    def _add(alias: str, path: str) -> None:
        key = _normalize_lora_alias(alias)
        if not key:
            return
        matches = alias_index[key]
        if path not in matches:
            matches.append(path)

    for full_path in iter_lora_files():
        resolved = Path(full_path).expanduser().resolve(strict=False)
        canonical_path = resolved.as_posix()
        filename = resolved.name
        stem = resolved.stem

        _add(stem, canonical_path)
        _add(filename, canonical_path)
        _add(canonical_path, canonical_path)

        if repo_root is not None:
            try:
                relative = resolved.relative_to(repo_root).as_posix()
            except Exception:
                relative = ""
            if relative:
                _add(relative, canonical_path)

    return dict(alias_index)


def _resolve_lora_path(
    *,
    alias_index: dict[str, list[str]],
    token_name: str,
) -> tuple[str | None, str | None]:
    normalized = _normalize_lora_alias(token_name)
    if not normalized:
        return None, "LoRA tag name is empty."

    matches = alias_index.get(normalized, [])
    if len(matches) == 1:
        return matches[0], None

    if not matches:
        return None, f"LoRA '{token_name}' not found in discovered LoRA inventory."

    preview = ", ".join(os.path.basename(path) for path in matches[:3])
    if len(matches) > 3:
        preview = f"{preview}, ..."
    return (
        None,
        (
            f"LoRA '{token_name}' is ambiguous ({len(matches)} matches: {preview}). "
            "Use a unique alias, relative path, or absolute path."
        ),
    )


class ExtraNetsParseError(ValueError):
    """Fail-loud parse error for invalid LoRA-like prompt tags."""

    def __init__(self, errors: list[str]) -> None:
        normalized = tuple(str(item).strip() for item in errors if str(item).strip())
        self.errors = normalized
        details = "; ".join(normalized) if normalized else "unknown parse error"
        super().__init__(details)


def _dedupe_lora_selections(selections: list[LoraSelection]) -> list[LoraSelection]:
    ordered_paths: list[str] = []
    last_by_path: dict[str, LoraSelection] = {}
    for selection in selections:
        path = str(selection.path or "")
        if not path:
            continue
        if path not in last_by_path:
            ordered_paths.append(path)
        last_by_path[path] = selection
    return [last_by_path[path] for path in ordered_paths]


@dataclass(slots=True)
class ParsedExtras:
    prompt: str
    loras: list[LoraSelection]


def parse_prompt_for_extras(
    prompt: str,
    *,
    _alias_index: dict[str, list[str]] | None = None,
) -> ParsedExtras:
    parsed, _ = _parse_prompt_for_extras(prompt, alias_index=_alias_index)
    return parsed


def _parse_prompt_for_extras(
    prompt: str,
    *,
    alias_index: dict[str, list[str]] | None = None,
) -> tuple[ParsedExtras, dict[str, list[str]] | None]:
    loras: list[LoraSelection] = []
    parse_errors: list[str] = []

    def _record_parse_error(raw_tag: str, message: str) -> None:
        parse_errors.append(f"{raw_tag}: {message}")

    text = str(prompt or "")
    cleaned_parts: list[str] = []
    cursor = 0

    def _consume(raw_tag: str) -> str:
        nonlocal alias_index
        parsed_tag = _TAG_RE.fullmatch(raw_tag)
        if parsed_tag is None:
            _record_parse_error(
                raw_tag,
                "invalid LoRA tag; expected <lora:name>, <lora:name:weight>, or "
                "<lora:name:text_encoder_weight:unet_weight>.",
            )
            return raw_tag

        name = str(parsed_tag.group("name") or "").strip()
        raw_weight = parsed_tag.group("weight")
        raw_unet_weight = parsed_tag.group("unet_weight")
        weight_text = "" if raw_weight is None else str(raw_weight).strip()
        unet_weight_text = "" if raw_unet_weight is None else str(raw_unet_weight).strip()

        if alias_index is None:
            alias_index = _build_lora_alias_index()
        path, resolve_error = _resolve_lora_path(alias_index=alias_index, token_name=name)
        if resolve_error is not None or path is None:
            _record_parse_error(raw_tag, resolve_error or "failed to resolve LoRA path")
            return raw_tag

        text_encoder_weight = 1.0
        if raw_weight is not None:
            try:
                parsed_weight = float(weight_text)
                if not math.isfinite(parsed_weight):
                    raise ValueError("LoRA text-encoder weight must be finite")
                text_encoder_weight = max(0.0, parsed_weight)
            except Exception as exc:
                _record_parse_error(
                    raw_tag,
                    f"invalid LoRA text-encoder weight '{weight_text}': {exc}",
                )
                return raw_tag

        unet_weight = text_encoder_weight
        if raw_unet_weight is not None:
            try:
                parsed_unet_weight = float(unet_weight_text)
                if not math.isfinite(parsed_unet_weight):
                    raise ValueError("LoRA UNet weight must be finite")
                unet_weight = max(0.0, parsed_unet_weight)
            except Exception as exc:
                _record_parse_error(
                    raw_tag,
                    f"invalid LoRA UNet weight '{unet_weight_text}': {exc}",
                )
                return raw_tag

        loras.append(
            LoraSelection(
                path=path,
                weight=text_encoder_weight,
                unet_weight=unet_weight,
                online=False,
            )
        )
        return ""

    for start, end, raw_tag in _iter_lora_like_segments(text):
        cleaned_parts.append(text[cursor:start])
        cleaned_parts.append(_consume(raw_tag))
        cursor = end
    cleaned_parts.append(text[cursor:])
    cleaned = "".join(cleaned_parts)
    if parse_errors:
        raise ExtraNetsParseError(parse_errors)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return ParsedExtras(prompt=cleaned, loras=loras), alias_index


def parse_prompts(prompts: List[str]) -> Tuple[List[str], List[LoraSelection]]:
    cleaned: List[str] = []
    acc: List[LoraSelection] = []
    alias_index: dict[str, list[str]] | None = None
    for prompt in prompts:
        parsed, alias_index = _parse_prompt_for_extras(prompt, alias_index=alias_index)
        cleaned.append(parsed.prompt)
        acc.extend(parsed.loras)
    return cleaned, _dedupe_lora_selections(acc)


__all__ = [
    "parse_prompts",
    "parse_prompt_for_extras",
    "ParsedExtras",
    "ExtraNetsParseError",
]
