"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Declarative key-style detection + strict keyspace resolver primitives for checkpoint state_dicts.
Provides fail-loud style detection and canonical keyspace resolution utilities used by model-family loaders to resolve multiple upstream
key layouts into one canonical lookup space without mutating source keys or relying on ad-hoc string-replace chains. Any attempt to
rewrite an incoming layer name outside explicit keyspace mapping now raises immediately; keymaps must map source keys deliberately instead of renaming them through generic preprocessing.

Symbols (top-level; keep in sync; no ghosts):
- `KeyMappingError` (exception): Raised when key-style detection or keyspace resolution fails (unknown style, ambiguous style, collisions).
- `KeyStyleDetectionError` (exception): Raised when key-style detection fails (unknown/ambiguous layout).
- `KeyStyle` (enum): Stable identifiers for common key layouts (Codex, Diffusers, LDM, OpenCLIP, WAN export, llama.cpp GGUF, HF).
- `SentinelKind` (enum): Sentinel matching strategy (exact/prefix/substring/regex).
- `KeySentinel` (dataclass): A single style signal used to detect a key layout.
- `KeyStyleSpec` (dataclass): A named key-style + its sentinel set and matching threshold.
- `KeyStyleDetector` (dataclass): Detects a style from a key list with strict ambiguity handling.
- `ResolvedKeyspace` (dataclass): Canonical resolver envelope (`style`, `canonical_to_source`, `metadata`, `view`).
- `fail_on_key_name_rewrite` (function): Fail-loud guard that rejects any attempt to rewrite an incoming source key by dropping known wrapper/prefix chains.
- `resolve_state_dict_keyspace` (function): Detects style + resolves a canonical keyspace view with strict fail-loud invariants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, Mapping, MutableMapping, Sequence, TypeVar

from apps.backend.runtime.state_dict.views import KeyspaceLookupView

_T = TypeVar("_T")


class KeyMappingError(RuntimeError):
    pass


class KeyStyleDetectionError(KeyMappingError):
    pass


def _raise_layer_name_mutation(*, operation: str, source_key: str, candidate_key: str) -> None:
    raise KeyMappingError(
        "Layer-name mutation is forbidden in this repository. "
        "Keymaps map keyspaces; they do not rewrite stored model keys. "
        f"operation={operation!r} source_key={source_key!r} candidate_key={candidate_key!r}"
    )


def _candidate_after_prefix_chain(source_key: str, prefixes: tuple[str, ...]) -> str:
    candidate_key = source_key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if candidate_key.startswith(prefix):
                candidate_key = candidate_key[len(prefix) :]
                changed = True
                break
    return candidate_key


class KeyStyle(str, Enum):
    CODEX = "codex"
    DIFFUSERS = "diffusers"
    LDM = "ldm"
    OPENCLIP = "openclip"
    WAN_EXPORT = "wan_export"
    LLAMA_GGUF = "llama_gguf"
    HF = "hf"


class SentinelKind(str, Enum):
    EXACT = "exact"
    PREFIX = "prefix"
    SUBSTRING = "substring"
    REGEX = "regex"


@dataclass(frozen=True, slots=True)
class KeySentinel:
    kind: SentinelKind
    pattern: str
    description: str = ""
    _compiled: re.Pattern[str] | None = None

    def __post_init__(self) -> None:
        if self.kind is SentinelKind.REGEX:
            object.__setattr__(self, "_compiled", re.compile(self.pattern))

    def matches(self, key: str, *, keys_set: frozenset[str] | None = None) -> bool:
        if self.kind is SentinelKind.EXACT:
            if keys_set is None:
                return key == self.pattern
            return self.pattern in keys_set
        if self.kind is SentinelKind.PREFIX:
            return key.startswith(self.pattern)
        if self.kind is SentinelKind.SUBSTRING:
            return self.pattern in key
        if self.kind is SentinelKind.REGEX:
            assert self._compiled is not None
            return self._compiled.search(key) is not None
        raise KeyMappingError(f"Unknown sentinel kind: {self.kind!r}")


@dataclass(frozen=True, slots=True)
class KeyStyleSpec:
    style: KeyStyle
    sentinels: tuple[KeySentinel, ...]
    min_sentinel_hits: int = 1


@dataclass(frozen=True, slots=True)
class KeyStyleDetector:
    name: str
    styles: tuple[KeyStyleSpec, ...]

    def detect(self, keys: Sequence[str]) -> KeyStyle:
        if not self.styles:
            raise KeyStyleDetectionError(f"{self.name}: no styles configured")
        if not keys:
            raise KeyStyleDetectionError(f"{self.name}: empty key list; cannot detect key style")

        keys_set: frozenset[str] = frozenset(keys)

        hits: dict[KeyStyle, int] = {}
        for spec in self.styles:
            hit_count = 0
            for sentinel in spec.sentinels:
                if sentinel.kind is SentinelKind.EXACT:
                    if sentinel.matches("", keys_set=keys_set):
                        hit_count += 1
                    continue
                if any(sentinel.matches(k, keys_set=keys_set) for k in keys):
                    hit_count += 1
            hits[spec.style] = hit_count

        matched: list[tuple[KeyStyle, int]] = []
        for spec in self.styles:
            score = hits.get(spec.style, 0)
            if score >= spec.min_sentinel_hits:
                matched.append((spec.style, score))

        preview = ", ".join(sorted(keys)[:10])
        if not matched:
            expected = "; ".join(
                f"{spec.style.value}: [{', '.join(s.pattern for s in spec.sentinels)}]"
                for spec in self.styles
            )
            raise KeyStyleDetectionError(
                f"{self.name}: could not detect key style (no sentinels matched). "
                f"expected one of: {expected}. sample_keys=[{preview}]"
            )

        if len(matched) == 1:
            return matched[0][0]

        max_score = max(score for _, score in matched)
        winners = [style for style, score in matched if score == max_score]
        if len(winners) == 1:
            return winners[0]

        scored = ", ".join(f"{style.value}={score}" for style, score in matched)
        raise KeyStyleDetectionError(
            f"{self.name}: ambiguous key style detection (matched multiple styles: {scored}). "
            f"sample_keys=[{preview}]"
        )


@dataclass(frozen=True, slots=True)
class ResolvedKeyspace(Generic[_T]):
    style: KeyStyle | str
    canonical_to_source: dict[str, str]
    metadata: dict[str, object]
    view: MutableMapping[str, _T]


def fail_on_key_name_rewrite(source_key: str, prefixes: tuple[str, ...]) -> str:
    candidate_key = _candidate_after_prefix_chain(source_key, prefixes)
    if candidate_key != source_key:
        _raise_layer_name_mutation(
            operation="fail_on_key_name_rewrite",
            source_key=source_key,
            candidate_key=candidate_key,
        )
    return source_key


def resolve_state_dict_keyspace(
    state_dict: MutableMapping[str, _T],
    *,
    detector: KeyStyleDetector,
    source_key_guard: Callable[[str], None] | None = None,
    mappers: Mapping[KeyStyle, Callable[[str], str]],
    view_factory: Callable[[MutableMapping[str, _T], dict[str, str]], MutableMapping[str, _T]] | None = None,
    output_validator: Callable[[Sequence[str]], None] | None = None,
    required_canonical_keys: Sequence[str] | None = None,
    forbidden_output_prefixes: Sequence[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ResolvedKeyspace[_T]:
    keys = list(state_dict.keys())
    source_keys = [str(key) for key in keys]
    if source_key_guard is not None:
        for source_key in source_keys:
            source_key_guard(source_key)
    style = detector.detect(source_keys)

    mapper = mappers.get(style)
    if mapper is None:
        raise KeyMappingError(f"{detector.name}: no mapper registered for style={style.value!r}")

    canonical_to_source: dict[str, str] = {}
    for source_key in source_keys:
        destination_key = mapper(source_key)
        previous_source = canonical_to_source.get(destination_key)
        if previous_source is not None and previous_source != source_key:
            raise KeyMappingError(
                f"{detector.name}: multiple source keys map to the same destination key: "
                f"dst={destination_key!r} srcs={previous_source!r},{source_key!r}"
            )
        canonical_to_source[destination_key] = source_key

    canonical_keys = list(canonical_to_source.keys())
    if output_validator is not None:
        output_validator(canonical_keys)

    if required_canonical_keys:
        missing = [key for key in required_canonical_keys if key not in canonical_to_source]
        if missing:
            raise KeyMappingError(
                f"{detector.name}: resolver output is missing required canonical keys. "
                f"missing_sample={missing[:10]}"
            )

    if forbidden_output_prefixes:
        offenders = [
            key
            for key in canonical_keys
            if key.startswith(tuple(str(prefix) for prefix in forbidden_output_prefixes))
        ]
        if offenders:
            raise KeyMappingError(
                f"{detector.name}: resolver produced non-canonical keys with forbidden prefixes. "
                f"offenders_sample={sorted(offenders)[:10]}"
            )

    factory = view_factory or (lambda base, mapping: KeyspaceLookupView(base, mapping))
    resolved_metadata = dict(metadata or {})
    resolved_metadata.setdefault("detector", detector.name)
    resolved_metadata.setdefault("source_keys", len(keys))
    resolved_metadata.setdefault("canonical_keys", len(canonical_keys))

    return ResolvedKeyspace(
        style=style,
        canonical_to_source=canonical_to_source,
        metadata=resolved_metadata,
        view=factory(state_dict, canonical_to_source),
    )


__all__ = [
    "KeyMappingError",
    "KeyStyleDetectionError",
    "KeyStyle",
    "SentinelKind",
    "KeySentinel",
    "KeyStyleSpec",
    "KeyStyleDetector",
    "ResolvedKeyspace",
    "fail_on_key_name_rewrite",
    "resolve_state_dict_keyspace",
]
