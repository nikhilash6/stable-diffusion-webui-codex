"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Fixed-default config for enabling LTX2 transformer-core streaming.
Parses only the existing boolean streaming surface for the first LTX2 streaming tranche. Tuning keys intentionally fail
loud here until the orchestrator fingerprint contract is widened to include them.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2StreamingConfig` (class): Minimal config for enabling LTX2 transformer-core streaming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Ltx2StreamingConfig:
    enabled: bool = False
    policy: str = "naive"
    blocks_per_segment: int = 1
    window_size: int = 1

    @classmethod
    def from_options(cls, options: Mapping[str, object] | None) -> "Ltx2StreamingConfig":
        if options is None:
            return cls()
        if not isinstance(options, Mapping):
            raise RuntimeError(f"LTX2 streaming options must be a mapping, got {type(options).__name__}.")
        if "codex_core_streaming" in options:
            raise RuntimeError(
                "LTX2 streaming assembly expects the normalized internal key `core_streaming_enabled`. "
                "Legacy `codex_core_streaming` should have been normalized before runtime assembly."
            )

        unsupported = [
            key
            for key in (
                "core_streaming_policy",
                "core_streaming_blocks_per_segment",
                "core_streaming_window_size",
                "core_streaming_auto_threshold_mb",
            )
            if key in options and options.get(key) is not None
        ]
        if unsupported:
            raise RuntimeError(
                "LTX2 core streaming tuning keys are not exposed in tranche 1 because the engine reload "
                f"fingerprint only normalizes the boolean flag today. Unsupported keys: {unsupported!r}."
            )

        raw = options.get("core_streaming_enabled")
        if raw is None:
            return cls()
        if isinstance(raw, bool):
            return cls(enabled=raw)
        raise RuntimeError(
            "LTX2 streaming option `core_streaming_enabled` must be a normalized boolean at engine assembly time; "
            f"got {type(raw).__name__}."
        )
