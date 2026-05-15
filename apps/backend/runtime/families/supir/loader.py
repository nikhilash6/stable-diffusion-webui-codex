"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR asset resolution helpers for canonical img2img/inpaint ownership.
Centralizes the validated asset resolution for SUPIR mode:
- consume the already-selected SDXL checkpoint record,
- reject unsupported checkpoint layouts (non-checkpoint/core-only/GGUF/refiner/non-SDXL),
- resolve SUPIR variant weights from `supir_models` roots.

This module does not instantiate runtime modules; it only resolves validated file paths so routers and runtime owners can fail loud
before sampling starts.

Symbols (top-level; keep in sync; no ghosts):
- `SupirResolvedAssets` (dataclass): Validated file paths required for a SUPIR mode run.
- `resolve_supir_assets` (function): Resolve + validate base checkpoint + SUPIR variant weights from canonical img2img selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from apps.backend.runtime.models.types import CheckpointFormat

from .errors import SupirBaseModelError
from .sdxl_guard import require_sdxl_base_checkpoint
from .weights import SupirVariant, resolve_supir_weights


@dataclass(frozen=True)
class SupirResolvedAssets:
    base_checkpoint: Path
    variant_ckpt: Path


def _checkpoint_format_value(record: Any) -> str | None:
    raw = getattr(record, "format", None)
    if isinstance(raw, CheckpointFormat):
        return raw.value
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    value = getattr(raw, "value", None)
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def resolve_supir_assets(*, checkpoint_record: Any, variant: SupirVariant, supir_models_roots: Sequence[Path]) -> SupirResolvedAssets:
    if checkpoint_record is None:
        raise SupirBaseModelError("SUPIR mode requires a resolved SDXL checkpoint record")

    filename = str(getattr(checkpoint_record, "filename", "") or "").strip()
    if not filename:
        raise SupirBaseModelError("Selected checkpoint record is missing filename metadata")
    base_path = Path(filename)

    checkpoint_format = _checkpoint_format_value(checkpoint_record)
    if checkpoint_format != CheckpointFormat.CHECKPOINT.value:
        raise SupirBaseModelError(
            "SUPIR mode requires a full SDXL checkpoint file (.safetensors/.ckpt); "
            f"got format={checkpoint_format!r}"
        )

    if bool(getattr(checkpoint_record, "core_only", False)) or base_path.suffix.lower() == ".gguf":
        raise SupirBaseModelError(
            "SUPIR mode requires a full SDXL checkpoint (.safetensors/.ckpt), not a core-only or GGUF checkpoint"
        )

    require_sdxl_base_checkpoint(base_path)
    weights = resolve_supir_weights(roots=list(supir_models_roots), variant=variant)
    return SupirResolvedAssets(base_checkpoint=base_path, variant_ckpt=weights.ckpt_path)


__all__ = [
    "SupirResolvedAssets",
    "resolve_supir_assets",
]
