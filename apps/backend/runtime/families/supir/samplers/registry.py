"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR sampler registry.
Provides a small, deterministic registry of SUPIR sampler IDs, UI labels, and their
canonical native sampler/scheduler tuples.

The sampler execution logic is implemented in the SUPIR runtime (not here). This module exists to:
    - keep id -> label/native tuple mapping centralized,
    - keep native sampler/scheduler derivation centralized,
    - avoid stringly-typed sampler selection and kwargs leakage.

Symbols (top-level; keep in sync; no ghosts):
- `list_supir_samplers` (function): Return all SUPIR sampler specs.
- `resolve_supir_sampler` (function): Resolve one canonical SUPIR sampler ID into a `SupirSamplerSpec` (fail loud).
- `iter_supir_sampler_labels` (function): Yield public sampler labels, optionally excluding dev entries.
"""

from __future__ import annotations

from typing import Iterable

from apps.backend.runtime.families.supir.errors import SupirConfigError

from .types import SupirSamplerId, SupirSamplerSpec


_REGISTRY: tuple[SupirSamplerSpec, ...] = (
    SupirSamplerSpec(
        sampler_id=SupirSamplerId.RESTORE_HEUN_EDM_STABLE,
        label="Restore Heun EDM (Stable)",
        stability="stable",
        supports_tiling=True,
        native_sampler="heun",
        native_scheduler="karras",
    ),
    SupirSamplerSpec(
        sampler_id=SupirSamplerId.RESTORE_EULER_EDM_STABLE,
        label="Restore Euler EDM (Stable)",
        stability="stable",
        supports_tiling=True,
        native_sampler="euler",
        native_scheduler="karras",
    ),
    SupirSamplerSpec(
        sampler_id=SupirSamplerId.RESTORE_DPMPP2M_STABLE,
        label="Restore DPM++ 2M (Stable)",
        stability="stable",
        supports_tiling=True,
        native_sampler="dpm++ 2m",
        native_scheduler="karras",
    ),
    SupirSamplerSpec(
        sampler_id=SupirSamplerId.RESTORE_HEUN_EDM_DEV,
        label="Restore Heun EDM (Dev)",
        stability="dev",
        supports_tiling=True,
        native_sampler="heun",
        native_scheduler="karras",
    ),
    SupirSamplerSpec(
        sampler_id=SupirSamplerId.RESTORE_EULER_EDM_DEV,
        label="Restore Euler EDM (Dev)",
        stability="dev",
        supports_tiling=True,
        native_sampler="euler",
        native_scheduler="karras",
    ),
    SupirSamplerSpec(
        sampler_id=SupirSamplerId.RESTORE_DPMPP2M_DEV,
        label="Restore DPM++ 2M (Dev)",
        stability="dev",
        supports_tiling=True,
        native_sampler="dpm++ 2m",
        native_scheduler="karras",
    ),
)


def list_supir_samplers() -> list[SupirSamplerSpec]:
    return list(_REGISTRY)


def resolve_supir_sampler(value: str, *, include_dev: bool = True) -> SupirSamplerSpec:
    raw = str(value or "").strip()
    if not raw:
        raise SupirConfigError("supir_sampler must be set")

    # Canonical public/runtime selector: enum-backed sampler id.
    try:
        sid = SupirSamplerId(raw)
    except Exception:
        raise SupirConfigError(f"Unknown SUPIR sampler id: {raw!r}") from None

    for spec in _REGISTRY:
        if not include_dev and spec.stability != "stable":
            continue
        if spec.sampler_id is sid:
            return spec
    raise SupirConfigError(f"SUPIR sampler id not registered: {sid.value!r}")


def iter_supir_sampler_labels(*, include_dev: bool = True) -> Iterable[str]:
    for spec in _REGISTRY:
        if not include_dev and spec.stability != "stable":
            continue
        yield spec.label


__all__ = [
    "iter_supir_sampler_labels",
    "list_supir_samplers",
    "resolve_supir_sampler",
]
