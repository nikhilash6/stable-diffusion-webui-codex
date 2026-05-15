"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Netflix VOID engine specification and typed runtime assembly.
Rehydrates the loader-produced Netflix VOID bundle contract into a dedicated engine runtime container, threads
normalized engine options into a small runtime scaffold, and exposes the canonical `vid2vid` method consumed by the
use-case layer.

Symbols (top-level; keep in sync; no ghosts):
- `NetflixVoidEngineRuntime` (dataclass): Loaded Netflix VOID engine runtime container.
- `NetflixVoidEngineSpec` (dataclass): Canonical Netflix VOID engine spec metadata.
- `assemble_netflix_void_runtime` (function): Assemble the loaded Netflix VOID runtime from local inventory.
- `NETFLIX_VOID_SPEC` (constant): Canonical Netflix VOID engine spec instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from apps.backend.core.requests import InferenceEvent, Vid2VidRequest
from apps.backend.runtime.families.netflix_void.loader import prepare_netflix_void_bundle_inputs
from apps.backend.runtime.families.netflix_void.model import NetflixVoidBundleInputs
from apps.backend.runtime.families.netflix_void.runtime import (
    NetflixVoidNativeComponents,
    build_netflix_void_native_components,
    run_netflix_void_vid2vid,
)
from apps.backend.runtime.model_registry.specs import ModelFamily


@dataclass(frozen=True, slots=True)
class NetflixVoidEngineRuntime:
    bundle_inputs: NetflixVoidBundleInputs
    native: NetflixVoidNativeComponents
    device: str
    dtype: str

    def run_vid2vid(self, *, request: Vid2VidRequest) -> Iterator[InferenceEvent]:
        return run_netflix_void_vid2vid(native=self.native, request=request)


@dataclass(frozen=True, slots=True)
class NetflixVoidEngineSpec:
    engine_id: str
    family: ModelFamily
    runtime_note: str


NETFLIX_VOID_SPEC = NetflixVoidEngineSpec(
    engine_id="netflix_void",
    family=ModelFamily.NETFLIX_VOID,
    runtime_note="Netflix VOID native vid2vid runtime",
)


def assemble_netflix_void_runtime(
    *,
    spec: NetflixVoidEngineSpec,
    model_ref: str,
    engine_options: Mapping[str, Any],
) -> NetflixVoidEngineRuntime:
    if spec.family is not ModelFamily.NETFLIX_VOID:
        raise RuntimeError(
            "Netflix VOID runtime assembly requires ModelFamily.NETFLIX_VOID; "
            f"got {spec.family.value!r}."
        )
    bundle_inputs = prepare_netflix_void_bundle_inputs(model_ref)
    native = build_netflix_void_native_components(bundle_inputs=bundle_inputs, engine_options=engine_options)
    return NetflixVoidEngineRuntime(
        bundle_inputs=bundle_inputs,
        native=native,
        device=native.device,
        dtype=native.dtype,
    )
