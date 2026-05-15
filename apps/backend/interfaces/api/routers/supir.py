"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR diagnostics API routes.
Exposes only inventory/readiness diagnostics for native SUPIR mode:
- SUPIR weights diagnostics (`GET /api/supir/models`)
- stable-only public sampler rows with backend-owned native sampler/scheduler metadata

Live SUPIR generation is owned by canonical SDXL `img2img` / inpaint and not by a standalone `/api/supir/enhance` route.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for SUPIR diagnostics endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

from apps.backend.infra.config.paths import get_paths_for
from apps.backend.runtime.families.supir.samplers.registry import list_supir_samplers
from apps.backend.runtime.families.supir.weights import SupirVariant, supir_weights_diagnostics


def build_router(
    *,
    codex_root: Path,
    opts_get,
    generation_provenance,
    save_generated_images,
) -> APIRouter:
    del codex_root, opts_get, generation_provenance, save_generated_images
    router = APIRouter()

    @router.get("/api/supir/models")
    async def get_supir_models() -> Dict[str, Any]:
        roots = [Path(path) for path in get_paths_for("supir_models")]
        return {
            "supir_models": supir_weights_diagnostics(roots=roots),
            "variants": [{"key": variant.value, "label": variant.value} for variant in SupirVariant],
            "samplers": [
                {
                    "id": spec.sampler_id.value,
                    "label": spec.label,
                    "stability": spec.stability,
                    "native_sampler": spec.native_sampler,
                    "native_scheduler": spec.native_scheduler,
                }
                for spec in list_supir_samplers()
                if spec.stability == "stable"
            ],
            "note": "Diagnostics-only surface. Native SUPIR mode runs through /api/img2img on SDXL.",
        }

    return router
