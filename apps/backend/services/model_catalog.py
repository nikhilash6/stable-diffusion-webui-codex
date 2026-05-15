"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Unified model-catalog invalidation + freshness authority.
Owns backend-side cache invalidation across paths config, model checkpoint registry, and inventory scans so
`/api/models` and `/api/models/inventory` share one freshness revision source, including process-local
family-specific discovery caches derived from current asset roots and refresh-coupled runtime asset caches such as IP-Adapter bundles.

Symbols (top-level; keep in sync; no ghosts):
- `current_models_revision` (function): Returns the current model-catalog revision (monotonic, process-local).
- `invalidate_model_catalog` (function): Invalidates paths/registry/inventory caches and bumps model-catalog revision.
- `refresh_model_catalog` (function): Forces registry + inventory refresh from current roots and bumps model-catalog revision.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import threading
from typing import Dict, List

_LOCK = threading.RLock()
_MODELS_REVISION = 0
_LOG = get_backend_logger("backend.model_catalog")


def _normalize_reason(reason: object) -> str:
    normalized = str(reason or "").strip()
    if normalized:
        return normalized
    return "unspecified"


def _next_revision_locked() -> int:
    global _MODELS_REVISION
    _MODELS_REVISION += 1
    return _MODELS_REVISION


def current_models_revision() -> int:
    with _LOCK:
        return int(_MODELS_REVISION)


def invalidate_model_catalog(*, reason: str) -> int:
    reason_text = _normalize_reason(reason)
    from apps.backend.infra.config.paths import invalidate_paths_cache
    from apps.backend.inventory import cache as inventory_cache
    from apps.backend.runtime.adapters.ip_adapter.assets import invalidate_ip_adapter_asset_cache
    from apps.backend.runtime.model_registry.ltx2_execution import invalidate_ltx2_execution_caches
    from apps.backend.runtime.models import api as model_api

    with _LOCK:
        invalidate_paths_cache()
        invalidate_ltx2_execution_caches()
        invalidate_ip_adapter_asset_cache()
        inventory_cache.invalidate()
        model_api.invalidate()
        revision = _next_revision_locked()

    _LOG.info("model-catalog: invalidated revision=%d reason=%s", revision, reason_text)
    return revision


def refresh_model_catalog(*, reason: str) -> tuple[int, Dict[str, List[Dict[str, str]]]]:
    reason_text = _normalize_reason(reason)
    from apps.backend.infra.config.paths import invalidate_paths_cache
    from apps.backend.inventory import cache as inventory_cache
    from apps.backend.runtime.adapters.ip_adapter.assets import invalidate_ip_adapter_asset_cache
    from apps.backend.runtime.model_registry.ltx2_execution import invalidate_ltx2_execution_caches
    from apps.backend.runtime.models import api as model_api

    with _LOCK:
        invalidate_paths_cache()
        invalidate_ltx2_execution_caches()
        invalidate_ip_adapter_asset_cache()
        model_api.refresh()
        checkpoint_count = len(model_api.list_checkpoints(refresh=False))
        inventory = inventory_cache.refresh()
        revision = _next_revision_locked()

    _LOG.info(
        "model-catalog: refreshed revision=%d reason=%s checkpoints=%d vaes=%d text_encoders=%d loras=%d wan22.gguf=%d metadata=%d",
        revision,
        reason_text,
        checkpoint_count,
        len(inventory.get("vaes", [])),
        len(inventory.get("text_encoders", [])),
        len(inventory.get("loras", [])),
        len(inventory.get("wan22", [])),
        len(inventory.get("metadata", [])),
    )
    return revision, inventory


__all__ = [
    "current_models_revision",
    "invalidate_model_catalog",
    "refresh_model_catalog",
]
