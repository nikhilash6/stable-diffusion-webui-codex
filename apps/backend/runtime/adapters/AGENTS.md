# apps/backend/runtime/adapters Overview
Date: 2025-10-28
Last Review: 2026-04-30
Status: Active

## Purpose
- Provides adapter infrastructure (LoRA, SafeTensors, future adapter types) applied to models at runtime.

## Key Files
- `base.py` — Common adapter interfaces.

## Subdirectories
- `ip_adapter/` — Canonical IP-Adapter runtime seam for validated assets, reference-image conditioning, slot layout, and request-scoped patch apply/restore.
- `lora/` — Full LoRA pipeline implementation (loader, mapping, ops, type definitions).

## Notes
- Add new adapter families alongside LoRA; keep loader/ops modular so engines can mix and match.
- LoRA mapping now reads `CodexCoreSignature` metadata (via `model_config.core_config`) to align alias resolution with architecture-aware loaders.
- 2026-01-02: Added standardized file header docstrings to `base.py` and `lora/*` modules (doc-only change; part of rollout).
