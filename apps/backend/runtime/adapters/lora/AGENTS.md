# apps/backend/runtime/adapters/lora Overview
Date: 2025-10-28
Last Review: 2026-03-31
Status: Active

## Purpose
- Implements the Codex-native LoRA pipeline (loading, tensor mapping, application ops, type definitions).

## Key Files
- `loader.py` — Loads LoRA weights safely with metadata checks.
- `mapping.py` — Maps LoRA weights onto target modules.
- `pipeline.py` — High-level orchestration used by engines/patchers.
- `preflight.py` — Shared structural preflight helpers (cheap SafeTensors header fast path + materialized patch-shape validation).
- `selections.py` — Process-wide LoRA selection state used by API endpoints and workflow builders.
- `types.py` — Dataclasses describing LoRA assets.

## Notes
- Keep this pipeline aligned with `apps/backend/patchers/lora_apply.py` and the options service so selections remain consistent.
- Prompt-owned LoRA selections may carry a text-encoder/default `weight` plus an optional `unet_weight`; when `unet_weight` is omitted, the UNet side inherits the text-encoder weight.
- 2026-01-02: Added standardized file header docstrings to `__init__.py`, `mapping.py`, `pipeline.py`, and `types.py` (doc-only change; part of rollout).
- 2026-02-18: `mapping.py` now builds UNet LoRA aliases from runtime state keys plus canonical SDXL checkpoint keymap normalization (`keymap_sdxl_checkpoint`) so SDXL `model.diffusion_model.*` wrappers and runtime-native key layouts resolve without custom fallback translation paths.
- 2026-03-07: `loader.py` now parses WAN22 modulation DIFF tensors (`*.diff_m`) only when the caller provides an explicit modulation logical target (`blocks.N.modulation` / `head.modulation`); do not add runtime remap or compatibility shims outside the caller/keyspace seam.
- 2026-03-07: The generic parser still recognizes `.diff`, `.diff_b`, and `.set_weight`, but not `.diff_m`. Any family-specific stage path that needs modulation tensors must either add explicit parser support with proven target semantics or classify `.diff_m` as unsupported instead of silently dropping it.
- 2026-03-31: `preflight.py` is the shared structural preflight seam for repo-owned file-based LoRA application. Use the cheap header path only for standard LoRA/DIFF/SET layouts that it can prove honestly; otherwise fall back to materialized patch parsing before mutation instead of inventing a second parser or weakening the fail-loud contract.
