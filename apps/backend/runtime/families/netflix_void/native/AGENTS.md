<!-- tags: backend, runtime, families, netflix_void, native, text-encoder -->
# apps/backend/runtime/families/netflix_void/native Overview
Date: 2026-04-03
Last Review: 2026-04-03
Status: Active

## Purpose
- Own the native Netflix VOID runtime implementation under `apps/**`.
- Keep base-bundle component resolution, tokenizer/text-encoder loading, and future transformer/VAE/scheduler ports inside the family-native seam.
- Preserve canonical keyspace resolution and safe-load contracts without `diffusers` runtime imports.

## Key Files
- `apps/backend/runtime/families/netflix_void/native/__init__.py` — Public export surface for native VOID runtime pieces.
- `apps/backend/runtime/families/netflix_void/native/bundle_io.py` — Base-bundle component dir/config/weights resolution helpers.
- `apps/backend/runtime/families/netflix_void/native/text_encoder.py` — Native T5 tokenizer/text-encoder loader built on `IntegratedT5` + canonical T5 keymap + `safe_load_state_dict(...)`.

## Expectations
- Keep this directory native-only. Do not import official Diffusers CogVideoX runtime/model/pipeline classes here.
- Use family-owned base-bundle resolution only; do not add generic component guessing or a second planner.
- Text loading must keep the canonical T5 keyspace resolver + `safe_load_state_dict(...)` path and must not add prefix strippers, eager copied remap dicts, or raw `module.load_state_dict(...)` shortcuts.
- Future transformer/VAE/scheduler ports should reuse `bundle_io.py` rather than inventing component-local path scanners.
