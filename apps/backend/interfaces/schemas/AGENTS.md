<!-- tags: backend, schemas, settings -->
# apps/backend/interfaces/schemas Overview
Date: 2026-01-24
Last Review: 2026-01-24
Status: Active

## Purpose
- Host backend-owned JSON schemas and generated registries that define stable UI/API contracts.

## Key Files
- `apps/backend/interfaces/schemas/settings_schema.json` — source-of-truth settings schema (categories/sections/fields) served to the WebUI.
- `apps/backend/interfaces/schemas/settings_registry.py` — generated Python registry used by the backend to serve schema + validate/prune persisted option values.

## Notes
- Do not edit `settings_registry.py` by hand; regenerate it after changing `settings_schema.json`:
  - `CODEX_ROOT=$PWD PYTHONPATH=$PWD python .sangoi/dev/tools/settings/generate_settings_registry.py`
- 2026-02-20: `codex_attention_backend` is canonicalized to `pytorch|xformers|split|quad`; legacy aliases (`torch-sdpa`, `sage`) are removed from the active contract.
- 2026-04-05: the settings schema/registry now exposes `codex_main_device` as the single runtime-device setting. Do not keep parallel `codex_core_device` / `codex_te_device` / `codex_vae_device` fields in the generated contract.
