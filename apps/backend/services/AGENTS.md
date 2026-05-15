# apps/backend/services Overview
Date: 2025-10-28
Last Review: 2026-03-31
Status: Active

## Purpose
- Encapsulates high-level backend services that expose functionality to the API: media encoding/decoding, options management, progress broadcasting, etc.

## Key Files
- `media_service.py` — Handles encode/decode operations and format policies.
- `live_preview_service.py` — Builds per-task live preview config from Settings and attaches encoded preview payloads to progress events.
- `options_store.py` — JSON-backed options store (`apps/settings_values.json`) and helpers used by API/runtime.
- `output_service.py` — Output naming + persistence helpers (writes under `CODEX_ROOT/output`).

## Notes
- Services should remain stateless apart from request-scoped state managed in `core/state.py`.
- When introducing new user-facing capabilities, add service wrappers here and expose them via the API schemas in `apps/backend/interfaces/`.
- `__init__.py` is a package marker (no re-exports); import services from their defining modules.
- 2026-01-01: Centralized live preview Settings parsing + SSE payload encoding/attachment in `live_preview_service.py` to keep API workers thin and avoid duplicating image encoding logic.
- 2025-12-29: `options_store.py` resolves `apps/settings_values.json` relative to `CODEX_ROOT` (required) so option reads/writes don’t depend on the process CWD.
- 2026-01-03: Added standardized file header docstrings to `services/*` modules (doc-only change; part of rollout).
- 2026-01-24: `options_store.py` now only exposes the minimal, registry-backed options surface (global defaults + smart flags + memory overrides); unknown keys are pruned from `apps/settings_values.json` at backend startup.
- 2026-01-24: `live_preview_service.py` no longer mutates `os.environ` to apply per-task preview settings; API worker threads use `LivePreviewTaskConfig.runtime_overrides()` (thread-local) around sampling.
- 2026-02-15: `options_store.py` now maintains `codex_options_revision` as persisted source-of-truth for generation contract checks; revision increments on option writes and is surfaced through options snapshots/API.
- 2026-04-05: `options_store.py` now persists one runtime-device option only: `codex_main_device`. Per-component device keys are gone from the active options surface; only per-component dtypes remain alongside it.
- 2026-02-21: `options_store.py` and `live_preview_service.py` now parse boolean options strictly (`true/false/1/0/yes/no/on/off`) and fail loud on invalid persisted values instead of permissive Python truthiness (e.g. `"false"` no longer coerces to `True`).
- 2026-02-21: `live_preview_service.py` now parses `show_progress_every_n_steps` via shared strict integer parsing and raises on malformed/negative values (no silent fallback to `0` that disabled previews implicitly).
- 2026-02-22: Rolled back the temporary global live-preview kill-switch; `live_preview_service.py` again derives runtime interval/method and SSE enablement from persisted preview options while keeping strict fail-loud option parsing.
- 2026-02-22: `live_preview_service.py` now parses `show_progress_type` and `live_previews_image_format` in strict fail-loud mode (invalid persisted enum-like values raise instead of silently defaulting).
- 2026-03-31: `live_preview_service.py` now consumes `core.state.live_preview_snapshot()` as the single preview read seam and only attaches preview payloads for the expected per-run owner token; task workers must not read raw preview fields from `BackendState` directly.
