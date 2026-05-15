# apps/backend/engines/netflix_void Overview
<!-- tags: backend, engines, netflix_void, video, vid2vid -->
Date: 2026-04-03
Last Review: 2026-04-05
Status: Active

## Purpose
- Host the parked `netflix_void` engine seam for the VOID vid2vid family.
- Keep `netflix_void` as an explicit fail-loud `BaseVideoEngine` placeholder until the native runtime is implemented.

## Key Files
- `apps/backend/engines/netflix_void/netflix_void.py` — `NetflixVoidEngine` parked placeholder (`engine_id="netflix_void"`, `tasks=()`, `load()`/`vid2vid()` raise `NotImplementedError`).
- `apps/backend/engines/netflix_void/spec.py` — Typed runtime assembly + runtime container kept for the future native implementation.
- `apps/backend/engines/netflix_void/factory.py` — Factory seam kept for the future native implementation.
- `apps/backend/engines/netflix_void/__init__.py` — Package export surface.

## Notes
- This package is a parked seam, not a fake runnable lane: `registration.register_netflix_void(...)` now raises `NotImplementedError`, and `register_default_engines(...)` must not re-register it until the native runtime is real.
- Base-bundle resolution and literal Pass 1/Pass 2 pairing remain under `apps/backend/runtime/families/netflix_void/**`; keep them as reference/runtime-owner seams, not as proof that the public engine is live.
- `/api/vid2vid` is parked before staging/task creation, so the engine package must remain a stub rather than a semi-live fallback path.
