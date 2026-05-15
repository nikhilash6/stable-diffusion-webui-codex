# apps/backend/engines/ltx2 Overview
<!-- tags: backend, engines, ltx2, video, gemma3 -->
Date: 2026-03-11
Last Review: 2026-03-26
Status: Active

## Purpose
- Host the native LTX2 engine seam for backend video generation.
- Keep `ltx2` as a thin `BaseVideoEngine` adapter over the loader-produced typed LTX2 bundle contract.

## Key Files
- `apps/backend/engines/ltx2/ltx2.py` — `Ltx2Engine` facade (`engine_id="ltx2"`, `expected_family=ModelFamily.LTX2`, capabilities `TXT2VID` / `IMG2VID`).
- `apps/backend/engines/ltx2/spec.py` — Runtime assembly that rehydrates the typed LTX2 bundle contract, loads the native family runtime, and exposes `run_txt2vid()` / `run_img2vid()` to the canonical use-cases.
- `apps/backend/engines/ltx2/factory.py` — Factory seam returning the loaded `Ltx2EngineRuntime`.
- `apps/backend/engines/ltx2/__init__.py` — Package export surface.

## Notes
- This package now owns the real native runtime handoff: registration, bundle ownership, canonical use-case handoff, and native generation dispatch are all real.
- `txt2vid` / `img2vid` ownership remains canonical in `apps/backend/use_cases/*`.
- Canonical video use-cases now consume the family-local `Ltx2RunResult` contract (`frames + AudioExportAsset + metadata`) and own cleanup of generated temp audio after export paths finish or fail.
- 2026-03-16: `spec.py` now threads an explicit generated-audio export policy from the canonical video use-cases into `run_txt2vid()` / `run_img2vid()` so runtime audio materialization follows saved-output truth instead of blindly writing temp WAVs.
- 2026-03-26: `spec.py` now exposes the explicit LTX `two_stage` lane through the same runtime seam as `one_stage`; profile truth stays in the router/checkpoint metadata, while runtime internals keep the request-scoped generator and stage-2 asset paths out of the public engine contract.
- The current core-streaming tranche stays backend-internal. `spec.py` may pass only the normalized boolean `core_streaming_enabled` into runtime assembly; it must not widen the public engine/result contract with streaming tuning or metadata in this slice.
- Do not route LTX2 through generic engine helpers or runtime key remap shims. The dedicated family runtime must stay native-only under `apps/**`.
