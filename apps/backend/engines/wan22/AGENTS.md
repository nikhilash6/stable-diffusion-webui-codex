<!-- tags: backend, engines, wan22, gguf -->

# apps/backend/engines/wan22 Overview
Date: 2025-12-06
Last Review: 2026-04-05
Status: Active

## Purpose
- WAN22 engine implementations (`txt2vid`, `img2vid`) that coordinate GGUF-backed runtime execution while keeping `vid2vid` explicitly parked.

## Key Files
- `apps/backend/engines/wan22/wan22_14b.py` — `Wan2214BEngine` (canonical 14B lane for `txt2vid`/`img2vid`; `vid2vid` now fail-loud parked).
- `apps/backend/engines/wan22/wan22_14b_animate.py` — `Wan22Animate14BEngine` (`img2vid` animate lane; engine id `wan22_14b_animate`; `vid2vid` now fail-loud parked).
- `apps/backend/engines/wan22/wan22_5b.py` — `Wan225BEngine` (GGUF-backed wrapper for 5B lane; strict file-only model validation).
- `apps/backend/engines/wan22/wan22_common.py` — shared WAN route/build/asset normalization helpers consumed by WAN22 engines.

## Current Behavior
- All active WAN22 engine lanes are GGUF-only wrappers; `load()` validates GGUF model input and stores runtime metadata without eager pipeline construction.
- `wan22_5b`, `wan22_14b`, and `wan22_14b_animate` now raise `NotImplementedError("wan vid2vid not yet implemented")` on `vid2vid()`; no WAN engine advertises `TaskType.VID2VID`.
- `wan22_14b_animate` keeps registration key `wan22_14b_animate` and exposes only the animate-flavored `img2vid` lane; canonical `vid2vid` orchestration stays parked until a real runtime lands.
- Model-path validation fails loud on empty refs, directories, non-`.gguf` inputs, missing files, and obvious wrong-lane weight labels (5B vs 14B).
- WAN22 stage/settings behavior is payload-driven through use-cases/runtime (`apps/backend/use_cases/*` + `apps/backend/runtime/families/wan22/*`).

## Execution Paths
- GGUF: strict assets via WAN22 runtime/use-case orchestration; model materialization occurs at execution time, not engine `load()`.

## Device/Dtype Policy
- Engine-local `device` override is rejected (`load(device=...)` must be unset/`auto`); mount device comes from memory manager defaults.
