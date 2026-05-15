<!-- tags: backend, runtime, families, layout -->

# apps/backend/runtime/families Overview
Date: 2026-01-17
Last Review: 2026-04-03
Status: Active

## Purpose
- Host model-family runtime code (WAN22/Flux/SD/ZImage/Chroma/LTX2/Netflix VOID) under a single `families/` root so `apps/backend/runtime/` stays reserved for generic, cross-family runtime modules (models/loaders, ops, sampling, memory, vision, etc.).

## Structure
- `wan22/`, `flux/`, `sd/`, `zimage/`, `chroma/` — family runtimes (implementation-specific).
- `ltx2/` — native-only LTX2 runtime seam (video + audio + external Gemma3 asset contract) with family-owned model, scheduler, and execution code under `apps/**`.
- `netflix_void/` — native-only Netflix VOID vid2vid scaffold (explicit base bundle + literal Pass 1/Pass 2 overlays; execution still fail-loud until the repo-owned runtime port lands).

## Notes
- Avoid facade-first imports; prefer importing the defining module within each family runtime.
- See the plan: `.sangoi/plans/2026-01-17-backend-runtime-families-layout.md`.
