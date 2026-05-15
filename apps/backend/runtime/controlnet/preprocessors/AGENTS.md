# AGENT — apps/backend/runtime/controlnet/preprocessors
Date: 2025-10-31
Last Review: 2026-05-11
Status: Active

## Purpose
- Host Codex-native preprocessing model modules for ControlNet (edge detectors, depth estimators, pose extractors, etc.).
- Keep package-level entrypoints absent until a real runtime/API consumer owns a preprocessor surface.
- Implement differentiable and deterministic transformations without importing legacy modules.

## Notes
- Model implementations live under `models/` (HED, PiDiNet, MLSD, lineart_anime, manga_line, LeReS, Zoe, etc.). Public preprocessor entrypoints and registration wiring are still pending.
- `__init__.py` is a package marker (no auto-registration or facade exports at import time); consumers should import concrete model modules from their owner paths when a real surface exists.
- Future batches (pose, segmentation, geometry) should add concrete owner modules and update the parity matrix (`.sangoi/backend/runtime/controlnet-parity.md`).
