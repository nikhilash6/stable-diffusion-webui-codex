# apps/backend/video/interpolation Overview
Date: 2025-10-28
Last Review: 2026-02-16
Status: Active

## Purpose
- Provides video frame interpolation helpers (currently RIFE) shared by txt2vid/img2vid pipelines.

## Key Files
- `rife.py` — In-repo RIFE adapter via `ccvfi`; resolves deterministic model paths and raises explicit errors when runtime/model assets are missing.

## Notes
- Keep interpolation helpers lightweight and stateless. Interpolation runtime errors must fail loud (no silent fallback rewriting).
- `rife47.pth` token resolves to repo-local runtime storage (`.uv/xdg-data/rife/rife47.pth`).
- 2026-02-16: `rife.py` now performs a one-shot runtime auto-provision attempt for the default checkpoint only (`rife47.pth`/default relative token with no `CODEX_RIFE_MODEL_PATH` override). Custom explicit model paths and env overrides still fail loud immediately.
- 2026-02-13: `rife.py` now applies explicit runtime cleanup (`unload` hook + CUDA cache cleanup + GC) in a `finally` path so lifecycle is deterministic on both success and failure.
- 2026-01-03: Added standardized file header docstrings to interpolation modules (doc-only change; part of rollout).
