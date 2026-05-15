# AGENT — apps/backend/patchers/controlnet/architectures
Date: 2025-10-31
Last Review: 2026-05-11
Status: Active

## Purpose
- Host architecture-specific ControlNet implementations (SD, SDXL, Flux, Chroma, adapters).
- Keep the package root as a marker; callers import concrete implementations from the owning family package.
- Centralize placeholders for unported architectures (e.g., ControlNet Lite, Flux control, Chroma control) with explicit `NotImplementedError`.

## Notes
- `sd/` contains the current production implementations (ControlNet, ControlNetLite placeholder, ControlLora, T2IAdapter).
- `sdxl/` re-exports SD behaviour until SDXL-specific differences are implemented.
- `flux/` and `chroma/` expose explicit factory placeholders; wire them up once runtimes exist.
