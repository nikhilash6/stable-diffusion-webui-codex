# AGENT — apps/backend/patchers/controlnet
Date: 2025-10-31
Last Review: 2026-05-11
Status: Active

## Purpose
- Provide Codex-native ControlNet patchers organised by architecture/family (SD, SDXL, Flux, Chroma).
- Expose clean builders for `ControlNet`, `ControlLora`, `T2IAdapter`, and advanced request helpers without relying on legacy modules.
- Bridge ControlNet modules with the runtime graph API (`ControlNode`, `ControlRequest`, `ControlComposite`).

## Components
- `__init__.py` – public facade exports for active patcher APIs and direct SD-family ControlNet module implementations.
- `base.py` – shared lifecycle helpers for control modules (hint management, weighting context, cloning).
- `weighting.py` – advanced weighting, mask application, and tensor broadcast utilities.
- `apply.py` – user-facing `apply_controlnet_advanced` that builds graph nodes with validation.
- `ops/lora.py` – LoRA-aware operations used by ControlNet LoRA builds.
- `architectures/` – package marker plus architecture family implementations:
  - `architectures/sd/` — production SD/SDXL-compatible modules (`ControlNet`, `ControlLora`, `T2IAdapter`) + explicit placeholders.
  - `architectures/sdxl/` — SDXL facade re-exporting SD behaviour until specialisation lands.
  - `architectures/flux/`, `architectures/chroma/` — explicit placeholders (raise `NotImplementedError`) until ported.

## Notes
- All modules operate exclusivamente em imports `apps.*`; código legado em snapshots de referência é apenas referência.
- Weight schedules and masks validate against runtime transformer options and emit descriptive failures.
- Any new model family integration must document itself here and update `.sangoi/plans/2026-01-27-controlnet-refactor.md`.
