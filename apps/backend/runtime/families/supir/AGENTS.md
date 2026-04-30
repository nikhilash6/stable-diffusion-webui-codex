<!-- tags: backend, runtime, supir, restore, sdxl -->

# apps/backend/runtime/families/supir Overview
Date: 2026-02-02
Last Review: 2026-04-30
Status: In progress

## Purpose
- Host SUPIR-specific runtime code for native SDXL img2img/inpaint SUPIR mode.
- Centralize SUPIR guardrails, typed parameter parsing, asset validation, and the dedicated runtime owner so the canonical img2img path can stay thin and fail-loud.

## Key files
- `weights.py` — SUPIR weights discovery/validation under the `supir_models` roots (`apps/paths.json`).
- `sdxl_guard.py` — SDXL base/refiner detection; enforces “reject SDXL Refiner” (SUPIR base must be SDXL base/finetune).
- `config.py` — Typed nested `img2img_extras.supir` config parsing for the tranche-1 public SUPIR surface.
- `nn/` — SUPIR neural network modules (GLVControl + LightGLVUNet + adapters).
- `samplers/` — SUPIR sampler IDs/specs and registry (Enum + dataclasses; no kwargs leakage).
- `loader.py` — Validates the already-selected SDXL checkpoint record and resolves SUPIR variant weights under `supir_models`.
- `runtime.py` — Canonical SUPIR mode execution owner called from `apps/backend/use_cases/img2img.py`.

## Notes
- **No LLaVA** support: prompts are optional; no image-to-text captioning path.
- **No fast VAE encode/decode**: do not implement any “fast VAE” shortcuts.
- Default OFF: VAE tiling and diffusion tiling (guardrails required when enabled).
- Keep router/runtime imports **import-light**: torch-heavy work must be inside functions.
- 2026-03-31: `/api/supir/models` is diagnostics-only; live SUPIR generation is owned by canonical SDXL `img2img.py`, not a standalone `/api/supir/enhance` route/task.
- 2026-04-01: the public/native SUPIR restore surface is currently bounded to `restoreCfgSTmin`; do not expose or accept structure-only / LPF knobs again until `runtime.py` owns a truthful non-placeholder execution path for them.
- 2026-04-02: `runtime.py::_resolve_loaded_sdxl_checkpoint(...)` must resolve the live checkpoint file from `engine._current_bundle.model_ref`; `bundle.source` is the bundle materialization format (`state_dict` / `diffusers`), not the filesystem path of the loaded SDXL base checkpoint.
- 2026-04-02: `runtime.py::_build_supir_runtime_modules(...)` must translate the active SDXL UNet `codex_config.transformer_depth` into GLVControl's per-level contract before constructing `nn/control.py`; the common UNet config stores SDXL depth per block (`[0,0,2,2,10,10]`-style), while `GLVControl` expects one uniform depth per encoder level (`[0,2,10]`-style) and should fail loud if a level is non-uniform.
- 2026-04-02: the official SUPIR variant checkpoint only owns the adapter overlay under `project_modules.*` plus the standalone `control_model.*` seam; `runtime.py::_build_supir_runtime_modules(...)` must load that overlay through the exact submodule owner (`supir_unet.project_modules`) instead of treating it as a full UNet state dict.
