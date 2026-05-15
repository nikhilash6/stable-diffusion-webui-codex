# apps/backend/engines/util Overview
<!-- tags: backend, engines, util, adapters -->
Date: 2025-10-28
Last Review: 2026-04-29
Status: Active

## Purpose
- Utility modules supporting engine execution (scheduler mapping, attention backend selection, etc.).

## Notes
- Add shared utilities here instead of duplicating helpers inside specific engine packages.
- `adapters.py` now keeps the stage owners split truthfully: top-level `extras.swap_model` becomes `SwapStageConfig` on `CodexProcessingTxt2Img.swap_model`, `extras.hires.swap_model` stays selector-only on `CodexHiresConfig.swap_model`, and `extras.refiner` / `extras.hires.refiner` become typed `RefinerConfig` owners only for the native SDXL refiner seams.
- Generic swap selectors must preserve family-native load selectors instead of flattening them away. Example: Z-Image `zimage_variant` belongs on `SwapModelConfig` for `swap_model` seams and must not be widened into refiner configs.
- `build_txt2img_processing` also wires smart flags from `Txt2ImgRequest` (`smart_offload`, `smart_fallback`, `smart_cache`) into `CodexProcessingTxt2Img` so pipeline stages can make per-job decisions.
- 2025-12-31: `build_img2img_processing` now wires `distilled_cfg_scale`/`image_cfg_scale` from request metadata and propagates smart flags into `CodexProcessingImg2Img` (needed for Flux/Kontext parity with txt2img).
- 2026-01-01: `build_{txt2img,img2img}_processing` now carries `clip_skip` into `processing.metadata` so workflows can treat it as a prompt control (without prompt-tag injection).
- 2026-01-29: `build_img2img_processing` now maps mask/inpaint controls from `Img2ImgRequest` into `CodexProcessingImg2Img` (mask enforcement + blur/invert/full-res/filled-content knobs).
- 2026-01-02: Added standardized file header docstrings to engine util modules (doc-only change; part of rollout).
- 2026-01-06: `schedulers.py` now expects canonical sampler/scheduler strings (lowercase; spaces/`++` preserved); empty/unknown values raise immediately.
- 2026-02-08: `build_{txt2img,img2img}_processing` now copies `extras.er_sde` mappings before storing overrides, preserving request-local ER-SDE options without shared mutable aliasing.
- 2026-02-08: `adapters._build_refiner_config` now uses swap-pointer semantics (`switch_at_step` → `RefinerConfig.swap_at_step`) instead of refiner step-count semantics.
- 2026-02-21: `attention_backend.py` now sources defaults from runtime memory config and applies diffusers SDPA flags from effective attention policy (with explicit warning fallback when flash appears unavailable).
- 2026-04-05: `adapters.build_img2img_processing(...)` now treats `mask` and `mask_round` as the only runtime img2img mask owners. Do not reintroduce `image_mask` / `round_image_mask` mirrors in processing setup or downstream hires/conditioning paths.
- 2026-03-08: `schedulers.apply_sampler_scheduler(...)` now maps the bridge-supported sampler set (`euler`, `euler a`, `heun`, `lms`, `ddim`, `dpm++ 2m`, `dpm++ 2m sde`, `dpm++ 2m sde heun`, `dpm 2`, `dpm 2 ancestral`, `uni-pc`) and rejects unsupported native-only lanes such as `uni-pc bh2` and `dpm++ 2s ancestral` explicitly with bridge-capability errors.
- 2026-03-09: `adapters._build_hires_config(...)` now preserves omitted hires sampler/scheduler overrides as `None` so inheritance is represented by omission instead of legacy `Use same*` sentinels or eager fallback to base sampling fields.
- 2026-04-29: `adapters._build_hires_config(...)` requires enabled hires active fields (`denoise`, `scale`, `resize_x`, `resize_y`, `steps`, `upscaler`) to be present before constructing `CodexHiresConfig`; do not reintroduce active-field default-fill at the adapter boundary.
- 2026-04-06: `build_img2img_processing(...)` now transfers `Img2ImgRequest.inpaint_mode` and the masked per-step scalars into `CodexProcessingImg2Img`; the adapter must not alias removed `mask_enforcement` names downstream.
- 2026-03-31: `build_img2img_processing(...)` now owns the typed SUPIR handoff: it parses only nested `img2img_extras.supir` into `CodexProcessingImg2Img.supir` and must never mirror `supir` into `processing.override_settings`.
