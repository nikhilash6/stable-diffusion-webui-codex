# apps/backend/runtime/processing Overview
Date: 2025-10-28
Last Review: 2026-03-22
Status: Active

## Purpose
- Shared preprocessing utilities (e.g., image conditioning, mask preparation) used before dispatching to engines.

## Notes
- 2026-03-22: `conditioners.py` now resolves a PyTorch-compatible img2img VAE-posterior seed from the processing object and forwards optional `encode_seed` into `encode_image_batch(...)`; this is the shared deterministic encode seam used by both unmasked and masked img2img prep.
- 2026-03-26: `conditioners.img2img_conditioning(...)` now resolves the runtime UNet channel contract from `diffusion_model.codex_config.in_channels`: plain UNets still get the zero `(B,5,1,1)` tensor, model-class inpaint UNets get real masked `c_concat`, and unsupported channel layouts fail loud. The owner is the runtime channel contract, not checkpoint names or heuristics.
- Centralize preprocessing logic here to avoid duplicating conversions in use cases or engines.
- `CodexProcessingBase` carries per-job smart flags (`smart_offload`, `smart_fallback`, `smart_cache`) so use-cases and engines can honor request-level overrides without consulting globals directly.
- 2026-01-02: Removed token-merging fields from processing dataclasses (feature is no longer supported).
- 2026-01-02: Added standardized file header docstrings to processing primitives (`__init__.py`, `conditioners.py`, `datatypes.py`) (doc-only change; part of rollout).
- 2026-04-06: `CodexProcessingImg2Img` now includes explicit `inpaint_mode` selection (`per_step_blend`, `post_sample_blend`, `fooocus_inpaint`, `brushnet`) for Codex-native masked img2img.
- 2026-02-03: Processing models use `CodexHiresConfig` for hires configuration (renamed).
- 2026-02-08: `datatypes.py` now includes `ErSdeOptions` and `SamplingPlan.er_sde` to carry normalized ER-SDE runtime options through pipeline stages.
- 2026-02-08: `processing.models.RefinerConfig` now uses `swap_at_step` (serialized as `switch_at_step`) to represent swap-pointer semantics; `CodexHiresConfig.update_from_payload` reads nested refiner pointers from `refiner.switch_at_step`.
- 2026-02-10: Batch 4C tightened `datatypes.py` typing contracts with a closed init-image mode selector (`Literal["pixel","latent"]`) and explicit `dict[str, object]` / `Mapping[str, object]` metadata surfaces for generation/video payload dataclasses.
- 2026-02-27: `conditioners.img2img_conditioning(...)` now masks-out the inpaint conditioning image (Forge/A1111 parity; default mask-weight behavior) before VAE encoding, reducing “conditioning sees removed content” drift for masked img2img.
- 2026-04-06: `CodexProcessingImg2Img` now carries `per_step_blend_strength` (`0..1`, default `1.0`) only for the generic `per_step_blend` branch; mode-specific SDXL accessories must not reinterpret that scalar.
- 2026-03-25: `CodexProcessingTxt2Img.swap_model` is now a first-pass stage config (`SwapStageConfig`), while `CodexHiresConfig.swap_model` stays selector-only (`SwapModelConfig`). Do not collapse those two seams back into one type.
- 2026-03-26: `PromptContext` now carries a dedicated request-owned `clip_skip` field instead of a generic prompt-control bag. Prompt-tag runtime controls are gone; prompt text is LoRA-only at the angle-bracket seam.
- 2026-03-31: `CodexProcessingImg2Img` now carries `supir: SupirModeConfig | None` as the single runtime owner for native SDXL img2img/inpaint SUPIR mode; do not mirror that owner back into generic override maps.
- 2026-04-05: `CodexProcessingImg2Img.mask` and `mask_round` are now the only runtime mask owners. `image_mask` / `round_image_mask` no longer exist on the processing model; masked prep must read the canonical fields only.
