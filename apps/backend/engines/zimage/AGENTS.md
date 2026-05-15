# apps/backend/engines/zimage
Date: 2025-12-12
Last Review: 2026-03-22
Status: Active

## Purpose
- Engine wiring for **Z Image** (Turbo/Base variants; ModelFamily `zimage`): loading core-only checkpoints, binding external Qwen3 text encoders + Flow16 VAEs, and exposing txt2img/img2img execution via the shared pipeline runner.

## Key Files
- `apps/backend/engines/zimage/spec.py` — Runtime assembly (external VAE/Qwen3 for core-only checkpoints) + flow predictor defaults.
- `apps/backend/engines/zimage/factory.py` — Factory seam returning `(runtime, CodexObjects)` for consistent engine assembly.
- `apps/backend/engines/zimage/zimage.py` — `ZImageEngine` implementation (prompt formatting, conditioning, VAE encode/decode semantics).
- `apps/backend/engines/zimage/__init__.py` — Package marker (no re-exports); import from `spec.py` / `zimage.py`.

## References (vendored assets)
- `apps/backend/huggingface/Tongyi-MAI/Z-Image-Turbo/scheduler/scheduler_config.json` — Turbo canonical `shift` + `num_train_timesteps` (`shift=3.0`).
- `apps/backend/huggingface/Tongyi-MAI/Z-Image/scheduler/scheduler_config.json` — Base canonical `shift` + `num_train_timesteps` (`shift=6.0`).
- `apps/backend/huggingface/Tongyi-MAI/Z-Image-Turbo/vae/config.json` — canonical `scaling_factor` + `shift_factor`.

## Notes / Decisions
- 2026-03-22: `ZImageEngine.encode_first_stage(...)` still uses the shared Flow16 first-stage lane, but now forwards optional `encode_seed` so img2img init-latent posterior sampling stays deterministic without a ZImage-only encode fork.
- 2026-03-20: `ZImageEngine` now treats `extras.zimage_variant` as the authoritative selector for Base vs Turbo; GGUF metadata is only a fallback when the request omitted the variant, and Z Image img2img must not inherit SD-family inpaint heuristics from checkpoint metadata.
- **Variant contract:** UI sends `extras.zimage_variant="turbo"|"base"` for the base request, and generic swap seams may also carry the same selector on `extras.swap_model` / `extras.hires.swap_model`; the backend forwards it to `engine_options["zimage_variant"]` so the orchestrator reloads the engine when the variant changes.
  - For Codex-produced GGUFs, the engine may also trust `codex.zimage.variant` metadata when it matches Codex provenance.
- **CFG semantics (diffusers parity):** Z-Image uses classic CFG; unconditional conditioning is used when `guidance_scale > 1` and negative prompts are supported for both variants.
- **VAE normalization:** decode must apply `vae.first_stage_model.process_out(latents)` before `vae.decode(...)` (Flux/Z-Image latent format).
- **Prompt wrappers:** `ZImageEngine._prepare_prompt_wrappers(...)` attaches `cfg_scale` to the prompt list from `processing.guidance_scale` so the UI “guidance” slider can be propagated into conditioning logs (and any future guidance embedding usage).
- **Diffusers-math sampler:** `standalone_sampler.sample_zimage_diffusers_math(...)` mirrors diffusers scheduler behavior (`shift=3.0`, `sigma_min=0.0`) and avoids double-negating the model output (core already returns `noise_pred=-v`).
- **Debugging:** set `CODEX_ZIMAGE_DEBUG_PROMPT=1` to log the formatted prompt string and `cfg_scale` used for the run.
- 2026-01-01: ZImage prompt conditioning now participates in `smart_cache` (`zimage.conditioning`) so repeated prompts don’t re-encode Qwen3 each time; `get_learned_conditioning(...)` returns the cross-attn tensor directly (no placeholder `vector/guidance` allocations).
- 2026-01-02: Added standardized file header docstrings to Z Image engine modules (doc-only change; part of rollout).
- 2026-01-03: Z Image runtime core is now stored as `ZImageEngineRuntime.denoiser` via `DenoiserPatcher` (no ControlNet graph).
- 2026-01-03: `ZImageEngine` now assembles via `CodexZImageFactory` (factory-first seam; reduces drift in `_build_components`).
- 2026-01-18: Z Image treats `vae_path`/`tenc_path` as **external asset selection** (not state-dict overrides) and the API requires sha-based selection (`vae_sha`/`tenc_sha`) for Z Image runs (no silent fallbacks).
- 2026-01-06: Refreshed `spec.py` header block wording to reflect optional external overrides for full checkpoints (doc-only change).
- 2026-01-08: `spec.flow_shift` now resolves from the vendored diffusers `scheduler_config.json` (HF mirror) instead of using family defaults, keeping scheduler parity as the source of truth.
- 2026-01-20: Removed unused dev-only ZImage artifacts (`diffusers_pipeline.py`, `test_diffusers.py`) — engine wiring lives in `spec.py` / `factory.py` / `zimage.py`.
- 2026-01-30: Removed dev-only Diffusers bypass flag (`CODEX_ZIMAGE_DIFFUSERS_BYPASS`) and downgraded Z-Image assembly/sampler logs to debug (default runs are quiet). External VAE/TEnc loading now follows memory-manager role dtypes (options `codex_vae_dtype` / `codex_te_dtype`) instead of a Z-Image-specific `dtype` override path.
- 2026-01-31: `ZImageEngine.encode_first_stage/decode_first_stage` now delegate to the shared `CodexDiffusionEngine` VAE stage implementation (timeline spans preserved; no behavior change).
- 2026-01-31: ZImage conditioning caching now uses the shared engine cache helpers (hit/miss metrics + CPU storage + explicit restore to device/dtype). `clip_skip` is handled as a default no-op by the base engine (Z-Image does not use CLIP).
- 2026-02-09: ZImage conditioning entrypoints now use `torch.no_grad()` (not `torch.inference_mode()`) to avoid caching inference tensors across requests (version-counter faults).
- 2026-02-11: `CodexZImageFactory` now wires `text_encoders[\"qwen3\"]` to the dedicated Qwen patcher wrapper (`runtime.qwen`) instead of the non-loadable text-processing engine object, so smart-offload TE staging can call memory-manager load/unload without resolver failures.
- 2026-02-18: sampling-path latent decode in `zimage.py` now loads/unloads VAE with the base canonical target (`self._vae_memory_target()`), avoiding wrapper-vs-patcher identity drift against shared engine unload cleanup.
- 2026-02-23: Z-Image runtime metadata defaults no longer hardcode backend device literals in engine/spec surfaces; runtime device fallback now resolves from memory-manager mount-device authority.
- 2026-02-23: `standalone_sampler.py::sample_zimage_diffusers_math(...)` no longer defaults `device=\"cuda\"`; unresolved device now resolves through memory-manager mount-device authority.
- 2026-03-06: `ZImageEngine` now pins `expected_family=ModelFamily.ZIMAGE` so expected-family loads bypass stale generic detector assumptions and flow through the vendored-HF signature + family-scoped GGUF keyspace path directly.
