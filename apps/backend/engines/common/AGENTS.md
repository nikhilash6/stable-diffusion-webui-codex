# apps/backend/engines/common Overview
Date: 2025-10-28
Last Review: 2026-03-22
Status: Active

## Purpose
- Shared engine utilities (base classes, mixins, helpers) reused across SD, Flux, Chroma, WAN22 engines.

## Notes
- 2026-03-22: shared `CodexDiffusionEngine.encode_first_stage(...)` now accepts optional `encode_seed` and forwards it into the VAE wrapper; image engines that use the canonical first-stage lane inherit deterministic img2img posterior sampling without custom engine-side logic, and callers that retry from seed must recreate the generator from that same seed.
- 2026-03-20: `base.py` load-path selectors are now request/inventory-authoritative for image runs (`checkpoint_core_only`, `model_format`, `vae_source`); missing selector truth fails loud instead of falling back from suffix/path presence, inpaint is no longer inferred from checkpoint shape/channels, and external SDXL VAE overrides reject wrapper-prefix rewrite attempts while still dropping the known non-weight metadata keys (`model_ema.decay`, `model_ema.num_updates`).
- `CodexDiffusionEngine` now subclasses `BaseInferenceEngine`; implement `_build_components(bundle, *, options)` to assemble runtime objects during `load()`.
- Engines receive pre-materialised `DiffusionModelBundle` instances; avoid invoking legacy loaders inside subclasses.
- Model family flags (`is_sd1`, `is_sd2`, `is_sd3`, `is_sdxl`) remain read-only; call `register_model_family(...)` inside `_build_components` after deriving the runtime.
- Lifecycle hooks: `_on_unload()` lets subclasses clear caches, while `status()` now reports `model_ref`, bundle source, and registered families.
- Tiling/CFG scale toggles remain available but emit structured logs when changed.
- 2026-01-25: SDXL VAE overrides (`vae_path`) now run through the SDXL VAE keymap (LDM→diffusers, no wrapper-prefix rewrite, drop `model_ema.decay` / `model_ema.num_updates`); SDXL override unexpected keys are fatal to prevent silent drift.
- 2026-01-22: `CodexDiffusionEngine.txt2img` now delegates to the canonical mode wrapper `apps/backend/use_cases/txt2img.py:run_txt2img` (Option A: engines are adapters; mode orchestration lives in use-cases).
- 2025-12-30: `CodexDiffusionEngine.load()` now calls `unload()` when already loaded, and `unload()` clears bound components from the memory manager (prevents duplicate model instances accumulating when reloading with different overrides).
- 2026-01-01: Engine docs note that `get_learned_conditioning(...)` may return either a dict or a cross-attn tensor (both supported by `compile_conditions`).
- 2026-01-02: Added standardized file header docstrings to shared engine modules (doc-only change; part of rollout).
- 2026-01-04: `CodexObjects` renamed `unet` → `denoiser`; engines store their sampling core patcher under `codex_objects.denoiser` (UNet for SD-family; transformer/DiT for Flux/Z-Image/WAN).
- 2026-01-06: VAE override (`vae_path`) now unwraps wrapper VAEs via `first_stage_model` before calling `safe_load_state_dict` (fixes `'VAE' object has no attribute 'state_dict'`).
- 2026-01-06: VAE/TE selection is explicit via `vae_source`/`tenc_source` + paths; core-only `.gguf` checkpoints never treat these paths as state-dict overrides, and ZImage always treats them as external selection (may be dir/gguf).
- 2026-01-06: Refreshed `base.py` file header blocks to document `vae_source`/`tenc_source` validation and core-only `.gguf` semantics (doc-only change).
- 2026-01-06: Generation metadata no longer falls back to `"Automatic"` for sampler/scheduler; missing values serialize as null to surface invalid inputs.
- 2026-01-08: `base.py` now imports `TextEncoderOverrideConfig` from `runtime.models.text_encoder_overrides` after the loader seam carve (no behavior change).
- 2026-01-14: Flux engines now pass `expected_family` into `resolve_diffusion_bundle(...)` so prefixed Flux checkpoints can use metadata-driven signatures instead of state-dict detector guesses.
- 2026-01-27: `BaseVideoEngine` now wires `_maybe_export_video(...)` to the ffmpeg exporter (mp4/webm/gif) so WAN txt2vid/img2vid can return `video {rel_path,mime}` (served under `/api/output/{rel_path}`).
- 2026-01-29: `CodexDiffusionEngine.img2img` now accepts `GenerationResult` from the canonical use-case, allowing masked img2img full-res paste-back to return decoded/composited PIL images without re-decoding in the engine wrapper.
- 2026-01-31: `CodexDiffusionEngine.img2img` now delegates to `apps/backend/use_cases/img2img.py:run_img2img` (Option A ownership); the base engine also provides default first-stage VAE `encode_first_stage/decode_first_stage` for image engines with optional decode-stats hooks.
- 2026-01-31: Added shared helper modules:
  - `runtime_lifecycle.py` (`require_runtime`) for consistent fail-fast runtime guards across engines.
  - `tensor_tree.py` (`detach_to_cpu` / `move_to_device`) for caching payload CPU↔device moves.
  - `prompt_wrappers.py` (`PromptListBase`) for common prompt metadata flags (negative + smart-cache override).
- 2026-02-01: Added shared helper modules:
  - `model_scopes.py` (`stage_scoped_model_load`) to enforce stage-scoped smart-offload semantics when loading text encoders.
  - `capabilities_presets.py` (constants) to dedupe common image-engine capabilities tuples without hiding fields.
- 2026-02-18: `model_scopes.stage_scoped_model_load(...)` now passes scoped event context into `memory_management.manager` so generic smart-offload `load`/`unload` emission remains centralized in manager ownership.
- 2026-02-18: `base.py` engine lifecycle logs now emit via global runtime event emitter (`emit_backend_event`) for a single backend event emission path.
- 2026-02-18: `base.py` now canonicalizes patcher-backed memory-manager targets (`component.patcher` when present, else component) across first-stage VAE encode/decode and engine-level unload cleanup (denoiser, VAE, clipvision, and text-encoders fallback when `patcher is None`), preventing wrapper-vs-patcher record splits.
- 2026-02-21: `BaseVideoEngine._maybe_export_video(...)` now parses `save_output` strictly (`parse_bool_value`) and fails loud on export errors (raises `VideoExportError` instead of returning `{saved:false}` fallback metadata).
- 2026-03-11: `BaseVideoEngine._maybe_export_video(...)` now accepts optional `audio_source_path` for mux-capable video exports and returns truthful `has_audio` metadata from the exporter result.
- 2026-02-22: `CodexDiffusionEngine.decode_first_stage(...)` now preserves denoiser residency whenever the denoiser is already loaded by loading VAE+denoiser as a keep-loaded set; this prevents preview-time VAE loads from unloading the active denoiser mid-sampling (Anima `llm_adapter` embedding CPU/CUDA mismatch path).
- 2026-01-31: `CodexDiffusionEngine` improvements:
  - `__init__` accepts an optional `logger=` to avoid subclass logger collisions and keep per-engine log namespaces consistent.
  - Conditioning cache helpers (`_get_cached_cond/_set_cached_cond`) accept per-call enable overrides and store arbitrary payloads (tensors/dicts/tuples), with hit/miss metrics.
  - Default `set_clip_skip` is a no-op for engines without `"clip"` in `required_text_encoders` (engines with CLIP must still implement clip-skip; fail loud otherwise).
- 2026-02-05: Added Anima-specific `tenc_path` invariant in `base.py`: Anima accepts exactly one external text encoder path and now fails loud when `tenc_path` is passed as `list[str]`.
- 2026-02-10: Batch 4C tightened `base.py` typing/fail-loud surfaces: `LoadOptions` + `EngineStatus` are explicit, `status()` now rejects non-string/blank `engine_id` and non-bool `loaded`, and normalized `TextEncoderOverrideConfig` is persisted in load options.
