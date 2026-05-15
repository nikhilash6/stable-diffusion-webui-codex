# apps/backend/engines/flux Overview
Date: 2025-12-06
Last Review: 2026-02-09
Status: Active

## Purpose
- Flux.1 family engines (Flux, Chroma, Kontext) leveraging the Flux runtime.

## Notes
- Ensure scheduler and runtime dependencies stay in sync with `apps/backend/runtime/families/flux/`.
- Shared assembly helpers live in `spec.py`; Flux-family engines assemble runtimes via `CodexFluxFamilyFactory` (`apps/backend/engines/flux/factory.py`) which wraps `assemble_flux_runtime`.
- `assemble_flux_runtime` accepts `engine_options` and validates streaming directives via `StreamingConfig` for Flux family assembly.
- `_maybe_enable_streaming_core` currently enforces a fail-loud contract: for `flux1`, any explicit/auto streaming enable request raises `NotImplementedError` until the capability contract is finalized.
- When streaming is disabled, `_maybe_enable_streaming_core` unwraps an already-streamed core so reloads stay on the canonical non-streaming runtime path.
- Flux-family engines expose `EngineCapabilities` and set distilled-CFG behaviour during `load()`; keep bundle assembly side-effect free.
- 2025-12-30: Flux.1 now wraps prompts with per-job metadata (`distilled_cfg_scale`, `smart_cache`) and the conditioning cache respects `smart_cache` and includes `distilled_cfg_scale` in its key (avoids stale embeddings when toggling cache or changing distilled CFG).
- 2026-01-01: `Flux.set_clip_skip(...)` now clears the conditioning cache to avoid returning stale pooled embeddings when `smart_cache` is enabled.
- 2026-01-02: Added standardized file header docstrings to Flux engine modules (doc-only change; part of rollout).
- 2026-01-03: Flux runtime now uses `DenoiserPatcher` (`FluxEngineRuntime.denoiser`) instead of `UnetPatcher`; ControlNet hooks remain UNet-only.
- 2026-01-03: Flux/Chroma/Kontext engines now assemble runtimes via `CodexFluxFamilyFactory` (keeps `_build_components` consistent and reduces drift).
- 2026-01-04: Flux Kontext is treated as a Flux.1 variant (`Flux.1-Kontext-dev`) and is co-located with the Flux family under `apps/backend/engines/flux/kontext.py`.
- 2026-01-04: Flux.1-family engine keys are `flux1` / `flux1_kontext` / `flux1_chroma` (no legacy aliases); clients must use canonical keys.
- 2026-01-06: Flux sampler allow-lists now use canonical `SamplerKind` strings (e.g. `euler a`, `dpm++ 2m`).
- 2026-01-20: Removed unused `flux_config.py` (no call sites; config lives in `spec.py` / factory assembly).
- 2026-01-25: `clip_skip=0` is now accepted as a “use default” sentinel for Flux CLIP branches (resets to the canonical default without requiring a separate UI toggle).
- 2026-01-31: Flux/Chroma engines now rely on the default `CodexDiffusionEngine.encode_first_stage/decode_first_stage` implementation for the common image-VAE semantics (no behavior change; reduces duplication).
- 2026-01-31: Kontext no longer duplicates `_build_components`; it inherits Flux runtime assembly (factory-driven) to reduce drift. Flux conditioning caching now uses shared cache helpers (CPU storage + device restore) and `_on_unload` clears the streaming controller reference.
- 2026-02-01: Flux clip-skip handling is now Flux-local (`apps/backend/engines/flux/_clip_skip.py`) and validates/reset semantics mirror SD (no compat shims). Text-encoder patcher load/unload is stage-scoped via `stage_scoped_model_load(...)` to avoid unload/reload churn under smart-offload.
- 2026-02-09: Flux/Chroma conditioning entrypoints now use `torch.no_grad()` (not `torch.inference_mode()`) to avoid caching inference tensors across requests (version-counter faults).
