# apps/backend/runtime/families/ltx2/streaming Overview
<!-- tags: backend, ltx2, streaming, runtime -->
Date: 2026-03-12
Last Review: 2026-03-26
Status: Active

## Purpose
Owns the LTX2 family-local core-streaming layer for the native transformer core.
This package reuses the shared streaming controller contract from
`apps/backend/runtime/streaming/controller.py` and only streams
`Ltx2VideoTransformer3DModel.transformer_blocks`.

## Key files
- `apps/backend/runtime/families/ltx2/streaming/specs.py` — segment/execution-plan dataclasses for `transformer_blocks`.
- `apps/backend/runtime/families/ltx2/streaming/controller.py` — shared-controller wrapper with LTX2 ownership.
- `apps/backend/runtime/families/ltx2/streaming/config.py` — fixed-default config parsing from the normalized internal boolean option surface only.
- `apps/backend/runtime/families/ltx2/streaming/wrapper.py` — streamed transformer wrapper using block hooks around the native forward loop while proxying the native transformer attributes consumed by `native/pipelines.py`.

## Notes
- This tranche streams only the transformer core. It does not stream connectors, VAE, audio VAE, or vocoder.
- The wrapper contract must stay honest for `native/pipelines.py`: proxy `config`, `rope`, `audio_rope`, `cache_context(...)`, and `forward(...)` without forking the giant native forward.
- The wrapper registers the real transformer under `_base`, so any runtime seam that targets named parameters directly (for example the temporary two-stage distilled-LoRA apply/revert path) must unwrap `base_model` first instead of targeting wrapper-prefixed `_base.*` names.
- Tuning remains internal in this tranche. `config.py` must consume only normalized `core_streaming_enabled`; legacy `codex_core_streaming` reaching this seam is a fail-loud upstream bug.
- WSL validation for giant real assets stays header-only / metadata-only unless the user explicitly asks otherwise.
