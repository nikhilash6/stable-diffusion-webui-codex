# apps/backend/runtime/vision/clip Overview
Date: 2025-10-31
Last Review: 2026-03-30
Status: Active

## Purpose
- Codex-native Clip vision runtime (model specs, state-dict tooling, encoder wrapper, preprocessing).

## Notes
- Raise `ClipVisionError` subclasses for all error paths; never fall back to silent prints.
- Specs live in `specs.py` and feed detection/registry helpers; extend via dataclasses/enums.
- `ClipVisionEncoder` handles device/dtype selection and logging; keep it free of UI concerns.
- Update `.sangoi/plans/2026-01-27-codex-legacy-backlog.md` entry #9 when behaviour changes or new variants land.
- 2026-01-02: Added standardized file header docstrings to CLIP vision runtime modules (doc-only change; part of rollout).
- 2026-01-18: `clip/__init__.py` is a package marker (no re-exports); import types/helpers from the defining modules (`encoder.py`, `errors.py`, `types.py`, etc.).
- 2026-03-29: `state_dict.py` is now a facade over `apps/backend/runtime/state_dict/keymap_clip_vision.py`; `ClipVisionEncoder.from_state_dict(...)` is the only public CLIP vision load seam and must normalize supported source keyspaces before `safe_load_state_dict(...)`.
- 2026-03-29: Adapter-local CLIP vision pre-normalizers are forbidden here. Do not reintroduce filtered copies, prefix stripping, bespoke rekey helpers, or raw `nn.Module.load_state_dict(...)` in IP-Adapter/image-encoder paths.
- 2026-03-29: Keyspace-resolution failures from the canonical keymap must be translated back into `ClipVisionError` subclasses at the CLIP seam; do not leak raw `KeyMappingError` out of `vision/clip/*`.
- 2026-03-29: `ClipVisionEncoder` must follow the same memory-owner birth/load pattern as the central loaders: resolve the owner device/dtype first, construct/mount the module under `using_codex_operations(**to_args, ...)`, then call `safe_load_state_dict(...)`. `ModelPatcher` only owns runtime offload/reload; it does not excuse a CPU-born load path.
- 2026-03-30: CLIP vision runtime birth/load/cache ownership belongs to `DeviceRole.CLIP_VISION`, not `TEXT_ENCODER`. The checkpoint source mapping may still be staged through text-encoder offload IO, but the runtime module itself must resolve device/dtype from the dedicated CLIP vision role before `safe_load_state_dict(...)`.
- 2026-03-30: `preprocess.py` must stay numerically aligned with `transformers.CLIPImageProcessor` for the supported CLIP variants and return canonical float32 CLIP pixel values. `ClipVisionEncoder.encode_pixels(...)` owns the final device/dtype cast into runtime execution, so callers that need official `zeros_like(pixel_values)` semantics (such as IP-Adapter Plus) must zero the already-preprocessed tensor instead of re-encoding a black image.
