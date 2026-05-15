# apps/backend/runtime/families/netflix_void Overview
<!-- tags: backend, runtime, families, netflix_void, video, vid2vid -->
Date: 2026-04-03
Last Review: 2026-04-03
Status: Active

## Purpose
- Host the repo-owned Netflix VOID family scaffold under `apps/**`.
- Freeze the explicit base-bundle + literal Pass 1/Pass 2 overlay contract while the native execution port lands in bounded family-owned slices.

## Key Files
- `apps/backend/runtime/families/netflix_void/config.py` — Canonical family constants, literal overlay filenames, and base-bundle validation helper.
- `apps/backend/runtime/families/netflix_void/model.py` — Typed bundle contracts (`NetflixVoidBaseBundle`, `NetflixVoidOverlayPair`, `NetflixVoidBundleInputs`).
- `apps/backend/runtime/families/netflix_void/loader.py` — Fail-loud base-bundle + Pass 1/Pass 2 pair resolution from local inventory.
- `apps/backend/runtime/families/netflix_void/native/` — Native component loaders/builders (currently base-bundle component IO + T5 text encoder).
- `apps/backend/runtime/families/netflix_void/preprocess.py` — Shared source-video + quadmask preprocessing (ffmpeg extraction, quadmask quantization/inversion, temporal padding).
- `apps/backend/runtime/families/netflix_void/runtime.py` — Runtime scaffold carrying normalized device/dtype options, native text-encoder hydration, and the explicit not-yet-implemented `run_netflix_void_vid2vid(...)` seam.
- `apps/backend/runtime/families/netflix_void/__init__.py` — Public family surface.

## Notes
- Do not import `.refs/**` from active code.
- Do not add key-rewrite helpers, prefix strippers, or eager `dict(...)` checkpoint materialization here.
- `netflix_void_execution.py` remains inventory/selector metadata only; execution ownership stays in this family runtime.
- Reuse `native/bundle_io.py` for new base-bundle components instead of inventing component-local path scanners.
- Automatic mask generation is out of scope for this directory; tranche A consumes a precomputed quadmask video only.
