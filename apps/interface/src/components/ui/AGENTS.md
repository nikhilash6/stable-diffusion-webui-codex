<!-- tags: frontend, components, ui, primitives -->
# apps/interface/src/components/ui Overview
Date: 2025-12-23
Last Review: 2026-03-29
Status: Active

## Purpose
- Small, reusable UI primitives used across views/components (modals, form widgets).

## Key Files
- `apps/interface/src/components/ui/Modal.vue` — Generic modal shell used by views (file browser, overrides, etc).
- `apps/interface/src/components/ui/SliderField.vue` — Shared slider layout (label left, input right, slider below).
- `apps/interface/src/components/ui/HoverTooltip.vue` — Reusable hover/focus tooltip primitive (title + multiline body) used by slider labels.
- `apps/interface/src/components/ui/NumberStepperInput.vue` — Numeric input with optional +/- stepper controls.
- `apps/interface/src/components/ui/DimensionPresetsGrid.vue` — Resolution preset buttons (2×2 grid), used by `BasicParametersCard.vue`.
- `apps/interface/src/components/ui/UpscalerTileControls.vue` — Tile presets + overlap + min tile + explicit OOM fallback toggle (shared by hires-fix + `/upscale`).
- `apps/interface/src/components/ui/JsonTreeView.vue` — Collapsible JSON renderer used by the metadata modal (`<details>/<summary>` tree).
- `apps/interface/src/components/ui/Dropzone.vue` — Drag-and-drop file picker primitive (presentational; emits `select`/`rejected`).
- `apps/interface/src/components/ui/ImageZoomOverlay.vue` — Full-screen image zoom overlay with shared close semantics (Esc and outside-click).
- `apps/interface/src/components/ui/VideoZoomOverlay.vue` — Full-screen video zoom overlay with controls-safe default interaction (`Pan: Off`), explicit drag-pan zoning (`Pan: On`), outside-click/Escape close, and double-click fullscreen suppression.
- `apps/interface/src/components/ui/InpaintMaskEditorOverlay.vue` — Full-screen inpaint mask editor overlay (brush/eraser/circle/polygon + zoom/pan + undo/redo).
- `apps/interface/src/components/ui/inpaint_mask_editor_engine.ts` — Pure mask-editing engine used by the inpaint editor overlay (draw ops + bounded history + mask helpers).

## Notes
- Keep components presentational: props in, emits out; no store calls or API fetching.
- Styling should live under `apps/interface/src/styles/components/` and use semantic classes.
- If a reusable overlay or primitive keeps a sanctioned CSS-contract seam, the contract is owned by `.sangoi/reference/ui/frontend-css-contracts.md` and the verifier config; this file stays pointer-only and must not enumerate live exception symbols or rationales.
- 2026-03-26: `InpaintMaskEditorOverlay.vue` now maps pointer coordinates from the actual mask-canvas rect/backing-store ratio instead of the generic content wrapper, and live brush/eraser moves coalesce through `requestAnimationFrame` while skipping full blur/crop preview recompute until commit/tool/param/source changes.
- 2026-01-03: Added standardized file header blocks to UI primitive components (doc-only change; part of rollout).
- 2026-01-13: `JsonTreeView.vue` supports expand/collapse-all signals (used by the metadata modal controls).
- 2026-01-29: Added `Dropzone.vue` with styles in `apps/interface/src/styles/components/cdx-dropzone.css`.
- 2026-02-04: `UpscalerTileControls.vue` now exposes `min_tile` as an Advanced control (keeps backend tile fallback behavior visible and configurable).
- 2026-02-08: `UpscalerTileControls.vue` now supports `presetVariant='resolution'` so hires tile preset buttons can match the Basic Parameters resolution-button pattern while keeping legacy toggle styling as default.
- 2026-02-17: Added `ImageZoomOverlay.vue`; `ResultViewer.vue` and init-image previews now reuse the same zoom behavior and close path (`Esc` and outside-click call the same close handler).
- 2026-02-18: `SliderField.vue` now supports optional rich label tooltips (hover/focus trigger with `?` badge) via shared `HoverTooltip.vue`.
- 2026-02-21: Added `InpaintMaskEditorOverlay.vue` + `inpaint_mask_editor_engine.ts` for practical inpaint mask authoring (binary mask export, brush/eraser/circle/polygon, deep undo/redo, reset/apply contract).
- 2026-02-21: `ImageZoomOverlay.vue` and `InpaintMaskEditorOverlay.vue` now apply pan offsets with explicit `left/top` values plus scale-only transforms, removing `transform: translate...` usage; inpaint brush cursor centering now uses top-left coordinate math instead of translate offsets.
- 2026-02-21: `InpaintMaskEditorOverlay.vue` now includes `Upload mask` in-toolbar import; uploaded masks are stretched to init-image dimensions and committed through engine history (`replaceMask`) so undo/redo reverts/restores imports.
- 2026-03-24: `InpaintMaskEditorOverlay.vue` now reuses parent-owned `maskBlur` / `maskedPadding` state via shared sliders, renders the same outward blur-spill band + cyan effective-crop box used by the inline thumbnail, requires explicit processing dimensions for crop preview math, and closes itself if that preview contract becomes unavailable. `maskInvert` now drives true WYSIWYG editing against the visible effective mask while apply/export still keep the raw draft mask plane unchanged at the storage boundary.
- 2026-03-24: `HoverTooltip.vue` keeps hover help keyboard-accessible via `:focus-visible`/descendant `:has(:focus-visible)` instead of raw `:focus-within`, so pointer clicks on toggles or `?` triggers no longer pin tooltips after the pointer leaves.
- 2026-03-01: `ImageZoomOverlay.vue` now supports optional WAN frame-guide editing for init-image previews: guide toggle, free image-size controls (AR locked), guide `W/H` controls, drag-based crop offsets, and source/scaled/frame/crop metadata bound to deterministic projection math.
- 2026-03-02: `ImageZoomOverlay.vue` frame-guide normalization now preserves user-selected `imageScale` while source dimensions are still unknown (no implicit clamp-to-1 during unresolved-source state); min-scale clamping is only applied after source/frame geometry is known.
- 2026-03-02: `Modal.vue` now supports optional per-instance panel classes (`panelClass`) and optional footer rendering (`showFooter`) so modal consumers can tune size/layout and remove redundant footer actions while keeping shared close semantics (`✕` + backdrop click).
- 2026-03-04: Added `VideoZoomOverlay.vue` as a dedicated video-only overlay component (separate from image overlay) with wheel zoom + drag pan, `Esc`/outside-click close, and `@dblclick.prevent.stop` to block native video fullscreen.
- 2026-03-04: `VideoZoomOverlay.vue` now defaults to controls-safe interaction (`Pan: Off`) and exposes explicit drag-pan zoning (`Pan: On`) so overlay pan does not accidentally suppress native media controls.
