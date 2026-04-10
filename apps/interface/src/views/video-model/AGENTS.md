# apps/interface/src/views/video-model Overview
Date: 2026-04-03
Last Review: 2026-04-03
Status: Active

## Purpose
- Hold view-local runtime helpers for `VideoModelTab.vue`.
- These helpers mount family-specific composables/watchers and expose slot props to the route-owned video view.

## Files
- `apps/interface/src/views/video-model/VideoModelTabWanRuntime.vue` — renderless WAN runtime helper for the canonical video tab view.
- `apps/interface/src/views/video-model/VideoModelTabWan22_5bRuntime.vue` — renderless exact WAN 2.2 5B single-stage runtime helper for the canonical video tab view.
- `apps/interface/src/views/video-model/VideoModelTabLtxRuntime.vue` — renderless LTX runtime helper for the canonical video tab view.

## Notes
- `VideoModelTab.vue` remains the body/layout owner. Do not move panel/card template ownership into this folder.
- Helpers here may own active-family-only side effects (bootstrap, auto-resume, temporal persistence, guided listeners, checkpoint-default watchers).
- Exported-video zoom overlay visibility is family-runtime-owned here (`videoZoomOpen` style state), while the actual `VideoZoomOverlay` component stays mounted in `VideoModelTab.vue`.
- LTX geometry/frame/profile warnings also belong here: keep `32/64` dimension alignment, `8n+1` frame blocking messages, and stale execution-profile blocking copy in the LTX runtime helper instead of burying that contract in shared video cards or QuickSettings.
- LTX Results header/history actions also belong here: `VideoModelTabLtxRuntime.vue` now wires the compact per-tab history strip plus `Save snapshot` / `Copy params` actions into the WAN-baseline `GenerationResultsPanel.vue` without reintroducing a second LTX-only Results surface.
- WAN history-details state and actions still belong here too, but `VideoModelTabWanRuntime.vue` now exports section data into the shared presentational `components/modals/RunHistoryDetailsModal.vue` mounted by `VideoModelTab.vue`; keep LTX out of that modal seam.
- WAN 2.2 exact branches stay split here: `VideoModelTabWanRuntime.vue` is the 14B/two-stage lane, and `VideoModelTabWan22_5bRuntime.vue` is the 5B/single-stage lane. Do not reintroduce a generic WAN runtime helper that guesses the engine shape from shared props.
- Keep LTX helper copy here blocking-only. Do not reintroduce always-on explanatory legends about execution profiles, output assets, or selector ownership into the body cards.
- Do not add shared presentational components here; shared UI belongs under `apps/interface/src/components/**`.
- Only the active video family may instantiate its runtime helper at a time.
