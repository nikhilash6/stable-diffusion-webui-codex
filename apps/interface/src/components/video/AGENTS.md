# apps/interface/src/components/video Overview
Date: 2026-04-03
Last Review: 2026-04-08
Status: Active

## Purpose
- Shared presentational video card components used by `apps/interface/src/views/VideoModelTab.vue` and future video-family branches.

## Key Files
- `VideoPromptStageCard.vue` — generic prompt card used by WAN/LTX prompt sections.
- `VideoCoreParamsCard.vue` — generic width/height/frames/FPS card.
- `VideoStageBasicParamsCard.vue` — generic sampler/steps/cfg/seed stage card.
- `VideoOutputCard.vue` — generic output/assets wrapper card.

## Notes
- Keep these components presentational only: no stores, no family composables, no bootstrap/listener side effects.
- Family-specific runtime ownership stays under `apps/interface/src/views/video-model/**`.
- Live img2vid init-image ownership no longer lives in this folder; `apps/interface/src/components/InitialImageBlock.vue` is the shared owner used by WAN/LTX while this folder keeps the remaining video-only cards.
- Shared styling for these components lives in `apps/interface/src/styles/components/video-generation-cards.css`.
