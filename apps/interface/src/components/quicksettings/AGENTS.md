# apps/interface/src/components/quicksettings Overview
<!-- tags: frontend, quicksettings, engines -->
Date: 2025-12-06
Last Review: 2026-04-08
Status: Active

## Purpose
- Compact engine/paths/performance controls rendered in the main header (`QuickSettingsBar.vue`), backed by the global `quicksettings` store.

## Key Files
- `QuickSettingsAssetBlock.vue` — Shared non-WAN quicksettings asset owner for checkpoint/VAE/text-encoder selectors, including the dedicated LTX branch in `QuickSettingsBar.vue`; mode toggles and family-only extras remain header-owned.
- `QuickSettingsPerf.vue` — Performance toggles shared across engines (Smart Offload/Fallback/Cache/Core Streaming) rendered in the Advanced nested area.
- `QuickSettingsWan.vue` — WAN22-specific quicksettings (Mode preset selector + `LightX2V` toggle button, high/low model dirs, text encoder/VAE selectors, plus a Refresh button).

## Notes
- `QuickSettingsAssetBlock.vue` stays presentational and non-WAN-only; engine-specific filtering, read-only routing, mode toggles, and WAN-only selectors still live in `QuickSettingsBar.vue`.
- `QuickSettingsBar.vue` renders a main row for engine selectors and a collapsible Advanced nested area (Smart toggles + GPU VRAM / Attention Backend / Overrides), with a left-side handle button.
- 2026-03-16: `QuickSettingsBar.vue` owns LTX `TXT2VID/IMG2VID` mode in the shared top row. The LTX branch still binds checkpoint/VAE/text encoder directly to `tab.params.{checkpoint,vae,textEncoder}`, filters text encoders through `ltx2_tenc`, keeps the TE selector visible even for non-core-only checkpoints, freezes the whole LTX row during active runs and LTX-specific hydration gaps, and preserves hidden init-image state when switching back to `TXT2VID`.
- 2026-03-16: On any `/models/:tabId` route, `QuickSettingsBar.vue` now waits for the hydrated tab object before rendering a family branch. While that route-tab hydration is pending, selector handlers no-op with a toast instead of falling back to global checkpoint/text-encoder writes; family-global VAE ownership remains unchanged after hydration completes, stale tab ids switch to an explicit not-found placeholder, and tab-load failures switch to an explicit load-failure placeholder.
- 2026-04-08: `QuickSettingsAssetBlock.vue` now accepts the shared disabled/family-label/dual-text-encoder contract for every non-WAN selector row, so `QuickSettingsBar.vue` can stop mirroring `Base`/`Flux`/`Flux2`/`Chroma`/`ZImage` wrappers while preserving the same add-path/metadata actions.
- The shared read-only wrapper in `QuickSettingsBar.vue` is structural, not cosmetic: if a family branch is nested under `fieldset.qs-readonly-fieldset`, that wrapper must preserve the main-row flex contract for all sibling `quicksettings-group` roots.
- 2026-04-08: `QuickSettingsAssetBlock.vue` now owns the shared optional text-encoder metadata/add-path controls used by LTX, Flux, Chroma, and Z-Image.
- `QuickSettingsPerf` uses toggle buttons (`.qs-toggle-btn`) for Smart Offload/Fallback/Cache/Core Streaming (no legacy switches).
- 2026-02-22: `QuickSettingsPerf.vue` adds an `Obliterate VRAM` action button (disabled while running), emitted to `QuickSettingsBar.vue` for backend-triggered VRAM cleanup (`POST /api/obliterate-vram`) with safe default external mode (`disabled`).
- Text encoder dropdowns display a compact label (`family/basename`) even when `/api/paths` or the inventory return long absolute paths; the full value is still posted back in the `<option value>`.
- For Flux-family tabs, `QuickSettingsBar` hides the base text encoder field but keeps family contracts honest: `flux1` renders dual CLIP/T5 selectors under `flux1_tenc`, while `flux2` renders a single `Qwen3-4B` selector under `flux2_tenc` for the current Klein 4B / base-4B slice. Wiring to backend overrides still reuses the shared sha-first `tenc_sha` path.
- 2026-01-28: Z-Image keeps a per-tab `Turbo` toggle (`tab.params.zimageTurbo`), and `QuickSettingsBar.vue` can still lock that toggle when the selected checkpoint carries trusted `codex.zimage.variant` metadata (Codex-produced GGUFs).
- 2026-04-08: Z-Image Turbo now mounts through `QuickSettingsAssetBlock.vue`'s `after-checkpoint` slot; `QuickSettingsBar.vue` still owns the IMG2IMG/INPAINT toggle group and keeps it aligned near `Refresh` instead of hiding that ordering inside the asset owner.
- 2026-03-06: `QuickSettingsBar.vue` owns IMG2IMG/INPAINT mode toggles in the top quicksettings row. FLUX.1 keeps INPAINT visible but disabled through `flux1_kontext`, while FLUX.2 Klein uses the shared taxonomy/request-engine gate: img2img is available for the current 4B/base-4B slice, and masked img2img/inpaint stays enabled when the resolved engine truthfully supports it.
- Shared header INPAINT gating must stay aligned with the body/runtime rule: masking requires `useInitImage`, `initSource.mode='img'`, a materialized init image, and backend capability support.
- 2025-12-27: Removed the `hideCheckpoint` toggle/prop; checkpoint selection is always rendered, and on `/models/:tabId` it is tab-scoped (`tab.params.checkpoint`, auto-seeded) while still filtering choices by engine-specific `*_ckpt` roots from `apps/paths.json` (plus user-added paths).
- 2025-12-14: WAN text encoder selector now lists explicit `.safetensors` / `.gguf` files under `wan22_tenc` and stores values as `wan22/<abs_path>` for consistent labeling; payload builders must normalize before sending to backend.
- 2025-12-14: WAN Metadata/VAE selectors now prefer concrete inventory paths (VAE constrained by `wan22_vae`), keeping the video endpoints strict about asset paths.
- 2026-01-17: WAN Metadata selector was removed; WAN preset Mode now drives the metadata repo id used by payloads.
- 2025-12-15: QuickSettings WAN groups now use `.qs-group-wan-*` sizing hooks so the header flex layout doesn’t collapse all controls to the left on wide screens.
- 2026-03-03: WAN model browse now uses a single compact `+` next to `LightX2V` (shared `wan22_ckpt` root); the duplicated per-select model `+` actions were removed.
- 2025-12-20: Replaced WAN “Format” with a `LightX2V` toggle; per-stage LoRA selection now lives in the WAN tab (High/Low Noise) when enabled.
- 2025-12-26: Removed the WAN Assets modal; metadata/text encoder/VAE selectors are now inline in the header quicksettings bar.
- 2025-12-26: QuickSettings buttons now use `qs-btn-secondary`/`qs-btn-outline` so they fill the `qs-row` height and keep a visible border (no fixed `2rem` height from `.btn-*` variants).
- 2025-12-28: Removed the obsolete “Diffusion in Low Bits” selectors and moved Smart toggles + GPU VRAM / Attention Backend / Overrides into a collapsible Advanced nested area (open by default, left-side handle); WAN `LightX2V` is a toggle button and the Guided gen header button is hidden for now.
- 2025-12-31: `QuickSettingsWan.vue` now declares `defineEmits(...)` for `browse*` + `update:*` events to avoid Vue “extraneous non-emits listeners” warnings with a fragment root template.
- 2026-01-03: Added standardized file header blocks to quicksettings components (doc-only change; part of rollout).
- 2026-02-20: VAE selectors now recognize canonical sentinel `built-in` across families; metadata buttons are disabled for sentinel values (`built-in`/`none`) to avoid invalid metadata lookups.
- 2026-02-21: `QuickSettingsWan.vue` mode selector now exposes only `I2V/T2V` presets (`14B/5B`); `V2V 14B` was removed to align UI with the current backend vid2vid-disabled contract.
