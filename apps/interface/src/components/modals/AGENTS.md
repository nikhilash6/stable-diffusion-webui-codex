<!-- tags: frontend, components, modals -->
# apps/interface/src/components/modals Overview
Date: 2025-12-04
Last Review: 2026-04-08
Status: Active

## Purpose
- Modal components used across the UI (prompt editing helpers, style editor, and global QuickSettings dialogs).

## Key files
- `apps/interface/src/components/ui/Modal.vue` — shared modal shell (header/body/footer, click-outside backdrop).
- `apps/interface/src/components/modals/AssetMetadataModal.vue` — read-only JSON metadata viewer for selected checkpoints/assets (QuickSettings info button).
- `apps/interface/src/components/modals/PromptAssetInsertModal.vue` — shared prompt-asset modal shell (search/weight/load/error/list chrome) reused by LoRA and TI wrappers.
- `apps/interface/src/components/modals/LoraModal.vue` — LoRA picker/insert helpers.
- `apps/interface/src/components/modals/TextualInversionModal.vue` — TI picker/insert helpers.
- `apps/interface/src/components/modals/QuickSettingsOverridesModal.vue` — runtime main-device + per-component dtype overrides UI.
- `apps/interface/src/components/modals/QuickSettingsAddPathModal.vue` — reusable quicksettings add-path workflow (scan candidates + add-one/add-all).
- `apps/interface/src/components/modals/StyleEditorModal.vue` — create/edit prompt styles.

## Notes
- Avoid `style="..."` in templates; prefer shared primitives and CSS in `apps/interface/src/styles/**`.
- `PromptAssetInsertModal.vue` is the shared owner for prompt-asset modal chrome (search, weight, lazy ensure-loaded trigger, refresh button, count, error, filtered list shell); keep LoRA/TI inventory loading and token-format semantics in `LoraModal.vue` / `TextualInversionModal.vue`.
- `LoraModal.vue` emits `<lora:filename:weight>` payloads with `target` + `action` (`add`/`remove`) so prompt surfaces can treat per-row actions as true toggles.
- `LoraModal.vue` accepts `showNegativeTarget` so prompt owners can hide the Negative action when the live prompt surface also hides the negative field.
- Keep modals presentational; stores and routing decisions live in views/stores.
- 2026-04-08: `RunHistoryDetailsModal.vue` is the shared presentational owner for image/WAN history-details chrome (preview, meta, summary, sections, params snapshot, footer buttons). Keep history persistence, section-building, and apply/load/copy behavior in the live image/WAN owners; LTX stays outside this modal seam.
- 2026-01-03: Added standardized file header blocks to modal components (doc-only change; part of rollout).
- 2026-01-13: `AssetMetadataModal.vue` adds in-view controls (Beautify + expand/collapse all) to switch between raw/nested file metadata and manage large trees.
- 2026-02-15: `QuickSettingsOverridesModal.vue` now reflects backend apply metadata; restart warning appears only when `/api/options` reports `restart_required[]`, otherwise it shows hot-apply guidance.
- 2026-04-05: `QuickSettingsOverridesModal.vue` now exposes exactly one device owner (`codex_main_device`) plus component dtype overrides. Do not reintroduce separate TE/VAE/Core device selects in the WebUI modal.
- 2026-02-17: `LoraModal.vue` now supports explicit inventory refresh (`refreshModelInventory`) and surfaces load errors in-modal while emitting filename-based LoRA prompt tokens (`<lora:filename:weight>`); SHA resolution is attached separately at request payload build time.
- 2026-02-28: `LoraModal.vue` continues to emit prompt-target token payloads (`<lora:filename:weight>`) while WAN stage LoRA arrays are derived in `useVideoGeneration` during request assembly.
- 2026-02-21: `StyleEditorModal.vue` now reuses the shared `ui/Modal.vue` shell (teleport + backdrop + footer slot) instead of rendering an ad-hoc modal container.
- 2026-02-21: Shared modal list spacing now uses `.modal-list-section` across TI/LoRA pickers; inline `style="margin-top: ..."` was removed.
- 2026-03-02: `LoraModal.vue` now uses a compact toolbar row (`Search`, `Weight`, `Refresh`, count), removes the redundant footer close button (header `✕` remains), uses explicit per-row Prompt/Negative toggle actions, and refreshes quicksettings LoRA SHA mappings from the same inventory payload used by the modal list.
- 2026-03-02: `AssetMetadataModal.vue` now keeps tree-view controls (`Beautify`, expand all, collapse all) visible after toggling Beautify off; object payloads always stay in `JsonTreeView` mode so controls no longer disappear until reload.
- 2026-03-02: `LoraModal.vue` list rows are now scrollable with zebra/hover states and right-aligned action toggles; each action emits `add/remove` instead of unlimited repeated insertions.
- 2026-03-03: Added `QuickSettingsAddPathModal.vue` as the reusable quicksettings add-path modal: sanitizes user paths, scans candidates without SHA, supports per-row add and sequential add-all progress, and keeps errors fail-loud for quicksettings toasts.
- 2026-03-03: `QuickSettingsAddPathModal.vue` scan action now includes an explicit spinner animation while in-flight.
- 2026-03-04: `QuickSettingsAddPathModal.vue` now enforces an explicit row FSM (`queued|adding|added|already_in_library|error`), computes add-all byte progress deterministically only when all pending rows have valid `size_bytes`, and fails loud on missing/invalid byte metadata.
