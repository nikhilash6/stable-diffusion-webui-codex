# apps.launcher.gui_tk
Date: 2026-01-25
Status: Active
Last Review: 2026-04-02

## Purpose
- Modular Tk/ttk GUI implementation for the Codex launcher (services + settings + logs).
- Keep the stable entrypoint at `apps/codex_launcher.py` thin; implementation lives here.

## Modules
- `__init__.py` – Public re-exports (`CodexLauncherApp`, `main`).
- `app.py` – Tk root window + tab wiring + background task polling.
- `controller.py` – Non-UI controller (store/services/log buffer + persistence helpers).
- `styles.py` – Palette + cross-platform font resolution + ttk styling.
- `form_schema.py` – Declarative form descriptor models (`FormSectionDescriptor`, `FormFieldDescriptor`, `FieldKind`).
- `form_renderer.py` – Shared descriptor-driven form renderer used by tabs for consistent layout/progressive disclosure.
- `widgets.py` – Scrollable container + small layout helpers.
- `tabs/services.py` – API/UI supervision tab plus launcher-owned next-start service prefs.
- `tabs/runtime.py` – Device defaults + attention mode + GGUF/LoRA + PyTorch alloc conf.
- `tabs/manual_env_vars.py` – API-only manual env overlay toggle + editor + validation (`KEY=VALUE` lines) used on next API start/restart after Save Settings.
- `tabs/diagnostics.py` – Preflight checks + grouped debug/logging/profiler env flags.
- `tabs/logs.py` – Log viewer (filter/search/export).

## Notes
- UI state auto-persisted via `LauncherProfileStore.save_meta()` is limited to tab index, window geometry, and advanced-controls visibility; launcher next-start service prefs (`external_terminal`, `frontend_dev_typecheck`) plus env changes require explicit `Save Settings`.
- Logs are structured (`CodexLogRecord`) and rendered incrementally to avoid UI freezes.
- Raw `tk.Text` surfaces must be themed explicitly and use `styles.resolve_fonts(...).mono`; do not hardcode Windows-only monospace fonts in tabs.
- 2026-01-30: Removed the dev-only Z-Image Diffusers bypass toggle (`CODEX_ZIMAGE_DIFFUSERS_BYPASS`) from `tabs/diagnostics.py`.
- 2026-01-31: `tabs/diagnostics.py` now exposes global profiling env flags (`CODEX_PROFILE*`) for backend torch-profiler runs.
- 2026-02-15: `tabs/diagnostics.py` now exposes launcher trace toggles (`CODEX_TRACE_CONTRACT`, `CODEX_TRACE_PROFILER`) alongside timeline/profile flags.
- 2026-02-18: `tabs/runtime.py` now exposes task cancel default mode (`CODEX_TASK_CANCEL_DEFAULT_MODE`) with strict choices (`immediate`, `after_current`) alongside existing task/safety knobs.
- 2026-02-21: `tabs/runtime.py` now exposes WAN img2vid chunk buffer mode (`CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE`) with strict choices (`hybrid`, `ram`, `ram+hd`).
- 2026-02-21: `tabs/runtime.py` now exposes attention mode selection (`sdpa_*|xformers|split|quad`) and persists it as launcher bootstrap env (`CODEX_ATTENTION_BACKEND`, `CODEX_ATTENTION_SDPA_POLICY`).
- 2026-02-22: `tabs/runtime.py` now uses `PYTORCH_CUDA_ALLOC_CONF` for allocator tuning.
- 2026-02-22: `tabs/runtime.py` no longer offers GGUF dequant cache levels (`lvl1`/`lvl2`) or ratio tuning UI; the setting is locked to `off` and stale dequant cache env tuning keys are cleared during runtime sync.
- 2026-02-23: `tabs/runtime.py` now uses a single `CODEX_MAIN_DEVICE` selector and mirrors it to `CODEX_CORE_DEVICE`/`CODEX_TE_DEVICE`/`CODEX_VAE_DEVICE` to enforce main-device invariance from launcher bootstrap.
- 2026-02-23: `tabs/runtime.py::reload()` no longer mutates mount/offload to main-device as a side effect while loading UI state; it now parses and preserves persisted `CODEX_MOUNT_DEVICE`/`CODEX_OFFLOAD_DEVICE` values.
- 2026-02-23: `tabs/runtime.py::reload()` now defaults invalid/missing offload values to CPU (not main device) to keep launcher bootstrap aligned with explicit offload de-residency semantics.
- 2026-02-23: Runtime/Diagnostics now support progressive disclosure (`Show advanced ...` toggles) so high-risk profiling/runtime knobs are hidden by default.
- 2026-02-23: Runtime tab now renders from declarative descriptors via `form_schema.py` + `form_renderer.py` (reduces manual widget boilerplate and centralizes form behavior).
- 2026-02-23: Services tab now shows resolved endpoints and quick actions (`Open`, API `Docs`) with no health polling/status row in the UI.
- 2026-02-23: Services tab now resolves effective UI port via repo-root `.webui-ui-<port>.pid` files (port-guard output) so endpoint/open follow fallback ports (`+10000/+20000`) when base `WEB_PORT` is busy.
- 2026-03-28: `controller.py` now propagates the running API handle's launcher-resolved port into UI starts, and `tabs/services.py` now prefers that runtime API port for endpoint/docs actions while the API handle is active.
- 2026-02-23: Visual revamp “Control Room”: updated palette/surfaces/buttons/inputs/status styles for stronger hierarchy and reduced legacy Tk look.
- 2026-02-23: Runtime settings are now split into dedicated top-level tabs (`Bootstrap`, `Engine`, `Safety`) for clearer navigation and less nested chrome.
- 2026-02-23: Advanced controls toggle moved to global footer (`app.py`) and now drives both Runtime and Diagnostics progressive disclosure.
- 2026-02-23: Services button-state refresh now updates only on state transition (avoids hover flicker caused by repeated poll-time reconfigure).
- 2026-02-23: Scrollable canvas border chrome removed (`widgets.py`) and dark scrollbar styling added (`styles.py`) to avoid legacy bright frame artifacts.
- 2026-02-23: Runtime explanatory text now renders per-field via contextual `?` help buttons (`HelpMode.DIALOG`) instead of detached multi-line blocks.
- 2026-02-23: Root app layout now uses `grid` (`notebook` row + fixed footer row) so the footer is preserved in reduced window sizes.
- 2026-02-23: Diagnostics tab now renders inside `ScrollableFrame`, preventing advanced controls from clipping/forcing footer loss on small windows.
- 2026-02-23: `ScrollableFrame` now binds wheel events to descendant widgets (not just canvas enter/leave), fixing mouse-wheel scroll over content controls when scrollbars are present.
- 2026-02-24: Runtime allocator UI now binds only `PYTORCH_CUDA_ALLOC_CONF` + `CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF`.
- 2026-04-01: `tabs/manual_env_vars.py` now owns the API-only manual env overlay toggle, editor, and validation together; `tabs/runtime.py` no longer mirrors that feature.
- 2026-03-01: Diagnostics trace controls are split into explicit categories (`CODEX_TRACE_INFERENCE_DEBUG`, `CODEX_TRACE_LOAD_PATCH_DEBUG`, `CODEX_TRACE_CALL_DEBUG`) with dedicated call-trace cap key (`CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC`).
- 2026-03-01: Footer `Show advanced controls` toggle now persists immediately in launcher meta (`show_advanced_controls`) and restores on next launcher start.
- 2026-04-01: `tabs/services.py` now owns launcher next-start service prefs (`external_terminal`, `frontend_dev_typecheck`) as working saved-state controls, and Start/Restart always use the last saved launcher config instead of unsaved edits.
- 2026-04-02: `controller.py` now caches committed launcher state between save/reload/start boundaries, keeps running API URLs tied to live service-handle host/port truth, resolves repo-root UI pid receipts against the active launcher-owned UI instance token when multiple same-repo fallback receipts exist, and ignores invalid saved Manual Env Vars overlays only for Services preview/open so the poll loop stays healthy while start/save validation remains fail-loud.
