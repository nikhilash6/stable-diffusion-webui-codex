# apps.launcher.gui_tk
Date: 2026-01-25
Status: Active
Last Review: 2026-05-18

## Purpose
- Modular Tk/ttk GUI implementation for the Codex launcher (services, bootstrap/engine/safety settings, manual env vars, diagnostics, logs).
- Keep the stable entrypoint at `apps/codex_launcher.py` thin; implementation lives here.

## Modules
- `__init__.py` - Public GUI entrypoint exports (`CodexLauncherApp`, `main`).
- `app.py` - Tk root window, tab registry wiring, background task polling, save/revert/exit flow, and persisted UI state.
- `tab_registry.py` - Ordered tab lifecycle registry for add/reload/refresh/advanced/dispose fan-out.
- `controller.py` - Non-UI controller around profile store, service handles, log buffer, and persistence helpers.
- `styles.py` - Palette, cross-platform fonts, ttk styles, and VRAM badge styles.
- `form_schema.py` - Declarative form descriptor models, including field-level visible VRAM impact metadata.
- `form_renderer.py` - Shared descriptor-driven form renderer used by tabs for consistent layout, help buttons, advanced visibility, and VRAM badges.
- `widgets.py` - Scrollable container and small layout helpers.
- `tabs/services.py` - API/UI supervision tab plus launcher-owned next-start service prefs.
- `tabs/bootstrap.py` - Main/mount/offload device selectors and attention mode.
- `tabs/engine.py` - CFG batching, GGUF/LoRA/WAN chunk buffer, and PyTorch allocator settings.
- `tabs/safety.py` - Task single-flight, cancellation, replay-buffer caps, and safeweights settings.
- `tabs/manual_env_vars.py` - API-only manual env overlay toggle/editor/validation.
- `tabs/diagnostics_sections.py` - Declarative Diagnostics tab section inventory.
- `tabs/diagnostics.py` - Environment checks plus descriptor-rendered diagnostics/debug/tracing/profiler/logging controls.
- `tabs/logs.py` - Log viewer (filter/search/export).

## Notes
- UI state auto-persisted via `LauncherProfileStore.save_meta()` is limited to tab index, window geometry, and advanced-controls visibility; launcher next-start service prefs plus env changes require explicit `Save Settings`.
- Runtime settings are top-level tabs: `Bootstrap`, `Engine`, and `Safety`; do not reintroduce a string-parameterized multi-tab runtime controller.
- `Diagnostics` is a troubleshooting surface; runtime behavior selectors such as CFG batching belong in `Engine`.
- Any control that materially changes VRAM behavior should pass `vram_metadata_for_key(...)` into its `FormFieldDescriptor` so the UI renders a direct `VRAM: LOW|MED|HIGH` badge.
- Raw `tk.Text` surfaces must be themed explicitly and use `styles.resolve_fonts(...).mono`; do not hardcode Windows-only monospace fonts in tabs.
- The global `Show advanced controls` footer toggle drives registry participants through `LauncherTabRegistry`.
