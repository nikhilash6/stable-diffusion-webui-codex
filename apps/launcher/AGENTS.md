# apps.launcher
Date: 2025-10-28
Status: Active
Last Review: 2026-05-18

## Purpose
- Provide reusable launcher infrastructure for Codex entrypoints: path resolution, environment checks, service launch preparation/supervision, segmented profile persistence, and shared launcher setting metadata.

## Modules
- `paths.py` - Resolve canonical data/model/output directories with strict normalisation.
- `log_buffer.py` - Thread-safe ring buffer for capturing launcher logs shared across services and UI.
- `checks.py` - Mode-aware environment validation for launcher prerequisites.
- `setting_registry.py` - Canonical launcher setting descriptors, choices, defaults, and visible VRAM impact metadata.
- `profile_meta.py` - Launcher meta shape, manual API env parser, mode-profile constants, and allocator constants.
- `profile_defaults.py` - Default launcher env maps and JSON env-map IO helpers for `.sangoi/launcher/areas` and `.sangoi/launcher/models`.
- `profile_store.py` - `LauncherProfileStore`, env mutation view, consistency cleanup, and legacy migration.
- `service_env.py` - Service env truth parsing and allocator contract enforcement.
- `api_args.py` - Launcher env snapshot to backend API CLI args.
- `ports.py` - API port parsing, fallback chain, and IPv4/IPv6 bind checks.
- `service_modes.py` - Launcher app-mode profile resolution and preflight checks.
- `service_process.py` - `CodexServiceSpec`, `PreparedServiceLaunch`, `prepare_service_launch(...)`, `CodexServiceHandle`, and service lifecycle state.
- `service_specs.py` - Default API/UI service specs and UI dev-service command construction.
- `settings.py` - Typed env-backed wrappers and strict cross-setting normalization helpers.
- `gui_tk/` - Tk-based GUI launcher package.
- `__init__.py` - Package marker only; import concrete launcher owners from their modules.

## Notes
- New shared launcher env settings must be registered in `setting_registry.py`; GUI and Docker TUI settings should consume that registry instead of duplicating choices/defaults.
- Service startup command/env preparation belongs in `prepare_service_launch(...)`; process mutation and `subprocess.Popen(...)` belong in `CodexServiceHandle.start(...)`.
- Profile persistence writes to `.sangoi/launcher/{meta,areas,models}` through `LauncherProfileStore`; env-map defaults come from `setting_registry.py` through `profile_defaults.py`.
- API-only manual env overlay settings live in `profile_meta.py` and the GUI `Manual Env Vars` tab; shell launch paths do not consume that overlay text.
- Launcher device authority is `CODEX_MAIN_DEVICE`; service API args mirror it to core/TE/VAE flags, while mount/offload remain explicit bootstrap selectors.
- Allocator contract keys are `PYTORCH_CUDA_ALLOC_CONF`, `CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF`, and `CODEX_CUDA_MALLOC`.
- `CODEX_CFG_BATCH_MODE=fused|split` is an Engine setting, default `fused`.
- `CODEX_VAE_TENSOR_STATS` and `CODEX_MEMORY_DEBUG` are opt-in Diagnostics settings and default off.
