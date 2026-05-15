# apps.launcher
Date: 2025-10-28
Status: Active
Last Review: 2026-04-02

## Purpose
- Provide reusable launcher infrastructure (path resolution, environment checks, service supervision, segmented profile persistence) for Codex entrypoints.

## Modules
- `paths.py` – Resolve canonical data/model/output directories with strict normalisation.
- `log_buffer.py` – Thread-safe ring buffer for capturing launcher logs shared across services and UI.
- `checks.py` – Environment validation (Python, Node/npm, Vite) with explicit diagnostics.
- `services.py` – Service specifications and process supervision helpers for API/UI processes.
- `profiles.py` – Segmented profile persistence (`.sangoi/launcher/`) with area/model separation and legacy migration support.
- `settings.py` – Typed launcher settings + validation helpers (env-backed, UI/service friendly).
- `gui_tk/` – Tk-based GUI launcher package (UI around profiles/checks/services).
- `__init__.py` – Re-exports public APIs for callers (`CodexPaths`, `run_launch_checks`, `LauncherProfileStore`, etc.).

## Notes
- New launcher features must surface through this package; avoid ad-hoc scripts accessing internal modules directly.
- Persistence writes to `.sangoi/launcher/{meta,areas,models}`; migrations should extend `profiles` rather than duplicating logic.
- Service command definitions should remain minimal and composition-friendly—prefer adding options via profile/env rather than bespoke subprocess code.
- The Tk GUI uses `logo.png` for `iconphoto` and (on Windows) `logo.ico` for `iconbitmap` so the taskbar icon is branded.
- Launcher meta (`.sangoi/launcher/meta.json`) stores `window_geometry` so the Tk GUI can restore size/position across runs.
- Launcher meta (`.sangoi/launcher/meta.json`) stores `window_geometry` and `show_advanced_controls` so the Tk GUI restores both window layout and advanced-controls visibility across runs.
- 2025-11-02: Windows “Services in new terminal” now wraps commands with `cmd.exe /K` and leaves stdin attached so the console stays open after exit for manual inspection.
- 2025-11-02: Launcher profiles persist diffusion/TE/VAE device + dtype choices via the Codex options snapshot, and `services.py` now forwards them as CLI flags (`--core-device`, `--te-device`, `--vae-device`, etc.) when spawning the API instead of relying on env vars.
- 2025-11-03: Launcher forwards conditioning diagnostics via `--debug-conditioning` when `CODEX_DEBUG_COND` is enabled in profiles/TUI.
- 2025-12-29: Launcher now resolves the repo root via `CODEX_ROOT` (shared helper) instead of `Path(__file__).parents[...]`, so Windows/WSL launch methods stay consistent.
- 2025-12-29: Launcher UI service now always receives `API_PORT` (prevents Vite proxy/API_PORT derivation from a fallback WEB_PORT), and the API service performs a strict preflight port check across IPv4/IPv6 localhost (helps diagnose WSL/Windows double-run and “localhost” split-brain).
- 2026-03-28: `services.py` now owns launcher-started API fallback before spawn (`7850 -> 17850 -> 27850`, or the same arithmetic chain for explicit overrides), stores the chosen API port as runtime truth only while the API handle is active, and clears that truth on stop/kill/unexpected exit so later UI/docs resolution cannot reuse stale state.
- 2026-03-01: Launcher no longer persists/forwards GGUF exec mode knobs (`CODEX_GGUF_EXEC`, `--gguf-exec`); GGUF runtime policy is fixed to forward dequantization and launcher only forwards LoRA runtime flags (`--lora-apply-mode`, `--lora-online-math`).
- 2026-01-24: Launcher profiles include explicit device defaults (`CODEX_CORE_DEVICE`, `CODEX_TE_DEVICE`, `CODEX_VAE_DEVICE`) and `services.py` forwards them to the backend as CLI flags (`--core-device`, `--te-device`, `--vae-device`) to avoid bootstrap-time fallback/prompt failures in non-interactive spawns (and profile consistency keeps these keys, no accidental pruning).
- 2026-01-02: Added standardized file header docstrings to launcher modules (doc-only change; part of rollout).
- 2026-01-03: Added standardized file header docstrings to remaining launcher modules (`__init__.py`, `checks.py`, `log_buffer.py`, `paths.py`) (doc-only change; part of rollout).
- 2026-01-06: Launcher Python preflight now matches `.python-version` (3.12.10) instead of allowing stale 3.10/3.11.
- 2026-01-21: Launcher profiles now default `PYTORCH_CUDA_ALLOC_CONF` (global PyTorch CUDA allocator tuning) to `max_split_size_mb:256,garbage_collection_threshold:0.8` when unset.
- 2026-03-01: Legacy launcher profiles containing `CODEX_GGUF_EXEC` are sanitized during runtime-env normalization (stale key removed, no compatibility shim).
- 2026-01-31: Launcher profiles now persist global profiling env flags (`CODEX_PROFILE*`) and the GUI diagnostics tab exposes them for backend torch-profiler runs.
- 2026-02-15: Launcher API arg forwarding now includes trace toggles (`CODEX_TRACE_CONTRACT` -> `--trace-contract`, `CODEX_TRACE_PROFILER` -> `--trace-profiler`) for backend bootstrap alignment.
- 2026-02-18: Launcher task/runtime profile defaults now persist `CODEX_TASK_CANCEL_DEFAULT_MODE` (`immediate|after_current`) as a backend bootstrap knob for task cancel policy.
- 2026-02-21: Launcher profiles/settings now persist and validate `CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE` (`hybrid|ram|ram+hd`) as a runtime bootstrap knob used by WAN22 img2vid chunk buffering policy.
- 2026-02-21: Launcher Runtime now owns attention bootstrap policy via `CODEX_ATTENTION_BACKEND` + `CODEX_ATTENTION_SDPA_POLICY`, forwarding `--attention-backend` and `--attention-sdpa-policy` to backend startup.
- 2026-02-22: Launcher profiles now write `PYTORCH_CUDA_ALLOC_CONF` for allocator tuning defaults.
- 2026-02-22: Removed GGUF dequant-forward run cache forwarding from launcher bootstrap args (`services.py` no longer emits `--gguf-dequant-cache*` flags); runtime env normalization now forces `CODEX_GGUF_DEQUANT_CACHE=off` and clears stale ratio/limit keys.
- 2026-02-23: Launcher now defines a global device authority via `CODEX_MAIN_DEVICE`; `services.py` forwards `--main-device` and mirrors core/TE/VAE flags to the same value to enforce single-device runtime invariant.
- 2026-02-23: `profiles.py` now treats `CODEX_*` runtime/device keys as area-scoped only (`core`): model overlays and non-core areas can no longer override `CODEX_MAIN_DEVICE`/`CODEX_MOUNT_DEVICE`/`CODEX_OFFLOAD_DEVICE` (prevents stale model JSON from defeating saved runtime-tab device settings).
- 2026-02-23: Launcher entrypoints now enforce `PYTORCH_CUDA_ALLOC_CONF` allocator contract keys and drop unsupported `*_ALLOC_CONF` variants from process env before backend spawn.
- 2026-02-23: `profiles.py` now sanitizes persisted allocator/profile env keys at load (drops unsupported `*_ALLOC_CONF` variants), backfills missing required defaults, and persists normalized env maps automatically when cleanup occurs.
- 2026-02-23: `profiles.py` save now prunes stale `areas/*.json`/`models/*.json` files not present in normalized mappings, so removed legacy areas (e.g. `wan`) are deleted from disk and do not trigger repeated load-time rewrites.
- 2026-02-23: launcher offload default is now explicit CPU: `services.py` forwards `--offload-device=cpu` when unset, and `profiles.py` defaults `CODEX_OFFLOAD_DEVICE=cpu` to avoid implicit same-device offload no-op states under Contract-R unload semantics.
- 2026-02-23: `services.py` now enforces `PYTORCH_CUDA_ALLOC_CONF` allocator backend when `CODEX_CUDA_MALLOC=1` (requires/ensures `backend:cudaMallocAsync`, fail-loud on conflicting backend entries).
- 2026-02-23: `services.py` now sanitizes unsupported allocator env keys (`PYTORCH_*_ALLOC_CONF` / `CODEX_ENABLE_DEFAULT_PYTORCH_*_ALLOC_CONF` variants) before subprocess spawn to keep runtime contract strict (`PYTORCH_CUDA_ALLOC_CONF` + `CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF` only).
- 2026-02-23: `profiles.py` now persists only canonical allocator keys in `.sangoi/launcher/*` and prunes unsupported allocator variants during normalization.
- 2026-02-23: `services.py` allocator contract sanitization now rejects unsupported allocator variants before spawn.
- 2026-02-23: `run-webui.sh` now makes `--cuda-malloc` / `CODEX_CUDA_MALLOC=1` effective by ensuring `PYTORCH_CUDA_ALLOC_CONF` includes `backend:cudaMallocAsync` (and failing loud on invalid/conflicting allocator config).
- 2026-02-23: Tk launcher UI (`gui_tk/`) added descriptor-driven form infrastructure (`form_schema.py`, `form_renderer.py`) and services overview upgrades (resolved endpoint/open actions, no health polling/status row) to keep launcher UX maintainable while staying a bootstrap orchestrator.
- 2026-02-23: `profiles.py` now guards env map reads/writes with an internal `RLock` to keep launcher env snapshots deterministic under concurrent GUI/background access.
- 2026-02-24: Launcher allocator contract is `PYTORCH_CUDA_ALLOC_CONF` + `CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF` (`run-webui.sh/.bat`, `profiles.py`, `services.py`, Tk runtime tab).
- 2026-02-25: Launcher profile meta now persists API-only manual env overlay settings (`manual_api_env_enabled`, `manual_api_env_text`), with strict `KEY=VALUE` parsing and fail-loud startup errors when invalid.
- 2026-03-01: Launcher core defaults now include split trace env knobs (`CODEX_TRACE_INFERENCE_DEBUG`, `CODEX_TRACE_LOAD_PATCH_DEBUG`, `CODEX_TRACE_CALL_DEBUG`, `CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC`) and the Tk diagnostics tab maps directly to these explicit keys.
- 2026-03-03: `profiles.py` now uses a single canonical default provider for `external_terminal` (enabled on Windows, disabled elsewhere) and applies it consistently for new profiles plus missing-key fallback paths (`LauncherMeta`, `_load_meta`, `_maybe_migrate_legacy`) while preserving explicit persisted values.
- 2026-03-31: Launcher-owned envs are runtime-global/bootstrap selectors only. Do not introduce family-specific debug/feature env names into launcher core defaults unless the launcher truly owns that runtime concept.
- 2026-04-01: Frontend dev typecheck boot policy is launcher-meta-owned (`frontend_dev_typecheck`), not a runtime env var. Launcher UI-service boot must choose between `apps/interface` scripts `dev:fast` and `dev:typecheck` from persisted meta, and manual terminal use stays script-driven.
- 2026-04-01: Launcher service starts/restarts now consume the last saved profile snapshot from disk, not unsaved working edits. API-only manual env overlay text is validated at save time, and UI/API endpoint truth is controller-owned from committed-vs-live service state.
- 2026-04-02: `controller.py` now caches the committed launcher profile between save/reload/start boundaries (no per-poll disk reload), running API URLs use service-handle live host/port truth instead of future saved overlay hosts, and running UI URLs resolve repo-root `.webui-ui-*.pid` files for port-guard fallback ports while matching the active launcher-owned UI instance token when multiple same-repo receipts exist. Invalid saved Manual Env Vars overlays are ignored only for URL preview/open while save/start/restart remain fail-loud.
