# apps Overview
Date: 2025-10-28
Last Review: 2026-05-02
Status: Active

## Purpose
- Host all active application code for the Codex rebuild. Each top-level package under `apps/` owns a distinct runtime surface (backend services, new Vue interface, launcher tooling).

## Subdirectories
- `backend/` — Codex-native backend stack (engines, runtimes, services, registries, HF asset mirrors). This is the authoritative implementation.
- `interface/` — Vue 3 + Vite application that replaces the legacy Gradio UI. Includes build tooling, public assets, and source modules.

## Key Files
- `launcher/` — Package exposing launcher infrastructure (`checks`, `services`, `profiles`, `paths`).
- `codex_launcher.py` — Tk-based GUI launcher entrypoint for managing API/UI services (Windows). Implementation lives in `launcher/gui_tk.py`.
- `docker_tui_launcher.py` — Docker-oriented terminal launcher entrypoint (interactive TUI + profile persistence + `run-webui.sh` delegation).
- `__init__.py` — Marks `apps` as a Python package so relative imports resolve cleanly across backend modules.

## Notes
- New code must target `apps/backend` and `apps/interface`. The launcher infrastructure lives under `apps/launcher/`; the GUI launcher entrypoint is `apps/codex_launcher.py` (Windows).
- Docker terminal launches should use `apps/docker_tui_launcher.py` (called by `run-webui-docker.sh`) so runtime env/profile configuration remains centralized in `LauncherProfileStore`.
- The GUI code should live under `apps/launcher/` so the entrypoint stays stable while the implementation evolves.
- 2026-01-20: Deprecated the curses TUI launcher (`apps/tui_launcher.py` + `run-tui.bat`) by moving it to repo-local `/.deprecated/` (ignored by Git).
- When adding new subpackages, create an `AGENTS.md` describing responsibilities to keep this overview accurate.
- 2026-02-28: Launcher bootstrap authority is `CODEX_MAIN_DEVICE`/`--main-device`; launcher services mirror it to `--core-device/--te-device/--vae-device` for parity and keep legacy per-component keys only as fallback input.
- 2026-02-28: Launcher runtime defaults persist in the launcher profile store (`.sangoi/launcher/profiles/*`); `run-webui.sh` also reads fallback device defaults from `apps/settings_values.json` (`codex_main_device` + component keys).
- 2026-02-28: Launcher/API startup actively forwards `CODEX_*` bootstrap/runtime knobs and allocator keys (for example `CODEX_MAIN_DEVICE`, `CODEX_MOUNT_DEVICE`, `CODEX_OFFLOAD_DEVICE`, `PYTORCH_CUDA_ALLOC_CONF`) instead of stripping them.
- 2026-02-28: Manual API env overlay (`manual_api_env_enabled`) is a GUI-launcher API-start feature; shell launcher paths (`run-webui.sh`) do not consume the launcher manual overlay text.
- 2026-05-02: Unset launcher LoRA apply mode uses `CODEX_LORA_APPLY_MODE=online`; explicit `merge` remains valid when a run should rewrite weights once at apply-time. Backend CLI also supports `--lora-apply-mode`.
- 2026-03-01: Launcher no longer exposes GGUF exec mode controls; GGUF runtime stays on forward dequantization by default and only LoRA online-math remains as an explicit (reserved/fail-loud) bootstrap flag.
- 2026-01-21: Launchers can set `PYTORCH_CUDA_ALLOC_CONF` to tune the PyTorch CUDA caching allocator (e.g. reduce fragmentation via `max_split_size_mb`).
- 2026-01-21: Launcher default for `PYTORCH_CUDA_ALLOC_CONF` is `max_split_size_mb:256,garbage_collection_threshold:0.8` (override via GUI or `./run-webui.sh --pytorch-cuda-alloc-conf ...`).
- 2025-11-03: Launcher UI exposes "Conditioning Debug" toggle, wiring to `CODEX_DEBUG_COND` and the backend `--debug-conditioning` flag.
- 2025-11-03: (Deprecated) "Pin Shared Memory" launcher toggle removed; `--pin-shared-memory` remains a CLI-only switch (no env overrides).
- 2025-11-03: Logging tab now includes a "Trace Debug" toggle that sets `CODEX_TRACE_DEBUG=1`, enabling the global call tracer behind `--trace-debug`.
- 2025-11-25: Launcher logging/debug UI gained sampler diagnostics toggles: `CODEX_LOG_SAMPLER` (per-step norms) and `CODEX_LOG_SIGMAS` (sigma ladder dump).
- 2026-01-02: Launchers now expose `CODEX_LOG_CFG_DELTA` (and `CODEX_LOG_CFG_DELTA_N`) to log the cond/uncond delta inside CFG for the first N steps (requires `CODEX_LOG_SAMPLER=1`).
- 2025-11-14: Launcher debug UI mirrors backend defaults for `CODEX_TRACE_DEBUG_MAX_PER_FUNC` (10 by default) so the displayed values stay in sync with `apps.backend.infra.config.args`.
- 2025-12-03: (Deprecated) "Force Native Sampler" launcher toggle removed; sampler routing is configured via Web UI / payload (no env overrides).
- 2025-12-28: `apps/settings_values.json` and `apps/interface/{tabs,workflows}.json` are backend-managed runtime state files; they are created/overwritten locally and are intentionally ignored by Git.
- 2025-12-29: Repo-root resolution across backend + launchers is now strict and `CODEX_ROOT`-anchored (no `__file__`/CWD fallbacks); launch via `run-webui.{bat,sh}` or set `CODEX_ROOT` explicitly.
- 2026-01-03: Added standardized file header docstrings to the remaining low-core `apps/` entrypoints (`__init__.py`, `backend/__init__.py`, and WebUI entrypoints/config) (doc-only change; part of rollout).
- 2026-02-05: Added Anima model roots to `apps/paths.json` (`anima_ckpt`, `anima_tenc`, `anima_vae`, `anima_loras`) to mirror existing per-family directory conventions.
- 2026-03-12: Added LTX2 model roots to `apps/paths.json` (`ltx2_ckpt`, `ltx2_tenc`, `ltx2_vae`, `ltx2_connectors`, `ltx2_loras`) so the backend provisions repo-local `models/ltx2*` folders for the current GGUF+side-asset intake lane.
- 2026-03-31: Env-var hygiene contract: family-prefixed env vars belong only to that family's owner seams; shared runtime features must use shared feature prefixes (for example sampling, sampler-model, IP-Adapter) instead of piggybacking on a model-family namespace.
