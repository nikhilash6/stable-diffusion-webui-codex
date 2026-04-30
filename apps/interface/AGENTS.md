<!-- tags: frontend, interface, overview -->
# apps/interface Overview
Date: 2025-10-28
Last Review: 2026-04-02
Status: Active

## Purpose
- Hosts the Codex Vue 3 + Vite frontend application that replaces the legacy Gradio UI.

## Subdirectories
- `src/` — TypeScript/Vue source code (components, views, stores, styles, API client).
- `public/` — Static assets served as-is (favicons, manifest, etc.).
- `tools/` — Developer tooling scripts (port guard helpers, verification tooling).

## Key Files
- `package.json` / `tsconfig.json` / `vite.config.ts` — Build and tooling configuration.
- `src/main.ts` / `src/styles.css` — Runtime CSS bootstrap (`main.ts -> styles.css`) and stylesheet root.
- `blocks.json` — Server-driven UI definition synced with backend.
- `presets.json` — UI presets served by the backend `/api/ui/presets` endpoint (source of truth for preset IDs/options).

## Notes
- Run `npm run dev` from this directory for local development; backend expects the build artifacts emitted by Vite.
- Keep source structure consistent with the guidance in `.sangoi/frontend/guidelines/`.
- CSS contract authority split:
  - `npm run verify:css-contracts` is the only direct CSS gate.
  - `npm run verify` is wrapper-only and chains CSS + type/build validation.
  - Detailed CSS contract truth lives in `.sangoi/reference/ui/frontend-css-contracts.md`; support docs here stay pointer-only.
- 2025-12-29: `vite.config.ts` ignores backend-persisted `tabs.json`/`workflows.json` changes to prevent Vite full-reloads during dev toggles.
- 2025-12-29: `tools/port-guard-dev.mjs` now checks IPv4+IPv6 bind targets (0.0.0.0/127.0.0.1/::/::1) to avoid localhost split-brain; when the base port is busy it probes `/api/version` to warn about an existing Codex instance (WSL/Windows) and writes repo-root `.webui-ui-<port>.pid` files for launcher/debugging truth.
- 2025-11-14: API requests are built via `src/api/payloads.ts` (Zod schemas) — payload builders trim prompts and always attach the per-tab engine/model metadata (even for img2img).
- 2025-12-03: Txt2Img prompt schema now rejects empty prompts at the frontend (`PromptSchema`), surfacing a validation error instead of silently sending `prompt=""` to the backend.
- 2026-02-28: Frontend follows root testing policy: manual validation by default; automated/unit tests are not maintained unless explicitly requested by the repo owner.
- 2026-03-03: `vite.config.ts` default `allowedHosts` baseline now includes only local loopback hosts (`localhost`, `127.0.0.1`, `::1`); extra hosts must be provided explicitly through `ALLOWED_HOSTS`.
- 2026-01-01: Updated `apps/interface/README.md` to reflect the repo-local `.venv` (and `run-webui.sh` as the recommended dev entrypoint).
- 2026-01-01: Added a branded `public/favicon.ico` and referenced it from `index.html` so the browser tab icon matches the project branding.
- 2026-01-03: Added standardized file header blocks to WebUI entrypoints/config (`vite.config.ts`, `src/{App,main,router}.ts/.vue`, `src/api/types.ts`) (doc-only change; part of rollout).
- 2026-01-21: Updated `blocks.json` WAN22 stage contract to `model_sha` + `loras[]` (JSON array entries `{sha, weight}`); legacy stage keys `lora_sha`/`lora_weight` are not accepted by backend WAN routes.
- 2026-01-23: WAN video dimensions now snap to multiples of 16 (rounded up; Diffusers parity) in the UI and payload builders to avoid backend 400s and silent patch-grid cropping.
- 2026-01-27: `package-lock.json` updated to match npm 11 (used by the repo-local `.nodeenv` installer) to avoid lockfile churn on fresh installs.
- 2026-02-06: Added `vue-tsc` typechecking via `npm run typecheck` so Vue SFC drift fails loud when you explicitly ask for the type gate.
- 2026-04-01: Frontend dev boot now splits into `npm run dev:fast` and `npm run dev:typecheck`; `npm run dev` defaults to `dev:fast`, while launcher-owned UI-service boot policy chooses between the two via launcher meta (Services tab), not a runtime env var.
- 2026-04-02: `tools/port-guard-dev.mjs` remains the owner of repo-root `.webui-ui-<port>.pid` receipts for UI fallback ports; when launcher starts the UI service it also passes a per-start `CODEX_LAUNCHER_UI_INSTANCE_TOKEN`, and launcher endpoint/open resolution must read those repo-root pid files instead of `apps/interface` while preferring the receipt that matches the active launcher-owned token.
- 2026-02-08: SDXL swap-model UI contract now uses explicit pointer semantics (`swapAtStep` in frontend state, serialized as `switch_at_step` in API payloads), replacing refiner step-count wording/behavior.
- 2026-03-05: FLUX.2 frontend requests now target the backend-owned Klein 4B / base-4B slice only: the UI keeps `flux2` first-class, uses one `Qwen3-4B` selector, and no longer aliases FLUX.2 img2img into FLUX.1 Kontext.
