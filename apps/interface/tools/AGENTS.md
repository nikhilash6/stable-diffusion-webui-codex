# apps/interface/tools Overview
Date: 2025-10-28
Last Review: 2026-04-02
Status: Active

## Purpose
- Contains scripts/utilities that support frontend development and contract validation.

## Key Files
- `port-guard-dev.mjs` — Dev server launcher wrapper that enforces safe/available UI ports.
- `verify-css-contracts.mjs` — Frontend CSS-contract verifier and owner behind `npm run verify:css-contracts`.
- `style-topology.mjs` — Ordered runtime CSS topology declaration for `apps/interface`.
- `css-contracts.config.json` — CSS-contract budgets and typed exception ownership.

## Notes
- Keep these scripts in sync with repository-shipped frontend docs (`apps/interface/README.md`, `apps/interface/AGENTS.md`, `apps/interface/src/styles/AGENTS.md`) so developers know how to invoke them.
- 2026-04-02: `port-guard-dev.mjs` must launch Vite through the explicit repo-local Node entrypoint (`node_modules/vite/bin/vite.js`) instead of `shell: true` + `.cmd` wrappers, so Windows dev boot stays warning-free and argument handling remains explicit.
- `npm run verify:css-contracts` is the only direct CSS gate.
- `npm run verify` is wrapper-only.
- Detailed topology, budgets, and typed exceptions live in `style-topology.mjs`, `css-contracts.config.json`, and `.sangoi/reference/ui/frontend-css-contracts.md`; this file stays pointer-only.
