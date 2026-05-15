# apps/interface/tools/css-contracts Overview
Date: 2026-03-29
Last Review: 2026-03-29
Status: Active

## Purpose
- Internal helper modules for the frontend CSS contract verifier.

## Key files
- `config.mjs` — Loads and validates `css-contracts.config.json`.
- `fs-utils.mjs` — Shared repo-path, IO, and line/column helpers.
- `topology.mjs` — Verifies `main.ts -> styles.css -> style-topology.mjs -> src/styles/**` parity.
- `source-analysis.mjs` — Scans Vue/TS source for class consumers, `:style`, scoped styles, and DOM/CSS-variable writes.
- `css-analysis.mjs` — Parses runtime CSS with PostCSS and reports selector/variable/declaration drift.
- `report.mjs` — Builds the JSON report and final fail-loud status.

## Notes
- This folder has no public CLI surface; the only public owner is `apps/interface/tools/verify-css-contracts.mjs`.
- Keep the helper split flat by concern. Do not introduce internal framework layers, compat wrappers, or second public entrypoints.
