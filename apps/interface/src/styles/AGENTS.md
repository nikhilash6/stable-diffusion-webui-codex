<!-- tags: frontend, styles, tokens, conventions -->
# apps/interface/src/styles Overview
Date: 2025-12-22
Last Review: 2026-03-29
Status: Active

## Purpose
- Own the shared frontend stylesheet tree for `apps/interface`.
- Keep styling predictable: tokens first, shared primitives second, feature/view CSS last.

## Runtime truth
- `apps/interface/src/main.ts` imports `apps/interface/src/styles.css`; this is the only runtime CSS bootstrap path.
- `apps/interface/src/styles.css` is the stylesheet root for runtime CSS modules.
- Tailwind v4 is wired through `@tailwindcss/vite`, and `styles.css` starts with `@import 'tailwindcss';`.
- `apps/interface/src/styles/EXAMPLE-*` files are reference-only and are not runtime owners.

## Authoring rules
- Put app-wide tokens and shared primitives in `apps/interface/src/styles.css`.
- Put reusable component CSS under `apps/interface/src/styles/components/*.css`.
- Put view-owned CSS under `apps/interface/src/styles/views/*.css`.
- Wire new runtime stylesheets through `apps/interface/src/styles.css`; do not create side-entry CSS paths.
- New shared semantic classes should prefer the `cdx-` prefix; established feature prefixes may continue where already owned.
- Do not add inline styles or ad-hoc DOM style writes as a convenience escape hatch.
- Avoid new SFC `<style>` / `<style scoped>` blocks for application UI styling; prefer `src/styles/**`.

## Validation
- `cd apps/interface && npm run verify:css-contracts` — only direct CSS-contract gate.
- `cd apps/interface && npm run verify` — wrapper-only frontend validation entrypoint.

## Contract authority
- Exact runtime topology, budgets, and typed exception ownership do not live in this file.
- See `apps/interface/tools/style-topology.mjs`, `apps/interface/tools/css-contracts.config.json`, and `.sangoi/reference/ui/frontend-css-contracts.md`.
