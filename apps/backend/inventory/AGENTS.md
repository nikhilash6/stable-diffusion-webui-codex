<!-- tags: backend, inventory, models, huggingface, wan22 -->

# apps/backend/inventory Overview
Date: 2025-10-28
Last Review: 2026-03-05
Status: Active

## Purpose
- Provides lightweight helpers for building and caching backend inventories (used during audits, module parity tracking, and diagnostics).

## Key Files
- `cache.py` — Dataclasses and helpers for persisting inventory snapshots to disk.
- `scanners/` — Import-light filesystem scanners shared between inventory and UI-facing registries (single source of truth for roots/exts).

## Notes
- When adding new inventory schemas, extend `cache.py` or add adjacent modules here so reporting logic stays centralized.
- Reference: `.sangoi/reference/models/model-assets-selection-and-inventory.md` documents the end-to-end contract (inventory → SHA selection → backend resolution).
- 2025-11-30: Inventory Hugging Face root now points at `apps/backend/huggingface` and uses a correct repo-root calculation, keeping metadata listings aligned with WAN22 engines and UI expectations.
- 2025-12-04: WAN22 inventory uses explicit GGUF roots from `apps/paths.json` (`wan22_ckpt`) with high/low stage detection preserved.
- 2025-12-04: Text encoder inventory uses per-engine roots from `apps/paths.json` (`sd15_tenc`, `sdxl_tenc`, `flux1_tenc`, `flux2_tenc`, `wan22_tenc`, `zimage_tenc`), deduplicating by path so engine-specific TEnc folders show up once under `/api/models/inventory`.
- 2025-12-04: Inventory also appends engine-specific VAEs from `apps/paths.json` (`flux1_vae`, `flux2_vae`, `zimage_vae`) so Flux/ZImage dropdowns can list concrete files without filename heuristics.
- 2025-12-06: Model inventory is now pre-warmed during backend bootstrap (`_bootstrap_runtime` calls `inventory.cache.refresh()`), so the first `/api/models/inventory` request no longer pays the full filesystem scan cost on demand.
- 2025-12-29: Inventory repo root now prefers `CODEX_ROOT` so scans don’t depend on the backend process CWD.
- 2026-01-04: Inventory now builds VAEs/LoRAs/text encoders/WAN22 GGUF via shared scanners (`inventory/scanners/*`) so `/api/models/inventory` and registries don’t drift on roots/extension policy.
- 2026-01-04: Vendored HF `{org}/{repo}` traversal was centralized in `inventory/scanners/vendored_hf.py` and reused for inventory metadata (keeps metadata/tokenizer listings consistent).
- 2026-01-03: Added standardized file header docstring to `cache.py` (doc-only change; part of rollout).
- 2026-01-21: Inventory now requires `sha256` for file assets; `cache.py` falls back to direct hashing when the registry cache fails and clears the sha→path cache on `init()/refresh()`.
- 2026-02-11: `cache.py` now exposes `resolve_vae_path_by_sha(...)` with a VAE-only SHA→path cache (`_SHA_TO_VAE_PATH`) so API contracts can reject non-VAE assets passed via `extras.vae_sha` before runtime load.
- 2026-03-12: Inventory discovery now includes `ltx2_tenc`, `ltx2_vae`, and `ltx2_loras` roots; generic inventory intentionally filters `mmproj` out of text-encoder listings and filters `audio_vae` bundles out of VAE listings so the real LTX 2.3 split pack does not get mislabeled.
