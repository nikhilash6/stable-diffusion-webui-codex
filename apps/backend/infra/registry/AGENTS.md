<!-- tags: backend, infra-registry, assets, text-encoders -->
# apps/backend/infra/registry Overview
Date: 2025-12-05
Last Review: 2026-05-17
Status: Active

## Purpose
- Host the remaining lightweight infra registry helpers for textual-inversion embeddings and configured text-encoder roots, safe to import without pulling heavy engine/runtime modules.

## Key Files
- `base.py` — Shared `AssetEntry` dataclass for active registry entries.
- `embeddings.py` — Textual inversion (TI) discovery + metadata.
- `text_encoder_roots.py` — Engine text encoder roots registry (per-family paths and stable labels).

## Notes
- These registries must stay thin: no engine loading and no public contract ownership. They are intended for API listings and diagnostics around the remaining embedding/text-encoder-root surfaces.
- 2025-12-05: Text encoder roots per engine (`*_tenc` in `apps/paths.json`) are now wired into the inventory layer (`apps/backend/inventory/cache.py`); future registry helpers for per-family TE overrides should live here, built on top of `get_paths_for("<engine>_tenc")`.
- 2026-01-04: Removed direct `apps/paths.json` reads in registries; use `apps/backend/infra/config/paths.py:get_paths_for` instead to keep repo-root expansion/dedup consistent.
- 2026-01-04: LoRA/VAE/WAN22 GGUF discovery uses the shared inventory scanners (`apps/backend/inventory/scanners/*`) so `/api/models/inventory` is the single source of truth for roots/extension policy.
- 2026-01-04: Tokenizer discovery now uses the shared vendored HF scanner (`apps/backend/inventory/scanners/vendored_hf.py`) to keep traversal/sorting consistent with inventory metadata.
- 2025-12-29: Added `zimage_tenc` to the text encoder roots registry so ZImage text-encoder roots show up in diagnostics and inventory-adjacent listings.
- 2025-12-29: Text encoder root labels (`TextEncoderRoot.name`) now prefer repo-relative paths when roots live under `CODEX_ROOT` (keeps override labels stable and avoids leaking absolute host paths).
- 2026-01-02: Added standardized file header docstrings to `base.py`, `embeddings.py`, `text_encoder_roots.py`, and package `__init__.py` (doc-only change; part of rollout).
- 2026-02-05: `text_encoder_roots.py` now maps `ModelFamily.ANIMA` to `apps/paths.json["anima_tenc"]` so registry listings cover Anima text-encoder roots.
- 2026-02-16: `text_encoder_roots.py` now exposes explicit WAN22 variant families (`WAN22_5B`, `WAN22_14B`, `WAN22_ANIMATE`) mapped to `wan22_tenc` roots.
- 2026-03-12: `text_encoder_roots.py` now maps `ModelFamily.LTX2` to `apps/paths.json["ltx2_tenc"]` so diagnostics and inventory-adjacent listings reflect the dedicated LTX2 text-encoder root.
- 2026-05-11: LoRA/VAE public discovery is owned by `apps/backend/inventory/scanners/*`, `/api/models/inventory`, and `apps/backend/runtime/models/registry.py` as applicable.
- 2026-05-17: `text_encoder_roots.py` maps `ModelFamily.QWEN_IMAGE` to `apps/paths.json["qwen_image_tenc"]` so Qwen Image text-encoder override labels stay family-scoped.
