<!-- tags: backend, inventory, scanners, assets -->
# apps/backend/inventory/scanners Overview
Date: 2026-01-04
Last Review: 2026-03-05
Status: Active

## Purpose
- Shared, import-light filesystem scanners for model assets (VAEs, LoRAs, etc.).
- Provides a single place to define root/extension policies so `inventory/cache.py` and UI-facing registries don’t drift.

## Key files
- `base.py`: Shared helpers for resolving repo/models roots and walking directories.
- `vaes.py`: VAE file discovery policy (per-family `*_vae` roots from `apps/paths.json`).
- `loras.py`: LoRA file discovery policy (per-family `*_loras` roots from `apps/paths.json`).
- `text_encoders.py`: Text encoder weight discovery policy (per-family `*_tenc` roots from `apps/paths.json`).
- `wan22_gguf.py`: WAN22 stage GGUF discovery policy (paths.json `wan22_ckpt`) + stage classifier.
- `vendored_hf.py`: Shared `{org}/{repo}` directory walk for vendored HF roots (tokenizers + metadata inventory).

## Notes
- Scanners must stay lightweight (no torch/transformers imports). Hashing and heavy metadata extraction lives in inventory/cache or runtime.
- Root configuration comes from `apps/backend/infra/config/paths.py:get_paths_for` and `CODEX_ROOT` via `infra/config/repo_root.py`.
- Scanners intentionally ignore ad-hoc files under `models/` (only explicit roots from `apps/paths.json` are scanned).
- 2026-01-17: `wan22_gguf.py` stage inference now recognizes `HN`/`LN` (and avoids false positives like `flow` → `low`) so WAN High/Low selectors populate correctly.
- 2026-02-05: Per-family scanner roots now include Anima (`anima_tenc`, `anima_vae`, `anima_loras`) so inventory discovery matches the new `models/anima*` layout.
- 2026-03-03: `wan22_gguf.py` now scans `wan22_ckpt` directory roots recursively for `.gguf` files (stable order), so nested WAN22 folder layouts from HF/manual extraction still populate `/api/models/inventory`.
- 2026-03-05: Per-family scanner roots now include Flux.2 keys (`flux2_tenc`, `flux2_vae`, `flux2_loras`) in addition to Flux.1, so discovery contracts are ready for Flux.2 model roots.
- 2026-03-12: Per-family scanner roots now also include `ltx2_tenc`, `ltx2_vae`, and `ltx2_loras`; generic scanners exclude `mmproj` projector GGUF files from text-encoder inventory and exclude `audio_vae` bundle files from generic VAE inventory.
