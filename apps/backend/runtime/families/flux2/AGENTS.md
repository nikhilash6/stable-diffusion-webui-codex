# apps/backend/runtime/families/flux2 Overview
<!-- tags: backend, runtime, families, flux2, qwen3 -->
Date: 2026-03-06
Last Review: 2026-03-22
Status: Active

## Purpose
- Host the FLUX.2 Klein 4B/base-4B runtime helpers used by the `flux2` engine: the normalized external 32-channel latent bridge, the sampler-facing transformer adapter, and the Qwen3 stacked-hidden-state prompt encoder.

## Key Files
- `apps/backend/runtime/families/flux2/runtime.py` — Active FLUX.2 latent/VAE helpers plus the sampler-facing `Flux2CoreAdapter` that accepts optional `image_latents`.
- `apps/backend/runtime/families/flux2/text_encoder.py` — FLUX.2 Qwen3-4B wrapper + prompt embedding engine.
- `apps/backend/runtime/families/flux2/model.py` — Older centered-latent bridge/predictor helpers retained in-tree; do not route the live engine/img2img seam through this file.
- `apps/backend/runtime/families/flux2/__init__.py` — Explicit re-export surface for the engine/spec seam.

## Notes
- External sampler state uses normalized external **32-channel latents**; `encode_flux2_external_latents(...)` / `decode_flux2_external_latents(...)` apply the FLUX.2 patchified batch-norm mean/std instead of Flow16-style scalar `process_in/out`.
- 2026-03-22: `encode_flux2_external_latents(...)` now forwards optional shared `encode_seed` into the VAE wrapper instead of silently dropping img2img posterior-seed requests at the FLUX.2 runtime seam.
- Qwen prompt encoding must follow the FLUX.2 Klein contract: `apply_chat_template(..., enable_thinking=False)`, `max_length=512`, and stacked hidden states from layers `9/18/27` → `7680` feature dim.
- 2026-03-06: FLUX.2 GGUF Qwen loads now reuse the ZImage Qwen keyspace resolver directly (`resolve_qwen3_gguf_keyspace`) and pass the lazy lookup mapping into native strict load (no materialized renamed state dict).
- The truthful active slice is **txt2img + image-conditioned img2img** and **Qwen-only**:
  - accept optional `image_latents` for FLUX.2 img2img
  - allow inert pooled/vector placeholder `y` only for sampler-contract compatibility
  - reject T5-side extras (`t5xxl_ids`, `t5xxl_weights`)
  - reject ControlNet/control conditioning
  - reject unsupported sampler extras fail-loud
- Distilled `FLUX.2-klein-4B` still has `guidance_embeds=false`; the engine layer resolves the effective distilled guidance scale and empty-negative contract before sampling instead of relying on transformer guidance embeddings.
- Do not copy `.refs/**` code into `apps/**`; extract intent and re-implement cleanly.
