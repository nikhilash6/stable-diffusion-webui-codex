# apps/backend/runtime/families/sd Overview
Date: 2025-10-28
Last Review: 2026-04-08
Status: Active

## Purpose
- Stable Diffusion (SD) runtime helpers used by SD engines (conditioning, pipelines, control modules).

## Subdirectories
- `cnets/` — ControlNet-specific helpers and wrappers.

## Notes
- 2026-03-20: `hires_fix.py` no longer branches on model-class `is_inpaint`; masked img2img behavior is request-owned through the shared mask stages instead of checkpoint heuristics.
- 2026-04-08: `fooocus_inpaint.py` and `brushnet.py` request-scoped sessions are now entered from `sampling_execute.py` only after canonical LoRA apply/reset activates the live sampling snapshot. Fooocus must respect `CODEX_LORA_APPLY_MODE` for its patch registration and restore the pre-session denoiser patch registry on exit.
- 2026-04-06: `fooocus_inpaint.py` owns the SDXL-only request-scoped Fooocus patch session (`fooocus_inpaint_head.pth` + `inpaint_v26.fooocus.patch`). Keep that branch exact-engine scoped and out of the shared `masked_img2img.py` stage.
- 2026-04-06: `fooocus_inpaint.py` also owns Fooocus checkpoint preflight for SDXL. Reject distilled/Turbo/Lightning/Hyper checkpoint selections from the router after checkpoint resolution; do not infer support from generic `sdxl` exact-engine truth alone.
- 2026-04-06: `brushnet.py` owns the SDXL-only request-scoped BrushNet lane pinned to `random_mask_brushnet_ckpt_sdxl_v0` under `sdxl_brushnet`. Keep that branch exact-engine scoped, keep the shared `masked_img2img.py` stage generic-only, and patch exact SDXL UNet inner layers instead of widening ControlNet truth or per-block alias glue.
- Keep SD runtime modules aligned with `apps/backend/engines/sd/`.
- 2026-01-02: Added standardized file header docstrings to `__init__.py` (doc-only change; part of rollout).
- 2026-02-01: Added `hires_fix.py` (hires pass init preparation; routes latent vs spandrel upscalers via the global upscalers runtime).
- 2026-02-20: `mmditx.py` attention wrapper now delegates to runtime dispatcher (`attention_function` with explicit PyTorch backend) instead of direct local SDPA call.
