# apps/backend/engines/flux2 Overview
<!-- tags: backend, engines, flux2, qwen3 -->
Date: 2026-03-08
Last Review: 2026-03-22
Status: Active

## Purpose
- Host the truthful FLUX.2 Klein 4B/base-4B engine seam for backend image generation.

## Key Files
- `apps/backend/engines/flux2/flux2.py` — `Flux2Engine` facade (Qwen conditioning + normalized-external FLUX.2 latent encode/decode overrides + txt2img/img2img task exposure).
- `apps/backend/engines/flux2/spec.py` — Runtime assembly for FLUX.2 loader-resolved components.
- `apps/backend/engines/flux2/factory.py` — Factory seam returning `(runtime, CodexObjects)`.
- `apps/backend/engines/flux2/img2img.py` — Dedicated FLUX.2 img2img wrapper that injects `image_latents` instead of reusing classic init-latent denoise semantics.
- `apps/backend/engines/flux2/__init__.py` — Package marker only.

## Notes
- 2026-03-22: `Flux2Engine.encode_first_stage(...)` now forwards the shared optional `encode_seed` into the FLUX.2 external-latent VAE encode helper, and `img2img.py` threads the same shared seed into masked `image_latents` conditioning so FLUX.2 img2img/inpaint no longer drops posterior-seed requests on either encode branch.
- 2026-03-08: `img2img.py` now preserves truthful open-ended sampling progress for native samplers without an honest bounded total (for example `dpm adaptive`) by emitting `percent=None`, `total_steps=None`, and non-fake sampling/decode metadata instead of coercing unknown totals to integers.
- 2026-03-09: `img2img.py` hires path records effective hires sampler/scheduler/steps/denoise/size metadata for the shared img2img response surface; hires sampler/scheduler/size stay request-owned overrides and prompt parsing is LoRA-only.
- Loader/parser already own FLUX.2 checkpoint detection + component loading; engine code must reuse the resolved bundle instead of inventing new loading paths.
- The active FLUX.2 backend seam is **txt2img + image-conditioned img2img/inpaint** for Klein 4B/base-4B:
  - `supports_img2img=true`
  - `supports_hires=true`
  - `supports_lora=false`
  - `encode_first_stage(...)` uses the local AutoencoderKLFlux2 batch-norm latent normalization contract; do not map FLUX.2 to Flow16/Flux.1 semantics.
  - Masked img2img/inpaint reuses the shared masked bundle/full-res composite path, but FLUX.2 still conditions through `image_latents` rather than SD-family `image_conditioning`.
  - Partial-denoise img2img is wired through sampler-native continuation (`init_latent` + `denoise_strength`) while keeping clean `image_latents` conditioning.
  - Unmasked hires img2img is wired through the shared hires-prep dispatcher plus the dedicated FLUX.2 second pass; masked hires still fails loud.
- FLUX.2 conditioning is Qwen-only cross-attention (`7680` dim). `get_learned_conditioning(...)` now returns sampler-ready dict conditioning (`crossattn` + inert `vector`) so `image_latents` can flow through the compiled-conditioning path without special-case shims.
- Sampler state is normalized external **32-channel** latent BCHW. Encode/decode apply the FLUX.2 VAE batch-norm mean/std over the patchified 128-channel latent space; do not map FLUX.2 to Flow16/Flux.1 latent semantics.
- Variant behavior is runtime-resolved:
  - `FLUX.2-klein-4B` → distilled checkpoint; non-empty negative prompts fail loud and the engine uses the request's distilled guidance scale as the effective sampling CFG.
  - `FLUX.2-klein-base-4B` → classic CFG / negative prompts
- Use `img2img.py` for FLUX.2 image-conditioned generation; the common `apps/backend/use_cases/img2img.py` route still models classic init-latent denoise / Kontext continuation and is not truthful for FLUX.2.
- 2026-03-24: FLUX.2 masked img2img now matches the classic owner split for `per_step_blend_strength`: exact `1.0` keeps the blend-total trio (`pre_denoiser` + `post_denoiser` + `post_sample`), while lower strengths must use `post_step_hook` + `post_sample` only.
- 2026-03-26: `img2img.py` now treats prompt parsing as LoRA-only on both base and hires paths, reuses `resolve_mask_enforcer_hooks(...)` instead of local branch duplication, opts out of SD-style masked `image_conditioning`, and passes the internal fixed-step continuation flag only from the hires second-pass seam.
- 2026-03-26: FLUX.2 hires prompt parsing stays LoRA-only, but second-pass prompt contexts must still inherit base/request LoRAs when the hires prompt omits them; prompt-local hires tags only override same-path weights, they do not clear inherited LoRAs by omission.
