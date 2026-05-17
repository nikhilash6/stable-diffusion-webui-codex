# apps/backend/runtime/families/qwen_image
Date: 2026-05-17
Last Review: 2026-05-17
Status: Active

## Purpose
- Host repo-owned Qwen Image family runtime contracts for the single `qwen_image` architecture family.
- Keep concrete Qwen Image payloads (`Qwen-Image-2512` txt2img and `Qwen-Image-Edit-2511` img2img edit) as internal variants, not separate engine or family ids.

## Key Files
- `apps/backend/runtime/families/qwen_image/config.py` - Internal variant metadata, prompt-template constants, dimensions, and image geometry helpers.
- `apps/backend/runtime/families/qwen_image/scheduler.py` - Diffusers-free FlowMatch Euler scheduler metadata validation plus Qwen Image sequence-length/shift helpers.
- `apps/backend/runtime/families/qwen_image/text_encoder.py` - Lightweight Qwen2.5-VL config validation plus variant-owned prompt-template planning.
- `apps/backend/runtime/families/qwen_image/transformer.py` - Lightweight `QwenImageTransformer2DModel` config and variant zero-conditioning validation.
- `apps/backend/runtime/families/qwen_image/vae.py` - Lightweight `AutoencoderKLQwenImage` metadata validation plus per-channel latent normalization helpers.
- `apps/backend/runtime/families/qwen_image/__init__.py` - Lightweight public family-runtime export surface.

## Notes / Decisions
- `Qwen-Image-2.0` is an architecture/frontier label in this repo, not a concrete checkpoint/repository contract.
- The canonical exact engine and family id is `qwen_image`; do not add `qwen_image_2512`, `qwen_image_edit_2511`, or `qwen_image_2_0` ids.
- `qwen_image_variant` is internal runtime/task metadata derived by the backend route/mode owner. Public request payloads must fail loud if they carry it.
- Qwen Image VAE assets must come from `qwen_image_vae` API roots and expose adjacent `AutoencoderKLQwenImage` config metadata; do not accept a generic inventory VAE just because the SHA appears in `inventory.vaes`.
- Reference trees under `.refs/**` and metadata mirrors under `apps/backend/huggingface/Qwen/**` are source frontiers only. Active code must stay repo-owned and Diffusers-free.
- Header/keyspace work must obey the root keymap law: do not strip prefixes, rewrite punctuation, materialize remapped state dicts, or normalize stored checkpoint layer names.
