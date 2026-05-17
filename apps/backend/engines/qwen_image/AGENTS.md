# apps/backend/engines/qwen_image
Date: 2026-05-17
Last Review: 2026-05-17
Status: Active

## Purpose
- Host the Qwen Image engine facade for the single `qwen_image` architecture family.
- Keep public request variants internal-only: task/router code derives `2512` for txt2img and `edit_2511` for img2img edit.

## Key Files
- `apps/backend/engines/qwen_image/qwen_image.py` — engine registration target; validates metadata-only Qwen Image bundle selection and fails loud until native runtime execution is implemented.
- `apps/backend/engines/qwen_image/__init__.py` — package marker only.

## Notes
- Do not introduce `qwen_image_2512` or `qwen_image_edit_2511` engine ids, aliases, path roots, or families.
- Qwen Image execution requires one Qwen2.5-VL-7B text encoder and one Qwen Image VAE selected through canonical sha-backed asset contracts; do not reuse Anima Qwen3-0.6B or WanVAE contracts.
- The facade may validate vendored HF metadata and variant/cache wiring before GPU/runtime implementation, but generation must fail with `NotImplementedError` until the native runtime owner exists.
