<!-- tags: backend, types, payloads, samplers, exports -->

# apps/backend/types Overview
Date: 2026-01-03
Last Review: 2026-04-29
Status: Active

## Purpose
- Holds lightweight backend type definitions and shared constants used across services, API schemas, and package facades.

## Key Files
- `__init__.py` — Package marker (no re-exports); import from defining modules.
- `payloads.py` — Frozen key sets used for request payload validation.
- `samplers.py` — Canonical sampler enum + apply outcome container.
- `exports.py` — Lazy export name groups for backend `__getattr__` facades.

## Notes
- Keep these modules import-light to avoid bootstrap cycles; prefer simple dataclasses/enums and frozen sets.
- 2026-01-03: Added standardized file header docstrings to `types/*` modules (doc-only change; part of rollout).
- 2026-01-06: `samplers.py` parsing is strict (no alias/case normalization); empty values are invalid.
- 2026-01-18: Removed `types/__init__.py` re-export facade to avoid another public surface; call sites should import from `payloads.py` / `samplers.py` / `exports.py`.
- 2026-01-19: `exports.py` now keeps patcher/service export sets intentionally empty (patchers/services are not re-exported from `apps.backend`).
- 2026-01-28: `payloads.ExtrasKeys` now includes `zimage_variant` for Z-Image Turbo/Base variant selection in request extras.
- 2026-02-03: `payloads.ExtrasKeys` now uses `hires` (legacy key removed).
- 2026-02-08: `payloads.Txt2ImgKeys.HIRES` now explicitly includes `refiner` and `distilled_cfg` to stay consistent with `extras.hires.*` parsing in API router validation.
- 2026-02-15: `payloads.Txt2ImgKeys` now includes explicit `settings_revision` contract key group so generation payload validation can enforce revision parity.
- 2026-02-16: WAN22 video/model key ownership is outside `payloads.py`; model-specific keymaps now live in `apps/backend/runtime/state_dict/*`.
- 2026-02-18: `payloads.ExtrasKeys.COMMON` now includes `guidance` so txt2img/img2img extras can carry strict guidance-policy overrides (`apg_*`, `guidance_rescale`, `cfg_trunc_ratio`, `renorm_cfg`) without contract drift.
- 2026-03-31: `payloads.ExtrasKeys.COMMON` stays shared-only; img2img-only nested owners such as `img2img_extras.supir` must be admitted by the route-specific allowlist instead of broadening the common extras key set.
- 2026-04-29: `payloads.Txt2ImgKeys.HIRES` is the complete nested `extras.hires` key owner; do not rebuild nested hires allowlists from top-level `CORE` / `DIFFUSION` groups.
