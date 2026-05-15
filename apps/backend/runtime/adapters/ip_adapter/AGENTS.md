# apps/backend/runtime/adapters/ip_adapter Overview
Date: 2026-03-30
Last Review: 2026-04-30
Status: Active

## Purpose
- Hosts the canonical IP-Adapter runtime seam: validated asset loading, reference-image embedding prep, slot-layout resolution, and request-scoped patch apply/restore.

## Key Files
- `assets.py` — Loads/caches the validated IP-Adapter asset bundle and owns layout-family detection.
- `layout.py` — Owns the canonical slot-to-UNet coordinate order per semantic engine.
- `probe.py` — Owns the bounded live diagnostics report for `reference image -> CLIP preprocess -> CLIP encode -> projector/resampler`.
- `preprocess.py` — Builds conditional/unconditional image tokens from the selected reference image.
- `modules.py` — Defines the projector/resampler modules and the shared attn2 replace patch implementation.
- `session.py` — Applies IP-Adapter to the active denoiser clone for one sampling pass and restores the baseline objects afterward.
- `types.py` — Typed config, source, asset, and embedding carriers for the runtime seam.

## Notes
- IP-Adapter slot order is not generic UNet discovery order. Keep the checkpoint slot contract in `layout.py`; do not bind slots straight from `UnetPatcher._iter_transformer_coordinates()`.
- SDXL slot order must be proven against translated `attn2.to_k.weight` parameter names derived from the repo's canonical SDXL config + diffusers→LDM map. For the official base/Plus SDXL checkpoints, the proved order is diffusers `down -> up -> mid` (LDM `input -> output -> middle`). Width-only matching is insufficient because the middle block and multiple output groups share the same width.
- `session.py` may validate against the generic UNet transformer inventory, but the authoritative order for slot assignment is the IP-Adapter semantic-engine layout.
- Conditional vs unconditional branch selection belongs in `modules.py::IpAdapterCrossAttentionPatch`; request prep owns token construction only.
- The bounded diagnostics route `/api/tests/ip-adapter/probe` is not a second generation API. `probe.py` owns the live conditioning receipt, while `interfaces/api/routers/tests.py` only resolves inventory-backed asset paths and repo-scoped reference-image paths.
- The bounded diagnostics route `/api/tests/ip-adapter/probe` must keep the canonical CUDA/runtime codepath but execute it in a subprocess. If the live CLIP/IP-Adapter probe OOMs or dies, the route must return a structured failure receipt instead of taking down the API host.
- Base-layout unconditional tokens are the zero vector in pooled CLIP embedding space; use `zeros_like(image_embeds)` before the base projector.
- Plus-layout unconditional tokens come from `zeros_like(pixel_values)` in already-preprocessed CLIP image space before the resampler projection; do not emulate this by encoding a black image through preprocessing.
- Image-encoder ownership is split on purpose: checkpoint IO may stage through the text-encoder offload lane, but the live CLIP vision runtime module and cached asset bundle must resolve device/dtype from `DeviceRole.CLIP_VISION`.
- Do not reintroduce adapter-local CLIP vision key rewriting or raw `nn.Module.load_state_dict(...)` paths here; image-encoder loading must stay on the canonical CLIP vision/state-dict seams.
- Token-batch expansion must preserve prompt order exactly. When runtime sampling needs more rows than the prepared IP-Adapter token batch, repeat each prepared row by an integer factor (`repeat_interleave`) and fail loud on non-integral or shrinking batch geometry; never truncate and never pad by repeating only the last row.
- Checkpoint bucket ownership stays lazy: nested `image_proj` / `ip_adapter` mappings may pass through as-is, and flat-prefixed checkpoints may only be exposed through explicit prefix-filter views. Do not strip `image_proj.` / `ip_adapter.` into eager copied dicts.
- Slot source-key ownership belongs to `IpAdapterKvProjectionSet.slot_specs`. `assets.py`, `layout.py`, `session.py`, and `IpAdapterCrossAttentionPatch` must consume the exact parsed checkpoint source keys; never reconstruct slot keys as `1,3,5...` from slot index math.
- `preprocess.py` owns the conditioning math for generation and diagnostics. `probe.py` may collect receipts and choose `crop`, but it must consume the shared conditioning helper instead of restating the projector/zero-path algorithm.
- The official encoder branch of `/api/tests/ip-adapter/probe` must compare `ClipVisionEncoder` against `CLIPVisionModelWithProjection` loaded from the canonical CLIP vision keyspace, not from a raw state dict with allowed full-CLIP extras like `logit_scale`.
- If isolated encoder/projector/resampler/attn2 proofs all match the official reference and the visual bug remains, debug the live integration seam with `CODEX_IP_ADAPTER_DEBUG=1` or `CODEX_IP_ADAPTER_DEBUG_PATCH=1`: `session.py` owns the slot-map preview and `modules.py` owns the first live patch-call receipts (block, slot, sigma window, cond/uncond geometry, and tensor stats).
- 2026-04-30: IP-Adapter context managers are side-effect-only and yield no session payload; do not reintroduce a session-carrier dataclass unless a real in-repo consumer is added in the same tranche.
