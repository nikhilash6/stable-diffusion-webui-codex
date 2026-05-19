# apps/backend/runtime/tools Overview
Date: 2025-12-31
Last Review: 2026-05-19
Status: Active

## Purpose
- Backend runtime “tools” that perform heavyweight offline-style operations (e.g. converting checkpoints) and are exposed via `/api/tools/*`.

## Key Files
- `apps/backend/runtime/tools/gguf_converter.py` — Converts SafeTensors (including sharded `*.safetensors.index.json`) to GGUF with quantization + verification.
- `apps/backend/runtime/tools/gguf_converter_specs.py` — Typed converter specs (one profile id per supported component family + quantization policy rule types).
- `apps/backend/runtime/tools/gguf_converter_profiles.py` — Profile registry: resolves source/native metadata normalizers, optional key mappings, and per-model dtype policies.
- `apps/backend/runtime/tools/gguf_converter_model_metadata.py` — Lists vendored model metadata (org/repo + supported components/config dirs) for the GGUF converter UI.
- `apps/backend/runtime/tools/gguf_converter_key_mapping.py` — Hugging Face → GGUF tensor-name mapping helpers (layer-indexed translations).
- `apps/backend/runtime/tools/gguf_converter_safetensors_source.py` — SafeTensors source helpers (single-file + sharded index/dir).
- `apps/backend/runtime/tools/safetensors_merge.py` — Merges safetensors sources (single file/index/directory) into one `.safetensors` output with typed progress.
- `apps/backend/runtime/tools/gguf_converter_quantization.py` — Quantization selector + generic per-tensor shape/block compatibility rules.
- `apps/backend/runtime/tools/gguf_converter_tensor_planner.py` — Tensor conversion planning helpers for the source/native converter surface (types + stored byte shapes + metadata normalizers).
- `apps/backend/runtime/tools/gguf_converter_types.py` — Public converter types (config, quantization enum, quant policy preset enum, progress, verification error).
- `apps/backend/runtime/tools/gguf_converter_metadata.py` — GGUF metadata injection helpers (provenance + arch + quant policy keys).
- `apps/backend/runtime/tools/gguf_converter_verify.py` — GGUF output verification helpers (tensor tables + spot-checks).

## Notes
- Tools should be deterministic, auditable, and fail loud (no silent fallbacks).
- When adding metadata to GGUF outputs, prefer stable keys and avoid leaking absolute local filesystem paths.
- 2026-01-13: GGUF converter metadata uses a Codex UI schema (`model.*`, `codex.*`, `gguf.*`) and avoids verbose conversion input keys (`codex.source_*`).
- 2026-01-13: GGUF converter supports cooperative cancellation (Tools API cancel flag) and the tools API defaults to no-overwrite when the output file already exists.
- 2026-03-07: GGUF converter emits source/native tensor names only for Flux, Qwen Image, Z-Image, WAN22, LTX-2, and Gemma3 text-encoder components; the tooling surface exposes one profile id per component and no layout-selection contract.
- 2026-05-17: GGUF converter supports Qwen Image transformer weights through the `qwen_image_transformer` profile (`arch=qwen_image`), preserves native tensor names with no key mapping, records `codex.qwen_image.*` metadata, verifies sharded SafeTensors payload presence before listing presets, and no longer treats unknown Diffusers transformer configs as the Llama fallback.
- 2026-05-18: GGUF converter policy presets are global through `quant_policy_preset=HQ|MQ|LQ`, with `MQ` as the default. `HQ` keeps conservative optional source-dtype preservation where profiles define it, `MQ` keeps balanced optional rules, and `LQ` uses only required invariants plus base requested quantization where profiles have no compact extras.
- 2026-05-19: Qwen Image `MQ` policy is a real middle preset for mixed quantized selectors: it keeps required/final tensors source-dtype, preserves input/timestep weights, keeps modulation edge bumps, and raises value projections plus MLP down/out projections to `Q6_K` with broader first/last and adjacent block precision bumps. Qwen Image policy metadata is profile-wide `v2`; `HQ` and `LQ` tensor policies remain conservative and compact respectively.
- 2026-01-14: Fixed `concat_dim0` streaming writes to allow variable dim0 sizes (required by Flux single-block `linear1` fusion: q/k/v + `proj_mlp`).
- 2026-01-14: Flux GGUF quantization keeps sensitive IO projection weights in float (F32/F16) and keeps Flux 1D tensors in F32 (biases + norm scales) to preserve output quality.
- 2026-01-14: GGUF converter dispatch is now profile-driven (typed registry): model-specific dtype “overrides” are formalized as per-model quantization policies (user `tensor_type_overrides` remain supported, but policy rules can be marked required).
- 2026-01-15: Removed a stale Flux planner dtype override injection that imported a deleted type; Flux dtype rules live in the profile quantization policy.
- 2026-01-15: Flux optional IO quality groups are gated by `quant_policy_preset`: `HQ` preserves source floating dtype, while `MQ`/`LQ` use the profile F16 baseline for those optional groups.
- 2026-01-15: GGUF converter now supports explicit `profile_id` selection (UI can avoid heuristics) and a vendored preset list for picking configs.
- 2026-01-15: Any profile rule that preserves a tensor uses the source floating dtype. User-selectable mixed float groups are not a live contract.
- 2026-01-16: Flux and WAN22 quality policy is profile-owned and gated by `quant_policy_preset`; Tools UI exposes the Mixed quant-variant toggle plus the global policy preset, not a dtype selector.
- 2026-01-16: Vendored selector now uses “model metadata” (org/repo + component) rather than listing raw config-dir presets.
- 2026-01-16: Vendored model metadata scanner no longer classifies `*ForCausalLM` configs as converter components and labels supported diffusion/text-encoder components for UI display.
- 2026-01-16: WAN22 presets label the two-stage denoiser split as `high_noise` (transformer) and `low_noise` (transformer_2) for clarity in Tools UI.
- 2026-01-16: WAN22 optional time/text embedder source-dtype preservation is the `HQ` quant-policy lane; `MQ`/`LQ` use the profile F16 baseline while required stability tensors stay required.
- 2026-01-16: WAN22 converter presets expose global `quant_policy_preset` quality selection; there is no Mixed AUTO/FP16/FP32 knob.
- 2026-01-16: GGUF converter verification reuses the conversion safetensors handle (avoids re-opening huge WAN22 weights twice; improves Windows stability).
- 2026-01-24: Added GGUF converter support for LTX-2 denoiser weights (`LTX2VideoTransformer3DModel`) and the LTX-2 Gemma3 text encoder via stable profile ids `ltx2_transformer` and `gemma3_tenc`.
- 2026-01-28: Z-Image conversions now record `codex.zimage.variant=turbo|base` when the source includes a diffusers `scheduler/scheduler_config.json` (shift=3.0/6.0), and the Z-Image metadata normalizer no longer defaults `_name_or_path` to Turbo when the input config is missing it.
- 2026-03-07: Root-repo tools emit base `.gguf` outputs only.
- 2026-03-07: The converter emits source/native tensor names only for Flux, Qwen Image, Z-Image, WAN22, LTX-2, and Gemma3 text-encoder profiles; mixed/required dtype policy rules target the tensor names that actually reach the GGUF writer.
- 2026-01-02: Added standardized file header docstrings to the tools facade (`__init__.py`) (doc-only change; part of rollout).
- 2026-05-18: Removed `precision_mode`, `float_group_overrides`, and the mixed float-group registry from the live converter contract. Stale payload keys fail loud at the Tools API boundary.
- 2026-03-05: Added `safetensors_merge.py` runtime tool (`merge_safetensors_source`) that resolves single/sharded SafeTensors layouts, validates each source payload as fully indexed/contiguous, exposes shared merge-config validation for the API preflight, rejects output paths that alias source shard/source files, and streams tensor payload bytes into one `.safetensors` output with progress callbacks and fail-loud path validation.
