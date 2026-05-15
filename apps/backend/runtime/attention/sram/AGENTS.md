# apps/backend/runtime/attention/sram Overview
Date: 2026-03-19
Last Review: 2026-03-28
Status: Active

## Purpose
- Generic SRAM/shared-memory attention runtime bridge for versioned CUDA backends.

## Key Files
- `apps/backend/runtime/attention/sram/__init__.py` — Generic mode parsing, retired-env rejection, extension load/build warmup, pre-shaped dispatch attempts, and runtime metrics.
- `apps/backend/runtime/attention/sram/splitkv_validation.py` — bounded live-validation helpers for the split-KV diagnostics route (request parsing, branch-gate mirror, CPU oracle, live case execution).

## Notes
- This namespace is generic by design: WAN22 is only the first consumer.
- Keep the active contract on pre-shaped attention tensors (`[B,H,S,D]`); do not reintroduce model-specific projection fusion here.
- Retired WAN-only env keys must fail loud; do not add translation shims.
- 2026-03-19: Bridge-side pre-shaped validation now rejects zero batch/head tuples and mismatched or empty K/V sequence lengths before kernel launch, so unsupported tuples fail at the contract seam instead of surfacing as CUDA-path runtime errors.
- 2026-03-20: Pre-shaped dispatch now accepts non-overlapping dense `[B,H,S,D]` layouts with contiguous head-dim lanes (`stride[-1] == 1`), so WAN22 self-attention can hand the bridge its permuted views without blind Q/K/V materialization. The bridge also mirrors the CUDA causal `q_len <= int32` bound before launch and expects `attn_fwd` to preserve the input layout in its output tensor.
- 2026-03-28: Rectangular causal tuples stay supported, but the active contract is now bottom-right aligned (`kv_index <= q_index + (kv_len - q_len)`), matching the comparable FlashAttention reference semantics instead of the old top-left prefix rule.
- 2026-03-28: The bridge still exposes a single `attn_fwd` surface, but the CUDA extension may now pick an internal split-KV forward+reduce path when a conservative device/shape heuristic and the temporary-buffer budget both pass; no new Python/runtime knobs were added for this tranche.
- 2026-03-28: `splitkv_validation.py` is the runtime-owned diagnostics seam for the bounded `/api/tests/attention/sram/splitkv` route; it mirrors both split-count selection and temp-budget fallback, stays bounded to locked internal tuples instead of arbitrary tensor execution, and returns exact operator receipts (`ok`, `phase`, `reason_code`, `reason_detail`) for expected runtime outcomes.
- 2026-03-28: the diagnostics seam may now call `warmup_extension_for_diagnostics(...)`, which forces a build-enabled extension load retry through the same runtime owner even when normal load-time JIT is off; launcher/startup behavior remains unchanged.
