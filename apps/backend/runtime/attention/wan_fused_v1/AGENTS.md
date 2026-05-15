# apps/backend/runtime/attention/wan_fused_v1 Overview
Date: 2026-02-25
Last Review: 2026-02-26
Status: Active

## Purpose
- Defines WAN fused-attention V1 contract helpers and runtime bridge points for optional CUDA fused kernels.
- Centralizes fail-loud validation for WAN fused self/cross attention request tuples.

## Key Files
- `apps/backend/runtime/attention/wan_fused_v1/__init__.py` — Runtime contract validator, mode resolver, and extension loader bridge (`prebuilt` -> `in_place` -> optional `jit`).
- `apps/backend/runtime/kernels/wan_fused_v1/wan_fused_v1_binding.cpp` — `torch.ops.wan_fused_v1.{self_fwd,cross_fwd}` registration and CPU/CUDA dispatch wiring.
- `apps/backend/runtime/kernels/wan_fused_v1/wan_fused_v1_kernels.cu` — CUDA entrypoints implementing self/cross V1 fused forward paths.

## Notes
- V1 scope is inference-only (`dropout=0`) and CUDA-only.
- Cross-attention in V1 requires RoPE on Q+K when fused path is enabled.
- Forced mode must fail loud on unsupported tuples or missing extension/kernel ops.
- Non-forced mode may return explicit reason codes and allow caller-level fallback.
- 2026-02-25: Added explicit load-time warmup API (`warmup_extension_for_load`) that resolves fused mode/env gates, triggers extension load/build before denoise, and raises fail-loud when warmup fails under `force` mode.
- 2026-02-25: Kernel-runtime invariants are now mapped to explicit contract code `E_WAN_FUSED_STREAMING_INVARIANT_VIOLATION` so failures in streaming-only guarantees are surfaced without generic error masking.
- 2026-02-25: Self fused dispatch no longer packs `w_qkv`/`b_qkv`; wrapper now passes `w_q/w_k/w_v` + optional biases separately to cut transient VRAM overhead before kernel dispatch.
- 2026-02-26: Runtime loader enforces extension ABI (`WAN_FUSED_V1_ABI=4`) and rejects stale modules to avoid silent signature mismatch after weight/bias contract changes (2D weights + 1D biases).
- 2026-02-26: Wrapper now passes `nn.Linear`-native projection weights as 2D `[out,in]` tensors and biases as 1D `[out]` tensors; CUDA kernels handle transpose views internally to avoid per-call transpose+contiguous materialization in Python.
- 2026-02-25: Loader now purges import cache for extension module names during stage fallback so an ABI-rejected prebuilt module does not poison in-place/JIT resolution in-process.
- 2026-02-26: Runtime now recognizes kernel-side telemetry controls (`CODEX_WAN_FUSED_V1_KERNEL_TRACE`, `CODEX_WAN_FUSED_V1_KERNEL_TRACE_KV`, `CODEX_WAN_FUSED_V1_KERNEL_TRACE_EVERY_Q`, `CODEX_WAN_FUSED_V1_KERNEL_TRACE_EVERY_KV`) for per-phase VRAM snapshots emitted by fused CUDA path.
- 2026-02-26: Attention-core resolution contract is explicit and resolver-owned via `resolve_effective_wan_fused_attn_core(mode) -> (attn_core, attn_core_source, attn_core_raw)`. `attn_core_source` tokens are emitted verbatim as `env|force_default|kernel_default` (no caller-side rewriting).
- 2026-02-26: `CODEX_WAN_FUSED_V1_ATTN_CORE=aten|cuda_experimental` remains optional; `cuda` aliases `cuda_experimental`. If unset, resolver defaults by mode: `force -> aten (force_default)`, otherwise `aten (kernel_default)`.
- 2026-02-26: WAN22 model/run telemetry now plumbs `attn_core`, `attn_core_source`, and `attn_core_raw` directly from resolver output, with no env mutation in the model/run hot path.
- 2026-02-26: Kernel workspace cap env is `CODEX_WAN_FUSED_V1_PRECOMPUTE_WORKSPACE_MB` (strict positive integer MB, default `512`), and invariant errors now report workspace cap plus score-tile-budgeting mode.
- 2026-02-26: Minimal env recommendations:
  - Force debugging (fail-loud, default core): `CODEX_WAN22_FUSED_ATTN_V1_MODE=force` (default effective core is `aten`; optionally pin `CODEX_WAN_FUSED_V1_ATTN_CORE=cuda_experimental`).
  - Production off (disable fused runtime path): `CODEX_WAN22_FUSED_ATTN_V1_MODE=off` (leave `CODEX_WAN_FUSED_V1_ATTN_CORE` unset).
