# apps/backend/runtime/kernels/attention_sram_v1 Overview
Date: 2026-03-19
Last Review: 2026-03-28
Status: Active

## Purpose
- Owns the generic `attention_sram_v1_cuda` extension used by `runtime/attention/sram`.

## Key Files
- `apps/backend/runtime/kernels/attention_sram_v1/setup.py` — local CUDAExtension build script for `attention_sram_v1_cuda`.
- `apps/backend/runtime/kernels/attention_sram_v1/attention_sram_v1_binding.cpp` — ABI export, op registration, and CPU fail-loud stubs.
- `apps/backend/runtime/kernels/attention_sram_v1/attention_sram_v1_kernels.cu` — narrow shared-memory attention forward kernel for pre-shaped `[B,H,S,D]` tensors plus optional generic RoPE helper.
- `apps/backend/runtime/kernels/attention_sram_v1/smoke_attention_sram_v1.py` — manual CUDA smoke harness for preflight/build/warmup/parity/fallback on a CUDA-capable host.

## Notes
- Keep this tree generic: no WAN-only naming, no fake backend selectors, no projection/norm/out-proj logic here.
- The active `attn_fwd` contract is narrow on purpose: CUDA, fp16, non-overlapping dense `[B,H,S,D]` with contiguous head-dim lanes (`stride[-1] == 1`), `head_dim=128`, boolean `is_causal`, and output layout preserved from `q`.
- `rope_blhd_` is optional scaffolding and must stay separate from the `attn_fwd` hot path.
- 2026-03-19: Kernel-side validation now mirrors the bridge on K/V sequence-length agreement, and the tile loop uses explicit `std::min(...)` selection so the first cut stays compileable under the narrow CUDA contract.
- 2026-03-20: The CUDA path now consumes stride-based `[B,H,S,D]` views directly and writes the output with the input layout, instead of forcing caller-side Q/K/V materialization just to satisfy the first SRAM cut.
- 2026-03-28: Rectangular causal masking is bottom-right aligned in the kernel (`kv_index <= q_index + (kv_len - q_len)`); if `q_len > kv_len`, fully masked early rows remain valid and produce zeros.
- 2026-03-28: `attn_fwd` now has one internal split-KV forward+reduce dispatch path behind a conservative occupancy/budget heuristic; the public CUDA extension ABI and the bridge-facing contract remain unchanged.
- 2026-03-29: The build seam now validates the same `CUDA_HOME/bin/nvcc(.exe)` tool that PyTorch `BuildExtension` will actually invoke, instead of treating a raw `PATH` hit as the compile owner; `setup.py` also uses platform-aware host C++ optimization flags (`/O2` on Windows, `-O3` elsewhere) instead of one POSIX-only `cxx` flag list.
- 2026-03-20: The manual smoke harness keeps build ownership explicit:
  - `build` is the only stage that may compile the extension,
  - `warmup` / `parity` / `fallback` / `full` run the bridge with `CODEX_ATTENTION_SRAM_JIT=0`,
  - `full` executes `preflight -> build -> warmup -> parity -> fallback` without re-triggering bridge-owned JIT.
- 2026-03-20: The operator-facing harness CLI is locked to the documented flow:
  - `parity`, `fallback`, and `full` use the fixed supported/unsupported tuples from the runbook,
  - only `--mode auto|force` stays user-selectable on the runtime stages,
  - widen the runbook first if the harness contract ever needs more CLI surface.
