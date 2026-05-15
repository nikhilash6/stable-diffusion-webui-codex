# apps/backend/runtime/kernels Overview
Status: Active

## Purpose
- Owns custom C++/CUDA extensions used by runtime code.

## Kernel Trees
- `attention_sram_v1/` — generic SRAM/shared-memory attention addon sources, build scripts, and the kernel-local manual CUDA smoke harness.
- `wan_fused_v1/` — retired WAN-specific prototype awaiting removal after generic cutover.

## Expectations
- Keep build inputs aligned with the runtime loaders that consume each extension.
- Kernel-local validation helpers are acceptable when they exercise one kernel tree directly, stay manual-only, are never imported by runtime code paths, and keep their CLI surface aligned to the documented operator flow instead of growing ad-hoc debug knobs.
- Remove retired kernel trees instead of leaving dormant sources behind.
