# apps/backend/runtime/ops Overview
Status: Active

## Purpose
- Shared execution ops and `torch.nn` shims used by engines and runtimes.

## Current behavior
- `using_codex_operations(..., weight_format="gguf")` selects the GGUF-aware op layer.
- Unsupported `weight_format` values must raise `NotImplementedError`.
- Root runtime paths accept base `.gguf` artifacts only; unsupported packed artifacts must fail loud.
- GGUF CPU cache policy applies only to CPU-resident weights.
- Operation context is request-local and global patch windows stay serialized.
