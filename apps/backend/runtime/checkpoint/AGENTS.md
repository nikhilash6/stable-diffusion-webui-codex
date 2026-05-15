# AGENT — Runtime Checkpoint IO

Purpose: Checkpoint IO helpers used by the runtime (safetensors / GGUF / guarded pickle).

Key files:
- `apps/backend/runtime/checkpoint/io.py`: `load_torch_file`, GGUF loaders, and config read helpers.
- `apps/backend/runtime/checkpoint/safetensors_header.py`: SafeTensors header-only readers (incl. primary dtype hint) for lightweight tooling/logging.

Notes:
- Keep checkpoint IO lightweight and avoid importing heavy runtime modules at import time.
- Favor SafeTensors and safe torch loading where possible; only fall back to guarded pickle when explicitly allowed.
- 2026-01-28: `io.py` now also exposes `read_gguf_metadata(...)` so engines can inspect trusted GGUF provenance without importing `apps.backend.quantization.*` directly (import guardrail compliance).
- 2026-02-15: `load_gguf_state_dict(...)` device argument is now used across WAN22/Flow16 call sites to place loaded tensors directly on the target runtime device (no hidden CPU-only assumption).
- 2026-03-11: `load_torch_file(...)` treats both `.safetensor` and `.safetensors` as lazy SafeTensors sources; singular-extension checkpoints must not fall through to guarded pickle loading.
- 2026-03-12: `io.py` also exposes `read_safetensors_metadata(...)` so runtime-family loaders can carry exact SafeTensors header config (for example the real LTX 2.3 wrapped vocoder config) without materializing a second state dict or inventing hardcoded defaults.
- 2026-03-12: `read_arbitrary_config(...)` now accepts both `config.json` and `scheduler_config.json` so vendored scheduler directories stay first-class config sources instead of special-case failures.
- 2026-03-29: `runtime/models/safety.py::safe_torch_load(...)` now treats native `torch.load(..., weights_only=True)` as the authoritative safe path when the active torch build supports it, enables `mmap=True` only for zip/new-serialization checkpoints, and keeps the restricted pre-validation / restricted-pickle path as a legacy fallback for torch builds without native `weights_only`, because modern torch zip checkpoints carry archive metadata and persistent-storage references that the fallback validator does not fully model.

Last Review: 2026-03-29
