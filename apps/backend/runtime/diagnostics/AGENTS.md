# AGENT — Runtime Diagnostics

Purpose: Central home for runtime diagnostics and debugging helpers (trace/timeline/pipeline debug, exception dump hooks).

Key files:
- `apps/backend/runtime/diagnostics/call_trace.py`: `sys.setprofile`-based call tracing for deep debugging.
- `apps/backend/runtime/diagnostics/contract_trace.py`: JSONL contract-trace sink (`logs/contract-trace`) with prompt hashing only.
- `apps/backend/runtime/diagnostics/error_summary.py`: bounded runtime-error classifier for concise console summaries and friendly public-safe projections.
- `apps/backend/runtime/diagnostics/exception_hook.py`: Sys/thread/asyncio exception dump hooks (writes to `logs/`).
- `apps/backend/runtime/diagnostics/pipeline_debug.py`: Pipeline debug flag + decorator helpers.
- `apps/backend/runtime/diagnostics/profiler.py`: Global torch-profiler wrapper (Perfetto trace export + transfer/cast totals; opt-in via `CODEX_PROFILE` or `CODEX_TRACE_PROFILER`).
- `apps/backend/runtime/diagnostics/timeline.py`: Inference “timeline” tracer (nested stage/event tracking + render/export).
- `apps/backend/runtime/diagnostics/trace.py`: Lightweight torch tracing helpers (`torch.nn.Module.to` patch + scoped sections).

Notes:
- Diagnostics should stay lightweight and avoid importing heavy ML dependencies at import time unless strictly required.
- If a failure is expected/optional, make it explicit; do not swallow unexpected errors.
- Contract-trace payloads must remain prompt-redacted (`prompt_hash` only).

Last Review: 2026-03-30

- 2026-02-18: `exception_hook.py` and `timeline.py` now route explicit stream emission through `apps.backend.infra.stdio` (`write_stderr`/`write_stdout`), preserving crash-path and timeline console contracts while centralizing primitive stream writes.
- 2026-03-30: `error_summary.py` is the bounded owner for known runtime-failure projections; `orchestrator.py` keeps full exception dumps in `exception_hook.py` logs while console logging only the concise summary returned by the classifier.
