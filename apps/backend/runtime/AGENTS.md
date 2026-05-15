# apps/backend/runtime Overview
<!-- tags: backend, runtime, overview -->
Date: 2025-10-30
Last Review: 2026-02-22
Status: Active

## Purpose
- Provides reusable runtime components shared across engines: attention kernels, adapters, text processing, memory policies, sampling utilities, model loaders, and model-family runtimes (SD, Flux, Chroma, ZImage, WAN22).

## Key Subdirectories
- `attention/` — Attention backends and related kernels.
- `adapters/` — Runtime adapters (e.g., LoRA, SafeTensors helpers).
- `diagnostics/` — Tracing/timeline/debug helpers (optional; used for diagnosis).
- `model_parser/` — Codex-native checkpoint parser plans and conversions replacing `huggingface_guess`.
- `text_processing/` — Tokenization, prompt parsing, and textual inversion helpers.
- `sampling/` — Sigma builders, schedulers, Philox integration, and sampling drivers.
- `memory/` — VRAM/CPU memory policies and management helpers.
- `ops/` — Low-level tensor operations leveraged by engines.
- `checkpoint/` — Checkpoint IO helpers (safetensors/GGUF/pickle + config reads).
- `state_dict/` — Lightweight state-dict views + small state-dict utilities.
- `models/` — Model registry/load helpers (checkpoints, VAEs, etc.).
- `families/` — Model/runtime-specific implementations by engine family (`sd/`, `flux/`, `chroma/`, `zimage/`, `wan22/`).
- `vision/` — Vision encoder runtimes (clip specs/registry/encoders) shared across engines and patchers.
- `processing/` — High-level input preprocessing utilities shared by use cases.
- `pipeline_stages/` — Shared pipeline helper stages consumed by canonical use-cases (Option A; no engine-specific pipelines).
- `streaming/` — Shared segment streaming controller primitives used by multiple family streaming wrappers (Flux/WAN22).
- `common/` — Shared building blocks (e.g., core (UNet/DiT) wrappers) used across runtimes.
- `misc/` — Smaller helper modules that don’t fit other buckets (logging, strict checks, etc.).
- `sampling_adapters/` — Sampling adapter wrappers used by samplers/patchers.
- `kernels/` — Custom CUDA/C++ kernels where required.

## Notes
- Keep runtime logic model-agnostic when possible; place model-specific code under `families/<family>/`.
- Avoid duplicating helpers across engines—centralize them here to maintain parity.
- 2026-01-24: Attention backend selection is now driven by the runtime memory config (seeded from `/api/options` key `codex_attention_backend` at bootstrap, and switchable at runtime via `memory_management.set_attention_backend(...)`).
- Runtime layout: family runtimes live under `apps/backend/runtime/families/<family>/`; keep `apps/backend/runtime/` for generic runtime modules shared across families (plan: `.sangoi/plans/2026-01-17-backend-runtime-families-layout.md`).
- 2025-12-03: Processing models expose `RefinerConfig` and carry refiner configs on `CodexProcessingTxt2Img`/`CodexHiresConfig` (global + hires) for stage-based refiner execution.
- 2026-02-08: `RefinerConfig` pointer semantics now use `swap_at_step` (serialized as `switch_at_step`) instead of refiner step counts.
- 2025-12-03: Sampler driver checks `backend_state.should_stop` each step and honors `/api/tasks/{id}/cancel` (immediate) by raising `RuntimeError("cancelled")` to abort sampling.
- 2025-11-03: `runtime.diagnostics.call_trace` exposes global function-call tracing via `enable()/disable()` and `enable_from_env()`. The API entrypoint wires this behind `--trace-debug`/`--trace-call-debug` and `CODEX_TRACE_CALL_DEBUG`, logging each Python function call at `DEBUG` using the `backend.calltrace` logger. Module scope is controlled by `CODEX_TRACE_CALL_DEBUG_MODULE_PREFIXES`, and per-function caps use `CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC` (default 10; `<=0` disables the cap).
- 2025-11-04: Added streaming materialization helpers to `runtime.state_dict.views.FilterPrefixView`/`LazySafetensorsDict` so safetensor-backed parser components load with a single handle instead of reopening per key (prevents Windows `torch_cpu.dll` crashes during SDXL parsing).
- 2025-11-14: Weight/bias fetch logs under `runtime.ops.operations` are now rate-limited via `CODEX_WEIGHT_FETCH_LOG_LIMIT` (default 10 per layer class). Set to `0` to disable the log entirely or raise when diagnosing dtype/offload issues.
- 2025-11-25: SDXL CLIP converters now handle OpenCLIP BigG resblock layouts without the double-`transformer` prefix bug; CLIP-G keys under `transformer.resblocks.*` are normalized to `transformer.text_model.encoder.*` (preserving `logit_scale`) so validations no longer warn on missing `layer_norm1`.
- 2025-12-15: `runtime/tools/gguf_converter.py` emits real quantized GGUF using the shared GGUF writer + quant kernels and streams tensor data instead of buffering entire checkpoints in memory.
- 2025-12-19: GGUF converter quant menu expanded to include `Q2_K/Q3_K/IQ4_NL`, mixed schemes (`Q4_K_M/Q5_K_M`), per-tensor override rules, and legacy `Q4_0/Q4_1/Q5_0/Q5_1/Q6_K` (in addition to `Q8_0/Q5_K/Q4_K`).
- 2025-12-30: GGUF converter now supports sharded SafeTensors inputs via `*.safetensors.index.json` (or by pointing at a directory containing the index); no manual shard merge required.
- 2025-12-29: Sampling and utils now avoid importing heavy runtime ops/quantization at module import time (keeps API startup and `/api/models`/QuickSettings paths scans lightweight).
- 2025-12-29: Runtime exception logging now prefers `CODEX_ROOT/logs` when `CODEX_ROOT` is set (prevents CWD-dependent log placement).
- 2026-01-31: `CODEX_LOG_FILE` now attaches a file handler to the `backend` logger hierarchy as well as the root logger, since `backend.propagate=False` would otherwise yield an empty log file for backend logs (launcher “Write to log file”).
- 2026-03-30: `runtime/logging.py` is the canonical repo-owned backend logging seam. `get_backend_logger(...)` returns `BackendLoggerProxy`, `emit_backend_message(...)` owns human-readable operational logs, `emit_backend_event(...)` owns structured event logs, `build_backend_uvicorn_log_config(...)` owns the repo-controlled uvicorn/bootstrap config, and `configure_backend_root_for_call_trace()` is the only sanctioned call-trace mutation path for the `backend` root logger. Raw `logging.getLogger(...)` stays inside `runtime/logging.py` internals and the explicit `uvicorn.access` integration in `interfaces/api/run_api.py`.
- 2026-02-18: `runtime/logging.py` applies richer default formatting (`%(name)s | %(message)s`), adds `format_log_message(...)` for event-style key/value log text, and supports Rich traceback/path toggles via `CODEX_LOG_RICH_TRACEBACKS` and `CODEX_LOG_RICH_SHOW_PATH`.
- 2026-02-20: `runtime/logging.py` now defaults to message-only console lines (module prefix hidden unless `CODEX_LOG_INCLUDE_LOGGER_NAME=1`), applies a dedicated Rich key/value highlighter (`CodexLogHighlighter`), and renders dense telemetry (`memory_before_*`, `memory_after_*`, `memory_current_*`) as structured multiline blocks for readability.
- 2026-02-20: Logging readability follow-up tightened Rich highlighter regex precedence (typed values override generic catch-all), constrained event highlighting to real event heads, reduced multiline chunk size for dense events, and lowered inline threshold so high-field diagnostics split into consistent multiline blocks.
- 2026-03-02: `TqdmAwareHandler` now emits bridged lines with explicit timestamp + level prefix (`[MM/DD/YY HH:MM:SS] LEVEL`) when writing through `tqdm.write(...)`, restoring temporal/type context for sampler event logs that pass through the tqdm bridge path.
- 2026-03-01: GGUF checkpoint loader contract is hardwired to forward dequantization when policy is omitted (`load_gguf_state_dict(..., dequantize=None) -> dequantize=False`); launcher/backend no longer expose GGUF exec bootstrap knobs.
- 2026-03-01: `--lora-online-math=activation` remains reserved/not implemented and fails loud as a standalone runtime contract.
- 2026-01-04: Added `runtime.checkpoint.io.load_gguf_state_dict(...)` as the canonical GGUF load wrapper so runtime codepaths honor global GGUF flags consistently (no direct loader calls).
- 2026-01-18: `runtime.checkpoint.io.load_gguf_state_dict(...)` supports explicit GGUF dequantization policy (`dequantize` + `computation_dtype`) so callers can centralize GGUF loads without importing `apps.backend.quantization.*` directly.
- 2026-01-01: Live preview utilities now live in `runtime/live_preview.py` (method enum, preview decode helper, and debug preview-factor fitting/logging) so workflows and API layers don’t duplicate preview logic.
- 2026-01-02: Added standardized file header docstrings to runtime modules (doc-only change; part of rollout).
- 2026-01-02: Added standardized file header docstrings to runtime package scaffolding (`__init__.py` and diagnostics modules) (doc-only change; part of rollout).
- 2026-01-03: Standardized upstream references in runtime docs/comments to prefer Hugging Face Diffusers as the behaviour baseline.
- 2026-02-09: Version-counter mitigation is handled at engine conditioning entrypoints (`torch.no_grad()`); runtime no longer carries inference-tensor materialization shims for this class of failure.
- 2026-02-22: `runtime/live_preview.py` now resolves cheap-preview projections by `(profile, channels)` and includes an initial Anima 16-channel bootstrap projection path (no VAE fallback in `Approx cheap`; missing/invalid projections skip preview with deduplicated warnings).
- 2026-02-22: Live preview profile resolution now maps known image-family engines to canonical profiles (`sd15`/`sdxl`/`flow16`) so `Approx cheap` can run on image pipelines that emit 4D 16-channel latents (for example SD3/SD3.5/Flux/ZImage/Anima) using a shared bootstrap projection when applicable.
- 2026-03-31: Env vars read from shared runtime modules must use shared feature namespaces. If a toggle is read outside `families/<family>/` or `engines/<family>/`, do not name it with that family prefix.
