# apps/backend/interfaces Overview
<!-- tags: backend, api, validation -->
Date: 2025-12-05
Last Review: 2026-03-06
Status: Active

## Purpose
- Defines API-facing schemas and adapters that expose backend capabilities to the Codex frontend and external clients.

## Subdirectories
- `api/` — FastAPI endpoint implementations and adapters.
- `schemas/` — Pydantic/dataclass schemas describing request/response payloads.

## Notes
- Keep schemas in sync with the frontend API client (`apps/interface/src/api`).
- Avoid embedding business logic here—delegate to services/use cases and focus on validation and serialization.
- Reference: `.sangoi/reference/models/model-assets-selection-and-inventory.md` captures the “how assets are listed + selected” contract (inventory → SHA selection → backend resolution).
- API workers should reuse a single `InferenceOrchestrator` instance per process to preserve engine caches/VRAM across requests. See `api/routers/generation.py` (`_ORCH` singleton).
- 2025-11-14: `/api/txt2img` enforces the semantic contract (e.g., `prompt`, `negative_prompt`, `width`, `extras.hires`); legacy compat keys were removed (requests must use the canonical selectors / payload fields).
- 2025-11-21: SPA static mount now registers after all `/api/*` routes to prevent POSTs from being intercepted by the UI fallback; invalid txt2/img2/video payloads raise HTTP errors instead of returning 200 with a background error.
- 2025-11-21: Module-level `app` remains available for ASGI servers, but the preferred entrypoint is the uvicorn factory `apps.backend.interfaces.api.run_api:create_api_app`. Factory and direct `:app` both build the same FastAPI instance.
- 2025-11-14: `create_api_app(argv, env)` is the canonical FastAPI factory; when launching uvicorn manually use `--factory apps.backend.interfaces.api.run_api:create_api_app` so the runtime bootstraps before serving (the TUI/launcher already calls it).
- 2025-12-03: `/api/txt2img` extras now accept `hires.refiner` alongside the global `extras.refiner`, raising HTTP 400 on malformed nested refiner configs.
- 2026-02-08: Swap-model payload contract now uses `switch_at_step` (global + hires nested refiner objects) with pointer semantics (`1 <= switch_at_step < total_steps`) instead of refiner step-count semantics.
- 2025-12-03: `/api/tasks/{task_id}/cancel` allows best-effort cancellation (immediate vs after_current flag); workers abort event streaming with `error: cancelled` when `mode=immediate`.
- 2025-12-03: `/api/options` now accepts `codex_{core,te,vae}_{device,dtype}` to set per-role backend/dtype via memory manager; device choices auto/cuda/cpu/mps/xpu/directml, dtype auto/fp16/bf16/fp32.
- 2025-12-05: `/api/txt2img` extras agora aceitam um objeto opcional `text_encoder_override` (family + label + components[]) validado como JSON; quando presente, o worker de txt2img o encaminha como `engine_options.text_encoder_override` para o orchestrator/engines, que por sua vez repassam o override ao `runtime.models.loader` (via `TextEncoderOverrideConfig`). A partir de 2025-12-06, `components[]` também aceita entradas no formato `alias=/abs/path/to/weights.safetensors` para overrides por arquivo (ex.: Flux), que o loader interpreta como `explicit_paths` sem depender de um endpoint dedicado de labels (roots vêm de `apps/paths.json` via `/api/paths`).
- 2025-12-05: `/api/options` e `/api/txt2img` expõem flags `codex_smart_offload`/`codex_smart_fallback`/`codex_smart_cache` (checkboxes na UI → `smart_offload`/`smart_fallback`/`smart_cache` no payload) para controlar descarregamento entre estágios, fallback para CPU em caso de OOM e caches de condicionamento SDXL; quando um job inclui `smart_cache`, o valor por-job prevalece sobre o snapshot global, permitindo rodar jobs mistos em uma mesma sessão.
- 2025-12-05: `/api/engines/capabilities` passa a incluir um bloco opcional `smart_cache` com contadores de hits/misses agregados (por bucket) para diagnóstico de caching de SDXL no runtime.
- 2025-12-05: `/api/memory` agora usa `apps.backend.runtime.memory.memory_management.memory_snapshot()` para expor um snapshot estruturado de VRAM/CPU (backend, dispositivo primário, probe, budgets, stats do torch e modelos carregados); clientes que só leem `total_vram_mb` continuam atendidos, mas novas UIs devem consumir o snapshot completo.
- 2025-12-06: `_bootstrap_runtime` agora pré-calcula o inventário de modelos (`apps.backend.inventory.cache.refresh()`) durante o bootstrap do backend, de forma que `/api/models/inventory` esteja quente quando a UI abrir o QuickSettings; a rota continua expondo `?refresh=true` e `POST /api/models/inventory/refresh` para rescans explícitos.
- 2025-12-14: `/api/txt2vid` e `/api/img2vid` populam `steps` em `Txt2VidRequest/Img2VidRequest` e o plano de vídeo (`build_video_plan`) lê `guidance_scale` (alinhamento de contrato com o runtime).
- 2025-12-16: Added `/api/vid2vid` (multipart: `video` upload + JSON `payload`) and `/api/output/{rel_path}` for root-scoped serving of exported videos.
- 2026-02-28: `/api/vid2vid` is intentionally parked in `generation.py` (HTTP 400 + explicit placeholder detail) until capability-driven router/runtime contract finalization; historical wan_animate notes remain implementation history only.
- 2025-12-19: `/api/tools/convert-gguf` expanded quantization menu and now accepts `tensor_type_overrides` (regex → quant per tensor) for mixed schemes and advanced tuning.
- 2025-12-29: `/api/paths` now resolves `apps/paths.json` via `CODEX_ROOT` (required), keeping QuickSettings path-based filtering stable across launchers and CWD changes.
- 2025-12-29: `run_api.py` no longer uses `os.getcwd()` for repo files (settings/blocks/tabs/workflows/presets/tmp); it uses the resolved project root so the backend behaves the same no matter where it’s launched from.
- 2025-12-29: `/api/models/inventory` now returns repo-relative `path` values (when under `CODEX_ROOT`) to avoid leaking absolute host paths to the UI; WAN video APIs resolve repo-relative `wan_*` paths back to absolute under `CODEX_ROOT` so runtime never depends on CWD.
- 2025-12-29: API port-guard (`port_free`) now checks IPv4 + IPv6 loopback/wildcard (0.0.0.0/127.0.0.1/::/::1) to avoid “localhost” split-brain where an IPv6-only listener exists but the guard only tested IPv4.
- 2025-12-30: Suppressed uvicorn access-log spam for `/api/tools/convert-gguf/{job_id}` polling; opt out via `CODEX_UVICORN_ACCESS_LOG_TOOLS=1`.
- 2025-12-31: `/api/img2img` now accepts `img2img_extras` (incl. `text_encoder_override` + `tenc_sha`), enforces `tenc_sha` for `.gguf`, and forwards request-level `engine_options` into the orchestrator (parity with `/api/txt2img`, needed for Flux/Kontext GGUF runs).
- 2025-12-31: `/api/img2img` now infers missing `img2img_width/img2img_height` from the init image (snapped to multiples of 8).
- 2026-01-22: `/api/img2img` no longer applies engine-specific Kontext defaults for omitted fields; request validation is uniform across engines (fail fast when required fields are missing).
- 2026-01-01: `/api/txt2img` now accepts `clip_skip`, and `/api/img2img` accepts `img2img_clip_skip` (wired into prompt controls before conditioning is computed).
- 2026-01-01: `/api/{txt2img,img2img}` now supports live preview streaming: backend reads UI settings (`show_progress_every_n_steps`, `show_progress_type`, `live_previews_image_format`) and attaches `preview_image`/`preview_step` to task `progress` SSE events when a new preview is available.
- 2026-01-01: Live preview config parsing + payload encoding/attachment now live in `apps/backend/services/live_preview_service.py` so `api/run_api.py` doesn’t duplicate preview logic.
- 2026-01-01: Added `--debug-preview-factors` (launcher arg) so the runtime can log best-fit latent→RGB preview factors (`[preview-factors]`) for deriving new `Approx cheap` mappings.
- 2026-01-02: `/api/{txt2img,img2img}` accepts checkpoint selection by SHA (10-char short hash or 64-char sha256) via `model` or `extras.model_sha`; VAE/TE selection is request-driven via `extras.vae_sha`/`extras.tenc_sha` (or `img2img_extras.*`).
- 2026-01-24: Settings schema/options were tightened: `/api/settings/schema` is served from `apps/backend/interfaces/schemas/settings_registry.py` (generated from `settings_schema.json`) with JSON fallback, and `apps/settings_values.json` is pruned against the registry on startup (unknown keys dropped; invalid values clamped).
- 2026-01-01: `/api/models` now accepts `?refresh=1` to re-scan checkpoint roots so the UI can pick up newly copied weights without restarting the backend.
- 2026-01-02: Added standardized file header docstrings to interface modules (doc-only change; part of rollout).
- 2026-01-04: Image-family engine keys are `flux1` / `flux1_kontext` / `flux1_chroma` plus the separate FLUX.2 key `flux2` (no legacy aliases); `run_api.py` resolves engine keys via the registry and rejects unknown keys with HTTP 400.
- 2026-01-04: Text encoder override labels are now derived from `apps/paths.json` (`*_tenc` roots via `/api/paths`) and inventory-derived TE files; tooling guardrail added to prevent direct `apps/paths.json` reads outside `infra/config/paths.py`.
- 2026-01-06: `/api/{txt2img,img2img}` now treats `.gguf` checkpoints as core-only: requires `vae_sha` + `tenc_sha` (tenc accepts arrays for multi-encoder models); ZImage enforces exactly 1 (Qwen3) and Flux.1 enforces exactly 2 (CLIP + T5).
- 2026-01-06: API workers now set `engine_options.vae_source`/`engine_options.tenc_source` (`built_in` vs `external`) to make asset selection explicit (pairs with `engine_options.vae_path`/`engine_options.tenc_path` when external).
- 2026-01-06: `/api/{samplers,schedulers}` now returns minimal entries; `/api/{txt2img,img2img}` validates canonical sampler/scheduler selection (including per-sampler scheduler compatibility) and fails fast with HTTP 400.
- 2026-01-06: `/api/{txt2vid,img2vid}` default sampler now uses `uni-pc` (scheduler `simple`) to match WAN22 diffusers scheduler metadata.
- 2026-01-08: `run_api.py` was modularized into router modules under `apps/backend/interfaces/api/routers` (composition-only entrypoint; route logic moved to focused router files).
- 2026-03-06: `/api/img2img` no longer hard-rejects FLUX.2 partial denoise at request parse-time now that the backend continuation path is real; the same router explicitly rejects masked FLUX.2 hires (`img2img_mask` + `img2img_extras.hires.enable`) until that backend path exists.
