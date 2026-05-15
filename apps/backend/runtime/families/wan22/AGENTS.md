<!-- tags: backend, runtime, wan22, gguf, streaming, transformer -->
# apps/backend/runtime/families/wan22 Overview
Status: Active
Last Review: 2026-03-31

## Purpose
- WAN 2.2 GGUF runtime components used by WAN engines.
- Native `WanTransformer2DModel` execution, scheduler orchestration, VAE IO, and streaming support.

## Key files
- `wan22.py` — public runtime facade used by WAN engines.
- `config.py` — run/stage config parsing and device/dtype resolution.
- `run.py` — txt2vid/img2vid orchestration and streaming entrypoints.
- `sampling.py` — stage sampling loops and block-progress wiring.
- `sdpa.py` — WAN22 SDPA policy/chunking wrapper and SRAM-mode context plumbing for self-attention integration.
- `scheduler.py` — WAN-specific scheduler helpers.
- `text_context.py` — local-files-only tokenizer/text-encoder loading for prompt embeddings.
- `stage_loader.py` — mounts base GGUF stage weights via `using_codex_operations(..., weight_format="gguf")`.
- `stage_lora.py` — stage LoRA application for base GGUF weights.
- `model.py` — transformer architecture and GGUF keyspace resolution.
- `vae_io.py` — VAE load/encode/decode helpers.
- `streaming/` — chunked/core-streaming infrastructure.

## Expectations
- Keep runtime behavior aligned with `apps/backend/engines/wan22/`.
- 2026-04-09: `config.py` now re-checks selected GGUF family truth through `runtime/model_registry/detectors/wan22.py::inspect_wan22_gguf_path(...)`; `wan_single` accepts only structural `WAN22_5B`, while `wan_high` / `wan_low` accept only structural `WAN22_14B`. Do not trust filename tokens or repo hints alone at this seam.
- Current Phase 1 SRAM integration is self-attention-only. WAN22 cross-attention stays on the PyTorch SDPA path until the generic backend proves speed and VRAM advantage on the narrow self-attention slice.
- WAN22 self-attention may hand the generic SRAM bridge non-overlapping dense `[B,H,S,D]` permute views. Do not force blind `.contiguous()` materialization on Q/K/V just to enter the SRAM path.
- `sdpa.py`, `run.py`, `sampling.py`, and `stage_loader.py` must use generic SRAM naming/telemetry; do not reintroduce WAN-only backend naming or `attn_core` selectors into the active path.
- Base `.gguf` artifacts are the supported root-path input; unsupported packed artifacts must fail loud.
- Exact stage ownership now matters in config/runtime too: WAN 2.2 5B uses `RunConfig.single` plus dedicated single-stage txt2vid/img2vid entrypoints, while 14B keeps `RunConfig.high` + `RunConfig.low`. Do not fabricate empty `high/low` stages for 5B convenience.
- WAN22 GGUF sampler support is intentionally narrow: `uni-pc` (optional solver hint), `euler`, and `euler a`. Non-lane labels (`uni-pc bh2`, `euler cfg++`, `euler a cfg++`) must fail loud and must not collapse to executable lanes.
- `text_context.py` must keep tokenizer/model loading local-files-only and strict on device/key mismatches.
- Stage and VAE placement remain owned by the memory manager.
- `stage_lora.py` is a no-remap seam: it may interpret WAN22 LoRA logical keys through `keymap_wan22_transformer.py`, but it must not invent runtime state-dict remaps or alias shims outside that seam.
- Current WAN22 stage-LoRA diagnostics must classify logical misses (`matched`, `resolver_none`, `resolved_target_missing`, `unsupported_i2v_branch`, `alias_collision`) and report unsupported tensor suffix families separately; the upstream I2V image branch (`k_img`, `v_img`, `norm_k_img`, `img_emb.proj.*`) stays explicitly unsupported in the local runtime.
- `stage_lora.py` owns truthful stage-LoRA coverage diagnostics. It must distinguish local matcher gaps, missing local targets, and unsupported image-branch/suffix families instead of collapsing them into opaque partial-coverage noise.
- `stage_lora.py` also owns the pre-patch structural compatibility gate for mapped standard LoRA pair weights plus the generic materialized patch-shape check for parsed WAN22 stage LoRAs. The mounted stage shape truth now comes from the live parameter owner, not `state_dict()` heuristics, and the cheap SafeTensors header pass must reject obvious mismatches before `sf.load_file(...)`.
- Wan2.1 480p-shape adapters must still fail loud with the dedicated incompatibility error on the larger `wan22_14b` runtime profile, but that diagnosis is now derived from the actual mismatch records instead of a hardcoded mounted-profile heuristic deciding compatibility.
- The current local WAN22 runtime does not expose the upstream I2V image branch (`k_img`, `v_img`, `norm_k_img`, `img_emb`). Stage LoRA reporting must keep those families explicitly unsupported until the runtime model grows the real branch.
- Unsupported I2V image-branch families must stay visible in diagnostics and natural partial-coverage reporting. The stage-LoRA seam may apply supported keys from mixed adapters, but it must not pretend the unsupported image branch was loaded. The default threshold-free path keeps that partial read natural; a non-zero `CODEX_WAN22_STAGE_LORA_MIN_MATCH_RATIO` still remains an explicit strictness control and may hard-fail low coverage.
