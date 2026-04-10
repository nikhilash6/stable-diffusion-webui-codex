### ABSOLUTE LAW — DO NOT TOUCH LAYER NAMES

Read this twice.

If a checkpoint says a layer is `foo.bar.baz`, then in this repository it stays `foo.bar.baz`.
Not `bar.baz`.
Not `foo_bar_baz`.
Not "normalized".
Not "close enough".

Keymaps here **map keyspaces** so the engine/runtime can understand how different ecosystems name the same conceptual weight.
They do **not** rename stored model keys.
They do **not** strip prefixes.
They do **not** rewrite punctuation.
They do **not** slide dots around.
They do **not** materialize remapped state dicts.

If two ecosystems use different names for the same conceptual weight, the keymap resolves that relationship explicitly and the engine interprets the stored key as-is.
If the layout is unsupported, ambiguous, or structurally incompatible, you fail loud and extend the keymap properly.
You do **not** "normalize" a checkpoint by rewriting layer names in memory. Ever.

Lazy mappings are law too.
Checkpoint/state-dict seams stay lazy by default.
Keymaps and loaders inspect checkpoints through mapping/view APIs first (`shape_of(...)`, lookup views, computed views) and only touch real tensor values at the owner seam that actually needs them.
You do **not** materialize eager `dict(...)` copies of checkpoint mappings as convenience glue.

---

### ACT II – WHERE THE TRUTH LIVES: `.sangoi` AND REUSE

Before you do anything else, you read `SUBSYSTEM-MAP-INDEX.md` first. You use it to find the real seam, then jump into `SUBSYSTEM-MAP.md` only for the bounded hotspot/pipeline/owner section you need.
If you don't know what to change, you don't guess — you search the index first.
`SUBSYSTEM-MAP-INDEX.md` is lookup-only.
`SUBSYSTEM-MAP.md` is discovery and navigation only: hotspot directory, pipeline node chains, owner seams, and pointers to generated artifacts/policy.
Contract authority lives here in root `AGENTS.md` (policy) and in `.sangoi/reference/**` (detailed contracts). Keep those roles split. Do not dump contract matrices back into `SUBSYSTEM-MAP.md`.
When touching `SUBSYSTEM-MAP-INDEX.md` or `SUBSYSTEM-MAP.md`, run this operational checklist before handoff:

- Keep it discovery-only (no contract matrices / drift ledgers).
- Keep the index/map split honest:
  - `SUBSYSTEM-MAP-INDEX.md` = lookup-only front door
  - `SUBSYSTEM-MAP.md` = discovery/node atlas only
- Ensure hotspot discoverability is explicit (`keymaps`, `vae_codex3d.py`, `hires_fix.py`).
- Update `SUBSYSTEM-MAP-INDEX.md` and `SUBSYSTEM-MAP.md` in the same tranche when any mapped node chain changes because of file moves, owner-path changes, public route additions/removals, or top-level owner functions changing.
- If a touched `apps/**` file is part of a mapped node chain or hotspot entry, refresh its file header block in the same tranche. This is additive to the standing rule that every touched `apps/**` file header must stay truthful.
- Regenerate backend index artifacts:
  - `backend_py_paths_file="$(mktemp /tmp/backend_py_paths.XXXXXX.txt)"`
  - `git ls-files apps/backend | rg "\\.py$" | LC_ALL=C sort > "$backend_py_paths_file"`
  - `python3 .sangoi/.tools/dump_apps_file_headers.py --out .sangoi/reports/tooling/apps-backend-file-header-blocks.md --root apps/backend --fail-on-missing`
  - `python3 .sangoi/.tools/build_backend_py_book_index.py --paths "$backend_py_paths_file" --headers .sangoi/reports/tooling/apps-backend-file-header-blocks.md --out .sangoi/reports/tooling/backend-py-book-index.md`
- Validate parity/checks:
  - `python3 .sangoi/.tools/build_backend_py_book_index.py --paths "$backend_py_paths_file" --headers .sangoi/reports/tooling/apps-backend-file-header-blocks.md --out .sangoi/reports/tooling/backend-py-book-index.md --check`
  - `bash .sangoi/.tools/link-check.sh .sangoi`
  - `bash .sangoi/.tools/link-check.sh .`

- If you touch an `apps/**` source file, you keep its **file header block** honest. If the purpose or top-level symbols changed, you update them.
  - What it is: the standardized top-of-file block containing `Repository:` + `SPDX-License-Identifier:` + `Purpose:` + `Symbols (top-level; keep in sync):`.
  - Where it lives: `.py` = module docstring (first statement); `.ts` = top block comment (`/* ... */`); `.vue` = top HTML comment (before `<template>`).
  - Standard: `.sangoi/policies/file-header-block.md`. Helper: `python3 .sangoi/.tools/review_apps_header_updates.py`.

---

### ACT III – GIT, COMMITS, AND HISTORY

`.sangoi/` is a separate Git repository and is ignored by this root repository.

- Root commits (`git add/commit` from this repo) do not include `.sangoi/**`.
- When a task targets `.sangoi`, run Git commands explicitly against that repo (`git -C .sangoi ...`).
- Keep commit/push operations split by repository and report both hashes when both repos change.

When your turn is done:

- You verify the **file header block** (top-of-file `Repository/SPDX/Purpose/Symbols`) for **every touched file** under `apps/**` (even if the diff "seems small”), and update Purpose/Symbols if needed.
- Use `python3 .sangoi/.tools/review_apps_header_updates.py --show-body-diff` to review "changed body, unchanged header” cases.
- If the touched `apps/**` file is referenced by a mapped node chain, hotspot entry, or owner seam in the subsystem docs, update the manual map/index in the same tranche instead of leaving the atlas stale.

If you touch dependencies or configs, you update the proper manifest or lockfile and note the impact.

---

### ACT IV – ARCHITECTURE, LEGACY, MODELS, PYTHON

- The default core for attention is PyTorch SDPA.
- You list risks, side effects, globals.
- Codex prefix or suffix is used where it actually adds meaning.
- `Codex` is an intentional project naming convention. Do **not** strip it just because a symbol looks long.
- If naming or structure is bad, fix the fake namespace, owner shape, module boundary, or alias soup. Do **not** "clean it up" by deleting the `Codex` prefix, and do **not** invent pseudo-namespaces like `CodexProcessing.Txt2Img` unless a real namespace or module exists.
- You always code in Codex style:
  - Dataclasses, enums and similar.
  - Small modules with clear seams.
  - Explicit and fail-loud errors.
  - Readable names.

Testing policy: do not add or maintain automated tests unless explicitly requested by the repo owner.
Prefer fail-loud runtime contracts and manual validation workflows.

When we say "pipeline" in this repo, we mean the whole trip:
Frontend command → API request → task_id → SSE events → model load → sampling → postprocess/encode → finished artifact.

Drift is not a vibe. Drift is a bug.
Drift is when the _same mode_ (txt2img/img2img/txt2vid/img2vid/vid2vid) takes a different trip depending on engine.
Drift Also counts as drift when any of this changes per engine for the same mode:

- Contract drift: request schema/defaults, progress semantics, preview semantics, error semantics, or result fields.
- Stage drift: normalize → resolve engine/device → ensure assets/load → plan → execute → postprocess/encode → emit (skipped, duplicated, re-ordered, or hidden).
- Ownership drift: routers doing pipeline work, engines owning modes, or use-cases bypassed.

**Policy (Option A): one canonical use-case per mode.**

- `apps/backend/use_cases/{txt2img,img2img,txt2vid,img2vid,vid2vid}.py` owns the mode pipeline.
- Engines are adapters and hooks. They load models and expose primitives. They do **not** re-implement the mode.
- Routers stay thin: validate + dispatch + stream.
- The orchestrator stays the coordinator: resolve engine/device, cache/reload, run, and emit events.
- Shared, reusable stages live in `apps/backend/runtime/pipeline_stages/`. If it's shared, it goes there. If it's not shared, it stays in the canonical use-case.

**Ownership law: one concept, one owner path.**

- If a concept already lives under a typed nested owner, it stays there. You do **not** mirror it into flat shadow fields for convenience.
- Examples of forbidden shadow-owner drift:
  - `processing.hires.swap_model` plus `processing.hires_swap_model`
  - `processing.hires.refiner` plus `processing.hires_refiner`
  - `processing.hires.*` plus `processing.hr_*`
  - nested selector/config ownership plus sibling `*_name` / `*_path` aliases
- If a callsite wants a flatter shape, redesign the callsite. Do **not** duplicate the owner.

**Native names stay native.**

- If a field is named after a native concept, it carries only that concept.
- `refiner` is the native SDXL refiner seam. It is **not** a generic model-swap bucket.
- Generic model swap must live under explicitly generic naming such as `swap_model`, with its own typed owner.
- `extras.swap_model` is the top-level first-pass stage config:
  - it owns `enable` + `switch_at_step` semantics for mid-generation base-pass swapping;
  - it is **not** selector-only.
- `extras.hires.swap_model` is the selector-only second-pass owner:
  - it replaces the whole hires engine for the second pass;
  - it does **not** grow stage-pointer fields.
- `extras.refiner` / `extras.hires.refiner` remain SDXL-native refiner stages only.
- When a public/runtime seam is renamed to the native owner, the old name dies everywhere in the same tranche: router payloads, frontend state, component props/emits, run helpers, docs, and AGENTS.
- `hires.checkpoint`-style ghosts are forbidden once `hires.swap_model` exists.

**Derived-plan law: execution-only, selector-free.**

- A derived plan/helper struct may carry computed execution values such as target size, step count, denoise, or chosen upscaler.
- A derived plan/helper struct must **not** own selectors, model references, checkpoint names, swap-model config, refiner config, modules, or any other request-shaped contract data.
- If a plan/helper struct needs to carry those fields, that struct should not exist; compute from the canonical typed owner instead.
- `HiResPlan`-style shadow containers are forbidden as a destination for contract ownership.

**Unsupported seams fail loud.**

- If a mode/surface does not support a typed seam yet, you do **not** hide the payload and continue.
- Hide or clear the control in the UI when possible, and still fail loud at request build/runtime boundaries if stale state survives.
- Example: img2img must not silently drop `swap_model` / refiner state that only exists for txt2img hires.
- Public-state law: if a seam exists in frontend state, request payloads, or router parsing, it must also have a real execution owner.
  - No hidden/store-only `swap_model` surfaces.
  - No request/runtime surfaces that quietly do nothing.

**Recurring failure classes: stop recreating these.**

- **Owner-path drift**
  - Symptom: a router, engine, or convenience helper starts owning mode/stage work that belongs to a canonical use-case or shared stage.
  - Do: keep mode ownership in `apps/backend/use_cases/*.py`; if exact sampling mutation is needed, install it from `apps/backend/use_cases/img2img.py` and enter it from `apps/backend/runtime/pipeline_stages/sampling_execute.py` after canonical LoRA activation.
  - Do **not**: wrap whole runtime branches earlier, add engine-specific mode pipelines, or move slot/layout ownership out of owner modules such as `apps/backend/runtime/adapters/ip_adapter/layout.py`.

- **Contract drift**
  - Symptom: UI, request, parser, processing, and runtime names/defaults/allowlists stop matching each other.
  - Do: cut over the whole seam in one tranche across `apps/interface/**`, `apps/backend/interfaces/api/routers/generation.py`, and the runtime/processing owners.
  - Do **not**: keep alias readers, stale allowlists, or half-renamed payloads.
  - Examples: `inpaintMode` / `img2img_inpaint_mode`; `switch_at_step`; `apps/backend/runtime/families/supir/runtime.py` resolving the loaded checkpoint from the canonical model reference.

- **Lazy/load-path drift**
  - Symptom: checkpoint helpers materialize mappings into `dict(...)`, move tensors too early, or rebuild IO semantics in convenience glue.
  - Do: keep mapping/view ownership in `apps/backend/runtime/state_dict/views.py`; keep checkpoint IO in `apps/backend/runtime/checkpoint/io.py`; keep logical-shape load truth at the loader seam.
  - Do **not**: materialize eager copies, normalize keys in helpers, or move load semantics into family-local shortcuts.

- **Keymap/keyspace drift**
  - Symptom: family-specific remap logic reappears in runtime helpers, stage loaders, or patch code instead of canonical keymap owners.
  - Do: put keyspace rules in `apps/backend/runtime/state_dict/*.py`, for example `apps/backend/runtime/state_dict/keymap_wan22_transformer.py`, then have runtime consumers call that owner.
  - Do **not**: add prefix strippers, rekey helpers, or stage-local logical remappers.

- **Parity drift**
  - Symptom: the same mode takes a different trip or emits different public semantics across engines/tabs without an approved canonical hook.
  - Do: keep mode trips aligned through shared owners such as `apps/backend/use_cases/img2img.py`, `apps/backend/runtime/pipeline_stages/video.py`, and the shared frontend results/history owners.
  - Do **not**: special-case one engine/tab by cloning request/result/progress logic into a peer-owned surface.
  - Example: Z-Image masked img2img stays on its truthful runtime path in `apps/backend/runtime/families/zimage/model.py`; it does **not** get shoved through SD-style image-conditioning just because another family uses that seam.

- **Fail-loud erosion**
  - Symptom: unsupported or incomplete states survive as silent fallback, quiet no-op, or “best effort” continuation.
  - Do: reject stale state at UI/request/runtime boundaries and raise explicit errors when a required seam has no real execution owner.
  - Do **not**: keep hidden/store-only surfaces, default missing mandatory stages, or silently accept zero compatible layers.
  - Examples: `apps/backend/patchers/lora_apply.py` must fail when compatibility is zero; `apps/backend/interfaces/api/routers/generation.py` must reject unsupported request surfaces before task creation.

**Pre-merge anti-rerun checklist**

- Is the owner path still singular and canonical?
- Did every public field/default/allowlist change across UI/request/runtime in the same tranche?
- Did the loader/keymap path stay lazy and single-owner?
- Did family-specific mapping/layout logic stay in the canonical owner module?
- Does the same mode still take the same trip across engines and tabs?
- Will unsupported or stale state fail loud instead of degrading quietly?

If an engine needs special behavior, you add a hook that the canonical use-case calls.
If you can't express it as a hook, you stop and redesign until you can.
No engine-specific pipelines. No zoo.

Imports outside `/apps` are banned.
Only `apps.*` lives in active code.

If a feature has not been implemented, you raise:

```python
NotImplementedError("<feature> not yet implemented")
```

Model loading is a minefield you cross with a map.
You follow `.sangoi/research/models/model-loading-efficient-2025-10.md`.

- Supporting a family in `diffusers` format does **not** delegate contract truth to external `diffusers` helpers/imports.
- If this repo supports a `diffusers` surface, classification, component requirements, and family-specific constraints stay in repo-owned loader/detector/parser seams.
- Family-native external asset slots stay explicit and named. Do **not** collapse multi-slot families into generic selector bags when the contract depends on slot identity.
- When debugging model/adapter/runtime integration, prefer bounded seam proofs before E2E guesses:
  - preprocess parity against the canonical processor;
  - encoder/load parity against the same checkpoint through the canonical repo seam;
  - projector/module parity against the real checkpoint tensors;
  - patch math parity in isolation;
  - binding/layout parity by translated parameter names, not width-only heuristics.
- Use real checkpoints plus the appropriate local/official reference shelf under `.refs/`, and keep each proof scoped to one seam so you know exactly what failed.

**IP-Adapter image-encoder postmortem: this was done wrong once, and never again.**

- A prior IP-Adapter implementation loaded the image encoder through bespoke helpers instead of the canonical loader stack:
  - `_normalize_image_encoder_state_dict(...)`
  - `cleaned_state_dict(...)`
  - `rekey_vision_state_dict(...)`
  - `convert_openclip_checkpoint(...)`
  - raw `nn.Module.load_state_dict(...)` inside `ClipVisionEncoder`
- That was wrong because it bypassed the exact repo mechanisms that already exist to keep model loading honest:
  - canonical keyspace resolution
  - lazy mapping/view ownership
  - `fail_on_key_name_rewrite(...)`
  - `safe_load_state_dict(...)`
- That bespoke path also violated the architectural rule already stated above:
  - it eagerly materialized checkpoint mappings into `dict(...)`
  - it rewrote stored keys in memory
  - it created component-specific loader drift
  - it caused avoidable RAM blow-ups during image-encoder load
  - it violated the canonical build/mount pattern by constructing the runtime module outside the memory-owner device/dtype path, then relying on later runtime offload wrappers to clean it up
- The correct rule is simple:
  - the IP-Adapter image encoder is **not** a special loader class
  - if a VAE or text encoder would load through canonical keymap/view resolution plus `safe_load_state_dict(...)`, then the image encoder must do the same
  - if a CLIP-vision layout needs support, extend the canonical loader/keymap ownership and keep it lazy
  - if the layout is unsupported, fail loud
  - if a component is memory-managed, the module itself must also be built/mounted through the canonical owner path before weight load: resolve the owner device/dtype first, construct under the same `using_codex_operations(**to_args, ...)` pattern used by the central loaders, place the module on that owner device/dtype, then call `safe_load_state_dict(...)`, and only after that rely on runtime offload/reload
- Never again:
  - do **not** add adapter-local “cleaned state dict” helpers, prefix strippers, rekey shims, or raw `module.load_state_dict(...)` shortcuts just to get an auxiliary component loading quickly
  - do **not** bypass the rewrite guard by normalizing keys before the canonical loader sees them
  - do **not** ship a bespoke image-encoder loader when the rest of the repo already has the right loading contract
  - do **not** assume that wrapping a module in `ModelPatcher` later makes an off-pattern build/load canonical; the birth/load path must already follow the memory-manager owner contract

Keymap law: see **ABSOLUTE LAW — DO NOT TOUCH LAYER NAMES** at the top of this file.
The same no-rename/no-strip/no-punctuation-rewrite rule applies during model loading and engine/runtime keyspace interpretation.

You prefer SafeTensors.
You call `torch.load(..., weights_only=True, mmap=True)` when it applies.

Keep Python disciplined.
You do not add shebangs to source files.

When agent-side verification requires running the WebUI/backend on CPU, use the repository-local `uv` toolchain and explicit CPU env overrides.

- Prefer local `uv`: `./.uv/bin/uv` (never system/global `uv` for this workflow).
- Required env for CPU lane: `CODEX_ROOT="$PWD" PYTHONPATH="$PWD" CODEX_TORCH_MODE=cpu CODEX_TORCH_BACKEND=cpu`.
- Example check command pattern:
  - `CODEX_ROOT="$PWD" PYTHONPATH="$PWD" CODEX_TORCH_MODE=cpu CODEX_TORCH_BACKEND=cpu ./.uv/bin/uv run --python .venv/bin/python --no-sync -m apps.backend.interfaces.api.run_api --help`
- WSL heavy-model safety rule: for LTX/WAN-class giant assets, default to header-only / metadata-only inspection in WSL. Do not materialize tensors, assemble full runtimes, initialize full pipelines, or run forward passes unless the user explicitly asks. Prefer GGUF metadata readers and SafeTensors header readers first.

---

### ACT V – FRONTEND, LAYOUT, AND CSS

If you want to change something in `apps/interface/src/styles`, you read the local `AGENTS.md` before you touch a single selector.
Ignore that, and your pull request does not pass.

Styles for `apps/interface/src/styles` are not a dumping ground.
Common rules belong where they will be reused.
Variants are named with intent.
Do not litter with vague utilities that hide confusion.

---

### ACT VI – `.refs/` REFERENCE SOURCES

This repo keeps a serious local reference shelf under `.refs/`. Use it before you reach for `web.run`.

What lives there right now:

- upstream or related codebases such as `ComfyUI`, `ComfyUI-GGUF`, `ComfyUI-SeedVR2_VideoUpscaler`, `Forge-A1111`, `diffusers`, `flash-attention`, `pytorch`, `sd-scripts`, `llama.cpp`, `k-diffusion`, `LyCORIS`, `adetailer`, `WanVideoWrapper`, `Stable-Video-Infinity`, `LightX2V`, `IP-Adapter` and `open-tv`
- git-backed extensions and nested repos inside `.refs/Forge-A1111`
- model/index/reference artifacts such as `hf-model-indexes`, normalized stream JSON dumps, and local Gemini helper/reference files

Rules:

- When you need upstream behavior, loader details, extension behavior, API shape, or model-family reference code, search `.refs/` first instead of flailing with `web.run`.
- Before researching any git-backed reference under `.refs/`, run `git -C <that-reference-repo> pull --ff-only` so you are reading the freshest code.
- In this repo, that pull-first rule applies to the current git-backed references under `.refs/`, including repositories like `ComfyUI`, `ComfyUI-GGUF`, `ComfyUI-SeedVR2_VideoUpscaler`, `diffusers`, `pytorch`, `sd-scripts`, `llama.cpp`, `k-diffusion`, `LyCORIS`, `adetailer`, `WanVideoWrapper`, `Stable-Video-Infinity`, `LightX2V`, `open-tv`, and the nested git-backed repos inside `.refs/Forge-A1111`.
- `.refs/` stays reference-only. Durable conclusions belong in tracked docs, comments, or contracts inside this repo, not as ephemeral memory.

You read them. You do not import them into `apps/**`. You do not copy them into active code. You extract the intent, then you re-implement it clean and the our good Codex style.
