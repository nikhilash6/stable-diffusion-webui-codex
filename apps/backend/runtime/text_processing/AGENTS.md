# Text Processing Runtime — AGENTS Notes
<!-- tags: backend, runtime, text-processing -->
Date: 2025-11-03
Last Review: 2026-02-18
Status: Active

## Scope
Applies to `apps/backend/runtime/text_processing/*` including `classic_engine.py`.

## Design Guidelines
- No legacy imports. Keep APIs Codex-native, clear, and typed.
- Preserve common WebUI behaviour where it still matches the live contract: chunking, emphasis, and LoRA prompt tags.
- Avoid hard dependencies on upstream internals that drift between HF releases.

## 2025-11-03 Update — CLIP TE Forward
- Removed reliance on `embeddings.position_ids` which is not guaranteed across HF CLIP variants.
- Generate `attention_mask` and `position_ids` per batch and pass only if the text encoder’s forward supports them (signature introspection).
- Emit actionable logs on precision fallback; never silence errors.

## 2025-12-03 — TE dtype alignment
- Classic CLIP engine now keeps embeddings in the selected TE dtype (no unconditional fp32 upcast); TE dtype is configured via Web UI / memory manager (no env override).
- T5 engine moves shared embeddings to the chosen TE dtype from memory manager instead of forcing fp32.

### Wrapper-first calling convention
- Classic engine now calls the text encoder via the wrapper (`self.text_encoder(...)`) instead of the inner transformer. The wrapper forwards to the Codex CLIP and harmonises the return shape with HF semantics.

## Invariants
- Token chunking is stable: BOS + chunk + EOS, padding with configured PAD id.
- Emphasis/Textual Inversion application remains before the final layer norm and after hidden state selection (`clip_skip`).

## Risks
- CLIP forward signatures vary across `transformers` versions; introspection guards the call but monitor logs for unusual masks being ignored.
- If future models drop `pooler_output`, the engine must gate pooled features accordingly.

## 2025-11-03 State-Dict Normalization (CLIP)
- The loader now normalizes CLIP state dicts that come rooted at `text_model.*` by lifting them under `transformer.*` so they match `IntegratedCLIP` keys. This avoids spurious `Missing/Unexpected` when checkpoints already use the modern `text_model.*` layout but omit the outer `transformer` prefix.

## Not Implemented
- Non-classic emphasis variants outside the configured registry will raise `NotImplementedError` when wired.

## 2026-01-02 — Notes
- 2026-01-02: Added standardized file header docstrings to `__init__.py`, `emphasis.py`, `extra_nets.py`, `parsing.py`, and `textual_inversion.py` (doc-only change; part of rollout).
- 2026-02-09: Version-counter mitigation is handled at engine conditioning entrypoints (`torch.no_grad()`); text processing must not rely on internal inference-mode toggles.
- 2026-02-17: `extra_nets.py` keeps strict filename/path alias LoRA resolution for prompt tags (`<lora:filename[:text_encoder_weight][:unet_weight]>`); SHA tokens are transport-only (`extras.lora_sha`) and are not valid prompt aliases.
- 2026-03-26: `extra_nets.py` is now lora-only. Only valid `<lora:name>`, `<lora:name:weight>`, and `<lora:name:text_encoder_weight:unet_weight>` tags have parser semantics here; invalid LoRA-like tags fail loud, TI/token-merge/clip-skip/sampler/scheduler/geometry/cfg/seed/denoise/tiling have no prompt-tag seam, and non-LoRA angle-bracket text is left literal in the prompt.
