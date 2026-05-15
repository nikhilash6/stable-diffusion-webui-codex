# apps/backend/runtime/common/nn Overview
Date: 2025-10-28
Last Review: 2026-02-11
Status: Active

## Purpose
- Shared neural network building blocks (layers, wrappers) used across multiple runtimes/engines.

## Notes
- Keep modules generic; specialize behaviour in the model-specific runtimes instead of here.
- 2026-01-02: Added standardized file header docstrings to `__init__.py`, `base.py`, `clip.py`, `t5.py`, `clip_text_cx.py`, and `unet/{__init__,config,utils}.py` (doc-only change; part of rollout).
- 2026-02-11: `clip.py:IntegratedCLIP` now supports projection layout selection (`text_projection_layout=linear|matmul`) and exposes `_MatmulProjection` so AUTO mode can keep native projection orientation without tensor transpose.
- 2026-02-20: `clip_text_cx.py` and `t5.py` attention lanes now route through runtime attention dispatcher helpers (pre-shaped/explicit PyTorch backend path) instead of direct local SDPA calls.
- 2026-03-21: `_MatmulProjection` now participates in Codex manual-cast semantics, preventing SDXL CLIP-G projection crashes when pooled activations run in compute `fp16` and projection weights remain in storage `bf16`.
