<!-- tags: backend, huggingface, zhen-nan, metadata-mirror -->

# apps/backend/huggingface/zhen-nan Overview
Date: 2026-05-23
Last Review: 2026-05-23
Status: Active

## Purpose
- Stores metadata-only local mirrors for public `zhen-nan` Hugging Face repositories used by Codex tooling.
- These mirrors provide config metadata for strict/offline planning and GGUF conversion presets without bundling model weights.

## Key files
- `apps/backend/huggingface/zhen-nan/L2P/` — metadata-only mirror for the public `zhen-nan/L2P` GGUF conversion denoiser preset.

## Notes
- Do not add `*.safetensors`, `*.safetensors.index.json`, `*.bin`, `*.pth`, `*.pt`, `*.ckpt`, `*.gguf`, or `*.onnx` here.
- Do not keep upstream `.gitattributes` files in this mirror unless this repository intentionally adopts Git LFS for the mirror.
- Tools/GGUF Converter presets use these configs as metadata only; the operator supplies the real SafeTensors file or folder through the separate conversion form field.
- L2P uses the shared Qwen3-4B text encoder metadata under `apps/backend/huggingface/Qwen/Qwen3-4B/`.
