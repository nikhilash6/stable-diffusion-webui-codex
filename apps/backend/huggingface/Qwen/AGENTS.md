<!-- tags: backend, huggingface, qwen, metadata-mirror -->

# apps/backend/huggingface/Qwen Overview
Date: 2026-05-15
Last Review: 2026-05-23
Status: Active

## Purpose
- Stores metadata-only local mirrors for public Qwen and Qwen Image Hugging Face repositories.
- These mirrors support strict/offline loader planning by keeping model index, component configs, tokenizer assets, processor assets, and weight-index metadata available without bundling model weights.

## Key files
- `apps/backend/huggingface/Qwen/Qwen-Image-2512/` — public text-to-image snapshot for the newest available open-weight Qwen Image generation repo in Hugging Face during the 2026-05-15 vendor pass.
- `apps/backend/huggingface/Qwen/Qwen-Image-Edit-2511/` — public image-edit snapshot using `QwenImageEditPlusPipeline`, including its `processor/` metadata.
- `apps/backend/huggingface/Qwen/Qwen3-4B/` — exact Qwen3-4B text-encoder metadata used by Z-Image and Z-Image L2P GGUF converter profiles.

## Notes
- Do not add `*.safetensors`, `*.bin`, `*.pth`, `*.pt`, `*.ckpt`, `*.gguf`, or `*.onnx` here.
- Do not keep upstream `.gitattributes` files in this mirror unless this repository intentionally adopts Git LFS for the mirror.
- `*.safetensors.index.json` files are metadata and are allowed; they do not contain tensor payloads.
- `processor/` files and `chat_template.jinja` are lightweight sidecars. They are not proof that a tokenizer or runtime component is loadable by themselves.
- `Qwen-Image-2.0` is not mirrored here until a concrete public Hugging Face repository is selected.
