# zhen-nan/L2P metadata mirror

Metadata-only Codex mirror for the public `zhen-nan/L2P` model.

This directory intentionally contains no model weights. The upstream Hugging Face repository publishes the large L2P SafeTensors payload but does not provide the denoiser `config.json` needed by the local GGUF converter preset scanner. Codex therefore keeps the minimal denoiser conversion metadata here and requires operators to select the real SafeTensors source path separately in Tools / GGUF Converter.

Files:

- `denoiser/config.json` — L2P pixel-space denoiser identity for the `zimage_l2p_denoiser` GGUF profile.

The Qwen3-4B text encoder used by L2P is listed separately under `apps/backend/huggingface/Qwen/Qwen3-4B/`.

Do not add model weights, fake SafeTensors indexes, GGUF files, ONNX files, or upstream `.gitattributes` files to this mirror.
