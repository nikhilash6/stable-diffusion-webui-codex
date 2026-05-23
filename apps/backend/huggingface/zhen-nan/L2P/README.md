# zhen-nan/L2P metadata mirror

Metadata-only Codex mirror for the public `zhen-nan/L2P` model.

This directory intentionally contains no model weights. The upstream Hugging Face repository publishes the large L2P SafeTensors payload but does not provide the component `config.json` tree needed by the local GGUF converter preset scanner. Codex therefore keeps the minimal conversion metadata here and requires operators to select the real SafeTensors source path separately in Tools / GGUF Converter.

Files:

- `denoiser/config.json` — L2P pixel-space denoiser identity for the `zimage_l2p_denoiser` GGUF profile.
- `text_encoder/config.json` — exact Qwen3-4B text-encoder identity for the `zimage_l2p_tenc` GGUF profile.

Do not add model weights, fake SafeTensors indexes, GGUF files, ONNX files, or upstream `.gitattributes` files to this mirror.
