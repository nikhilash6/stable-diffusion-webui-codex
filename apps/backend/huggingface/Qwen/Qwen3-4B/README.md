# Qwen/Qwen3-4B metadata mirror

Metadata-only Codex mirror for the Qwen3-4B text encoder used by Z-Image and Z-Image L2P.

This directory intentionally contains no model weights. Tools / GGUF Converter uses `text_encoder/config.json` only as the exact Qwen3-4B identity for the `zimage_l2p_tenc` profile; operators select the real SafeTensors file, SafeTensors index, folder, or generated GGUF separately.

Files:

- `text_encoder/config.json` — exact Qwen3-4B text-encoder identity for the `zimage_l2p_tenc` GGUF profile.

Do not add model weights, fake SafeTensors indexes, GGUF files, ONNX files, or upstream `.gitattributes` files to this mirror.
