<!-- tags: backend, runtime, zimage-l2p, pixel-dit -->

# apps/backend/runtime/families/zimage_l2p Overview
Date: 2026-05-23
Last Review: 2026-05-24
Status: Active

## Purpose
- Own the public `zhen-nan/L2P` pixel-space Z-Image derivative runtime.
- Keep L2P separate from `apps/backend/runtime/families/zimage/`, which remains the latent Z-Image Turbo/Base runtime owner.

## Structure
- `l2p_model.py` — native L2P DiT/local-decoder model and strict SafeTensors/GGUF state-dict loader.
- `__init__.py` — package exports for the L2P runtime owner.

## Notes
- L2P is a no-VAE RGB pixel-space model; do not add dummy VAE paths or latent Z-Image aliases.
- Native checkpoint keys stay native (`all_x_embedder.16-1`, `local_decoder`, `noise_refiner`, `context_refiner`, `layers`).
- GGUF support must preserve the same native lookup keyspace; converter/keymap fixes belong at the GGUF/converter seam, not in runtime prefix strippers.
- 2026-05-23: L2P timestep embeddings must use a floating activation dtype owned by the caller/model compute path. Packed GGUF storage dtypes such as `torch.int8` must never drive timestep MLP activations; direct embedder calls may use floating `weight.computation_dtype` or floating weight dtype, and otherwise fail loud.
- 2026-05-24: L2P attention remains explicitly forced to PyTorch SDPA in `l2p_model.py`; the runtime logs the first actual attention call with backend, effective SDPA policy, Q/K/V shapes, dtype, device, heads, and head dim so operator logs show the attention path used during sampling.
