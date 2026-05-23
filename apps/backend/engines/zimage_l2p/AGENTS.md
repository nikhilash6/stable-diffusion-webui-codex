<!-- tags: backend, engines, zimage-l2p -->

# apps/backend/engines/zimage_l2p Overview
Date: 2026-05-23
Last Review: 2026-05-23
Status: Active

## Purpose
- Own the `zimage_l2p` engine facade for public `zhen-nan/L2P` txt2img.
- Keep the engine as an adapter/hook provider for the canonical txt2img use-case; do not create a parallel txt2img pipeline.

## Structure
- `zimage_l2p.py` — `ZImageL2PEngine` facade, conditioning, and pixel sampling hook.
- `spec.py` — runtime assembly for L2P core + external Qwen3-4B text encoder.
- `factory.py` — `CodexObjects` construction for the common engine lifecycle.
- `standalone_sampler.py` — minimal pixel-space FlowMatch Euler sampler used by the canonical txt2img hook.
- `__init__.py` — package exports.

## Notes
- Engine id is exactly `zimage_l2p`; no aliases.
- First tranche is exact 1024x1024 txt2img, batch size/count 1, no hires, no img2img, no VAE, no LoRA.
- SafeTensors and GGUF are both first-class for the denoiser core and Qwen3-4B TEnc.
