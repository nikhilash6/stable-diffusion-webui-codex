<!-- tags: backend, gguf, io -->
# apps/backend/quantization/gguf Overview
Status: Active

## Purpose
- Base GGUF schema and IO helpers used by converters and runtime loaders.

## Key files
- `constants.py` — GGUF constants, enums, and metadata keys.
- `reader.py` — memmap-based `GGUFReader`.
- `writer.py` — GGUF writer implementation.
- `quant_shapes.py` — logical-shape ↔ byte-shape helpers for quantized tensors.

## Expectations
- Keep this package dependency-light: NumPy + stdlib only.
- Preserve reader/writer behavior for base `.gguf` files.
- Keep quantization math in `apps/backend/quantization/kernels/*`, not here.
