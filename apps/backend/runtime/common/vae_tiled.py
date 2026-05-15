"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared VAE tiled geometry typing/policy + tile-window iterator.
Defines strict typed geometry/window dataclasses, family-aware decode tiled-geometry defaults, and deterministic tile-window enumeration
with context padding for tiled VAE encode/decode paths.

Symbols (top-level; keep in sync; no ghosts):
- `VaeTileGeometry` (dataclass): Immutable/validated tiled geometry contract (`tile_x`, `tile_y`, `overlap`).
- `VaeTileWindow` (dataclass): Immutable tiled core/context bounds descriptor for stitched tiled passes.
- `DEFAULT_VAE_DECODE_TILED_GEOMETRY` (constant): Default decode tiled geometry used by non-Anima families.
- `ANIMA_VAE_DECODE_TILED_GEOMETRY` (constant): Anima-specific decode tiled geometry override.
- `resolve_vae_decode_tiled_geometry` (function): Family-aware decode tiled geometry resolver with fail-loud validation.
- `iter_vae_tile_windows` (function): Ordered tile-window iterator with strict geometry/padding validation.
- `__all__` (constant): Explicit export list for tiled helper symbols.
"""

from __future__ import annotations

from dataclasses import dataclass

from apps.backend.runtime.model_registry.specs import ModelFamily


@dataclass(frozen=True, slots=True)
class VaeTileGeometry:
    tile_x: int
    tile_y: int
    overlap: int

    def __post_init__(self) -> None:
        if not isinstance(self.tile_x, int) or not isinstance(self.tile_y, int) or not isinstance(self.overlap, int):
            raise RuntimeError(
                "Invalid VAE tiled geometry types: "
                f"tile_x={type(self.tile_x)!r} tile_y={type(self.tile_y)!r} overlap={type(self.overlap)!r}."
            )
        if self.tile_x <= 0 or self.tile_y <= 0:
            raise RuntimeError(
                f"Invalid VAE tiled geometry tile size: tile_x={self.tile_x} tile_y={self.tile_y}."
            )
        if self.overlap < 0:
            raise RuntimeError(f"Invalid VAE tiled geometry overlap: overlap={self.overlap}.")


@dataclass(frozen=True, slots=True)
class VaeTileWindow:
    core_y0: int
    core_y1: int
    core_x0: int
    core_x1: int
    context_y0: int
    context_y1: int
    context_x0: int
    context_x1: int


DEFAULT_VAE_DECODE_TILED_GEOMETRY = VaeTileGeometry(tile_x=64, tile_y=64, overlap=16)
ANIMA_VAE_DECODE_TILED_GEOMETRY = VaeTileGeometry(tile_x=48, tile_y=48, overlap=24)


def resolve_vae_decode_tiled_geometry(*, family: ModelFamily | None) -> VaeTileGeometry:
    if family is None:
        return DEFAULT_VAE_DECODE_TILED_GEOMETRY
    if not isinstance(family, ModelFamily):
        raise RuntimeError(f"Invalid VAE family type for tiled geometry resolver: {type(family)!r}.")
    if family is ModelFamily.ANIMA:
        return ANIMA_VAE_DECODE_TILED_GEOMETRY
    return DEFAULT_VAE_DECODE_TILED_GEOMETRY


def iter_vae_tile_windows(
    *,
    height: int,
    width: int,
    tile_y: int,
    tile_x: int,
    pad_y: int,
    pad_x: int,
):
    if height <= 0 or width <= 0:
        raise RuntimeError(f"Invalid tiled VAE geometry: height={height} width={width}.")
    if tile_y <= 0 or tile_x <= 0:
        raise RuntimeError(f"Invalid tiled VAE tile size: tile_y={tile_y} tile_x={tile_x}.")
    if pad_y < 0 or pad_x < 0:
        raise RuntimeError(f"Invalid tiled VAE padding: pad_y={pad_y} pad_x={pad_x}.")

    for core_y0 in range(0, height, tile_y):
        core_y1 = min(height, core_y0 + tile_y)
        context_y0 = max(0, core_y0 - pad_y)
        context_y1 = min(height, core_y1 + pad_y)
        for core_x0 in range(0, width, tile_x):
            core_x1 = min(width, core_x0 + tile_x)
            context_x0 = max(0, core_x0 - pad_x)
            context_x1 = min(width, core_x1 + pad_x)
            yield VaeTileWindow(
                core_y0=core_y0,
                core_y1=core_y1,
                core_x0=core_x0,
                core_x1=core_x1,
                context_y0=context_y0,
                context_y1=context_y1,
                context_x0=context_x0,
                context_x1=context_x1,
            )


__all__ = [
    "ANIMA_VAE_DECODE_TILED_GEOMETRY",
    "DEFAULT_VAE_DECODE_TILED_GEOMETRY",
    "VaeTileGeometry",
    "VaeTileWindow",
    "iter_vae_tile_windows",
    "resolve_vae_decode_tiled_geometry",
]
