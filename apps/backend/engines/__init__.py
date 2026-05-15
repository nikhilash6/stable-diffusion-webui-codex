"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Engine package facade and default registration entry points.
Exposes `register_default_engines(...)` and lazily resolves optional/large engine classes to avoid heavy imports during startup.
Includes the Anima, FLUX.2, and LTX2 engine facades; parked placeholder families such as `netflix_void`
remain importable by name but are not part of default runtime registration.

Symbols (top-level; keep in sync; no ghosts):
- `EngineLoadError` (class): Error raised when an engine fails to load required resources.
- `EngineExecutionError` (class): Error raised when an engine fails during inference execution.
- `register_default_engines` (function): Registers the canonical engine set into the registry.
- `_ENGINE_EXPORTS` (constant): Lazy export map `{name: (module_path, attr)}` for engine class exports.
- `__getattr__` (function): Lazy import hook for engine class exports.
- `__all__` (constant): Explicit export list for the engine facade.
"""

from __future__ import annotations

# tags: backend, engines, lazy-imports

from importlib import import_module
from typing import TYPE_CHECKING

from apps.backend.core.exceptions import EngineExecutionError, EngineLoadError
from apps.backend.core.registry import EngineRegistry

if TYPE_CHECKING:
    # Keep the surface import-light for runtime. For type-checkers, expose names without importing engine modules,
    # which may rely on optional deps and/or heavy import graphs.
    from typing import Any as Chroma
    from typing import Any as Flux
    from typing import Any as Flux2Engine
    from typing import Any as Ltx2Engine
    from typing import Any as NetflixVoidEngine
    from typing import Any as Kontext
    from typing import Any as StableDiffusion
    from typing import Any as StableDiffusion2
    from typing import Any as StableDiffusion3
    from typing import Any as StableDiffusionXL
    from typing import Any as StableDiffusionXLRefiner
    from typing import Any as Wan2214BEngine
    from typing import Any as Wan225BEngine
    from typing import Any as Wan22Animate14BEngine
    from typing import Any as ZImageEngine
    from typing import Any as AnimaEngine


def register_default_engines(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    """Register the canonical set of engines into the provided registry."""

    registration = import_module("apps.backend.engines.registration")

    from apps.backend.core.registry import registry as _global_registry

    target = registry or _global_registry

    def _maybe_register(key: str, fn) -> None:  # type: ignore[no-untyped-def]
        if replace:
            fn(registry=target, replace=True)
            return
        try:
            target.get_descriptor(key)
            return
        except Exception:
            fn(registry=target, replace=False)

    _maybe_register("sd15", registration.register_sd15)
    _maybe_register("sdxl", registration.register_sdxl)
    _maybe_register("flux1", registration.register_flux)
    _maybe_register("flux2", registration.register_flux2)
    _maybe_register("ltx2", registration.register_ltx2)
    _maybe_register("flux1_kontext", registration.register_kontext)
    _maybe_register("flux1_chroma", registration.register_chroma)
    _maybe_register("sd20", registration.register_sd20)
    _maybe_register("zimage", registration.register_zimage)
    _maybe_register("anima", registration.register_anima)
    # WAN22 GGUF lanes are explicit and variant-specific.
    _maybe_register("wan22_5b", registration.register_wan22_5b)
    _maybe_register("wan22_14b", registration.register_wan22_14b)
    _maybe_register("wan22_14b_animate", registration.register_wan22_14b_animate)


__all__ = [
    "EngineLoadError",
    "EngineExecutionError",
    "register_default_engines",
    "StableDiffusion",
    "StableDiffusion2",
    "StableDiffusion3",
    "StableDiffusionXL",
    "StableDiffusionXLRefiner",
    "Flux",
    "Flux2Engine",
    "Ltx2Engine",
    "NetflixVoidEngine",
    "Kontext",
    "Chroma",
    "ZImageEngine",
    "AnimaEngine",
    "Wan22Animate14BEngine",
    "Wan2214BEngine",
    "Wan225BEngine",
]

_ENGINE_EXPORTS = {
    "StableDiffusion": ("apps.backend.engines.sd.sd15", "StableDiffusion"),
    "StableDiffusion2": ("apps.backend.engines.sd.sd20", "StableDiffusion2"),
    "StableDiffusion3": ("apps.backend.engines.sd.sd35", "StableDiffusion3"),
    "StableDiffusionXL": ("apps.backend.engines.sd.sdxl", "StableDiffusionXL"),
    "StableDiffusionXLRefiner": ("apps.backend.engines.sd.sdxl", "StableDiffusionXLRefiner"),
    "Flux": ("apps.backend.engines.flux.flux", "Flux"),
    "Flux2Engine": ("apps.backend.engines.flux2.flux2", "Flux2Engine"),
    "Ltx2Engine": ("apps.backend.engines.ltx2.ltx2", "Ltx2Engine"),
    "NetflixVoidEngine": ("apps.backend.engines.netflix_void.netflix_void", "NetflixVoidEngine"),
    "Kontext": ("apps.backend.engines.flux.kontext", "Kontext"),
    "Chroma": ("apps.backend.engines.flux.chroma", "Chroma"),
    "ZImageEngine": ("apps.backend.engines.zimage.zimage", "ZImageEngine"),
    "AnimaEngine": ("apps.backend.engines.anima.anima", "AnimaEngine"),
    "Wan22Animate14BEngine": ("apps.backend.engines.wan22.wan22_14b_animate", "Wan22Animate14BEngine"),
    "Wan2214BEngine": ("apps.backend.engines.wan22.wan22_14b", "Wan2214BEngine"),
    "Wan225BEngine": ("apps.backend.engines.wan22.wan22_5b", "Wan225BEngine"),
}

def __getattr__(name: str):  # pragma: no cover - runtime dispatch
    if name in _ENGINE_EXPORTS:
        module_name, attr = _ENGINE_EXPORTS[name]
        mod = import_module(module_name)
        value = getattr(mod, attr)
        globals()[name] = value
        return value

    raise AttributeError(name)
