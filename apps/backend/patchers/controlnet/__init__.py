"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public facade for Codex-native ControlNet patcher APIs.
Re-exports the advanced attach helper and core SD-family ControlNet module implementations used by the backend.

Symbols (top-level; keep in sync; no ghosts):
- `apply_controlnet_advanced` (function): Clones a UNet patcher and appends an advanced ControlNet node.
- `ControlNet` (class): Stable Diffusion ControlNet module implementation.
- `ControlNetLite` (class): Placeholder class for ControlNet-Lite variants (not yet ported).
- `ControlLiteConfig` (dataclass): Placeholder config for ControlNet-Lite variants.
- `ControlLora` (class): ControlNet LoRA module that materialises a ControlNet on demand.
- `T2IAdapter` (class): Adapter-based control module for T2I-Adapter weights.
- `load_t2i_adapter` (function): Loads a T2I-Adapter state dict into a runnable module.
- `__all__` (constant): Explicit re-export list for the package facade.
"""

from .apply import apply_controlnet_advanced
from .architectures.sd.control import ControlNet
from .architectures.sd.control_lite import ControlNetLite, ControlLiteConfig
from .architectures.sd.lora import ControlLora
from .architectures.sd.t2i_adapter import T2IAdapter, load_t2i_adapter

__all__ = [
    "apply_controlnet_advanced",
    "ControlNet",
    "ControlLora",
    "ControlNetLite",
    "ControlLiteConfig",
    "T2IAdapter",
    "load_t2i_adapter",
]
