"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: IP-Adapter runtime package exports.
Provides the shared typed contracts and the request-scoped sampling apply context used by the canonical image pipelines.

Symbols (top-level; keep in sync; no ghosts):
- `apply_ip_adapter_for_sampling` (function): Context manager that patches the active sampling denoiser for one sampling pass and restores it afterwards.
- `IpAdapterConfig` (dataclass): Typed processing/runtime owner for one IP-Adapter application.
- `IpAdapterSourceConfig` (dataclass): Typed nested source owner for IP-Adapter reference images.
"""

from .session import apply_ip_adapter_for_sampling
from .types import IpAdapterConfig, IpAdapterSourceConfig

__all__ = ["IpAdapterConfig", "IpAdapterSourceConfig", "apply_ip_adapter_for_sampling"]
