"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Removed ControlNet model facade (no compat shims).
This package existed as a legacy-facing import surface. It is intentionally not supported in Codex: import the architecture modules from
their concrete family packages such as `apps.backend.patchers.controlnet.architectures.sd`, or use the public facade at
`apps.backend.patchers.controlnet`.

Symbols (top-level; keep in sync; no ghosts):
- `__all__` (constant): Empty export list (module import is intentionally rejected).
"""

__all__: list[str] = []

raise ImportError(
    "apps.backend.patchers.controlnet.models has been removed.\n"
    "Use apps.backend.patchers.controlnet (public facade) or concrete architecture family packages such as "
    "apps.backend.patchers.controlnet.architectures.sd."
)
