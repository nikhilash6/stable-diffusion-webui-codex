"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Router package for the FastAPI API surface.
Keeps endpoint groups in focused modules and lets run_api assemble the app via router factories.
Includes the standalone upscaling surface (`upscale` router).
Includes the SUPIR diagnostics surface (`supir` router).
Includes bounded diagnostics routes (`tests` router).

Symbols (top-level; keep in sync; no ghosts):
- `__all__` (constant): Export list for router modules.
"""

__all__ = [
    "generation",
    "models",
    "options",
    "paths",
    "settings",
    "supir",
    "system",
    "tasks",
    "tests",
    "tools",
    "upscale",
    "ui",
]
