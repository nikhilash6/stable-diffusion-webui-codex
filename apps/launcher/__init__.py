"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Package marker for launcher infrastructure.
Concrete launcher owners live in focused modules such as `profile_store.py`, `profile_meta.py`, `service_process.py`, `service_specs.py`, `checks.py`, and `paths.py`.

Symbols (top-level; keep in sync; no ghosts):
- `__all__` (constant): Empty explicit export list; import concrete launcher owners from their modules.
"""

from __future__ import annotations

__all__: list[str] = []
