"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher service environment normalization helpers.
Owns boolean env parsing plus strict allocator env contract checks and cudaMallocAsync allocator mutation before launcher-spawned services run.

Symbols (top-level; keep in sync; no ghosts):
- `env_truthy` (function): Parses common truthy env tokens.
- `sanitize_allocator_env_contract` (function): Rejects unsupported allocator env keys before subprocess spawn.
- `parse_pytorch_cuda_alloc_conf` (function): Parses `PYTORCH_CUDA_ALLOC_CONF` entries.
- `ensure_cuda_malloc_async_allocator_env` (function): Ensures allocator backend is `cudaMallocAsync` when requested.
"""

from __future__ import annotations

from typing import MutableMapping

from apps.launcher.profile_meta import ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY


def env_truthy(value: object) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def sanitize_allocator_env_contract(env: MutableMapping[str, str], *, scope_label: str) -> None:
    supported_alloc_key = "PYTORCH_CUDA_ALLOC_CONF"
    supported_toggle_key = ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY
    unsupported_keys: list[str] = []
    for key in list(env.keys()):
        if key.startswith("PYTORCH_") and key.endswith("_ALLOC_CONF") and key != supported_alloc_key:
            unsupported_keys.append(key)
            continue
        if (
            key.startswith("CODEX_ENABLE_DEFAULT_PYTORCH_")
            and key.endswith("_ALLOC_CONF")
            and key != supported_toggle_key
        ):
            unsupported_keys.append(key)
    if unsupported_keys:
        keys = ", ".join(sorted(unsupported_keys))
        raise ValueError(
            f"Unsupported allocator env key(s) for {scope_label}: {keys}. "
            "Supported keys: PYTORCH_CUDA_ALLOC_CONF and "
            f"{supported_toggle_key}."
        )


def parse_pytorch_cuda_alloc_conf(raw_conf: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for raw_entry in str(raw_conf or "").split(","):
        token = raw_entry.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                "Invalid PYTORCH_CUDA_ALLOC_CONF entry "
                f"{token!r}: expected 'key:value' format."
            )
        key, value = token.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                "Invalid PYTORCH_CUDA_ALLOC_CONF entry "
                f"{token!r}: expected non-empty 'key:value' parts."
            )
        entries.append((key, value))
    return entries


def ensure_cuda_malloc_async_allocator_env(env: MutableMapping[str, str]) -> None:
    target_backend = "cudaMallocAsync"
    target_backend_norm = target_backend.lower()
    raw_alloc_conf = str(env.get("PYTORCH_CUDA_ALLOC_CONF", "") or "").strip()
    if not raw_alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = f"backend:{target_backend}"
        return

    entries = parse_pytorch_cuda_alloc_conf(raw_alloc_conf)
    backend_index: int | None = None
    for index, (key, _value) in enumerate(entries):
        if key.strip().lower() == "backend":
            if backend_index is not None:
                raise ValueError(
                    "Invalid PYTORCH_CUDA_ALLOC_CONF: multiple 'backend' entries found. "
                    "Use exactly one backend directive."
                )
            backend_index = index

    if backend_index is None:
        entries.append(("backend", target_backend))
        env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(f"{key}:{value}" for key, value in entries)
        return

    configured_backend = entries[backend_index][1]
    if configured_backend.replace(" ", "").lower() != target_backend_norm:
        raise ValueError(
            "CODEX_CUDA_MALLOC=1 requires PYTORCH_CUDA_ALLOC_CONF backend:cudaMallocAsync, "
            f"but found backend:{configured_backend}. "
            "Set PYTORCH_CUDA_ALLOC_CONF with backend:cudaMallocAsync or disable CODEX_CUDA_MALLOC."
        )
    env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(f"{key}:{value}" for key, value in entries)
