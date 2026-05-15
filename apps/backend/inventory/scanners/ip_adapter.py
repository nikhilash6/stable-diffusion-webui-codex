"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: IP-Adapter asset discovery policy used by backend inventories.
Defines adapter-model and image-encoder locations from explicit `apps/paths.json` keys (`ip_adapter_models`,
`ip_adapter_image_encoders`) and yields weight-file paths in stable order while failing loud on unreadable configured roots or
unsupported configured file entries.

Symbols (top-level; keep in sync; no ghosts):
- `IP_ADAPTER_MODEL_EXTS` (constant): Recognized IP-Adapter adapter-weight file extensions.
- `IP_ADAPTER_IMAGE_ENCODER_EXTS` (constant): Recognized IP-Adapter image-encoder weight file extensions.
- `_files_in_dir_recursive_strict` (function): Recursively lists supported asset files in a directory and raises on unreadable paths.
- `_list_configured_roots` (function): Resolves one configured IP-Adapter root list and rejects unsupported existing file entries.
- `_iter_ip_adapter_files` (function): Yields supported IP-Adapter asset files from configured roots (recursive for directories).
- `list_ip_adapter_model_roots` (function): Resolves configured IP-Adapter adapter-model roots.
- `list_ip_adapter_image_encoder_roots` (function): Resolves configured IP-Adapter image-encoder roots.
- `iter_ip_adapter_model_files` (function): Yields adapter-weight files under `ip_adapter_models`.
- `iter_ip_adapter_image_encoder_files` (function): Yields image-encoder weight files under `ip_adapter_image_encoders`.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

from apps.backend.infra.config.paths import get_paths_for

from .base import dedupe_keep_order

IP_ADAPTER_MODEL_EXTS: tuple[str, ...] = (".safetensors", ".bin", ".pt", ".pth")
IP_ADAPTER_IMAGE_ENCODER_EXTS: tuple[str, ...] = (".safetensors", ".bin", ".pt", ".pth")


def _files_in_dir_recursive_strict(dir_path: str, *, exts: Sequence[str], label: str) -> list[str]:
    if not dir_path or not os.path.isdir(dir_path):
        return []

    out: list[str] = []
    exts_lc = tuple(str(ext).lower() for ext in exts)

    def _raise_oserror(exc: OSError) -> None:
        raise exc

    try:
        for dirpath, dirnames, filenames in os.walk(dir_path, onerror=_raise_oserror):
            dirnames.sort(key=lambda s: s.lower())
            for name in sorted(filenames, key=lambda s: s.lower()):
                if name.lower().endswith(exts_lc):
                    out.append(os.path.join(dirpath, name))
    except OSError as exc:
        raise RuntimeError(f"Failed to read {label} inventory root {dir_path!r}: {exc}") from exc

    return out


def _list_configured_roots(*, key: str, exts: Sequence[str], label: str) -> list[str]:
    roots: list[str] = []
    exts_lc = tuple(str(ext).lower() for ext in exts)
    allowed_exts = ", ".join(exts_lc)

    for path in get_paths_for(key):
        if not path:
            continue
        if os.path.isdir(path):
            roots.append(path)
            continue
        if os.path.isfile(path):
            if not path.lower().endswith(exts_lc):
                raise RuntimeError(
                    f"Unsupported {label} file configured via {key!r}: {path!r}. "
                    f"Expected one of: {allowed_exts}."
                )
            roots.append(path)
            continue
        if os.path.exists(path):
            raise RuntimeError(
                f"Unsupported {label} inventory entry configured via {key!r}: {path!r}. "
                f"Expected a directory or one of: {allowed_exts}."
            )

    return dedupe_keep_order(roots)


def _iter_ip_adapter_files(
    *,
    key: str,
    exts: Sequence[str],
    label: str,
    roots: Sequence[str] | None = None,
) -> Iterable[str]:
    use_roots = list(roots) if roots is not None else _list_configured_roots(key=key, exts=exts, label=label)
    out: list[str] = []
    exts_lc = tuple(str(ext).lower() for ext in exts)
    allowed_exts = ", ".join(exts_lc)

    for root in use_roots:
        if not root:
            continue
        if os.path.isfile(root):
            if not root.lower().endswith(exts_lc):
                raise RuntimeError(
                    f"Unsupported {label} file configured via {key!r}: {root!r}. "
                    f"Expected one of: {allowed_exts}."
                )
            out.append(root)
            continue
        if os.path.isdir(root):
            out.extend(_files_in_dir_recursive_strict(root, exts=exts_lc, label=label))
            continue
        if os.path.exists(root):
            raise RuntimeError(
                f"Unsupported {label} inventory entry configured via {key!r}: {root!r}. "
                f"Expected a directory or one of: {allowed_exts}."
            )

    return dedupe_keep_order(out)


def list_ip_adapter_model_roots(models_root: str | None = None) -> list[str]:
    return _list_configured_roots(key="ip_adapter_models", exts=IP_ADAPTER_MODEL_EXTS, label="IP-Adapter model")


def list_ip_adapter_image_encoder_roots(models_root: str | None = None) -> list[str]:
    return _list_configured_roots(
        key="ip_adapter_image_encoders",
        exts=IP_ADAPTER_IMAGE_ENCODER_EXTS,
        label="IP-Adapter image encoder",
    )


def iter_ip_adapter_model_files(models_root: str | None = None, *, roots: Sequence[str] | None = None) -> Iterable[str]:
    return _iter_ip_adapter_files(
        key="ip_adapter_models",
        exts=IP_ADAPTER_MODEL_EXTS,
        label="IP-Adapter model",
        roots=roots,
    )


def iter_ip_adapter_image_encoder_files(
    models_root: str | None = None,
    *,
    roots: Sequence[str] | None = None,
) -> Iterable[str]:
    return _iter_ip_adapter_files(
        key="ip_adapter_image_encoders",
        exts=IP_ADAPTER_IMAGE_ENCODER_EXTS,
        label="IP-Adapter image encoder",
        roots=roots,
    )


__all__ = [
    "IP_ADAPTER_IMAGE_ENCODER_EXTS",
    "IP_ADAPTER_MODEL_EXTS",
    "iter_ip_adapter_image_encoder_files",
    "iter_ip_adapter_model_files",
    "list_ip_adapter_image_encoder_roots",
    "list_ip_adapter_model_roots",
]
