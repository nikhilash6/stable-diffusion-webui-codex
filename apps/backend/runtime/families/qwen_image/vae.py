"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen Image VAE metadata validation and latent normalization helpers.
Owns the lightweight `AutoencoderKLQwenImage` config contract and per-channel latent mean/std math used by Qwen Image
encode/decode seams, without building or loading the heavyweight VAE module.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageVaeConfig` (dataclass): Strict metadata contract for `AutoencoderKLQwenImage`.
- `qwen_image_denormalize_latents` (function): Apply inverse per-channel Qwen Image VAE latent normalization.
- `qwen_image_normalize_latents` (function): Apply per-channel Qwen Image VAE latent normalization.
- `qwen_image_validate_external_vae_path` (function): Validate a selected external Qwen Image VAE asset path/root/config.
- `qwen_image_vae_config_from_mapping` (function): Validate and convert a VAE config mapping.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import torch

from .config import QWEN_IMAGE_LATENT_CHANNELS


@dataclass(frozen=True, slots=True)
class QwenImageVaeConfig:
    class_name: str
    z_dim: int
    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.class_name != "AutoencoderKLQwenImage":
            raise ValueError("Qwen Image VAE class must be AutoencoderKLQwenImage")
        if self.z_dim != QWEN_IMAGE_LATENT_CHANNELS:
            raise ValueError(f"Qwen Image VAE z_dim must be {QWEN_IMAGE_LATENT_CHANNELS}")
        if len(self.latents_mean) != QWEN_IMAGE_LATENT_CHANNELS:
            raise ValueError(f"Qwen Image VAE latents_mean must have {QWEN_IMAGE_LATENT_CHANNELS} values")
        if len(self.latents_std) != QWEN_IMAGE_LATENT_CHANNELS:
            raise ValueError(f"Qwen Image VAE latents_std must have {QWEN_IMAGE_LATENT_CHANNELS} values")
        for index, value in enumerate(self.latents_mean):
            if not math.isfinite(float(value)):
                raise ValueError(f"Qwen Image VAE latents_mean[{index}] must be finite")
        for index, value in enumerate(self.latents_std):
            if not math.isfinite(float(value)) or float(value) == 0.0:
                raise ValueError(f"Qwen Image VAE latents_std[{index}] must be finite and non-zero")


def _float_tuple(values: object, *, field: str, context: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise RuntimeError(f"{context}: {field} must be a sequence of {QWEN_IMAGE_LATENT_CHANNELS} numbers.")
    result: list[float] = []
    for index, value in enumerate(values):
        try:
            result.append(float(value))
        except Exception as exc:  # noqa: BLE001 - strict metadata validation
            raise RuntimeError(f"{context}: {field}[{index}] must be numeric; got {value!r}.") from exc
    return tuple(result)


def qwen_image_vae_config_from_mapping(
    config: Mapping[str, object],
    *,
    context: str = "Qwen Image VAE metadata",
) -> QwenImageVaeConfig:
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{context}: VAE config must be a mapping.")
    try:
        vae_config = QwenImageVaeConfig(
            class_name=str(config.get("_class_name") or "").strip(),
            z_dim=int(config.get("z_dim") or 0),
            latents_mean=_float_tuple(config.get("latents_mean"), field="latents_mean", context=context),
            latents_std=_float_tuple(config.get("latents_std"), field="latents_std", context=context),
        )
    except ValueError as exc:
        raise RuntimeError(f"{context}: {exc}") from exc
    return vae_config


def _read_vae_config(path: Path, *, context: str) -> Mapping[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"{context}: Qwen Image VAE config not found: {path}") from exc
    except Exception as exc:  # noqa: BLE001 - strict metadata validation
        raise RuntimeError(f"{context}: invalid Qwen Image VAE config JSON at {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise RuntimeError(f"{context}: Qwen Image VAE config must be a JSON object: {path}")
    return data


def _config_path_for_vae_asset(path: Path) -> Path:
    if path.is_dir():
        return path / "config.json"
    return path.parent / "config.json"


def _is_under_allowed_root(path: Path, roots: Sequence[object]) -> bool:
    resolved_path = path.resolve()
    for raw_root in roots:
        if not isinstance(raw_root, str) or not raw_root.strip():
            continue
        root = Path(raw_root.strip()).expanduser().resolve()
        if root.is_file():
            if resolved_path == root:
                return True
            continue
        try:
            resolved_path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def qwen_image_validate_external_vae_path(
    raw_path: object,
    *,
    allowed_roots: Sequence[object] = (),
    context: str = "Qwen Image external VAE",
) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError(f"{context}: path must be a non-empty string.")
    path = Path(raw_path.strip()).expanduser()
    if not path.exists():
        raise RuntimeError(f"{context}: path not found: {path}")
    if allowed_roots and not _is_under_allowed_root(path, allowed_roots):
        roots_text = ", ".join(str(root) for root in allowed_roots if isinstance(root, str) and root.strip())
        raise RuntimeError(
            f"{context}: path must be under qwen_image_vae roots; got {path}. Roots: {roots_text or '<none>'}."
        )

    config_path = _config_path_for_vae_asset(path)
    qwen_image_vae_config_from_mapping(_read_vae_config(config_path, context=context), context=str(config_path))
    return str(path)


def _latent_stat_tensor(latents: torch.Tensor, values: tuple[float, ...], *, field: str) -> torch.Tensor:
    import torch

    if not isinstance(latents, torch.Tensor):
        raise TypeError("latents must be a torch.Tensor")
    if latents.ndim < 3:
        raise RuntimeError(f"Qwen Image VAE latents must have at least 3 dimensions; got shape={tuple(latents.shape)}.")
    channels = int(latents.shape[1])
    if channels != QWEN_IMAGE_LATENT_CHANNELS:
        raise RuntimeError(
            f"Qwen Image VAE latent channel mismatch: expected {QWEN_IMAGE_LATENT_CHANNELS}, got {channels}."
        )
    stat = torch.tensor(values, device=latents.device, dtype=latents.dtype)
    shape = (1, QWEN_IMAGE_LATENT_CHANNELS, *([1] * (latents.ndim - 2)))
    if stat.numel() != QWEN_IMAGE_LATENT_CHANNELS:
        raise RuntimeError(f"Qwen Image VAE {field} must have {QWEN_IMAGE_LATENT_CHANNELS} values.")
    return stat.reshape(shape)


def qwen_image_normalize_latents(latents: torch.Tensor, config: QwenImageVaeConfig) -> torch.Tensor:
    mean = _latent_stat_tensor(latents, config.latents_mean, field="latents_mean")
    std = _latent_stat_tensor(latents, config.latents_std, field="latents_std")
    return (latents - mean) / std


def qwen_image_denormalize_latents(latents: torch.Tensor, config: QwenImageVaeConfig) -> torch.Tensor:
    mean = _latent_stat_tensor(latents, config.latents_mean, field="latents_mean")
    std = _latent_stat_tensor(latents, config.latents_std, field="latents_std")
    return latents * std + mean


__all__ = [
    "QwenImageVaeConfig",
    "qwen_image_denormalize_latents",
    "qwen_image_normalize_latents",
    "qwen_image_validate_external_vae_path",
    "qwen_image_vae_config_from_mapping",
]
