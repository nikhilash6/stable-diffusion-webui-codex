"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Deterministic noise generation helpers for image pipelines.
Implements image latent noise generation with support for GPU/CPU/Philox sources, subseeds (slerp), and resize-from seeding policies.

Symbols (top-level; keep in sync; no ghosts):
- `NoiseSourceKind` (enum): Selects the noise source backend (GPU/CPU/Philox).
- `NoiseSettings` (dataclass): Noise source configuration (source + eta seed delta + optional forced device).
- `_slerp` (function): Spherical interpolation between two noise tensors (used for subseeds).
- `_resolve_generator_device` (function): Chooses the device used by the underlying generator given settings/target device.
- `_seed_to_tensor` (function): Generates a deterministic normal tensor for a given seed and shape.
- `ImageRNG` (dataclass): Deterministic noise batch generator honoring seed/subseed/resize policies.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import enum
import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import torch

from .devices import cpu, default_device
from .philox import PhiloxGenerator

_LOGGER = get_backend_logger(f"{__name__}.rng")


class NoiseSourceKind(str, enum.Enum):
    GPU = "gpu"
    CPU = "cpu"
    PHILOX = "philox"

    @staticmethod
    def from_string(value: str | None) -> "NoiseSourceKind":
        if value is None:
            return NoiseSourceKind.GPU
        key = value.strip().lower()
        for member in NoiseSourceKind:
            if key in {member.value, member.name.lower()}:
                return member
        raise ValueError(f"Unknown noise source '{value}'")


@dataclass
class NoiseSettings:
    source: NoiseSourceKind = NoiseSourceKind.GPU
    eta_noise_seed_delta: int = 0
    force_device: torch.device | None = None


def _slerp(amount: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    low_norm = torch.nn.functional.normalize(low, dim=1)
    high_norm = torch.nn.functional.normalize(high, dim=1)
    dot = (low_norm * high_norm).sum(1)

    if torch.mean(dot) > 0.9995:
        return low.lerp(high, amount)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (
        torch.sin((1.0 - amount) * omega) / so
    ).unsqueeze(1) * low + (
        torch.sin(amount * omega) / so
    ).unsqueeze(1) * high
    return res


def _resolve_generator_device(settings: NoiseSettings, target_device: torch.device) -> torch.device:
    if settings.force_device is not None:
        return settings.force_device
    if settings.source is NoiseSourceKind.CPU:
        return cpu()
    return target_device


def _seed_to_tensor(shape: Iterable[int], seed: int, *, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return torch.randn(tuple(shape), generator=gen, device=device)


@dataclass
class ImageRNG:
    """Generate deterministic latent noise batches respecting seed policies."""

    shape: Tuple[int, int, int]
    seeds: Sequence[int]
    subseeds: Sequence[int]
    subseed_strength: float
    seed_resize_from_h: int = 0
    seed_resize_from_w: int = 0
    settings: NoiseSettings = field(default_factory=NoiseSettings)
    device: torch.device = field(default_factory=default_device)

    def __post_init__(self) -> None:
        self.shape = tuple(int(dim) for dim in self.shape)
        self._target = self.device
        self._generator_device = _resolve_generator_device(self.settings, self._target)
        self._generators: List[PhiloxGenerator | torch.Generator] = [
            self._create_generator(seed) for seed in self.seeds
        ]
        self._is_first = True

    def _create_generator(self, seed: int) -> PhiloxGenerator | torch.Generator:
        if self.settings.source is NoiseSourceKind.PHILOX:
            return PhiloxGenerator(int(seed))
        generator = torch.Generator(device=self._generator_device)
        generator.manual_seed(int(seed))
        return generator

    def _noise_from_generator(
        self,
        generator: PhiloxGenerator | torch.Generator,
        shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        if isinstance(generator, PhiloxGenerator):
            return generator.randn(shape, device=self._target)
        noise = torch.randn(shape, generator=generator, device=self._generator_device)
        if noise.device == self._target:
            return noise
        return noise.to(self._target)

    def _initial_noise(self) -> torch.Tensor:
        seeds = list(self.seeds) or [0]
        noise_shape = self.shape
        resize = (
            self.seed_resize_from_h > 0 and self.seed_resize_from_w > 0
        )
        if resize:
            noise_shape = (
                self.shape[0],
                int(self.seed_resize_from_h) // 8,
                int(self.seed_resize_from_w) // 8,
            )

        batch = []
        for index, seed in enumerate(seeds):
            generator = self._generators[index]
            primary = self._noise_from_generator(generator, noise_shape)

            if self.subseeds and self.subseed_strength != 0.0:
                subseed = self.subseeds[index] if index < len(self.subseeds) else 0
                secondary = _seed_to_tensor(noise_shape, subseed, device=self._generator_device)
                if secondary.device != self._target:
                    secondary = secondary.to(self._target)
                primary = _slerp(float(self.subseed_strength), primary.unsqueeze(0), secondary.unsqueeze(0))[0]

            if resize and noise_shape != self.shape:
                full = self._noise_from_generator(generator, self.shape)
                dx = (self.shape[2] - noise_shape[2]) // 2
                dy = (self.shape[1] - noise_shape[1]) // 2
                tmp = full.clone()
                tmp[:, dy : dy + noise_shape[1], dx : dx + noise_shape[2]] = primary
                primary = tmp

            batch.append(primary)

        eta_delta = int(self.settings.eta_noise_seed_delta or 0)
        if eta_delta:
            self._generators = [self._create_generator(seed + eta_delta) for seed in seeds]

        stacked = torch.stack(batch)
        return stacked

    def next(self) -> torch.Tensor:
        if self._is_first:
            self._is_first = False
            noise = self._initial_noise()
            _LOGGER.debug(
                "rng.initial batch=%d shape=%s source=%s device=%s",
                noise.shape[0],
                tuple(noise.shape[1:]),
                self.settings.source.value,
                self._target,
            )
            return noise

        samples = [
            self._noise_from_generator(generator, self.shape)
            for generator in self._generators
        ]
        stacked = torch.stack(samples)
        return stacked


__all__ = ["ImageRNG", "NoiseSettings", "NoiseSourceKind"]
