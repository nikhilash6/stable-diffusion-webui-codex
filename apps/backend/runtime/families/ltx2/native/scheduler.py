"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native FlowMatch-Euler scheduler helpers for the LTX2 runtime lane.
Rebuilds the small scheduler surface the native LTX2 execution helpers need from the vendored scheduler config,
including dynamic flow-shift resolution, terminal stretching, and Euler stepping, without importing Diffusers
scheduler classes.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2FlowMatchEulerStepOutput` (dataclass): Small scheduler step result carrying `prev_sample`.
- `Ltx2FlowMatchEulerScheduler` (class): Config-driven native FlowMatch-Euler scheduler used by LTX2 txt2vid/img2vid.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from apps.backend.runtime.model_registry.flow_shift import FlowShiftMode, flow_shift_spec_from_config


@dataclass(frozen=True, slots=True)
class Ltx2FlowMatchEulerStepOutput:
    prev_sample: torch.Tensor


class Ltx2FlowMatchEulerScheduler:
    order: int = 1
    init_noise_sigma: float = 1.0

    def __init__(
        self,
        *,
        config: Mapping[str, object],
        config_path: str | None = None,
    ) -> None:
        self.config: dict[str, Any] = dict(config)
        self._config_path = config_path
        self._flow_shift_spec = flow_shift_spec_from_config(self.config, config_path=config_path)
        self._num_train_timesteps = int(self.config.get("num_train_timesteps", 1000) or 1000)
        if self._num_train_timesteps <= 0:
            raise RuntimeError(
                "LTX2 FlowMatchEuler scheduler requires num_train_timesteps > 0; "
                f"got {self._num_train_timesteps}."
            )

        self._invert_sigmas = bool(self.config.get("invert_sigmas") is True)
        self._stochastic_sampling = bool(self.config.get("stochastic_sampling") is True)
        self._use_karras_sigmas = bool(self.config.get("use_karras_sigmas") is True)
        self._use_exponential_sigmas = bool(self.config.get("use_exponential_sigmas") is True)
        self._use_beta_sigmas = bool(self.config.get("use_beta_sigmas") is True)
        if sum(
            int(flag)
            for flag in (
                self._use_karras_sigmas,
                self._use_exponential_sigmas,
                self._use_beta_sigmas,
            )
        ) > 1:
            raise RuntimeError(
                "LTX2 FlowMatchEuler scheduler config may enable at most one of use_karras_sigmas, "
                "use_exponential_sigmas, or use_beta_sigmas."
            )

        base_timesteps = np.linspace(1.0, float(self._num_train_timesteps), self._num_train_timesteps, dtype=np.float32)[
            ::-1
        ].copy()
        base_sigmas = base_timesteps / float(self._num_train_timesteps)
        if self._flow_shift_spec.mode is FlowShiftMode.FIXED:
            base_sigmas = self._apply_shift(base_sigmas, effective_shift=self._flow_shift_spec.resolve_effective_shift())

        self.timesteps = torch.from_numpy(base_sigmas * float(self._num_train_timesteps)).to(dtype=torch.float32)
        self.sigmas = torch.from_numpy(base_sigmas).to(dtype=torch.float32)
        self.sigma_min = float(self.sigmas[-1].item())
        self.sigma_max = float(self.sigmas[0].item())
        self.num_inference_steps = int(self.timesteps.numel())
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
        *,
        config_path: str | None = None,
    ) -> "Ltx2FlowMatchEulerScheduler":
        return cls(config=config, config_path=config_path)

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = int(begin_index)

    def _resolve_effective_shift(
        self,
        *,
        sequence_length: int | None,
        mu: float | None,
    ) -> float:
        if self._flow_shift_spec.mode is FlowShiftMode.FIXED:
            return float(self._flow_shift_spec.resolve_effective_shift())
        if mu is not None and sequence_length is not None:
            raise RuntimeError("LTX2 scheduler.set_timesteps() accepts only one of `mu` or `sequence_length`.")
        if mu is None and sequence_length is None:
            raise RuntimeError(
                "LTX2 dynamic FlowMatchEuler scheduling requires `sequence_length` (preferred) or `mu`."
            )
        if mu is None:
            return float(self._flow_shift_spec.resolve_effective_shift(seq_len=int(sequence_length)))
        time_shift_type = str(self.config.get("time_shift_type", "exponential") or "exponential").strip().lower()
        if time_shift_type == "linear":
            return float(mu)
        if time_shift_type == "exponential":
            return float(math.exp(float(mu)))
        raise RuntimeError(f"LTX2 scheduler config has unsupported time_shift_type={time_shift_type!r}.")

    @staticmethod
    def _apply_shift(sigmas: np.ndarray, *, effective_shift: float) -> np.ndarray:
        shift = float(effective_shift)
        if not math.isfinite(shift) or shift <= 0.0:
            raise RuntimeError(f"LTX2 FlowMatchEuler effective shift must be finite > 0; got {shift!r}.")
        return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)

    def _stretch_shift_to_terminal(self, sigmas: np.ndarray) -> np.ndarray:
        shift_terminal_raw = self.config.get("shift_terminal")
        if shift_terminal_raw is None:
            return sigmas
        shift_terminal = float(shift_terminal_raw)
        if shift_terminal == 0.0:
            return sigmas
        if not math.isfinite(shift_terminal) or shift_terminal >= 1.0:
            raise RuntimeError(
                "LTX2 scheduler shift_terminal must be finite and < 1.0 when provided; "
                f"got {shift_terminal!r}."
            )
        one_minus_z = 1.0 - sigmas
        scale_factor = one_minus_z[-1] / (1.0 - shift_terminal)
        if scale_factor == 0.0:
            raise RuntimeError("LTX2 scheduler cannot stretch sigmas to shift_terminal with a zero scale factor.")
        return 1.0 - (one_minus_z / scale_factor)

    def _convert_to_karras(self, in_sigmas: np.ndarray, *, num_inference_steps: int) -> np.ndarray:
        sigma_min = float(self.config.get("sigma_min", in_sigmas[-1]))
        sigma_max = float(self.config.get("sigma_max", in_sigmas[0]))
        rho = 7.0
        ramp = np.linspace(0.0, 1.0, int(num_inference_steps), dtype=np.float32)
        min_inv_rho = sigma_min ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

    def _convert_to_exponential(self, in_sigmas: np.ndarray, *, num_inference_steps: int) -> np.ndarray:
        sigma_min = float(self.config.get("sigma_min", in_sigmas[-1]))
        sigma_max = float(self.config.get("sigma_max", in_sigmas[0]))
        return np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), int(num_inference_steps), dtype=np.float32))

    def _convert_to_beta(self, in_sigmas: np.ndarray, *, num_inference_steps: int) -> np.ndarray:
        try:
            import scipy.stats  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("LTX2 scheduler beta sigmas require scipy.") from exc
        sigma_min = float(self.config.get("sigma_min", in_sigmas[-1]))
        sigma_max = float(self.config.get("sigma_max", in_sigmas[0]))
        alpha = 0.6
        beta = 0.6
        return np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1.0 - np.linspace(0.0, 1.0, int(num_inference_steps), dtype=np.float32)
                ]
            ],
            dtype=np.float32,
        )

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: torch.device | str | None = None,
        sigmas: Sequence[float] | np.ndarray | None = None,
        timesteps: Sequence[float] | np.ndarray | None = None,
        *,
        sequence_length: int | None = None,
        mu: float | None = None,
    ) -> torch.Tensor:
        if num_inference_steps is None and sigmas is None and timesteps is None:
            raise RuntimeError("LTX2 scheduler.set_timesteps() requires num_inference_steps, sigmas, or timesteps.")
        if sigmas is not None and timesteps is not None and len(sigmas) != len(timesteps):
            raise RuntimeError("LTX2 scheduler sigmas and timesteps must have identical lengths.")

        if num_inference_steps is not None:
            num_inference_steps = int(num_inference_steps)
            if num_inference_steps <= 0:
                raise RuntimeError(
                    f"LTX2 scheduler num_inference_steps must be > 0; got {num_inference_steps!r}."
                )
        else:
            inferred_len = len(sigmas) if sigmas is not None else len(timesteps)  # type: ignore[arg-type]
            num_inference_steps = int(inferred_len)

        if timesteps is not None:
            timestep_array = np.asarray(list(timesteps), dtype=np.float32)
        elif sigmas is None:
            timestep_array = np.linspace(
                self.sigma_max * float(self._num_train_timesteps),
                self.sigma_min * float(self._num_train_timesteps),
                int(num_inference_steps),
                dtype=np.float32,
            )
        else:
            timestep_array = None

        if sigmas is None:
            if timestep_array is None:
                raise RuntimeError("LTX2 scheduler internal error: timestep_array unexpectedly missing.")
            sigma_array = timestep_array / float(self._num_train_timesteps)
        else:
            sigma_array = np.asarray(list(sigmas), dtype=np.float32)
            if int(sigma_array.shape[0]) != int(num_inference_steps):
                raise RuntimeError(
                    "LTX2 scheduler sigmas length must match num_inference_steps; "
                    f"got sigmas={int(sigma_array.shape[0])} num_inference_steps={int(num_inference_steps)}."
                )

        effective_shift = self._resolve_effective_shift(sequence_length=sequence_length, mu=mu)
        sigma_array = self._apply_shift(sigma_array, effective_shift=effective_shift)
        sigma_array = self._stretch_shift_to_terminal(sigma_array)

        if self._use_karras_sigmas:
            sigma_array = self._convert_to_karras(sigma_array, num_inference_steps=int(num_inference_steps))
        elif self._use_exponential_sigmas:
            sigma_array = self._convert_to_exponential(sigma_array, num_inference_steps=int(num_inference_steps))
        elif self._use_beta_sigmas:
            sigma_array = self._convert_to_beta(sigma_array, num_inference_steps=int(num_inference_steps))

        target_device = None if device is None else torch.device(device)
        sigma_tensor = torch.from_numpy(sigma_array).to(dtype=torch.float32, device=target_device)
        if timestep_array is None:
            timestep_tensor = sigma_tensor * float(self._num_train_timesteps)
        else:
            timestep_tensor = torch.from_numpy(timestep_array).to(dtype=torch.float32, device=target_device)

        if self._invert_sigmas:
            sigma_tensor = 1.0 - sigma_tensor
            timestep_tensor = sigma_tensor * float(self._num_train_timesteps)
            sigma_tensor = torch.cat([sigma_tensor, torch.ones(1, dtype=sigma_tensor.dtype, device=sigma_tensor.device)])
        else:
            sigma_tensor = torch.cat([sigma_tensor, torch.zeros(1, dtype=sigma_tensor.dtype, device=sigma_tensor.device)])

        self.num_inference_steps = int(num_inference_steps)
        self.timesteps = timestep_tensor
        self.sigmas = sigma_tensor
        self._step_index = None
        self._begin_index = None
        return self.timesteps

    def index_for_timestep(
        self,
        timestep: int | float | torch.Tensor,
        schedule_timesteps: torch.Tensor | None = None,
    ) -> int:
        schedule = self.timesteps if schedule_timesteps is None else schedule_timesteps
        if not isinstance(schedule, torch.Tensor) or schedule.ndim != 1:
            raise RuntimeError("LTX2 scheduler expected 1D schedule_timesteps tensor.")
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() != 1:
                raise RuntimeError(
                    f"LTX2 scheduler expected scalar timestep tensor; got shape={tuple(int(dim) for dim in timestep.shape)!r}."
                )
            ts = timestep.to(device=schedule.device, dtype=schedule.dtype)
        else:
            ts = torch.tensor(float(timestep), device=schedule.device, dtype=schedule.dtype)
        matches = torch.isclose(schedule, ts, atol=1e-6, rtol=0.0).nonzero(as_tuple=False)
        if matches.numel() == 0:
            raise RuntimeError(f"LTX2 scheduler timestep {float(ts.item())!r} is not present in scheduler.timesteps.")
        position = 1 if int(matches.shape[0]) > 1 else 0
        return int(matches[position].item())

    def _init_step_index(self, timestep: int | float | torch.Tensor) -> None:
        if self._begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = int(self._begin_index or 0)

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(item) for item in timestep]
        elif self.step_index is not None:
            step_indices = [int(self.step_index)] * int(timestep.shape[0])
        else:
            step_indices = [int(self.begin_index)] * int(timestep.shape[0])
        sigma = self.sigmas.to(device=sample.device, dtype=sample.dtype)[step_indices].flatten()
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma * noise + (1.0 - sigma) * sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | float | torch.Tensor,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> Ltx2FlowMatchEulerStepOutput | tuple[torch.Tensor]:
        if self._step_index is None:
            self._init_step_index(timestep)
        if self._step_index is None:
            raise RuntimeError("LTX2 scheduler failed to initialize step_index.")
        if self._step_index < 0 or self._step_index >= int(self.timesteps.numel()):
            raise RuntimeError(
                "LTX2 scheduler step_index is out of range "
                f"(step_index={self._step_index} timesteps={int(self.timesteps.numel())})."
            )
        if self._step_index + 1 >= int(self.sigmas.numel()):
            raise RuntimeError(
                "LTX2 scheduler sigma index is out of range "
                f"(step_index={self._step_index} sigmas={int(self.sigmas.numel())})."
            )

        expected_timestep = float(self.timesteps[self._step_index].item())
        got_timestep = float(timestep.item()) if isinstance(timestep, torch.Tensor) else float(timestep)
        if not math.isclose(got_timestep, expected_timestep, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                "LTX2 scheduler.step() received a timestep that does not match the current step index. "
                f"Expected {expected_timestep}, got {got_timestep}."
            )

        sample_fp32 = sample.to(dtype=torch.float32)
        model_output_fp32 = model_output.to(dtype=torch.float32)
        sigmas = self.sigmas.to(device=sample_fp32.device, dtype=sample_fp32.dtype)
        sigma = sigmas[self._step_index]
        sigma_next = sigmas[self._step_index + 1]

        if self._stochastic_sampling:
            x0 = sample_fp32 - sigma * model_output_fp32
            noise = torch.randn(
                sample_fp32.shape,
                generator=generator,
                device=sample_fp32.device,
                dtype=sample_fp32.dtype,
            )
            prev_sample = (1.0 - sigma_next) * x0 + sigma_next * noise
        else:
            dt = sigma_next - sigma
            prev_sample = sample_fp32 + dt * model_output_fp32

        self._step_index += 1
        prev_sample = prev_sample.to(dtype=model_output.dtype)
        if not return_dict:
            return (prev_sample,)
        return Ltx2FlowMatchEulerStepOutput(prev_sample=prev_sample)

    def __len__(self) -> int:
        return int(self._num_train_timesteps)


__all__ = [
    "Ltx2FlowMatchEulerScheduler",
    "Ltx2FlowMatchEulerStepOutput",
]
