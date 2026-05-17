"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF scheduler helpers (Diffusers-free).
Provides a strict reader for `scheduler_config.json` (vendored HF mirror) and a WAN-only scheduler implementation that
matches the flow-sigmas schedule used by WAN2.2 (source-of-truth: `scheduler_config.json`), without importing Diffusers.
Includes explicit mixed-precision stability guards for UniPC corrector linear solves, per-device/dtype sigma cache reuse
for hot-path step calls (avoiding repeated scalar device materialization), and an experimental FlowMatch-Euler lane for
sampler experiments with caller-owned generator support for stochastic Euler updates.

Symbols (top-level; keep in sync; no ghosts):
- `WanSchedulerOutput` (dataclass): Minimal scheduler step output with `prev_sample` (Diffusers-compatible surface).
- `WanFlowMatchEulerScheduler` (class): WAN flow-match Euler scheduler surface for experimental sampler overrides.
- `WanUniPCFlowScheduler` (class): WAN-only UniPC multistep scheduler for flow-prediction models (BH1/BH2; order<=2).
- `build_wan_flow_match_euler_scheduler` (function): Build the WAN FlowMatch-Euler scheduler from vendored scheduler config + resolved `flow_shift`.
- `build_wan_unipc_flow_scheduler` (function): Build the WAN UniPC scheduler from vendored scheduler config and a resolved `flow_shift`.
- `infer_high_steps_from_boundary_ratio` (function): Derive default High/Low stage step split from `boundary_ratio` and vendor metadata.
- `load_wan_scheduler_config` (function): Read and validate a WAN `scheduler_config.json` mapping from a vendor dir.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch


def load_wan_scheduler_config(vendor_dir: str) -> Mapping[str, object]:
    root = Path(os.path.expanduser(str(vendor_dir))).resolve()

    config_path: Path | None = None
    for fname in ("scheduler_config.json", "config.json"):
        candidate = root / "scheduler" / fname
        if candidate.is_file():
            config_path = candidate
            break
    if config_path is None:
        raise RuntimeError(f"WAN22 GGUF: missing scheduler_config.json under: {str(root)!r}")

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - strict decode
        raise RuntimeError(f"WAN22 GGUF: invalid scheduler config JSON: {str(config_path)}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"WAN22 GGUF: scheduler config must be a JSON object: {str(config_path)}")
    return data


def _require_str(cfg: Mapping[str, object], key: str, *, label: str) -> str:
    raw = cfg.get(key)
    val = str(raw or "").strip()
    if not val:
        raise RuntimeError(f"WAN22 GGUF: missing {key} in {label}")
    return val


def _require_int(cfg: Mapping[str, object], key: str, *, label: str, default: int | None = None) -> int:
    raw = cfg.get(key, default)
    try:
        return int(raw)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"WAN22 GGUF: invalid {key}={raw!r} in {label}") from exc


def _require_bool(cfg: Mapping[str, object], key: str, *, label: str, default: bool = False) -> bool:
    raw = cfg.get(key, default)
    return bool(raw is True)

def _optional_str(cfg: Mapping[str, object], key: str) -> str | None:
    raw = cfg.get(key)
    if raw is None:
        return None
    val = str(raw).strip()
    return val or None


def _build_flow_sigmas(
    *,
    steps: int,
    flow_shift: float,
    num_train_timesteps: int,
    final_sigmas_type: str,
) -> torch.Tensor:
    steps = max(int(steps), 1)
    if not math.isfinite(float(flow_shift)) or float(flow_shift) <= 0:
        raise RuntimeError(f"WAN22 GGUF: invalid flow_shift={flow_shift!r} (expected finite > 0)")
    if int(num_train_timesteps) <= 0:
        raise RuntimeError(
            f"WAN22 GGUF: invalid num_train_timesteps={int(num_train_timesteps)} (expected > 0)"
        )

    # Match Diffusers UniPCMultistepScheduler when `use_flow_sigmas=True`:
    # sigmas = linspace(1, 1/num_train_timesteps, steps+1)[:-1]
    base = torch.linspace(
        1.0,
        1.0 / float(int(num_train_timesteps)),
        int(steps) + 1,
        dtype=torch.float32,
    )[:-1]

    # Fixed shifting for WAN2.2 (dynamic shifting is not supported here).
    shift = float(flow_shift)
    sigmas = shift * base / (1.0 + (shift - 1.0) * base)

    # Avoid log(1 - sigma) edge cases upstream by nudging the first sigma off exactly-1.
    eps = 1e-6
    if abs(float(sigmas[0]) - 1.0) < eps:
        sigmas[0] = sigmas[0] - eps

    if final_sigmas_type == "zero":
        sigma_last = torch.zeros(1, dtype=torch.float32)
    elif final_sigmas_type == "sigma_min":
        sigma_last = sigmas[-1:]
    else:
        raise RuntimeError(
            f"WAN22 GGUF: unsupported final_sigmas_type={final_sigmas_type!r} (expected 'zero' or 'sigma_min')"
        )

    return torch.cat([sigmas, sigma_last]).to("cpu")


@dataclass(frozen=True)
class WanSchedulerOutput:
    prev_sample: torch.Tensor


class WanFlowMatchEulerScheduler:
    """WAN flow-match Euler scheduler surface used by the GGUF stage sampler."""

    init_noise_sigma: float = 1.0
    order: int = 1

    def __init__(
        self,
        *,
        sigmas: torch.Tensor,
        timesteps: torch.Tensor,
        stochastic_sampling: bool,
    ) -> None:
        if not isinstance(sigmas, torch.Tensor):
            raise TypeError("sigmas must be a torch.Tensor")
        if not isinstance(timesteps, torch.Tensor):
            raise TypeError("timesteps must be a torch.Tensor")
        if sigmas.ndim != 1:
            raise RuntimeError(f"WAN22 GGUF: expected 1D sigmas, got shape={tuple(sigmas.shape)}")
        if timesteps.ndim != 1:
            raise RuntimeError(f"WAN22 GGUF: expected 1D timesteps, got shape={tuple(timesteps.shape)}")
        if sigmas.numel() < 2:
            raise RuntimeError("WAN22 GGUF: sigma ladder must have at least 2 elements")
        if timesteps.numel() != sigmas.numel() - 1:
            raise RuntimeError(
                "WAN22 GGUF: timesteps must align with sigmas "
                f"(timesteps={int(timesteps.numel())} sigmas={int(sigmas.numel())})."
            )

        self.sigmas = sigmas.detach().to(device="cpu", dtype=torch.float32)
        self.timesteps = timesteps.detach().to(device="cpu", dtype=torch.float32)
        self._stochastic_sampling = bool(stochastic_sampling)
        self._sigmas_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._step_index: int | None = None
        self._begin_index: int | None = None
        self.config = SimpleNamespace(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            stochastic_sampling=self._stochastic_sampling,
        )

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = int(begin_index)

    def _sigmas_for(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._sigmas_cache.get(key)
        if cached is None:
            cached = self.sigmas.to(device=device, dtype=dtype)
            self._sigmas_cache[key] = cached
        return cached

    def index_for_timestep(self, timestep: int | float | torch.Tensor, schedule_timesteps: torch.Tensor | None = None) -> int:
        schedule = self.timesteps if schedule_timesteps is None else schedule_timesteps
        if not isinstance(schedule, torch.Tensor) or schedule.ndim != 1:
            raise RuntimeError("WAN22 GGUF: schedule_timesteps must be a 1D tensor.")

        ts = timestep
        if not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts, dtype=schedule.dtype, device=schedule.device)
        elif ts.numel() != 1:
            raise RuntimeError(f"WAN22 GGUF: expected scalar timestep, got shape={tuple(ts.shape)}")
        else:
            ts = ts.to(device=schedule.device, dtype=schedule.dtype)

        indices = (schedule == ts).nonzero(as_tuple=False)
        if indices.numel() == 0:
            raise RuntimeError(f"WAN22 GGUF: timestep {float(ts.item())!r} is not present in scheduler.timesteps.")
        pos = 1 if int(indices.shape[0]) > 1 else 0
        return int(indices[pos].item())

    def _init_step_index(self, timestep: int | float | torch.Tensor) -> None:
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = int(self._begin_index or 0)

    def scale_model_input(self, sample: torch.Tensor, timestep: Any) -> torch.Tensor:
        return sample

    def step(
        self,
        *,
        model_output: torch.Tensor,
        timestep: int | float | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> WanSchedulerOutput:
        if self._step_index is None:
            self._init_step_index(timestep)

        assert self._step_index is not None
        if self._step_index < 0 or self._step_index >= int(self.timesteps.numel()):
            raise RuntimeError(
                "WAN22 GGUF: scheduler internal step_index out of range "
                f"(step_index={self._step_index} timesteps={int(self.timesteps.numel())})."
            )
        if self._step_index + 1 >= int(self.sigmas.numel()):
            raise RuntimeError(
                "WAN22 GGUF: scheduler sigma index out of range "
                f"(step_index={self._step_index} sigmas={int(self.sigmas.numel())})."
            )

        expected = float(self.timesteps[self._step_index].item())
        try:
            got = float(timestep.item()) if isinstance(timestep, torch.Tensor) else float(timestep)
        except Exception as exc:  # noqa: BLE001 - strict input validation
            raise RuntimeError(f"WAN22 GGUF: invalid timestep value: {timestep!r}") from exc
        if not math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                "WAN22 GGUF: timestep mismatch for scheduler.step(). "
                f"Expected timesteps[{self._step_index}]={expected}, got {got}."
            )

        sample_fp32 = sample.to(dtype=torch.float32)
        model_output_fp32 = model_output.to(dtype=torch.float32)
        sigmas = self._sigmas_for(device=sample_fp32.device, dtype=sample_fp32.dtype)
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
        return WanSchedulerOutput(prev_sample=prev_sample.to(dtype=sample.dtype))


class WanUniPCFlowScheduler:
    """WAN-only scheduler surface used by the GGUF stage sampler.

    This is a **strict**, WAN-only implementation mirroring Diffusers `UniPCMultistepScheduler` behavior for
    WAN2.2 configs (flow sigmas + flow prediction). It is intentionally not a general-purpose scheduler.
    """

    init_noise_sigma: float = 1.0
    order: int = 1

    def __init__(
        self,
        *,
        sigmas: torch.Tensor,
        timesteps: torch.Tensor,
        solver_order: int,
        solver_type: str,
        predict_x0: bool,
        prediction_type: str,
        lower_order_final: bool,
        disable_corrector: Sequence[int],
    ) -> None:
        if not isinstance(sigmas, torch.Tensor):
            raise TypeError("sigmas must be a torch.Tensor")
        if not isinstance(timesteps, torch.Tensor):
            raise TypeError("timesteps must be a torch.Tensor")
        if sigmas.ndim != 1:
            raise RuntimeError(f"WAN22 GGUF: expected 1D sigmas, got shape={tuple(sigmas.shape)}")
        if timesteps.ndim != 1:
            raise RuntimeError(f"WAN22 GGUF: expected 1D timesteps, got shape={tuple(timesteps.shape)}")
        if sigmas.numel() < 2:
            raise RuntimeError("WAN22 GGUF: sigma ladder must have at least 2 elements")
        if timesteps.numel() != sigmas.numel() - 1:
            raise RuntimeError(
                "WAN22 GGUF: timesteps must align with sigmas "
                f"(timesteps={int(timesteps.numel())} sigmas={int(sigmas.numel())})."
            )

        solver_order = int(solver_order)
        if solver_order not in (1, 2):
            raise NotImplementedError(
                f"WAN22 GGUF: UniPC solver_order={solver_order} is not supported yet (supported: 1, 2)."
            )

        solver_type_norm = str(solver_type or "").strip().lower()
        if solver_type_norm not in {"bh1", "bh2"}:
            raise NotImplementedError(
                f"WAN22 GGUF: UniPC solver_type={solver_type!r} is not supported (supported: 'bh1', 'bh2')."
            )

        pred_type_norm = str(prediction_type or "").strip().lower()
        if pred_type_norm != "flow_prediction":
            raise NotImplementedError(
                "WAN22 GGUF: UniPC scheduler only supports prediction_type='flow_prediction' "
                f"(got {prediction_type!r})."
            )
        if not bool(predict_x0 is True):
            raise NotImplementedError("WAN22 GGUF: UniPC scheduler requires predict_x0=true for WAN2.2.")

        self.sigmas = sigmas.detach().to(device="cpu", dtype=torch.float32)
        self.timesteps = timesteps.detach().to(device="cpu", dtype=torch.int64)
        self._sigmas_cache: dict[tuple[str, str], torch.Tensor] = {}

        self._solver_order = solver_order
        self._solver_type = solver_type_norm
        self._predict_x0 = True
        self._prediction_type = pred_type_norm
        self._lower_order_final = bool(lower_order_final is True)
        self._disable_corrector = {int(x) for x in disable_corrector}

        # Diffusers-compatible surface: allow reading `.config.*` in logs/tests when needed.
        self.config = SimpleNamespace(
            solver_order=self._solver_order,
            solver_type=self._solver_type,
            prediction_type=self._prediction_type,
            predict_x0=self._predict_x0,
            lower_order_final=self._lower_order_final,
            disable_corrector=sorted(self._disable_corrector),
            use_flow_sigmas=True,
        )

        # Multistep state (mirrors Diffusers reset in set_timesteps()).
        self.model_outputs: list[torch.Tensor | None] = [None] * self._solver_order
        self.lower_order_nums: int = 0
        self.last_sample: torch.Tensor | None = None
        self.this_order: int = 0
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def _sigmas_for(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._sigmas_cache.get(key)
        if cached is None:
            cached = self.sigmas.to(device=device, dtype=dtype)
            self._sigmas_cache[key] = cached
        return cached

    def _sigma_to_alpha_sigma_t(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_t = 1 - sigma
        sigma_t = sigma
        return alpha_t, sigma_t

    def _convert_model_output(self, model_output: torch.Tensor, *, sample: torch.Tensor) -> torch.Tensor:
        if self._step_index is None:
            raise RuntimeError("WAN22 GGUF: scheduler.step_index is not initialized (call step() first).")
        if self._prediction_type != "flow_prediction" or not self._predict_x0:
            raise NotImplementedError("WAN22 GGUF: only flow_prediction + predict_x0=true is implemented.")

        sigmas = self._sigmas_for(device=sample.device, dtype=sample.dtype)
        sigma = sigmas[self._step_index]
        return sample - sigma * model_output

    def _B_h(self, hh: torch.Tensor) -> torch.Tensor:
        if self._solver_type == "bh1":
            return hh
        if self._solver_type == "bh2":
            return torch.expm1(hh)
        raise NotImplementedError(f"WAN22 GGUF: unsupported solver_type={self._solver_type!r}")

    def _multistep_uni_p_bh_update(self, *, sample: torch.Tensor, order: int) -> torch.Tensor:
        if self._step_index is None:
            raise RuntimeError("WAN22 GGUF: scheduler.step_index is not initialized (call step() first).")
        if order not in (1, 2):
            raise NotImplementedError(f"WAN22 GGUF: UniPC order={order} is not supported (supported: 1, 2).")
        if self._step_index + 1 >= int(self.sigmas.numel()):
            raise RuntimeError("WAN22 GGUF: scheduler.step() called past the end of the sigma ladder.")

        m0 = self.model_outputs[-1]
        if m0 is None:
            raise RuntimeError("WAN22 GGUF: missing current model_output history (m0).")

        x = sample
        sigmas = self._sigmas_for(device=x.device, dtype=x.dtype)
        sigma_t = sigmas[self._step_index + 1]
        sigma_s0 = sigmas[self._step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        hh = -h
        h_phi_1 = torch.expm1(hh)
        B_h = self._B_h(hh)

        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if order == 1:
            return x_t_.to(x.dtype)

        if self._step_index < 1:
            raise RuntimeError("WAN22 GGUF: UniPC order=2 requires step_index>=1.")
        mi = self.model_outputs[-2]
        if mi is None:
            raise RuntimeError("WAN22 GGUF: UniPC order=2 requires 2 model outputs, but history is incomplete.")

        sigma_si = sigmas[self._step_index - 1]
        alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(sigma_si)
        lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
        rk = (lambda_si - lambda_s0) / h
        D1 = (mi - m0) / rk

        pred_res = 0.5 * D1
        x_t = x_t_ - alpha_t * B_h * pred_res
        return x_t.to(x.dtype)

    def _multistep_uni_c_bh_update(
        self,
        *,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        if self._step_index is None:
            raise RuntimeError("WAN22 GGUF: scheduler.step_index is not initialized (call step() first).")
        if self._step_index < 1:
            raise RuntimeError("WAN22 GGUF: UniPC corrector requires step_index>=1.")
        if order not in (1, 2):
            raise NotImplementedError(f"WAN22 GGUF: UniPC corrector order={order} is not supported (supported: 1, 2).")

        m0 = self.model_outputs[-1]
        if m0 is None:
            raise RuntimeError("WAN22 GGUF: missing previous model_output history (m0).")

        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigmas = self._sigmas_for(device=x_t.device, dtype=x_t.dtype)
        sigma_t = sigmas[self._step_index]
        sigma_s0 = sigmas[self._step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        B_h = self._B_h(hh)

        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        D1_t = model_t - m0

        device = x_t.device
        if order == 1:
            x_out = x_t_ - alpha_t * B_h * (0.5 * D1_t)
            return x_out.to(x.dtype)

        if self._step_index < 2:
            raise RuntimeError("WAN22 GGUF: UniPC corrector order=2 requires step_index>=2.")
        mi = self.model_outputs[-2]
        if mi is None:
            raise RuntimeError("WAN22 GGUF: UniPC corrector order=2 requires 2 model outputs, but history is incomplete.")

        sigma_si = sigmas[self._step_index - 2]
        alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(sigma_si)
        lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
        rk = (lambda_si - lambda_s0) / h
        D1 = (mi - m0) / rk

        rks = torch.tensor([rk, 1.0], device=device, dtype=x.dtype)
        R = []
        b = []
        factorial_i = 1
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        Rm = torch.stack(R)
        bv = torch.stack(b)
        if Rm.dtype in (torch.float16, torch.bfloat16):
            solve_dtype = torch.float32
            Rm = Rm.to(dtype=solve_dtype)
            bv = bv.to(dtype=solve_dtype)
        rhos_c = torch.linalg.solve(Rm, bv).to(dtype=x.dtype)

        corr_res = rhos_c[0] * D1
        x_out = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        return x_out.to(x.dtype)

    def index_for_timestep(self, timestep: int | torch.Tensor, schedule_timesteps: torch.Tensor | None = None) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        ts = timestep
        if isinstance(ts, torch.Tensor):
            if ts.numel() != 1:
                raise RuntimeError(f"WAN22 GGUF: expected scalar timestep, got shape={tuple(ts.shape)}")
            ts = ts.to(schedule_timesteps.device)
        index_candidates = (schedule_timesteps == ts).nonzero()

        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()
        return int(step_index)

    def _init_step_index(self, timestep: int | torch.Tensor) -> None:
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = int(self._begin_index or 0)

    def scale_model_input(self, sample: torch.Tensor, timestep: Any) -> torch.Tensor:
        return sample

    def step(
        self,
        *,
        model_output: torch.Tensor,
        timestep: Any,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> WanSchedulerOutput:
        del generator
        if self._step_index is None:
            self._init_step_index(timestep)

        assert self._step_index is not None
        if self._step_index < 0 or self._step_index >= int(self.timesteps.numel()):
            raise RuntimeError(
                "WAN22 GGUF: scheduler internal step_index out of range "
                f"(step_index={self._step_index} timesteps={int(self.timesteps.numel())})."
            )

        try:
            expected = int(self.timesteps[self._step_index].item())
            got = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        except Exception as exc:  # noqa: BLE001 - strict input validation
            raise RuntimeError(f"WAN22 GGUF: invalid timestep value: {timestep!r}") from exc
        if got != expected:
            raise RuntimeError(
                "WAN22 GGUF: timestep mismatch for scheduler.step(). "
                f"Expected timesteps[{self._step_index}]={expected}, got {got}."
            )

        use_corrector = (
            self._step_index > 0
            and (self._step_index - 1) not in self._disable_corrector
            and self.last_sample is not None
        )

        model_output_convert = self._convert_model_output(model_output, sample=sample)
        if use_corrector:
            sample = self._multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self._solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output_convert

        if self._lower_order_final:
            this_order = min(self._solver_order, int(self.timesteps.numel()) - self._step_index)
        else:
            this_order = self._solver_order
        self.this_order = min(int(this_order), int(self.lower_order_nums) + 1)
        if self.this_order <= 0:
            raise RuntimeError("WAN22 GGUF: UniPC warmup produced invalid order (this_order<=0).")

        self.last_sample = sample
        prev_sample = self._multistep_uni_p_bh_update(sample=sample, order=self.this_order)

        if self.lower_order_nums < self._solver_order:
            self.lower_order_nums += 1

        self._step_index += 1
        return WanSchedulerOutput(prev_sample=prev_sample)


def infer_high_steps_from_boundary_ratio(
    *,
    total_steps: int,
    boundary_ratio: float,
    vendor_dir: str,
    flow_shift: float,
) -> int:
    total_steps = int(total_steps)
    if total_steps < 2:
        raise RuntimeError(f"WAN22 GGUF: total_steps must be >= 2, got {total_steps}")
    if not (0.0 < float(boundary_ratio) < 1.0):
        raise RuntimeError(f"WAN22 GGUF: boundary_ratio must be in (0,1), got {boundary_ratio!r}")

    cfg = load_wan_scheduler_config(vendor_dir)
    class_name = _require_str(cfg, "_class_name", label="scheduler_config.json")
    if class_name != "UniPCMultistepScheduler":
        raise RuntimeError(
            f"WAN22 GGUF: unsupported scheduler _class_name={class_name!r} (expected 'UniPCMultistepScheduler')"
        )

    if not bool(cfg.get("use_flow_sigmas") is True):
        raise RuntimeError("WAN22 GGUF: scheduler_config.json must set use_flow_sigmas=true for WAN2.2")

    if _require_bool(cfg, "use_dynamic_shifting", label="scheduler_config.json", default=False):
        raise RuntimeError(
            "WAN22 GGUF: dynamic shifting is not supported in the GGUF runtime yet (use_dynamic_shifting=true)."
        )
    if _optional_str(cfg, "shift_terminal") is not None:
        raise RuntimeError(
            "WAN22 GGUF: scheduler_config.json uses shift_terminal, which is not supported in the GGUF runtime yet."
        )

    num_train_timesteps = _require_int(cfg, "num_train_timesteps", label="scheduler_config.json", default=1000)
    final_sigmas_type = str(cfg.get("final_sigmas_type") or "zero").strip().lower() or "zero"

    sigmas = _build_flow_sigmas(
        steps=total_steps,
        flow_shift=float(flow_shift),
        num_train_timesteps=num_train_timesteps,
        final_sigmas_type=final_sigmas_type,
    )

    # Match prior behavior: compute a boundary timestep in training-step space.
    boundary_timestep = float(boundary_ratio) * float(num_train_timesteps)
    train_timesteps = (sigmas[:-1] * float(num_train_timesteps)).to(dtype=torch.int64)
    hi_steps = int((train_timesteps >= boundary_timestep).sum().item())
    hi_steps = max(1, min(hi_steps, total_steps - 1))
    return hi_steps


def build_wan_unipc_flow_scheduler(
    *,
    steps: int,
    vendor_dir: str,
    flow_shift: float,
) -> WanUniPCFlowScheduler:
    cfg = load_wan_scheduler_config(vendor_dir)
    class_name = _require_str(cfg, "_class_name", label="scheduler_config.json")
    if class_name != "UniPCMultistepScheduler":
        raise RuntimeError(
            f"WAN22 GGUF: unsupported scheduler _class_name={class_name!r} (expected 'UniPCMultistepScheduler')"
        )
    if not bool(cfg.get("use_flow_sigmas") is True):
        raise RuntimeError("WAN22 GGUF: scheduler_config.json must set use_flow_sigmas=true for WAN2.2")
    if _require_bool(cfg, "use_dynamic_shifting", label="scheduler_config.json", default=False):
        raise RuntimeError(
            "WAN22 GGUF: dynamic shifting is not supported in the GGUF runtime yet (use_dynamic_shifting=true)."
        )
    if _optional_str(cfg, "shift_terminal") is not None:
        raise RuntimeError(
            "WAN22 GGUF: scheduler_config.json uses shift_terminal, which is not supported in the GGUF runtime yet."
        )
    num_train_timesteps = _require_int(cfg, "num_train_timesteps", label="scheduler_config.json", default=1000)
    final_sigmas_type = str(cfg.get("final_sigmas_type") or "zero").strip().lower() or "zero"
    sigmas = _build_flow_sigmas(
        steps=int(steps),
        flow_shift=float(flow_shift),
        num_train_timesteps=num_train_timesteps,
        final_sigmas_type=final_sigmas_type,
    )
    timesteps = (sigmas[:-1] * float(num_train_timesteps)).to(device="cpu", dtype=torch.int64)

    solver_order = _require_int(cfg, "solver_order", label="scheduler_config.json", default=2)
    solver_type = str(cfg.get("solver_type") or "bh2").strip()
    prediction_type = str(cfg.get("prediction_type") or "").strip()
    predict_x0 = bool(cfg.get("predict_x0") is True)
    lower_order_final = bool(cfg.get("lower_order_final") is True)
    disable_corrector_raw = cfg.get("disable_corrector") or []
    if not isinstance(disable_corrector_raw, list):
        raise RuntimeError(
            "WAN22 GGUF: scheduler_config.json disable_corrector must be a list of integers, "
            f"got {type(disable_corrector_raw).__name__}."
        )
    try:
        disable_corrector = [int(x) for x in disable_corrector_raw]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"WAN22 GGUF: scheduler_config.json disable_corrector must be a list of integers, got {disable_corrector_raw!r}."
        ) from exc

    if bool(cfg.get("thresholding") is True):
        raise RuntimeError("WAN22 GGUF: thresholding=true is not supported in the GGUF runtime (unexpected WAN config).")

    return WanUniPCFlowScheduler(
        sigmas=sigmas,
        timesteps=timesteps,
        solver_order=solver_order,
        solver_type=solver_type,
        predict_x0=predict_x0,
        prediction_type=prediction_type,
        lower_order_final=lower_order_final,
        disable_corrector=disable_corrector,
    )


def build_wan_flow_match_euler_scheduler(
    *,
    steps: int,
    vendor_dir: str,
    flow_shift: float,
    stochastic_sampling: bool = False,
) -> WanFlowMatchEulerScheduler:
    cfg = load_wan_scheduler_config(vendor_dir)
    class_name = _require_str(cfg, "_class_name", label="scheduler_config.json")
    if class_name not in {"UniPCMultistepScheduler", "FlowMatchEulerDiscreteScheduler"}:
        raise RuntimeError(
            "WAN22 GGUF: experimental FlowMatch-Euler lane requires metadata scheduler "
            f"'UniPCMultistepScheduler' or 'FlowMatchEulerDiscreteScheduler', got {class_name!r}."
        )
    if _require_bool(cfg, "use_dynamic_shifting", label="scheduler_config.json", default=False):
        raise RuntimeError(
            "WAN22 GGUF: dynamic shifting is not supported in the GGUF runtime yet (use_dynamic_shifting=true)."
        )
    if _optional_str(cfg, "shift_terminal") is not None:
        raise RuntimeError(
            "WAN22 GGUF: scheduler_config.json uses shift_terminal, which is not supported in the GGUF runtime yet."
        )

    num_train_timesteps = _require_int(cfg, "num_train_timesteps", label="scheduler_config.json", default=1000)
    final_sigmas_type = str(cfg.get("final_sigmas_type") or "zero").strip().lower() or "zero"
    sigmas = _build_flow_sigmas(
        steps=int(steps),
        flow_shift=float(flow_shift),
        num_train_timesteps=num_train_timesteps,
        final_sigmas_type=final_sigmas_type,
    )
    timesteps = (sigmas[:-1] * float(num_train_timesteps)).to(device="cpu", dtype=torch.float32)
    return WanFlowMatchEulerScheduler(
        sigmas=sigmas,
        timesteps=timesteps,
        stochastic_sampling=bool(stochastic_sampling),
    )


__all__ = [
    "WanFlowMatchEulerScheduler",
    "WanSchedulerOutput",
    "build_wan_flow_match_euler_scheduler",
    "WanUniPCFlowScheduler",
    "build_wan_unipc_flow_scheduler",
    "infer_high_steps_from_boundary_ratio",
    "load_wan_scheduler_config",
]
