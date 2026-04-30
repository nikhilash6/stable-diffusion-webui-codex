"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Optional sampler extras (UniPC, Restart, DDPM).
Provides lightweight noise-schedule utilities, a pure restart-step planner, and step samplers used by optional/experimental
sampling paths (native-only).

Symbols (top-level; keep in sync; no ghosts):
- `NoiseScheduleVP` (class): VP noise schedule helper (discrete/linear/cosine) providing alpha/sigma/lambda accessors for samplers.
- `get_sigmas_karras` (function): Karras sigma schedule builder with a terminal zero.
- `to_d` (function): Sigma-space ODE derivative helper for denoised predictions.
- `model_wrapper` (function): Wraps a model into a VP-space epsilon predictor with optional guidance modes.
- `sample_unipc` (function): Minimal UniPC sampler implementation operating over a sigma schedule.
- `RestartStepPlan` (dataclass): One executable restart step with explicit sigma bounds and optional renoise scale.
- `build_restart_step_plan` (function): Pure restart planner that expands restart segments over an active sigma ladder.
- `restart_sampler` (function): Restart sampling wrapper (supports restart segments + noise injection).
- `default_noise_sampler` (function): Returns a default noise sampler closure for stochastic samplers.
- `generic_step_sampler` (function): Generic sampler driver that iterates sigmas and calls a provided step function.
- `DDPMSampler_step` (function): Single-step DDPM update function used by the generic step sampler.
- `sample_ddpm` (function): DDPM sampler wrapper using `generic_step_sampler`.
"""

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import torch
from tqdm import trange

def get_sigmas_karras(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rho: float = 7.0,
) -> torch.Tensor:
    """Karras schedule helper (returns sigmas with a terminal 0)."""
    if steps <= 0:
        raise ValueError("steps must be >= 1")
    ramp = torch.linspace(0, 1, steps, device=device, dtype=dtype)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + (min_inv - max_inv) * ramp) ** rho
    terminal = torch.zeros(1, device=device, dtype=dtype)
    return torch.cat([sigmas, terminal])


def to_d(x: torch.Tensor, sigma: torch.Tensor | float, denoised: torch.Tensor) -> torch.Tensor:
    """ODE derivative helper used by sigma-space samplers."""
    if isinstance(sigma, torch.Tensor):
        if torch.any(sigma == 0):
            return torch.zeros_like(x)
        return (x - denoised) / sigma
    sigma_f = float(sigma)
    if sigma_f == 0.0:
        return torch.zeros_like(x)
    return (x - denoised) / sigma_f


class NoiseScheduleVP:
    def __init__(self, schedule="discrete", *, betas=None, alphas_cumprod=None, continuous_beta_0=0.1, continuous_beta_1=20.0):
        if schedule not in {"discrete", "linear", "cosine"}:
            raise ValueError(f"Unsupported noise schedule {schedule}")
        self.schedule = schedule
        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                if alphas_cumprod is None:
                    raise ValueError("alphas_cumprod required for discrete schedule")
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.0
            self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1))
            self.log_alpha_array = log_alphas.reshape((1, -1))
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.0
            self.cosine_t_max = (
                math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi)
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0))
            self.schedule = schedule
            self.T = 0.9946 if schedule == "cosine" else 1.0

    def marginal_log_mean_coeff(self, t):
        if self.schedule == "discrete":
            return torch.interp(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        if self.schedule == "linear":
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        log_alpha = torch.log(torch.cos((t + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0))
        return log_alpha - self.cosine_log_alpha_0

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std


def model_wrapper(model, noise_schedule: NoiseScheduleVP, model_type="noise", model_kwargs=None, guidance_type="uncond", condition=None, unconditional_condition=None, guidance_scale=1.0):
    model_kwargs = model_kwargs or {}
    def _fn(x, t_continuous):
        t = t_continuous
        if noise_schedule.schedule == "discrete":
            t = t * 999
        sigma = noise_schedule.marginal_std(t)
        model_sigma = sigma
        model_input = x / noise_schedule.marginal_alpha(t).view(x.shape[0], *((1,) * (x.ndim - 1)))
        eps = model(model_input, model_sigma, **model_kwargs)
        if model_type == "x_start":
            x0_pred = eps
            eps = (x - x0_pred * noise_schedule.marginal_alpha(t).view(x.shape[0], *((1,) * (x.ndim - 1)))) / noise_schedule.marginal_std(t).view(x.shape[0], *((1,) * (x.ndim - 1)))
        elif model_type == "v":
            eps = noise_schedule.marginal_std(t).view(x.shape[0], *((1,) * (x.ndim - 1))) * eps + noise_schedule.marginal_alpha(t).view(x.shape[0], *((1,) * (x.ndim - 1))) * x
        if guidance_type == "uncond" or guidance_type is None:
            return eps
        if guidance_type == "classifier":
            raise NotImplementedError
        if guidance_type == "classifier-free":
            cond = condition
            uncond = unconditional_condition
            if cond is None or uncond is None:
                return eps
            eps_uncond, eps_cond = eps.chunk(2)
            return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        raise ValueError(f"Unknown guidance type {guidance_type}")
    return _fn


def sample_unipc(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = extra_args or {}
    # Convert sigmas (descending) to t in [0,1]
    # Use VP discrete schedule; len(sigmas)-1 steps.
    alphas_cumprod = 1.0 / (sigmas ** 2 + 1.0)
    noise_schedule = NoiseScheduleVP("discrete", alphas_cumprod=alphas_cumprod[:-1].flip(0))
    model_fn = model_wrapper(model, noise_schedule, model_kwargs=extra_args, guidance_type=None)

    # Basic UniPC (order=2) adapted from official implementation
    timesteps = noise_schedule.t_array.to(x.device).view(-1)
    # map sigmas length to timesteps length; pad to match
    if timesteps.numel() < sigmas.numel():
        timesteps = torch.linspace(0, 1, sigmas.numel(), device=x.device)

    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur = timesteps[i]
        t_next = timesteps[i + 1]
        h = t_next - t_cur
        eps = model_fn(x, t_cur)
        x = x + h * eps
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": x - eps})
    return x


@dataclass(frozen=True)
class RestartStepPlan:
    sigma_current: float
    sigma_next: float
    renoise_scale: float = 0.0


def build_restart_step_plan(
    sigmas: torch.Tensor,
    *,
    restart_list: Mapping[float, Sequence[float]] | None = None,
) -> list[RestartStepPlan]:
    if sigmas.ndim != 1:
        raise RuntimeError(f"Restart planner expects a 1D sigma schedule; got shape={tuple(sigmas.shape)}.")
    if int(sigmas.numel()) < 2:
        raise RuntimeError("Restart planner requires at least two sigma entries (start and end).")
    if not bool(torch.all(torch.isfinite(sigmas))):
        raise RuntimeError("Restart planner requires a finite sigma schedule.")

    working_sigmas = sigmas
    steps = int(working_sigmas.numel()) - 1
    normalized_restart_list: Mapping[float, Sequence[float]]
    if restart_list is None:
        if steps >= 20:
            restart_steps = 9
            restart_times = 1
            if steps >= 36:
                restart_steps = steps // 4
                restart_times = 2
            base_steps = steps - restart_steps * restart_times
            if base_steps < 1:
                raise RuntimeError(
                    "Restart auto-plan produced an invalid base step count "
                    f"(steps={steps}, restart_steps={restart_steps}, restart_times={restart_times})."
                )
            working_sigmas = get_sigmas_karras(
                base_steps,
                float(working_sigmas[-2].item()),
                float(working_sigmas[0].item()),
                device=working_sigmas.device,
                dtype=working_sigmas.dtype,
            )
            normalized_restart_list = {0.1: (restart_steps + 1, restart_times, 2.0)}
        else:
            normalized_restart_list = {}
    else:
        normalized_restart_list = restart_list

    restart_map: dict[int, tuple[int, int, float]] = {}
    for sigma_anchor, raw_value in normalized_restart_list.items():
        if len(raw_value) != 3:
            raise RuntimeError(
                "Restart planner requires restart_list entries shaped as "
                "{sigma_anchor: [restart_steps, restart_times, restart_max]}."
            )
        restart_steps_raw, restart_times_raw, restart_max_raw = raw_value
        restart_steps = int(restart_steps_raw)
        restart_times = int(restart_times_raw)
        restart_max = float(restart_max_raw)
        if restart_steps < 2:
            raise RuntimeError(f"Restart planner requires restart_steps >= 2; got {restart_steps}.")
        if restart_times < 1:
            raise RuntimeError(f"Restart planner requires restart_times >= 1; got {restart_times}.")
        if not math.isfinite(restart_max) or restart_max <= 0.0:
            raise RuntimeError(f"Restart planner requires finite positive restart_max; got {restart_max}.")
        anchor_index = int(torch.argmin(torch.abs(working_sigmas - float(sigma_anchor)), dim=0).item())
        restart_map[anchor_index] = (restart_steps, restart_times, restart_max)

    raw_step_pairs: list[tuple[float, float]] = []
    for index in range(len(working_sigmas) - 1):
        sigma_current = float(working_sigmas[index].item())
        sigma_next = float(working_sigmas[index + 1].item())
        raw_step_pairs.append((sigma_current, sigma_next))
        restart_spec = restart_map.get(index + 1)
        if restart_spec is None:
            continue
        restart_steps, restart_times, restart_max = restart_spec
        restart_min_index = index + 1
        restart_max_index = int(torch.argmin(torch.abs(working_sigmas - restart_max), dim=0).item())
        if restart_max_index >= restart_min_index:
            raise RuntimeError(
                "Restart planner requires restart_max to resolve to an earlier higher-sigma anchor "
                f"(restart_max={restart_max}, min_index={restart_min_index}, max_index={restart_max_index})."
            )
        restart_sigmas = get_sigmas_karras(
            restart_steps,
            float(working_sigmas[restart_min_index].item()),
            float(working_sigmas[restart_max_index].item()),
            device=working_sigmas.device,
            dtype=working_sigmas.dtype,
        )[:-1]
        for _ in range(restart_times):
            raw_step_pairs.extend(
                (float(cur.item()), float(nxt.item()))
                for cur, nxt in zip(restart_sigmas[:-1], restart_sigmas[1:])
            )

    step_plan: list[RestartStepPlan] = []
    last_sigma: float | None = None
    for sigma_current, sigma_next in raw_step_pairs:
        if sigma_current < 0.0 or sigma_next < 0.0:
            raise RuntimeError(
                "Restart planner requires non-negative sigma values "
                f"(sigma_current={sigma_current}, sigma_next={sigma_next})."
            )
        renoise_scale = 0.0
        if last_sigma is not None and last_sigma < sigma_current:
            renoise_scale_sq = sigma_current**2 - last_sigma**2
            if renoise_scale_sq < -1e-8:
                raise RuntimeError(
                    "Restart planner produced a negative renoise variance "
                    f"(sigma_current={sigma_current}, last_sigma={last_sigma})."
                )
            renoise_scale = math.sqrt(max(renoise_scale_sq, 0.0))
        step_plan.append(
            RestartStepPlan(
                sigma_current=sigma_current,
                sigma_next=sigma_next,
                renoise_scale=renoise_scale,
            )
        )
        last_sigma = sigma_next
    return step_plan


@torch.no_grad()
def restart_sampler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    restart_list=None,
    noise_sampler: Callable[[torch.Tensor], torch.Tensor] | None = None,
):
    """Restart sampling (Restart Sampling for Improving Generative Processes, 2023).

    Optionally inserts restart segments built with Karras sigmas, applies Heun/Euler steps, and injects
    noise between segments. Parameter semantics match the runtime config surface while using the shared
    pure restart planner in this module.
    """

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = (lambda reference: torch.randn_like(reference)) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    step_id = 0

    def _heun_step(x_in: torch.Tensor, sigma_cur: torch.Tensor, sigma_next: torch.Tensor, *, second_order: bool = True):
        nonlocal step_id
        denoised = model(x_in, sigma_cur * s_in, **extra_args)
        d = to_d(x_in, sigma_cur, denoised)
        if callback is not None:
            callback({"x": x_in, "i": step_id, "sigma": sigma_next, "sigma_hat": sigma_cur, "denoised": denoised})
        dt = sigma_next - sigma_cur
        if sigma_next == 0 or not second_order:
            x_out = x_in + d * dt
        else:
            x_euler = x_in + d * dt
            denoised_2 = model(x_euler, sigma_next * s_in, **extra_args)
            d_2 = to_d(x_euler, sigma_next, denoised_2)
            x_out = x_in + 0.5 * (d + d_2) * dt
        step_id += 1
        return x_out

    step_plan = build_restart_step_plan(sigmas, restart_list=restart_list)
    for step_id_plan in trange(len(step_plan), disable=disable):
        plan = step_plan[step_id_plan]
        if plan.renoise_scale > 0.0:
            x = x + noise_sampler(x) * s_noise * plan.renoise_scale
        sigma_current = torch.tensor(plan.sigma_current, device=x.device, dtype=x.dtype)
        sigma_next = torch.tensor(plan.sigma_next, device=x.device, dtype=x.dtype)
        x = _heun_step(x, sigma_current, sigma_next)
    return x


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


def generic_step_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, step_function=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        x = step_function(x / torch.sqrt(1.0 + sigmas[i] ** 2.0), sigmas[i], sigmas[i + 1], (x - denoised) / sigmas[i], noise_sampler)
        if sigmas[i + 1] != 0:
            x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2.0)
    return x


def DDPMSampler_step(x, sigma, sigma_prev, noise, noise_sampler):
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
    alpha = (alpha_cumprod / alpha_cumprod_prev)

    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        mu += ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise_sampler(sigma, sigma_prev)
    return mu


@torch.no_grad()
def sample_ddpm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    return generic_step_sampler(model, x, sigmas, extra_args, callback, disable, noise_sampler, DDPMSampler_step)
