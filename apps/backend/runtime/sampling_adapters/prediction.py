"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Prediction adapters and helpers (FlowMatch / EDM variants) used by schedulers/samplers.
Provides prediction modules that transform model outputs into the form expected by different sigma schedules (including FlowMatch Euler),
plus helper utilities for beta schedules and SNR/sigma rescaling. Flow-match helpers keep shift-aware discrete-flow semantics:
`percent_to_sigma` applies the same shift transform used by the schedule, and discrete-flow predictors support an explicit timestep multiplier.
This module is Diffusers-free and uses Codex-native flow-shift helpers.

Symbols (top-level; keep in sync; no ghosts):
- `betas_for_alpha_bar` (function): Builds beta schedule from an alpha-bar function (discretized diffusion schedule helper).
- `make_beta_schedule` (function): Creates standard beta schedules (linear/cosine/sqrt variants) as tensors.
- `time_snr_shift` (function): Applies an SNR-based time shift parameterization.
- `rescale_zero_terminal_snr_sigmas` (function): Rescales sigmas for “zero terminal SNR” behavior.
- `SIMPLE_SCHEDULE_MODE_TAIL_DOWNSAMPLE_SIGMAS` (constant): Predictor opt-in for SIMPLE sigma downsample from the predictor ladder tail.
- `SIMPLE_SCHEDULE_MODE_FLOWMATCH_SHIFTED_LINSPACE` (constant): Predictor default SIMPLE mode (legacy shifted-linspace behavior).
- `FLOW_SIMPLE_SCHEDULE_MODES` (constant): Allowed SIMPLE schedule mode values for flow predictors.
- `AbstractPrediction` (class): Base prediction module interface (torch.nn.Module) for mapping model output → prediction.
- `Prediction` (class): Standard prediction implementation (inherits `AbstractPrediction`).
- `PredictionEDM` (class): EDM-style prediction wrapper (inherits `Prediction`).
- `PredictionContinuousEDM` (class): Continuous EDM prediction base (sigma-continuous mapping).
- `PredictionContinuousV` (class): Continuous V-prediction implementation (inherits continuous EDM base).
- `PredictionFlow` (class): Flow prediction implementation (for flow-matching style training/inference).
- `PredictionDiscreteFlow` (class): Discrete flow prediction implementation (alias of `PredictionFlow`; multiplier support).
- `FlowMatchEulerPrediction` (class): FlowMatch Euler discrete prediction module (supports explicit shift/alpha and seq-len derived shift).
- `prediction_from_diffusers_scheduler` (function): Builds the appropriate prediction module from a diffusers scheduler instance.
"""

import math
import torch
import numpy as np

from apps.backend.runtime.model_registry.flow_shift import calculate_shift


SIMPLE_SCHEDULE_MODE_TAIL_DOWNSAMPLE_SIGMAS = "tail_downsample_sigmas"
SIMPLE_SCHEDULE_MODE_FLOWMATCH_SHIFTED_LINSPACE = "flowmatch_shifted_linspace"
FLOW_SIMPLE_SCHEDULE_MODES = frozenset(
    {
        SIMPLE_SCHEDULE_MODE_TAIL_DOWNSAMPLE_SIGMAS,
        SIMPLE_SCHEDULE_MODE_FLOWMATCH_SHIFTED_LINSPACE,
    }
)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clamp(betas, min=0, max=0.999)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def rescale_zero_terminal_snr_sigmas(sigmas):
    alphas_cumprod = 1 / ((sigmas * sigmas) + 1)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= (alphas_bar_sqrt_T)

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas_bar[-1] = 4.8973451890853435e-08
    return ((1 - alphas_bar) / alphas_bar) ** 0.5


class AbstractPrediction(torch.nn.Module):
    def __init__(self, sigma_data=1.0, prediction_type='epsilon'):
        super().__init__()
        self.sigma_data = sigma_data
        self.prediction_type = prediction_type
        assert self.prediction_type in ['epsilon', 'const', 'v_prediction', 'edm']

    def calculate_input(self, sigma, noise):
        if self.prediction_type == 'const':
            return noise
        else:
            sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
            return noise / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        if self.prediction_type == 'v_prediction':
            return model_input * self.sigma_data ** 2 / (
                    sigma ** 2 + self.sigma_data ** 2) - model_output * sigma * self.sigma_data / (
                    sigma ** 2 + self.sigma_data ** 2) ** 0.5
        elif self.prediction_type == 'edm':
            return model_input * self.sigma_data ** 2 / (
                    sigma ** 2 + self.sigma_data ** 2) + model_output * sigma * self.sigma_data / (
                    sigma ** 2 + self.sigma_data ** 2) ** 0.5
        else:
            return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        if self.prediction_type == 'const':
            return sigma * noise + (1.0 - sigma) * latent_image
        else:
            if max_denoise:
                noise = noise * torch.sqrt(1.0 + sigma ** 2.0)
            else:
                noise = noise * sigma

            noise += latent_image
            return noise

    def inverse_noise_scaling(self, sigma, latent):
        if self.prediction_type == 'const':
            return latent / (1.0 - sigma)
        else:
            return latent


class Prediction(AbstractPrediction):
    def __init__(self, sigma_data=1.0, prediction_type='eps', beta_schedule='linear', linear_start=0.00085,
                 linear_end=0.012, timesteps=1000):
        super().__init__(sigma_data=sigma_data, prediction_type=prediction_type)
        self.register_schedule(given_betas=None, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=8e-3)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5

        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('sigmas', sigmas.float())
        self.register_buffer('log_sigmas', sigmas.log().float())
        return

    def set_sigmas(self, sigmas):
        self.register_buffer('sigmas', sigmas.float())
        self.register_buffer('log_sigmas', sigmas.log().float())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep):
        t = torch.clamp(timestep.float().to(self.log_sigmas.device), min=0, max=(len(self.sigmas) - 1))
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent * 999.0)).item()


class PredictionEDM(Prediction):
    def timestep(self, sigma):
        return 0.25 * sigma.log()

    def sigma(self, timestep):
        return (timestep / 0.25).exp()


class PredictionContinuousEDM(AbstractPrediction):
    def __init__(self, sigma_data=1.0, prediction_type='eps', sigma_min=0.002, sigma_max=120.0):
        super().__init__(sigma_data=sigma_data, prediction_type=prediction_type)
        self.set_parameters(sigma_min, sigma_max, sigma_data)

    def set_parameters(self, sigma_min, sigma_max, sigma_data):
        self.sigma_data = sigma_data
        sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), 1000).exp()

        self.register_buffer('sigmas', sigmas)
        self.register_buffer('log_sigmas', sigmas.log())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return 0.25 * sigma.log()

    def sigma(self, timestep):
        return (timestep / 0.25).exp()

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent

        log_sigma_min = math.log(self.sigma_min)
        return math.exp((math.log(self.sigma_max) - log_sigma_min) * percent + log_sigma_min)


class PredictionContinuousV(PredictionContinuousEDM):
    def timestep(self, sigma):
        return sigma.atan() / math.pi * 2

    def sigma(self, timestep):
        return (timestep * math.pi / 2).tan()


class PredictionFlow(AbstractPrediction):
    def __init__(
        self,
        sigma_data: float = 1.0,
        prediction_type: str = "const",
        *,
        shift: float = 1.0,
        multiplier: float = 1000.0,
        timesteps: int = 1000,
        simple_schedule_mode: str | None = None,
    ):
        super().__init__(sigma_data=sigma_data, prediction_type=prediction_type)
        self.shift = float(shift)
        self.multiplier = float(multiplier)
        if simple_schedule_mode is None:
            # Default behavior: keep legacy SIMPLE schedule semantics (FlowMatch-style shifted-linspace).
            # The sigma scheduler opts-in to tail-downsample SIMPLE scheduling only when set to `tail_downsample_sigmas`.
            self.simple_schedule_mode = SIMPLE_SCHEDULE_MODE_FLOWMATCH_SHIFTED_LINSPACE
        else:
            value = str(simple_schedule_mode).strip().lower()
            if value not in FLOW_SIMPLE_SCHEDULE_MODES:
                raise ValueError(
                    "Invalid simple_schedule_mode for flow predictor: "
                    f"{simple_schedule_mode!r} "
                    f"(expected one of {sorted(FLOW_SIMPLE_SCHEDULE_MODES)})."
                )
            self.simple_schedule_mode = value
        if timesteps <= 0:
            raise ValueError("timesteps must be >= 1")
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / float(timesteps)) * self.multiplier)
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return time_snr_shift(self.shift, 1.0 - percent)


class PredictionDiscreteFlow(PredictionFlow):
    """Discrete flow predictor with explicit shift-aware timestep scaling.

    This wrapper exists for explicit naming; the implementation is
    identical to `PredictionFlow` (shift + multiplier-based timestep mapping).
    """

    def __init__(
        self,
        sigma_data: float = 1.0,
        prediction_type: str = "const",
        *,
        shift: float = 1.0,
        timesteps: int = 1000,
        multiplier: float = 1000.0,
        simple_schedule_mode: str | None = None,
    ):
        super().__init__(
            sigma_data=sigma_data,
            prediction_type=prediction_type,
            shift=shift,
            multiplier=multiplier,
            timesteps=timesteps,
            simple_schedule_mode=simple_schedule_mode,
        )


class FlowMatchEulerPrediction(AbstractPrediction):
    def __init__(
        self,
        seq_len=4096,
        base_seq_len=256,
        max_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
        pseudo_timestep_range=10000,
        shift=None,
        time_shift_type="exponential",
    ):
        super().__init__(sigma_data=1.0, prediction_type='const')
        # Credits: semantics match diffusers FlowMatchEulerDiscreteScheduler (sigma=1) and kohya-ss sd-scripts reference loops.
        self.mu = None
        self.shift = None
        self.pseudo_timestep_range = pseudo_timestep_range
        self.apply_mu_transform(
            seq_len=seq_len,
            base_seq_len=base_seq_len,
            max_seq_len=max_seq_len,
            base_shift=base_shift,
            max_shift=max_shift,
            shift=shift,
            time_shift_type=time_shift_type,
        )

    def apply_mu_transform(
        self,
        seq_len=4096,
        base_seq_len=256,
        max_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
        shift=None,
        time_shift_type="exponential",
    ):
        # TODO: Add an UI option to let user choose whether to call this in each generation to bind latent size to sigmas
        # And some cases may want their own mu values or other parameters
        if shift is None:
            mu = calculate_shift(
                image_seq_len=seq_len,
                base_seq_len=base_seq_len,
                max_seq_len=max_seq_len,
                base_shift=base_shift,
                max_shift=max_shift,
            )
            self.mu = float(mu)

            kind = str(time_shift_type or "exponential").strip().lower()
            if kind == "linear":
                self.shift = float(mu)
            elif kind == "exponential":
                self.shift = float(math.exp(float(mu)))
            else:
                raise ValueError(f"Invalid time_shift_type={time_shift_type!r} (expected 'exponential' or 'linear')")
        else:
            self.mu = None
            self.shift = float(shift)
        if self.shift <= 0:
            raise ValueError("shift must be > 0")
        sigmas = torch.arange(1, self.pseudo_timestep_range + 1, 1) / self.pseudo_timestep_range
        # Apply the effective shift (alpha) using the standard rational shift form.
        sigmas = time_snr_shift(self.shift, sigmas)
        self.register_buffer('sigmas', sigmas)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        # timestep is expected to be in range [0, pseudo_timestep_range-1] as indices
        # Map to actual sigma values from the pre-computed buffer
        if isinstance(timestep, torch.Tensor):
            # Clamp and convert to indices
            idx = timestep.long().clamp(0, self.pseudo_timestep_range - 1)
            # Move sigmas to same device as index
            sigmas = self.sigmas.to(idx.device)
            return sigmas[idx]
        else:
            idx = int(max(0, min(timestep, self.pseudo_timestep_range - 1)))
            return self.sigmas[idx]

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return time_snr_shift(float(self.shift), 1.0 - percent)


def prediction_from_diffusers_scheduler(scheduler):
    cfg = getattr(scheduler, "config", None)
    if cfg is None:
        raise NotImplementedError(f"Failed to recognize {scheduler}")

    pred_type = getattr(cfg, "prediction_type", None)
    if isinstance(pred_type, str):
        pred_type = pred_type.lower()

    beta_schedule = getattr(cfg, "beta_schedule", None)
    if pred_type in ["epsilon", "v_prediction"] and beta_schedule in ("scaled_linear", "linear"):
        sigma_data = getattr(cfg, "sigma_data", None)
        try:
            sigma_data_value = float(sigma_data) if sigma_data is not None else 1.0
        except Exception:  # noqa: BLE001 - defensive parsing
            sigma_data_value = 1.0

        beta_start = getattr(cfg, "beta_start", 0.00085)
        beta_end = getattr(cfg, "beta_end", 0.012)
        timesteps = getattr(cfg, "num_train_timesteps", 1000)

        return Prediction(
            sigma_data=sigma_data_value,
            prediction_type=pred_type,
            beta_schedule="linear",
            linear_start=beta_start,
            linear_end=beta_end,
            timesteps=timesteps,
        )

    raise NotImplementedError(f"Failed to recognize {scheduler}")
