"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 flow-matching video sampler for 5D latents.
Implements a sigma schedule and Euler ODE integration with classifier-free guidance for WAN-style video latents shaped `[B, C, T, H, W]`,
plus a convenience helper that decodes sampled latents through a VAE.

Symbols (top-level; keep in sync; no ghosts):
- `WAN_FLOW_MULTIPLIER_DEFAULT` (constant): Default multiplier applied to sigma when constructing model timestep inputs.
- `_default_mount_device` (function): Resolves default sampler device from memory-manager mount-device authority.
- `get_flow_sigmas` (function): Builds a shifted sigma schedule for flow-matching from 1→0.
- `WanVideoSampler` (class): Sampler wrapper around the WAN transformer (CFG + Euler integration).
- `sample_txt2vid` (function): High-level txt2vid helper (samples latents then decodes via VAE).
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

import logging
from typing import Callable, Optional

import torch
from torch import nn

from apps.backend.runtime.memory import memory_management

logger = get_backend_logger("backend.runtime.wan22.sampler")

WAN_FLOW_MULTIPLIER_DEFAULT = 1000.0


def _default_mount_device() -> torch.device:
    mount_device = memory_management.manager.mount_device()
    if not isinstance(mount_device, torch.device):
        raise RuntimeError(
            "WAN22 sampler requires memory manager mount_device() to return torch.device "
            f"(got {type(mount_device).__name__})."
        )
    return mount_device


def get_flow_sigmas(
    num_steps: int,
    shift: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate sigma schedule for flow-matching.
    
    Uses shifted linear schedule (timesteps from 1.0 to 0.0).
    
    Args:
        num_steps: Number of sampling steps.
        shift: Flow shift parameter (mu) from the model's scheduler_config.json.
        device: Target device.
        dtype: Target dtype.
    
    Returns:
        Sigma tensor of shape [num_steps + 1] from sigma_max to 0.
    """
    resolved_device = _default_mount_device() if device is None else device
    # Simple linear schedule from 1.0 to 0.0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=resolved_device, dtype=dtype)
    
    # Apply shift (higher shift = more noise at start)
    sigmas = shift * timesteps / (1 + (shift - 1) * timesteps)
    
    return sigmas


class WanVideoSampler:
    """Flow-matching video sampler for WAN models.
    
    Implements Euler ODE integration with CFG for 5D video latents.
    """
    
    def __init__(
        self,
        transformer: nn.Module,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.transformer = transformer
        self.device = _default_mount_device() if device is None else device
        self.dtype = dtype
        self._logger = get_backend_logger(__name__)
    
    @torch.inference_mode()
    def sample(
        self,
        shape: tuple[int, ...],  # [B, C, T, H, W]
        *,
        cond: torch.Tensor,  # [B, L, D] positive conditioning
        uncond: Optional[torch.Tensor] = None,  # [B, L, D] negative conditioning
        num_steps: int = 20,
        cfg_scale: float = 7.5,
        flow_shift: float,
        flow_multiplier: float = WAN_FLOW_MULTIPLIER_DEFAULT,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """Sample video latents using flow-matching Euler integration.
        
        Args:
            shape: Output shape [B, C, T, H, W].
            cond: Positive text conditioning [B, L, D].
            uncond: Negative text conditioning (optional, uses zeros if None).
            num_steps: Number of sampling steps.
            cfg_scale: Classifier-free guidance scale.
            flow_shift: Flow shift parameter.
            flow_multiplier: Multiplier applied to the model timestep input (sigma -> timestep).
            seed: Random seed (optional).
            callback: Progress callback(step, total, latent).
        
        Returns:
            Sampled video latents [B, C, T, H, W].
        """
        B, C, T, H, W = shape
        device = self.device
        dtype = self.dtype
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Initialize with noise
        x = torch.randn(shape, device=device, dtype=dtype)
        
        # Create sigma schedule
        sigmas = get_flow_sigmas(num_steps, shift=flow_shift, device=device, dtype=torch.float32)
        
        # Handle uncond
        if uncond is None:
            uncond = torch.zeros_like(cond)
        
        emit_backend_message(
            "WAN sampling",
            logger=self._logger.name,
            shape=shape,
            steps=num_steps,
            cfg=cfg_scale,
            shift=flow_shift,
            multiplier=float(flow_multiplier),
        )
        
        # Euler ODE integration
        for i in range(num_steps):
            t = sigmas[i]
            t_next = sigmas[i + 1]
            
            # Model expects timestep-like scale (sigma -> sigma * multiplier).
            # Keep the sigma ladder itself in [0,1] for dt integration.
            timestep = torch.full((B,), float(t) * float(flow_multiplier), device=device, dtype=torch.float32)
            
            # CFG: run model twice (conditional + unconditional)
            # Batch both together for efficiency
            x_input = torch.cat([x, x], dim=0)
            cond_input = torch.cat([cond, uncond], dim=0)
            timestep_input = torch.cat([timestep, timestep], dim=0)
            
            # Model forward pass
            v_pred = self.transformer(
                x_input,
                timestep_input,
                cond_input,
            )
            
            # Split predictions
            v_cond, v_uncond = v_pred.chunk(2, dim=0)
            
            # Apply CFG
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            
            # Euler step: x_next = x + (t_next - t) * v
            dt = float(t_next) - float(t)
            x = x + dt * v
            
            # Progress callback
            if callback is not None:
                callback(i + 1, num_steps, x)
            
            if (i + 1) % 5 == 0 or i == 0:
                emit_backend_message(
                    "WAN sampling step",
                    logger=self._logger.name,
                    level=logging.DEBUG,
                    step=i + 1,
                    total=num_steps,
                    t=float(t),
                    t_next=float(t_next),
                    norm=float(x.norm()),
                )
        
        emit_backend_message("WAN sampling complete", logger=self._logger.name)
        return x


def sample_txt2vid(
    transformer: nn.Module,
    vae: nn.Module,
    *,
    cond: torch.Tensor,
    uncond: Optional[torch.Tensor] = None,
    width: int = 768,
    height: int = 432,
    num_frames: int = 16,
    num_steps: int = 20,
    cfg_scale: float = 7.5,
    flow_shift: float,
    flow_multiplier: float = WAN_FLOW_MULTIPLIER_DEFAULT,
    seed: Optional[int] = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
) -> torch.Tensor:
    """High-level txt2vid sampling function.
    
    Args:
        transformer: WanTransformer2DModel.
        vae: VAE decoder.
        cond: Text conditioning from T5.
        uncond: Negative conditioning (optional).
        width: Output width.
        height: Output height.
        num_frames: Number of frames.
        num_steps: Sampling steps.
        cfg_scale: CFG scale.
        flow_shift: Flow shift.
        flow_multiplier: Multiplier applied to the model timestep input (sigma -> timestep).
        seed: Random seed.
        device: Device.
        dtype: Dtype.
        callback: Progress callback.
    
    Returns:
        Decoded video tensor [B, C, T, H, W].
    """
    # Compute latent dimensions (WAN uses 4x compression with patch size 2,2)
    latent_h = height // 8
    latent_w = width // 8
    latent_c = 16  # WAN latent channels
    
    # Shape: [B, C, T, H, W]
    B = cond.shape[0]
    shape = (B, latent_c, num_frames, latent_h, latent_w)
    
    # Create sampler
    sampler = WanVideoSampler(
        transformer,
        device=(_default_mount_device() if device is None else device),
        dtype=dtype,
    )
    
    # Sample latents
    latents = sampler.sample(
        shape,
        cond=cond,
        uncond=uncond,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        flow_shift=flow_shift,
        flow_multiplier=flow_multiplier,
        seed=seed,
        callback=callback,
    )
    
    # Decode through VAE
    emit_backend_message("Decoding latents through VAE", logger=logger.name)
    with torch.inference_mode():
        video = vae.decode(latents)
    
    return video
