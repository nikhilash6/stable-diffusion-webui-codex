"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Standalone Z Image sampler using Diffusers scheduler math.
Runs the Z-Image transformer directly while following `FlowMatchEulerDiscreteScheduler` semantics (including `shift` and terminal sigma handling) without the native sampler driver stack.

Symbols (top-level; keep in sync; no ghosts):
- `_default_zimage_sampler_device` (function): Resolves default sampler device identity from memory-manager mount authority.
- `sample_zimage_diffusers_math` (function): Samples Z Image latents using Diffusers math/scheduler steps with optional classic CFG
  (positive/negative embeddings) and explicit `shift` selection (no double-negation of the model output).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from apps.backend.runtime.memory import memory_management

logger = get_backend_logger("backend.zimage.standalone")


def _default_zimage_sampler_device() -> str:
    mount_device = memory_management.manager.mount_device()
    if not hasattr(mount_device, "type"):
        raise RuntimeError(
            "ZImage standalone sampler requires memory manager mount_device() to return torch.device."
        )
    return str(mount_device)


def sample_zimage_diffusers_math(
    transformer: torch.nn.Module,
    text_embeddings: torch.Tensor,
    negative_text_embeddings: Optional[torch.Tensor] = None,
    *,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 9,
    guidance_scale: float = 0.0,
    shift: float = 3.0,
    generator: Optional[torch.Generator] = None,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    latent_channels: int = 16,
    patch_size: int = 2,
) -> torch.Tensor:
    """Sample using Diffusers FlowMatchEulerDiscreteScheduler.
    
    This is a standalone sampler that:
    - Uses OUR transformer directly (GGUF-loaded ZImageTransformer2DModel)
    - Uses OUR text embeddings (from ZImageTextEncoder)
    - Uses Diffusers scheduler for timestep/sigma management
    - Matches diffusers schedule semantics (shift + sigma_min=0)
    
    Args:
        transformer: Our ZImageTransformer2DModel
        text_embeddings: Pre-encoded text embeddings from our encoder [B, seq, hidden]
        negative_text_embeddings: Negative prompt embeddings for CFG [B, seq, hidden] (required when guidance_scale > 1)
        height: Image height
        width: Image width
        num_inference_steps: Sampling steps (diffusers ZImagePipeline recommends 9 by default)
        guidance_scale: CFG scale (classic CFG; enabled when > 1)
        shift: Flow-match schedule shift (Turbo=3.0, Base=6.0)
        generator: Optional RNG generator
        device: Target device
        dtype: Computation dtype
        latent_channels: Number of latent channels (16)
        patch_size: Patch size (2)
        
    Returns:
        latents: Final denoised latents [B, C, H//8, W//8]
    """
    device_name = str(device or _default_zimage_sampler_device())
    batch_size = text_embeddings.shape[0]
    apply_cfg = guidance_scale > 1.0
    if apply_cfg:
        if negative_text_embeddings is None:
            raise ValueError("negative_text_embeddings is required when guidance_scale > 1")
        if tuple(negative_text_embeddings.shape) != tuple(text_embeddings.shape):
            raise ValueError(
                f"negative_text_embeddings shape {tuple(negative_text_embeddings.shape)} does not match "
                f"text_embeddings shape {tuple(text_embeddings.shape)}"
            )
    
    # Calculate latent dimensions (VAE downscale = 8)
    vae_scale = 8
    latent_height = height // vae_scale
    latent_width = width // vae_scale
    
    # Create scheduler (HF scheduler_config.json: use_dynamic_shifting=false, shift depends on model variant).
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=float(shift),
    )
    # Match diffusers ZImagePipeline: force a terminal sigma of 0.0 (double-zero tail after set_timesteps).
    scheduler.sigma_min = 0.0
    scheduler.set_timesteps(num_inference_steps, device=device_name)
    timesteps = scheduler.timesteps
    
    logger.debug(
        "[diffusers-sampler] steps=%d, timesteps=%s",
        num_inference_steps, 
        [round(float(t), 3) for t in timesteps[:4].tolist()]
    )
    
    # Initialize latents
    latents_shape = (batch_size, latent_channels, latent_height, latent_width)
    latents = torch.randn(
        latents_shape,
        generator=generator,
        device=device_name,
        dtype=torch.float32,  # Scheduler expects float32
    )
    
    # Flow-matching starts with pure noise (sigma=1), no scaling needed
    
    logger.debug("[diffusers-sampler] latents shape=%s", latents.shape)
    
    # Sampling loop
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # Prepare model input
            latent_model_input = latents.to(dtype)
            
            # Call our transformer
            # Our transformer expects `sigma` in [1→0] (1=start/noise, 0=end/clean).
            # It internally uses `t_inv = 1 - sigma` and applies `t_scale` (1000.0) to match diffusers.
            #
            # Scheduler returns timesteps t = sigma * 1000 (1000=start → 0=end), so sigma = t/1000.
            sigma = float(t) / 1000.0
            if apply_cfg:
                sigma_tensor = torch.full((batch_size * 2,), sigma, device=device_name, dtype=dtype)
                latent_model_input = latent_model_input.repeat(2, 1, 1, 1)
                context = torch.cat([negative_text_embeddings, text_embeddings], dim=0).to(dtype)
                model_output = transformer(
                    latent_model_input,
                    sigma_tensor,
                    context=context,
                )
                # Our Z-Image core returns the negated velocity (noise_pred) already.
                noise_pred = model_output.float()
                neg, pos = noise_pred.chunk(2, dim=0)
                noise_pred = pos + float(guidance_scale) * (pos - neg)
            else:
                sigma_tensor = torch.full((batch_size,), sigma, device=device, dtype=dtype)
                model_output = transformer(
                    latent_model_input,
                    sigma_tensor,
                    context=text_embeddings.to(dtype),
                )
                # Our Z-Image core returns the negated velocity (noise_pred) already.
                noise_pred = model_output.float()
            
            # Compute previous sample using scheduler
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # Log progress
            if i < 3 or i == len(timesteps) - 1:
                norm_x = float(latents.norm())
                norm_pred = float(noise_pred.norm())
                logger.debug(
                    "[diffusers-sampler] step=%d/%d t=%.3f norm(x)=%.2f norm(pred)=%.2f",
                    i + 1, len(timesteps), float(t), norm_x, norm_pred
                )
    
    return latents


def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents to images using VAE.
    
    Args:
        vae: Our VAE wrapper with .decode() and .first_stage_model
        latents: Latents [B, C, H, W] in normalized space
        
    Returns:
        images: Tensor [B, 3, H*8, W*8] in range [0, 1]
    """
    # Denormalize latents for VAE
    # VAE expects: (latents / scaling_factor) + shift_factor
    scaling_factor = 0.3611  # From flux VAE
    shift_factor = 0.1159
    
    latents = (latents / scaling_factor) + shift_factor
    
    # Decode
    images = vae.decode(latents)
    
    # Convert to [0, 1] range
    images = (images + 1.0) / 2.0
    images = images.clamp(0, 1)
    
    return images


__all__ = ["sample_zimage_diffusers_math", "decode_latents"]
