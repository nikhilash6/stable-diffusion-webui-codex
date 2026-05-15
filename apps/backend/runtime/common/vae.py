"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared Flow-family VAE utilities (Flow16 + FLUX.2 32-channel AutoencoderKL variants).
Defines the canonical Flow16 config parity used by diffusers (no quant/post-quant conv) plus the FLUX.2 AutoencoderKLFlux2 config contract,
with helpers to locate and load those VAEs from either a diffusers directory or a single weights file with device-aware checkpoint loading paths.

Symbols (top-level; keep in sync; no ghosts):
- `FLOW16_VAE_CONFIG` (constant): Canonical diffusers-like config dict for Flow16 VAEs (16 latent channels, scaling/shift factors).
- `FLUX2_VAE_CONFIG` (constant): Canonical diffusers-like config dict for FLUX.2 AutoencoderKLFlux2 (32 latent channels + patch BN).
- `prepare_external_vae_override_state_dict` (function): Validates incoming VAE key names plus allowed SDXL/Flow16 metadata before engine-side override lane handling.
- `load_flow16_vae` (function): Loads a Flow16 VAE from a directory or weights file with strict latent-channel validation and device-aware state-dict ingestion.
- `load_flux2_vae` (function): Loads a FLUX.2 AutoencoderKLFlux2 from a directory or weights file with strict 32-channel + BN contract validation.
- `find_flow16_vae` (function): Searches candidate directories for a valid Flow16 VAE path.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import os
from typing import Mapping, Optional

import torch

from apps.backend.infra.config.vae_layout_lane import VaeLayoutLane
from apps.backend.runtime.common.vae_lane_policy import (
    detect_vae_layout,
    resolve_vae_layout_lane,
    validate_vae_key_names,
    uses_ldm_native_lane,
)
from apps.backend.runtime.common.vae_ldm import AutoencoderKL_LDM, sanitize_ldm_vae_config
from apps.backend.runtime.model_registry.specs import ModelFamily

logger = get_backend_logger("backend.runtime.common.vae")


# Configuration for 16-channel flow-based VAE (used by Flux, Z Image)
# NOTE: Flow16 VAE config mirrors the canonical diffusers configs shipped for:
# - `apps/backend/huggingface/black-forest-labs/FLUX.1-dev/vae/config.json`
# - `apps/backend/huggingface/Tongyi-MAI/Z-Image-Turbo/vae/config.json`
#
# In particular: these VAEs disable quant/post-quant convs (`use_quant_conv=false`)
# so the weight files may legitimately omit `quant_conv.*` and `post_quant_conv.*`.
FLOW16_VAE_CONFIG = {
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 16,  # 16-channel latent space
    "latents_mean": None,
    "latents_std": None,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 1024,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
    "use_post_quant_conv": False,
    "use_quant_conv": False,
}

FLUX2_VAE_CONFIG = {
    "act_fn": "silu",
    "batch_norm_eps": 0.0001,
    "batch_norm_momentum": 0.1,
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 32,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "patch_size": [2, 2],
    "sample_size": 1024,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
    "use_post_quant_conv": True,
    "use_quant_conv": True,
}


def _validate_flux2_vae_contract(vae: object, *, vae_path: str) -> None:
    cfg = getattr(vae, "config", None)
    latent_channels = int(getattr(cfg, "latent_channels", 0) or 0)
    if latent_channels != 32:
        raise ValueError(
            f"Incompatible FLUX.2 VAE at {vae_path}: expected latent_channels=32, got {latent_channels}."
        )
    patch_size = getattr(cfg, "patch_size", None)
    if list(patch_size) != [2, 2]:
        raise ValueError(
            f"Incompatible FLUX.2 VAE at {vae_path}: expected patch_size=[2, 2], got {patch_size!r}."
        )
    bn = getattr(vae, "bn", None)
    if bn is None:
        raise ValueError(f"Incompatible FLUX.2 VAE at {vae_path}: missing required batch-norm module `bn`.")
    running_mean = getattr(bn, "running_mean", None)
    running_var = getattr(bn, "running_var", None)
    if not isinstance(running_mean, torch.Tensor) or int(running_mean.numel()) != 128:
        raise ValueError(
            f"Incompatible FLUX.2 VAE at {vae_path}: expected bn.running_mean with 128 patch channels."
        )
    if not isinstance(running_var, torch.Tensor) or int(running_var.numel()) != 128:
        raise ValueError(
            f"Incompatible FLUX.2 VAE at {vae_path}: expected bn.running_var with 128 patch channels."
        )


def prepare_external_vae_override_state_dict(
    *,
    state_dict: dict[str, object],
    family: ModelFamily,
) -> Mapping[str, object]:
    """Validate an external VAE override enough for lane-specific engine handling.

    This helper is intentionally narrower than the full SDXL/Flow16 keyspace resolver:
    - always rejects any wrapper-prefix rewrite attempt;
    - for SDXL/Flow16 families, always drops only the known training metadata keys
      and fails loud on unknown non-weight keys;
    - leaves lane-specific keyspace mapping to the caller after lane resolution.
    """

    prepared: Mapping[str, object] = validate_vae_key_names(state_dict)
    if family in (
        ModelFamily.SDXL,
        ModelFamily.SDXL_REFINER,
        ModelFamily.FLUX,
        ModelFamily.FLUX_KONTEXT,
        ModelFamily.ZIMAGE,
    ):
        from apps.backend.runtime.state_dict.keymap_sdxl_vae import strip_known_sdxl_vae_metadata

        prepared = strip_known_sdxl_vae_metadata(prepared)  # type: ignore[arg-type]
    return prepared


def load_flux2_vae(
    vae_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
) -> object:
    """Load a FLUX.2 AutoencoderKLFlux2 from a path with strict 32-channel validation."""
    from diffusers import AutoencoderKLFlux2

    logger.info("Loading FLUX.2 VAE from: %s", vae_path)

    try:
        if os.path.isdir(vae_path):
            vae = AutoencoderKLFlux2.from_pretrained(vae_path, torch_dtype=dtype)
        else:
            suffix = os.path.splitext(vae_path)[1].lower()
            if suffix == ".gguf":
                from apps.backend.runtime.checkpoint.io import load_gguf_state_dict

                state_dict = load_gguf_state_dict(
                    vae_path,
                    dequantize=True,
                    computation_dtype=dtype,
                    device=device,
                )
            else:
                from apps.backend.runtime.checkpoint.io import load_torch_file

                state_dict = load_torch_file(vae_path, device=device)

            state_dict = validate_vae_key_names(state_dict)
            vae = AutoencoderKLFlux2.from_config(FLUX2_VAE_CONFIG)
            expected_total = len(vae.state_dict())
            missing, unexpected = vae.load_state_dict(state_dict, strict=False)

            if missing:
                ratio = len(missing) / max(expected_total, 1)
                if ratio > 0.05:
                    raise ValueError(
                        f"Incompatible FLUX.2 VAE at {vae_path}: missing {len(missing)}/{expected_total} keys. "
                        "Please supply a matching AutoencoderKLFlux2 weights file."
                    )
                logger.warning("FLUX.2 VAE missing keys (%d): %s", len(missing), missing[:5])
            if unexpected:
                logger.debug("FLUX.2 VAE unexpected keys (%d): %s", len(unexpected), unexpected[:5])

            vae = vae.to(dtype=dtype)

        _validate_flux2_vae_contract(vae, vae_path=vae_path)

        if device:
            vae = vae.to(device=device)

        param_count = sum(p.numel() for p in vae.parameters())
        logger.info("Loaded FLUX.2 VAE: %d params, dtype=%s", param_count, dtype)
        return vae
    except Exception as e:
        logger.error("Failed to load FLUX.2 VAE from %s: %s", vae_path, e)
        raise ValueError(f"Failed to load FLUX.2 VAE from {vae_path}: {e}") from e


def load_flow16_vae(
    vae_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
    family: ModelFamily = ModelFamily.ZIMAGE,
) -> object:
    """Load a 16-channel flow VAE from a path.
    
    This function handles loading from:
    - Single weights file (.safetensors/.gguf/.ckpt/.pt)
    - Diffusers directory format
    
    Args:
        vae_path: Path to VAE file or directory.
        dtype: Target dtype for the model.
        device: Target device (default: None = keep on CPU initially).
    
    Returns:
        Loaded AutoencoderKL model.
    
    Raises:
        ValueError: If loading fails.
    """
    from diffusers import AutoencoderKL
    
    logger.info("Loading Flow16 VAE from: %s", vae_path)

    try:
        if os.path.isdir(vae_path):
            # Diffusers directory format
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
        else:
            suffix = os.path.splitext(vae_path)[1].lower()
            if suffix == ".gguf":
                # VAE GGUFs are small enough to safely dequantize upfront; this also avoids
                # requiring quantized Conv2d ops in the VAE graph.
                from apps.backend.runtime.checkpoint.io import load_gguf_state_dict

                state_dict = load_gguf_state_dict(
                    vae_path,
                    dequantize=True,
                    computation_dtype=dtype,
                    device=device,
                )
            else:
                # Single-file weights (safetensors/ckpt/pt)
                from apps.backend.runtime.checkpoint.io import load_torch_file

                state_dict = load_torch_file(vae_path, device=device)

            state_dict = validate_vae_key_names(state_dict)
            vae_layout = detect_vae_layout(state_dict)
            vae_lane = resolve_vae_layout_lane(family=family, layout=vae_layout)
            using_ldm_native = uses_ldm_native_lane(vae_lane)

            if using_ldm_native:
                vae = AutoencoderKL_LDM.from_config(sanitize_ldm_vae_config(FLOW16_VAE_CONFIG))
            else:
                # Resolve LDM-style Flow16 VAE keyspace into diffusers lookup semantics when diffusers lane is active.
                from apps.backend.runtime.state_dict.keymap_sdxl_vae import resolve_sdxl_vae_keyspace

                state_dict = resolve_sdxl_vae_keyspace(state_dict).view
                vae = AutoencoderKL.from_config(FLOW16_VAE_CONFIG)

            expected_total = len(vae.state_dict())
            missing, unexpected = vae.load_state_dict(state_dict, strict=False)
            
            if missing:
                logger.warning("VAE missing keys (%d): %s", len(missing), missing[:5])
            if unexpected:
                logger.debug("VAE unexpected keys (%d): %s", len(unexpected), unexpected[:5])

            # Fail loudly if this is not actually a Flow16 VAE.
            # A mismatched 4-channel VAE will otherwise decode pure noise.
            if missing:
                ratio = len(missing) / max(expected_total, 1)
                if ratio > 0.05:
                    raise ValueError(
                        f"Incompatible Flow16 VAE at {vae_path}: missing {len(missing)}/{expected_total} keys "
                        f"(lane={vae_lane.value}, layout={vae_layout}). Please supply a matching 16-channel Flow VAE."
                    )
            
            vae = vae.to(dtype=dtype)
        
        if device:
            vae = vae.to(device=device)
        
        param_count = sum(p.numel() for p in vae.parameters())
        logger.info("Loaded Flow16 VAE: %d params, dtype=%s", param_count, dtype)
        return vae
        
    except Exception as e:
        logger.error("Failed to load Flow16 VAE from %s: %s", vae_path, e)
        raise ValueError(f"Failed to load VAE from {vae_path}: {e}") from e


def find_flow16_vae(search_paths: list[str]) -> Optional[str]:
    """Find a Flow16 VAE in the given search paths.
    
    Args:
        search_paths: List of paths to search (files or directories).
    
    Returns:
        Path to VAE if found, None otherwise.
    """
    for path in search_paths:
        if not path:
            continue
            
        if os.path.isdir(path):
            # Check for diffusers-format VAE directory
            if os.path.exists(os.path.join(path, "config.json")):
                logger.info("Found VAE directory: %s", path)
                return path
            
            # Check for safetensors files
            for f in os.listdir(path):
                if f.endswith(".safetensors") or f.endswith(".gguf"):
                    vae_path = os.path.join(path, f)
                    logger.info("Found VAE file: %s", vae_path)
                    return vae_path
                    
        elif os.path.isfile(path) and (path.endswith(".safetensors") or path.endswith(".gguf")):
            logger.info("Found VAE file: %s", path)
            return path
    
    return None


__all__ = [
    "FLOW16_VAE_CONFIG",
    "FLUX2_VAE_CONFIG",
    "prepare_external_vae_override_state_dict",
    "load_flux2_vae",
    "load_flow16_vae",
    "find_flow16_vae",
]
