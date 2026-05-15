"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Legacy centered-latent FLUX.2 Klein bridge retained alongside the normalized external-latent runtime seam.
Keeps Codex sampler state in centered raw 32-channel latent BCHW space while adapting to the upstream FLUX.2 transformer/VAE
contract (normalized 128-channel 2x2-patch space packed as sequence tokens). The bridge is linear in sampler space:
noise init scales raw Gaussian noise through patch-space batch-norm std, model inputs divide by that std, and model
outputs are mapped back through the inverse patchification when this adapter is used directly.

Symbols (top-level; keep in sync; no ghosts):
- `FLUX2_LATENT_CHANNELS` (constant): Raw FLUX.2 VAE latent channels.
- `FLUX2_PATCH_CHANNELS` (constant): Patchified FLUX.2 transformer channels (`32 * 2 * 2 = 128`).
- `patchify_flux2_latents` (function): Convert raw 32-channel BCHW latents into 128-channel 2x2 patch space.
- `unpatchify_flux2_latents` (function): Convert 128-channel patch-space BCHW latents back into raw 32-channel space.
- `pack_flux2_latents` (function): Convert BCHW patch-space latents into `(B, HW, C)` token layout.
- `unpack_flux2_latents` (function): Convert `(B, HW, C)` token layout back into BCHW patch-space latents.
- `prepare_flux2_text_ids` (function): Build FLUX.2 4D text position ids `(B, S, 4)`.
- `prepare_flux2_latent_ids` (function): Build FLUX.2 4D image position ids `(B, HW, 4)` for patch-space latents.
- `flux2_patch_bn_stats` (function): Read and validate FLUX.2 VAE batch-norm stats for patch-space normalization.
- `flux2_decode_offset_raw` (function): Convert patch-space batch-norm mean into raw 32-channel latent offset.
- `encode_flux2_external_latents` (function): Encode pixels into centered raw 32-channel sampler latents (`raw - BN_mean_offset`).
- `decode_flux2_external_latents` (function): Decode centered raw 32-channel sampler latents by restoring the BN mean offset first.
- `Flux2NoisePrediction` (class): FlowMatch predictor with FLUX.2-specific raw-space noise initialization.
- `Flux2KleinTransformerAdapter` (class): Wrapper translating raw sampler latents to/from the upstream transformer contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from apps.backend.runtime.model_registry.specs import CodexCoreArchitecture
from apps.backend.runtime.sampling_adapters.prediction import FlowMatchEulerPrediction

FLUX2_LATENT_CHANNELS = 32
FLUX2_PATCH_CHANNELS = FLUX2_LATENT_CHANNELS * 4


def _unwrap_flux2_vae_model(vae_like: object) -> nn.Module:
    model = getattr(vae_like, "first_stage_model", vae_like)
    model = getattr(model, "_base", model)
    if not isinstance(model, nn.Module):
        raise RuntimeError(
            "FLUX.2 VAE helper expected a torch.nn.Module-compatible AutoencoderKLFlux2; "
            f"got {type(model).__name__}."
        )
    return model


def patchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise RuntimeError(f"FLUX.2 patchify expects BCHW latents; got shape={tuple(latents.shape)}.")
    batch_size, channels, height, width = latents.shape
    if int(channels) != FLUX2_LATENT_CHANNELS:
        raise RuntimeError(
            "FLUX.2 patchify expects 32 latent channels; "
            f"got channels={channels} shape={tuple(latents.shape)}."
        )
    if (height % 2) != 0 or (width % 2) != 0:
        raise RuntimeError(
            "FLUX.2 latents must have even spatial dimensions for 2x2 patch packing; "
            f"got shape={tuple(latents.shape)}."
        )
    latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch_size, channels * 4, height // 2, width // 2)


def unpatchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise RuntimeError(f"FLUX.2 unpatchify expects BCHW latents; got shape={tuple(latents.shape)}.")
    batch_size, channels, height, width = latents.shape
    if int(channels) != FLUX2_PATCH_CHANNELS:
        raise RuntimeError(
            "FLUX.2 unpatchify expects 128 patch channels; "
            f"got channels={channels} shape={tuple(latents.shape)}."
        )
    latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, channels // 4, height * 2, width * 2)


def pack_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise RuntimeError(f"FLUX.2 pack expects BCHW latents; got shape={tuple(latents.shape)}.")
    batch_size, channels, height, width = latents.shape
    return latents.reshape(batch_size, channels, height * width).permute(0, 2, 1)


def unpack_flux2_latents(latents: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if latents.ndim != 3:
        raise RuntimeError(f"FLUX.2 unpack expects BNC latents; got shape={tuple(latents.shape)}.")
    batch_size, seq_len, channels = latents.shape
    expected_seq_len = int(height) * int(width)
    if int(seq_len) != expected_seq_len:
        raise RuntimeError(
            "FLUX.2 unpack sequence mismatch: "
            f"got seq_len={seq_len}, expected {expected_seq_len} for height={height} width={width}."
        )
    return latents.permute(0, 2, 1).reshape(batch_size, channels, int(height), int(width))


def prepare_flux2_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
    if prompt_embeds.ndim != 3:
        raise RuntimeError(
            f"FLUX.2 text ids expect prompt_embeds shape (B,S,C); got {tuple(prompt_embeds.shape)}."
        )
    batch_size, seq_len, _ = prompt_embeds.shape
    ids = torch.zeros((seq_len, 4), device=prompt_embeds.device, dtype=torch.long)
    ids[:, 3] = torch.arange(seq_len, device=prompt_embeds.device, dtype=torch.long)
    return ids.unsqueeze(0).expand(batch_size, -1, -1)


def prepare_flux2_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise RuntimeError(f"FLUX.2 latent ids expect BCHW latents; got shape={tuple(latents.shape)}.")
    batch_size, _, height, width = latents.shape
    h_ids = torch.arange(height, device=latents.device, dtype=torch.long).view(height, 1).expand(height, width)
    w_ids = torch.arange(width, device=latents.device, dtype=torch.long).view(1, width).expand(height, width)
    ids = torch.zeros((height * width, 4), device=latents.device, dtype=torch.long)
    ids[:, 1] = h_ids.reshape(-1)
    ids[:, 2] = w_ids.reshape(-1)
    return ids.unsqueeze(0).expand(batch_size, -1, -1)


def flux2_patch_bn_stats(vae: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    bn = getattr(vae, "bn", None)
    cfg = getattr(vae, "config", None)
    if bn is None:
        raise RuntimeError("FLUX.2 VAE is missing the required `bn` BatchNorm2d module.")
    running_mean = getattr(bn, "running_mean", None)
    running_var = getattr(bn, "running_var", None)
    eps = getattr(cfg, "batch_norm_eps", None)
    if not isinstance(running_mean, torch.Tensor) or running_mean.ndim != 1:
        raise RuntimeError("FLUX.2 VAE batch-norm running_mean must be a 1D tensor.")
    if not isinstance(running_var, torch.Tensor) or running_var.ndim != 1:
        raise RuntimeError("FLUX.2 VAE batch-norm running_var must be a 1D tensor.")
    if int(running_mean.shape[0]) != FLUX2_PATCH_CHANNELS or int(running_var.shape[0]) != FLUX2_PATCH_CHANNELS:
        raise RuntimeError(
            "FLUX.2 VAE batch-norm channel contract mismatch. "
            f"Expected {FLUX2_PATCH_CHANNELS} patch channels, got mean={tuple(running_mean.shape)} var={tuple(running_var.shape)}."
        )
    if eps is None:
        raise RuntimeError("FLUX.2 VAE config is missing `batch_norm_eps`.")
    eps_value = float(eps)
    if eps_value <= 0.0:
        raise RuntimeError(f"FLUX.2 VAE batch_norm_eps must be > 0, got {eps_value}.")
    mean = running_mean.detach().float().view(1, FLUX2_PATCH_CHANNELS, 1, 1)
    std = torch.sqrt(running_var.detach().float().view(1, FLUX2_PATCH_CHANNELS, 1, 1) + eps_value)
    if torch.any(std <= 0):
        raise RuntimeError("FLUX.2 VAE batch-norm std contains non-positive values.")
    return mean, std


def flux2_decode_offset_raw(
    vae: nn.Module,
    *,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
    mean, _ = flux2_patch_bn_stats(vae)
    if height is None or width is None:
        return unpatchify_flux2_latents(mean)
    if int(height) <= 0 or int(width) <= 0:
        raise RuntimeError(f"FLUX.2 raw decode offset expects positive spatial size; got height={height} width={width}.")
    if (int(height) % 2) != 0 or (int(width) % 2) != 0:
        raise RuntimeError(
            "FLUX.2 raw decode offset expects even latent spatial dimensions; "
            f"got height={height} width={width}."
        )
    tiled_mean = mean.expand(1, FLUX2_PATCH_CHANNELS, int(height) // 2, int(width) // 2)
    return unpatchify_flux2_latents(tiled_mean)


def encode_flux2_external_latents(vae_like: object, pixel_samples_bhwc: torch.Tensor) -> torch.Tensor:
    if not hasattr(vae_like, "encode"):
        raise TypeError("FLUX.2 encode helper requires a VAE wrapper exposing `encode(...)`.")
    raw_latents = vae_like.encode(pixel_samples_bhwc)
    if not isinstance(raw_latents, torch.Tensor):
        raise RuntimeError(
            "FLUX.2 VAE encode returned invalid latent output; "
            f"got {type(raw_latents).__name__}."
        )
    if raw_latents.ndim != 4:
        raise RuntimeError(f"FLUX.2 VAE encode must return BCHW latents; got shape={getattr(raw_latents, 'shape', None)}.")
    offset = flux2_decode_offset_raw(
        _unwrap_flux2_vae_model(vae_like),
        height=int(raw_latents.shape[2]),
        width=int(raw_latents.shape[3]),
    ).to(
        device=raw_latents.device,
        dtype=raw_latents.dtype,
    )
    return raw_latents - offset


def decode_flux2_external_latents(vae_like: object, centered_latents: torch.Tensor) -> torch.Tensor:
    if not hasattr(vae_like, "decode"):
        raise TypeError("FLUX.2 decode helper requires a VAE wrapper exposing `decode(...)`.")
    if centered_latents.ndim != 4:
        raise RuntimeError(
            f"FLUX.2 decode helper expects centered BCHW latents; got shape={getattr(centered_latents, 'shape', None)}."
        )
    offset = flux2_decode_offset_raw(
        _unwrap_flux2_vae_model(vae_like),
        height=int(centered_latents.shape[2]),
        width=int(centered_latents.shape[3]),
    ).to(
        device=centered_latents.device,
        dtype=centered_latents.dtype,
    )
    raw_latents = centered_latents + offset
    return vae_like.decode(raw_latents)


class Flux2NoisePrediction(FlowMatchEulerPrediction):
    """FLUX.2 FlowMatch predictor with raw-space noise initialization."""

    def __init__(self, *, patch_std: torch.Tensor) -> None:
        if not isinstance(patch_std, torch.Tensor) or tuple(patch_std.shape) != (1, FLUX2_PATCH_CHANNELS, 1, 1):
            raise RuntimeError(
                "Flux2NoisePrediction requires patch_std shaped (1, 128, 1, 1); "
                f"got {getattr(patch_std, 'shape', None)}."
            )
        super().__init__(
            seq_len=4096,
            base_seq_len=256,
            max_seq_len=4096,
            base_shift=0.5,
            max_shift=1.15,
            pseudo_timestep_range=1000,
            time_shift_type="exponential",
        )
        self.register_buffer("_flux2_patch_std", patch_std.detach().float(), persistent=False)

    def noise_scaling(self, sigma, noise, latent_image, max_denoise: bool = False):
        del max_denoise
        if noise.ndim != 4:
            raise RuntimeError(f"FLUX.2 noise_scaling expects BCHW noise; got shape={tuple(noise.shape)}.")
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        patch_noise = patchify_flux2_latents(noise)
        patch_std = self._flux2_patch_std.to(device=noise.device, dtype=noise.dtype)
        scaled_noise = unpatchify_flux2_latents(patch_noise * patch_std)
        return sigma * scaled_noise + (1.0 - sigma) * latent_image


class Flux2KleinTransformerAdapter(nn.Module):
    """Wrap a diffusers FLUX.2 transformer so Codex samplers can operate in raw 32-channel latent space."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: nn.Module,
        storage_dtype: torch.dtype | str,
        computation_dtype: torch.dtype,
        load_device: torch.device,
        offload_device: torch.device,
        initial_device: torch.device,
    ) -> None:
        super().__init__()
        self.inner = transformer
        self._vae = vae
        self.storage_dtype = storage_dtype
        self.computation_dtype = computation_dtype
        self.load_device = load_device
        self.offload_device = offload_device
        self.initial_device = initial_device
        self.architecture = CodexCoreArchitecture.FLOW_TRANSFORMER
        self.codex_config = SimpleNamespace(context_dim=7680, adm_in_channels=None)
        self.lora_loader = None
        self.num_classes = None

        cfg = getattr(transformer, "config", None)
        if cfg is None:
            raise RuntimeError("FLUX.2 transformer is missing config metadata.")
        in_channels = int(getattr(cfg, "in_channels", 0) or 0)
        patch_size = int(getattr(cfg, "patch_size", 0) or 0)
        context_dim = int(getattr(cfg, "joint_attention_dim", 0) or 0)
        guidance_embeds = bool(getattr(cfg, "guidance_embeds", False))
        if in_channels != FLUX2_PATCH_CHANNELS or patch_size != 1 or context_dim != 7680 or guidance_embeds:
            raise RuntimeError(
                "Unsupported FLUX.2 transformer config for truthful Klein 4B/base-4B runtime. "
                f"in_channels={in_channels} patch_size={patch_size} context_dim={context_dim} guidance_embeds={guidance_embeds}."
            )

        patch_mean, patch_std = flux2_patch_bn_stats(vae)
        self.register_buffer("_patch_mean", patch_mean, persistent=False)
        self.register_buffer("_patch_std", patch_std, persistent=False)
        self.register_buffer("_decode_offset_raw", unpatchify_flux2_latents(patch_mean), persistent=False)

    @property
    def patch_std(self) -> torch.Tensor:
        return self._patch_std

    @property
    def decode_offset_raw(self) -> torch.Tensor:
        return self._decode_offset_raw

    @property
    def dtype(self) -> torch.dtype:
        if isinstance(self.computation_dtype, torch.dtype):
            return self.computation_dtype
        try:
            return next(self.inner.parameters()).dtype
        except Exception:
            return torch.float32

    def decode_offset_for(self, reference: torch.Tensor) -> torch.Tensor:
        if reference.ndim != 4:
            raise RuntimeError(
                f"FLUX.2 decode offset reference must be BCHW latents; got shape={tuple(reference.shape)}."
            )
        offset = flux2_decode_offset_raw(
            self._vae,
            height=int(reference.shape[2]),
            width=int(reference.shape[3]),
        )
        return offset.to(device=reference.device, dtype=reference.dtype)

    def _validate_extra_conds(self, sample: torch.Tensor, extra: dict[str, Any]) -> torch.Tensor | None:
        image_latents = extra.pop("image_latents", None)
        pooled = extra.pop("y", None)
        unsupported = sorted(str(key) for key, value in extra.items() if value is not None)
        if unsupported:
            raise RuntimeError(
                "FLUX.2 conditioning contract mismatch: unsupported sampler extras "
                f"{unsupported}. Expected cross-attention plus optional inert pooled vector/image_latents."
            )
        if pooled is not None:
            if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
                raise RuntimeError(
                    "FLUX.2 pooled/vector conditioning placeholder must be a 2D tensor when provided; "
                    f"got shape={getattr(pooled, 'shape', None)}."
                )
            if int(pooled.shape[0]) != int(sample.shape[0]):
                raise RuntimeError(
                    "FLUX.2 pooled/vector conditioning batch mismatch: "
                    f"vector.B={int(pooled.shape[0])} expected={int(sample.shape[0])}."
                )
        if image_latents is None:
            return None
        if not isinstance(image_latents, torch.Tensor) or image_latents.ndim != 4:
            raise RuntimeError(
                "FLUX.2 image conditioning must be a 4D tensor (B,C,H,W); "
                f"got shape={getattr(image_latents, 'shape', None)}."
            )
        if tuple(int(dim) for dim in image_latents.shape) != tuple(int(dim) for dim in sample.shape):
            raise RuntimeError(
                "FLUX.2 image conditioning must match the sampled latent shape exactly. "
                f"image_latents={tuple(int(dim) for dim in image_latents.shape)} "
                f"sampled={tuple(int(dim) for dim in sample.shape)}."
            )
        return image_latents.to(device=sample.device, dtype=sample.dtype)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        control=None,
        transformer_options=None,
        **extra: Any,
    ) -> torch.Tensor:
        del control, transformer_options
        image_latents = self._validate_extra_conds(x, extra)

        if x.ndim != 4:
            raise RuntimeError(f"FLUX.2 adapter expects BCHW latents; got shape={tuple(x.shape)}.")
        if int(x.shape[1]) != FLUX2_LATENT_CHANNELS:
            raise RuntimeError(
                "FLUX.2 adapter expects 32 latent channels; "
                f"got shape={tuple(x.shape)}."
            )
        if context.ndim != 3 or int(context.shape[-1]) != 7680:
            raise RuntimeError(
                "FLUX.2 conditioning must be a 3D tensor with feature dim 7680; "
                f"got shape={tuple(context.shape)}."
            )

        patch_latents = patchify_flux2_latents(x)
        patch_std = self._patch_std.to(device=patch_latents.device, dtype=patch_latents.dtype)
        normalized_patch = patch_latents / patch_std
        packed_latents = pack_flux2_latents(normalized_patch)
        primary_token_count = int(packed_latents.shape[1])
        img_ids = prepare_flux2_latent_ids(normalized_patch)
        txt_ids = prepare_flux2_text_ids(context)

        if image_latents is not None:
            image_patch_latents = patchify_flux2_latents(image_latents)
            normalized_image_patch = image_patch_latents / patch_std
            packed_image_latents = pack_flux2_latents(normalized_image_patch)
            image_ids = prepare_flux2_latent_ids(normalized_image_patch)
            packed_latents = torch.cat((packed_latents, packed_image_latents), dim=1)
            img_ids = torch.cat((img_ids, image_ids), dim=1)

        batch_size = int(x.shape[0])
        if timestep.ndim == 0:
            timestep = timestep.view(1).expand(batch_size)
        elif timestep.ndim == 1 and int(timestep.shape[0]) == 1 and batch_size != 1:
            timestep = timestep.expand(batch_size)
        elif timestep.ndim != 1 or int(timestep.shape[0]) != batch_size:
            raise RuntimeError(
                "FLUX.2 timestep tensor must be shape (B,); "
                f"got shape={tuple(timestep.shape)} for batch={batch_size}."
            )

        output = self.inner(
            hidden_states=packed_latents,
            encoder_hidden_states=context,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        output = output[:, :primary_token_count, :]

        patch_height = int(normalized_patch.shape[2])
        patch_width = int(normalized_patch.shape[3])
        patch_output = unpack_flux2_latents(output, height=patch_height, width=patch_width)
        return unpatchify_flux2_latents(patch_output * patch_std)


__all__ = [
    "FLUX2_LATENT_CHANNELS",
    "FLUX2_PATCH_CHANNELS",
    "Flux2KleinTransformerAdapter",
    "Flux2NoisePrediction",
    "decode_flux2_external_latents",
    "encode_flux2_external_latents",
    "flux2_decode_offset_raw",
    "flux2_patch_bn_stats",
    "pack_flux2_latents",
    "patchify_flux2_latents",
    "prepare_flux2_latent_ids",
    "prepare_flux2_text_ids",
    "unpack_flux2_latents",
    "unpatchify_flux2_latents",
]
