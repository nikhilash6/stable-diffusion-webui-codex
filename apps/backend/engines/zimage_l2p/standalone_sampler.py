"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Pixel-space FlowMatch sampler for Z-Image L2P.
Implements the upstream Z-Image scheduler math for exact 1024x1024 L2P txt2img without VAE decode or latent-space fallthrough,
including runtime CFG fused/split batching, conservative prompt-length eligibility, CUDA-OOM fallback, and effective sampling logs.

Symbols (top-level; keep in sync; no ghosts):
- `build_zimage_l2p_sigmas` (function): Builds shifted FlowMatch sigma schedule for L2P.
- `_require_prompt_embeddings` (function): Validates per-prompt L2P text embeddings.
- `_move_embeddings` (function): Moves prompt embeddings to the active denoiser device/dtype.
- `_is_cuda_oom` (function): Classifies CUDA OOM failures for fused-CFG fallback.
- `_rounded_l2p_tokens` (function): Computes L2P text token length after `_SEQ_MULTI_OF` padding.
- `_can_fuse_l2p_cfg` (function): Checks whether cond/uncond prompt embeddings can share one fused CFG forward.
- `_run_l2p_split_cfg` (function): Executes serial cond/uncond CFG forwards.
- `_run_l2p_fused_cfg` (function): Executes fused uncond+cond CFG in one model forward.
- `sample_zimage_l2p_pixel_txt2img` (function): Runs classic-CFG L2P pixel-space Euler sampling and returns decoded RGB tensors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from apps.backend.infra.config.env_flags import env_str
from apps.backend.runtime.families.zimage_l2p.l2p_model import _SEQ_MULTI_OF, _l2p_sdpa_policy_label
from apps.backend.runtime.logging import emit_backend_message
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import AttentionBackend


_CFG_BATCH_MODE_ENV = "CODEX_CFG_BATCH_MODE"


def build_zimage_l2p_sigmas(
    *,
    steps: int,
    denoising_strength: float = 1.0,
    shift: float = 3.0,
    device: torch.device | str,
) -> torch.Tensor:
    """Build the shifted sigma schedule used by the upstream L2P Z-Image scheduler."""

    if int(steps) <= 0:
        raise RuntimeError(f"Z-Image L2P requires positive steps; got {steps}.")
    sigma_start = float(denoising_strength)
    if sigma_start <= 0.0 or sigma_start > 1.0:
        raise RuntimeError(f"Z-Image L2P denoising_strength must be in (0, 1]; got {sigma_start}.")
    raw = torch.linspace(sigma_start, 0.0, int(steps) + 1, dtype=torch.float32, device=device)[:-1]
    return float(shift) * raw / (1.0 + (float(shift) - 1.0) * raw)


def _require_prompt_embeddings(value: object, *, label: str, batch_size: int) -> list[torch.Tensor]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(f"Z-Image L2P {label} conditioning must be a list of per-prompt tensors.")
    tensors = list(value)
    if len(tensors) != int(batch_size):
        raise RuntimeError(
            f"Z-Image L2P {label} conditioning batch mismatch: "
            f"got {len(tensors)} tensors for batch={batch_size}."
        )
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or int(tensor.shape[-1]) != 2560:
            raise RuntimeError(
                f"Z-Image L2P {label} conditioning tensor {index} must be [tokens,2560]; "
                f"got type={type(tensor).__name__} shape={getattr(tensor, 'shape', None)}."
            )
    return tensors


def _move_embeddings(
    embeddings: Sequence[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    return [tensor.to(device=device, dtype=dtype) for tensor in embeddings]


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_types = []
    for owner, name in ((torch, "OutOfMemoryError"), (torch, "CUDAOutOfMemoryError"), (torch.cuda, "OutOfMemoryError")):
        candidate = getattr(owner, name, None)
        if isinstance(candidate, type):
            oom_types.append(candidate)
    if oom_types and isinstance(exc, tuple(oom_types)):  # type: ignore[arg-type]
        return True
    message = str(exc).lower()
    return (
        "cuda out of memory" in message
        or "cublas_status_alloc_failed" in message
        or "cudnn_status_alloc_failed" in message
    )


def _rounded_l2p_tokens(tensor: torch.Tensor) -> int:
    tokens = int(tensor.shape[0])
    padding = (-tokens) % int(_SEQ_MULTI_OF)
    return tokens + padding


def _can_fuse_l2p_cfg(
    cond_list: Sequence[torch.Tensor],
    uncond_list: Sequence[torch.Tensor],
) -> tuple[bool, str]:
    if len(cond_list) != len(uncond_list):
        return False, f"cond/uncond batch mismatch: {len(cond_list)} != {len(uncond_list)}"
    for index, (cond_tensor, uncond_tensor) in enumerate(zip(cond_list, uncond_list)):
        if cond_tensor.ndim != 2 or uncond_tensor.ndim != 2:
            return False, f"prompt embed rank mismatch at sample {index}"
        if int(cond_tensor.shape[1]) != int(uncond_tensor.shape[1]):
            return False, (
                f"prompt embed dim mismatch at sample {index}: "
                f"{tuple(cond_tensor.shape)} vs {tuple(uncond_tensor.shape)}"
            )
        cond_tokens = _rounded_l2p_tokens(cond_tensor)
        uncond_tokens = _rounded_l2p_tokens(uncond_tensor)
        if cond_tokens != uncond_tokens:
            return False, (
                f"rounded prompt token length mismatch at sample {index}: "
                f"{cond_tokens} != {uncond_tokens}"
            )
    return True, "ok"


def _run_l2p_split_cfg(
    *,
    model: torch.nn.Module,
    model_input: torch.Tensor,
    sigma_batch: torch.Tensor,
    cond_list: Sequence[torch.Tensor],
    uncond_list: Sequence[torch.Tensor],
    cfg_scale: float,
) -> torch.Tensor:
    cond_pred = model(model_input, sigma_batch, prompt_embeds=cond_list).float()
    uncond_pred = model(model_input, sigma_batch, prompt_embeds=uncond_list).float()
    return uncond_pred + float(cfg_scale) * (cond_pred - uncond_pred)


def _run_l2p_fused_cfg(
    *,
    model: torch.nn.Module,
    model_input: torch.Tensor,
    sigma_batch: torch.Tensor,
    cond_list: Sequence[torch.Tensor],
    uncond_list: Sequence[torch.Tensor],
    cfg_scale: float,
) -> torch.Tensor:
    batch_size = int(model_input.shape[0])
    fused_input = torch.cat([model_input, model_input], dim=0)
    fused_sigma = torch.cat([sigma_batch, sigma_batch], dim=0)
    fused_embeds = list(uncond_list) + list(cond_list)
    fused_pred = model(fused_input, fused_sigma, prompt_embeds=fused_embeds).float()
    if int(fused_pred.shape[0]) != batch_size * 2:
        raise RuntimeError(
            "Z-Image L2P fused CFG expected model output batch "
            f"{batch_size * 2}, got {tuple(fused_pred.shape)}."
        )
    uncond_pred, cond_pred = fused_pred.split(batch_size, dim=0)
    return uncond_pred + float(cfg_scale) * (cond_pred - uncond_pred)


@torch.inference_mode()
def sample_zimage_l2p_pixel_txt2img(
    *,
    model: torch.nn.Module,
    noise: torch.Tensor,
    cond: object,
    uncond: object | None,
    steps: int,
    guidance_scale: float,
    flow_shift: float = 3.0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> torch.Tensor:
    """Run L2P pixel-space Euler sampling and return decoded RGB `[B,3,H,W]` in `[-1, 1]`."""

    if noise.ndim != 4 or int(noise.shape[1]) != 3:
        raise RuntimeError(f"Z-Image L2P noise must be [B,3,H,W]; got {tuple(noise.shape)}.")
    batch_size = int(noise.shape[0])
    height = int(noise.shape[2])
    width = int(noise.shape[3])
    if height != 1024 or width != 1024:
        raise RuntimeError(f"Z-Image L2P first tranche supports exactly 1024x1024; got {width}x{height}.")

    device = noise.device
    model_dtype = getattr(model, "computation_dtype", None)
    if not isinstance(model_dtype, torch.dtype):
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32

    cond_list = _move_embeddings(
        _require_prompt_embeddings(cond, label="positive", batch_size=batch_size),
        device=device,
        dtype=model_dtype,
    )
    uncond_list = None
    cfg_scale = float(guidance_scale)
    if cfg_scale > 1.0:
        uncond_list = _move_embeddings(
            _require_prompt_embeddings(uncond, label="negative", batch_size=batch_size),
            device=device,
            dtype=model_dtype,
        )

    sample = noise.to(device=device, dtype=torch.float32)
    sigmas = build_zimage_l2p_sigmas(steps=int(steps), shift=float(flow_shift), device=device)
    sigma_next = torch.cat([sigmas[1:], torch.zeros((1,), dtype=sigmas.dtype, device=device)], dim=0)

    requested_cfg_batch_mode = env_str(
        _CFG_BATCH_MODE_ENV,
        default="fused",
        allowed={"fused", "split"},
    )
    has_cfg = uncond_list is not None and cfg_scale > 1.0
    fused_allowed = False
    fused_block_reason = "cfg disabled"
    if has_cfg:
        fused_allowed, fused_block_reason = _can_fuse_l2p_cfg(cond_list, uncond_list)
    if not has_cfg:
        effective_cfg_batch_mode = "none"
    elif requested_cfg_batch_mode == "split":
        effective_cfg_batch_mode = "split"
        fused_block_reason = "requested split"
    elif fused_allowed:
        effective_cfg_batch_mode = "fused"
    else:
        effective_cfg_batch_mode = "split"

    emit_backend_message(
        "[zimage_l2p] sampling config",
        logger=__name__,
        requested_cfg_batch_mode=requested_cfg_batch_mode,
        effective_cfg_batch_mode=effective_cfg_batch_mode,
        cfg_scale=cfg_scale,
        steps=int(steps),
        fused_allowed=fused_allowed,
        fused_block_reason=fused_block_reason,
        input_shape=tuple(sample.shape),
        dtype=str(model_dtype),
        device=str(device),
        attention_backend=AttentionBackend.PYTORCH.value,
        attention_sdpa_policy=_l2p_sdpa_policy_label(),
    )

    fused_enabled_for_run = effective_cfg_batch_mode == "fused"
    first_forward_logged = False
    for step_index, sigma in enumerate(sigmas):
        sigma_batch = sigma.repeat(batch_size).to(device=device)
        model_input = sample.to(dtype=model_dtype)
        if not first_forward_logged:
            emit_backend_message(
                "[zimage_l2p] first denoiser forward",
                logger=__name__,
                effective_cfg_batch_mode=effective_cfg_batch_mode,
                model_input_shape=tuple(model_input.shape),
                sigma_shape=tuple(sigma_batch.shape),
                cond_tokens=tuple(int(tensor.shape[0]) for tensor in cond_list),
                uncond_tokens=tuple(int(tensor.shape[0]) for tensor in uncond_list) if uncond_list is not None else (),
                dtype=str(model_input.dtype),
                device=str(model_input.device),
            )
            first_forward_logged = True
        if not has_cfg:
            model_output = model(model_input, sigma_batch, prompt_embeds=cond_list).float()
        elif fused_enabled_for_run and uncond_list is not None:
            try:
                model_output = _run_l2p_fused_cfg(
                    model=model,
                    model_input=model_input,
                    sigma_batch=sigma_batch,
                    cond_list=cond_list,
                    uncond_list=uncond_list,
                    cfg_scale=cfg_scale,
                )
            except Exception as exc:
                if not _is_cuda_oom(exc):
                    raise
                fused_enabled_for_run = False
                effective_cfg_batch_mode = "split"
                emit_backend_message(
                    "[zimage_l2p] fused CFG OOM; falling back to split",
                    logger=__name__,
                    level="WARNING",
                    step=int(step_index + 1),
                    steps=int(steps),
                    reason=str(exc),
                )
                memory_management.manager.soft_empty_cache()
                model_output = _run_l2p_split_cfg(
                    model=model,
                    model_input=model_input,
                    sigma_batch=sigma_batch,
                    cond_list=cond_list,
                    uncond_list=uncond_list,
                    cfg_scale=cfg_scale,
                )
        elif uncond_list is not None:
            model_output = _run_l2p_split_cfg(
                model=model,
                model_input=model_input,
                sigma_batch=sigma_batch,
                cond_list=cond_list,
                uncond_list=uncond_list,
                cfg_scale=cfg_scale,
            )
        else:
            raise RuntimeError("Z-Image L2P CFG state mismatch: uncond embeddings are missing while CFG is active.")
        sample = sample + model_output * (sigma_next[step_index] - sigma)
        if progress_callback is not None:
            progress_callback(
                {
                    "step": int(step_index + 1),
                    "steps": int(steps),
                    "sigma": float(sigma.detach().cpu().item()),
                }
            )

    return sample.clamp(-1.0, 1.0)


__all__ = ["build_zimage_l2p_sigmas", "sample_zimage_l2p_pixel_txt2img"]
