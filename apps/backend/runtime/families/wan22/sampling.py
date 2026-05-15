"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF sampling helpers (geometry + scheduler + per-stage sampling loops).
Builds patch geometry, prepares per-stage latent tensors, and runs the stage sampling loop (generator yields progress events); CFG execution uses sequential cond/uncond passes to lower VRAM peaks, I2V conditioning channels are cached once per stage loop to avoid redundant per-step buffer copies, and scheduler aliases/sampler overrides are validated fail-loud against the real WAN22 runtime lanes (`uni-pc`, `euler`, `euler a`) without collapsing cfg++ labels into plain Euler paths.
Per-step compute runs under `torch.inference_mode()` to reduce overhead (model assembly/load stays outside inference mode); block-progress callback wiring is strict/mandatory and must be provided through `transformer_options` by the WAN unified progress adapter.

Symbols (top-level; keep in sync; no ghosts):
- `PatchGeometry` (dataclass): Patch/tile geometry configuration used to infer latent/video shapes.
- `latent_dimensions` (function): Computes latent tensor dimensions from a `PatchGeometry` description.
- `resize_latents_hw` (function): Resizes latents to a target H/W (used for compatibility across stages/sizes).
- `ensure_latent_shape` (function): Validates/reshapes latent tensors to the expected `PatchGeometry` layout.
- `infer_patch_geometry` (function): Infers patch geometry defaults from config and requested latent size.
- `make_scheduler` (function): Builds the WAN22 scheduler from vendored metadata (`scheduler_config.json`); scheduler overrides stay strict, sampler overrides must resolve to a real WAN22 lane, cfg++ sampler labels are rejected instead of remapped, and effective sampler metadata can be returned for run reporting.
- `resolve_init_noise_sigma` (function): Resolves the scheduler initial noise sigma (`init_noise_sigma`) for seeding parity with Diffusers.
- `_assert_finite_tensor` (function): Fail-loud finite check helper with stage/step context and numeric summaries.
- `cfg_merge` (function): Classifier-free guidance merge helper (uncond/cond + scale).
- `time_snr_shift` (function): Time/SNR shift helper used in scheduler-time transformations.
- `prepare_stage_seed_latents` (function): Prepares seeded stage latents (for determinism across runs/stages).
- `build_i2v_mask4` (function): Builds the 4-channel I2V first-frame mask (Diffusers-compatible; latent time scale=4).
- `assemble_i2v_state` (function): Assembles I2V model state `[lat16 + mask4 + img16]` (order-aware, strict).
- `sample_stage_latents` (function): Core latent sampling for a single WAN stage (high/low) using the selected scheduler/sampler; requires adapter-wired `transformer_options` for block-progress callback hookup.
- `sample_stage_latents_generator` (function): Generator version of stage sampling for streaming progress (yields intermediate states; CFG path runs sequential cond/uncond passes, I2V conditioning channels are cached once per stage loop, and non-CFG timestep buffers are reused), with strict fail-loud block-progress emitter wiring.
"""

from __future__ import annotations
from apps.backend.runtime.logging import BackendLoggerProxy, emit_backend_message, get_backend_logger

import math
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from apps.backend.runtime.attention.sram import sram_attention_runtime_metrics_set_stage
from apps.backend.runtime.sampling.block_progress import (
    BLOCK_PROGRESS_CALLBACK_KEY,
    resolve_block_progress_callback,
)

from .config import WAN_FLOW_MULTIPLIER, resolve_i2v_order
from .diagnostics import (
    get_logger,
    log_cuda_mem,
    log_numerics_enabled,
    log_sigmas_enabled,
    log_t_mapping,
    summarize_tensor,
    summarize_numerics,
)


@dataclass(frozen=True)
class PatchGeometry:
    grid: Tuple[int, int, int]
    token_count: int
    token_dim: int
    in_channels: int
    patch_kernel: Tuple[int, int, int]


def latent_dimensions(geom: PatchGeometry) -> Tuple[int, int, int]:
    kT, kH, kW = geom.patch_kernel
    return (
        int(geom.grid[0] * kT),
        int(geom.grid[1] * kH),
        int(geom.grid[2] * kW),
    )


def resize_latents_hw(x: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    import torch.nn.functional as F

    if x.ndim == 5:
        b, c, t, h, w = x.shape
        if h == height and w == width:
            return x
        xt = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        xt = F.interpolate(xt, size=(int(height), int(width)), mode="bilinear", align_corners=False)
        xt = xt.view(b, t, c, height, width).permute(0, 2, 1, 3, 4).contiguous()
        return xt

    if x.ndim == 4:
        b, c, h, w = x.shape
        if h == height and w == width:
            return x
        return F.interpolate(x, size=(int(height), int(width)), mode="bilinear", align_corners=False)

    return x


def ensure_latent_shape(x: torch.Tensor, geom: PatchGeometry) -> torch.Tensor:
    t_target, h_target, w_target = latent_dimensions(geom)
    if x.ndim != 5:
        raise RuntimeError(f"WAN22: expected 5D latents [B,C,T,H,W], got shape={tuple(x.shape)}")
    if x.shape[2] == t_target and x.shape[3] == h_target and x.shape[4] == w_target:
        return x
    return resize_latents_hw(x, height=h_target, width=w_target)


def infer_patch_geometry(model: Any, *, t: int, h_lat: int, w_lat: int) -> PatchGeometry:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("WAN22: expected model with .config (WanTransformer2DModel)")
    kT, kH, kW = tuple(int(x) for x in getattr(cfg, "patch_size", (1, 2, 2)))
    if t < kT or h_lat < kH or w_lat < kW:
        raise RuntimeError(
            f"WAN22: invalid latent shape for patch_embed: T={t} H={h_lat} W={w_lat} kernel={(kT, kH, kW)}"
        )
    gT = int(t - kT + 1)
    gH = int(((h_lat - kH) // kH) + 1)
    gW = int(((w_lat - kW) // kW) + 1)
    token_count = int(gT * gH * gW)
    c_out = int(getattr(cfg, "d_model", 0) or 0)
    c_in = int(getattr(cfg, "in_channels", 0) or 0)
    return PatchGeometry(
        grid=(gT, gH, gW),
        token_count=token_count,
        token_dim=c_out,
        in_channels=c_in,
        patch_kernel=(kT, kH, kW),
    )


def make_scheduler(
    steps: int,
    *,
    metadata_dir: str,
    flow_shift: float,
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
    return_effective_sampler: bool = False,
):
    """Instantiate the WAN22 scheduler from vendored metadata (Diffusers-free).

    Source of truth is `model_index.json` + `scheduler/scheduler_config.json` shipped with the official repos.
    Scheduler construction remains metadata-driven (WAN flow lane): UniPC is the default metadata lane, while
    Euler-family sampler overrides can opt into an experimental FlowMatch-Euler runtime scheduler.
    """

    import json
    import os

    if not metadata_dir:
        raise RuntimeError("WAN22 GGUF: metadata_dir is required to build the scheduler (missing WAN metadata).")

    raw_sampler = str(sampler or "").strip().lower()
    raw_scheduler = str(scheduler or "").strip().lower()
    if raw_scheduler in {"inherit", "auto", "default"}:
        raise RuntimeError(
            "WAN22 GGUF: scheduler aliases ('inherit'/'auto'/'default') are not supported; "
            "use 'simple'."
        )

    vendor_dir = os.path.expanduser(str(metadata_dir))
    scheduler_dir = os.path.join(vendor_dir, "scheduler")
    if not os.path.isdir(scheduler_dir):
        parent = os.path.dirname(vendor_dir)
        scheduler_dir = os.path.join(parent, "scheduler") if parent else ""
        if scheduler_dir and os.path.isdir(scheduler_dir):
            vendor_dir = parent
        else:
            raise RuntimeError(
                f"WAN22 GGUF: metadata_dir must be a diffusers repo dir (or a tokenizer dir whose parent is one): {metadata_dir!r}"
            )

    config_path = None
    for fname in ("scheduler_config.json", "config.json"):
        candidate = os.path.join(vendor_dir, "scheduler", fname)
        if os.path.isfile(candidate):
            config_path = candidate
            break
    if not config_path:
        raise RuntimeError(f"WAN22 GGUF: scheduler config not found under: {vendor_dir!r} (expected scheduler_config.json)")

    try:
        config_raw = json.loads(open(config_path, encoding="utf-8").read())
    except Exception as exc:  # noqa: BLE001 - strict decode
        raise RuntimeError(f"WAN22 GGUF: invalid scheduler config JSON: {config_path}: {exc}") from exc
    if not isinstance(config_raw, dict):
        raise RuntimeError(f"WAN22 GGUF: scheduler config must be a JSON object: {config_path}")

    class_name = str(config_raw.get("_class_name") or "").strip()
    if not class_name:
        raise RuntimeError(f"WAN22 GGUF: scheduler config missing _class_name: {config_path}")

    if raw_scheduler and raw_scheduler != "simple":
        raise RuntimeError(
            f"WAN22 GGUF: unsupported scheduler override {scheduler!r}; expected 'simple'."
        )

    from apps.backend.types.samplers import SamplerKind
    from .scheduler import (
        build_wan_flow_match_euler_scheduler,
        build_wan_unipc_flow_scheduler,
    )

    sampler_lane: str = SamplerKind.UNI_PC.value
    sampler_solver_hint: str | None = None
    if raw_sampler:
        parts = raw_sampler.split()
        sampler_name = parts[0]
        if sampler_name == SamplerKind.UNI_PC.value:
            if len(parts) > 2:
                raise RuntimeError(
                    f"WAN22 GGUF: sampler override must be 'uni-pc' or 'uni-pc <solver_hint>', got {sampler!r}."
                )
            if len(parts) == 2:
                sampler_solver_hint = parts[1]
                if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", str(sampler_solver_hint)) is None:
                    raise RuntimeError(
                        f"WAN22 GGUF: invalid UniPC solver hint in sampler override {sampler!r}; "
                        "use lowercase [a-z0-9._-] tokens only."
                    )
        else:
            try:
                sampler_kind = SamplerKind.from_string(raw_sampler)
            except Exception as exc:
                raise RuntimeError(
                    f"WAN22 GGUF: unsupported sampler override {sampler!r}. "
                    "Supported WAN22 sampler lanes: 'uni-pc' (optional solver hint), 'euler', 'euler a'."
                ) from exc
            if sampler_kind in {SamplerKind.UNI_PC, SamplerKind.UNI_PC_BH2}:
                sampler_lane = SamplerKind.UNI_PC.value
                if sampler_kind is SamplerKind.UNI_PC_BH2:
                    sampler_solver_hint = "bh2"
            elif sampler_kind is SamplerKind.EULER:
                sampler_lane = SamplerKind.EULER.value
            elif sampler_kind is SamplerKind.EULER_A:
                sampler_lane = SamplerKind.EULER_A.value
            else:
                raise RuntimeError(
                    f"WAN22 GGUF: unsupported sampler override {sampler!r}. "
                    "Supported WAN22 sampler lanes: 'uni-pc' (optional solver hint), 'euler', 'euler a'."
                )

    if sampler_lane == SamplerKind.EULER.value:
        get_backend_logger("backend.runtime.wan22.sampling").warning(
            "WAN22 GGUF: sampler=%r routed to experimental FlowMatch-Euler scheduler lane.",
            sampler,
        )
        scheduler_obj = build_wan_flow_match_euler_scheduler(
            steps=max(1, int(steps)),
            vendor_dir=vendor_dir,
            flow_shift=float(flow_shift),
            stochastic_sampling=False,
        )
        if return_effective_sampler:
            return scheduler_obj, SamplerKind.EULER.value
        return scheduler_obj

    if sampler_lane == SamplerKind.EULER_A.value:
        get_backend_logger("backend.runtime.wan22.sampling").warning(
            "WAN22 GGUF: sampler=%r routed to experimental FlowMatch-Euler stochastic scheduler lane.",
            sampler,
        )
        scheduler_obj = build_wan_flow_match_euler_scheduler(
            steps=max(1, int(steps)),
            vendor_dir=vendor_dir,
            flow_shift=float(flow_shift),
            stochastic_sampling=True,
        )
        if return_effective_sampler:
            return scheduler_obj, SamplerKind.EULER_A.value
        return scheduler_obj

    if class_name != "UniPCMultistepScheduler":
        raise RuntimeError(
            f"WAN22 GGUF: unsupported metadata scheduler {class_name!r} in {config_path}; expected UniPCMultistepScheduler."
        )

    config_solver = str(config_raw.get("solver_type") or "").strip().lower() or None
    if sampler_solver_hint is not None:
        if config_solver is None:
            raise RuntimeError(
                "WAN22 GGUF: sampler override requests UniPC solver hint "
                f"{sampler_solver_hint!r}, but metadata scheduler has no solver_type."
            )
        if sampler_solver_hint != config_solver:
            raise RuntimeError(
                "WAN22 GGUF: sampler override UniPC solver hint mismatch "
                f"(requested={sampler_solver_hint!r} metadata={config_solver!r})."
            )

    scheduler_obj = build_wan_unipc_flow_scheduler(
        steps=max(1, int(steps)),
        vendor_dir=vendor_dir,
        flow_shift=float(flow_shift),
    )
    effective_sampler = SamplerKind.UNI_PC.value if config_solver is None else f"{SamplerKind.UNI_PC.value} {config_solver}"
    if return_effective_sampler:
        return scheduler_obj, effective_sampler
    return scheduler_obj


def resolve_init_noise_sigma(scheduler: Any) -> float:
    """Return the scheduler-defined initial noise sigma (Diffusers parity).

    Diffusers pipelines scale the initial Gaussian noise by `scheduler.init_noise_sigma`.
    WAN22 GGUF uses the same behavior; `WAN_FLOW_MULTIPLIER` is only for model timestep inputs.
    """

    raw = getattr(scheduler, "init_noise_sigma", None)
    if raw is None:
        return 1.0
    try:
        val = float(raw)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"WAN22 GGUF: invalid scheduler.init_noise_sigma={raw!r}") from exc
    if not math.isfinite(val) or val <= 0:
        raise RuntimeError(f"WAN22 GGUF: invalid scheduler.init_noise_sigma={raw!r} (expected finite > 0)")
    return val


def _assert_finite_tensor(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    stage_name: str,
    local_step: int,
    total_steps: int,
    global_idx: int,
    timestep: Any,
) -> None:
    if torch.isfinite(tensor).all():
        return
    bad = int((~torch.isfinite(tensor)).sum().item())
    try:
        timestep_repr = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
    except Exception:
        timestep_repr = str(timestep)
    raise RuntimeError(
        "WAN22 GGUF: non-finite tensor in stage sampling "
        f"(stage={stage_name} step={int(local_step)}/{int(total_steps)} idx={int(global_idx)} "
        f"timestep={timestep_repr} tensor={tensor_name} bad={bad}; "
        f"{summarize_numerics(tensor, name=tensor_name)})."
    )


def cfg_merge(uncond: torch.Tensor, cond: torch.Tensor, scale: float | None) -> torch.Tensor:
    if scale is None:
        return cond
    guidance = float(scale)
    uncond.mul_(1.0 - guidance)
    uncond.add_(cond, alpha=guidance)
    return uncond


def time_snr_shift(alpha: float, t: float) -> float:
    # Same functional form as time_snr_shift used in reference implementations
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def prepare_stage_seed_latents(
    latents: torch.Tensor,
    target_geom: PatchGeometry,
    *,
    logger: BackendLoggerProxy | None,
) -> torch.Tensor:
    c_src = int(latents.shape[1])
    c_dst = int(target_geom.in_channels)
    if c_src == c_dst:
        return ensure_latent_shape(latents, target_geom)
    if c_src >= 16 and c_dst == 16:
        sliced = latents[:, :16, ...] if resolve_i2v_order() == "lat_first" else latents[:, -16:, ...]
        return ensure_latent_shape(sliced, target_geom)
    if c_src == 16 and c_dst == 36:
        raise RuntimeError(
            "WAN22 GGUF: cannot assemble I2V (Cin=36) state from 16-channel latents alone. "
            "Build the full I2V state explicitly (noise latents + mask4 + image_latents) in the img2vid runner."
        )
    raise RuntimeError(f"Cannot adapt latent channels from {c_src} to {c_dst}; unsupported hand-off configuration")



def build_i2v_mask4(
    *,
    batch: int,
    num_frames: int,
    latent_frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    scale_factor_temporal: int = 4,
) -> torch.Tensor:
    """Build the 4-channel I2V mask (Diffusers-compatible).

    Diffusers (WanImageToVideoPipeline.prepare_latents) builds a 4-channel mask by:
    - creating a 1-channel per-frame mask at *output* time resolution,
    - repeating the first frame mask by `scale_factor_temporal`,
    - reshaping into `[B, 4, T_lat, H_lat, W_lat]` where `T_lat=(T_out-1)//scale+1`.

    For the I2V case (no `last_image`), the mask is 1 on the first latent-time chunk and 0 elsewhere.
    """

    if num_frames <= 0:
        raise RuntimeError(f"build_i2v_mask4: num_frames must be > 0, got {num_frames}")
    if scale_factor_temporal <= 0:
        raise RuntimeError(f"build_i2v_mask4: scale_factor_temporal must be > 0, got {scale_factor_temporal}")
    if num_frames % scale_factor_temporal != 1:
        raise RuntimeError(
            "build_i2v_mask4: num_frames must satisfy num_frames % scale_factor_temporal == 1 "
            f"(num_frames={num_frames} scale={scale_factor_temporal})"
        )

    expected_latent = int((int(num_frames) - 1) // int(scale_factor_temporal) + 1)
    if int(latent_frames) != expected_latent:
        raise RuntimeError(
            "build_i2v_mask4: latent_frames mismatch "
            f"(latent_frames={int(latent_frames)} expected={expected_latent} from num_frames={num_frames} scale={scale_factor_temporal})"
        )

    expected_frames = int(latent_frames) * int(scale_factor_temporal)
    # mask_lat_size: [B,1,T_out_expanded,H_lat,W_lat]
    # Expanded timeline matches Diffusers semantics: first frame is repeated over the first latent-time chunk.
    mask_lat_size = torch.zeros((int(batch), 1, expected_frames, int(height), int(width)), device=device, dtype=dtype)
    mask_lat_size[:, :, : int(scale_factor_temporal), :, :] = 1
    if int(mask_lat_size.shape[2]) != expected_frames:
        raise RuntimeError(
            "build_i2v_mask4: internal frame count mismatch after expansion "
            f"(got={int(mask_lat_size.shape[2])} expected={expected_frames})"
        )

    # Reshape and transpose to `[B,4,T_lat,H_lat,W_lat]`
    mask_lat_size = mask_lat_size.view(int(batch), -1, int(scale_factor_temporal), int(height), int(width))
    mask_lat_size = mask_lat_size.transpose(1, 2)
    if tuple(mask_lat_size.shape) != (int(batch), int(scale_factor_temporal), int(latent_frames), int(height), int(width)):
        raise RuntimeError(
            "build_i2v_mask4: unexpected output shape "
            f"(got={tuple(mask_lat_size.shape)} expected={(int(batch), int(scale_factor_temporal), int(latent_frames), int(height), int(width))})"
        )
    return mask_lat_size


def assemble_i2v_state(
    latents: torch.Tensor,
    *,
    mask4: torch.Tensor,
    image_latents: torch.Tensor,
    expected_cin: int,
    logger: BackendLoggerProxy | None,
) -> torch.Tensor:
    """Assemble I2V model input state to match expected Cin for patch embedding.

    This is the strict, canonical assembly for WAN I2V:
    - `latents`: noise latents (16ch) at latent time resolution
    - `mask4`: 4-channel first-frame mask at latent time resolution
    - `image_latents`: VAE-encoded video_condition latents (16ch) at latent time resolution
    """

    if latents.ndim != 5:
        raise RuntimeError(f"assemble_i2v_state: expected 5D latents [B,C,T,H,W], got {tuple(latents.shape)}")
    if mask4.ndim != 5:
        raise RuntimeError(f"assemble_i2v_state: expected 5D mask4 [B,4,T,H,W], got {tuple(mask4.shape)}")
    if image_latents.ndim != 5:
        raise RuntimeError(f"assemble_i2v_state: expected 5D image_latents [B,16,T,H,W], got {tuple(image_latents.shape)}")
    if latents.dtype != mask4.dtype or latents.dtype != image_latents.dtype:
        raise RuntimeError(
            "assemble_i2v_state: dtype mismatch between latents/mask4/image_latents "
            f"(latents={latents.dtype} mask4={mask4.dtype} image_latents={image_latents.dtype})."
        )
    if latents.device != mask4.device or latents.device != image_latents.device:
        raise RuntimeError(
            "assemble_i2v_state: device mismatch between latents/mask4/image_latents "
            f"(latents={latents.device} mask4={mask4.device} image_latents={image_latents.device})."
        )

    b, c_lat, t, h, w = latents.shape
    if mask4.shape[0] != b or mask4.shape[2:] != (t, h, w):
        raise RuntimeError(
            "assemble_i2v_state: mask4 shape mismatch "
            f"(mask4={tuple(mask4.shape)} latents={tuple(latents.shape)})"
        )
    if image_latents.shape[0] != b or image_latents.shape[2:] != (t, h, w):
        raise RuntimeError(
            "assemble_i2v_state: image_latents shape mismatch "
            f"(image_latents={tuple(image_latents.shape)} latents={tuple(latents.shape)})"
        )
    if int(mask4.shape[1]) != 4:
        raise RuntimeError(f"assemble_i2v_state: expected mask4 to have 4 channels, got {int(mask4.shape[1])}")
    if int(image_latents.shape[1]) != 16:
        raise RuntimeError(
            f"assemble_i2v_state: expected image_latents to have 16 channels, got {int(image_latents.shape[1])}"
        )

    expected = int(c_lat) + 4 + 16
    if int(expected_cin) != expected:
        raise RuntimeError(
            "assemble_i2v_state: expected_cin mismatch "
            f"(expected_cin={int(expected_cin)} expected={expected} from latents={int(c_lat)} + mask4 + img16)"
        )

    order = resolve_i2v_order()
    assembled = latents.new_empty((int(b), int(expected_cin), int(t), int(h), int(w)))
    if order == "lat_first":
        assembled[:, : int(c_lat), ...] = latents
        assembled[:, int(c_lat) : int(c_lat) + 4, ...] = mask4
        assembled[:, int(c_lat) + 4 :, ...] = image_latents
        layout = f"[lat{int(c_lat)} + mask4 + img16]"
    else:
        assembled[:, :4, ...] = mask4
        assembled[:, 4:20, ...] = image_latents
        assembled[:, 20:, ...] = latents
        layout = f"[mask4 + img16 + lat{int(c_lat)}]"
    if int(assembled.shape[1]) != int(expected_cin):
        raise RuntimeError(
            f"assemble_i2v_state: produced C={int(assembled.shape[1])}, expected C_in={int(expected_cin)} ({layout})."
        )
    if logger is not None:
        emit_backend_message(
            "[wan22.gguf] i2v assemble",
            logger=logger.name,
            order=order,
            layout=layout,
            channels=int(assembled.shape[1]),
        )
    return assembled


def sample_stage_latents(
    *,
    model: Any,
    geom: PatchGeometry,
    steps: int,
    cfg_scale: Optional[float],
    prompt_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    logger: BackendLoggerProxy | None,
    sampler_name: Optional[str] = None,
    scheduler_name: Optional[str] = None,
    metadata_dir: Optional[str] = None,
    scheduler_obj: Any | None = None,
    timestep_start: int = 0,
    timestep_end: Optional[int] = None,
    seed: Optional[int] = None,
    state_init: Optional[torch.Tensor] = None,
    on_progress: Optional[Any] = None,
    log_mem_interval: Optional[int] = None,
    flow_shift: float,
    flow_multiplier: float = WAN_FLOW_MULTIPLIER,
    stage_name: str = "stage",
    transformer_options: dict[str, Any] | None = None,
) -> torch.Tensor:
    gen = sample_stage_latents_generator(
        model=model,
        geom=geom,
        steps=steps,
        cfg_scale=cfg_scale,
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        device=device,
        dtype=dtype,
        logger=logger,
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
        metadata_dir=metadata_dir,
        scheduler_obj=scheduler_obj,
        timestep_start=timestep_start,
        timestep_end=timestep_end,
        seed=seed,
        state_init=state_init,
        log_mem_interval=log_mem_interval,
        flow_shift=flow_shift,
        flow_multiplier=flow_multiplier,
        stage_name=stage_name,
        emit_logs=(on_progress is None),
        transformer_options=transformer_options,
    )

    while True:
        try:
            event = next(gen)
        except StopIteration as stop:
            return stop.value
        if on_progress:
            payload = {k: event[k] for k in ("step", "total", "percent", "eta_seconds", "step_seconds") if k in event}
            try:
                on_progress(**payload)
            except Exception as exc:
                emit_backend_message(
                    "[wan22.gguf] progress callback raised",
                    logger=logger.name if logger is not None else "backend.runtime.wan22.gguf",
                    level=logging.DEBUG,
                    error=str(exc),
                )


def sample_stage_latents_generator(
    *,
    model: Any,
    geom: PatchGeometry,
    steps: int,
    cfg_scale: Optional[float],
    prompt_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    logger: BackendLoggerProxy | None,
    sampler_name: Optional[str] = None,
    scheduler_name: Optional[str] = None,
    metadata_dir: Optional[str] = None,
    scheduler_obj: Any | None = None,
    timestep_start: int = 0,
    timestep_end: Optional[int] = None,
    seed: Optional[int] = None,
    state_init: Optional[torch.Tensor] = None,
    log_mem_interval: Optional[int] = None,
    flow_shift: float,
    flow_multiplier: float = WAN_FLOW_MULTIPLIER,
    stage_name: str = "stage",
    emit_logs: bool = True,
    transformer_options: dict[str, Any] | None = None,
):
    log = get_logger(logger)
    sram_attention_runtime_metrics_set_stage(stage_name)
    scheduler_state_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    t_lat, h_lat, w_lat = latent_dimensions(geom)
    steps = max(int(steps), 1)

    if transformer_options is None:
        model_transformer_options: dict[str, Any] | None = None
    elif isinstance(transformer_options, dict):
        model_transformer_options = transformer_options
    else:
        raise RuntimeError(
            "WAN22 GGUF: transformer_options must be a dict when provided "
            f"(got {type(transformer_options).__name__})."
        )
    block_progress_callback = resolve_block_progress_callback(model_transformer_options)
    if block_progress_callback is None:
        raise RuntimeError(
            "WAN22 GGUF: missing block-progress emitter wiring for stage sampling "
            f"(stage={stage_name!r}). Route this branch through the WAN unified progress adapter and "
            f"provide transformer_options['{BLOCK_PROGRESS_CALLBACK_KEY}']."
        )
    emit_logs = bool(emit_logs and block_progress_callback is None)

    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("WAN22: expected model with .config (WanTransformer2DModel)")
    cin = int(getattr(cfg, "in_channels", 0) or 0)
    cout = int(getattr(cfg, "latent_channels", 0) or 0)
    if cin <= 0 or cout <= 0:
        raise RuntimeError(f"WAN22: invalid model channels (in_channels={cin}, latent_channels={cout})")

    if int(geom.in_channels) != cin:
        raise RuntimeError(
            f"WAN22: geometry/model mismatch (geom.cin={int(geom.in_channels)} vs model.in_channels={cin})."
        )

    scheduler = scheduler_obj
    if scheduler is None:
        if metadata_dir is None:
            raise RuntimeError("WAN22 GGUF: metadata_dir is required for stage sampling (missing WAN metadata).")
        scheduler = make_scheduler(
            steps,
            metadata_dir=metadata_dir,
            flow_shift=flow_shift,
            sampler=sampler_name,
            scheduler=scheduler_name,
        )

    timesteps = scheduler.timesteps
    total_all = len(timesteps)

    start = int(timestep_start or 0)
    end = int(timestep_end) if timestep_end is not None else int(total_all)
    if start < 0 or end < 0 or start > end or end > total_all:
        raise RuntimeError(
            f"WAN22 GGUF: invalid timestep slice start={start} end={end} (timesteps={total_all})."
        )
    total = int(end - start)
    if total <= 0:
        raise RuntimeError("WAN22 GGUF: timestep slice is empty (no steps to run).")
    if scheduler_obj is not None and int(steps) != total:
        raise RuntimeError(
            f"WAN22 GGUF: step count mismatch for stage {stage_name!r} (steps={int(steps)} slice={total})."
        )
    if state_init is None and start != 0:
        raise RuntimeError("WAN22 GGUF: state_init is required when starting from a non-zero timestep index.")

    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None or len(sigmas) not in (total_all, total_all + 1):
        raise RuntimeError(
            f"WAN22 GGUF: scheduler {scheduler.__class__.__name__} is missing a usable sigma ladder "
            f"(sigmas_len={len(sigmas) if sigmas is not None else None} timesteps={total_all})."
        )

    batch = int(state_init.shape[0]) if state_init is not None else 1
    shape = (batch, int(geom.in_channels), t_lat, h_lat, w_lat)

    if state_init is not None:
        state = ensure_latent_shape(state_init, geom)
        if state.device != device or state.dtype != scheduler_state_dtype:
            state = state.to(device=device, dtype=scheduler_state_dtype)
        if not state.is_contiguous():
            state = state.contiguous()
        # Drop the caller reference as soon as the sampler state is materialized to avoid dual-live retention.
        state_init = None
    else:
        if cin != cout:
            raise RuntimeError(
                "WAN22 GGUF: state_init is required when model.in_channels != model.latent_channels "
                f"(in_channels={cin}, latent_channels={cout})."
            )
        if seed is not None and int(seed) >= 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            state = torch.randn(shape, generator=generator, device=device, dtype=scheduler_state_dtype)
        else:
            state = torch.randn(shape, device=device, dtype=scheduler_state_dtype)
        init_noise_sigma = resolve_init_noise_sigma(scheduler)
        state = state * float(init_noise_sigma)

    if scheduler_state_dtype != dtype:
        log.info(
            "[wan22.gguf] %s scheduler-state dtype island: model_dtype=%s scheduler_dtype=%s",
            stage_name,
            str(dtype),
            str(scheduler_state_dtype),
        )

    parity_idxs = {start, max(start, start + total // 2 - 1), max(start, end - 1)}

    if log_sigmas_enabled():
        sigmas = getattr(scheduler, "sigmas", None)
        if isinstance(sigmas, torch.Tensor):
            log.info(
                "[wan22.gguf] %s schedule: scheduler=%s timesteps=%d sigmas=%s",
                stage_name,
                scheduler.__class__.__name__,
                int(total),
                summarize_tensor(sigmas),
            )
        log_t_mapping(scheduler, timesteps, label=stage_name, logger=logger)

    yield {"type": "progress", "stage": stage_name, "step": 0, "total": total, "percent": 0.0}

    import time

    t0 = time.perf_counter()
    last = t0

    order = resolve_i2v_order()
    effective_cfg_scale: float | None = None
    if cfg_scale is not None:
        cfg_value = float(cfg_scale)
        if not math.isfinite(cfg_value):
            raise RuntimeError(f"WAN22 GGUF: cfg_scale must be finite; got {cfg_scale!r}.")
        if math.isclose(cfg_value, 1.0, rel_tol=0.0, abs_tol=1e-6):
            log.debug(
                "[wan22.gguf] %s: cfg_scale=1.0 -> disabling CFG branch (single conditional pass).",
                stage_name,
            )
        else:
            effective_cfg_scale = cfg_value

    if effective_cfg_scale is not None:
        if int(prompt_embeds.shape[0]) != int(batch):
            raise RuntimeError(
                "WAN22 GGUF: prompt embeds batch does not match latent batch for CFG "
                f"(prompt_B={int(prompt_embeds.shape[0])} latent_B={int(batch)})."
            )
        if int(negative_embeds.shape[0]) != int(batch):
            raise RuntimeError(
                "WAN22 GGUF: negative embeds batch does not match latent batch for CFG "
                f"(negative_B={int(negative_embeds.shape[0])} latent_B={int(batch)})."
            )
    else:
        if int(prompt_embeds.shape[0]) != int(batch):
            raise RuntimeError(
                "WAN22 GGUF: prompt embeds batch does not match latent batch for non-CFG "
                f"(prompt_B={int(prompt_embeds.shape[0])} latent_B={int(batch)})."
            )
    has_conditioning = int(state.shape[1]) != cout
    cond_channels = 0
    state_cond_model_static: torch.Tensor | None = None
    if has_conditioning:
        if int(state.shape[1]) != cin:
            raise RuntimeError(
                f"WAN22 GGUF: latent state channels C={int(state.shape[1])} does not match expected in_channels={cin}."
            )
        if order == "lat_first":
            state_cond_static = state[:, cout:, ...]
        else:
            state_cond_static = state[:, :-cout, ...]
        if state_cond_static.dtype == dtype:
            state_cond_model_static = state_cond_static
        else:
            # I2V conditioning channels are invariant across steps in this loop; cast once.
            state_cond_model_static = state_cond_static.to(dtype=dtype)
        cond_channels = int(state_cond_model_static.shape[1])
    model_state_buffer: torch.Tensor | None = None
    cfg_timestep_buffer: torch.Tensor | None = None
    non_cfg_timestep_buffer: torch.Tensor | None = None
    state_lat_scaled_model_buffer: torch.Tensor | None = None
    eps_scheduler_buffer: torch.Tensor | None = None

    for local_idx, idx in enumerate(range(start, end)):
        timestep = timesteps[idx]
        sigma_value = float(sigmas[idx])
        di_timestep = float(sigma_value) * float(flow_multiplier)
        step_number = int(local_idx + 1)

        _assert_finite_tensor(
            state,
            tensor_name="state_in",
            stage_name=stage_name,
            local_step=step_number,
            total_steps=total,
            global_idx=idx,
            timestep=timestep,
        )

        if log_sigmas_enabled() and idx in parity_idxs:
            log.info(
                "[wan22.gguf] %s t-in[%d/%d]: idx=%d sigma=%.6g flow_multiplier=%.1f di_timestep=%.6g sched_timestep=%s",
                stage_name,
                step_number,
                total,
                idx,
                float(sigma_value),
                float(flow_multiplier),
                float(di_timestep),
                str(timestep),
            )

        with torch.inference_mode():
            if not has_conditioning:
                if int(state.shape[1]) != cout:
                    raise RuntimeError(
                        f"WAN22 GGUF: latent state channels C={int(state.shape[1])} does not match expected latent_channels={cout}."
                    )
                state_lat = state
            else:
                if int(state.shape[1]) != cin:
                    raise RuntimeError(
                        f"WAN22 GGUF: latent state channels C={int(state.shape[1])} does not match expected in_channels={cin}."
                    )

                # Inpainting-style I2V: state is [latents + conditioning], while the model predicts only the latent channels.
                if order == "lat_first":
                    state_lat = state[:, :cout, ...]
                else:
                    state_lat = state[:, -cout:, ...]

            state_lat_scaled = state_lat
            scaler = getattr(scheduler, "scale_model_input", None)
            if callable(scaler):
                state_lat_scaled = scaler(state_lat, timestep)
            _assert_finite_tensor(
                state_lat_scaled,
                tensor_name="state_lat_scaled",
                stage_name=stage_name,
                local_step=step_number,
                total_steps=total,
                global_idx=idx,
                timestep=timestep,
            )

            if state_lat_scaled.dtype == dtype:
                state_lat_scaled_model = state_lat_scaled
            else:
                if (
                    state_lat_scaled_model_buffer is None
                    or tuple(state_lat_scaled_model_buffer.shape) != tuple(state_lat_scaled.shape)
                    or state_lat_scaled_model_buffer.dtype != dtype
                    or state_lat_scaled_model_buffer.device != state_lat_scaled.device
                ):
                    state_lat_scaled_model_buffer = torch.empty(
                        tuple(state_lat_scaled.shape),
                        device=state_lat_scaled.device,
                        dtype=dtype,
                    )
                state_lat_scaled_model_buffer.copy_(state_lat_scaled)
                state_lat_scaled_model = state_lat_scaled_model_buffer
            if not has_conditioning:
                model_state = state_lat_scaled_model
            else:
                if state_cond_model_static is None:
                    raise RuntimeError("WAN22 GGUF: conditioning tensor is missing for I2V latent state.")
                latent_channels = int(state_lat_scaled_model.shape[1])
                expected_shape = (
                    int(state_lat_scaled_model.shape[0]),
                    int(state_lat_scaled_model.shape[1]) + int(cond_channels),
                    int(state_lat_scaled_model.shape[2]),
                    int(state_lat_scaled_model.shape[3]),
                    int(state_lat_scaled_model.shape[4]),
                )
                if (
                    model_state_buffer is None
                    or tuple(model_state_buffer.shape) != expected_shape
                    or model_state_buffer.dtype != dtype
                    or model_state_buffer.device != state_lat_scaled_model.device
                ):
                    model_state_buffer = torch.empty(expected_shape, device=state_lat_scaled_model.device, dtype=dtype)
                    if order == "lat_first":
                        model_state_buffer[:, latent_channels:, ...].copy_(state_cond_model_static)
                    else:
                        model_state_buffer[:, :cond_channels, ...].copy_(state_cond_model_static)
                if order == "lat_first":
                    model_state_buffer[:, :latent_channels, ...].copy_(state_lat_scaled_model)
                else:
                    model_state_buffer[:, cond_channels:, ...].copy_(state_lat_scaled_model)
                model_state = model_state_buffer

            if effective_cfg_scale is None:
                model_batch = int(model_state.shape[0])
                if (
                    non_cfg_timestep_buffer is None
                    or int(non_cfg_timestep_buffer.shape[0]) != model_batch
                    or non_cfg_timestep_buffer.device != device
                ):
                    non_cfg_timestep_buffer = torch.empty((model_batch,), device=device, dtype=torch.float32)
                non_cfg_timestep_buffer.fill_(float(di_timestep))
                eps_model = model(
                    model_state,
                    non_cfg_timestep_buffer,
                    prompt_embeds,
                    transformer_options=model_transformer_options,
                )
                _assert_finite_tensor(
                    eps_model,
                    tensor_name="model_output",
                    stage_name=stage_name,
                    local_step=step_number,
                    total_steps=total,
                    global_idx=idx,
                    timestep=timestep,
                )
            else:
                model_batch = int(model_state.shape[0])
                if (
                    cfg_timestep_buffer is None
                    or int(cfg_timestep_buffer.shape[0]) != model_batch
                    or cfg_timestep_buffer.device != device
                ):
                    cfg_timestep_buffer = torch.empty((model_batch,), device=device, dtype=torch.float32)
                cfg_timestep_buffer.fill_(float(di_timestep))

                v_cond = model(
                    model_state,
                    cfg_timestep_buffer,
                    prompt_embeds,
                    transformer_options=model_transformer_options,
                )
                _assert_finite_tensor(
                    v_cond,
                    tensor_name="model_output_cfg_cond",
                    stage_name=stage_name,
                    local_step=step_number,
                    total_steps=total,
                    global_idx=idx,
                    timestep=timestep,
                )
                v_uncond = model(
                    model_state,
                    cfg_timestep_buffer,
                    negative_embeds,
                    transformer_options=model_transformer_options,
                )
                _assert_finite_tensor(
                    v_uncond,
                    tensor_name="model_output_cfg_uncond",
                    stage_name=stage_name,
                    local_step=step_number,
                    total_steps=total,
                    global_idx=idx,
                    timestep=timestep,
                )
                eps_model = cfg_merge(v_uncond, v_cond, effective_cfg_scale)
                _assert_finite_tensor(
                    eps_model,
                    tensor_name="cfg_merge_output",
                    stage_name=stage_name,
                    local_step=step_number,
                    total_steps=total,
                    global_idx=idx,
                    timestep=timestep,
                )

            if eps_model.ndim != 5 or eps_model.shape[0] != state.shape[0] or eps_model.shape[2:] != state.shape[2:]:
                raise RuntimeError(
                    f"WAN22 GGUF: model output shape {tuple(eps_model.shape)} does not match latent state {tuple(state.shape)} "
                    f"(patch_size={geom.patch_kernel} grid={geom.grid})"
                )

            if int(eps_model.shape[1]) != cout:
                raise RuntimeError(
                    f"WAN22 GGUF: model output channels C={int(eps_model.shape[1])} does not match expected latent_channels={cout}."
                )
            if eps_model.dtype == scheduler_state_dtype:
                eps = eps_model
            else:
                if (
                    eps_scheduler_buffer is None
                    or tuple(eps_scheduler_buffer.shape) != tuple(eps_model.shape)
                    or eps_scheduler_buffer.dtype != scheduler_state_dtype
                    or eps_scheduler_buffer.device != eps_model.device
                ):
                    eps_scheduler_buffer = torch.empty(
                        tuple(eps_model.shape),
                        device=eps_model.device,
                        dtype=scheduler_state_dtype,
                    )
                eps_scheduler_buffer.copy_(eps_model)
                eps = eps_scheduler_buffer

            if not has_conditioning:
                out = scheduler.step(model_output=eps, timestep=timestep, sample=state_lat)
                state = out.prev_sample
                _assert_finite_tensor(
                    state,
                    tensor_name="scheduler_prev_sample",
                    stage_name=stage_name,
                    local_step=step_number,
                    total_steps=total,
                    global_idx=idx,
                    timestep=timestep,
                )
            else:
                if state_lat.shape != eps.shape:
                    raise RuntimeError(
                        f"WAN22 GGUF: model output shape {tuple(eps.shape)} does not match latent slice {tuple(state_lat.shape)} "
                        f"(patch_size={geom.patch_kernel} grid={geom.grid})"
                    )

                out = scheduler.step(model_output=eps, timestep=timestep, sample=state_lat)
                lat_next = out.prev_sample
                _assert_finite_tensor(
                    lat_next,
                    tensor_name="scheduler_prev_sample",
                    stage_name=stage_name,
                    local_step=step_number,
                    total_steps=total,
                    global_idx=idx,
                    timestep=timestep,
                )
                if order == "lat_first":
                    state[:, :cout, ...] = lat_next
                else:
                    state[:, -cout:, ...] = lat_next
                _assert_finite_tensor(
                    state,
                    tensor_name="state_out",
                    stage_name=stage_name,
                    local_step=step_number,
                    total_steps=total,
                    global_idx=idx,
                    timestep=timestep,
                )

            if log_numerics_enabled() and idx in parity_idxs:
                log.info(
                    "[wan22.gguf] %s numerics[%d/%d]: %s | %s",
                    stage_name,
                    step_number,
                    total,
                    summarize_numerics(eps, name="eps_step"),
                    summarize_numerics(state, name="state_step"),
                )

        pct = float(local_idx + 1) / float(max(1, total))
        if log_mem_interval is not None:
            n = int(log_mem_interval or 0)
            if n > 0 and ((local_idx + 1) % n) == 0:
                log_cuda_mem(logger, label=f"{stage_name}-step-{local_idx + 1}")

        now = time.perf_counter()
        step_dt = now - last
        elapsed = now - t0
        remain = max(0, total - (local_idx + 1))
        eta = (elapsed / max(1, local_idx + 1)) * remain
        last = now

        if emit_logs and ((local_idx + 1) % 5 == 0 or local_idx + 1 == total):
            log.info("[wan22.gguf] %s step %d/%d (%.1f%%)", stage_name.upper(), local_idx + 1, total, pct * 100.0)

        yield {
            "type": "progress",
            "stage": stage_name,
            "step": local_idx + 1,
            "total": total,
            "percent": pct,
            "eta_seconds": eta,
            "step_seconds": step_dt,
        }

    return state
