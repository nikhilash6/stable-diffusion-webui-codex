"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Torch-bound sampling inner loop (kept separate so `apps.backend.runtime.sampling` stays import-light for API/UI imports).
Implements conditioning batching, CFG routing (including optional APG/rescale/trunc/renorm guidance policy), and sampling lifecycle hooks
(prepare/cleanup) for native samplers.
Sampling prepare/cleanup delegates generic smart-offload load/unload event emission to the memory manager.
Emits optional profiling sections (torch-profiler `record_function`) at key seams when `CODEX_PROFILE` is enabled.
Supports an opt-in CFG cond+uncond fused batch mode (`CODEX_CFG_BATCH_MODE=fused|split`) that can reduce `apply_model` calls when memory allows,
with a best-effort fallback to split on CUDA OOM.

Symbols (top-level; keep in sync; no ghosts):
- `get_area_and_mult` (function): Computes per-conditioning spatial area crop + mask multiplier (supports `area`, `mask`, `strength`,
  and timestep gates) and returns the prepared slice for batching.
- `cond_equal_size` (function): Checks whether two compiled conditionings are size-compatible for batching.
- `can_concat_cond` (function): Checks whether two conditioning entries can be concatenated into the same UNet batch (area/control/patch compat).
- `cond_cat` (function): Concatenates a list of compiled conditioning dicts into a single dict with canonical keys (`c_crossattn`, `y`, `c_concat`).
- `compute_cond_mark` (function): Builds a cond/uncond mark tensor aligned to the sigma ladder (used for chunked batching/indexing).
- `compute_cond_indices` (function): Computes flat indices for conditional vs unconditional slices in a packed `(batch*sigmas)` tensor layout.
- `calc_cond_uncond_batch` (function): Runs batched UNet calls to compute conditional/unconditional predictions with area masks, memory-aware
  batching, and strict conditioning validation (no fallbacks).
- `sampling_function_inner` (function): Core CFG math and hook routing; handles distilled/turbo `uncond=None`, optional deep debug logs,
  sampler pre/post cfg modifiers, and optional APG/rescale/trunc/renorm policy execution.
- `sampling_function` (function): Wrapper around `sampling_function_inner` for the denoiser interface; applies conditioning modifiers and
  control/image concat plumbing, returning denoised + (cond/uncond) predictions.
- `sampling_prepare` (function): Pre-sampling hook; activates ControlNet runtime, loads required models to GPU, and prepares smart-offload state.
- `sampling_cleanup` (function): Post-sampling hook; cleans up ControlNet, smart-offload state, unloads models, and triggers op cache cleanup.
"""

import torch
import math
import collections
import logging
from typing import Any, Mapping
from apps.backend.runtime.diagnostics.error_summary import summarize_exception_for_console
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.smart_offload import smart_offload_enabled
from apps.backend.runtime import utils
from apps.backend.infra.config.env_flags import env_flag, env_int, env_str
from apps.backend.runtime.diagnostics.profiler import profiler
from .condition import Condition, compile_conditions, compile_weighted_conditions


_DEBUG_LOGGER = get_backend_logger(__name__)

_SAMPLING_INNER_DEBUG_COUNT = 0

_GUIDANCE_POLICY_KEY = "codex_guidance_policy"
_GUIDANCE_STEP_INDEX_KEY = "codex_guidance_step_index"
_GUIDANCE_TOTAL_STEPS_KEY = "codex_guidance_total_steps"
_GUIDANCE_APG_MOMENTUM_BUFFER_KEY = "codex_guidance_apg_momentum_buffer"
_GUIDANCE_WARNED_SAMPLER_CFG_KEY = "codex_guidance_sampler_cfg_warned"


def _rescale_noise_cfg(guided_noise: torch.Tensor, conditional_noise: torch.Tensor, rescale_factor: float) -> torch.Tensor:
    if rescale_factor <= 0.0:
        return guided_noise
    spatial_dims = tuple(range(1, conditional_noise.ndim))
    conditional_std = conditional_noise.std(dim=spatial_dims, keepdim=True)
    guided_std = guided_noise.std(dim=spatial_dims, keepdim=True)
    safe_guided_std = guided_std.clamp(min=1e-12)
    rescaled = guided_noise * (conditional_std / safe_guided_std)
    return rescale_factor * rescaled + (1.0 - rescale_factor) * guided_noise


def _guidance_policy_from_options(model_options: Mapping[str, Any]) -> dict[str, Any] | None:
    policy = model_options.get(_GUIDANCE_POLICY_KEY)
    if policy is None:
        return None
    if not isinstance(policy, dict):
        raise RuntimeError(
            f"Invalid model_options.{_GUIDANCE_POLICY_KEY}: expected object, got {type(policy).__name__}."
        )
    return policy


def _guidance_progress_ratio(model_options: Mapping[str, Any]) -> float:
    raw_step_index = model_options.get(_GUIDANCE_STEP_INDEX_KEY, 0)
    raw_total_steps = model_options.get(_GUIDANCE_TOTAL_STEPS_KEY, 1)
    try:
        step_index = int(raw_step_index)
        total_steps = int(raw_total_steps)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Invalid guidance step metadata in model_options: "
            f"{_GUIDANCE_STEP_INDEX_KEY}={raw_step_index!r} {_GUIDANCE_TOTAL_STEPS_KEY}={raw_total_steps!r}"
        ) from exc
    if total_steps <= 1:
        return 0.0
    return float(max(0.0, min(1.0, step_index / float(total_steps - 1))))


def _apply_apg_guidance(
    cond_pred: torch.Tensor,
    uncond_pred: torch.Tensor,
    cond_scale: float,
    *,
    policy: Mapping[str, Any],
    model_options: dict[str, Any],
) -> torch.Tensor:
    diff = cond_pred - uncond_pred
    dims = tuple(range(1, diff.ndim))

    momentum = float(policy.get("apg_momentum", 0.0) or 0.0)
    if momentum > 0.0:
        previous = model_options.get(_GUIDANCE_APG_MOMENTUM_BUFFER_KEY)
        if not isinstance(previous, torch.Tensor) or previous.shape != diff.shape:
            previous = torch.zeros_like(diff)
        elif previous.dtype != diff.dtype or previous.device != diff.device:
            previous = previous.to(device=diff.device, dtype=diff.dtype)
        diff = diff + momentum * previous
        model_options[_GUIDANCE_APG_MOMENTUM_BUFFER_KEY] = diff.detach()
    else:
        model_options.pop(_GUIDANCE_APG_MOMENTUM_BUFFER_KEY, None)

    norm_threshold = float(policy.get("apg_norm_threshold", 0.0) or 0.0)
    if norm_threshold > 0.0:
        diff_norm = torch.linalg.vector_norm(diff, dim=dims, keepdim=True).clamp(min=1e-12)
        threshold = torch.full_like(diff_norm, fill_value=norm_threshold)
        scale_factor = torch.minimum(torch.ones_like(diff_norm), threshold / diff_norm)
        diff = diff * scale_factor

    cond_unit = torch.nn.functional.normalize(cond_pred.double(), dim=dims)
    diff_double = diff.double()
    diff_parallel = (diff_double * cond_unit).sum(dim=dims, keepdim=True) * cond_unit
    diff_orthogonal = diff_double - diff_parallel
    eta = float(policy.get("apg_eta", 0.0) or 0.0)
    normalized_update = (diff_orthogonal + eta * diff_parallel).to(dtype=diff.dtype)
    guided = uncond_pred + cond_scale * normalized_update

    apg_rescale = float(policy.get("apg_rescale", 0.0) or 0.0)
    if apg_rescale > 0.0:
        guided = _rescale_noise_cfg(guided, cond_pred, apg_rescale)
    return guided


def _apply_guidance_policy(
    cond_pred: torch.Tensor,
    uncond_pred: torch.Tensor,
    *,
    cond_scale: float,
    edit_strength: float,
    model_options: dict[str, Any],
    policy: Mapping[str, Any],
) -> torch.Tensor:
    cond_scale_effective = float(cond_scale) * float(edit_strength)
    cfg_trunc_ratio_raw = policy.get("cfg_trunc_ratio", None)
    cfg_enabled = True
    if cfg_trunc_ratio_raw is not None:
        cfg_trunc_ratio = float(cfg_trunc_ratio_raw)
        if cfg_trunc_ratio < 0.0 or cfg_trunc_ratio > 1.0:
            raise RuntimeError(
                f"Invalid guidance cfg_trunc_ratio={cfg_trunc_ratio!r}; expected range [0, 1]."
            )
        progress_ratio = _guidance_progress_ratio(model_options)
        cfg_enabled = progress_ratio < cfg_trunc_ratio

    if not cfg_enabled or math.isclose(cond_scale_effective, 1.0):
        model_options.pop(_GUIDANCE_APG_MOMENTUM_BUFFER_KEY, None)
        return cond_pred

    step_index_raw = model_options.get(_GUIDANCE_STEP_INDEX_KEY, 0)
    try:
        step_index = int(step_index_raw)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Invalid guidance step index in model_options: {_GUIDANCE_STEP_INDEX_KEY}={step_index_raw!r}"
        ) from exc

    apg_enabled = bool(policy.get("apg_enabled", False))
    apg_start_step = int(policy.get("apg_start_step", 0) or 0)
    if apg_enabled and step_index >= apg_start_step:
        guided = _apply_apg_guidance(
            cond_pred,
            uncond_pred,
            cond_scale_effective,
            policy=policy,
            model_options=model_options,
        )
    else:
        guided = uncond_pred + (cond_pred - uncond_pred) * cond_scale_effective
        model_options.pop(_GUIDANCE_APG_MOMENTUM_BUFFER_KEY, None)
        guidance_rescale = float(policy.get("guidance_rescale", 0.0) or 0.0)
        if guidance_rescale > 0.0:
            guided = _rescale_noise_cfg(guided, cond_pred, guidance_rescale)

    renorm_cfg = float(policy.get("renorm_cfg", 0.0) or 0.0)
    if renorm_cfg > 0.0:
        dims = tuple(range(1, cond_pred.ndim))
        cond_norm = torch.linalg.vector_norm(cond_pred, dim=dims, keepdim=True)
        max_norm = cond_norm * renorm_cfg
        guided_norm = torch.linalg.vector_norm(guided, dim=dims, keepdim=True).clamp(min=1e-12)
        scale_factor = torch.minimum(torch.ones_like(guided_norm), max_norm / guided_norm)
        guided = guided * scale_factor

    return guided


def get_area_and_mult(conds, x_in, timestep_in):
    area = (x_in.shape[2], x_in.shape[3], 0, 0)
    strength = 1.0

    if 'timestep_start' in conds:
        timestep_start = conds['timestep_start']
        if timestep_in[0] > timestep_start:
            return None
    if 'timestep_end' in conds:
        timestep_end = conds['timestep_end']
        if timestep_in[0] < timestep_end:
            return None
    if 'area' in conds:
        area = conds['area']
    if 'strength' in conds:
        strength = conds['strength']

    input_x = x_in[:, :, area[2]:area[0] + area[2], area[3]:area[1] + area[3]]

    if 'mask' in conds:
        mask_strength = 1.0
        if "mask_strength" in conds:
            mask_strength = conds["mask_strength"]
        mask = conds['mask']
        assert (mask.shape[1] == x_in.shape[2])
        assert (mask.shape[2] == x_in.shape[3])
        mask = mask[:, area[2]:area[0] + area[2], area[3]:area[1] + area[3]] * mask_strength
        mask = mask.unsqueeze(1).repeat(input_x.shape[0] // mask.shape[0], input_x.shape[1], 1, 1)
    else:
        mask = torch.ones_like(input_x)
    mult = mask * strength

    if 'mask' not in conds:
        rr = 8
        if mult.shape[2] < rr or mult.shape[3] < rr:
            # Preserve legacy slicing semantics for tiny areas (negative/empty slice edge cases).
            if area[2] != 0:
                for t in range(rr):
                    mult[:, :, t:1 + t, :] *= ((1.0 / rr) * (t + 1))
            if (area[0] + area[2]) < x_in.shape[2]:
                for t in range(rr):
                    mult[:, :, area[0] - 1 - t:area[0] - t, :] *= ((1.0 / rr) * (t + 1))
            if area[3] != 0:
                for t in range(rr):
                    mult[:, :, :, t:1 + t] *= ((1.0 / rr) * (t + 1))
            if (area[1] + area[3]) < x_in.shape[3]:
                for t in range(rr):
                    mult[:, :, :, area[1] - 1 - t:area[1] - t] *= ((1.0 / rr) * (t + 1))
        else:
            ramp = torch.arange(1, rr + 1, device=mult.device, dtype=mult.dtype) / float(rr)
            ramp_h = ramp.view(1, 1, rr, 1)
            ramp_h_reverse = torch.flip(ramp_h, dims=(2,))
            ramp_w = ramp.view(1, 1, 1, rr)
            ramp_w_reverse = torch.flip(ramp_w, dims=(3,))
            if area[2] != 0:
                mult[:, :, :rr, :] *= ramp_h
            if (area[0] + area[2]) < x_in.shape[2]:
                mult[:, :, area[0] - rr:area[0], :] *= ramp_h_reverse
            if area[3] != 0:
                mult[:, :, :, :rr] *= ramp_w
            if (area[1] + area[3]) < x_in.shape[3]:
                mult[:, :, :, area[1] - rr:area[1]] *= ramp_w_reverse

    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area)

    control = conds.get('control', None)

    patches = None
    cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'control', 'patches'])
    return cond_obj(input_x, mult, conditioning, area, control, patches)


def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    if c1.keys() != c2.keys():
        return False
    for k in c1:
        if not c1[k].can_concat(c2[k]):
            return False
    return True


def can_concat_cond(c1, c2):
    if c1.input_x.shape != c2.input_x.shape:
        return False

    def objects_concatable(obj1, obj2):
        if (obj1 is None) != (obj2 is None):
            return False
        if obj1 is not None:
            if obj1 is not obj2:
                return False
        return True

    if not objects_concatable(c1.control, c2.control):
        return False

    if not objects_concatable(c1.patches, c2.patches):
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)


def cond_cat(c_list):
    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out


def compute_cond_mark(cond_or_uncond, sigmas):
    cond_or_uncond_size = int(sigmas.shape[0])
    cond_mark = torch.as_tensor(cond_or_uncond, device=sigmas.device, dtype=sigmas.dtype)
    return cond_mark.repeat_interleave(cond_or_uncond_size)


def compute_cond_indices(cond_or_uncond, sigmas):
    sigma_count = int(sigmas.shape[0])
    if sigma_count <= 0:
        return [], []

    cond_flags = torch.as_tensor(cond_or_uncond, dtype=torch.int64)
    if cond_flags.numel() == 0:
        return [], []

    flat_indices = torch.arange(cond_flags.numel() * sigma_count, dtype=torch.int64)
    flat_indices = flat_indices.view(cond_flags.numel(), sigma_count)
    cond_indices = flat_indices[cond_flags == 0].reshape(-1).tolist()
    uncond_indices = flat_indices[cond_flags != 0].reshape(-1).tolist()
    return cond_indices, uncond_indices


def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
    out_cond = torch.zeros_like(x_in)
    out_count = torch.full_like(x_in, 1e-37)

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.full_like(x_in, 1e-37)

    COND = 0
    UNCOND = 1

    cfg_batch_mode = env_str(
        "CODEX_CFG_BATCH_MODE",
        default="fused",
        allowed={"fused", "split"},
    )
    fused_enabled = cfg_batch_mode == "fused"
    force_fused_retry = env_str(
        "CODEX_CFG_FUSED_FORCE_RETRY",
        default="0",
        allowed={"0", "1"},
    ) == "1"
    fused_disabled_logged = False
    timestep_repeat_cache: dict[int, torch.Tensor] = {}

    to_run = [
        (prepared, COND)
        for prepared in (get_area_and_mult(item, x_in, timestep) for item in cond)
        if prepared is not None
    ]
    if uncond is not None:
        to_run.extend(
            (prepared, UNCOND)
            for prepared in (get_area_and_mult(item, x_in, timestep) for item in uncond)
            if prepared is not None
        )

    def _is_cuda_oom(exc: Exception) -> bool:
        oom_types = []
        for name in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
            t = getattr(torch, name, None)
            if isinstance(t, type):
                oom_types.append(t)
        cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if isinstance(cuda_oom_type, type):
            oom_types.append(cuda_oom_type)
        if oom_types and isinstance(exc, tuple(oom_types)):  # type: ignore[arg-type]
            return True
        msg = str(exc).lower()
        return ("out of memory" in msg) and ("cuda" in msg or "cudnn" in msg)

    def _run_batch(batch_indices: list[int]) -> None:
        # Materialize the batch from to_run without mutating it until the forward succeeds.
        items = [to_run[idx] for idx in batch_indices]

        batch_chunks = len(items)
        input_x = [prepared.input_x for prepared, _flag in items]
        mult = [prepared.mult for prepared, _flag in items]
        c = [prepared.conditioning for prepared, _flag in items]
        area = [prepared.area for prepared, _flag in items]
        cond_or_uncond = [flag for _prepared, flag in items]
        control = items[-1][0].control if items else None
        patches = items[-1][0].patches if items else None
        input_x_cat = torch.cat(input_x)
        c_dict = cond_cat(c)

        # Validate assembled conditioning before UNet call (no fallbacks)
        if 'c_crossattn' not in c_dict or not isinstance(c_dict['c_crossattn'], torch.Tensor) or c_dict['c_crossattn'].ndim != 3:
            raise ValueError(
                f"Missing or invalid 'c_crossattn' for UNet: got type={type(c_dict.get('c_crossattn'))} "
                f"shape={getattr(c_dict.get('c_crossattn'), 'shape', None)} (expected 3D tensor BxSxC)."
            )
        needs_y = getattr(model, 'diffusion_model', None) is not None and getattr(model.diffusion_model, 'num_classes', None) is not None
        if needs_y:
            if 'y' not in c_dict or not isinstance(c_dict['y'], torch.Tensor) or c_dict['y'].ndim != 2:
                raise ValueError(
                    "UNet requires ADM 'y' vector (2D tensor BxV) but it is missing or invalid. "
                    "Ensure SDXL pooled embedding is wired as 'vector' and compiled to 'y'."
                )

        # Align dtype/device for conditioning tensors (often triggers device transfers).
        with profiler.section("sampling.cond_align"):
            target_dtype = getattr(model, 'computation_dtype', None) or input_x_cat.dtype
            dev = input_x_cat.device
            c_dict['c_crossattn'] = c_dict['c_crossattn'].to(dtype=target_dtype, device=dev)
            if 'y' in c_dict and isinstance(c_dict['y'], torch.Tensor):
                c_dict['y'] = c_dict['y'].to(device=dev)
            if 'c_concat' in c_dict and isinstance(c_dict['c_concat'], torch.Tensor):
                c_dict['c_concat'] = c_dict['c_concat'].to(device=dev)
            if batch_chunks <= 1:
                timestep_ = timestep
            else:
                timestep_ = timestep_repeat_cache.get(batch_chunks)
                if timestep_ is None:
                    if timestep.shape[0] == 1:
                        timestep_ = timestep.expand((batch_chunks,) + tuple(timestep.shape[1:]))
                    else:
                        timestep_repeat_shape = (batch_chunks,) + (1,) * max(timestep.ndim - 1, 0)
                        timestep_ = timestep.repeat(timestep_repeat_shape)
                    timestep_repeat_cache[batch_chunks] = timestep_

        transformer_options = {}
        if 'transformer_options' in model_options:
            transformer_options = model_options['transformer_options'].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        transformer_options["cond_mark"] = compute_cond_mark(cond_or_uncond=cond_or_uncond, sigmas=timestep)
        transformer_options["cond_indices"], transformer_options["uncond_indices"] = compute_cond_indices(cond_or_uncond=cond_or_uncond, sigmas=timestep)

        if _DEBUG_LOGGER.isEnabledFor(logging.DEBUG):
            emit_backend_message(
                "Control batch",
                logger=__name__,
                level=logging.DEBUG,
                size=len(cond_or_uncond),
                cond=sum(1 for flag in cond_or_uncond if flag == COND),
                uncond=sum(1 for flag in cond_or_uncond if flag != COND),
                sigma_shape=tuple(timestep.shape),
            )

        c_dict['transformer_options'] = transformer_options

        if control:
            control.set_transformer_options(transformer_options)
            control_cond = c_dict.copy()  # get_control may change items in this dict, so we need to copy it
            with profiler.section("sampling.controlnet.get_control"):
                try:
                    c_dict['control'] = control.get_control(input_x_cat, timestep_, control_cond, len(cond_or_uncond))
                except Exception as err:
                    emit_backend_message(
                        "ControlNet get_control failed",
                        logger=__name__,
                        level="ERROR",
                        error=summarize_exception_for_console(err),
                    )
                    raise
            c_dict['control_model'] = control

        with profiler.section("sampling.apply_model"):
            if 'model_function_wrapper' in model_options:
                output = model_options['model_function_wrapper'](
                    model.apply_model,
                    {"input": input_x_cat, "timestep": timestep_, "c": c_dict, "cond_or_uncond": cond_or_uncond},
                ).chunk(batch_chunks)
            else:
                output = model.apply_model(input_x_cat, timestep_, **c_dict).chunk(batch_chunks)

        for o in range(batch_chunks):
            if cond_or_uncond[o] == COND:
                out_cond[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_count[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += mult[o]
            else:
                out_uncond[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_uncond_count[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += mult[o]

        # Remove processed items only after success.
        for idx in sorted(batch_indices, reverse=True):
            to_run.pop(idx)

    def _batch_flags(batch_indices: list[int]) -> set[int]:
        return {int(to_run[idx][1]) for idx in batch_indices}

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        if not fused_enabled and len(to_batch_temp) > 1:
            first_flag = int(to_run[to_batch_temp[0]][1])
            to_batch_temp = [idx for idx in to_batch_temp if int(to_run[idx][1]) == first_flag]
        to_batch = to_batch_temp[:1]

        if memory_management.manager.signal_empty_cache:
            memory_management.manager.soft_empty_cache(force=True)

        free_memory = memory_management.manager.get_free_memory(x_in.device)

        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[:len(to_batch_temp) // i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        if force_fused_retry and fused_enabled and len(to_batch_temp) == 2 and len(to_batch) < 2:
            # Best-effort fused CFG batch: try cond+uncond in one forward even when memory heuristics say "no".
            # Disabled by default; enable only via CODEX_CFG_FUSED_FORCE_RETRY=1.
            # If it OOMs, fall back to the existing split path for this run.
            flags = _batch_flags(to_batch_temp)
            if flags == {COND, UNCOND}:
                try:
                    _run_batch(to_batch_temp)
                    continue
                except Exception as exc:
                    if not _is_cuda_oom(exc):
                        raise
                    if not fused_disabled_logged:
                        emit_backend_message(
                            "[cfg-batch] fused attempt OOM; falling back to split. Try reducing resolution/CFG or disabling fused CFG batching.",
                            logger=__name__,
                            level="WARNING",
                            mode=cfg_batch_mode,
                        )
                        fused_disabled_logged = True
                    fused_enabled = False
                    try:
                        memory_management.manager.soft_empty_cache()
                    except Exception:
                        pass

        try:
            _run_batch(to_batch)
        except Exception as exc:
            if not fused_enabled or not _is_cuda_oom(exc):
                raise
            if _batch_flags(to_batch) != {COND, UNCOND}:
                raise
            if not fused_disabled_logged:
                emit_backend_message(
                    "[cfg-batch] fused batch OOM; falling back to split. Try reducing resolution/CFG or disabling fused CFG batching.",
                    logger=__name__,
                    level="WARNING",
                    mode=cfg_batch_mode,
                )
                fused_disabled_logged = True
            fused_enabled = False
            try:
                memory_management.manager.soft_empty_cache()
            except Exception:
                pass
            continue

    out_cond /= out_count
    del out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    return out_cond, out_uncond


def sampling_function_inner(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None, return_full=False):
    edit_strength = sum((item['strength'] if 'strength' in item else 1) for item in cond)

    if math.isclose(cond_scale, 1.0) and not model_options.get("disable_cfg1_optimization", False):
        uncond_ = None
    else:
        uncond_ = uncond

    for fn in model_options.get("sampler_pre_cfg_function", []):
        model, cond, uncond_, x, timestep, model_options = fn(model, cond, uncond_, x, timestep, model_options)

    with profiler.section("sampling.calc_cond_uncond_batch"):
        cond_pred, uncond_pred = calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)

    # Optional deep diagnostics for flow models (Z Image/Flux): log CFG routing and tensor norms.
    global _SAMPLING_INNER_DEBUG_COUNT
    debug_enabled = env_flag("CODEX_SAMPLING_DEBUG") or env_flag("CODEX_SAMPLING_DEBUG_INNER")
    debug_limit = env_int("CODEX_SAMPLING_DEBUG_INNER_N", 3, min_value=0)
    if debug_enabled and _SAMPLING_INNER_DEBUG_COUNT < debug_limit:
        try:
            sigma0 = float(timestep.detach().view(-1)[0].item()) if isinstance(timestep, torch.Tensor) else float(timestep)
        except Exception:
            sigma0 = float("nan")
        try:
            cond_norm = float(cond_pred.detach().float().norm().item()) if isinstance(cond_pred, torch.Tensor) else float("nan")
        except Exception:
            cond_norm = float("nan")
        try:
            uncond_norm = float(uncond_pred.detach().float().norm().item()) if isinstance(uncond_pred, torch.Tensor) else float("nan")
        except Exception:
            uncond_norm = float("nan")
        emit_backend_message(
            "[sampling-debug] sampling_inner",
            logger=__name__,
            sigma=sigma0,
            cond_scale=float(cond_scale),
            edit_strength=float(edit_strength),
            uncond_present=uncond_ is not None,
            cond_norm=cond_norm,
            uncond_norm=uncond_norm,
        )
        _SAMPLING_INNER_DEBUG_COUNT += 1

    # Distilled / turbo models may omit unconditional conditioning entirely.
    # In that case, skip CFG math and return the conditional prediction as-is.
    if uncond_ is None:
        model_options.pop(_GUIDANCE_APG_MOMENTUM_BUFFER_KEY, None)
        cfg_result = cond_pred
    elif "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        guidance_policy = _guidance_policy_from_options(model_options)
        if guidance_policy is None:
            if not math.isclose(edit_strength, 1.0):
                cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * edit_strength
            else:
                cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale
            model_options.pop(_GUIDANCE_APG_MOMENTUM_BUFFER_KEY, None)
        else:
            cfg_result = _apply_guidance_policy(
                cond_pred,
                uncond_pred,
                cond_scale=cond_scale,
                edit_strength=edit_strength,
                model_options=model_options,
                policy=guidance_policy,
            )
    if "sampler_cfg_function" in model_options and _guidance_policy_from_options(model_options) is not None:
        if not bool(model_options.get(_GUIDANCE_WARNED_SAMPLER_CFG_KEY, False)):
            emit_backend_message(
                "Guidance policy is ignored because sampler_cfg_function is active.",
                logger=__name__,
                level="WARNING",
            )
            model_options[_GUIDANCE_WARNED_SAMPLER_CFG_KEY] = True

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    if return_full:
        return cfg_result, cond_pred, uncond_pred

    return cfg_result


def sampling_function(self, denoiser_params, cond_scale, cond_composition):
    denoiser_patcher = self.inner_model.inner_model.codex_objects.denoiser
    model = denoiser_patcher.model
    control = getattr(denoiser_patcher, "controlnet_linked_list", None)
    extra_concat_condition = getattr(denoiser_patcher, "extra_concat_condition", None)
    x = denoiser_params.x
    timestep = denoiser_params.sigma
    uncond = compile_conditions(denoiser_params.text_uncond)
    cond = compile_weighted_conditions(denoiser_params.text_cond, cond_composition)
    model_options = denoiser_patcher.model_options
    seed = self.p.seeds[0]

    if extra_concat_condition is not None:
        image_cond_in = extra_concat_condition
    else:
        image_cond_in = denoiser_params.image_cond

    if isinstance(image_cond_in, torch.Tensor):
        if image_cond_in.shape[0] == x.shape[0] \
                and image_cond_in.shape[2] == x.shape[2] \
                and image_cond_in.shape[3] == x.shape[3]:
            if uncond is not None:
                for i in range(len(uncond)):
                    uncond[i]['model_conds']['c_concat'] = Condition(image_cond_in)
            for i in range(len(cond)):
                cond[i]['model_conds']['c_concat'] = Condition(image_cond_in)

    if control:
        for h in cond:
            h['control'] = control
    if control and uncond is not None:
        for h in uncond:
            h['control'] = control

    for modifier in model_options.get('conditioning_modifiers', []):
        model, x, timestep, uncond, cond, cond_scale, model_options, seed = modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed)

    denoised, cond_pred, uncond_pred = sampling_function_inner(model, x, timestep, uncond, cond, cond_scale, model_options, seed, return_full=True)
    return denoised, cond_pred, uncond_pred


def sampling_prepare(denoiser, x):
    B, C, H, W = x.shape

    memory_estimation_function = denoiser.model_options.get('memory_peak_estimation_modifier', denoiser.memory_required)

    denoiser_inference_memory = memory_estimation_function([B * 2, C, H, W])
    additional_inference_memory = int(getattr(denoiser, "extra_preserved_memory_during_sampling", 0) or 0)
    additional_model_patchers = list(getattr(denoiser, "extra_model_patchers_during_sampling", []) or [])

    control_runtime = denoiser.activate_control() if hasattr(denoiser, "activate_control") else None
    real_model = denoiser.model

    def percent_to_timestep_function(p):  # type: ignore[no-untyped-def]
        return real_model.predictor.percent_to_sigma(p)

    setattr(denoiser, "_codex_smart_offload_models", [])
    try:
        if control_runtime:
            control_runtime.prepare(real_model, percent_to_timestep_function)
            additional_inference_memory += control_runtime.inference_memory_requirements(denoiser.model_dtype())
            additional_model_patchers += control_runtime.get_models()
            if _DEBUG_LOGGER.isEnabledFor(logging.DEBUG):
                emit_backend_message(
                    "Control runtime activated",
                    logger=__name__,
                    level=logging.DEBUG,
                    extra_memory=additional_inference_memory,
                    models=len(additional_model_patchers),
                )

        if denoiser.has_online_lora():
            lora_memory = utils.nested_compute_size(
                denoiser.lora_patches,
                element_size=utils.dtype_to_element_size(denoiser.model.computation_dtype),
            )
            additional_inference_memory += lora_memory

        models_to_load = [denoiser] + additional_model_patchers
        memory_management.manager.load_models(
            models=models_to_load,
            memory_required=denoiser_inference_memory,
            hard_memory_preservation=additional_inference_memory,
            source="runtime.sampling.inner_loop.sampling_prepare",
            stage="sampling_prepare",
        )

        if smart_offload_enabled():
            setattr(denoiser, "_codex_smart_offload_models", models_to_load)

        if denoiser.has_online_lora():
            utils.nested_move_to_device(
                denoiser.lora_patches,
                device=denoiser.current_device,
                dtype=denoiser.model.computation_dtype,
            )

        if control_runtime and _DEBUG_LOGGER.isEnabledFor(logging.DEBUG):
            emit_backend_message(
                "Control runtime prepared",
                logger=__name__,
                level=logging.DEBUG,
                model_type=type(real_model).__name__,
            )
    except Exception as exc:
        try:
            sampling_cleanup(denoiser)
        except Exception as cleanup_exc:
            raise RuntimeError(
                "sampling_prepare failed and cleanup failed; runtime residency may be inconsistent "
                f"(prepare_error={exc}, cleanup_error={cleanup_exc})"
            ) from cleanup_exc
        raise

    return


def sampling_cleanup(denoiser):
    if denoiser.has_online_lora():
        utils.nested_move_to_device(denoiser.lora_patches, device=denoiser.offload_device)
    control_runtime = getattr(denoiser, "controlnet_linked_list", None)
    if control_runtime:
        control_runtime.cleanup()
        if _DEBUG_LOGGER.isEnabledFor(logging.DEBUG):
            emit_backend_message(
                "Control runtime cleaned up after sampling",
                logger=__name__,
                level=logging.DEBUG,
            )
    if hasattr(denoiser, "clear_control"):
        denoiser.clear_control()
    if smart_offload_enabled():
        models_to_unload = getattr(denoiser, "_codex_smart_offload_models", [])
        for model in models_to_unload:
            memory_management.manager.unload_model(
                model,
                source="runtime.sampling.inner_loop.sampling_cleanup",
                stage="sampling_cleanup",
            )
        setattr(denoiser, "_codex_smart_offload_models", [])
    from apps.backend.runtime.ops import cleanup_cache
    cleanup_cache()
    return
