"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 GGUF runtime config types and small parsing helpers.
Defines the dataclasses used by the WAN22 GGUF runners (RunConfig/StageConfig) and small env-driven knobs, including
geometry validation (e.g. `height/width % 16 == 0`), metadata-derived sampler/scheduler defaults, strict WAN VAE
config-source contract checks (bundle dir or file+config), plus strict `gguf_sdpa_policy` validation and WAN scheduler contract validation.
Also parses no-stretch img2vid guide controls (`img2vid_image_scale` + normalized crop offsets) into run config and now keeps the exact
WAN 2.2 stage owner truthful: `single` for 5B, `high` + `low` for 14B.

Symbols (top-level; keep in sync; no ghosts):
- `WAN_FLOW_MULTIPLIER` (constant): Multiplier applied to shifted sigma to build the model timestep input.
- `StageConfig` (dataclass): Stage-level configuration (stage model selection + prompt/negative + sampler/scheduler/steps/cfg/flow_shift + ordered LoRA sequence).
- `RunConfig` (dataclass): Full run configuration (geometry, prompts, devices/dtypes, assets, and the truthful exact stage owner: `single` or `high` + `low`).
- `_coerce_int` (function): Best-effort coercion of optional values to `int` (returns `None` on failure).
- `_coerce_float` (function): Best-effort coercion of optional values to `float` (returns `None` on failure).
- `_coerce_bool` (function): Best-effort coercion of optional values to `bool` (returns `None` on failure).
- `_normalize_wan22_sampler_value` (function): Validates WAN22 sampler overrides against real runtime lanes, canonicalizes accepted values, and can enforce metadata-compatible UniPC solver hints.
- `_normalize_wan22_scheduler_value` (function): Validates/canonicalizes WAN22 scheduler values (`simple`) fail-loud.
- `as_torch_dtype` (function): Parses dtype strings into torch dtypes (with validation).
- `resolve_device_name` (function): Normalizes device names (`cuda`/`cpu`/etc) into runtime-compatible values.
- `resolve_i2v_order` (function): Resolves the image-to-video conditioning channel order policy.
- `resolve_wan_flow_multiplier` (function): Resolves WAN timestep multiplier from scheduler metadata (`num_train_timesteps`).
- `build_wan22_gguf_run_config` (function): Builds a validated GGUF `RunConfig` from a request-like object and its extras mapping (including strict VAE path + config-source validation).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
from typing import Any, Mapping, Optional

import torch

from apps.backend.runtime.logging import BackendLoggerProxy, emit_backend_message
from apps.backend.runtime.memory import memory_management
from .paths import normalize_win_path

WAN_FLOW_MULTIPLIER = 1000.0


@dataclass(frozen=True)
class StageConfig:
    model_dir: str
    prompt: Optional[str]
    negative_prompt: Optional[str]
    sampler: str
    scheduler: str
    steps: int
    cfg_scale: Optional[float]
    flow_shift: float
    loras: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RunConfig:
    width: int
    height: int
    fps: int
    num_frames: int
    guidance_scale: Optional[float]
    dtype: str
    device: str
    seed: Optional[int] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    init_image: Optional[object] = None
    vae_dir: Optional[str] = None
    vae_config_dir: Optional[str] = None
    text_encoder_dir: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    metadata_dir: Optional[str] = None
    wan_engine_variant: Optional[str] = None  # exact API dispatch hint ('5b'/'14b' or exact WAN22 engine key)
    single: Optional[StageConfig] = None
    high: Optional[StageConfig] = None
    low: Optional[StageConfig] = None
    # Memory/attention controls (optional)
    sdpa_policy: Optional[str] = None  # 'auto' | 'mem_efficient' | 'flash' | 'math'
    attention_mode: str = "global"  # 'global' | 'sliding'
    attn_chunk_size: Optional[int] = None  # split attention along sequence if set (>0)
    gguf_cache_policy: Optional[str] = None  # dequant cache removed; only 'none' | 'off' accepted
    gguf_cache_limit_mb: Optional[int] = None  # dequant cache removed; must be omitted or 0
    log_mem_interval: Optional[int] = None  # log CUDA mem every N steps if >0
    # Aggressive offload controls
    aggressive_offload: bool = False  # legacy switch; maps to offload_level=2 (balanced) when offload_level is unset
    te_device: Optional[str] = None  # 'cuda' | 'cpu' (None = follow cfg.device)
    # New: coarse-grained offload profile (takes precedence over aggressive_offload if provided)
    # 0 = baseline (no stage-boundary cache clear; stage teardown still unloads stage models),
    # 1 = light (offload TE/VAE only), 2 = balanced (stage cache clear only when pressured),
    # 3 = aggressive (always clear between stages)
    offload_level: Optional[int] = None
    chunk_buffer_mode: str = "hybrid"  # img2vid chunk buffering strategy: 'hybrid' | 'ram' | 'ram+hd'
    img2vid_image_scale: float | None = None  # no-stretch init-image guide scale (>0); None = auto-fit minimum
    img2vid_crop_offset_x: float = 0.5  # normalized crop offset in [0,1]
    img2vid_crop_offset_y: float = 0.5  # normalized crop offset in [0,1]


def as_torch_dtype(dtype: str) -> torch.dtype:
    key = str(dtype or "").strip().lower()
    if key in {"fp16", "float16", "f16"}:
        return torch.float16
    if key in {"bf16", "bfloat16"}:
        return getattr(torch, "bfloat16", torch.float16)
    if key in {"fp32", "float32", "f32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype!r} (expected fp16/bf16/fp32)")


def resolve_device_name(name: str | None) -> str:
    raw = "" if name is None else str(name).strip()
    s = raw.lower()
    manager = getattr(memory_management, "manager", None)
    if manager is None or not hasattr(manager, "mount_device"):
        raise RuntimeError("WAN22: memory manager is required to resolve runtime device.")
    mount_device = manager.mount_device()
    if not isinstance(mount_device, torch.device):
        raise RuntimeError(
            "WAN22: memory manager mount_device() must return torch.device "
            f"(got {type(mount_device).__name__})."
        )
    mount_name = str(mount_device).lower().strip()

    if s in {"cpu"}:
        return "cpu"

    if s in {"auto", ""}:
        if mount_name == "cpu" or mount_name.startswith("cuda"):
            return mount_name
        raise RuntimeError(
            "WAN22: memory manager mount device is unsupported for WAN runtime "
            f"(mount_device={mount_device!s}; expected cpu/cuda)."
        )

    # Accept explicit CUDA device strings (cuda, cuda:0, etc).
    if s == "gpu":
        s = "cuda"
    if s.startswith("cuda"):
        cuda_available = bool(getattr(manager.hardware_probe, "cuda_available", False))
        if cuda_available:
            return s
        raise RuntimeError(
            f"WAN22: device={raw!r} requested but CUDA is unavailable in memory-manager probe; "
            "set device='cpu' explicitly."
        )

    raise ValueError(f"Unsupported device: {raw!r} (expected 'auto', 'cpu', or 'cuda').")


def resolve_i2v_order() -> str:
    """Return channel order for I2V concatenation.

    - 'lat_first': latents(16) then cond extras (mask4+img16).
    - 'lat_last' : cond extras first then latents(16).
    Defaults to 'lat_first'. (Env overrides removed; payload-driven only.)
    """
    return "lat_first"


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        return float(value)
    except Exception:
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _normalize_wan22_sampler_value(
    *,
    field_name: str,
    value: Any,
    expected_unipc_solver_hint: str | None = None,
) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"WAN22 GGUF: {field_name} must be a string, got: {value!r}")
    normalized = value.strip().lower()
    if not normalized:
        raise RuntimeError(f"WAN22 GGUF: {field_name} must not be empty when provided.")
    from apps.backend.types.samplers import SamplerKind

    parts = normalized.split()
    sampler_name = parts[0]
    solver_hint = parts[1] if len(parts) == 2 else None

    if sampler_name == SamplerKind.UNI_PC.value:
        if len(parts) > 2:
            raise RuntimeError(
                f"WAN22 GGUF: {field_name} must be 'uni-pc' or 'uni-pc <solver_hint>', got: {value!r}"
            )
        if solver_hint is None:
            return SamplerKind.UNI_PC.value
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", solver_hint) is None:
            raise RuntimeError(
                f"WAN22 GGUF: {field_name} has invalid UniPC solver hint {solver_hint!r}; "
                "use lowercase [a-z0-9._-] tokens only."
            )
        if expected_unipc_solver_hint is None:
            raise RuntimeError(
                f"WAN22 GGUF: {field_name} solver hint {solver_hint!r} is unsupported; "
                "metadata scheduler has no solver_type."
            )
        if solver_hint != expected_unipc_solver_hint:
            raise RuntimeError(
                f"WAN22 GGUF: {field_name} solver hint mismatch "
                f"(requested={solver_hint!r} metadata={expected_unipc_solver_hint!r})."
            )
        return f"{SamplerKind.UNI_PC.value} {solver_hint}"

    try:
        sampler_kind = SamplerKind.from_string(normalized)
    except Exception as exc:
        raise RuntimeError(
            f"WAN22 GGUF: unsupported {field_name}={value!r}. "
            "Supported WAN22 sampler lanes: 'uni-pc' (optional solver hint), 'euler', 'euler a'."
        ) from exc

    if sampler_kind is SamplerKind.UNI_PC:
        return sampler_kind.value
    if sampler_kind is SamplerKind.EULER:
        return SamplerKind.EULER.value
    if sampler_kind is SamplerKind.EULER_A:
        return SamplerKind.EULER_A.value

    raise RuntimeError(
        f"WAN22 GGUF: unsupported {field_name}={value!r}. "
        "Supported WAN22 sampler lanes: 'uni-pc' (optional solver hint), 'euler', 'euler a'."
    )


def _normalize_wan22_scheduler_value(*, field_name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"WAN22 GGUF: {field_name} must be a string, got: {value!r}")
    normalized = value.strip().lower()
    if not normalized:
        raise RuntimeError(f"WAN22 GGUF: {field_name} must not be empty when provided.")
    if normalized != "simple":
        raise RuntimeError(
            f"WAN22 GGUF: {field_name} must be 'simple', got: {value!r}."
        )
    return normalized


def resolve_wan_flow_multiplier(metadata_dir: str) -> float:
    from .scheduler import load_wan_scheduler_config

    vendor_dir = str(metadata_dir or "").strip()
    if not vendor_dir:
        raise RuntimeError("WAN22 GGUF: cannot resolve flow multiplier without metadata_dir.")
    vendor_dir = os.path.expanduser(vendor_dir)
    if not os.path.isdir(os.path.join(vendor_dir, "scheduler")):
        parent = os.path.dirname(vendor_dir)
        if parent and os.path.isdir(os.path.join(parent, "scheduler")):
            vendor_dir = parent
    cfg = load_wan_scheduler_config(vendor_dir)
    raw = cfg.get("num_train_timesteps")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RuntimeError(
            "WAN22 GGUF: scheduler_config.json must define integer 'num_train_timesteps' for flow multiplier."
        )
    if raw <= 0:
        raise RuntimeError(
            f"WAN22 GGUF: scheduler_config.json has invalid num_train_timesteps={raw!r} (expected > 0)."
        )
    return float(raw)


def build_wan22_gguf_run_config(
    *,
    request: Any,
    device: str | None,
    dtype: str,
    logger: BackendLoggerProxy | None = None,
) -> RunConfig:
    """Build a validated WAN22 GGUF RunConfig from a request-like object.

    Contract: this is a pure mapping layer (no implicit fallbacks, no filesystem guessing).

    Expected `request` attrs (via getattr):
    - prompt / negative_prompt
    - width / height / fps / num_frames / steps / guidance_scale / seed
    - sampler / scheduler
    - init_image (img2vid only)
    - extras: mapping that includes WAN GGUF asset paths and stage overrides
    """
    ex_raw = getattr(request, "extras", {}) or {}
    extras: dict[str, Any] = dict(ex_raw) if isinstance(ex_raw, Mapping) else {}

    raw_wan_engine_variant = extras.get("wan_engine_variant")
    wan_engine_variant: str | None = None
    if raw_wan_engine_variant is not None:
        if not isinstance(raw_wan_engine_variant, str):
            raise RuntimeError(
                "WAN22 GGUF: 'wan_engine_variant' must be a string when provided, "
                f"got {type(raw_wan_engine_variant).__name__}.",
            )
        normalized_variant = raw_wan_engine_variant.strip().lower()
        variant_map = {
            "5b": "5b",
            "14b": "14b",
            "wan22_5b": "5b",
            "wan22_14b": "14b",
            "wan22_14b_animate": "14b",
        }
        wan_engine_variant = variant_map.get(normalized_variant)
        if wan_engine_variant is None:
            allowed = ", ".join(sorted(variant_map))
            raise RuntimeError(
                "WAN22 GGUF: invalid 'wan_engine_variant'="
                f"{raw_wan_engine_variant!r}. Allowed: {allowed}.",
            )

    vae_path = str(extras.get("wan_vae_path") or "").strip() or None

    if extras.get("wan_text_encoder_dir"):
        raise ValueError("WAN22: 'wan_text_encoder_dir' is unsupported in sha-only mode; provide 'wan_text_encoder_path' instead.")

    te_path = str(extras.get("wan_text_encoder_path") or "").strip() or None

    meta_dir = None
    if extras.get("wan_metadata_dir"):
        meta_dir = str(extras.get("wan_metadata_dir") or "").strip() or None
    elif extras.get("wan_tokenizer_dir"):
        # Allow providing tokenizer dir; scheduler_config resolution supports parent fallback.
        meta_dir = str(extras.get("wan_tokenizer_dir") or "").strip() or None

    if not te_path:
        raise RuntimeError(
            "WAN22 GGUF requires a text encoder weights file; provide 'wan_text_encoder_path' (resolved from sha selection)."
        )
    if not vae_path:
        raise RuntimeError(
            "WAN22 GGUF requires a VAE bundle directory; provide 'wan_vae_path' (resolved from sha selection)."
        )
    if not meta_dir:
        raise RuntimeError("WAN22 GGUF requires tokenizer metadata; provide 'wan_metadata_dir' or 'wan_tokenizer_dir'.")

    te_path = os.path.expanduser(te_path)
    te_lower = te_path.lower()
    if not (te_lower.endswith(".safetensors") or te_lower.endswith(".gguf")):
        raise RuntimeError("WAN22 GGUF: 'wan_text_encoder_path' must be a '.safetensors' or '.gguf' file, got: %s" % te_path)
    if not os.path.isfile(te_path):
        raise RuntimeError(f"WAN22 GGUF: text encoder weights not found: {te_path}")

    vae_path = os.path.expanduser(vae_path)
    vae_config_dir: str | None = None
    if os.path.isdir(vae_path):
        config_path = os.path.join(vae_path, "config.json")
        if not os.path.isfile(config_path):
            raise RuntimeError(f"WAN22 GGUF: VAE bundle is missing config.json: {vae_path}")
        weights_candidates = (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
            "model.safetensors",
            "model.bin",
            "pytorch_model.bin",
        )
        if not any(os.path.isfile(os.path.join(vae_path, name)) for name in weights_candidates):
            raise RuntimeError(
                "WAN22 GGUF: VAE bundle is missing weights file "
                f"(expected one of {weights_candidates}) under: {vae_path}"
            )
        vae_config_dir = vae_path
    elif os.path.isfile(vae_path):
        sibling_dir = os.path.dirname(vae_path)
        sibling_config = os.path.join(sibling_dir, "config.json")
        metadata_root = os.path.expanduser(str(meta_dir or ""))
        metadata_candidates = (
            os.path.join(metadata_root, "vae"),
            os.path.join(os.path.dirname(metadata_root), "vae"),
        )
        if os.path.isfile(sibling_config):
            vae_config_dir = sibling_dir
        else:
            for candidate in metadata_candidates:
                if os.path.isfile(os.path.join(candidate, "config.json")):
                    vae_config_dir = candidate
                    break
            if not vae_config_dir:
                raise RuntimeError(
                    "WAN22 GGUF: file VAE path requires config.json at sibling path or metadata repo "
                    f"(missing for VAE file: {vae_path}; checked metadata candidates: {metadata_candidates})."
                )
    else:
        raise RuntimeError(f"WAN22 GGUF: VAE path not found: {vae_path}")

    ws_raw = extras.get("wan_single") if isinstance(extras.get("wan_single"), dict) else None
    wh_raw = extras.get("wan_high") if isinstance(extras.get("wan_high"), dict) else None
    wl_raw = extras.get("wan_low") if isinstance(extras.get("wan_low"), dict) else None

    has_single = isinstance(ws_raw, dict)
    has_high = isinstance(wh_raw, dict)
    has_low = isinstance(wl_raw, dict)
    if has_single:
        if has_high or has_low:
            raise RuntimeError(
                "WAN22 GGUF: mixed stage shape is unsupported; use either 'wan_single' or ('wan_high' + 'wan_low')."
            )
        stage_layout = "single"
    elif has_high or has_low:
        if not (has_high and has_low):
            missing_stage = "wan_high" if not has_high else "wan_low"
            raise RuntimeError(
                f"WAN22 GGUF: 14B stage shape requires both 'wan_high' and 'wan_low' (missing {missing_stage})."
            )
        stage_layout = "dual"
    else:
        raise RuntimeError(
            "WAN22 GGUF requires either 'wan_single' or both 'wan_high' and 'wan_low' in request.extras."
        )

    forbidden = ("lightning", "lora_path", "lora_sha", "lora_weight")
    stage_items = (("wan_single", ws_raw),) if stage_layout == "single" else (("wan_high", wh_raw), ("wan_low", wl_raw))
    for stage_name, stage_cfg in stage_items:
        if not isinstance(stage_cfg, dict):
            continue
        for key in forbidden:
            if stage_cfg.get(key) not in (None, ""):
                if key == "lora_path":
                    raise RuntimeError(
                        f"WAN22 GGUF: '{stage_name}.lora_path' is not supported (use '{stage_name}.loras')."
                    )
                if key in {"lora_sha", "lora_weight"}:
                    raise RuntimeError(
                        f"WAN22 GGUF: '{stage_name}.{key}' is not supported (use '{stage_name}.loras')."
                    )
                raise RuntimeError(f"WAN22 GGUF: '{stage_name}.{key}' is not supported (use Diffusers path).")

    total_steps = int(getattr(request, "steps", 12) or 12)
    if total_steps < 2:
        raise RuntimeError(f"WAN22 GGUF requires steps >= 2, got: {total_steps}")
    default_cfg = getattr(request, "guidance_scale", None)

    vendor_dir = str(meta_dir or "").strip()
    vendor_dir = os.path.expanduser(vendor_dir)
    if not os.path.isdir(os.path.join(vendor_dir, "scheduler")):
        parent = os.path.dirname(vendor_dir)
        if parent and os.path.isdir(os.path.join(parent, "scheduler")):
            vendor_dir = parent
    model_index_path = os.path.join(vendor_dir, "model_index.json")
    if not os.path.isfile(model_index_path):
        raise RuntimeError(f"WAN22 GGUF: missing model_index.json under: {vendor_dir!r}")
    try:
        model_index = json.loads(open(model_index_path, encoding="utf-8").read())
    except Exception as exc:  # noqa: BLE001 - strict decode
        raise RuntimeError(f"WAN22 GGUF: invalid model_index.json under {vendor_dir!r}: {exc}") from exc
    if not isinstance(model_index, dict):
        raise RuntimeError(f"WAN22 GGUF: model_index.json must be a JSON object: {model_index_path}")
    def _has_component(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (list, tuple)):
            return any(item is not None for item in value)
        return True

    class_name = str(model_index.get("_class_name") or "").strip().lower()
    has_transformer_2 = _has_component(model_index.get("transformer_2"))
    has_image_encoder = _has_component(model_index.get("image_encoder"))
    metadata_variant_hint: str | None = None
    if "wananimatepipeline" in class_name or "animate" in str(model_index_path).lower():
        metadata_variant_hint = "14b_animate"
    elif has_image_encoder and not has_transformer_2:
        metadata_variant_hint = "14b_animate"
    elif has_transformer_2:
        metadata_variant_hint = "14b"
    elif model_index.get("expand_timesteps") is not None:
        metadata_variant_hint = "5b"

    boundary_ratio = None
    if stage_layout == "dual":
        boundary_ratio_raw = model_index.get("boundary_ratio")
        if boundary_ratio_raw is None:
            raise RuntimeError(f"WAN22 GGUF: model_index.json missing boundary_ratio: {model_index_path}")
        try:
            boundary_ratio = float(boundary_ratio_raw)
        except Exception as exc:  # noqa: BLE001 - strict parsing
            raise RuntimeError(f"WAN22 GGUF: invalid boundary_ratio={boundary_ratio_raw!r} in {model_index_path}") from exc
        if not (0.0 < boundary_ratio < 1.0):
            raise RuntimeError(
                f"WAN22 GGUF: boundary_ratio must be in (0,1), got {boundary_ratio} in {model_index_path}"
            )

    from apps.backend.runtime.model_registry.flow_shift import flow_shift_spec_from_repo_dir

    default_flow_shift = flow_shift_spec_from_repo_dir(vendor_dir).resolve()
    if stage_layout == "single":
        if metadata_variant_hint is not None and metadata_variant_hint != "5b":
            raise RuntimeError(
                "WAN22 GGUF: single-stage payload requires 5B metadata, "
                f"but metadata hints {metadata_variant_hint!r} under {model_index_path}."
            )
        if wan_engine_variant is not None and wan_engine_variant != "5b":
            raise RuntimeError(
                "WAN22 GGUF: single-stage payload requires requested variant '5b', "
                f"got {raw_wan_engine_variant!r}."
            )
        single_flow_shift_override = None
        if isinstance(ws_raw, dict) and ws_raw.get("flow_shift") is not None:
            single_flow_shift_override = _coerce_float(ws_raw.get("flow_shift"))
            if single_flow_shift_override is None:
                raise RuntimeError(
                    f"WAN22 GGUF: wan_single.flow_shift must be a float, got: {ws_raw.get('flow_shift')!r}"
                )
        effective_flow_shift = (
            float(single_flow_shift_override)
            if single_flow_shift_override is not None
            else float(default_flow_shift)
        )
        default_steps_single = int(total_steps)
    else:
        if metadata_variant_hint is not None and metadata_variant_hint != "14b":
            raise RuntimeError(
                "WAN22 GGUF: dual-stage payload requires 14B metadata, "
                f"but metadata hints {metadata_variant_hint!r} under {model_index_path}."
            )
        if wan_engine_variant is not None and wan_engine_variant != "14b":
            raise RuntimeError(
                "WAN22 GGUF: dual-stage payload requires requested variant '14b', "
                f"got {raw_wan_engine_variant!r}."
            )
        hi_flow_shift_override = None
        if isinstance(wh_raw, dict) and wh_raw.get("flow_shift") is not None:
            hi_flow_shift_override = _coerce_float(wh_raw.get("flow_shift"))
            if hi_flow_shift_override is None:
                raise RuntimeError(
                    f"WAN22 GGUF: wan_high.flow_shift must be a float, got: {wh_raw.get('flow_shift')!r}"
                )
        lo_flow_shift_override = None
        if isinstance(wl_raw, dict) and wl_raw.get("flow_shift") is not None:
            lo_flow_shift_override = _coerce_float(wl_raw.get("flow_shift"))
            if lo_flow_shift_override is None:
                raise RuntimeError(
                    f"WAN22 GGUF: wan_low.flow_shift must be a float, got: {wl_raw.get('flow_shift')!r}"
                )
        if hi_flow_shift_override is not None and lo_flow_shift_override is not None:
            if float(hi_flow_shift_override) != float(lo_flow_shift_override):
                raise RuntimeError(
                    "WAN22 GGUF: high/low flow_shift mismatch. "
                    f"wan_high.flow_shift={hi_flow_shift_override} wan_low.flow_shift={lo_flow_shift_override}. "
                    "Schedule must be continuous."
                )
        if hi_flow_shift_override is not None:
            effective_flow_shift = float(hi_flow_shift_override)
        elif lo_flow_shift_override is not None:
            effective_flow_shift = float(lo_flow_shift_override)
        else:
            effective_flow_shift = float(default_flow_shift)

        hi_steps_override = None
        if isinstance(wh_raw, dict) and wh_raw.get("steps") is not None:
            raise RuntimeError("WAN22 GGUF: wan_high.steps is unsupported; use request.steps.")
        lo_steps_override = None
        if isinstance(wl_raw, dict) and wl_raw.get("steps") is not None:
            lo_steps_override = _coerce_int(wl_raw.get("steps"))
            if lo_steps_override is None:
                raise RuntimeError(f"WAN22 GGUF: wan_low.steps must be an int, got: {wl_raw.get('steps')!r}")

        if hi_steps_override is not None and lo_steps_override is not None:
            default_steps_high = int(hi_steps_override)
            default_steps_low = int(lo_steps_override)
            if default_steps_high < 1 or default_steps_low < 1:
                raise RuntimeError(
                    f"WAN22 GGUF: stage steps must be >= 1 (wan_high.steps={default_steps_high} wan_low.steps={default_steps_low})."
                )
            if (default_steps_high + default_steps_low) != int(total_steps):
                raise RuntimeError(
                    "WAN22 GGUF: stage steps must sum to request.steps for schedule continuity "
                    f"(request.steps={total_steps} wan_high.steps={default_steps_high} wan_low.steps={default_steps_low})."
                )
        elif hi_steps_override is not None:
            default_steps_high = int(hi_steps_override)
            default_steps_low = int(total_steps - default_steps_high)
            if default_steps_high < 1 or default_steps_low < 1:
                raise RuntimeError(
                    "WAN22 GGUF: stage steps must sum to request.steps for schedule continuity "
                    f"(request.steps={total_steps} wan_high.steps={default_steps_high} wan_low.steps={default_steps_low})."
                )
        elif lo_steps_override is not None:
            default_steps_low = int(lo_steps_override)
            default_steps_high = int(total_steps - default_steps_low)
            if default_steps_high < 1 or default_steps_low < 1:
                raise RuntimeError(
                    "WAN22 GGUF: stage steps must sum to request.steps for schedule continuity "
                    f"(request.steps={total_steps} wan_high.steps={default_steps_high} wan_low.steps={default_steps_low})."
                )
        else:
            from .scheduler import infer_high_steps_from_boundary_ratio

            if boundary_ratio is None:
                raise RuntimeError("WAN22 GGUF: dual-stage scheduler split requires metadata boundary_ratio.")
            hi_steps = infer_high_steps_from_boundary_ratio(
                total_steps=total_steps,
                boundary_ratio=boundary_ratio,
                vendor_dir=vendor_dir,
                flow_shift=float(effective_flow_shift),
            )
            default_steps_high = int(hi_steps)
            default_steps_low = int(total_steps - hi_steps)

    def _metadata_sampler_scheduler_defaults() -> tuple[str, str]:
        scheduler_dir = os.path.join(vendor_dir, "scheduler")
        config_path = None
        for filename in ("scheduler_config.json", "config.json"):
            candidate = os.path.join(scheduler_dir, filename)
            if os.path.isfile(candidate):
                config_path = candidate
                break
        if not config_path:
            raise RuntimeError(
                "WAN22 GGUF: missing scheduler config under metadata dir "
                f"(expected '{os.path.join(scheduler_dir, 'scheduler_config.json')}' "
                f"or '{os.path.join(scheduler_dir, 'config.json')}')."
            )
        try:
            scheduler_config = json.loads(open(config_path, encoding="utf-8").read())
        except Exception as exc:  # noqa: BLE001 - strict decode
            raise RuntimeError(f"WAN22 GGUF: invalid scheduler config JSON: {config_path}: {exc}") from exc
        if not isinstance(scheduler_config, dict):
            raise RuntimeError(f"WAN22 GGUF: scheduler config must be a JSON object: {config_path}")
        class_name = str(scheduler_config.get("_class_name") or "").strip()
        if class_name == "UniPCMultistepScheduler":
            raw_solver_type = scheduler_config.get("solver_type")
            if raw_solver_type is None:
                return ("uni-pc", "simple")
            if not isinstance(raw_solver_type, str):
                raise RuntimeError(
                    "WAN22 GGUF: scheduler config solver_type must be a string when provided "
                    f"(path={config_path} value={raw_solver_type!r})."
                )
            solver_type = raw_solver_type.strip().lower()
            if not solver_type:
                return ("uni-pc", "simple")
            return (f"uni-pc {solver_type}", "simple")
        if not class_name:
            raise RuntimeError(f"WAN22 GGUF: scheduler config missing _class_name: {config_path}")
        raise RuntimeError(
            f"WAN22 GGUF: unsupported metadata scheduler {class_name!r} in {config_path}; "
            "expected UniPCMultistepScheduler."
        )

    metadata_sampler_default, metadata_scheduler_default = _metadata_sampler_scheduler_defaults()

    raw_metadata_sampler_parts = str(metadata_sampler_default).strip().lower().split()
    metadata_default_unipc_solver_hint = (
        raw_metadata_sampler_parts[1]
        if len(raw_metadata_sampler_parts) == 2 and raw_metadata_sampler_parts[0] == "uni-pc"
        else None
    )

    metadata_sampler_default = _normalize_wan22_sampler_value(
        field_name="metadata.scheduler_config.sampler_default",
        value=metadata_sampler_default,
        expected_unipc_solver_hint=metadata_default_unipc_solver_hint,
    )
    metadata_scheduler_default = _normalize_wan22_scheduler_value(
        field_name="metadata.scheduler_config.scheduler_default",
        value=metadata_scheduler_default,
    )
    metadata_sampler_parts = str(metadata_sampler_default).strip().lower().split()
    metadata_unipc_solver_hint = (
        metadata_sampler_parts[1]
        if len(metadata_sampler_parts) == 2 and metadata_sampler_parts[0] == "uni-pc"
        else None
    )

    def _stage_opts(
        raw: dict | None,
        *,
        stage: str,
        default_steps: int,
    ) -> tuple[str, Optional[str], Optional[str], int, Optional[float], Optional[str], Optional[str], Optional[float], Optional[int], tuple[tuple[str, float], ...]]:
        if not isinstance(raw, dict):
            raise RuntimeError(f"WAN22 GGUF requires {stage}.model_dir (resolved from model_sha).")
        model_dir = str(raw.get("model_dir") or "").strip()
        if not model_dir:
            raise RuntimeError(f"WAN22 GGUF requires {stage}.model_dir (resolved from model_sha).")
        model_dir = normalize_win_path(os.path.expanduser(model_dir))
        if not model_dir.lower().endswith(".gguf"):
            raise RuntimeError(f"WAN22 GGUF: {stage} model must be a .gguf file, got: {model_dir}")
        if not os.path.isfile(model_dir):
            raise RuntimeError(f"WAN22 GGUF: {stage} model not found: {model_dir}")
        from apps.backend.runtime.model_registry.detectors.wan22 import inspect_wan22_gguf_path
        from apps.backend.runtime.model_registry.specs import ModelFamily

        expected_family = ModelFamily.WAN22_5B if stage == "wan_single" else ModelFamily.WAN22_14B
        try:
            structural_metadata = inspect_wan22_gguf_path(model_dir)
        except Exception as exc:
            raise RuntimeError(f"WAN22 GGUF: failed structural inspection for {stage} model {model_dir}: {exc}") from exc
        if structural_metadata.family != expected_family:
            raise RuntimeError(
                f"WAN22 GGUF: {stage} model family mismatch; expected {expected_family.value}, "
                f"detected {structural_metadata.family.value}: {model_dir}"
            )

        if stage in {"wan_single", "wan_high"}:
            for removed_key in ("prompt", "negative_prompt", "steps", "cfg_scale", "sampler", "scheduler", "seed"):
                if raw.get(removed_key) is not None:
                    raise RuntimeError(
                        f"WAN22 GGUF: {stage}.{removed_key} is unsupported; use the top-level request owner."
                    )
            stage_prompt = str(getattr(request, "prompt", None) or "").strip() or None
            if stage_prompt is None:
                raise RuntimeError(f"WAN22 GGUF: request.prompt must be a non-empty string for {stage}.")
            stage_negative_prompt = str(getattr(request, "negative_prompt", None) or "").strip()
        else:
            raw_prompt = raw.get("prompt")
            if raw_prompt is None:
                stage_prompt = None
            elif isinstance(raw_prompt, str):
                stage_prompt = raw_prompt.strip() or None
            else:
                raise RuntimeError(f"WAN22 GGUF: {stage}.prompt must be a string, got: {raw_prompt!r}")

            raw_negative_prompt = raw.get("negative_prompt")
            if raw_negative_prompt is None:
                stage_negative_prompt = None
            elif isinstance(raw_negative_prompt, str):
                stage_negative_prompt = raw_negative_prompt.strip()
            else:
                raise RuntimeError(
                    f"WAN22 GGUF: {stage}.negative_prompt must be a string, got: {raw_negative_prompt!r}"
                )

        if stage in {"wan_single", "wan_high"}:
            steps = int(default_steps)
            if int(steps) < 1:
                raise RuntimeError(f"WAN22 GGUF: {stage}.steps must be >= 1, got: {steps}")
            cfg_scale = default_cfg
            sampler = None
            scheduler = None
        else:
            raw_steps = raw.get("steps")
            steps = _coerce_int(raw_steps)
            if raw_steps is not None and steps is None:
                raise RuntimeError(f"WAN22 GGUF: {stage}.steps must be an int, got: {raw_steps!r}")
            steps = int(steps) if steps is not None else int(default_steps)
            if int(steps) < 1:
                raise RuntimeError(f"WAN22 GGUF: {stage}.steps must be >= 1, got: {steps}")

            raw_cfg_scale = raw.get("cfg_scale")
            if raw_cfg_scale is None:
                cfg_scale = default_cfg
            else:
                cfg_scale = _coerce_float(raw_cfg_scale)
                if cfg_scale is None:
                    raise RuntimeError(f"WAN22 GGUF: {stage}.cfg_scale must be a float, got: {raw_cfg_scale!r}")

            raw_sampler = raw.get("sampler")
            if raw_sampler is None:
                sampler = None
            else:
                sampler = _normalize_wan22_sampler_value(
                    field_name=f"{stage}.sampler",
                    value=raw_sampler,
                    expected_unipc_solver_hint=metadata_unipc_solver_hint,
                )

            raw_scheduler = raw.get("scheduler")
            if raw_scheduler is None:
                scheduler = None
            else:
                scheduler = _normalize_wan22_scheduler_value(field_name=f"{stage}.scheduler", value=raw_scheduler)

        raw_flow_shift = raw.get("flow_shift")
        if raw_flow_shift is None:
            flow_shift = None
        else:
            flow_shift = _coerce_float(raw_flow_shift)
            if flow_shift is None:
                raise RuntimeError(f"WAN22 GGUF: {stage}.flow_shift must be a float, got: {raw_flow_shift!r}")

        if stage in {"wan_single", "wan_high"}:
            seed = None
        else:
            raw_seed = raw.get("seed")
            if raw_seed is None:
                seed = None
            else:
                seed = _coerce_int(raw_seed)
                if seed is None:
                    raise RuntimeError(f"WAN22 GGUF: {stage}.seed must be an int, got: {raw_seed!r}")

        raw_loras = raw.get("loras")
        loras: list[tuple[str, float]] = []
        if raw_loras is None:
            pass
        elif not isinstance(raw_loras, list):
            raise RuntimeError(f"WAN22 GGUF: {stage}.loras must be an array when provided.")
        else:
            from apps.backend.inventory.scanners.loras import iter_lora_files

            known_lora_paths = {
                os.path.normcase(os.path.realpath(os.path.expanduser(str(path))))
                for path in iter_lora_files()
            }
            if not known_lora_paths:
                raise RuntimeError(
                    f"WAN22 GGUF: {stage}.loras was provided, but no LoRA assets are available in inventory."
                )
            for index, raw_lora in enumerate(raw_loras):
                if not isinstance(raw_lora, dict):
                    raise RuntimeError(f"WAN22 GGUF: {stage}.loras[{index}] must be an object.")
                unknown_lora_keys = sorted(set(raw_lora.keys()) - {"sha", "weight"})
                if unknown_lora_keys:
                    raise RuntimeError(
                        f"WAN22 GGUF: {stage}.loras[{index}] has unexpected key(s): {', '.join(unknown_lora_keys)}."
                    )
                lora_sha = str(raw_lora.get("sha") or "").strip().lower()
                if not lora_sha:
                    raise RuntimeError(f"WAN22 GGUF: {stage}.loras[{index}].sha is required.")
                if not re.fullmatch(r"[0-9a-f]{64}", lora_sha):
                    raise RuntimeError(
                        f"WAN22 GGUF: {stage}.loras[{index}].sha must be sha256 (64 lowercase hex)."
                    )
                from apps.backend.inventory.cache import resolve_asset_by_sha

                resolved = resolve_asset_by_sha(lora_sha)
                if not resolved:
                    raise RuntimeError(f"WAN22 GGUF: {stage}.loras[{index}].sha not found in inventory: {lora_sha}")
                lora_path = normalize_win_path(os.path.expanduser(str(resolved)))
                if not lora_path.lower().endswith(".safetensors"):
                    raise RuntimeError(
                        f"WAN22 GGUF: {stage}.loras[{index}].sha must resolve to a .safetensors file: {lora_sha}"
                    )
                canonical_lora_path = os.path.normcase(os.path.realpath(os.path.expanduser(lora_path)))
                if canonical_lora_path not in known_lora_paths:
                    raise RuntimeError(
                        f"WAN22 GGUF: {stage}.loras[{index}].sha resolved to a non-LoRA asset path: {lora_path}. "
                        "Select a SHA from inventory.loras."
                    )
                if not os.path.isfile(lora_path):
                    raise RuntimeError(f"WAN22 GGUF: {stage} LoRA file not found: {lora_path}")
                raw_weight = raw_lora.get("weight")
                if raw_weight is None:
                    lora_weight = 1.0
                else:
                    if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                        raise RuntimeError(
                            f"WAN22 GGUF: {stage}.loras[{index}].weight must be numeric when provided, got: {raw_weight!r}"
                        )
                    lora_weight = float(raw_weight)
                    if not math.isfinite(lora_weight):
                        raise RuntimeError(
                            f"WAN22 GGUF: {stage}.loras[{index}].weight must be finite, got: {raw_weight!r}"
                        )
                loras.append((lora_path, lora_weight))
        return (
            model_dir,
            stage_prompt,
            stage_negative_prompt,
            steps,
            cfg_scale,
            sampler,
            scheduler,
            flow_shift,
            seed,
            tuple(loras),
        )

    single_dir = single_prompt = single_negative_prompt = single_sampler = single_scheduler = None
    single_steps = single_cfg = single_flow_shift = single_seed = None
    single_loras: tuple[tuple[str, float], ...] = ()
    hi_dir = hi_prompt = hi_negative_prompt = hi_sampler = hi_scheduler = None
    hi_steps = hi_cfg = hi_flow_shift = hi_seed = None
    hi_loras: tuple[tuple[str, float], ...] = ()
    lo_dir = lo_prompt = lo_negative_prompt = lo_sampler = lo_scheduler = None
    lo_steps = lo_cfg = lo_flow_shift = None
    lo_loras: tuple[tuple[str, float], ...] = ()

    if stage_layout == "single":
        (
            single_dir,
            single_prompt,
            single_negative_prompt,
            single_steps,
            single_cfg,
            single_sampler,
            single_scheduler,
            single_flow_shift,
            single_seed,
            single_loras,
        ) = _stage_opts(ws_raw, stage="wan_single", default_steps=default_steps_single)
        single_flow_shift = effective_flow_shift
        explicit_stage_steps = False
    else:
        (
            hi_dir,
            hi_prompt,
            hi_negative_prompt,
            hi_steps,
            hi_cfg,
            hi_sampler,
            hi_scheduler,
            hi_flow_shift,
            hi_seed,
            hi_loras,
        ) = _stage_opts(wh_raw, stage="wan_high", default_steps=default_steps_high)
        (
            lo_dir,
            lo_prompt,
            lo_negative_prompt,
            lo_steps,
            lo_cfg,
            lo_sampler,
            lo_scheduler,
            lo_flow_shift,
            _lo_seed,
            lo_loras,
        ) = _stage_opts(wl_raw, stage="wan_low", default_steps=default_steps_low)

        explicit_stage_steps = bool((wh_raw and wh_raw.get("steps") is not None) or (wl_raw and wl_raw.get("steps") is not None))
        if explicit_stage_steps and (int(hi_steps) + int(lo_steps)) != int(total_steps):
            raise RuntimeError(
                "WAN22 GGUF: stage steps must sum to request.steps for schedule continuity "
                f"(request.steps={total_steps} wan_high.steps={hi_steps} wan_low.steps={lo_steps})."
            )

        hi_flow_shift = effective_flow_shift
        lo_flow_shift = effective_flow_shift

    seed = getattr(request, "seed", None)
    if stage_layout == "dual" and hi_seed is not None:
        seed = hi_seed

    request_sampler = getattr(request, "sampler", None)
    if request_sampler is None:
        sampler_fallback = metadata_sampler_default
    else:
        sampler_fallback = _normalize_wan22_sampler_value(
            field_name="request.sampler",
            value=request_sampler,
            expected_unipc_solver_hint=metadata_unipc_solver_hint,
        )
    request_scheduler = getattr(request, "scheduler", None)
    if request_scheduler is None:
        scheduler_fallback = metadata_scheduler_default
    else:
        scheduler_fallback = _normalize_wan22_scheduler_value(field_name="request.scheduler", value=request_scheduler)

    tokenizer_dir = str(extras.get("wan_tokenizer_dir") or "").strip() or None

    offload_level_raw = extras.get("gguf_offload_level")
    if offload_level_raw is None:
        offload_level = None
    else:
        offload_level = _coerce_int(offload_level_raw)
        if offload_level is None:
            raise RuntimeError(
                "WAN22 GGUF: 'gguf_offload_level' must be an integer when provided, "
                f"got {offload_level_raw!r}."
            )
        if offload_level < 0:
            raise RuntimeError(
                "WAN22 GGUF: 'gguf_offload_level' must be >= 0 when provided, "
                f"got {offload_level!r}."
            )

    if logger is not None:
        try:
            emit_backend_message(
                "[wan22.gguf] assets",
                logger=logger.name,
                metadata=os.path.basename(str(meta_dir)) if meta_dir else None,
                te=os.path.basename(str(te_path)) if te_path else None,
                vae=os.path.basename(str(vae_path)) if vae_path else None,
            )
        except Exception:
            pass

    width = int(getattr(request, "width", 768) or 768)
    height = int(getattr(request, "height", 432) or 432)
    if height % 16 != 0 or width % 16 != 0:
        raise RuntimeError(f"WAN22 GGUF: height and width have to be divisible by 16 but are {height} and {width}.")

    aggressive_offload_raw = extras.get("gguf_offload", False)
    aggressive_offload = _coerce_bool(aggressive_offload_raw)
    if aggressive_offload is None:
        raise RuntimeError(
            f"WAN22 GGUF: 'gguf_offload' must be a boolean when provided, got {aggressive_offload_raw!r}."
        )

    if "gguf_te_impl" in extras:
        raise RuntimeError(
            "WAN22 GGUF: 'gguf_te_impl' was removed. WAN22 text-encoder execution is GGUF-only."
        )
    if "gguf_te_kernel_required" in extras:
        raise RuntimeError(
            "WAN22 GGUF: 'gguf_te_kernel_required' was removed. WAN22 text-encoder execution is GGUF-only."
        )

    attention_mode_raw = extras.get("gguf_attention_mode")
    attention_mode: str = "global"
    if attention_mode_raw is not None:
        attention_mode = str(attention_mode_raw).strip().lower()
        if attention_mode not in {"global", "sliding"}:
            raise RuntimeError(
                "WAN22 GGUF: 'gguf_attention_mode' must be 'global' or 'sliding' when provided, "
                f"got {attention_mode_raw!r}."
            )

    attn_chunk_size = (
        int(extras.get("gguf_attn_chunk", 0))
        if extras.get("gguf_attn_chunk") not in (None, "", 0)
        else None
    )
    if attention_mode == "sliding" and attn_chunk_size is None:
        attn_chunk_size = 1024

    sdpa_policy_raw = extras.get("gguf_sdpa_policy")
    sdpa_policy: str | None = None
    if sdpa_policy_raw is not None:
        sdpa_policy = str(sdpa_policy_raw).strip().lower()
        if sdpa_policy not in {"auto", "mem_efficient", "flash", "math"}:
            raise RuntimeError(
                "WAN22 GGUF: 'gguf_sdpa_policy' must be one of "
                "'auto', 'mem_efficient', 'flash', or 'math' when provided, "
                f"got {sdpa_policy_raw!r}."
            )
    chunk_buffer_mode_raw = extras.get("img2vid_chunk_buffer_mode")
    chunk_buffer_mode = "hybrid"
    if chunk_buffer_mode_raw is not None:
        chunk_buffer_mode = str(chunk_buffer_mode_raw).strip().lower()
        if chunk_buffer_mode not in {"hybrid", "ram", "ram+hd"}:
            raise RuntimeError(
                "WAN22 GGUF: 'img2vid_chunk_buffer_mode' must be one of "
                "('hybrid','ram','ram+hd') when provided, "
                f"got {chunk_buffer_mode_raw!r}."
            )

    image_scale_raw = extras.get("img2vid_image_scale")
    if image_scale_raw is None or image_scale_raw == "":
        img2vid_image_scale: float | None = None
    else:
        if isinstance(image_scale_raw, bool):
            raise RuntimeError(
                "WAN22 GGUF: 'img2vid_image_scale' must be a finite float > 0 when provided, "
                f"got {type(image_scale_raw).__name__}."
            )
        parsed_image_scale = _coerce_float(image_scale_raw)
        if parsed_image_scale is None or not math.isfinite(parsed_image_scale) or parsed_image_scale <= 0.0:
            raise RuntimeError(
                "WAN22 GGUF: 'img2vid_image_scale' must be a finite float > 0 when provided, "
                f"got {image_scale_raw!r}."
            )
        img2vid_image_scale = float(parsed_image_scale)

    def _parse_crop_offset(field_name: str, *, default: float) -> float:
        raw_value = extras.get(field_name)
        if raw_value is None or raw_value == "":
            return float(default)
        parsed = _coerce_float(raw_value)
        if parsed is None or not math.isfinite(parsed):
            raise RuntimeError(
                f"WAN22 GGUF: '{field_name}' must be a finite float in [0,1] when provided, got {raw_value!r}."
            )
        if parsed < 0.0 or parsed > 1.0:
            raise RuntimeError(
                f"WAN22 GGUF: '{field_name}' must be within [0,1] when provided, got {parsed!r}."
            )
        return float(parsed)

    img2vid_crop_offset_x = _parse_crop_offset("img2vid_crop_offset_x", default=0.5)
    img2vid_crop_offset_y = _parse_crop_offset("img2vid_crop_offset_y", default=0.5)

    cache_policy_raw = extras.get("gguf_cache_policy")
    cache_policy: str | None = None
    if cache_policy_raw is not None:
        if not isinstance(cache_policy_raw, str):
            raise RuntimeError(
                "WAN22 GGUF: 'gguf_cache_policy' must be a string when provided, "
                f"got {cache_policy_raw!r}."
            )
        normalized_cache_policy = str(cache_policy_raw).strip().lower()
        if normalized_cache_policy in {"", "none", "off"}:
            cache_policy = "none"
        else:
            raise RuntimeError(
                "WAN22 GGUF: invalid 'gguf_cache_policy'. "
                f"Expected 'none' or 'off', got {cache_policy_raw!r}."
            )

    cache_limit_raw = extras.get("gguf_cache_limit_mb")
    cache_limit_mb: int | None = None
    if cache_limit_raw not in (None, "", 0):
        cache_limit_mb = _coerce_int(cache_limit_raw)
        if cache_limit_mb is None or cache_limit_mb < 0:
            raise RuntimeError(
                "WAN22 GGUF: 'gguf_cache_limit_mb' must be a non-negative integer when provided, "
                f"got {cache_limit_raw!r}."
            )
    if cache_policy is None and cache_limit_mb is not None:
        raise RuntimeError("WAN22 GGUF: 'gguf_cache_limit_mb' requires 'gguf_cache_policy'.")
    if cache_policy == "none" and cache_limit_mb not in (None, 0):
        raise RuntimeError(
            "WAN22 GGUF: 'gguf_cache_limit_mb' must be omitted or 0 when cache policy is 'none'/'off'."
        )

    resolved_single_scheduler = None
    resolved_high_scheduler = None
    resolved_low_scheduler = None
    if stage_layout == "single":
        resolved_single_scheduler = str(single_scheduler or scheduler_fallback).strip().lower()
        if not resolved_single_scheduler:
            raise RuntimeError(
                "WAN22 GGUF: failed to resolve wan_single.scheduler from stage/request/metadata defaults."
            )
    else:
        resolved_high_scheduler = str(hi_scheduler or scheduler_fallback).strip().lower()
        resolved_low_scheduler = str(lo_scheduler or scheduler_fallback).strip().lower()
        if not resolved_high_scheduler:
            raise RuntimeError(
                "WAN22 GGUF: failed to resolve wan_high.scheduler from stage/request/metadata defaults."
            )
        if not resolved_low_scheduler:
            raise RuntimeError(
                "WAN22 GGUF: failed to resolve wan_low.scheduler from stage/request/metadata defaults."
            )

    return RunConfig(
        width=width,
        height=height,
        fps=int(getattr(request, "fps", 24) or 24),
        num_frames=int(getattr(request, "num_frames", 17) or 17),
        guidance_scale=getattr(request, "guidance_scale", None),
        dtype=str(dtype or "fp16"),
        device=resolve_device_name(device),
        seed=seed,
        prompt=getattr(request, "prompt", None),
        negative_prompt=getattr(request, "negative_prompt", None),
        init_image=getattr(request, "init_image", None),
        vae_dir=vae_path,
        vae_config_dir=vae_config_dir,
        text_encoder_dir=te_path,
        tokenizer_dir=tokenizer_dir,
        metadata_dir=meta_dir,
        wan_engine_variant=wan_engine_variant,
        sdpa_policy=sdpa_policy,
        attention_mode=attention_mode,
        attn_chunk_size=attn_chunk_size,
        gguf_cache_policy=cache_policy,
        gguf_cache_limit_mb=cache_limit_mb,
        log_mem_interval=(
            int(extras.get("gguf_log_mem_interval", 0)) if extras.get("gguf_log_mem_interval") not in (None, "", 0) else None
        ),
        aggressive_offload=aggressive_offload,
        offload_level=offload_level,
        chunk_buffer_mode=chunk_buffer_mode,
        img2vid_image_scale=img2vid_image_scale,
        img2vid_crop_offset_x=img2vid_crop_offset_x,
        img2vid_crop_offset_y=img2vid_crop_offset_y,
        te_device=(str(extras.get("gguf_te_device")).lower() if extras.get("gguf_te_device") is not None else None),
        single=(
            StageConfig(
                model_dir=str(single_dir),
                prompt=single_prompt,
                negative_prompt=single_negative_prompt,
                sampler=str(single_sampler or sampler_fallback),
                scheduler=str(resolved_single_scheduler),
                steps=max(1, int(single_steps)),
                cfg_scale=single_cfg,
                flow_shift=float(single_flow_shift),
                loras=single_loras,
            )
            if stage_layout == "single"
            else None
        ),
        high=(
            StageConfig(
                model_dir=str(hi_dir),
                prompt=hi_prompt,
                negative_prompt=hi_negative_prompt,
                sampler=str(hi_sampler or sampler_fallback),
                scheduler=str(resolved_high_scheduler),
                steps=max(1, int(hi_steps)),
                cfg_scale=hi_cfg,
                flow_shift=float(hi_flow_shift),
                loras=hi_loras,
            )
            if stage_layout == "dual"
            else None
        ),
        low=(
            StageConfig(
                model_dir=str(lo_dir),
                prompt=lo_prompt,
                negative_prompt=lo_negative_prompt,
                sampler=str(lo_sampler or sampler_fallback),
                scheduler=str(resolved_low_scheduler),
                steps=max(1, int(lo_steps)),
                cfg_scale=lo_cfg,
                flow_shift=float(lo_flow_shift),
                loras=lo_loras,
            )
            if stage_layout == "dual"
            else None
        ),
    )
