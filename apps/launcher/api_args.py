"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher API backend CLI argument builder.
Converts validated launcher env snapshots into backend `run_api.py` CLI flags without owning subprocess lifecycle or profile persistence.

Symbols (top-level; keep in sync; no ghosts):
- `api_backend_args_from_env` (function): Converts launcher env values into backend CLI args.
"""

from __future__ import annotations

from typing import List, Mapping

from apps.launcher.service_env import env_truthy
from apps.launcher.settings import DEVICE_CHOICES


def api_backend_args_from_env(env: Mapping[str, str]) -> List[str]:
    args: List[str] = []
    allowed_devices = set(DEVICE_CHOICES)

    def _append_device_arg(*, env_key: str, flag: str, fallback: str = "") -> None:
        raw_value = str(env.get(env_key, "") or "").strip().lower()
        if not raw_value:
            raw_value = str(fallback or "").strip().lower()
        if not raw_value:
            return
        if raw_value not in allowed_devices:
            allowed = ", ".join(sorted(allowed_devices))
            raise ValueError(f"{env_key} must be one of: {allowed} (got {raw_value!r}).")
        args.append(f"--{flag}={raw_value}")

    raw_main_device = str(env.get("CODEX_MAIN_DEVICE", "") or "").strip().lower()
    if raw_main_device:
        if raw_main_device not in allowed_devices:
            allowed = ", ".join(sorted(allowed_devices))
            raise ValueError(f"CODEX_MAIN_DEVICE must be one of: {allowed} (got {raw_main_device!r}).")
        args.append(f"--main-device={raw_main_device}")
        args.append(f"--core-device={raw_main_device}")
        args.append(f"--te-device={raw_main_device}")
        args.append(f"--vae-device={raw_main_device}")
    else:
        raw_core_device = str(env.get("CODEX_CORE_DEVICE", "") or "").strip().lower()
        if raw_core_device:
            if raw_core_device not in allowed_devices:
                allowed = ", ".join(sorted(allowed_devices))
                raise ValueError(f"CODEX_CORE_DEVICE must be one of: {allowed} (got {raw_core_device!r}).")
            args.append(f"--core-device={raw_core_device}")

        raw_te_device = str(env.get("CODEX_TE_DEVICE", "") or "").strip().lower()
        if raw_te_device:
            if raw_te_device not in allowed_devices:
                allowed = ", ".join(sorted(allowed_devices))
                raise ValueError(f"CODEX_TE_DEVICE must be one of: {allowed} (got {raw_te_device!r}).")
            args.append(f"--te-device={raw_te_device}")

        raw_vae_device = str(env.get("CODEX_VAE_DEVICE", "") or "").strip().lower()
        if raw_vae_device:
            if raw_vae_device not in allowed_devices:
                allowed = ", ".join(sorted(allowed_devices))
                raise ValueError(f"CODEX_VAE_DEVICE must be one of: {allowed} (got {raw_vae_device!r}).")
            args.append(f"--vae-device={raw_vae_device}")

    _append_device_arg(env_key="CODEX_MOUNT_DEVICE", flag="mount-device", fallback=raw_main_device)
    _append_device_arg(env_key="CODEX_OFFLOAD_DEVICE", flag="offload-device", fallback="cpu")

    raw_attention_backend = str(env.get("CODEX_ATTENTION_BACKEND", "") or "").strip().lower()
    if raw_attention_backend:
        if raw_attention_backend not in {"pytorch", "xformers", "split", "quad"}:
            raise ValueError(
                "CODEX_ATTENTION_BACKEND must be one of: pytorch, xformers, split, quad "
                f"(got {raw_attention_backend!r}).",
            )
        args.append(f"--attention-backend={raw_attention_backend}")
        raw_sdpa_policy = str(env.get("CODEX_ATTENTION_SDPA_POLICY", "") or "").strip().lower()
        if raw_attention_backend == "pytorch" and raw_sdpa_policy:
            if raw_sdpa_policy not in {"auto", "flash", "mem_efficient", "math"}:
                raise ValueError(
                    "CODEX_ATTENTION_SDPA_POLICY must be one of: auto, flash, mem_efficient, math "
                    f"(got {raw_sdpa_policy!r}).",
                )
            args.append(f"--attention-sdpa-policy={raw_sdpa_policy}")

    raw_lora_mode = str(env.get("CODEX_LORA_APPLY_MODE", "") or "").strip().lower()
    if raw_lora_mode:
        args.append(f"--lora-apply-mode={raw_lora_mode}")

    raw_lora_math = str(env.get("CODEX_LORA_ONLINE_MATH", "") or "").strip().lower()
    if raw_lora_math:
        args.append(f"--lora-online-math={raw_lora_math}")

    if env_truthy(env.get("CODEX_CUDA_MALLOC")):
        args.append("--cuda-malloc")

    if env_truthy(env.get("CODEX_TRACE_CONTRACT")):
        args.append("--trace-contract")
    if env_truthy(env.get("CODEX_TRACE_PROFILER")):
        args.append("--trace-profiler")

    return args
