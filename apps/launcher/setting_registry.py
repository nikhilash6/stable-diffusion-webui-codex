"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical launcher setting registry for env defaults, choices, ownership, and visible VRAM impact metadata.
Owns launcher-managed setting descriptors so profile defaults, Tk forms, and the Docker TUI consume one source of truth for shared runtime selectors.

Symbols (top-level; keep in sync; no ghosts):
- `VramImpactLevel` (class): Three-level visible VRAM impact scale (`low`, `medium`, `high`).
- `VramImpactMetadata` (dataclass): Per-setting static or value-sensitive VRAM impact metadata.
- `LauncherSettingDescriptor` (dataclass): Canonical metadata for one launcher-managed env setting.
- `DEVICE_CHOICES` (constant): Allowed values for launcher device selectors.
- `CFG_BATCH_MODE_CHOICES` (constant): Allowed values for `CODEX_CFG_BATCH_MODE`.
- `TASK_EVENT_BUFFER_MAX_EVENTS_DEFAULT` (constant): Default max SSE events buffered per task.
- `TASK_EVENT_BUFFER_MAX_MB_DEFAULT` (constant): Default max SSE MB buffered per task.
- `TASK_CANCEL_DEFAULT_MODE_CHOICES` (constant): Allowed values for `CODEX_TASK_CANCEL_DEFAULT_MODE`.
- `ATTENTION_BACKEND_CHOICES` (constant): Allowed values for `CODEX_ATTENTION_BACKEND`.
- `ATTENTION_SDPA_POLICY_CHOICES` (constant): Allowed values for `CODEX_ATTENTION_SDPA_POLICY`.
- `LAUNCHER_ATTENTION_MODE_CHOICES` (constant): Allowed launcher UI attention mode values.
- `GGUF_DEQUANT_CACHE_CHOICES` (constant): Allowed values for `CODEX_GGUF_DEQUANT_CACHE`.
- `WAN22_IMG2VID_CHUNK_BUFFER_MODE_CHOICES` (constant): Allowed values for `CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE`.
- `LORA_APPLY_CHOICES` (constant): Allowed values for `CODEX_LORA_APPLY_MODE`.
- `LORA_ONLINE_MATH_CHOICES` (constant): Allowed values for `CODEX_LORA_ONLINE_MATH`.
- `DEFAULT_SETTING_DESCRIPTORS` (constant): Ordered launcher-managed descriptors used to build default env maps.
- `SETTING_DESCRIPTORS_BY_KEY` (constant): Descriptor lookup by env key.
- `launcher_default_core_env` (function): Builds default `areas/core` env values from the registry.
- `setting_descriptor_for_key` (function): Returns the descriptor for an env key.
- `vram_metadata_for_key` (function): Returns visible VRAM impact metadata for an env key when defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import os
from typing import Mapping, Sequence

from apps.launcher.profile_meta import (
    CODEX_CUDA_MALLOC_KEY,
    DEFAULT_PYTORCH_CUDA_ALLOC_CONF,
    ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
)


class VramImpactLevel(StrEnum):
    """Visible launcher VRAM impact levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def badge_text(self) -> str:
        if self is VramImpactLevel.LOW:
            return "VRAM: LOW"
        if self is VramImpactLevel.MEDIUM:
            return "VRAM: MED"
        return "VRAM: HIGH"

    @property
    def style_name(self) -> str:
        if self is VramImpactLevel.LOW:
            return "Vram.Low.TLabel"
        if self is VramImpactLevel.MEDIUM:
            return "Vram.Medium.TLabel"
        return "Vram.High.TLabel"


@dataclass(frozen=True, slots=True)
class VramImpactMetadata:
    """Static or value-sensitive visible VRAM impact metadata."""

    static: VramImpactLevel | None = None
    by_value: Mapping[str, VramImpactLevel] = field(default_factory=dict)

    def resolve(self, value: object | None) -> VramImpactLevel | None:
        if self.static is not None:
            return self.static
        text_value = str(value or "").strip().lower()
        if text_value in self.by_value:
            return self.by_value[text_value]
        normalized = _normalize_vram_value(value)
        return self.by_value.get(normalized)


@dataclass(frozen=True, slots=True)
class LauncherSettingDescriptor:
    """Canonical metadata for one launcher-managed env setting."""

    key: str
    owner_tab: str
    default: str
    choices: Sequence[str] = ()
    default_from_environment: bool = False
    persist_default: bool = True
    restart_required: bool = True
    advanced: bool = False
    vram: VramImpactMetadata | None = None

    def default_value(self) -> str:
        if not self.default_from_environment:
            return self.default
        return str(os.getenv(self.key, self.default))


DEVICE_CHOICES: tuple[str, ...] = ("auto", "cuda", "cpu", "mps", "xpu", "directml")
CFG_BATCH_MODE_CHOICES: tuple[str, ...] = ("fused", "split")
TASK_EVENT_BUFFER_MAX_EVENTS_DEFAULT = 5000
TASK_EVENT_BUFFER_MAX_MB_DEFAULT = 64
TASK_CANCEL_DEFAULT_MODE_CHOICES: tuple[str, ...] = ("immediate", "after_current")
ATTENTION_BACKEND_CHOICES: tuple[str, ...] = ("pytorch", "xformers", "split", "quad")
ATTENTION_SDPA_POLICY_CHOICES: tuple[str, ...] = ("auto", "flash", "mem_efficient", "math")
LAUNCHER_ATTENTION_MODE_CHOICES: tuple[str, ...] = (
    "sdpa_auto",
    "sdpa_flash",
    "sdpa_mem_efficient",
    "sdpa_math",
    "xformers",
    "split",
    "quad",
)
GGUF_DEQUANT_CACHE_CHOICES: tuple[str, ...] = ("off",)
WAN22_IMG2VID_CHUNK_BUFFER_MODE_CHOICES: tuple[str, ...] = ("hybrid", "ram", "ram+hd")
LORA_APPLY_CHOICES: tuple[str, ...] = ("merge", "online")
LORA_ONLINE_MATH_CHOICES: tuple[str, ...] = ("weight_merge",)

_ENABLED_HIGH_VRAM: Mapping[str, VramImpactLevel] = {
    "enabled": VramImpactLevel.HIGH,
    "disabled": VramImpactLevel.LOW,
}

_ENABLED_MEDIUM_VRAM: Mapping[str, VramImpactLevel] = {
    "enabled": VramImpactLevel.MEDIUM,
    "disabled": VramImpactLevel.LOW,
}

_SINGLE_FLIGHT_VRAM: Mapping[str, VramImpactLevel] = {
    "enabled": VramImpactLevel.LOW,
    "disabled": VramImpactLevel.HIGH,
}

_DEVICE_VRAM: Mapping[str, VramImpactLevel] = {
    "auto": VramImpactLevel.MEDIUM,
    "cuda": VramImpactLevel.HIGH,
    "cpu": VramImpactLevel.LOW,
    "mps": VramImpactLevel.HIGH,
    "xpu": VramImpactLevel.HIGH,
    "directml": VramImpactLevel.HIGH,
}


DEFAULT_SETTING_DESCRIPTORS: tuple[LauncherSettingDescriptor, ...] = (
    LauncherSettingDescriptor("CODEX_PIPELINE_DEBUG", "diagnostics", "0", default_from_environment=True),
    LauncherSettingDescriptor(
        "CODEX_CFG_BATCH_MODE",
        "engine",
        "fused",
        choices=CFG_BATCH_MODE_CHOICES,
        default_from_environment=True,
        vram=VramImpactMetadata(
            by_value={
                "fused": VramImpactLevel.HIGH,
                "split": VramImpactLevel.LOW,
            }
        ),
    ),
    LauncherSettingDescriptor(
        "CODEX_VAE_TENSOR_STATS",
        "diagnostics",
        "0",
        default_from_environment=True,
        vram=VramImpactMetadata(by_value=_ENABLED_HIGH_VRAM),
    ),
    LauncherSettingDescriptor(
        "CODEX_MEMORY_DEBUG",
        "diagnostics",
        "0",
        default_from_environment=True,
        vram=VramImpactMetadata(by_value=_ENABLED_HIGH_VRAM),
    ),
    LauncherSettingDescriptor(
        "CODEX_SINGLE_FLIGHT",
        "safety",
        "1",
        default_from_environment=True,
        vram=VramImpactMetadata(by_value=_SINGLE_FLIGHT_VRAM),
    ),
    LauncherSettingDescriptor(
        "CODEX_TASK_EVENT_BUFFER_MAX_EVENTS",
        "safety",
        str(TASK_EVENT_BUFFER_MAX_EVENTS_DEFAULT),
        default_from_environment=True,
        advanced=True,
    ),
    LauncherSettingDescriptor(
        "CODEX_TASK_EVENT_BUFFER_MAX_MB",
        "safety",
        str(TASK_EVENT_BUFFER_MAX_MB_DEFAULT),
        default_from_environment=True,
        advanced=True,
    ),
    LauncherSettingDescriptor(
        "CODEX_TASK_CANCEL_DEFAULT_MODE",
        "safety",
        "immediate",
        choices=TASK_CANCEL_DEFAULT_MODE_CHOICES,
        default_from_environment=True,
    ),
    LauncherSettingDescriptor("CODEX_SAFE_WEIGHTS", "safety", "0", default_from_environment=True),
    LauncherSettingDescriptor("CODEX_PROFILE", "diagnostics", "0", default_from_environment=True),
    LauncherSettingDescriptor("CODEX_PROFILE_TRACE", "diagnostics", "1", default_from_environment=True),
    LauncherSettingDescriptor(
        "CODEX_PROFILE_RECORD_SHAPES",
        "diagnostics",
        "0",
        default_from_environment=True,
        vram=VramImpactMetadata(by_value=_ENABLED_MEDIUM_VRAM),
    ),
    LauncherSettingDescriptor(
        "CODEX_PROFILE_PROFILE_MEMORY",
        "diagnostics",
        "1",
        default_from_environment=True,
        vram=VramImpactMetadata(by_value=_ENABLED_MEDIUM_VRAM),
    ),
    LauncherSettingDescriptor(
        "CODEX_PROFILE_WITH_STACK",
        "diagnostics",
        "0",
        default_from_environment=True,
        vram=VramImpactMetadata(by_value=_ENABLED_HIGH_VRAM),
    ),
    LauncherSettingDescriptor("CODEX_PROFILE_TOP_N", "diagnostics", "25", default_from_environment=True),
    LauncherSettingDescriptor("CODEX_PROFILE_MAX_STEPS", "diagnostics", "0", default_from_environment=True),
    LauncherSettingDescriptor("CODEX_TRACE_INFERENCE_DEBUG", "diagnostics", "0", default_from_environment=True),
    LauncherSettingDescriptor("CODEX_TRACE_LOAD_PATCH_DEBUG", "diagnostics", "0", default_from_environment=True),
    LauncherSettingDescriptor("CODEX_TRACE_CALL_DEBUG", "diagnostics", "0", default_from_environment=True),
    LauncherSettingDescriptor(
        "CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC",
        "diagnostics",
        "10",
        default_from_environment=True,
    ),
    LauncherSettingDescriptor(
        "CODEX_MAIN_DEVICE",
        "bootstrap",
        "auto",
        choices=DEVICE_CHOICES,
        vram=VramImpactMetadata(static=VramImpactLevel.HIGH),
    ),
    LauncherSettingDescriptor(
        "CODEX_MOUNT_DEVICE",
        "bootstrap",
        "auto",
        choices=DEVICE_CHOICES,
        vram=VramImpactMetadata(static=VramImpactLevel.HIGH),
    ),
    LauncherSettingDescriptor(
        "CODEX_OFFLOAD_DEVICE",
        "bootstrap",
        "cpu",
        choices=DEVICE_CHOICES,
        vram=VramImpactMetadata(
            by_value={
                "auto": VramImpactLevel.MEDIUM,
                "cuda": VramImpactLevel.HIGH,
                "cpu": VramImpactLevel.LOW,
                "mps": VramImpactLevel.HIGH,
                "xpu": VramImpactLevel.HIGH,
                "directml": VramImpactLevel.HIGH,
            }
        ),
    ),
    LauncherSettingDescriptor("CODEX_CORE_DEVICE", "bootstrap", "auto", choices=DEVICE_CHOICES),
    LauncherSettingDescriptor("CODEX_TE_DEVICE", "bootstrap", "auto", choices=DEVICE_CHOICES),
    LauncherSettingDescriptor("CODEX_VAE_DEVICE", "bootstrap", "auto", choices=DEVICE_CHOICES),
    LauncherSettingDescriptor(
        "CODEX_GGUF_DEQUANT_CACHE",
        "engine",
        "off",
        choices=GGUF_DEQUANT_CACHE_CHOICES,
        advanced=True,
        vram=VramImpactMetadata(by_value={"off": VramImpactLevel.LOW}),
    ),
    LauncherSettingDescriptor(
        "CODEX_ATTENTION_BACKEND",
        "bootstrap",
        "pytorch",
        choices=ATTENTION_BACKEND_CHOICES,
        vram=VramImpactMetadata(static=VramImpactLevel.MEDIUM),
    ),
    LauncherSettingDescriptor(
        "CODEX_ATTENTION_SDPA_POLICY",
        "bootstrap",
        "auto",
        choices=ATTENTION_SDPA_POLICY_CHOICES,
        vram=VramImpactMetadata(static=VramImpactLevel.MEDIUM),
    ),
    LauncherSettingDescriptor(
        "CODEX_ATTENTION_MODE",
        "bootstrap",
        "sdpa_auto",
        choices=LAUNCHER_ATTENTION_MODE_CHOICES,
        persist_default=False,
        vram=VramImpactMetadata(static=VramImpactLevel.MEDIUM),
    ),
    LauncherSettingDescriptor(
        "CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE",
        "engine",
        "hybrid",
        choices=WAN22_IMG2VID_CHUNK_BUFFER_MODE_CHOICES,
        default_from_environment=True,
        advanced=True,
        vram=VramImpactMetadata(
            by_value={
                "hybrid": VramImpactLevel.MEDIUM,
                "ram": VramImpactLevel.HIGH,
                "ram+hd": VramImpactLevel.LOW,
            }
        ),
    ),
    LauncherSettingDescriptor(
        "CODEX_LORA_APPLY_MODE",
        "engine",
        "online",
        choices=LORA_APPLY_CHOICES,
        vram=VramImpactMetadata(
            by_value={
                "online": VramImpactLevel.MEDIUM,
                "merge": VramImpactLevel.HIGH,
            }
        ),
    ),
    LauncherSettingDescriptor(
        "CODEX_LORA_ONLINE_MATH",
        "engine",
        "weight_merge",
        choices=LORA_ONLINE_MATH_CHOICES,
        advanced=True,
        vram=VramImpactMetadata(by_value={"weight_merge": VramImpactLevel.MEDIUM}),
    ),
    LauncherSettingDescriptor(
        "PYTORCH_CUDA_ALLOC_CONF",
        "engine",
        DEFAULT_PYTORCH_CUDA_ALLOC_CONF,
        persist_default=False,
        advanced=True,
        vram=VramImpactMetadata(static=VramImpactLevel.MEDIUM),
    ),
    LauncherSettingDescriptor(
        ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
        "engine",
        "1",
        advanced=True,
        vram=VramImpactMetadata(by_value=_ENABLED_MEDIUM_VRAM),
    ),
    LauncherSettingDescriptor(
        CODEX_CUDA_MALLOC_KEY,
        "engine",
        "0",
        advanced=True,
        vram=VramImpactMetadata(by_value=_ENABLED_MEDIUM_VRAM),
    ),
    LauncherSettingDescriptor("CODEX_DEBUG_COND", "diagnostics", "0", persist_default=False),
    LauncherSettingDescriptor("CODEX_LOG_SAMPLER", "diagnostics", "0", persist_default=False),
    LauncherSettingDescriptor("CODEX_LOG_CFG_DELTA", "diagnostics", "0", persist_default=False),
    LauncherSettingDescriptor("CODEX_LOG_SIGMAS", "diagnostics", "0", persist_default=False),
    LauncherSettingDescriptor("CODEX_DUMP_LATENTS", "diagnostics", "0", persist_default=False),
    LauncherSettingDescriptor("CODEX_TIMELINE", "diagnostics", "0", persist_default=False),
    LauncherSettingDescriptor("CODEX_TRACE_CONTRACT", "diagnostics", "0", persist_default=False),
    LauncherSettingDescriptor("CODEX_TRACE_PROFILER", "diagnostics", "0", persist_default=False),
    LauncherSettingDescriptor("CODEX_LOG_CFG_DELTA_N", "diagnostics", "2", persist_default=False),
    LauncherSettingDescriptor("CODEX_DUMP_LATENTS_PATH", "diagnostics", "", persist_default=False),
    LauncherSettingDescriptor("CODEX_LOG_DEBUG", "diagnostics", "0", persist_default=False),
    LauncherSettingDescriptor("CODEX_LOG_INFO", "diagnostics", "1", persist_default=False),
    LauncherSettingDescriptor("CODEX_LOG_WARNING", "diagnostics", "1", persist_default=False),
    LauncherSettingDescriptor("CODEX_LOG_ERROR", "diagnostics", "1", persist_default=False),
    LauncherSettingDescriptor("CODEX_LOG_FILE", "diagnostics", "", persist_default=False),
)

SETTING_DESCRIPTORS_BY_KEY: Mapping[str, LauncherSettingDescriptor] = {
    descriptor.key: descriptor for descriptor in DEFAULT_SETTING_DESCRIPTORS
}


def launcher_default_core_env() -> dict[str, str]:
    """Build the default launcher `areas/core` env map from persisted registry descriptors."""
    return {
        descriptor.key: descriptor.default_value()
        for descriptor in DEFAULT_SETTING_DESCRIPTORS
        if descriptor.persist_default
    }


def setting_descriptor_for_key(key: str) -> LauncherSettingDescriptor | None:
    return SETTING_DESCRIPTORS_BY_KEY.get(str(key))


def vram_metadata_for_key(key: str) -> VramImpactMetadata | None:
    descriptor = setting_descriptor_for_key(key)
    return descriptor.vram if descriptor is not None else None


def _normalize_vram_value(value: object | None) -> str:
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return "enabled"
    if text in {"0", "false", "no", "off"}:
        return "disabled"
    return text
