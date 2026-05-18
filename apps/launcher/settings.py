"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed launcher settings and validation helpers.
Provides typed env-backed wrappers (no Tk dependency) over choices owned by `setting_registry.py` so UI/service code avoids stringly-typed lookups and scattered normalization rules.
Includes strict normalization for attention bootstrap keys (`CODEX_ATTENTION_BACKEND`, `CODEX_ATTENTION_SDPA_POLICY`) and
task cancel default mode (`CODEX_TASK_CANCEL_DEFAULT_MODE`) alongside task buffer/safety knobs.
GGUF/LoRA normalization resolves missing LoRA apply mode to `online` while preserving explicit `merge` values.

Symbols (top-level; keep in sync; no ghosts):
- `SettingValidationError` (exception): Raised when a launcher setting value is invalid.
- `ChoiceSetting` (dataclass): Typed view over a string setting constrained to a fixed set of choices.
- `BoolSetting` (dataclass): Typed view over a boolean setting serialized as "1"/"0".
- `IntSetting` (dataclass): Typed view over an integer setting serialized as a string.
- `DEVICE_CHOICES` (constant): Allowed values for `CODEX_*_DEVICE`.
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
- `attention_mode_to_backend_policy` (function): Maps launcher UI attention mode (`sdpa_*|xformers|split|quad`) to backend + SDPA policy.
- `backend_policy_to_attention_mode` (function): Maps normalized backend + SDPA policy to launcher UI attention mode.
- `normalize_attention_env` (function): Normalizes attention env keys (`CODEX_ATTENTION_BACKEND`, `CODEX_ATTENTION_SDPA_POLICY`) enforcing cross-setting invariants.
- `normalize_gguf_lora_env` (function): Normalizes GGUF/LoRA/WAN img2vid chunk-buffer env keys enforcing cross-setting invariants.
- `normalize_task_runtime_env` (function): Normalizes task/runtime env keys (single-flight, safeweights, task SSE buffer caps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

from apps.launcher.setting_registry import (
    ATTENTION_BACKEND_CHOICES,
    ATTENTION_SDPA_POLICY_CHOICES,
    CFG_BATCH_MODE_CHOICES,
    DEVICE_CHOICES,
    GGUF_DEQUANT_CACHE_CHOICES,
    LAUNCHER_ATTENTION_MODE_CHOICES,
    LORA_APPLY_CHOICES,
    LORA_ONLINE_MATH_CHOICES,
    TASK_CANCEL_DEFAULT_MODE_CHOICES,
    TASK_EVENT_BUFFER_MAX_EVENTS_DEFAULT,
    TASK_EVENT_BUFFER_MAX_MB_DEFAULT,
    WAN22_IMG2VID_CHUNK_BUFFER_MODE_CHOICES,
    setting_descriptor_for_key,
)


class SettingValidationError(ValueError):
    pass


def _normalize_lower(value: str) -> str:
    return str(value).strip().lower()


@dataclass(frozen=True, slots=True)
class ChoiceSetting:
    key: str
    default: str
    choices: tuple[str, ...]
    normalize: Callable[[str], str] = _normalize_lower

    def parse(self, raw: str | None) -> str:
        if raw is None:
            return self.default
        value = self.normalize(str(raw))
        if not value:
            return self.default
        if value not in self.choices:
            allowed = ", ".join(self.choices)
            raise SettingValidationError(f"{self.key} must be one of: {allowed} (got {raw!r}).")
        return value

    def get(self, env: Mapping[str, str]) -> str:
        return self.parse(env.get(self.key))

    def set(self, env: MutableMapping[str, str], value: str) -> None:
        env[self.key] = self.parse(value)


@dataclass(frozen=True, slots=True)
class BoolSetting:
    key: str
    default: bool = False

    def parse(self, raw: str | None) -> bool:
        if raw is None:
            return bool(self.default)
        value = str(raw).strip().lower()
        if not value:
            return bool(self.default)
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise SettingValidationError(f"{self.key} must be boolean (got {raw!r}).")

    def get(self, env: Mapping[str, str]) -> bool:
        return self.parse(env.get(self.key))

    def set(self, env: MutableMapping[str, str], value: bool) -> None:
        env[self.key] = "1" if bool(value) else "0"


@dataclass(frozen=True, slots=True)
class IntSetting:
    key: str
    default: int
    minimum: int | None = None
    maximum: int | None = None

    def parse(self, raw: str | None) -> int:
        if raw is None:
            return int(self.default)
        s = str(raw).strip()
        if not s:
            return int(self.default)
        try:
            value = int(s)
        except Exception as exc:
            raise SettingValidationError(f"{self.key} must be an integer (got {raw!r}).") from exc
        if self.minimum is not None and value < self.minimum:
            raise SettingValidationError(f"{self.key} must be >= {self.minimum} (got {value}).")
        if self.maximum is not None and value > self.maximum:
            raise SettingValidationError(f"{self.key} must be <= {self.maximum} (got {value}).")
        return value

    def get(self, env: Mapping[str, str]) -> int:
        return self.parse(env.get(self.key))

    def set(self, env: MutableMapping[str, str], value: int) -> None:
        value = int(value)
        if self.minimum is not None and value < self.minimum:
            raise SettingValidationError(f"{self.key} must be >= {self.minimum} (got {value}).")
        if self.maximum is not None and value > self.maximum:
            raise SettingValidationError(f"{self.key} must be <= {self.maximum} (got {value}).")
        env[self.key] = str(value)


def _setting_default(key: str) -> str:
    descriptor = setting_descriptor_for_key(key)
    if descriptor is None:
        raise RuntimeError(f"Launcher setting descriptor missing for {key}")
    return descriptor.default_value()


def _setting_bool_default(key: str) -> bool:
    return BoolSetting(key, default=False).parse(_setting_default(key))


def attention_mode_to_backend_policy(mode: str) -> tuple[str, str]:
    normalized_mode = ChoiceSetting(
        "CODEX_ATTENTION_MODE",
        default=_setting_default("CODEX_ATTENTION_MODE"),
        choices=LAUNCHER_ATTENTION_MODE_CHOICES,
    ).parse(mode)
    if normalized_mode == "sdpa_auto":
        return "pytorch", "auto"
    if normalized_mode == "sdpa_flash":
        return "pytorch", "flash"
    if normalized_mode == "sdpa_mem_efficient":
        return "pytorch", "mem_efficient"
    if normalized_mode == "sdpa_math":
        return "pytorch", "math"
    if normalized_mode in {"xformers", "split", "quad"}:
        return normalized_mode, "auto"
    raise SettingValidationError(f"CODEX_ATTENTION_MODE must be one of: {', '.join(LAUNCHER_ATTENTION_MODE_CHOICES)}.")


def backend_policy_to_attention_mode(backend: str, sdpa_policy: str) -> str:
    normalized_backend = ChoiceSetting(
        "CODEX_ATTENTION_BACKEND",
        default=_setting_default("CODEX_ATTENTION_BACKEND"),
        choices=ATTENTION_BACKEND_CHOICES,
    ).parse(backend)
    normalized_policy = ChoiceSetting(
        "CODEX_ATTENTION_SDPA_POLICY",
        default=_setting_default("CODEX_ATTENTION_SDPA_POLICY"),
        choices=ATTENTION_SDPA_POLICY_CHOICES,
    ).parse(sdpa_policy)
    if normalized_backend == "pytorch":
        if normalized_policy == "flash":
            return "sdpa_flash"
        if normalized_policy == "mem_efficient":
            return "sdpa_mem_efficient"
        if normalized_policy == "math":
            return "sdpa_math"
        return "sdpa_auto"
    return normalized_backend


def normalize_attention_env(env: MutableMapping[str, str]) -> tuple[str, str]:
    backend = ChoiceSetting(
        "CODEX_ATTENTION_BACKEND",
        default=_setting_default("CODEX_ATTENTION_BACKEND"),
        choices=ATTENTION_BACKEND_CHOICES,
    ).get(env)
    sdpa_policy = ChoiceSetting(
        "CODEX_ATTENTION_SDPA_POLICY",
        default=_setting_default("CODEX_ATTENTION_SDPA_POLICY"),
        choices=ATTENTION_SDPA_POLICY_CHOICES,
    ).get(env)
    if backend != "pytorch":
        sdpa_policy = _setting_default("CODEX_ATTENTION_SDPA_POLICY")
    env["CODEX_ATTENTION_BACKEND"] = backend
    env["CODEX_ATTENTION_SDPA_POLICY"] = sdpa_policy
    return backend, sdpa_policy


def normalize_gguf_lora_env(env: MutableMapping[str, str]) -> tuple[str, str, str, str]:
    """Normalize GGUF/LoRA/WAN img2vid chunk-buffer env keys enforcing cross-setting invariants.

    Returns (gguf_dequant_cache, lora_apply_mode, lora_online_math, chunk_buffer_mode) as normalized values.
    """

    # Do not silently coerce reserved values (e.g. activation math mode).
    # Validation must remain fail-loud and aligned with backend flag contracts.

    env.pop("CODEX_GGUF_DEQUANT_CACHE_RATIO", None)
    env.pop("CODEX_GGUF_DEQUANT_CACHE_LIMIT_MB", None)
    env.pop("CODEX_GGUF_EXEC", None)
    gguf_dequant_cache = ChoiceSetting(
        "CODEX_GGUF_DEQUANT_CACHE",
        default=_setting_default("CODEX_GGUF_DEQUANT_CACHE"),
        choices=GGUF_DEQUANT_CACHE_CHOICES,
    ).get(env)
    lora_apply = ChoiceSetting(
        "CODEX_LORA_APPLY_MODE",
        default=_setting_default("CODEX_LORA_APPLY_MODE"),
        choices=LORA_APPLY_CHOICES,
    ).get(env)
    lora_math = ChoiceSetting(
        "CODEX_LORA_ONLINE_MATH",
        default=_setting_default("CODEX_LORA_ONLINE_MATH"),
        choices=LORA_ONLINE_MATH_CHOICES,
    ).get(env)
    chunk_buffer_mode = ChoiceSetting(
        "CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE",
        default=_setting_default("CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE"),
        choices=WAN22_IMG2VID_CHUNK_BUFFER_MODE_CHOICES,
    ).get(env)

    # math only valid on online mode
    if lora_apply != "online":
        lora_math = _setting_default("CODEX_LORA_ONLINE_MATH")

    env["CODEX_GGUF_DEQUANT_CACHE"] = gguf_dequant_cache
    env["CODEX_LORA_APPLY_MODE"] = lora_apply
    env["CODEX_LORA_ONLINE_MATH"] = lora_math
    env["CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE"] = chunk_buffer_mode

    return gguf_dequant_cache, lora_apply, lora_math, chunk_buffer_mode


def normalize_task_runtime_env(env: MutableMapping[str, str]) -> tuple[bool, bool, int, int, str]:
    """Normalize task/runtime env keys and enforce invariants.

    Returns (single_flight, safe_weights, buffer_max_events, buffer_max_mb, cancel_default_mode).
    """

    single_flight_setting = BoolSetting("CODEX_SINGLE_FLIGHT", default=_setting_bool_default("CODEX_SINGLE_FLIGHT"))
    safeweights_setting = BoolSetting("CODEX_SAFE_WEIGHTS", default=_setting_bool_default("CODEX_SAFE_WEIGHTS"))
    max_events_setting = IntSetting(
        "CODEX_TASK_EVENT_BUFFER_MAX_EVENTS",
        default=TASK_EVENT_BUFFER_MAX_EVENTS_DEFAULT,
        minimum=1,
    )
    max_mb_setting = IntSetting(
        "CODEX_TASK_EVENT_BUFFER_MAX_MB",
        default=TASK_EVENT_BUFFER_MAX_MB_DEFAULT,
        minimum=1,
    )
    cancel_default_mode_setting = ChoiceSetting(
        "CODEX_TASK_CANCEL_DEFAULT_MODE",
        default=_setting_default("CODEX_TASK_CANCEL_DEFAULT_MODE"),
        choices=TASK_CANCEL_DEFAULT_MODE_CHOICES,
    )

    single_flight = single_flight_setting.get(env)
    safeweights = safeweights_setting.get(env)
    max_events = max_events_setting.get(env)
    max_mb = max_mb_setting.get(env)
    cancel_default_mode = cancel_default_mode_setting.get(env)

    single_flight_setting.set(env, single_flight)
    safeweights_setting.set(env, safeweights)
    max_events_setting.set(env, max_events)
    max_mb_setting.set(env, max_mb)
    cancel_default_mode_setting.set(env, cancel_default_mode)

    return single_flight, safeweights, max_events, max_mb, cancel_default_mode
