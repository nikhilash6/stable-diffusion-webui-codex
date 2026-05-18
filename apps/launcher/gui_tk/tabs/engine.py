"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Engine settings tab for the Tk launcher.
Owns API-restart sampling, GGUF, LoRA, WAN chunk-buffer, and PyTorch allocator settings with visible VRAM impact badges on residency-affecting controls.

Symbols (top-level; keep in sync; no ghosts):
- `EngineTab` (class): Tk tab for engine/runtime behavior selectors and allocator settings.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Sequence

from apps.launcher.profile_meta import (
    CODEX_CUDA_MALLOC_KEY,
    DEFAULT_PYTORCH_CUDA_ALLOC_CONF,
    ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
)
from apps.launcher.setting_registry import setting_descriptor_for_key, vram_metadata_for_key
from apps.launcher.settings import (
    BoolSetting,
    CFG_BATCH_MODE_CHOICES,
    ChoiceSetting,
    GGUF_DEQUANT_CACHE_CHOICES,
    LORA_APPLY_CHOICES,
    LORA_ONLINE_MATH_CHOICES,
    SettingValidationError,
    WAN22_IMG2VID_CHUNK_BUFFER_MODE_CHOICES,
    normalize_gguf_lora_env,
)

from ..controller import LauncherController
from ..form_renderer import FormRenderer
from ..form_schema import FieldKind, FormFieldDescriptor, FormSectionDescriptor, HelpMode
from .runtime_common import RuntimeFormTabBase


def _descriptor_default(key: str) -> str:
    descriptor = setting_descriptor_for_key(key)
    if descriptor is None:
        raise RuntimeError(f"Launcher setting descriptor missing for {key}")
    return descriptor.default_value()


def _descriptor_bool_default(key: str) -> bool:
    return BoolSetting(key, default=False).parse(_descriptor_default(key))


class EngineTab(RuntimeFormTabBase):
    def __init__(self, controller: LauncherController, *, canvas_bg: str, mark_changed: Callable[[], None]) -> None:
        super().__init__(controller, canvas_bg=canvas_bg, mark_changed=mark_changed)
        self._var_cfg_batch_mode = tk.StringVar()
        self._var_lora_apply_mode = tk.StringVar()
        self._var_gguf_dequant_cache = tk.StringVar()
        self._var_wan_chunk_buffer_mode = tk.StringVar()
        self._var_lora_online_math = tk.StringVar()
        self._var_pytorch_alloc_conf = tk.StringVar()
        self._var_default_alloc_conf_enabled = tk.BooleanVar()
        self._var_cuda_malloc = tk.BooleanVar()
        self._lora_math_combo: ttk.Combobox | None = None
        self._gguf_dequant_cache_combo: ttk.Combobox | None = None

    def _sections_for_view(self) -> Sequence[FormSectionDescriptor]:
        return [
            FormSectionDescriptor(
                title="Sampling / GGUF / LoRA / PyTorch",
                fields=[
                    FormFieldDescriptor(
                        field_id="cfg_batch_mode",
                        kind=FieldKind.CHOICE,
                        label="CFG Cond+Uncond batch mode (requires API restart):",
                        variable=self._var_cfg_batch_mode,
                        choices=list(CFG_BATCH_MODE_CHOICES),
                        on_change=lambda: self._sync_runtime_deps(mark_changed=True),
                        width=12,
                        vram=vram_metadata_for_key("CODEX_CFG_BATCH_MODE"),
                        help_mode=HelpMode.DIALOG,
                        help_title="CFG Cond+Uncond batch mode",
                        help_text=(
                            "Env var: CODEX_CFG_BATCH_MODE\n"
                            "fused: allows compatible cond+uncond CFG work to run in one model batch when memory allows (default).\n"
                            "split: keeps cond and uncond work in separate model calls to lower peak VRAM."
                        ),
                    ),
                    FormFieldDescriptor(
                        field_id="lora_apply_mode",
                        kind=FieldKind.CHOICE,
                        label="LoRA apply mode (requires API restart):",
                        variable=self._var_lora_apply_mode,
                        choices=list(LORA_APPLY_CHOICES),
                        on_change=lambda: self._sync_runtime_deps(mark_changed=True),
                        width=12,
                        vram=vram_metadata_for_key("CODEX_LORA_APPLY_MODE"),
                        help_mode=HelpMode.DIALOG,
                        help_title="LoRA apply mode",
                        help_text=(
                            "online: applies LoRA patches on-the-fly during forward (default).\n"
                            "merge: rewrites weights once at apply-time."
                        ),
                    ),
                    FormFieldDescriptor(
                        field_id="gguf_dequant_cache",
                        kind=FieldKind.CHOICE,
                        label="GGUF dequant cache (requires API restart):",
                        variable=self._var_gguf_dequant_cache,
                        choices=list(GGUF_DEQUANT_CACHE_CHOICES),
                        on_change=lambda: self._sync_runtime_deps(mark_changed=True),
                        width=10,
                        advanced=True,
                        vram=vram_metadata_for_key("CODEX_GGUF_DEQUANT_CACHE"),
                        help_mode=HelpMode.DIALOG,
                        help_title="GGUF dequant cache",
                        help_text="Removed in this build: GGUF dequant run cache levels (lvl1/lvl2).\nValue is locked to 'off'.",
                    ),
                    FormFieldDescriptor(
                        field_id="wan_chunk_buffer_mode",
                        kind=FieldKind.CHOICE,
                        label="WAN img2vid chunk buffer mode (requires API restart):",
                        variable=self._var_wan_chunk_buffer_mode,
                        choices=list(WAN22_IMG2VID_CHUNK_BUFFER_MODE_CHOICES),
                        on_change=lambda: self._sync_runtime_deps(mark_changed=True),
                        width=10,
                        advanced=True,
                        vram=vram_metadata_for_key("CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE"),
                        help_mode=HelpMode.DIALOG,
                        help_title="WAN chunk buffer mode",
                        help_text=(
                            "Env var: CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE\n"
                            "hybrid: auto-select RAM or RAM+disk by chunk memory estimate.\n"
                            "ram: keep chunk buffers only in RAM.\n"
                            "ram+hd: spool chunk buffers to RAM+disk (bounded RAM)."
                        ),
                    ),
                    FormFieldDescriptor(
                        field_id="lora_online_math",
                        kind=FieldKind.CHOICE,
                        label="LoRA online math (requires API restart):",
                        variable=self._var_lora_online_math,
                        choices=list(LORA_ONLINE_MATH_CHOICES),
                        on_change=lambda: self._sync_runtime_deps(mark_changed=True),
                        width=16,
                        advanced=True,
                        vram=vram_metadata_for_key("CODEX_LORA_ONLINE_MATH"),
                        help_mode=HelpMode.DIALOG,
                        help_title="LoRA online math",
                        help_text=(
                            "weight_merge: current online behavior (materializes patched weights per-forward).\n"
                            "activation math is reserved for future packed-kernel LoRA support (not exposed in this build)."
                        ),
                    ),
                    FormFieldDescriptor(
                        field_id="pytorch_alloc_conf",
                        kind=FieldKind.ENTRY,
                        label="PyTorch CUDA alloc conf (requires API restart):",
                        variable=self._var_pytorch_alloc_conf,
                        on_change=self._on_alloc_conf_changed,
                        width=56,
                        advanced=True,
                        vram=vram_metadata_for_key("PYTORCH_CUDA_ALLOC_CONF"),
                        help_mode=HelpMode.DIALOG,
                        help_title="PyTorch alloc conf",
                        help_text=(
                            "Env var: PYTORCH_CUDA_ALLOC_CONF\n"
                            f"Default value: {DEFAULT_PYTORCH_CUDA_ALLOC_CONF}\n"
                            f"Default toggle env: {ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY}"
                        ),
                    ),
                    FormFieldDescriptor(
                        field_id="default_alloc_toggle",
                        kind=FieldKind.CHECK,
                        label="Apply default PyTorch alloc conf when unset (requires API restart):",
                        variable=self._var_default_alloc_conf_enabled,
                        on_change=self._on_default_alloc_conf_toggle_changed,
                        advanced=True,
                        vram=vram_metadata_for_key(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY),
                        help_mode=HelpMode.DIALOG,
                        help_title="Default alloc conf toggle",
                        help_text=(
                            f"Env var: {ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY}\n"
                            "When enabled and PYTORCH_CUDA_ALLOC_CONF is empty, launcher injects the default alloc config."
                        ),
                    ),
                    FormFieldDescriptor(
                        field_id="cuda_malloc_toggle",
                        kind=FieldKind.CHECK,
                        label="Enable cudaMallocAsync backend (requires API restart):",
                        variable=self._var_cuda_malloc,
                        on_change=self._on_cuda_malloc_changed,
                        advanced=True,
                        vram=vram_metadata_for_key(CODEX_CUDA_MALLOC_KEY),
                        help_mode=HelpMode.DIALOG,
                        help_title="cudaMallocAsync backend",
                        help_text=(
                            f"Env var: {CODEX_CUDA_MALLOC_KEY}\n"
                            "When enabled, launcher forwards backend flag '--cuda-malloc'."
                        ),
                    ),
                ],
            ),
        ]

    def _after_render(self, renderer: FormRenderer) -> None:
        gguf_combo = renderer.widget_for("gguf_dequant_cache")
        lora_combo = renderer.widget_for("lora_online_math")
        self._gguf_dequant_cache_combo = gguf_combo if isinstance(gguf_combo, ttk.Combobox) else None
        self._lora_math_combo = lora_combo if isinstance(lora_combo, ttk.Combobox) else None

    def reload(self) -> None:
        env = self._controller.store.env

        def _get(key: str) -> str:
            default = _descriptor_default(key)
            raw = str(env.get(key, default) or "").strip().lower()
            return raw if raw else default

        runtime_settings_sanitized = False
        cfg_batch_setting = ChoiceSetting(
            "CODEX_CFG_BATCH_MODE",
            default=_descriptor_default("CODEX_CFG_BATCH_MODE"),
            choices=CFG_BATCH_MODE_CHOICES,
        )
        try:
            cfg_batch_mode = cfg_batch_setting.get(env)
        except SettingValidationError as exc:
            cfg_batch_mode = _descriptor_default("CODEX_CFG_BATCH_MODE")
            env["CODEX_CFG_BATCH_MODE"] = cfg_batch_mode
            messagebox.showerror("Invalid runtime setting", str(exc))
            runtime_settings_sanitized = True
        self._var_cfg_batch_mode.set(cfg_batch_mode)
        self._var_lora_apply_mode.set(_get("CODEX_LORA_APPLY_MODE"))
        self._var_gguf_dequant_cache.set(_get("CODEX_GGUF_DEQUANT_CACHE"))
        self._var_wan_chunk_buffer_mode.set(_get("CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE"))
        self._var_lora_online_math.set(_get("CODEX_LORA_ONLINE_MATH"))

        default_alloc_setting = BoolSetting(
            ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
            default=_descriptor_bool_default(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY),
        )
        try:
            default_alloc_enabled = default_alloc_setting.get(env)
        except SettingValidationError as exc:
            default_alloc_enabled = _descriptor_bool_default(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY)
            default_alloc_setting.set(env, default_alloc_enabled)
            messagebox.showerror("Invalid runtime setting", str(exc))
            runtime_settings_sanitized = True
        self._var_default_alloc_conf_enabled.set(bool(default_alloc_enabled))

        cuda_malloc_setting = BoolSetting(CODEX_CUDA_MALLOC_KEY, default=_descriptor_bool_default(CODEX_CUDA_MALLOC_KEY))
        try:
            cuda_malloc_enabled = cuda_malloc_setting.get(env)
        except SettingValidationError as exc:
            cuda_malloc_enabled = _descriptor_bool_default(CODEX_CUDA_MALLOC_KEY)
            cuda_malloc_setting.set(env, cuda_malloc_enabled)
            messagebox.showerror("Invalid runtime setting", str(exc))
            runtime_settings_sanitized = True
        self._var_cuda_malloc.set(bool(cuda_malloc_enabled))
        alloc = str(env.get("PYTORCH_CUDA_ALLOC_CONF", "") or "").strip()
        if not alloc and default_alloc_enabled:
            alloc = DEFAULT_PYTORCH_CUDA_ALLOC_CONF
        self._var_pytorch_alloc_conf.set(alloc)

        self._sync_runtime_deps(mark_changed=False)
        if runtime_settings_sanitized:
            self._mark_dirty()
        self._apply_advanced_visibility()

    def _on_alloc_conf_changed(self) -> None:
        key = "PYTORCH_CUDA_ALLOC_CONF"
        value = str(self._var_pytorch_alloc_conf.get() or "").strip()
        if not value:
            try:
                del self._controller.store.env[key]
            except KeyError:
                pass
        else:
            self._controller.store.env[key] = value
        self._mark_dirty()

    def _on_default_alloc_conf_toggle_changed(self) -> None:
        env = self._controller.store.env
        enabled = bool(self._var_default_alloc_conf_enabled.get())
        BoolSetting(
            ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
            default=_descriptor_bool_default(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY),
        ).set(env, enabled)
        if not enabled and "PYTORCH_CUDA_ALLOC_CONF" not in env:
            self._var_pytorch_alloc_conf.set("")
        if enabled and "PYTORCH_CUDA_ALLOC_CONF" not in env and not str(self._var_pytorch_alloc_conf.get() or "").strip():
            self._var_pytorch_alloc_conf.set(DEFAULT_PYTORCH_CUDA_ALLOC_CONF)
        self._mark_dirty()

    def _on_cuda_malloc_changed(self) -> None:
        BoolSetting(CODEX_CUDA_MALLOC_KEY, default=_descriptor_bool_default(CODEX_CUDA_MALLOC_KEY)).set(
            self._controller.store.env,
            bool(self._var_cuda_malloc.get()),
        )
        self._mark_dirty()

    def _sync_runtime_deps(self, *, mark_changed: bool) -> None:
        env = self._controller.store.env
        env["CODEX_CFG_BATCH_MODE"] = (
            str(self._var_cfg_batch_mode.get() or "").strip().lower() or _descriptor_default("CODEX_CFG_BATCH_MODE")
        )
        try:
            cfg_batch_mode = ChoiceSetting(
                "CODEX_CFG_BATCH_MODE",
                default=_descriptor_default("CODEX_CFG_BATCH_MODE"),
                choices=CFG_BATCH_MODE_CHOICES,
            ).get(env)
        except SettingValidationError as exc:
            env["CODEX_CFG_BATCH_MODE"] = _descriptor_default("CODEX_CFG_BATCH_MODE")
            cfg_batch_mode = _descriptor_default("CODEX_CFG_BATCH_MODE")
            messagebox.showerror("Invalid runtime setting", str(exc))
            mark_changed = True
        self._var_cfg_batch_mode.set(cfg_batch_mode)
        env.pop("CODEX_GGUF_EXEC", None)
        env["CODEX_GGUF_DEQUANT_CACHE"] = (
            str(self._var_gguf_dequant_cache.get() or "").strip().lower()
            or _descriptor_default("CODEX_GGUF_DEQUANT_CACHE")
        )
        env.pop("CODEX_GGUF_DEQUANT_CACHE_RATIO", None)
        env.pop("CODEX_GGUF_DEQUANT_CACHE_LIMIT_MB", None)
        env["CODEX_LORA_APPLY_MODE"] = (
            str(self._var_lora_apply_mode.get() or "").strip().lower() or _descriptor_default("CODEX_LORA_APPLY_MODE")
        )
        env["CODEX_LORA_ONLINE_MATH"] = (
            str(self._var_lora_online_math.get() or "").strip().lower() or _descriptor_default("CODEX_LORA_ONLINE_MATH")
        )
        env["CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE"] = (
            str(self._var_wan_chunk_buffer_mode.get() or "").strip().lower()
            or _descriptor_default("CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE")
        )
        try:
            gguf_cache, lora_apply, lora_math, chunk_buffer_mode = normalize_gguf_lora_env(env)
        except SettingValidationError as exc:
            env.pop("CODEX_GGUF_EXEC", None)
            env["CODEX_GGUF_DEQUANT_CACHE"] = _descriptor_default("CODEX_GGUF_DEQUANT_CACHE")
            env.pop("CODEX_GGUF_DEQUANT_CACHE_RATIO", None)
            env.pop("CODEX_GGUF_DEQUANT_CACHE_LIMIT_MB", None)
            env["CODEX_LORA_APPLY_MODE"] = _descriptor_default("CODEX_LORA_APPLY_MODE")
            env["CODEX_LORA_ONLINE_MATH"] = _descriptor_default("CODEX_LORA_ONLINE_MATH")
            env["CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE"] = _descriptor_default("CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE")
            gguf_cache, lora_apply, lora_math, chunk_buffer_mode = normalize_gguf_lora_env(env)
            messagebox.showerror("Invalid runtime setting", str(exc))
            mark_changed = True

        self._var_gguf_dequant_cache.set(gguf_cache)
        self._var_lora_apply_mode.set(lora_apply)
        self._var_lora_online_math.set(lora_math)
        self._var_wan_chunk_buffer_mode.set(chunk_buffer_mode)

        if self._gguf_dequant_cache_combo is not None:
            self._gguf_dequant_cache_combo.configure(state="readonly")
        if self._lora_math_combo is not None:
            self._lora_math_combo.configure(state="readonly" if lora_apply == "online" else "disabled")
        if mark_changed:
            self._mark_dirty()
