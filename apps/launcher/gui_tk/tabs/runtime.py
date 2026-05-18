"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime settings tab for the Tk launcher.
Edits bootstrap-critical main-device defaults and global runtime/task knobs that must exist before the API starts (main/mount/offload devices, CFG batching, GGUF/LoRA,
task single-flight, task cancel default mode, task SSE buffer caps, upscaler safeweights). Offload device defaults to CPU on missing/invalid values to preserve explicit de-residency semantics.
LoRA apply mode resolves missing launcher values to `online` while preserving explicit `merge` selections.
Allocator defaults are managed through `PYTORCH_CUDA_ALLOC_CONF` and `CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF`.
API-only manual env overlay ownership lives in `Manual Env Vars`, not in runtime selectors.

Symbols (top-level; keep in sync; no ghosts):
- `RuntimeTab` (class): Runtime settings tab (device defaults + attention/CFG batch mode + LoRA + `PYTORCH_CUDA_ALLOC_CONF`/cuda-malloc toggles).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from apps.launcher.profiles import (
    CODEX_CUDA_MALLOC_KEY,
    DEFAULT_PYTORCH_CUDA_ALLOC_CONF,
    ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
)
from apps.launcher.settings import (
    BoolSetting,
    CFG_BATCH_MODE_CHOICES,
    ChoiceSetting,
    DEVICE_CHOICES,
    IntSetting,
    WAN22_IMG2VID_CHUNK_BUFFER_MODE_CHOICES,
    SettingValidationError,
    TASK_CANCEL_DEFAULT_MODE_CHOICES,
    TASK_EVENT_BUFFER_MAX_EVENTS_DEFAULT,
    TASK_EVENT_BUFFER_MAX_MB_DEFAULT,
    attention_mode_to_backend_policy,
    backend_policy_to_attention_mode,
    normalize_gguf_lora_env,
    normalize_attention_env,
    normalize_task_runtime_env,
)

from ..controller import LauncherController
from ..form_renderer import FormRenderer
from ..form_schema import FieldKind, FormFieldDescriptor, FormSectionDescriptor, HelpMode
from ..widgets import ScrollableFrame


_ATTENTION_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("SDPA - Auto", "sdpa_auto"),
    ("SDPA - Flash", "sdpa_flash"),
    ("SDPA - Mem Efficient", "sdpa_mem_efficient"),
    ("SDPA - Math", "sdpa_math"),
    ("xFormers", "xformers"),
    ("Split (Chunked)", "split"),
    ("Quad (Sub-Quadratic)", "quad"),
)
_ATTENTION_LABEL_TO_MODE = {label: mode for label, mode in _ATTENTION_MODE_OPTIONS}
_ATTENTION_MODE_TO_LABEL = {mode: label for label, mode in _ATTENTION_MODE_OPTIONS}


class RuntimeTab:
    def __init__(
        self,
        controller: LauncherController,
        *,
        canvas_bg: str,
        mark_changed: Callable[[], None],
        section: str,
    ) -> None:
        self._controller = controller
        self._canvas_bg = canvas_bg
        self._mark_changed = mark_changed
        normalized_section = str(section or "").strip().lower()
        if normalized_section not in {"bootstrap", "engine", "safety"}:
            raise ValueError(f"Unknown runtime section: {section!r}")
        self._section = normalized_section

        self.frame: ttk.Frame | None = None

        self._var_main_device = tk.StringVar()
        self._var_mount_device = tk.StringVar()
        self._var_offload_device = tk.StringVar()
        self._var_attention_mode = tk.StringVar()

        self._var_cfg_batch_mode = tk.StringVar()
        self._var_lora_apply_mode = tk.StringVar()
        self._var_gguf_dequant_cache = tk.StringVar()
        self._var_wan_chunk_buffer_mode = tk.StringVar()
        self._var_lora_online_math = tk.StringVar()
        self._var_pytorch_alloc_conf = tk.StringVar()
        self._var_default_alloc_conf_enabled = tk.BooleanVar()
        self._var_cuda_malloc = tk.BooleanVar()
        self._var_single_flight = tk.BooleanVar()
        self._var_safeweights = tk.BooleanVar()
        self._var_task_cancel_default_mode = tk.StringVar()
        self._var_task_buffer_max_events = tk.StringVar()
        self._var_task_buffer_max_mb = tk.StringVar()
        self._advanced_visible = False

        self._lora_math_combo: ttk.Combobox | None = None
        self._gguf_dequant_cache_combo: ttk.Combobox | None = None
        self._form_renderer: FormRenderer | None = None

    def build(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        scroll = ScrollableFrame(frame, canvas_bg=self._canvas_bg)
        scroll.pack(fill="both", expand=True, padx=8, pady=8)
        body = scroll.inner
        body.columnconfigure(0, weight=1)

        renderer = FormRenderer(body)
        sections = self._sections_for_view()
        renderer.render_sections(0, sections)
        self._form_renderer = renderer

        gguf_combo = renderer.widget_for("gguf_dequant_cache")
        lora_combo = renderer.widget_for("lora_online_math")
        self._gguf_dequant_cache_combo = gguf_combo if isinstance(gguf_combo, ttk.Combobox) else None
        self._lora_math_combo = lora_combo if isinstance(lora_combo, ttk.Combobox) else None
        self._apply_advanced_visibility()

        self.frame = frame
        self.reload()
        return frame

    def _sections_for_view(self) -> list[FormSectionDescriptor]:
        if self._section == "bootstrap":
            return [
                FormSectionDescriptor(
                    title="Main Device (bootstrap)",
                    fields=[
                        FormFieldDescriptor(
                            field_id="main_device",
                            kind=FieldKind.CHOICE,
                            label="Main device (requires API restart):",
                            variable=self._var_main_device,
                            choices=list(DEVICE_CHOICES),
                            on_change=self._on_main_device_changed,
                            help_mode=HelpMode.DIALOG,
                            help_title="Main device",
                            help_text=(
                                "Backend flag: --main-device.\n"
                                "Launcher mirrors main device to core/TE/VAE for invariant bootstrap behavior."
                            ),
                        ),
                        FormFieldDescriptor(
                            field_id="mount_device",
                            kind=FieldKind.CHOICE,
                            label="Mount device (requires API restart):",
                            variable=self._var_mount_device,
                            choices=list(DEVICE_CHOICES),
                            on_change=self._on_mount_device_changed,
                            help_mode=HelpMode.DIALOG,
                            help_title="Mount device",
                            help_text=(
                                "Backend flag: --mount-device.\n"
                                "When unset, mount defaults to main device."
                            ),
                        ),
                        FormFieldDescriptor(
                            field_id="offload_device",
                            kind=FieldKind.CHOICE,
                            label="Offload device (requires API restart):",
                            variable=self._var_offload_device,
                            choices=list(DEVICE_CHOICES),
                            on_change=self._on_offload_device_changed,
                            help_mode=HelpMode.DIALOG,
                            help_title="Offload device",
                            help_text=(
                                "Backend flag: --offload-device.\n"
                                "Default is CPU.\n"
                                "Offload cannot match non-CPU mount device."
                            ),
                        ),
                        FormFieldDescriptor(
                            field_id="attention_mode",
                            kind=FieldKind.CHOICE,
                            label="Attention mode (requires API restart):",
                            variable=self._var_attention_mode,
                            choices=[label for label, _mode in _ATTENTION_MODE_OPTIONS],
                            on_change=self._on_attention_mode_changed,
                            width=20,
                            help_mode=HelpMode.DIALOG,
                            help_title="Attention mode",
                            help_text=(
                                "sdpa_auto lets PyTorch pick the best SDPA kernel.\n"
                                "sdpa_flash/sdpa_mem_efficient/sdpa_math force a specific SDPA policy.\n"
                                "xformers/split/quad select non-SDPA attention backends.\n"
                                "Backend flags: --attention-backend + optional --attention-sdpa-policy."
                            ),
                        ),
                    ],
                ),
            ]

        if self._section == "engine":
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
                            choices=["merge", "online"],
                            on_change=lambda: self._sync_runtime_deps(mark_changed=True),
                            width=12,
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
                            choices=["off"],
                            on_change=lambda: self._sync_runtime_deps(mark_changed=True),
                            width=10,
                            advanced=True,
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
                            choices=["weight_merge"],
                            on_change=lambda: self._sync_runtime_deps(mark_changed=True),
                            width=16,
                            advanced=True,
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

        if self._section == "safety":
            return [
                FormSectionDescriptor(
                    title="Tasks / Safety",
                    fields=[
                        FormFieldDescriptor(
                            field_id="single_flight",
                            kind=FieldKind.CHECK,
                            label="Single-flight inference (requires API restart):",
                            variable=self._var_single_flight,
                            on_change=lambda: self._sync_task_deps(mark_changed=True),
                            help_mode=HelpMode.DIALOG,
                            help_title="Single-flight inference",
                            help_text=(
                                "Env var: CODEX_SINGLE_FLIGHT\n"
                                "When enabled (default), GPU-heavy tasks (generation/video/upscale/SUPIR) are serialized to avoid global-state races."
                            ),
                        ),
                        FormFieldDescriptor(
                            field_id="task_cancel_mode",
                            kind=FieldKind.CHOICE,
                            label="Task cancel default mode (requires API restart):",
                            variable=self._var_task_cancel_default_mode,
                            choices=list(TASK_CANCEL_DEFAULT_MODE_CHOICES),
                            on_change=lambda: self._sync_task_deps(mark_changed=True),
                            width=14,
                            help_mode=HelpMode.DIALOG,
                            help_title="Task cancel default mode",
                            help_text=(
                                "Env var: CODEX_TASK_CANCEL_DEFAULT_MODE\n"
                                "immediate: cancels in-flight generation now.\n"
                                "after_current: finish current image job (1st pass + hires/decode/cleanup) before stopping."
                            ),
                        ),
                        FormFieldDescriptor(
                            field_id="safeweights_mode",
                            kind=FieldKind.CHECK,
                            label="Upscalers safeweights mode (requires API restart):",
                            variable=self._var_safeweights,
                            on_change=lambda: self._sync_task_deps(mark_changed=True),
                            help_mode=HelpMode.DIALOG,
                            help_title="Upscalers safeweights mode",
                            help_text=(
                                "Env var: CODEX_SAFE_WEIGHTS\n"
                                "When enabled, upscaler weights must be .safetensors (blocks .pt/.pth at discovery, download, and load-time)."
                            ),
                        ),
                    ],
                ),
                FormSectionDescriptor(
                    title="Tasks / Safety (Advanced)",
                    advanced=True,
                    fields=[
                        FormFieldDescriptor(
                            field_id="task_buffer_max_events",
                            kind=FieldKind.ENTRY_COMMIT,
                            label="Task SSE buffer max events (requires API restart):",
                            variable=self._var_task_buffer_max_events,
                            on_change=self._commit_task_buffer_max_events,
                            width=12,
                            advanced=True,
                            help_mode=HelpMode.DIALOG,
                            help_title="Task SSE buffer max events",
                            help_text=(
                                "Env var: CODEX_TASK_EVENT_BUFFER_MAX_EVENTS\n"
                                "Caps in-memory per-task replay events for reconnect/resume."
                            ),
                        ),
                        FormFieldDescriptor(
                            field_id="task_buffer_max_mb",
                            kind=FieldKind.ENTRY_COMMIT,
                            label="Task SSE buffer max MB (requires API restart):",
                            variable=self._var_task_buffer_max_mb,
                            on_change=self._commit_task_buffer_max_mb,
                            width=12,
                            advanced=True,
                            help_mode=HelpMode.DIALOG,
                            help_title="Task SSE buffer max MB",
                            help_text=(
                                "Env var: CODEX_TASK_EVENT_BUFFER_MAX_MB\n"
                                "Caps in-memory per-task replay size for reconnect/resume."
                            ),
                        ),
                    ],
                ),
            ]

        raise ValueError(f"Unhandled runtime section: {self._section}")

    def reload(self) -> None:
        env = self._controller.store.env

        def _get(key: str, default: str) -> str:
            raw = str(env.get(key, default) or "").strip().lower()
            return raw if raw else default

        raw_main = _get("CODEX_MAIN_DEVICE", "")
        if not raw_main:
            raw_main = _get("CODEX_CORE_DEVICE", "auto")
        try:
            main_device = ChoiceSetting(
                "CODEX_MAIN_DEVICE",
                default="auto",
                choices=DEVICE_CHOICES,
            ).parse(raw_main)
        except SettingValidationError as exc:
            main_device = "auto"
            messagebox.showerror("Invalid runtime setting", str(exc))
        self._set_main_device_env(main_device, mark_changed=False)

        raw_mount = _get("CODEX_MOUNT_DEVICE", main_device)
        raw_offload = _get("CODEX_OFFLOAD_DEVICE", "cpu")
        try:
            self._set_mount_device_env(raw_mount, mark_changed=False)
        except SettingValidationError as exc:
            messagebox.showerror("Invalid runtime setting", str(exc))
            self._set_mount_device_env(self._var_main_device.get(), mark_changed=False)
        try:
            self._set_offload_device_env(raw_offload, mark_changed=False)
        except SettingValidationError as exc:
            messagebox.showerror("Invalid runtime setting", str(exc))
            self._set_offload_device_env("cpu", mark_changed=False)
        try:
            attn_backend, attn_sdpa_policy = normalize_attention_env(env)
        except SettingValidationError as exc:
            self._controller.store.env["CODEX_ATTENTION_BACKEND"] = "pytorch"
            self._controller.store.env["CODEX_ATTENTION_SDPA_POLICY"] = "auto"
            attn_backend, attn_sdpa_policy = normalize_attention_env(self._controller.store.env)
            messagebox.showerror("Invalid runtime setting", str(exc))
        attention_mode = backend_policy_to_attention_mode(attn_backend, attn_sdpa_policy)
        self._var_attention_mode.set(_ATTENTION_MODE_TO_LABEL.get(attention_mode, "SDPA - Auto"))

        runtime_settings_sanitized = False

        if self._section == "engine":
            cfg_batch_setting = ChoiceSetting(
                "CODEX_CFG_BATCH_MODE",
                default="fused",
                choices=CFG_BATCH_MODE_CHOICES,
            )
            try:
                cfg_batch_mode = cfg_batch_setting.get(env)
            except SettingValidationError as exc:
                cfg_batch_mode = "fused"
                env["CODEX_CFG_BATCH_MODE"] = cfg_batch_mode
                messagebox.showerror("Invalid runtime setting", str(exc))
                runtime_settings_sanitized = True
            self._var_cfg_batch_mode.set(cfg_batch_mode)
        else:
            self._var_cfg_batch_mode.set(str(env.get("CODEX_CFG_BATCH_MODE", "fused") or "fused").strip().lower())

        self._var_lora_apply_mode.set(_get("CODEX_LORA_APPLY_MODE", "online"))
        self._var_gguf_dequant_cache.set(_get("CODEX_GGUF_DEQUANT_CACHE", "off"))
        self._var_wan_chunk_buffer_mode.set(_get("CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE", "hybrid"))
        self._var_lora_online_math.set(_get("CODEX_LORA_ONLINE_MATH", "weight_merge"))

        default_alloc_setting = BoolSetting(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY, default=True)
        try:
            default_alloc_enabled = default_alloc_setting.get(env)
        except SettingValidationError as exc:
            default_alloc_enabled = True
            default_alloc_setting.set(env, default_alloc_enabled)
            messagebox.showerror("Invalid runtime setting", str(exc))
            runtime_settings_sanitized = True
        self._var_default_alloc_conf_enabled.set(bool(default_alloc_enabled))

        cuda_malloc_setting = BoolSetting(CODEX_CUDA_MALLOC_KEY, default=False)
        try:
            cuda_malloc_enabled = cuda_malloc_setting.get(env)
        except SettingValidationError as exc:
            cuda_malloc_enabled = False
            cuda_malloc_setting.set(env, cuda_malloc_enabled)
            messagebox.showerror("Invalid runtime setting", str(exc))
            runtime_settings_sanitized = True
        self._var_cuda_malloc.set(bool(cuda_malloc_enabled))
        alloc = str(env.get("PYTORCH_CUDA_ALLOC_CONF", "") or "").strip()
        if not alloc and default_alloc_enabled:
            alloc = DEFAULT_PYTORCH_CUDA_ALLOC_CONF
        self._var_pytorch_alloc_conf.set(alloc)

        try:
            single_flight, safeweights, max_events, max_mb, cancel_default_mode = normalize_task_runtime_env(env)
        except SettingValidationError as exc:
            self._controller.store.env["CODEX_SINGLE_FLIGHT"] = "1"
            self._controller.store.env["CODEX_SAFE_WEIGHTS"] = "0"
            self._controller.store.env["CODEX_TASK_EVENT_BUFFER_MAX_EVENTS"] = str(TASK_EVENT_BUFFER_MAX_EVENTS_DEFAULT)
            self._controller.store.env["CODEX_TASK_EVENT_BUFFER_MAX_MB"] = str(TASK_EVENT_BUFFER_MAX_MB_DEFAULT)
            self._controller.store.env["CODEX_TASK_CANCEL_DEFAULT_MODE"] = "immediate"
            single_flight, safeweights, max_events, max_mb, cancel_default_mode = normalize_task_runtime_env(env)
            messagebox.showerror("Invalid task setting", str(exc))

        self._var_single_flight.set(bool(single_flight))
        self._var_safeweights.set(bool(safeweights))
        self._var_task_cancel_default_mode.set(str(cancel_default_mode))
        self._var_task_buffer_max_events.set(str(int(max_events)))
        self._var_task_buffer_max_mb.set(str(int(max_mb)))

        self._sync_runtime_deps(mark_changed=False)
        if runtime_settings_sanitized:
            self._mark_changed()
        self._apply_advanced_visibility()

    def set_advanced_visible(self, visible: bool) -> None:
        self._advanced_visible = bool(visible)
        self._apply_advanced_visibility()

    def _apply_advanced_visibility(self) -> None:
        if self._form_renderer is not None:
            self._form_renderer.set_advanced_visible(self._advanced_visible)

    def _commit_int_setting(
        self,
        *,
        var: tk.StringVar,
        key: str,
        default: int,
        minimum: int,
        error_title: str,
    ) -> None:
        env = self._controller.store.env
        setting = IntSetting(key, default=default, minimum=minimum)
        try:
            value = setting.parse(str(var.get() or ""))
        except SettingValidationError as exc:
            messagebox.showerror(error_title, str(exc))
            value = int(default)
        setting.set(env, value)
        var.set(str(value))
        self._mark_changed()

    def _commit_task_buffer_max_events(self) -> None:
        self._commit_int_setting(
            var=self._var_task_buffer_max_events,
            key="CODEX_TASK_EVENT_BUFFER_MAX_EVENTS",
            default=TASK_EVENT_BUFFER_MAX_EVENTS_DEFAULT,
            minimum=1,
            error_title="Invalid task setting",
        )

    def _commit_task_buffer_max_mb(self) -> None:
        self._commit_int_setting(
            var=self._var_task_buffer_max_mb,
            key="CODEX_TASK_EVENT_BUFFER_MAX_MB",
            default=TASK_EVENT_BUFFER_MAX_MB_DEFAULT,
            minimum=1,
            error_title="Invalid task setting",
        )

    # ------------------------------------------------------------------ env helpers

    def _set_main_device_env(self, value: str, *, mark_changed: bool) -> None:
        env = self._controller.store.env
        normalized = str(value or "").strip().lower() or "auto"
        normalized = ChoiceSetting(
            "CODEX_MAIN_DEVICE",
            default="auto",
            choices=DEVICE_CHOICES,
        ).parse(normalized)
        env["CODEX_MAIN_DEVICE"] = normalized
        env["CODEX_CORE_DEVICE"] = normalized
        env["CODEX_TE_DEVICE"] = normalized
        env["CODEX_VAE_DEVICE"] = normalized
        env["CODEX_MOUNT_DEVICE"] = normalized
        self._var_main_device.set(normalized)
        self._var_mount_device.set(normalized)
        current_offload = str(env.get("CODEX_OFFLOAD_DEVICE", "cpu") or "").strip().lower() or "cpu"
        try:
            self._set_offload_device_env(current_offload, mark_changed=False)
        except SettingValidationError:
            self._set_offload_device_env("cpu", mark_changed=False)
        if mark_changed:
            self._mark_changed()

    def _set_mount_device_env(self, value: str, *, mark_changed: bool) -> None:
        env = self._controller.store.env
        default_device = str(self._var_main_device.get() or "auto").strip().lower() or "auto"
        normalized = str(value or "").strip().lower() or default_device
        normalized = ChoiceSetting(
            "CODEX_MOUNT_DEVICE",
            default=default_device,
            choices=DEVICE_CHOICES,
        ).parse(normalized)
        env["CODEX_MOUNT_DEVICE"] = normalized
        self._var_mount_device.set(normalized)
        current_offload = str(env.get("CODEX_OFFLOAD_DEVICE", "cpu") or "").strip().lower() or "cpu"
        try:
            self._set_offload_device_env(current_offload, mark_changed=False)
        except SettingValidationError:
            self._set_offload_device_env("cpu", mark_changed=False)
        if mark_changed:
            self._mark_changed()

    def _set_offload_device_env(self, value: str, *, mark_changed: bool) -> None:
        env = self._controller.store.env
        default_device = "cpu"
        normalized = str(value or "").strip().lower() or default_device
        normalized = ChoiceSetting(
            "CODEX_OFFLOAD_DEVICE",
            default=default_device,
            choices=DEVICE_CHOICES,
        ).parse(normalized)
        resolved_main = ChoiceSetting(
            "CODEX_MAIN_DEVICE",
            default="auto",
            choices=DEVICE_CHOICES,
        ).parse(str(self._var_main_device.get() or "auto").strip().lower() or "auto")
        resolved_mount = ChoiceSetting(
            "CODEX_MOUNT_DEVICE",
            default=resolved_main,
            choices=DEVICE_CHOICES,
        ).parse(str(self._var_mount_device.get() or resolved_main).strip().lower() or resolved_main)
        if resolved_main == "cpu" and normalized != "cpu":
            raise SettingValidationError(
                "Offload device must be CPU when main device is CPU (no CPU->accelerator unload target)."
            )
        if resolved_mount == "cpu" and normalized != "cpu":
            raise SettingValidationError(
                "Offload device must be CPU when mount device is CPU (no CPU->accelerator unload target)."
            )
        if resolved_mount not in {"cpu", "auto"} and normalized == resolved_mount:
            raise SettingValidationError(
                "Offload device cannot match mount device for non-CPU unload; use CPU for de-residency."
            )
        env["CODEX_OFFLOAD_DEVICE"] = normalized
        self._var_offload_device.set(normalized)
        if mark_changed:
            self._mark_changed()

    def _on_main_device_changed(self) -> None:
        try:
            self._set_main_device_env(self._var_main_device.get(), mark_changed=True)
        except SettingValidationError as exc:
            messagebox.showerror("Invalid runtime setting", str(exc))
            self._set_main_device_env("auto", mark_changed=True)

    def _on_mount_device_changed(self) -> None:
        try:
            self._set_mount_device_env(self._var_mount_device.get(), mark_changed=True)
        except SettingValidationError as exc:
            messagebox.showerror("Invalid runtime setting", str(exc))
            self._set_mount_device_env(self._var_main_device.get(), mark_changed=True)

    def _on_offload_device_changed(self) -> None:
        try:
            self._set_offload_device_env(self._var_offload_device.get(), mark_changed=True)
        except SettingValidationError as exc:
            messagebox.showerror("Invalid runtime setting", str(exc))
            self._set_offload_device_env("cpu", mark_changed=True)

    def _on_attention_mode_changed(self) -> None:
        env = self._controller.store.env
        try:
            raw_mode = str(self._var_attention_mode.get() or "").strip()
            attention_mode = _ATTENTION_LABEL_TO_MODE.get(raw_mode, raw_mode)
            backend, sdpa_policy = attention_mode_to_backend_policy(attention_mode)
        except SettingValidationError as exc:
            messagebox.showerror("Invalid runtime setting", str(exc))
            backend, sdpa_policy = "pytorch", "auto"
        env["CODEX_ATTENTION_BACKEND"] = backend
        env["CODEX_ATTENTION_SDPA_POLICY"] = sdpa_policy
        attention_mode = backend_policy_to_attention_mode(backend, sdpa_policy)
        self._var_attention_mode.set(_ATTENTION_MODE_TO_LABEL.get(attention_mode, "SDPA - Auto"))
        self._mark_changed()

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
        self._mark_changed()

    def _on_default_alloc_conf_toggle_changed(self) -> None:
        env = self._controller.store.env
        enabled = bool(self._var_default_alloc_conf_enabled.get())
        BoolSetting(
            ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
            default=True,
        ).set(env, enabled)
        if not enabled and "PYTORCH_CUDA_ALLOC_CONF" not in env:
            self._var_pytorch_alloc_conf.set("")
        if enabled and "PYTORCH_CUDA_ALLOC_CONF" not in env and not str(self._var_pytorch_alloc_conf.get() or "").strip():
            self._var_pytorch_alloc_conf.set(DEFAULT_PYTORCH_CUDA_ALLOC_CONF)
        self._mark_changed()

    def _on_cuda_malloc_changed(self) -> None:
        BoolSetting(
            CODEX_CUDA_MALLOC_KEY,
            default=False,
        ).set(self._controller.store.env, bool(self._var_cuda_malloc.get()))
        self._mark_changed()

    def _sync_task_deps(self, *, mark_changed: bool) -> None:
        env = self._controller.store.env

        BoolSetting("CODEX_SINGLE_FLIGHT", default=True).set(env, bool(self._var_single_flight.get()))
        BoolSetting("CODEX_SAFE_WEIGHTS", default=False).set(env, bool(self._var_safeweights.get()))
        env["CODEX_TASK_CANCEL_DEFAULT_MODE"] = str(self._var_task_cancel_default_mode.get() or "").strip().lower()
        env["CODEX_TASK_EVENT_BUFFER_MAX_EVENTS"] = str(self._var_task_buffer_max_events.get() or "").strip()
        env["CODEX_TASK_EVENT_BUFFER_MAX_MB"] = str(self._var_task_buffer_max_mb.get() or "").strip()

        try:
            single_flight, safeweights, max_events, max_mb, cancel_default_mode = normalize_task_runtime_env(env)
        except SettingValidationError as exc:
            env["CODEX_SINGLE_FLIGHT"] = "1"
            env["CODEX_SAFE_WEIGHTS"] = "0"
            env["CODEX_TASK_EVENT_BUFFER_MAX_EVENTS"] = str(TASK_EVENT_BUFFER_MAX_EVENTS_DEFAULT)
            env["CODEX_TASK_EVENT_BUFFER_MAX_MB"] = str(TASK_EVENT_BUFFER_MAX_MB_DEFAULT)
            env["CODEX_TASK_CANCEL_DEFAULT_MODE"] = "immediate"
            single_flight, safeweights, max_events, max_mb, cancel_default_mode = normalize_task_runtime_env(env)
            messagebox.showerror("Invalid task setting", str(exc))
            mark_changed = True

        self._var_single_flight.set(bool(single_flight))
        self._var_safeweights.set(bool(safeweights))
        self._var_task_cancel_default_mode.set(str(cancel_default_mode))
        self._var_task_buffer_max_events.set(str(int(max_events)))
        self._var_task_buffer_max_mb.set(str(int(max_mb)))

        if mark_changed:
            self._mark_changed()

    # ------------------------------------------------------------------ dependency logic

    def _sync_runtime_deps(self, *, mark_changed: bool) -> None:
        env = self._controller.store.env
        if self._section == "engine":
            env["CODEX_CFG_BATCH_MODE"] = (
                str(self._var_cfg_batch_mode.get() or "").strip().lower() or "fused"
            )
            try:
                cfg_batch_mode = ChoiceSetting(
                    "CODEX_CFG_BATCH_MODE",
                    default="fused",
                    choices=CFG_BATCH_MODE_CHOICES,
                ).get(env)
            except SettingValidationError as exc:
                env["CODEX_CFG_BATCH_MODE"] = "fused"
                cfg_batch_mode = "fused"
                messagebox.showerror("Invalid runtime setting", str(exc))
                mark_changed = True
            self._var_cfg_batch_mode.set(cfg_batch_mode)
        env.pop("CODEX_GGUF_EXEC", None)
        env["CODEX_GGUF_DEQUANT_CACHE"] = str(self._var_gguf_dequant_cache.get() or "").strip().lower() or "off"
        env.pop("CODEX_GGUF_DEQUANT_CACHE_RATIO", None)
        env.pop("CODEX_GGUF_DEQUANT_CACHE_LIMIT_MB", None)
        env["CODEX_LORA_APPLY_MODE"] = str(self._var_lora_apply_mode.get() or "").strip().lower() or "online"
        env["CODEX_LORA_ONLINE_MATH"] = str(self._var_lora_online_math.get() or "").strip().lower() or "weight_merge"
        env["CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE"] = (
            str(self._var_wan_chunk_buffer_mode.get() or "").strip().lower() or "hybrid"
        )
        try:
            gguf_cache, lora_apply, lora_math, chunk_buffer_mode = normalize_gguf_lora_env(env)
        except SettingValidationError as exc:
            env.pop("CODEX_GGUF_EXEC", None)
            env["CODEX_GGUF_DEQUANT_CACHE"] = "off"
            env.pop("CODEX_GGUF_DEQUANT_CACHE_RATIO", None)
            env.pop("CODEX_GGUF_DEQUANT_CACHE_LIMIT_MB", None)
            env["CODEX_LORA_APPLY_MODE"] = "online"
            env["CODEX_LORA_ONLINE_MATH"] = "weight_merge"
            env["CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE"] = "hybrid"
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
            self._mark_changed()
