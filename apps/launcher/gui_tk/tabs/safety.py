"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Safety settings tab for the Tk launcher.
Owns task serialization, cancellation, task replay buffer caps, and upscaler safeweights runtime settings.

Symbols (top-level; keep in sync; no ghosts):
- `SafetyTab` (class): Tk tab for task/safety runtime settings.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from typing import Callable, Sequence

from apps.launcher.setting_registry import setting_descriptor_for_key, vram_metadata_for_key
from apps.launcher.settings import (
    BoolSetting,
    IntSetting,
    SettingValidationError,
    TASK_CANCEL_DEFAULT_MODE_CHOICES,
    normalize_task_runtime_env,
)

from ..controller import LauncherController
from ..form_schema import FieldKind, FormFieldDescriptor, FormSectionDescriptor, HelpMode
from .runtime_common import RuntimeFormTabBase


def _descriptor_default(key: str) -> str:
    descriptor = setting_descriptor_for_key(key)
    if descriptor is None:
        raise RuntimeError(f"Launcher setting descriptor missing for {key}")
    return descriptor.default_value()


def _descriptor_bool_default(key: str) -> bool:
    return BoolSetting(key, default=False).parse(_descriptor_default(key))


def _descriptor_int_default(key: str) -> int:
    return int(_descriptor_default(key))


class SafetyTab(RuntimeFormTabBase):
    def __init__(self, controller: LauncherController, *, canvas_bg: str, mark_changed: Callable[[], None]) -> None:
        super().__init__(controller, canvas_bg=canvas_bg, mark_changed=mark_changed)
        self._var_single_flight = tk.BooleanVar()
        self._var_safeweights = tk.BooleanVar()
        self._var_task_cancel_default_mode = tk.StringVar()
        self._var_task_buffer_max_events = tk.StringVar()
        self._var_task_buffer_max_mb = tk.StringVar()

    def _sections_for_view(self) -> Sequence[FormSectionDescriptor]:
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
                        vram=vram_metadata_for_key("CODEX_SINGLE_FLIGHT"),
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

    def reload(self) -> None:
        env = self._controller.store.env
        try:
            single_flight, safeweights, max_events, max_mb, cancel_default_mode = normalize_task_runtime_env(env)
        except SettingValidationError as exc:
            self._controller.store.env["CODEX_SINGLE_FLIGHT"] = _descriptor_default("CODEX_SINGLE_FLIGHT")
            self._controller.store.env["CODEX_SAFE_WEIGHTS"] = _descriptor_default("CODEX_SAFE_WEIGHTS")
            self._controller.store.env["CODEX_TASK_EVENT_BUFFER_MAX_EVENTS"] = _descriptor_default(
                "CODEX_TASK_EVENT_BUFFER_MAX_EVENTS"
            )
            self._controller.store.env["CODEX_TASK_EVENT_BUFFER_MAX_MB"] = _descriptor_default("CODEX_TASK_EVENT_BUFFER_MAX_MB")
            self._controller.store.env["CODEX_TASK_CANCEL_DEFAULT_MODE"] = _descriptor_default("CODEX_TASK_CANCEL_DEFAULT_MODE")
            single_flight, safeweights, max_events, max_mb, cancel_default_mode = normalize_task_runtime_env(env)
            messagebox.showerror("Invalid task setting", str(exc))
        self._var_single_flight.set(bool(single_flight))
        self._var_safeweights.set(bool(safeweights))
        self._var_task_cancel_default_mode.set(str(cancel_default_mode))
        self._var_task_buffer_max_events.set(str(int(max_events)))
        self._var_task_buffer_max_mb.set(str(int(max_mb)))
        self._apply_advanced_visibility()

    def _commit_int_setting(self, *, var: tk.StringVar, key: str, default: int, minimum: int, error_title: str) -> None:
        env = self._controller.store.env
        setting = IntSetting(key, default=default, minimum=minimum)
        try:
            value = setting.parse(str(var.get() or ""))
        except SettingValidationError as exc:
            messagebox.showerror(error_title, str(exc))
            value = int(default)
        setting.set(env, value)
        var.set(str(value))
        self._mark_dirty()

    def _commit_task_buffer_max_events(self) -> None:
        self._commit_int_setting(
            var=self._var_task_buffer_max_events,
            key="CODEX_TASK_EVENT_BUFFER_MAX_EVENTS",
            default=_descriptor_int_default("CODEX_TASK_EVENT_BUFFER_MAX_EVENTS"),
            minimum=1,
            error_title="Invalid task setting",
        )

    def _commit_task_buffer_max_mb(self) -> None:
        self._commit_int_setting(
            var=self._var_task_buffer_max_mb,
            key="CODEX_TASK_EVENT_BUFFER_MAX_MB",
            default=_descriptor_int_default("CODEX_TASK_EVENT_BUFFER_MAX_MB"),
            minimum=1,
            error_title="Invalid task setting",
        )

    def _sync_task_deps(self, *, mark_changed: bool) -> None:
        env = self._controller.store.env
        BoolSetting("CODEX_SINGLE_FLIGHT", default=_descriptor_bool_default("CODEX_SINGLE_FLIGHT")).set(
            env,
            bool(self._var_single_flight.get()),
        )
        BoolSetting("CODEX_SAFE_WEIGHTS", default=_descriptor_bool_default("CODEX_SAFE_WEIGHTS")).set(
            env,
            bool(self._var_safeweights.get()),
        )
        env["CODEX_TASK_CANCEL_DEFAULT_MODE"] = str(self._var_task_cancel_default_mode.get() or "").strip().lower()
        env["CODEX_TASK_EVENT_BUFFER_MAX_EVENTS"] = str(self._var_task_buffer_max_events.get() or "").strip()
        env["CODEX_TASK_EVENT_BUFFER_MAX_MB"] = str(self._var_task_buffer_max_mb.get() or "").strip()

        try:
            single_flight, safeweights, max_events, max_mb, cancel_default_mode = normalize_task_runtime_env(env)
        except SettingValidationError as exc:
            env["CODEX_SINGLE_FLIGHT"] = _descriptor_default("CODEX_SINGLE_FLIGHT")
            env["CODEX_SAFE_WEIGHTS"] = _descriptor_default("CODEX_SAFE_WEIGHTS")
            env["CODEX_TASK_EVENT_BUFFER_MAX_EVENTS"] = _descriptor_default("CODEX_TASK_EVENT_BUFFER_MAX_EVENTS")
            env["CODEX_TASK_EVENT_BUFFER_MAX_MB"] = _descriptor_default("CODEX_TASK_EVENT_BUFFER_MAX_MB")
            env["CODEX_TASK_CANCEL_DEFAULT_MODE"] = _descriptor_default("CODEX_TASK_CANCEL_DEFAULT_MODE")
            single_flight, safeweights, max_events, max_mb, cancel_default_mode = normalize_task_runtime_env(env)
            messagebox.showerror("Invalid task setting", str(exc))
            mark_changed = True

        self._var_single_flight.set(bool(single_flight))
        self._var_safeweights.set(bool(safeweights))
        self._var_task_cancel_default_mode.set(str(cancel_default_mode))
        self._var_task_buffer_max_events.set(str(int(max_events)))
        self._var_task_buffer_max_mb.set(str(int(max_mb)))
        if mark_changed:
            self._mark_dirty()
