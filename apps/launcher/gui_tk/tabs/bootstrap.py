"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Bootstrap settings tab for the Tk launcher.
Owns API-restart device selectors and attention backend/policy settings that must be present before backend startup.

Symbols (top-level; keep in sync; no ghosts):
- `BootstrapTab` (class): Tk tab for main/mount/offload devices and attention mode.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from typing import Callable, Sequence

from apps.launcher.setting_registry import setting_descriptor_for_key, vram_metadata_for_key
from apps.launcher.settings import (
    ChoiceSetting,
    DEVICE_CHOICES,
    SettingValidationError,
    attention_mode_to_backend_policy,
    backend_policy_to_attention_mode,
    normalize_attention_env,
)

from ..controller import LauncherController
from ..form_schema import FieldKind, FormFieldDescriptor, FormSectionDescriptor, HelpMode
from .runtime_common import RuntimeFormTabBase

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


def _descriptor_default(key: str) -> str:
    descriptor = setting_descriptor_for_key(key)
    if descriptor is None:
        raise RuntimeError(f"Launcher setting descriptor missing for {key}")
    return descriptor.default_value()


class BootstrapTab(RuntimeFormTabBase):
    def __init__(self, controller: LauncherController, *, canvas_bg: str, mark_changed: Callable[[], None]) -> None:
        super().__init__(controller, canvas_bg=canvas_bg, mark_changed=mark_changed)
        self._var_main_device = tk.StringVar()
        self._var_mount_device = tk.StringVar()
        self._var_offload_device = tk.StringVar()
        self._var_attention_mode = tk.StringVar()

    def _sections_for_view(self) -> Sequence[FormSectionDescriptor]:
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
                        vram=vram_metadata_for_key("CODEX_MAIN_DEVICE"),
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
                        vram=vram_metadata_for_key("CODEX_MOUNT_DEVICE"),
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
                        vram=vram_metadata_for_key("CODEX_OFFLOAD_DEVICE"),
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
                        vram=vram_metadata_for_key("CODEX_ATTENTION_MODE"),
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

    def reload(self) -> None:
        env = self._controller.store.env

        def _get(key: str, default: str) -> str:
            raw = str(env.get(key, default) or "").strip().lower()
            return raw if raw else default

        raw_main = _get("CODEX_MAIN_DEVICE", "")
        if not raw_main:
            raw_main = _get("CODEX_CORE_DEVICE", _descriptor_default("CODEX_CORE_DEVICE"))
        try:
            main_device = ChoiceSetting(
                "CODEX_MAIN_DEVICE",
                default=_descriptor_default("CODEX_MAIN_DEVICE"),
                choices=DEVICE_CHOICES,
            ).parse(raw_main)
        except SettingValidationError as exc:
            main_device = _descriptor_default("CODEX_MAIN_DEVICE")
            messagebox.showerror("Invalid runtime setting", str(exc))
        self._set_main_device_env(main_device, mark_changed=False)

        raw_mount = _get("CODEX_MOUNT_DEVICE", main_device)
        raw_offload = _get("CODEX_OFFLOAD_DEVICE", _descriptor_default("CODEX_OFFLOAD_DEVICE"))
        try:
            self._set_mount_device_env(raw_mount, mark_changed=False)
        except SettingValidationError as exc:
            messagebox.showerror("Invalid runtime setting", str(exc))
            self._set_mount_device_env(self._var_main_device.get(), mark_changed=False)
        try:
            self._set_offload_device_env(raw_offload, mark_changed=False)
        except SettingValidationError as exc:
            messagebox.showerror("Invalid runtime setting", str(exc))
            self._set_offload_device_env(_descriptor_default("CODEX_OFFLOAD_DEVICE"), mark_changed=False)
        try:
            attn_backend, attn_sdpa_policy = normalize_attention_env(env)
        except SettingValidationError as exc:
            self._controller.store.env["CODEX_ATTENTION_BACKEND"] = _descriptor_default("CODEX_ATTENTION_BACKEND")
            self._controller.store.env["CODEX_ATTENTION_SDPA_POLICY"] = _descriptor_default("CODEX_ATTENTION_SDPA_POLICY")
            attn_backend, attn_sdpa_policy = normalize_attention_env(self._controller.store.env)
            messagebox.showerror("Invalid runtime setting", str(exc))
        attention_mode = backend_policy_to_attention_mode(attn_backend, attn_sdpa_policy)
        self._var_attention_mode.set(
            _ATTENTION_MODE_TO_LABEL.get(attention_mode, _ATTENTION_MODE_TO_LABEL[_descriptor_default("CODEX_ATTENTION_MODE")])
        )
        self._apply_advanced_visibility()

    def _set_main_device_env(self, value: str, *, mark_changed: bool) -> None:
        env = self._controller.store.env
        normalized = str(value or "").strip().lower() or _descriptor_default("CODEX_MAIN_DEVICE")
        normalized = ChoiceSetting(
            "CODEX_MAIN_DEVICE",
            default=_descriptor_default("CODEX_MAIN_DEVICE"),
            choices=DEVICE_CHOICES,
        ).parse(normalized)
        env["CODEX_MAIN_DEVICE"] = normalized
        env["CODEX_CORE_DEVICE"] = normalized
        env["CODEX_TE_DEVICE"] = normalized
        env["CODEX_VAE_DEVICE"] = normalized
        env["CODEX_MOUNT_DEVICE"] = normalized
        self._var_main_device.set(normalized)
        self._var_mount_device.set(normalized)
        default_offload = _descriptor_default("CODEX_OFFLOAD_DEVICE")
        current_offload = str(env.get("CODEX_OFFLOAD_DEVICE", default_offload) or "").strip().lower() or default_offload
        try:
            self._set_offload_device_env(current_offload, mark_changed=False)
        except SettingValidationError:
            self._set_offload_device_env(default_offload, mark_changed=False)
        if mark_changed:
            self._mark_dirty()

    def _set_mount_device_env(self, value: str, *, mark_changed: bool) -> None:
        env = self._controller.store.env
        default_device = str(self._var_main_device.get() or _descriptor_default("CODEX_MAIN_DEVICE")).strip().lower()
        default_device = default_device or _descriptor_default("CODEX_MAIN_DEVICE")
        normalized = str(value or "").strip().lower() or default_device
        normalized = ChoiceSetting("CODEX_MOUNT_DEVICE", default=default_device, choices=DEVICE_CHOICES).parse(normalized)
        env["CODEX_MOUNT_DEVICE"] = normalized
        self._var_mount_device.set(normalized)
        default_offload = _descriptor_default("CODEX_OFFLOAD_DEVICE")
        current_offload = str(env.get("CODEX_OFFLOAD_DEVICE", default_offload) or "").strip().lower() or default_offload
        try:
            self._set_offload_device_env(current_offload, mark_changed=False)
        except SettingValidationError:
            self._set_offload_device_env(default_offload, mark_changed=False)
        if mark_changed:
            self._mark_dirty()

    def _set_offload_device_env(self, value: str, *, mark_changed: bool) -> None:
        env = self._controller.store.env
        normalized = str(value or "").strip().lower() or _descriptor_default("CODEX_OFFLOAD_DEVICE")
        normalized = ChoiceSetting(
            "CODEX_OFFLOAD_DEVICE",
            default=_descriptor_default("CODEX_OFFLOAD_DEVICE"),
            choices=DEVICE_CHOICES,
        ).parse(normalized)
        resolved_main = ChoiceSetting(
            "CODEX_MAIN_DEVICE",
            default=_descriptor_default("CODEX_MAIN_DEVICE"),
            choices=DEVICE_CHOICES,
        ).parse(
            str(self._var_main_device.get() or _descriptor_default("CODEX_MAIN_DEVICE")).strip().lower()
            or _descriptor_default("CODEX_MAIN_DEVICE")
        )
        resolved_mount = ChoiceSetting("CODEX_MOUNT_DEVICE", default=resolved_main, choices=DEVICE_CHOICES).parse(
            str(self._var_mount_device.get() or resolved_main).strip().lower() or resolved_main
        )
        if resolved_main == "cpu" and normalized != "cpu":
            raise SettingValidationError("Offload device must be CPU when main device is CPU (no CPU->accelerator unload target).")
        if resolved_mount == "cpu" and normalized != "cpu":
            raise SettingValidationError("Offload device must be CPU when mount device is CPU (no CPU->accelerator unload target).")
        if resolved_mount not in {"cpu", "auto"} and normalized == resolved_mount:
            raise SettingValidationError("Offload device cannot match mount device for non-CPU unload; use CPU for de-residency.")
        env["CODEX_OFFLOAD_DEVICE"] = normalized
        self._var_offload_device.set(normalized)
        if mark_changed:
            self._mark_dirty()

    def _on_main_device_changed(self) -> None:
        try:
            self._set_main_device_env(self._var_main_device.get(), mark_changed=True)
        except SettingValidationError as exc:
            messagebox.showerror("Invalid runtime setting", str(exc))
            self._set_main_device_env(_descriptor_default("CODEX_MAIN_DEVICE"), mark_changed=True)

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
            self._set_offload_device_env(_descriptor_default("CODEX_OFFLOAD_DEVICE"), mark_changed=True)

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
        self._mark_dirty()
