"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Diagnostics tab for the Tk launcher.
Shows launcher environment preflight checks plus descriptor-rendered backend debug, runtime-diagnostics, trace, profiler, and log-level controls grouped by troubleshooting purpose.

Symbols (top-level; keep in sync; no ghosts):
- `DiagnosticsTab` (class): Diagnostics tab controller for checks and debug/runtime-diagnostics/logging/profiling settings.
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Iterable, List, Tuple

from apps.launcher.checks import CodexLaunchCheck
from apps.launcher.setting_registry import setting_descriptor_for_key, vram_metadata_for_key
from apps.launcher.settings import BoolSetting, IntSetting, SettingValidationError

from ..controller import LauncherController
from ..form_renderer import FormRenderer
from ..form_schema import FieldKind, FormFieldDescriptor, FormSectionDescriptor
from ..widgets import ScrollableFrame
from .diagnostics_sections import DIAGNOSTICS_SECTIONS, LOG_FILE_LABEL, LOG_LEVEL_FLAGS, DiagnosticsEntrySpec


def _descriptor_default(key: str) -> str:
    descriptor = setting_descriptor_for_key(key)
    if descriptor is None:
        raise RuntimeError(f"Launcher setting descriptor missing for {key}")
    return descriptor.default_value()


def _descriptor_bool_default(key: str) -> bool:
    return BoolSetting(key, default=False).parse(_descriptor_default(key))


def _descriptor_int_default(key: str) -> int:
    return int(_descriptor_default(key))


class DiagnosticsTab:
    def __init__(
        self,
        controller: LauncherController,
        *,
        mark_changed: Callable[[], None],
        run_checks_async: Callable[[], None],
        canvas_bg: str,
    ) -> None:
        self._controller = controller
        self._mark_changed = mark_changed
        self._run_checks_async = run_checks_async
        self._canvas_bg = str(canvas_bg)

        self.frame: ttk.Frame | None = None
        self._checks_tree: ttk.Treeview | None = None
        self._form_renderer: FormRenderer | None = None

        self._debug_flags: Dict[str, tk.BooleanVar] = {}
        self._log_levels: Dict[str, tk.BooleanVar] = {}
        self._entry_vars: Dict[str, tk.StringVar] = {}
        self._var_log_file = tk.BooleanVar()
        self._advanced_visible = False
        self._cfg_delta_trace_ids: List[Tuple[tk.Variable, str]] = []

    def build(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        scroll = ScrollableFrame(frame, canvas_bg=self._canvas_bg)
        scroll.pack(fill="both", expand=True)
        body = scroll.inner
        body.columnconfigure(0, weight=1)

        self._build_checks_box(body)
        self._init_form_variables()
        renderer = FormRenderer(body, padx=8)
        renderer.render_sections(1, self._form_sections())
        self._form_renderer = renderer

        self.reload()
        self.frame = frame
        return frame

    def reload(self) -> None:
        env = self._controller.store.env
        for section in DIAGNOSTICS_SECTIONS:
            for spec in section.flags:
                self._debug_flags[spec.key].set(BoolSetting(spec.key, default=_descriptor_bool_default(spec.key)).get(env))
            for spec in section.entries:
                default = _descriptor_default(spec.key)
                self._entry_vars[spec.field_id].set(str(env.get(spec.key, default) or default))

        for spec in LOG_LEVEL_FLAGS:
            self._log_levels[spec.key].set(BoolSetting(spec.key, default=_descriptor_bool_default(spec.key)).get(env))

        self._var_log_file.set(bool(str(env.get("CODEX_LOG_FILE", "") or "").strip()))
        self._install_cfg_delta_guard()
        self._apply_advanced_visibility()

    def render_checks(self, checks: Iterable[CodexLaunchCheck]) -> None:
        tree = self._checks_tree
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)
        for chk in checks:
            tree.insert("", "end", values=(chk.name, "yes" if chk.ok else "no", chk.detail))

    def set_advanced_visible(self, visible: bool) -> None:
        self._advanced_visible = bool(visible)
        self._apply_advanced_visibility()

    def validate_int_settings(self) -> None:
        env = self._controller.store.env
        try:
            IntSetting("CODEX_LOG_CFG_DELTA_N", default=_descriptor_int_default("CODEX_LOG_CFG_DELTA_N"), minimum=1).get(env)
            IntSetting(
                "CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC",
                default=_descriptor_int_default("CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC"),
                minimum=0,
            ).get(env)
            IntSetting("CODEX_PROFILE_TOP_N", default=_descriptor_int_default("CODEX_PROFILE_TOP_N"), minimum=1, maximum=500).get(env)
            IntSetting(
                "CODEX_PROFILE_MAX_STEPS",
                default=_descriptor_int_default("CODEX_PROFILE_MAX_STEPS"),
                minimum=0,
                maximum=10_000,
            ).get(env)
        except SettingValidationError as exc:
            raise RuntimeError(str(exc)) from exc

    def _build_checks_box(self, body: ttk.Frame) -> None:
        checks_box = ttk.LabelFrame(body, text="  Environment Checks  ", padding=14)
        checks_box.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        checks_box.columnconfigure(0, weight=1)
        checks_box.rowconfigure(0, weight=1)

        tree_frame = ttk.Frame(checks_box)
        tree_frame.grid(row=0, column=0, sticky="ew")
        tree_frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(tree_frame, columns=("name", "ok", "detail"), show="headings", height=6)
        tree.heading("name", text="Check")
        tree.heading("ok", text="OK")
        tree.heading("detail", text="Detail")
        tree.column("name", width=170, minwidth=140, anchor="w", stretch=False)
        tree.column("ok", width=64, minwidth=64, anchor="center", stretch=False)
        tree.column("detail", width=680, minwidth=320, anchor="w", stretch=True)
        tree.grid(row=0, column=0, sticky="ew")
        xscroll = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        xscroll.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        tree.configure(xscrollcommand=xscroll.set)
        self._checks_tree = tree

        checks_actions = ttk.Frame(checks_box)
        checks_actions.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        checks_actions.columnconfigure(0, weight=1)
        ttk.Label(
            checks_actions,
            text="These checks validate launcher prerequisites only; service runtime failures still show up in Logs.",
            style="Muted.TLabel",
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(checks_actions, text="↻ Re-run checks", command=self._run_checks_async).grid(row=0, column=1, sticky="e")

    def _init_form_variables(self) -> None:
        env = self._controller.store.env
        for section in DIAGNOSTICS_SECTIONS:
            for spec in section.flags:
                self._debug_flags[spec.key] = tk.BooleanVar(
                    value=BoolSetting(spec.key, default=_descriptor_bool_default(spec.key)).get(env)
                )
            for spec in section.entries:
                default = _descriptor_default(spec.key)
                self._entry_vars[spec.field_id] = tk.StringVar(value=str(env.get(spec.key, default) or default))
        for spec in LOG_LEVEL_FLAGS:
            self._log_levels[spec.key] = tk.BooleanVar(
                value=BoolSetting(spec.key, default=_descriptor_bool_default(spec.key)).get(env)
            )
        self._var_log_file.set(bool(str(env.get("CODEX_LOG_FILE", "") or "").strip()))

    def _form_sections(self) -> list[FormSectionDescriptor]:
        sections: list[FormSectionDescriptor] = []
        for section in DIAGNOSTICS_SECTIONS:
            fields: list[FormFieldDescriptor] = []
            for spec in section.flags:
                fields.append(
                    FormFieldDescriptor(
                        field_id=spec.key.lower(),
                        kind=FieldKind.CHECK,
                        label=spec.label,
                        variable=self._debug_flags[spec.key],
                        on_change=lambda key=spec.key: self._set_bool(key, self._debug_flags[key].get()),
                        advanced=spec.advanced,
                        vram=vram_metadata_for_key(spec.key),
                    )
                )
            for spec in section.entries:
                fields.append(self._entry_field(spec))
            sections.append(FormSectionDescriptor(title=section.title, fields=tuple(fields), advanced=section.advanced))

        logging_fields: list[FormFieldDescriptor] = []
        for spec in LOG_LEVEL_FLAGS:
            logging_fields.append(
                FormFieldDescriptor(
                    field_id=spec.key.lower(),
                    kind=FieldKind.CHECK,
                    label=spec.label,
                    variable=self._log_levels[spec.key],
                    on_change=lambda key=spec.key: self._set_bool(key, self._log_levels[key].get()),
                )
            )
        logging_fields.append(
            FormFieldDescriptor(
                field_id="codex_log_file",
                kind=FieldKind.CHECK,
                label=LOG_FILE_LABEL,
                variable=self._var_log_file,
                on_change=self._toggle_log_file,
            )
        )
        sections.append(FormSectionDescriptor(title="Logging", fields=tuple(logging_fields)))
        return sections

    def _entry_field(self, spec: DiagnosticsEntrySpec) -> FormFieldDescriptor:
        return FormFieldDescriptor(
            field_id=spec.field_id,
            kind=FieldKind.ENTRY,
            label=spec.label,
            variable=self._entry_vars[spec.field_id],
            width=spec.width,
            advanced=spec.advanced,
            on_change=lambda entry=spec: self._set_text(entry.key, self._entry_vars[entry.field_id].get()),
        )

    def _set_bool(self, key: str, enabled: bool) -> None:
        BoolSetting(key, default=_descriptor_bool_default(key)).set(self._controller.store.env, enabled)
        self._mark_changed()

    def _set_text(self, key: str, value: str) -> None:
        self._controller.store.env[key] = str(value).strip()
        self._mark_changed()

    def _apply_advanced_visibility(self) -> None:
        if self._form_renderer is not None:
            self._form_renderer.set_advanced_visible(self._advanced_visible)

    def _clear_cfg_delta_guard(self) -> None:
        for variable, trace_id in self._cfg_delta_trace_ids:
            try:
                variable.trace_remove("write", trace_id)
            except Exception:
                continue
        self._cfg_delta_trace_ids.clear()

    def _install_cfg_delta_guard(self) -> None:
        if "CODEX_LOG_SAMPLER" not in self._debug_flags or "CODEX_LOG_CFG_DELTA" not in self._debug_flags:
            return
        self._clear_cfg_delta_guard()
        sampler_var = self._debug_flags["CODEX_LOG_SAMPLER"]
        delta_var = self._debug_flags["CODEX_LOG_CFG_DELTA"]
        guard = {"active": False}

        def _on_delta_changed(*_args: object) -> None:
            if guard["active"] or (not delta_var.get()):
                return
            if sampler_var.get():
                return
            guard["active"] = True
            try:
                sampler_var.set(True)
                BoolSetting("CODEX_LOG_SAMPLER", default=_descriptor_bool_default("CODEX_LOG_SAMPLER")).set(
                    self._controller.store.env,
                    True,
                )
                self._mark_changed()
            finally:
                guard["active"] = False

        def _on_sampler_changed(*_args: object) -> None:
            if guard["active"] or sampler_var.get():
                return
            if not delta_var.get():
                return
            guard["active"] = True
            try:
                delta_var.set(False)
                BoolSetting("CODEX_LOG_CFG_DELTA", default=_descriptor_bool_default("CODEX_LOG_CFG_DELTA")).set(
                    self._controller.store.env,
                    False,
                )
                self._mark_changed()
            finally:
                guard["active"] = False

        delta_trace_id = delta_var.trace_add("write", _on_delta_changed)
        sampler_trace_id = sampler_var.trace_add("write", _on_sampler_changed)
        self._cfg_delta_trace_ids.append((delta_var, delta_trace_id))
        self._cfg_delta_trace_ids.append((sampler_var, sampler_trace_id))
        _on_delta_changed()

    def _toggle_log_file(self) -> None:
        enabled = bool(self._var_log_file.get())
        env = self._controller.store.env
        if enabled:
            if not str(env.get("CODEX_LOG_FILE", "") or "").strip():
                logs_dir = self._controller.codex_root / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                env["CODEX_LOG_FILE"] = str(logs_dir / f"codex-{stamp}.log")
                self._mark_changed()
        else:
            try:
                del env["CODEX_LOG_FILE"]
            except KeyError:
                pass
            self._mark_changed()
