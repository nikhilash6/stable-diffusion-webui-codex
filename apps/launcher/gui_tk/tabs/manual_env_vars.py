"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Manual API environment overlay tab for the Tk launcher.
Co-locates the API-only overlay enable toggle, plain-text editor, and validation feedback for `KEY=VALUE` lines that are applied on the next
API start/restart after Save Settings.

Symbols (top-level; keep in sync; no ghosts):
- `ManualEnvVarsTab` (class): Tab controller for manual API env overlay enable/edit/validation flow.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from apps.launcher.profiles import DEFAULT_MANUAL_API_ENV_TEXT

from ..controller import LauncherController
from ..styles import Palette, resolve_fonts
from ..widgets import ScrollableFrame


class ManualEnvVarsTab:
    def __init__(
        self,
        controller: LauncherController,
        *,
        canvas_bg: str,
        palette: Palette,
        mark_changed: Callable[[], None],
    ) -> None:
        self._controller = controller
        self._canvas_bg = str(canvas_bg)
        self._palette = palette
        self._mark_changed = mark_changed
        self.frame: ttk.Frame | None = None
        self._editor: tk.Text | None = None
        self._enabled_var = tk.BooleanVar()
        self._validation_var = tk.StringVar()
        self._editing_programmatically = False
        self._validation_label: ttk.Label | None = None

    def build(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        scroll = ScrollableFrame(frame, canvas_bg=self._canvas_bg)
        scroll.pack(fill="both", expand=True, padx=8, pady=8)
        body = scroll.inner
        body.columnconfigure(0, weight=1)

        section = ttk.LabelFrame(body, text="  Manual Env Vars (API only)  ", padding=14)
        section.grid(row=0, column=0, sticky="nsew")
        section.columnconfigure(0, weight=1)
        section.rowconfigure(3, weight=1)

        ttk.Label(
            section,
            text=(
                "One env var per line as KEY=VALUE. Blank lines and lines starting with # are ignored.\n"
                "The overlay is stored in plain text in .sangoi/launcher/meta.json and affects only API starts/restarts after Save Settings.\n"
                "Default template:\n"
                f"{DEFAULT_MANUAL_API_ENV_TEXT}"
            ),
            style="Muted.TLabel",
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        ttk.Checkbutton(
            section,
            text="Enable Manual Env Vars overlay for API start/restart",
            variable=self._enabled_var,
            command=self._on_enabled_changed,
            style="Toggle.TCheckbutton",
        ).grid(row=1, column=0, sticky="w")

        validation_label = ttk.Label(section, textvariable=self._validation_var, style="Muted.TLabel", justify="left")
        validation_label.grid(row=2, column=0, sticky="w", pady=(8, 10))
        self._validation_label = validation_label

        editor_frame = ttk.Frame(section)
        editor_frame.grid(row=3, column=0, sticky="nsew")
        editor_frame.columnconfigure(0, weight=1)
        editor_frame.rowconfigure(0, weight=1)

        editor = tk.Text(
            editor_frame,
            width=98,
            height=20,
            wrap="none",
            undo=True,
            bg=self._palette.bg2,
            fg=self._palette.fg0,
            insertbackground=self._palette.fg0,
            selectbackground=self._palette.accent,
            selectforeground=self._palette.bg0,
            highlightthickness=1,
            highlightbackground=self._palette.line,
            highlightcolor=self._palette.accent,
            relief="flat",
            padx=10,
            pady=10,
            font=resolve_fonts(frame).mono,
        )
        editor.grid(row=0, column=0, sticky="nsew")
        editor.bind("<<Modified>>", self._on_editor_modified)
        self._editor = editor

        yscroll = ttk.Scrollbar(editor_frame, orient="vertical", command=editor.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(editor_frame, orient="horizontal", command=editor.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        editor.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.frame = frame
        self.reload()
        return frame

    def reload(self) -> None:
        current = str(getattr(self._controller.store.meta, "manual_api_env_text", DEFAULT_MANUAL_API_ENV_TEXT) or "")
        self._enabled_var.set(bool(getattr(self._controller.store.meta, "manual_api_env_enabled", False)))
        self._set_editor_text(current)
        self._refresh_validation_state()

    def _set_editor_text(self, value: str) -> None:
        if self._editor is None:
            return
        self._editing_programmatically = True
        try:
            self._editor.delete("1.0", tk.END)
            if value:
                self._editor.insert("1.0", str(value))
            self._editor.edit_modified(False)
        finally:
            self._editing_programmatically = False

    def _on_enabled_changed(self) -> None:
        self._controller.update_manual_api_env_enabled(bool(self._enabled_var.get()))
        self._mark_changed()
        self._refresh_validation_state()

    def _on_editor_modified(self, _event: tk.Event[tk.Text]) -> None:
        editor = self._editor
        if editor is None:
            return
        if self._editing_programmatically:
            editor.edit_modified(False)
            return
        if not editor.edit_modified():
            return
        editor.edit_modified(False)
        content = editor.get("1.0", "end-1c")
        self._controller.update_manual_api_env_text(str(content))
        self._mark_changed()
        self._refresh_validation_state()

    def _refresh_validation_state(self) -> None:
        label = self._validation_label
        if label is None:
            return
        if not bool(self._enabled_var.get()):
            self._validation_var.set(
                "Disabled. Save Settings, then enable this overlay when you want the API-only env vars applied on the next start/restart."
            )
            label.configure(style="Muted.TLabel")
            return
        error = self._controller.manual_api_env_error()
        if error:
            self._validation_var.set(f"Invalid overlay: {error}")
            label.configure(style="Status.Error.TLabel")
            return
        self._validation_var.set("Valid API-only overlay. Applied only on the next API start/restart after Save Settings.")
        label.configure(style="Status.Running.TLabel")
