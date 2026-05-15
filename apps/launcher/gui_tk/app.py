"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Main Tk application for the Codex launcher GUI.
Builds the window + tabs (including `Manual Env Vars`), seeds launcher service handles from persisted launcher meta, wires background tasks,
and provides a stable `main()` entrypoint used by `apps/codex_launcher.py`.

Symbols (top-level; keep in sync; no ghosts):
- `CodexLauncherApp` (class): Tk app; orchestrates tabs + controller + polling.
- `main` (function): Starts the Tk GUI launcher.
"""

from __future__ import annotations

import os
from pathlib import Path
from queue import Queue
import re
import threading
import time
import tkinter as tk
import traceback
from tkinter import messagebox, ttk
from typing import Callable

from apps.backend.infra.config.repo_root import get_repo_root
from apps.launcher.checks import CodexLaunchCheck, run_launch_checks
from apps.launcher.log_buffer import CodexLogBuffer
from apps.launcher.profiles import LauncherProfileStore
from apps.launcher.services import default_services

from .controller import LauncherController
from .styles import Palette, apply_style
from .tabs import DiagnosticsTab, LogsTab, ManualEnvVarsTab, RuntimeTab, ServicesTab


_GEOMETRY_RE = re.compile(r"^\d+x\d+(?:[+-]\d+[+-]\d+)?$")


class CodexLauncherApp(tk.Tk):
    POLL_INTERVAL_MS = 400
    TASK_POLL_INTERVAL_MS = 50

    def __init__(self) -> None:
        super().__init__()

        self._palette = Palette()
        self._task_queue: "Queue[Callable[[], None]]" = Queue()

        self.title("Codex Launcher")
        self.minsize(760, 540)

        self.codex_root = get_repo_root()

        store = LauncherProfileStore.load()
        log_buffer = CodexLogBuffer(capacity=4000)
        services = default_services(
            log_buffer=log_buffer,
            mode_profile=str(getattr(store.meta, "app_mode_profile", "") or ""),
            frontend_dev_typecheck=bool(getattr(store.meta, "frontend_dev_typecheck", False)),
        )
        self._controller = LauncherController(
            codex_root=self.codex_root,
            store=store,
            log_buffer=log_buffer,
            services=services,
        )
        self._restore_window_geometry()
        self._set_window_icon()

        self._unsaved_changes = False
        self._status_text = tk.StringVar(value="Ready")
        self._show_advanced_controls = tk.BooleanVar(
            value=bool(getattr(self._controller.store.meta, "show_advanced_controls", False))
        )

        apply_style(self, self._palette)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))

        self._services_tab = ServicesTab(
            self._controller,
            run_in_thread=self._run_in_thread,
            set_status=self._set_status,
            mark_changed=self._mark_changed,
        )
        self._runtime_bootstrap_tab = RuntimeTab(
            self._controller,
            canvas_bg=self._palette.bg1,
            mark_changed=self._mark_changed,
            section="bootstrap",
        )
        self._runtime_engine_tab = RuntimeTab(
            self._controller,
            canvas_bg=self._palette.bg1,
            mark_changed=self._mark_changed,
            section="engine",
        )
        self._runtime_safety_tab = RuntimeTab(
            self._controller,
            canvas_bg=self._palette.bg1,
            mark_changed=self._mark_changed,
            section="safety",
        )
        self._manual_env_vars_tab = ManualEnvVarsTab(
            self._controller,
            canvas_bg=self._palette.bg1,
            palette=self._palette,
            mark_changed=self._mark_changed,
        )
        self._diagnostics_tab = DiagnosticsTab(
            self._controller,
            mark_changed=self._mark_changed,
            run_checks_async=self._run_checks_async,
            canvas_bg=self._palette.bg1,
        )
        self._logs_tab = LogsTab(
            self._controller,
            palette=self._palette,
            set_status=self._set_status,
        )

        tabs = [
            ("Services", self._services_tab.build(self._notebook), 0),
            ("Bootstrap", self._runtime_bootstrap_tab.build(self._notebook), 0),
            ("Engine", self._runtime_engine_tab.build(self._notebook), 0),
            ("Safety", self._runtime_safety_tab.build(self._notebook), 2),
            ("Manual Env Vars", self._manual_env_vars_tab.build(self._notebook), 0),
            ("Diagnostics", self._diagnostics_tab.build(self._notebook), 0),
            ("Logs", self._logs_tab.build(self._notebook), 0),
        ]
        for name, frame, underline in tabs:
            self._notebook.add(frame, text=name, underline=underline)
        self._notebook.enable_traversal()

        self._tab_change_guard = True
        try:
            self._restore_tab_index()
        finally:
            self._tab_change_guard = False
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_status_bar()
        self._apply_advanced_controls_visibility()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(self.POLL_INTERVAL_MS, self._poll)
        self.after(self.TASK_POLL_INTERVAL_MS, self._poll_tasks)

        self._run_checks_async()

    # ------------------------------------------------------------------ window icon

    def _set_window_icon(self) -> None:
        logo_png = self.codex_root / "logo.png"
        logo_ico = self.codex_root / "logo.ico"
        if logo_png.is_file():
            try:
                base = tk.PhotoImage(file=str(logo_png))
                width = int(base.width()) or 0
                height = int(base.height()) or 0
                if width > 0 and height > 0:
                    sizes = (16, 32, 64, 128)
                    icons: list[tk.PhotoImage] = [base]
                    for size in sizes:
                        factor = max(1, min(width, height) // size)
                        icons.append(base.subsample(factor, factor))
                    self._icon_images = icons  # type: ignore[attr-defined]
                    self.iconphoto(True, *icons)
            except Exception as exc:
                self._controller.log_buffer.log("launcher", f"Failed to set window icon from {logo_png}: {exc}", stream="event")

        if os.name == "nt" and logo_ico.is_file():
            try:
                self.iconbitmap(default=str(logo_ico))
            except Exception:
                try:
                    self.iconbitmap(str(logo_ico))
                except Exception as exc:
                    self._controller.log_buffer.log(
                        "launcher",
                        f"Failed to set Windows iconbitmap from {logo_ico}: {exc}",
                        stream="event",
                    )

    # ------------------------------------------------------------------ status bar

    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self, style="Section.Toolbar.TFrame")
        bar.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        action_group = ttk.Frame(bar, style="Section.Toolbar.TFrame")
        action_group.pack(side="left")
        ttk.Button(action_group, text="Save Settings", style="Primary.TButton", command=self._save).pack(side="left", padx=(0, 8))
        ttk.Button(action_group, text="Revert", command=self._revert).pack(side="left", padx=(0, 8))
        ttk.Button(action_group, text="Exit Without Saving", style="Quiet.TButton", command=self._exit_no_save).pack(side="left")

        prefs_group = ttk.Frame(bar, style="Section.Toolbar.TFrame")
        prefs_group.pack(side="left", padx=(16, 0))
        ttk.Checkbutton(
            prefs_group,
            text="Show advanced controls",
            variable=self._show_advanced_controls,
            command=self._on_show_advanced_controls_changed,
            style="Toggle.TCheckbutton",
        ).pack(side="left")
        ttk.Label(bar, textvariable=self._status_text, style="Statusline.TLabel").pack(side="right")

    # ------------------------------------------------------------------ status/dirty

    def _set_status(self, msg: str) -> None:
        self._status_text.set(str(msg))

    def _apply_advanced_controls_visibility(self) -> None:
        visible = bool(self._show_advanced_controls.get())
        self._runtime_bootstrap_tab.set_advanced_visible(visible)
        self._runtime_engine_tab.set_advanced_visible(visible)
        self._runtime_safety_tab.set_advanced_visible(visible)
        self._diagnostics_tab.set_advanced_visible(visible)

    def _on_show_advanced_controls_changed(self) -> None:
        self._apply_advanced_controls_visibility()
        enabled = bool(self._show_advanced_controls.get())
        if bool(getattr(self._controller.store.meta, "show_advanced_controls", False)) == enabled:
            return
        try:
            self._controller.persist_show_advanced_controls(enabled)
        except Exception as exc:
            self._controller.log_buffer.log(
                "launcher",
                f"failed to persist advanced-controls state: {exc}",
                stream="event",
            )

    def _mark_changed(self) -> None:
        self._unsaved_changes = True
        self._set_status("Unsaved changes. Start/Restart uses last saved settings until you save.")

    def _log_exception(self, label: str, exc: BaseException) -> None:
        self._controller.log_buffer.log("launcher", f"{label} failed: {exc}", stream="event")
        for line in traceback.format_exception(exc):
            for chunk in str(line).rstrip("\n").splitlines():
                self._controller.log_buffer.log("launcher", chunk, stream="event")

    # ------------------------------------------------------------------ background tasks

    def _enqueue_task(self, func: Callable[[], None]) -> None:
        self._task_queue.put(func)

    def _poll_tasks(self) -> None:
        while True:
            try:
                task = self._task_queue.get_nowait()
            except Exception:
                break
            try:
                task()
            except Exception as exc:
                self._log_exception("task", exc)
        self.after(self.TASK_POLL_INTERVAL_MS, self._poll_tasks)

    def _run_in_thread(self, label: str, func: Callable[[], None]) -> None:
        def _worker() -> None:
            try:
                func()
            except Exception as exc:
                self._log_exception(label, exc)

                def _show() -> None:
                    messagebox.showerror("Launcher Error", f"{label} failed:\n\n{exc}\n\nSee the Logs tab for details.")

                self._enqueue_task(_show)

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------ checks

    def _run_checks_async(self) -> None:
        def _worker() -> None:
            try:
                results = run_launch_checks(
                    mode_profile=str(getattr(self._controller.store.meta, "app_mode_profile", "") or ""),
                )
            except Exception as exc:
                results = [CodexLaunchCheck(name="launch-checks", ok=False, detail=str(exc))]

            def _apply() -> None:
                self._diagnostics_tab.render_checks(results)

            self._enqueue_task(_apply)

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------ polling

    def _poll(self) -> None:
        try:
            self._services_tab.refresh()
            self._logs_tab.refresh()
        except Exception as exc:
            self._log_exception("poll", exc)
        finally:
            self.after(self.POLL_INTERVAL_MS, self._poll)

    # ------------------------------------------------------------------ save/revert/exit

    def _save(self) -> bool:
        try:
            self._diagnostics_tab.validate_int_settings()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return False
        try:
            self._controller.save_settings()
        except Exception as exc:
            self._log_exception("save", exc)
            messagebox.showerror("Save failed", str(exc))
            return False
        self._unsaved_changes = False
        self._set_status("Settings saved")
        return True

    def _revert(self) -> None:
        if self._unsaved_changes and not messagebox.askyesno("Revert changes", "Discard unsaved changes?"):
            return
        self._controller.reload_store()
        self._services_tab.reload()
        self._runtime_bootstrap_tab.reload()
        self._runtime_engine_tab.reload()
        self._runtime_safety_tab.reload()
        self._manual_env_vars_tab.reload()
        self._diagnostics_tab.reload()
        self._logs_tab.reload()
        self._show_advanced_controls.set(bool(getattr(self._controller.store.meta, "show_advanced_controls", False)))
        self._apply_advanced_controls_visibility()
        self._unsaved_changes = False
        self._set_status("Reverted to saved settings")

    def _exit_no_save(self) -> None:
        self._services_tab.dispose()
        self.destroy()

    def _on_close(self) -> None:
        self._persist_ui_state()
        if self._unsaved_changes:
            if messagebox.askyesno("Unsaved changes", "Save settings before exiting?"):
                if not self._save():
                    return
        self._services_tab.dispose()
        self.destroy()

    # ------------------------------------------------------------------ UI state (tab index)

    def _restore_window_geometry(self) -> None:
        stored = str(getattr(self._controller.store.meta, "window_geometry", "") or "").strip()
        if not stored:
            self.geometry("900x680")
            return
        if not _GEOMETRY_RE.match(stored):
            self.geometry("900x680")
            self._controller.log_buffer.log("launcher", f"Ignoring invalid window_geometry {stored!r}", stream="event")
            return
        try:
            self.geometry(stored)
        except Exception as exc:
            self.geometry("900x680")
            self._controller.log_buffer.log("launcher", f"Failed to apply window_geometry {stored!r}: {exc}", stream="event")

    def _restore_tab_index(self) -> None:
        try:
            idx = int(getattr(self._controller.store.meta, "tab_index", 0))
        except Exception:
            idx = 0
        tabs = list(self._notebook.tabs())
        if not tabs:
            return
        if idx < 0 or idx >= len(tabs):
            idx = 0
        self._notebook.select(tabs[idx])

    def _persist_ui_state(self) -> None:
        self._persist_window_geometry()
        self._persist_advanced_controls()

        tabs = list(self._notebook.tabs())
        current = self._notebook.select()
        if not current or current not in tabs:
            return
        idx = tabs.index(current)
        try:
            self._controller.persist_tab_index(idx)
        except Exception as exc:
            self._controller.log_buffer.log("launcher", f"failed to persist UI state: {exc}", stream="event")

    def _persist_window_geometry(self) -> None:
        try:
            self.update_idletasks()
            geometry = str(self.geometry() or "").strip()
        except Exception as exc:
            self._controller.log_buffer.log("launcher", f"failed to read window geometry: {exc}", stream="event")
            return
        if not geometry or not _GEOMETRY_RE.match(geometry):
            return
        if str(getattr(self._controller.store.meta, "window_geometry", "") or "").strip() == geometry:
            return
        try:
            self._controller.persist_window_geometry(geometry)
        except Exception as exc:
            self._controller.log_buffer.log("launcher", f"failed to persist window geometry: {exc}", stream="event")

    def _persist_advanced_controls(self) -> None:
        enabled = bool(self._show_advanced_controls.get())
        if bool(getattr(self._controller.store.meta, "show_advanced_controls", False)) == enabled:
            return
        try:
            self._controller.persist_show_advanced_controls(enabled)
        except Exception as exc:
            self._controller.log_buffer.log(
                "launcher",
                f"failed to persist advanced-controls state: {exc}",
                stream="event",
            )

    def _on_tab_changed(self, _event: tk.Event) -> None:
        if self._tab_change_guard:
            return
        tabs = list(self._notebook.tabs())
        current = self._notebook.select()
        if not current or current not in tabs:
            return
        idx = tabs.index(current)
        if int(getattr(self._controller.store.meta, "tab_index", 0)) == idx:
            return
        try:
            self._controller.persist_tab_index(idx)
        except Exception as exc:
            self._controller.log_buffer.log("launcher", f"failed to persist tab index: {exc}", stream="event")


def main() -> None:
    app = CodexLauncherApp()
    app.mainloop()
