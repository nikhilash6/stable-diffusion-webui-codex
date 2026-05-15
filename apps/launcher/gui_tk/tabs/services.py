"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Services tab for the Tk launcher.
Shows API/UI status, launcher-owned next-start preferences, and lifecycle actions backed by `CodexServiceHandle`.
Consumes controller-owned committed-vs-live service truth so service URLs and next-start policy stay honest when the launcher has unsaved changes.

Symbols (top-level; keep in sync; no ghosts):
- `ServicesTab` (class): Services tab view/controller for the launcher GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Dict
import webbrowser

from apps.launcher.profiles import LAUNCHER_MODE_PROFILE_DEV_SERVICE
from apps.launcher.services import ServiceStatus

from ..controller import LauncherController


@dataclass(slots=True)
class _ServiceCard:
    status_var: tk.StringVar
    info_var: tk.StringVar
    endpoint_var: tk.StringVar
    status_label: ttk.Label
    endpoint_label: ttk.Label
    start_btn: ttk.Button
    restart_btn: ttk.Button
    stop_btn: ttk.Button
    kill_btn: ttk.Button
    open_btn: ttk.Button
    open_docs_btn: ttk.Button | None


class ServicesTab:
    def __init__(
        self,
        controller: LauncherController,
        *,
        run_in_thread: Callable[[str, Callable[[], None]], None],
        set_status: Callable[[str], None],
        mark_changed: Callable[[], None],
    ) -> None:
        self._controller = controller
        self._run_in_thread = run_in_thread
        self._set_status = set_status
        self._mark_changed = mark_changed

        self.frame: ttk.Frame | None = None
        self._external_terminal_var = tk.BooleanVar(value=controller.working_external_terminal())
        self._frontend_dev_typecheck_var = tk.BooleanVar(value=controller.working_frontend_dev_typecheck())
        self._cards: Dict[str, _ServiceCard] = {}
        self._boot_note_var = tk.StringVar(
            value="Service starts use last saved settings. Save Settings before Start/Restart to apply changes."
        )

    def build(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        frame.columnconfigure(0, weight=1)

        header = ttk.LabelFrame(frame, text="  Services  ", padding=14)
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        prefs = ttk.Frame(header)
        prefs.grid(row=0, column=0, sticky="w")
        prefs.columnconfigure(0, weight=1)

        pref_row = 0
        if self._controller.external_terminal_supported:
            ttk.Checkbutton(
                prefs,
                text="Launch in external terminal (Windows)",
                variable=self._external_terminal_var,
                command=self._on_toggle_external_terminal,
                style="Toggle.TCheckbutton",
            ).grid(row=pref_row, column=0, sticky="w")
            pref_row += 1
        else:
            ttk.Label(prefs, text="External terminal: Windows only", style="Muted.TLabel").grid(
                row=pref_row, column=0, sticky="w"
            )
            self._external_terminal_var.set(False)
            pref_row += 1

        if str(getattr(self._controller.store.meta, "app_mode_profile", "") or "") == LAUNCHER_MODE_PROFILE_DEV_SERVICE:
            ttk.Checkbutton(
                prefs,
                text="Run frontend typecheck before Vite startup",
                variable=self._frontend_dev_typecheck_var,
                command=self._on_frontend_dev_typecheck_changed,
                style="Toggle.TCheckbutton",
            ).grid(row=pref_row, column=0, sticky="w", pady=(4 if pref_row else 0, 0))
            pref_row += 1

        ttk.Label(prefs, textvariable=self._boot_note_var, style="Muted.TLabel", justify="left").grid(
            row=pref_row,
            column=0,
            sticky="w",
            pady=(8, 0),
        )

        btns = ttk.Frame(header)
        btns.grid(row=0, column=1, sticky="ne", padx=(16, 0))
        ttk.Button(btns, text="Start All", command=self._start_all).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Stop All", command=self._stop_all).pack(side="left")

        row = 1
        for svc_name in self._controller.service_names:
            card_frame = ttk.LabelFrame(frame, text=f"  {svc_name}  ", padding=16, style="Service.Card.TLabelframe")
            card_frame.grid(row=row, column=0, sticky="ew", padx=8, pady=(0, 10))
            card_frame.columnconfigure(1, weight=1)
            card_frame.columnconfigure(2, weight=1)

            status_var = tk.StringVar(value="STOPPED")
            info_var = tk.StringVar(value="")
            endpoint_var = tk.StringVar(value="-")

            ttk.Label(card_frame, text="Status:").grid(row=0, column=0, sticky="w", padx=(0, 10))
            status_lbl = ttk.Label(card_frame, textvariable=status_var, style="Status.Stopped.TLabel")
            status_lbl.grid(row=0, column=1, sticky="w")
            ttk.Label(card_frame, textvariable=info_var, style="Service.Info.TLabel").grid(row=0, column=2, sticky="e")

            ttk.Label(card_frame, text="Endpoint:").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(6, 0))
            endpoint_lbl = ttk.Label(card_frame, textvariable=endpoint_var, style="Service.Endpoint.TLabel")
            endpoint_lbl.grid(row=1, column=1, columnspan=2, sticky="w", pady=(6, 0))

            btn_row = ttk.Frame(card_frame)
            btn_row.grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 0))

            start_btn = ttk.Button(btn_row, text="Start", width=12, command=lambda n=svc_name: self._start(n))
            restart_btn = ttk.Button(btn_row, text="Restart", width=12, command=lambda n=svc_name: self._restart(n))
            stop_btn = ttk.Button(btn_row, text="Stop", width=12, command=lambda n=svc_name: self._stop(n))
            kill_btn = ttk.Button(btn_row, text="Kill", width=12, command=lambda n=svc_name: self._kill(n))
            open_docs_btn: ttk.Button | None = None
            if svc_name == "API":
                open_docs_btn = ttk.Button(
                    btn_row,
                    text="Docs",
                    width=10,
                    command=lambda n=svc_name: self._open_service(n, target="docs"),
                )
            open_btn = ttk.Button(btn_row, text="Open", width=10, command=lambda n=svc_name: self._open_service(n, target="root"))

            for index, button in enumerate(
                [start_btn, restart_btn, stop_btn, kill_btn, *( [open_docs_btn] if open_docs_btn is not None else []), open_btn]
            ):
                if button is None:
                    continue
                button.grid(row=0, column=index, sticky="w", padx=(0 if index == 0 else 8, 0))

            self._cards[svc_name] = _ServiceCard(
                status_var=status_var,
                info_var=info_var,
                endpoint_var=endpoint_var,
                status_label=status_lbl,
                endpoint_label=endpoint_lbl,
                start_btn=start_btn,
                restart_btn=restart_btn,
                stop_btn=stop_btn,
                kill_btn=kill_btn,
                open_btn=open_btn,
                open_docs_btn=open_docs_btn,
            )
            row += 1

        self.frame = frame
        return frame

    def reload(self) -> None:
        self._external_terminal_var.set(self._controller.working_external_terminal())
        self._frontend_dev_typecheck_var.set(self._controller.working_frontend_dev_typecheck())

    def refresh(self) -> None:
        now = time.time()
        for svc_name, card in self._cards.items():
            svc = self._controller.services[svc_name]
            status = svc.status
            pid = svc.pid or 0
            uptime = "-"
            if svc.started_at and status == ServiceStatus.RUNNING:
                uptime = f"{int(now - svc.started_at)}s"
            last_exit = svc.last_exit_code

            card.status_var.set(status.value.upper())
            info_bits = []
            if pid:
                info_bits.append(f"PID {pid}")
            if uptime != "-":
                info_bits.append(f"Uptime {uptime}")
            if last_exit is not None and status != ServiceStatus.RUNNING:
                info_bits.append(f"Last exit {last_exit}")
            card.info_var.set(" | ".join(info_bits))
            root_url, _docs_url = self._controller.service_urls(svc_name)
            card.endpoint_var.set(root_url)

            if status == ServiceStatus.RUNNING:
                card.status_label.configure(style="Status.Running.TLabel")
            elif status == ServiceStatus.ERROR:
                card.status_label.configure(style="Status.Error.TLabel")
            else:
                card.status_label.configure(style="Status.Stopped.TLabel")

            self._set_widget_enabled(card.start_btn, enabled=(status != ServiceStatus.RUNNING))
            self._set_widget_enabled(card.restart_btn, enabled=(status == ServiceStatus.RUNNING))
            self._set_widget_enabled(card.stop_btn, enabled=(status == ServiceStatus.RUNNING))
            self._set_widget_enabled(card.kill_btn, enabled=(status == ServiceStatus.RUNNING))
            if card.open_docs_btn is not None:
                self._set_widget_enabled(card.open_docs_btn, enabled=(status == ServiceStatus.RUNNING))
            self._set_widget_enabled(card.open_btn, enabled=(status == ServiceStatus.RUNNING))

    @staticmethod
    def _set_widget_enabled(widget: ttk.Widget, *, enabled: bool) -> None:
        target_state = "normal" if enabled else "disabled"
        current_state = str(widget.cget("state"))
        if current_state == target_state:
            return
        widget.configure(state=target_state)

    def dispose(self) -> None:
        return

    def _open_service(self, service_name: str, *, target: str) -> None:
        root_url, docs_url = self._controller.service_urls(service_name)
        target_url = root_url
        if target == "docs":
            if docs_url is None:
                messagebox.showerror("Launcher Error", f"{service_name} does not expose a docs URL.")
                return
            target_url = docs_url
        try:
            opened = webbrowser.open(target_url)
        except Exception as exc:
            messagebox.showerror("Launcher Error", f"Failed to open {target_url}:\n\n{exc}")
            return
        if not opened:
            messagebox.showerror("Launcher Error", f"Browser did not open {target_url}.")
            return
        self._set_status(f"Opened {target_url}")

    def _on_toggle_external_terminal(self) -> None:
        self._controller.update_external_terminal(bool(self._external_terminal_var.get()))
        self._mark_changed()

    def _on_frontend_dev_typecheck_changed(self) -> None:
        self._controller.update_frontend_dev_typecheck(bool(self._frontend_dev_typecheck_var.get()))
        self._mark_changed()

    def _start_all(self) -> None:
        self._set_status("Starting services from last saved settings…")
        self._run_in_thread("Start All", self._controller.start_all)

    def _stop_all(self) -> None:
        self._set_status("Stopping services…")
        self._run_in_thread("Stop All", lambda: self._controller.stop_all(wait=5.0))

    def _start(self, name: str) -> None:
        self._set_status(f"{name} starting from last saved settings…")
        self._run_in_thread(f"Start {name}", lambda: self._controller.start_service(name))

    def _restart(self, name: str) -> None:
        self._set_status(f"{name} restarting from last saved settings…")
        self._run_in_thread(f"Restart {name}", lambda: self._controller.restart_service(name))

    def _stop(self, name: str) -> None:
        self._set_status(f"{name} stopping…")
        self._run_in_thread(f"Stop {name}", lambda: self._controller.stop_service(name, wait=5.0))

    def _kill(self, name: str) -> None:
        if not messagebox.askyesno("Confirm kill", f"Force kill {name}?"):
            return
        self._set_status(f"{name} killing…")
        self._run_in_thread(f"Kill {name}", lambda: self._controller.kill_service(name, wait=5.0))
