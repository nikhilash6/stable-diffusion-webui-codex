"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Logs tab for the Tk launcher.
Displays structured launcher/service logs with filtering, search highlighting, and incremental rendering.

Symbols (top-level; keep in sync; no ghosts):
- `LogsTab` (class): Log viewer tab (filter/search/export).
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Dict, List

from apps.launcher.log_buffer import CodexLogRecord, format_log_record

from ..controller import LauncherController
from ..styles import Palette, resolve_fonts


class LogsTab:
    def __init__(
        self,
        controller: LauncherController,
        *,
        palette: Palette,
        set_status: Callable[[str], None],
    ) -> None:
        self._controller = controller
        self._palette = palette
        self._set_status = set_status

        self.frame: ttk.Frame | None = None

        self._var_filter = tk.StringVar(value="All")
        self._var_search = tk.StringVar(value="")
        self._var_autoscroll = tk.BooleanVar(value=True)
        self._search_after_id: str | None = None

        self._filter_buttons: Dict[str, ttk.Button] = {}

        self._text: ScrolledText | None = None
        self._last_rendered_id: int | None = None
        self._last_filter: str = "All"
        self._force_redraw = False

    def build(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        top = ttk.Frame(frame)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        ttk.Label(top, text="Source:").pack(side="left")

        for text, value in (("All", "All"), ("Launcher", "launcher"), ("API", "API"), ("UI", "UI")):
            btn = ttk.Button(top, text=text, style="Filter.TButton", command=lambda v=value: self._set_filter(v))
            btn.pack(side="left", padx=(6, 0))
            self._filter_buttons[value] = btn
        self._update_filter_buttons()

        ttk.Label(top, text="Search:").pack(side="left", padx=(16, 0))
        entry = ttk.Entry(top, textvariable=self._var_search, width=28)
        entry.pack(side="left", padx=(6, 0))
        entry.bind("<KeyRelease>", lambda _e: self._schedule_search_highlight())
        ttk.Button(top, text="✕", width=3, command=self._clear_search).pack(side="left", padx=(6, 0))

        text = ScrolledText(
            frame,
            wrap="word",
            state="disabled",
            bg=self._palette.bg2,
            fg=self._palette.fg0,
            insertbackground=self._palette.fg0,
            font=resolve_fonts(frame).mono,
        )
        text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        text.tag_config("search_match", background=self._palette.accent, foreground=self._palette.bg0)
        self._text = text

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        ttk.Button(btn_row, text="Clear", command=self._clear).pack(side="left")
        ttk.Button(btn_row, text="Copy", command=self._copy).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Export", command=self._export).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(btn_row, text="Auto-scroll", variable=self._var_autoscroll).pack(side="right")

        self.frame = frame
        return frame

    def reload(self) -> None:
        self._force_redraw = True

    def refresh(self) -> None:
        if self._text is None:
            return

        records = self._controller.log_buffer.snapshot()
        current_filter = str(self._var_filter.get() or "All")

        if self._force_redraw or current_filter != self._last_filter:
            self._force_redraw = False
            self._last_filter = current_filter
            self._last_rendered_id = None
            self._replace([format_log_record(r) for r in records if self._match_filter(r, current_filter)])
            if records:
                self._last_rendered_id = records[-1].id
            return

        last_id = self._last_rendered_id
        if last_id is None:
            self._replace([format_log_record(r) for r in records if self._match_filter(r, current_filter)])
            if records:
                self._last_rendered_id = records[-1].id
            return

        new_lines = [format_log_record(r) for r in records if r.id > last_id and self._match_filter(r, current_filter)]
        if new_lines:
            self._append(new_lines)
        if records:
            self._last_rendered_id = records[-1].id

    # ------------------------------------------------------------------ filter + search

    def _match_filter(self, record: CodexLogRecord, filter_value: str) -> bool:
        if filter_value == "All":
            return True
        return str(record.source) == filter_value

    def _set_filter(self, value: str) -> None:
        self._var_filter.set(str(value))
        self._update_filter_buttons()
        self._force_redraw = True

    def _update_filter_buttons(self) -> None:
        selected = str(self._var_filter.get() or "All")
        for value, btn in self._filter_buttons.items():
            style = "Filter.Selected.TButton" if value == selected else "Filter.TButton"
            btn.configure(style=style)

    def _clear_search(self) -> None:
        self._var_search.set("")
        self._schedule_search_highlight()

    def _schedule_search_highlight(self) -> None:
        if self._text is None:
            return
        if self._search_after_id:
            try:
                self._text.after_cancel(self._search_after_id)
            except Exception:
                pass
        self._search_after_id = self._text.after(180, self._apply_search_highlights_full)

    def _apply_search_highlights_full(self) -> None:
        if self._text is None:
            return
        text = self._text
        text.tag_remove("search_match", "1.0", "end")
        query = str(self._var_search.get() or "").strip()
        if not query:
            return
        self._apply_search_highlights_range("1.0", "end")

    def _apply_search_highlights_range(self, start: str, end: str) -> None:
        if self._text is None:
            return
        query = str(self._var_search.get() or "").strip()
        if not query:
            return
        text = self._text
        idx = start
        while True:
            match = text.search(query, idx, stopindex=end, nocase=True, regexp=False)
            if not match:
                break
            last = f"{match}+{len(query)}c"
            text.tag_add("search_match", match, last)
            idx = last

    # ------------------------------------------------------------------ widget mutations

    def _replace(self, lines: List[str]) -> None:
        if self._text is None:
            return
        text = self._text
        text.configure(state="normal")
        text.delete("1.0", "end")
        if lines:
            text.insert("end", "\n".join(lines) + "\n")
        text.configure(state="disabled")
        self._apply_search_highlights_full()
        if self._var_autoscroll.get():
            text.see("end")

    def _append(self, lines: List[str]) -> None:
        if self._text is None:
            return
        text = self._text
        start_idx = text.index("end-1c")
        text.configure(state="normal")
        for line in lines:
            text.insert("end", line + "\n")
        end_idx = text.index("end-1c")
        text.configure(state="disabled")
        self._apply_search_highlights_range(start_idx, end_idx)
        if self._var_autoscroll.get():
            text.see("end")

    # ------------------------------------------------------------------ actions

    def _clear(self) -> None:
        self._controller.log_buffer.clear()
        self._force_redraw = True

    def _copy(self) -> None:
        if self._text is None:
            return
        payload = self._text.get("1.0", "end-1c")
        self._text.clipboard_clear()
        self._text.clipboard_append(payload)
        self._set_status("Logs copied to clipboard")

    def _export(self) -> None:
        if self._text is None:
            return
        logs_dir = self._controller.codex_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = logs_dir / f"codex-launcher-{stamp}.log"
        out_path.write_text(self._text.get("1.0", "end-1c"), encoding="utf-8")
        self._set_status(f"Exported logs to {out_path}")
