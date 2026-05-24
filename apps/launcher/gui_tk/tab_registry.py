"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tk launcher tab lifecycle registry.
Owns tab order plus reload/refresh/advanced/dispose fan-out so `app.py` orchestrates by capability instead of hard-coded repeated calls.

Symbols (top-level; keep in sync; no ghosts):
- `LauncherTabEntry` (dataclass): One built launcher tab registration.
- `LauncherTabRegistry` (class): Ordered tab registry with lifecycle fan-out helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from tkinter import ttk
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class LauncherTabEntry:
    tab_id: str
    label: str
    underline: int
    tab: Any
    advanced_participant: bool = False
    refresh_participant: bool = False
    dispose_participant: bool = False


class LauncherTabRegistry:
    def __init__(self, entries: Iterable[LauncherTabEntry]) -> None:
        self._entries = tuple(entries)
        ids = [entry.tab_id for entry in self._entries]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate launcher tab id(s): {ids!r}")

    def add_to_notebook(self, notebook: ttk.Notebook) -> None:
        for entry in self._entries:
            frame = entry.tab.build(notebook)
            notebook.add(frame, text=entry.label, underline=entry.underline)

    def tab(self, tab_id: str) -> Any:
        for entry in self._entries:
            if entry.tab_id == tab_id:
                return entry.tab
        raise KeyError(tab_id)

    def apply_advanced_visible(self, visible: bool) -> None:
        for entry in self._entries:
            if not entry.advanced_participant:
                continue
            entry.tab.set_advanced_visible(visible)

    def reload_all(self) -> None:
        for entry in self._entries:
            entry.tab.reload()

    def refresh_all(self) -> None:
        for entry in self._entries:
            if not entry.refresh_participant:
                continue
            entry.tab.refresh()

    def dispose_all(self) -> None:
        for entry in self._entries:
            if not entry.dispose_participant:
                continue
            entry.tab.dispose()
