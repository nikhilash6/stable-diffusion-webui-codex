"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared Tk form shell for launcher runtime setting tabs.
Owns only scrollable form rendering and advanced visibility plumbing; concrete Bootstrap, Engine, and Safety tabs own their setting descriptors and env writes.

Symbols (top-level; keep in sync; no ghosts):
- `RuntimeFormTabBase` (class): Base renderer/advanced plumbing for runtime setting tabs.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Sequence

from ..controller import LauncherController
from ..form_renderer import FormRenderer
from ..form_schema import FormSectionDescriptor
from ..widgets import ScrollableFrame


class RuntimeFormTabBase:
    def __init__(self, controller: LauncherController, *, canvas_bg: str, mark_changed: Callable[[], None]) -> None:
        self._controller = controller
        self._canvas_bg = canvas_bg
        self._mark_changed = mark_changed
        self.frame: ttk.Frame | None = None
        self._advanced_visible = False
        self._form_renderer: FormRenderer | None = None

    def build(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        scroll = ScrollableFrame(frame, canvas_bg=self._canvas_bg)
        scroll.pack(fill="both", expand=True, padx=8, pady=8)
        body = scroll.inner
        body.columnconfigure(0, weight=1)

        renderer = FormRenderer(body)
        renderer.render_sections(0, self._sections_for_view())
        self._form_renderer = renderer
        self._after_render(renderer)
        self._apply_advanced_visibility()

        self.frame = frame
        self.reload()
        return frame

    def set_advanced_visible(self, visible: bool) -> None:
        self._advanced_visible = bool(visible)
        self._apply_advanced_visibility()

    def _apply_advanced_visibility(self) -> None:
        if self._form_renderer is not None:
            self._form_renderer.set_advanced_visible(self._advanced_visible)

    def _sections_for_view(self) -> Sequence[FormSectionDescriptor]:
        raise NotImplementedError("runtime tab sections must be implemented by concrete tabs")

    def _after_render(self, renderer: FormRenderer) -> None:
        _ = renderer

    def reload(self) -> None:
        raise NotImplementedError("runtime tab reload must be implemented by concrete tabs")

    def _mark_dirty(self) -> None:
        self._mark_changed()
