"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared declarative form renderer for launcher Tk tabs.
Renders section/field descriptors into ttk widgets, tracks advanced rows for progressive disclosure, and exposes widget references for post-render state tweaks.

Symbols (top-level; keep in sync; no ghosts):
- `FormRenderer` (class): Renders `FormSectionDescriptor` + `FormFieldDescriptor` models into a grid-based ttk form.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, Iterable, List

from .form_schema import FieldKind, FormFieldDescriptor, FormSectionDescriptor, HelpMode


class FormRenderer:
    """Grid-based renderer for launcher tab forms."""

    def __init__(self, parent: ttk.Frame, *, label_column: int = 0, value_column: int = 1, padx: int = 16) -> None:
        self._parent = parent
        self._label_column = int(label_column)
        self._value_column = int(value_column)
        self._padx = int(padx)

        self._advanced_widgets: List[tk.Widget] = []
        self._field_widgets: Dict[str, tk.Widget] = {}

    def render_sections(self, row: int, sections: Iterable[FormSectionDescriptor]) -> int:
        current_row = int(row)
        for section in sections:
            current_row = self._render_section(current_row, section)
        return current_row

    def set_advanced_visible(self, visible: bool) -> None:
        if visible:
            for widget in self._advanced_widgets:
                widget.grid()
            return
        for widget in self._advanced_widgets:
            widget.grid_remove()

    def widget_for(self, field_id: str) -> tk.Widget | None:
        return self._field_widgets.get(str(field_id))

    def _render_section(self, row: int, section: FormSectionDescriptor) -> int:
        section_frame = ttk.LabelFrame(self._parent, text=f"  {section.title}  ", padding=14)
        section_frame.grid(
            row=row,
            column=self._label_column,
            columnspan=3,
            sticky="ew",
            padx=self._padx,
            pady=(18 if row > 0 else 0, 10),
        )
        section_frame.columnconfigure(0, weight=0)
        section_frame.columnconfigure(1, weight=1)
        section_frame.columnconfigure(2, weight=0)
        self._mark_advanced(section.advanced, section_frame)

        inner_row = 0
        for descriptor in section.fields:
            inner_row = self._render_field(section_frame, inner_row, descriptor, section_advanced=section.advanced)

        for help_text in section.help_texts:
            help_label = ttk.Label(section_frame, text=str(help_text), justify="left", style="Muted.TLabel")
            help_label.grid(
                row=inner_row,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(0, 8),
            )
            self._mark_advanced(section.advanced, help_label)
            inner_row += 1

        return row + 1

    def _render_field(
        self,
        parent: ttk.LabelFrame,
        row: int,
        descriptor: FormFieldDescriptor,
        *,
        section_advanced: bool,
    ) -> int:
        is_advanced = bool(section_advanced or descriptor.advanced)
        label_widget = ttk.Label(parent, text=descriptor.label)
        label_widget.grid(row=row, column=0, sticky="w", pady=8)

        widget: tk.Widget
        if descriptor.kind == FieldKind.CHOICE:
            combo = ttk.Combobox(
                parent,
                textvariable=descriptor.variable,  # type: ignore[arg-type]
                values=list(descriptor.choices),
                state="readonly",
                width=int(descriptor.width),
            )
            combo.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=8)
            combo.bind("<<ComboboxSelected>>", lambda _event: descriptor.on_change())
            widget = combo
        elif descriptor.kind == FieldKind.CHECK:
            check = ttk.Checkbutton(
                parent,
                variable=descriptor.variable,  # type: ignore[arg-type]
                command=descriptor.on_change,
                style="Toggle.TCheckbutton",
            )
            check.grid(row=row, column=1, sticky="w", padx=(10, 0), pady=8)
            widget = check
        elif descriptor.kind == FieldKind.ENTRY:
            entry = ttk.Entry(
                parent,
                textvariable=descriptor.variable,  # type: ignore[arg-type]
                width=int(descriptor.width),
            )
            entry.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=8)
            entry.bind("<KeyRelease>", lambda _event: descriptor.on_change())
            widget = entry
        elif descriptor.kind == FieldKind.ENTRY_COMMIT:
            entry = ttk.Entry(
                parent,
                textvariable=descriptor.variable,  # type: ignore[arg-type]
                width=int(descriptor.width),
            )
            entry.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=8)
            entry.bind("<FocusOut>", lambda _event: descriptor.on_change())
            entry.bind("<Return>", lambda _event: descriptor.on_change())
            widget = entry
        else:
            raise ValueError(f"Unknown field kind: {descriptor.kind}")

        self._field_widgets[descriptor.field_id] = widget
        help_button: tk.Widget | None = None
        if descriptor.help_text and descriptor.help_mode == HelpMode.DIALOG:
            title = str(descriptor.help_title or descriptor.label or "Field Help")
            message = str(descriptor.help_text)
            help_button = ttk.Button(
                parent,
                text="?",
                width=3,
                style="Help.TButton",
                takefocus=0,
                command=lambda t=title, m=message: messagebox.showinfo(t, m),
            )
            help_button.grid(
                row=row,
                column=2,
                sticky="w",
                padx=(6, 0),
                pady=8,
            )

        marked_widgets: list[tk.Widget] = [label_widget, widget]
        if help_button is not None:
            marked_widgets.append(help_button)
        self._mark_advanced(is_advanced, *marked_widgets)
        row += 1

        if descriptor.help_text and descriptor.help_mode == HelpMode.INLINE:
            help_label = ttk.Label(parent, text=str(descriptor.help_text), justify="left", style="Muted.TLabel")
            help_label.grid(
                row=row,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(0, 8),
            )
            self._mark_advanced(is_advanced, help_label)
            row += 1

        return row

    def _mark_advanced(self, advanced: bool, *widgets: tk.Widget) -> None:
        if not advanced:
            return
        for widget in widgets:
            self._advanced_widgets.append(widget)
