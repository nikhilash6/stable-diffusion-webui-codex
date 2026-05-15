"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tk/ttk styles used by the Codex launcher GUI.
Centralizes palette values, cross-platform font resolution, and ttk style configuration so the app/tabs do not hardcode colors or fonts.

Symbols (top-level; keep in sync; no ghosts):
- `Palette` (dataclass): Color palette used by the GUI.
- `LauncherFonts` (dataclass): Cached Tk font set used by the launcher UI.
- `resolve_fonts` (function): Returns the cached launcher font set for a Tk root.
- `apply_style` (function): Applies ttk theme/style configuration to a Tk root window.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk


@dataclass(frozen=True, slots=True)
class Palette:
    bg0: str = "#0d1118"
    bg1: str = "#141b27"
    bg2: str = "#1b2534"
    fg0: str = "#e8eef8"
    fg_muted: str = "#9bb0c9"
    accent: str = "#57a6ff"
    accent_hover: str = "#7ebcff"
    accent_active: str = "#2f8df4"
    line: str = "#2a3a52"
    ok: str = "#50d2a0"
    warn: str = "#f0c66c"
    err: str = "#ff6f91"


@dataclass(frozen=True, slots=True)
class LauncherFonts:
    body: tkfont.Font
    body_strong: tkfont.Font
    tab: tkfont.Font
    button: tkfont.Font
    mono: tkfont.Font
    small: tkfont.Font
    small_strong: tkfont.Font


def _copy_font(
    base: tkfont.Font,
    *,
    family: str | None = None,
    size: int | None = None,
    weight: str | None = None,
) -> tkfont.Font:
    copied = base.copy()
    if family is not None:
        copied.configure(family=family)
    if size is not None:
        copied.configure(size=int(size))
    if weight is not None:
        copied.configure(weight=weight)
    return copied


def resolve_fonts(root: tk.Misc) -> LauncherFonts:
    cached = getattr(root, "_codex_launcher_fonts", None)
    if isinstance(cached, LauncherFonts):
        return cached

    default_font = tkfont.nametofont("TkDefaultFont")
    fixed_font = tkfont.nametofont("TkFixedFont")

    body_family = "Segoe UI" if os.name == "nt" else str(default_font.cget("family"))
    mono_family = "Consolas" if os.name == "nt" else str(fixed_font.cget("family"))

    body = _copy_font(default_font, family=body_family, size=10)
    body_strong = _copy_font(body, weight="bold")
    tab = _copy_font(body, size=9, weight="bold")
    button = _copy_font(body, size=9, weight="bold")
    mono = _copy_font(fixed_font, family=mono_family, size=10)
    small = _copy_font(body, size=9)
    small_strong = _copy_font(body, size=9, weight="bold")

    fonts = LauncherFonts(
        body=body,
        body_strong=body_strong,
        tab=tab,
        button=button,
        mono=mono,
        small=small,
        small_strong=small_strong,
    )
    setattr(root, "_codex_launcher_fonts", fonts)
    return fonts


def apply_style(root: tk.Tk, palette: Palette) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")
    fonts = resolve_fonts(root)

    root.configure(bg=palette.bg0)

    style.configure(".", background=palette.bg0, foreground=palette.fg0, font=fonts.body)

    style.configure("TFrame", background=palette.bg0)
    style.configure("Section.Toolbar.TFrame", background=palette.bg0)
    style.configure("TLabel", background=palette.bg0, foreground=palette.fg0)
    style.configure("Muted.TLabel", background=palette.bg0, foreground=palette.fg_muted)
    style.configure("Section.Header.TLabel", background=palette.bg0, foreground=palette.accent, font=fonts.body_strong)
    style.configure("Statusline.TLabel", background=palette.bg0, foreground=palette.fg_muted, font=fonts.small)

    style.configure(
        "TNotebook",
        background=palette.bg0,
        borderwidth=1,
        relief="solid",
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
    )
    style.configure(
        "TNotebook.Tab",
        background=palette.bg1,
        foreground=palette.fg_muted,
        padding=[14, 8],
        borderwidth=1,
        relief="flat",
        font=fonts.tab,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", palette.bg2), ("active", palette.bg2)],
        foreground=[("selected", palette.fg0), ("active", palette.fg0)],
    )

    style.configure(
        "TLabelframe",
        background=palette.bg1,
        bordercolor=palette.line,
        relief="solid",
        borderwidth=1,
        lightcolor=palette.line,
        darkcolor=palette.line,
    )
    style.configure(
        "TLabelframe.Label",
        background=palette.bg1,
        foreground=palette.accent,
        font=fonts.small_strong,
    )
    style.configure(
        "Service.Card.TLabelframe",
        background=palette.bg1,
        bordercolor=palette.line,
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "Service.Card.TLabelframe.Label",
        background=palette.bg1,
        foreground=palette.fg0,
        font=fonts.body_strong,
    )

    style.configure(
        "TButton",
        background=palette.bg2,
        foreground=palette.fg0,
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
        relief="solid",
        padding=[14, 7],
        font=fonts.button,
    )
    style.map(
        "TButton",
        background=[("active", palette.accent_hover), ("pressed", palette.accent_active), ("disabled", palette.bg1)],
        foreground=[("active", palette.bg0), ("pressed", palette.bg0), ("disabled", palette.fg_muted)],
    )
    style.configure(
        "Primary.TButton",
        background=palette.accent,
        foreground=palette.bg0,
        bordercolor=palette.accent_active,
        lightcolor=palette.accent_active,
        darkcolor=palette.accent_active,
        relief="solid",
        padding=[14, 7],
        font=fonts.button,
    )
    style.map(
        "Primary.TButton",
        background=[("active", palette.accent_hover), ("pressed", palette.accent_active), ("disabled", palette.bg1)],
        foreground=[("active", palette.bg0), ("pressed", palette.bg0), ("disabled", palette.fg_muted)],
    )
    style.configure(
        "Quiet.TButton",
        background=palette.bg1,
        foreground=palette.fg_muted,
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
        relief="solid",
        padding=[14, 7],
        font=fonts.button,
    )
    style.map(
        "Quiet.TButton",
        background=[("active", palette.bg2), ("pressed", palette.bg2), ("disabled", palette.bg1)],
        foreground=[("active", palette.fg0), ("pressed", palette.fg0), ("disabled", palette.fg_muted)],
    )
    style.configure(
        "Help.TButton",
        background=palette.bg1,
        foreground=palette.accent,
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
        relief="solid",
        padding=[4, 2],
        font=fonts.small_strong,
    )
    style.map(
        "Help.TButton",
        background=[("active", palette.accent_hover), ("pressed", palette.accent_active)],
        foreground=[("active", palette.bg0), ("pressed", palette.bg0)],
    )

    style.configure("Filter.TButton", background=palette.bg1, foreground=palette.fg0, padding=[10, 4], font=fonts.small)
    style.map(
        "Filter.TButton",
        background=[("active", palette.bg2), ("disabled", palette.bg1)],
        foreground=[("active", palette.fg0), ("disabled", palette.fg_muted)],
    )
    style.configure(
        "Filter.Selected.TButton",
        background=palette.accent,
        foreground=palette.bg0,
        padding=[10, 4],
        font=fonts.small_strong,
    )
    style.map(
        "Filter.Selected.TButton",
        background=[("active", palette.accent_hover), ("disabled", palette.bg1)],
        foreground=[("active", palette.bg0), ("disabled", palette.fg_muted)],
    )

    style.configure("TCheckbutton", background=palette.bg0, foreground=palette.fg0, font=fonts.small)
    style.map(
        "TCheckbutton",
        background=[("active", palette.bg0), ("disabled", palette.bg0)],
        foreground=[("disabled", palette.fg_muted)],
    )
    style.configure("Toggle.TCheckbutton", background=palette.bg0, foreground=palette.fg0, font=fonts.small)
    style.map(
        "Toggle.TCheckbutton",
        background=[("active", palette.bg0), ("disabled", palette.bg0)],
        foreground=[("disabled", palette.fg_muted)],
    )

    style.configure(
        "TEntry",
        fieldbackground=palette.bg2,
        background=palette.bg2,
        foreground=palette.fg0,
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
        insertcolor=palette.fg0,
        font=fonts.body,
    )
    style.map(
        "TEntry",
        fieldbackground=[("disabled", palette.bg1)],
        foreground=[("disabled", palette.fg_muted)],
    )

    style.configure(
        "TCombobox",
        fieldbackground=palette.bg2,
        background=palette.bg2,
        foreground=palette.fg0,
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
        arrowcolor=palette.fg0,
        font=fonts.body,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", palette.bg2), ("disabled", palette.bg1)],
        foreground=[("readonly", palette.fg0), ("disabled", palette.fg_muted)],
    )
    root.option_add("*TCombobox*Listbox.background", palette.bg2)
    root.option_add("*TCombobox*Listbox.foreground", palette.fg0)
    root.option_add("*TCombobox*Listbox.selectBackground", palette.accent)
    root.option_add("*TCombobox*Listbox.selectForeground", palette.bg0)

    style.configure(
        "Treeview",
        background=palette.bg2,
        fieldbackground=palette.bg2,
        foreground=palette.fg0,
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
        font=fonts.small,
    )
    style.configure(
        "Treeview.Heading",
        background=palette.bg1,
        foreground=palette.accent,
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
        relief="solid",
        font=fonts.small_strong,
    )
    style.map("Treeview", background=[("selected", palette.accent)], foreground=[("selected", palette.bg0)])

    style.configure(
        "Vertical.TScrollbar",
        background=palette.bg2,
        troughcolor=palette.bg1,
        bordercolor=palette.line,
        arrowcolor=palette.fg_muted,
        lightcolor=palette.line,
        darkcolor=palette.line,
    )
    style.map(
        "Vertical.TScrollbar",
        background=[("active", palette.bg2), ("pressed", palette.accent_active)],
        arrowcolor=[("active", palette.fg0), ("pressed", palette.bg0)],
    )
    style.configure(
        "Horizontal.TScrollbar",
        background=palette.bg2,
        troughcolor=palette.bg1,
        bordercolor=palette.line,
        arrowcolor=palette.fg_muted,
        lightcolor=palette.line,
        darkcolor=palette.line,
    )
    style.map(
        "Horizontal.TScrollbar",
        background=[("active", palette.bg2), ("pressed", palette.accent_active)],
        arrowcolor=[("active", palette.fg0), ("pressed", palette.bg0)],
    )

    style.configure("Service.Info.TLabel", background=palette.bg1, foreground=palette.fg_muted, font=fonts.small)
    style.configure("Service.Endpoint.TLabel", background=palette.bg1, foreground=palette.fg0, font=fonts.mono)

    style.configure("Status.Running.TLabel", background=palette.bg1, foreground=palette.ok, font=fonts.body_strong)
    style.configure("Status.Stopped.TLabel", background=palette.bg1, foreground=palette.warn, font=fonts.body_strong)
    style.configure("Status.Error.TLabel", background=palette.bg1, foreground=palette.err, font=fonts.body_strong)
