"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Declarative schema models for launcher Tk forms.
Defines typed descriptors for sections and fields so tabs can render settings declaratively instead of hand-writing each widget row, including visible VRAM impact metadata.

Symbols (top-level; keep in sync; no ghosts):
- `FieldKind` (class): Supported form input kinds for launcher setting rows.
- `HelpMode` (class): Supported help rendering modes for field descriptions (inline text or dialog).
- `FormFieldDescriptor` (dataclass): Declarative field definition consumed by the shared form renderer.
- `FormSectionDescriptor` (dataclass): Declarative section definition containing ordered form fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import tkinter as tk
from typing import Callable, Sequence

from apps.launcher.setting_registry import VramImpactMetadata


class FieldKind(StrEnum):
    """Supported input kinds for launcher form descriptors."""

    CHOICE = "choice"
    CHECK = "check"
    ENTRY = "entry"
    ENTRY_COMMIT = "entry_commit"


class HelpMode(StrEnum):
    """Supported help rendering modes for field descriptions."""

    INLINE = "inline"
    DIALOG = "dialog"


@dataclass(frozen=True, slots=True)
class FormFieldDescriptor:
    """Declarative field definition rendered by `FormRenderer`."""

    field_id: str
    kind: FieldKind
    label: str
    variable: tk.Variable
    on_change: Callable[[], None]
    choices: Sequence[str] = ()
    width: int = 18
    advanced: bool = False
    vram: VramImpactMetadata | None = None
    help_text: str | None = None
    help_mode: HelpMode = HelpMode.INLINE
    help_title: str | None = None


@dataclass(frozen=True, slots=True)
class FormSectionDescriptor:
    """Declarative section definition with ordered fields."""

    title: str
    fields: Sequence[FormFieldDescriptor] = field(default_factory=tuple)
    help_texts: Sequence[str] = field(default_factory=tuple)
    advanced: bool = False
