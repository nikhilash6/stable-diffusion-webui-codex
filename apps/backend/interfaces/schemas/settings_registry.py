# Auto-generated from settings_schema.json. Do not edit schema content by hand; update the JSON schema and regenerate.
"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Generated Python registry of UI settings schema (categories/sections/fields + IDs).
Used by the backend API to serve stable settings schema/values to the WebUI; source of truth is `settings_schema.json`.

Symbols (top-level; keep in sync; no ghosts):
- `SettingType` (enum): Enumerates supported settings field types (checkbox/slider/radio/dropdown/etc).
- `CategoryDef` (dataclass): Category metadata (id + label) for grouping sections in the UI.
- `SectionDef` (dataclass): Section metadata (key + label + optional category id) for grouping fields.
- `FieldDef` (dataclass): Field schema (type/label/default/min/max/choices) describing one UI setting input.
- `schema_to_json` (function): Converts the registry into a JSON-serializable dict (used by schema endpoints).
- `field_index` (function): Builds `{field_key → FieldDef}` mapping for quick lookup/validation.
- `CategoryId` (enum): Stable category identifiers used by the schema.
- `SectionId` (enum): Stable section identifiers used by the schema.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import StrEnum
from typing import Any, Optional

class SettingType(StrEnum):
    CHECKBOX = "checkbox"
    SLIDER = "slider"
    RADIO = "radio"
    DROPDOWN = "dropdown"
    NUMBER = "number"
    TEXT = "text"
    COLOR = "color"
    HTML = "html"

@dataclass
class CategoryDef:
    id: "CategoryId"
    label: str

@dataclass
class SectionDef:
    key: "SectionId"
    label: str
    category_id: Optional["CategoryId"]

@dataclass
class FieldDef:
    key: str
    label: str
    type: SettingType
    section: "SectionId"
    default: Any | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[Any] | None = None
    choices_source: str | None = None

def schema_to_json() -> dict:
    return {
        "categories": [asdict(c) | {"id": c.id.value} for c in CATEGORIES],
        "sections": [asdict(s) | {"key": s.key.value, "category_id": s.category_id.value if s.category_id else None} for s in SECTIONS],
        "fields": [
            {k: (v.value if hasattr(v, "value") else v) for k, v in (asdict(f) | {"type": f.type.value, "section": f.section.value}).items()}
            for f in FIELDS
        ],
        "version": 1,
        "source": "settings_registry.py"
    }

def field_index() -> dict[str, FieldDef]:
    return {f.key: f for f in FIELDS}



class CategoryId(StrEnum):
    SAVING = 'saving'
    UI = 'ui'
    SYSTEM = 'system'

CATEGORIES: list[CategoryDef] = [
    CategoryDef(CategoryId.SAVING, 'Saving images'),
    CategoryDef(CategoryId.UI, 'User Interface'),
    CategoryDef(CategoryId.SYSTEM, 'System'),
]


class SectionId(StrEnum):
    SAVING_IMAGES = 'saving-images'
    UI = 'ui'
    SYSTEM = 'system'

SECTIONS: list[SectionDef] = [
    SectionDef(SectionId.SAVING_IMAGES, 'Saving images/grids', CategoryId.SAVING),
    SectionDef(SectionId.UI, 'Live previews', CategoryId.UI),
    SectionDef(SectionId.SYSTEM, 'System', CategoryId.SYSTEM),
]


FIELDS: list[FieldDef] = [
    FieldDef(key='samples_save', label='Always save all generated images', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='samples_format', label='File format for images', type=SettingType.DROPDOWN, section=SectionId.SAVING_IMAGES, default='png', min=None, max=None, step=None, choices=['png', 'jpeg', 'webp'], choices_source=None),
    FieldDef(key='jpeg_quality', label='Quality for saved jpeg and avif images', type=SettingType.SLIDER, section=SectionId.SAVING_IMAGES, default=80, min=1, max=100, step=1, choices=None, choices_source=None),
    FieldDef(key='webp_lossless', label='Use lossless compression for webp images', type=SettingType.CHECKBOX, section=SectionId.SAVING_IMAGES, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='live_previews_enable', label='Show live previews of the created image', type=SettingType.CHECKBOX, section=SectionId.UI, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='live_previews_image_format', label='Live preview file format', type=SettingType.RADIO, section=SectionId.UI, default='png', min=None, max=None, step=None, choices=['jpeg', 'png', 'webp'], choices_source=None),
    FieldDef(key='show_progress_every_n_steps', label='Live preview display period', type=SettingType.SLIDER, section=SectionId.UI, default=10, min=-1, max=32, step=1, choices=None, choices_source=None),
    FieldDef(key='show_progress_type', label='Live preview method', type=SettingType.RADIO, section=SectionId.UI, default='Approx cheap', min=None, max=None, step=None, choices=['Full', 'Approx cheap'], choices_source=None),
    FieldDef(key='codex_export_video', label='(Codex) Export video (tasks that return video)', type=SettingType.CHECKBOX, section=SectionId.SYSTEM, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='codex_attention_backend', label='(Codex) Attention backend', type=SettingType.DROPDOWN, section=SectionId.SYSTEM, default='pytorch', min=None, max=None, step=None, choices=['pytorch', 'xformers', 'split', 'quad'], choices_source=None),
    FieldDef(key='codex_main_device', label='(Codex) Main device', type=SettingType.DROPDOWN, section=SectionId.SYSTEM, default='auto', min=None, max=None, step=None, choices=['auto', 'cuda', 'cpu', 'mps', 'xpu', 'directml'], choices_source=None),
    FieldDef(key='codex_core_dtype', label='(Codex) Core storage dtype', type=SettingType.DROPDOWN, section=SectionId.SYSTEM, default='auto', min=None, max=None, step=None, choices=['auto', 'fp16', 'bf16', 'fp32'], choices_source=None),
    FieldDef(key='codex_core_compute_dtype', label='(Codex) Core compute dtype', type=SettingType.DROPDOWN, section=SectionId.SYSTEM, default='auto', min=None, max=None, step=None, choices=['auto', 'fp16', 'bf16', 'fp32'], choices_source=None),
    FieldDef(key='codex_te_dtype', label='(Codex) Text encoder storage dtype', type=SettingType.DROPDOWN, section=SectionId.SYSTEM, default='auto', min=None, max=None, step=None, choices=['auto', 'fp16', 'bf16', 'fp32'], choices_source=None),
    FieldDef(key='codex_te_compute_dtype', label='(Codex) Text encoder compute dtype', type=SettingType.DROPDOWN, section=SectionId.SYSTEM, default='auto', min=None, max=None, step=None, choices=['auto', 'fp16', 'bf16', 'fp32'], choices_source=None),
    FieldDef(key='codex_vae_dtype', label='(Codex) VAE storage dtype', type=SettingType.DROPDOWN, section=SectionId.SYSTEM, default='auto', min=None, max=None, step=None, choices=['auto', 'fp16', 'bf16', 'fp32'], choices_source=None),
    FieldDef(key='codex_vae_compute_dtype', label='(Codex) VAE compute dtype', type=SettingType.DROPDOWN, section=SectionId.SYSTEM, default='auto', min=None, max=None, step=None, choices=['auto', 'fp16', 'bf16', 'fp32'], choices_source=None),
    FieldDef(key='codex_vae_by_family', label='(Codex) VAE by family (JSON map)', type=SettingType.TEXT, section=SectionId.SYSTEM, default='{}', min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='codex_smart_offload', label='(Codex) Smart offload (TE/VAE unload between stages)', type=SettingType.CHECKBOX, section=SectionId.SYSTEM, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='codex_smart_fallback', label='(Codex) Smart fallback (CPU on OOM)', type=SettingType.CHECKBOX, section=SectionId.SYSTEM, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='codex_smart_cache', label='(Codex) Smart cache (SDXL conditioning cache)', type=SettingType.CHECKBOX, section=SectionId.SYSTEM, default=True, min=None, max=None, step=None, choices=None, choices_source=None),
    FieldDef(key='codex_core_streaming', label='(Codex) Core streaming (experimental)', type=SettingType.CHECKBOX, section=SectionId.SYSTEM, default=False, min=None, max=None, step=None, choices=None, choices_source=None),
]
