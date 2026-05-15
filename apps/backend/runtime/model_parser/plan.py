"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model parser plan execution (component materialization + keymap logging).
Executes a `ParserPlan` against a state dict by materializing component tensors (including lazy views), recording key-map diagnostics, and
returning a `ParserContext` used by loaders to wire component modules.

Symbols (top-level; keep in sync; no ghosts):
- `_keymap_paths` (function): Resolves the keymap log directory/file path under `CODEX_ROOT/logs/`.
- `_append_keymap_record` (function): Appends a JSON record to the parser keymap log (best-effort).
- `_materialize_component` (function): Materializes one component tensor mapping (supports lazy `.materialize(...)`) and emits trace events.
- `execute_plan` (function): Executes a parser plan over a state dict and returns a populated `ParserContext`.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

from collections.abc import MutableMapping
from typing import Any, Dict

import logging
import json
from pathlib import Path

from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.runtime.models.state_dict import try_filter_state_dict
from apps.backend.runtime.diagnostics.trace import event as trace_event

from .errors import MissingComponentError
from .specs import ComponentState, ParserContext, ParserPlan


_log = get_backend_logger("backend.model_parser")


def _keymap_paths() -> tuple[Path, Path]:
    root = get_repo_root() / "logs"
    return root, root / "parser_keymap.log"


def _append_keymap_record(record: dict[str, Any]) -> None:
    try:
        keymap_dir, keymap_path = _keymap_paths()
        keymap_dir.mkdir(parents=True, exist_ok=True)
        with keymap_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, indent=2))
            handle.write("\n")
    except Exception:
        _log.debug("Failed to append keymap record", exc_info=True)


def _materialize_component(component: ComponentState, context: ParserContext) -> Dict[str, Any]:
    tensors = component.tensors
    materializer = getattr(tensors, "materialize", None)
    try:
        if callable(materializer):
            mapping: Dict[str, Any] = {}
            try:
                result, mapping = materializer(return_mapping=True)
            except TypeError:
                result = materializer()
            if not isinstance(result, dict):
                result = dict(result)
            trace_event(
                "parser_materialize",
                component=component.name,
                keys=len(result),
                strategy="lazy_materialize",
            )
        else:
            result = dict(tensors.items())
            mapping = {}
            trace_event(
                "parser_materialize",
                component=component.name,
                keys=len(result),
                strategy="dict_copy",
            )
        if not isinstance(mapping, dict) or not mapping:
            mapping = {key: key for key in result.keys()}
        value_sources = {id(value): mapping.get(key, key) for key, value in result.items()}
        comp_meta = context.metadata.setdefault("component_keymap", {}).setdefault(
            component.name,
            {"pre": {}, "post": {}, "value_sources": {}},
        )
        comp_meta["pre"] = mapping
        comp_meta["value_sources"] = value_sources
        try:
            signature_family = getattr(context.signature, "family", None)
        except Exception:
            signature_family = None
        _append_keymap_record(
            {
                "component": component.name,
                "stage": "pre",
                "count": len(mapping),
                "signature_family": getattr(signature_family, "value", signature_family),
                "mapping": mapping,
            }
        )
        _log.info("[parser] component=%s pre_conversion_keys=%d", component.name, len(mapping))
        return result
    except Exception as exc:
        raise RuntimeError(f"Failed to materialize component '{component.name}'") from exc


def execute_plan(plan: ParserPlan, state_dict: MutableMapping[str, Any], *, signature) -> ParserContext:
    context = ParserContext(root_state=state_dict, signature=signature, plan=plan)

    # Split components first so converters can assume presence.
    for split in plan.splits:
        view = try_filter_state_dict(state_dict, split.prefixes, new_prefix=split.strip_prefix or "")
        length = len(view)
        if length == 0:
            if split.required:
                raise MissingComponentError(split.name, detail=f"prefixes {tuple(split.prefixes)} not found")
            continue
        dtype = None
        device = None
        sample_key = next(iter(view), None)
        source_format = str(getattr(view, "source_format", "")).strip().lower()
        if sample_key is not None and source_format in {"safetensor", "safetensors"}:
            dtype_hint = getattr(view, "primary_dtype_hint", None)
            if isinstance(dtype_hint, str) and dtype_hint.strip():
                dtype = dtype_hint.strip().lower()
            view_device = getattr(view, "device", None)
            if isinstance(view_device, str) and view_device.strip():
                device = view_device.strip().lower()
            else:
                base = getattr(view, "_base", None)
                base_device = getattr(base, "device", None)
                if isinstance(base_device, str) and base_device.strip():
                    device = base_device.strip().lower()
        elif sample_key is not None:
            try:
                t = view[sample_key]
                dtype = getattr(getattr(t, "dtype", None), "name", None)
                device = getattr(getattr(t, "device", None), "type", None)
            except Exception:
                pass
        trace_event("parser_split", component=split.name, count=length, dtype=dtype, device=device)
        # Do NOT materialize the whole component here; keep the filtered mapping lazy.
        context.components[split.name] = ComponentState(name=split.name, tensors=view)

    # Apply converters sequentially.
    for converter in plan.converters:
        component = context.components.get(converter.component)
        if component is None:
            # Optional components may omit converters.
            continue
        trace_event("parser_convert_start", component=converter.component, function=converter.function.__name__)
        materialized = _materialize_component(component, context)
        updated = converter.function(materialized, context)
        if not isinstance(updated, dict):
            raise TypeError(f"Converter {converter.function.__name__} must return dict, got {type(updated)!r}")
        component.tensors = updated
        comp_meta = context.metadata.setdefault("component_keymap", {}).setdefault(
            converter.component,
            {"pre": {}, "post": {}, "value_sources": {}},
        )
        value_sources = comp_meta.get("value_sources", {})
        post_mapping: Dict[str, Any] = {}
        for key, value in updated.items():
            source = value_sources.get(id(value))
            if source is None:
                source = comp_meta.get("pre", {}).get(key)
            post_mapping[key] = source
        comp_meta["post"] = post_mapping
        trace_event("parser_keymap", component=converter.component, mapped=len(post_mapping))
        try:
            signature_family = getattr(context.signature, "family", None)
        except Exception:
            signature_family = None
        _append_keymap_record(
            {
                "component": converter.component,
                "stage": "post",
                "count": len(post_mapping),
                "signature_family": getattr(signature_family, "value", signature_family),
                "mapping": post_mapping,
            }
        )
        _log.info("[parser] component=%s post_conversion_keys=%d", converter.component, len(post_mapping))
        trace_event("parser_convert_done", component=converter.component, keys=len(component.tensors))

    # Run validations
    for validation in plan.validations:
        trace_event("parser_validate", name=validation.name)
        validation.function(context)

    return context
