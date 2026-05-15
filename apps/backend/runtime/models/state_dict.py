"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: State dict loading and key-mapping helpers with trace/keymap logging.
Provides relaxed load helpers that log missing/unexpected keys and write diagnostics to `logs/parser_keymap.log`, plus utilities for
filtering and renaming keys (including transformer-style conversions) without eagerly materializing tensors (SafeTensors is only materialized up-front on Windows).
Structural conversion operations (e.g. fused in_proj -> split Q/K/V) are globally policy-gated by
`CODEX_WEIGHT_STRUCTURAL_CONVERSION` (`auto`=forbid, `convert`=allow).

Symbols (top-level; keep in sync; no ghosts):
- `_append_key_record` (function): Appends a JSON record to the parser keymap log.
- `load_state_dict` (function): Loads a state dict with strict=False and logs missing/unexpected keys.
- `state_dict_has` (function): Returns True if any state-dict key starts with a prefix.
- `filter_state_dict_with_prefix` (function): Returns a lazy `FilterPrefixView` filtered by prefix (optional re-prefix).
- `try_filter_state_dict` (function): Tries multiple prefixes and returns the first matching lazy view (or empty view).
- `transformers_convert` (function): Renames legacy transformer keys to HF-style encoder keys (incl. Q/K/V split).
- `state_dict_key_replace` (function): Applies a direct `{old: new}` key replacement mapping.
- `state_dict_prefix_replace` (function): Rewrites keys using multiple prefix replacements (optionally into a new dict).
- `safe_load_state_dict` (function): Conservative loader that copies tensors key-by-key, supports caller-owned allowed-missing prefix suppression for proven staged partial loads, and returns (missing, unexpected) (Windows: materializes SafeTensors once).
"""

import torch
import logging
import json
import sys
from pathlib import Path
from apps.backend.runtime.logging import get_backend_logger

from apps.backend.infra.config.weight_structural_conversion import (
    ENV_WEIGHT_STRUCTURAL_CONVERSION,
    is_structural_weight_conversion_enabled,
)
from apps.backend.runtime import trace as _trace
from apps.backend.runtime.state_dict.views import FilterPrefixView

_log = get_backend_logger("backend.state_dict")
_KEYMAP_DIR = Path("logs")
_KEYMAP_PATH = _KEYMAP_DIR / "parser_keymap.log"


def _append_key_record(record: dict[str, object]) -> None:
    try:
        _KEYMAP_DIR.mkdir(parents=True, exist_ok=True)
        with _KEYMAP_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, indent=2))
            handle.write("\n")
    except Exception:
        _log.debug("Failed to append key record", exc_info=True)


def load_state_dict(model, sd, ignore_errors=[], log_name=None, ignore_start=None):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [x for x in missing if x not in ignore_errors]
    unexpected = [x for x in unexpected if x not in ignore_errors]

    if isinstance(ignore_start, str):
        missing = [x for x in missing if not x.startswith(ignore_start)]
        unexpected = [x for x in unexpected if not x.startswith(ignore_start)]

    log_name = log_name or type(model).__name__
    if len(missing) > 0:
        _log.warning('%s Missing: %d keys', log_name, len(missing))
        # Sample a few keys at DEBUG for diagnostics
        _log.debug("%s missing_count=%d sample=%s", log_name, len(missing), missing[:10])
        _log.info("%s missing_keys=%s", log_name, missing)
        _append_key_record({
            "component": log_name,
            "stage": "load_missing",
            "count": len(missing),
            "keys": missing,
        })
    if len(unexpected) > 0:
        _log.warning('%s Unexpected: %d keys', log_name, len(unexpected))
        _log.debug("%s unexpected_count=%d sample=%s", log_name, len(unexpected), unexpected[:10])
        _log.info("%s unexpected_keys=%s", log_name, unexpected)
        _append_key_record({
            "component": log_name,
            "stage": "load_unexpected",
            "count": len(unexpected),
            "keys": unexpected,
        })
    _trace.event("load_state_dict_done", name=log_name, missing=len(missing), unexpected=len(unexpected))
    return


def state_dict_has(sd, prefix):
    return any(x.startswith(prefix) for x in sd.keys())


def filter_state_dict_with_prefix(sd, prefix, new_prefix=''):
    """Return a lazy view filtered by `prefix`, optionally re-prefixed.

    Avoid materializing tensors while building the subset; deletion from the
    base mapping is skipped to preserve laziness and stability.
    """
    return FilterPrefixView(sd, prefix, new_prefix)


def try_filter_state_dict(sd, prefix_list, new_prefix=''):
    for prefix in prefix_list:
        if state_dict_has(sd, prefix):
            return filter_state_dict_with_prefix(sd, prefix, new_prefix)
    return FilterPrefixView(sd, "__no_such_prefix__/")  # empty view


def transformers_convert(sd, prefix_from, prefix_to, number):
    allow_structural_conversion = is_structural_weight_conversion_enabled()
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                if not allow_structural_conversion:
                    raise RuntimeError(
                        "transformers_convert: structural conversion is disabled by policy "
                        f"({ENV_WEIGHT_STRUCTURAL_CONVERSION}=auto). "
                        "Cannot split OpenCLIP fused in_proj tensors into q/k/v projections. "
                        f"Set {ENV_WEIGHT_STRUCTURAL_CONVERSION}=convert to allow."
                    )
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from*x:shape_from*(x + 1)]
    return sd


def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def safe_load_state_dict(model, sd, *, log_name=None, ignore_missing_prefixes=()):
    """Conservative loader: iterates model keys and copies tensors one by one.

    Avoids materializing all tensors and reduces device/dtype edge cases.
    Emits periodic trace events. Returns (missing, unexpected) like nn.Module.load_state_dict.
    """
    from collections.abc import Mapping
    log_name = log_name or type(model).__name__

    model_state = model.state_dict()
    model_keys = list(model_state.keys())

    # If sd is a lazy safetensors dict, materialize it once on Windows to avoid reopening
    # the file repeatedly (torch_cpu.dll crash prevention). On non-Windows platforms we keep
    # lazy access so weights stream one tensor at a time.
    if sys.platform.startswith("win"):
        materializer = getattr(sd, "materialize", None)
        if callable(materializer):
            try:
                sd = materializer()
            except Exception:
                pass  # Fall back to lazy access if materialize fails

    sd_keys = list(sd.keys()) if isinstance(sd, Mapping) and hasattr(sd, 'keys') else []

    missing = []
    loaded = 0
    # Begin: diagnostics handled by trace/logger upstream (no console prints)
    for k in model_keys:
        try:
            t = sd[k]
        except Exception:
            missing.append(k)
            continue
        p = model_state.get(k)
        if not isinstance(t, torch.Tensor) or p is None:
            missing.append(k)
            continue
        try:
            target_device = p.device
            # Avoid unnecessary CPU staging: copy directly to target when possible
            if isinstance(t, torch.Tensor):
                if t.device == target_device:
                    t_cast = t.detach().to(dtype=p.dtype)
                    p.copy_(t_cast)
                else:
                    t_cast = t.detach().to(device=target_device, dtype=p.dtype)
                    p.copy_(t_cast)
            else:
                raise TypeError("state_dict value is not a tensor")
        except Exception:
            _log.exception("safe_load_state_dict: failed key=%s", k)
            missing.append(k)
            continue
        loaded += 1
        if loaded % 200 == 0:
            _trace.event("load_state_dict_progress", name=log_name, loaded=loaded)

    unexpected = [k for k in sd_keys if k not in model_keys]
    if ignore_missing_prefixes:
        normalized_prefixes = tuple(str(prefix) for prefix in ignore_missing_prefixes if str(prefix))
        if normalized_prefixes:
            missing = [key for key in missing if not any(str(key).startswith(prefix) for prefix in normalized_prefixes)]

    if missing:
        _log.warning('%s Missing: %d keys', log_name, len(missing))
        _log.debug("%s missing_count=%d sample=%s", log_name, len(missing), missing[:10])
        _append_key_record({
            "component": log_name,
            "stage": "safe_load_missing",
            "count": len(missing),
            "keys": missing,
        })
    if unexpected:
        _log.warning('%s Unexpected: %d keys', log_name, len(unexpected))
        _log.debug("%s unexpected_count=%d sample=%s", log_name, len(unexpected), unexpected[:10])
        _append_key_record({
            "component": log_name,
            "stage": "safe_load_unexpected",
            "count": len(unexpected),
            "keys": unexpected,
        })
    _trace.event("load_state_dict_done", name=log_name, missing=len(missing), unexpected=len(unexpected))
    return missing, unexpected
