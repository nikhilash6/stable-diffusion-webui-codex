"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared VAE layout/lane policy helpers for loader and engine VAE paths.
Provides fail-loud VAE key-name validation, robust layout detection (`ldm` vs `diffusers`), and strict lane resolution from global policy
(`CODEX_VAE_LAYOUT_LANE=auto|ldm_native|diffusers_native`) with family-aware fail-loud checks.
WAN22 variant families (`WAN22_5B`/`WAN22_14B`/`WAN22_ANIMATE`) are always constrained to native LDM VAE lane.

Symbols (top-level; keep in sync; no ghosts):
- `LDM_NATIVE_VAE_FAMILIES` (constant): Families that support native LDM VAE lane.
- `_WAN22_FAMILIES` (constant): WAN22 variant families that are hard-pinned to native LDM VAE lane.
- `validate_vae_key_names` (function): Fails loud when a VAE checkpoint would require wrapper/prefix rewriting before load.
- `detect_vae_layout` (function): Detects VAE keyspace layout (`ldm` or `diffusers`) with ambiguity checks.
- `resolve_vae_layout_lane` (function): Resolves effective lane (`ldm_native`/`diffusers_native`) from global policy + family + layout.
- `uses_ldm_native_lane` (function): True when effective lane is native LDM.
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from apps.backend.infra.config.vae_layout_lane import VaeLayoutLane, read_vae_layout_lane
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.state_dict.key_mapping import KeyMappingError

LDM_NATIVE_VAE_FAMILIES = {
    ModelFamily.FLUX,
    ModelFamily.FLUX_KONTEXT,
    ModelFamily.SDXL,
    ModelFamily.SDXL_REFINER,
    ModelFamily.ZIMAGE,
    ModelFamily.WAN22_5B,
    ModelFamily.WAN22_14B,
    ModelFamily.WAN22_ANIMATE,
}

_WAN22_FAMILIES = {
    ModelFamily.WAN22_5B,
    ModelFamily.WAN22_14B,
    ModelFamily.WAN22_ANIMATE,
}


def validate_vae_key_names(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail loud if a VAE checkpoint would require wrapper-prefix rewriting."""
    prefixes = (
        "first_stage_model.",
        "vae.",
        "model.",
        "module.",
    )

    for raw_key in state_dict.keys():
        source_key = str(raw_key)
        candidate_key = source_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if candidate_key.startswith(prefix):
                    candidate_key = candidate_key[len(prefix) :]
                    changed = True
                    break
        if candidate_key != source_key:
            raise KeyMappingError(
                "VAE key-name mutation is forbidden in this repository. "
                "Do not strip wrapper prefixes from stored VAE keys; extend the keyspace understanding explicitly instead. "
                f"source_key={source_key!r} candidate_key={candidate_key!r}"
            )
    return state_dict


def detect_vae_layout(state_dict: Mapping[str, Any]) -> str:
    """Detect VAE keyspace layout (`ldm` or `diffusers`) with fail-loud ambiguity checks."""
    if not hasattr(state_dict, "keys"):
        return "diffusers"

    has_ldm = False
    has_diffusers = False

    for raw_key in state_dict.keys():
        if not isinstance(raw_key, str):
            continue
        key = raw_key
        if (
            key.startswith("encoder.down.")
            or key.startswith("decoder.up.")
            or ".mid.attn_1." in key
            or ".mid.block_" in key
        ):
            has_ldm = True
        if (
            key.startswith("encoder.down_blocks.")
            or key.startswith("decoder.up_blocks.")
            or ".mid_block.attentions." in key
            or ".mid_block.resnets." in key
        ):
            has_diffusers = True
        if has_ldm and has_diffusers:
            raise RuntimeError(
                "VAE layout detection is ambiguous (mixed LDM and diffusers keyspaces). "
                "Provide a single-layout VAE or set CODEX_VAE_LAYOUT_LANE explicitly for diagnostics."
            )

    if has_ldm:
        return "ldm"
    return "diffusers"


def resolve_vae_layout_lane(
    *,
    family: Optional[ModelFamily],
    layout: str,
) -> VaeLayoutLane:
    """Resolve effective VAE lane from global policy + family + detected layout."""
    requested = read_vae_layout_lane()
    family_label = getattr(family, "value", str(family)) if family is not None else "<unknown>"
    supports_ldm_native = family in LDM_NATIVE_VAE_FAMILIES

    if family in _WAN22_FAMILIES:
        if layout != "ldm":
            raise RuntimeError(
                "WAN22 VAE lane requires LDM VAE keyspace layout; detected layout=%s." % layout
            )
        if requested is VaeLayoutLane.DIFFUSERS_NATIVE:
            raise RuntimeError(
                "CODEX_VAE_LAYOUT_LANE=diffusers_native is not supported for family %s; "
                "WAN22 variants require ldm_native."
                % family_label
            )
        return VaeLayoutLane.LDM_NATIVE

    if requested is VaeLayoutLane.AUTO:
        if layout == "ldm":
            if not supports_ldm_native:
                raise RuntimeError(
                    "Detected LDM VAE layout for family %s, but native LDM lane is not supported for this family. "
                    "Set CODEX_VAE_LAYOUT_LANE=diffusers_native only if an explicit keymap for this family exists."
                    % family_label
                )
            return VaeLayoutLane.LDM_NATIVE
        return VaeLayoutLane.DIFFUSERS_NATIVE

    if requested is VaeLayoutLane.LDM_NATIVE:
        if not supports_ldm_native:
            raise RuntimeError(
                "CODEX_VAE_LAYOUT_LANE=ldm_native is not supported for family %s." % family_label
            )
        if layout != "ldm":
            raise RuntimeError(
                "CODEX_VAE_LAYOUT_LANE=ldm_native requires LDM VAE keyspace; detected layout=%s for family %s."
                % (layout, family_label)
            )
        return VaeLayoutLane.LDM_NATIVE

    # diffusers_native
    if layout == "ldm" and not supports_ldm_native:
        raise RuntimeError(
            "CODEX_VAE_LAYOUT_LANE=diffusers_native cannot process LDM VAE layout for family %s "
            "because no family keymap is registered."
            % family_label
        )
    return VaeLayoutLane.DIFFUSERS_NATIVE


def uses_ldm_native_lane(lane: VaeLayoutLane) -> bool:
    return lane is VaeLayoutLane.LDM_NATIVE


__all__ = [
    "LDM_NATIVE_VAE_FAMILIES",
    "detect_vae_layout",
    "resolve_vae_layout_lane",
    "validate_vae_key_names",
    "uses_ldm_native_lane",
]
