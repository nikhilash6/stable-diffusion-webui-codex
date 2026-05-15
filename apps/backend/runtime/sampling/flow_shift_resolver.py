"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Resolve flow-match `flow_shift` for sampling from canonical scheduler_config.json sources (diffusers).
Used by the sampling context builder for predictors with `prediction_type='const'` (flow-match). Resolution prefers:
- `predictor.flow_shift_spec` overrides (explicit fixed/dynamic spec),
- diffusers repo directories that include `scheduler/scheduler_config.json`,
- vendored Hugging Face mirrors for special cases (e.g. Z-Image variants),
- fixed `FamilyRuntimeSpec.flow_shift` when a family defines a true invariant (e.g. Anima shift=3.0).

Symbols (top-level; keep in sync; no ghosts):
- `FlowShiftResolution` (dataclass): Flow-shift resolution result (effective shift + spec + source metadata).
- `resolve_flow_shift_for_sampling` (function): Resolve the effective flow shift (fixed/dynamic) for the current run.
- `__all__` (constant): Export list for the resolver helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apps.backend.runtime.model_registry.flow_shift import FlowShiftSpec
from apps.backend.runtime.logging import emit_backend_message


@dataclass(frozen=True)
class FlowShiftResolution:
    effective_shift: float
    spec: FlowShiftSpec
    repo_dir: str | None
    source: str


def resolve_flow_shift_for_sampling(
    sd_model,
    predictor,
    *,
    height: int | None,
    width: int | None,
) -> FlowShiftResolution:
    # Flow-match models must resolve flow_shift from the canonical scheduler_config.json.
    from apps.backend.infra.config.repo_root import get_repo_root
    from apps.backend.runtime.model_registry.family_runtime import get_family_spec
    from apps.backend.runtime.model_registry.flow_shift import (
        FlowShiftMode,
        FlowShiftSource,
        flow_shift_spec_from_repo_dir,
    )
    from apps.backend.runtime.model_registry.specs import ModelFamily

    spec_obj: FlowShiftSpec | None = None
    repo_dir: str | None = None
    source = "unknown"
    raw_spec = getattr(predictor, "flow_shift_spec", None)
    if isinstance(raw_spec, FlowShiftSpec):
        spec_obj = raw_spec
        source = "predictor.flow_shift_spec"
        repo_dir = None
    else:
        bundle = getattr(sd_model, "_current_bundle", None)
        repo_ref = getattr(bundle, "model_ref", None)
        if isinstance(repo_ref, str):
            repo_path = Path(repo_ref)
            if repo_path.is_dir():
                spec_obj = flow_shift_spec_from_repo_dir(repo_path)
                source = "diffusers_repo"
                repo_dir = str(repo_path)

    if spec_obj is None:
        # Z-Image core-only checkpoints typically do not ship scheduler configs with the weights file.
        # Resolve flow_shift from the vendored HF mirror based on the requested/loaded variant.
        bundle = getattr(sd_model, "_current_bundle", None)
        family = getattr(bundle, "family", None)
        if family is ModelFamily.ZIMAGE:
            variant_raw = getattr(sd_model, "zimage_variant", None) or getattr(sd_model, "_zimage_variant", None)
            variant = str(variant_raw or "").strip().lower()
            if variant in {"turbo", "base"}:
                repo_root = get_repo_root()
                hf_root = repo_root / "apps" / "backend" / "huggingface"
                rid = "Tongyi-MAI/Z-Image-Turbo" if variant == "turbo" else "Tongyi-MAI/Z-Image"
                vendor = hf_root / rid.replace("/", "/")
                if not vendor.is_dir():
                    raise RuntimeError(
                        f"Z-Image variant={variant!r} requires vendored HF assets at {vendor}. "
                        "Ensure the directory exists under apps/backend/huggingface/."
                    )
                spec_obj = flow_shift_spec_from_repo_dir(vendor)
                source = f"vendored_zimage_variant:{variant}"
                repo_dir = str(vendor)
            else:
                raise RuntimeError(
                    "Z-Image requires an explicit variant to resolve flow_shift. "
                    "Provide engine option zimage_variant='turbo'|'base' (UI: Turbo toggle), "
                    "or load a diffusers repo directory that includes scheduler/scheduler_config.json."
                )

    if spec_obj is None:
        # If the model isn't a diffusers directory, try resolving the canonical
        # scheduler config from the vendored Hugging Face mirror using the
        # detected repo_hint.
        bundle = getattr(sd_model, "_current_bundle", None)
        sig = getattr(bundle, "signature", None)
        repo_hint = getattr(sig, "repo_hint", None) if sig is not None else None
        if isinstance(repo_hint, str) and repo_hint.strip():
            family = getattr(bundle, "family", None)
            if not (isinstance(family, ModelFamily) and get_family_spec(family).flow_shift is not None):
                repo_root = get_repo_root()
                vendor_root = repo_root / "apps" / "backend" / "huggingface"
                vendor = vendor_root / repo_hint
                if not vendor.is_dir():
                    # Some detectors use a full HF repo id as repo_hint, but the vendored
                    # mirror may be stored under a shorter directory name (e.g., "Chroma").
                    vendor = vendor_root / Path(repo_hint).name
                if vendor.is_dir():
                    spec_obj = flow_shift_spec_from_repo_dir(vendor)
                    source = "vendored_repo_hint"
                    repo_dir = str(vendor)

    if spec_obj is None:
        bundle = getattr(sd_model, "_current_bundle", None)
        family = getattr(bundle, "family", None)
        if isinstance(family, ModelFamily):
            fam = get_family_spec(family)
            raw_shift = getattr(fam, "flow_shift", None)
            if raw_shift is not None:
                value = float(raw_shift)
                if value <= 0.0:
                    raise RuntimeError(f"Invalid family runtime flow_shift={value} for family={family.value!r}")
                spec_obj = FlowShiftSpec(
                    mode=FlowShiftMode.FIXED,
                    source=FlowShiftSource.OVERRIDE,
                    value=value,
                    config_path=None,
                )
                source = f"family_runtime:{family.value}"
                repo_dir = None

    if spec_obj is None:
        raise RuntimeError(
            "Flow-match sampling requires a scheduler_config.json to resolve flow_shift, but none was found. "
            "Load a diffusers repo with scheduler/ configs, ensure the engine provides vendored HF assets, "
            "or provide a fixed flow_shift override via the family runtime spec or predictor.flow_shift_spec."
        )

    effective_shift: float
    if spec_obj.mode is FlowShiftMode.DYNAMIC:
        if height is None or width is None:
            raise RuntimeError("Dynamic flow_shift requires explicit height/width for seq_len calculation.")
        bundle = getattr(sd_model, "_current_bundle", None)
        family = getattr(bundle, "family", None)
        if not isinstance(family, ModelFamily):
            raise RuntimeError("Dynamic flow_shift requires a known ModelFamily on the loaded bundle.")
        fam = get_family_spec(family)
        scale = int(fam.latent_scale_factor)
        patch = int(fam.patch_size)
        if scale <= 0 or patch <= 0:
            raise RuntimeError(f"Invalid latent_scale_factor/patch_size for family={family}: {scale}/{patch}")
        step = scale * patch
        if (int(height) % step) != 0 or (int(width) % step) != 0:
            raise RuntimeError(
                f"Invalid size for dynamic flow shift: {int(width)}x{int(height)} (expected multiples of {step})."
            )
        seq_len = (int(height) // scale // patch) * (int(width) // scale // patch)
        effective_shift = spec_obj.resolve_effective_shift(seq_len=seq_len)
    else:
        effective_shift = spec_obj.resolve_effective_shift()

    cfg_path = getattr(spec_obj, "config_path", None)
    if isinstance(cfg_path, str) and cfg_path.strip():
        cfg_path_display = cfg_path
        repo_dir_display = repo_dir or "<unknown>"
        try:
            repo_root = get_repo_root()
            cfg_path_display = str(Path(cfg_path).resolve().relative_to(repo_root))
            if repo_dir is not None:
                repo_dir_display = str(Path(repo_dir).resolve().relative_to(repo_root))
        except Exception:  # noqa: BLE001 - best-effort diagnostics
            pass
        emit_backend_message(
            "sigmas: using flow_shift from scheduler_config.json",
            logger=__name__,
            config=cfg_path_display,
            repo_dir=repo_dir_display,
            source=source,
            mode=getattr(spec_obj.mode, "value", spec_obj.mode),
        )

    return FlowShiftResolution(effective_shift=effective_shift, spec=spec_obj, repo_dir=repo_dir, source=source)


__all__ = [
    "resolve_flow_shift_for_sampling",
    "FlowShiftResolution",
]
