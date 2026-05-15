"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Request-scoped Fooocus inpaint patch session for SDXL masked img2img.
Resolves pinned Fooocus assets (`fooocus_inpaint_head.pth` + `inpaint_v26.fooocus.patch`) from the dedicated SDXL root,
builds the inpaint-head feature from the prepared masked latent bundle, patches a cloned SDXL denoiser for one sampling pass,
honors the global LoRA apply mode for its patch registration, and restores the pre-session denoiser patch registry afterwards.

Symbols (top-level; keep in sync; no ghosts):
- `FooocusInpaintAssets` (dataclass): Exact pinned Fooocus asset paths for the active repo/workspace.
- `InpaintHead` (class): Minimal local convolution head matching the pinned Fooocus inpaint-head checkpoint layout.
- `resolve_fooocus_inpaint_assets` (function): Resolve the pinned Fooocus assets from `sdxl_fooocus_inpaint`.
- `ensure_fooocus_checkpoint_supported` (function): Reject unsupported SDXL checkpoint variants for the Fooocus inpaint lane.
- `_detect_unsupported_variant_markers` (function): Classify unsupported Fooocus checkpoint variant markers from tokenized checkpoint labels.
- `apply_fooocus_inpaint_for_sampling` (function): Context manager that patches one SDXL denoiser clone for the current masked sampling pass.
- `_resolve_unique_asset_path` (function): Resolve one unique pinned Fooocus asset path across all configured `sdxl_fooocus_inpaint` roots.
- `_as_mapping` (function): Enforce mapping-shaped patch payload fragments before exact key ownership checks.
- `_load_inpaint_head` (function): Build and strict-load the pinned Fooocus inpaint head on the request-scoped runtime device.
- `_load_patch_state` (function): Read the pinned Fooocus patch payload with strict mapping ownership.
- `_build_inpaint_feature` (function): Build the masked latent/image conditioning feature consumed by the Fooocus input-block patch.
- `_build_input_block_patch` (function): Create the runtime input-block patch closure that injects Fooocus inpaint features per batch.
- `_build_patch_dict` (function): Translate the pinned Fooocus patch payload into the exact target-map patch dictionary expected by the local patcher seam.
- `_ensure_fooocus_calculator_registered` (function): Register the Fooocus weight calculator exactly once with the local patcher seam.
- `_calculate_weight_fooocus` (function): Apply one Fooocus patch weight payload to a target tensor without renaming stored patch keys.
- `_resolve_runtime_device_and_dtype` (function): Resolve the request-scoped device/dtype from the cloned denoiser owner before loading Fooocus assets.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import torch
import torch.nn.functional as F

from apps.backend.infra.config.lora_apply_mode import LoraApplyMode, read_lora_apply_mode
from apps.backend.infra.config.paths import get_paths_for
from apps.backend.patchers.lora_registry import extra_weight_calculators
from apps.backend.runtime.adapters.lora import model_lora_keys_unet
from apps.backend.runtime.checkpoint.io import load_torch_file
from apps.backend.runtime.logging import get_backend_logger
from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime.pipeline_stages.masked_img2img import MaskedImg2ImgBundle

logger = get_backend_logger("backend.runtime.families.sd.fooocus_inpaint")

_FOOOCUS_INPAINT_ROOT = "sdxl_fooocus_inpaint"
_FOOOCUS_HEAD_FILENAME = "fooocus_inpaint_head.pth"
_FOOOCUS_PATCH_FILENAME = "inpaint_v26.fooocus.patch"
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class FooocusInpaintAssets:
    head_path: str
    patch_path: str


class InpaintHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3), device="cpu", dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(input=padded, weight=self.head)


def resolve_fooocus_inpaint_assets() -> FooocusInpaintAssets:
    roots = get_paths_for(_FOOOCUS_INPAINT_ROOT)
    if not roots:
        raise RuntimeError(
            "Fooocus Inpaint requires paths.json key 'sdxl_fooocus_inpaint' with pinned assets "
            f"'{_FOOOCUS_HEAD_FILENAME}' and '{_FOOOCUS_PATCH_FILENAME}'."
        )
    return FooocusInpaintAssets(
        head_path=_resolve_unique_asset_path(roots=roots, filename=_FOOOCUS_HEAD_FILENAME),
        patch_path=_resolve_unique_asset_path(roots=roots, filename=_FOOOCUS_PATCH_FILENAME),
    )


def ensure_fooocus_checkpoint_supported(checkpoint_record: object) -> None:
    family_hint = str(getattr(checkpoint_record, "family_hint", "") or "").strip().lower()
    if family_hint and family_hint != "sdxl":
        raise RuntimeError(
            f"Fooocus Inpaint requires an SDXL checkpoint selection; inventory family_hint={family_hint!r} is unsupported."
        )

    label_fragments: list[str] = []
    for attr_name in ("title", "name", "model_name", "filename", "path"):
        raw_value = getattr(checkpoint_record, attr_name, None)
        if isinstance(raw_value, str) and raw_value.strip():
            value = raw_value.strip()
            if attr_name in {"filename", "path"}:
                value = Path(value).name
            label_fragments.append(value)

    metadata_fragments: list[str] = []
    metadata = getattr(checkpoint_record, "metadata", None)
    if isinstance(metadata, Mapping):
        for key, raw_value in metadata.items():
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            if str(key).strip().lower() not in {"variant", "model_variant", "repo_variant", "repo_hint"}:
                continue
            metadata_fragments.append(raw_value.strip())

    evidence_fragments = list(label_fragments)
    if metadata_fragments:
        evidence_fragments.append(" ".join(metadata_fragments))

    matched_markers = _detect_unsupported_variant_markers(evidence_fragments)
    if not matched_markers:
        return

    checkpoint_label = next(
        (
            candidate
            for candidate in (
                getattr(checkpoint_record, "title", None),
                getattr(checkpoint_record, "model_name", None),
                getattr(checkpoint_record, "name", None),
                getattr(checkpoint_record, "filename", None),
            )
            if isinstance(candidate, str) and candidate.strip()
        ),
        "<unknown checkpoint>",
    )
    raise RuntimeError(
        "Fooocus Inpaint does not support distilled, Turbo, Lightning, or Hyper SDXL checkpoints. "
        f"Selected checkpoint {checkpoint_label!r} matched unsupported variant marker(s): {', '.join(matched_markers)}."
    )


def _detect_unsupported_variant_markers(candidates: list[str]) -> list[str]:
    matched: list[str] = []
    for candidate in candidates:
        tokens = [token for token in _NON_ALNUM_RE.split(candidate.lower()) if token]
        token_set = set(tokens)
        if not tokens:
            continue
        if ("distilled" in token_set or "distill" in token_set) and "distilled" not in matched:
            matched.append("distilled")
        if "lightning" in token_set and "lightning" not in matched:
            matched.append("lightning")
        if "turbo" in token_set and "turbo" not in matched:
            matched.append("turbo")
        if "hyper" in token_set:
            has_hyper_context = (
                "sdxl" in token_set
                or any(token.endswith("step") or token.endswith("steps") for token in token_set)
                or any(token[:-4].isdigit() for token in token_set if token.endswith("step"))
                or any(token[:-5].isdigit() for token in token_set if token.endswith("steps"))
            )
            if has_hyper_context and "hyper" not in matched:
                matched.append("hyper")
    return matched


@contextlib.contextmanager
def apply_fooocus_inpaint_for_sampling(*, processing, masked_bundle: MaskedImg2ImgBundle) -> Iterator[FooocusInpaintAssets]:
    engine = getattr(processing, "sd_model", None)
    if engine is None:
        raise RuntimeError("Fooocus Inpaint requires processing.sd_model before sampling begins.")
    engine_id = str(getattr(engine, "engine_id", "") or "").strip().lower()
    if engine_id != "sdxl":
        raise RuntimeError(f"Fooocus Inpaint requires exact engine id 'sdxl'; got '{engine_id or '<empty>'}'.")
    assets = resolve_fooocus_inpaint_assets()
    previous_codex_objects = engine.codex_objects
    previous_denoiser = previous_codex_objects.denoiser
    patched_codex_objects = previous_codex_objects.shallow_copy()
    patched_denoiser = previous_denoiser.clone()
    try:
        runtime_device, _runtime_dtype = _resolve_runtime_device_and_dtype(patched_denoiser=patched_denoiser)
        inpaint_head = _load_inpaint_head(assets.head_path, device=runtime_device)
        inpaint_feature = _build_inpaint_feature(
            masked_bundle=masked_bundle,
            inpaint_head=inpaint_head,
            device=runtime_device,
        )
        patched_denoiser.set_model_input_block_patch(_build_input_block_patch(inpaint_feature))
        patch_state = _load_patch_state(assets.patch_path)
        target_model = getattr(patched_denoiser, "model", None)
        if target_model is None:
            raise RuntimeError("Fooocus Inpaint requires a cloned denoiser exposing '.model'.")
        target_map = model_lora_keys_unet(target_model, {})
        target_map.update({key: key for key in target_model.state_dict().keys()})
        patch_dict = _build_patch_dict(loaded_patch=patch_state, target_map=target_map)
        _ensure_fooocus_calculator_registered()
        apply_mode = read_lora_apply_mode()
        online_mode = apply_mode is LoraApplyMode.ONLINE
        patched = patched_denoiser.add_patches(
            filename=assets.patch_path,
            patches=patch_dict,
            online_mode=online_mode,
        )
        patched_denoiser.refresh_loras()
        not_patched = sorted(str(key) for key in patch_dict.keys() if key not in patched)
        if not_patched:
            raise RuntimeError(
                "Fooocus Inpaint patch failed to bind all resolved weights. "
                f"Unpatched keys: {', '.join(not_patched[:16])}"
            )
        patched_codex_objects.denoiser = patched_denoiser
        engine.codex_objects = patched_codex_objects
        logger.info(
            "Applying Fooocus Inpaint for engine=%s head=%s patch=%s apply_mode=%s",
            engine_id,
            assets.head_path,
            assets.patch_path,
            apply_mode.value,
        )
        yield assets
    finally:
        engine.codex_objects = previous_codex_objects
        try:
            previous_denoiser.refresh_loras()
        except Exception as exc:
            raise RuntimeError(
                "Fooocus Inpaint cleanup failed while restoring the pre-session denoiser patch registry."
            ) from exc


def _resolve_unique_asset_path(*, roots: list[str], filename: str) -> str:
    matches: list[str] = []
    for root in roots:
        root_path = Path(str(root)).expanduser()
        if not root_path.exists():
            continue
        for candidate in root_path.rglob(filename):
            if candidate.is_file():
                matches.append(str(candidate.resolve(strict=False)))
    unique_matches = sorted(dict.fromkeys(matches))
    if not unique_matches:
        raise RuntimeError(
            f"Fooocus Inpaint asset '{filename}' was not found under any configured '{_FOOOCUS_INPAINT_ROOT}' root: "
            + ", ".join(roots)
        )
    if len(unique_matches) > 1:
        raise RuntimeError(
            f"Fooocus Inpaint asset '{filename}' is ambiguous under '{_FOOOCUS_INPAINT_ROOT}': "
            + ", ".join(unique_matches)
        )
    return unique_matches[0]


def _as_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    raise RuntimeError(f"{label} must resolve to a mapping state dict; got {type(value).__name__}.")


def _load_inpaint_head(path: str, *, device: torch.device) -> InpaintHead:
    module = InpaintHead().to(device=device, dtype=torch.float32)
    state_dict = _as_mapping(load_torch_file(path, safe_load=True, device="cpu"), label="fooocus_inpaint_head")
    missing, unexpected = safe_load_state_dict(module, state_dict, log_name="FooocusInpaintHead")
    if missing or unexpected:
        raise RuntimeError(
            "Fooocus Inpaint head failed strict load. "
            f"missing={missing} unexpected={unexpected}"
        )
    module.eval()
    return module


def _load_patch_state(path: str) -> Mapping[str, object]:
    return _as_mapping(load_torch_file(path, safe_load=True, device="cpu"), label="fooocus_inpaint_patch")


def _build_inpaint_feature(*, masked_bundle: MaskedImg2ImgBundle, inpaint_head: InpaintHead, device: torch.device) -> torch.Tensor:
    latent_image = masked_bundle.init_latent.to(device=device, dtype=torch.float32)
    latent_mask = masked_bundle.latent_masked.to(device=device, dtype=torch.float32)
    if latent_mask.shape[0] == 1 and latent_image.shape[0] > 1:
        latent_mask = latent_mask.expand(latent_image.shape[0], -1, -1, -1)
    feed = torch.cat([latent_mask.round(), latent_image], dim=1)
    return inpaint_head(feed)


def _build_input_block_patch(inpaint_feature: torch.Tensor):
    def input_block_patch(h, transformer_options):
        block = transformer_options.get("block") if isinstance(transformer_options, dict) else None
        if isinstance(block, (tuple, list)) and len(block) > 1 and int(block[1]) == 0:
            return h + inpaint_feature.to(device=h.device, dtype=h.dtype)
        return h

    return input_block_patch


def _build_patch_dict(*, loaded_patch: Mapping[str, object], target_map: Mapping[str, object]) -> dict[str, tuple[str, object]]:
    patch_dict: dict[str, tuple[str, object]] = {}
    for target in target_map.values():
        if not isinstance(target, str):
            continue
        payload = loaded_patch.get(target)
        if payload is None:
            continue
        patch_dict[target] = ("fooocus", payload)
    if not patch_dict:
        raise RuntimeError("Fooocus Inpaint patch resolved zero matching UNet weights for the active SDXL denoiser.")
    return patch_dict


def _ensure_fooocus_calculator_registered() -> None:
    if extra_weight_calculators.get("fooocus") is _calculate_weight_fooocus:
        return
    extra_weight_calculators["fooocus"] = _calculate_weight_fooocus


def _calculate_weight_fooocus(weight: torch.Tensor, alpha: float, payload: object) -> torch.Tensor:
    if not isinstance(payload, (tuple, list)) or len(payload) < 3:
        raise RuntimeError(
            "Fooocus Inpaint patch payload must be a tuple/list of at least three tensors: (quantized, min, max)."
        )
    quantized, weight_min, weight_max = payload[:3]
    if not torch.is_tensor(quantized) or not torch.is_tensor(weight_min) or not torch.is_tensor(weight_max):
        raise RuntimeError("Fooocus Inpaint patch payload tensors are invalid.")
    quantized_tensor = quantized.to(device=weight.device, dtype=torch.float32)
    if quantized_tensor.shape != weight.shape:
        raise RuntimeError(
            f"Fooocus Inpaint patch shape mismatch: {tuple(quantized_tensor.shape)} != {tuple(weight.shape)}."
        )
    weight_min_tensor = weight_min.to(device=weight.device, dtype=torch.float32)
    weight_max_tensor = weight_max.to(device=weight.device, dtype=torch.float32)
    restored = (quantized_tensor / 255.0) * (weight_max_tensor - weight_min_tensor) + weight_min_tensor
    return weight + float(alpha) * restored.to(device=weight.device, dtype=weight.dtype)


def _resolve_runtime_device_and_dtype(*, patched_denoiser) -> tuple[torch.device, torch.dtype]:
    diffusion_model = getattr(getattr(patched_denoiser, "model", None), "diffusion_model", None)
    if diffusion_model is None:
        raise RuntimeError("Fooocus Inpaint requires patched_denoiser.model.diffusion_model to resolve runtime device/dtype.")
    for parameter in diffusion_model.parameters():
        if isinstance(parameter, torch.Tensor):
            return parameter.device, parameter.dtype
    raise RuntimeError("Fooocus Inpaint could not resolve an active diffusion-model parameter for runtime device/dtype.")
