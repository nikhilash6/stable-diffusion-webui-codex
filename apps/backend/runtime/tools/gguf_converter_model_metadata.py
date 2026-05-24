"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vendored config metadata for the GGUF converter UI.
Scans the local Hugging Face mirror under `apps/backend/huggingface/**` and exposes “model metadata” entries (org/repo)
with supported conversion components (Flux/Qwen Image/ZImage/Z-Image L2P/WAN22/LTX2 denoisers plus supported text encoders).
The L2P Qwen3-4B text encoder is listed from the `Qwen/Qwen3-4B` mirror, not as a `zhen-nan/L2P` subcomponent.
The preset list is metadata-only; conversion weights are supplied separately by the operator.

Symbols (top-level; keep in sync; no ghosts):
- `GGUFConverterModelComponent` (dataclass): Convertible component entry (config dir + one truthful profile id).
- `GGUFConverterModelMetadata` (dataclass): Model entry (org/repo + components).
- `_iter_candidate_config_dirs` (function): Iterates candidate config directories within a repo (root + subdirs).
- `_classify_config` (function): Classifies a config.json into a converter component kind + profile id.
- `list_vendored_gguf_converter_model_metadata` (function): Lists supported model metadata from the vendored HF mirror.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from apps.backend.inventory.scanners.vendored_hf import iter_vendored_hf_repos
from apps.backend.runtime.tools import gguf_converter_tensor_planner as _tensor_planner


@dataclass(frozen=True, slots=True)
class GGUFConverterModelComponent:
    id: str
    label: str
    config_dir: str
    kind: str
    profile_id: str | None = None


@dataclass(frozen=True, slots=True)
class GGUFConverterModelMetadata:
    id: str
    label: str
    org: str
    repo: str
    components: tuple[GGUFConverterModelComponent, ...]


def _iter_candidate_config_dirs(repo_dir: str) -> Iterable[tuple[str, str]]:
    yield "", repo_dir
    try:
        entries = sorted(os.listdir(repo_dir), key=lambda s: s.lower())
    except Exception:
        return
    for name in entries:
        full = os.path.join(repo_dir, name)
        if os.path.isdir(full):
            yield name, full


def _classify_config(cfg: dict[str, Any], *, org: str, repo: str, subdir: str) -> tuple[str, str | None]:
    class_name = str(cfg.get("_class_name") or "").strip()
    model_type = str(cfg.get("model_type") or "").strip()
    text_cfg = cfg.get("text_config")
    text_model_type = str(text_cfg.get("model_type") or "").strip() if isinstance(text_cfg, dict) else ""
    is_l2p_repo = org == "zhen-nan" and repo == "L2P"
    is_l2p_tenc_repo = org == "Qwen" and repo == "Qwen3-4B"
    if class_name == "FluxTransformer2DModel":
        return ("flux_transformer", "flux_transformer")
    if _tensor_planner.is_qwen_image_transformer_config(cfg):
        return ("qwen_image_transformer", "qwen_image_transformer")
    if _tensor_planner.is_qwen_image_text_encoder_config(cfg):
        return ("qwen_image_tenc", "qwen_image_tenc")
    if is_l2p_repo and subdir == "denoiser" and _tensor_planner.is_zimage_l2p_denoiser_config(cfg):
        return ("zimage_l2p_denoiser", "zimage_l2p_denoiser")
    if is_l2p_tenc_repo and subdir == "text_encoder" and _tensor_planner.is_zimage_l2p_text_encoder_config(cfg):
        return ("zimage_l2p_tenc", "zimage_l2p_tenc")
    if class_name == "ZImageTransformer2DModel":
        return ("zimage_transformer", "zimage_transformer")
    if class_name in {"WanTransformer3DModel", "WanModel"}:
        return ("wan22_transformer", "wan22_transformer")
    if class_name == "LTX2VideoTransformer3DModel":
        return ("ltx2_transformer", "ltx2_transformer")
    if model_type == "gemma3" or text_model_type == "gemma3_text":
        return ("gemma3_tenc", "gemma3_tenc")

    return ("unknown", None)


def list_vendored_gguf_converter_model_metadata(*, codex_root: Path) -> list[GGUFConverterModelMetadata]:
    vendored_root = codex_root / "apps" / "backend" / "huggingface"
    models: list[GGUFConverterModelMetadata] = []

    for org, repo, repo_dir in iter_vendored_hf_repos(str(vendored_root)):
        # WAN22 repositories may expose a two-stage denoiser split (high-noise vs low-noise)
        # under either Diffusers (`transformer`/`transformer_2`) or upstream (`high_noise_model`/`low_noise_model`).
        wan_two_stage = False
        for low_stage_dir in ("transformer_2", "low_noise_model"):
            candidate = os.path.join(repo_dir, low_stage_dir)
            if os.path.isfile(os.path.join(candidate, "config.json")):
                wan_two_stage = True
                break

        components: list[GGUFConverterModelComponent] = []
        for subdir, config_dir in _iter_candidate_config_dirs(repo_dir):
            cfg_path = os.path.join(config_dir, "config.json")
            if not os.path.isfile(cfg_path):
                continue

            try:
                cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            except Exception:
                continue

            kind, profile_id = _classify_config(cfg, org=org, repo=repo, subdir=subdir)
            if kind == "unknown":
                continue

            component_id = subdir or "root"
            component_label = subdir or "root"
            if kind in {
                "flux_transformer",
                "qwen_image_transformer",
                "zimage_transformer",
                "zimage_l2p_denoiser",
                "ltx2_transformer",
            }:
                component_label = "denoiser"
            if kind in {"gemma3_tenc", "qwen_image_tenc", "zimage_l2p_tenc"}:
                component_label = "text_encoder"
            if kind == "wan22_transformer" and wan_two_stage:
                if component_id in {"transformer", "high_noise_model"}:
                    component_label = "high_noise"
                elif component_id in {"transformer_2", "low_noise_model"}:
                    component_label = "low_noise"
            components.append(
                GGUFConverterModelComponent(
                    id=component_id,
                    label=component_label,
                    config_dir=str(Path(config_dir).resolve()),
                    kind=kind,
                    profile_id=profile_id,
                )
            )

        if not components:
            continue

        kind_priority = {
            "flux_transformer": 0,
            "qwen_image_transformer": 0,
            "zimage_transformer": 0,
            "zimage_l2p_denoiser": 0,
            "wan22_transformer": 0,
            "ltx2_transformer": 0,
            "gemma3_tenc": 1,
            "qwen_image_tenc": 1,
            "zimage_l2p_tenc": 1,
        }
        components.sort(key=lambda c: (kind_priority.get(c.kind, 9), c.label.lower()))
        model_id = f"{org}/{repo}"
        models.append(
            GGUFConverterModelMetadata(
                id=model_id,
                label=model_id,
                org=org,
                repo=repo,
                components=tuple(components),
            )
        )

    models.sort(key=lambda m: m.label.lower())
    return models


__all__ = [
    "GGUFConverterModelComponent",
    "GGUFConverterModelMetadata",
    "list_vendored_gguf_converter_model_metadata",
]
