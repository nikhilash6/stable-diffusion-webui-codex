"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GGUF metadata injection helpers for the converter.
Adds provenance/source metadata and minimal architecture keys required by loader tooling
(including Qwen Image transformer metadata and `codex.zimage.variant` when detectable from scheduler configs).

Symbols (top-level; keep in sync; no ghosts):
- `_is_hf_repo_id` (function): Returns True when a string looks like a Hugging Face repo id (`org/repo`).
- `add_basic_metadata` (function): Adds standard provenance/license metadata keys into the output GGUF.
"""

from __future__ import annotations

import datetime as _dt
import re
import json
from pathlib import Path

from apps.backend.quantization.gguf import GGUFWriter
from apps.backend.infra.config.provenance import CODEX_GENERATED_BY, CODEX_REPO_URL, best_effort_git_commit
from apps.backend.runtime.tools.gguf_converter_types import QuantizationType


def _is_hf_repo_id(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate:
        return False
    if candidate.startswith((".", "/", "\\")):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", candidate))


def add_basic_metadata(
    writer: GGUFWriter,
    arch: str,
    config: dict,
    quant: QuantizationType,
    *,
    config_path: Path,
    safetensors_path: str,
) -> None:
    # `GGUFWriter` writes `general.architecture` eagerly in `__init__`.
    # Codex uses a custom metadata schema, so remove it from the output.
    try:
        for shard in writer.kv_data:
            if isinstance(shard, dict):
                shard.pop("general.architecture", None)
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[4]
    commit = best_effort_git_commit(repo_root)
    writer.add_string("codex.quantized_by", CODEX_GENERATED_BY)
    writer.add_string("codex.repository", CODEX_REPO_URL)
    if commit:
        writer.add_string("codex.commit", commit)

    model_name = str(config.get("_name_or_path") or config.get("name") or "model")
    writer.add_string("model.name", model_name)
    writer.add_string("model.architecture", str(arch))

    upstream = str(config.get("_name_or_path") or "").strip()
    if _is_hf_repo_id(upstream):
        writer.add_string("model.repository", f"https://huggingface.co/{upstream}")

    # Minimal model metadata; loaders in this repo generally key off tensor names and shapes.
    writer.add_uint32("model.context_length", int(config.get("max_position_embeddings", 4096)))
    writer.add_uint32("model.embedding_length", int(config.get("hidden_size", 4096)))
    writer.add_uint32("model.block_count", int(config.get("num_hidden_layers", 32)))
    writer.add_uint32("model.attention.head_count", int(config.get("num_attention_heads", 32)))
    writer.add_uint32("model.attention.head_count_kv", int(config.get("num_key_value_heads", 8)))
    writer.add_float32("model.rope.freq_base", float(config.get("rope_theta", 10000.0)))
    writer.add_float32("model.attention.layer_norm_rms_epsilon", float(config.get("rms_norm_eps", 1e-6)))

    qwen_variant = str(config.get("codex.qwen_image.variant") or "").strip()
    if qwen_variant:
        axes_raw = config.get("codex.qwen_image.axes_dims_rope")
        if not isinstance(axes_raw, (list, tuple)) or not axes_raw:
            raise RuntimeError("Qwen Image GGUF metadata requires codex.qwen_image.axes_dims_rope")
        axes_dims_rope = [int(value) for value in axes_raw]
        writer.add_string("codex.qwen_image.variant", qwen_variant)
        writer.add_bool("codex.qwen_image.zero_cond_t", bool(config.get("codex.qwen_image.zero_cond_t")))
        writer.add_uint32("codex.qwen_image.joint_attention_dim", int(config.get("codex.qwen_image.joint_attention_dim", 0)))
        writer.add_uint32("codex.qwen_image.in_channels", int(config.get("codex.qwen_image.in_channels", 0)))
        writer.add_uint32("codex.qwen_image.out_channels", int(config.get("codex.qwen_image.out_channels", 0)))
        writer.add_uint32("codex.qwen_image.patch_size", int(config.get("codex.qwen_image.patch_size", 0)))
        writer.add_array("codex.qwen_image.axes_dims_rope", axes_dims_rope)

    writer.add_string("gguf.quantized_at_utc", _dt.datetime.now(tz=_dt.timezone.utc).isoformat())
    writer.add_string("gguf.quantization", str(quant.value))

    # Z-Image Turbo/Base disambiguation: when converting from a diffusers-style directory
    # layout, the scheduler_config.json contains the canonical `shift` (3.0 turbo / 6.0 base).
    #
    # This metadata is trusted by the WebUI only when Codex provenance keys are present,
    # so leaving it unset when we cannot prove the source is fine.
    try:
        is_zimage = str(arch).strip().lower() == "zimage" or str(config.get("model_type") or "").strip().lower() == "zimage"
        if is_zimage:
            cfg_dir = Path(config_path).resolve().parent
            candidates = [
                cfg_dir / "scheduler" / "scheduler_config.json",
                cfg_dir.parent / "scheduler" / "scheduler_config.json",
            ]
            for cand in candidates:
                if not cand.is_file():
                    continue
                data = json.loads(cand.read_text(encoding="utf-8"))
                raw_shift = data.get("shift")
                if raw_shift is None:
                    continue
                try:
                    shift = float(raw_shift)
                except Exception:
                    continue
                if abs(shift - 3.0) < 1e-3:
                    writer.add_string("codex.zimage.variant", "turbo")
                    break
                if abs(shift - 6.0) < 1e-3:
                    writer.add_string("codex.zimage.variant", "base")
                    break
    except Exception:
        pass


__all__ = [
    "add_basic_metadata",
]
