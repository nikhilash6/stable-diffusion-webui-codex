"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GGUF metadata injection helpers for the converter.
Adds provenance, quantization recipe/policy, and minimal architecture keys required by loader tooling
(including Qwen Image/L2P profile/component metadata and `codex.zimage.variant` when detectable from scheduler configs).

Symbols (top-level; keep in sync; no ghosts):
- `_is_hf_repo_id` (function): Returns True when a string looks like a Hugging Face repo id (`org/repo`).
- `add_basic_metadata` (function): Adds standard provenance, architecture, quant recipe, and optional quant-policy metadata keys into the output GGUF.
"""

from __future__ import annotations

import datetime as _dt
import re
import json
from pathlib import Path

from apps.backend.quantization.gguf import GGUFWriter
from apps.backend.infra.config.provenance import CODEX_GENERATED_BY, CODEX_REPO_URL, best_effort_git_commit
from apps.backend.runtime.tools.gguf_converter_quantization import QuantizationRecipeSpec
from apps.backend.runtime.tools.gguf_converter_types import QuantPolicyPreset


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
    recipe: QuantizationRecipeSpec,
    *,
    quant_policy: str | None,
    quant_policy_preset: QuantPolicyPreset | None,
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

    l2p_component = str(config.get("codex.zimage_l2p.component") or "").strip()
    if l2p_component:
        l2p_profile_id = str(config.get("codex.zimage_l2p.profile_id") or "").strip()
        l2p_family = str(config.get("codex.zimage_l2p.family") or "").strip()
        if not l2p_profile_id:
            raise RuntimeError("L2P GGUF metadata requires codex.zimage_l2p.profile_id")
        if l2p_family != "zimage_l2p":
            raise RuntimeError("L2P GGUF metadata requires codex.zimage_l2p.family='zimage_l2p'")
        if config.get("codex.zimage_l2p.pixel_space") is not True:
            raise RuntimeError("L2P GGUF metadata requires codex.zimage_l2p.pixel_space=true")
        writer.add_string("codex.zimage_l2p.profile_id", l2p_profile_id)
        writer.add_string("codex.zimage_l2p.component", l2p_component)
        writer.add_string("codex.zimage_l2p.family", l2p_family)
        writer.add_bool("codex.zimage_l2p.pixel_space", True)
        writer.add_bool("codex.zimage_l2p.requires_vae", bool(config.get("codex.zimage_l2p.requires_vae") is True))
        if l2p_component == "denoiser":
            if l2p_profile_id != "zimage_l2p_denoiser":
                raise RuntimeError("L2P denoiser GGUF metadata requires profile_id='zimage_l2p_denoiser'")
            axes_dims = config.get("codex.zimage_l2p.axes_dims")
            axes_lens = config.get("codex.zimage_l2p.axes_lens")
            if not isinstance(axes_dims, (list, tuple)) or len(axes_dims) != 3:
                raise RuntimeError("L2P denoiser GGUF metadata requires codex.zimage_l2p.axes_dims")
            if not isinstance(axes_lens, (list, tuple)) or len(axes_lens) != 3:
                raise RuntimeError("L2P denoiser GGUF metadata requires codex.zimage_l2p.axes_lens")
            writer.add_uint32("codex.zimage_l2p.patch_size", int(config.get("codex.zimage_l2p.patch_size", 16)))
            writer.add_uint32(
                "codex.zimage_l2p.frame_patch_size",
                int(config.get("codex.zimage_l2p.frame_patch_size", 1)),
            )
            writer.add_uint32("codex.zimage_l2p.in_channels", int(config.get("codex.zimage_l2p.in_channels", 3)))
            writer.add_uint32("codex.zimage_l2p.context_dim", int(config.get("codex.zimage_l2p.context_dim", 2560)))
            writer.add_uint32(
                "codex.zimage_l2p.num_refiner_layers",
                int(config.get("codex.zimage_l2p.num_refiner_layers", 2)),
            )
            writer.add_array("codex.zimage_l2p.axes_dims", [int(value) for value in axes_dims])
            writer.add_array("codex.zimage_l2p.axes_lens", [int(value) for value in axes_lens])
            writer.add_bool("codex.zimage_l2p.local_decoder", bool(config.get("codex.zimage_l2p.local_decoder") is True))
        elif l2p_component == "tenc":
            if l2p_profile_id != "zimage_l2p_tenc":
                raise RuntimeError("L2P TEnc GGUF metadata requires profile_id='zimage_l2p_tenc'")
            writer.add_string("codex.zimage_l2p.tenc_slot", str(config.get("codex.zimage_l2p.tenc_slot") or "qwen3_4b"))
            writer.add_uint32(
                "codex.zimage_l2p.qwen_hidden_size",
                int(config.get("codex.zimage_l2p.qwen_hidden_size", 2560)),
            )
            writer.add_uint32(
                "codex.zimage_l2p.qwen_layers",
                int(config.get("codex.zimage_l2p.qwen_layers", 36)),
            )
            writer.add_uint32(
                "codex.zimage_l2p.qwen_heads",
                int(config.get("codex.zimage_l2p.qwen_heads", 32)),
            )
            writer.add_uint32(
                "codex.zimage_l2p.qwen_kv_heads",
                int(config.get("codex.zimage_l2p.qwen_kv_heads", 8)),
            )
            writer.add_uint32(
                "codex.zimage_l2p.qwen_vocab",
                int(config.get("codex.zimage_l2p.qwen_vocab", 151936)),
            )
        else:
            raise RuntimeError(f"Unsupported L2P GGUF component metadata: {l2p_component!r}")

    writer.add_string("gguf.quantized_at_utc", _dt.datetime.now(tz=_dt.timezone.utc).isoformat())
    writer.add_file_type(int(recipe.llama_file_type))
    writer.add_quantization_version(int(recipe.quantization_version))
    writer.add_string("gguf.quantization", recipe.recipe.value)
    writer.add_string("codex.quant_recipe", recipe.recipe.value)
    writer.add_string("codex.quant_base_type", recipe.default_tensor_type.value)
    if quant_policy is not None and quant_policy_preset is not None:
        writer.add_string("codex.quant_policy", str(quant_policy))
        writer.add_string("codex.quant_policy_preset", str(quant_policy_preset.value))

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
