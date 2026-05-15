"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native Netflix VOID T5 text-encoder runtime loader.
Loads the base-bundle T5 tokenizer + encoder through repo-owned IntegratedT5/keymap/safe-load seams, preserving local
weights/config ownership and the canonical no-key-rewrite birth/load path.

Symbols (top-level; keep in sync; no ghosts):
- `NetflixVoidTextEncoderRuntime` (dataclass): Loaded T5 wrapper + tokenizer pair for the native Netflix VOID runtime.
- `load_netflix_void_text_encoder_runtime` (function): Load the family-owned T5 encoder/tokenizer from the resolved base bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from transformers import AutoTokenizer, modeling_utils

from apps.backend.runtime.common.nn.t5 import IntegratedT5
from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime.checkpoint.io import load_torch_file
from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.state_dict.keymap_t5_text_encoder import resolve_t5_text_encoder_keyspace
from apps.backend.runtime.text_processing.t5_engine import T5TextProcessingEngine

from ..model import NetflixVoidBaseBundle
from .bundle_io import (
    read_netflix_void_component_config,
    resolve_netflix_void_component_dir,
    resolve_netflix_void_component_weights_path,
)

_ALLOWED_T5_CLASS_NAMES = frozenset({"T5EncoderModel", "UMT5EncoderModel"})
_REQUIRED_T5_CONFIG_KEYS = frozenset(
    {
        "d_ff",
        "d_model",
        "dense_act_fn",
        "is_gated_act",
        "model_type",
        "num_heads",
        "num_layers",
        "vocab_size",
    }
)


@dataclass(frozen=True, slots=True)
class NetflixVoidTextEncoderRuntime:
    wrapper: IntegratedT5
    tokenizer: Any
    prompt_engine: T5TextProcessingEngine
    tokenizer_dir: str
    weights_path: str


def _validate_t5_config(raw_config: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    config = dict(raw_config)
    class_name = config.get("_class_name")
    if class_name is not None and str(class_name) not in _ALLOWED_T5_CLASS_NAMES:
        raise RuntimeError(
            f"Netflix VOID text-encoder config `_class_name` must be one of {tuple(sorted(_ALLOWED_T5_CLASS_NAMES))!r}, got {class_name!r}."
        )
    missing = sorted(key for key in _REQUIRED_T5_CONFIG_KEYS if key not in config)
    if missing:
        raise RuntimeError(
            f"Netflix VOID text-encoder config is missing required keys {missing!r} at {context!r}."
        )
    return config


def _resolve_prompt_engine_min_length(tokenizer: Any) -> int:
    raw_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(raw_length, int) and 0 < raw_length < 1_000_000:
        return max(1, int(raw_length))
    return 256


def load_netflix_void_text_encoder_runtime(
    *,
    base_bundle: NetflixVoidBaseBundle,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> NetflixVoidTextEncoderRuntime:
    if not isinstance(device, torch.device):
        raise RuntimeError(f"Netflix VOID text-encoder loader requires torch.device, got {type(device).__name__}.")
    if not isinstance(torch_dtype, torch.dtype):
        raise RuntimeError(
            f"Netflix VOID text-encoder loader requires torch.dtype, got {type(torch_dtype).__name__}."
        )

    tokenizer_dir = resolve_netflix_void_component_dir(base_bundle, component_name="tokenizer")
    text_encoder_dir = resolve_netflix_void_component_dir(base_bundle, component_name="text_encoder")
    config = _validate_t5_config(
        read_netflix_void_component_config(base_bundle, component_name="text_encoder"),
        context=str(text_encoder_dir),
    )
    weights_path = resolve_netflix_void_component_weights_path(base_bundle, component_name="text_encoder")

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        use_fast=True,
    )

    raw_state_dict = load_torch_file(str(weights_path), device="cpu")
    if not hasattr(raw_state_dict, "keys"):
        raise RuntimeError(
            "Netflix VOID text-encoder loader requires a mapping state_dict, got "
            f"{type(raw_state_dict).__name__}."
        )
    resolved_state_dict = resolve_t5_text_encoder_keyspace(raw_state_dict).view

    to_args = dict(device=device, dtype=torch_dtype)
    with modeling_utils.no_init_weights():
        with using_codex_operations(**to_args, manual_cast_enabled=True):
            wrapper = IntegratedT5(config).to(**to_args)
    wrapper.transformer.compute_dtype = torch.float32 if torch_dtype in {torch.float16, torch.bfloat16} else torch_dtype

    missing, unexpected = safe_load_state_dict(
        wrapper,
        resolved_state_dict,
        log_name="netflix_void.t5",
        ignore_missing_prefixes=("logit_scale",),
    )
    if missing or unexpected:
        raise RuntimeError(
            "Netflix VOID text-encoder strict load failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}."
        )
    wrapper.eval()

    prompt_engine = T5TextProcessingEngine(
        text_encoder=wrapper,
        tokenizer=tokenizer,
        min_length=_resolve_prompt_engine_min_length(tokenizer),
    )
    return NetflixVoidTextEncoderRuntime(
        wrapper=wrapper,
        tokenizer=tokenizer,
        prompt_engine=prompt_engine,
        tokenizer_dir=str(tokenizer_dir),
        weights_path=str(weights_path),
    )


__all__ = ["NetflixVoidTextEncoderRuntime", "load_netflix_void_text_encoder_runtime"]
