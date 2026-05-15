"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF text conditioning builder (tokenizer + text encoder).
Loads tokenizer metadata and text encoder weights from local paths only, applies strict embedding-key alias normalization for GGUF T5 variants,
forces forward-only GGUF dequantization for TE loads with explicit target-device routing, and then builds prompt/negative embeddings for the WAN GGUF runtime.
When smart offload requests a direct text-encoder offload transition, emits canonical INFO audit events via `backend.smart_offload`.
That transition event is tagged via the canonical `SmartOffloadAction.DIRECT_OFFLOAD` enum action.

Symbols (top-level; keep in sync; no ghosts):
- `WAN22_DEFAULT_MAX_SEQUENCE_LENGTH` (constant): Default token length used for WAN22 prompt embeddings (aligns with Diffusers default).
- `_prompt_clean` (function): Diffusers-style prompt cleaning (optional ftfy + HTML unescape + whitespace collapse).
- `_resolve_max_sequence_length` (function): Chooses a safe tokenizer max length, clamped to `WAN22_DEFAULT_MAX_SEQUENCE_LENGTH`.
- `_raise_unsupported_packed_text_encoder` (function): Raises the canonical root-runtime error for removed packed GGUF text-encoder artifacts.
- `_place_gguf_non_quant_tensors` (function): Moves non-quantized TE params/buffers to target device and applies floating dtype casts while preserving integer buffers.
- `get_text_context` (function): Builds text conditioning/context (single or batched prompt + negative prompt inputs) for the WAN transformer with strict fail-loud text-encoder key validation, device-aware TE weight loading, and global GGUF dequant policy alignment.
"""

from __future__ import annotations

import html
import os
import re
from collections.abc import Sequence
from typing import Any, Optional, Tuple

import torch

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.smart_offload import SmartOffloadAction, log_smart_offload_action
from apps.backend.runtime.ops.operations_gguf import is_packed_gguf_artifact
from .config import as_torch_dtype, resolve_device_name
from .diagnostics import get_logger


WAN22_DEFAULT_MAX_SEQUENCE_LENGTH = 512


def _prompt_clean(text: str) -> str:
    text = str(text or "")
    try:
        import ftfy  # type: ignore

        text = ftfy.fix_text(text)
    except ModuleNotFoundError:
        pass
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _resolve_max_sequence_length(tok: Any) -> int:
    raw = getattr(tok, "model_max_length", None)
    try:
        raw_int = int(raw) if raw is not None else WAN22_DEFAULT_MAX_SEQUENCE_LENGTH
    except Exception:
        raw_int = WAN22_DEFAULT_MAX_SEQUENCE_LENGTH

    # Some tokenizers expose an absurd sentinel (e.g. 1e30); clamp to WAN defaults.
    max_len = min(raw_int, WAN22_DEFAULT_MAX_SEQUENCE_LENGTH)
    if max_len <= 0:
        max_len = WAN22_DEFAULT_MAX_SEQUENCE_LENGTH
    return int(max_len)


def _raise_unsupported_packed_text_encoder() -> None:
    raise RuntimeError(
        "WAN22 GGUF: packed GGUF text-encoder artifacts are not supported on the root runtime path. "
        "Load the base `.gguf` text encoder artifact instead."
    )


def _is_gguf_quantized_tensor(tensor_obj: Any) -> bool:
    if getattr(tensor_obj, "qtype", None) is not None:
        return True
    if is_packed_gguf_artifact(tensor_obj):
        _raise_unsupported_packed_text_encoder()
    return False


def _place_gguf_non_quant_tensors(
    module: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, int]:
    moved_params = 0
    moved_buffers = 0
    with torch.no_grad():
        for submodule_name, submodule_obj in module.named_modules():
            for param_name, parameter in submodule_obj.named_parameters(recurse=False):
                if parameter is None or _is_gguf_quantized_tensor(parameter):
                    continue
                if getattr(parameter, "is_meta", False):
                    raise RuntimeError(
                        "WAN22 GGUF: unresolved meta parameter in text encoder after load: "
                        f"module={submodule_name or '<root>'} name={param_name}"
                    )
                target_dtype = dtype if torch.is_floating_point(parameter) else parameter.dtype
                if parameter.device != device or parameter.dtype != target_dtype:
                    parameter.data = parameter.data.to(device=device, dtype=target_dtype)
                    moved_params += 1
            for buffer_name, buffer in submodule_obj.named_buffers(recurse=False):
                if buffer is None or _is_gguf_quantized_tensor(buffer):
                    continue
                if getattr(buffer, "is_meta", False):
                    raise RuntimeError(
                        "WAN22 GGUF: unresolved meta buffer in text encoder after load: "
                        f"module={submodule_name or '<root>'} name={buffer_name}"
                    )
                target_dtype = dtype if torch.is_floating_point(buffer) else buffer.dtype
                if buffer.device != device or buffer.dtype != target_dtype:
                    submodule_obj._buffers[buffer_name] = buffer.to(device=device, dtype=target_dtype)
                    moved_buffers += 1
    return moved_params, moved_buffers


def get_text_context(
    model_dir: str,
    prompt: str | Sequence[str],
    negative: Optional[str | Sequence[str]],
    *,
    device: str,
    dtype: str,
    text_encoder_dir: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
    vae_dir: Optional[str] = None,  # unused (kept for compatibility with existing call sites)
    model_key: Optional[str] = None,  # unused (kept for compatibility with existing call sites)
    metadata_dir: Optional[str] = None,
    logger: Any = None,
    offload_after: bool = True,
    te_device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GGUF path: use Transformers tokenizer + encoder only; do NOT fall back to Diffusers.

    - Never downloads; never calls Diffusers. If not found, raises an explicit, actionable error.
    - Tokenizer/config are read from local folders only (metadata repo or explicit dirs).
    """

    _ = (model_dir, vae_dir, model_key)  # kept for signature compatibility
    log = get_logger(logger)

    # Normalize device strings early (call sites sometimes pass 'auto').
    device = resolve_device_name(device)
    if te_device is not None:
        te_device = resolve_device_name(te_device)

    from transformers import AutoConfig, AutoTokenizer

    try:
        from transformers import UMT5EncoderModel as _Enc
    except Exception:
        from transformers import T5EncoderModel as _Enc  # type: ignore

    # Resolve tokenizer dir: prefer explicit tokenizer_dir; else infer from metadata_dir/tokenizer*
    tk_dir = tokenizer_dir
    if (not tk_dir) and metadata_dir:
        cand = os.path.join(metadata_dir, "tokenizer")
        cand2 = os.path.join(metadata_dir, "tokenizer_2")
        if os.path.isdir(cand):
            tk_dir = cand
        elif os.path.isdir(cand2):
            tk_dir = cand2

    te_path = text_encoder_dir
    te_file: Optional[str] = None
    if te_path and os.path.isfile(te_path) and te_path.lower().endswith((".safetensors", ".gguf")):
        te_file = te_path
        te_path = os.path.dirname(te_path)
    if tk_dir and os.path.isfile(tk_dir):
        tk_dir = os.path.dirname(tk_dir)

    if not tk_dir or not os.path.isdir(tk_dir):
        raise RuntimeError(
            "WAN22 GGUF: tokenizer metadata missing or invalid; provide 'wan_metadata_dir' or 'wan_tokenizer_dir'."
        )

    try:
        tok = AutoTokenizer.from_pretrained(tk_dir, use_fast=True, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"WAN22 GGUF: failed to load tokenizer from '{tk_dir}': {exc}") from exc

    max_sequence_length = _resolve_max_sequence_length(tok)
    if isinstance(prompt, str):
        prompt_list = [_prompt_clean(prompt)]
    elif isinstance(prompt, Sequence) and not isinstance(prompt, (bytes, bytearray)):
        prompt_list = []
        for item in prompt:
            if not isinstance(item, str):
                raise RuntimeError(
                    "WAN22 GGUF: prompt sequence must contain only strings, "
                    f"got {type(item).__name__}."
                )
            prompt_list.append(_prompt_clean(item))
    else:
        raise RuntimeError(
            "WAN22 GGUF: prompt must be a string or sequence of strings, "
            f"got {type(prompt).__name__}."
        )
    if not prompt_list:
        raise RuntimeError("WAN22 GGUF: prompt sequence must not be empty.")

    negative_list: list[str]
    if negative is None:
        negative_list = ["" for _ in prompt_list]
    elif isinstance(negative, str):
        negative_list = [_prompt_clean(negative)]
    elif isinstance(negative, Sequence) and not isinstance(negative, (bytes, bytearray)):
        negative_list = []
        for item in negative:
            if not isinstance(item, str):
                raise RuntimeError(
                    "WAN22 GGUF: negative prompt sequence must contain only strings, "
                    f"got {type(item).__name__}."
                )
            negative_list.append(_prompt_clean(item))
    else:
        raise RuntimeError(
            "WAN22 GGUF: negative prompt must be a string, sequence of strings, or None, "
            f"got {type(negative).__name__}."
        )

    if len(negative_list) == 1 and len(prompt_list) > 1:
        negative_list = negative_list * len(prompt_list)
    if len(negative_list) != len(prompt_list):
        raise RuntimeError(
            "WAN22 GGUF: prompt and negative prompt batch sizes must match "
            f"(prompt={len(prompt_list)} negative={len(negative_list)})."
        )

    log.info(
        "[wan22.gguf] tokenizer loaded: dir=%s model_max_len=%s effective_max_len=%d",
        tk_dir,
        str(getattr(tok, "model_max_length", None)),
        int(max_sequence_length),
    )

    te_dev_eff = (te_device or device).strip().lower()
    if te_dev_eff == "gpu":
        te_dev_eff = "cuda"
    if te_dev_eff == "auto":
        te_dev_eff = device.strip().lower()
        if te_dev_eff == "gpu":
            te_dev_eff = "cuda"
    if te_dev_eff not in {"cpu", "cuda"} and not re.fullmatch(r"cuda:\d+", te_dev_eff):
        raise RuntimeError(
            "WAN22 GGUF: 'gguf_te_device' must be one of "
            f"'auto', 'cpu', 'cuda', or 'cuda:<index>' (got {te_device!r})."
        )

    # CPU TE requires fp32 (avoid implicit casts / weird numerics)
    if te_dev_eff == "cpu" and str(dtype).lower().strip() not in {"fp32", "float32"}:
        dtype = "fp32"

    # Strict: require a TE weights file; directory-based TE loading is not supported in WAN22 GGUF.
    if te_file is None:
        raise RuntimeError(
            "WAN22 GGUF: 'wan_text_encoder_path' (.safetensors or .gguf file) is required. Directory-based text encoders are not supported."
        )

    te_impl_label = "gguf" if te_file.lower().endswith(".gguf") else "safetensors"
    log.info(
        "[wan22.gguf] text-encoder: impl=%s device=%s",
        te_impl_label,
        te_dev_eff,
    )

    if not metadata_dir or not os.path.isdir(metadata_dir):
        raise RuntimeError("WAN22 GGUF: 'wan_metadata_dir' is required when providing 'wan_text_encoder_path'.")
    enc_dir = os.path.join(metadata_dir, "text_encoder")
    if not os.path.isdir(enc_dir):
        raise RuntimeError(f"WAN22 GGUF: expected text encoder config under metadata repo: '{enc_dir}'")

    try:
        cfg = AutoConfig.from_pretrained(enc_dir, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"WAN22 GGUF: failed to read text encoder config from '{enc_dir}': {exc}") from exc

    use_dev_name = te_dev_eff.lower().strip()
    target_device = use_dev_name
    te_is_gguf = te_file.lower().endswith(".gguf")
    if te_is_gguf:
        from transformers import modeling_utils as hf_modeling_utils
        from apps.backend.runtime.ops.operations import using_codex_operations

        with using_codex_operations(
            device=torch.device(target_device),
            dtype=as_torch_dtype(dtype),
            manual_cast_enabled=True,
            weight_format="gguf",
        ):
            with hf_modeling_utils.no_init_weights():
                enc = _Enc(cfg)
    else:
        enc = _Enc(cfg)
    try:
        if te_file.lower().endswith(".safetensors"):
            from safetensors.torch import load_file as _load_st

            sd = _load_st(te_file, device=target_device)
        else:
            from apps.backend.runtime.checkpoint.io import load_gguf_state_dict

            sd = load_gguf_state_dict(
                te_file,
                dequantize=False,
                computation_dtype=as_torch_dtype(dtype),
                device=target_device,
            )
            shared_weight = sd.get("shared.weight")
            encoder_embed_weight = sd.get("encoder.embed_tokens.weight")
            if shared_weight is not None and encoder_embed_weight is None:
                sd["encoder.embed_tokens.weight"] = shared_weight
            elif encoder_embed_weight is not None and shared_weight is None:
                sd["shared.weight"] = encoder_embed_weight
            elif shared_weight is not None and encoder_embed_weight is not None:
                shared_shape = tuple(int(dim) for dim in getattr(shared_weight, "shape", ()))
                embed_shape = tuple(int(dim) for dim in getattr(encoder_embed_weight, "shape", ()))
                if shared_shape != embed_shape:
                    raise RuntimeError(
                        "WAN22 GGUF: text encoder embedding alias shape mismatch "
                        f"(shared.weight={shared_shape} encoder.embed_tokens.weight={embed_shape})."
                    )
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "WAN22 GGUF: text encoder strict load failed: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
            )
    except Exception as exc:
        raise RuntimeError(f"WAN22 GGUF: failed to load text encoder weights '{te_file}': {exc}") from exc

    dev = torch.device(use_dev_name)
    if not te_is_gguf:
        try:
            enc = enc.to(device=dev, dtype=as_torch_dtype(dtype))
        except Exception:
            enc = enc.to(device=dev)

    if te_is_gguf:
        _place_gguf_non_quant_tensors(enc, device=dev, dtype=as_torch_dtype(dtype))

        requested_compute_dtype = as_torch_dtype(dtype)

        def _apply_compute_dtype(module_obj: Any) -> None:
            weight = getattr(module_obj, "weight", None)
            if weight is None or not hasattr(weight, "computation_dtype"):
                return
            try:
                setattr(weight, "computation_dtype", requested_compute_dtype)
            except Exception:
                return

        _apply_compute_dtype(getattr(enc, "shared", None))
        encoder_obj = getattr(enc, "encoder", None)
        _apply_compute_dtype(getattr(encoder_obj, "embed_tokens", None))

    def _do(clean_texts: Sequence[str]) -> torch.Tensor:
        inputs = tok(
            list(clean_texts),
            padding="max_length",
            truncation=True,
            max_length=int(max_sequence_length),
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            out = enc(**inputs).last_hidden_state
            mask = inputs.get("attention_mask", None)
            if mask is not None:
                out = out * mask.to(dtype=out.dtype).unsqueeze(-1)
            return out.to(as_torch_dtype(dtype))

    p = _do(prompt_list)
    n = _do(negative_list)

    cfg_hidden = int(getattr(getattr(enc, "config", None), "hidden_size", p.shape[-1]))
    if int(p.shape[-1]) != cfg_hidden:
        raise RuntimeError(f"WAN22 GGUF: TE hidden_size mismatch: output={int(p.shape[-1])} config={cfg_hidden}")

    log.info(
        "[wan22.gguf] TE outputs: prompt=%s negative=%s dtype=%s device=%s",
        tuple(p.shape),
        tuple(n.shape),
        str(p.dtype),
        str(p.device),
    )

    if offload_after:
        if not te_is_gguf:
            offload_device = memory_management.manager.offload_device()
            if not isinstance(offload_device, torch.device):
                raise RuntimeError(
                    "WAN22 GGUF: memory manager offload_device() must return torch.device "
                    f"(got {type(offload_device).__name__})."
                )
            moved_to_cpu = False
            try:
                enc.to(offload_device)
                moved_to_cpu = True
            except Exception as exc:
                raise RuntimeError(
                    "WAN22 GGUF: failed to offload text encoder to memory-manager offload device "
                    f"(from_device={dev} to_device={offload_device})."
                ) from exc
            if moved_to_cpu and dev.type == "cuda":
                log.info(
                    "[wan22.gguf] text-encoder offloaded to %s (smart_offload)",
                    offload_device,
                )
                log_smart_offload_action(
                    SmartOffloadAction.DIRECT_OFFLOAD,
                    source="runtime.families.wan22.text_context",
                    component="text_encoder",
                    from_device=str(dev),
                    to_device=str(offload_device),
                )
        del enc
        memory_management.manager.soft_empty_cache(force=True)

    return p, n
