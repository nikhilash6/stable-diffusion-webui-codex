"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Deterministic LoRA loader/applier with transactional backups.
Applies patch dictionaries onto model parameters with device/dtype management, supporting GGUF packed parameters.
Supports explicit merge precision mode (`CODEX_LORA_MERGE_MODE`) and refresh signature mode (`CODEX_LORA_REFRESH_SIGNATURE`).
Fails loud when removed packed GGUF artifacts reach the root runtime path.

Symbols (top-level; keep in sync; no ghosts):
- `_trace_load_patch_debug_enabled` (function): Returns whether verbose per-patch load tracing is enabled via env flag.
- `_raise_packed_gguf_unsupported` (function): Raises the canonical root-runtime error for removed packed GGUF artifacts during LoRA refresh.
- `get_parameter_devices` (function): Captures current parameter device mapping for later restoration.
- `set_parameter_devices` (function): Restores parameters to a previously captured device mapping.
- `_numpy_safe_quantize_input` (function): Materializes a CPU float tensor into a NumPy-safe array for GGUF re-quantization, promoting BF16 to FP32 before the NumPy bridge.
- `CodexLoraLoader` (class): High-level loader/applier that integrates mapping, device placement, and progress reporting (tqdm).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import hashlib
import logging
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from apps.backend.infra.config.env_flags import env_flag
from apps.backend.infra.config.lora_merge_mode import LoraMergeMode, read_lora_merge_mode
from apps.backend.infra.config.lora_refresh_signature import (
    LoraRefreshSignatureMode,
    read_lora_refresh_signature_mode,
)
from apps.backend.quantization.api import quantize_numpy
from apps.backend.quantization.tensor import CodexParameter
from apps.backend.runtime import utils
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.ops.operations_gguf import dequantize_tensor, is_packed_gguf_artifact

from .lora_merge import merge_lora_to_weight
from .lora_types import LoraPatchEntry

logger = get_backend_logger("backend.patchers.lora")


def _trace_load_patch_debug_enabled() -> bool:
    return env_flag("CODEX_TRACE_LOAD_PATCH_DEBUG", default=False)


def _raise_packed_gguf_unsupported(*, target: str | None = None) -> None:
    detail = f" target={target!r}." if target is not None else ""
    raise RuntimeError(
        "LoRA cannot run on packed GGUF artifacts on the root runtime path. "
        "Load the base `.gguf` artifact instead."
        f"{detail}"
    )


def get_parameter_devices(model) -> Dict[str, torch.device]:
    return {key: p.device for key, p in model.named_parameters()}


def set_parameter_devices(model, parameter_devices: Mapping[str, torch.device]) -> None:
    for key, device in parameter_devices.items():
        parameter = utils.get_attr(model, key)
        if parameter.device != device:
            parameter = utils.tensor2parameter(parameter.to(device=device))
            utils.set_attr_raw(model, key, parameter)


def _numpy_safe_quantize_input(tensor: torch.Tensor):
    cpu_tensor = tensor.detach().to(device="cpu")
    if cpu_tensor.dtype == torch.bfloat16:
        cpu_tensor = cpu_tensor.to(dtype=torch.float32)
    return cpu_tensor.numpy()


class CodexLoraLoader:
    """Deterministic LoRA loader with transactional backups and structured logging."""

    def __init__(self, model):
        self.model = model
        self.backup: Dict[str, torch.Tensor] = {}
        self.online_parents: List[torch.nn.Module] = []
        self.loaded_signature = ""

    @torch.no_grad()
    def refresh(
        self,
        lora_patches: MutableMapping[Tuple[str, float, float, bool], Dict[str, List[LoraPatchEntry]]],
        *,
        offload_device: torch.device | str | None = None,
        force_refresh: bool = False,
    ) -> None:
        if lora_patches and any(is_packed_gguf_artifact(param) for _key, param in self.model.named_parameters()):
            _raise_packed_gguf_unsupported()
        merge_mode = read_lora_merge_mode()
        merge_dtype = torch.float64 if merge_mode is LoraMergeMode.PRECISE else torch.float32
        signature_mode = read_lora_refresh_signature_mode()
        signature = self._signature(lora_patches, mode=signature_mode)
        resolved_offload_device = (
            memory_management.manager.offload_device()
            if offload_device is None
            else torch.device(offload_device)
        )
        trace_load_patch_debug = _trace_load_patch_debug_enabled() and logger.isEnabledFor(logging.DEBUG)
        if signature == self.loaded_signature and not force_refresh:
            if trace_load_patch_debug:
                logger.debug("LoRA loader refresh skipped (no changes).")
            return

        grouped = self._group_patches(lora_patches, trace_load_patch_debug=trace_load_patch_debug)
        memory_management.manager.signal_empty_cache = True
        parameter_devices = get_parameter_devices(self.model)

        self._restore_backups(parameter_devices)

        offline_groups = sum(1 for (_, online) in grouped.keys() if not online)
        logger.info(
            "Refreshing LoRA patches: groups=%d offline=%d online=%d merge_mode=%s signature_mode=%s",
            len(grouped),
            offline_groups,
            len(grouped) - offline_groups,
            merge_mode.value,
            signature_mode.value,
        )

        offline_total = sum(len(patches) for (key, online), patches in grouped.items() if not online and patches)
        progress = tqdm(total=offline_total, desc="lora merge", unit="patch") if offline_total else None

        try:
            for (param_key, online_mode), entries in grouped.items():
                if not entries:
                    continue
                if online_mode:
                    self._register_online(param_key, entries)
                    continue

                parent_layer, child_key, parameter = utils.get_attr_with_parent(self.model, param_key)
                if not isinstance(parameter, torch.nn.Parameter):
                    raise TypeError(f"LoRA target {param_key} is not a torch.nn.Parameter.")
                if is_packed_gguf_artifact(parameter):
                    _raise_packed_gguf_unsupported(target=param_key)
                if param_key not in self.backup:
                    if isinstance(parameter, CodexParameter) and parameter.qtype is not None:
                        self.backup[param_key] = parameter.copy_with_data(
                            parameter.data.detach().to(device=resolved_offload_device).clone()
                        )
                    else:
                        self.backup[param_key] = parameter.detach().to(device=resolved_offload_device).clone()

                gguf_parameter = None
                tensor = parameter

                if hasattr(parameter, "bnb_quantized"):
                    raise NotImplementedError(
                        "NF4/FP4 is not supported for LoRA. "
                        f"Found bnb_quantized weight at {param_key!r}. Convert the base model to GGUF or use a safetensors fp16/bf16/fp32 checkpoint."
                    )
                elif isinstance(parameter, CodexParameter) and parameter.qtype is not None:
                    gguf_parameter = parameter
                    tensor = dequantize_tensor(parameter)
                else:
                    tensor = parameter.data

                try:
                    merged = merge_lora_to_weight(
                        entries,
                        tensor,
                        key=param_key,
                        computation_dtype=merge_dtype,
                    )
                except RuntimeError as err:
                    if "out of memory" not in str(err).lower():
                        raise
                    logger.warning(
                        "LoRA merge OOM on %s; offloading to %s and retrying",
                        param_key,
                        resolved_offload_device,
                    )
                    self._offload_model(parameter_devices, resolved_offload_device)
                    memory_management.manager.soft_empty_cache()
                    merged = merge_lora_to_weight(
                        entries,
                        tensor,
                        key=param_key,
                        computation_dtype=merge_dtype,
                    )

                if gguf_parameter is None:
                    merged = merged.to(dtype=parameter.dtype, device=parameter.device)

                if gguf_parameter is not None:
                    # Re-quantize offline-merged weights back into GGUF packed storage.
                    # We do this explicitly (no implicit dtype casts): storage stays byte-packed.
                    qtype = gguf_parameter.qtype
                    if qtype is None:
                        raise RuntimeError(f"Unexpected GGUF parameter without qtype: {param_key}")

                    packed = quantize_numpy(_numpy_safe_quantize_input(merged), qtype)
                    restored = CodexParameter(
                        packed,
                        qtype=qtype,
                        shape=tuple(merged.shape),
                        computation_dtype=gguf_parameter.computation_dtype,
                    ).to(device=parameter.device, dtype=gguf_parameter.computation_dtype)
                    utils.set_attr_raw(self.model, param_key, restored)
                else:
                    utils.set_attr_raw(self.model, param_key, torch.nn.Parameter(merged, requires_grad=False))

                if progress is not None:
                    progress.update(len(entries))
                if trace_load_patch_debug:
                    logger.debug("Applied %d LoRA patches to %s", len(entries), param_key)

            self.loaded_signature = signature
        finally:
            if progress is not None:
                progress.close()
            set_parameter_devices(self.model, parameter_devices)

    def _restore_backups(self, parameter_devices: Mapping[str, torch.device]) -> None:
        for module in self.online_parents:
            if hasattr(module, "codex_online_loras"):
                del module.codex_online_loras
        self.online_parents.clear()

        for key, tensor in self.backup.items():
            target_device = parameter_devices.get(key, tensor.device)
            if isinstance(tensor, CodexParameter) and tensor.qtype is not None:
                restored = tensor.to(device=target_device, dtype=tensor.computation_dtype)
                utils.set_attr_raw(self.model, key, restored)
                continue
            restored = tensor.to(device=target_device).clone()
            utils.set_attr_raw(self.model, key, torch.nn.Parameter(restored, requires_grad=False))
        self.backup.clear()

    def _register_online(self, param_key: str, entries: Sequence[LoraPatchEntry]) -> None:
        parent_layer, child_key, parameter = utils.get_attr_with_parent(self.model, param_key)
        if not hasattr(parent_layer, "codex_online_loras"):
            parent_layer.codex_online_loras = {}
        parent_layer.codex_online_loras[child_key] = list(entries)
        if parent_layer not in self.online_parents:
            self.online_parents.append(parent_layer)
        trace_load_patch_debug = _trace_load_patch_debug_enabled() and logger.isEnabledFor(logging.DEBUG)
        if trace_load_patch_debug:
            logger.debug("Registered %d online LoRA patches for %s", len(entries), param_key)

    def _group_patches(
        self,
        lora_patches: Mapping[Tuple[str, float, float, bool], Dict[str, List[LoraPatchEntry]]],
        *,
        trace_load_patch_debug: bool,
    ) -> Dict[Tuple[str, bool], List[LoraPatchEntry]]:
        grouped: Dict[Tuple[str, bool], List[LoraPatchEntry]] = {}
        for (filename, strength_patch, strength_model, online_mode), param_map in lora_patches.items():
            for param_key, patches in param_map.items():
                target = grouped.setdefault((param_key, online_mode), [])
                target.extend(patches)
                if trace_load_patch_debug:
                    logger.debug(
                        "Queued %d patches for %s (file=%s strength_patch=%.3f strength_model=%.3f online=%s)",
                        len(patches),
                        param_key,
                        filename,
                        strength_patch,
                        strength_model,
                        online_mode,
                    )
        return grouped

    def _signature(
        self,
        lora_patches: Mapping[Tuple[str, float, float, bool], Dict[str, List[LoraPatchEntry]]],
        *,
        mode: LoraRefreshSignatureMode | None = None,
    ) -> str:
        selected_mode = read_lora_refresh_signature_mode() if mode is None else mode
        if selected_mode is LoraRefreshSignatureMode.CONTENT_SHA256:
            return self._content_signature(lora_patches)
        items = []
        for key in sorted(lora_patches.keys()):
            param_map = lora_patches[key]
            for param_key in sorted(param_map.keys()):
                items.append((key, param_key, len(param_map[param_key])))
        return f"structural:{items}"

    def _content_signature(
        self,
        lora_patches: Mapping[Tuple[str, float, float, bool], Dict[str, List[LoraPatchEntry]]],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"codex-lora-refresh-signature:v1")
        for key in sorted(lora_patches.keys()):
            self._hash_signature_value(digest, key, path="bundle_key")
            param_map = lora_patches[key]
            for param_key in sorted(param_map.keys()):
                digest.update(b"|param|")
                self._hash_signature_value(digest, param_key, path="param_key")
                entries = param_map[param_key]
                self._hash_signature_value(digest, len(entries), path="entry_count")
                for entry_index, entry in enumerate(entries):
                    digest.update(b"|entry|")
                    self._hash_signature_value(digest, entry_index, path="entry_index")
                    self._hash_signature_value(digest, entry, path="entry")
        return f"content_sha256:{digest.hexdigest()}"

    @staticmethod
    def _hash_signature_value(digest: Any, value: object, *, path: str) -> None:
        if value is None:
            digest.update(b"none")
            return
        if isinstance(value, bool):
            digest.update(b"bool")
            digest.update(b"1" if value else b"0")
            return
        if isinstance(value, int):
            digest.update(b"int")
            digest.update(str(value).encode("utf-8"))
            return
        if isinstance(value, float):
            digest.update(b"float")
            digest.update(repr(value).encode("utf-8"))
            return
        if isinstance(value, str):
            digest.update(b"str")
            digest.update(value.encode("utf-8"))
            return
        if isinstance(value, bytes):
            digest.update(b"bytes")
            digest.update(value)
            return
        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            raw = tensor.to(device="cpu").view(torch.uint8).numpy().tobytes()
            digest.update(raw)
            return
        if isinstance(value, Mapping):
            digest.update(b"mapping")
            for key in sorted(value.keys(), key=lambda item: str(item)):
                CodexLoraLoader._hash_signature_value(digest, key, path=f"{path}.key")
                CodexLoraLoader._hash_signature_value(digest, value[key], path=f"{path}.value")
            return
        if isinstance(value, Sequence):
            digest.update(b"sequence")
            digest.update(str(len(value)).encode("utf-8"))
            for index, item in enumerate(value):
                CodexLoraLoader._hash_signature_value(digest, index, path=f"{path}.index")
                CodexLoraLoader._hash_signature_value(digest, item, path=f"{path}.item")
            return
        if callable(value):
            digest.update(b"callable")
            module = getattr(value, "__module__", "")
            qualname = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
            digest.update(f"{module}:{qualname}".encode("utf-8"))
            return
        raise TypeError(
            "Unsupported value in content signature at {path}: {type_name}".format(
                path=path,
                type_name=type(value).__name__,
            )
        )

    def _offload_model(self, parameter_devices: Mapping[str, torch.device], offload_device: torch.device) -> None:
        for key in parameter_devices.keys():
            parameter = utils.get_attr(self.model, key)
            if isinstance(parameter, CodexParameter) and parameter.qtype is not None:
                utils.set_attr_raw(
                    self.model,
                    key,
                    parameter.to(device=offload_device, dtype=parameter.computation_dtype),
                )
                continue
            utils.set_attr_raw(
                self.model,
                key,
                torch.nn.Parameter(parameter.to(device=offload_device).clone(), requires_grad=False),
            )

__all__ = [
    "CodexLoraLoader",
    "get_parameter_devices",
    "set_parameter_devices",
]
