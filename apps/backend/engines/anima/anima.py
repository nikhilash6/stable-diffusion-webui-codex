"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima engine facade for txt2img/img2img.
Defines the `AnimaEngine` class and integrates with the common `CodexDiffusionEngine` lifecycle.
Implements Anima conditioning using dual tokenization (Qwen embeddings + T5 ids/weights/attention-mask) and exposes canonical engine hooks used by shared pipelines.

Symbols (top-level; keep in sync; no ghosts):
- `_ANIMA_FACTORY` (constant): Factory used to assemble the Anima runtime and `CodexObjects`.
- `_AnimaPromptList` (class): Prompt wrapper carrying shared prompt metadata flags for Anima conditioning.
- `_canonical_device_label` (function): Normalize device identity labels for strict runtime consistency checks.
- `AnimaEngine` (class): Engine facade registered under engine id `anima`.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Any, Iterable, Mapping

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.common.model_scopes import stage_scoped_model_load
from apps.backend.engines.common.prompt_wrappers import PromptListBase
from apps.backend.engines.common.runtime_lifecycle import require_runtime
from apps.backend.engines.common.tensor_tree import detach_to_cpu
from apps.backend.runtime.model_registry.capabilities import ENGINE_SURFACES, SemanticEngine
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.models.loader import DiffusionModelBundle
from apps.backend.runtime.families.anima.text_encoder import (
    resolve_anima_qwen_max_length,
    resolve_anima_t5_max_length,
    tokenize_t5_with_weights,
)

from .factory import CodexAnimaFactory
from .spec import ANIMA_SPEC, AnimaEngineRuntime


logger = get_backend_logger("backend.engines.anima")
_ANIMA_FACTORY = CodexAnimaFactory(spec=ANIMA_SPEC)


class _AnimaPromptList(PromptListBase):
    def __init__(
        self,
        items: Iterable[str],
        *,
        is_negative_prompt: bool,
        smart_cache: bool | None,
    ) -> None:
        super().__init__(items, is_negative_prompt=is_negative_prompt, smart_cache=smart_cache)


def _canonical_device_label(value: object, *, field_name: str) -> str:
    if isinstance(value, torch.device):
        device = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise RuntimeError(f"Anima runtime assembly returned empty `{field_name}` device identity.")
        try:
            device = torch.device(raw)
        except Exception as exc:  # noqa: BLE001 - fail-loud identity parsing
            raise RuntimeError(
                f"Anima runtime assembly returned invalid `{field_name}` device identity: {raw!r}"
            ) from exc
    else:
        raise RuntimeError(
            f"Anima runtime assembly returned invalid `{field_name}` device identity; expected torch.device or str."
        )

    if device.type == "cuda" and device.index is None:
        return "cuda:0"
    return str(device)


class AnimaEngine(CodexDiffusionEngine):
    """Anima engine (Cosmos Predict2 + Anima adapter)."""

    engine_id = "anima"
    expected_family = ModelFamily.ANIMA

    def __init__(self) -> None:
        super().__init__()
        self._runtime: AnimaEngineRuntime | None = None

    def capabilities(self) -> EngineCapabilities:
        surface = ENGINE_SURFACES[SemanticEngine.ANIMA]
        tasks: list[TaskType] = []
        if surface.supports_txt2img:
            tasks.append(TaskType.TXT2IMG)
        if surface.supports_img2img:
            tasks.append(TaskType.IMG2IMG)
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=tuple(tasks),
            model_types=("anima",),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
        )

    def _prepare_prompt_wrappers(
        self,
        texts: list[str],
        proc: Any,
        *,
        is_negative: bool,
    ) -> _AnimaPromptList:
        smart_flag = getattr(proc, "smart_cache", None)
        smart_value = None if smart_flag is None else bool(smart_flag)
        return _AnimaPromptList(
            [str(t or "") for t in texts],
            is_negative_prompt=is_negative,
            smart_cache=smart_value,
        )

    @property
    def required_text_encoders(self) -> tuple[str, ...]:
        return ("qwen3",)

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        assembly = _ANIMA_FACTORY.assemble(bundle, options=options)
        runtime = assembly.runtime
        self._runtime = runtime
        runtime_device_label = _canonical_device_label(getattr(runtime, "device", None), field_name="runtime.device")
        denoiser_load_device = getattr(runtime.denoiser, "load_device", None)
        if denoiser_load_device is None:
            raise RuntimeError(
                "Anima runtime assembly returned denoiser without `load_device` for device consistency checks."
            )
        denoiser_device_label = _canonical_device_label(
            denoiser_load_device,
            field_name="denoiser.load_device",
        )
        if denoiser_device_label != runtime_device_label:
            raise RuntimeError(
                "Anima runtime device mismatch: runtime.device="
                f"{runtime_device_label} but denoiser.load_device={denoiser_device_label}"
            )
        runtime_compute_dtype = getattr(runtime, "core_compute_dtype", None)
        if not isinstance(runtime_compute_dtype, str) or not runtime_compute_dtype:
            raise RuntimeError(
                "Anima runtime assembly returned invalid `core_compute_dtype`; expected non-empty dtype label."
            )
        self._device = runtime_device_label
        self._dtype = runtime_compute_dtype
        logger.debug("Anima runtime assembled")
        return assembly.codex_objects

    def _on_unload(self) -> None:
        self._runtime = None

    def _require_runtime(self) -> AnimaEngineRuntime:
        return require_runtime(self._runtime, label=self.engine_id)

    @torch.no_grad()
    def get_learned_conditioning(self, prompt: list[str]):
        runtime = self._require_runtime()
        qwen_patcher = self.codex_objects.text_encoders["qwen3"].patcher

        texts = tuple(str(x or "") for x in prompt)
        is_negative = bool(getattr(prompt, "is_negative_prompt", False))
        smart_flag = getattr(prompt, "smart_cache", None)
        use_cache = bool(smart_flag) if smart_flag is not None else self.smart_cache_enabled
        qwen_default = resolve_anima_qwen_max_length()
        qwen_max_length = int(getattr(runtime.text.qwen3_text, "max_length", qwen_default) or qwen_default)
        t5_max_length = resolve_anima_t5_max_length()
        cache_key = (texts, is_negative, qwen_max_length, t5_max_length)

        cached = self._get_cached_cond(cache_key, bucket_name="anima.conditioning", enabled=use_cache)
        if isinstance(cached, dict) and all(
            key in cached for key in ("crossattn", "t5xxl_ids", "t5xxl_weights", "t5xxl_attention_mask")
        ):
            target_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
            core_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE)
            return {
                "crossattn": cached["crossattn"].to(device=target_device, dtype=core_dtype),
                "t5xxl_ids": cached["t5xxl_ids"].to(device=target_device, dtype=torch.long),
                "t5xxl_weights": cached["t5xxl_weights"].to(device=target_device, dtype=core_dtype),
                "t5xxl_attention_mask": cached["t5xxl_attention_mask"].to(device=target_device, dtype=torch.long),
            }

        with stage_scoped_model_load(
            qwen_patcher,
            smart_offload_enabled=self.smart_offload_enabled,
            manager=memory_management.manager,
        ):
            crossattn = runtime.text.qwen3_text(list(texts))
            if not isinstance(crossattn, torch.Tensor) or crossattn.ndim != 3:
                raise RuntimeError(
                    f"Anima Qwen text encoder returned invalid crossattn tensor: "
                    f"type={type(crossattn).__name__} shape={getattr(crossattn, 'shape', None)}"
                )

            core_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE)
            crossattn = crossattn.to(dtype=core_dtype)

            t5_batch = tokenize_t5_with_weights(
                tokenizer=runtime.text.t5_tokenizer,
                texts=list(texts),
                max_length=t5_max_length,
            )
            if t5_batch.input_ids.ndim != 2 or t5_batch.weights.ndim != 2 or t5_batch.attention_mask.ndim != 2:
                raise RuntimeError(
                    "Anima T5 tokenization produced invalid tensor ranks: "
                    f"ids_ndim={t5_batch.input_ids.ndim} weights_ndim={t5_batch.weights.ndim} mask_ndim={t5_batch.attention_mask.ndim}"
                )
            if t5_batch.input_ids.shape != t5_batch.weights.shape or t5_batch.input_ids.shape != t5_batch.attention_mask.shape:
                raise RuntimeError(
                    "Anima T5 tokenization ids/weights/mask shape mismatch: "
                    f"ids={tuple(t5_batch.input_ids.shape)} weights={tuple(t5_batch.weights.shape)} mask={tuple(t5_batch.attention_mask.shape)}"
                )
            if int(t5_batch.input_ids.shape[0]) != int(crossattn.shape[0]):
                raise RuntimeError(
                    "Anima conditioning batch mismatch between Qwen embeddings and T5 tokenization: "
                    f"crossattn.B={int(crossattn.shape[0])} t5.B={int(t5_batch.input_ids.shape[0])}"
                )

            target_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
            cond = {
                "crossattn": crossattn.to(device=target_device, dtype=core_dtype),
                "t5xxl_ids": t5_batch.input_ids.to(device=target_device, dtype=torch.long),
                "t5xxl_weights": t5_batch.weights.to(device=target_device, dtype=core_dtype),
                "t5xxl_attention_mask": t5_batch.attention_mask.to(device=target_device, dtype=torch.long),
            }

            if use_cache:
                self._set_cached_cond(cache_key, detach_to_cpu(cond), enabled=use_cache)
            return cond

    @torch.no_grad()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        runtime = self._require_runtime()
        prompt_text = str(prompt or "")

        qwen_tokens = runtime.text.qwen3_text.tokenize([prompt_text])
        if not (isinstance(qwen_tokens, list) and qwen_tokens and isinstance(qwen_tokens[0], list)):
            raise RuntimeError("Anima Qwen tokenizer returned invalid tokenization output for prompt length calculation.")
        qwen_len = len(qwen_tokens[0])
        qwen_default = resolve_anima_qwen_max_length()
        qwen_max = int(getattr(runtime.text.qwen3_text, "max_length", qwen_default) or qwen_default)

        t5_max = resolve_anima_t5_max_length()
        t5_batch = tokenize_t5_with_weights(
            tokenizer=runtime.text.t5_tokenizer,
            texts=[prompt_text],
            max_length=t5_max,
        )
        if t5_batch.input_ids.ndim != 2:
            raise RuntimeError(
                f"Anima T5 tokenizer returned invalid ids tensor rank for prompt length calculation: ndim={t5_batch.input_ids.ndim}"
            )
        t5_len = int(t5_batch.attention_mask[0].sum().item()) if t5_batch.attention_mask.numel() > 0 else 0

        current = max(qwen_len, t5_len)
        maximum = max(current, qwen_max, t5_max)
        return current, maximum
