"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z Image engine (Turbo/Base variants) for txt2img/img2img.
Implements prompt formatting, conditioning, and execution for Z Image using a runtime assembled from a core transformer checkpoint + external assets
(text encoder + Flow16 VAE) with memory-manager storage/compute dtypes and vendored HF metadata under `apps/backend/huggingface/Tongyi-MAI/**`.
Sampling-path latent decode uses the shared canonical VAE memory target helper to keep manager identity aligned with engine unload cleanup.

Symbols (top-level; keep in sync; no ghosts):
- `_ZImagePromptList` (class): List-like prompt wrapper that carries per-run metadata (CFG scale, smart-cache policy, negative marker).
- `ZImageEngine` (class): `CodexDiffusionEngine` implementation for Z Image txt2img/img2img; loads/keeps runtime, formats prompts, builds conditioning,
  runs the shared txt2img pipeline, and records cache/timeline telemetry (contains nested helpers for prompt metadata and capability gating).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Any, Mapping, Optional

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.common.model_scopes import stage_scoped_model_load
from apps.backend.engines.common.prompt_wrappers import PromptListBase
from apps.backend.engines.common.runtime_lifecycle import require_runtime
from apps.backend.engines.common.tensor_tree import detach_to_cpu, move_to_device
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.models.loader import DiffusionModelBundle
from apps.backend.runtime.diagnostics.timeline import timeline_node
from apps.backend.runtime.families.zimage.debug import env_flag, env_int, truncate_text
from apps.backend.runtime.model_registry.specs import ModelFamily

from .factory import CodexZImageFactory
from .spec import ZImageEngineRuntime

logger = get_backend_logger("backend.engines.zimage.zimage")


class _ZImagePromptList(PromptListBase):
    """List-like prompt wrapper used to carry per-run metadata."""

    def __init__(
        self,
        items: list[str],
        *,
        cfg_scale: float,
        is_negative_prompt: bool,
        smart_cache: bool | None,
    ) -> None:
        self.cfg_scale = float(cfg_scale)
        super().__init__(items, is_negative_prompt=is_negative_prompt, smart_cache=smart_cache)


class ZImageEngine(CodexDiffusionEngine):
    """Z Image engine (Turbo/Base variants)."""

    engine_id = "zimage"
    expected_family = ModelFamily.ZIMAGE

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[ZImageEngineRuntime] = None
        self._device = str(memory_management.manager.mount_device())
        self._dtype = "fp32"
        self._zimage_variant: str = "turbo"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG, TaskType.IMG2IMG),
            model_types=("zimage", "zimage-turbo"),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
        )

    @property
    def zimage_variant(self) -> str:
        return self._zimage_variant

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        """Build engine components."""
        raw_variant = options.get("zimage_variant")
        variant = str(raw_variant or "").strip().lower()

        inferred: str | None = None
        try:
            # Prefer trusted Codex provenance when available. This is primarily
            # intended for GGUF files produced by our converter.
            ref = str(getattr(bundle, "model_ref", "") or "").strip()
            if ref.lower().endswith(".gguf"):
                from apps.backend.runtime.checkpoint.io import read_gguf_metadata
                from apps.backend.infra.config.provenance import CODEX_GENERATED_BY, CODEX_REPO_URL

                meta = read_gguf_metadata(ref)
                codex_repo = str(meta.get("codex.repository") or "").strip()
                codex_by = str(meta.get("codex.quantized_by") or "").strip()
                if codex_repo == CODEX_REPO_URL and codex_by == CODEX_GENERATED_BY:
                    v = str(meta.get("codex.zimage.variant") or "").strip().lower()
                    if v in {"turbo", "base"}:
                        inferred = v
        except Exception:
            inferred = None

        if not variant and inferred is not None:
            variant = inferred
        elif inferred is not None and variant != inferred:
            logger.warning(
                "Z-Image: explicit request zimage_variant=%r overrides trusted GGUF metadata variant=%r.",
                variant,
                inferred,
            )

        if variant not in {"turbo", "base"}:
            raise RuntimeError(
                "Z-Image requires an explicit variant. Provide request extras.zimage_variant='turbo'|'base' "
                "(UI: Turbo toggle), or use a Codex-produced GGUF that declares codex.zimage.variant."
            )

        self._zimage_variant = variant

        # Z-Image uses classic CFG semantics (diffusers parity): unconditional conditioning is used when guidance_scale > 1.
        # Do not mark this engine as "distilled guidance" (single-branch conditioning).
        self.use_distilled_cfg_scale = False

        flow_shift_override: float | None = None
        vendor_dir = None
        try:
            from apps.backend.infra.config.repo_root import get_repo_root
            from apps.backend.runtime.model_registry.flow_shift import flow_shift_spec_from_repo_dir

            repo_root = get_repo_root()
            hf_root = repo_root / "apps" / "backend" / "huggingface"
            repo_id = "Tongyi-MAI/Z-Image-Turbo" if variant == "turbo" else "Tongyi-MAI/Z-Image"
            vendor_dir = hf_root / repo_id.replace("/", "/")
            if not vendor_dir.is_dir():
                raise RuntimeError(
                    f"Missing vendored HF assets for Z-Image variant={variant!r}: {vendor_dir}. "
                    "Populate the directory under apps/backend/huggingface/ and retry."
                )
            flow_shift_override = float(flow_shift_spec_from_repo_dir(vendor_dir).resolve_effective_shift())
        except Exception as exc:
            raise RuntimeError(f"Failed to resolve flow_shift for Z-Image variant={variant!r}: {exc}") from exc

        logger.debug(
            "Z Image build: variant=%s flow_shift=%.3f vae_path=%s tenc_path=%s",
            variant,
            float(flow_shift_override),
            options.get("vae_path"),
            options.get("tenc_path"),
        )

        # Assemble runtime with a variant-specific spec (flow-shift affects schedule parity).
        from apps.backend.engines.zimage.spec import ZImageEngineSpec

        assembly = CodexZImageFactory(spec=ZImageEngineSpec(_flow_shift_override=flow_shift_override)).assemble(
            bundle,
            options=options,
        )
        runtime = assembly.runtime
        runtime_device = getattr(runtime, "device", None)
        if runtime_device is None:
            raise RuntimeError("Z-Image runtime contract violation: assembled runtime is missing `device`.")
        expected_core_device = str(memory_management.manager.get_device(DeviceRole.CORE))
        if str(runtime_device) != expected_core_device:
            raise RuntimeError(
                "Z-Image runtime contract violation: assembled runtime device does not match memory-manager CORE "
                f"device (runtime={runtime_device}, expected={expected_core_device})."
            )
        runtime_dtype = getattr(runtime, "core_compute_dtype", None)
        if runtime_dtype is None:
            raise RuntimeError("Z-Image runtime contract violation: assembled runtime is missing `core_compute_dtype`.")
        runtime_dtype_name = str(runtime_dtype)
        if runtime_dtype_name not in {"bf16", "fp16", "fp32"}:
            raise RuntimeError(
                "Z-Image runtime contract violation: assembled runtime has invalid `core_compute_dtype` "
                f"{runtime_dtype_name!r} (allowed: bf16, fp16, fp32)."
            )
        self._runtime = runtime
        self._device = str(runtime_device)
        self._dtype = runtime_dtype_name

        if vendor_dir is not None:
            tokenizer_dir = vendor_dir / "tokenizer"
            runtime.text.qwen3_text.text_encoder.set_tokenizer_path_hint(str(tokenizer_dir))
        logger.debug("Z Image runtime assembled")

        return assembly.codex_objects

    @property
    def required_text_encoders(self) -> tuple[str, ...]:
        """Z Image uses Qwen3 text encoder, not CLIP."""
        return ("qwen3",)

    def _on_unload(self) -> None:
        self._runtime = None

    def _require_runtime(self) -> ZImageEngineRuntime:
        return require_runtime(self._runtime, label=self.engine_id)

    def _prepare_prompt_wrappers(
        self,
        texts: list[str],
        proc: Any,
        *,
        is_negative: bool,
    ) -> _ZImagePromptList:
        raw_cfg = getattr(proc, "guidance_scale", None)
        default_scale = 1.0
        try:
            cfg_scale = float(raw_cfg) if raw_cfg is not None else default_scale
        except Exception:
            cfg_scale = default_scale
        smart_flag = getattr(proc, "smart_cache", None)
        smart_value = None if smart_flag is None else bool(smart_flag)
        return _ZImagePromptList(
            [str(t or "") for t in texts],
            cfg_scale=cfg_scale,
            is_negative_prompt=is_negative,
            smart_cache=smart_value,
        )

    @timeline_node("text_encoding", "get_learned_conditioning")
    @torch.no_grad()
    def get_learned_conditioning(self, prompts: list[str]):
        """Encode prompts using Qwen3."""
        runtime = self._require_runtime()
        qwen_patcher = self.codex_objects.text_encoders["qwen3"].patcher
        debug_dtype = env_flag("CODEX_PIPELINE_DEBUG", False) and logger.isEnabledFor(logging.DEBUG)

        texts = tuple(str(x or "") for x in prompts)
        is_negative = bool(getattr(prompts, "is_negative_prompt", False))
        smart_flag = getattr(prompts, "smart_cache", None)
        use_cache = bool(smart_flag) if smart_flag is not None else self.smart_cache_enabled
        max_length = getattr(runtime.text.qwen3_text, "max_length", None)
        cache_key = (texts, is_negative, max_length)

        cached = self._get_cached_cond(cache_key, bucket_name="zimage.conditioning", enabled=use_cache)
        if isinstance(cached, torch.Tensor):
            target_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
            core_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE)
            if debug_dtype:
                logger.debug(
                    "[zimage-dtype] conditioning cache hit: cached.dtype=%s cached.device=%s -> device=%s dtype=%s",
                    str(cached.dtype),
                    str(cached.device),
                    str(target_device),
                    str(core_dtype),
                )
            return move_to_device(cached, device=target_device, dtype=core_dtype)

        with stage_scoped_model_load(
            qwen_patcher,
            smart_offload_enabled=self.smart_offload_enabled,
            manager=memory_management.manager,
        ):
            # Per diffusers reference: pass prompts directly - the tokenizer's
            # apply_chat_template with enable_thinking=True handles formatting.
            # Do NOT add manual system prompt or <Prompt Start> markers.
            if env_flag("CODEX_ZIMAGE_DEBUG_PROMPT", False) and prompts:
                logger.info(
                    "[zimage-debug] prompt0=%s",
                    truncate_text(prompts[0], limit=env_int("CODEX_ZIMAGE_DEBUG_TEXT_MAX", 400)),
                )
            cond = runtime.text.qwen3_text(prompts)
            core_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE)
            if cond.dtype != core_dtype:
                if debug_dtype:
                    logger.debug(
                        "[zimage-dtype] conditioning cast: cond.dtype=%s -> %s",
                        str(cond.dtype),
                        str(core_dtype),
                    )
                cond = cond.to(dtype=core_dtype)
            if use_cache:
                self._set_cached_cond(cache_key, detach_to_cpu(cond), enabled=use_cache)

        raw_cfg = getattr(prompts, "cfg_scale", None)
        default_scale = 1.0
        try:
            scale_value = float(raw_cfg) if raw_cfg is not None else default_scale
        except Exception:
            scale_value = default_scale
        if env_flag("CODEX_ZIMAGE_DEBUG_PROMPT", False):
            logger.info("[zimage-debug] cfg_scale=%.3f", scale_value)

        return cond

    @torch.no_grad()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        runtime = self._require_runtime()
        tokens = runtime.text.qwen3_text.tokenize([prompt])
        length = len(tokens[0])
        return length, max(512, length)

    @timeline_node("vae", "encode_first_stage")
    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor, *, encode_seed: int | None = None) -> torch.Tensor:
        # Match Flux/Z-Image Flow16 VAE semantics:
        # - VAE wrapper expects pixel samples as BHWC in [0, 1]
        # - Latents used by the flow core must be normalized via process_in()
        return super().encode_first_stage(x, encode_seed=encode_seed)

    @timeline_node("vae", "decode_first_stage")
    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        # Match Flux/Z-Image Flow16 VAE semantics:
        # - Model operates in normalized latent space
        # - VAE decode expects denormalized latents via process_out()
        return super().decode_first_stage(x)

    @torch.inference_mode()
    def sample_with_diffusers(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 9,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
    ) -> list:
        """Run generation using Diffusers ZImagePipeline directly.
        
        This bypasses all Codex sampling and uses Diffusers scheduler exactly
        as in the reference implementation. For debugging/validation.
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt for CFG (required when guidance_scale > 1)
            height: Image height
            width: Image width  
            num_inference_steps: Sampling steps
            guidance_scale: CFG scale (classic CFG; enabled when > 1)
            seed: Random seed
        
        Returns:
            List of PIL images
        """
        from .standalone_sampler import sample_zimage_diffusers_math, decode_latents
        from PIL import Image
        import numpy as np
        
        runtime = self._require_runtime()
        
        logger.debug("[zimage] Running standalone Diffusers-math sampler")
        
        # Step 1: Encode prompt(s) using OUR working text encoder.
        prompts_list = _ZImagePromptList(
            [prompt] if isinstance(prompt, str) else list(prompt),
            cfg_scale=float(guidance_scale),
            is_negative_prompt=False,
            smart_cache=None,
        )
        cond = self.get_learned_conditioning(prompts_list)
        text_embeddings = cond["crossattn"] if isinstance(cond, dict) else cond  # [B, seq, hidden]
        negative_text_embeddings = None
        if float(guidance_scale) > 1.0:
            neg_list = _ZImagePromptList(
                [str(negative_prompt or "")],
                cfg_scale=float(guidance_scale),
                is_negative_prompt=True,
                smart_cache=None,
            )
            uncond = self.get_learned_conditioning(neg_list)
            negative_text_embeddings = uncond["crossattn"] if isinstance(uncond, dict) else uncond
        
        logger.debug("[zimage] text_embeddings: shape=%s dtype=%s", text_embeddings.shape, text_embeddings.dtype)
        
        # Step 2: Get transformer (raw model, not wrapped)
        transformer_model = runtime.denoiser.model.diffusion_model
        
        # Load transformer to GPU
        memory_management.manager.load_model(runtime.denoiser)
        
        try:
            # Step 3: Sample using Diffusers scheduler + negation
            if self._dtype == "bf16":
                computation_dtype = torch.bfloat16
            elif self._dtype == "fp16":
                computation_dtype = torch.float16
            elif self._dtype == "fp32":
                computation_dtype = torch.float32
            else:
                raise ValueError(f"Invalid Z Image dtype {self._dtype!r} (allowed: bf16, fp16, fp32)")
            shift = 3.0 if self._zimage_variant == "turbo" else 6.0
            
            if seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(seed)
            else:
                generator = None
            
            latents = sample_zimage_diffusers_math(
                transformer=transformer_model,
                text_embeddings=text_embeddings,
                negative_text_embeddings=negative_text_embeddings,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                shift=shift,
                generator=generator,
                device=self._device,
                dtype=computation_dtype,
            )
            
            logger.debug("[zimage] sampling done, latents: shape=%s dtype=%s", latents.shape, latents.dtype)
            
        finally:
            if self.smart_offload_enabled:
                memory_management.manager.unload_model(runtime.denoiser)
        
        # Step 4: Decode latents to images
        vae_target = self._vae_memory_target()
        memory_management.manager.load_model(vae_target)
        try:
            images_tensor = decode_latents(runtime.vae, latents)
            
            # Convert to PIL images
            images = []
            for i in range(images_tensor.shape[0]):
                img_np = images_tensor[i].permute(1, 2, 0).cpu().float().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                images.append(Image.fromarray(img_np))
            
            logger.debug("[zimage] decoded %d images", len(images))
            return images
            
        finally:
            if self.smart_offload_enabled:
                memory_management.manager.unload_model(vae_target)
