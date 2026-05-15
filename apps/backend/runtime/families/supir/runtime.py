"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical SUPIR mode runtime owner for SDXL img2img/inpaint.
Builds the request-scoped SUPIR sampling runtime on top of the already-loaded SDXL engine:
- reuse the selected SDXL VAE + text conditioning instead of spawning a parallel pipeline,
- mount the SUPIR control/UNet branch request-scoped for the canonical img2img owner,
- treat the SUPIR variant checkpoint as a bounded `project_modules` overlay plus standalone control-model seam instead of a full second UNet state dict,
- run restore sampling through the backend-owned native sampler/scheduler tuple with a SUPIR-specific post-CFG restore hook,
- thread the public restore-window knob directly into that restore hook,
- optionally return pre-decoded color-fixed output so the shared image egress can stay truthful.

Symbols (top-level; keep in sync; no ghosts):
- `_conditioning_cache_metadata` (function): Publish SUPIR-specific metadata for conditioning cache hits.
- `_resolve_loaded_sdxl_checkpoint` (function): Resolve the active file-backed SDXL base checkpoint from the loaded engine bundle.
- `_resolve_supir_variant_checkpoint` (function): Resolve the selected SUPIR variant weights checkpoint path.
- `_ensure_supir_vae_seam` (function): Validate that the loaded SDXL engine exposes the required VAE seam.
- `_use_supir_vae` (function): Load and, when needed, unload the active VAE memory target for one bounded stage.
- `_encode_first_stage_with_denoise` (function): Build the Stage-1 latent/reference pair with optional denoise.
- `_build_stage1_reference` (function): Construct the SUPIR Stage-1 reference tensor from the encoded input.
- `_resolve_supir_control_transformer_depth` (function): Translate SDXL per-block transformer depth into GLVControl's per-level contract.
- `_build_supir_runtime_modules` (function): Build the request-scoped SUPIR UNet and control modules on top of the loaded SDXL base.
- `_build_restore_post_cfg` (function): Build the SUPIR restore post-CFG hook from the public restore window settings.
- `_apply_supir_sampling_session` (function): Mount the bounded SUPIR sampling session and restore engine state afterward.
- `_decode_supir_output` (function): Decode SUPIR samples back into image-space output with the selected color-fix path.
- `run_supir_img2img` (function): Execute one bounded SUPIR-mode img2img/inpaint pass through the canonical owner path.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from apps.backend.runtime.processing.datatypes import (
    ConditioningPayload,
    GenerationResult,
    PromptContext,
    SamplingPlan,
)
from apps.backend.runtime.processing.models import CodexProcessingImg2Img


def _conditioning_cache_metadata(processing: CodexProcessingImg2Img) -> dict[str, object]:
    return {
        "conditioning_cache_hit": bool(getattr(processing, "_codex_conditioning_cache_hit", False)),
        "supir_mode": True,
    }


def _resolve_loaded_sdxl_checkpoint(engine: Any) -> Path:
    bundle = getattr(engine, "_current_bundle", None)
    model_ref = str(getattr(bundle, "model_ref", "") or "").strip()
    if not model_ref:
        raise RuntimeError("SUPIR mode requires an engine with a loaded SDXL checkpoint bundle model_ref.")
    path = Path(model_ref).expanduser()
    if path.is_file():
        return path
    bundle_source = str(getattr(bundle, "source", "") or "").strip() or "unknown"
    if path.is_dir():
        raise RuntimeError(
            "SUPIR mode requires a file-backed SDXL checkpoint; "
            f"loaded bundle model_ref is a directory: {path} (bundle_source={bundle_source})"
        )
    raise RuntimeError(
        "SUPIR mode requires a file-backed SDXL checkpoint; "
        f"loaded bundle model_ref is not a file: {path} (bundle_source={bundle_source})"
    )


def _resolve_supir_variant_checkpoint(variant: Any) -> Path:
    from apps.backend.infra.config.paths import get_paths_for

    from .weights import resolve_supir_weights

    roots = [Path(path).expanduser() for path in get_paths_for("supir_models")]
    resolved = resolve_supir_weights(roots=roots, variant=variant)
    return resolved.ckpt_path


def _ensure_supir_vae_seam(engine: Any) -> Any:
    import copy

    vae = getattr(getattr(engine, "codex_objects", None), "vae", None)
    if vae is None:
        raise RuntimeError("SUPIR mode requires a loaded VAE on the active SDXL engine.")
    first_stage = getattr(vae, "first_stage_model", None)
    if first_stage is None:
        raise RuntimeError("SUPIR mode requires vae.first_stage_model on the active SDXL engine.")
    base = getattr(first_stage, "_base", first_stage)
    if not hasattr(base, "encoder"):
        raise RuntimeError("SUPIR mode requires a VAE exposing an encoder for Stage-1 precompute.")
    if not hasattr(base, "quant_conv"):
        raise RuntimeError("SUPIR mode requires a VAE exposing quant_conv for Stage-1 precompute.")
    if not hasattr(base, "denoise_encoder"):
        base.denoise_encoder = copy.deepcopy(base.encoder)
    if not hasattr(base, "denoise_encoder_s1"):
        base.denoise_encoder_s1 = base.denoise_encoder
    return vae


@contextlib.contextmanager
def _use_supir_vae(engine: Any, *, stage: str):
    from apps.backend.runtime.memory import memory_management

    if hasattr(engine, "_vae_memory_target"):
        vae_target = engine._vae_memory_target()
    else:
        vae_target = getattr(getattr(engine, "codex_objects", None), "vae", None)
    if vae_target is None:
        raise RuntimeError("SUPIR mode could not resolve the active VAE memory target.")

    memory_management.manager.load_model(
        vae_target,
        source="supir.runtime",
        stage=stage,
        component_hint="vae",
    )
    unload_vae = bool(getattr(engine, "smart_offload_enabled", False))
    try:
        yield
    finally:
        if unload_vae:
            memory_management.manager.unload_model(
                vae_target,
                source="supir.runtime",
                stage=stage,
                component_hint="vae",
            )


def _encode_first_stage_with_denoise(
    engine: Any,
    source_tensor,
    *,
    use_stage1_encoder: bool,
):
    import torch

    from apps.backend.runtime.common.vae_ldm import DiagonalGaussianDistribution

    vae = _ensure_supir_vae_seam(engine)
    first_stage = vae.first_stage_model
    base = getattr(first_stage, "_base", first_stage)
    encoder_name = "denoise_encoder_s1" if use_stage1_encoder else "denoise_encoder"
    encoder = getattr(base, encoder_name, None)
    if encoder is None:
        raise RuntimeError(f"SUPIR mode requires VAE seam '{encoder_name}'.")

    with _use_supir_vae(engine, stage="supir.stage1.precompute"):
        parameter = next(base.parameters(), None)
        if parameter is None:
            raise RuntimeError("SUPIR mode could not resolve a VAE parameter for Stage-1 precompute.")
        x = source_tensor.to(device=parameter.device, dtype=parameter.dtype)
        with torch.inference_mode():
            h = encoder(x)
            quant_conv = getattr(base, "quant_conv", None)
            moments = quant_conv(h) if quant_conv is not None else h
            posterior = DiagonalGaussianDistribution(moments)
            z = posterior.mode()
            return first_stage.process_in(z)


def _build_stage1_reference(engine: Any, source_tensor):
    import torch

    from apps.backend.runtime.processing.conditioners import decode_latent_batch, encode_image_batch

    control_latent = _encode_first_stage_with_denoise(
        engine,
        source_tensor,
        use_stage1_encoder=True,
    )
    x_stage1 = decode_latent_batch(
        engine,
        control_latent,
        target_device=control_latent.device,
        stage="supir.stage1.decode",
    )
    x_stage1_reference = x_stage1.detach().to(device="cpu", dtype=torch.float32)
    anchor_latent = encode_image_batch(
        engine,
        x_stage1,
        target_device=control_latent.device,
        stage="supir.stage1.anchor.encode",
    )
    return control_latent, anchor_latent, x_stage1_reference


def _resolve_supir_control_transformer_depth(base_config: Any) -> int | tuple[int, ...]:
    channel_mult = tuple(int(value) for value in getattr(base_config, "channel_mult", ()) or ())
    if not channel_mult:
        raise RuntimeError("SUPIR mode requires a UNet codex_config with non-empty channel_mult.")

    raw_transformer_depth = getattr(base_config, "transformer_depth", 0)
    if isinstance(raw_transformer_depth, int):
        return int(raw_transformer_depth)

    transformer_depth_values = tuple(int(value) for value in raw_transformer_depth)
    if len(transformer_depth_values) == 1:
        return int(transformer_depth_values[0])
    if len(transformer_depth_values) == len(channel_mult):
        return transformer_depth_values

    if hasattr(base_config, "expanded_num_res_blocks"):
        num_res_blocks_per_level = tuple(int(value) for value in base_config.expanded_num_res_blocks())
    else:
        raw_num_res_blocks = getattr(base_config, "num_res_blocks", ())
        if isinstance(raw_num_res_blocks, int):
            num_res_blocks_per_level = tuple(int(raw_num_res_blocks) for _ in channel_mult)
        else:
            num_res_blocks_per_level = tuple(int(value) for value in raw_num_res_blocks)
            if len(num_res_blocks_per_level) == 1:
                num_res_blocks_per_level = tuple(num_res_blocks_per_level[0] for _ in channel_mult)

    if len(num_res_blocks_per_level) != len(channel_mult):
        raise RuntimeError(
            "SUPIR mode could not resolve per-level num_res_blocks from the active SDXL codex_config: "
            f"num_res_blocks={num_res_blocks_per_level!r} channel_mult={channel_mult!r}"
        )

    total_transformer_blocks = sum(num_res_blocks_per_level)
    if len(transformer_depth_values) != total_transformer_blocks:
        raise RuntimeError(
            "SUPIR mode could not translate the active SDXL transformer_depth into GLVControl per-level form: "
            f"transformer_depth={transformer_depth_values!r} num_res_blocks={num_res_blocks_per_level!r}"
        )

    per_level_depths: list[int] = []
    offset = 0
    for level_index, block_count in enumerate(num_res_blocks_per_level):
        level_values = transformer_depth_values[offset : offset + block_count]
        offset += block_count
        if not level_values:
            raise RuntimeError(
                "SUPIR mode found an empty transformer-depth slice while translating the active SDXL codex_config: "
                f"level={level_index} num_res_blocks={num_res_blocks_per_level!r}"
            )
        first_value = int(level_values[0])
        if any(int(value) != first_value for value in level_values[1:]):
            raise RuntimeError(
                "SUPIR mode requires uniform transformer_depth within each encoder level for GLVControl: "
                f"level={level_index} values={tuple(int(value) for value in level_values)!r}"
            )
        per_level_depths.append(first_value)
    return tuple(per_level_depths)


def _build_supir_runtime_modules(*, base_diffusion_model: Any, variant_checkpoint: Path) -> tuple[Any, Any]:
    import torch

    from apps.backend.runtime.checkpoint.io import load_torch_file
    from apps.backend.runtime.models.state_dict import safe_load_state_dict, try_filter_state_dict
    from apps.backend.runtime.ops.operations import using_codex_operations

    from .nn import GLVControl, LightGLVUNet

    base_config = getattr(base_diffusion_model, "codex_config", None)
    if base_config is None:
        raise RuntimeError("SUPIR mode requires the active SDXL UNet to expose codex_config.")
    unet_kwargs = asdict(base_config)
    control_transformer_depth = _resolve_supir_control_transformer_depth(base_config)

    base_parameter = next(base_diffusion_model.parameters(), None)
    if base_parameter is None:
        raise RuntimeError("SUPIR mode could not resolve an active SDXL UNet parameter.")
    construct_device = getattr(base_diffusion_model, "load_device", base_parameter.device)
    construct_dtype = getattr(base_diffusion_model, "dtype", base_parameter.dtype)

    with using_codex_operations(device=construct_device, dtype=construct_dtype, manual_cast_enabled=True):
        supir_unet = LightGLVUNet(
            mode="XL-base",
            project_type="ZeroSFT",
            project_channel_scale=2.0,
            **unet_kwargs,
        ).to(device=construct_device, dtype=construct_dtype)
        control_model = GLVControl(
            in_channels=int(unet_kwargs["in_channels"]),
            model_channels=int(unet_kwargs["model_channels"]),
            out_channels=int(unet_kwargs["out_channels"]),
            num_res_blocks=unet_kwargs["num_res_blocks"],
            dropout=float(unet_kwargs["dropout"]),
            channel_mult=tuple(unet_kwargs["channel_mult"]),
            conv_resample=bool(unet_kwargs["conv_resample"]),
            dims=int(unet_kwargs["dims"]),
            num_classes=unet_kwargs["num_classes"],
            use_checkpoint=bool(unet_kwargs["use_checkpoint"]),
            num_heads=int(unet_kwargs["num_heads"]),
            num_head_channels=int(unet_kwargs["num_head_channels"]),
            use_scale_shift_norm=bool(unet_kwargs["use_scale_shift_norm"]),
            resblock_updown=bool(unet_kwargs["resblock_updown"]),
            use_spatial_transformer=bool(unet_kwargs["use_spatial_transformer"]),
            transformer_depth=control_transformer_depth,
            context_dim=unet_kwargs["context_dim"],
            disable_self_attentions=unet_kwargs["disable_self_attentions"],
            num_attention_blocks=unet_kwargs["num_attention_blocks"],
            disable_middle_self_attn=bool(unet_kwargs["disable_middle_self_attn"]),
            use_linear_in_transformer=bool(unet_kwargs["use_linear_in_transformer"]),
            adm_in_channels=unet_kwargs["adm_in_channels"],
            transformer_depth_middle=unet_kwargs["transformer_depth_middle"],
            input_upscale=1,
        ).to(device=construct_device, dtype=construct_dtype)

    missing_base, unexpected_base = safe_load_state_dict(
        supir_unet,
        base_diffusion_model.state_dict(),
        log_name="supir.base_unet",
        ignore_missing_prefixes=("project_modules.",),
    )
    if unexpected_base:
        raise RuntimeError(
            "SUPIR base UNet bootstrap encountered unexpected SDXL keys: "
            f"sample={unexpected_base[:10]}"
        )
    if missing_base:
        raise RuntimeError(
            "SUPIR base UNet bootstrap failed; missing non-SUPIR keys after loading the active SDXL UNet: "
            f"sample={missing_base[:10]}"
        )

    variant_state = load_torch_file(str(variant_checkpoint), safe_load=True, device="cpu")
    project_state = try_filter_state_dict(
        variant_state,
        ("model.diffusion_model.project_modules.", "diffusion_model.project_modules."),
    )
    control_state = try_filter_state_dict(variant_state, ("model.control_model.", "control_model."))

    project_first_key = next(iter(project_state.keys()), None)
    if project_first_key is None:
        raise RuntimeError(
            "SUPIR variant checkpoint is missing the SUPIR UNet adapter seam under an explicit diffusion-model keyspace."
        )
    control_first_key = next(iter(control_state.keys()), None)
    if control_first_key is None:
        raise RuntimeError(
            "SUPIR variant checkpoint is missing the SUPIR control-model seam under an explicit control-model keyspace."
        )

    missing_overlay, unexpected_overlay = safe_load_state_dict(
        supir_unet.project_modules,
        project_state,
        log_name="supir.variant_project_modules",
    )
    if missing_overlay or unexpected_overlay:
        raise RuntimeError(
            "SUPIR variant project-modules overlay failed (strict): "
            f"missing_sample={missing_overlay[:10]} unexpected_sample={unexpected_overlay[:10]}"
        )

    missing_control, unexpected_control = safe_load_state_dict(control_model, control_state, log_name="supir.control")
    if missing_control or unexpected_control:
        raise RuntimeError(
            "SUPIR control-model load failed (strict): "
            f"missing_sample={missing_control[:10]} unexpected_sample={unexpected_control[:10]}"
        )
    return supir_unet, control_model


def _build_restore_post_cfg(
    *,
    x_center,
    restoration_scale: float,
    restore_cfg_s_tmin: float,
    sigma_max: float,
):
    import torch

    if sigma_max <= 0.0:
        raise RuntimeError(f"SUPIR mode requires a positive sigma_max; got {sigma_max!r}.")

    def _sigma_scalar(value) -> float:
        if torch.is_tensor(value):
            return float(value.reshape(-1)[0].item())
        return float(value)

    def _expand_like(tensor, reference):
        if tensor.device != reference.device or tensor.dtype != reference.dtype:
            tensor = tensor.to(device=reference.device, dtype=reference.dtype)
        if int(tensor.shape[0]) == int(reference.shape[0]):
            return tensor
        if int(tensor.shape[0]) != 1:
            raise RuntimeError(
                "SUPIR restore anchor batch mismatch: "
                f"anchor_batch={int(tensor.shape[0])} runtime_batch={int(reference.shape[0])}."
            )
        return tensor.expand(reference.shape[0], *tensor.shape[1:])

    def _apply_restore(args: dict[str, object]):
        den = args.get("denoised")
        sigma = args.get("sigma")
        if den is None or sigma is None:
            raise RuntimeError("SUPIR restore hook requires `denoised` and `sigma`.")
        if restore_cfg_s_tmin > 0.0 and _sigma_scalar(sigma) <= float(restore_cfg_s_tmin):
            return den
        anchor = _expand_like(x_center, den)
        sigma_tensor = torch.as_tensor(sigma, device=den.device, dtype=den.dtype).reshape(-1, 1, 1, 1)
        if int(sigma_tensor.shape[0]) == 1 and int(den.shape[0]) != 1:
            sigma_tensor = sigma_tensor.expand(int(den.shape[0]), 1, 1, 1)
        scale = (sigma_tensor / float(sigma_max)).clamp_min(0.0).pow(float(restoration_scale))
        return den - (den - anchor) * scale

    return _apply_restore


@contextlib.contextmanager
def _apply_supir_sampling_session(
    *,
    engine: Any,
    processing: CodexProcessingImg2Img,
    control_latent,
    x_center,
    variant_checkpoint: Path,
):
    import torch

    from apps.backend.patchers.base import set_model_options_post_cfg_function
    from apps.backend.runtime.sampling.driver import CodexSampler
    from apps.backend.runtime.sampling_adapters.sampler_model import SamplerModel

    config = getattr(processing, "supir", None)
    if config is None:
        raise RuntimeError("SUPIR sampling session requires processing.supir.")

    previous_codex_objects = getattr(engine, "codex_objects", None)
    previous_original = getattr(engine, "codex_objects_original", None)
    previous_after_lora = getattr(engine, "codex_objects_after_applying_lora", None)
    if previous_codex_objects is None:
        raise RuntimeError("SUPIR mode requires engine.codex_objects on the active SDXL engine.")

    patched_codex_objects = previous_codex_objects.shallow_copy()
    patched_denoiser = previous_codex_objects.denoiser.clone()
    base_sampler_model = getattr(patched_denoiser, "model", None)
    if base_sampler_model is None:
        raise RuntimeError("SUPIR mode requires a denoiser patcher exposing a sampler model.")
    base_diffusion_model = getattr(base_sampler_model, "diffusion_model", None)
    if base_diffusion_model is None:
        raise RuntimeError("SUPIR mode requires the active denoiser to expose diffusion_model.")
    predictor = getattr(base_sampler_model, "predictor", None)
    if predictor is None:
        raise RuntimeError("SUPIR mode requires the active denoiser predictor.")
    sigma_max = float(getattr(predictor, "sigma_max", 0.0) or 0.0)
    if sigma_max <= 0.0:
        raise RuntimeError("SUPIR mode requires a predictor exposing sigma_max > 0.")

    supir_unet, control_model = _build_supir_runtime_modules(
        base_diffusion_model=base_diffusion_model,
        variant_checkpoint=variant_checkpoint,
    )

    runtime_parameter = next(supir_unet.parameters(), None)
    if runtime_parameter is None:
        raise RuntimeError("SUPIR mode could not resolve a runtime parameter on the SUPIR UNet.")
    runtime_device = runtime_parameter.device
    runtime_dtype = runtime_parameter.dtype
    control_latent = control_latent.to(device=runtime_device, dtype=runtime_dtype)
    x_center = x_center.to(device=runtime_device, dtype=runtime_dtype)

    class _SupirControlDiffusionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unet = supir_unet
            self.control_model = control_model
            self.control_latent = control_latent
            self.sigma_max = float(sigma_max)
            self.control_scale = float(config.control_scale)
            self.load_device = getattr(base_diffusion_model, "load_device", runtime_device)
            self.offload_device = getattr(base_diffusion_model, "offload_device", runtime_device)
            self.initial_device = getattr(base_diffusion_model, "initial_device", runtime_device)
            self.storage_dtype = getattr(base_diffusion_model, "storage_dtype", runtime_dtype)
            self.computation_dtype = getattr(base_diffusion_model, "computation_dtype", runtime_dtype)
            self.dtype = getattr(base_diffusion_model, "dtype", runtime_dtype)
            self.codex_config = getattr(self.unet, "codex_config", None)
            self.num_classes = getattr(self.unet, "num_classes", None)
            self._current_sigma = None

        def set_runtime_sigma(self, sigma) -> None:
            self._current_sigma = sigma

        def _resolve_control_scale(self) -> float:
            if self._current_sigma is None:
                raise RuntimeError("SUPIR control diffusion model received a forward pass without runtime sigma.")
            return float(self.control_scale)

        @torch.inference_mode()
        def forward(
            self,
            x,
            timesteps=None,
            context=None,
            y=None,
            transformer_options=None,
            **kwargs,
        ):
            del kwargs
            control = self.control_model(
                x=self.control_latent,
                timesteps=timesteps,
                xt=x,
                context=context,
                y=y,
            )
            return self.unet(
                x,
                timesteps=timesteps,
                context=context,
                y=y,
                control=control,
                control_scale=self._resolve_control_scale(),
                transformer_options=transformer_options,
            )

    class _SupirSamplerModel(SamplerModel):
        def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
            del control
            self.diffusion_model.set_runtime_sigma(t)
            return super().apply_model(
                x,
                t,
                c_concat=c_concat,
                c_crossattn=c_crossattn,
                control=None,
                transformer_options=transformer_options,
                **kwargs,
            )

    supir_diffusion_model = _SupirControlDiffusionModel()
    supir_sampler_model = _SupirSamplerModel(
        supir_diffusion_model,
        diffusers_scheduler=None,
        predictor=predictor,
        config=getattr(base_sampler_model, "config", None),
    )
    patched_denoiser.model = supir_sampler_model
    patched_denoiser.model_options = set_model_options_post_cfg_function(
        patched_denoiser.model_options,
        _build_restore_post_cfg(
            x_center=x_center,
            restoration_scale=float(config.restoration_scale),
            restore_cfg_s_tmin=float(config.restore_cfg_s_tmin),
            sigma_max=float(sigma_max),
        ),
    )
    patched_codex_objects.denoiser = patched_denoiser

    original_sampler_name = getattr(processing, "sampler_name", None)
    original_scheduler_name = getattr(processing, "scheduler", None)
    original_sampler = getattr(processing, "sampler", None)
    mapped_sampler_name = str(config.sampler.native_sampler)
    mapped_scheduler_name = str(config.sampler.native_scheduler)
    processing.sampler_name = mapped_sampler_name
    processing.scheduler = mapped_scheduler_name
    processing.sampler = CodexSampler(engine, algorithm=mapped_sampler_name)
    if hasattr(processing, "update_extra_param"):
        processing.update_extra_param("SUPIR mode", config.variant.value)
        processing.update_extra_param("SUPIR sampler", config.sampler.label)

    engine.codex_objects = patched_codex_objects
    if previous_original is not None:
        engine.codex_objects_original = patched_codex_objects
    if previous_after_lora is not None:
        engine.codex_objects_after_applying_lora = patched_codex_objects
    try:
        yield
    finally:
        engine.codex_objects = previous_codex_objects
        if previous_original is not None:
            engine.codex_objects_original = previous_original
        if previous_after_lora is not None:
            engine.codex_objects_after_applying_lora = previous_after_lora
        processing.sampler_name = original_sampler_name
        processing.scheduler = original_scheduler_name
        processing.sampler = original_sampler


def _decode_supir_output(*, engine: Any, samples, color_fix_mode: str, x_stage1_reference):
    from apps.backend.runtime.memory import memory_management
    from apps.backend.runtime.processing.conditioners import decode_latent_batch

    from .colorfix import adaptive_instance_normalization, wavelet_reconstruction

    decoded = decode_latent_batch(
        engine,
        samples,
        target_device=memory_management.manager.cpu_device,
        stage="supir.decode(final)",
    )
    mode = str(color_fix_mode or "None")
    if mode == "Wavelet":
        decoded = wavelet_reconstruction(decoded, x_stage1_reference)
    elif mode == "AdaIN":
        decoded = adaptive_instance_normalization(decoded, x_stage1_reference)
    return decoded.to(device=memory_management.manager.cpu_device, dtype=decoded.dtype)


def run_supir_img2img(
    processing: CodexProcessingImg2Img,
    *,
    plan: SamplingPlan,
    payload: ConditioningPayload,
    prompt_context: PromptContext,
    rng: Any,
    noise,
    source_tensor,
    pre_denoiser_hook=None,
    post_denoiser_hook=None,
    post_step_hook=None,
    post_sample_hook=None,
) -> GenerationResult:
    import torch

    from apps.backend.runtime.pipeline_stages.sampling_execute import execute_sampling

    from .sdxl_guard import require_sdxl_base_checkpoint

    if not isinstance(processing, CodexProcessingImg2Img):
        raise TypeError("run_supir_img2img expects CodexProcessingImg2Img")
    config = getattr(processing, "supir", None)
    if config is None:
        raise RuntimeError("SUPIR mode runtime requires processing.supir.")
    if prompt_context.loras:
        raise NotImplementedError("SUPIR mode with LoRA selections is not yet implemented")
    if not torch.is_tensor(source_tensor) or source_tensor.ndim != 4:
        raise TypeError("SUPIR mode requires source_tensor as a BCHW torch.Tensor")

    engine = getattr(processing, "sd_model", None)
    if engine is None:
        raise RuntimeError("SUPIR mode requires processing.sd_model.")

    base_checkpoint = _resolve_loaded_sdxl_checkpoint(engine)
    require_sdxl_base_checkpoint(base_checkpoint)
    variant_checkpoint = _resolve_supir_variant_checkpoint(config.variant)

    control_latent, x_center, x_stage1_reference = _build_stage1_reference(engine, source_tensor)

    mapped_sampler_name = str(config.sampler.native_sampler)
    mapped_scheduler_name = str(config.sampler.native_scheduler)
    supir_plan = replace(
        plan,
        sampler_name=mapped_sampler_name,
        scheduler_name=mapped_scheduler_name,
    )

    with _apply_supir_sampling_session(
        engine=engine,
        processing=processing,
        control_latent=control_latent,
        x_center=x_center,
        variant_checkpoint=variant_checkpoint,
    ):
        samples = execute_sampling(
            processing,
            supir_plan,
            payload,
            prompt_context,
            prompt_context.loras,
            rng=rng,
            noise=noise,
            image_conditioning=None,
            allow_txt2img_conditioning_fallback=False,
            img2img_fix_steps=False,
            init_latent=None,
            start_at_step=0,
            denoise_strength=None,
            pre_denoiser_hook=pre_denoiser_hook,
            post_denoiser_hook=post_denoiser_hook,
            post_step_hook=post_step_hook,
            post_sample_hook=post_sample_hook,
        )

    metadata = _conditioning_cache_metadata(processing)
    metadata["supir_variant"] = config.variant.value
    metadata["supir_sampler"] = config.sampler.label
    metadata["supir_native_sampler"] = mapped_sampler_name
    metadata["supir_native_scheduler"] = mapped_scheduler_name
    metadata["supir_restore_cfg_s_tmin"] = float(config.restore_cfg_s_tmin)
    decoded = None
    if str(config.color_fix) != "None":
        decoded = _decode_supir_output(
            engine=engine,
            samples=samples,
            color_fix_mode=str(config.color_fix),
            x_stage1_reference=x_stage1_reference,
        )
    return GenerationResult(
        samples=samples,
        decoded=decoded,
        metadata=metadata,
        decode_engine=engine,
    )


__all__ = ["run_supir_img2img"]
