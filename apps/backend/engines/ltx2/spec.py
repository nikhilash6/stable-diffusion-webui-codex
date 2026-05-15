"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LTX2 engine specification and truthful runtime assembly.
Rehydrates the loader-produced typed LTX2 bundle contract into a dedicated native runtime container, threads normalized
internal engine options into that assembly, exposes the explicit `one_stage` / `two_stage` stage helpers on the
runtime container, and lets the registered `ltx2` engine execute canonical `txt2vid` / `img2vid` without drifting
into WAN paths.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2EngineRuntime` (dataclass): Loaded LTX2 engine runtime container holding bundle inputs plus the assembled native components.
- `Ltx2EngineSpec` (dataclass): Canonical LTX2 engine spec metadata.
- `assemble_ltx2_runtime` (function): Assemble the loaded LTX2 engine runtime from a loader-produced diffusion bundle.
- `LTX2_SPEC` (constant): Canonical LTX2 engine spec instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from apps.backend.runtime.families.ltx2.model import Ltx2BundleInputs
from apps.backend.runtime.families.ltx2.runtime import (
    Ltx2NativeComponents,
    build_ltx2_request_generator,
    build_ltx2_native_components,
    decode_ltx2_stage_result,
    refine_ltx2_img2vid_two_stage,
    refine_ltx2_txt2vid_two_stage,
    require_ltx2_bundle_inputs,
    run_ltx2_img2vid,
    run_ltx2_txt2vid,
    sample_ltx2_img2vid_stage,
    sample_ltx2_txt2vid_stage,
    upsample_ltx2_two_stage_video_latents,
)
from apps.backend.runtime.model_registry.specs import ModelFamily

if TYPE_CHECKING:
    from apps.backend.core.requests import Img2VidRequest, Txt2VidRequest
    from apps.backend.runtime.families.ltx2.runtime import Ltx2RunResult
    from apps.backend.runtime.pipeline_stages.video import GeneratedAudioExportPolicy, Ltx2TwoStageGeometry
    from apps.backend.runtime.processing.datatypes import VideoPlan


@dataclass(frozen=True, slots=True)
class Ltx2EngineRuntime:
    bundle_inputs: Ltx2BundleInputs
    native: Ltx2NativeComponents
    device: str
    dtype: str

    def build_request_generator(self, *, request: Any) -> Any:
        return build_ltx2_request_generator(native=self.native, request=request)

    def run_txt2vid(
        self,
        *,
        request: "Txt2VidRequest",
        plan: "VideoPlan",
        generated_audio_export_policy: "GeneratedAudioExportPolicy",
    ) -> "Ltx2RunResult":
        return run_ltx2_txt2vid(
            native=self.native,
            request=request,
            plan=plan,
            generated_audio_export_policy=generated_audio_export_policy,
        )

    def sample_txt2vid_stage(
        self,
        *,
        request: "Txt2VidRequest",
        plan: "VideoPlan",
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        noise_scale: float = 0.0,
        latents: Any | None = None,
        audio_latents: Any | None = None,
        sigmas: Sequence[float] | None = None,
        generator: Any | None = None,
    ) -> Any:
        return sample_ltx2_txt2vid_stage(
            native=self.native,
            request=request,
            plan=plan,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            latents=latents,
            audio_latents=audio_latents,
            sigmas=sigmas,
            generator=generator,
        )

    def run_img2vid(
        self,
        *,
        request: "Img2VidRequest",
        plan: "VideoPlan",
        generated_audio_export_policy: "GeneratedAudioExportPolicy",
    ) -> "Ltx2RunResult":
        return run_ltx2_img2vid(
            native=self.native,
            request=request,
            plan=plan,
            generated_audio_export_policy=generated_audio_export_policy,
        )

    def sample_img2vid_stage(
        self,
        *,
        request: "Img2VidRequest",
        plan: "VideoPlan",
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        noise_scale: float = 0.0,
        latents: Any | None = None,
        audio_latents: Any | None = None,
        sigmas: Sequence[float] | None = None,
        generator: Any | None = None,
    ) -> Any:
        return sample_ltx2_img2vid_stage(
            native=self.native,
            request=request,
            plan=plan,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            latents=latents,
            audio_latents=audio_latents,
            sigmas=sigmas,
            generator=generator,
        )

    def upsample_two_stage_video_latents(
        self,
        *,
        request: Any,
        stage_result: Any,
        geometry: "Ltx2TwoStageGeometry",
    ) -> Any:
        return upsample_ltx2_two_stage_video_latents(
            bundle_inputs=self.bundle_inputs,
            native=self.native,
            request=request,
            stage_result=stage_result,
            geometry=geometry,
        )

    def refine_txt2vid_two_stage(
        self,
        *,
        request: "Txt2VidRequest",
        plan: "VideoPlan",
        geometry: "Ltx2TwoStageGeometry",
        upscaled_video_latents: Any,
        stage1_result: Any,
        generator: Any | None = None,
    ) -> Any:
        return refine_ltx2_txt2vid_two_stage(
            native=self.native,
            request=request,
            plan=plan,
            geometry=geometry,
            upscaled_video_latents=upscaled_video_latents,
            stage1_result=stage1_result,
            generator=generator,
        )

    def refine_img2vid_two_stage(
        self,
        *,
        request: "Img2VidRequest",
        plan: "VideoPlan",
        geometry: "Ltx2TwoStageGeometry",
        upscaled_video_latents: Any,
        stage1_result: Any,
        generator: Any | None = None,
    ) -> Any:
        return refine_ltx2_img2vid_two_stage(
            native=self.native,
            request=request,
            plan=plan,
            geometry=geometry,
            upscaled_video_latents=upscaled_video_latents,
            stage1_result=stage1_result,
            generator=generator,
        )

    def decode_stage_result(
        self,
        *,
        request: Any,
        plan: "VideoPlan",
        stage_result: Any,
        generated_audio_export_policy: "GeneratedAudioExportPolicy",
        pipeline_name: str,
        metadata_extra: Mapping[str, Any] | None = None,
    ) -> "Ltx2RunResult":
        return decode_ltx2_stage_result(
            native=self.native,
            request=request,
            plan=plan,
            stage_result=stage_result,
            generated_audio_export_policy=generated_audio_export_policy,
            pipeline_name=pipeline_name,
            metadata_extra=metadata_extra,
        )


@dataclass(frozen=True, slots=True)
class Ltx2EngineSpec:
    name: str = "ltx2"
    family: ModelFamily = ModelFamily.LTX2


def assemble_ltx2_runtime(
    *,
    spec: Ltx2EngineSpec,
    bundle,
    engine_options: Mapping[str, Any] | None = None,
) -> Ltx2EngineRuntime:
    if getattr(bundle, "family", None) is not spec.family:
        raise RuntimeError(
            f"LTX2 engine assembly expected bundle family {spec.family.value!r}, "
            f"got {getattr(getattr(bundle, 'family', None), 'value', getattr(bundle, 'family', None))!r}."
        )

    inputs = require_ltx2_bundle_inputs(bundle)
    options = dict(engine_options or {})
    raw_dtype = str(options.get("dtype", "bf16") or "bf16").strip().lower()
    if raw_dtype not in {"fp16", "bf16", "fp32"}:
        raise RuntimeError(f"LTX2 engine dtype must be one of fp16|bf16|fp32, got {raw_dtype!r}.")
    raw_device = str(options.get("device", "auto") or "auto").strip().lower()
    if raw_device not in {"auto", "cpu", "cuda"}:
        raise RuntimeError(f"LTX2 engine device must be one of auto|cpu|cuda, got {raw_device!r}.")
    native = build_ltx2_native_components(
        bundle_inputs=inputs,
        device=raw_device,
        dtype=raw_dtype,
        engine_options=options,
    )

    return Ltx2EngineRuntime(
        bundle_inputs=inputs,
        native=native,
        device=native.device_label,
        dtype=native.dtype_label,
    )


LTX2_SPEC = Ltx2EngineSpec()
