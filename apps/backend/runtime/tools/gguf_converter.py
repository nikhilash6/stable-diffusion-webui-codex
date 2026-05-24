"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GGUF converter tool (SafeTensors → GGUF) with optional quantization, metadata injection, and verification.
Uses explicit converter profiles for supported text encoders and transformer/denoiser components, including sharded HF-style weights.

Symbols (top-level; keep in sync; no ghosts):
- `QuantizationRecipe` (enum): Public file-level quantization recipe selector.
- `QuantPolicyPreset` (enum): Quantization policy preset selector.
- `ConversionConfig` (dataclass): Conversion configuration (input/output paths, quantization choices, profile/policy selection, and regex dtype overrides).
- `ConversionProgress` (dataclass): Progress/report structure for long conversions (stage counters, timings, and status fields).
- `ConversionPreflight` (dataclass): Resolved conversion contract returned by `preflight_conversion_contract`.
- `GGUFConversionCancelled` (exception): Raised when a conversion is cancelled via a cooperative cancel signal.
- `GGUFVerificationError` (exception): Raised when a written GGUF file fails validation/verification.
- `_format_size_delta` (function): Formats underflow promotion size deltas for operator logs.
- `_log_q8_underflow_reports` (function): Emits per-tensor Q8_0 underflow promotion/retention reports.
- `_log_k_underflow_reports` (function): Emits per-tensor Q4_K/Q5_K underflow promotion/retention reports.
- `preflight_conversion_contract` (function): Resolve and validate recipe/profile/policy before heavy tensor IO.
- `convert_safetensors_to_gguf` (function): Main conversion entrypoint; reads SafeTensors (incl. sharded), quantizes tensors, writes GGUF,
  and optionally verifies the output (uses many helpers above).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from apps.backend.quantization.api import quantize_numpy
from apps.backend.quantization.gguf import (
    GGMLQuantizationType,
    GGUFWriter,
)
from apps.backend.runtime.tools import gguf_converter_metadata as _metadata
from apps.backend.runtime.tools import gguf_converter_k_underflow as _k_underflow
from apps.backend.runtime.tools import gguf_converter_profiles as _profiles
from apps.backend.runtime.tools import gguf_converter_q8_underflow as _q8_underflow
from apps.backend.runtime.tools import gguf_converter_quantization as _quantization
from apps.backend.runtime.tools import gguf_converter_safetensors_source as _safetensors_source
from apps.backend.runtime.tools import gguf_converter_tensor_planner as _tensor_planner
from apps.backend.runtime.tools import gguf_converter_verify as _verify
from apps.backend.runtime.tools.gguf_converter_specs import (
    CompiledTensorTypeRule,
    ConverterProfileSpec,
    GGUFArch,
)
from apps.backend.runtime.tools.gguf_converter_types import (
    ConversionConfig,
    ConversionProgress,
    GGUFVerificationError,
    QuantPolicyPreset,
    QuantizationRecipe,
    normalize_optional_quant_policy_preset,
    normalize_quantization_recipe,
)

logger = get_backend_logger("backend.runtime.tools.gguf_converter")


class GGUFConversionCancelled(Exception):
    """Raised when a conversion is cancelled via a cooperative cancel signal."""


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    config_path: Path
    model_config: dict[str, Any]
    profile: ConverterProfileSpec
    recipe: QuantizationRecipe
    recipe_info: _quantization.QuantizationRecipeSpec
    quant_policy_preset: QuantPolicyPreset | None
    quant_policy_id: str | None
    requested_type: GGMLQuantizationType
    dtype_rules: list[CompiledTensorTypeRule]
    arch: str
    metadata_config: dict[str, Any]
    key_mapping: dict[str, str]


def _format_size_delta(delta_bytes: int | None) -> str:
    if delta_bytes is None:
        return "unknown"
    sign = "+" if delta_bytes >= 0 else "-"
    return f"{sign}{GGUFWriter.format_n_bytes_to_str(abs(delta_bytes))}"


def _log_q8_underflow_reports(reports: tuple[_q8_underflow.Q8UnderflowReport, ...]) -> None:
    if not reports:
        return

    promoted_count = sum(1 for report in reports if report.promoted)
    retained_count = len(reports) - promoted_count
    logger.warning(
        "Q8_0 stored-scale underflow affected %d tensor(s); promoted=%d retained_q8_0=%d",
        len(reports),
        promoted_count,
        retained_count,
    )
    for report in reports:
        ratio_percent = report.affected_ratio * 100.0
        if report.promoted:
            assert report.selected_ggml_type is not None
            logger.warning(
                "Promoting Q8_0 tensor %s -> %s due to stored-scale underflow: "
                "affected=%d/%d (%.6f%%), size_delta=%s, reason=%s",
                report.gguf_name,
                report.selected_ggml_type.name,
                report.affected_blocks,
                report.total_blocks,
                ratio_percent,
                _format_size_delta(report.size_delta_bytes),
                report.reason,
            )
        else:
            logger.warning(
                "Keeping Q8_0 tensor %s with non-material stored-scale underflow: "
                "affected=%d/%d (%.6f%%), reason=%s",
                report.gguf_name,
                report.affected_blocks,
                report.total_blocks,
                ratio_percent,
                report.reason,
            )


def _log_k_underflow_reports(reports: tuple[_k_underflow.KUnderflowReport, ...]) -> None:
    if not reports:
        return

    promoted_count = sum(1 for report in reports if report.promoted)
    retained_count = len(reports) - promoted_count
    logger.warning(
        "Q4_K/Q5_K stored-scale underflow affected %d tensor(s); promoted=%d retained_k=%d",
        len(reports),
        promoted_count,
        retained_count,
    )
    for report in reports:
        ratio_percent = report.affected_ratio * 100.0
        if report.promoted:
            assert report.selected_ggml_type is not None
            logger.warning(
                "Promoting %s tensor %s -> %s due to stored-scale underflow: "
                "affected=%d/%d (%.6f%%), size_delta=%s, reason=%s",
                report.ggml_type.name,
                report.gguf_name,
                report.selected_ggml_type.name,
                report.affected_blocks,
                report.total_blocks,
                ratio_percent,
                _format_size_delta(report.size_delta_bytes),
                report.reason,
            )
        else:
            logger.warning(
                "Keeping %s tensor %s with non-material stored-scale underflow: "
                "affected=%d/%d (%.6f%%), reason=%s",
                report.ggml_type.name,
                report.gguf_name,
                report.affected_blocks,
                report.total_blocks,
                ratio_percent,
                report.reason,
            )


def preflight_conversion_contract(
    config: ConversionConfig,
    *,
    quant_policy_preset_explicit: bool = False,
) -> ConversionPreflight:
    """Resolve and validate a GGUF conversion contract before heavy tensor IO."""

    config_path = _safetensors_source.resolve_config_json_path(config.config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    profile_id = getattr(config, "profile_id", None)
    if profile_id:
        profile = _profiles.profile_by_id(str(profile_id))
        if not profile.detect(model_config):
            cfg_hint = model_config.get("_class_name") or model_config.get("architectures") or model_config.get("model_type") or "unknown"
            raise RuntimeError(f"profile_id {profile.id.value!r} does not match the selected config ({cfg_hint!r})")
    else:
        profile = _profiles.resolve_profile(model_config)

    recipe = normalize_quantization_recipe(config.quantization)
    profile.quant_policy.require_recipe_supported(recipe)
    recipe_info = _quantization.recipe_spec(recipe)

    requested_policy = normalize_optional_quant_policy_preset(getattr(config, "quant_policy_preset", None))
    if quant_policy_preset_explicit and requested_policy is None:
        allowed = ", ".join(preset.value for preset in QuantPolicyPreset)
        raise ValueError(f"Invalid quant_policy_preset None; expected one of: {allowed}")
    if recipe_info.is_float:
        if quant_policy_preset_explicit or requested_policy is not None:
            raise ValueError(f"quant_policy_preset is not accepted for float recipe {recipe.value}")
        effective_policy: QuantPolicyPreset | None = None
    elif requested_policy is not None:
        if not profile.quant_policy.supports_policy_preset(
            recipe=recipe,
            policy_preset=requested_policy,
            model_config=model_config,
        ):
            raise ValueError(
                f"quant_policy_preset {requested_policy.value!r} has no supported effect for "
                f"profile {profile.id.value!r} recipe {recipe.value!r}"
            )
        effective_policy = requested_policy
    else:
        effective_policy = profile.quant_policy.default_policy_preset(recipe=recipe, model_config=model_config)

    requested_type = recipe_info.ggml_type
    dtype_rules = profile.quant_policy.compile(
        recipe=recipe,
        policy_preset=effective_policy,
        model_config=model_config,
        user_rules=config.tensor_type_overrides,
    )
    quant_policy_id = None
    if effective_policy is not None:
        quant_policy_id = f"{profile.quant_policy.id}_{effective_policy.value.lower()}_v{profile.quant_policy.version}"

    if profile.arch is GGUFArch.LLAMA:
        arch = str(model_config.get("model_type") or "llama")
    else:
        arch = profile.arch.value

    metadata_config = (
        profile.metadata_normalizer(model_config) if profile.metadata_normalizer is not None else dict(model_config)
    )

    key_mapping: dict[str, str] = {}
    if profile.key_mapping is not None:
        key_mapping = profile.key_mapping.build(model_config)

    return ConversionPreflight(
        config_path=config_path,
        model_config=model_config,
        profile=profile,
        recipe=recipe,
        recipe_info=recipe_info,
        quant_policy_preset=effective_policy,
        quant_policy_id=quant_policy_id,
        requested_type=requested_type,
        dtype_rules=dtype_rules,
        arch=arch,
        metadata_config=metadata_config,
        key_mapping=key_mapping,
    )


def convert_safetensors_to_gguf(
    config: ConversionConfig,
    progress_callback: Optional[Callable[[ConversionProgress], None]] = None,
    *,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> str:
    """Convert a Safetensors file to GGUF format.

    Args:
        config: Conversion configuration
        progress_callback: Optional callback for progress updates

    Returns:
        Path to the output GGUF file
    """
    progress = ConversionProgress(status="loading_config")

    def update_progress():
        if progress_callback:
            progress_callback(progress)

    def check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            progress.status = "cancelled"
            progress.error = "cancelled"
            update_progress()
            raise GGUFConversionCancelled("cancelled")

    update_progress()
    check_cancel()

    preflight = preflight_conversion_contract(config)
    model_config = preflight.model_config
    config_path = preflight.config_path
    profile = preflight.profile
    requested_type = preflight.requested_type
    dtype_rules = preflight.dtype_rules
    arch = preflight.arch
    metadata_config = preflight.metadata_config
    key_mapping = preflight.key_mapping

    logger.info(
        "Loaded config: %s (profile=%s, recipe=%s, policy=%s)",
        model_config.get("_class_name") or model_config.get("model_type") or "unknown",
        profile.id.value,
        preflight.recipe.value,
        preflight.quant_policy_preset.value if preflight.quant_policy_preset is not None else "none",
    )
    check_cancel()
    
    # Load safetensors
    progress.status = "loading_weights"
    update_progress()
    check_cancel()
    
    logger.info("Loading safetensors: %s", config.safetensors_path)
    
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _safetensors_source.open_safetensors_source(config.safetensors_path) as sf:
        tensor_names = list(sf.keys())
        check_cancel()

        # Shared streaming row chunk size for pre-header analysis and tensor writes.
        chunk_rows = 1024

        plans = _tensor_planner.plan_tensors(tensor_names, sf, key_mapping, requested_type, dtype_rules)

        def note_q8_underflow_scan(plan: _tensor_planner.TensorPlan) -> None:
            progress.current_tensor = f"q8 underflow scan: {plan.gguf_name}"
            update_progress()

        promotion_result = _q8_underflow.apply_q8_underflow_promotions(
            plans,
            sf,
            check_cancel=check_cancel,
            chunk_rows=chunk_rows,
            on_scan_tensor=note_q8_underflow_scan,
        )
        plans = promotion_result.plans
        progress.current_tensor = ""
        _log_q8_underflow_reports(promotion_result.reports)

        def note_k_underflow_scan(plan: _tensor_planner.TensorPlan) -> None:
            progress.current_tensor = f"k underflow scan: {plan.gguf_name}"
            update_progress()

        k_promotion_result = _k_underflow.apply_k_underflow_promotions(
            plans,
            sf,
            check_cancel=check_cancel,
            chunk_rows=chunk_rows,
            on_scan_tensor=note_k_underflow_scan,
        )
        plans = k_promotion_result.plans
        progress.current_tensor = ""
        _log_k_underflow_reports(k_promotion_result.reports)

        progress.total_steps = len(plans)
        check_cancel()

        writer = GGUFWriter(path=str(output_path), arch=arch)
        _metadata.add_basic_metadata(
            writer,
            arch,
            metadata_config,
            preflight.recipe_info,
            quant_policy=preflight.quant_policy_id,
            quant_policy_preset=preflight.quant_policy_preset,
            config_path=config_path,
            safetensors_path=config.safetensors_path,
        )

        for plan in plans:
            raw_dtype = None if plan.ggml_type in {GGMLQuantizationType.F16, GGMLQuantizationType.F32} else plan.ggml_type
            writer.add_tensor_info(
                plan.gguf_name,
                tensor_shape=plan.stored_shape,
                tensor_dtype=plan.stored_dtype,
                tensor_nbytes=plan.stored_nbytes,
                raw_dtype=raw_dtype,
            )

        progress.status = "converting"
        update_progress()
        check_cancel()

        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()

            assert writer.fout is not None
            out = writer.fout[0]
            writer.write_padding(out, out.tell())

            def _write_bf16_bytes(tensor: torch.Tensor) -> int:
                bf16_tensor = tensor.to(torch.bfloat16).contiguous()
                bf16_bits = bf16_tensor.view(torch.uint16).contiguous()
                out.write(bf16_bits.numpy().tobytes(order="C"))
                return int(bf16_bits.numel() * 2)

            for i, plan in enumerate(plans):
                check_cancel()
                progress.current_step = i + 1
                progress.current_tensor = plan.gguf_name
                update_progress()

                bytes_written = 0

                if plan.op == "copy":
                    sl = sf.get_slice(plan.src_name)
                    shape = tuple(int(x) for x in sl.get_shape())
                    if shape != plan.raw_shape:
                        raise RuntimeError(
                            f"Tensor shape changed during conversion for {plan.src_name}: {shape} vs {plan.raw_shape}"
                        )

                    if plan.ggml_type == GGMLQuantizationType.F16:
                        target_dtype = torch.float16
                        if len(shape) == 1:
                            t = sl[:].to(target_dtype).contiguous()
                            out.write(t.numpy().tobytes(order="C"))
                            bytes_written += t.numel() * 2
                        elif len(shape) == 2:
                            rows = shape[0]
                            for start in range(0, rows, chunk_rows):
                                check_cancel()
                                chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                                out.write(chunk.numpy().tobytes(order="C"))
                                bytes_written += chunk.numel() * 2
                        else:
                            t = sf.get_tensor(plan.src_name).to(target_dtype).contiguous()
                            out.write(t.numpy().tobytes(order="C"))
                            bytes_written += t.numel() * 2

                    elif plan.ggml_type == GGMLQuantizationType.F32:
                        target_dtype = torch.float32
                        if len(shape) == 1:
                            t = sl[:].to(target_dtype).contiguous()
                            out.write(t.numpy().tobytes(order="C"))
                            bytes_written += t.numel() * 4
                        elif len(shape) == 2:
                            rows = shape[0]
                            for start in range(0, rows, chunk_rows):
                                check_cancel()
                                chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                                out.write(chunk.numpy().tobytes(order="C"))
                                bytes_written += chunk.numel() * 4
                        else:
                            t = sf.get_tensor(plan.src_name).to(target_dtype).contiguous()
                            out.write(t.numpy().tobytes(order="C"))
                            bytes_written += t.numel() * 4

                    elif plan.ggml_type == GGMLQuantizationType.BF16:
                        if len(shape) == 1:
                            bytes_written += _write_bf16_bytes(sl[:])
                        elif len(shape) == 2:
                            rows = shape[0]
                            for start in range(0, rows, chunk_rows):
                                check_cancel()
                                chunk = sl[start : min(rows, start + chunk_rows)]
                                bytes_written += _write_bf16_bytes(chunk)
                        else:
                            t = sf.get_tensor(plan.src_name)
                            bytes_written += _write_bf16_bytes(t)

                    else:
                        if len(shape) == 1:
                            # By policy we keep 1D tensors in float-like dtypes, so this indicates a planning bug.
                            raise RuntimeError(f"Unexpected quantized 1D tensor plan for {plan.src_name}: {shape}")

                        if len(shape) == 2:
                            rows = shape[0]
                            for start in range(0, rows, chunk_rows):
                                check_cancel()
                                chunk = sl[start : min(rows, start + chunk_rows)].to(torch.float32).contiguous()
                                arr = chunk.numpy()
                                try:
                                    q = quantize_numpy(arr, plan.ggml_type)
                                except Exception as exc:
                                    raise RuntimeError(
                                        f"Failed to quantize tensor {plan.src_name} to {plan.ggml_type.name}: {exc}"
                                    ) from exc
                                out.write(q.tobytes(order="C"))
                                bytes_written += q.nbytes
                        else:
                            t = sf.get_tensor(plan.src_name).to(torch.float32).contiguous()
                            arr = t.numpy()
                            try:
                                q = quantize_numpy(arr, plan.ggml_type)
                            except Exception as exc:
                                raise RuntimeError(
                                    f"Failed to quantize tensor {plan.src_name} to {plan.ggml_type.name}: {exc}"
                                ) from exc
                            out.write(q.tobytes(order="C"))
                            bytes_written += q.nbytes

                elif plan.op == "swap_halves":
                    # Swap first and second halves along dim0 (for Diffusers→BFL shift/scale reorder).
                    check_cancel()
                    t = sf.get_tensor(plan.src_name)
                    shape = tuple(t.shape)
                    if shape != plan.raw_shape:
                        raise RuntimeError(
                            f"swap_halves shape mismatch for {plan.gguf_name}: expected {plan.raw_shape}, got {shape}"
                        )
                    half = shape[0] // 2
                    if shape[0] % 2 != 0:
                        raise RuntimeError(
                            f"swap_halves requires even dim0 for {plan.gguf_name}: got {shape[0]}"
                        )
                    # Swap: [second_half, first_half]
                    first_half = t[:half]
                    second_half = t[half:]
                    swapped = torch.cat([second_half, first_half], dim=0)

                    if plan.ggml_type == GGMLQuantizationType.F16:
                        swapped = swapped.to(torch.float16).contiguous()
                        out.write(swapped.numpy().tobytes(order="C"))
                        bytes_written += swapped.numel() * 2
                    elif plan.ggml_type == GGMLQuantizationType.BF16:
                        bytes_written += _write_bf16_bytes(swapped)
                    elif plan.ggml_type == GGMLQuantizationType.F32:
                        swapped = swapped.to(torch.float32).contiguous()
                        out.write(swapped.numpy().tobytes(order="C"))
                        bytes_written += swapped.numel() * 4
                    else:
                        # Quantized: write in chunks
                        swapped = swapped.to(torch.float32).contiguous()
                        arr = swapped.numpy()
                        try:
                            q = quantize_numpy(arr, plan.ggml_type)
                        except Exception as exc:
                            raise RuntimeError(
                                f"Failed to quantize tensor {plan.gguf_name} to {plan.ggml_type.name}: {exc}"
                            ) from exc
                        out.write(q.tobytes(order="C"))
                        bytes_written += q.nbytes

                elif plan.op == "concat_dim0":
                    check_cancel()
                    if not plan.src_names:
                        raise RuntimeError(f"concat_dim0 plan has no sources for {plan.gguf_name}")

                    slices = [sf.get_slice(name) for name in plan.src_names]
                    shapes = [tuple(int(x) for x in sl.get_shape()) for sl in slices]
                    base_shape = shapes[0]
                    rank = len(base_shape)
                    if any(len(s) != rank for s in shapes[1:]):
                        raise RuntimeError(f"concat_dim0 source rank mismatch for {plan.gguf_name}: {shapes}")
                    if rank == 2:
                        trailing = base_shape[1:]
                        if any(s[1:] != trailing for s in shapes[1:]):
                            raise RuntimeError(
                                f"concat_dim0 source trailing dims mismatch for {plan.gguf_name}: {shapes}"
                            )

                    if rank == 1:
                        expected_shape = (sum(int(s[0]) for s in shapes),)
                    elif rank == 2:
                        expected_shape = (sum(int(s[0]) for s in shapes), int(base_shape[1]))
                    else:
                        raise RuntimeError(
                            f"concat_dim0 expects 1D/2D tensors for {plan.gguf_name}, got {base_shape}"
                        )

                    if expected_shape != plan.raw_shape:
                        raise RuntimeError(
                            f"concat_dim0 planned shape mismatch for {plan.gguf_name}: expected {expected_shape}, planned {plan.raw_shape}"
                        )

                    if plan.ggml_type == GGMLQuantizationType.F16:
                        target_dtype = torch.float16
                        if rank == 1:
                            for sl in slices:
                                check_cancel()
                                t = sl[:].to(target_dtype).contiguous()
                                out.write(t.numpy().tobytes(order="C"))
                                bytes_written += t.numel() * 2
                        else:
                            for sl, shape in zip(slices, shapes, strict=True):
                                rows = int(shape[0])
                                for start in range(0, rows, chunk_rows):
                                    check_cancel()
                                    chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                                    out.write(chunk.numpy().tobytes(order="C"))
                                    bytes_written += chunk.numel() * 2

                    elif plan.ggml_type == GGMLQuantizationType.BF16:
                        if rank == 1:
                            for sl in slices:
                                check_cancel()
                                bytes_written += _write_bf16_bytes(sl[:])
                        else:
                            for sl, shape in zip(slices, shapes, strict=True):
                                rows = int(shape[0])
                                for start in range(0, rows, chunk_rows):
                                    check_cancel()
                                    chunk = sl[start : min(rows, start + chunk_rows)]
                                    bytes_written += _write_bf16_bytes(chunk)

                    elif plan.ggml_type == GGMLQuantizationType.F32:
                        target_dtype = torch.float32
                        if rank == 1:
                            for sl in slices:
                                check_cancel()
                                t = sl[:].to(target_dtype).contiguous()
                                out.write(t.numpy().tobytes(order="C"))
                                bytes_written += t.numel() * 4
                        else:
                            for sl, shape in zip(slices, shapes, strict=True):
                                rows = int(shape[0])
                                for start in range(0, rows, chunk_rows):
                                    check_cancel()
                                    chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                                    out.write(chunk.numpy().tobytes(order="C"))
                                    bytes_written += chunk.numel() * 4

                    else:
                        if rank != 2:
                            raise RuntimeError(
                                f"Unexpected quantized concat_dim0 tensor plan for {plan.gguf_name}: {base_shape}"
                            )
                        for sl, shape in zip(slices, shapes, strict=True):
                            rows = int(shape[0])
                            for start in range(0, rows, chunk_rows):
                                check_cancel()
                                chunk = sl[start : min(rows, start + chunk_rows)].to(torch.float32).contiguous()
                                arr = chunk.numpy()
                                try:
                                    q = quantize_numpy(arr, plan.ggml_type)
                                except Exception as exc:
                                    raise RuntimeError(
                                        f"Failed to quantize tensor {plan.gguf_name} to {plan.ggml_type.name}: {exc}"
                                    ) from exc
                                out.write(q.tobytes(order="C"))
                                bytes_written += q.nbytes

                else:
                    raise RuntimeError(f"Unknown tensor op for {plan.gguf_name}: {plan.op!r}")

                if bytes_written != plan.stored_nbytes:
                    raise RuntimeError(
                        f"Byte count mismatch for {plan.gguf_name}: wrote {bytes_written}, expected {plan.stored_nbytes}"
                    )
                writer.write_padding(out, plan.stored_nbytes)
        finally:
            writer.close()

        logger.info("GGUF file written: %s", output_path)
        check_cancel()

        # Verification step: validate the generated file.
        #
        # Important: reuse the already-open safetensors handle from the conversion pass to avoid
        # re-opening huge WAN22 checkpoints on Windows (observed to crash sporadically in some environments).
        progress.status = "verifying"
        update_progress()
        check_cancel()

        _verify.verify_gguf_file(
            gguf_path=str(output_path),
            source_safetensors=config.safetensors_path,
            tensor_plans=plans,
            key_mapping=key_mapping,
            source_handle=sf,
        )
    
    progress.status = "complete"
    progress.current_step = progress.total_steps
    update_progress()
    
    logger.info("GGUF conversion and verification complete: %s", output_path)
    return str(output_path)


__all__ = [
    "ConversionConfig",
    "ConversionProgress", 
    "ConversionPreflight",
    "GGUFConversionCancelled",
    "QuantPolicyPreset",
    "QuantizationRecipe",
    "GGUFVerificationError",
    "convert_safetensors_to_gguf",
    "preflight_conversion_contract",
]
