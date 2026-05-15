"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Bounded live diagnostics for the IP-Adapter conditioning seam.
Provides strict diagnostics request parsing plus operator-facing receipts for the
reference-image -> CLIP preprocess -> CLIP encode -> projector/resampler path
without duplicating the public generation pipeline.

Symbols (top-level; keep in sync; no ghosts):
- `IpAdapterProbeInvalidRequest` (exception): Raised for invalid diagnostics payloads.
- `IpAdapterProbeRequest` (dataclass): Normalized bounded request for the IP-Adapter probe route.
- `IpAdapterProbeTensorStats` (dataclass): Numeric summary for one tensor receipt.
- `IpAdapterProbeImageMeta` (dataclass): Metadata summary for the resolved reference image.
- `IpAdapterProbeReport` (dataclass): Structured live diagnostics response for the IP-Adapter probe route.
- `parse_ip_adapter_probe_request` (function): Validates and normalizes the bounded diagnostics request payload.
- `parse_ip_adapter_probe_report` (function): Validates and normalizes the child-process diagnostics receipt payload.
- `run_ip_adapter_probe` (function): Executes the live IP-Adapter conditioning probe and returns a structured report.
- `run_ip_adapter_probe_subprocess` (function): Runs the live probe in a child Python process so CUDA/OOM faults do not kill the API host.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Final

import numpy as np
from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection

from apps.backend.runtime.adapters.ip_adapter.assets import prepare_ip_adapter_assets_for_paths
from apps.backend.runtime.adapters.ip_adapter.preprocess import _prepare_ip_adapter_conditioning
from apps.backend.runtime.checkpoint.io import load_torch_file
from apps.backend.runtime.load_authority import LoadAuthorityStage, coordinator_load_permit
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime.vision.clip.state_dict import normalize_clip_vision_state_dict_with_layout
from apps.backend.services.media_service import MediaService

_ALLOWED_SOURCE_KINDS: Final[frozenset[str]] = frozenset({"uploaded", "path"})


class IpAdapterProbeInvalidRequest(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IpAdapterProbeRequest:
    model_path: str
    image_encoder_path: str
    source_kind: str
    reference_image_data: str | None
    reference_image_path: str | None
    crop: bool
    compare_official_encoder: bool


@dataclass(frozen=True, slots=True)
class IpAdapterProbeTensorStats:
    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    finite: bool
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    l2_norm: float | None


@dataclass(frozen=True, slots=True)
class IpAdapterProbeImageMeta:
    mode: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class IpAdapterProbeReport:
    ok: bool
    phase: str
    reason_code: str | None
    reason_detail: str | None
    model_path: str
    image_encoder_path: str
    source_kind: str
    crop: bool
    layout: str | None
    uses_hidden_states: bool | None
    slot_count: int | None
    token_count: int | None
    output_cross_attention_dim: int | None
    internal_cross_attention_dim: int | None
    encoder_variant: str | None
    encoder_hidden_size: int | None
    encoder_projection_dim: int | None
    reference_image: IpAdapterProbeImageMeta | None
    source_pixels: IpAdapterProbeTensorStats | None
    preprocessed_pixels: IpAdapterProbeTensorStats | None
    image_embeds: IpAdapterProbeTensorStats | None
    penultimate_hidden_states: IpAdapterProbeTensorStats | None
    condition_tokens: IpAdapterProbeTensorStats | None
    uncondition_tokens: IpAdapterProbeTensorStats | None
    condition_uncondition_max_abs_diff: float | None
    condition_uncondition_mean_abs_diff: float | None
    official_compare_enabled: bool
    official_compare_root: str | None
    official_preprocessed_pixels: IpAdapterProbeTensorStats | None
    official_image_embeds: IpAdapterProbeTensorStats | None
    official_penultimate_hidden_states: IpAdapterProbeTensorStats | None
    official_preprocessed_max_abs_diff: float | None
    official_preprocessed_mean_abs_diff: float | None
    official_image_embeds_max_abs_diff: float | None
    official_image_embeds_mean_abs_diff: float | None
    official_penultimate_hidden_states_max_abs_diff: float | None
    official_penultimate_hidden_states_mean_abs_diff: float | None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def parse_ip_adapter_probe_request(payload: Any) -> IpAdapterProbeRequest:
    if not isinstance(payload, dict):
        raise IpAdapterProbeInvalidRequest("payload must be a JSON object")
    allowed_keys = {
        "model_path",
        "image_encoder_path",
        "source_kind",
        "reference_image_data",
        "reference_image_path",
        "crop",
        "compare_official_encoder",
    }
    unknown_keys = sorted(str(key) for key in payload.keys() if key not in allowed_keys)
    if unknown_keys:
        raise IpAdapterProbeInvalidRequest(f"unknown payload keys: {', '.join(unknown_keys)}")
    model_path = _require_non_empty_str(payload.get("model_path"), field_name="model_path")
    image_encoder_path = _require_non_empty_str(payload.get("image_encoder_path"), field_name="image_encoder_path")
    source_kind = _require_non_empty_str(payload.get("source_kind"), field_name="source_kind").lower()
    if source_kind not in _ALLOWED_SOURCE_KINDS:
        allowed = ", ".join(sorted(_ALLOWED_SOURCE_KINDS))
        raise IpAdapterProbeInvalidRequest(f"source_kind must be one of: {allowed}")
    crop = _parse_bool(payload.get("crop", True), field_name="crop")
    reference_image_data = payload.get("reference_image_data")
    reference_image_path = payload.get("reference_image_path")
    compare_official_encoder = _parse_bool(payload.get("compare_official_encoder", False), field_name="compare_official_encoder")
    if source_kind == "uploaded":
        if not isinstance(reference_image_data, str) or not reference_image_data.strip():
            raise IpAdapterProbeInvalidRequest("reference_image_data is required when source_kind='uploaded'")
        if reference_image_path is not None:
            raise IpAdapterProbeInvalidRequest("reference_image_path is only valid when source_kind='path'")
        normalized_reference_image_data = reference_image_data.strip()
        normalized_reference_image_path = None
    else:
        if not isinstance(reference_image_path, str) or not reference_image_path.strip():
            raise IpAdapterProbeInvalidRequest("reference_image_path is required when source_kind='path'")
        if reference_image_data is not None:
            raise IpAdapterProbeInvalidRequest("reference_image_data is only valid when source_kind='uploaded'")
        normalized_reference_image_data = None
        normalized_reference_image_path = reference_image_path.strip()
    return IpAdapterProbeRequest(
        model_path=model_path,
        image_encoder_path=image_encoder_path,
        source_kind=source_kind,
        reference_image_data=normalized_reference_image_data,
        reference_image_path=normalized_reference_image_path,
        crop=crop,
        compare_official_encoder=compare_official_encoder,
    )


def run_ip_adapter_probe(request: IpAdapterProbeRequest) -> IpAdapterProbeReport:
    try:
        assets = prepare_ip_adapter_assets_for_paths(
            model_path=request.model_path,
            image_encoder_path=request.image_encoder_path,
        )
    except Exception as exc:
        return _failure_report(
            request=request,
            phase="assets",
            reason_code="E_IP_ADAPTER_PROBE_ASSET_LOAD",
            reason_detail=str(exc),
        )
    try:
        reference_image = _load_reference_image(request)
    except Exception as exc:
        return _failure_report(
            request=request,
            phase="reference_image",
            reason_code="E_IP_ADAPTER_PROBE_REFERENCE_IMAGE",
            reason_detail=str(exc),
            assets=assets,
        )
    try:
        source_pixels, processed, encoded, condition, uncondition = _prepare_ip_adapter_conditioning(
            image=reference_image,
            assets=assets,
            crop=request.crop,
        )
        max_abs_diff, mean_abs_diff = _tensor_difference(condition, uncondition)
        official_compare_root = None
        official_preprocessed_pixels = None
        official_image_embeds = None
        official_penultimate_hidden_states = None
        official_preprocessed_max_abs_diff = None
        official_preprocessed_mean_abs_diff = None
        official_image_embeds_max_abs_diff = None
        official_image_embeds_mean_abs_diff = None
        official_penultimate_hidden_states_max_abs_diff = None
        official_penultimate_hidden_states_mean_abs_diff = None
        if request.compare_official_encoder:
            try:
                (
                    official_compare_root,
                    official_preprocessed_pixels,
                    official_image_embeds,
                    official_penultimate_hidden_states,
                    (official_preprocessed_max_abs_diff, official_preprocessed_mean_abs_diff),
                    (official_image_embeds_max_abs_diff, official_image_embeds_mean_abs_diff),
                    (
                        official_penultimate_hidden_states_max_abs_diff,
                        official_penultimate_hidden_states_mean_abs_diff,
                    ),
                ) = _run_official_encoder_compare(
                    image_encoder_path=request.image_encoder_path,
                    reference_image=reference_image,
                    crop=request.crop,
                    processed=processed,
                    encoded=encoded,
                    assets=assets,
                )
            except Exception as exc:
                return _failure_report(
                    request=request,
                    phase="official_encoder_compare",
                    reason_code="E_IP_ADAPTER_PROBE_OFFICIAL_ENCODER_COMPARE",
                    reason_detail=str(exc),
                    assets=assets,
                    reference_image=reference_image,
                    source_pixels=source_pixels,
                )
        return IpAdapterProbeReport(
            ok=True,
            phase="complete",
            reason_code=None,
            reason_detail=None,
            model_path=request.model_path,
            image_encoder_path=request.image_encoder_path,
            source_kind=request.source_kind,
            crop=bool(request.crop),
            layout=assets.layout.value,
            uses_hidden_states=bool(assets.uses_hidden_states),
            slot_count=int(assets.slot_count),
            token_count=int(assets.token_count),
            output_cross_attention_dim=int(assets.output_cross_attention_dim),
            internal_cross_attention_dim=int(assets.internal_cross_attention_dim),
            encoder_variant=assets.image_encoder_runtime.spec.variant.value,
            encoder_hidden_size=int(assets.image_encoder_runtime.spec.hidden_size),
            encoder_projection_dim=int(assets.image_encoder_runtime.spec.projection_dim),
            reference_image=IpAdapterProbeImageMeta(
                mode=str(reference_image.mode),
                width=int(reference_image.width),
                height=int(reference_image.height),
            ),
            source_pixels=_tensor_stats(source_pixels),
            preprocessed_pixels=_tensor_stats(processed),
            image_embeds=_tensor_stats(encoded.image_embeds),
            penultimate_hidden_states=_tensor_stats(encoded.penultimate_hidden_states),
            condition_tokens=_tensor_stats(condition),
            uncondition_tokens=_tensor_stats(uncondition),
            condition_uncondition_max_abs_diff=max_abs_diff,
            condition_uncondition_mean_abs_diff=mean_abs_diff,
            official_compare_enabled=bool(request.compare_official_encoder),
            official_compare_root=official_compare_root,
            official_preprocessed_pixels=_tensor_stats(official_preprocessed_pixels),
            official_image_embeds=_tensor_stats(official_image_embeds),
            official_penultimate_hidden_states=_tensor_stats(official_penultimate_hidden_states),
            official_preprocessed_max_abs_diff=official_preprocessed_max_abs_diff,
            official_preprocessed_mean_abs_diff=official_preprocessed_mean_abs_diff,
            official_image_embeds_max_abs_diff=official_image_embeds_max_abs_diff,
            official_image_embeds_mean_abs_diff=official_image_embeds_mean_abs_diff,
            official_penultimate_hidden_states_max_abs_diff=official_penultimate_hidden_states_max_abs_diff,
            official_penultimate_hidden_states_mean_abs_diff=official_penultimate_hidden_states_mean_abs_diff,
        )
    except Exception as exc:
        return _failure_report(
            request=request,
            phase="conditioning",
            reason_code="E_IP_ADAPTER_PROBE_CONDITIONING",
            reason_detail=str(exc),
            assets=assets,
            reference_image=reference_image,
            source_pixels=locals().get("source_pixels"),
        )


def run_ip_adapter_probe_subprocess(
    request: IpAdapterProbeRequest,
    *,
    timeout_seconds: int = 600,
) -> IpAdapterProbeReport:
    payload = {
        "model_path": request.model_path,
        "image_encoder_path": request.image_encoder_path,
        "source_kind": request.source_kind,
        "reference_image_data": request.reference_image_data,
        "reference_image_path": request.reference_image_path,
        "crop": request.crop,
        "compare_official_encoder": request.compare_official_encoder,
    }
    repo_root = Path(__file__).resolve().parents[5]
    env = dict(os.environ)
    env.setdefault("CODEX_ROOT", str(repo_root))
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [part for part in existing_pythonpath.split(os.pathsep) if part]
    if str(repo_root) not in pythonpath_parts:
        pythonpath_parts.insert(0, str(repo_root))
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    command = [sys.executable, "-m", "apps.backend.runtime.adapters.ip_adapter.probe", "--child"]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=int(timeout_seconds),
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return _failure_report(
            request=request,
            phase="subprocess",
            reason_code="E_IP_ADAPTER_PROBE_TIMEOUT",
            reason_detail=f"probe child timed out after {int(timeout_seconds)}s",
        )
    except Exception as exc:
        return _failure_report(
            request=request,
            phase="subprocess",
            reason_code="E_IP_ADAPTER_PROBE_SUBPROCESS",
            reason_detail=str(exc),
        )
    stderr_tail = (completed.stderr or "").strip().splitlines()[-20:]
    stderr_detail = " | ".join(stderr_tail)
    if completed.returncode != 0:
        detail = f"probe child exited with code {completed.returncode}"
        if stderr_detail:
            detail = f"{detail}: {stderr_detail}"
        return _failure_report(
            request=request,
            phase="subprocess",
            reason_code="E_IP_ADAPTER_PROBE_SUBPROCESS",
            reason_detail=detail,
        )
    try:
        output = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = f"probe child returned invalid JSON: {exc}"
        if stderr_detail:
            detail = f"{detail}: {stderr_detail}"
        return _failure_report(
            request=request,
            phase="subprocess",
            reason_code="E_IP_ADAPTER_PROBE_SUBPROCESS_OUTPUT",
            reason_detail=detail,
        )
    try:
        normalized_output = parse_ip_adapter_probe_report(output)
    except Exception as exc:
        detail = f"probe child returned invalid report: {exc}"
        if stderr_detail:
            detail = f"{detail}: {stderr_detail}"
        return _failure_report(
            request=request,
            phase="subprocess",
            reason_code="E_IP_ADAPTER_PROBE_SUBPROCESS_OUTPUT",
            reason_detail=detail,
        )
    return normalized_output


def _load_reference_image(request: IpAdapterProbeRequest) -> Image.Image:
    if request.source_kind == "uploaded":
        return MediaService().decode_image(request.reference_image_data).convert("RGB")
    assert request.reference_image_path is not None
    image_path = Path(request.reference_image_path)
    if not image_path.is_file():
        raise RuntimeError(f"reference image path does not exist: {request.reference_image_path!r}")
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _image_to_bhwc_tensor(image: Image.Image) -> torch.Tensor:
    rgb_image = image.convert("RGB")
    array = np.asarray(rgb_image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(f"IP-Adapter probe reference image must decode to HWC RGB; got shape={array.shape}.")
    return torch.from_numpy(array / 255.0).unsqueeze(0)


def _tensor_stats(tensor: torch.Tensor | None) -> IpAdapterProbeTensorStats | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    detached = tensor.detach()
    working = detached.float().cpu()
    numel = int(working.numel())
    finite = bool(torch.isfinite(working).all().item()) if numel > 0 else True
    if numel == 0:
        minimum = maximum = mean = std = l2_norm = None
    else:
        minimum = float(working.min().item())
        maximum = float(working.max().item())
        mean = float(working.mean().item())
        std = float(working.std(unbiased=False).item()) if numel > 1 else 0.0
        l2_norm = float(torch.linalg.vector_norm(working).item())
    return IpAdapterProbeTensorStats(
        shape=tuple(int(dim) for dim in detached.shape),
        dtype=str(detached.dtype),
        device=str(detached.device),
        numel=numel,
        finite=finite,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        l2_norm=l2_norm,
    )


def _tensor_difference(left: torch.Tensor, right: torch.Tensor) -> tuple[float | None, float | None]:
    left_cpu = left.detach().float().cpu()
    right_cpu = right.detach().float().cpu()
    if left_cpu.shape != right_cpu.shape:
        raise RuntimeError(
            "IP-Adapter probe expected condition/uncondition tensors to share the same shape; "
            f"got {tuple(left_cpu.shape)} vs {tuple(right_cpu.shape)}."
        )
    delta = (left_cpu - right_cpu).abs()
    if delta.numel() == 0:
        return None, None
    return float(delta.max().item()), float(delta.mean().item())


def _run_official_encoder_compare(
    *,
    image_encoder_path: str,
    reference_image: Image.Image,
    crop: bool,
    processed: torch.Tensor,
    encoded,
    assets,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, tuple[float | None, float | None], tuple[float | None, float | None], tuple[float | None, float | None]]:
    encoder_root = _resolve_official_encoder_root(image_encoder_path)
    processor = CLIPImageProcessor.from_pretrained(encoder_root)
    image_size = int(assets.image_encoder_runtime.spec.preprocess.image_size)
    try:
        memory_management.manager.unload_model(
            assets.image_encoder_runtime.patcher,
            source="runtime.adapters.ip_adapter.probe.official_encoder_compare",
            stage="official_encoder_compare",
            component_hint="ClipVisionEncoder",
            event_reason="free_current_encoder_before_official_compare",
        )
    except Exception:
        pass
    config = CLIPVisionConfig.from_pretrained(encoder_root)
    runtime_device = assets.image_encoder_runtime.load_device
    runtime_dtype = assets.image_encoder_runtime.runtime_dtype
    official_model = CLIPVisionModelWithProjection(config).to(device=runtime_device, dtype=runtime_dtype)
    official_model.eval()
    raw_state = load_torch_file(
        image_encoder_path,
        safe_load=True,
        device=memory_management.manager.get_offload_device(DeviceRole.TEXT_ENCODER),
    )
    normalized_state, resolved_layout = normalize_clip_vision_state_dict_with_layout(raw_state)
    missing, unexpected = safe_load_state_dict(
        official_model,
        normalized_state,
        log_name="IPAdapterProbeOfficialClipVision",
    )
    if missing or unexpected:
        raise RuntimeError(
            "Official CLIP vision compare load mismatch: "
            f"source_style={resolved_layout.source_style} "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}."
        )
    official_processed = processor(
        images=[reference_image],
        do_center_crop=bool(crop),
        size={"shortest_edge": image_size} if crop else {"height": image_size, "width": image_size},
        crop_size={"height": image_size, "width": image_size},
        return_tensors="pt",
    ).pixel_values.to(device=runtime_device, dtype=runtime_dtype)
    with torch.inference_mode():
        outputs = official_model(
            pixel_values=official_processed,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states or ()
    if len(hidden_states) < 1:
        raise RuntimeError("Official CLIP vision compare did not return hidden states.")
    penultimate_index = -2 if len(hidden_states) >= 2 else -1
    official_image_embeds = outputs.image_embeds.detach().float().cpu()
    official_penultimate = hidden_states[penultimate_index].detach().float().cpu()
    official_processed_cpu = official_processed.detach().float().cpu()
    processed_diff = _tensor_difference(processed, official_processed_cpu)
    image_embeds_diff = _tensor_difference(encoded.image_embeds, official_image_embeds)
    penultimate_diff = _tensor_difference(encoded.penultimate_hidden_states, official_penultimate)
    return (
        str(encoder_root),
        official_processed_cpu,
        official_image_embeds,
        official_penultimate,
        processed_diff,
        image_embeds_diff,
        penultimate_diff,
    )


def _resolve_official_encoder_root(image_encoder_path: str) -> str:
    path = Path(image_encoder_path)
    root = path if path.is_dir() else path.parent
    config_path = root / "config.json"
    preprocessor_path = root / "preprocessor_config.json"
    if not config_path.is_file():
        raise RuntimeError(
            "Official CLIP vision compare requires a directory root with config.json; "
            f"derived root '{root}' from '{image_encoder_path}'."
        )
    if not preprocessor_path.is_file():
        raise RuntimeError(
            "Official CLIP vision compare requires preprocessor_config.json beside config.json; "
            f"derived root '{root}' from '{image_encoder_path}'."
        )
    return str(root)


def _failure_report(
    *,
    request: IpAdapterProbeRequest,
    phase: str,
    reason_code: str,
    reason_detail: str,
    assets: Any | None = None,
    reference_image: Image.Image | None = None,
    source_pixels: torch.Tensor | None = None,
) -> IpAdapterProbeReport:
    image_meta = None
    if isinstance(reference_image, Image.Image):
        image_meta = IpAdapterProbeImageMeta(
            mode=str(reference_image.mode),
            width=int(reference_image.width),
            height=int(reference_image.height),
        )
    return IpAdapterProbeReport(
        ok=False,
        phase=str(phase),
        reason_code=str(reason_code),
        reason_detail=str(reason_detail),
        model_path=request.model_path,
        image_encoder_path=request.image_encoder_path,
        source_kind=request.source_kind,
        crop=bool(request.crop),
        layout=getattr(getattr(assets, "layout", None), "value", None),
        uses_hidden_states=getattr(assets, "uses_hidden_states", None),
        slot_count=getattr(assets, "slot_count", None),
        token_count=getattr(assets, "token_count", None),
        output_cross_attention_dim=getattr(assets, "output_cross_attention_dim", None),
        internal_cross_attention_dim=getattr(assets, "internal_cross_attention_dim", None),
        encoder_variant=getattr(getattr(getattr(assets, "image_encoder_runtime", None), "spec", None), "variant", None).value
        if getattr(getattr(getattr(assets, "image_encoder_runtime", None), "spec", None), "variant", None) is not None
        else None,
        encoder_hidden_size=getattr(getattr(getattr(assets, "image_encoder_runtime", None), "spec", None), "hidden_size", None),
        encoder_projection_dim=getattr(getattr(getattr(assets, "image_encoder_runtime", None), "spec", None), "projection_dim", None),
        reference_image=image_meta,
        source_pixels=_tensor_stats(source_pixels),
        preprocessed_pixels=None,
        image_embeds=None,
        penultimate_hidden_states=None,
        condition_tokens=None,
        uncondition_tokens=None,
        condition_uncondition_max_abs_diff=None,
        condition_uncondition_mean_abs_diff=None,
        official_compare_enabled=bool(request.compare_official_encoder),
        official_compare_root=None,
        official_preprocessed_pixels=None,
        official_image_embeds=None,
        official_penultimate_hidden_states=None,
        official_preprocessed_max_abs_diff=None,
        official_preprocessed_mean_abs_diff=None,
        official_image_embeds_max_abs_diff=None,
        official_image_embeds_mean_abs_diff=None,
        official_penultimate_hidden_states_max_abs_diff=None,
        official_penultimate_hidden_states_mean_abs_diff=None,
    )


def _require_non_empty_str(raw: Any, *, field_name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise IpAdapterProbeInvalidRequest(f"{field_name} must be a non-empty string")
    return raw.strip()


def _parse_bool(raw: Any, *, field_name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    raise IpAdapterProbeInvalidRequest(f"{field_name} must be a boolean")


def parse_ip_adapter_probe_report(payload: Any) -> IpAdapterProbeReport:
    if not isinstance(payload, dict):
        raise IpAdapterProbeInvalidRequest("probe report payload must be an object")
    return IpAdapterProbeReport(
        ok=bool(payload.get("ok")),
        phase=str(payload.get("phase")),
        reason_code=payload.get("reason_code"),
        reason_detail=payload.get("reason_detail"),
        model_path=str(payload.get("model_path")),
        image_encoder_path=str(payload.get("image_encoder_path")),
        source_kind=str(payload.get("source_kind")),
        crop=bool(payload.get("crop")),
        layout=payload.get("layout"),
        uses_hidden_states=payload.get("uses_hidden_states"),
        slot_count=payload.get("slot_count"),
        token_count=payload.get("token_count"),
        output_cross_attention_dim=payload.get("output_cross_attention_dim"),
        internal_cross_attention_dim=payload.get("internal_cross_attention_dim"),
        encoder_variant=payload.get("encoder_variant"),
        encoder_hidden_size=payload.get("encoder_hidden_size"),
        encoder_projection_dim=payload.get("encoder_projection_dim"),
        reference_image=_parse_image_meta(payload.get("reference_image")),
        source_pixels=_parse_tensor_stats(payload.get("source_pixels")),
        preprocessed_pixels=_parse_tensor_stats(payload.get("preprocessed_pixels")),
        image_embeds=_parse_tensor_stats(payload.get("image_embeds")),
        penultimate_hidden_states=_parse_tensor_stats(payload.get("penultimate_hidden_states")),
        condition_tokens=_parse_tensor_stats(payload.get("condition_tokens")),
        uncondition_tokens=_parse_tensor_stats(payload.get("uncondition_tokens")),
        condition_uncondition_max_abs_diff=payload.get("condition_uncondition_max_abs_diff"),
        condition_uncondition_mean_abs_diff=payload.get("condition_uncondition_mean_abs_diff"),
        official_compare_enabled=bool(payload.get("official_compare_enabled", False)),
        official_compare_root=payload.get("official_compare_root"),
        official_preprocessed_pixels=_parse_tensor_stats(payload.get("official_preprocessed_pixels")),
        official_image_embeds=_parse_tensor_stats(payload.get("official_image_embeds")),
        official_penultimate_hidden_states=_parse_tensor_stats(payload.get("official_penultimate_hidden_states")),
        official_preprocessed_max_abs_diff=_optional_float(payload.get("official_preprocessed_max_abs_diff")),
        official_preprocessed_mean_abs_diff=_optional_float(payload.get("official_preprocessed_mean_abs_diff")),
        official_image_embeds_max_abs_diff=_optional_float(payload.get("official_image_embeds_max_abs_diff")),
        official_image_embeds_mean_abs_diff=_optional_float(payload.get("official_image_embeds_mean_abs_diff")),
        official_penultimate_hidden_states_max_abs_diff=_optional_float(payload.get("official_penultimate_hidden_states_max_abs_diff")),
        official_penultimate_hidden_states_mean_abs_diff=_optional_float(payload.get("official_penultimate_hidden_states_mean_abs_diff")),
    )


def _parse_image_meta(payload: Any) -> IpAdapterProbeImageMeta | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise IpAdapterProbeInvalidRequest("probe image meta payload must be an object")
    return IpAdapterProbeImageMeta(
        mode=str(payload.get("mode")),
        width=int(payload.get("width")),
        height=int(payload.get("height")),
    )


def _parse_tensor_stats(payload: Any) -> IpAdapterProbeTensorStats | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise IpAdapterProbeInvalidRequest("probe tensor stats payload must be an object")
    return IpAdapterProbeTensorStats(
        shape=tuple(int(dim) for dim in payload.get("shape", ())),
        dtype=str(payload.get("dtype")),
        device=str(payload.get("device")),
        numel=int(payload.get("numel")),
        finite=bool(payload.get("finite")),
        minimum=_optional_float(payload.get("minimum")),
        maximum=_optional_float(payload.get("maximum")),
        mean=_optional_float(payload.get("mean")),
        std=_optional_float(payload.get("std")),
        l2_norm=_optional_float(payload.get("l2_norm")),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] != "--child":
        raise SystemExit("Use `python -m apps.backend.runtime.adapters.ip_adapter.probe --child`.")
    request = parse_ip_adapter_probe_request(json.load(sys.stdin))
    with coordinator_load_permit(
        owner="api.tests.ip_adapter.probe",
        stage=LoadAuthorityStage.MATERIALIZE,
    ):
        report = run_ip_adapter_probe(request)
    json.dump(report.to_payload(), sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
