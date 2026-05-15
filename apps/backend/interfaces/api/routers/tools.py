"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tools API routes (GGUF conversion + SafeTensors merge + file browser + PNG metadata inspection).
Provides long-running conversion/merge job tracking, filesystem browsing for file picker dialogs, and small utility endpoints used by the UI.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for tools endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from io import BytesIO
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Final

from fastapi import APIRouter, Body, File, HTTPException, UploadFile

from apps.backend.core.strict_values import parse_bool_value


def build_router(*, codex_root: Path) -> APIRouter:
    router = APIRouter()

    class _ToolJobStatus(str, Enum):
        PENDING = "pending"
        LOADING_CONFIG = "loading_config"
        LOADING_WEIGHTS = "loading_weights"
        CONVERTING = "converting"
        VERIFYING = "verifying"
        MERGING_SAFETENSORS = "merging_safetensors"
        FINALIZING = "finalizing"
        CANCELLING = "cancelling"
        CANCELLED = "cancelled"
        COMPLETE = "complete"
        ERROR = "error"

    _TERMINAL_JOB_STATUSES: Final[set[_ToolJobStatus]] = {
        _ToolJobStatus.COMPLETE,
        _ToolJobStatus.ERROR,
        _ToolJobStatus.CANCELLED,
    }
    _ALLOWED_JOB_STATUS_TRANSITIONS: Final[dict[_ToolJobStatus, set[_ToolJobStatus]]] = {
        _ToolJobStatus.PENDING: {
            _ToolJobStatus.LOADING_CONFIG,
            _ToolJobStatus.LOADING_WEIGHTS,
            _ToolJobStatus.MERGING_SAFETENSORS,
            _ToolJobStatus.FINALIZING,
            _ToolJobStatus.CANCELLING,
            _ToolJobStatus.CANCELLED,
            _ToolJobStatus.ERROR,
        },
        _ToolJobStatus.LOADING_CONFIG: {
            _ToolJobStatus.LOADING_WEIGHTS,
            _ToolJobStatus.CANCELLING,
            _ToolJobStatus.CANCELLED,
            _ToolJobStatus.ERROR,
        },
        _ToolJobStatus.LOADING_WEIGHTS: {
            _ToolJobStatus.MERGING_SAFETENSORS,
            _ToolJobStatus.CONVERTING,
            _ToolJobStatus.CANCELLING,
            _ToolJobStatus.CANCELLED,
            _ToolJobStatus.ERROR,
        },
        _ToolJobStatus.CONVERTING: {
            _ToolJobStatus.VERIFYING,
            _ToolJobStatus.FINALIZING,
            _ToolJobStatus.CANCELLING,
            _ToolJobStatus.CANCELLED,
            _ToolJobStatus.ERROR,
        },
        _ToolJobStatus.VERIFYING: {
            _ToolJobStatus.FINALIZING,
            _ToolJobStatus.CANCELLING,
            _ToolJobStatus.CANCELLED,
            _ToolJobStatus.ERROR,
        },
        _ToolJobStatus.MERGING_SAFETENSORS: {
            _ToolJobStatus.FINALIZING,
            _ToolJobStatus.CANCELLING,
            _ToolJobStatus.CANCELLED,
            _ToolJobStatus.ERROR,
        },
        _ToolJobStatus.FINALIZING: {
            _ToolJobStatus.CANCELLING,
            _ToolJobStatus.COMPLETE,
            _ToolJobStatus.CANCELLED,
            _ToolJobStatus.ERROR,
        },
        _ToolJobStatus.CANCELLING: {
            _ToolJobStatus.LOADING_CONFIG,
            _ToolJobStatus.LOADING_WEIGHTS,
            _ToolJobStatus.CONVERTING,
            _ToolJobStatus.VERIFYING,
            _ToolJobStatus.MERGING_SAFETENSORS,
            _ToolJobStatus.FINALIZING,
            _ToolJobStatus.COMPLETE,
            _ToolJobStatus.CANCELLED,
            _ToolJobStatus.ERROR,
        },
        _ToolJobStatus.CANCELLED: set(),
        _ToolJobStatus.COMPLETE: set(),
        _ToolJobStatus.ERROR: set(),
    }

    @dataclass(slots=True)
    class _ToolJobState:
        status: _ToolJobStatus
        progress: float
        current_tensor: str
        error: str | None
        output_path: str

        def to_payload(self) -> Dict[str, Any]:
            return {
                "status": self.status.value,
                "progress": self.progress,
                "current_tensor": self.current_tensor,
                "error": self.error,
                "output_path": self.output_path,
            }

    @dataclass(slots=True)
    class _GgufConversionControl:
        cancel_event: threading.Event
        tmp_path: Path
        final_path: Path

    @dataclass(slots=True)
    class _SafetensorsMergeControl:
        source_path: Path
        tmp_path: Path
        final_path: Path

    _gguf_conversion_jobs: Dict[str, _ToolJobState] = {}
    _gguf_conversion_controls: Dict[str, _GgufConversionControl] = {}
    _safetensors_merge_jobs: Dict[str, _ToolJobState] = {}
    _safetensors_merge_controls: Dict[str, _SafetensorsMergeControl] = {}
    _job_state_lock = threading.RLock()
    _UNSET = object()

    def _normalize_job_status(raw: _ToolJobStatus | str) -> _ToolJobStatus:
        if isinstance(raw, _ToolJobStatus):
            return raw
        text = str(raw or "").strip()
        try:
            return _ToolJobStatus(text)
        except ValueError as exc:
            raise RuntimeError(f"Unknown tool job status: {text!r}") from exc

    def _set_job_state(
        job: _ToolJobState,
        *,
        status: _ToolJobStatus | str | None = None,
        progress: float | None = None,
        current_tensor: str | None = None,
        error: object = _UNSET,
    ) -> None:
        if status is not None:
            next_status = _normalize_job_status(status)
            if next_status != job.status:
                allowed = _ALLOWED_JOB_STATUS_TRANSITIONS.get(job.status, set())
                if next_status not in allowed:
                    raise RuntimeError(f"Invalid tool-job status transition: {job.status.value} -> {next_status.value}")
            job.status = next_status
        if progress is not None:
            job.progress = float(progress)
        if current_tensor is not None:
            job.current_tensor = str(current_tensor)
        if error is not _UNSET:
            job.error = None if error is None else str(error)

    def _alloc_nonexistent_path(parent: Path, *, prefix: str, suffix: str) -> Path:
        for _ in range(64):
            token = uuid.uuid4().hex[:10]
            candidate = parent / f"{prefix}{token}{suffix}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Failed to allocate a unique output path under: {str(parent)!r}")

    @router.get("/api/tools/gguf-converter/presets")
    async def list_gguf_converter_presets() -> Dict[str, Any]:
        from apps.backend.runtime.tools.gguf_converter_float_groups import float_groups_for_profile_id
        from apps.backend.runtime.tools.gguf_converter_model_metadata import list_vendored_gguf_converter_model_metadata

        models = list_vendored_gguf_converter_model_metadata(codex_root=codex_root)
        profile_ids: set[str] = set()
        for model in models:
            for comp in model.components:
                if comp.profile_id:
                    profile_ids.add(comp.profile_id)

        float_groups: dict[str, Any] = {}
        for pid in sorted(profile_ids):
            float_groups[pid] = [
                {"id": g.id, "label": g.label, "patterns": list(g.patterns)}
                for g in float_groups_for_profile_id(pid)
            ]

        return {
            "models": [
                {
                    "id": m.id,
                    "label": m.label,
                    "org": m.org,
                    "repo": m.repo,
                    "components": [
                        {
                            "id": c.id,
                            "label": c.label,
                            "config_dir": c.config_dir,
                            "kind": c.kind,
                            "profile_id": c.profile_id,
                        }
                        for c in m.components
                    ],
                }
                for m in models
            ],
            "float_groups": float_groups,
        }

    @router.post("/api/tools/convert-gguf")
    async def convert_to_gguf(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Start a GGUF conversion job."""
        from apps.backend.runtime.tools.gguf_converter import (
            ConversionConfig,
            GGUFConversionCancelled,
            QuantizationType,
            convert_safetensors_to_gguf,
        )
        from apps.backend.runtime.tools.gguf_converter_types import (
            normalize_mixed_float_override,
            normalize_precision_mode,
        )

        job_id = str(uuid.uuid4())[:8]

        allowed_payload_keys = {
            "config_path",
            "safetensors_path",
            "output_path",
            "overwrite",
            "quantization",
            "tensor_type_overrides",
            "profile_id",
            "float_group_overrides",
            "precision_mode",
        }
        unknown_keys = sorted({str(key) for key in payload.keys() if str(key) not in allowed_payload_keys})
        if unknown_keys:
            allowed_msg = ", ".join(sorted(allowed_payload_keys))
            unknown_msg = ", ".join(repr(key) for key in unknown_keys)
            raise HTTPException(
                status_code=400,
                detail=f"Unknown /api/tools/convert-gguf payload key(s): {unknown_msg} (allowed: {allowed_msg})",
            )

        # Validate paths
        config_path = payload.get("config_path", "")
        safetensors_path = payload.get("safetensors_path", "")
        output_path = payload.get("output_path", "")
        try:
            overwrite = parse_bool_value(payload.get("overwrite"), field="overwrite", default=False)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        quant_str = payload.get("quantization", "F16")
        overrides_raw = payload.get("tensor_type_overrides", [])
        profile_id_raw = payload.get("profile_id", None)
        float_group_overrides_raw = payload.get("float_group_overrides", {})
        precision_mode_raw = payload.get("precision_mode", None)

        if not config_path or not safetensors_path or not output_path:
            raise HTTPException(status_code=400, detail="Missing required paths")

        if not os.path.exists(config_path) and not os.path.exists(os.path.join(config_path, "config.json")):
            raise HTTPException(status_code=400, detail=f"Config not found: {config_path}")

        if not os.path.exists(safetensors_path):
            raise HTTPException(status_code=400, detail=f"Safetensors not found: {safetensors_path}")

        final_path = Path(os.path.expanduser(str(output_path))).resolve()

        if final_path.name.lower().endswith(".codexpack.gguf"):
            raise HTTPException(
                status_code=400,
                detail="Packed `.codexpack.gguf` outputs are unsupported on the root tools API. Use a base `.gguf` output path.",
            )
        if not final_path.name.lower().endswith(".gguf"):
            raise HTTPException(status_code=400, detail="Output path must end with `.gguf`.")

        if final_path.exists() and not overwrite:
            raise HTTPException(status_code=409, detail=f"Output file already exists: {final_path}")
        if final_path.exists() and final_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Output path is a directory: {final_path}")

        try:
            quant = QuantizationType(quant_str)
        except ValueError as exc:
            allowed = ", ".join(q.value for q in QuantizationType)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid quantization: {quant_str!r} (allowed: {allowed})",
            ) from exc

        profile_id: str | None = None
        if profile_id_raw is not None:
            profile_id = str(profile_id_raw).strip() or None
            if profile_id is not None:
                from apps.backend.runtime.tools.gguf_converter_profiles import profile_by_id

                try:
                    profile_by_id(profile_id)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            precision_mode = normalize_precision_mode(precision_mode_raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        float_group_overrides: dict[str, str] = {}
        if float_group_overrides_raw is None:
            float_group_overrides = {}
        elif isinstance(float_group_overrides_raw, dict):
            for k, v in float_group_overrides_raw.items():
                group_id = str(k or "").strip()
                if not group_id:
                    raise HTTPException(status_code=400, detail="float_group_overrides contains an empty group id")
                try:
                    float_group_overrides[group_id] = normalize_mixed_float_override(v)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=str(exc),
                    ) from exc
        else:
            raise HTTPException(status_code=400, detail="float_group_overrides must be an object/dict when provided")

        if precision_mode is not None and float_group_overrides:
            raise HTTPException(
                status_code=400,
                detail="precision_mode cannot be combined with float_group_overrides.",
            )

        if any(v != "auto" for v in float_group_overrides.values()):
            if profile_id is None:
                raise HTTPException(status_code=400, detail="float_group_overrides requires profile_id")

            from apps.backend.runtime.tools.gguf_converter_float_groups import float_groups_for_profile_id

            allowed = {g.id for g in float_groups_for_profile_id(profile_id)}
            for gid, choice in float_group_overrides.items():
                if choice == "auto":
                    continue
                if gid not in allowed:
                    allowed_msg = ", ".join(sorted(allowed)) if allowed else "(none)"
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown float dtype group for profile {profile_id!r}: {gid!r} (allowed: {allowed_msg})",
                    )

        tensor_type_overrides: list[str] = []
        if isinstance(overrides_raw, str):
            tensor_type_overrides = [ln.strip() for ln in overrides_raw.splitlines() if ln.strip()]
        elif isinstance(overrides_raw, list):
            tensor_type_overrides = [str(x).strip() for x in overrides_raw if str(x).strip()]

        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_handle = tempfile.NamedTemporaryFile(
            prefix=f"{final_path.stem}.",
            suffix=f".part-{job_id}{final_path.suffix or '.gguf'}",
            dir=str(final_path.parent),
            delete=False,
        )
        tmp_path = Path(tmp_handle.name)
        tmp_handle.close()

        cancel_event = threading.Event()
        with _job_state_lock:
            _gguf_conversion_jobs[job_id] = _ToolJobState(
                status=_ToolJobStatus.PENDING,
                progress=0.0,
                current_tensor="",
                error=None,
                output_path=str(final_path),
            )
            _gguf_conversion_controls[job_id] = _GgufConversionControl(
                cancel_event=cancel_event,
                tmp_path=tmp_path,
                final_path=final_path,
            )

        def run_conversion() -> None:
            try:
                with _job_state_lock:
                    job = _gguf_conversion_jobs[job_id]
                    ctrl = _gguf_conversion_controls[job_id]

                config = ConversionConfig(
                    config_path=config_path,
                    safetensors_path=safetensors_path,
                    output_path=str(ctrl.tmp_path),
                    profile_id=profile_id,
                    quantization=quant,
                    tensor_type_overrides=tensor_type_overrides,
                    float_group_overrides=float_group_overrides,
                    precision_mode=precision_mode,
                )

                def progress_cb(prog):
                    status = _normalize_job_status(prog.status)
                    percent = float(prog.progress_percent)
                    if status is _ToolJobStatus.COMPLETE:
                        # The converter reports completion before we atomically move the temp
                        # file into place; keep polling until final rename finishes.
                        status = _ToolJobStatus.FINALIZING
                        percent = min(99.9, percent)
                    with _job_state_lock:
                        _set_job_state(
                            job,
                            status=status,
                            progress=percent,
                            current_tensor=prog.current_tensor,
                        )

                convert_safetensors_to_gguf(
                    config,
                    progress_callback=progress_cb,
                    should_cancel=lambda: bool(ctrl.cancel_event.is_set()),
                )

                with _job_state_lock:
                    _set_job_state(job, status=_ToolJobStatus.FINALIZING, progress=99.9)
                os.replace(str(ctrl.tmp_path), str(ctrl.final_path))

                with _job_state_lock:
                    _set_job_state(job, status=_ToolJobStatus.COMPLETE, progress=100)

            except GGUFConversionCancelled:
                with _job_state_lock:
                    job = _gguf_conversion_jobs[job_id]
                    _set_job_state(job, status=_ToolJobStatus.CANCELLED, error=None)
                try:
                    with _job_state_lock:
                        ctrl = _gguf_conversion_controls[job_id]
                    tmp = ctrl.tmp_path
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
            except Exception as exc:
                try:
                    with _job_state_lock:
                        ctrl = _gguf_conversion_controls[job_id]
                    tmp = ctrl.tmp_path
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
                with _job_state_lock:
                    job = _gguf_conversion_jobs[job_id]
                    _set_job_state(job, status=_ToolJobStatus.ERROR, error=str(exc))

        thread = threading.Thread(target=run_conversion, daemon=True)
        thread.start()

        return {"job_id": job_id, "status": "started"}

    @router.get("/api/tools/convert-gguf/{job_id}")
    async def get_gguf_conversion_status(job_id: str) -> Dict[str, Any]:
        with _job_state_lock:
            job = _gguf_conversion_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            return job.to_payload()

    @router.post("/api/tools/convert-gguf/{job_id}/cancel")
    async def cancel_gguf_conversion(job_id: str) -> Dict[str, Any]:
        with _job_state_lock:
            job = _gguf_conversion_jobs.get(job_id)
            ctrl = _gguf_conversion_controls.get(job_id)
            if job is None or ctrl is None:
                raise HTTPException(status_code=404, detail="Job not found")

            if job.status in _TERMINAL_JOB_STATUSES:
                raise HTTPException(status_code=409, detail=f"Job is not cancellable (status={job.status.value})")

            ctrl.cancel_event.set()
            _set_job_state(job, status=_ToolJobStatus.CANCELLING)
        return {"ok": True}

    @router.post("/api/tools/merge-safetensors")
    async def merge_safetensors(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Start a safetensors merge job."""
        from apps.backend.runtime.tools.safetensors_merge import (
            SafetensorsMergeConfig,
            SafetensorsMergeProgress,
            merge_safetensors_source,
            validate_safetensors_merge_config,
        )

        source_path_raw = str(payload.get("source_path") or "").strip()
        output_path_raw = str(payload.get("output_path") or "").strip()
        try:
            overwrite = parse_bool_value(payload.get("overwrite"), field="overwrite", default=False)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not source_path_raw:
            raise HTTPException(status_code=400, detail="source_path is required")
        if not output_path_raw:
            raise HTTPException(status_code=400, detail="output_path is required")

        try:
            source_path, final_path, _resolved = validate_safetensors_merge_config(
                SafetensorsMergeConfig(
                    source_path=source_path_raw,
                    output_path=output_path_raw,
                    overwrite=overwrite,
                )
            )
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        job_id = str(uuid.uuid4())[:8]
        tmp_path = _alloc_nonexistent_path(
            final_path.parent,
            prefix=f"{final_path.stem}.part-{job_id}-",
            suffix=final_path.suffix or ".safetensors",
        )

        with _job_state_lock:
            _safetensors_merge_jobs[job_id] = _ToolJobState(
                status=_ToolJobStatus.PENDING,
                progress=0.0,
                current_tensor="",
                error=None,
                output_path=str(final_path),
            )
            _safetensors_merge_controls[job_id] = _SafetensorsMergeControl(
                source_path=source_path,
                tmp_path=tmp_path,
                final_path=final_path,
            )

        def run_merge() -> None:
            try:
                with _job_state_lock:
                    job = _safetensors_merge_jobs[job_id]
                    ctrl = _safetensors_merge_controls[job_id]

                config = SafetensorsMergeConfig(
                    source_path=str(ctrl.source_path),
                    output_path=str(ctrl.tmp_path),
                    overwrite=False,
                )

                def progress_cb(prog: SafetensorsMergeProgress) -> None:
                    status = _normalize_job_status(prog.status)
                    percent = float(prog.progress_percent)
                    if status is _ToolJobStatus.COMPLETE:
                        # Runtime merge reports complete before the final atomic rename.
                        status = _ToolJobStatus.FINALIZING
                        percent = min(99.9, percent)
                    with _job_state_lock:
                        _set_job_state(
                            job,
                            status=status,
                            progress=percent,
                            current_tensor=prog.current_tensor,
                        )

                merge_safetensors_source(config, progress_callback=progress_cb)

                with _job_state_lock:
                    _set_job_state(job, status=_ToolJobStatus.FINALIZING, progress=99.9)

                if ctrl.final_path.exists() and ctrl.final_path.is_dir():
                    raise IsADirectoryError(f"output_path is a directory: {ctrl.final_path}")
                if ctrl.final_path.exists() and not overwrite:
                    raise FileExistsError(f"output file already exists: {ctrl.final_path}")

                os.replace(str(ctrl.tmp_path), str(ctrl.final_path))

                with _job_state_lock:
                    _set_job_state(job, status=_ToolJobStatus.COMPLETE, progress=100.0, error=None)
            except Exception as exc:
                try:
                    with _job_state_lock:
                        ctrl = _safetensors_merge_controls[job_id]
                    if ctrl.tmp_path.exists():
                        ctrl.tmp_path.unlink()
                except Exception:
                    pass
                with _job_state_lock:
                    job = _safetensors_merge_jobs[job_id]
                    _set_job_state(job, status=_ToolJobStatus.ERROR, error=str(exc))

        thread = threading.Thread(target=run_merge, daemon=True)
        thread.start()

        return {"job_id": job_id, "status": "started"}

    @router.get("/api/tools/merge-safetensors/{job_id}")
    async def get_merge_safetensors_status(job_id: str) -> Dict[str, Any]:
        with _job_state_lock:
            job = _safetensors_merge_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            return job.to_payload()

    @router.get("/api/tools/browse-files")
    async def browse_files(path: str = "", extensions: str = "") -> Dict[str, Any]:
        """Browse files/directories for file picker."""
        if not path:
            path = str(codex_root / "models")

        if not os.path.exists(path):
            return {"path": path, "exists": False, "items": []}

        if os.path.isfile(path):
            return {"path": path, "exists": True, "is_file": True, "items": []}

        ext_list = [e.strip().lower() for e in extensions.split(",") if e.strip()] if extensions else []

        items = []
        try:
            for entry in os.scandir(path):
                if entry.is_dir():
                    items.append({"name": entry.name, "type": "directory"})
                elif entry.is_file():
                    if not ext_list or any(entry.name.lower().endswith(ext) for ext in ext_list):
                        items.append(
                            {
                                "name": entry.name,
                                "type": "file",
                                "size": entry.stat().st_size,
                            }
                        )
        except PermissionError:
            pass

        items.sort(key=lambda x: (0 if x["type"] == "directory" else 1, x["name"].lower()))

        return {
            "path": path,
            "exists": True,
            "is_file": False,
            "parent": str(Path(path).parent),
            "items": items,
        }

    @router.post("/api/tools/pnginfo/analyze")
    async def analyze_pnginfo(file: UploadFile = File(...)) -> Dict[str, Any]:
        """Extract PNG text metadata for the PNG Info UI."""
        try:
            raw = await file.read()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read upload: {exc}") from exc

        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload")

        max_bytes = 50 * 1024 * 1024
        if len(raw) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large (max {max_bytes} bytes)")

        try:
            from PIL import Image  # type: ignore

            img = Image.open(BytesIO(raw))
            img.load()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

        fmt = str(getattr(img, "format", "") or "").upper()
        if fmt != "PNG":
            raise HTTPException(status_code=415, detail="Only PNG is supported")

        metadata: dict[str, str] = {}
        info = getattr(img, "info", None)
        if isinstance(info, dict):
            for k, v in info.items():
                if isinstance(k, str) and isinstance(v, str):
                    text = v.strip()
                    if text:
                        metadata[k] = text

        # Some PIL versions expose textual chunks on `.text` as well.
        text_map = getattr(img, "text", None)
        if isinstance(text_map, dict):
            for k, v in text_map.items():
                if isinstance(k, str) and isinstance(v, str):
                    text = v.strip()
                    if text:
                        metadata.setdefault(k, text)

        return {
            "width": int(getattr(img, "width", 0) or 0),
            "height": int(getattr(img, "height", 0) or 0),
            "metadata": metadata,
        }

    return router
