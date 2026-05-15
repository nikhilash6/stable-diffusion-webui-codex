"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Task orchestration helpers for upscaler endpoints.
Keeps `/api/upscale` and `/api/upscalers/download` routers thin by centralizing the worker boilerplate:
status/progress/result/end/error + cancellation checks.
Uses the shared inference gate for the upscale worker when `CODEX_SINGLE_FLIGHT=1` and always marks tasks finished via `TaskEntry.mark_finished`.
Any cancel mode may abort while waiting on the inference gate; once running, only `immediate` interrupts the active work loop.
The upscalers download task verifies file integrity against the HF manifest (`upscalers/manifest.json`, schema v1) when available,
and returns a per-destination `sha256_by_path` mapping in the task result `info` payload.

Symbols (top-level; keep in sync; no ghosts):
- `_parse_bool_option` (function): Strict parser for bool-like option values used by upscale task settings.
- `run_upscale_task` (function): Runs an upscale task worker (single-image v1).
- `run_upscaler_download_task` (function): Runs an HF download task worker for curated upscaler weights.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import io
import json
import hashlib
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

from apps.backend.interfaces.api.inference_gate import acquire_inference_gate, release_inference_gate, single_flight_enabled
from apps.backend.interfaces.api.public_errors import (
    build_cancelled_task_error,
    build_missing_result_task_error,
    build_public_task_error,
)
from apps.backend.interfaces.api.task_registry import TaskCancelMode, TaskEntry

logger = get_backend_logger("backend.api.tasks.upscale")

_HF_MANIFEST_PATH = "upscalers/manifest.json"


def _parse_bool_option(value: object, *, field: str, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    raise RuntimeError(
        f"Invalid '{field}': expected bool or one of "
        f"('true','false','1','0','yes','no','on','off'), got {value!r}."
    )


@dataclass(slots=True)
class _DownloadItem:
    hf_path: str
    dst_path: Path


def _copy_to_dst_atomic_and_hash_sha256(
    *,
    src: Path,
    dst: Path,
    expected_sha256: str | None,
    label: str,
) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise RuntimeError(f"Destination exists: {dst}")

    tmp = dst.with_name(f".{dst.name}.tmp-{uuid4().hex}")
    try:
        digest = hashlib.sha256()
        with src.open("rb") as src_handle, tmp.open("xb") as dst_handle:
            for chunk in iter(lambda: src_handle.read(1024 * 1024), b""):
                digest.update(chunk)
                dst_handle.write(chunk)
        sha256 = digest.hexdigest()
        if expected_sha256 and sha256.lower() != str(expected_sha256).strip().lower():
            raise RuntimeError(f"SHA256 mismatch for {label}: expected {expected_sha256}, got {sha256}")
        shutil.copystat(src, tmp)
        tmp.replace(dst)
        return sha256
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            dst.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def run_upscale_task(
    *,
    task_id: str,
    payload: dict[str, Any],
    image_bytes: bytes,
    entry: TaskEntry,
    device: str,
    opts_get: Callable[..., Any],
    generation_provenance: Mapping[str, str],
    save_generated_images: Callable[..., Any],
) -> None:
    """Run a standalone upscaling task (single image)."""

    def push(event: dict[str, Any]) -> None:
        entry.push_event(event)

    push({"type": "status", "stage": "queued"})

    def worker() -> None:
        acquired = False
        success = False
        try:
            if single_flight_enabled():
                push({"type": "status", "stage": "waiting_for_inference"})

            acquired = acquire_inference_gate(
                should_cancel=lambda: bool(entry.cancel_requested),
            )
            if not acquired:
                entry.error = build_cancelled_task_error()
                return

            push({"type": "status", "stage": "running"})
            from apps.backend.interfaces.api.device_selection import apply_primary_device

            apply_primary_device(device)

            if entry.cancel_requested and entry.cancel_mode is TaskCancelMode.IMMEDIATE:
                entry.error = build_cancelled_task_error()
                return

            from PIL import Image  # type: ignore

            from apps.backend.use_cases.upscale import UpscaleParams, upscale_pil_image
            from apps.backend.interfaces.api.tasks.generation_tasks import encode_images
            from apps.backend.core.engine_interface import TaskType

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            params = UpscaleParams.from_payload(payload)

            # Tile progress callback.
            def on_tile(step: int, total: int) -> None:
                if entry.cancel_requested and entry.cancel_mode is TaskCancelMode.IMMEDIATE:
                    raise RuntimeError("cancelled")
                percent = None
                try:
                    percent = (float(step) / float(total)) * 100.0 if total > 0 else None
                except Exception:
                    percent = None
                push(
                    {
                        "type": "progress",
                        "stage": "upscale",
                        "percent": percent,
                        "step": int(step),
                        "total_steps": int(total) if total else None,
                        "eta_seconds": None,
                    }
                )

            out = upscale_pil_image(img, params=params, progress_callback=on_tile)
            result_images = [out]

            info_obj: dict[str, Any] = {
                "upscale": params.to_dict(),
                **generation_provenance,
            }

            if _parse_bool_option(
                opts_get("samples_save", True),
                field="options.samples_save",
                default=True,
            ):
                save_generated_images(result_images, task=TaskType.UPSCALE, info=info_obj, metadata=generation_provenance)

            result = {
                "images": encode_images(result_images, metadata=generation_provenance),
                "info": info_obj,
            }
            entry.result = {"status": "completed", "result": result}
            success = True
        except Exception as err:  # pragma: no cover - surfaces runtime errors
            entry.error = build_public_task_error(err)
            success = False
        finally:
            if success:
                result_obj = entry.result.get("result") if isinstance(entry.result, dict) else None
                if not isinstance(result_obj, dict):
                    entry.error = build_missing_result_task_error()
                    success = False
            entry.mark_finished(success=success)
            entry.schedule_cleanup(task_id)
            if acquired:
                try:
                    release_inference_gate()
                except Exception as exc:
                    logger.warning(
                        "inference gate release failed in upscale worker (task_id=%s): %s",
                        task_id,
                        exc,
                        exc_info=False,
                    )

    threading.Thread(target=worker, name=f"upscale-task-{task_id}", daemon=True).start()


def run_upscaler_download_task(
    *,
    task_id: str,
    items: Sequence[_DownloadItem],
    entry: TaskEntry,
    hf_repo_id: str,
    hf_revision: str | None,
) -> None:
    """Download allowlisted upscaler weights from a curated HF repo into local model roots."""

    def push(event: dict[str, Any]) -> None:
        entry.push_event(event)

    push({"type": "status", "stage": "queued"})

    def worker() -> None:
        success = False
        written_paths: list[Path] = []
        try:
            push({"type": "status", "stage": "running"})

            from huggingface_hub import hf_hub_download  # type: ignore

            total = len(items)
            completed = 0
            sha256_by_path: dict[str, str] = {}

            expected_sha256_by_hf_path: dict[str, str] = {}
            manifest_errors: list[str] = []
            try:
                local_manifest = hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=_HF_MANIFEST_PATH,
                    revision=hf_revision,
                )
                with open(local_manifest, "r", encoding="utf-8") as handle:
                    raw_manifest = json.load(handle)
                from apps.backend.interfaces.api.upscalers_manifest import validate_upscalers_manifest

                result = validate_upscalers_manifest(raw_manifest)
                manifest_errors = list(result.errors or [])
                expected_sha256_by_hf_path = {
                    str(hf_path): str(meta.get("sha256"))
                    for hf_path, meta in (result.weights_by_hf_path or {}).items()
                    if isinstance(hf_path, str) and isinstance(meta, dict) and isinstance(meta.get("sha256"), str) and meta.get("sha256")
                }
            except Exception as exc:
                manifest_errors = [f"manifest unavailable: {exc}"]
                expected_sha256_by_hf_path = {}

            for item in items:
                if entry.cancel_requested and entry.cancel_mode is TaskCancelMode.IMMEDIATE:
                    raise RuntimeError("cancelled")

                completed += 1
                push(
                    {
                        "type": "progress",
                        "stage": "download",
                        "percent": (completed / total) * 100.0 if total else None,
                        "step": completed,
                        "total_steps": total,
                        "eta_seconds": None,
                    }
                )

                local_tmp = hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=item.hf_path,
                    revision=hf_revision,
                )
                src_path = Path(str(local_tmp)).resolve()
                dst_path = item.dst_path

                expected = expected_sha256_by_hf_path.get(item.hf_path)
                sha256_by_path[str(dst_path)] = _copy_to_dst_atomic_and_hash_sha256(
                    src=src_path,
                    dst=dst_path,
                    expected_sha256=expected,
                    label=item.hf_path,
                )
                written_paths.append(dst_path)

            try:
                from apps.backend.runtime.vision.upscalers.registry import invalidate_upscalers_cache

                invalidate_upscalers_cache()
            except Exception:
                pass

            result = {
                "files": [str(p) for p in written_paths],
                "sha256_by_path": sha256_by_path,
                "manifest_errors": manifest_errors,
            }
            entry.result = {"status": "completed", "result": {"images": [], "info": result}}
            success = True
        except Exception as err:  # pragma: no cover
            for path in written_paths:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
            entry.error = build_public_task_error(err)
            success = False
        finally:
            if success:
                result_obj = entry.result.get("result") if isinstance(entry.result, dict) else None
                if not isinstance(result_obj, dict):
                    entry.error = build_missing_result_task_error()
                    success = False
            entry.mark_finished(success=success)
            entry.schedule_cleanup(task_id)

    threading.Thread(target=worker, name=f"upscalers-download-task-{task_id}", daemon=True).start()


__all__ = ["run_upscale_task", "run_upscaler_download_task"]
