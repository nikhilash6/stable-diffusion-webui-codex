"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: System/diagnostic API routes (health, version, memory, VRAM cleanup).
Exposes lightweight endpoints used by the UI footer/diagnostics overlays and a fail-loud VRAM cleanup entrypoint.

Symbols (top-level; keep in sync; no ghosts):
- `_parse_optional_int` (function): Parses integer-like text (supports mixed unit strings like `123 MiB`).
- `_parse_compute_csv_line` (function): Parses one `nvidia-smi --query-compute-apps` CSV row into a process descriptor.
- `_query_gpu_compute_processes` (function): Returns external GPU compute processes from `nvidia-smi` plus warnings/availability.
- `_normalize_process_basename` (function): Resolves a stable lowercase executable basename from process path/name strings.
- `_is_critical_process_name` (function): Flags critical OS/shell/UI process names that must not be terminated.
- `_kill_process` (function): Terminates a PID cross-platform (POSIX `SIGKILL`, Windows `taskkill`).
- `_protected_pids` (function): Returns PID set that must not be killed by the obliterate endpoint (current + parent).
- `ObliterateExternalKillMode` (enum): External termination mode for `/api/obliterate-vram`.
- `ObliterateVramRequest` (model): Request payload for `/api/obliterate-vram`.
- `build_router` (function): Build the APIRouter for system endpoints.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import gc
import logging
import ntpath
import os
import re
import signal
import subprocess
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from apps.backend.runtime.load_authority import (
    LoadAuthorityStage,
    coordinator_load_permit,
)

_LOG = get_backend_logger(__name__)

_CRITICAL_PROCESS_BASENAMES: set[str] = {
    "explorer.exe",
    "dwm.exe",
    "winlogon.exe",
    "csrss.exe",
    "smss.exe",
    "services.exe",
    "lsass.exe",
    "svchost.exe",
    "taskmgr.exe",
    "startmenuexperiencehost.exe",
    "shellexperiencehost.exe",
    "searchhost.exe",
    "code.exe",
    "powershell.exe",
    "pwsh.exe",
    "cmd.exe",
    "conhost.exe",
    "windowsterminal.exe",
    "wt.exe",
}


class ObliterateExternalKillMode(str, Enum):
    DISABLED = "disabled"
    ALL = "all"


class ObliterateVramRequest(BaseModel):
    external_kill_mode: ObliterateExternalKillMode = ObliterateExternalKillMode.DISABLED


def _parse_optional_int(raw_value: str) -> Optional[int]:
    text = str(raw_value or "").strip()
    if not text:
        return None
    if text.lower() in {"n/a", "na", "-"}:
        return None
    direct = re.fullmatch(r"-?\d+", text)
    if direct is not None:
        try:
            return int(text)
        except Exception:
            return None
    match = re.search(r"-?\d+", text)
    if match is None:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def _parse_compute_csv_line(raw_line: str) -> Optional[Dict[str, Any]]:
    line = str(raw_line or "").strip()
    if not line:
        return None
    columns = [segment.strip() for segment in line.split(",")]
    if not columns:
        return None
    pid = _parse_optional_int(columns[0])
    if pid is None or pid <= 0:
        return None
    process_name = columns[1] if len(columns) > 1 else ""
    used_gpu_memory_mb = _parse_optional_int(columns[2]) if len(columns) > 2 else None
    gpu_uuid = columns[3] if len(columns) > 3 else ""
    return {
        "pid": pid,
        "process_name": process_name,
        "used_gpu_memory_mb": used_gpu_memory_mb,
        "gpu_uuid": gpu_uuid,
    }


def _query_gpu_compute_processes() -> Tuple[list[Dict[str, Any]], list[str], bool]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return [], ["nvidia-smi was not found; external GPU process cleanup skipped."], False
    except Exception as exc:  # pragma: no cover - defensive command execution guard
        return [], [f"nvidia-smi probe failed: {exc}"], False

    warnings: list[str] = []
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"exit={completed.returncode}"
        warnings.append(f"nvidia-smi query failed: {detail}")
        return [], warnings, True

    processes: list[Dict[str, Any]] = []
    seen_pids: set[int] = set()
    for raw_line in (completed.stdout or "").splitlines():
        parsed = _parse_compute_csv_line(raw_line)
        if parsed is None:
            line = str(raw_line or "").strip()
            if line:
                warnings.append(f"ignored malformed nvidia-smi row: {line}")
            continue
        pid = int(parsed["pid"])
        if pid in seen_pids:
            continue
        seen_pids.add(pid)
        processes.append(parsed)

    return processes, warnings, True


def _normalize_process_basename(process_name: str) -> str:
    raw = str(process_name or "").strip().strip('"').strip("'")
    if not raw:
        return ""
    # nvidia-smi on Windows commonly returns absolute paths with backslashes.
    from_nt = ntpath.basename(raw)
    from_posix = os.path.basename(raw)
    base = from_nt or from_posix or raw
    return base.strip().lower()


def _is_critical_process_name(process_name: str) -> bool:
    base = _normalize_process_basename(process_name)
    if not base:
        return False
    return base in _CRITICAL_PROCESS_BASENAMES


def _kill_process(pid: int) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or f"taskkill exit={result.returncode}"
            raise RuntimeError(detail)
        return

    os.kill(pid, signal.SIGKILL)


def _protected_pids() -> set[int]:
    protected: set[int] = {os.getpid()}
    parent_pid = os.getppid()
    if isinstance(parent_pid, int) and parent_pid > 1:
        protected.add(parent_pid)
    return protected


def build_router(*, app_version: str) -> APIRouter:
    router = APIRouter()

    @router.get("/api/health")
    def health() -> Dict[str, bool]:
        return {"ok": True}

    @router.get("/api/version")
    def version_info() -> Dict[str, Any]:
        """Return backend version details for footer display."""
        # Git commit
        git_commit: Optional[str] = None
        try:
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            git_commit = os.environ.get("GIT_COMMIT") or os.environ.get("VITE_GIT_COMMIT") or None

        # Python
        import sys

        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Torch/CUDA (optional)
        torch_ver: Optional[str] = None
        cuda_ver: Optional[str] = None
        try:
            import torch  # type: ignore

            torch_ver = getattr(torch, "__version__", None)
            cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
        except Exception:
            pass

        return {
            "app_version": app_version,
            "git_commit": git_commit,
            "python_version": py_ver,
            "torch_version": torch_ver,
            "cuda_version": cuda_ver,
        }

    @router.get("/api/memory")
    def memory() -> Dict[str, Any]:
        """Return a snapshot of current VRAM/CPU memory state."""
        try:
            from apps.backend.runtime import memory_management as _mm  # type: ignore

            snap = _mm.memory_snapshot()
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"memory snapshot failed: {exc}")

        probe = snap.get("probe", {}) or {}
        totals = snap.get("totals", {}) or {}
        torch_stats = snap.get("torch", {}) or {}
        attention = snap.get("attention", {}) or {}

        raw_total_vram_mb = probe.get("total_vram_mb", None)
        if raw_total_vram_mb is None:
            total_vram_mb = 0
        else:
            if isinstance(raw_total_vram_mb, bool):
                raise HTTPException(
                    status_code=500,
                    detail="memory snapshot contract error: probe.total_vram_mb must be int-like, got bool.",
                )
            try:
                total_vram_mb = int(raw_total_vram_mb)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "memory snapshot contract error: invalid probe.total_vram_mb "
                        f"value={raw_total_vram_mb!r}."
                    ),
                ) from exc
            if total_vram_mb < 0:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "memory snapshot contract error: probe.total_vram_mb must be >= 0 "
                        f"(got {total_vram_mb})."
                    ),
                )

        return {
            "device_backend": snap.get("device_backend"),
            "primary_device": snap.get("primary_device"),
            "total_vram_mb": total_vram_mb,
            "probe": probe,
            "budgets": snap.get("budgets", {}),
            "attention": attention,
            "torch": torch_stats,
            "models": snap.get("models", []),
            "totals": totals,
        }

    @router.post("/api/obliterate-vram")
    def obliterate_vram(payload: Optional[ObliterateVramRequest] = None) -> Dict[str, Any]:
        effective_payload = payload or ObliterateVramRequest()
        report: Dict[str, Any] = {
            "ok": True,
            "message": "",
            "internal": {
                "runtime_unload_models": False,
                "runtime_soft_empty_cache": False,
                "gguf_cache_cleared": False,
                "gc_collect_ran": False,
                "torch_cuda_cache_cleared": False,
            },
            "internal_failures": [],
            "external": {
                "kill_mode": effective_payload.external_kill_mode.value,
                "nvidia_smi_available": False,
                "detected_processes": [],
                "terminated_pids": [],
                "skipped": [],
                "failures": [],
            },
            "warnings": [],
        }

        try:
            from apps.backend.runtime import memory_management as memory_state  # type: ignore
        except Exception as exc:
            report["internal_failures"].append(f"memory_manager_import:{exc}")
        else:
            try:
                with coordinator_load_permit(
                    owner="api.routers.system.obliterate_vram",
                    stage=LoadAuthorityStage.CLEANUP,
                ):
                    memory_state.manager.unload_all_models()
                report["internal"]["runtime_unload_models"] = True
            except Exception as exc:
                report["internal_failures"].append(f"runtime_unload_models:{exc}")

            try:
                memory_state.manager.soft_empty_cache(force=True)
                report["internal"]["runtime_soft_empty_cache"] = True
            except Exception as exc:
                report["internal_failures"].append(f"runtime_soft_empty_cache:{exc}")

        try:
            gc.collect()
            report["internal"]["gc_collect_ran"] = True
        except Exception as exc:  # pragma: no cover - defensive
            report["warnings"].append(f"gc_collect_failed:{exc}")

        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                report["internal"]["torch_cuda_cache_cleared"] = True
        except Exception as exc:
            report["internal_failures"].append(f"torch_cuda_cache_clear:{exc}")

        detected_processes, external_warnings, nvidia_available = _query_gpu_compute_processes()
        report["external"]["nvidia_smi_available"] = bool(nvidia_available)
        report["external"]["detected_processes"] = detected_processes
        report["warnings"].extend(external_warnings)

        protected = _protected_pids()
        if effective_payload.external_kill_mode == ObliterateExternalKillMode.DISABLED:
            report["warnings"].append("external_gpu_termination_disabled_by_default")
            for process in detected_processes:
                report["external"]["skipped"].append(
                    {
                        "pid": int(process["pid"]),
                        "reason": "external_kill_disabled",
                    }
                )
        else:
            for process in detected_processes:
                pid = int(process["pid"])
                process_name = str(process.get("process_name", ""))
                if pid in protected:
                    report["external"]["skipped"].append({"pid": pid, "reason": "protected_pid"})
                    continue
                if _is_critical_process_name(process_name):
                    report["external"]["skipped"].append({"pid": pid, "reason": "critical_process_name"})
                    continue
                try:
                    _kill_process(pid)
                    report["external"]["terminated_pids"].append(pid)
                except ProcessLookupError:
                    report["external"]["skipped"].append({"pid": pid, "reason": "already_exited"})
                except Exception as exc:
                    report["external"]["failures"].append({"pid": pid, "error": str(exc)})

        if not report["external"]["nvidia_smi_available"]:
            report["warnings"].append("external_gpu_cleanup_unavailable")

        failure_count = len(report["internal_failures"]) + len(report["external"]["failures"])
        if failure_count > 0:
            report["ok"] = False
            report["message"] = (
                "Obliterate VRAM finished with failures. "
                f"internal_failures={len(report['internal_failures'])}, "
                f"external_failures={len(report['external']['failures'])}"
            )
        else:
            killed = len(report["external"]["terminated_pids"])
            if effective_payload.external_kill_mode == ObliterateExternalKillMode.DISABLED:
                detected = len(report["external"]["detected_processes"])
                report["message"] = (
                    "Obliterate VRAM finished successfully (internal cleanup only). "
                    f"external_detected={detected}, external_killed=0"
                )
            else:
                report["message"] = f"Obliterate VRAM finished successfully. external_killed={killed}"

        _LOG.info(
            "Obliterate VRAM result: ok=%s killed=%d internal_failures=%d external_failures=%d warnings=%d",
            report["ok"],
            len(report["external"]["terminated_pids"]),
            len(report["internal_failures"]),
            len(report["external"]["failures"]),
            len(report["warnings"]),
        )
        return report

    return router
