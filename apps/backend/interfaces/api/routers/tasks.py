"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Task status and output file API routes.
Exposes SSE task events (with bounded replay/resume via `after` / `Last-Event-ID` + monotonic `id:`), cancellation, and output file access under
CODEX_ROOT/output.
Cancellation endpoint supports `immediate` and `after_current`; default mode is env-configurable via
`CODEX_TASK_CANCEL_DEFAULT_MODE` with strict fail-loud validation.

Symbols (top-level; keep in sync; no ghosts):
- `_default_task_cancel_mode` (function): Resolves and strictly validates `CODEX_TASK_CANCEL_DEFAULT_MODE`.
- `DEFAULT_TASK_CANCEL_MODE` (constant): Effective default cancel mode used when `/api/tasks/{id}/cancel` omits `mode`.
- `build_router` (function): Build the APIRouter for task endpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from apps.backend.interfaces.api.public_errors import PublicTaskError
from apps.backend.interfaces.api.task_registry import (
    TaskCancelMode,
    TaskEventType,
    get_task,
    parse_task_cancel_mode,
    request_task_cancel,
    restore_task_cancel_request,
)


def _default_task_cancel_mode() -> TaskCancelMode:
    raw_mode = os.getenv("CODEX_TASK_CANCEL_DEFAULT_MODE", TaskCancelMode.IMMEDIATE.value)
    try:
        return parse_task_cancel_mode(raw_mode)
    except ValueError as exc:
        raise RuntimeError(
            "Invalid CODEX_TASK_CANCEL_DEFAULT_MODE value; expected 'immediate' or 'after_current' "
            f"(got {raw_mode!r})."
        ) from exc


DEFAULT_TASK_CANCEL_MODE = _default_task_cancel_mode()


def build_router(*, codex_root: Path, backend_state: Any) -> APIRouter:
    router = APIRouter()

    def _snapshot_error_payload(error: PublicTaskError) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "error": error.message,
            "error_code": error.code.value,
        }
        if error.error_id is not None:
            payload["error_id"] = error.error_id
        return payload

    def _event_error_payload(error: PublicTaskError) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": TaskEventType.ERROR.value,
            "message": error.message,
            "code": error.code.value,
        }
        if error.error_id is not None:
            payload["error_id"] = error.error_id
        return payload

    @router.get("/api/output/{rel_path:path}")
    async def get_output_file(rel_path: str) -> FileResponse:
        root = (codex_root / "output").resolve()
        raw = str(rel_path or "").lstrip("/").replace("\\", "/")
        target = (root / raw).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid output path") from None
        if not target.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(str(target))

    @router.get("/api/tasks/{task_id}")
    async def task_status(task_id: str) -> Dict[str, Any]:
        entry = get_task(task_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if entry.done.done():
            if entry.error:
                entry.schedule_cleanup(task_id)
                payload: Dict[str, Any] = {
                    "status": "error",
                    "last_event_id": entry.last_event_id(),
                }
                payload.update(_snapshot_error_payload(entry.error))
                return payload
            entry.schedule_cleanup(task_id)
            out = entry.result
            if not isinstance(out, dict):
                raise RuntimeError("Task completed without result payload.")
            result_obj = out.get("result")
            if not isinstance(result_obj, dict):
                raise RuntimeError("Task result payload must include a dict 'result' field.")
            out.setdefault("last_event_id", entry.last_event_id())
            return out
        snap = entry.snapshot_running()
        snap["task_id"] = task_id
        return snap

    @router.get("/api/tasks/{task_id}/events")
    async def task_events(task_id: str, request: Request, after: int | None = None) -> StreamingResponse:
        entry = get_task(task_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Task not found")

        def _sse(*, event_id: int, json_payload: str) -> str:
            return f"id: {event_id}\n" f"data: {json_payload}\n\n"

        async def event_stream():
            last_id = 0
            if isinstance(after, int) and after >= 0:
                last_id = int(after)
            else:
                try:
                    raw = request.headers.get("Last-Event-ID")
                    if raw is not None and str(raw).strip():
                        last_id = max(0, int(str(raw).strip(), 10))
                except Exception:
                    last_id = 0

            while True:
                gap, items = entry.iter_events_after(last_id)
                if gap:
                    oldest, newest = entry.buffer_window()
                    gap_id = max(int(last_id) + 1, (int(oldest) - 1) if int(oldest) > 0 else int(last_id) + 1)
                    gap_payload = {
                        "type": TaskEventType.GAP.value,
                        "oldest_event_id": int(oldest),
                        "newest_event_id": int(newest),
                        "last_event_id": int(entry.last_event_id()),
                    }
                    yield _sse(event_id=gap_id, json_payload=json.dumps(gap_payload))
                    # After a gap, replay whatever we still have. If the buffer is empty, avoid looping gaps.
                    if int(oldest) > 0:
                        last_id = int(gap_id)
                    else:
                        base_last_id = int(entry.last_event_id()) - (2 if entry.done.done() else 0)
                        last_id = max(int(gap_id), max(0, base_last_id))
                    continue

                for ev in items:
                    last_id = int(ev.event_id)
                    yield _sse(event_id=int(ev.event_id), json_payload=ev.json_payload)

                if entry.done.done():
                    # Terminal emission: always end the stream (contract).
                    terminal_ids = entry.terminal_event_ids()
                    if terminal_ids is None:
                        # Defensive: should be impossible when done is set.
                        break

                    primary_id, end_id = terminal_ids
                    if int(last_id) < int(primary_id):
                        if entry.error:
                            err_payload = _event_error_payload(entry.error)
                            yield _sse(event_id=int(primary_id), json_payload=json.dumps(err_payload))
                        else:
                            if not isinstance(entry.result, dict):
                                raise RuntimeError("Task completed without result payload.")
                            result_obj = entry.result.get("result")
                            if not isinstance(result_obj, dict):
                                raise RuntimeError("Task result payload must include a dict 'result' field.")
                            result_payload: dict[str, Any] = {"type": TaskEventType.RESULT.value}
                            result_payload.update(result_obj)
                            yield _sse(event_id=int(primary_id), json_payload=json.dumps(result_payload))
                        last_id = int(primary_id)

                    if int(last_id) < int(end_id):
                        yield _sse(event_id=int(end_id), json_payload=json.dumps({"type": TaskEventType.END.value}))
                        last_id = int(end_id)

                    entry.schedule_cleanup(task_id)
                    break

                await entry.wait_for_event_or_done(after_event_id=int(last_id))

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.post("/api/tasks/{task_id}/cancel")
    async def task_cancel(task_id: str, payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="cancel payload must be an object")
        try:
            mode = parse_task_cancel_mode(payload.get("mode", DEFAULT_TASK_CANCEL_MODE.value))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        entry = get_task(task_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Task not found")
        previous_cancel_requested = bool(entry.cancel_requested)
        previous_cancel_mode = entry.cancel_mode
        task_has_started = entry.snapshot_running().get("started_at_ms") is not None and not entry.done.done()

        stop_generating = None
        if mode is TaskCancelMode.IMMEDIATE:
            stop_generating = getattr(backend_state, "stop_generating", None)
            if not callable(stop_generating):
                raise RuntimeError("backend_state.stop_generating is required for immediate cancellation")

        ok = request_task_cancel(task_id, mode=mode)
        if not ok:
            raise HTTPException(status_code=404, detail="Task not found")

        if mode is TaskCancelMode.IMMEDIATE and task_has_started:
            try:
                stop_generating()
            except Exception:
                restore_task_cancel_request(
                    task_id,
                    cancel_requested=previous_cancel_requested,
                    cancel_mode=previous_cancel_mode,
                )
                raise
        return {"status": "cancelling", "mode": mode.value}

    return router
