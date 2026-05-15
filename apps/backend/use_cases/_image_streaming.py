"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared streaming helpers for image mode wrappers (txt2img/img2img).
Provides seed normalization, worker-thread execution (with smart runtime override propagation), sampling progress polling, decode normalization (including pre-decode cache flush + CPU-target decode transfer), and common `info` metadata building.

Symbols (top-level; keep in sync; no ghosts):
- `_resolve_seed_plan` (function): Normalize request seed + batch_total into (seed, all_seeds, subseeds, subseed_strength).
- `_ImageProgressProfile` (dataclass): Narrow use-case-local config for image `ProgressEvent` shaping (phase weights, optional encode phase, additive payload fields).
- `_resolve_progress_owner_token` (function): Resolve the per-run progress-owner token used to isolate raw backend snapshots across overlapping image tasks.
- `_seed_progress_owner_token` (function): Install the expected owner token into the raw backend state before worker-side encode/sampling begins.
- `_task_context_from_worker_name` (function): Normalize worker thread names into task-context tokens for structured logs.
- `_log_runtime_override_failure` (function): Emit classified smart-runtime override failure events with task/request context.
- `_normalize_runtime_overrides` (function): Validate/normalize worker smart runtime overrides (fail-loud contract, explicit transient path).
- `_run_inference_worker` (function): Run a callable in a daemon thread while propagating smart runtime overrides and capturing output/error/timings.
- `_iter_sampling_progress` (function): Poll `backend_state` and yield phase-aware progress snapshots (sampling + VAE encode/decode blocks) until a worker signals completion.
- `_iter_image_progress_events` (function): Convert shared phase snapshots into truthful image `ProgressEvent`s for txt2img/img2img/FLUX.2 wrappers.
- `_decode_generation_output` (function): Normalize `GenerationResult`/tensor output into a list of PIL images and decode timing (pre-decode cache flush + CPU-target decode transfer).
- `_build_common_info` (function): Build the shared `info` dict for image tasks (engine/task/dims/seed/sampler/scheduler/prompts/extra/timings).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Iterator, Mapping, Sequence
from uuid import uuid4

from apps.backend.core.requests import ProgressEvent


_SEED_MASK = 0x7FFFFFFF
_SMART_RUNTIME_OVERRIDE_KEYS: tuple[str, str, str] = (
    "smart_offload",
    "smart_fallback",
    "smart_cache",
)
_SMART_RUNTIME_OVERRIDE_KEYSET = frozenset(_SMART_RUNTIME_OVERRIDE_KEYS)


def _normalize_seed(value: int) -> int:
    return int(value) & _SEED_MASK


def _resolve_seed_plan(
    *,
    seed: int | None,
    batch_total: int,
) -> tuple[int, list[int], list[int], float]:
    """Resolve a request seed into per-image seeds for a batch.

    Rules (Codex semantics):
    - If seed is missing or < 0: generate a random seed per image.
    - Else: use seed+i for each image.
    """
    import secrets

    total = max(1, int(batch_total))
    raw_seed = None if seed is None else int(seed)
    if raw_seed is None or raw_seed < 0:
        seeds = [_normalize_seed(secrets.randbits(32)) for _ in range(total)]
        base = seeds[0]
    else:
        base = _normalize_seed(raw_seed)
        seeds = [_normalize_seed(base + idx) for idx in range(total)]

    subseeds = [-1 for _ in range(total)]
    return base, seeds, subseeds, 0.0


@dataclass(slots=True)
class _WorkerOutcome:
    output: Any = None
    error: BaseException | None = None
    success: bool = False
    sampling_start: float | None = None
    sampling_end: float | None = None


@dataclass(frozen=True, slots=True)
class _ImageProgressProfile:
    encode_weight: float
    sampling_weight: float
    decode_weight: float
    emit_encode_phase: bool = True
    extra_data: Mapping[str, Any] | None = None
    include_sampling_block_alias: bool = False


def _task_context_from_worker_name(name: str) -> str:
    worker_name = str(name or "").strip()
    if not worker_name:
        return "unknown"
    if worker_name.endswith("-worker"):
        return worker_name[: -len("-worker")] or worker_name
    return worker_name


def _resolve_progress_owner_token(*, task_context: str, task_id: str | None) -> str:
    normalized_task_id = str(task_id or "").strip()
    if normalized_task_id:
        return f"task:{normalized_task_id}"
    normalized_context = _task_context_from_worker_name(task_context)
    return f"worker:{normalized_context}:{uuid4().hex}"


def _seed_progress_owner_token(*, progress_owner_token: str) -> None:
    from apps.backend.core.state import state as backend_state

    backend_state.set_progress_owner_token(progress_owner_token)


def _log_runtime_override_failure(
    *,
    category: str,
    worker_name: str,
    request_context: str,
    error: BaseException,
    details: Mapping[str, object] | None = None,
    transient: bool,
) -> None:
    from apps.backend.runtime.logging import emit_backend_event

    payload: dict[str, object] = {
        "category": str(category),
        "task_context": _task_context_from_worker_name(worker_name),
        "request_context": str(request_context or "unknown"),
        "worker_name": str(worker_name),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if details:
        payload.update({str(key): value for key, value in details.items()})
    emit_backend_event(
        "smart_cache.runtime_overrides_failure",
        logger="smart_offload",
        level=(logging.WARNING if transient else logging.ERROR),
        **payload,
    )


def _normalize_runtime_overrides(
    *,
    runtime_overrides: Mapping[str, bool | None] | None,
    worker_name: str,
    request_context: str,
) -> dict[str, bool | None]:
    from apps.backend.runtime.memory.smart_offload import current_smart_runtime_overrides

    source_overrides: Mapping[str, object]
    if runtime_overrides is None:
        try:
            source_overrides = current_smart_runtime_overrides()
        except Exception as exc:  # noqa: BLE001 - explicit transient classification/logging
            wrapped = RuntimeError(
                "Failed to snapshot smart runtime overrides before worker launch."
            )
            _log_runtime_override_failure(
                category="transient",
                worker_name=worker_name,
                request_context=request_context,
                error=wrapped,
                details={"stage": "current_smart_runtime_overrides", "source_error": str(exc)},
                transient=True,
            )
            raise wrapped from exc
    else:
        source_overrides = runtime_overrides

    if not isinstance(source_overrides, Mapping):
        err = TypeError(
            "runtime_overrides must be Mapping[str, bool | None] when provided "
            f"(got {type(source_overrides).__name__})."
        )
        _log_runtime_override_failure(
            category="contract",
            worker_name=worker_name,
            request_context=request_context,
            error=err,
            details={"stage": "runtime_overrides.mapping_type"},
            transient=False,
        )
        raise err

    unknown_keys = sorted(
        str(key)
        for key in source_overrides.keys()
        if key not in _SMART_RUNTIME_OVERRIDE_KEYSET
    )
    if unknown_keys:
        err = ValueError(
            "runtime_overrides received unknown keys; expected only "
            f"{sorted(_SMART_RUNTIME_OVERRIDE_KEYSET)} (got {unknown_keys})."
        )
        _log_runtime_override_failure(
            category="contract",
            worker_name=worker_name,
            request_context=request_context,
            error=err,
            details={"stage": "runtime_overrides.unknown_keys", "unknown_keys": unknown_keys},
            transient=False,
        )
        raise err

    normalized: dict[str, bool | None] = {}
    for key in _SMART_RUNTIME_OVERRIDE_KEYS:
        raw_value = source_overrides.get(key, None)
        if raw_value is not None and not isinstance(raw_value, bool):
            err = TypeError(
                "runtime_overrides values must be bool | None "
                f"(key={key!r}, got {type(raw_value).__name__})."
            )
            _log_runtime_override_failure(
                category="contract",
                worker_name=worker_name,
                request_context=request_context,
                error=err,
                details={
                    "stage": "runtime_overrides.value_type",
                    "key": key,
                    "value_type": type(raw_value).__name__,
                },
                transient=False,
            )
            raise err
        normalized[key] = raw_value
    return normalized


def _run_inference_worker(
    *,
    name: str,
    fn: Callable[[], Any],
    runtime_overrides: Mapping[str, bool | None] | None = None,
) -> tuple["threading.Event", _WorkerOutcome]:
    import contextvars
    import threading
    import time

    from apps.backend.runtime.live_preview import (
        live_preview_method,
        preview_interval_steps,
        preview_runtime_overrides,
    )

    request_context = threading.current_thread().name
    effective_runtime_overrides = _normalize_runtime_overrides(
        runtime_overrides=runtime_overrides,
        worker_name=name,
        request_context=request_context,
    )

    outcome = _WorkerOutcome()
    done = threading.Event()
    effective_preview_interval = int(preview_interval_steps(default=0))
    effective_preview_method = live_preview_method()
    worker_context = contextvars.copy_context()

    def _worker_body() -> None:
        from apps.backend.runtime.memory.smart_offload import smart_runtime_overrides

        try:
            with smart_runtime_overrides(**effective_runtime_overrides):
                with preview_runtime_overrides(
                    interval_steps=effective_preview_interval,
                    method=effective_preview_method,
                ):
                    outcome.sampling_start = time.perf_counter()
                    outcome.output = fn()
                    outcome.success = True
        except BaseException as exc:  # noqa: BLE001
            outcome.error = exc
        finally:
            outcome.sampling_end = time.perf_counter()
            done.set()

    def _worker() -> None:
        worker_context.run(_worker_body)

    threading.Thread(target=_worker, name=name, daemon=True).start()
    return done, outcome


def _iter_sampling_progress(
    *,
    done: "threading.Event",
    outcome: _WorkerOutcome | None = None,
    poll_interval_s: float = 0.12,
    expected_owner_token: str | None = None,
) -> Iterator[tuple[str, int, int | None, int, int, float | None, int, int | None, int, int]]:
    import time

    from apps.backend.core.state import state as backend_state

    def _has_block_progress(*, block_index: int, block_total: int) -> bool:
        return block_total > 0 and 0 < block_index < block_total

    def _has_vae_progress(*, phase: str, block_index: int, block_total: int) -> bool:
        return phase in {"encode", "decode"} and block_total > 0 and block_index > 0

    t0 = time.perf_counter()
    last_snapshot: tuple[int, int | None, int, int] = (-1, None, -1, -1)
    last_vae_snapshot = ("", -1, -1)
    last_owned_sampling_context: tuple[int, int | None, int, int] | None = None
    vae_phase_started_at: dict[str, float] = {}
    normalized_expected_owner_token = str(expected_owner_token or "").strip()
    while True:
        owner_token, step, total, block_index, block_total = backend_state.sampling_snapshot()
        (
            vae_owner_token,
            vae_phase,
            vae_block_index,
            vae_block_total,
            vae_sampling_step,
            vae_sampling_total,
        ) = backend_state.vae_progress_snapshot()

        owner_token = str(owner_token or "").strip()
        vae_owner_token = str(vae_owner_token or "").strip()
        sampling_owner_matches = not normalized_expected_owner_token or owner_token == normalized_expected_owner_token
        vae_owner_matches = not normalized_expected_owner_token or vae_owner_token == normalized_expected_owner_token
        if not sampling_owner_matches:
            step, total, block_index, block_total = 0, None, 0, 0
        if not vae_owner_matches:
            vae_phase, vae_block_index, vae_block_total, vae_sampling_step, vae_sampling_total = "", 0, 0, 0, None

        vae_phase = str(vae_phase or "").strip().lower()
        if vae_phase not in {"encode", "decode"}:
            vae_phase = ""
            vae_block_index = 0
            vae_block_total = 0
            vae_sampling_step = 0
            vae_sampling_total = None

        total = None if total is None else max(0, int(total))
        step = max(0, min(int(step), total if total is not None and total > 0 else int(step)))
        block_total = max(0, int(block_total))
        block_index = max(0, int(block_index))
        if block_total > 0:
            block_index = min(block_index, block_total)
        vae_block_total = max(0, int(vae_block_total))
        vae_block_index = max(0, int(vae_block_index))
        if vae_block_total > 0:
            vae_block_index = min(vae_block_index, vae_block_total)
        vae_sampling_total = None if vae_sampling_total is None else max(0, int(vae_sampling_total))
        vae_sampling_step = max(
            0,
            min(
                int(vae_sampling_step),
                vae_sampling_total if vae_sampling_total is not None and vae_sampling_total > 0 else int(vae_sampling_step),
            ),
        )

        done_now = done.is_set()
        at_full_block_boundary = (
            total is not None
            and block_total > 0
            and block_index >= block_total
            and step < total
        )
        emit_step = step
        emit_block_index = block_index
        emit_block_total = block_total
        promote_completed_step = at_full_block_boundary
        if promote_completed_step:
            # Full block-boundary snapshots represent a completed step before the
            # backend tick lands. Promote them to the corresponding completed-step
            # snapshot instead of suppressing progress.
            emit_step = min(total, step + 1)
            emit_block_index = 0
            emit_block_total = 0
        if sampling_owner_matches and (
            emit_step > 0
            or emit_block_index > 0
            or (total is not None and total > 0)
        ):
            last_owned_sampling_context = (emit_step, total, emit_block_index, emit_block_total)

        current_snapshot = (emit_step, total, emit_block_index, emit_block_total)
        should_emit = (
            (emit_step > 0 or emit_block_index > 0)
            and current_snapshot != last_snapshot
        )
        if should_emit:
            elapsed = time.perf_counter() - t0
            completed_units = float(emit_step)
            if _has_block_progress(block_index=emit_block_index, block_total=emit_block_total):
                completed_units += float(emit_block_index) / float(emit_block_total)
            if total is not None:
                completed_units = min(float(total), completed_units)
                eta = (
                    (elapsed * (float(total) - completed_units) / completed_units)
                    if completed_units > 0.0
                    else None
                )
            else:
                eta = None
            yield (
                "sampling",
                emit_step,
                total,
                emit_block_index,
                emit_block_total,
                eta,
                emit_step,
                total,
                emit_block_index,
                emit_block_total,
            )
            last_snapshot = current_snapshot

        current_vae_snapshot = (vae_phase, vae_block_index, vae_block_total)
        should_emit_vae = (
            _has_vae_progress(phase=vae_phase, block_index=vae_block_index, block_total=vae_block_total)
            and current_vae_snapshot != last_vae_snapshot
        )
        if should_emit_vae:
            now = time.perf_counter()
            if vae_phase not in vae_phase_started_at:
                vae_phase_started_at[vae_phase] = now
            elapsed_phase = max(0.0, now - vae_phase_started_at[vae_phase])
            completed_blocks = float(min(vae_block_total, vae_block_index))
            vae_eta = (
                (elapsed_phase * (float(vae_block_total) - completed_blocks) / completed_blocks)
                if completed_blocks > 0.0
                else None
            )
            vae_sampling_context = last_owned_sampling_context
            if vae_sampling_total is not None and vae_sampling_total > 0:
                vae_sampling_context = (
                    vae_sampling_step,
                    vae_sampling_total,
                    0,
                    0,
                )
            if vae_sampling_context is None:
                vae_sampling_context = (emit_step, total, emit_block_index, emit_block_total)
            vae_sampling_step_current, vae_sampling_total_current, vae_sampling_block_index, vae_sampling_block_total = (
                vae_sampling_context
            )
            yield (
                vae_phase,
                vae_block_index,
                vae_block_total,
                vae_block_index,
                vae_block_total,
                vae_eta,
                vae_sampling_step_current,
                vae_sampling_total_current,
                vae_sampling_block_index,
                vae_sampling_block_total,
            )
            last_vae_snapshot = current_vae_snapshot

        if done_now:
            break

        time.sleep(float(poll_interval_s))


def _iter_image_progress_events(
    *,
    done: "threading.Event",
    outcome: _WorkerOutcome,
    progress_owner_token: str | None,
    profile: _ImageProgressProfile,
) -> Iterator[ProgressEvent]:
    normalized_extra_data = dict(profile.extra_data or {})
    sampling_block_total_hint = 0
    encode_weight = float(profile.encode_weight)
    sampling_weight = float(profile.sampling_weight)
    decode_weight = float(profile.decode_weight)
    total_progress_base = encode_weight

    for (
        phase,
        phase_step,
        phase_total,
        phase_block_index,
        phase_block_total,
        phase_eta,
        sampling_step,
        sampling_total,
        sampling_block_index,
        sampling_block_total,
    ) in _iter_sampling_progress(
        done=done,
        outcome=outcome,
        expected_owner_token=progress_owner_token,
    ):
        if phase == "encode":
            if not profile.emit_encode_phase:
                continue
            encode_ratio = (
                min(float(phase_step), float(phase_total)) / float(phase_total)
                if phase_total > 0
                else 0.0
            )
            total_percent = (
                encode_weight * encode_ratio
                if sampling_total is not None and sampling_total > 0
                else None
            )
            data_payload: dict[str, Any] = {
                "block_index": int(phase_block_index),
                "block_total": int(phase_block_total),
                "total_phase": "encode",
                "total_percent": (float(total_percent) if total_percent is not None else None),
                "phase_step": int(phase_step),
                "phase_total_steps": int(phase_total),
                "phase_eta_seconds": (float(phase_eta) if phase_eta is not None else None),
            }
            data_payload.update(normalized_extra_data)
            yield ProgressEvent(
                stage="encoding",
                percent=(encode_ratio * 100.0 if phase_total > 0 else None),
                step=(int(phase_step) if phase_total > 0 else None),
                total_steps=(int(phase_total) if phase_total > 0 else None),
                eta_seconds=phase_eta,
                message=f"VAE encode block {phase_step}/{phase_total}",
                data=data_payload,
            )
            continue

        if phase == "sampling":
            if sampling_block_total > 0:
                sampling_block_total_hint = int(sampling_block_total)
            effective_sampling_block_total = (
                int(sampling_block_total)
                if sampling_block_total > 0
                else int(sampling_block_total_hint)
            )
            has_block_progress = 0 < sampling_block_index < sampling_block_total
            if sampling_total is not None and sampling_total > 0:
                completed_units = float(sampling_step)
                if has_block_progress:
                    completed_units += float(sampling_block_index) / float(sampling_block_total)
                sampling_ratio = min(float(sampling_total), completed_units) / float(sampling_total)
                progress_percent = sampling_ratio * 100.0
                pct = max(5.0, min(99.0, progress_percent))
                total_percent = total_progress_base + (sampling_weight * sampling_ratio)
                phase_step_blocks = int(phase_step)
                phase_total_blocks = int(phase_total)
                if effective_sampling_block_total > 0:
                    completed_sampling_steps = max(0, min(int(sampling_step), int(sampling_total)))
                    intra_step_blocks = max(0, min(int(sampling_block_index), int(effective_sampling_block_total)))
                    phase_total_blocks = int(sampling_total) * int(effective_sampling_block_total)
                    phase_step_blocks = min(
                        int(phase_total_blocks),
                        (int(completed_sampling_steps) * int(effective_sampling_block_total)) + int(intra_step_blocks),
                    )
                if has_block_progress:
                    message = (
                        f"Sampling step {min(sampling_step + 1, sampling_total)}/{sampling_total} "
                        f"(block {sampling_block_index}/{sampling_block_total})"
                    )
                else:
                    message = f"Sampling step {sampling_step}/{sampling_total}"
            else:
                pct = None
                total_percent = None
                phase_step_blocks = None
                phase_total_blocks = None
                if has_block_progress:
                    message = f"Sampling step {sampling_step} (block {sampling_block_index}/{sampling_block_total})"
                else:
                    message = f"Sampling step {sampling_step}"
            data_payload = {
                "block_index": int(sampling_block_index),
                "block_total": int(sampling_block_total),
                "total_phase": "sampling",
                "total_percent": (float(total_percent) if total_percent is not None else None),
                "phase_step": (int(phase_step_blocks) if phase_step_blocks is not None else None),
                "phase_total_steps": (int(phase_total_blocks) if phase_total_blocks is not None else None),
                "phase_eta_seconds": (float(phase_eta) if phase_eta is not None else None),
            }
            if profile.include_sampling_block_alias and effective_sampling_block_total > 0:
                data_payload["sampling_block_index"] = int(sampling_block_index)
                data_payload["sampling_block_total"] = int(effective_sampling_block_total)
            data_payload.update(normalized_extra_data)
            yield ProgressEvent(
                stage="sampling",
                percent=pct,
                step=sampling_step,
                total_steps=(sampling_total if sampling_total is not None and sampling_total > 0 else None),
                eta_seconds=phase_eta,
                message=message,
                data=data_payload,
            )
            continue

        if phase == "decode":
            decode_ratio = (
                min(float(phase_step), float(phase_total)) / float(phase_total)
                if phase_total > 0
                else 0.0
            )
            total_percent = (
                min(100.0, total_progress_base + sampling_weight + (decode_weight * decode_ratio))
                if sampling_total is not None and sampling_total > 0
                else None
            )
            data_payload = {
                "block_index": int(phase_block_index),
                "block_total": int(phase_block_total),
                "total_phase": "decode",
                "total_percent": (float(total_percent) if total_percent is not None else None),
                "phase_step": int(phase_step),
                "phase_total_steps": int(phase_total),
                "phase_eta_seconds": (float(phase_eta) if phase_eta is not None else None),
                "sampling_step": int(sampling_step),
                "sampling_total_steps": (
                    int(sampling_total) if sampling_total is not None and sampling_total > 0 else None
                ),
            }
            data_payload.update(normalized_extra_data)
            yield ProgressEvent(
                stage="decoding",
                percent=(decode_ratio * 100.0 if phase_total > 0 else None),
                step=(int(phase_step) if phase_total > 0 else None),
                total_steps=(int(phase_total) if phase_total > 0 else None),
                eta_seconds=phase_eta,
                message=f"VAE decode block {phase_step}/{phase_total}",
                data=data_payload,
            )


def _decode_generation_output(
    *,
    engine: Any,
    output: Any,
    task_label: str,
) -> tuple[list[object], float]:
    import gc
    import time

    import torch

    from apps.backend.runtime.memory import memory_management
    from apps.backend.runtime.memory.smart_offload_invariants import (
        enforce_smart_offload_post_decode_residency,
    )
    from apps.backend.runtime.processing.conditioners import decode_latent_batch
    from apps.backend.runtime.processing.datatypes import GenerationResult
    from apps.backend.runtime.pipeline_stages.image_io import latents_to_pil

    decoded_images: Any | None = None
    latents: Any = None
    metadata: dict[str, Any] = {}
    metadata_error: RuntimeError | None = None
    decode_engine = engine
    if isinstance(output, GenerationResult):
        latents = output.samples
        decoded_images = output.decoded
        if getattr(output, "decode_engine", None) is not None:
            decode_engine = output.decode_engine
        if not isinstance(output.metadata, dict):
            metadata_error = RuntimeError(
                f"{task_label} pipeline returned metadata as {type(output.metadata).__name__}; expected dict."
            )
        else:
            metadata = output.metadata
    else:
        latents = output
        decoded_images = None

    raw_cache_hit = metadata.get("conditioning_cache_hit", False)
    if not isinstance(raw_cache_hit, bool):
        metadata_error = RuntimeError(
            f"{task_label} pipeline metadata['conditioning_cache_hit'] must be bool; got {type(raw_cache_hit).__name__}."
        )

    decode_start = time.perf_counter()
    try:
        if metadata_error is None:
            if decoded_images is not None:
                if isinstance(decoded_images, torch.Tensor):
                    images = latents_to_pil(decoded_images)
                elif isinstance(decoded_images, list):
                    try:
                        from PIL import Image as _PILImage

                        if not all(isinstance(img, _PILImage.Image) for img in decoded_images):
                            raise TypeError("decoded images are not PIL.Image.Image")
                    except Exception as exc:
                        raise RuntimeError(
                            f"{task_label} pipeline returned decoded images, but they are not a PIL image list"
                        ) from exc
                    images = decoded_images
                else:
                    raise RuntimeError(
                        f"{task_label} pipeline returned decoded images, expected torch.Tensor or list[PIL.Image.Image]"
                    )
            else:
                if not isinstance(latents, torch.Tensor):
                    raise RuntimeError(
                        f"{task_label} pipeline returned {type(latents).__name__}; expected torch.Tensor (latents)"
                    )
                gc.collect()
                memory_management.manager.soft_empty_cache(force=True)
                # Intentional exception: this egress path materializes decoded tensors on CPU
                # for immediate PIL conversion (`latents_to_pil`), independent of model offload policy.
                cpu_decode_target = memory_management.manager.cpu_device
                decoded = decode_latent_batch(
                    decode_engine,
                    latents,
                    target_device=cpu_decode_target,
                    stage=f"{task_label}.decode(pre)",
                )
                images = latents_to_pil(decoded)
    finally:
        enforce_smart_offload_post_decode_residency(
            decode_engine,
            stage=f"{task_label}.decode",
        )

    if metadata_error is not None:
        raise metadata_error

    decode_end = time.perf_counter()
    decode_ms = max(0.0, (decode_end - decode_start) * 1000.0)
    return list(images), decode_ms


def _build_common_info(
    *,
    engine_id: str,
    task: str,
    proc: Any,
    seed: int,
    all_seeds: Sequence[int],
    extra_params: Mapping[str, object],
    timings_ms: Mapping[str, float],
    mode_info: Mapping[str, object] | None = None,
) -> dict[str, object]:
    info: dict[str, object] = {
        "engine": str(engine_id),
        "task": str(task),
        "width": int(getattr(proc, "width", 0) or 0),
        "height": int(getattr(proc, "height", 0) or 0),
        "steps": int(getattr(proc, "steps", 0) or 0),
        "guidance_scale": float(getattr(proc, "guidance_scale", 0.0) or 0.0),
        "sampler": (str(getattr(proc, "sampler_name", "")).strip() or None),
        "scheduler": (str(getattr(proc, "scheduler", "")).strip() or None),
        "seed": int(seed),
        "all_seeds": [int(s) for s in (all_seeds or [])],
    }

    prompt = str(getattr(proc, "primary_prompt", "") or "").strip()
    negative = str(getattr(proc, "primary_negative_prompt", "") or "").strip()
    if prompt:
        info["prompt"] = prompt
    if negative:
        info["negative_prompt"] = negative
    if extra_params:
        info["extra"] = dict(extra_params)
    if timings_ms:
        info["timings_ms"] = dict(timings_ms)
    if mode_info:
        info.update(dict(mode_info))
    return info
