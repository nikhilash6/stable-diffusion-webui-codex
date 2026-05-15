"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Txt2img entry point using the staged pipeline runner.
Delegates latent generation to `Txt2ImgPipelineRunner` and provides a canonical event-emitting wrapper used by engines/orchestrator.
The wrapper executes sampling + decode + post-cleanup inside the same worker-thread envelope so model residency/offload policies remain single-owner per job.
Worker-thread smart runtime overrides are propagated through `_image_streaming._run_inference_worker(...)`, cleanup hooks always run in a `finally` block, and the wrapper seeds a per-run progress-owner token before worker-side VAE/sampling progress begins.

Symbols (top-level; keep in sync; no ghosts):
- `_logger` (constant): Module logger for the txt2img use case.
- `_RUNNER` (constant): Singleton `Txt2ImgPipelineRunner` instance.
- `generate_txt2img` (function): Runs the txt2img pipeline runner and returns a `GenerationResult` (samples + optional decoded output).
- `run_txt2img` (function): Canonical txt2img mode wrapper (phase-aware progress polling, decode, and result events).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from typing import Any, Iterator, Sequence

from apps.backend.runtime.processing.datatypes import GenerationResult
from apps.backend.runtime.processing.models import CodexProcessingTxt2Img
from apps.backend.runtime.diagnostics.pipeline_debug import pipeline_trace
from apps.backend.runtime.diagnostics.timeline import timeline
from .txt2img_pipeline.runner import Txt2ImgPipelineRunner


_logger = get_backend_logger("backend.use_cases.txt2img")
_RUNNER = Txt2ImgPipelineRunner()


@pipeline_trace
def generate_txt2img(
    processing,
    conditioning,
    unconditional_conditioning,
    seeds: Sequence[int],
    subseeds: Sequence[int],
    subseed_strength: float,
    prompts: Sequence[str],
) -> GenerationResult:
    if not isinstance(processing, CodexProcessingTxt2Img):
        raise TypeError("generate_txt2img expects CodexProcessingTxt2Img")

    timeline_enabled = bool(timeline.enabled)
    model_engine_id = str(getattr(getattr(processing, "sd_model", None), "engine_id", "unknown") or "unknown")
    capture_name = f"txt2img:{model_engine_id}"
    with timeline.capture(name=capture_name) as capture:
        result = _RUNNER.run(
            processing=processing,
            conditioning_data=conditioning,
            unconditional_data=unconditional_conditioning,
            seeds=seeds,
            subseeds=subseeds,
            subseed_strength=subseed_strength,
            prompts=prompts,
        )
    if timeline_enabled:
        if capture is None:
            raise RuntimeError(
                "CODEX_TIMELINE is enabled but timeline capture context did not produce a capture object."
            )
        if len(capture.events) == 0:
            raise RuntimeError(
                "CODEX_TIMELINE is enabled but txt2img captured zero timeline events. "
                "Ensure timeline_node instrumentation remains active."
            )
    return result


def run_txt2img(*, engine, request) -> Iterator["InferenceEvent"]:
    """Run txt2img as a canonical event stream.

    This wrapper owns the mode-level concerns (seed defaults, progress polling, decode + result packaging).
    Engines should delegate here rather than implementing per-mode pipelines.
    """

    import json

    from apps.backend.core.requests import InferenceEvent, ResultEvent, Txt2ImgRequest
    from apps.backend.engines.util.adapters import build_txt2img_processing
    from apps.backend.runtime.text_processing import (
        clear_last_extra_generation_params,
        snapshot_last_extra_generation_params,
    )

    from ._image_streaming import (
        _build_common_info,
        _decode_generation_output,
        _ImageProgressProfile,
        _iter_image_progress_events,
        _resolve_seed_plan,
        _resolve_progress_owner_token,
        _run_inference_worker,
        _seed_progress_owner_token,
    )

    if not isinstance(request, Txt2ImgRequest):
        raise TypeError("run_txt2img expects Txt2ImgRequest")

    engine.ensure_loaded()

    proc = build_txt2img_processing(request)
    proc.sd_model = engine
    import threading

    task_context = str(threading.current_thread().name or "").strip() or "unknown-thread"
    setattr(proc, "_codex_pipeline_mode", "txt2img")
    task_id: str | None = None
    marker = "-task-"
    if marker in task_context:
        candidate = task_context.split(marker, 1)[1].strip()
        if candidate:
            task_id = candidate
    if task_id is not None:
        setattr(proc, "_codex_task_id", task_id)
        setattr(proc, "_codex_correlation_id", task_id)
        setattr(proc, "_codex_hires_correlation_id", task_id)
        setattr(proc, "_codex_correlation_source", "task_id")
    progress_owner_token = _resolve_progress_owner_token(task_context=task_context, task_id=task_id)
    setattr(proc, "_codex_progress_owner_token", progress_owner_token)
    _seed_progress_owner_token(progress_owner_token=progress_owner_token)

    base_seed, seeds, subseeds, subseed_strength = _resolve_seed_plan(
        seed=getattr(request, "seed", None),
        batch_total=proc.batch_total,
    )
    proc.seed = base_seed
    proc.seeds = list(seeds)
    proc.subseed = -1
    proc.subseeds = list(subseeds)

    prompts = list(getattr(proc, "prompts", []) or []) or [proc.prompt]
    smart_flags = {
        "smart_offload": bool(getattr(proc, "smart_offload", False)),
        "smart_fallback": bool(getattr(proc, "smart_fallback", False)),
        "smart_cache": bool(getattr(proc, "smart_cache", False)),
    }

    def _generate() -> dict[str, object]:
        import time

        cleanup_targets: list[Any] = [engine]
        sampling_start = 0.0
        sampling_end = 0.0
        active_decode_engine: Any = engine

        try:
            clear_last_extra_generation_params()
            sampling_start = time.perf_counter()
            output = generate_txt2img(
                processing=proc,
                conditioning=None,
                unconditional_conditioning=None,
                seeds=seeds,
                subseeds=subseeds,
                subseed_strength=subseed_strength,
                prompts=prompts,
            )
            sampling_end = time.perf_counter()

            output_decode_engine = getattr(output, "decode_engine", None)
            active_decode_engine = output_decode_engine if output_decode_engine is not None else getattr(proc, "sd_model", None)
            if active_decode_engine is None:
                active_decode_engine = engine
            elif active_decode_engine is not engine:
                _logger.info("txt2img decode will use active pipeline model instance (swap/refiner path).")
            if active_decode_engine is not None and not any(existing is active_decode_engine for existing in cleanup_targets):
                cleanup_targets.append(active_decode_engine)

            images, decode_ms = _decode_generation_output(
                engine=active_decode_engine,
                output=output,
                task_label="txt2img",
            )

            all_seeds = list(getattr(proc, "all_seeds", []) or []) or list(seeds)
            seed_value = int(all_seeds[0]) if all_seeds else int(base_seed)

            extra_params: dict[str, object] = {}
            try:
                extra_params.update(snapshot_last_extra_generation_params())
                extra_params.update(getattr(proc, "extra_generation_params", {}) or {})
            except Exception:  # noqa: BLE001
                extra_params = getattr(proc, "extra_generation_params", {}) or {}

            timings: dict[str, float] = {
                "sampling_ms": max(0.0, (sampling_end - sampling_start) * 1000.0),
                "decode_ms": float(decode_ms),
            }

            mode_info: dict[str, object] = {}
            if bool(getattr(getattr(proc, "hires", None), "enabled", False)):
                try:
                    mode_info["hires"] = getattr(proc, "hires", None).as_dict()
                except Exception:  # noqa: BLE001
                    pass
                effective_hires_sampling = getattr(proc, "_codex_effective_hires_sampling", None)
                if isinstance(effective_hires_sampling, dict) and effective_hires_sampling:
                    mode_info["effective_hires_sampling"] = dict(effective_hires_sampling)

            info = _build_common_info(
                engine_id=engine.engine_id,
                task="txt2img",
                proc=proc,
                seed=seed_value,
                all_seeds=all_seeds,
                extra_params=extra_params,
                timings_ms=timings,
                mode_info=mode_info,
            )
            return {"images": images, "info": json.dumps(info)}
        finally:
            processing_model = getattr(proc, "sd_model", None)
            if processing_model is not None and not any(existing is processing_model for existing in cleanup_targets):
                cleanup_targets.append(processing_model)
            for target in cleanup_targets:
                post_cleanup = getattr(target, "_post_txt2img_cleanup", None)
                if callable(post_cleanup):
                    post_cleanup()

    done, outcome = _run_inference_worker(
        name=f"{engine.engine_id}-txt2img-worker",
        fn=_generate,
        runtime_overrides=smart_flags,
    )

    yield from _iter_image_progress_events(
        done=done,
        outcome=outcome,
        progress_owner_token=progress_owner_token,
        profile=_ImageProgressProfile(
            encode_weight=0.0,
            sampling_weight=90.0,
            decode_weight=10.0,
            emit_encode_phase=False,
        ),
    )

    if outcome.error is not None:
        raise outcome.error

    payload = outcome.output
    if not isinstance(payload, dict):
        raise RuntimeError(
            "txt2img worker returned invalid payload type; expected dict with 'images' and 'info'. "
            f"Got {type(payload).__name__}."
        )
    images = payload.get("images")
    info = payload.get("info")
    if not isinstance(images, list):
        raise RuntimeError("txt2img worker payload field 'images' must be list.")
    if not isinstance(info, str):
        raise RuntimeError("txt2img worker payload field 'info' must be JSON string.")
    yield ResultEvent(payload={"images": images, "info": info})
