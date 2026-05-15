"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend-owned image automation loop orchestration for image tabs.
Resolves per-iteration prompt, seed, wildcard expansion, folder-backed img2img init images, folder-backed IP-Adapter reference images, delay,
and `after_current` stop boundaries while delegating concrete image execution back to the canonical txt2img/img2img request owners.

Symbols (top-level; keep in sync; no ghosts):
- `ImageAutomationImmediateCancel` (exception): Signals an immediate automation cancellation boundary.
- `AutomationIterationPlan` (dataclass): Scalar per-iteration execution metadata used for progress and iteration events.
- `AutomationRunResult` (dataclass): Final last-iteration result plus terminal automation summary.
- `run_image_automation` (function): Executes the backend-owned automation loop around a shared image-execution callback.
"""

from __future__ import annotations

import base64
import io
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image, ImageOps

from apps.backend.core.requests import ImageAutomationRequest
from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.runtime.text_processing import expand_wildcards

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".avif", ".jxl"}
_PROMPT_PREVIEW_LIMIT = 160
_DELAY_POLL_SECONDS = 0.1


class ImageAutomationImmediateCancel(RuntimeError):
    """Raised when automation receives an immediate cancellation signal."""


@dataclass(frozen=True)
class AutomationIterationPlan:
    iteration_index: int
    iteration_total: int | None
    loop_mode: str
    requested_seed: int | None
    prompt_preview: str
    source_label: str | None


@dataclass(frozen=True)
class AutomationRunResult:
    last_result: dict[str, Any]
    automation_summary: dict[str, Any]


def _prompt_field_name(mode: str) -> str:
    if mode == "txt2img":
        return "prompt"
    if mode == "img2img":
        return "img2img_prompt"
    raise ValueError(f"Unsupported automation mode {mode!r}.")


def _seed_field_name(mode: str) -> str:
    if mode == "txt2img":
        return "seed"
    if mode == "img2img":
        return "img2img_seed"
    raise ValueError(f"Unsupported automation mode {mode!r}.")


def _img2img_init_field_name(mode: str) -> str:
    if mode != "img2img":
        raise ValueError(f"Init-image automation is unsupported for mode {mode!r}.")
    return "img2img_init_image"


def _extras_field_name(mode: str) -> str:
    if mode == "txt2img":
        return "extras"
    if mode == "img2img":
        return "img2img_extras"
    raise ValueError(f"Unsupported automation mode {mode!r}.")


def _coerce_cancel_mode(raw_mode: object) -> str:
    text = str(raw_mode or "").strip().lower()
    return text if text in {"immediate", "after_current"} else "immediate"


def _prompt_preview(text: object) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= _PROMPT_PREVIEW_LIMIT:
        return raw
    return raw[: _PROMPT_PREVIEW_LIMIT - 1].rstrip() + "…"


def _seed_from_info(info_obj: object, *, fallback: int | None) -> int | None:
    if isinstance(info_obj, Mapping):
        seed_value = info_obj.get("seed")
        if isinstance(seed_value, bool):
            return fallback
        if isinstance(seed_value, int):
            return seed_value
        if isinstance(seed_value, float) and seed_value.is_integer():
            return int(seed_value)
        if isinstance(seed_value, str):
            token = seed_value.strip()
            if token:
                try:
                    return int(token, 10)
                except Exception:
                    return fallback
    return fallback


def _repo_relative_label(path: Path) -> str:
    resolved = path.resolve(strict=False)
    root = get_repo_root().resolve(strict=False)
    try:
        return resolved.relative_to(root).as_posix()
    except Exception:
        return resolved.as_posix()


def _prompt_entries(request: ImageAutomationRequest) -> list[str]:
    prompt_source = request.prompt_source
    if prompt_source.kind == "current":
        return []
    text = str(prompt_source.text or "")
    entries = [line.strip() for line in text.splitlines() if line.strip()]
    if not entries:
        raise ValueError("prompt_source.text must include at least one non-empty prompt line.")
    return entries


def _compose_prompt(*, base_prompt: str, prompt_entry: str, insert_position: str) -> str:
    if insert_position == "replace":
        return prompt_entry
    if insert_position == "prepend":
        return " ".join(part for part in (prompt_entry.strip(), base_prompt.strip()) if part).strip()
    if insert_position == "append":
        return " ".join(part for part in (base_prompt.strip(), prompt_entry.strip()) if part).strip()
    raise ValueError(f"Unsupported prompt insert_position {insert_position!r}.")


def _resolved_iteration_prompt(
    *,
    request: ImageAutomationRequest,
    base_prompt: str,
    prompt_entries: list[str],
    iteration_index: int,
    rng: random.Random,
) -> str:
    prompt_source = request.prompt_source
    prompt = base_prompt
    if prompt_source.kind == "list":
        prompt_entry = prompt_entries[(iteration_index - 1) % len(prompt_entries)]
        prompt = _compose_prompt(
            base_prompt=base_prompt,
            prompt_entry=prompt_entry,
            insert_position=prompt_source.insert_position,
        )
    if prompt_source.wildcard_mode == "expand":
        wildcard_root = str(prompt_source.wildcard_root or "")
        if not wildcard_root:
            raise ValueError("Wildcard expansion requires a non-empty wildcard_root.")
        prompt = expand_wildcards(prompt, wildcard_root=wildcard_root, rng=rng)
    return prompt


def _requested_seed(*, request: ImageAutomationRequest, base_seed: int | None, iteration_index: int, rng: random.Random) -> int | None:
    seed_policy = request.seed_policy
    if seed_policy.mode == "random":
        return rng.randrange(0, 2**31)
    if seed_policy.mode == "fixed":
        return int(base_seed) if base_seed is not None else None
    if seed_policy.mode == "increment":
        if base_seed is None:
            return None
        return int(base_seed) + ((iteration_index - 1) * int(seed_policy.increment_step))
    raise ValueError(f"Unsupported seed_policy.mode {seed_policy.mode!r}.")


def _sort_folder_entries(paths: list[Path], *, sort_by: str) -> list[Path]:
    if sort_by == "name":
        return sorted(paths, key=lambda item: item.name.lower())
    if sort_by == "size":
        return sorted(paths, key=lambda item: (item.stat().st_size, item.name.lower()))
    if sort_by == "created_at":
        return sorted(paths, key=lambda item: (item.stat().st_ctime, item.name.lower()))
    if sort_by == "modified_at":
        return sorted(paths, key=lambda item: (item.stat().st_mtime, item.name.lower()))
    raise ValueError(f"Unsupported folder sort field {sort_by!r}.")


def _selected_folder_images(
    *,
    folder_path: str,
    selection_mode: str,
    selection_count: int | None,
    order: str,
    sort_by: str,
    rng: random.Random,
) -> list[Path]:
    root = Path(folder_path).expanduser().resolve(strict=False)
    if not root.exists():
        raise ValueError(f"Automation source folder does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Automation source folder is not a directory: {root}")

    files = [
        entry
        for entry in root.iterdir()
        if entry.is_file() and entry.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not files:
        raise ValueError(f"Automation source folder has no supported images: {root}")

    if order == "sorted":
        ordered = _sort_folder_entries(files, sort_by=sort_by)
    elif order == "random":
        ordered = list(files)
        rng.shuffle(ordered)
    else:
        raise ValueError(f"Unsupported folder order {order!r}.")

    if selection_mode == "all":
        return ordered
    if selection_mode == "count":
        if selection_count is None or selection_count < 1:
            raise ValueError("Folder selection_mode='count' requires count >= 1.")
        if selection_count > len(ordered):
            raise ValueError(
                f"Folder selection count {selection_count} exceeds available images {len(ordered)} in {root}."
            )
        return ordered[:selection_count]
    raise ValueError(f"Unsupported folder selection_mode {selection_mode!r}.")


def _refresh_random_folder_cycle(
    *,
    selected_folder_images: list[Path] | None,
    iteration_index: int,
    folder_path: str | None,
    selection_mode: str | None,
    selection_count: int | None,
    order: str,
    sort_by: str | None,
    rng: random.Random,
) -> None:
    if selected_folder_images is None or order != "random" or iteration_index <= 1:
        return
    if len(selected_folder_images) < 1:
        raise ValueError("Random folder automation requires at least one selected image.")
    if (iteration_index - 1) % len(selected_folder_images) != 0:
        return
    if not folder_path:
        raise ValueError("Random folder automation requires a non-empty folder path.")
    refreshed = _selected_folder_images(
        folder_path=folder_path,
        selection_mode=str(selection_mode or "all"),
        selection_count=selection_count,
        order=order,
        sort_by=str(sort_by or "name"),
        rng=rng,
    )
    selected_folder_images[:] = refreshed


def _encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _target_size_from_payload(payload: Mapping[str, Any]) -> tuple[int, int] | None:
    width = payload.get("img2img_width")
    height = payload.get("img2img_height")
    if isinstance(width, bool) or isinstance(height, bool):
        return None
    if not isinstance(width, int) or not isinstance(height, int):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _materialize_folder_image(*, path: Path, use_crop: bool, payload: Mapping[str, Any]) -> str:
    with Image.open(path) as image_file:
        image = image_file.convert("RGB")
        if use_crop:
            target_size = _target_size_from_payload(payload)
            if target_size is not None:
                resampling = getattr(Image, "Resampling", Image)
                image = ImageOps.fit(image, target_size, method=resampling.LANCZOS)
        return _encode_image(image)


def _ip_adapter_payload(*, mode: str, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    extras = payload.get(_extras_field_name(mode))
    if not isinstance(extras, Mapping):
        return None
    ip_adapter = extras.get("ip_adapter")
    if not isinstance(ip_adapter, Mapping):
        return None
    return ip_adapter


def _resolve_img2img_source(
    *,
    request: ImageAutomationRequest,
    base_template: Mapping[str, Any],
    payload: dict[str, Any],
    iteration_index: int,
    selected_folder_images: list[Path] | None,
    rng: random.Random,
) -> str | None:
    init_source = request.init_source
    if init_source is None:
        raise ValueError("img2img automation requires init_source.")
    init_field = _img2img_init_field_name(request.mode)
    if init_source.kind == "uploaded_current":
        current_image = base_template.get(init_field)
        if not isinstance(current_image, str) or not current_image.strip():
            raise ValueError("img2img automation init_source.kind='uploaded_current' requires template.img2img_init_image.")
        payload[init_field] = current_image
        return None
    if init_source.kind != "server_folder":
        raise ValueError(f"Unsupported img2img init_source.kind {init_source.kind!r}.")
    if not selected_folder_images:
        raise ValueError("img2img server_folder automation requires at least one selected image.")
    _refresh_random_folder_cycle(
        selected_folder_images=selected_folder_images,
        iteration_index=iteration_index,
        folder_path=init_source.folder_path,
        selection_mode=init_source.selection_mode,
        selection_count=init_source.count,
        order=str(init_source.order or "sorted"),
        sort_by=init_source.sort_by,
        rng=rng,
    )
    source_path = selected_folder_images[(iteration_index - 1) % len(selected_folder_images)]
    payload[init_field] = _materialize_folder_image(
        path=source_path,
        use_crop=bool(init_source.use_crop),
        payload=payload,
    )
    return _repo_relative_label(source_path)


def _selected_ip_adapter_folder_images(
    *,
    request: ImageAutomationRequest,
    base_template: Mapping[str, Any],
    rng: random.Random,
) -> list[Path] | None:
    ip_adapter = _ip_adapter_payload(mode=request.mode, payload=base_template)
    if ip_adapter is None:
        return None
    source = ip_adapter.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("IP-Adapter automation requires a source object when enabled.")
    if str(source.get("kind") or "") != "server_folder":
        return None
    folder_path = str(source.get("folder_path") or "").strip()
    if not folder_path:
        raise ValueError("IP-Adapter server_folder automation requires source.folder_path.")
    selection_mode = str(source.get("selection_mode") or "all")
    count = source.get("count")
    if count is not None and not isinstance(count, int):
        raise ValueError("IP-Adapter server_folder automation requires integer source.count when provided.")
    return _selected_folder_images(
        folder_path=folder_path,
        selection_mode=selection_mode,
        selection_count=count,
        order=str(source.get("order") or "sorted"),
        sort_by=str(source.get("sort_by") or "name"),
        rng=rng,
    )


def _resolve_ip_adapter_source(
    *,
    request: ImageAutomationRequest,
    payload: dict[str, Any],
    iteration_index: int,
    selected_folder_images: list[Path] | None,
    rng: random.Random,
) -> str | None:
    extras_key = _extras_field_name(request.mode)
    extras = payload.get(extras_key)
    if not isinstance(extras, dict):
        return None
    ip_adapter = extras.get("ip_adapter")
    if not isinstance(ip_adapter, dict) or not bool(ip_adapter.get("enabled", False)):
        return None
    source = ip_adapter.get("source")
    if not isinstance(source, dict):
        raise ValueError("IP-Adapter automation requires source object.")
    kind = str(source.get("kind") or "").strip()
    if kind == "uploaded":
        reference_image_data = source.get("reference_image_data")
        if not isinstance(reference_image_data, str) or not reference_image_data.strip():
            raise ValueError("IP-Adapter uploaded automation source requires reference_image_data.")
        return None
    if kind == "same_as_init":
        if request.mode != "img2img":
            raise ValueError("IP-Adapter source.kind='same_as_init' is only valid for img2img automation.")
        init_field = _img2img_init_field_name(request.mode)
        init_image_data = payload.get(init_field)
        if not isinstance(init_image_data, str) or not init_image_data.strip():
            raise ValueError("IP-Adapter same_as_init requires the current iteration init image.")
        ip_adapter["source"] = {
            "kind": "uploaded",
            "reference_image_data": init_image_data,
        }
        return "Same as init image"
    if kind != "server_folder":
        raise ValueError(f"Unsupported IP-Adapter automation source.kind {kind!r}.")
    if not selected_folder_images:
        raise ValueError("IP-Adapter server_folder automation requires at least one selected image.")
    _refresh_random_folder_cycle(
        selected_folder_images=selected_folder_images,
        iteration_index=iteration_index,
        folder_path=str(source.get("folder_path") or "").strip(),
        selection_mode=str(source.get("selection_mode") or "all"),
        selection_count=source.get("count") if isinstance(source.get("count"), int) else None,
        order=str(source.get("order") or "sorted"),
        sort_by=str(source.get("sort_by") or "name"),
        rng=rng,
    )
    source_path = selected_folder_images[(iteration_index - 1) % len(selected_folder_images)]
    ip_adapter["source"] = {
        "kind": "uploaded",
        "reference_image_data": _materialize_folder_image(
            path=source_path,
            use_crop=False,
            payload=payload,
        ),
    }
    return _repo_relative_label(source_path)


def _sleep_with_cancel(
    *,
    delay_ms: int,
    cancel_snapshot: Callable[[], tuple[bool, object]],
) -> str | None:
    remaining = max(0.0, float(delay_ms) / 1000.0)
    while remaining > 0.0:
        cancel_requested, cancel_mode = cancel_snapshot()
        if cancel_requested:
            if _coerce_cancel_mode(cancel_mode) == "after_current":
                return "cancelled_after_current"
            raise ImageAutomationImmediateCancel("cancelled")
        interval = min(_DELAY_POLL_SECONDS, remaining)
        time.sleep(interval)
        remaining -= interval
    return None


def _progress_event(plan: AutomationIterationPlan) -> dict[str, Any]:
    return {
        "type": "progress",
        "stage": "automation_iteration",
        "message": f"Starting automation iteration {plan.iteration_index}",
        "data": {
            "loop_mode": plan.loop_mode,
            "iteration_index": plan.iteration_index,
            "iteration_total": plan.iteration_total,
            "current_seed": plan.requested_seed,
            "current_prompt_preview": plan.prompt_preview,
        },
    }


def _iteration_event(*, plan: AutomationIterationPlan, result: Mapping[str, Any], info_obj: object) -> dict[str, Any]:
    return {
        "type": "automation_iteration",
        "iteration_index": plan.iteration_index,
        "images": list(result.get("images", []) or []),
        "info": info_obj,
        "seed": _seed_from_info(info_obj, fallback=plan.requested_seed),
        "prompt_preview": plan.prompt_preview,
        "source_label": plan.source_label,
    }


def run_image_automation(
    request: ImageAutomationRequest,
    *,
    execute_iteration: Callable[[dict[str, Any]], dict[str, Any]],
    emit_iteration: Callable[[dict[str, Any]], None],
    emit_progress: Callable[[dict[str, Any]], None],
    cancel_snapshot: Callable[[], tuple[bool, object]],
    rng: random.Random | None = None,
) -> AutomationRunResult:
    if request.mode not in {"txt2img", "img2img"}:
        raise ValueError(f"Unsupported automation mode {request.mode!r}.")

    generator = rng if rng is not None else random.Random()
    base_template = deepcopy(dict(request.template or {}))
    prompt_field = _prompt_field_name(request.mode)
    seed_field = _seed_field_name(request.mode)
    base_prompt = str(base_template.get(prompt_field) or "")
    base_seed_raw = base_template.get(seed_field)
    base_seed = int(base_seed_raw) if isinstance(base_seed_raw, int) and not isinstance(base_seed_raw, bool) else None
    if (base_seed is None or base_seed < 0) and request.seed_policy.mode in {"fixed", "increment"}:
        base_seed = generator.randrange(0, 2**31)
    prompt_entries = _prompt_entries(request)
    selected_init_images: list[Path] | None = None
    if request.mode == "img2img":
        init_source = request.init_source
        if init_source is None:
            raise ValueError("img2img automation requires init_source.")
        if init_source.kind == "server_folder":
            if not init_source.folder_path:
                raise ValueError("img2img server_folder automation requires init_source.folder_path.")
            selected_init_images = _selected_folder_images(
                folder_path=init_source.folder_path,
                selection_mode=str(init_source.selection_mode or "all"),
                selection_count=init_source.count,
                order=str(init_source.order or "sorted"),
                sort_by=str(init_source.sort_by or "name"),
                rng=generator,
            )
    selected_ip_adapter_images = _selected_ip_adapter_folder_images(
        request=request,
        base_template=base_template,
        rng=generator,
    )

    if request.loop.mode == "count":
        total_iterations = request.loop.count
        if total_iterations is None:
            iteration_lengths = [
                len(images)
                for images in (selected_init_images, selected_ip_adapter_images)
                if images is not None
            ]
            if not iteration_lengths:
                raise ValueError(
                    "automation loop.mode='count' requires loop.count unless a folder-backed init source or IP-Adapter source selects all images."
                )
            total_iterations = max(iteration_lengths)
    else:
        total_iterations = None

    attempted_iterations = 0
    successful_iterations = 0
    last_result: dict[str, Any] | None = None
    last_error: Exception | None = None
    summary_seed: int | None = None
    summary_prompt_preview = ""
    summary_source_label: str | None = None
    stop_reason = "count_completed" if request.loop.mode == "count" else "cancelled_after_current"

    while True:
        if total_iterations is not None and attempted_iterations >= total_iterations:
            stop_reason = "count_completed"
            break

        cancel_requested, cancel_mode = cancel_snapshot()
        if cancel_requested:
            if successful_iterations == 0:
                raise ImageAutomationImmediateCancel("cancelled")
            if _coerce_cancel_mode(cancel_mode) == "after_current":
                stop_reason = "cancelled_after_current"
                break
            raise ImageAutomationImmediateCancel("cancelled")

        iteration_index = attempted_iterations + 1
        payload = deepcopy(base_template)
        resolved_prompt = _resolved_iteration_prompt(
            request=request,
            base_prompt=base_prompt,
            prompt_entries=prompt_entries,
            iteration_index=iteration_index,
            rng=generator,
        )
        payload[prompt_field] = resolved_prompt
        requested_seed = _requested_seed(
            request=request,
            base_seed=base_seed,
            iteration_index=iteration_index,
            rng=generator,
        )
        payload[seed_field] = -1 if requested_seed is None else int(requested_seed)
        source_labels: list[str] = []
        if request.mode == "img2img":
            init_source_label = _resolve_img2img_source(
                request=request,
                base_template=base_template,
                payload=payload,
                iteration_index=iteration_index,
                selected_folder_images=selected_init_images,
                rng=generator,
            )
            if init_source_label:
                source_labels.append(f"Init: {init_source_label}")
        ip_adapter_source_label = _resolve_ip_adapter_source(
            request=request,
            payload=payload,
            iteration_index=iteration_index,
            selected_folder_images=selected_ip_adapter_images,
            rng=generator,
        )
        if ip_adapter_source_label:
            source_labels.append(f"IP-Adapter: {ip_adapter_source_label}")
        source_label = " | ".join(source_labels) if source_labels else None

        plan = AutomationIterationPlan(
            iteration_index=iteration_index,
            iteration_total=total_iterations,
            loop_mode=request.loop.mode,
            requested_seed=requested_seed,
            prompt_preview=_prompt_preview(resolved_prompt),
            source_label=source_label,
        )
        emit_progress(_progress_event(plan))
        attempted_iterations = iteration_index

        try:
            iteration_result = execute_iteration(deepcopy(payload))
        except ImageAutomationImmediateCancel:
            raise
        except Exception as exc:
            last_error = exc
            if last_result is None:
                raise
            if request.loop.stop_on_error:
                raise
            emit_progress(
                {
                    "type": "progress",
                    "stage": "automation_iteration_error",
                    "message": str(exc),
                    "data": {
                        "iteration_index": plan.iteration_index,
                        "iteration_total": plan.iteration_total,
                        "current_seed": plan.requested_seed,
                        "current_prompt_preview": plan.prompt_preview,
                    },
                }
            )
            if total_iterations is not None and attempted_iterations >= total_iterations:
                break
            delay_stop_reason = _sleep_with_cancel(delay_ms=request.loop.delay_ms, cancel_snapshot=cancel_snapshot)
            if delay_stop_reason is not None:
                stop_reason = delay_stop_reason
                break
            continue

        info_obj = iteration_result.get("info")
        emit_iteration(_iteration_event(plan=plan, result=iteration_result, info_obj=info_obj))
        successful_iterations = iteration_index
        last_result = dict(iteration_result)
        summary_seed = _seed_from_info(info_obj, fallback=plan.requested_seed)
        summary_prompt_preview = plan.prompt_preview
        summary_source_label = plan.source_label

        if total_iterations is not None and attempted_iterations >= total_iterations:
            stop_reason = "count_completed"
            break

        cancel_requested, cancel_mode = cancel_snapshot()
        if cancel_requested and _coerce_cancel_mode(cancel_mode) == "after_current":
            stop_reason = "cancelled_after_current"
            break
        if cancel_requested:
            raise ImageAutomationImmediateCancel("cancelled")

        delay_stop_reason = _sleep_with_cancel(delay_ms=request.loop.delay_ms, cancel_snapshot=cancel_snapshot)
        if delay_stop_reason is not None:
            stop_reason = delay_stop_reason
            break

    if last_result is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Image automation completed without a successful iteration result.")

    return AutomationRunResult(
        last_result=last_result,
        automation_summary={
            "iteration_index": successful_iterations,
            "iteration_total": total_iterations,
            "seed": summary_seed,
            "prompt_preview": summary_prompt_preview,
            "source_label": summary_source_label,
            "stop_reason": stop_reason,
        },
    )
