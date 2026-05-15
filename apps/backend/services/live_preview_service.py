"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Live preview task configuration and progress-event attachment.
Builds per-task preview settings, encodes a resized preview image, and attaches it to SSE progress payloads when enabled.

Symbols (top-level; keep in sync; no ghosts):
- `LivePreviewImageFormat` (enum): Supported output formats for encoded preview images.
- `_coerce_bool_option` (function): Strict bool parser for preview-related option values.
- `_coerce_int_option` (function): Strict integer parser for preview-related option values.
- `_coerce_image_format_option` (function): Strict image format parser for preview output settings.
- `_coerce_preview_method_option` (function): Strict preview method parser for runtime option values.
- `LivePreviewEncodedImage` (dataclass): Encoded preview payload (`format` + base64 `data`).
- `LivePreviewTaskConfig` (dataclass): Preview config for a task; can apply per-task runtime overrides for the sampling runtime.
- `LivePreviewService` (class): Builds preview config, encodes images, and attaches previews to progress events.
- `__all__` (constant): Explicit export list for this module.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import base64
from contextlib import contextmanager
import io
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, Optional

from apps.backend.core.strict_values import parse_bool_value, parse_int_value
from apps.backend.core.state import state as backend_state
from apps.backend.runtime.live_preview import (
    LivePreviewMethod,
    debug_preview_factors_enabled,
    preview_runtime_overrides,
)

logger = get_backend_logger(__name__)


class LivePreviewImageFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"

    @staticmethod
    def from_string(value: str | None, *, default: "LivePreviewImageFormat" = PNG) -> "LivePreviewImageFormat":
        key = (value or "").strip().lower()
        if key in {"jpg", "jpeg"}:
            return LivePreviewImageFormat.JPEG
        if key == "png":
            return LivePreviewImageFormat.PNG
        if key == "webp":
            return LivePreviewImageFormat.WEBP
        return default


def _coerce_bool_option(value: object, *, key: str, default: bool) -> bool:
    try:
        return parse_bool_value(value, field=f"options.{key}", default=default)
    except RuntimeError as exc:
        raise RuntimeError(f"Invalid boolean option '{key}': {exc}") from exc


def _coerce_int_option(value: object, *, key: str, default: int, minimum: int | None = None) -> int:
    try:
        return parse_int_value(value, field=f"options.{key}", default=default, minimum=minimum)
    except RuntimeError as exc:
        raise RuntimeError(f"Invalid integer option '{key}': {exc}") from exc


def _coerce_image_format_option(
    value: object,
    *,
    key: str,
    default: LivePreviewImageFormat,
) -> LivePreviewImageFormat:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized == "":
        raise RuntimeError(f"Invalid image format option '{key}': expected non-empty value, got {value!r}")
    if normalized in {"jpg", "jpeg"}:
        return LivePreviewImageFormat.JPEG
    if normalized == "png":
        return LivePreviewImageFormat.PNG
    if normalized == "webp":
        return LivePreviewImageFormat.WEBP
    raise RuntimeError(
        f"Invalid image format option '{key}': expected one of ['jpeg', 'png', 'webp'], got {value!r}",
    )


def _coerce_preview_method_option(
    value: object,
    *,
    key: str,
    default: LivePreviewMethod,
) -> LivePreviewMethod:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"full", "vae"}:
        return LivePreviewMethod.FULL
    if normalized == "":
        raise RuntimeError(f"Invalid preview method option '{key}': expected non-empty value, got {value!r}")
    if normalized in {"approx cheap", "approx_cheap", "approx-cheap", "cheap"}:
        return LivePreviewMethod.APPROX_CHEAP
    raise RuntimeError(
        "Invalid preview method option "
        f"'{key}': expected one of ['full', 'vae', 'approx cheap', 'approx_cheap', 'approx-cheap', 'cheap'], got {value!r}",
    )


@dataclass(frozen=True)
class LivePreviewEncodedImage:
    format: str
    data: str

    def as_dict(self) -> dict[str, str]:
        return {"format": self.format, "data": self.data}


@dataclass(frozen=True)
class LivePreviewTaskConfig:
    runtime_interval_steps: int
    runtime_method: LivePreviewMethod
    sse_enabled: bool
    image_format: LivePreviewImageFormat
    max_dim: int = 512

    @contextmanager
    def runtime_overrides(self) -> Iterator[None]:
        """Apply per-task preview settings to the current thread runtime."""
        with preview_runtime_overrides(
            interval_steps=int(self.runtime_interval_steps),
            method=self.runtime_method,
        ):
            yield


class LivePreviewService:
    """Build live preview config and attach preview payloads to progress events."""

    def build_task_config(self, opts_get: Callable[[str, object], object]) -> LivePreviewTaskConfig:
        enabled = _coerce_bool_option(
            opts_get("live_previews_enable", True),
            key="live_previews_enable",
            default=True,
        )
        image_format = _coerce_image_format_option(
            opts_get("live_previews_image_format", LivePreviewImageFormat.PNG.value),
            key="live_previews_image_format",
            default=LivePreviewImageFormat.PNG,
        )

        period_raw = opts_get("show_progress_every_n_steps", 10)
        period = _coerce_int_option(
            period_raw,
            key="show_progress_every_n_steps",
            default=10,
            minimum=-1,
        )

        method = _coerce_preview_method_option(
            opts_get("show_progress_type", LivePreviewMethod.APPROX_CHEAP.value),
            key="show_progress_type",
            default=LivePreviewMethod.APPROX_CHEAP,
        )

        # `show_progress_every_n_steps=-1` is a supported persisted sentinel that disables previews.
        # SSE preview payloads are gated by the explicit UI setting plus a positive period.
        sse_enabled = enabled and period > 0

        runtime_interval = period if sse_enabled else 0
        if debug_preview_factors_enabled() and runtime_interval <= 0:
            runtime_interval = 10

        return LivePreviewTaskConfig(
            runtime_interval_steps=runtime_interval,
            runtime_method=method,
            sse_enabled=sse_enabled,
            image_format=image_format,
            max_dim=512,
        )

    def encode_preview_image(self, image: object, *, fmt: LivePreviewImageFormat, max_dim: int) -> Optional[LivePreviewEncodedImage]:
        try:
            from PIL import Image  # type: ignore
        except Exception:
            return None

        if not isinstance(image, Image.Image):
            return None

        img = image
        try:
            w, h = img.size
            max_side = max(int(w), int(h))
            if max_side > int(max_dim) > 0:
                scale = float(max_dim) / float(max_side)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                img = img.resize((new_w, new_h), resample=resample)
        except Exception:
            img = image

        buf = io.BytesIO()
        if fmt == LivePreviewImageFormat.PNG:
            img.save(buf, format="PNG")
        elif fmt == LivePreviewImageFormat.WEBP:
            img.save(buf, format="WEBP", quality=80)
        else:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=80)

        return LivePreviewEncodedImage(format=fmt.value, data=base64.b64encode(buf.getvalue()).decode("ascii"))

    def maybe_attach_to_progress_event(
        self,
        event: dict[str, Any],
        entry: Any,
        *,
        config: LivePreviewTaskConfig,
        expected_owner_token: str | None = None,
    ) -> None:
        if not config.sse_enabled:
            return

        preview_owner_token, preview_id, preview_image, preview_step = backend_state.live_preview_snapshot()
        normalized_expected_owner_token = str(expected_owner_token or "").strip()
        if normalized_expected_owner_token and str(preview_owner_token or "").strip() != normalized_expected_owner_token:
            return
        if preview_id <= 0:
            return

        last_sent = int(getattr(entry, "last_preview_id_sent", 0) or 0)
        if preview_id == last_sent:
            return

        encoded = self.encode_preview_image(
            preview_image,
            fmt=config.image_format,
            max_dim=int(config.max_dim),
        )
        if not encoded:
            return

        try:
            setattr(entry, "last_preview_id_sent", preview_id)
        except Exception:
            pass

        event["preview_image"] = encoded.as_dict()
        if preview_step > 0:
            event["preview_step"] = preview_step


__all__ = [
    "LivePreviewEncodedImage",
    "LivePreviewImageFormat",
    "LivePreviewService",
    "LivePreviewTaskConfig",
]
