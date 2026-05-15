"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Netflix VOID source-video + quadmask preprocessing helpers.
Loads the source video and precomputed quadmask video through the shared ffmpeg owners, validates their bounded
alignment contract, resizes them to the requested runtime geometry, quantizes/inverts the quadmask according to the
upstream-local VOID inference path, and applies the same temporal-padding policy before native inference.

Symbols (top-level; keep in sync; no ghosts):
- `NetflixVoidPreparedInputs` (dataclass): Prepared source-video + quadmask tensors plus bounded metadata for native runtime use.
- `prepare_netflix_void_vid2vid_inputs` (function): Load, validate, resize, quantize, and temporally pad one VOID vid2vid request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Sequence
from uuid import uuid4

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from apps.backend.core.requests import Vid2VidRequest
from apps.backend.infra.config.repo_root import repo_scratch_path
from apps.backend.video.io.ffmpeg import VideoProbe, extract_frames, probe_video

from .config import NETFLIX_VOID_DEFAULT_MAX_VIDEO_LENGTH


@dataclass(frozen=True, slots=True)
class NetflixVoidPreparedInputs:
    video: torch.Tensor
    mask_video: torch.Tensor
    prompt: str
    source_fps: float
    output_fps: int
    frame_count: int
    width: int
    height: int


def _ensure_probe_alignment(*, source_probe: VideoProbe, mask_probe: VideoProbe) -> None:
    if source_probe.width != mask_probe.width or source_probe.height != mask_probe.height:
        raise RuntimeError(
            "Netflix VOID source video and quadmask must share the same pixel geometry before resize; "
            f"got video={source_probe.width}x{source_probe.height}, mask={mask_probe.width}x{mask_probe.height}."
        )

    if abs(float(source_probe.fps) - float(mask_probe.fps)) > 0.05:
        raise RuntimeError(
            "Netflix VOID source video and quadmask must share the same fps before inference; "
            f"got video={source_probe.fps:.6f}, mask={mask_probe.fps:.6f}."
        )

    if source_probe.frame_count is not None and mask_probe.frame_count is not None:
        if int(source_probe.frame_count) != int(mask_probe.frame_count):
            raise RuntimeError(
                "Netflix VOID source video and quadmask must share the same frame count before inference; "
                f"got video={source_probe.frame_count}, mask={mask_probe.frame_count}."
            )


def _load_rgb_frame_sequence(frame_paths: Sequence[str]) -> torch.Tensor:
    if not frame_paths:
        raise RuntimeError("Netflix VOID source-video extraction produced 0 frames.")

    arrays: list[np.ndarray] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as handle:
            arrays.append(np.array(handle.convert("RGB"), dtype=np.uint8))
    stacked = np.stack(arrays, axis=0)
    tensor = torch.from_numpy(stacked).permute(3, 0, 1, 2).float() / 255.0
    return tensor


def _load_mask_frame_sequence(frame_paths: Sequence[str]) -> torch.Tensor:
    if not frame_paths:
        raise RuntimeError("Netflix VOID quadmask extraction produced 0 frames.")

    arrays: list[np.ndarray] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as handle:
            arrays.append(np.array(handle.convert("L"), dtype=np.uint8))
    stacked = np.stack(arrays, axis=0)
    tensor = torch.from_numpy(stacked).unsqueeze(0).float()
    return tensor


def _resize_video_tensor(video_tensor: torch.Tensor, *, width: int, height: int) -> torch.Tensor:
    resized = F.interpolate(video_tensor, size=(int(height), int(width)), mode="area")
    return resized.unsqueeze(0)


def _resize_mask_tensor(mask_tensor: torch.Tensor, *, width: int, height: int) -> torch.Tensor:
    resized = F.interpolate(mask_tensor, size=(int(height), int(width)), mode="area")
    return resized.unsqueeze(0)


def _quantize_quadmask(mask_tensor: torch.Tensor) -> torch.Tensor:
    quantized = mask_tensor.clone()
    quantized = torch.where(quantized <= 31.0, 0.0, quantized)
    quantized = torch.where((quantized > 31.0) & (quantized <= 95.0), 63.0, quantized)
    quantized = torch.where((quantized > 95.0) & (quantized <= 191.0), 127.0, quantized)
    quantized = torch.where(quantized > 191.0, 255.0, quantized)
    if not bool(torch.any(quantized < 255.0)):
        raise RuntimeError("Netflix VOID quadmask contains no removable or affected region after quantization.")
    return (255.0 - quantized) / 255.0


def _resolve_temporal_target_length(*, current_length: int, minimum_length: int, maximum_length: int) -> int:
    target_length = (int(current_length) // 4) * 4 + 1
    if target_length < int(current_length):
        target_length += 4
    if (target_length // 4) % 2 == 0:
        target_length += 4
    target_length = min(int(maximum_length), target_length)
    target_length = max(int(minimum_length), target_length)
    return int(target_length)


def _slice_tensor_along_dim(tensor: torch.Tensor, *, dim: int, size: int) -> torch.Tensor:
    slices = [slice(None)] * tensor.ndim
    slices[int(dim)] = slice(0, int(size))
    return tensor[tuple(slices)]


def _temporal_pad_to_target(tensor: torch.Tensor, *, dim: int, target_length: int) -> torch.Tensor:
    padded = _slice_tensor_along_dim(tensor, dim=dim, size=target_length)
    while int(padded.size(dim)) < int(target_length):
        flipped = torch.flip(padded, dims=[int(dim)])
        padded = torch.cat([padded, flipped], dim=int(dim))
        padded = _slice_tensor_along_dim(padded, dim=dim, size=target_length)
    return padded


def prepare_netflix_void_vid2vid_inputs(request: Vid2VidRequest) -> NetflixVoidPreparedInputs:
    source_video_path = str(getattr(request, "video_path", "") or "").strip()
    mask_video_path = str(getattr(request, "mask_video_path", "") or "").strip()
    if not source_video_path:
        raise RuntimeError("Netflix VOID preprocessing requires 'video_path'.")
    if not mask_video_path:
        raise RuntimeError("Netflix VOID preprocessing requires 'mask_video_path'.")

    source_probe = probe_video(source_video_path)
    mask_probe = probe_video(mask_video_path)
    _ensure_probe_alignment(source_probe=source_probe, mask_probe=mask_probe)

    target_width = int(getattr(request, "width", 0) or 0)
    target_height = int(getattr(request, "height", 0) or 0)
    if target_width <= 0 or target_height <= 0:
        raise RuntimeError(f"Netflix VOID preprocessing requires positive width/height, got {target_width}x{target_height}.")

    requested_frames = int(getattr(request, "num_frames", 0) or 0)
    if requested_frames <= 0:
        raise RuntimeError(f"Netflix VOID preprocessing requires positive num_frames, got {requested_frames}.")
    if requested_frames > int(NETFLIX_VOID_DEFAULT_MAX_VIDEO_LENGTH):
        raise RuntimeError(
            "Netflix VOID preprocessing does not support num_frames above the current max video length "
            f"({int(NETFLIX_VOID_DEFAULT_MAX_VIDEO_LENGTH)}), got {requested_frames}."
        )

    scratch_root = repo_scratch_path("netflix_void_preprocess", uuid4().hex)
    video_frames_dir = scratch_root / "video_frames"
    mask_frames_dir = scratch_root / "mask_frames"

    try:
        source_frame_paths = extract_frames(
            source_video_path,
            out_dir=str(video_frames_dir),
            max_frames=int(NETFLIX_VOID_DEFAULT_MAX_VIDEO_LENGTH),
        )
        mask_frame_paths = extract_frames(
            mask_video_path,
            out_dir=str(mask_frames_dir),
            max_frames=int(NETFLIX_VOID_DEFAULT_MAX_VIDEO_LENGTH),
        )

        if len(source_frame_paths) != len(mask_frame_paths):
            raise RuntimeError(
                "Netflix VOID preprocessing requires source video and quadmask to extract the same frame count; "
                f"got video={len(source_frame_paths)}, mask={len(mask_frame_paths)}."
            )

        video_tensor = _resize_video_tensor(
            _load_rgb_frame_sequence(source_frame_paths),
            width=target_width,
            height=target_height,
        )
        mask_tensor = _resize_mask_tensor(
            _load_mask_frame_sequence(mask_frame_paths),
            width=target_width,
            height=target_height,
        )
        mask_tensor = _quantize_quadmask(mask_tensor)

        target_length = _resolve_temporal_target_length(
            current_length=len(source_frame_paths),
            minimum_length=requested_frames,
            maximum_length=int(NETFLIX_VOID_DEFAULT_MAX_VIDEO_LENGTH),
        )
        padded_video = _temporal_pad_to_target(video_tensor, dim=2, target_length=target_length)
        padded_mask = _temporal_pad_to_target(mask_tensor, dim=2, target_length=target_length)

        return NetflixVoidPreparedInputs(
            video=padded_video,
            mask_video=padded_mask,
            prompt=str(getattr(request, "prompt", "") or ""),
            source_fps=float(source_probe.fps),
            output_fps=int(getattr(request, "fps", 0) or round(float(source_probe.fps)) or 12),
            frame_count=int(target_length),
            width=target_width,
            height=target_height,
        )
    finally:
        shutil.rmtree(Path(scratch_root), ignore_errors=True)
