"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Audio-component and export-asset helpers for the native LTX2 seam.
Validates parser-owned audio VAE fragments plus the supported vocoder stored layouts, splits combined LTX 2.3 audio
bundles (`audio_vae.*` + `vocoder.*`), materializes generated LTX2 audio into a temporary WAV file, and wraps that file
into the shared `AudioExportAsset` contract used by canonical video export paths.

Symbols (top-level; keep in sync; no ghosts):
- `is_ltx2_wrapped_vocoder_state` (function): Detect whether a parser-owned LTX2 vocoder mapping uses the real 2.3 wrapped owner groups.
- `validate_ltx2_audio_bundle_contract` (function): Validate required LTX2 audio-VAE and vocoder sentinel keys.
- `split_ltx2_audio_bundle_state` (function): Split a combined LTX audio bundle file into `audio_vae` and `vocoder` component mappings.
- `build_ltx2_audio_export_asset` (function): Build a shared `AudioExportAsset` from a generated LTX2 audio file.
- `materialize_ltx2_generated_audio_asset` (function): Write generated waveform data to a temp WAV and wrap it as `AudioExportAsset`.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Mapping
import wave

import numpy as np
import torch

from apps.backend.runtime.pipeline_stages.video import AudioExportAsset

_LEGACY_VOCODER_TOP_LEVEL_PREFIXES = frozenset({"conv_pre", "ups", "resblocks", "conv_post"})
_WRAPPED_23_VOCODER_TOP_LEVEL_PREFIXES = frozenset({"bwe_generator", "mel_stft", "vocoder"})


def is_ltx2_wrapped_vocoder_state(vocoder_state: Mapping[str, Any]) -> bool:
    raw_keys = tuple(str(key) for key in vocoder_state.keys())
    if not raw_keys:
        raise RuntimeError("LTX2 vocoder bundle is empty after split.")

    top_level_prefixes = {key.split(".", 1)[0] for key in raw_keys}
    has_legacy = bool(top_level_prefixes & _LEGACY_VOCODER_TOP_LEVEL_PREFIXES)
    has_wrapped_23 = bool(top_level_prefixes & _WRAPPED_23_VOCODER_TOP_LEVEL_PREFIXES)
    if has_legacy and has_wrapped_23:
        raise RuntimeError(
            "LTX2 vocoder bundle mixes legacy raw and 2.3 wrapper groups. "
            f"Top-level prefixes={sorted(top_level_prefixes)!r}."
        )
    return has_wrapped_23


def validate_ltx2_audio_bundle_contract(
    *,
    audio_vae_state: Mapping[str, Any],
    vocoder_state: Mapping[str, Any],
) -> None:
    if "per_channel_statistics.mean-of-means" not in audio_vae_state:
        raise RuntimeError("LTX2 audio VAE bundle is missing `per_channel_statistics.mean-of-means`.")

    raw_keys = tuple(str(key) for key in vocoder_state.keys())
    top_level_prefixes = {key.split(".", 1)[0] for key in raw_keys}
    has_wrapped_23 = is_ltx2_wrapped_vocoder_state(vocoder_state)

    if has_wrapped_23:
        unexpected = sorted(top_level_prefixes - _WRAPPED_23_VOCODER_TOP_LEVEL_PREFIXES)
        if unexpected:
            raise RuntimeError(
                "Unsupported LTX2 2.3 wrapped vocoder bundle layout. "
                f"Unexpected top-level prefixes={unexpected!r}."
            )
        required_wrapped = (
            "bwe_generator.conv_pre.weight",
            "bwe_generator.conv_post.weight",
            "mel_stft.mel_basis",
            "vocoder.conv_pre.weight",
            "vocoder.conv_post.weight",
        )
        missing_wrapped = [key for key in required_wrapped if key not in vocoder_state]
        if missing_wrapped:
            raise RuntimeError(
                "LTX2 2.3 wrapped vocoder bundle is missing required keys: "
                f"{missing_wrapped!r}"
            )
        return

    required_legacy = ("conv_pre.weight", "conv_post.weight")
    missing_legacy = [key for key in required_legacy if key not in vocoder_state]
    if missing_legacy:
        raise RuntimeError(f"LTX2 vocoder bundle is missing required keys: {missing_legacy!r}")

    unexpected = sorted(top_level_prefixes - _LEGACY_VOCODER_TOP_LEVEL_PREFIXES)
    if unexpected:
        raise RuntimeError(
            "Unsupported legacy LTX2 vocoder bundle layout. "
            f"Unexpected top-level prefixes={unexpected!r}."
        )


def split_ltx2_audio_bundle_state(state_dict: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    audio_vae_state: dict[str, Any] = {}
    vocoder_state: dict[str, Any] = {}

    for raw_key, value in state_dict.items():
        key = str(raw_key)
        if key.startswith("audio_vae."):
            stripped = key[len("audio_vae.") :]
            if not stripped:
                raise RuntimeError("LTX2 combined audio bundle contains an empty `audio_vae.` key.")
            if stripped in audio_vae_state:
                raise RuntimeError(f"LTX2 combined audio bundle collides in audio_vae after prefix split: {stripped!r}.")
            audio_vae_state[stripped] = value
            continue
        if key.startswith("vocoder."):
            stripped = key[len("vocoder.") :]
            if not stripped:
                raise RuntimeError("LTX2 combined audio bundle contains an empty `vocoder.` key.")
            if stripped in vocoder_state:
                raise RuntimeError(f"LTX2 combined audio bundle collides in vocoder after prefix split: {stripped!r}.")
            vocoder_state[stripped] = value

    if not audio_vae_state:
        raise RuntimeError(
            "LTX2 combined audio bundle is missing `audio_vae.*` tensors. "
            "Provide the real combined LTX audio bundle file, not a video VAE or unrelated side asset."
        )
    if not vocoder_state:
        raise RuntimeError(
            "LTX2 combined audio bundle is missing `vocoder.*` tensors. "
            "Provide the real combined LTX audio bundle file, not a partial audio-only asset."
        )

    validate_ltx2_audio_bundle_contract(audio_vae_state=audio_vae_state, vocoder_state=vocoder_state)
    return audio_vae_state, vocoder_state


def build_ltx2_audio_export_asset(
    path: str,
    *,
    owned_temp: bool,
    sample_rate_hz: int | None = None,
    channels: int | None = None,
) -> AudioExportAsset:
    resolved = os.path.expanduser(str(path or "").strip())
    if not resolved:
        raise RuntimeError("LTX2 generated audio export path is empty.")
    if not os.path.isfile(resolved):
        raise RuntimeError(f"LTX2 generated audio export path not found: {resolved}")
    if sample_rate_hz is not None and int(sample_rate_hz) <= 0:
        raise RuntimeError(f"LTX2 generated audio sample_rate_hz must be > 0, got {sample_rate_hz!r}.")
    if channels is not None and int(channels) <= 0:
        raise RuntimeError(f"LTX2 generated audio channels must be > 0, got {channels!r}.")

    return AudioExportAsset(
        path=resolved,
        owned_temp=bool(owned_temp),
        sample_rate_hz=(None if sample_rate_hz is None else int(sample_rate_hz)),
        channels=(None if channels is None else int(channels)),
    )


def materialize_ltx2_generated_audio_asset(
    audio: Any,
    *,
    sample_rate_hz: int,
    owned_temp: bool = True,
) -> AudioExportAsset | None:
    if audio is None:
        return None

    if isinstance(audio, torch.Tensor):
        array = audio.detach().to(torch.float32).cpu().numpy()
    else:
        array = np.asarray(audio)

    if array.ndim == 0:
        raise RuntimeError("LTX2 generated audio tensor is scalar; expected waveform data.")
    if array.ndim == 3:
        if int(array.shape[0]) != 1:
            raise RuntimeError(
                "LTX2 generated audio batch size must be 1 for canonical runtime use-cases; "
                f"got batch={array.shape[0]!r}."
            )
        array = array[0]
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise RuntimeError(
            "LTX2 generated audio must be 1D, 2D, or single-batch 3D; "
            f"got shape={tuple(int(dim) for dim in array.shape)!r}."
        )

    if array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
        channels = int(array.shape[0])
        samples_first = array.transpose(1, 0)
    elif array.shape[1] <= 8 and array.shape[0] > array.shape[1]:
        channels = int(array.shape[1])
        samples_first = array
    else:
        raise RuntimeError(
            "LTX2 generated audio shape is ambiguous; expected [channels, samples] or [samples, channels] "
            f"with <= 8 channels, got shape={tuple(int(dim) for dim in array.shape)!r}."
        )

    if channels <= 0:
        raise RuntimeError(f"LTX2 generated audio has invalid channel count: {channels!r}")

    waveform = np.nan_to_num(samples_first.astype(np.float32, copy=False), copy=False)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).round().astype("<i2", copy=False)

    fd, path = tempfile.mkstemp(prefix="ltx2_audio_", suffix=".wav")
    os.close(fd)
    try:
        with wave.open(path, "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate_hz))
            handle.writeframes(pcm16.tobytes())
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass
        raise

    return build_ltx2_audio_export_asset(
        path,
        owned_temp=owned_temp,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
    )
