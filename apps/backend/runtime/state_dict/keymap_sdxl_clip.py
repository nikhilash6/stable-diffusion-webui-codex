"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical CLIP key-style detection + explicit source-style mapping (generic + SDXL wrappers) into Codex IntegratedCLIP state_dict layout.
Supports HF `text_model.*`, OpenCLIP legacy `transformer.resblocks.*`, Codex-canonical `transformer.text_model.*`, and the known SDXL
wrapper/container surfaces that expose those same CLIP layouts. Validates known buffers/weights, fails loud on unknown non-weight keys,
and exposes generic layout-detection/keyspace helpers used by loader/parser seams.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_sdxl_clip_l_keyspace` (function): Keymap for SDXL base CLIP-L (`text_encoder`) into Codex IntegratedCLIP keys.
- `resolve_sdxl_clip_g_keyspace` (function): Keymap for SDXL base CLIP-G (`text_encoder_2`) into Codex IntegratedCLIP keys.
- `detect_clip_layout_metadata` (function): Detects CLIP key style + native layout metadata (QKV/projection orientation).
- `resolve_clip_keyspace_with_layout` (function): Generic CLIP keymap with explicit QKV/projection layout controls.
- `resolve_sdxl_clip_l_keyspace_with_layout` (function): SDXL CLIP-L keymap wrapper returning resolved layout metadata.
- `resolve_sdxl_clip_g_keyspace_with_layout` (function): SDXL CLIP-G keymap wrapper returning resolved layout metadata.

Notes: Target keyspace matches `apps/backend/runtime/common/nn/clip.py:IntegratedCLIP` (and related Codex CLIP wrappers).
Key policies (non-exhaustive): interpret known wrapper/container surfaces explicitly per key; drop HF-only buffers (`*.position_ids`) and refuse other unknown non-weight keys;
canonicalize native `logit_scale` when present, fail loud on duplicate sources, and synthesize the IntegratedCLIP default (`ln(100)`) when the source omits it;
map optional projection weights into `transformer.text_projection.weight` (CLIP-G; lazy transpose); for OpenCLIP-style fused attention weights (`attn.in_proj_{weight,bias}`),
expose either split Q/K/V projections or fused `in_proj` keys.
The QKV layout can be selected via the `qkv_impl` argument (`"auto"`, `"split"` or `"fused"`).
`"auto"` keeps the native layout: OpenCLIP-style weights stay fused, HF/Codex weights stay split.
Structural conversion operations in this keymap (QKV split/fuse and projection transpose) are globally policy-gated by
`CODEX_WEIGHT_STRUCTURAL_CONVERSION` (`auto`=forbid, `convert`=allow).
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from math import log
from typing import Literal, TypeVar

from apps.backend.infra.config.weight_structural_conversion import (
    ENV_WEIGHT_STRUCTURAL_CONVERSION,
    is_structural_weight_conversion_enabled,
)
from apps.backend.runtime.state_dict.key_mapping import (
    KeyMappingError,
    KeyStyle,
    KeyStyleDetector,
    KeyStyleSpec,
    ResolvedKeyspace,
    KeySentinel,
    SentinelKind,
)

_T = TypeVar("_T")
_QKVImpl = Literal["auto", "split", "fused"]
_ProjectionOrientation = Literal["none", "linear", "matmul"]
_ProjectionOrientationTarget = Literal["auto", "linear", "matmul"]

_WRAPPER_PREFIXES: tuple[str, ...] = (
    "conditioner.embedders.0.transformer.",
    "conditioner.embedders.0.model.",
    "conditioner.embedders.0.",
    "conditioner.embedders.1.transformer.",
    "conditioner.embedders.1.model.",
    "conditioner.embedders.1.",
    "cond_stage_model.model.",
    "cond_stage_model.",
    "text_encoders.clip_l.",
    "text_encoders.clip_g.",
    "text_encoders.clip_h.",
    "clip_l.",
    "clip_g.",
    "clip_h.",
    "model.text_model.",
    "model.",
)

_LOGIT_KEYS: tuple[str, ...] = (
    "logit_scale",
    "transformer.logit_scale",
    "transformer.text_model.logit_scale",
)

_PROJ_KEYS: tuple[str, ...] = (
    "transformer.text_projection.weight",
    "transformer.text_projection",
    "text_projection.weight",
    "text_projection",
)

_DETECTOR = KeyStyleDetector(
    name="sdxl_clip_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "transformer.text_model.embeddings."),
                KeySentinel(SentinelKind.PREFIX, "transformer.text_model.encoder.layers."),
            ),
            min_sentinel_hits=1,
        ),
        KeyStyleSpec(
            style=KeyStyle.HF,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "text_model.embeddings."),
                KeySentinel(SentinelKind.PREFIX, "text_model.encoder.layers."),
            ),
            min_sentinel_hits=1,
        ),
        KeyStyleSpec(
            style=KeyStyle.OPENCLIP,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "transformer.resblocks."),
                KeySentinel(SentinelKind.EXACT, "token_embedding.weight"),
                KeySentinel(SentinelKind.EXACT, "positional_embedding"),
            ),
            min_sentinel_hits=1,
        ),
    ),
)

_ESSENTIAL_KEYS_SPLIT: tuple[str, ...] = (
    "transformer.text_model.embeddings.token_embedding.weight",
    "transformer.text_model.embeddings.position_embedding.weight",
    "transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
    "transformer.text_model.final_layer_norm.weight",
)

_ESSENTIAL_KEYS_FUSED: tuple[str, ...] = (
    "transformer.text_model.embeddings.token_embedding.weight",
    "transformer.text_model.embeddings.position_embedding.weight",
    "transformer.text_model.encoder.layers.0.self_attn.in_proj.weight",
    "transformer.text_model.final_layer_norm.weight",
)


@dataclass(frozen=True, slots=True)
class _Direct:
    key: str


@dataclass(frozen=True, slots=True)
class _SliceQKV:
    key: str
    index: int  # 0=q,1=k,2=v


@dataclass(frozen=True, slots=True)
class _ConcatQKV:
    q_key: str
    k_key: str
    v_key: str


@dataclass(frozen=True, slots=True)
class _Transpose:
    key: str


@dataclass(frozen=True, slots=True)
class _DefaultLogitScale:
    value: float


_Spec = _Direct | _SliceQKV | _ConcatQKV | _Transpose | _DefaultLogitScale


@dataclass(frozen=True, slots=True)
class ClipLayoutMetadata:
    qkv_layout: Literal["split", "fused"]
    projection_orientation: _ProjectionOrientation
    source_style: str | None = None


class _SDXLCLIPKeymapView(MutableMapping[str, _T]):
    """Lazy key mapping view supporting QKV slicing + projection transpose."""

    def __init__(self, base: MutableMapping[str, _T], mapping: dict[str, _Spec]):
        self._base = base
        self._map = dict(mapping)
        self._keys = tuple(mapping.keys())
        # Cache only derived tensors (slices/defaults) so we don't keep the whole
        # text encoder resident when streaming from SafeTensors.
        self._derived_cache: dict[str, _T] = {}
        self._source_cache: dict[str, _T] = {}

    @staticmethod
    def _slice_in_proj(value: _T, index: int) -> _T:
        try:
            shape = getattr(value, "shape", None)
            if not shape:
                return value
            if len(shape) < 1:
                return value
            total = int(shape[0])
            if total % 3 != 0:
                raise KeyMappingError(f"OpenCLIP in_proj first dim is not divisible by 3 (shape={shape!r})")
            chunk = total // 3
            start = index * chunk
            end = (index + 1) * chunk
            return value[start:end]
        except Exception as exc:
            raise KeyMappingError(f"Failed to slice OpenCLIP in_proj tensor (index={index})") from exc

    @staticmethod
    def _transpose_2d(value: _T) -> _T:
        try:
            ndim = getattr(value, "ndim", None)
            if ndim != 2:
                return value
            transposed = value.transpose(0, 1)
            contiguous = getattr(transposed, "contiguous", None)
            if callable(contiguous):
                return contiguous()
            return transposed
        except Exception as exc:
            raise KeyMappingError("Failed to transpose projection tensor") from exc

    @staticmethod
    def _concat_qkv(values: tuple[_T, _T, _T]) -> _T:
        try:
            import torch

            q, k, v = values
            if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                raise TypeError("Expected torch.Tensor values for QKV concatenation")
            return torch.cat((q, k, v), dim=0)
        except Exception as exc:
            raise KeyMappingError("Failed to concatenate Q/K/V tensors into fused in_proj") from exc

    def _get_source(self, key: str) -> _T:
        cached = self._source_cache.get(key)
        if cached is not None:
            return cached
        v = self._base[key]
        self._source_cache[key] = v
        return v

    def __getitem__(self, k: str) -> _T:
        spec = self._map[k]
        if isinstance(spec, _Direct):
            return self._base[spec.key]
        elif isinstance(spec, _SliceQKV):
            cached = self._derived_cache.get(k)
            if cached is not None:
                return cached
            base_tensor = self._get_source(spec.key)
            v = self._slice_in_proj(base_tensor, spec.index)
        elif isinstance(spec, _ConcatQKV):
            cached = self._derived_cache.get(k)
            if cached is not None:
                return cached
            q = self._get_source(spec.q_key)
            k_tensor = self._get_source(spec.k_key)
            v_tensor = self._get_source(spec.v_key)
            v = self._concat_qkv((q, k_tensor, v_tensor))
        elif isinstance(spec, _Transpose):
            cached = self._derived_cache.get(k)
            if cached is not None:
                return cached
            v = self._transpose_2d(self._base[spec.key])
        elif isinstance(spec, _DefaultLogitScale):
            cached = self._derived_cache.get(k)
            if cached is not None:
                return cached
            import torch

            v = torch.tensor(float(spec.value))  # type: ignore[assignment]
        else:  # pragma: no cover - defensive
            raise KeyError(k)

        if not isinstance(spec, _ConcatQKV):
            self._derived_cache[k] = v
        return v

    def __setitem__(self, k: str, v: _T) -> None:
        self._derived_cache.pop(k, None)
        self._map[k] = _Direct(k)
        self._base[k] = v
        if k not in self._keys:
            self._keys = (*self._keys, k)

    def __delitem__(self, k: str) -> None:
        self._derived_cache.pop(k, None)
        spec = self._map.pop(k, None)
        if isinstance(spec, _Direct) and spec.key in self._base:
            del self._base[spec.key]
        if k in self._keys:
            self._keys = tuple(x for x in self._keys if x != k)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, k: object) -> bool:
        return k in self._map

    def keys(self):
        return list(self._keys)

    def items(self):
        for k in self._keys:
            yield k, self[k]


def _map_source_key_to_clip_lookup_key(key: str) -> str:
    source_key = str(key)
    if _is_supported_clip_root_key(source_key):
        return source_key
    for wrapper_prefix in _WRAPPER_PREFIXES:
        if source_key.startswith(wrapper_prefix):
            candidate_key = source_key[len(wrapper_prefix) :]
            if _is_supported_clip_root_key(candidate_key):
                return candidate_key
            break
    return source_key


def _is_supported_clip_root_key(key: str) -> bool:
    return (
        key.startswith(("transformer.text_model.", "text_model.", "transformer.resblocks."))
        or key in _LOGIT_KEYS
        or key in _PROJ_KEYS
        or key in {"token_embedding.weight", "positional_embedding", "ln_final.weight", "ln_final.bias"}
    )


def _has_native_fused_qkv_keys(keys: Sequence[str]) -> bool:
    for key in keys:
        if ".attn.in_proj_weight" in key or ".attn.in_proj_bias" in key:
            return True
        if ".self_attn.in_proj.weight" in key or ".self_attn.in_proj.bias" in key:
            return True
    return False


def _has_native_split_qkv_keys(keys: Sequence[str]) -> bool:
    for key in keys:
        if ".self_attn.q_proj." in key or ".self_attn.k_proj." in key or ".self_attn.v_proj." in key:
            return True
        if ".attn.q_proj." in key or ".attn.k_proj." in key or ".attn.v_proj." in key:
            return True
    return False


def _projection_orientation_for_style(*, style: KeyStyle, has_projection: bool) -> _ProjectionOrientation:
    if not has_projection:
        return "none"
    if style is KeyStyle.CODEX:
        return "linear"
    return "matmul"


def _style_from_source(source_style: str | None) -> KeyStyle | None:
    if source_style is None:
        return None
    if source_style == KeyStyle.CODEX.value:
        return KeyStyle.CODEX
    if source_style == KeyStyle.HF.value:
        return KeyStyle.HF
    if source_style == KeyStyle.OPENCLIP.value:
        return KeyStyle.OPENCLIP
    raise KeyMappingError(
        f"sdxl_clip: invalid cached source_style={source_style!r} (expected one of: codex, hf, openclip)"
    )


def _validate_layout_metadata(layout_metadata: ClipLayoutMetadata) -> None:
    if layout_metadata.qkv_layout not in {"split", "fused"}:
        raise KeyMappingError(
            "sdxl_clip: invalid cached qkv_layout=%r (expected one of: split, fused)"
            % (layout_metadata.qkv_layout,)
        )
    if layout_metadata.projection_orientation not in {"none", "linear", "matmul"}:
        raise KeyMappingError(
            "sdxl_clip: invalid cached projection_orientation=%r (expected one of: none, linear, matmul)"
            % (layout_metadata.projection_orientation,)
        )
    _style_from_source(layout_metadata.source_style)


def _spec_source(spec: _Spec) -> str:
    if isinstance(spec, _Direct):
        return str(spec.key)
    if isinstance(spec, _SliceQKV):
        return str(spec.key)
    if isinstance(spec, _ConcatQKV):
        return f"{spec.q_key}|{spec.k_key}|{spec.v_key}"
    if isinstance(spec, _Transpose):
        return str(spec.key)
    if isinstance(spec, _DefaultLogitScale):
        return f"<default:{spec.value}>"
    raise KeyMappingError(f"sdxl_clip: unsupported mapping spec type={type(spec).__name__}")


def clip_layout_metadata_from_resolved(resolved: ResolvedKeyspace[object]) -> ClipLayoutMetadata:
    metadata = dict(getattr(resolved, "metadata", {}) or {})
    qkv_layout = str(metadata.get("qkv_layout", "")).strip().lower()
    projection_orientation = str(metadata.get("projection_orientation", "")).strip().lower()
    source_style_value = metadata.get("source_style")
    source_style = str(source_style_value).strip().lower() if source_style_value is not None else None
    if not qkv_layout:
        raise KeyMappingError("sdxl_clip: resolved keyspace metadata is missing qkv_layout")
    if not projection_orientation:
        raise KeyMappingError("sdxl_clip: resolved keyspace metadata is missing projection_orientation")
    layout = ClipLayoutMetadata(
        qkv_layout=qkv_layout,  # type: ignore[arg-type]
        projection_orientation=projection_orientation,  # type: ignore[arg-type]
        source_style=source_style or None,
    )
    _validate_layout_metadata(layout)
    return layout


def detect_clip_layout_metadata(
    state_dict: MutableMapping[str, _T],
    *,
    keep_projection: bool,
) -> ClipLayoutMetadata:
    if len(state_dict) == 1 and "state_dict" in state_dict:
        inner = state_dict.get("state_dict")
        if isinstance(inner, MutableMapping):
            state_dict = inner

    raw_keys = list(state_dict.keys())
    lookup_keys = [_map_source_key_to_clip_lookup_key(raw_key) for raw_key in raw_keys]
    keys_for_style = [key for key in lookup_keys if not key.endswith(".position_ids")]
    style = _DETECTOR.detect(keys_for_style)

    has_fused = _has_native_fused_qkv_keys(lookup_keys)
    has_split = _has_native_split_qkv_keys(lookup_keys)
    if has_fused and has_split:
        raise KeyMappingError("sdxl_clip: ambiguous native qkv layout detection (both fused and split keys found)")
    if has_fused:
        native_qkv_layout: Literal["split", "fused"] = "fused"
    elif has_split:
        native_qkv_layout = "split"
    else:
        native_qkv_layout = "fused" if style is KeyStyle.OPENCLIP else "split"

    has_projection = keep_projection and any(key in _PROJ_KEYS for key in lookup_keys)
    native_projection_orientation = _projection_orientation_for_style(style=style, has_projection=has_projection)
    return ClipLayoutMetadata(
        qkv_layout=native_qkv_layout,
        projection_orientation=native_projection_orientation,
        source_style=style.value,
    )


def _resolve_clip_keyspace(
    state_dict: MutableMapping[str, _T],
    *,
    num_layers: int,
    keep_projection: bool,
    qkv_impl: _QKVImpl,
    projection_orientation: _ProjectionOrientationTarget,
    layout_metadata: ClipLayoutMetadata | None,
    require_projection: bool,
) -> ResolvedKeyspace[_T]:
    if qkv_impl not in ("auto", "split", "fused"):
        raise KeyMappingError(
            f"sdxl_clip: invalid qkv_impl={qkv_impl!r} (expected one of: auto, split, fused)"
        )
    if projection_orientation not in ("auto", "linear", "matmul"):
        raise KeyMappingError(
            f"sdxl_clip: invalid projection_orientation={projection_orientation!r} "
            "(expected one of: auto, linear, matmul)"
        )

    if len(state_dict) == 1 and "state_dict" in state_dict:
        inner = state_dict.get("state_dict")
        if isinstance(inner, MutableMapping):
            state_dict = inner

    allow_structural_conversion = is_structural_weight_conversion_enabled()

    raw_keys = list(state_dict.keys())
    lookup_key_pairs = [(raw, _map_source_key_to_clip_lookup_key(raw)) for raw in raw_keys]
    if layout_metadata is not None:
        _validate_layout_metadata(layout_metadata)
        detected_layout = layout_metadata
    else:
        detected_layout = detect_clip_layout_metadata(state_dict, keep_projection=keep_projection)

    cached_style = _style_from_source(detected_layout.source_style)
    if cached_style is None:
        keys_for_style = [key for _, key in lookup_key_pairs if not key.endswith(".position_ids")]
        style = _DETECTOR.detect(keys_for_style)
    else:
        style = cached_style

    mapping: dict[str, _Spec] = {}
    seen_logit: str | None = None
    seen_proj: str | None = None
    resolved_qkv_layout: Literal["split", "fused"] = (
        detected_layout.qkv_layout if qkv_impl == "auto" else qkv_impl
    )
    wants_fused_qkv = resolved_qkv_layout == "fused"
    requires_qkv_conversion = detected_layout.qkv_layout != resolved_qkv_layout
    if requires_qkv_conversion and not allow_structural_conversion:
        conversion = f"{detected_layout.qkv_layout}->{resolved_qkv_layout}"
        raise KeyMappingError(
            "sdxl_clip: structural conversion is disabled by policy "
            f"({ENV_WEIGHT_STRUCTURAL_CONVERSION}=auto). Requested qkv_impl={qkv_impl!r} "
            f"requires {conversion}. Set {ENV_WEIGHT_STRUCTURAL_CONVERSION}=convert to allow."
        )

    if keep_projection:
        resolved_projection_orientation: _ProjectionOrientation = (
            detected_layout.projection_orientation
            if projection_orientation == "auto"
            else projection_orientation
        )
    else:
        resolved_projection_orientation = "none"

    requires_projection_conversion = (
        keep_projection
        and detected_layout.projection_orientation != "none"
        and resolved_projection_orientation != detected_layout.projection_orientation
    )
    if requires_projection_conversion and not allow_structural_conversion:
        raise KeyMappingError(
            "sdxl_clip: structural conversion is disabled by policy "
            f"({ENV_WEIGHT_STRUCTURAL_CONVERSION}=auto). Requested projection_orientation="
            f"{projection_orientation!r} requires {detected_layout.projection_orientation}->{resolved_projection_orientation}. "
            f"Set {ENV_WEIGHT_STRUCTURAL_CONVERSION}=convert to allow."
        )

    qkv_weights_by_layer: dict[int, dict[str, str]] = {}
    qkv_biases_by_layer: dict[int, dict[str, str]] = {}

    def _put(dst: str, spec: _Spec) -> None:
        prev = mapping.get(dst)
        if prev is not None:
            raise KeyMappingError(f"sdxl_clip: duplicate destination key {dst!r} (collision)")
        mapping[dst] = spec

    def _capture_qkv(dst_key: str, raw_key: str) -> bool:
        # For fused-QKV output, consume split q/k/v projections from HF/Codex input keys
        # and later synthesize `self_attn.in_proj.{weight,bias}` via concatenation.
        if not wants_fused_qkv:
            return False
        parts = dst_key.split(".")
        if len(parts) < 8:
            return False
        if parts[0] != "transformer" or parts[1] != "text_model":
            return False
        if parts[2] != "encoder" or parts[3] != "layers" or not parts[4].isdigit():
            return False
        layer = int(parts[4])
        if layer < 0 or layer >= int(num_layers):
            return False
        if parts[5] != "self_attn":
            return False
        proj = parts[6]
        if proj not in {"q_proj", "k_proj", "v_proj"}:
            return False
        tail = parts[7:]
        if len(tail) != 1 or tail[0] not in {"weight", "bias"}:
            return False
        proj_letter = proj[0]
        if tail[0] == "weight":
            qkv_weights_by_layer.setdefault(layer, {})[proj_letter] = raw_key
        else:
            qkv_biases_by_layer.setdefault(layer, {})[proj_letter] = raw_key
        return True

    for raw_key, key in lookup_key_pairs:
        if key.endswith(".position_ids"):
            continue

        if key in _LOGIT_KEYS:
            if seen_logit is not None:
                raise KeyMappingError(
                    f"sdxl_clip: multiple logit_scale sources: {seen_logit!r},{key!r}"
                )
            seen_logit = key
            _put("logit_scale", _Direct(raw_key))
            continue

        if key in _PROJ_KEYS:
            if not keep_projection:
                continue
            if seen_proj is not None:
                raise KeyMappingError(f"sdxl_clip: multiple text_projection sources: {seen_proj!r},{key!r}")
            seen_proj = key
            if resolved_projection_orientation == "none":
                raise KeyMappingError(
                    "sdxl_clip: projection key was found but resolved orientation is 'none'; "
                    "this indicates inconsistent cached/override metadata."
                )
            if resolved_projection_orientation == detected_layout.projection_orientation:
                _put("transformer.text_projection.weight", _Direct(raw_key))
            else:
                _put("transformer.text_projection.weight", _Transpose(raw_key))
            continue

        if style is KeyStyle.CODEX:
            if key.startswith("transformer.text_model."):
                if _capture_qkv(key, raw_key):
                    continue
                _put(key, _Direct(raw_key))
                continue
            raise KeyMappingError(f"sdxl_clip: unsupported CODEX key {key!r}")

        if style is KeyStyle.HF:
            if key.startswith("text_model."):
                dst = f"transformer.{key}"
                if _capture_qkv(dst, raw_key):
                    continue
                _put(dst, _Direct(raw_key))
                continue
            raise KeyMappingError(f"sdxl_clip: unsupported HF key {key!r}")

        if style is KeyStyle.OPENCLIP:
            if key == "positional_embedding":
                _put("transformer.text_model.embeddings.position_embedding.weight", _Direct(raw_key))
                continue
            if key == "token_embedding.weight":
                _put("transformer.text_model.embeddings.token_embedding.weight", _Direct(raw_key))
                continue
            if key in {"ln_final.weight", "ln_final.bias"}:
                suffix = "weight" if key.endswith(".weight") else "bias"
                _put(f"transformer.text_model.final_layer_norm.{suffix}", _Direct(raw_key))
                continue
            if not key.startswith("transformer.resblocks."):
                raise KeyMappingError(f"sdxl_clip: unsupported OpenCLIP key {key!r}")

            parts = key.split(".")
            if len(parts) < 5 or parts[0] != "transformer" or parts[1] != "resblocks" or not parts[2].isdigit():
                raise KeyMappingError(f"sdxl_clip: unsupported OpenCLIP resblock key {key!r}")
            layer = int(parts[2])
            if layer < 0 or layer >= int(num_layers):
                raise KeyMappingError(
                    f"sdxl_clip: OpenCLIP resblock index out of range (layer={layer}, num_layers={num_layers}) for key={key!r}"
                )

            tail = parts[3:]
            base = f"transformer.text_model.encoder.layers.{layer}."

            # OpenCLIP layout:
            # - ln_1/ln_2: transformer.resblocks.{i}.ln_1.{weight,bias}
            # - mlp: transformer.resblocks.{i}.mlp.c_fc.{weight,bias} / c_proj.{weight,bias}
            # - attn: transformer.resblocks.{i}.attn.{in_proj_weight,in_proj_bias,out_proj.{weight,bias}}
            if len(tail) == 2 and tail[0] in {"ln_1", "ln_2"} and tail[1] in {"weight", "bias"}:
                mapped = "layer_norm1" if tail[0] == "ln_1" else "layer_norm2"
                _put(base + f"{mapped}.{tail[1]}", _Direct(raw_key))
                continue

            if len(tail) == 3 and tail[0] == "mlp" and tail[1] in {"c_fc", "c_proj"} and tail[2] in {"weight", "bias"}:
                mapped = "mlp.fc1" if tail[1] == "c_fc" else "mlp.fc2"
                _put(base + f"{mapped}.{tail[2]}", _Direct(raw_key))
                continue

            if len(tail) == 2 and tail[0] == "attn" and tail[1] in {"in_proj_weight", "in_proj_bias"}:
                suffix = "weight" if tail[1].endswith("_weight") else "bias"
                if wants_fused_qkv:
                    _put(base + f"self_attn.in_proj.{suffix}", _Direct(raw_key))
                else:
                    _put(base + f"self_attn.q_proj.{suffix}", _SliceQKV(raw_key, 0))
                    _put(base + f"self_attn.k_proj.{suffix}", _SliceQKV(raw_key, 1))
                    _put(base + f"self_attn.v_proj.{suffix}", _SliceQKV(raw_key, 2))
                continue

            if len(tail) == 3 and tail[0] == "attn" and tail[1] == "out_proj" and tail[2] in {"weight", "bias"}:
                _put(base + f"self_attn.out_proj.{tail[2]}", _Direct(raw_key))
                continue

            raise KeyMappingError(f"sdxl_clip: unsupported OpenCLIP resblock key {key!r}")

        raise KeyMappingError(f"sdxl_clip: unsupported detected style={style.value!r}")

    if "logit_scale" not in mapping:
        _put("logit_scale", _DefaultLogitScale(log(100.0)))

    if keep_projection and require_projection and "transformer.text_projection.weight" not in mapping:
        raise KeyMappingError(
            "sdxl_clip: projection weights are required for this encoder but were not found "
            "(expected one of: %s)" % (", ".join(_PROJ_KEYS),)
        )

    if wants_fused_qkv:
        for layer, weights in qkv_weights_by_layer.items():
            if set(weights.keys()) != {"q", "k", "v"}:
                missing = sorted({"q", "k", "v"} - set(weights.keys()))
                raise KeyMappingError(
                    "sdxl_clip: cannot build fused in_proj weights (missing split projections). "
                    f"layer={layer} missing={missing}"
                )
            base = f"transformer.text_model.encoder.layers.{layer}."
            _put(base + "self_attn.in_proj.weight", _ConcatQKV(weights["q"], weights["k"], weights["v"]))

        for layer, biases in qkv_biases_by_layer.items():
            if set(biases.keys()) != {"q", "k", "v"}:
                missing = sorted({"q", "k", "v"} - set(biases.keys()))
                raise KeyMappingError(
                    "sdxl_clip: cannot build fused in_proj bias (missing split projections). "
                    f"layer={layer} missing={missing}"
                )
            base = f"transformer.text_model.encoder.layers.{layer}."
            _put(base + "self_attn.in_proj.bias", _ConcatQKV(biases["q"], biases["k"], biases["v"]))

    essentials = _ESSENTIAL_KEYS_FUSED if wants_fused_qkv else _ESSENTIAL_KEYS_SPLIT
    missing_essentials = [key for key in essentials if key not in mapping]
    if missing_essentials:
        sample = ", ".join(missing_essentials[:3])
        raise KeyMappingError(
            "sdxl_clip: key mapping failed (missing essential tensors). "
            f"missing_sample=[{sample}] style={style.value}"
        )

    # Ensure output is canonical (no source-keyspace remnants).
    forbidden = []
    for out_key in mapping.keys():
        if out_key.startswith("text_model.") or out_key.startswith("transformer.resblocks."):
            forbidden.append(out_key)
    if forbidden:
        raise KeyMappingError(f"sdxl_clip: produced non-canonical keys (sample={sorted(forbidden)[:10]})")

    resolved_layout = ClipLayoutMetadata(
        qkv_layout=resolved_qkv_layout,
        projection_orientation=resolved_projection_orientation,
        source_style=style.value,
    )
    canonical_to_source = {destination: _spec_source(spec) for destination, spec in mapping.items()}
    return ResolvedKeyspace(
        style=style,
        canonical_to_source=canonical_to_source,
        metadata={
            "resolver": "sdxl_clip",
            "qkv_layout": resolved_layout.qkv_layout,
            "projection_orientation": resolved_layout.projection_orientation,
            "source_style": resolved_layout.source_style,
            "keep_projection": bool(keep_projection),
        },
        view=_SDXLCLIPKeymapView(state_dict, mapping),
    )


def resolve_clip_keyspace_with_layout(
    state_dict: MutableMapping[str, _T],
    *,
    num_layers: int,
    keep_projection: bool,
    qkv_impl: _QKVImpl = "auto",
    projection_orientation: _ProjectionOrientationTarget = "auto",
    layout_metadata: ClipLayoutMetadata | None = None,
    require_projection: bool = True,
) -> ResolvedKeyspace[_T]:
    return _resolve_clip_keyspace(
        state_dict,
        num_layers=num_layers,
        keep_projection=keep_projection,
        qkv_impl=qkv_impl,
        projection_orientation=projection_orientation,
        layout_metadata=layout_metadata,
        require_projection=require_projection,
    )


def resolve_sdxl_clip_l_keyspace_with_layout(
    state_dict: MutableMapping[str, _T],
    *,
    qkv_impl: _QKVImpl = "auto",
    layout_metadata: ClipLayoutMetadata | None = None,
) -> ResolvedKeyspace[_T]:
    """Keymap SDXL base CLIP-L weights (text_encoder) into Codex IntegratedCLIP keys."""

    return resolve_clip_keyspace_with_layout(
        state_dict,
        num_layers=12,
        keep_projection=False,
        qkv_impl=qkv_impl,
        projection_orientation="auto",
        layout_metadata=layout_metadata,
        require_projection=False,
    )


def resolve_sdxl_clip_l_keyspace(
    state_dict: MutableMapping[str, _T],
    *,
    qkv_impl: _QKVImpl = "auto",
) -> ResolvedKeyspace[_T]:
    return resolve_sdxl_clip_l_keyspace_with_layout(
        state_dict,
        qkv_impl=qkv_impl,
        layout_metadata=None,
    )


def resolve_sdxl_clip_g_keyspace_with_layout(
    state_dict: MutableMapping[str, _T],
    *,
    qkv_impl: _QKVImpl = "auto",
    projection_orientation: _ProjectionOrientationTarget = "auto",
    layout_metadata: ClipLayoutMetadata | None = None,
) -> ResolvedKeyspace[_T]:
    """Keymap SDXL base CLIP-G weights (text_encoder_2) into Codex IntegratedCLIP keys."""

    return resolve_clip_keyspace_with_layout(
        state_dict,
        num_layers=32,
        keep_projection=True,
        qkv_impl=qkv_impl,
        projection_orientation=projection_orientation,
        layout_metadata=layout_metadata,
        require_projection=True,
    )


def resolve_sdxl_clip_g_keyspace(
    state_dict: MutableMapping[str, _T],
    *,
    qkv_impl: _QKVImpl = "auto",
    projection_orientation: _ProjectionOrientationTarget = "auto",
) -> ResolvedKeyspace[_T]:
    return resolve_sdxl_clip_g_keyspace_with_layout(
        state_dict,
        qkv_impl=qkv_impl,
        projection_orientation=projection_orientation,
        layout_metadata=None,
    )


__all__ = [
    "ClipLayoutMetadata",
    "clip_layout_metadata_from_resolved",
    "detect_clip_layout_metadata",
    "resolve_clip_keyspace_with_layout",
    "resolve_sdxl_clip_g_keyspace_with_layout",
    "resolve_sdxl_clip_g_keyspace",
    "resolve_sdxl_clip_l_keyspace_with_layout",
    "resolve_sdxl_clip_l_keyspace",
]
