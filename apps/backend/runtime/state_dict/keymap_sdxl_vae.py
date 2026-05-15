"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDXL VAE key-style detection + keyspace resolution (LDM-style + DIFFUSERS legacy aliases → canonical diffusers AutoencoderKL).
Resolves common LDM layouts into diffusers keyspace via lookup views, rejects wrapper/prefix rewrites, and fails loud on unsupported layouts.
Supports legacy mid-attention aliases under `mid.attn_1.*`, prefixed `mid.attn_1.to_*`, `mid.block_1.*`, and DIFFUSERS `mid_block.attentions.*.{query,key,value,proj_attn}.*`.
Drops only known training metadata keys (`model_ema.decay` / `model_ema.num_updates`) and fails loud on other unknown keys.
Projection normalization is lane-based and explicit:
- `linear_2d`: canonical pass-through (`[C_out, C_in]`);
- `conv1x1_4d`: canonical pass-through (`[C_out, C_in, 1, 1]`);
- any other shape fails loud.

Symbols (top-level; keep in sync; no ghosts):
- `strip_known_sdxl_vae_metadata` (function): Drops only the allowed SDXL/Flow16 non-weight metadata keys and fails loud on unknown non-weight keys.
- `resolve_sdxl_vae_keyspace` (function): Resolves SDXL/Flow16 VAE keys into canonical keyspace (`ResolvedKeyspace`).
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import TypeVar

from apps.backend.runtime.state_dict.key_mapping import (
    fail_on_key_name_rewrite,
    KeyMappingError,
    KeySentinel,
    KeyStyle,
    KeyStyleDetector,
    KeyStyleSpec,
    ResolvedKeyspace,
    SentinelKind,
    resolve_state_dict_keyspace,
)

_T = TypeVar("_T")

_PREFIXES = (
    "module.",
    "model.",
    "vae.",
    "first_stage_model.",
)

_WEIGHT_PREFIXES = (
    "encoder.",
    "decoder.",
    "quant_conv.",
    "post_quant_conv.",
)

# Some SDXL VAE weights files include training metadata tensors (e.g., EMA decay).
# These keys are not part of diffusers `AutoencoderKL.state_dict()`.
#
# Policy: allow (and drop) only the known metadata keys; anything else is an error.
_DROPPED_KEYS = (
    "model_ema.decay",
    "model_ema.num_updates",
)

_DETECTOR = KeyStyleDetector(
    name="sdxl_vae_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.DIFFUSERS,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "encoder.down_blocks."),
                KeySentinel(SentinelKind.PREFIX, "decoder.up_blocks."),
                KeySentinel(SentinelKind.SUBSTRING, ".mid_block.attentions.0."),
            ),
            min_sentinel_hits=1,
        ),
        KeyStyleSpec(
            style=KeyStyle.LDM,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "encoder.down."),
                KeySentinel(SentinelKind.PREFIX, "decoder.up."),
                KeySentinel(SentinelKind.SUBSTRING, ".mid.attn_1."),
                KeySentinel(SentinelKind.SUBSTRING, ".mid.block_"),
            ),
            min_sentinel_hits=1,
        ),
    ),
)


class _SDXLVAEKeyspaceView(MutableMapping[str, _T]):
    """Lazy keyspace lookup view with SDXL VAE projection-lane validation.

    Canonical mid-attention projection keys accept either 2D linear weights
    (`[C_out, C_in]`) or native 1x1 Conv2d weights (`[C_out, C_in, 1, 1]`).
    Any other shape fails loud.
    """

    def __init__(
        self,
        base: MutableMapping[str, _T],
        mapping: dict[str, str],
    ):
        self._base = base
        self._map = dict(mapping)
        self._cache: dict[str, _T] = {}

    @staticmethod
    def _is_projection_weight(key: str) -> bool:
        if ".mid_block.attentions.0." not in key:
            return False
        return key.endswith(
            (
                ".to_q.weight",
                ".to_k.weight",
                ".to_v.weight",
                ".to_out.0.weight",
            )
        )

    @staticmethod
    def _projection_lane(value: _T) -> str:
        ndim = getattr(value, "ndim", None)
        shape = getattr(value, "shape", None)
        if ndim == 2 and shape and len(shape) == 2:
            return "linear_2d"
        if ndim == 4 and shape and len(shape) == 4 and tuple(shape[-2:]) == (1, 1):
            return "conv1x1_4d"
        return "unsupported"

    @staticmethod
    def _shape_tuple(value: _T) -> tuple[int, ...] | None:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        try:
            return tuple(int(x) for x in shape)
        except Exception:
            return None

    def _validate_projection_weight(self, *, key: str, value: _T) -> _T:
        lane = self._projection_lane(value)
        if lane == "linear_2d":
            return value
        if lane == "conv1x1_4d":
            return value
        raise KeyMappingError(
            "SDXL VAE mid-attention projection lane mismatch. "
            "Expected 2D [C_out, C_in] or 4D [C_out, C_in, 1, 1]. "
            f"key={key!r} ndim={getattr(value, 'ndim', None)} shape={self._shape_tuple(value)}"
        )

    def __getitem__(self, k: str) -> _T:
        cached = self._cache.get(k)
        if cached is not None:
            return cached

        v = self._base[self._map[k]]
        if self._is_projection_weight(k):
            v = self._validate_projection_weight(key=k, value=v)
            self._cache[k] = v
        return v

    def __setitem__(self, k: str, v: _T) -> None:
        self._cache.pop(k, None)
        self._map[k] = k
        self._base[k] = v

    def __delitem__(self, k: str) -> None:
        self._cache.pop(k, None)
        old = self._map.pop(k, None)
        if old is not None and old in self._base:
            del self._base[old]

    def __iter__(self):
        return iter(self._map.keys())

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, k: object) -> bool:
        return k in self._map

    def keys(self):
        return list(self._map.keys())

    def items(self):
        for k in self._map.keys():
            yield k, self[k]


class _FilteredKeysView(MutableMapping[str, _T]):
    """Restrict a mutable mapping to an explicit key set (no eager tensor reads)."""

    def __init__(self, base: MutableMapping[str, _T], keys: Sequence[str]):
        self._base = base
        self._keys = tuple(keys)
        self._keys_set = set(self._keys)

    def __getitem__(self, k: str) -> _T:
        return self._base[k]

    def __setitem__(self, k: str, v: _T) -> None:
        self._base[k] = v
        if k not in self._keys_set:
            self._keys_set.add(k)
            self._keys = (*self._keys, k)

    def __delitem__(self, k: str) -> None:
        del self._base[k]
        if k in self._keys_set:
            self._keys_set.remove(k)
            self._keys = tuple(x for x in self._keys if x != k)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, k: object) -> bool:
        return k in self._keys_set

    def keys(self):
        return list(self._keys)


def strip_known_sdxl_vae_metadata(state_dict: MutableMapping[str, _T]) -> MutableMapping[str, _T]:
    """Drop only the known SDXL/Flow16 non-weight metadata keys (fail loud on any other non-weight key)."""

    raw_keys = list(state_dict.keys())
    kept_raw_keys: list[str] = []
    unknown_non_weight: list[str] = []

    for raw_key in raw_keys:
        source_key = fail_on_key_name_rewrite(str(raw_key), _PREFIXES)
        if source_key.startswith(_WEIGHT_PREFIXES):
            kept_raw_keys.append(raw_key)
            continue
        if source_key in _DROPPED_KEYS:
            continue
        unknown_non_weight.append(source_key)

    if unknown_non_weight:
        sample = sorted(set(unknown_non_weight))[:10]
        raise KeyMappingError(
            "SDXL VAE keyspace resolver refuses unknown non-weight keys. "
            f"unknown_sample={sample}"
        )

    return _FilteredKeysView(state_dict, kept_raw_keys)


def resolve_sdxl_vae_keyspace(state_dict: MutableMapping[str, _T]) -> ResolvedKeyspace[_T]:
    """Resolve SDXL/Flow16 VAE keys into diffusers AutoencoderKL layout (fail loud).

    Accepted inputs:
    - Diffusers-style VAE keys: `encoder.down_blocks.*`, `decoder.up_blocks.*`, `*.mid_block.*`
      (legacy mid-attention aliases `query/key/value/proj_attn` are canonicalized)
    - LDM-style SDXL VAE keys: `encoder.down.*`, `decoder.up.*`, `*.mid.attn_1.*`

    Output:
    - Canonical diffusers AutoencoderKL keys (no wrapper prefixes).
    - Mid-attention projection weights use explicit shape lanes:
      - 2D linear weights pass through untouched;
      - 1×1 conv weights pass through untouched;
      - non-canonical projection shapes fail loud.
    """

    filtered = strip_known_sdxl_vae_metadata(state_dict)

    def _diffusers_to_canonical(key: str) -> str:
        prefix = ""
        if key.startswith("encoder.mid_block.attentions."):
            prefix = "encoder.mid_block.attentions."
        elif key.startswith("decoder.mid_block.attentions."):
            prefix = "decoder.mid_block.attentions."
        else:
            return key

        suffix = key[len(prefix) :]
        parts = suffix.split(".")
        if len(parts) < 3:
            return key

        attention_index = parts[0]
        if not attention_index.isdigit():
            return key

        head = parts[1]
        rest = ".".join(parts[2:])
        if not rest:
            return key

        table = {
            "query": "to_q",
            "key": "to_k",
            "value": "to_v",
            "proj_attn": "to_out.0",
        }
        mapped = table.get(head)
        if mapped is None:
            return key
        if head == "proj_attn" and rest.startswith("0."):
            mapped = "to_out"
        return f"{prefix}{attention_index}.{mapped}.{rest}"

    def _ldm_to_diffusers(key: str) -> str:
        new_key = key

        if key.startswith("encoder.down."):
            parts = key.split(".")
            # encoder.down.{i}.block.{j}.rest...
            if len(parts) >= 6 and parts[2].isdigit() and parts[4].isdigit() and parts[3] == "block":
                i = int(parts[2])
                j = int(parts[4])
                rest = ".".join(parts[5:])
                new_key = f"encoder.down_blocks.{i}.resnets.{j}.{rest}"
            # encoder.down.{i}.downsample.rest...
            elif len(parts) >= 4 and parts[2].isdigit() and parts[3] == "downsample":
                i = int(parts[2])
                rest = ".".join(parts[4:])
                new_key = f"encoder.down_blocks.{i}.downsamplers.0.{rest}"

        elif key.startswith("decoder.up."):
            parts = key.split(".")
            # decoder.up.{k}.block.{j}.rest...
            if len(parts) >= 6 and parts[2].isdigit() and parts[4].isdigit() and parts[3] == "block":
                k = int(parts[2])
                j = int(parts[4])
                # SDXL VAE indexes up_blocks in reverse order vs. LDM-style up.{k}
                i = 3 - k
                rest = ".".join(parts[5:])
                new_key = f"decoder.up_blocks.{i}.resnets.{j}.{rest}"
            # decoder.up.{k}.upsample.rest...
            elif len(parts) >= 4 and parts[2].isdigit() and parts[3] == "upsample":
                k = int(parts[2])
                i = 3 - k
                rest = ".".join(parts[4:])
                new_key = f"decoder.up_blocks.{i}.upsamplers.0.{rest}"

        elif key.startswith("encoder.mid.block_") or key.startswith("decoder.mid.block_"):
            parts = key.split(".")
            prefix = "encoder" if parts[0] == "encoder" else "decoder"
            table = {
                "q": "to_q",
                "k": "to_k",
                "v": "to_v",
                "proj_out": "to_out.0",
                "norm": "group_norm",
                # older / alternative naming
                "query": "to_q",
                "key": "to_k",
                "value": "to_v",
                "proj_attn": "to_out.0",
                # already-prefixed variants seen in some exports
                "to_q": "to_q",
                "to_k": "to_k",
                "to_v": "to_v",
                "to_out": "to_out.0",
            }

            # Some SDXL VAE exports encode mid attention heads under `mid.block_1.*`
            # (or `mid.block_1.attn_1.*`) instead of `mid.attn_1.*`.
            if len(parts) >= 5 and parts[2] == "block_1":
                head = None
                rest = ""
                if parts[3] in table and len(parts) >= 5:
                    head = parts[3]
                    rest = ".".join(parts[4:])
                elif len(parts) >= 6 and parts[3] == "attn_1" and parts[4] in table:
                    head = parts[4]
                    rest = ".".join(parts[5:])
                if head is not None and rest:
                    mapped = table[head]
                    if head == "to_out" and rest.startswith("0."):
                        mapped = "to_out"
                    new_key = f"{prefix}.mid_block.attentions.0.{mapped}.{rest}"

            if len(parts) >= 4 and parts[2].startswith("block_") and new_key == key:
                try:
                    block_index = int(parts[2].split("_", 1)[1]) - 1
                except (IndexError, ValueError):
                    block_index = 0
                rest = ".".join(parts[3:])
                new_key = f"{prefix}.mid_block.resnets.{block_index}.{rest}"

        elif key.startswith("encoder.mid.attn_1.") or key.startswith("decoder.mid.attn_1."):
            is_encoder = key.startswith("encoder.")
            base = "encoder.mid.attn_1." if is_encoder else "decoder.mid.attn_1."
            suffix = key[len(base) :]
            prefix = "encoder" if is_encoder else "decoder"

            try:
                head, rest = suffix.split(".", 1)
            except ValueError:
                head, rest = suffix, ""

            table = {
                "q": "to_q",
                "k": "to_k",
                "v": "to_v",
                "proj_out": "to_out.0",
                "norm": "group_norm",
                # older / alternative naming
                "query": "to_q",
                "key": "to_k",
                "value": "to_v",
                "proj_attn": "to_out.0",
                # already-prefixed variants seen in some exports
                "to_q": "to_q",
                "to_k": "to_k",
                "to_v": "to_v",
                "to_out": "to_out.0",
            }

            mapped = table.get(head)
            if mapped is not None and rest:
                if head == "to_out" and rest.startswith("0."):
                    mapped = "to_out"
                new_key = f"{prefix}.mid_block.attentions.0.{mapped}.{rest}"

        if "nin_shortcut." in key:
            parts = key.split(".")
            # encoder.down.{i}.block.0.nin_shortcut.{weight,bias}
            if key.startswith("encoder.down.") and len(parts) >= 7 and parts[2].isdigit() and parts[3] == "block" and parts[4] == "0":
                i = int(parts[2])
                rest = ".".join(parts[6:])
                new_key = f"encoder.down_blocks.{i}.resnets.0.conv_shortcut.{rest}"
            # decoder.up.{k}.block.0.nin_shortcut.{weight,bias}
            elif key.startswith("decoder.up.") and len(parts) >= 7 and parts[2].isdigit() and parts[3] == "block" and parts[4] == "0":
                k = int(parts[2])
                i = 3 - k
                rest = ".".join(parts[6:])
                new_key = f"decoder.up_blocks.{i}.resnets.0.conv_shortcut.{rest}"

        if key.startswith("encoder.norm_out."):
            rest = key[len("encoder.norm_out.") :]
            new_key = f"encoder.conv_norm_out.{rest}"
        elif key.startswith("decoder.norm_out."):
            rest = key[len("decoder.norm_out.") :]
            new_key = f"decoder.conv_norm_out.{rest}"

        return new_key

    def _validate_output(keys: Sequence[str]) -> None:
        offenders: list[str] = []

        def _is_forbidden(k: str) -> bool:
            return (
                k.startswith("encoder.down.")
                or k.startswith("decoder.up.")
                or k.startswith("encoder.mid.attn_1.")
                or k.startswith("decoder.mid.attn_1.")
                or k.startswith("encoder.mid.block_")
                or k.startswith("decoder.mid.block_")
                or k.startswith("encoder.norm_out.")
                or k.startswith("decoder.norm_out.")
                or ".nin_shortcut." in k
                or (
                    (k.startswith("encoder.mid_block.attentions.") or k.startswith("decoder.mid_block.attentions."))
                    and (".query." in k or ".key." in k or ".value." in k or ".proj_attn." in k)
                )
            )

        for k in keys:
            if _is_forbidden(k):
                offenders.append(k)

        if offenders:
            sample = sorted(offenders)[:10]
            raise KeyMappingError(
                "SDXL VAE keyspace resolver produced non-canonical keys (mapping incomplete). "
                f"offenders_sample={sample}"
            )

        # Full VAE files always include down_blocks; require it when the tensor set is large enough.
        if len(keys) > 64 and not any(k.startswith("encoder.down_blocks.") for k in keys):
            preview = ", ".join(sorted(keys)[:10])
            raise KeyMappingError(
                "SDXL VAE keyspace resolver output is missing required encoder.down_blocks.* keys. "
                f"sample_keys=[{preview}]"
            )

    mappers = {
        KeyStyle.DIFFUSERS: _diffusers_to_canonical,
        KeyStyle.LDM: _ldm_to_diffusers,
    }
    resolved = resolve_state_dict_keyspace(
        filtered,
        detector=_DETECTOR,
        mappers=mappers,
        view_factory=lambda base, mapping: _SDXLVAEKeyspaceView(base, mapping),
        output_validator=_validate_output,
    )
    resolved.metadata.setdefault("resolver", "sdxl_vae")
    return resolved


__all__ = ["resolve_sdxl_vae_keyspace", "strip_known_sdxl_vae_metadata"]
