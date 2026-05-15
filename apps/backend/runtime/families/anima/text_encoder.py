"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima Qwen3-0.6B text encoder runtime + offline tokenizers (Qwen + T5).
Loads sha-selected Qwen3-0.6B weights through strict Qwen keyspace resolution or GGUF keyspace resolution into the native Qwen3
implementation and provides the
family-owned prompt/tokenization contract used by runtime conditioning and API prompt counting. Also loads an offline T5 tokenizer used only
for dual-tokenization (token ids + weights + attention mask).

No runtime downloads are allowed: tokenizers must be resolved from vendored `apps/backend/huggingface/**` assets or explicit paths.

Symbols (top-level; keep in sync; no ghosts):
- `AnimaQwenTextEncoder` (class): Qwen3-0.6B encoder wrapper (native model + tokenizer).
- `AnimaQwenTextProcessingEngine` (class): Thin adapter exposing `__call__`, `tokenize`, and `tokenize_with_weights`.
- `AnimaT5TokenBatch` (dataclass): T5 tokenization batch (`input_ids`, `weights`, `attention_mask`) for Anima conditioning.
- `load_anima_qwen_tokenizer` (function): Offline slow `Qwen2Tokenizer` loader for Anima prompt/token parity.
- `load_anima_qwen3_06b_text_encoder` (function): Strict loader for Qwen3-0.6B weights (safetensors or GGUF; sha-selected).
- `load_anima_t5_tokenizer` (function): Offline T5 tokenizer loader (used for Anima dual-tokenization ids/weights).
- `resolve_anima_qwen_max_length` (function): Resolve fail-loud max-length policy for Anima Qwen tokenization.
- `resolve_anima_t5_max_length` (function): Resolve fail-loud max-length policy for Anima T5 tokenization.
- `tokenize_qwen_with_weights` (function): Qwen tokenization helper with optional `return_word_ids` metadata parity.
- `tokenize_t5_with_weights` (function): Offline T5 tokenization producing ids+weights+mask tensors for Anima conditioning.
- `count_anima_prompt_tokens` (function): Count Anima prompt tokens with the same family-owned runtime tokenization rules.
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, cast

import torch
import torch.nn as nn

from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.runtime.checkpoint.io import load_gguf_state_dict, load_torch_file
from apps.backend.runtime.checkpoint.safetensors_header import read_safetensors_header
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.state_dict.keymap_qwen_text_encoder import resolve_qwen_text_encoder_keyspace

logger = get_backend_logger("backend.runtime.anima.text_encoder")

ANIMA_MAX_LENGTH_DEFAULT = 99999999
_ESCAPED_RIGHT_PAREN = "\0\1"
_ESCAPED_LEFT_PAREN = "\0\2"
_QWEN3_06B_HIDDEN_SIZE = 1024
_QWEN3_06B_REQUIRED_SHAPES: dict[str, tuple[int, ...]] = {
    "model.layers.0.self_attn.q_proj.weight": (2048, 1024),
    "model.layers.0.self_attn.k_proj.weight": (1024, 1024),
    "model.layers.0.self_attn.v_proj.weight": (1024, 1024),
    "model.layers.0.self_attn.o_proj.weight": (1024, 2048),
    "model.layers.0.self_attn.q_norm.weight": (128,),
    "model.layers.0.self_attn.k_norm.weight": (128,),
    "model.layers.0.mlp.gate_proj.weight": (3072, 1024),
    "model.norm.weight": (1024,),
}


@dataclass(frozen=True, slots=True)
class AnimaT5TokenBatch:
    input_ids: torch.Tensor
    weights: torch.Tensor
    attention_mask: torch.Tensor


def _resolve_positive_length_env(*, env_var: str, default: int) -> int:
    raw = str(os.getenv(env_var, str(default)) or str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be an integer > 0, got: {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{env_var} must be > 0, got: {value}")
    return value


def resolve_anima_qwen_max_length() -> int:
    return _resolve_positive_length_env(
        env_var="CODEX_ANIMA_QWEN_MAX_LENGTH",
        default=ANIMA_MAX_LENGTH_DEFAULT,
    )


def resolve_anima_t5_max_length() -> int:
    return _resolve_positive_length_env(
        env_var="CODEX_ANIMA_T5_MAX_LENGTH",
        default=ANIMA_MAX_LENGTH_DEFAULT,
    )


def _resolve_dir_candidates(*, env_var: str, explicit: str | None, candidates: Iterable[Path]) -> list[Path]:
    repo_root = get_repo_root()
    out: list[Path] = []

    env_value = os.getenv(env_var)
    if env_value:
        out.append(Path(os.path.expanduser(env_value.strip())))
    if explicit:
        out.insert(0, Path(os.path.expanduser(str(explicit).strip())))

    out.extend(candidates)

    normalized: list[Path] = []
    for p in out:
        raw = str(p).strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        try:
            path = path.resolve()
        except Exception:
            path = path.absolute()
        normalized.append(path)
    return normalized


def _load_tokenizer_dir(
    *,
    env_var: str,
    explicit: str | None,
    candidates: Iterable[Path],
    loader_name: str,
    load_tokenizer: Callable[[str], Any],
) -> Any:
    tried: list[str] = []
    errors: list[str] = []
    for p in _resolve_dir_candidates(env_var=env_var, explicit=explicit, candidates=candidates):
        tried.append(str(p))
        if not p.exists() or not p.is_dir():
            continue
        try:
            tok = load_tokenizer(str(p))
            emit_backend_message(
                "Loaded tokenizer",
                logger=logger.name,
                loader=loader_name,
                path=p,
                env=env_var,
            )
            return tok
        except Exception as exc:  # noqa: BLE001 - try next candidate
            errors.append(f"{p}: {type(exc).__name__}: {exc}")

    detail = "\n".join(errors) if errors else "<no load errors captured>"
    raise RuntimeError(
        f"Failed to load an offline {loader_name} for {env_var}. "
        f"Set {env_var} or vendor the tokenizer under apps/backend/huggingface. "
        f"Tried: {tried}\nErrors:\n{detail}"
    )


def _escape_important(text: str) -> str:
    return str(text).replace("\\)", _ESCAPED_RIGHT_PAREN).replace("\\(", _ESCAPED_LEFT_PAREN)


def _unescape_important(text: str) -> str:
    return str(text).replace(_ESCAPED_RIGHT_PAREN, ")").replace(_ESCAPED_LEFT_PAREN, "(")


def _parse_parentheses(text: str) -> list[str]:
    result: list[str] = []
    current = ""
    nesting_level = 0
    for char in str(text):
        if char == "(":
            if nesting_level == 0:
                if current:
                    result.append(current)
                    current = "("
                else:
                    current = "("
            else:
                current += char
            nesting_level += 1
            continue
        if char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current + ")")
                current = ""
            else:
                current += char
            continue
        current += char
    if current:
        result.append(current)
    return result


def _token_weights(text: str, current_weight: float) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for item in _parse_parentheses(text):
        weight = current_weight
        if len(item) >= 2 and item[0] == "(" and item[-1] == ")":
            inner = item[1:-1]
            split_at = inner.rfind(":")
            weight *= 1.1
            if split_at > 0:
                try:
                    weight = float(inner[split_at + 1 :])
                    inner = inner[:split_at]
                except Exception:
                    pass
            out.extend(_token_weights(inner, weight))
            continue
        out.append((item, current_weight))
    return out


def _anima_weighted_segments(text: str) -> list[tuple[str, float]]:
    escaped = _escape_important(str(text or ""))
    weighted = _token_weights(escaped, 1.0)
    resolved = [(_unescape_important(segment), float(weight)) for segment, weight in weighted]
    for segment, _weight in resolved:
        for word in str(segment).split():
            if word.startswith("embedding:"):
                embedding_name = str(word[len("embedding:") :]).strip() or "<empty>"
                raise NotImplementedError(
                    "Anima textual inversion embeddings are not yet implemented. "
                    f"Unsupported prompt token: embedding:{embedding_name}"
                )
    return resolved


def _tokenize_segment_ids(*, tokenizer: Any, segment_text: str, tokenizer_adds_end_token: bool) -> list[int]:
    tokenized = tokenizer(
        str(segment_text),
        padding=False,
        truncation=False,
        verbose=False,
    )
    token_ids = tokenized.get("input_ids")
    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], int):
        ids_out = [int(token_id) for token_id in token_ids]
    elif isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        ids_out = [int(token_id) for token_id in token_ids[0]]
    else:
        raise RuntimeError("Prompt tokenizer did not return input_ids as list[int] or list[list[int]].")
    if tokenizer_adds_end_token and ids_out:
        ids_out = ids_out[:-1]
    return ids_out


def _default_qwen_tokenizer_candidates() -> list[Path]:
    repo_root = get_repo_root()
    return [
        repo_root / "apps" / "backend" / "huggingface" / "circlestone-labs" / "Anima" / "qwen25_tokenizer",
    ]


def _default_t5_tokenizer_candidates() -> list[Path]:
    repo_root = get_repo_root()
    return [
        repo_root / "apps" / "backend" / "huggingface" / "circlestone-labs" / "Anima" / "t5_tokenizer",
    ]


@dataclass(frozen=True, slots=True)
class _QwenTokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


QwenWeightedToken = tuple[int, float] | tuple[int, float, int]


def _normalize_qwen_weighted_token_entry(entry: object, *, return_word_ids: bool) -> QwenWeightedToken:
    if not isinstance(entry, (tuple, list)):
        raise RuntimeError(
            "Qwen weighted token entry must be tuple/list; "
            f"got {type(entry).__name__}."
        )
    if len(entry) < 2:
        raise RuntimeError(
            "Qwen weighted token entry must contain at least (token_id, weight). "
            f"Got len={len(entry)} entry={entry!r}."
        )

    try:
        token_id = int(entry[0])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Qwen weighted token has non-int token_id: {entry[0]!r}.") from exc
    try:
        weight = float(entry[1])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Qwen weighted token has non-float weight: {entry[1]!r}.") from exc

    if not return_word_ids:
        return (token_id, weight)

    if len(entry) < 3:
        raise RuntimeError(
            "Qwen weighted token entry is missing word_id while return_word_ids=True. "
            f"entry={entry!r}"
        )
    try:
        word_id = int(entry[2])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Qwen weighted token has non-int word_id: {entry[2]!r}.") from exc
    return (token_id, weight, word_id)


def _resolve_pad_token_id_for_qwen(*, tokenizer: Any, context: str) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise RuntimeError(f"{context}: tokenizer missing pad_token_id.")
    try:
        pad_id_int = int(pad_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{context}: tokenizer pad_token_id must be int; got {pad_id!r}.") from exc
    token_count: int | None = None
    try:
        token_count = int(len(tokenizer))
    except Exception:
        token_count = None
    if token_count is not None and not (0 <= pad_id_int < token_count):
        raise RuntimeError(
            f"{context}: tokenizer pad_token_id out of range: "
            f"pad_token_id={pad_id_int} len={token_count}."
        )
    return pad_id_int


def _tokenize_qwen_row(
    *,
    tokenizer: Any,
    text: str,
    max_length: int,
    return_word_ids: bool,
) -> tuple[list[QwenWeightedToken], bool]:
    row: list[QwenWeightedToken] = []
    word_id = 1
    for segment, _segment_weight in _anima_weighted_segments(text):
        segment_ids = _tokenize_segment_ids(
            tokenizer=tokenizer,
            segment_text=segment,
            tokenizer_adds_end_token=False,
        )
        for token_id in segment_ids:
            # ComfyUI Anima keeps Qwen token weights flat at 1.0 even when prompt weighting
            # is parsed for segment boundaries and T5-side weights.
            entry = (token_id, 1.0, word_id) if return_word_ids else (token_id, 1.0)
            row.append(_normalize_qwen_weighted_token_entry(entry, return_word_ids=return_word_ids))
        if segment_ids:
            word_id += 1

    synthetic_masked_pad = False
    if not row:
        pad_id = _resolve_pad_token_id_for_qwen(
            tokenizer=tokenizer,
            context="Anima Qwen weighted tokenization produced an empty sequence",
        )
        entry = (pad_id, 1.0, 0) if return_word_ids else (pad_id, 1.0)
        row.append(_normalize_qwen_weighted_token_entry(entry, return_word_ids=return_word_ids))
        synthetic_masked_pad = True

    if len(row) > max_length:
        raise ValueError(
            "Anima Qwen tokenization exceeded max_length=%d (len=%d). "
            "Reduce prompt length or increase CODEX_ANIMA_QWEN_MAX_LENGTH."
            % (max_length, len(row))
        )
    return row, synthetic_masked_pad


def tokenize_qwen_with_weights(
    *,
    tokenizer: Any,
    texts: list[str],
    max_length: int = ANIMA_MAX_LENGTH_DEFAULT,
    return_word_ids: bool = False,
) -> list[list[QwenWeightedToken]]:
    """Tokenize Qwen prompts into tuples with Qwen weights pinned to `1.0` for Comfy parity."""
    max_length = int(max_length)
    if max_length <= 0:
        raise ValueError("max_length must be > 0")

    out: list[list[QwenWeightedToken]] = []
    for raw in texts:
        row, _synthetic_masked_pad = _tokenize_qwen_row(
            tokenizer=tokenizer,
            text=str(raw or ""),
            max_length=max_length,
            return_word_ids=return_word_ids,
        )
        out.append(row)
    return out


class AnimaQwenTextEncoder(nn.Module):
    """Qwen3-0.6B text encoder (native)."""

    def __init__(self, *, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self._tokenizer: Any | None = None
        self._tokenizer_path_hint: str | None = None

    def set_tokenizer_path_hint(self, tokenizer_path: str | None) -> None:
        value = str(tokenizer_path).strip() if tokenizer_path is not None else ""
        self._tokenizer_path_hint = value or None

    def _require_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        hint = self._tokenizer_path_hint
        tok = load_anima_qwen_tokenizer(hint)
        self._tokenizer = tok
        return tok

    def tokenize(self, texts: list[str], *, max_length: int) -> _QwenTokenBatch:
        tok = self._require_tokenizer()
        token_rows: list[list[QwenWeightedToken]] = []
        synthetic_masked_pad_rows: list[bool] = []
        for text in texts:
            row, synthetic_masked_pad = _tokenize_qwen_row(
                tokenizer=tok,
                text=str(text or ""),
                max_length=int(max_length),
                return_word_ids=False,
            )
            token_rows.append(row)
            synthetic_masked_pad_rows.append(synthetic_masked_pad)

        pad_id = _resolve_pad_token_id_for_qwen(
            tokenizer=tok,
            context="Anima Qwen tokenizer produced an empty sequence for prompt(s)",
        )
        batch_size = len(token_rows)
        max_len = max((len(row) for row in token_rows), default=1)
        input_ids = torch.full((batch_size, max_len), fill_value=pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for index, row in enumerate(token_rows):
            token_ids = [int(token_id) for token_id, _weight in row]
            input_ids[index, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            if not synthetic_masked_pad_rows[index]:
                attention_mask[index, : len(token_ids)] = 1
        return _QwenTokenBatch(input_ids=input_ids, attention_mask=attention_mask)

    def tokenize_with_weights(
        self,
        texts: list[str],
        *,
        max_length: int,
        return_word_ids: bool = False,
    ) -> list[list[QwenWeightedToken]]:
        tok = self._require_tokenizer()
        return tokenize_qwen_with_weights(
            tokenizer=tok,
            texts=texts,
            max_length=max_length,
            return_word_ids=return_word_ids,
        )

    @torch.no_grad()
    def encode(self, texts: list[str], *, max_length: int) -> torch.Tensor:
        batch = self.tokenize(texts, max_length=max_length)
        input_ids = batch.input_ids.to(device=self.device, dtype=torch.long)
        attention_mask = batch.attention_mask.to(device=self.device, dtype=torch.long)

        hidden, _intermediate = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(f"Qwen3 model returned invalid hidden states: {type(hidden).__name__} shape={getattr(hidden,'shape',None)}")
        return hidden.to(dtype=self.dtype)

    @property
    def device(self) -> torch.device:
        preferred = getattr(self.model, "preferred_device", None)
        if isinstance(preferred, torch.device):
            return preferred
        if isinstance(preferred, str):
            try:
                return torch.device(preferred)
            except Exception:
                pass
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return memory_management.manager.cpu_device

    @property
    def dtype(self) -> torch.dtype:
        preferred = getattr(self.model, "preferred_dtype", None)
        if isinstance(preferred, torch.dtype):
            return preferred
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32


class AnimaQwenTextProcessingEngine:
    """Thin adapter providing a consistent callable interface around `AnimaQwenTextEncoder`."""

    def __init__(self, text_encoder: AnimaQwenTextEncoder, *, max_length: int = ANIMA_MAX_LENGTH_DEFAULT) -> None:
        self.text_encoder = text_encoder
        self.max_length = int(max_length)

    def __call__(self, texts: list[str]) -> torch.Tensor:
        return self.text_encoder.encode(texts, max_length=self.max_length)

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        batch = self.text_encoder.tokenize(texts, max_length=self.max_length)
        return batch.input_ids.tolist()

    def tokenize_with_weights(
        self,
        texts: list[str],
        *,
        return_word_ids: bool = False,
    ) -> list[list[QwenWeightedToken]]:
        return self.text_encoder.tokenize_with_weights(
            texts,
            max_length=self.max_length,
            return_word_ids=return_word_ids,
        )


def _validate_qwen3_06b_header(*, header: Mapping[str, object], context: str) -> None:
    header_keys = {str(key): value for key, value in header.items()}
    try:
        resolved = resolve_qwen_text_encoder_keyspace(
            header_keys,
            allow_lm_head_aux=True,
            allow_visual_aux=False,
            require_backbone_keys=True,
        )
        normalized_header: Mapping[str, object] = resolved.view
    except Exception as exc:  # noqa: BLE001 - surfaced as strict header validation context
        raise RuntimeError(f"Qwen3-0.6B header key mapping failed: {exc} ({context})") from exc

    def _shape(key: str) -> tuple[int, ...] | None:
        meta = normalized_header.get(key)
        if isinstance(meta, dict):
            shape = meta.get("shape")
            if isinstance(shape, (list, tuple)) and all(isinstance(x, (int, float)) for x in shape):
                return tuple(int(x) for x in shape)
        return None

    embed = _shape("model.embed_tokens.weight")
    if embed is None or len(embed) != 2:
        raise RuntimeError(f"Qwen3-0.6B header missing model.embed_tokens.weight shape: {context}")
    vocab, hidden = int(embed[0]), int(embed[1])
    if hidden != _QWEN3_06B_HIDDEN_SIZE:
        raise RuntimeError(
            f"Qwen3-0.6B embed dim mismatch for {context}: got {hidden}, expected {_QWEN3_06B_HIDDEN_SIZE}."
        )
    if vocab <= 0:
        raise RuntimeError(f"Qwen3-0.6B vocab_size invalid for {context}: {vocab}.")

    _validate_qwen3_06b_required_shapes(shape_for=_shape, context=context)


def _shape_of_mapping(mapping: Mapping[str, object], key: str) -> tuple[int, ...] | None:
    shape_getter = getattr(mapping, "shape_of", None)
    if callable(shape_getter):
        try:
            shape = shape_getter(key)
        except Exception:
            shape = None
        if shape is not None:
            try:
                return tuple(int(v) for v in shape)
            except Exception:
                return None
    value = mapping.get(key)
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(v) for v in shape)
    except Exception:
        return None


def _validate_qwen3_06b_required_shapes(
    *,
    shape_for: Callable[[str], tuple[int, ...] | None],
    context: str,
) -> None:
    for key, expected_shape in _QWEN3_06B_REQUIRED_SHAPES.items():
        shape = shape_for(key)
        if shape is None:
            raise RuntimeError(
                "Qwen3-0.6B weights file does not look like a compatible Qwen text-encoder checkpoint. "
                f"Missing key: {key} ({context})"
            )
        if tuple(shape) != tuple(expected_shape):
            raise RuntimeError(
                f"Qwen3-0.6B shape mismatch for {key} at {context}: "
                f"got {shape}, expected {expected_shape}."
            )


def _validate_qwen3_06b_state_dict(*, state_dict: Mapping[str, object], context: str) -> None:
    embed_shape = _shape_of_mapping(state_dict, "model.embed_tokens.weight")
    if embed_shape is None or len(embed_shape) != 2:
        raise RuntimeError(f"Qwen3-0.6B state_dict missing model.embed_tokens.weight shape: {context}")
    vocab_size, hidden_size = int(embed_shape[0]), int(embed_shape[1])
    if hidden_size != _QWEN3_06B_HIDDEN_SIZE:
        raise RuntimeError(
            f"Qwen3-0.6B embed dim mismatch for {context}: got {hidden_size}, expected {_QWEN3_06B_HIDDEN_SIZE}."
        )
    if vocab_size <= 0:
        raise RuntimeError(f"Qwen3-0.6B vocab_size invalid for {context}: {vocab_size}.")

    _validate_qwen3_06b_required_shapes(
        shape_for=lambda key: _shape_of_mapping(state_dict, key),
        context=context,
    )


def load_anima_qwen3_06b_text_encoder(
    tenc_path: str,
    *,
    torch_dtype: torch.dtype,
    device: torch.device | str,
) -> AnimaQwenTextEncoder:
    raw = str(tenc_path or "").strip()
    if not raw:
        raise ValueError("Anima Qwen3-0.6B text encoder path is required.")
    p = Path(os.path.expanduser(raw))
    try:
        p = p.resolve()
    except Exception:
        p = p.absolute()
    if device is None:
        raise ValueError("Anima Qwen3-0.6B text encoder loader requires an explicit owner device.")
    if not p.exists() or not p.is_file():
        raise RuntimeError(f"Anima Qwen3-0.6B text encoder path not found: {p}")
    load_device = torch.device(device)
    to_args = dict(device=load_device, dtype=torch_dtype)

    from apps.backend.runtime.families.zimage.qwen3 import Qwen3_06B, resolve_qwen3_gguf_keyspace

    suffix = p.suffix.lower()
    if suffix in {".safetensor", ".safetensors"}:
        header = read_safetensors_header(p)
        _validate_qwen3_06b_header(header=header, context=str(p))

        sd = load_torch_file(str(p), device="cpu")
        if not isinstance(sd, Mapping):
            raise RuntimeError(f"Anima Qwen3-0.6B loader returned non-mapping state_dict: {type(sd).__name__}")
        non_string_keys = [repr(key) for key in sd.keys() if not isinstance(key, str)]
        if non_string_keys:
            raise RuntimeError(
                "Anima Qwen3-0.6B state_dict keys must be strings. "
                f"non_string_keys_sample={non_string_keys[:10]}"
            )
        state_dict = cast(Mapping[str, torch.Tensor], sd)
        try:
            resolved = resolve_qwen_text_encoder_keyspace(
                state_dict,
                allow_lm_head_aux=True,
                allow_visual_aux=False,
                require_backbone_keys=True,
            )
            key_style = resolved.style
            sd = resolved.view
            style_label = key_style.value if hasattr(key_style, "value") else str(key_style)
            emit_backend_message(
                "Anima Qwen3-0.6B keymap style",
                logger=logger.name,
                level=logging.DEBUG,
                style=style_label,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as strict load-time context
            raise RuntimeError(f"Anima Qwen3-0.6B key mapping failed: {exc}") from exc
        _validate_qwen3_06b_state_dict(state_dict=sd, context=f"{p} (resolved keyspace)")

        with using_codex_operations(**to_args, manual_cast_enabled=True):
            model = Qwen3_06B(dtype=torch_dtype).to(**to_args)
    elif suffix == ".gguf":
        try:
            gguf_state_dict = load_gguf_state_dict(
                str(p),
                dequantize=False,
                computation_dtype=torch_dtype,
                device=load_device,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as strict load-time context
            raise RuntimeError(f"Anima Qwen3-0.6B GGUF load failed: {exc}") from exc
        if not isinstance(gguf_state_dict, Mapping):
            raise RuntimeError(
                "Anima Qwen3-0.6B GGUF loader returned non-mapping state_dict: "
                f"{type(gguf_state_dict).__name__}"
            )
        non_string_keys = [repr(key) for key in gguf_state_dict.keys() if not isinstance(key, str)]
        if non_string_keys:
            raise RuntimeError(
                "Anima Qwen3-0.6B GGUF state_dict keys must be strings. "
                f"non_string_keys_sample={non_string_keys[:10]}"
            )
        try:
            sd = resolve_qwen3_gguf_keyspace(gguf_state_dict, num_layers=28)
        except Exception as exc:  # noqa: BLE001 - surfaced as strict load-time context
            raise RuntimeError(f"Anima Qwen3-0.6B GGUF keyspace resolution failed: {exc}") from exc
        _validate_qwen3_06b_state_dict(state_dict=sd, context=f"{p} (GGUF resolved keyspace)")

        with using_codex_operations(**to_args, manual_cast_enabled=True, weight_format="gguf"):
            model = Qwen3_06B(dtype=torch_dtype).to(**to_args)
            model.load_sd(sd)
    else:
        raise ValueError(f"Anima Qwen3-0.6B text encoder must be a .safetensors or .gguf file, got: {p}")

    model.preferred_device = load_device
    model.preferred_dtype = torch_dtype
    if suffix in {".safetensor", ".safetensors"}:
        missing, unexpected = safe_load_state_dict(model, sd, log_name="anima.qwen3_06b")
        if missing or unexpected:
            raise RuntimeError(
                "Anima Qwen3-0.6B strict load failed: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
            )

    model.eval()
    return AnimaQwenTextEncoder(model=model)


def load_anima_qwen_tokenizer(tokenizer_path: str | None = None) -> Any:
    try:
        from transformers import Qwen2Tokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("transformers.Qwen2Tokenizer is required to load the Anima Qwen tokenizer.") from exc

    tok = _load_tokenizer_dir(
        env_var="CODEX_ANIMA_QWEN_TOKENIZER_PATH",
        explicit=tokenizer_path,
        candidates=_default_qwen_tokenizer_candidates(),
        loader_name="Anima Qwen tokenizer (slow Qwen2Tokenizer)",
        load_tokenizer=lambda path: Qwen2Tokenizer.from_pretrained(path, local_files_only=True),
    )
    if not isinstance(tok, Qwen2Tokenizer):
        raise RuntimeError(
            f"Anima Qwen tokenizer loaded unexpected class {type(tok).__name__}; expected Qwen2Tokenizer."
        )
    if bool(getattr(tok, "is_fast", False)):
        raise RuntimeError("Anima Qwen tokenizer must load through the slow Qwen2Tokenizer path, not a fast tokenizer.")
    return tok


def load_anima_t5_tokenizer(tokenizer_path: str | None = None) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("transformers is required to load the Anima T5 tokenizer.") from exc

    tok = _load_tokenizer_dir(
        env_var="CODEX_ANIMA_T5_TOKENIZER_PATH",
        explicit=tokenizer_path,
        candidates=_default_t5_tokenizer_candidates(),
        loader_name="Anima T5 tokenizer",
        load_tokenizer=lambda path: AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True),
    )

    token_count: int | None = None
    try:
        token_count = int(len(tok))
    except Exception:
        token_count = None
    max_tokens = 32128
    if token_count is not None and token_count > max_tokens:
        raise RuntimeError(f"Anima T5 tokenizer is too large: got len={token_count}, expected <= {max_tokens}.")

    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        raise RuntimeError("Anima T5 tokenizer missing pad_token_id.")
    try:
        pad_id = int(pad_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Anima T5 tokenizer pad_token_id must be an int; got {pad_id!r}.") from exc
    if token_count is not None and not (0 <= pad_id < token_count):
        raise RuntimeError(f"Anima T5 tokenizer pad_token_id out of range: pad_token_id={pad_id} len={token_count}.")

    return tok


def _resolve_t5_pad_token_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise RuntimeError("Anima T5 tokenizer missing pad_token_id.")
    try:
        return int(pad_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Anima T5 tokenizer pad_token_id must be an int; got {pad_id!r}.") from exc


def _resolve_t5_end_token_id(tokenizer: Any) -> int:
    end_id = getattr(tokenizer, "eos_token_id", None)
    if end_id is None:
        raise RuntimeError("Anima T5 tokenizer missing eos_token_id required for Comfy parity.")
    try:
        return int(end_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Anima T5 tokenizer eos_token_id must be an int; got {end_id!r}.") from exc


def tokenize_t5_with_weights(
    *,
    tokenizer: Any,
    texts: list[str],
    max_length: int = ANIMA_MAX_LENGTH_DEFAULT,
) -> AnimaT5TokenBatch:
    """Tokenize texts into `(input_ids, weights, attention_mask)` for the Anima adapter."""
    max_length = int(max_length)
    if max_length <= 0:
        raise ValueError("max_length must be > 0")

    pad_id = _resolve_t5_pad_token_id(tokenizer)
    end_id = _resolve_t5_end_token_id(tokenizer)

    all_ids: list[list[int]] = []
    all_weights: list[list[float]] = []
    all_masks: list[list[int]] = []

    for raw in texts:
        ids: list[int] = []
        weights: list[float] = []
        for segment, weight in _anima_weighted_segments(str(raw or "")):
            segment_ids = _tokenize_segment_ids(
                tokenizer=tokenizer,
                segment_text=segment,
                tokenizer_adds_end_token=True,
            )
            for token_id in segment_ids:
                ids.append(int(token_id))
                weights.append(float(weight))
        ids.append(end_id)
        weights.append(1.0)
        if len(ids) > max_length:
            raise ValueError(
                "Anima T5 tokenization exceeded max_length=%d (len=%d). Reduce prompt length or increase CODEX_ANIMA_T5_MAX_LENGTH."
                % (max_length, len(ids))
            )
        all_ids.append(ids)
        all_weights.append(weights)
        all_masks.append([1] * len(ids))

    batch = len(all_ids)
    max_len = max(len(x) for x in all_ids) if all_ids else 1

    ids_out = torch.full((batch, max_len), pad_id, dtype=torch.long)
    w_out = torch.zeros((batch, max_len), dtype=torch.float32)
    mask_out = torch.zeros((batch, max_len), dtype=torch.long)
    for i, (ids, w, mask) in enumerate(zip(all_ids, all_weights, all_masks, strict=True)):
        ids_out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        w_out[i, : len(w)] = torch.tensor(w, dtype=torch.float32)
        mask_out[i, : len(mask)] = torch.tensor(mask, dtype=torch.long)

    return AnimaT5TokenBatch(
        input_ids=ids_out,
        weights=w_out,
        attention_mask=mask_out,
    )


def count_anima_prompt_tokens(
    prompt: str,
    *,
    qwen_tokenizer: Any,
    t5_tokenizer: Any,
    qwen_max_length: int | None = None,
    t5_max_length: int | None = None,
) -> int:
    qwen_max = int(qwen_max_length) if qwen_max_length is not None else resolve_anima_qwen_max_length()
    t5_max = int(t5_max_length) if t5_max_length is not None else resolve_anima_t5_max_length()
    qwen_tokens = tokenize_qwen_with_weights(
        tokenizer=qwen_tokenizer,
        texts=[str(prompt or "")],
        max_length=qwen_max,
        return_word_ids=False,
    )
    t5_batch = tokenize_t5_with_weights(
        tokenizer=t5_tokenizer,
        texts=[str(prompt or "")],
        max_length=t5_max,
    )
    t5_count = int(t5_batch.attention_mask[0].sum().item()) if t5_batch.attention_mask.numel() > 0 else 0
    return max(len(qwen_tokens[0]), t5_count)


__all__ = [
    "AnimaT5TokenBatch",
    "AnimaQwenTextEncoder",
    "AnimaQwenTextProcessingEngine",
    "load_anima_qwen_tokenizer",
    "load_anima_qwen3_06b_text_encoder",
    "load_anima_t5_tokenizer",
    "resolve_anima_qwen_max_length",
    "resolve_anima_t5_max_length",
    "tokenize_qwen_with_weights",
    "tokenize_t5_with_weights",
    "count_anima_prompt_tokens",
]
