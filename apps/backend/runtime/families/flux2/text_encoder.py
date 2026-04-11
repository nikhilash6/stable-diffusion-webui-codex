"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: FLUX.2 Klein Qwen3-4B text encoder runtime for the truthful 4B/base-4B slice.
Wraps either a native Codex Qwen3-4B model (safetensors/GGUF overrides) or a Hugging Face `Qwen3ForCausalLM`
instance (vendored diffusers repos), applies the FLUX.2 chat template (`enable_thinking=False`), and returns the
concatenated intermediate hidden states `(9, 18, 27)` expected by `Flux2KleinPipeline`.

Symbols (top-level; keep in sync; no ghosts):
- `FLUX2_QWEN_HIDDEN_SIZE` (constant): Supported Qwen hidden size for the truthful FLUX.2 slice.
- `FLUX2_QWEN_HIDDEN_LAYERS` (constant): Intermediate hidden-state layers concatenated for FLUX.2 conditioning.
- `Flux2TextEncoder` (class): Text encoder wrapper supporting native Qwen3-4B and HF `Qwen3ForCausalLM` models.
- `Flux2TextProcessingEngine` (class): Thin callable/text-tokenization adapter around `Flux2TextEncoder`.
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import torch
import torch.nn as nn

from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.state_dict.keymap_qwen_text_encoder import resolve_qwen_text_encoder_keyspace

logger = get_backend_logger("backend.runtime.flux2.text_encoder")

FLUX2_QWEN_HIDDEN_SIZE = 2560
FLUX2_QWEN_HIDDEN_LAYERS: tuple[int, int, int] = (9, 18, 27)
_FLUX2_QWEN_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class Flux2TextEncoder(nn.Module):
    """FLUX.2 Qwen3-4B text encoder wrapper."""

    def __init__(
        self,
        qwen_model: nn.Module,
        *,
        hidden_size: int = FLUX2_QWEN_HIDDEN_SIZE,
        hidden_state_layers: Sequence[int] = FLUX2_QWEN_HIDDEN_LAYERS,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        self.model = qwen_model
        self.hidden_size = int(hidden_size)
        self.hidden_state_layers = tuple(int(v) for v in hidden_state_layers)
        self._tokenizer = tokenizer
        self._tokenizer_path_hint: str | None = None
        self._validate_hidden_size(self.hidden_size)

    @staticmethod
    def _validate_hidden_size(hidden_size: int) -> None:
        if int(hidden_size) != FLUX2_QWEN_HIDDEN_SIZE:
            raise RuntimeError(
                "Unsupported FLUX.2 text encoder hidden size. "
                f"Expected {FLUX2_QWEN_HIDDEN_SIZE}, got {hidden_size}."
            )

    def set_tokenizer(self, tokenizer: Any | None) -> None:
        self._tokenizer = tokenizer

    def set_tokenizer_path_hint(self, tokenizer_path: str | None) -> None:
        value = str(tokenizer_path).strip() if tokenizer_path is not None else ""
        self._tokenizer_path_hint = value or None

    @classmethod
    def from_pretrained_model(
        cls,
        model: nn.Module,
        *,
        tokenizer: Any | None = None,
    ) -> "Flux2TextEncoder":
        hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError(
                "FLUX.2 HF Qwen3 text encoder is missing hidden_size metadata."
            )
        return cls(model, hidden_size=int(hidden_size), tokenizer=tokenizer)

    @classmethod
    def from_gguf(cls, gguf_path: str, *, torch_dtype: torch.dtype = torch.bfloat16) -> "Flux2TextEncoder":
        from apps.backend.runtime.checkpoint.io import load_gguf_state_dict
        from apps.backend.runtime.families.zimage.qwen3 import (
            Qwen3_4B,
            Qwen3Config,
            resolve_qwen3_gguf_keyspace,
        )

        emit_backend_message(
            "Loading FLUX.2 Qwen3-4B text encoder from GGUF",
            logger=logger.name,
            path=gguf_path,
        )
        gguf_state_dict = load_gguf_state_dict(gguf_path)
        state_dict = resolve_qwen3_gguf_keyspace(gguf_state_dict, num_layers=36)

        with using_codex_operations(weight_format="gguf", manual_cast_enabled=True, device=None, dtype=torch_dtype):
            model = Qwen3_4B(Qwen3Config(), dtype=torch_dtype)
            model.load_sd(state_dict)
            model = model.to(dtype=torch_dtype)

        return cls(model, hidden_size=FLUX2_QWEN_HIDDEN_SIZE)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "Flux2TextEncoder":
        from apps.backend.runtime.families.zimage.qwen3 import Qwen3_4B, Qwen3Config

        if not isinstance(state_dict, Mapping):
            raise RuntimeError(
                "FLUX.2 Qwen3-4B state_dict must be a mapping; "
                f"got {type(state_dict).__name__}."
            )
        for raw_key in state_dict.keys():
            if not isinstance(raw_key, str):
                raise RuntimeError(
                    "FLUX.2 Qwen3-4B state_dict keys must be strings. "
                    f"Got {type(raw_key).__name__}."
                )

        emit_backend_message(
            "Loading FLUX.2 Qwen3-4B text encoder from state_dict",
            logger=logger.name,
            keys=len(state_dict),
        )
        resolved = resolve_qwen_text_encoder_keyspace(
            state_dict,
            allow_lm_head_aux=True,
            allow_visual_aux=True,
            require_backbone_keys=True,
        )
        resolved_state_dict = resolved.view

        with using_codex_operations(manual_cast_enabled=True, device=None, dtype=torch_dtype):
            model = Qwen3_4B(Qwen3Config(), dtype=torch_dtype)
            model.load_sd(resolved_state_dict)
            model = model.to(dtype=torch_dtype)

        return cls(model, hidden_size=FLUX2_QWEN_HIDDEN_SIZE)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return memory_management.manager.cpu_device

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def load_tokenizer(self, tokenizer_path: str | None = None) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("transformers is required to load the FLUX.2 tokenizer.") from exc

        repo_root = get_repo_root()
        candidates: list[Path] = []

        env_override = os.getenv("CODEX_FLUX2_TOKENIZER_PATH")
        if env_override:
            candidates.append(Path(os.path.expanduser(env_override.strip())))
        if tokenizer_path:
            candidates.insert(0, Path(os.path.expanduser(str(tokenizer_path).strip())))
        hint = getattr(self, "_tokenizer_path_hint", None)
        if hint:
            candidates.append(Path(os.path.expanduser(hint)))

        candidates.extend(
            [
                repo_root / "apps" / "backend" / "huggingface" / "black-forest-labs" / "FLUX.2-klein-4B" / "tokenizer",
                repo_root / "apps" / "backend" / "huggingface" / "black-forest-labs" / "FLUX.2-klein-base-4B" / "tokenizer",
            ]
        )

        tried: list[str] = []
        errors: list[str] = []
        for raw in candidates:
            path = raw
            if not path.is_absolute():
                path = repo_root / path
            try:
                path = path.resolve()
            except Exception:
                path = path.absolute()
            tried.append(str(path))
            if not path.exists() or not path.is_dir():
                continue
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=True)
                emit_backend_message(
                    "Loaded FLUX.2 tokenizer",
                    logger=logger.name,
                    path=path,
                )
                return self._tokenizer
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {type(exc).__name__}: {exc}")

        detail = "\n".join(errors) if errors else "<no load errors captured>"
        raise RuntimeError(
            "Failed to load an offline tokenizer for FLUX.2. "
            "Set CODEX_FLUX2_TOKENIZER_PATH or vendor the tokenizer under apps/backend/huggingface/black-forest-labs. "
            f"Tried: {tried}\nErrors:\n{detail}"
        )

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
        *,
        apply_template: bool = True,
    ) -> Dict[str, torch.Tensor]:
        tokenizer = self.load_tokenizer()

        processed: list[str] = []
        for raw in texts:
            text = str(raw or "")
            if not apply_template or text.startswith("<|im_start|>") or text.startswith("<|start_header_id|>"):
                processed.append(text)
                continue

            rendered: str | None = None
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    rendered = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                        [{"role": "user", "content": text}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    try:
                        rendered = tokenizer.apply_chat_template(
                            [{"role": "user", "content": text}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except Exception:
                        rendered = None
                except Exception:
                    rendered = None
            processed.append(rendered if isinstance(rendered, str) and rendered else _FLUX2_QWEN_TEMPLATE.format(text))

        tokens = tokenizer(
            processed,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=int(max_length),
        )
        return {
            "input_ids": tokens["input_ids"].to(self.device),
            "attention_mask": tokens["attention_mask"].to(self.device),
        }

    def _stack_hidden_states(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        if not hidden_states:
            raise RuntimeError("FLUX.2 Qwen3 did not return any hidden states.")
        max_index = max(self.hidden_state_layers)
        if len(hidden_states) <= max_index:
            raise RuntimeError(
                "FLUX.2 Qwen3 hidden-state contract mismatch: "
                f"need layers {self.hidden_state_layers}, got {len(hidden_states)} states."
            )
        selected = [hidden_states[idx] for idx in self.hidden_state_layers]
        if any(not isinstance(state, torch.Tensor) or state.ndim != 3 for state in selected):
            raise RuntimeError("FLUX.2 Qwen3 selected hidden states must be 3D tensors (B,S,C).")
        out = torch.stack(selected, dim=1)
        batch_size, num_layers, seq_len, hidden_dim = out.shape
        if int(hidden_dim) != FLUX2_QWEN_HIDDEN_SIZE:
            raise RuntimeError(
                "Unsupported FLUX.2 Qwen3 hidden-state width after selection. "
                f"Expected {FLUX2_QWEN_HIDDEN_SIZE}, got {hidden_dim}."
            )
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)
        return prompt_embeds.to(dtype=self.dtype)

    def _encode_native(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        model = self.model
        backbone = getattr(model, "model", None)
        if backbone is None:
            raise RuntimeError("Native FLUX.2 Qwen3 wrapper is missing the `.model` backbone.")

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        hidden_states = backbone.embed_tokens(input_ids)
        compute_dtype = getattr(model, "compute_dtype", None)
        if isinstance(compute_dtype, torch.dtype) and compute_dtype != hidden_states.dtype:
            hidden_states = hidden_states.to(dtype=compute_dtype)

        batch_size = int(hidden_states.shape[0])
        seq_len = int(hidden_states.shape[1])
        causal_mask = backbone._build_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            attention_mask=attention_mask,
        )

        all_hidden_states: list[torch.Tensor] = [hidden_states]
        for layer in backbone.layers:
            hidden_states = layer(hidden_states, causal_mask)
            all_hidden_states.append(hidden_states)

        return self._stack_hidden_states(all_hidden_states)

    def _encode_hf(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not isinstance(hidden_states, (tuple, list)):
            raise RuntimeError("FLUX.2 HF Qwen3 output is missing `hidden_states`.")
        return self._stack_hidden_states(hidden_states)

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        max_length: int = 512,
        *,
        apply_template: bool = True,
    ) -> torch.Tensor:
        tokens = self.tokenize(texts, max_length=max_length, apply_template=apply_template)
        if hasattr(self.model, "load_sd"):
            return self._encode_native(tokens)
        return self._encode_hf(tokens)

    def forward(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        return self.encode(texts, max_length=max_length)


class Flux2TextProcessingEngine:
    """Thin callable/tokenization adapter for FLUX.2 Qwen3 conditioning."""

    def __init__(
        self,
        text_encoder: Flux2TextEncoder,
        *,
        max_length: int = 512,
    ) -> None:
        self.text_encoder = text_encoder
        self.max_length = int(max_length)

    def __call__(self, texts: List[str]) -> torch.Tensor:
        return self.text_encoder.encode(texts, max_length=self.max_length)

    def tokenize(self, texts: Iterable[str]) -> List[List[int]]:
        tokens = self.text_encoder.tokenize(list(texts), max_length=self.max_length)
        return tokens["input_ids"].tolist()

    def prompt_lengths(self, prompt: str) -> tuple[int, int]:
        tokens = self.text_encoder.tokenize([str(prompt or "")], max_length=self.max_length)
        attention_mask = tokens["attention_mask"]
        current = int(attention_mask[0].sum().item())
        return current, self.max_length


__all__ = [
    "FLUX2_QWEN_HIDDEN_LAYERS",
    "FLUX2_QWEN_HIDDEN_SIZE",
    "Flux2TextEncoder",
    "Flux2TextProcessingEngine",
]
