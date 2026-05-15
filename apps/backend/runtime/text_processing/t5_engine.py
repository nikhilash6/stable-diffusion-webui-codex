"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Codex-native prompt processing for T5 text encoders (chunking + emphasis + padding).
Tokenizes prompts, applies emphasis weights, chunks/pads sequences to the required minimum length, and runs the T5 text encoder with the
selected TE device/dtype policy.

Symbols (top-level; keep in sync; no ghosts):
- `PromptChunkFix` (constant): Namedtuple describing a fix-up embedding splice (offset + embedding tensor).
- `PromptChunk` (dataclass): One token chunk (tokens + per-token multipliers) with append/pad helpers.
- `PromptEncoding` (dataclass): Chunked encoding container (chunks + token count) with max-length helper.
- `T5TextProcessingEngine` (class): Prompt processor/encoder for T5-based text encoders.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch

from apps.backend.infra.config.args import dynamic_args
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from . import emphasis, parsing

logger = get_backend_logger("backend.runtime.text_processing.t5_engine")

PromptChunkFix = namedtuple("PromptChunkFix", ["offset", "embedding"])


@dataclass
class PromptChunk:
    tokens: List[int] = field(default_factory=list)
    multipliers: List[float] = field(default_factory=list)

    def append(self, token: int, weight: float) -> None:
        self.tokens.append(token)
        self.multipliers.append(weight)

    def pad(self, target_length: int, pad_token: int, pad_multiplier: float = 1.0) -> None:
        deficit = target_length - len(self.tokens)
        if deficit <= 0:
            return
        self.tokens.extend([pad_token] * deficit)
        self.multipliers.extend([pad_multiplier] * deficit)

    def clone(self) -> "PromptChunk":
        return PromptChunk(tokens=self.tokens.copy(), multipliers=self.multipliers.copy())


@dataclass
class PromptEncoding:
    chunks: List[PromptChunk]
    token_count: int

    @property
    def max_chunk_length(self) -> int:
        return max((len(chunk.tokens) for chunk in self.chunks), default=0)


class T5TextProcessingEngine:
    """Codex-native prompt encoder for T5-based text encoders."""

    def __init__(self, text_encoder, tokenizer, emphasis_name: str = "Original", min_length: int = 256):
        super().__init__()
        if min_length <= 0:
            raise ValueError("min_length must be positive")

        self.text_encoder = text_encoder.transformer
        self.tokenizer = tokenizer
        self.emphasis_name = emphasis_name
        self.min_length = min_length
        self.id_end = 1
        self.id_pad = 0

        vocab = self.tokenizer.get_vocab()
        self.comma_token = vocab.get(",</w>", None)
        self.token_mults: Dict[int, float] = self._compute_token_multipliers(vocab)

        self._ensure_emphasis_instance()

    def _ensure_emphasis_instance(self) -> None:
        emphasis_key = dynamic_args.get("emphasis_name", self.emphasis_name)
        emphasis_cls = emphasis.get_current_option(emphasis_key)
        self.emphasis = emphasis_cls()
        logger.debug("Initialized emphasis strategy '%s'", emphasis_key)

    @staticmethod
    def _compute_token_multipliers(vocab: Dict[str, int]) -> Dict[int, float]:
        token_mults: Dict[int, float] = {}
        for text, identifier in vocab.items():
            mult = 1.0
            for char in text:
                if char == "[":
                    mult /= 1.1
                elif char == "]":
                    mult *= 1.1
                elif char == "(":
                    mult *= 1.1
                elif char == ")":
                    mult /= 1.1
            if mult != 1.0:
                token_mults[identifier] = mult
        return token_mults

    def tokenize(self, texts: Sequence[str]) -> List[List[int]]:
        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")
        result = self.tokenizer(texts, truncation=False, add_special_tokens=False, verbose=False)["input_ids"]
        logger.debug("Tokenized %d prompts", len(result))
        return result

    def encode_with_transformers(self, tokens: torch.Tensor) -> torch.Tensor:
        device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
        dtype = memory_management.manager.dtype_for_role(DeviceRole.TEXT_ENCODER)
        tokens = tokens.to(device=device, dtype=torch.long)
        if hasattr(self.text_encoder, "shared"):
            self.text_encoder.shared.to(device=device, dtype=dtype)
        outputs = self.text_encoder(input_ids=tokens)
        return outputs

    def _finalize_chunk(self, chunk: PromptChunk, chunks: List[PromptChunk]) -> None:
        chunk.append(self.id_end, 1.0)
        current_length = len(chunk.tokens)
        remaining = self.min_length - current_length
        if remaining > 0:
            chunk.pad(self.min_length, self.id_pad)
        chunks.append(chunk)

    def tokenize_line(self, line: str) -> PromptEncoding:
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
        tokenized_segments = self.tokenize([text for text, _ in parsed])

        chunks: List[PromptChunk] = []
        chunk = PromptChunk()
        token_count = 0

        def next_chunk() -> None:
            nonlocal chunk, token_count
            self._finalize_chunk(chunk, chunks)
            token_count += len(chunks[-1].tokens)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized_segments, parsed):
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue
            for token in tokens:
                chunk.append(token, weight)

        if chunk.tokens or not chunks:
            next_chunk()

        encoding = PromptEncoding(chunks=chunks, token_count=token_count)
        logger.debug(
            "Tokenized line '%s' into %d chunk(s) (%d tokens)",
            line,
            len(encoding.chunks),
            encoding.token_count,
        )
        return encoding

    def __call__(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("No prompts provided to T5TextProcessingEngine")

        self._ensure_emphasis_instance()

        cache: Dict[str, List[torch.Tensor]] = {}
        embeddings: List[torch.Tensor] = []

        for line in texts:
            if line in cache:
                embeddings.extend(cache[line])
                continue

            encoding = self.tokenize_line(line)
            max_len = max(encoding.max_chunk_length, self.min_length)
            chunk_embeddings: List[torch.Tensor] = []

            for chunk in encoding.chunks:
                padded_chunk = chunk.clone()
                padded_chunk.pad(max_len, self.id_pad)
                embedding = self.process_tokens([padded_chunk.tokens], [padded_chunk.multipliers])[0]
                chunk_embeddings.append(embedding)

            cache[line] = chunk_embeddings
            embeddings.extend(chunk_embeddings)

        stacked = torch.stack(embeddings)
        logger.debug("Generated T5 embeddings shape=%s", tuple(stacked.shape))
        return stacked

    def process_tokens(self, batch_tokens: Sequence[Sequence[int]], batch_multipliers: Sequence[Sequence[float]]) -> torch.Tensor:
        if len(batch_tokens) != len(batch_multipliers):
            raise ValueError("batch_tokens and batch_multipliers must have the same length")
        if not batch_tokens:
            return torch.empty(0)

        tokens = torch.tensor(batch_tokens, dtype=torch.long)
        encoded = self.encode_with_transformers(tokens)

        multipliers = torch.tensor(batch_multipliers, dtype=encoded.dtype, device=encoded.device)
        self.emphasis.tokens = batch_tokens
        self.emphasis.multipliers = multipliers
        self.emphasis.z = encoded
        self.emphasis.after_transformers()

        return self.emphasis.z
