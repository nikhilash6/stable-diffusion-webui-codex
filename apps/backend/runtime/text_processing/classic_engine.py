"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Classic (legacy-style) text processing engine for prompt parsing and conditioning assembly.
Implements emphasis parsing, textual inversion embedding injection, and token chunking for CLIP-based text encoders, integrating with Codex runtime memory management for device placement and safe execution.
Matches WebUI `clip_skip` semantics: `clip_skip=1` uses the post-final-layer-norm embedding (`last_hidden_state`).

Symbols (top-level; keep in sync; no ghosts):
- `PromptChunkFix` (namedtuple): Single embedding “fix” applied at a specific token offset (used by textual inversion).
- `PromptChunk` (class): Container for tokens, multipliers, and embedding fixes produced by parsing/chunking.
- `CLIPEmbeddingForTextualInversion` (class): Wrapper module that intercepts CLIP embedding lookups and injects textual inversion vectors
  (contains nested logic for batching fixes and restoring state).
- `clear_last_extra_generation_params` (function): Clears thread-local extra generation params at request boundaries.
- `snapshot_last_extra_generation_params` (function): Returns a copy of thread-local extra generation params for response metadata.
- `ClassicTextProcessingEngine` (class): Main text processing engine; parses prompts (including negative/emphasis), manages embeddings DB,
  and produces encoder inputs/embeddings for downstream diffusion runtimes (contains nested helpers for chunking and extra params tracking).
"""

import logging
import math
import torch
import os
import threading
from apps.backend.runtime.logging import get_backend_logger

from collections import namedtuple
from collections.abc import Iterator, MutableMapping
from . import parsing, emphasis
from .textual_inversion import EmbeddingDatabase
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.infra.config.args import dynamic_args


PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])
logger = get_backend_logger("backend.text_processing.classic")

_last_extra_generation_params_local = threading.local()


def _current_extra_generation_params() -> dict[str, str]:
    params = getattr(_last_extra_generation_params_local, "params", None)
    if params is None:
        params = {}
        _last_extra_generation_params_local.params = params
    return params


class _ThreadLocalExtraGenerationParams(MutableMapping[str, str]):
    """Dict-like proxy backed by thread-local storage."""

    def _state(self) -> dict[str, str]:
        return _current_extra_generation_params()

    def __getitem__(self, key: str) -> str:
        return self._state()[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._state()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._state()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._state())

    def __len__(self) -> int:
        return len(self._state())


last_extra_generation_params: MutableMapping[str, str] = _ThreadLocalExtraGenerationParams()


def clear_last_extra_generation_params() -> None:
    _current_extra_generation_params().clear()


def snapshot_last_extra_generation_params() -> dict[str, str]:
    return dict(_current_extra_generation_params())


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


class CLIPEmbeddingForTextualInversion(torch.nn.Module):
    def __init__(self, wrapped, embeddings, textual_inversion_key='clip_l'):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings
        self.textual_inversion_key = textual_inversion_key
        self.weight = self.wrapped.weight

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = embedding.vec[self.textual_inversion_key] if isinstance(embedding.vec, dict) else embedding.vec
                emb = emb.to(inputs_embeds)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]]).to(dtype=inputs_embeds.dtype)

            vecs.append(tensor)

        return torch.stack(vecs)


class ClassicTextProcessingEngine:
    def __init__(
            self, text_encoder, tokenizer, chunk_length=75,
            embedding_dir=None, embedding_key='clip_l', embedding_expected_shape=768, emphasis_name="Original",
            text_projection=False, minimal_clip_skip=1, clip_skip=1, return_pooled=False, final_layer_norm=True
    ):
        super().__init__()

        self.embeddings = EmbeddingDatabase(tokenizer, embedding_expected_shape)

        if isinstance(embedding_dir, str):
            self.embeddings.add_embedding_dir(embedding_dir)
            self.embeddings.load_textual_inversion_embeddings()

        self.embedding_key = embedding_key

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self._current_device: torch.device | None = None
        self._current_dtype: torch.dtype | None = None

        self.emphasis = emphasis.get_current_option(dynamic_args.get('emphasis_name', 'original'))()
        self.text_projection = text_projection
        self.minimal_clip_skip = minimal_clip_skip
        self.clip_skip = clip_skip
        self.return_pooled = return_pooled
        self.final_layer_norm = final_layer_norm

        self.chunk_length = chunk_length

        self.id_start = self.tokenizer.bos_token_id
        self.id_end = self.tokenizer.eos_token_id
        self.id_pad = self.tokenizer.pad_token_id

        model_embeddings = text_encoder.transformer.text_model.embeddings
        model_embeddings.token_embedding = CLIPEmbeddingForTextualInversion(model_embeddings.token_embedding, self.embeddings, textual_inversion_key=embedding_key)

        vocab = self.tokenizer.get_vocab()

        self.comma_token = vocab.get(',</w>', None)

        self.token_mults = {}

        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

    def empty_chunk(self):
        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count):
        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize(self, texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False, verbose=False)["input_ids"]

        return tokenized

    def _apply_precision(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._current_device == device and self._current_dtype == dtype:
            return
        self.text_encoder.to(device=device, dtype=dtype)
        self._current_device = device
        self._current_dtype = dtype

    def encode_with_transformers(self, tokens):
        """Encode token ids with a CLIP text transformer.

        Robust to HF CLIP variants that do not register `position_ids` on
        the embeddings module. Generates `attention_mask` and `position_ids`
        on-the-fly and only passes them if the transformer's forward
        signature supports the respective parameters.
        """
        import inspect

        target_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)

        while True:
            desired_dtype = memory_management.manager.dtype_for_role(DeviceRole.TEXT_ENCODER)
            self._apply_precision(target_device, desired_dtype)

            # Ensure embedding weights use a stable compute dtype to avoid overflow
            # regardless of TE compute dtype. We do NOT rely on a `position_ids`
            # attribute existing on embeddings (many CLIP variants do not expose it).
            embeddings = self.text_encoder.transformer.text_model.embeddings
            if hasattr(embeddings, "position_embedding"):
                embeddings.position_embedding = embeddings.position_embedding.to(dtype=desired_dtype)
            if hasattr(embeddings, "token_embedding"):
                embeddings.token_embedding = embeddings.token_embedding.to(dtype=desired_dtype)

            tokens_device = tokens.to(target_device)

            # Build masks/ids once, on device. Only forward what the model supports.
            batch, seqlen = tokens_device.shape[0], tokens_device.shape[1]
            attention_mask = (tokens_device != self.id_pad).to(dtype=torch.long, device=target_device)
            position_ids = torch.arange(seqlen, device=target_device).unsqueeze(0).expand(batch, -1)

            # Prefer the high-level wrapper to reduce coupling on internal structure
            fwd = self.text_encoder.forward
            try:
                sig = inspect.signature(fwd)
                allowed = set(sig.parameters.keys())
            except (ValueError, TypeError):  # builtins/torchscript edge cases
                allowed = {"input_ids", "attention_mask", "position_ids", "output_hidden_states"}

            kwargs = {"output_hidden_states": True}
            if "attention_mask" in allowed:
                kwargs["attention_mask"] = attention_mask
            if "position_ids" in allowed:
                kwargs["position_ids"] = position_ids

            # Route via wrapper to let it normalise return types across variants
            outputs = self.text_encoder(tokens_device, **kwargs)

            effective_clip_skip = max(int(self.clip_skip), int(self.minimal_clip_skip))
            hidden_states = getattr(outputs, "hidden_states", None)

            # Match WebUI semantics: clip_skip=1 means "no skip" and must use the
            # *post* final-layer-norm embedding (HF: outputs.last_hidden_state).
            if effective_clip_skip <= 1:
                z = getattr(outputs, "last_hidden_state", None)
                if not isinstance(z, torch.Tensor):
                    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
                        raise RuntimeError("CLIP output is missing both last_hidden_state and hidden_states.")
                    z = hidden_states[-1]
                    final_ln = getattr(getattr(self.text_encoder.transformer, "text_model", None), "final_layer_norm", None)
                    if callable(final_ln):
                        z = final_ln(z)
                    else:
                        raise RuntimeError(
                            "CLIP output is missing last_hidden_state and the text model lacks final_layer_norm; "
                            "cannot produce a normalized clip_skip=1 embedding."
                        )
            else:
                if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
                    raise RuntimeError("CLIP output is missing hidden_states (required for clip_skip > 1).")
                if effective_clip_skip > len(hidden_states):
                    raise ValueError(
                        f"clip_skip={effective_clip_skip} exceeds available hidden_states ({len(hidden_states)})."
                    )
                z = hidden_states[-effective_clip_skip]
                if self.final_layer_norm:
                    # final_layer_norm lives on the inner text model
                    z = self.text_encoder.transformer.text_model.final_layer_norm(z)

            pooled_output = outputs.pooler_output if self.return_pooled else None
            if pooled_output is not None and self.text_projection and self.embedding_key != 'clip_l':
                pooled_output = self.text_encoder.transformer.text_projection(pooled_output)

            if pooled_output is not None:
                z.pooled = pooled_output

            has_nan = torch.isnan(z).any() or (pooled_output is not None and torch.isnan(pooled_output).any())
            if has_nan:
                logger.warning(
                    "CLIP encoding produced NaNs on %s using dtype %s; requesting precision fallback.",
                    target_device,
                    str(desired_dtype),
                )
                next_dtype = memory_management.manager.report_precision_failure(
                    DeviceRole.TEXT_ENCODER,
                    location="clip.encode",
                    reason="NaN detected in CLIP output",
                )
                if next_dtype is None:
                    hint = memory_management.manager.precision_hint(DeviceRole.TEXT_ENCODER)
                    raise RuntimeError(
                        f"Text encoder produced NaNs on {target_device} with dtype {desired_dtype}. {hint}"
                    )
                self._apply_precision(target_device, next_dtype)
                memory_management.manager.soft_empty_cache(force=True)
                continue

            return z

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                comma_padding_backtrack = 20

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                elif comma_padding_backtrack != 0 and len(chunk.tokens) == self.chunk_length and last_comma != -1 and len(chunk.tokens) - last_comma <= comma_padding_backtrack:
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                embedding, embedding_length_in_tokens = self.embeddings.find_embedding_at_position(tokens, position)
                if embedding is None:
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1
                    continue

                emb_len = int(embedding.vectors)
                if len(chunk.tokens) + emb_len > self.chunk_length:
                    next_chunk()

                chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))

                chunk.tokens += [0] * emb_len
                chunk.multipliers += [weight] * emb_len
                position += embedding_length_in_tokens

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)

        return chunks, token_count

    def process_texts(self, texts):
        token_count = 0

        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)

                cache[line] = chunks

            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def __call__(self, texts):
        self.emphasis = emphasis.get_current_option(dynamic_args.get('emphasis_name', 'original'))()

        batch_chunks, token_count = self.process_texts(texts)

        used_embeddings = {}
        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            self.embeddings.fixes = [x.fixes for x in batch_chunk]

            for fixes in self.embeddings.fixes:
                for _position, embedding in fixes:
                    used_embeddings[embedding.name] = embedding

            z = self.process_tokens(tokens, multipliers)
            zs.append(z)

        if used_embeddings:
            names = []

            for name, embedding in used_embeddings.items():
                logger.info('[Textual Inversion] Used Embedding [%s] in CLIP of [%s]', name, self.embedding_key)
                names.append(name.replace(":", "").replace(",", ""))

            if "TI" in last_extra_generation_params:
                last_extra_generation_params["TI"] += ", " + ", ".join(names)
            else:
                last_extra_generation_params["TI"] = ", ".join(names)

        if any(x for x in texts if "(" in x or "[" in x) and self.emphasis.name != "Original":
            last_extra_generation_params["Emphasis"] = self.emphasis.name

        if self.return_pooled:
            return torch.hstack(zs), zs[0].pooled
        else:
            return torch.hstack(zs)

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        tokens = torch.asarray(remade_batch_tokens)

        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index + 1:tokens.shape[1]] = self.id_pad

        z = self.encode_with_transformers(tokens)

        pooled = getattr(z, 'pooled', None)

        self.emphasis.tokens = remade_batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(z)
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        if pooled is not None:
            z.pooled = pooled

        return z
