"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen3-4B text encoder wrapper for Z Image and Z-Image L2P (GGUF or safetensors) with strict fail-loud load semantics.
Wraps the Qwen3 model used by Z Image (Turbo/Base variants) and L2P for text encoding, preferring vendored HF tokenizers under `apps/backend/huggingface/Tongyi-MAI/**`.
GGUF loads are constructed under `using_codex_operations(weight_format="gguf")` so `torch.nn` layers can load packed `CodexParameter` weights.
Safetensors loads apply strict generic Qwen key-style normalization before native strict model load.
This module follows the “Flux pattern” by providing a small text-processing engine wrapper for consistent interfaces.

Symbols (top-level; keep in sync; no ghosts):
- `ZImageTextEncoder` (class): nn.Module wrapper for Qwen3-4B; supports loading from GGUF/safetensors with strict key validation, tokenization, and embedding extraction
  (contains nested helpers for tokenizer loading, chat templating, debug tracing, and encode/tokenize/masked-encode APIs).
- `ZImageTextProcessingEngine` (class): Thin adapter providing a consistent callable interface (`__call__`, `tokenize`) around `ZImageTextEncoder`.
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.state_dict.keymap_qwen_text_encoder import resolve_qwen_text_encoder_keyspace

from .debug import env_flag, env_int, find_indices, summarize_ints, tensor_stats, truncate_text

logger = get_backend_logger("backend.runtime.zimage.text_encoder")

# Chat template for Qwen3 (reference template)
QWEN3_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class ZImageTextEncoder(nn.Module):
    """Text encoder for Z Image using Qwen3-4B.
    
    Wraps a Qwen3-4B model for text encoding. The model can be:
    - A safetensors checkpoint (qwen_3_4b.safetensors)
    - A GGUF quantized model (Qwen3-4B-Q8_0.gguf)
    - An FP8 quantized model (qwen3_4b_fp8_scaled.safetensors)
    
    The model uses a chat template format for text encoding.
    """
    
    def __init__(
        self,
        qwen_model: nn.Module,
        hidden_size: int = 2560,
        layer_idx: int = -2,
    ):
        """Initialize the text encoder wrapper.
        
        Args:
            qwen_model: The underlying Qwen3-4B model.
            hidden_size: Hidden dimension of the model (2560 for Qwen3-4B).
            layer_idx: Which hidden layer to use (-2 = second-to-last).
        """
        super().__init__()
        self.model = qwen_model
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self._tokenizer = None
        self._tokenizer_path_hint: str | None = None

    def set_tokenizer_path_hint(self, tokenizer_path: str | None) -> None:
        """Set a preferred tokenizer directory for lazy loading."""
        value = str(tokenizer_path).strip() if tokenizer_path is not None else ""
        self._tokenizer_path_hint = value or None
    
    @classmethod
    def from_gguf(cls, gguf_path: str, torch_dtype: torch.dtype = torch.bfloat16) -> "ZImageTextEncoder":
        """Load text encoder from GGUF file using native Qwen3_4B.
        
        Args:
            gguf_path: Path to the GGUF file.
            torch_dtype: Target dtype for the model.
        
        Returns:
            ZImageTextEncoder instance.
        """
        emit_backend_message("Loading Qwen3 text encoder from GGUF", logger=logger.name, path=gguf_path)
        
        # Import native Qwen3 implementation
        from .qwen3 import Qwen3_4B, Qwen3Config, resolve_qwen3_gguf_keyspace
        
        # Load GGUF state dict using our infrastructure
        from apps.backend.runtime.ops.operations import using_codex_operations
        from apps.backend.runtime.checkpoint.io import load_gguf_state_dict
        
        try:
            # Load GGUF file
            emit_backend_message("Loading GGUF state dict", logger=logger.name)
            gguf_state_dict = load_gguf_state_dict(gguf_path)
            emit_backend_message("Loaded GGUF tensors", logger=logger.name, tensors=len(gguf_state_dict))
            
            # Resolve GGUF keys into the native lookup keyspace without renaming tensors
            emit_backend_message("Resolving GGUF keyspace to native lookup view", logger=logger.name)
            state_dict = resolve_qwen3_gguf_keyspace(gguf_state_dict, num_layers=36)
            emit_backend_message("Resolved native lookup keys", logger=logger.name, keys=len(state_dict))
            
            # Use Codex operations context to enable GGUF tensor support
            # This patches torch.nn.Linear to CodexOperationsGGUF.Linear which handles CodexParameter
            with using_codex_operations(weight_format='gguf', manual_cast_enabled=True, device=None, dtype=torch_dtype):
                # Create model with native Qwen3_4B inside the context
                config = Qwen3Config()
                model = Qwen3_4B(config, dtype=torch_dtype)
                
                debug_run = env_flag("CODEX_ZIMAGE_DEBUG_TENC_RUN", False)
                if debug_run:
                    emit_backend_message(
                        "[zimage-debug] tenc.gguf module types",
                        logger=logger.name,
                        embed_tokens=type(model.model.embed_tokens).__name__,
                        q_proj=type(model.model.layers[0].self_attn.q_proj).__name__,
                    )

                    emb_key = "model.embed_tokens.weight"
                    if emb_key in state_dict:
                        emb_tensor = state_dict[emb_key]
                        emit_backend_message(
                            "[zimage-debug] tenc.gguf tensor",
                            logger=logger.name,
                            key=emb_key,
                            tensor_type=type(emb_tensor).__name__,
                            shape=getattr(emb_tensor, "shape", None),
                        )
                        if hasattr(emb_tensor, "real_shape"):
                            emit_backend_message(
                                "[zimage-debug] tenc.gguf tensor real_shape",
                                logger=logger.name,
                                key=emb_key,
                                real_shape=emb_tensor.real_shape,
                            )

                # Load state dict - patched layers will handle GGUF tensors correctly
                try:
                    model.load_sd(state_dict)
                except RuntimeError as e:
                    emit_backend_message(
                        "RuntimeError during load_sd",
                        logger=logger.name,
                        level=logging.ERROR,
                        error=str(e),
                    )
                    emit_backend_message(
                        "Model embedding weight shape",
                        logger=logger.name,
                        level=logging.ERROR,
                        shape=model.model.embed_tokens.weight.shape,
                    )
                    raise
                
                model = model.to(dtype=torch_dtype)
            
            param_count = sum(p.numel() for p in model.parameters())
            emit_backend_message(
                "GGUF text encoder loaded",
                logger=logger.name,
                params=param_count,
                params_b=param_count / 1e9,
            )
            
            return cls(model, hidden_size=2560, layer_idx=-2)
            
        except ImportError as e:
            emit_backend_message(
                "GGUF loader/ops not available",
                logger=logger.name,
                level=logging.ERROR,
                error=str(e),
            )
            raise ValueError(f"GGUF loader/ops not available: {e}")
        except Exception as e:
            emit_backend_message(
                "Failed to load GGUF text encoder",
                logger=logger.name,
                level=logging.ERROR,
                error=str(e),
            )
            raise ValueError(f"Failed to load GGUF text encoder from {gguf_path}: {e}")
    
    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ZImageTextEncoder":
        """Load text encoder from state_dict.
        
        Args:
            state_dict: State dict with model weights.
            torch_dtype: Target dtype for the model.
        
        Returns:
            ZImageTextEncoder instance.
        """
        if not isinstance(state_dict, Mapping):
            raise RuntimeError(
                "Z Image Qwen3-4B state_dict must be a mapping; "
                f"got {type(state_dict).__name__}."
            )
        for raw_key in state_dict.keys():
            if not isinstance(raw_key, str):
                raise RuntimeError(
                    "Z Image Qwen3-4B state_dict keys must be strings. "
                    f"Got {type(raw_key).__name__}."
                )
        emit_backend_message(
            "Loading Qwen3 text encoder from state_dict",
            logger=logger.name,
            keys=len(state_dict),
        )

        try:
            resolved = resolve_qwen_text_encoder_keyspace(
                state_dict,
                allow_lm_head_aux=True,
                allow_visual_aux=True,
                require_backbone_keys=True,
            )
            key_style = resolved.style
            resolved_state_dict = resolved.view
            style_label = key_style.value if hasattr(key_style, "value") else str(key_style)
            emit_backend_message(
                "ZImage Qwen3 keymap style",
                logger=logger.name,
                level=logging.DEBUG,
                style=style_label,
            )

            # Use native Qwen3_4B implementation (compatible with the exported format)
            from .qwen3 import Qwen3_4B, Qwen3Config
            from apps.backend.runtime.ops.operations import using_codex_operations
            
            with using_codex_operations(manual_cast_enabled=True, device=None, dtype=torch_dtype):
                config = Qwen3Config()
                model = Qwen3_4B(config, dtype=torch_dtype)

                # Load weights - native implementation has compatible key format
                model.load_sd(resolved_state_dict)
                
                # Move to target dtype
                model.to(dtype=torch_dtype)
            
            encoder = cls(model, hidden_size=2560, layer_idx=-2)
            emit_backend_message("Safetensors text encoder loaded successfully", logger=logger.name)
            return encoder
            
        except Exception as e:
            emit_backend_message(
                "Failed to load text encoder from state_dict",
                logger=logger.name,
                level=logging.ERROR,
                error=str(e),
            )
            raise ValueError(f"Failed to load text encoder from state_dict: {e}")
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return memory_management.manager.cpu_device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model parameters."""
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32
    
    def load_tokenizer(self, tokenizer_path: Optional[str] = None):
        """Load the Qwen tokenizer.
        
        Args:
            tokenizer_path: Path to tokenizer files. If None, uses default.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            emit_backend_message(
                "transformers library required for Qwen tokenizer",
                logger=logger.name,
                level=logging.ERROR,
            )
            raise

        # Prefer explicit config, then vendored Hugging Face assets, then bundled tokenizer snapshot.
        repo_root = get_repo_root()
        runtime_root = Path(__file__).resolve().parents[1]

        candidates: list[str] = []
        env_override = os.getenv("CODEX_ZIMAGE_TOKENIZER_PATH")
        if env_override:
            candidates.append(env_override)
        if tokenizer_path:
            candidates.insert(0, tokenizer_path)

        hint = getattr(self, "_tokenizer_path_hint", None)
        if hint:
            candidates.append(hint)

        hf_tokenizer_turbo = (
            repo_root / "apps" / "backend" / "huggingface" / "Tongyi-MAI" / "Z-Image-Turbo" / "tokenizer"
        )
        candidates.append(str(hf_tokenizer_turbo))
        hf_tokenizer_base = repo_root / "apps" / "backend" / "huggingface" / "Tongyi-MAI" / "Z-Image" / "tokenizer"
        candidates.append(str(hf_tokenizer_base))

        bundled_tokenizer = runtime_root / "text_processing" / "tokenizers" / "qwen25_tokenizer"
        candidates.append(str(bundled_tokenizer))

        errors: list[str] = []
        for raw in candidates:
            raw = str(raw).strip()
            if not raw:
                continue
            p = Path(os.path.expanduser(raw))
            if not p.is_absolute():
                p = repo_root / p
            try:
                p = p.resolve()
            except Exception:
                p = p.absolute()
            if not p.exists():
                continue
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(str(p), local_files_only=True, use_fast=True)
                emit_backend_message("Qwen tokenizer loaded", logger=logger.name, path=str(p))
                return
            except Exception as exc:  # noqa: BLE001 - try next candidate
                errors.append(f"{p}: {type(exc).__name__}: {exc}")

        detail = "\n".join(errors) if errors else "<no load errors captured>"
        raise RuntimeError(
            "Failed to load a Qwen tokenizer for Z Image. "
            "Set CODEX_ZIMAGE_TOKENIZER_PATH or provide tokenizer_path explicitly. "
            f"Tried: {candidates}\nErrors:\n{detail}"
        )
    
    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
        apply_template: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text with optional chat template.
        
        Args:
            texts: List of text prompts.
            max_length: Maximum token length.
            apply_template: Whether to apply the chat template.
        
        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors.
        """
        if self._tokenizer is None:
            self.load_tokenizer()
        
        debug_text = env_flag("CODEX_ZIMAGE_DEBUG_TENC_TEXT", False)
        debug_tokens = env_flag("CODEX_ZIMAGE_DEBUG_TENC_TOKENS", False)
        debug_decode = env_flag("CODEX_ZIMAGE_DEBUG_TENC_DECODE", False)
        text_max = env_int("CODEX_ZIMAGE_DEBUG_TEXT_MAX", 400)

        # Behavior: if the user already provided a chat template,
        # do not wrap again.
        if apply_template:
            wrapped: list[str] = []
            for raw in texts:
                s = str(raw)
                if s.startswith("<|im_start|>") or s.startswith("<|start_header_id|>"):
                    wrapped.append(s)
                else:
                    # Prefer the tokenizer's built-in chat template when available.
                    # This keeps us aligned with the tokenizer files shipped with Hugging Face model assets.
                    rendered = None
                    if hasattr(self._tokenizer, "apply_chat_template"):
                        try:
                            # Z Image uses enable_thinking=True per diffusers reference
                            rendered = self._tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                                [{"role": "user", "content": s}],
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=True,  # Critical for Z Image
                            )
                        except TypeError:
                            # Fallback if tokenizer doesn't support enable_thinking
                            try:
                                rendered = self._tokenizer.apply_chat_template(
                                    [{"role": "user", "content": s}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                )
                            except Exception:
                                rendered = None
                        except Exception:
                            rendered = None
                    wrapped.append(rendered if isinstance(rendered, str) and rendered else QWEN3_TEMPLATE.format(s))
            texts = wrapped

        if debug_text and texts:
            emit_backend_message(
                "[zimage-debug] tenc.tokenize",
                logger=logger.name,
                apply_template=apply_template,
                text0=truncate_text(texts[0], limit=text_max),
            )
        
        tokens = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if debug_tokens and "input_ids" in tokens:
            try:
                ids0 = tokens["input_ids"][0].tolist()
                pad_id = getattr(self._tokenizer, "pad_token_id", None)
                emit_backend_message(
                    "[zimage-debug] tenc.tokens",
                    logger=logger.name,
                    shape=tuple(tokens["input_ids"].shape),
                    pad_id=str(pad_id),
                    ids=summarize_ints([int(v) for v in ids0], window=12),
                )
                # Common Qwen token ids (observed in bundled tokenizer snapshots): im_start=151644, im_end=151645.
                # We log indices to help compare template slicing logic.
                for name, tok in (("im_start", 151644), ("im_end", 151645)):
                    idxs = find_indices([int(v) for v in ids0], tok, limit=8)
                    if idxs:
                        emit_backend_message(
                            "[zimage-debug] tenc.tokens marker",
                            logger=logger.name,
                            name=name,
                            token=tok,
                            idx=idxs,
                        )
                if isinstance(pad_id, int):
                    pad_idxs = find_indices([int(v) for v in ids0], int(pad_id), limit=8)
                    if pad_idxs:
                        emit_backend_message(
                            "[zimage-debug] tenc.tokens pad",
                            logger=logger.name,
                            idx=pad_idxs,
                        )
            except Exception as exc:
                emit_backend_message(
                    "[zimage-debug] failed to summarize token ids",
                    logger=logger.name,
                    level=logging.ERROR,
                    error=str(exc),
                )

        if debug_decode and "input_ids" in tokens:
            try:
                ids0 = tokens["input_ids"][0].tolist()
                decoded = self._tokenizer.decode(ids0)
                emit_backend_message(
                    "[zimage-debug] tenc.decode",
                    logger=logger.name,
                    text0=truncate_text(decoded, limit=text_max),
                )
            except Exception as exc:
                emit_backend_message(
                    "[zimage-debug] failed to decode token ids",
                    logger=logger.name,
                    level=logging.ERROR,
                    error=str(exc),
                )
        
        return {
            "input_ids": tokens["input_ids"].to(self.device),
            "attention_mask": tokens["attention_mask"].to(self.device),
        }
    
    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        max_length: int = 512,
        apply_template: bool = True,
    ) -> torch.Tensor:
        """Encode texts to embeddings.
        
        Args:
            texts: List of text prompts.
            max_length: Maximum token length.
            apply_template: Whether to apply the chat template.
        
        Returns:
            Text embeddings [B, L, hidden_size].
        """
        tokens = self.tokenize(texts, max_length, apply_template)

        debug_run = env_flag("CODEX_ZIMAGE_DEBUG_TENC_RUN", False)
        if debug_run:
            tensor_stats(logger.name, "tenc.input_ids", tokens.get("input_ids"))
            tensor_stats(logger.name, "tenc.attention_mask", tokens.get("attention_mask"))
        
        # Check if this is our native Qwen3_4B model or a HuggingFace model
        # Native Qwen3_4B returns (hidden_states, intermediate) tuple
        # HuggingFace models return an object with .hidden_states attribute
        is_native_model = hasattr(self.model, 'load_sd')  # Native Qwen3_4B has load_sd method

        layer_idx = self.layer_idx
        layer_override = os.getenv("CODEX_ZIMAGE_QWEN_LAYER_IDX")
        if layer_override is not None:
            try:
                layer_idx = int(str(layer_override).strip())
            except Exception:
                layer_idx = self.layer_idx
        final_norm = env_flag("CODEX_ZIMAGE_QWEN_FINAL_NORM_INTERMEDIATE", True)
        if debug_run:
            emit_backend_message(
                "[zimage-debug] tenc.encode",
                logger=logger.name,
                native=bool(is_native_model),
                layer_idx=str(layer_idx),
                final_norm_intermediate=bool(final_norm),
            )
        
        if is_native_model:
            # Native Qwen3_4B: use intermediate_output to get second-to-last layer
            hidden_states, intermediate = self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                intermediate_output=layer_idx,  # -2 = second-to-last layer (default)
                final_layer_norm_intermediate=bool(final_norm),
            )
            # intermediate contains the second-to-last layer output (with final norm applied)
            hidden = intermediate if intermediate is not None else hidden_states
        else:
            # HuggingFace model: use output_hidden_states
            outputs = self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
            )
            # Use second-to-last layer (like CLIP)
            hidden = outputs.hidden_states[layer_idx]

        if debug_run:
            tensor_stats(logger.name, "tenc.hidden", hidden)
        
        return hidden.to(self.dtype)

    @torch.no_grad()
    def encode_masked(
        self,
        texts: List[str],
        max_length: int = 512,
        apply_template: bool = True,
    ) -> list[torch.Tensor]:
        """Encode texts and return one non-padding embedding tensor per prompt.

        L2P consumes a list of variable-length Qwen embeddings after applying the tokenizer
        attention mask. Returning padded `[B, S, C]` tensors here would change the native
        L2P conditioning contract and corrupt RoPE positions.
        """
        tokens = self.tokenize(texts, max_length, apply_template)
        hidden = self.encode(texts, max_length=max_length, apply_template=apply_template)
        attention_mask = tokens["attention_mask"].to(device=hidden.device, dtype=torch.bool)
        if hidden.ndim != 3:
            raise RuntimeError(f"Qwen masked encode expected hidden [B,S,C], got {tuple(hidden.shape)}.")
        if attention_mask.shape[:2] != hidden.shape[:2]:
            raise RuntimeError(
                "Qwen masked encode attention-mask shape mismatch: "
                f"mask={tuple(attention_mask.shape)} hidden={tuple(hidden.shape)}"
            )
        result: list[torch.Tensor] = []
        for index in range(int(hidden.shape[0])):
            item = hidden[index][attention_mask[index]]
            if item.ndim != 2 or int(item.shape[-1]) != int(hidden.shape[-1]) or int(item.shape[0]) <= 0:
                raise RuntimeError(
                    "Qwen masked encode produced an invalid per-prompt tensor: "
                    f"index={index} shape={tuple(item.shape)}"
                )
            result.append(item.to(dtype=self.dtype))
        return result
    
    def forward(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> torch.Tensor:
        """Forward pass."""
        return self.encode(texts, max_length)


class ZImageTextProcessingEngine:
    """Text processing engine for Z Image (matches Flux pattern).
    
    This class wraps the text encoder and provides a consistent interface
    for text encoding across different model families.
    """
    
    def __init__(
        self,
        text_encoder: ZImageTextEncoder,
        emphasis_name: str = "Original",
        max_length: int = 512,
    ):
        """Initialize the text processing engine.
        
        Args:
            text_encoder: The Z Image text encoder.
            emphasis_name: Name of emphasis style (unused).
            max_length: Maximum token length.
        """
        self.text_encoder = text_encoder
        self.emphasis_name = emphasis_name
        self.max_length = max_length
    
    def __call__(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings."""
        return self.text_encoder.encode(texts, self.max_length)

    def encode_masked(self, texts: List[str]) -> list[torch.Tensor]:
        """Encode texts as a masked list for L2P conditioning."""
        return self.text_encoder.encode_masked(texts, self.max_length)
    
    def tokenize(self, texts: List[str]) -> List[List[int]]:
        """Tokenize without encoding."""
        tokens = self.text_encoder.tokenize(texts, self.max_length)
        return tokens["input_ids"].tolist()
