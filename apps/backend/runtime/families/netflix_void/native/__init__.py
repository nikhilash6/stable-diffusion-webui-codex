"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native Netflix VOID runtime package marker plus explicit text-encoder export.
Owns the repo-native component loaders/builders used by the Netflix VOID family runtime.

Symbols (top-level; keep in sync; no ghosts):
- `NetflixVoidTextEncoderRuntime` (dataclass): Loaded T5 wrapper + tokenizer pair for the native VOID runtime.
- `load_netflix_void_text_encoder_runtime` (function): Load the family-owned T5 encoder/tokenizer runtime.
"""

from .text_encoder import NetflixVoidTextEncoderRuntime, load_netflix_void_text_encoder_runtime

__all__ = ["NetflixVoidTextEncoderRuntime", "load_netflix_void_text_encoder_runtime"]
