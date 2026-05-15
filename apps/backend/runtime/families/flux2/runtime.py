"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Truthful FLUX.2 Klein runtime helpers for the backend engine seam.
Bridges the shared Codex sampler onto Diffusers `Flux2Transformer2DModel` + `AutoencoderKLFlux2` by exposing a
sampler-facing core adapter, special 2x2 latent patchify/unpatchify + batch-norm normalization helpers, and the Qwen3
prompt-embedding contract used by FLUX.2 Klein 4B/base-4B (`enable_thinking=False`, layers 9/18/27`). The adapter also
owns FLUX.2 image-conditioned img2img semantics by appending external VAE latents as `image_latents` tokens instead of
pretending FLUX.2 uses classic init-latent denoise strength.

Symbols (top-level; keep in sync; no ghosts):
- `Flux2CoreConfig` (dataclass): Minimal sampler-facing config surface (`context_dim`) for the FLUX.2 adapter.
- `_require_even_spatial_dims` (function): Validate 2x2 latent patchify preconditions.
- `_require_patchified_channels` (function): Validate patchified latent channel counts before unpatchify.
- `_require_flux2_bn_state` (function): Resolve AutoencoderKLFlux2 batch-norm stats/config for latent normalization.
- `patchify_flux2_latents` (function): Convert 32-channel external latents into patchified 128-channel FLUX.2 latents.
- `unpatchify_flux2_latents` (function): Convert patchified 128-channel FLUX.2 latents into external 32-channel latents.
- `normalize_flux2_patchified_latents` (function): Apply AutoencoderKLFlux2 batch-norm normalization to patchified latents.
- `denormalize_flux2_patchified_latents` (function): Reverse AutoencoderKLFlux2 batch-norm normalization for patchified latents.
- `encode_flux2_external_latents` (function): Encode pixels through `VAE.encode(...)` and export normalized external 32-channel latents, forwarding optional shared `encode_seed`.
- `decode_flux2_external_latents` (function): Decode normalized external 32-channel latents through `VAE.decode(...)`.
- `_pack_flux2_latents` (function): Convert patchified BCHW latents into transformer token format `(B, L, C)`.
- `_unpack_flux2_latents` (function): Convert transformer token format `(B, L, C)` back into patchified BCHW latents.
- `_prepare_text_ids` (function): Build FLUX.2 4D text position ids `(T,H,W,L)` for prompt embeddings.
- `_prepare_latent_ids` (function): Build FLUX.2 4D latent position ids `(T,H,W,L)` for packed image latents.
- `Flux2TextProcessingEngine` (class): Qwen3 embedding engine for FLUX.2 (stacked hidden states -> 7680-dim prompt tokens).
- `Flux2CoreAdapter` (class): `nn.Module` adapter exposing the sampler-compatible FLUX.2 forward contract on external latents, including optional `image_latents` token concatenation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


_FLUX2_EXTERNAL_LATENT_CHANNELS = 32
_FLUX2_PATCH_FACTOR = 2
_FLUX2_PATCHIFIED_LATENT_CHANNELS = _FLUX2_EXTERNAL_LATENT_CHANNELS * (_FLUX2_PATCH_FACTOR ** 2)
_FLUX2_QWEN_HIDDEN_STATES_LAYERS: tuple[int, int, int] = (9, 18, 27)


@dataclass(frozen=True, slots=True)
class Flux2CoreConfig:
    context_dim: int = 7680


def _require_even_spatial_dims(latents: torch.Tensor, *, label: str) -> tuple[int, int, int, int]:
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor; got {type(latents).__name__}.")
    if latents.ndim != 4:
        raise ValueError(f"{label} must be BCHW; got shape={tuple(latents.shape)}.")
    batch, channels, height, width = map(int, latents.shape)
    if height <= 0 or width <= 0:
        raise ValueError(f"{label} spatial dims must be > 0; got shape={tuple(latents.shape)}.")
    if (height % _FLUX2_PATCH_FACTOR) != 0 or (width % _FLUX2_PATCH_FACTOR) != 0:
        raise ValueError(
            f"{label} height/width must be multiples of {_FLUX2_PATCH_FACTOR}; got shape={tuple(latents.shape)}."
        )
    return batch, channels, height, width


def _require_patchified_channels(latents: torch.Tensor, *, label: str) -> tuple[int, int, int, int]:
    batch, channels, height, width = _require_even_spatial_dims(latents, label=label)
    if channels != _FLUX2_PATCHIFIED_LATENT_CHANNELS:
        raise ValueError(
            f"{label} must have {_FLUX2_PATCHIFIED_LATENT_CHANNELS} patchified channels; "
            f"got shape={tuple(latents.shape)}."
        )
    return batch, channels, height, width


def _require_flux2_bn_state(vae_model: object, *, channels: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    bn = getattr(vae_model, "bn", None)
    if bn is None:
        raise RuntimeError("FLUX.2 VAE contract violation: AutoencoderKLFlux2 is missing batch-norm module `bn`.")
    running_mean = getattr(bn, "running_mean", None)
    running_var = getattr(bn, "running_var", None)
    if not isinstance(running_mean, torch.Tensor) or not isinstance(running_var, torch.Tensor):
        raise RuntimeError("FLUX.2 VAE contract violation: `vae.bn` is missing running_mean/running_var tensors.")
    if int(running_mean.numel()) != channels or int(running_var.numel()) != channels:
        raise RuntimeError(
            "FLUX.2 VAE contract violation: batch-norm channel count mismatch. "
            f"running_mean={tuple(running_mean.shape)} running_var={tuple(running_var.shape)} expected_channels={channels}."
        )

    eps = getattr(getattr(vae_model, "config", None), "batch_norm_eps", None)
    if eps is None:
        eps = getattr(bn, "eps", None)
    try:
        eps_value = float(eps)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("FLUX.2 VAE contract violation: missing numeric `batch_norm_eps`.") from exc
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise RuntimeError(f"FLUX.2 VAE contract violation: invalid batch_norm_eps={eps_value!r}.")

    mean = running_mean.view(1, channels, 1, 1).to(device=device, dtype=dtype)
    std = torch.sqrt(running_var.view(1, channels, 1, 1).to(device=device, dtype=dtype) + eps_value)
    return mean, std


def patchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = _require_even_spatial_dims(latents, label="FLUX.2 latents")
    if channels != _FLUX2_EXTERNAL_LATENT_CHANNELS:
        raise ValueError(
            f"FLUX.2 external latents must have {_FLUX2_EXTERNAL_LATENT_CHANNELS} channels; "
            f"got shape={tuple(latents.shape)}."
        )
    latents = latents.view(
        batch,
        channels,
        height // _FLUX2_PATCH_FACTOR,
        _FLUX2_PATCH_FACTOR,
        width // _FLUX2_PATCH_FACTOR,
        _FLUX2_PATCH_FACTOR,
    )
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(
        batch,
        _FLUX2_PATCHIFIED_LATENT_CHANNELS,
        height // _FLUX2_PATCH_FACTOR,
        width // _FLUX2_PATCH_FACTOR,
    )


def unpatchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    batch, _channels, height, width = _require_patchified_channels(latents, label="FLUX.2 patchified latents")
    latents = latents.reshape(
        batch,
        _FLUX2_EXTERNAL_LATENT_CHANNELS,
        _FLUX2_PATCH_FACTOR,
        _FLUX2_PATCH_FACTOR,
        height,
        width,
    )
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(
        batch,
        _FLUX2_EXTERNAL_LATENT_CHANNELS,
        height * _FLUX2_PATCH_FACTOR,
        width * _FLUX2_PATCH_FACTOR,
    )


def normalize_flux2_patchified_latents(latents: torch.Tensor, *, vae_model: object) -> torch.Tensor:
    _require_patchified_channels(latents, label="FLUX.2 patchified latents")
    mean, std = _require_flux2_bn_state(
        vae_model,
        channels=int(latents.shape[1]),
        dtype=latents.dtype,
        device=latents.device,
    )
    return (latents - mean) / std


def denormalize_flux2_patchified_latents(latents: torch.Tensor, *, vae_model: object) -> torch.Tensor:
    _require_patchified_channels(latents, label="FLUX.2 normalized patchified latents")
    mean, std = _require_flux2_bn_state(
        vae_model,
        channels=int(latents.shape[1]),
        dtype=latents.dtype,
        device=latents.device,
    )
    return (latents * std) + mean


def encode_flux2_external_latents(
    vae,
    pixel_samples_bhwc: torch.Tensor,
    *,
    encode_seed: int | None = None,
) -> torch.Tensor:
    if not hasattr(vae, "encode"):
        raise TypeError("FLUX.2 encode helper requires a VAE wrapper exposing `encode(...)`.")
    raw_latents = vae.encode(pixel_samples_bhwc, encode_seed=encode_seed)
    patchified = patchify_flux2_latents(raw_latents)
    normalized = normalize_flux2_patchified_latents(patchified, vae_model=vae.first_stage_model)
    return unpatchify_flux2_latents(normalized)


def decode_flux2_external_latents(vae, external_latents: torch.Tensor) -> torch.Tensor:
    if not hasattr(vae, "decode"):
        raise TypeError("FLUX.2 decode helper requires a VAE wrapper exposing `decode(...)`.")
    patchified = patchify_flux2_latents(external_latents)
    denormalized = denormalize_flux2_patchified_latents(patchified, vae_model=vae.first_stage_model)
    raw_latents = unpatchify_flux2_latents(denormalized)
    return vae.decode(raw_latents)


def _pack_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(f"FLUX.2 packed latent input must be BCHW; got shape={tuple(latents.shape)}.")
    batch, channels, height, width = map(int, latents.shape)
    return latents.reshape(batch, channels, height * width).permute(0, 2, 1)


def _unpack_flux2_latents(tokens: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise ValueError(f"FLUX.2 transformer output must be BLC; got shape={getattr(tokens, 'shape', None)}.")
    batch, seq_len, channels = map(int, tokens.shape)
    expected_seq_len = int(height) * int(width)
    if seq_len != expected_seq_len:
        raise RuntimeError(
            "FLUX.2 transformer output token count mismatch. "
            f"got_seq_len={seq_len} expected_seq_len={expected_seq_len} height={height} width={width}."
        )
    return tokens.permute(0, 2, 1).reshape(batch, channels, int(height), int(width))


def _prepare_text_ids(*, batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    if batch <= 0 or seq_len <= 0:
        raise ValueError(f"Invalid FLUX.2 text id dimensions: batch={batch} seq_len={seq_len}.")
    ids = torch.zeros((seq_len, 4), device=device, dtype=torch.long)
    ids[:, 3] = torch.arange(seq_len, device=device, dtype=torch.long)
    return ids.unsqueeze(0).expand(batch, -1, -1)


def _prepare_latent_ids(*, batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    if batch <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"Invalid FLUX.2 latent id dimensions: batch={batch} height={height} width={width}.")
    h_ids = torch.arange(height, device=device, dtype=torch.long).view(height, 1).expand(height, width)
    w_ids = torch.arange(width, device=device, dtype=torch.long).view(1, width).expand(height, width)
    ids = torch.zeros((height * width, 4), device=device, dtype=torch.long)
    ids[:, 1] = h_ids.reshape(-1)
    ids[:, 2] = w_ids.reshape(-1)
    return ids.unsqueeze(0).expand(batch, -1, -1)


class Flux2TextProcessingEngine:
    """Qwen3 prompt embedding engine for FLUX.2 Klein 4B/base-4B."""

    def __init__(
        self,
        *,
        text_encoder: object,
        tokenizer: object | None,
        max_length: int = 512,
        hidden_states_layers: Sequence[int] = _FLUX2_QWEN_HIDDEN_STATES_LAYERS,
    ) -> None:
        self.text_encoder = text_encoder
        self._tokenizer = tokenizer
        self.max_length = int(max_length)
        self.hidden_states_layers = tuple(int(layer) for layer in hidden_states_layers)
        if self.max_length <= 0:
            raise ValueError(f"FLUX.2 max_length must be > 0; got {self.max_length}.")
        if not self.hidden_states_layers:
            raise ValueError("FLUX.2 hidden_states_layers must not be empty.")

    def _resolve_tokenizer(self):
        tokenizer = self._tokenizer
        if tokenizer is not None:
            return tokenizer
        loader = getattr(self.text_encoder, "load_tokenizer", None)
        if callable(loader):
            loader()
            tokenizer = getattr(self.text_encoder, "_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "FLUX.2 text tokenizer is unavailable. Expected loader-provided tokenizer component or "
                "a text encoder wrapper that can lazy-load one."
            )
        self._tokenizer = tokenizer
        return tokenizer

    def _render_chat_prompts(self, texts: Sequence[str]) -> list[str]:
        tokenizer = self._resolve_tokenizer()
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("FLUX.2 tokenizer must expose apply_chat_template(...).")

        rendered: list[str] = []
        for raw in texts:
            messages = [{"role": "user", "content": str(raw or "")}]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "FLUX.2 tokenizer must support apply_chat_template(..., enable_thinking=False)."
                ) from exc
            if not isinstance(prompt, str):
                raise RuntimeError(
                    "FLUX.2 tokenizer apply_chat_template(...) returned non-string prompt: "
                    f"{type(prompt).__name__}."
                )
            rendered.append(prompt)
        return rendered

    def _tokenize_batch(
        self,
        texts: Sequence[str],
        *,
        padding: str | bool,
        truncation: bool,
        max_length: int | None,
    ) -> dict[str, torch.Tensor]:
        tokenizer = self._resolve_tokenizer()
        rendered = self._render_chat_prompts(texts)
        tokens = tokenizer(
            rendered,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        input_ids = tokens.get("input_ids")
        attention_mask = tokens.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            raise RuntimeError("FLUX.2 tokenizer must return tensor `input_ids` and `attention_mask`.")
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise RuntimeError(
                "FLUX.2 tokenizer returned invalid tensor ranks: "
                f"input_ids.ndim={input_ids.ndim} attention_mask.ndim={attention_mask.ndim}."
            )
        if tuple(input_ids.shape) != tuple(attention_mask.shape):
            raise RuntimeError(
                "FLUX.2 tokenizer returned mismatched input_ids/attention_mask shapes: "
                f"input_ids={tuple(input_ids.shape)} attention_mask={tuple(attention_mask.shape)}."
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _select_hidden_states(self, *, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        selected: list[torch.Tensor] = []
        total_layers = len(hidden_states)
        for raw_index in self.hidden_states_layers:
            idx = int(raw_index)
            if idx < 0:
                idx = total_layers + idx
            if idx < 0 or idx >= total_layers:
                raise RuntimeError(
                    "FLUX.2 hidden-state layer selection is out of range. "
                    f"index={raw_index} total_layers={total_layers}."
                )
            layer_tensor = hidden_states[idx]
            if not isinstance(layer_tensor, torch.Tensor) or layer_tensor.ndim != 3:
                raise RuntimeError(
                    "FLUX.2 text encoder returned invalid hidden-state tensor for selected layer "
                    f"{raw_index}: shape={getattr(layer_tensor, 'shape', None)}."
                )
            selected.append(layer_tensor)
        return torch.stack(selected, dim=1)

    def _encode_native_qwen(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        qwen_wrapper = getattr(self.text_encoder, "model", None)
        backbone = getattr(qwen_wrapper, "model", None)
        if backbone is None:
            raise RuntimeError("FLUX.2 native Qwen wrapper is missing `.model` backbone for hidden-state extraction.")
        output = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = None
        if isinstance(output, tuple) and len(output) >= 2:
            hidden_states = output[1]
        elif hasattr(output, "hidden_states"):
            hidden_states = getattr(output, "hidden_states")
        if not isinstance(hidden_states, (tuple, list)):
            raise RuntimeError("FLUX.2 native Qwen backbone did not return hidden_states.")
        return self._select_hidden_states(hidden_states=hidden_states)

    def _encode_hf_qwen(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = getattr(output, "hidden_states", None)
        if not isinstance(hidden_states, (tuple, list)):
            raise RuntimeError("FLUX.2 Hugging Face Qwen model did not return hidden_states.")
        return self._select_hidden_states(hidden_states=hidden_states)

    @staticmethod
    def _model_device(model: object) -> torch.device:
        try:
            return next(model.parameters()).device  # type: ignore[call-arg]
        except StopIteration:
            return torch.device("cpu")
        except Exception:
            return torch.device("cpu")

    @staticmethod
    def _model_dtype(model: object) -> torch.dtype:
        try:
            return next(model.parameters()).dtype  # type: ignore[call-arg]
        except StopIteration:
            return torch.float32
        except Exception:
            return torch.float32

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self._tokenize_batch(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        is_native_wrapper = hasattr(self.text_encoder, "load_tokenizer") and hasattr(self.text_encoder, "model")
        model_ref = getattr(self.text_encoder, "model", None) if is_native_wrapper else self.text_encoder
        if model_ref is None:
            raise RuntimeError("FLUX.2 text encoder is missing the underlying Qwen model reference.")

        device = self._model_device(model_ref)
        dtype = self._model_dtype(model_ref)
        input_ids = tokens["input_ids"].to(device=device)
        attention_mask = tokens["attention_mask"].to(device=device)

        if is_native_wrapper:
            stacked = self._encode_native_qwen(input_ids=input_ids, attention_mask=attention_mask)
        else:
            stacked = self._encode_hf_qwen(input_ids=input_ids, attention_mask=attention_mask)

        stacked = stacked.to(device=device, dtype=dtype)
        batch, num_layers, seq_len, hidden_dim = map(int, stacked.shape)
        return stacked.permute(0, 2, 1, 3).reshape(batch, seq_len, num_layers * hidden_dim)

    def __call__(self, texts: Sequence[str]) -> torch.Tensor:
        return self.encode(texts)

    def prompt_lengths(self, prompt: str) -> tuple[int, int]:
        tokens = self._tokenize_batch(
            [prompt],
            padding=False,
            truncation=False,
            max_length=None,
        )
        current = int(tokens["attention_mask"][0].sum().item())
        return min(current, self.max_length), self.max_length

    def tokenize(self, texts: Sequence[str]) -> list[list[int]]:
        tokens = self._tokenize_batch(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return tokens["input_ids"].tolist()


class Flux2CoreAdapter(nn.Module):
    """Sampler-facing FLUX.2 adapter on external normalized 32-channel latents."""

    def __init__(self, *, transformer: nn.Module, context_dim: int = 7680) -> None:
        super().__init__()
        self.transformer = transformer
        self.codex_config = Flux2CoreConfig(context_dim=int(context_dim))
        self.storage_dtype = getattr(transformer, "storage_dtype", torch.float32)
        self.computation_dtype = getattr(transformer, "computation_dtype", torch.float32)
        self.load_device = getattr(transformer, "load_device", torch.device("cpu"))
        self.offload_device = getattr(transformer, "offload_device", torch.device("cpu"))
        self.initial_device = getattr(transformer, "initial_device", self.offload_device)
        self.architecture = getattr(transformer, "architecture", None)
        self.num_classes = None

    @property
    def dtype(self) -> torch.dtype:
        if isinstance(self.computation_dtype, torch.dtype):
            return self.computation_dtype
        try:
            return next(self.transformer.parameters()).dtype
        except Exception:
            return torch.float32

    def _validate_forward_inputs(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor) -> tuple[int, int, int]:
        batch, channels, height, width = _require_even_spatial_dims(x, label="FLUX.2 sampler latents")
        if channels != _FLUX2_EXTERNAL_LATENT_CHANNELS:
            raise ValueError(
                f"FLUX.2 sampler latents must have {_FLUX2_EXTERNAL_LATENT_CHANNELS} channels; "
                f"got shape={tuple(x.shape)}."
            )
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError(
                f"FLUX.2 context must be a 3D tensor (B,S,C); got shape={getattr(context, 'shape', None)}."
            )
        if int(context.shape[0]) != batch:
            raise ValueError(
                f"FLUX.2 context batch mismatch: context.B={int(context.shape[0])} expected={batch}."
            )
        if int(context.shape[2]) != int(self.codex_config.context_dim):
            raise ValueError(
                "FLUX.2 context feature mismatch: "
                f"got={int(context.shape[2])} expected={int(self.codex_config.context_dim)}."
            )
        if not isinstance(timestep, torch.Tensor):
            raise TypeError(f"FLUX.2 timestep must be a tensor; got {type(timestep).__name__}.")
        if timestep.ndim == 0:
            timestep = timestep.reshape(1)
        if timestep.ndim != 1:
            raise ValueError(f"FLUX.2 timestep must be rank-1; got shape={tuple(timestep.shape)}.")
        if int(timestep.shape[0]) != batch:
            raise ValueError(
                f"FLUX.2 timestep batch mismatch: timestep.B={int(timestep.shape[0])} expected={batch}."
            )
        return batch, height // _FLUX2_PATCH_FACTOR, width // _FLUX2_PATCH_FACTOR

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        control=None,
        transformer_options=None,
        **extra_conds: object,
    ) -> torch.Tensor:
        if control is not None:
            raise NotImplementedError("FLUX.2 does not support ControlNet/control conditioning in this engine seam.")
        image_latents = extra_conds.pop("image_latents", None)
        pooled = extra_conds.pop("y", None)
        unsupported = sorted(str(key) for key in extra_conds.keys())
        if unsupported:
            raise ValueError(
                "FLUX.2 conditioning contract mismatch: unsupported sampler extras "
                f"{unsupported}. Expected cross-attention plus optional inert pooled vector/image_latents."
            )

        if pooled is not None:
            if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
                raise ValueError(
                    "FLUX.2 pooled/vector conditioning placeholder must be a 2D tensor when provided; "
                    f"got shape={getattr(pooled, 'shape', None)}."
                )
            if int(pooled.shape[0]) != int(x.shape[0]):
                raise ValueError(
                    "FLUX.2 pooled/vector conditioning batch mismatch: "
                    f"vector.B={int(pooled.shape[0])} expected={int(x.shape[0])}."
                )

        batch, packed_height, packed_width = self._validate_forward_inputs(x, timestep, context)
        patchified = patchify_flux2_latents(x)
        hidden_states = _pack_flux2_latents(patchified)
        primary_token_count = int(hidden_states.shape[1])
        img_ids = _prepare_latent_ids(
            batch=batch,
            height=packed_height,
            width=packed_width,
            device=x.device,
        )
        txt_ids = _prepare_text_ids(
            batch=batch,
            seq_len=int(context.shape[1]),
            device=context.device,
        )

        if image_latents is not None:
            if not isinstance(image_latents, torch.Tensor) or image_latents.ndim != 4:
                raise ValueError(
                    "FLUX.2 image conditioning must be a 4D tensor (B,C,H,W); "
                    f"got shape={getattr(image_latents, 'shape', None)}."
                )
            if tuple(int(dim) for dim in image_latents.shape) != tuple(int(dim) for dim in x.shape):
                raise ValueError(
                    "FLUX.2 image conditioning must match the sampled latent shape exactly. "
                    f"image_latents={tuple(int(dim) for dim in image_latents.shape)} "
                    f"sampled={tuple(int(dim) for dim in x.shape)}."
                )
            image_latents = image_latents.to(device=x.device, dtype=x.dtype)
            image_patchified = patchify_flux2_latents(image_latents)
            image_hidden_states = _pack_flux2_latents(image_patchified)
            image_ids = _prepare_latent_ids(
                batch=batch,
                height=packed_height,
                width=packed_width,
                device=image_latents.device,
            )
            hidden_states = torch.cat((hidden_states, image_hidden_states), dim=1)
            img_ids = torch.cat((img_ids, image_ids), dim=1)

        joint_attention_kwargs = None
        if isinstance(transformer_options, dict):
            joint_attention_kwargs = transformer_options.get("joint_attention_kwargs")
            if joint_attention_kwargs is not None and not isinstance(joint_attention_kwargs, dict):
                raise TypeError(
                    "FLUX.2 transformer_options['joint_attention_kwargs'] must be a dict when provided."
                )

        output = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=context,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )
        if not isinstance(output, tuple) or not output:
            raise RuntimeError("FLUX.2 transformer returned invalid output contract; expected tuple(sample, ...).")
        sample = output[0]
        sample = sample[:, :primary_token_count, :]
        unpacked = _unpack_flux2_latents(sample, height=packed_height, width=packed_width)
        return unpatchify_flux2_latents(unpacked)


__all__ = [
    "Flux2CoreAdapter",
    "Flux2CoreConfig",
    "Flux2TextProcessingEngine",
    "decode_flux2_external_latents",
    "denormalize_flux2_patchified_latents",
    "encode_flux2_external_latents",
    "normalize_flux2_patchified_latents",
    "patchify_flux2_latents",
    "unpatchify_flux2_latents",
]
