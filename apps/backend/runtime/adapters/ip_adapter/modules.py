"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: IP-Adapter runtime modules and attention patch implementations.
Provides the tranche-1 projection/resampler modules, exact KV weight holders, and the shared attn2 replace patch used by the
IP-Adapter runtime stage.

Symbols (top-level; keep in sync; no ghosts):
- `ImageProjectionModel` (class): Base IP-Adapter image projection module for pooled CLIP image embeddings.
- `MlpProjectionModel` (class): Full-feature MLP projection module (unsupported in tranche 1 but kept as a typed loader target).
- `PerceiverAttention` (class): Resampler attention block used by IP-Adapter Plus layouts.
- `Resampler` (class): IP-Adapter Plus resampler implementation.
- `IpAdapterKvSlotSpec` (dataclass): Canonical slot/source-key metadata derived from one IP-Adapter checkpoint KV inventory.
- `IpAdapterKvProjectionSet` (class): Exact-source-key holder for IP-Adapter `to_k_ip` / `to_v_ip` linear projections.
- `IpAdapterCrossAttentionPatch` (class): Shared attn2 replace patch that augments base cross-attention with image-conditioned K/V paths.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass

import torch
from torch import nn

from apps.backend.infra.config.env_flags import env_flag, env_int
from apps.backend.runtime.attention import attention_function
from apps.backend.runtime.logging import emit_backend_message

_IP_ADAPTER_PATCH_DEBUG_EMIT_COUNT: ContextVar[int] = ContextVar(
    "ip_adapter_patch_debug_emit_count",
    default=0,
)


@dataclass(frozen=True)
class IpAdapterKvSlotSpec:
    slot_number: int
    k_source_key: str
    v_source_key: str
    input_dim: int
    output_dim: int


class ImageProjectionModel(nn.Module):
    def __init__(self, *, cross_attention_dim: int, clip_embeddings_dim: int, clip_extra_context_tokens: int) -> None:
        super().__init__()
        self.cross_attention_dim = int(cross_attention_dim)
        self.clip_extra_context_tokens = int(clip_extra_context_tokens)
        self.proj = nn.Linear(int(clip_embeddings_dim), self.clip_extra_context_tokens * self.cross_attention_dim)
        self.norm = nn.LayerNorm(self.cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(image_embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        return self.norm(tokens)


class MlpProjectionModel(nn.Module):
    def __init__(self, *, cross_attention_dim: int, clip_embeddings_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(int(clip_embeddings_dim), int(clip_embeddings_dim)),
            nn.GELU(),
            nn.Linear(int(clip_embeddings_dim), int(cross_attention_dim)),
            nn.LayerNorm(int(cross_attention_dim)),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.proj(image_embeds)


class FeedForward(nn.Sequential):
    def __init__(self, *, dim: int, mult: int = 4) -> None:
        inner_dim = int(dim * mult)
        super().__init__(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias=False),
        )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.scale = float(dim_head) ** -0.5
        self.dim_head = int(dim_head)
        self.heads = int(heads)
        inner_dim = self.dim_head * self.heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, inputs: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        inputs = self.norm1(inputs)
        latents = self.norm2(latents)
        batch_size, latent_length, _ = latents.shape
        queries = self._reshape_heads(self.to_q(latents))
        kv_inputs = torch.cat((inputs, latents), dim=-2)
        keys, values = self.to_kv(kv_inputs).chunk(2, dim=-1)
        keys = self._reshape_heads(keys)
        values = self._reshape_heads(values)
        scale = 1.0 / math.sqrt(math.sqrt(self.dim_head))
        weights = (queries * scale) @ (keys * scale).transpose(-2, -1)
        weights = torch.softmax(weights.float(), dim=-1).to(dtype=weights.dtype)
        attended = weights @ values
        attended = attended.permute(0, 2, 1, 3).reshape(batch_size, latent_length, -1)
        return self.to_out(attended)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, length, width = tensor.shape
        tensor = tensor.view(batch_size, length, self.heads, -1)
        return tensor.transpose(1, 2).reshape(batch_size, self.heads, length, -1)


class Resampler(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int,
        heads: int,
        num_queries: int,
        embedding_dim: int,
        output_dim: int,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, int(num_queries), int(dim)) / (float(dim) ** 0.5))
        self.proj_in = nn.Linear(int(embedding_dim), int(dim))
        self.proj_out = nn.Linear(int(dim), int(output_dim))
        self.norm_out = nn.LayerNorm(int(output_dim))
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=int(dim), dim_head=int(dim_head), heads=int(heads)),
                        FeedForward(dim=int(dim), mult=int(ff_mult)),
                    ]
                )
                for _ in range(int(depth))
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latents = self.latents.repeat(inputs.size(0), 1, 1)
        inputs = self.proj_in(inputs)
        for attention_layer, feed_forward in self.layers:
            latents = attention_layer(inputs, latents) + latents
            latents = feed_forward(latents) + latents
        return self.norm_out(self.proj_out(latents))


class IpAdapterKvProjectionSet(nn.Module):
    def __init__(self, state_dict) -> None:
        super().__init__()
        module_dict: OrderedDict[str, nn.Module] = OrderedDict()
        self._source_key_to_module_name: dict[str, str] = {}
        self.slot_specs = self.inspect_state_dict(state_dict)
        for source_key, weight in self._ordered_weight_items(state_dict):
            module_name = self._module_name_from_source_key(source_key)
            layer = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=False)
            layer.weight.data.copy_(weight)
            module_dict[module_name] = layer
            self._source_key_to_module_name[source_key] = module_name
        self.to_kvs = nn.ModuleDict(module_dict)
        self.slot_count = len(self.slot_specs)

    @classmethod
    def inspect_state_dict(cls, state_dict) -> tuple[IpAdapterKvSlotSpec, ...]:
        grouped_weights: dict[int, dict[str, tuple[str, torch.Tensor]]] = {}
        for source_key, weight in cls._ordered_weight_items(state_dict):
            slot_number, kind = cls._parse_source_key(source_key)
            grouped_weights.setdefault(slot_number, {})[kind] = (source_key, weight)
        slot_specs: list[IpAdapterKvSlotSpec] = []
        for slot_number in sorted(grouped_weights):
            slot_pair = grouped_weights[slot_number]
            try:
                k_source_key, k_weight = slot_pair["to_k_ip"]
                v_source_key, v_weight = slot_pair["to_v_ip"]
            except KeyError as exc:
                raise RuntimeError(
                    f"IP-Adapter KV state dict is missing matched to_k_ip/to_v_ip pairs for slot {slot_number}."
                ) from exc
            if k_weight.ndim != 2:
                raise RuntimeError(
                    f"IP-Adapter KV weight '{k_source_key}' must be 2-D; got shape={tuple(k_weight.shape)}."
                )
            if v_weight.ndim != 2:
                raise RuntimeError(
                    f"IP-Adapter KV weight '{v_source_key}' must be 2-D; got shape={tuple(v_weight.shape)}."
                )
            if tuple(k_weight.shape) != tuple(v_weight.shape):
                raise RuntimeError(
                    "IP-Adapter KV slot pair mismatch: "
                    f"slot={slot_number} k_shape={tuple(k_weight.shape)} v_shape={tuple(v_weight.shape)}."
                )
            slot_specs.append(
                IpAdapterKvSlotSpec(
                    slot_number=int(slot_number),
                    k_source_key=str(k_source_key),
                    v_source_key=str(v_source_key),
                    input_dim=int(k_weight.shape[1]),
                    output_dim=int(k_weight.shape[0]),
                )
            )
        return tuple(slot_specs)

    @staticmethod
    def _ordered_weight_items(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
        items: list[tuple[str, torch.Tensor]] = []
        seen_slots: dict[int, set[str]] = {}
        for source_key, weight in state_dict.items():
            if not isinstance(source_key, str):
                raise RuntimeError("IP-Adapter KV state dict keys must be strings.")
            if not isinstance(weight, torch.Tensor):
                raise RuntimeError(f"IP-Adapter KV entry '{source_key}' must be a torch.Tensor.")
            slot_number, kind = IpAdapterKvProjectionSet._parse_source_key(source_key)
            seen_slots.setdefault(slot_number, set()).add(kind)
            items.append((source_key, weight))
        missing = [
            slot_number
            for slot_number, kinds in sorted(seen_slots.items())
            if kinds != {"to_k_ip", "to_v_ip"}
        ]
        if missing:
            raise RuntimeError(
                "IP-Adapter KV state dict is missing matched to_k_ip/to_v_ip pairs for slot(s): "
                + ", ".join(str(slot) for slot in missing)
            )
        return sorted(items, key=lambda item: IpAdapterKvProjectionSet._sort_key(item[0]))

    @staticmethod
    def _parse_source_key(source_key: str) -> tuple[int, str]:
        if not source_key.endswith(".weight"):
            raise RuntimeError(f"Unsupported IP-Adapter KV source key '{source_key}'. Expected '*.weight'.")
        stem = source_key[:-len(".weight")]
        parts = stem.split(".")
        if len(parts) != 2 or parts[1] not in {"to_k_ip", "to_v_ip"}:
            raise RuntimeError(
                f"Unsupported IP-Adapter KV source key '{source_key}'. Expected '<odd_slot>.to_k_ip.weight' or '.to_v_ip.weight'."
            )
        try:
            slot_number = int(parts[0])
        except ValueError as exc:
            raise RuntimeError(f"Unsupported IP-Adapter KV source key '{source_key}'.") from exc
        if slot_number < 1 or slot_number % 2 == 0:
            raise RuntimeError(
                f"Unsupported IP-Adapter KV source key '{source_key}'. Slot number must be an odd integer >= 1."
            )
        return slot_number, parts[1]

    @staticmethod
    def _slot_number_from_source_key(source_key: str) -> int:
        return IpAdapterKvProjectionSet._parse_source_key(source_key)[0]

    @staticmethod
    def _sort_key(source_key: str) -> tuple[int, int]:
        slot_number, kind = IpAdapterKvProjectionSet._parse_source_key(source_key)
        kind_rank = 0 if kind == "to_k_ip" else 1
        return slot_number, kind_rank

    @staticmethod
    def _module_name_from_source_key(source_key: str) -> str:
        slot_number, kind = IpAdapterKvProjectionSet._parse_source_key(source_key)
        return f"slot_{slot_number}_{kind}"

    def projection(self, source_key: str) -> nn.Linear:
        try:
            module_name = self._source_key_to_module_name[source_key]
        except KeyError as exc:
            raise RuntimeError(f"Missing IP-Adapter projection for source key '{source_key}'.") from exc
        return self.to_kvs[module_name]


class IpAdapterCrossAttentionPatch(nn.Module):
    def __init__(
        self,
        *,
        slot_index: int,
        k_source_key: str,
        v_source_key: str,
        weight: float,
        sigma_start: float,
        sigma_end: float,
        ip_layers: IpAdapterKvProjectionSet,
        condition: torch.Tensor,
        uncondition: torch.Tensor,
    ) -> None:
        super().__init__()
        self.slot_index = int(slot_index)
        self.weight = float(weight)
        self.sigma_start = float(sigma_start)
        self.sigma_end = float(sigma_end)
        self.ip_layers = ip_layers
        self.register_buffer("condition_tokens", condition, persistent=False)
        self.register_buffer("uncondition_tokens", uncondition, persistent=False)
        self.k_source_key = str(k_source_key)
        self.v_source_key = str(v_source_key)

    def forward(
        self,
        queries: torch.Tensor,
        context_keys: torch.Tensor,
        context_values: torch.Tensor,
        extra_options: dict[str, object],
    ) -> torch.Tensor:
        base = attention_function(queries, context_keys, context_values, int(extra_options["n_heads"]))
        sigma_value = self._sigma_value(extra_options)
        sigma_active = self._sigma_is_active(extra_options, sigma_value=sigma_value)
        if not sigma_active:
            result = base.to(dtype=queries.dtype)
            self._maybe_debug_emit(
                extra_options=extra_options,
                sigma_value=sigma_value,
                sigma_active=sigma_active,
                queries=queries,
                context_keys=context_keys,
                context_values=context_values,
                cond_tokens=None,
                uncond_tokens=None,
                ip_keys=None,
                ip_values=None,
                base=base,
                conditioned=None,
                result=result,
                batch_prompt=None,
            )
            return result
        cond_or_uncond = extra_options.get("cond_or_uncond")
        if not isinstance(cond_or_uncond, (list, tuple)) or len(cond_or_uncond) == 0:
            raise RuntimeError("IP-Adapter attention patch requires non-empty extra_options['cond_or_uncond'].")
        batch_total = int(queries.shape[0])
        batch_prompt = batch_total // len(cond_or_uncond)
        if batch_prompt < 1:
            raise RuntimeError(
                f"Invalid IP-Adapter batch geometry: batch_total={batch_total} cond_or_uncond={len(cond_or_uncond)}."
            )
        cond_tokens = self._expand_tokens(self.condition_tokens, batch_prompt=batch_prompt)
        uncond_tokens = self._expand_tokens(self.uncondition_tokens, batch_prompt=batch_prompt)
        k_cond = self.ip_layers.projection(self.k_source_key)(cond_tokens)
        k_uncond = self.ip_layers.projection(self.k_source_key)(uncond_tokens)
        v_cond = self.ip_layers.projection(self.v_source_key)(cond_tokens)
        v_uncond = self.ip_layers.projection(self.v_source_key)(uncond_tokens)
        ip_keys = self._select_cfg_branch(k_cond, k_uncond, cond_or_uncond).to(dtype=queries.dtype)
        ip_values = self._select_cfg_branch(v_cond, v_uncond, cond_or_uncond).to(dtype=queries.dtype)
        conditioned = attention_function(queries, ip_keys, ip_values, int(extra_options["n_heads"]))
        result = base.to(dtype=queries.dtype) + (conditioned.to(dtype=queries.dtype) * self.weight)
        self._maybe_debug_emit(
            extra_options=extra_options,
            sigma_value=sigma_value,
            sigma_active=sigma_active,
            queries=queries,
            context_keys=context_keys,
            context_values=context_values,
            cond_tokens=cond_tokens,
            uncond_tokens=uncond_tokens,
            ip_keys=ip_keys,
            ip_values=ip_values,
            base=base,
            conditioned=conditioned,
            result=result,
            batch_prompt=batch_prompt,
        )
        return result

    @staticmethod
    def _sigma_value(extra_options: dict[str, object]) -> float | None:
        sigma_value = extra_options.get("sigmas")
        if isinstance(sigma_value, torch.Tensor) and sigma_value.numel() > 0:
            return float(sigma_value.flatten()[0].item())
        return None

    def _sigma_is_active(self, extra_options: dict[str, object], *, sigma_value: float | None = None) -> bool:
        sigma = sigma_value if sigma_value is not None else self._sigma_value(extra_options)
        if sigma is not None:
            if sigma > self.sigma_start or sigma < self.sigma_end:
                return False
        return True

    @staticmethod
    def _expand_tokens(tokens: torch.Tensor, *, batch_prompt: int) -> torch.Tensor:
        token_batch = int(tokens.shape[0])
        if token_batch <= 0:
            raise RuntimeError("IP-Adapter token batch is empty.")
        if token_batch == batch_prompt:
            return tokens
        if batch_prompt < token_batch:
            raise RuntimeError(
                "IP-Adapter token batch mismatch: "
                f"runtime batch_prompt={batch_prompt} is smaller than prepared token batch={token_batch}."
            )
        if batch_prompt % token_batch != 0:
            raise RuntimeError(
                "IP-Adapter token batch mismatch: "
                f"runtime batch_prompt={batch_prompt} is not an integer multiple of prepared token batch={token_batch}."
            )
        repeat_factor = batch_prompt // token_batch
        return tokens.repeat_interleave(repeat_factor, dim=0)

    @staticmethod
    def _select_cfg_branch(
        cond_tokens: torch.Tensor,
        uncond_tokens: torch.Tensor,
        cond_or_uncond: list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        selected: list[torch.Tensor] = []
        for branch in cond_or_uncond:
            if int(branch) == 0:
                selected.append(cond_tokens)
            elif int(branch) == 1:
                selected.append(uncond_tokens)
            else:
                raise RuntimeError(
                    f"Unsupported cond_or_uncond branch {branch!r}; expected 0 (cond) or 1 (uncond)."
                )
        return torch.cat(selected, dim=0)

    @classmethod
    def _debug_enabled(cls) -> bool:
        return env_flag("CODEX_IP_ADAPTER_DEBUG") or env_flag("CODEX_IP_ADAPTER_DEBUG_PATCH")

    @classmethod
    def _debug_limit(cls) -> int:
        return env_int("CODEX_IP_ADAPTER_DEBUG_PATCH_N", 6, min_value=0)

    @classmethod
    def reset_debug_counter(cls) -> None:
        _IP_ADAPTER_PATCH_DEBUG_EMIT_COUNT.set(0)

    @classmethod
    def _debug_counter(cls) -> int:
        return int(_IP_ADAPTER_PATCH_DEBUG_EMIT_COUNT.get())

    @classmethod
    def _increment_debug_counter(cls) -> None:
        _IP_ADAPTER_PATCH_DEBUG_EMIT_COUNT.set(cls._debug_counter() + 1)

    @staticmethod
    def _tensor_stats(label: str, tensor: torch.Tensor | None) -> str:
        if tensor is None or not torch.is_tensor(tensor):
            return f"{label}=<none>"
        with torch.no_grad():
            data = tensor.detach()
            stats = data.float()
            return (
                f"{label}:shape={tuple(data.shape)} dtype={data.dtype} dev={data.device} "
                f"min={float(stats.min().item()):.6g} max={float(stats.max().item()):.6g} "
                f"mean={float(stats.mean().item()):.6g} std={float(stats.std(unbiased=False).item()):.6g} "
                f"norm={float(stats.norm().item()):.6g}"
            )

    def _maybe_debug_emit(
        self,
        *,
        extra_options: dict[str, object],
        sigma_value: float | None,
        sigma_active: bool,
        queries: torch.Tensor,
        context_keys: torch.Tensor,
        context_values: torch.Tensor,
        cond_tokens: torch.Tensor | None,
        uncond_tokens: torch.Tensor | None,
        ip_keys: torch.Tensor | None,
        ip_values: torch.Tensor | None,
        base: torch.Tensor,
        conditioned: torch.Tensor | None,
        result: torch.Tensor,
        batch_prompt: int | None,
    ) -> None:
        if not self._debug_enabled():
            return
        if self._debug_counter() >= self._debug_limit():
            return
        cond_or_uncond = extra_options.get("cond_or_uncond")
        cond_summary = None
        if isinstance(cond_or_uncond, (list, tuple)):
            cond_summary = [int(branch) for branch in cond_or_uncond]
        emit_backend_message(
            "[ip-adapter-debug] patch",
            logger=__name__,
            slot_index=self.slot_index,
            block=extra_options.get("block"),
            transformer_index=extra_options.get("transformer_index"),
            block_index=extra_options.get("block_index"),
            k_source_key=self.k_source_key,
            v_source_key=self.v_source_key,
            weight=self.weight,
            sigma=sigma_value,
            sigma_active=sigma_active,
            sigma_start=self.sigma_start,
            sigma_end=self.sigma_end,
            cond_or_uncond=cond_summary,
            batch_total=int(queries.shape[0]),
            batch_prompt=batch_prompt,
            n_heads=extra_options.get("n_heads"),
        )
        emit_backend_message(f"[ip-adapter-debug] {self._tensor_stats('queries', queries)}", logger=__name__)
        emit_backend_message(f"[ip-adapter-debug] {self._tensor_stats('context_keys', context_keys)}", logger=__name__)
        emit_backend_message(f"[ip-adapter-debug] {self._tensor_stats('context_values', context_values)}", logger=__name__)
        emit_backend_message(f"[ip-adapter-debug] {self._tensor_stats('condition_tokens', cond_tokens)}", logger=__name__)
        emit_backend_message(f"[ip-adapter-debug] {self._tensor_stats('uncondition_tokens', uncond_tokens)}", logger=__name__)
        emit_backend_message(f"[ip-adapter-debug] {self._tensor_stats('ip_keys', ip_keys)}", logger=__name__)
        emit_backend_message(f"[ip-adapter-debug] {self._tensor_stats('ip_values', ip_values)}", logger=__name__)
        emit_backend_message(f"[ip-adapter-debug] {self._tensor_stats('base', base)}", logger=__name__)
        emit_backend_message(f"[ip-adapter-debug] {self._tensor_stats('conditioned', conditioned)}", logger=__name__)
        emit_backend_message(f"[ip-adapter-debug] {self._tensor_stats('result', result)}", logger=__name__)
        self._increment_debug_counter()
