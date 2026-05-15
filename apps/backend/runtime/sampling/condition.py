"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Conditioning helpers for diffusion sampling (tensor and dict-based conditioning payloads).
Enforces shape invariants and wraps conditioning tensors in small helper classes used by samplers and legacy adapters, including optional
dual-tokenization extras (`t5xxl_ids`/`t5xxl_weights`/`t5xxl_attention_mask`) required by Anima conditioning.

Symbols (top-level; keep in sync; no ghosts):
- `repeat_to_batch_size` (function): Repeat/slice a tensor batch dimension to match a target batch size.
- `lcm` (function): Least common multiple helper used for cross-attn concatenation.
- `Condition` (class): Base wrapper for a conditioning tensor with concat/processing helpers.
- `ConditionNoiseShape` (class): Conditioning wrapper that slices by a noise-shape area before batching.
- `ConditionCrossAttn` (class): Cross-attention wrapper supporting LCM-based concat along sequence length.
- `ConditionConstant` (class): Wrapper for constant (non-tensor) conditioning values.
- `compile_conditions` (function): Normalize a conditioning payload (tensor or dict) into the canonical structure used by samplers.
- `compile_weighted_conditions` (function): Compile a list of weighted conditioning indices into per-strength condition bundles.
- `logger` (constant): Module logger used for DEBUG diagnostics.
"""

import math
import logging
import torch
from apps.backend.runtime.logging import emit_backend_message


def repeat_to_batch_size(tensor, batch_size):
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    if tensor.shape[0] < batch_size:
        reps = [math.ceil(batch_size / tensor.shape[0])] + [1] * (len(tensor.shape) - 1)
        return tensor.repeat(reps)[:batch_size]
    return tensor


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


class Condition:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(repeat_to_batch_size(self.cond, batch_size).to(device))

    def can_concat(self, other):
        if self.cond.shape != other.cond.shape:
            return False
        return True

    def concat(self, others):
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
        return torch.cat(conds)


class ConditionNoiseShape(Condition):
    def process_cond(self, batch_size, device, area, **kwargs):
        data = self.cond[:, :, area[2]:area[0] + area[2], area[3]:area[1] + area[3]]
        return self._copy_with(repeat_to_batch_size(data, batch_size).to(device))


class ConditionCrossAttn(Condition):
    @staticmethod
    def _require_positive_seq_len(cond: torch.Tensor, *, where: str) -> int:
        if not isinstance(cond, torch.Tensor):
            raise ValueError(
                f"ConditionCrossAttn {where} must be a torch.Tensor; got type={type(cond).__name__}."
            )
        if cond.ndim != 3:
            raise ValueError(
                f"ConditionCrossAttn {where} must be 3D (B,S,C); got shape={tuple(cond.shape)}."
            )
        seq_len = int(cond.shape[1])
        if seq_len <= 0:
            raise ValueError(
                "ConditionCrossAttn sequence length must be > 0 for concat math; "
                f"got shape={tuple(cond.shape)} at {where}."
            )
        return seq_len

    def can_concat(self, other):
        seq_len_self = self._require_positive_seq_len(self.cond, where="self")
        other_cond = getattr(other, "cond", None)
        seq_len_other = self._require_positive_seq_len(other_cond, where="other")
        s1 = self.cond.shape
        s2 = other_cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]:
                return False

            mult_min = lcm(seq_len_self, seq_len_other)
            diff = mult_min // min(seq_len_self, seq_len_other)
            if diff > 4:
                return False
        return True

    def concat(self, others):
        conds_with_lengths: list[tuple[torch.Tensor, int]] = []
        self_len = self._require_positive_seq_len(self.cond, where="self")
        conds_with_lengths.append((self.cond, self_len))
        crossattn_max_len = self_len
        for idx, x in enumerate(others):
            c = getattr(x, "cond", None)
            c_len = self._require_positive_seq_len(c, where=f"others[{idx}]")
            crossattn_max_len = lcm(crossattn_max_len, c_len)
            conds_with_lengths.append((c, c_len))

        out = []
        for c, c_len in conds_with_lengths:
            if c_len < crossattn_max_len:
                c = c.repeat(1, crossattn_max_len // c_len, 1)
            out.append(c)
        return torch.cat(out)


class ConditionConstant(Condition):
    def __init__(self, cond):
        self.cond = cond

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(self.cond)

    def can_concat(self, other):
        if self.cond != other.cond:
            return False
        return True

    def concat(self, others):
        return self.cond


def compile_conditions(cond):
    if cond is None:
        return None

    if isinstance(cond, torch.Tensor):
        # Legacy path: only cross-attn provided.
        if cond.ndim != 3:
            raise ValueError(f"cross-attn tensor must be 3D (B,S,C); got shape={tuple(cond.shape)}")
        result = dict(
            cross_attn=cond,
            model_conds=dict(
                c_crossattn=ConditionCrossAttn(cond),
            ),
        )
        return [result]

    # Dict-based path: require cross-attn and validate optional extras.
    if not isinstance(cond, dict):
        raise TypeError(f"conditioning must be Tensor or dict; got {type(cond).__name__}")
    if 'crossattn' not in cond:
        raise ValueError("conditioning dict missing required key 'crossattn'")

    cross_attn = cond['crossattn']
    has_vector = 'vector' in cond
    pooled_output = cond['vector'] if has_vector else None

    if not isinstance(cross_attn, torch.Tensor) or cross_attn.ndim != 3:
        raise ValueError(f"'crossattn' must be a 3D tensor (B,S,C); got {type(cross_attn).__name__} shape={getattr(cross_attn,'shape',None)}")

    result = dict(
        cross_attn=cross_attn,
        model_conds=dict(
            c_crossattn=ConditionCrossAttn(cross_attn),
        ),
    )
    if has_vector:
        if not isinstance(pooled_output, torch.Tensor) or pooled_output.ndim != 2:
            raise ValueError(f"'vector' must be a 2D tensor (B,V); got {type(pooled_output).__name__} shape={getattr(pooled_output,'shape',None)}")
        if int(cross_attn.shape[0]) != int(pooled_output.shape[0]):
            raise ValueError(
                "conditioning batch mismatch: "
                f"crossattn.B={int(cross_attn.shape[0])} vector.B={int(pooled_output.shape[0])}"
            )
        result['pooled_output'] = pooled_output
        result['model_conds']['y'] = Condition(pooled_output)

    if 'guidance' in cond:
        guidance = cond['guidance']
        if not isinstance(guidance, torch.Tensor):
            raise ValueError(f"'guidance' must be a tensor; got {type(guidance).__name__}")
        if guidance.ndim != 1:
            raise ValueError(f"'guidance' must be a 1D tensor (B,); got shape={tuple(guidance.shape)}")
        if int(guidance.shape[0]) != int(cross_attn.shape[0]):
            raise ValueError(
                "conditioning batch mismatch: "
                f"crossattn.B={int(cross_attn.shape[0])} guidance.B={int(guidance.shape[0])}"
            )
        result['model_conds']['guidance'] = Condition(guidance)

    has_t5_ids = 't5xxl_ids' in cond
    has_t5_weights = 't5xxl_weights' in cond
    has_t5_mask = 't5xxl_attention_mask' in cond
    if len({has_t5_ids, has_t5_weights, has_t5_mask}) != 1:
        raise ValueError(
            "conditioning must provide 't5xxl_ids', 't5xxl_weights', and 't5xxl_attention_mask' together"
        )

    if has_t5_ids:
        t5xxl_ids = cond['t5xxl_ids']
        t5xxl_weights = cond['t5xxl_weights']
        t5xxl_attention_mask = cond['t5xxl_attention_mask']
        if not isinstance(t5xxl_ids, torch.Tensor) or t5xxl_ids.ndim != 2:
            raise ValueError(
                f"'t5xxl_ids' must be a 2D tensor (B,S); got {type(t5xxl_ids).__name__} "
                f"shape={getattr(t5xxl_ids, 'shape', None)}"
            )
        if not isinstance(t5xxl_weights, torch.Tensor) or t5xxl_weights.ndim != 2:
            raise ValueError(
                f"'t5xxl_weights' must be a 2D tensor (B,S); got {type(t5xxl_weights).__name__} "
                f"shape={getattr(t5xxl_weights, 'shape', None)}"
            )
        if not isinstance(t5xxl_attention_mask, torch.Tensor) or t5xxl_attention_mask.ndim != 2:
            raise ValueError(
                f"'t5xxl_attention_mask' must be a 2D tensor (B,S); got {type(t5xxl_attention_mask).__name__} "
                f"shape={getattr(t5xxl_attention_mask, 'shape', None)}"
            )
        if t5xxl_ids.shape != t5xxl_weights.shape:
            raise ValueError(
                "conditioning shape mismatch: "
                f"t5xxl_ids={tuple(t5xxl_ids.shape)} t5xxl_weights={tuple(t5xxl_weights.shape)}"
            )
        if t5xxl_ids.shape != t5xxl_attention_mask.shape:
            raise ValueError(
                "conditioning shape mismatch: "
                f"t5xxl_ids={tuple(t5xxl_ids.shape)} t5xxl_attention_mask={tuple(t5xxl_attention_mask.shape)}"
            )
        if int(t5xxl_ids.shape[0]) != int(cross_attn.shape[0]):
            raise ValueError(
                "conditioning batch mismatch: "
                f"crossattn.B={int(cross_attn.shape[0])} t5xxl_ids.B={int(t5xxl_ids.shape[0])}"
            )
        result['model_conds']['t5xxl_ids'] = Condition(t5xxl_ids.to(dtype=torch.long))
        result['model_conds']['t5xxl_weights'] = Condition(t5xxl_weights)
        result['model_conds']['t5xxl_attention_mask'] = Condition(t5xxl_attention_mask.to(dtype=torch.long))

    if 'image_latents' in cond and cond['image_latents'] is not None:
        image_latents = cond['image_latents']
        if not isinstance(image_latents, torch.Tensor) or image_latents.ndim != 4:
            raise ValueError(
                f"'image_latents' must be a 4D tensor (B,C,H,W); got {type(image_latents).__name__} "
                f"shape={getattr(image_latents,'shape',None)}"
            )
        result['model_conds']['image_latents'] = Condition(image_latents)

    emit_backend_message(
        "compiled conditions",
        logger=__name__,
        level=logging.DEBUG,
        cross_attn=tuple(cross_attn.shape),
        pooled=None if pooled_output is None else tuple(pooled_output.shape),
    )
    return [result]


def compile_weighted_conditions(cond, weights):
    transposed = list(map(list, zip(*weights)))
    results = []

    for cond_pre in transposed:
        current_indices = []
        current_weight = 0
        for i, w in cond_pre:
            current_indices.append(i)
            current_weight = w

        if hasattr(cond, 'advanced_indexing'):
            feed = cond.advanced_indexing(current_indices)
        else:
            feed = cond[current_indices]

        h = compile_conditions(feed)
        h[0]['strength'] = current_weight
        results += h

    return results
