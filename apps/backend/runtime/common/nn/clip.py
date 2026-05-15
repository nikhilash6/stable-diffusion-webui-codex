"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: CLIP wrapper module used to integrate transformer backbones with optional text projection.

Symbols (top-level; keep in sync; no ghosts):
- `_MatmulProjection` (class): Projection head with matmul-oriented weight semantics (`x @ weight`) that honors Codex manual-cast runtime precision.
- `IntegratedCLIP` (class): Wraps a CLIP-like transformer and exposes HF-style forward outputs.
"""

import torch

from apps.backend.runtime.ops.operations import get_operation_context, main_stream_worker, weights_manual_cast


class _MatmulProjection(torch.nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        ctx = get_operation_context()
        dtype = ctx.dtype or torch.float32
        self.weight = torch.nn.Parameter(torch.empty((embed_dim, embed_dim), device=ctx.device, dtype=dtype))
        self.bias = None
        self.parameters_manual_cast = ctx.manual_cast_enabled
        torch.nn.init.normal_(self.weight, std=embed_dim ** -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.parameters_manual_cast:
            weight, _bias, signal = weights_manual_cast(self, hidden_states)
            with main_stream_worker(weight, None, signal):
                return hidden_states @ weight
        return hidden_states @ self.weight


class IntegratedCLIP(torch.nn.Module):
    def __init__(
        self,
        cls,
        config,
        add_text_projection: bool = False,
        text_projection_layout: str = "linear",
    ):
        super().__init__()
        self.transformer = cls(config)
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))

        if add_text_projection:
            embed_dim = config.hidden_size
            if text_projection_layout == "linear":
                self.transformer.text_projection = torch.nn.Linear(embed_dim, embed_dim, bias=False)
            elif text_projection_layout == "matmul":
                self.transformer.text_projection = _MatmulProjection(embed_dim)
            else:
                raise ValueError(
                    "IntegratedCLIP text_projection_layout must be one of: linear, matmul; "
                    f"got: {text_projection_layout!r}"
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ):
        # Forward mask/position_ids when available; fall back gracefully otherwise.
        kwargs = {"output_hidden_states": output_hidden_states}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids

        outputs = self.transformer(input_ids, **kwargs)

        if return_dict:
            return outputs
        return (outputs.last_hidden_state, outputs.pooler_output, outputs.hidden_states)
