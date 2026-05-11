"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Script-hook helpers for pipeline orchestration.
Keeps processing-level script callbacks and shared job metadata isolated from sampler execution logic.

Symbols (top-level; keep in sync; no ghosts):
- `run_process_scripts` (function): Run processing scripts (legacy-compatible) when present.
- `set_shared_job` (function): Update shared job metadata for batch runs.
- `collect_lora_selections` (function): Merge global selections with prompt-local LoRA descriptors, with prompt-local same-path weights winning.
- `run_before_sampling_hooks` (function): Invoke before-sampling hooks (scripts + shared job metadata).
- `run_post_sample_hooks` (function): Invoke post-sample hooks, returning potentially modified samples.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch

from apps.backend.runtime.adapters.lora import selections as lora_selections
from apps.backend.core.state import state as backend_state
from apps.backend.runtime.processing.datatypes import PromptContext


def run_process_scripts(processing: Any) -> None:
    """Execute legacy script hooks if present."""
    script_runner = getattr(processing, "scripts", None)
    if script_runner is not None and hasattr(script_runner, "process"):
        script_runner.process(processing)


def set_shared_job(processing: Any) -> None:
    """Update shared backend job metadata for batch runs."""
    if getattr(processing, "iterations", 1) <= 1:
        return
    backend_state.begin(job=f"Batch 1 out of {processing.iterations}")


def collect_lora_selections(prompt_loras: Sequence[Any]) -> list[Any]:
    """Merge global selections with prompt-local LoRA descriptors."""
    ordered_paths: list[str] = []
    last_by_path: dict[str, Any] = {}
    try:
        all_selections: Iterable[Any] = list(lora_selections.get_selections()) + list(prompt_loras)
    except Exception:
        all_selections = list(prompt_loras)
    for sel in all_selections:
        path = getattr(sel, "path", None)
        if not path:
            continue
        path_key = str(path)
        if path_key not in last_by_path:
            ordered_paths.append(path_key)
        last_by_path[path_key] = sel
    return [last_by_path[path] for path in ordered_paths]


def run_before_sampling_hooks(
    processing: Any,
    prompt_context: PromptContext,
    seeds: Sequence[int],
    subseeds: Sequence[int],
) -> None:
    """Invoke before-sampling hooks on processing scripts."""
    script_runner = getattr(processing, "scripts", None)
    if script_runner is None:
        return

    hook_kwargs = {
        "batch_number": 0,
        "prompts": prompt_context.prompts,
        "seeds": list(seeds),
        "subseeds": list(subseeds),
        "negative_prompts": prompt_context.negative_prompts,
    }

    if hasattr(script_runner, "before_process_batch"):
        script_runner.before_process_batch(processing, **hook_kwargs)

    if hasattr(script_runner, "process_batch"):
        script_runner.process_batch(processing, **hook_kwargs)

    set_shared_job(processing)


def run_post_sample_hooks(processing: Any, samples: torch.Tensor) -> torch.Tensor:
    """Invoke post-sample hooks, returning the potentially modified samples."""
    script_runner = getattr(processing, "scripts", None)
    if script_runner is None or not hasattr(script_runner, "post_sample"):
        return samples

    class _Args:
        def __init__(self, value: torch.Tensor) -> None:
            self.samples = value

    args = _Args(samples)
    script_runner.post_sample(processing, args)
    return getattr(args, "samples", samples)
