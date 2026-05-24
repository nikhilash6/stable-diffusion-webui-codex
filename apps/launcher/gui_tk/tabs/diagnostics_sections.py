"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Declarative Diagnostics tab section inventory.
Owns diagnostics/debug/logging/profiler grouping so `DiagnosticsTab` renders controls through the shared form renderer instead of manual row construction.

Symbols (top-level; keep in sync; no ghosts):
- `DiagnosticsFlagSpec` (dataclass): One boolean diagnostics env control.
- `DiagnosticsEntrySpec` (dataclass): One text/integer diagnostics env control.
- `DiagnosticsSectionSpec` (dataclass): Grouped diagnostics section inventory.
- `DIAGNOSTICS_SECTIONS` (constant): Ordered diagnostics/debug/tracing/profiler sections.
- `LOG_LEVEL_FLAGS` (constant): Ordered logging-level flags and defaults.
- `LOG_FILE_LABEL` (constant): Label for the log-file toggle control.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DiagnosticsFlagSpec:
    key: str
    label: str
    advanced: bool = False


@dataclass(frozen=True, slots=True)
class DiagnosticsEntrySpec:
    field_id: str
    key: str
    label: str
    width: int
    advanced: bool = False


@dataclass(frozen=True, slots=True)
class DiagnosticsSectionSpec:
    title: str
    flags: tuple[DiagnosticsFlagSpec, ...] = field(default_factory=tuple)
    entries: tuple[DiagnosticsEntrySpec, ...] = field(default_factory=tuple)
    advanced: bool = False


DIAGNOSTICS_SECTIONS: tuple[DiagnosticsSectionSpec, ...] = (
    DiagnosticsSectionSpec(
        title="Sampling + Pipeline",
        flags=(
            DiagnosticsFlagSpec("CODEX_DEBUG_COND", "Conditioning Debug"),
            DiagnosticsFlagSpec("CODEX_LOG_SAMPLER", "Sampler Verbose Logs"),
            DiagnosticsFlagSpec("CODEX_LOG_CFG_DELTA", "CFG Delta Logs (requires Sampler Verbose Logs)"),
            DiagnosticsFlagSpec("CODEX_LOG_SIGMAS", "Sigma Ladder Logs"),
            DiagnosticsFlagSpec("CODEX_PIPELINE_DEBUG", "Pipeline Debug", advanced=True),
            DiagnosticsFlagSpec("CODEX_DUMP_LATENTS", "Dump Latents", advanced=True),
            DiagnosticsFlagSpec("CODEX_TIMELINE", "Timeline Trace (TVA-style execution timeline)", advanced=True),
        ),
        entries=(
            DiagnosticsEntrySpec("cfg_delta_steps", "CODEX_LOG_CFG_DELTA_N", "CFG Delta Steps (N):", 10),
        ),
    ),
    DiagnosticsSectionSpec(
        title="Runtime Diagnostics",
        advanced=True,
        flags=(
            DiagnosticsFlagSpec("CODEX_VAE_TENSOR_STATS", "VAE Tensor Stats (requires DEBUG logging)", advanced=True),
            DiagnosticsFlagSpec("CODEX_MEMORY_DEBUG", "Memory Debug (also enabled by DEBUG logging)", advanced=True),
        ),
    ),
    DiagnosticsSectionSpec(
        title="Deep Traces + Contract",
        advanced=True,
        flags=(
            DiagnosticsFlagSpec("CODEX_TRACE_INFERENCE_DEBUG", "Trace Debug: Inference", advanced=True),
            DiagnosticsFlagSpec("CODEX_TRACE_LOAD_PATCH_DEBUG", "Trace Debug: Load/Patch", advanced=True),
            DiagnosticsFlagSpec("CODEX_TRACE_CALL_DEBUG", "Trace Debug: Call Trace", advanced=True),
            DiagnosticsFlagSpec("CODEX_TRACE_CONTRACT", "Contract Trace (JSONL in logs/contract-trace)", advanced=True),
            DiagnosticsFlagSpec("CODEX_TRACE_PROFILER", "Contract Profiler Toggle (maps to --trace-profiler)", advanced=True),
        ),
        entries=(
            DiagnosticsEntrySpec(
                "trace_call_max",
                "CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC",
                "Call trace max / func (0=unlimited):",
                10,
                advanced=True,
            ),
            DiagnosticsEntrySpec("dump_latents_path", "CODEX_DUMP_LATENTS_PATH", "Dump latents path:", 40, advanced=True),
        ),
    ),
    DiagnosticsSectionSpec(
        title="Profiler",
        advanced=True,
        flags=(
            DiagnosticsFlagSpec("CODEX_PROFILE", "Global Profiler (torch.profiler)", advanced=True),
            DiagnosticsFlagSpec("CODEX_PROFILE_TRACE", "Profiler: export Perfetto trace", advanced=True),
            DiagnosticsFlagSpec("CODEX_PROFILE_RECORD_SHAPES", "Profiler: record shapes", advanced=True),
            DiagnosticsFlagSpec("CODEX_PROFILE_PROFILE_MEMORY", "Profiler: profile memory", advanced=True),
            DiagnosticsFlagSpec("CODEX_PROFILE_WITH_STACK", "Profiler: include stacks (very heavy)", advanced=True),
        ),
        entries=(
            DiagnosticsEntrySpec("profile_top_n", "CODEX_PROFILE_TOP_N", "Profiler top ops (N):", 10, advanced=True),
            DiagnosticsEntrySpec("profile_max_steps", "CODEX_PROFILE_MAX_STEPS", "Profiler max steps (0=all):", 10, advanced=True),
        ),
    ),
)

LOG_LEVEL_FLAGS: tuple[DiagnosticsFlagSpec, ...] = (
    DiagnosticsFlagSpec("CODEX_LOG_DEBUG", "DEBUG (verbose)"),
    DiagnosticsFlagSpec("CODEX_LOG_INFO", "INFO"),
    DiagnosticsFlagSpec("CODEX_LOG_WARNING", "WARNING"),
    DiagnosticsFlagSpec("CODEX_LOG_ERROR", "ERROR"),
)
LOG_FILE_LABEL = "Write to log file (logs/codex-*.log)"
