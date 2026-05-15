"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Diagnostics tab for the Tk launcher.
Shows launcher environment preflight checks plus backend debug, trace, profiler, and log-level controls grouped by troubleshooting purpose.

Symbols (top-level; keep in sync; no ghosts):
- `DiagnosticsTab` (class): Diagnostics tab controller for checks and debug/logging/profiling settings.
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Iterable, List, Tuple

from apps.launcher.checks import CodexLaunchCheck
from apps.launcher.settings import BoolSetting, ChoiceSetting, CFG_BATCH_MODE_CHOICES, IntSetting, SettingValidationError

from ..controller import LauncherController
from ..widgets import ScrollableFrame


TRACE_DEBUG_DEFAULT = "10"


class DiagnosticsTab:
    def __init__(
        self,
        controller: LauncherController,
        *,
        mark_changed: Callable[[], None],
        run_checks_async: Callable[[], None],
        canvas_bg: str,
    ) -> None:
        self._controller = controller
        self._mark_changed = mark_changed
        self._run_checks_async = run_checks_async
        self._canvas_bg = str(canvas_bg)

        self.frame: ttk.Frame | None = None
        self._checks_tree: ttk.Treeview | None = None

        self._debug_flags: Dict[str, tk.BooleanVar] = {}
        self._log_levels: Dict[str, tk.BooleanVar] = {}

        self._var_cfg_delta_n = tk.StringVar()
        self._var_cfg_batch_mode = tk.StringVar()
        self._var_trace_max = tk.StringVar()
        self._var_dump_path = tk.StringVar()
        self._var_profile_top_n = tk.StringVar()
        self._var_profile_max_steps = tk.StringVar()
        self._var_log_file = tk.BooleanVar()
        self._advanced_visible = False
        self._advanced_widgets: List[tk.Widget] = []
        self._cfg_delta_trace_ids: List[Tuple[tk.Variable, str]] = []

    def build(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        scroll = ScrollableFrame(frame, canvas_bg=self._canvas_bg)
        scroll.pack(fill="both", expand=True)
        body = scroll.inner
        body.columnconfigure(0, weight=1)

        checks_box = ttk.LabelFrame(body, text="  Environment Checks  ", padding=14)
        checks_box.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        checks_box.columnconfigure(0, weight=1)
        checks_box.rowconfigure(0, weight=1)

        tree_frame = ttk.Frame(checks_box)
        tree_frame.grid(row=0, column=0, sticky="ew")
        tree_frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(tree_frame, columns=("name", "ok", "detail"), show="headings", height=6)
        tree.heading("name", text="Check")
        tree.heading("ok", text="OK")
        tree.heading("detail", text="Detail")
        tree.column("name", width=170, minwidth=140, anchor="w", stretch=False)
        tree.column("ok", width=64, minwidth=64, anchor="center", stretch=False)
        tree.column("detail", width=680, minwidth=320, anchor="w", stretch=True)
        tree.grid(row=0, column=0, sticky="ew")
        xscroll = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        xscroll.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        tree.configure(xscrollcommand=xscroll.set)
        self._checks_tree = tree

        checks_actions = ttk.Frame(checks_box)
        checks_actions.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        checks_actions.columnconfigure(0, weight=1)
        ttk.Label(
            checks_actions,
            text="These checks validate launcher prerequisites only; service runtime failures still show up in Logs.",
            style="Muted.TLabel",
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(checks_actions, text="↻ Re-run checks", command=self._run_checks_async).grid(row=0, column=1, sticky="e")

        sampling_box = ttk.LabelFrame(body, text="  Sampling + Pipeline  ", padding=14)
        sampling_box.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        sampling_box.columnconfigure(0, weight=1)

        trace_box = ttk.LabelFrame(body, text="  Deep Traces + Contract  ", padding=14)
        trace_box.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        trace_box.columnconfigure(0, weight=1)

        profiler_box = ttk.LabelFrame(body, text="  Profiler  ", padding=14)
        profiler_box.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        profiler_box.columnconfigure(0, weight=1)

        logging_box = ttk.LabelFrame(body, text="  Logging  ", padding=14)
        logging_box.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
        logging_box.columnconfigure(0, weight=1)

        sampling_flags = [
            ("CODEX_DEBUG_COND", "Conditioning Debug", False),
            ("CODEX_LOG_SAMPLER", "Sampler Verbose Logs", False),
            ("CODEX_LOG_CFG_DELTA", "CFG Delta Logs (requires Sampler Verbose Logs)", False),
            ("CODEX_LOG_SIGMAS", "Sigma Ladder Logs", False),
            ("CODEX_PIPELINE_DEBUG", "Pipeline Debug", True),
            ("CODEX_DUMP_LATENTS", "Dump Latents", True),
            ("CODEX_TIMELINE", "Timeline Trace (TVA-style execution timeline)", True),
        ]
        row = 0
        for key, label, advanced in sampling_flags:
            row = self._add_flag_toggle(sampling_box, row, key=key, label=label, advanced=advanced)
        row = self._add_entry(
            sampling_box,
            row,
            label="CFG Delta Steps (N):",
            var=self._var_cfg_delta_n,
            width=10,
            on_change=lambda: self._set_text("CODEX_LOG_CFG_DELTA_N", self._var_cfg_delta_n.get()),
        )
        _ = self._add_choice(
            sampling_box,
            row,
            label="CFG Cond+Uncond Batch Mode:",
            var=self._var_cfg_batch_mode,
            choices=CFG_BATCH_MODE_CHOICES,
            on_change=lambda: self._set_text("CODEX_CFG_BATCH_MODE", self._var_cfg_batch_mode.get()),
            advanced=True,
        )

        trace_flags = [
            ("CODEX_TRACE_INFERENCE_DEBUG", "Trace Debug: Inference"),
            ("CODEX_TRACE_LOAD_PATCH_DEBUG", "Trace Debug: Load/Patch"),
            ("CODEX_TRACE_CALL_DEBUG", "Trace Debug: Call Trace"),
            ("CODEX_TRACE_CONTRACT", "Contract Trace (JSONL in logs/contract-trace)"),
            ("CODEX_TRACE_PROFILER", "Contract Profiler Toggle (maps to --trace-profiler)"),
        ]
        self._register_advanced(trace_box)
        row = 0
        for key, label in trace_flags:
            row = self._add_flag_toggle(trace_box, row, key=key, label=label, advanced=True)
        row = self._add_entry(
            trace_box,
            row,
            label="Call trace max / func (0=unlimited):",
            var=self._var_trace_max,
            width=10,
            on_change=lambda: self._set_text("CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC", self._var_trace_max.get()),
            advanced=True,
        )
        _ = self._add_entry(
            trace_box,
            row,
            label="Dump latents path:",
            var=self._var_dump_path,
            width=40,
            on_change=lambda: self._set_text("CODEX_DUMP_LATENTS_PATH", self._var_dump_path.get()),
            advanced=True,
        )

        profiler_flags = [
            ("CODEX_PROFILE", "Global Profiler (torch.profiler)"),
            ("CODEX_PROFILE_TRACE", "Profiler: export Perfetto trace"),
            ("CODEX_PROFILE_RECORD_SHAPES", "Profiler: record shapes"),
            ("CODEX_PROFILE_PROFILE_MEMORY", "Profiler: profile memory"),
            ("CODEX_PROFILE_WITH_STACK", "Profiler: include stacks (very heavy)"),
        ]
        self._register_advanced(profiler_box)
        row = 0
        for key, label in profiler_flags:
            row = self._add_flag_toggle(profiler_box, row, key=key, label=label, advanced=True)
        row = self._add_entry(
            profiler_box,
            row,
            label="Profiler top ops (N):",
            var=self._var_profile_top_n,
            width=10,
            on_change=lambda: self._set_text("CODEX_PROFILE_TOP_N", self._var_profile_top_n.get()),
            advanced=True,
        )
        _ = self._add_entry(
            profiler_box,
            row,
            label="Profiler max steps (0=all):",
            var=self._var_profile_max_steps,
            width=10,
            on_change=lambda: self._set_text("CODEX_PROFILE_MAX_STEPS", self._var_profile_max_steps.get()),
            advanced=True,
        )

        log_defaults = {
            "CODEX_LOG_DEBUG": False,
            "CODEX_LOG_INFO": True,
            "CODEX_LOG_WARNING": True,
            "CODEX_LOG_ERROR": True,
        }
        row = 0
        for key, label in (
            ("CODEX_LOG_DEBUG", "DEBUG (verbose)"),
            ("CODEX_LOG_INFO", "INFO"),
            ("CODEX_LOG_WARNING", "WARNING"),
            ("CODEX_LOG_ERROR", "ERROR"),
        ):
            var = tk.BooleanVar(value=BoolSetting(key, default=bool(log_defaults[key])).get(self._controller.store.env))
            self._log_levels[key] = var
            checkbox = ttk.Checkbutton(
                logging_box,
                text=label,
                variable=var,
                command=lambda k=key: self._set_bool(k, self._log_levels[k].get()),
                style="Toggle.TCheckbutton",
            )
            checkbox.grid(row=row, column=0, sticky="w", pady=2)
            row += 1

        self._var_log_file.set(bool(str(self._controller.store.env.get("CODEX_LOG_FILE", "") or "").strip()))
        log_file_toggle = ttk.Checkbutton(
            logging_box,
            text="Write to log file (logs/codex-*.log)",
            variable=self._var_log_file,
            command=self._toggle_log_file,
            style="Toggle.TCheckbutton",
        )
        log_file_toggle.grid(row=row, column=0, sticky="w", pady=(10, 2))

        self.reload()
        self.frame = frame
        return frame

    def reload(self) -> None:
        env = self._controller.store.env
        for key, var in self._debug_flags.items():
            var.set(BoolSetting(key, default=False).get(env))

        log_defaults = {
            "CODEX_LOG_DEBUG": False,
            "CODEX_LOG_INFO": True,
            "CODEX_LOG_WARNING": True,
            "CODEX_LOG_ERROR": True,
        }
        for key, var in self._log_levels.items():
            var.set(BoolSetting(key, default=bool(log_defaults.get(key, True))).get(env))

        self._var_cfg_delta_n.set(str(env.get("CODEX_LOG_CFG_DELTA_N", "2") or "2"))
        self._var_cfg_batch_mode.set(ChoiceSetting("CODEX_CFG_BATCH_MODE", default="fused", choices=CFG_BATCH_MODE_CHOICES).get(env))
        self._var_trace_max.set(str(env.get("CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC", TRACE_DEBUG_DEFAULT) or TRACE_DEBUG_DEFAULT))
        self._var_dump_path.set(str(env.get("CODEX_DUMP_LATENTS_PATH", "") or ""))
        self._var_profile_top_n.set(str(env.get("CODEX_PROFILE_TOP_N", "25") or "25"))
        self._var_profile_max_steps.set(str(env.get("CODEX_PROFILE_MAX_STEPS", "0") or "0"))
        self._var_log_file.set(bool(str(env.get("CODEX_LOG_FILE", "") or "").strip()))

        self._install_cfg_delta_guard()
        self._apply_advanced_visibility()

    def render_checks(self, checks: Iterable[CodexLaunchCheck]) -> None:
        tree = self._checks_tree
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)
        for chk in checks:
            tree.insert("", "end", values=(chk.name, "yes" if chk.ok else "no", chk.detail))

    def _add_flag_toggle(self, parent: ttk.LabelFrame, row: int, *, key: str, label: str, advanced: bool) -> int:
        var = tk.BooleanVar(value=BoolSetting(key, default=False).get(self._controller.store.env))
        self._debug_flags[key] = var
        checkbox = ttk.Checkbutton(
            parent,
            text=label,
            variable=var,
            command=lambda k=key: self._set_bool(k, self._debug_flags[k].get()),
            style="Toggle.TCheckbutton",
        )
        checkbox.grid(row=row, column=0, sticky="w", pady=2)
        if advanced:
            self._register_advanced(checkbox)
        return row + 1

    def _set_bool(self, key: str, enabled: bool) -> None:
        BoolSetting(key, default=False).set(self._controller.store.env, enabled)
        self._mark_changed()

    def _set_text(self, key: str, value: str) -> None:
        self._controller.store.env[key] = str(value).strip()
        self._mark_changed()

    def _add_entry(
        self,
        parent: ttk.LabelFrame,
        row: int,
        *,
        label: str,
        var: tk.StringVar,
        width: int,
        on_change: Callable[[], None],
        advanced: bool = False,
    ) -> int:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=8)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="w", padx=(10, 0), pady=8)
        entry.bind("<KeyRelease>", lambda _e: on_change())
        if advanced:
            self._register_advanced(label_widget, entry)
        return row + 1

    def _add_choice(
        self,
        parent: ttk.LabelFrame,
        row: int,
        *,
        label: str,
        var: tk.StringVar,
        choices: tuple[str, ...],
        on_change: Callable[[], None],
        advanced: bool = False,
    ) -> int:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=8)
        combo = ttk.Combobox(parent, textvariable=var, width=20, state="readonly")
        combo["values"] = list(choices)
        combo.grid(row=row, column=1, sticky="w", padx=(10, 0), pady=8)
        combo.bind("<<ComboboxSelected>>", lambda _e: on_change())
        if advanced:
            self._register_advanced(label_widget, combo)
        return row + 1

    def _register_advanced(self, *widgets: tk.Widget) -> None:
        self._advanced_widgets.extend(widgets)

    def set_advanced_visible(self, visible: bool) -> None:
        self._advanced_visible = bool(visible)
        self._apply_advanced_visibility()

    def _apply_advanced_visibility(self) -> None:
        visible = bool(self._advanced_visible)
        for widget in self._advanced_widgets:
            if visible:
                widget.grid()
            else:
                widget.grid_remove()

    def _clear_cfg_delta_guard(self) -> None:
        for variable, trace_id in self._cfg_delta_trace_ids:
            try:
                variable.trace_remove("write", trace_id)
            except Exception:
                continue
        self._cfg_delta_trace_ids.clear()

    def _install_cfg_delta_guard(self) -> None:
        if "CODEX_LOG_SAMPLER" not in self._debug_flags or "CODEX_LOG_CFG_DELTA" not in self._debug_flags:
            return
        self._clear_cfg_delta_guard()
        sampler_var = self._debug_flags["CODEX_LOG_SAMPLER"]
        delta_var = self._debug_flags["CODEX_LOG_CFG_DELTA"]
        guard = {"active": False}

        def _on_delta_changed(*_args: object) -> None:
            if guard["active"] or (not delta_var.get()):
                return
            if sampler_var.get():
                return
            guard["active"] = True
            try:
                sampler_var.set(True)
                BoolSetting("CODEX_LOG_SAMPLER", default=False).set(self._controller.store.env, True)
                self._mark_changed()
            finally:
                guard["active"] = False

        def _on_sampler_changed(*_args: object) -> None:
            if guard["active"] or sampler_var.get():
                return
            if not delta_var.get():
                return
            guard["active"] = True
            try:
                delta_var.set(False)
                BoolSetting("CODEX_LOG_CFG_DELTA", default=False).set(self._controller.store.env, False)
                self._mark_changed()
            finally:
                guard["active"] = False

        delta_trace_id = delta_var.trace_add("write", _on_delta_changed)
        sampler_trace_id = sampler_var.trace_add("write", _on_sampler_changed)
        self._cfg_delta_trace_ids.append((delta_var, delta_trace_id))
        self._cfg_delta_trace_ids.append((sampler_var, sampler_trace_id))
        _on_delta_changed()

    def _toggle_log_file(self) -> None:
        enabled = bool(self._var_log_file.get())
        env = self._controller.store.env
        if enabled:
            if not str(env.get("CODEX_LOG_FILE", "") or "").strip():
                logs_dir = self._controller.codex_root / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                env["CODEX_LOG_FILE"] = str(logs_dir / f"codex-{stamp}.log")
                self._mark_changed()
        else:
            try:
                del env["CODEX_LOG_FILE"]
            except KeyError:
                pass
            self._mark_changed()

    def validate_int_settings(self) -> None:
        env = self._controller.store.env
        try:
            IntSetting("CODEX_LOG_CFG_DELTA_N", default=2, minimum=1).get(env)
            IntSetting("CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC", default=int(TRACE_DEBUG_DEFAULT), minimum=0).get(env)
            IntSetting("CODEX_PROFILE_TOP_N", default=25, minimum=1, maximum=500).get(env)
            IntSetting("CODEX_PROFILE_MAX_STEPS", default=0, minimum=0, maximum=10_000).get(env)
        except SettingValidationError as exc:
            raise RuntimeError(str(exc)) from exc
