"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Docker-friendly terminal launcher (TUI) for Codex WebUI.
Provides an interactive terminal configuration flow for launcher runtime env keys (device/allocator/attention/runtime
knobs), consumes shared option metadata from the launcher setting registry, persists them via `LauncherProfileStore`, and starts `run-webui.sh` with profile values injected.

Symbols (top-level; keep in sync; no ghosts):
- `DockerTuiOption` (dataclass): Declarative description of one TUI-configurable runtime option.
- `_registry_option` (function): Builds a TUI option from a shared launcher setting descriptor.
- `main` (function): CLI entrypoint; optionally runs interactive configuration and then executes `run-webui.sh`.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

_codex_root_raw = str(os.environ.get("CODEX_ROOT", "") or "").strip()
if not _codex_root_raw:
    raise EnvironmentError("CODEX_ROOT not set. Launch via run-webui-docker.sh or set CODEX_ROOT explicitly.")

_codex_root = Path(_codex_root_raw)
if str(_codex_root) not in sys.path:
    sys.path.insert(0, str(_codex_root))

from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.infra.config.lora_merge_mode import LoraMergeMode
from apps.backend.infra.config.lora_refresh_signature import LoraRefreshSignatureMode
from apps.launcher.profile_meta import CODEX_CUDA_MALLOC_KEY, ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY
from apps.launcher.profile_store import LauncherProfileStore
from apps.launcher.setting_registry import SETTING_DESCRIPTORS_BY_KEY
from apps.launcher.settings import (
    normalize_attention_env,
    normalize_gguf_lora_env,
    normalize_task_runtime_env,
)


LOGGER = logging.getLogger("codex.docker_tui")

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_LORA_MERGE_MODE_CHOICES: tuple[str, ...] = tuple(mode.value for mode in LoraMergeMode)
_LORA_REFRESH_SIGNATURE_CHOICES: tuple[str, ...] = tuple(mode.value for mode in LoraRefreshSignatureMode)


@dataclass(frozen=True, slots=True)
class DockerTuiOption:
    key: str
    label: str
    kind: str
    choices: tuple[str, ...] = ()
    allow_blank: bool = False
    default: str = ""


def _registry_option(
    key: str,
    label: str,
    kind: str,
    *,
    allow_blank: bool = False,
) -> DockerTuiOption:
    descriptor = SETTING_DESCRIPTORS_BY_KEY[key]
    return DockerTuiOption(
        key,
        label,
        kind,
        choices=tuple(descriptor.choices),
        allow_blank=allow_blank,
        default=descriptor.default,
    )


_OPTIONS: tuple[DockerTuiOption, ...] = (
    _registry_option("CODEX_MAIN_DEVICE", "Main device", "choice"),
    _registry_option("CODEX_MOUNT_DEVICE", "Mount device", "choice"),
    _registry_option("CODEX_OFFLOAD_DEVICE", "Offload device", "choice"),
    _registry_option("CODEX_ATTENTION_BACKEND", "Attention backend", "choice"),
    _registry_option("CODEX_ATTENTION_SDPA_POLICY", "Attention SDPA policy", "choice"),
    _registry_option("CODEX_LORA_APPLY_MODE", "LoRA apply mode", "choice"),
    _registry_option("CODEX_LORA_ONLINE_MATH", "LoRA online math", "choice"),
    _registry_option("CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE", "WAN22 img2vid chunk buffer mode", "choice"),
    DockerTuiOption(
        "CODEX_LORA_MERGE_MODE",
        "LoRA merge mode",
        "choice",
        choices=_LORA_MERGE_MODE_CHOICES,
        default="fast",
    ),
    DockerTuiOption(
        "CODEX_LORA_REFRESH_SIGNATURE",
        "LoRA refresh signature",
        "choice",
        choices=_LORA_REFRESH_SIGNATURE_CHOICES,
        default="content_sha256",
    ),
    _registry_option(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY, "Enable default PYTORCH_CUDA_ALLOC_CONF", "bool"),
    _registry_option("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "text", allow_blank=True),
    _registry_option(CODEX_CUDA_MALLOC_KEY, "Enable --cuda-malloc", "bool"),
    DockerTuiOption("API_PORT_OVERRIDE", "API port override", "port", allow_blank=True),
    DockerTuiOption("WEB_PORT", "Web port", "port", allow_blank=True),
)


def _truthy(raw: object) -> bool:
    return str(raw or "").strip().lower() in _TRUE_VALUES


def _normalize_bool_token(raw: str, *, key: str) -> str:
    value = str(raw or "").strip().lower()
    if not value:
        return "0"
    if value in _TRUE_VALUES:
        return "1"
    if value in _FALSE_VALUES:
        return "0"
    raise ValueError(f"{key} must be boolean (accepted: 1/0/true/false/yes/no/on/off).")


def _normalize_choice(raw: str, *, key: str, choices: Sequence[str]) -> str:
    value = str(raw or "").strip().lower()
    if value not in choices:
        allowed = ", ".join(choices)
        raise ValueError(f"{key} must be one of: {allowed} (got {raw!r}).")
    return value


def _normalize_port(raw: str, *, key: str, allow_blank: bool) -> str:
    value = str(raw or "").strip()
    if not value:
        if allow_blank:
            return ""
        raise ValueError(f"{key} requires a value.")
    if not value.isdigit():
        raise ValueError(f"{key} must be an integer in 1..65535 (got {raw!r}).")
    port = int(value)
    if port < 1 or port > 65535:
        raise ValueError(f"{key} must be in 1..65535 (got {port}).")
    return str(port)


def _parse_pytorch_cuda_alloc_conf(raw_conf: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for token in str(raw_conf or "").split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid PYTORCH_CUDA_ALLOC_CONF entry {token!r}: expected 'key:value'.")
        key, value = token.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                f"Invalid PYTORCH_CUDA_ALLOC_CONF entry {token!r}: expected non-empty 'key:value'.",
            )
        entries.append((key, value))
    return entries


def _normalize_allocator_env(core_env: dict[str, str]) -> None:
    toggle_default = SETTING_DESCRIPTORS_BY_KEY[ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY].default
    toggle = _normalize_bool_token(
        core_env.get(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY, toggle_default),
        key=ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
    )
    core_env[ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY] = toggle

    cuda_malloc_default = SETTING_DESCRIPTORS_BY_KEY[CODEX_CUDA_MALLOC_KEY].default
    cuda_malloc = _normalize_bool_token(core_env.get(CODEX_CUDA_MALLOC_KEY, cuda_malloc_default), key=CODEX_CUDA_MALLOC_KEY)
    core_env[CODEX_CUDA_MALLOC_KEY] = cuda_malloc

    raw_conf = str(core_env.get("PYTORCH_CUDA_ALLOC_CONF", "") or "").strip()
    if not raw_conf:
        # Keep this key absent in persisted profile when unset; launcher store/build will inject
        # the default allocator value when enabled and unset.
        core_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        return

    entries = _parse_pytorch_cuda_alloc_conf(raw_conf)
    backend_idx: int | None = None
    for index, (key, _value) in enumerate(entries):
        if key.strip().lower() == "backend":
            if backend_idx is not None:
                raise ValueError("PYTORCH_CUDA_ALLOC_CONF cannot contain multiple backend entries.")
            backend_idx = index

    if cuda_malloc == "1":
        target_backend = "cudamallocasync"
        if backend_idx is None:
            entries.append(("backend", "cudaMallocAsync"))
        else:
            backend = str(entries[backend_idx][1]).replace(" ", "").lower()
            if backend != target_backend:
                raise ValueError(
                    "CODEX_CUDA_MALLOC=1 requires PYTORCH_CUDA_ALLOC_CONF backend:cudaMallocAsync.",
                )

    core_env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(f"{key}:{value}" for key, value in entries)


def _normalize_core_env(core_env: dict[str, str]) -> None:
    for option in _OPTIONS:
        raw_value = str(core_env.get(option.key, option.default) or "").strip()
        if not raw_value and option.default:
            raw_value = option.default
        if option.kind == "choice":
            core_env[option.key] = _normalize_choice(raw_value, key=option.key, choices=option.choices)
            continue
        if option.kind == "bool":
            core_env[option.key] = _normalize_bool_token(raw_value, key=option.key)
            continue
        if option.kind == "port":
            normalized_port = _normalize_port(raw_value, key=option.key, allow_blank=option.allow_blank)
            if normalized_port:
                core_env[option.key] = normalized_port
            else:
                core_env.pop(option.key, None)
            continue
        if option.kind == "text":
            if raw_value:
                core_env[option.key] = raw_value
            else:
                core_env.pop(option.key, None)
            continue
        raise ValueError(f"Unknown option kind {option.kind!r} for {option.key}.")

    normalize_attention_env(core_env)
    normalize_gguf_lora_env(core_env)
    normalize_task_runtime_env(core_env)
    _normalize_allocator_env(core_env)


def _render_summary_table(console: Console, env_map: Mapping[str, str]) -> None:
    table = Table(title="Codex Docker Launcher Profile", show_lines=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    for option in _OPTIONS:
        value = str(env_map.get(option.key, "") or "").strip()
        table.add_row(option.key, value or "<empty>")
    console.print(table)


def _prompt_choice(console: Console, *, label: str, key: str, current: str, choices: Sequence[str]) -> str:
    allowed = "/".join(choices)
    while True:
        response = Prompt.ask(f"{label} ({allowed})", default=current)
        try:
            return _normalize_choice(response, key=key, choices=choices)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")


def _prompt_bool(console: Console, *, label: str, current: str) -> str:
    default_value = _truthy(current)
    return "1" if Confirm.ask(label, default=default_value) else "0"


def _prompt_text(console: Console, *, label: str, current: str, allow_blank: bool) -> str:
    hint = "Use '-' to clear."
    response = Prompt.ask(f"{label} ({hint})", default=current or "")
    trimmed = str(response).strip()
    if trimmed == "-":
        return ""
    if not trimmed and not allow_blank:
        return current
    return trimmed


def _prompt_port(console: Console, *, label: str, key: str, current: str, allow_blank: bool) -> str:
    while True:
        response = Prompt.ask(f"{label} (1..65535, '-' to clear)", default=current or "")
        response = str(response).strip()
        if response == "-":
            response = ""
        try:
            return _normalize_port(response, key=key, allow_blank=allow_blank)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")


def _configure_interactively(console: Console, store: LauncherProfileStore) -> None:
    core_env = store.areas.setdefault("core", {})
    _normalize_core_env(core_env)

    console.print("[bold cyan]Codex Docker TUI[/bold cyan]")
    console.print(
        "Configure launcher runtime env keys once and persist them under `.sangoi/launcher/`.",
    )
    _render_summary_table(console, core_env)

    if not Confirm.ask("Edit settings now?", default=True):
        return

    for option in _OPTIONS:
        current = str(core_env.get(option.key, "") or "").strip()
        if option.kind == "choice":
            core_env[option.key] = _prompt_choice(
                console,
                label=option.label,
                key=option.key,
                current=current,
                choices=option.choices,
            )
            continue
        if option.kind == "bool":
            core_env[option.key] = _prompt_bool(console, label=option.label, current=current)
            continue
        if option.kind == "text":
            core_env[option.key] = _prompt_text(
                console,
                label=option.label,
                current=current,
                allow_blank=option.allow_blank,
            )
            continue
        if option.kind == "port":
            core_env[option.key] = _prompt_port(
                console,
                label=option.label,
                key=option.key,
                current=current,
                allow_blank=option.allow_blank,
            )
            continue
        raise RuntimeError(f"Unsupported option kind {option.kind!r}.")

    _normalize_core_env(core_env)
    store.save()
    console.print("[green]Launcher profile saved.[/green]")
    _render_summary_table(console, core_env)


def _build_runtime_env(store: LauncherProfileStore) -> dict[str, str]:
    runtime_env = os.environ.copy()
    profile_env = store.build_env()
    for key, value in profile_env.items():
        runtime_env.setdefault(key, value)

    codex_root = str(get_repo_root())
    runtime_env["CODEX_ROOT"] = codex_root
    existing_pythonpath = str(runtime_env.get("PYTHONPATH", "") or "").strip()
    pythonpath_parts = [part for part in existing_pythonpath.split(":") if part] if existing_pythonpath else []
    if codex_root not in pythonpath_parts:
        pythonpath_parts.insert(0, codex_root)
    runtime_env["PYTHONPATH"] = ":".join(pythonpath_parts)
    runtime_env.setdefault("PYTHONUNBUFFERED", "1")
    runtime_env.setdefault("FORCE_COLOR", "1")
    return runtime_env


def _should_run_tui(args: argparse.Namespace) -> bool:
    interactive_session = sys.stdin.isatty() and sys.stdout.isatty()
    env_toggle = str(os.environ.get("CODEX_DOCKER_TUI", "1") or "").strip().lower()
    env_enabled = env_toggle in {"", "1", "true", "yes", "on"}

    if args.no_tui:
        return False
    if args.tui:
        if not interactive_session:
            raise RuntimeError("--tui requires an interactive TTY (stdin/stdout).")
        return True
    if args.non_interactive:
        return False
    if not interactive_session:
        return False
    return env_enabled


def _parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="docker_tui_launcher.py",
        description="Docker-friendly terminal launcher for Codex WebUI.",
    )
    parser.add_argument("--tui", action="store_true", help="Force interactive terminal configuration.")
    parser.add_argument("--no-tui", action="store_true", help="Skip interactive configuration and run directly.")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive prompts (for headless/container automation).",
    )
    parser.add_argument(
        "--configure-only",
        action="store_true",
        help="Run configuration flow and exit without starting API/UI.",
    )
    args, forward_args = parser.parse_known_args(argv)
    if args.tui and args.no_tui:
        parser.error("--tui and --no-tui are mutually exclusive.")
    return args, forward_args


def _run_webui(forward_args: Iterable[str], runtime_env: Mapping[str, str]) -> int:
    codex_root = get_repo_root()
    entrypoint = codex_root / "run-webui.sh"
    if not entrypoint.exists():
        raise FileNotFoundError(f"Missing run entrypoint: {entrypoint}")
    if not os.access(entrypoint, os.X_OK):
        raise PermissionError(f"Entrypoint is not executable: {entrypoint}")

    command = [str(entrypoint), *list(forward_args)]
    LOGGER.info("Executing %s", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=str(codex_root),
        env=dict(runtime_env),
        check=False,
    )
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    args, forward_args = _parse_args(list(argv or sys.argv[1:]))
    console = Console()
    store = LauncherProfileStore.load()
    core_env = store.areas.setdefault("core", {})
    previous_env = dict(core_env)
    _normalize_core_env(core_env)
    if core_env != previous_env:
        store.save()

    if _should_run_tui(args):
        _configure_interactively(console, store)

    if args.configure_only:
        return 0

    runtime_env = _build_runtime_env(store)
    return _run_webui(forward_args, runtime_env)


if __name__ == "__main__":
    raise SystemExit(main())
