"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher service port parsing and fallback resolution.
Owns API port candidate chains and local IPv4/IPv6 bind checks used before spawning launcher-managed backend services.

Symbols (top-level; keep in sync; no ghosts):
- `extract_cli_port` (function): Extracts a `--port` value from a command list.
- `parse_port_like_value` (function): Parses integer-like port values with bounds checks.
- `api_port_candidate_chain` (function): Builds the launcher API fallback port chain.
- `resolve_api_runtime_port` (function): Resolves an available API port and whether fallback was used.
- `port_free_everywhere` (function): Validates bindability on common IPv4/IPv6 local hosts.
"""

from __future__ import annotations

from contextlib import closing
import errno
import socket
from typing import List, Mapping


def extract_cli_port(command: List[str]) -> int | None:
    for idx, token in enumerate(command):
        if token == "--port" and idx + 1 < len(command):
            try:
                return int(command[idx + 1])
            except Exception:
                return None
        if token.startswith("--port="):
            try:
                return int(token.split("=", 1)[1])
            except Exception:
                return None
    return None


def parse_port_like_value(raw_value: object) -> int | None:
    try:
        parsed = int(str(raw_value or "").strip())
    except (TypeError, ValueError):
        return None
    if parsed < 1 or parsed > 65535:
        return None
    return int(parsed)


def api_port_candidate_chain(base_port: int) -> tuple[int, int, int]:
    normalized_base = int(base_port)
    return (
        normalized_base,
        normalized_base + 10000,
        normalized_base + 20000,
    )


def resolve_api_runtime_port(*, command: List[str], env: Mapping[str, str]) -> tuple[int, int, bool]:
    requested_port = extract_cli_port(command)
    if requested_port is None:
        requested_port = parse_port_like_value(env.get("API_PORT_OVERRIDE") or env.get("API_PORT"))
    if requested_port is None:
        requested_port = 7850
    blocked_details: list[str] = []
    for index, candidate in enumerate(api_port_candidate_chain(requested_port)):
        if candidate < 1 or candidate > 65535:
            blocked_details.append(f"{candidate} (out_of_range)")
            continue
        ok, blocked = port_free_everywhere(candidate)
        if ok:
            return int(candidate), int(requested_port), bool(index != 0)
        blocked_details.append(f"{candidate} ({blocked or 'busy'})")
    blocked_summary = ", ".join(blocked_details)
    raise RuntimeError(
        "No free API port in launcher fallback chain. "
        f"Tried: {blocked_summary}. "
        "You may already have Codex running (WSL/Windows) or another service bound on IPv4/IPv6 localhost. "
        "Stop the other instance or set API_PORT_OVERRIDE/WEB_PORT to a free pair."
    )


def port_free_everywhere(port: int) -> tuple[bool, str]:
    def _can_bind(family: int, host: str) -> tuple[bool, str]:
        try:
            with closing(socket.socket(family, socket.SOCK_STREAM)) as socket_handle:
                socket_handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if family == socket.AF_INET6:
                    socket_handle.bind((host, port, 0, 0))
                else:
                    socket_handle.bind((host, port))
                return True, ""
        except OSError as exc:
            if getattr(exc, "errno", None) in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
                return True, ""
            code = getattr(exc, "errno", None)
            return False, f"host={host} errno={code}"

    for family, host in (
        (socket.AF_INET, "0.0.0.0"),
        (socket.AF_INET, "127.0.0.1"),
        (socket.AF_INET6, "::"),
        (socket.AF_INET6, "::1"),
    ):
        ok, detail = _can_bind(family, host)
        if not ok:
            return False, detail
    return True, ""
