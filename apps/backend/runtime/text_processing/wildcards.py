"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Repo-owned wildcard expansion helpers for prompt automation.
Expands `__token__` prompt wildcards from repo-fenced text files under a wildcard root, choosing one non-empty line per token and failing loud on missing,
empty, circular, or recursion-overflow expansions.

Symbols (top-level; keep in sync; no ghosts):
- `DEFAULT_WILDCARD_DIR` (constant): Default repo-relative wildcard directory under `CODEX_ROOT`.
- `default_wildcard_root` (function): Returns the default absolute wildcard root under the repository.
- `expand_wildcards` (function): Expands `__token__` placeholders from wildcard text files under a repo-fenced root.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from apps.backend.infra.config.repo_root import get_repo_root

DEFAULT_WILDCARD_DIR = Path("input") / "wildcards"
_TOKEN_RE = re.compile(r"__([A-Za-z0-9][A-Za-z0-9_./-]*)__")
_DEFAULT_MAX_DEPTH = 12


def default_wildcard_root() -> Path:
    return get_repo_root() / DEFAULT_WILDCARD_DIR


def _normalize_token(token: str) -> Path:
    candidate = str(token or "").strip().replace("\\", "/").strip("/")
    if not candidate:
        raise ValueError("Wildcard token cannot be empty.")
    token_path = Path(candidate)
    if token_path.is_absolute():
        raise ValueError(f"Wildcard token {token!r} must be relative.")
    normalized_parts = []
    for part in token_path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError(f"Wildcard token {token!r} must not traverse parent directories.")
        normalized_parts.append(part)
    if not normalized_parts:
        raise ValueError(f"Wildcard token {token!r} is invalid.")
    return Path(*normalized_parts)


def _resolve_token_file(*, root: Path, token: str) -> Path:
    candidate = (root / _normalize_token(token)).with_suffix(".txt")
    resolved = candidate.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except Exception as exc:
        raise ValueError(
            f"Wildcard token {token!r} resolves outside wildcard root: {resolved}",
        ) from exc
    return resolved


def _read_token_options(path: Path, *, token: str) -> list[str]:
    if not path.is_file():
        raise ValueError(f"Missing wildcard token {token!r} at {path}.")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Wildcard token {token!r} is empty at {path}.")
    return lines


def expand_wildcards(
    text: str,
    *,
    wildcard_root: str | Path,
    rng: random.Random | None = None,
    max_depth: int = _DEFAULT_MAX_DEPTH,
) -> str:
    source_text = str(text or "")
    if "__" not in source_text:
        return source_text
    if max_depth < 1:
        raise ValueError(f"Wildcard max_depth must be >= 1 (got {max_depth}).")

    root = Path(wildcard_root).expanduser()
    generator = rng if rng is not None else random.Random()

    def _expand(fragment: str, *, depth: int, stack: tuple[str, ...]) -> str:
        if depth > max_depth:
            raise ValueError(
                "Wildcard expansion exceeded recursion limit "
                f"({max_depth}) while expanding {' -> '.join(stack) if stack else '<root>'}."
            )

        def _replace(match: re.Match[str]) -> str:
            token = match.group(1)
            if token in stack:
                chain = " -> ".join((*stack, token))
                raise ValueError(f"Circular wildcard expansion detected: {chain}.")
            token_file = _resolve_token_file(root=root, token=token)
            options = _read_token_options(token_file, token=token)
            choice = generator.choice(options)
            return _expand(choice, depth=depth + 1, stack=(*stack, token))

        return _TOKEN_RE.sub(_replace, fragment)

    return _expand(source_text, depth=0, stack=tuple())
