"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime-owned load authority permits and guards.
Defines a typed permit contract (`stage` + `owner`) used by coordinator/runtime seams to authorize
load/materialization-sensitive operations. Violations fail loud with explicit `LOAD_AUTHORITY_VIOLATION`
errors (no silent fallback).

Symbols (top-level; keep in sync; no ghosts):
- `LoadAuthorityStage` (enum): Contract stages for coordinator-owned load/materialization lifecycles.
- `LoadAuthorityPermit` (dataclass): Active permit payload (`owner`, `stage`, `acknowledged`) used by guard checks.
- `LoadAuthorityViolationError` (class): Fail-loud violation raised when load authority checks fail.
- `current_load_permit` (function): Returns the currently active permit (or `None`).
- `coordinator_load_permit` (context manager): Installs an active permit for coordinator-owned execution scopes.
- `require_load_authority` (function): Guard helper that validates active permit ownership/ack/stage.
- `guarded_load_entrypoint` (decorator): Wrapper for risky load entrypoints requiring an active permit.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Callable, Iterator, Sequence, TypeVar


class LoadAuthorityStage(str, Enum):
    """Coordinator/runtime stage identifiers for load authority enforcement."""

    LOAD = "load"
    MATERIALIZE = "materialize"
    UNLOAD = "unload"
    RELOAD = "reload"
    CLEANUP = "cleanup"


@dataclass(frozen=True, slots=True)
class LoadAuthorityPermit:
    """Typed runtime load permit (`stage` + `owner`)."""

    owner: str
    stage: LoadAuthorityStage
    acknowledged: bool = True


class LoadAuthorityViolationError(RuntimeError):
    """Raised when a guarded load/materialization entrypoint lacks authority."""

    code = "LOAD_AUTHORITY_VIOLATION"

    def __init__(
        self,
        *,
        action: str,
        reason: str,
        permit: LoadAuthorityPermit | None,
    ) -> None:
        action_text = str(action or "").strip() or "unknown_action"
        reason_text = str(reason or "").strip() or "unspecified"
        if permit is None:
            permit_desc = "none"
        else:
            permit_desc = (
                f"owner={permit.owner!r} stage={permit.stage.value!r} acknowledged={bool(permit.acknowledged)!r}"
            )
        super().__init__(
            f"{self.code}: action={action_text!r} reason={reason_text!r} active_permit={permit_desc}"
        )


_ACTIVE_LOAD_PERMIT: ContextVar[LoadAuthorityPermit | None] = ContextVar(
    "codex_active_load_permit",
    default=None,
)


def current_load_permit() -> LoadAuthorityPermit | None:
    """Return the currently active coordinator load permit (if any)."""

    return _ACTIVE_LOAD_PERMIT.get()


@contextmanager
def coordinator_load_permit(
    *,
    owner: str,
    stage: LoadAuthorityStage,
    acknowledged: bool = True,
) -> Iterator[LoadAuthorityPermit]:
    """Install a runtime-owned load permit for the active context."""

    owner_text = str(owner or "").strip()
    if not owner_text:
        raise ValueError("coordinator_load_permit requires a non-empty owner.")
    permit = LoadAuthorityPermit(
        owner=owner_text,
        stage=LoadAuthorityStage(stage),
        acknowledged=bool(acknowledged),
    )
    token = _ACTIVE_LOAD_PERMIT.set(permit)
    try:
        yield permit
    finally:
        _ACTIVE_LOAD_PERMIT.reset(token)


def require_load_authority(
    action: str,
    *,
    allowed_stages: Sequence[LoadAuthorityStage] | None = None,
    require_acknowledged: bool = True,
) -> LoadAuthorityPermit:
    """Validate active load authority for a risky entrypoint."""

    permit = current_load_permit()
    if permit is None:
        raise LoadAuthorityViolationError(
            action=action,
            reason="missing_active_permit",
            permit=None,
        )
    if require_acknowledged and not permit.acknowledged:
        raise LoadAuthorityViolationError(
            action=action,
            reason="permit_not_acknowledged",
            permit=permit,
        )

    if allowed_stages is not None:
        allowed_values = tuple(LoadAuthorityStage(stage) for stage in allowed_stages)
        if permit.stage not in allowed_values:
            allowed_text = ",".join(stage.value for stage in allowed_values)
            raise LoadAuthorityViolationError(
                action=action,
                reason=f"stage_not_allowed(active={permit.stage.value}, allowed={allowed_text})",
                permit=permit,
            )
    return permit


_CallableT = TypeVar("_CallableT", bound=Callable[..., object])


def guarded_load_entrypoint(
    *,
    action: str,
    allowed_stages: Sequence[LoadAuthorityStage] | None = None,
    require_acknowledged: bool = True,
) -> Callable[[_CallableT], _CallableT]:
    """Decorator enforcing load authority before executing a risky entrypoint."""

    def _decorate(func: _CallableT) -> _CallableT:
        @wraps(func)
        def _wrapped(*args: object, **kwargs: object) -> object:
            require_load_authority(
                action=action,
                allowed_stages=allowed_stages,
                require_acknowledged=require_acknowledged,
            )
            return func(*args, **kwargs)

        return _wrapped  # type: ignore[return-value]

    return _decorate


__all__ = [
    "LoadAuthorityPermit",
    "LoadAuthorityStage",
    "LoadAuthorityViolationError",
    "coordinator_load_permit",
    "current_load_permit",
    "guarded_load_entrypoint",
    "require_load_authority",
]
