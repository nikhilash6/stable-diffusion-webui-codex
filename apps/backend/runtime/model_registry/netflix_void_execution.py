"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Truthful Netflix VOID checkpoint classification and pairing metadata.
Classifies discoverable Netflix VOID overlay checkpoints from explicit local filename rules,
enforces the tranche-A literal sibling pairing contract (`void_pass1.safetensors` +
`void_pass2.safetensors` in the same directory), and emits the namespaced checkpoint metadata
forwarded by `/api/models` without introducing a second detector/alias lane.

Symbols (top-level; keep in sync; no ghosts):
- `NETFLIX_VOID_KIND_PASS1` (constant): Classified public Pass 1 checkpoint kind.
- `NETFLIX_VOID_KIND_PASS2` (constant): Classified follow-on Pass 2 checkpoint kind.
- `NETFLIX_VOID_KIND_UNKNOWN` (constant): Unclassified or unsupported checkpoint kind.
- `NETFLIX_VOID_METADATA_KIND_KEY` (constant): `/api/models` metadata key for checkpoint kind.
- `NETFLIX_VOID_METADATA_PAIR_READY_KEY` (constant): `/api/models` metadata key for literal sibling-pair readiness.
- `NetflixVoidCheckpointClassification` (dataclass): Internal classification result for one discoverable overlay checkpoint.
- `classify_netflix_void_checkpoint` (function): Apply the literal filename/sibling-pair contract to one record.
- `build_netflix_void_checkpoint_metadata` (function): Build namespaced Netflix VOID metadata forwarded by `/api/models`.
- `netflix_void_checkpoint_kind` (function): Return the classified checkpoint kind for one record.
- `netflix_void_pair_ready` (function): Return whether the record has the required literal sibling partner.
- `netflix_void_record_is_publicly_selectable` (function): Return whether the record is a public Pass 1 selector candidate.
- `netflix_void_record_is_publicly_runnable` (function): Return whether the record is a public Pass 1 selector with a ready sibling pair.
- `resolve_netflix_void_pass2_partner` (function): Resolve the literal sibling Pass 2 overlay path for one public Pass 1 record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apps.backend.runtime.models.types import CheckpointFormat, CheckpointRecord

NETFLIX_VOID_KIND_PASS1 = "pass1"
NETFLIX_VOID_KIND_PASS2 = "pass2"
NETFLIX_VOID_KIND_UNKNOWN = "unknown"

NETFLIX_VOID_METADATA_KIND_KEY = "netflix_void_checkpoint_kind"
NETFLIX_VOID_METADATA_PAIR_READY_KEY = "netflix_void_pair_ready"

_PASS1_FILENAME = "void_pass1.safetensors"
_PASS2_FILENAME = "void_pass2.safetensors"


@dataclass(frozen=True)
class NetflixVoidCheckpointClassification:
    checkpoint_kind: str
    pair_ready: bool
    partner_path: str | None


def _normalized_checkpoint_format(record: CheckpointRecord) -> str:
    raw_format = record.format
    if isinstance(raw_format, CheckpointFormat):
        return raw_format.value
    return str(raw_format or "").strip().lower()


def _normalized_filename(record: CheckpointRecord) -> str:
    return Path(str(record.filename or record.path or "")).name.strip().lower()


def _build_classification(
    *,
    checkpoint_kind: str,
    partner_path: Path | None,
) -> NetflixVoidCheckpointClassification:
    partner = str(partner_path) if partner_path is not None and partner_path.is_file() else None
    return NetflixVoidCheckpointClassification(
        checkpoint_kind=checkpoint_kind,
        pair_ready=partner is not None,
        partner_path=partner,
    )


def classify_netflix_void_checkpoint(record: CheckpointRecord) -> NetflixVoidCheckpointClassification:
    if _normalized_checkpoint_format(record) != CheckpointFormat.CHECKPOINT.value:
        return NetflixVoidCheckpointClassification(
            checkpoint_kind=NETFLIX_VOID_KIND_UNKNOWN,
            pair_ready=False,
            partner_path=None,
        )

    filename = _normalized_filename(record)
    if filename == _PASS1_FILENAME:
        return _build_classification(
            checkpoint_kind=NETFLIX_VOID_KIND_PASS1,
            partner_path=Path(record.directory) / _PASS2_FILENAME,
        )
    if filename == _PASS2_FILENAME:
        return _build_classification(
            checkpoint_kind=NETFLIX_VOID_KIND_PASS2,
            partner_path=Path(record.directory) / _PASS1_FILENAME,
        )
    return NetflixVoidCheckpointClassification(
        checkpoint_kind=NETFLIX_VOID_KIND_UNKNOWN,
        pair_ready=False,
        partner_path=None,
    )


def build_netflix_void_checkpoint_metadata(record: CheckpointRecord) -> dict[str, object]:
    classification = classify_netflix_void_checkpoint(record)
    return {
        NETFLIX_VOID_METADATA_KIND_KEY: classification.checkpoint_kind,
        NETFLIX_VOID_METADATA_PAIR_READY_KEY: classification.pair_ready,
    }


def netflix_void_checkpoint_kind(record: CheckpointRecord) -> str:
    return classify_netflix_void_checkpoint(record).checkpoint_kind


def netflix_void_pair_ready(record: CheckpointRecord) -> bool:
    return classify_netflix_void_checkpoint(record).pair_ready


def netflix_void_record_is_publicly_selectable(record: CheckpointRecord) -> bool:
    return classify_netflix_void_checkpoint(record).checkpoint_kind == NETFLIX_VOID_KIND_PASS1


def netflix_void_record_is_publicly_runnable(record: CheckpointRecord) -> bool:
    classification = classify_netflix_void_checkpoint(record)
    return classification.checkpoint_kind == NETFLIX_VOID_KIND_PASS1 and classification.pair_ready


def resolve_netflix_void_pass2_partner(record: CheckpointRecord) -> str:
    classification = classify_netflix_void_checkpoint(record)
    if classification.checkpoint_kind != NETFLIX_VOID_KIND_PASS1:
        raise RuntimeError("Netflix VOID Pass 2 partner resolution requires a literal Pass 1 checkpoint selector.")
    if not classification.partner_path:
        raise RuntimeError(
            f"Netflix VOID Pass 1 checkpoint {record.filename!r} is missing literal sibling {_PASS2_FILENAME!r}."
        )
    return classification.partner_path
