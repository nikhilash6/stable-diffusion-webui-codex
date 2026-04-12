/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared helpers for generation stale-settings conflict handling.
Normalizes backend conflict errors (`409`) into a revision value and produces user-facing retry guidance for manual reruns.

Symbols (top-level; keep in sync; no ghosts):
- `resolveSettingsRevisionConflict` (function): Returns `current_revision` for stale-settings conflicts (`409`), else `null`.
- `formatSettingsRevisionConflictMessage` (function): Builds actionable user message for stale-settings conflicts.
*/

import { getApiErrorStatus, getCurrentRevisionFromError } from '../api/client'

export function resolveSettingsRevisionConflict(error: unknown): number | null {
  if (getApiErrorStatus(error) !== 409) return null
  return getCurrentRevisionFromError(error)
}

export function formatSettingsRevisionConflictMessage(
  currentRevision: number,
  retryInstruction = 'retry generation manually',
): string {
  const revision = Math.max(0, Math.trunc(Number(currentRevision)))
  return `Settings changed on the backend (revision ${revision}). The UI synced the latest revision; ${retryInstruction}.`
}
