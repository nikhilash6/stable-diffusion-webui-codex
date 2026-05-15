/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared frontend helpers for advanced guidance capability/toggle handling.
Provides small pure helpers for checking per-engine advanced-guidance support and building the canonical Basic Parameters
advanced-guidance toggle patch without duplicating capability probes across cards.

Symbols (top-level; keep in sync; no ghosts):
- `hasGuidanceSupport` (function): Returns whether one advanced-guidance control is supported by the current capability map.
- `hasAnyGuidanceSupport` (function): Returns whether any advanced-guidance control is supported by the current capability map.
- `buildGuidanceAdvancedTogglePatch` (function): Builds the canonical Basic Parameters advanced-guidance toggle patch.
*/

import type { GuidanceAdvancedCapabilities } from '../api/types'
import type { GuidanceAdvancedParams } from '../stores/model_tabs'

export function hasGuidanceSupport(
  support: GuidanceAdvancedCapabilities | null | undefined,
  control: keyof GuidanceAdvancedCapabilities,
): boolean {
  return Boolean(support?.[control])
}

export function hasAnyGuidanceSupport(support: GuidanceAdvancedCapabilities | null | undefined): boolean {
  if (!support) return false
  return Object.values(support).some((flag) => flag === true)
}

export function buildGuidanceAdvancedTogglePatch(
  nextEnabled: boolean,
  support: GuidanceAdvancedCapabilities | null | undefined,
): Partial<GuidanceAdvancedParams> {
  const patch: Partial<GuidanceAdvancedParams> = { enabled: nextEnabled }
  if (hasGuidanceSupport(support, 'apg_enabled')) patch.apgEnabled = nextEnabled
  if (hasGuidanceSupport(support, 'cfg_trunc_ratio')) patch.cfgTruncEnabled = nextEnabled
  return patch
}
