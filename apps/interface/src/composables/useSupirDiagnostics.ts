/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared SUPIR diagnostics/readiness owner for SDXL img2img/inpaint UI surfaces.
Caches the diagnostics-only `/api/supir/models` payload once, exposes readonly reactive fetch state,
supports explicit cache invalidation for Refresh, and derives one truthful selection/blocking contract
for both `QuickSettingsBar.vue` and `ImageModelTab.vue` without duplicating sampler metadata or stale-selection logic.

Symbols (top-level; keep in sync; no ghosts):
- `SupirVariantChoice` (interface): UI-ready SUPIR variant row with install status.
- `SupirSelectionState` (interface): Derived SUPIR selection/blocking contract for the current tab state.
- `ensureSupirDiagnosticsLoaded` (function): Lazily loads the shared SUPIR diagnostics payload once.
- `reloadSupirDiagnostics` (function): Invalidates and reloads the shared SUPIR diagnostics payload after an explicit refresh.
- `resolveSupirSelectionState` (function): Derives variants, samplers, selected rows, and blocking reason from shared diagnostics + current tab state.
- `useSupirDiagnostics` (function): Returns readonly shared SUPIR diagnostics refs plus the lazy loader.
*/

import { readonly, ref } from 'vue'

import { fetchSupirModels, invalidateSupirModelsCache } from '../api/client'
import type { SupirModelsResponse, SupirSamplerInfo } from '../api/types'

export interface SupirVariantChoice {
  value: string
  label: string
  available: boolean
}

export interface SupirSelectionState {
  variantChoices: SupirVariantChoice[]
  samplerChoices: SupirSamplerInfo[]
  availableVariants: SupirVariantChoice[]
  selectedVariantInstalled: boolean
  selectedSamplerAvailable: boolean
  selectionValid: boolean
  selectedSamplerInfo: SupirSamplerInfo | null
  blockingReason: string
}

const supirDiagnostics = ref<SupirModelsResponse | null>(null)
const supirDiagnosticsLoading = ref(false)
const supirDiagnosticsError = ref('')
const supirDiagnosticsAttempted = ref(false)
let supirDiagnosticsReloadQueued = false

async function loadSupirDiagnostics(force: boolean): Promise<void> {
  if (supirDiagnosticsLoading.value) {
    if (force) supirDiagnosticsReloadQueued = true
    return
  }
  if (!force && (supirDiagnostics.value || supirDiagnosticsAttempted.value)) return
  supirDiagnosticsAttempted.value = true
  supirDiagnosticsLoading.value = true
  supirDiagnosticsError.value = ''
  try {
    supirDiagnostics.value = await fetchSupirModels()
  } catch (error) {
    supirDiagnosticsError.value = error instanceof Error ? error.message : String(error)
  } finally {
    supirDiagnosticsLoading.value = false
  }
  if (supirDiagnosticsReloadQueued) {
    supirDiagnosticsReloadQueued = false
    invalidateSupirModelsCache()
    supirDiagnostics.value = null
    supirDiagnosticsError.value = ''
    supirDiagnosticsAttempted.value = false
    await loadSupirDiagnostics(true)
  }
}

export async function ensureSupirDiagnosticsLoaded(): Promise<void> {
  await loadSupirDiagnostics(false)
}

export async function reloadSupirDiagnostics(): Promise<void> {
  invalidateSupirModelsCache()
  supirDiagnostics.value = null
  supirDiagnosticsError.value = ''
  supirDiagnosticsAttempted.value = false
  await loadSupirDiagnostics(true)
}

export function resolveSupirSelectionState(options: {
  supported: boolean
  selectedVariant: string
  selectedSampler: string
  guidanceAdvancedEnabled?: boolean
}): SupirSelectionState {
  const diagnostics = supirDiagnostics.value
  const expected = diagnostics?.supir_models?.expected ?? {}
  const variantChoices: SupirVariantChoice[] = (diagnostics?.variants ?? []).map((entry) => ({
    value: String(entry.key || '').trim(),
    label: String(entry.label || entry.key || '').trim(),
    available: Boolean(expected[String(entry.key || '').trim()]?.present),
  }))
  const samplerChoices: SupirSamplerInfo[] = Array.from(new Map(
    (diagnostics?.samplers ?? [])
      .map((entry) => ({
        ...entry,
        id: String(entry.id || '').trim(),
        label: String(entry.label || '').trim(),
        stability: (entry.stability === 'dev' ? 'dev' : 'stable') as SupirSamplerInfo['stability'],
        native_sampler: String(entry.native_sampler || '').trim(),
        native_scheduler: String(entry.native_scheduler || '').trim(),
      }))
      .filter((entry) => entry.id && entry.label && entry.native_sampler && entry.native_scheduler)
      .map((entry) => [entry.id, entry] as const),
  ).values())
  const availableVariants = variantChoices.filter((entry) => entry.available && (entry.value === 'v0F' || entry.value === 'v0Q'))
  const normalizedVariant = String(options.selectedVariant || '').trim()
  const normalizedSampler = String(options.selectedSampler || '').trim()
  const selectedVariantInstalled = variantChoices.some((entry) => entry.value === normalizedVariant && entry.available)
  const selectedSamplerInfo = samplerChoices.find((entry) => entry.id === normalizedSampler) ?? null
  const selectedSamplerAvailable = selectedSamplerInfo !== null
  const selectionValid = selectedVariantInstalled && selectedSamplerAvailable

  let blockingReason = ''
  if (options.supported) {
    if (supirDiagnosticsLoading.value) {
      blockingReason = 'Loading SUPIR diagnostics…'
    } else if (supirDiagnosticsError.value) {
      blockingReason = `Failed to load SUPIR diagnostics: ${supirDiagnosticsError.value}`
    } else if (!diagnostics) {
      blockingReason = 'SUPIR diagnostics are not loaded yet.'
    } else if (Boolean(options.guidanceAdvancedEnabled)) {
      blockingReason = 'SUPIR mode cannot be enabled while Advanced Guidance/APG is active. Disable Advanced Guidance first.'
    } else if (availableVariants.length === 0) {
      blockingReason = 'No SUPIR variants are installed under the configured supir_models roots.'
    } else if (samplerChoices.length === 0) {
      blockingReason = 'The backend did not report any SUPIR samplers.'
    } else if (!selectedVariantInstalled) {
      blockingReason = `Selected SUPIR variant '${normalizedVariant}' is not installed.`
    } else if (!selectedSamplerAvailable) {
      blockingReason = `Selected SUPIR sampler '${normalizedSampler}' is unavailable.`
    }
  }

  return {
    variantChoices,
    samplerChoices,
    availableVariants,
    selectedVariantInstalled,
    selectedSamplerAvailable,
    selectionValid,
    selectedSamplerInfo,
    blockingReason,
  }
}

export function useSupirDiagnostics() {
  return {
    diagnostics: readonly(supirDiagnostics),
    loading: readonly(supirDiagnosticsLoading),
    error: readonly(supirDiagnosticsError),
    attempted: readonly(supirDiagnosticsAttempted),
    ensureSupirDiagnosticsLoaded,
    reloadSupirDiagnostics,
  }
}
