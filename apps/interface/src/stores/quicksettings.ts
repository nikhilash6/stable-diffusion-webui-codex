/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: QuickSettings global store (models/options + asset SHA/variant selection).
Loads lists from `/api/*`, persists option changes via `/api/options`, and maintains cached inventory-driven choice lists plus SHA/variant maps for VAEs/text encoders/WAN GGUF
so UI selections resolve to backend SHA-based assets (no raw-path inputs). It also caches IP-Adapter model/image-encoder option lists from the canonical inventory snapshot,
owns the global runtime-device override plus component storage/compute dtype overrides applied via options, and reads the current `/api/options` revision through the shared
API-client monotonic cache for generation payload contracts (`settings_revision`) plus bounded conditional option writes that must fail loud instead of overwriting newer owner state.
Text-encoder choices are sourced from inventory files constrained by `*_tenc` roots (not folder roots), and stale root-label overrides are
sanitized so `tenc_sha` resolution remains deterministic across families (including Qwen Image, Anima, and LTX2). Qwen Image text-encoder labels
resolve only through `qwen_image_tenc` root-scoped inventory rows, while Z-Image L2P text-encoder labels resolve through `zimage_tenc` root-scoped rows under
the distinct `zimage_l2p/<path>` label prefix. Inventory slot metadata is cached alongside
SHA mappings so SDXL core-only requests can emit explicit `tenc1_sha` / `tenc2_sha` selectors without guessing label order. VAE state defaults to canonical `built-in`
when no persisted value exists, request preflight can enforce fail-loud non-empty selection via `requireVaeSelection`, and LoRA SHA mappings
are refreshed through the store-owned inventory flow (`fetchInventoryWithLoraHydration` + `hydrateLoraShaMap`). Workflow/history restore can also
sync the VAE choice/SHA cache from a fresh inventory snapshot through `hydrateVaeInventorySnapshot(...)` so immediate generate uses the same VAE truth
that restore/apply just validated. FLUX.2 override persistence
stays truthful to the current Klein 4B / base-4B slice by keeping at most one `flux2/*` Qwen selector.

Symbols (top-level; keep in sync; no ghosts):
- `normalizeTextEncoderSelectionLabels` (function): Normalizes persisted/current TE override labels, deduping values and capping FLUX.2 to one selector.
- `useQuicksettingsStore` (store): Pinia store that owns QuickSettings state + actions; includes nested loaders (`loadModels/loadVaes/...`),
  setters that call API updates, inventory hydrators (`fetchInventoryWithLoraHydration`, `hydrateLoraShaMap`, `hydrateVaeInventorySnapshot`, `hydratePathsSnapshot`), and resolvers that map UI labels → inventory SHA/slot/variant
  (`resolve*Sha` helpers plus `resolveTextEncoderSlot`, `resolveWanGgufVariant`, including LoRA).
  It also exports checkpoint helpers (`resolveModelInfo`, `requireModelInfo`, `resolveFlux2CheckpointVariant`, `resolveLtxCheckpointExecutionMetadata`) so image/video requests can fail loud on stale checkpoint picks
  and FLUX.2 guidance semantics can be derived from the selected model without extra request fields. Inventory-owned cached refs now include VAE inventory rows and
  model path roots alongside VAE/text-encoder/IP-Adapter choice lists so restore/apply and QuickSettingsBar share one freshness owner for VAE canonicalization.
*/

import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { InventoryResponse, ModelInfo } from '../api/types'
import {
  fetchModelsWithFreshness,
  fetchOptions,
  updateOptions,
  fetchModelInventory,
  fetchFreshModelInventory,
  fetchFreshPaths,
  refreshModelInventoryAsync,
  fetchMemory,
  getModelCatalogInvalidationVersion,
  getCachedOptionsRevision,
  invalidateModelCatalogCaches,
  promoteCachedOptionsRevision,
} from '../api/client'
import type { ModelsFreshnessMarker } from '../api/client'

const TEXT_ENCODER_OVERRIDES_STORAGE_KEY = 'codex.quicksettings.text_encoder_overrides'
const DEVICE_STORAGE_KEY = 'codex.quicksettings.device'
const VAE_STORAGE_KEY = 'codex.quicksettings.vae'
const VAE_BY_FAMILY_STORAGE_KEY = 'codex.quicksettings.vae_by_family'
const VAE_BY_FAMILY_OPTION_KEY = 'codex_vae_by_family'
const DEFAULT_VAE_SELECTION = 'built-in'
const NONE_VAE_SELECTION = 'none'
const VAE_FAMILIES = ['sd15', 'sdxl', 'flux1', 'flux2', 'chroma', 'zimage', 'qwen_image', 'anima', 'ltx2'] as const
type VaeFamily = (typeof VAE_FAMILIES)[number]
type WanGgufVariant = NonNullable<InventoryResponse['wan22']['gguf'][number]['variant']>

const TEXT_ENCODER_FAMILY_KEYS: Array<[string, string]> = [
  ['sd15', 'sd15_tenc'],
  ['sdxl', 'sdxl_tenc'],
  ['flux1', 'flux1_tenc'],
  ['flux2', 'flux2_tenc'],
  ['qwen_image', 'qwen_image_tenc'],
  ['anima', 'anima_tenc'],
  ['ltx2', 'ltx2_tenc'],
  ['wan22', 'wan22_tenc'],
  ['zimage', 'zimage_tenc'],
  ['zimage_l2p', 'zimage_tenc'],
]

const TEXT_ENCODER_PREFIXES = ['sd15', 'sdxl', 'flux1', 'flux2', 'anima', 'chroma', 'ltx2', 'wan22', 'zimage', 'zimage_l2p']
const FLUX2_UNSUPPORTED_VARIANT_MARKERS = [
  'flux.2-klein-base-9b',
  'flux2-klein-base-9b',
  'flux.2-klein-9b',
  'flux2-klein-9b',
  'base-9b',
  '/9b/',
  '-9b',
] as const
const FLUX2_BASE_VARIANT_MARKERS = [
  'flux.2-klein-base-4b',
  'flux2-klein-base-4b',
  'base-4b',
  'base_4b',
  '/base/',
] as const

function normalizeRevision(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.max(0, Math.trunc(value))
  }
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return null
    if (/^-?\d+$/.test(trimmed)) {
      return Math.max(0, Math.trunc(Number(trimmed)))
    }
  }
  return null
}

function normalizePath(raw: string): string {
  const normalized = String(raw || '').trim().replace(/\\+/g, '/')
  if (normalized.length <= 1) return normalized
  return normalized.replace(/\/+$/g, '')
}

function isRecordObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function normalizeFlux2Variant(value: unknown): 'distilled' | 'base' | null {
  const normalized = normalizePath(String(value || '')).toLowerCase()
  if (!normalized) return null
  if (normalized === 'base') return 'base'
  if (normalized === 'distilled' || normalized === 'klein') return 'distilled'
  if (FLUX2_BASE_VARIANT_MARKERS.some((marker) => normalized.includes(marker))) return 'base'
  return null
}

export type LtxCheckpointKind = 'dev' | 'distilled' | 'unknown'

export interface LtxCheckpointExecutionMetadata {
  checkpointKind: LtxCheckpointKind
  allowedExecutionProfiles: string[]
  defaultExecutionProfile: string | null
  defaultSteps: number | null
  defaultGuidanceScale: number | null
}

function normalizeLtxCheckpointKind(value: unknown): LtxCheckpointKind | null {
  const normalized = normalizePath(String(value || '')).toLowerCase()
  if (normalized === 'dev' || normalized === 'distilled' || normalized === 'unknown') return normalized
  return null
}

function parseLtxExecutionProfiles(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null
  const out: string[] = []
  const seen = new Set<string>()
  for (const entry of value) {
    const normalized = String(entry || '').trim()
    if (!normalized || seen.has(normalized)) continue
    seen.add(normalized)
    out.push(normalized)
  }
  return out
}

function parseOptionalNonNegativeInteger(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isInteger(value) || value < 0) return null
  return value
}

function parseOptionalNonNegativeNumber(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return null
  return value
}

function appendUniqueCandidate(target: string[], seen: Set<string>, value: unknown): void {
  const normalized = normalizePath(String(value || ''))
  if (!normalized) return
  const key = normalized.toLowerCase()
  if (seen.has(key)) return
  seen.add(key)
  target.push(normalized)
}

function normalizeTextEncoderSelectionLabels(labels: readonly string[]): string[] {
  const next: string[] = []
  const seen = new Set<string>()
  let flux2Selected = false

  for (const raw of labels) {
    const normalized = normalizePath(String(raw || ''))
    if (!normalized) continue
    if (seen.has(normalized)) continue
    if (normalized.startsWith('flux2/')) {
      if (flux2Selected) continue
      flux2Selected = true
    }
    seen.add(normalized)
    next.push(normalized)
  }

  return next
}

function pathMatchesRoot(filePath: string, rootPath: string): boolean {
  if (!filePath || !rootPath) return false
  const fileNorm = normalizePath(filePath)
  const rootNorm = normalizePath(rootPath)
  if (!fileNorm || !rootNorm) return false
  const candidates = new Set<string>()
  candidates.add(rootNorm)
  if (rootNorm.startsWith('/')) {
    candidates.add(rootNorm.slice(1))
    const modelsIdx = rootNorm.lastIndexOf('/models/')
    if (modelsIdx >= 0) {
      candidates.add(rootNorm.slice(modelsIdx + 1))
    }
  }

  for (const candidate of candidates) {
    if (!candidate) continue
    if (fileNorm === candidate || fileNorm.startsWith(candidate + '/')) return true
    if (fileNorm.includes('/' + candidate + '/') || fileNorm.endsWith('/' + candidate)) return true
  }
  return false
}

function lookupTextEncoderShaFromMap(map: Map<string, string>, label: string): string | undefined {
  const normalized = normalizePath(label)
  if (!normalized) return undefined
  const withoutPrefix = normalized.includes('/') ? normalized.split('/').slice(1).join('/') : normalized
  const tail = normalized.split('/').pop() || ''
  return map.get(normalized) || map.get(withoutPrefix) || map.get(tail)
}

function normalizeVaeSelection(value: string | null | undefined): string {
  const raw = String(value || '').trim()
  if (!raw) return ''
  const lower = raw.toLowerCase()
  if (lower === 'automatic' || lower === 'built in' || lower === 'built-in') {
    return DEFAULT_VAE_SELECTION
  }
  if (lower === 'none') return NONE_VAE_SELECTION
  return raw
}

function isVaeFamily(value: string): value is VaeFamily {
  return (VAE_FAMILIES as readonly string[]).includes(value)
}

function serializeVaeByFamilyOption(values: Partial<Record<VaeFamily, string>>): string {
  const payload: Partial<Record<VaeFamily, string>> = {}
  for (const family of VAE_FAMILIES) {
    const normalized = normalizeVaeSelection(values[family] ?? '')
    if (!normalized) continue
    payload[family] = normalized
  }
  return JSON.stringify(payload)
}

function parseVaeByFamilyOption(value: unknown): Partial<Record<VaeFamily, string>> {
  let parsed: unknown = value
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) {
      throw new Error(`Invalid options.${VAE_BY_FAMILY_OPTION_KEY}: expected JSON object string, got empty string.`)
    }
    try {
      parsed = JSON.parse(trimmed)
    } catch (error) {
      throw new Error(
        `Invalid options.${VAE_BY_FAMILY_OPTION_KEY}: expected JSON object string (${String(error)}).`,
      )
    }
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`Invalid options.${VAE_BY_FAMILY_OPTION_KEY}: expected object.`)
  }
  const next: Partial<Record<VaeFamily, string>> = {}
  for (const [key, rawValue] of Object.entries(parsed as Record<string, unknown>)) {
    const family = String(key || '').trim().toLowerCase()
    if (!isVaeFamily(family)) {
      throw new Error(`Invalid options.${VAE_BY_FAMILY_OPTION_KEY}: unknown family '${key}'.`)
    }
    if (typeof rawValue !== 'string') {
      throw new Error(`Invalid options.${VAE_BY_FAMILY_OPTION_KEY}: family '${family}' must map to a string.`)
    }
    const normalized = normalizeVaeSelection(rawValue)
    if (!normalized) {
      throw new Error(`Invalid options.${VAE_BY_FAMILY_OPTION_KEY}: family '${family}' has empty VAE selection.`)
    }
    next[family] = normalized
  }
  return next
}

function buildLoraShaMapFromInventory(inventory: Pick<InventoryResponse, 'loras'> | null | undefined): Map<string, string> {
  const loraMap = new Map<string, string>()
  if (!inventory || !Array.isArray(inventory.loras)) return loraMap

  for (const lora of inventory.loras) {
    const sha = typeof lora.sha256 === 'string' ? lora.sha256.trim().toLowerCase() : ''
    if (!sha || !/^[0-9a-f]{64}$/.test(sha)) continue
    const name = typeof lora.name === 'string' ? lora.name.trim() : ''
    const rawPath = typeof lora.path === 'string' ? lora.path.trim() : ''
    const normPath = rawPath ? rawPath.replace(/\\+/g, '/') : ''
    const basename = normPath ? normPath.split('/').pop() : name
    const keys = new Set<string>()
    if (name) keys.add(name)
    if (rawPath) keys.add(rawPath)
    if (normPath) keys.add(normPath)
    if (basename) keys.add(String(basename))
    for (const key of keys) {
      loraMap.set(key, sha)
    }
  }
  return loraMap
}

function buildInventoryChoiceList(
  items: Array<{ name?: string; path?: string }> | null | undefined,
): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  if (!Array.isArray(items)) return out

  for (const item of items) {
    const path = normalizePath(String(item?.path || ''))
    if (path) {
      appendUniqueCandidate(out, seen, path)
      continue
    }
    appendUniqueCandidate(out, seen, item?.name)
  }

  return out
}

export const useQuicksettingsStore = defineStore('quicksettings', () => {
  const models = ref<ModelInfo[]>([])
  const modelsFreshness = ref<ModelsFreshnessMarker | null>(null)
  const currentModel = ref<string>('')
  const vaeChoices = ref<string[]>([])
  const inventoryVaesSnapshot = ref<InventoryResponse['vaes']>([])
  const pathsConfigSnapshot = ref<Record<string, string[]>>({})
  let assetSnapshotEpoch = 0
  const currentVae = ref<string>(DEFAULT_VAE_SELECTION)
  const vaeByFamily = ref<Partial<Record<VaeFamily, string>>>({})
  const textEncoderChoices = ref<string[]>([])
  const currentTextEncoders = ref<string[]>([])
  const ipAdapterModelChoices = ref<string[]>([])
  const ipAdapterImageEncoderChoices = ref<string[]>([])
  const textEncoderRootLabels = ref<Set<string>>(new Set())
  // SHA256 lookup maps populated from inventory
  const textEncoderShaMap = ref<Map<string, string>>(new Map())
  const textEncoderSlotMap = ref<Map<string, string>>(new Map())
  const vaeShaMap = ref<Map<string, string>>(new Map())
  const loraShaMap = ref<Map<string, string>>(new Map())
  const wanGgufShaMap = ref<Map<string, string>>(new Map())
  const wanGgufVariantMap = ref<Map<string, WanGgufVariant>>(new Map())
  const deviceChoices = ref<{ value: string; label: string }[]>([
    { value: 'cuda', label: 'CUDA' },
    { value: 'cpu', label: 'CPU' },
    { value: 'mps', label: 'MPS' },
    { value: 'xpu', label: 'XPU' },
    { value: 'directml', label: 'DirectML' },
  ])
  const currentDevice = ref<string>('cuda')
  const mainDevice = ref<string>('auto')
  const dtypeChoices = ref<string[]>(['auto', 'fp16', 'bf16', 'fp32'])
  const coreDtype = ref<string>('auto')
  const coreComputeDtype = ref<string>('auto')
  const teDtype = ref<string>('auto')
  const teComputeDtype = ref<string>('auto')
  const vaeDtype = ref<string>('auto')
  const vaeComputeDtype = ref<string>('auto')
  const smartOffload = ref<boolean>(false)
  const smartFallback = ref<boolean>(false)
  const smartCache = ref<boolean>(true)
  const coreStreaming = ref<boolean>(false)
  const lastAppliedNowMessages = ref<string[]>([])
  const lastRestartRequiredMessages = ref<string[]>([])
  let modelsRequestSerial = 0
  let optionsRequestSerial = 0

  function loadTextEncoderOverridesFromStorage(): void {
    try {
      const raw = localStorage.getItem(TEXT_ENCODER_OVERRIDES_STORAGE_KEY)
      if (!raw) return
      const parsed = JSON.parse(raw)
      if (!Array.isArray(parsed)) return
      currentTextEncoders.value = normalizeTextEncoderSelectionLabels(
        parsed
          .map((entry) => String(entry).trim())
          .filter((entry) => entry.length > 0),
      )
    } catch (err) {
      console.warn('[quicksettings] failed to load text encoder overrides from localStorage', err)
    }
  }

  function saveTextEncoderOverridesToStorage(labels: string[]): void {
    try {
      localStorage.setItem(TEXT_ENCODER_OVERRIDES_STORAGE_KEY, JSON.stringify(labels))
    } catch (err) {
      console.warn('[quicksettings] failed to persist text encoder overrides to localStorage', err)
    }
  }

  function sanitizeTextEncoderOverrides(): void {
    if (textEncoderRootLabels.value.size === 0 && textEncoderShaMap.value.size === 0) return
    const next: string[] = []
    const seen = new Set<string>()
    const canValidateSha = textEncoderShaMap.value.size > 0

    for (const entry of currentTextEncoders.value) {
      const raw = String(entry || '').trim()
      if (!raw) continue
      const normalized = normalizePath(raw)
      if (!normalized) continue
      const lower = normalized.toLowerCase()
      const isSha = lower.length === 64 && /^[0-9a-f]+$/.test(lower)
      if (isSha) {
        if (!seen.has(lower)) {
          seen.add(lower)
          next.push(lower)
        }
        continue
      }

      const resolvedSha = lookupTextEncoderShaFromMap(textEncoderShaMap.value, normalized)
      if (textEncoderRootLabels.value.has(normalized) && !resolvedSha) continue

      if (!canValidateSha || resolvedSha) {
        if (!seen.has(normalized)) {
          seen.add(normalized)
          next.push(normalized)
        }
      }
    }

    const normalizedNext = normalizeTextEncoderSelectionLabels(next)
    const changed =
      normalizedNext.length !== currentTextEncoders.value.length ||
      normalizedNext.some((label, index) => label !== currentTextEncoders.value[index])
    if (changed) {
      currentTextEncoders.value = normalizedNext
      saveTextEncoderOverridesToStorage(normalizedNext)
    }
  }

  function loadDeviceFromStorage(): void {
    try {
      const raw = localStorage.getItem(DEVICE_STORAGE_KEY)
      if (!raw) return
      const normalized = String(raw).trim().toLowerCase()
      if (!normalized) return
      if (deviceChoices.value.some((d) => d.value === normalized)) {
        currentDevice.value = normalized
      }
    } catch (err) {
      console.warn('[quicksettings] failed to load device from localStorage', err)
    }
  }

  function saveDeviceToStorage(device: string): void {
    try {
      localStorage.setItem(DEVICE_STORAGE_KEY, String(device))
    } catch (err) {
      console.warn('[quicksettings] failed to persist device to localStorage', err)
    }
  }

  function clearDeviceStorage(): void {
    try {
      localStorage.removeItem(DEVICE_STORAGE_KEY)
    } catch (err) {
      console.warn('[quicksettings] failed to clear device from localStorage', err)
    }
  }

  function normalizeRuntimeDevice(raw: unknown): string | null {
    const normalized = String(raw || '').trim().toLowerCase()
    if (!normalized) return null
    if (normalized === 'gpu') return 'cuda'
    if (normalized === 'dml') return 'directml'
    if (normalized.startsWith('cuda')) return 'cuda'
    if (deviceChoices.value.some((entry) => entry.value === normalized)) return normalized
    return null
  }

  function normalizeMainDeviceSetting(raw: unknown): string {
    const normalized = String(raw || '').trim().toLowerCase()
    if (normalized === 'auto') return 'auto'
    return normalizeRuntimeDevice(normalized) || 'auto'
  }

  async function syncCurrentDeviceFromBackendAuthority(): Promise<void> {
    try {
      const memory = await fetchMemory()
      const normalized = normalizeRuntimeDevice((memory as any).primary_device ?? (memory as any).device_backend)
      if (!normalized) return
      currentDevice.value = normalized
      saveDeviceToStorage(normalized)
    } catch (err) {
      console.warn('[quicksettings] failed to sync device from backend authority', err)
    }
  }

  function loadVaeFromStorage(): void {
    try {
      const raw = localStorage.getItem(VAE_STORAGE_KEY)
      if (!raw) {
        currentVae.value = DEFAULT_VAE_SELECTION
        return
      }
      const normalized = normalizeVaeSelection(raw)
      currentVae.value = normalized || DEFAULT_VAE_SELECTION
    } catch (err) {
      console.warn('[quicksettings] failed to load VAE selection from localStorage', err)
    }
  }

  function loadVaeByFamilyFromStorage(): void {
    try {
      const raw = localStorage.getItem(VAE_BY_FAMILY_STORAGE_KEY)
      if (!raw) {
        vaeByFamily.value = {}
        return
      }
      const parsed = JSON.parse(raw)
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        vaeByFamily.value = {}
        return
      }
      const next: Partial<Record<VaeFamily, string>> = {}
      for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
        const family = String(key || '').trim().toLowerCase()
        if (!isVaeFamily(family)) continue
        const normalized = normalizeVaeSelection(typeof value === 'string' ? value : '')
        if (!normalized) continue
        next[family] = normalized
      }
      vaeByFamily.value = next
    } catch (err) {
      console.warn('[quicksettings] failed to load VAE-by-family map from localStorage', err)
    }
  }

  function saveVaeToStorage(label: string): void {
    try {
      const normalized = normalizeVaeSelection(label)
      if (!normalized) {
        localStorage.removeItem(VAE_STORAGE_KEY)
        return
      }
      localStorage.setItem(VAE_STORAGE_KEY, normalized)
    } catch (err) {
      console.warn('[quicksettings] failed to persist VAE selection to localStorage', err)
    }
  }

  function saveVaeByFamilyToStorage(values: Partial<Record<VaeFamily, string>>): void {
    try {
      const entries = Object.entries(values).filter((entry) => {
        const family = String(entry[0] || '').trim().toLowerCase()
        if (!isVaeFamily(family)) return false
        return Boolean(normalizeVaeSelection(String(entry[1] || '')))
      })
      if (!entries.length) {
        localStorage.removeItem(VAE_BY_FAMILY_STORAGE_KEY)
        return
      }
      const payload: Partial<Record<VaeFamily, string>> = {}
      for (const [family, value] of entries) {
        const normalized = normalizeVaeSelection(String(value || ''))
        if (!normalized) continue
        payload[family as VaeFamily] = normalized
      }
      localStorage.setItem(VAE_BY_FAMILY_STORAGE_KEY, JSON.stringify(payload))
    } catch (err) {
      console.warn('[quicksettings] failed to persist VAE-by-family map to localStorage', err)
    }
  }

  function getPersistedVaeForFamily(family: string): string {
    const normalizedFamily = String(family || '').trim().toLowerCase()
    if (!isVaeFamily(normalizedFamily)) return ''
    const fromFamily = normalizeVaeSelection(vaeByFamily.value[normalizedFamily] ?? '')
    return fromFamily || ''
  }

  function getVaeForFamily(family: string): string {
    const normalizedFamily = String(family || '').trim().toLowerCase()
    if (!isVaeFamily(normalizedFamily)) {
      return currentVae.value || DEFAULT_VAE_SELECTION
    }
    const fromFamily = getPersistedVaeForFamily(normalizedFamily)
    if (fromFamily) return fromFamily
    return DEFAULT_VAE_SELECTION
  }

  function getSettingsRevision(): number {
    return Math.max(0, Math.trunc(getCachedOptionsRevision()))
  }

  async function refreshSettingsRevision(fallbackRevision?: number): Promise<number> {
    try {
      await fetchOptions()
    } catch (error) {
      const fallback = normalizeRevision(fallbackRevision)
      if (fallback !== null) {
        promoteCachedOptionsRevision(fallback)
      } else {
        throw error
      }
    }
    return getSettingsRevision()
  }

  async function applyOptionUpdate(payload: Record<string, unknown>): Promise<void> {
    const response = await updateOptions(payload, { expectedRevision: getSettingsRevision() })
    const appliedNowRaw = (response as any).applied_now
    const restartRequiredRaw = (response as any).restart_required
    lastAppliedNowMessages.value = Array.isArray(appliedNowRaw) ? appliedNowRaw.map((item) => String(item)) : []
    lastRestartRequiredMessages.value = Array.isArray(restartRequiredRaw) ? restartRequiredRaw.map((item) => String(item)) : []
  }

  function clearOptionApplyMessages(): void {
    lastAppliedNowMessages.value = []
    lastRestartRequiredMessages.value = []
  }

  async function init(): Promise<void> {
    await Promise.all([
      loadModels(),
      loadVaes(),
      loadTextEncoders(),
      loadOptions(),
    ])
  }

  function applyModelsResponse(args: {
    response: { models: ModelInfo[]; current: string | null }
    freshness: ModelsFreshnessMarker
    requestSerial: number
  }): void {
    const { response, freshness, requestSerial } = args
    // Ignore stale responses that resolve after a newer models request.
    if (requestSerial !== modelsRequestSerial) return
    models.value = response.models
    modelsFreshness.value = freshness
    if (!currentModel.value && response.current) {
      currentModel.value = response.current
    }
  }

  async function loadModelsList(options?: { refresh?: boolean; invalidate?: boolean }): Promise<void> {
    const requestSerial = ++modelsRequestSerial
    const { response, freshness } = await fetchModelsWithFreshness({
      refresh: options?.refresh === true,
      invalidate: options?.invalidate === true,
    })
    applyModelsResponse({
      response,
      freshness,
      requestSerial,
    })
  }

  async function loadModels(): Promise<void> {
    await loadModelsList({ refresh: false })
  }

  function invalidateModelsList(): number {
    return invalidateModelCatalogCaches()
  }

  async function refreshModelsList(): Promise<void> {
    // Explicit invalidation is centralized here (no scattered ad-hoc refresh semantics).
    invalidateModelsList()
    await loadModelsList({ refresh: true, invalidate: false })
  }

  async function loadOptions(): Promise<void> {
    loadDeviceFromStorage()
    loadVaeFromStorage()
    loadVaeByFamilyFromStorage()
    loadTextEncoderOverridesFromStorage()
    sanitizeTextEncoderOverrides()

    const requestEpoch = assetSnapshotEpoch
    const requestSerial = ++optionsRequestSerial
    const isCurrentOptionsRequest = (): boolean =>
      requestSerial === optionsRequestSerial && requestEpoch === assetSnapshotEpoch

    const res = await fetchOptions()
    if (!isCurrentOptionsRequest()) return
    const opts = res.values
    const hasVaeByFamilyOption = Object.prototype.hasOwnProperty.call(opts, VAE_BY_FAMILY_OPTION_KEY)
    if (hasVaeByFamilyOption) {
      const nextFromBackend = parseVaeByFamilyOption((opts as Record<string, unknown>)[VAE_BY_FAMILY_OPTION_KEY])
      vaeByFamily.value = nextFromBackend
      saveVaeByFamilyToStorage(nextFromBackend)
    } else if (Object.keys(vaeByFamily.value).length > 0) {
      if (!isCurrentOptionsRequest()) return
      try {
        await applyOptionUpdate({
          [VAE_BY_FAMILY_OPTION_KEY]: serializeVaeByFamilyOption(vaeByFamily.value),
        })
      } catch (error) {
        if (!isCurrentOptionsRequest()) return
        throw error
      }
      if (!isCurrentOptionsRequest()) return
    }
    if (typeof (opts as any).codex_main_device === 'string') {
      mainDevice.value = normalizeMainDeviceSetting((opts as any).codex_main_device)
      if (mainDevice.value === 'auto') {
        await syncCurrentDeviceFromBackendAuthority()
      } else {
        currentDevice.value = mainDevice.value
        saveDeviceToStorage(mainDevice.value)
      }
    }
    if (typeof (opts as any).codex_core_dtype === 'string') coreDtype.value = (opts as any).codex_core_dtype
    if (typeof (opts as any).codex_core_compute_dtype === 'string') coreComputeDtype.value = (opts as any).codex_core_compute_dtype
    if (typeof (opts as any).codex_te_dtype === 'string') teDtype.value = (opts as any).codex_te_dtype
    if (typeof (opts as any).codex_te_compute_dtype === 'string') teComputeDtype.value = (opts as any).codex_te_compute_dtype
    if (typeof (opts as any).codex_vae_dtype === 'string') vaeDtype.value = (opts as any).codex_vae_dtype
    if (typeof (opts as any).codex_vae_compute_dtype === 'string') vaeComputeDtype.value = (opts as any).codex_vae_compute_dtype
    if (typeof (opts as any).codex_smart_offload === 'boolean') {
      smartOffload.value = (opts as any).codex_smart_offload
    }
    if (typeof (opts as any).codex_smart_fallback === 'boolean') {
      smartFallback.value = (opts as any).codex_smart_fallback
    }
    if (typeof (opts as any).codex_smart_cache === 'boolean') {
      smartCache.value = (opts as any).codex_smart_cache
    }
    if (typeof (opts as any).codex_core_streaming === 'boolean') {
      coreStreaming.value = (opts as any).codex_core_streaming
    }
  }

  function hydrateInventoryChoiceCaches(
    inventory: Pick<InventoryResponse, 'vaes' | 'ip_adapter_models' | 'ip_adapter_image_encoders'> | null | undefined,
  ): void {
    vaeChoices.value = buildVaeChoiceList(inventory)
    ipAdapterModelChoices.value = buildInventoryChoiceList(inventory?.ip_adapter_models)
    ipAdapterImageEncoderChoices.value = buildInventoryChoiceList(inventory?.ip_adapter_image_encoders)
  }

  function buildVaeChoiceList(
    inventory: Pick<InventoryResponse, 'vaes'> | null | undefined,
  ): string[] {
    const seen = new Set<string>()
    const vaeOut: string[] = [DEFAULT_VAE_SELECTION, NONE_VAE_SELECTION]
    for (const item of inventory?.vaes || []) {
      const name = String(item?.name || '').trim()
      if (!name || seen.has(name)) continue
      seen.add(name)
      if (!vaeOut.includes(name)) vaeOut.push(name)
    }
    return vaeOut
  }

  function buildVaeShaMap(
    inventory: Pick<InventoryResponse, 'vaes'> | null | undefined,
  ): Map<string, string> {
    const vaeMap = new Map<string, string>()
    for (const vae of inventory?.vaes || []) {
      const sha = typeof vae.sha256 === 'string' ? vae.sha256 : ''
      if (!sha) continue
      const name = typeof vae.name === 'string' ? vae.name : ''
      const rawPath = typeof vae.path === 'string' ? vae.path : ''
      const normPath = rawPath ? rawPath.replace(/\\+/g, '/') : ''
      const keys = new Set<string>()
      if (name) keys.add(name)
      if (rawPath) keys.add(rawPath)
      if (normPath) keys.add(normPath)
      const basename = normPath ? normPath.split('/').pop() : name
      if (basename) keys.add(basename)
      if (normPath) {
        for (const prefix of TEXT_ENCODER_PREFIXES) {
          keys.add(`${prefix}/${normPath}`)
        }
      }
      for (const key of keys) {
        vaeMap.set(key, sha)
      }
    }
    return vaeMap
  }

  function hydrateVaeInventorySnapshot(
    inventory: Pick<InventoryResponse, 'vaes'> | null | undefined,
  ): void {
    inventoryVaesSnapshot.value = Array.isArray(inventory?.vaes) ? [...inventory.vaes] : []
    vaeChoices.value = buildVaeChoiceList(inventory)
    vaeShaMap.value = buildVaeShaMap(inventory)
  }

  function hydratePathsSnapshot(paths: Record<string, string[]> | null | undefined): void {
    const snapshot: Record<string, string[]> = {}
    for (const [key, values] of Object.entries(paths || {})) {
      snapshot[key] = Array.isArray(values) ? values.map((value) => String(value || '')) : []
    }
    pathsConfigSnapshot.value = snapshot
  }

  function bumpAssetSnapshotEpoch(): number {
    const next = assetSnapshotEpoch + 1
    assetSnapshotEpoch = Number.isSafeInteger(next) ? next : 1
    return assetSnapshotEpoch
  }

  function getAssetSnapshotEpoch(): number {
    return assetSnapshotEpoch
  }

  async function loadVaes(): Promise<void> {
    const requestEpoch = assetSnapshotEpoch
    const inv = await fetchModelInventory()
    if (requestEpoch !== assetSnapshotEpoch) return
    hydrateInventoryChoiceCaches(inv)
  }

  async function loadTextEncoders(): Promise<void> {
    const requestEpoch = assetSnapshotEpoch
    const [pathsRes, inv] = await Promise.all([fetchFreshPaths(), fetchModelInventory()])
    if (requestEpoch !== assetSnapshotEpoch) return
    const paths = ((pathsRes as any)?.paths || {}) as Record<string, string[]>
    hydratePathsSnapshot(paths)

    const rootsByFamily = new Map<string, string[]>()
    const rootLabels = new Set<string>()
    for (const [family, key] of TEXT_ENCODER_FAMILY_KEYS) {
      const roots = (Array.isArray(paths[key]) ? paths[key] : [])
        .map((entry) => normalizePath(String(entry || '')))
        .filter((entry) => entry.length > 0)
      rootsByFamily.set(family, roots)
      for (const root of roots) {
        rootLabels.add(`${family}/${root}`)
      }
    }
    textEncoderRootLabels.value = rootLabels

    const labels = new Set<string>()
    const shaMap = new Map<string, string>()
    const slotMap = new Map<string, string>()
    for (const te of inv.text_encoders || []) {
      const sha = typeof te.sha256 === 'string' ? te.sha256.trim().toLowerCase() : ''
      if (!sha || !/^[0-9a-f]{64}$/.test(sha)) continue
      const name = typeof te.name === 'string' ? te.name.trim() : ''
      const rawPath = typeof te.path === 'string' ? te.path.trim() : ''
      const normPath = normalizePath(rawPath)
      const basename = normPath ? normPath.split('/').pop() || '' : ''
      const slot = typeof te.slot === 'string' ? te.slot.trim() : ''

      const matchedFamilies: string[] = []
      if (normPath) {
        for (const [family] of TEXT_ENCODER_FAMILY_KEYS) {
          const roots = rootsByFamily.get(family) || []
          if (roots.some((root) => pathMatchesRoot(normPath, root))) {
            matchedFamilies.push(family)
          }
        }
      }
      for (const family of matchedFamilies) {
        labels.add(`${family}/${normPath}`)
      }

      const mapKeys = new Set<string>()
      if (name) mapKeys.add(name)
      if (rawPath) mapKeys.add(rawPath)
      if (normPath) mapKeys.add(normPath)
      if (basename) mapKeys.add(basename)
      if (normPath) {
        for (const family of matchedFamilies) {
          mapKeys.add(`${family}/${normPath}`)
        }
        for (const prefix of TEXT_ENCODER_PREFIXES) {
          mapKeys.add(`${prefix}/${normPath}`)
        }
      }
      for (const key of mapKeys) {
        shaMap.set(key, sha)
        if (slot) slotMap.set(key, slot)
      }
    }
    textEncoderChoices.value = Array.from(labels).sort()
    textEncoderShaMap.value = shaMap
    textEncoderSlotMap.value = slotMap

    vaeShaMap.value = buildVaeShaMap(inv)

    const wanMap = new Map<string, string>()
    const wanVariantMap = new Map<string, WanGgufVariant>()
    const wanFiles = (inv as any)?.wan22?.gguf
    if (Array.isArray(wanFiles)) {
      for (const w of wanFiles) {
        const sha = typeof w?.sha256 === 'string' ? w.sha256 : ''
        if (!sha) continue
        const variant = w?.variant === 'wan22_5b' || w?.variant === 'wan22_14b' || w?.variant === 'wan22_14b_animate'
          ? w.variant
          : null
        const name = typeof w?.name === 'string' ? w.name : ''
        const rawPath = typeof w?.path === 'string' ? w.path : ''
        const normPath = rawPath ? rawPath.replace(/\\+/g, '/') : ''
        const keys = new Set<string>()
        if (name) keys.add(name)
        if (rawPath) keys.add(rawPath)
        if (normPath) keys.add(normPath)
        const basename = normPath ? normPath.split('/').pop() : name
        if (basename) keys.add(basename)
        for (const key of keys) {
          wanMap.set(key, sha)
          if (variant) wanVariantMap.set(key, variant)
        }
      }
    }
    wanGgufShaMap.value = wanMap
    wanGgufVariantMap.value = wanVariantMap

    hydrateInventoryChoiceCaches(inv)
    hydrateLoraShaMap(inv)
    sanitizeTextEncoderOverrides()
  }

  function hydrateLoraShaMap(inventory: Pick<InventoryResponse, 'loras'> | null | undefined): void {
    loraShaMap.value = buildLoraShaMapFromInventory(inventory)
  }

  async function fetchInventoryWithLoraHydration(
    options: { refresh?: boolean; signal?: AbortSignal } = {},
  ): Promise<InventoryResponse> {
    let inventory: InventoryResponse
    if (options.refresh === true) {
      inventory = await refreshModelInventoryAsync({ signal: options.signal })
    } else {
      let usedFreshRetry = false
      while (true) {
        const requestVersion = getModelCatalogInvalidationVersion()
        inventory = usedFreshRetry
          ? await fetchFreshModelInventory()
          : await fetchModelInventory()
        if (requestVersion === getModelCatalogInvalidationVersion()) {
          break
        }
        usedFreshRetry = true
      }
    }
    hydrateInventoryChoiceCaches(inventory)
    hydrateLoraShaMap(inventory)
    return inventory
  }

  function resolveTextEncoderSha(label: string | null | undefined): string | undefined {
    if (!label) return undefined
    const normalized = label.replace(/\\+/g, '/')

    const lower = normalized.trim().toLowerCase()
    if (lower.length === 64 && /^[0-9a-f]+$/.test(lower)) {
      return lower
    }
    if (normalized.startsWith('qwen_image/')) {
      return textEncoderShaMap.value.get(normalized)
    }

    return lookupTextEncoderShaFromMap(textEncoderShaMap.value, normalized)
  }

  function resolveTextEncoderSlot(label: string | null | undefined): string | undefined {
    if (!label) return undefined
    const normalized = label.replace(/\\+/g, '/')
    if (normalized.startsWith('qwen_image/')) {
      return textEncoderSlotMap.value.get(normalized)
    }
    return textEncoderSlotMap.value.get(normalized)
  }

  function resolveModelInfo(label: string | null | undefined): ModelInfo | undefined {
    const raw = String(label || '').trim()
    if (!raw) return undefined
    if (models.value.length === 0) return undefined

    const normalized = raw.replace(/\\+/g, '/')
    const tail = normalized.split('/').pop() || ''
    const lower = raw.toLowerCase()
    const isHex = /^[0-9a-f]+$/.test(lower)
    const looksLikeSha = (lower.length === 10 || lower.length === 64) && isHex

    for (const model of models.value) {
      if (!model) continue
      const modelHash = String(model.hash || '').trim().toLowerCase()
      if (looksLikeSha && modelHash && modelHash === lower) return model

      if (raw === model.title || raw === model.name || raw === model.filename) return model

      const fileNorm = String(model.filename || '').replace(/\\+/g, '/')
      if (normalized && normalized === fileNorm) return model

      const fileTail = fileNorm.split('/').pop() || ''
      if (tail && (tail === model.title || tail === model.name || tail === fileTail)) return model
    }
    return undefined
  }

  function requireModelInfo(label: string | null | undefined): ModelInfo {
    const model = resolveModelInfo(label)
    if (model) return model
    throw new Error('Selected checkpoint is invalid or stale. Refresh model inventory and re-select the checkpoint.')
  }

  function resolveFlux2CheckpointVariant(
    source: string | ModelInfo | null | undefined,
  ): 'distilled' | 'base' | null {
    const raw = typeof source === 'string' ? String(source || '').trim() : ''
    const model = typeof source === 'string' ? resolveModelInfo(source) : source

    if (model) {
      const metadata = isRecordObject(model.metadata) ? model.metadata : null
      if (metadata) {
        const explicitVariant = normalizeFlux2Variant(
          metadata.flux2_variant
          ?? metadata.variant
          ?? metadata.model_variant
          ?? metadata['codex.flux2.variant'],
        )
        if (explicitVariant) return explicitVariant

        const isDistilled = metadata.is_distilled
        if (typeof isDistilled === 'boolean') return isDistilled ? 'distilled' : 'base'

        const rawMetadata = isRecordObject(metadata.raw) ? metadata.raw : null
        if (rawMetadata) {
          const rawVariant = normalizeFlux2Variant(
            rawMetadata['codex.flux2.variant']
            ?? rawMetadata.flux2_variant
            ?? rawMetadata.variant,
          )
          if (rawVariant) return rawVariant
          const rawIsDistilled = rawMetadata.is_distilled
          if (typeof rawIsDistilled === 'boolean') return rawIsDistilled ? 'distilled' : 'base'
        }
      }
    }

    const candidates: string[] = []
    const seen = new Set<string>()
    if (model) {
      appendUniqueCandidate(candidates, seen, model.title)
      appendUniqueCandidate(candidates, seen, model.name)
      appendUniqueCandidate(candidates, seen, model.model_name)
      appendUniqueCandidate(candidates, seen, model.filename)
      appendUniqueCandidate(candidates, seen, model.family_hint)
      const metadata = isRecordObject(model.metadata) ? model.metadata : null
      if (metadata) {
        appendUniqueCandidate(candidates, seen, metadata.repo_hint)
        appendUniqueCandidate(candidates, seen, metadata.repo_id)
        appendUniqueCandidate(candidates, seen, metadata.huggingface_repo)
        appendUniqueCandidate(candidates, seen, metadata._name_or_path)
      }
    }
    appendUniqueCandidate(candidates, seen, raw)

    if (candidates.length === 0) return null

    for (const candidate of candidates) {
      const normalized = normalizePath(candidate).toLowerCase()
      if (!normalized) continue
      if (FLUX2_UNSUPPORTED_VARIANT_MARKERS.some((marker) => normalized.includes(marker))) {
        return null
      }
      if (FLUX2_BASE_VARIANT_MARKERS.some((marker) => normalized.includes(marker))) {
        return 'base'
      }
    }

    return 'distilled'
  }

  function resolveLtxCheckpointExecutionMetadata(
    source: string | ModelInfo | null | undefined,
  ): LtxCheckpointExecutionMetadata | null {
    const model = typeof source === 'string' ? resolveModelInfo(source) : source
    if (!model) return null
    const metadata = isRecordObject(model.metadata) ? model.metadata : null
    if (!metadata) return null

    const checkpointKind = normalizeLtxCheckpointKind(metadata.ltx_checkpoint_kind)
    if (!checkpointKind) return null

    const allowedExecutionProfiles = parseLtxExecutionProfiles(metadata.ltx_allowed_execution_profiles) ?? []
    const rawDefaultProfile = String(metadata.ltx_default_execution_profile || '').trim()
    const defaultExecutionProfile = rawDefaultProfile || null
    const defaultSteps = parseOptionalNonNegativeInteger(metadata.ltx_default_steps)
    const defaultGuidanceScale = parseOptionalNonNegativeNumber(metadata.ltx_default_guidance_scale)

    return {
      checkpointKind,
      allowedExecutionProfiles,
      defaultExecutionProfile,
      defaultSteps,
      defaultGuidanceScale,
    }
  }

  function resolveModelSha(label: string | null | undefined): string | undefined {
    const raw = String(label || '').trim()
    if (!raw) return undefined

    const lower = raw.toLowerCase()
    if ((lower.length === 10 || lower.length === 64) && /^[0-9a-f]+$/.test(lower)) {
      return lower
    }

    const model = resolveModelInfo(raw)
    const sha = model ? String(model.hash || '').trim() : ''
    return sha || undefined
  }

  function resolveVaeSha(label: string | null | undefined): string | undefined {
    const raw = normalizeVaeSelection(label)
    if (!raw) return undefined

    if (raw === DEFAULT_VAE_SELECTION || raw === NONE_VAE_SELECTION) {
      return undefined
    }
    const lower = raw.toLowerCase()
    if (lower.length === 64 && /^[0-9a-f]+$/.test(lower)) {
      return lower
    }

    const normalized = raw.replace(/\\+/g, '/')
    const withoutPrefix = normalized.includes('/') ? normalized.split('/').slice(1).join('/') : normalized
    const tail = normalized.split('/').pop() || ''
    return (
      vaeShaMap.value.get(normalized) ||
      vaeShaMap.value.get(withoutPrefix) ||
      vaeShaMap.value.get(tail)
    )
  }

  function resolveWanGgufSha(label: string | null | undefined): string | undefined {
    const raw = String(label || '').trim()
    if (!raw) return undefined

    const lower = raw.toLowerCase()
    if (lower.length === 64 && /^[0-9a-f]+$/.test(lower)) {
      return lower
    }

    const normalized = raw.replace(/\\+/g, '/')
    const tail = normalized.split('/').pop() || ''
    return wanGgufShaMap.value.get(normalized) || wanGgufShaMap.value.get(tail)
  }

  function resolveWanGgufVariant(label: string | null | undefined): WanGgufVariant | undefined {
    const raw = String(label || '').trim()
    if (!raw) return undefined
    const normalized = raw.replace(/\\+/g, '/')
    const tail = normalized.split('/').pop() || ''
    return wanGgufVariantMap.value.get(normalized) || wanGgufVariantMap.value.get(tail)
  }

  function resolveLoraSha(label: string | null | undefined): string | undefined {
    const raw = String(label || '').trim()
    if (!raw) return undefined

    const lower = raw.toLowerCase()
    if (lower.length === 64 && /^[0-9a-f]+$/.test(lower)) {
      return lower
    }

    const normalized = raw.replace(/\\+/g, '/')
    const withoutPrefix = normalized.includes('/') ? normalized.split('/').slice(1).join('/') : normalized
    const tail = normalized.split('/').pop() || ''
    return (
      loraShaMap.value.get(raw) ||
      loraShaMap.value.get(normalized) ||
      loraShaMap.value.get(withoutPrefix) ||
      loraShaMap.value.get(tail)
    )
  }

  function isModelCoreOnly(label: string | null | undefined): boolean {
    const raw = String(label || '').trim()
    if (!raw) return false

    const model = resolveModelInfo(raw)
    if (!model) {
      return false
    }
    return typeof model.core_only === 'boolean' ? model.core_only : false
  }

  async function setVae(label: string): Promise<void> {
    const normalized = normalizeVaeSelection(label)
    currentVae.value = normalized
    saveVaeToStorage(normalized)
  }

  async function setVaeForFamily(
    family: string,
    label: string,
    options?: { expectedAssetSnapshotEpoch?: number; persist?: boolean },
  ): Promise<void> {
    const normalized = normalizeVaeSelection(label) || DEFAULT_VAE_SELECTION
    const normalizedFamily = String(family || '').trim().toLowerCase()
    const expectedAssetSnapshotEpoch = options?.expectedAssetSnapshotEpoch
    const persist = options?.persist !== false
    if (
      typeof expectedAssetSnapshotEpoch === 'number'
      && Number.isSafeInteger(expectedAssetSnapshotEpoch)
      && expectedAssetSnapshotEpoch !== assetSnapshotEpoch
    ) {
      return
    }
    if (!isVaeFamily(normalizedFamily)) {
      currentVae.value = normalized
      saveVaeToStorage(normalized)
      return
    }

    const previousCurrentVae = currentVae.value
    const previousVaeByFamily = { ...vaeByFamily.value }
    const nextVaeByFamily: Partial<Record<VaeFamily, string>> = {
      ...vaeByFamily.value,
      [normalizedFamily]: normalized,
    }

    currentVae.value = normalized
    saveVaeToStorage(normalized)
    vaeByFamily.value = nextVaeByFamily
    saveVaeByFamilyToStorage(nextVaeByFamily)
    if (!persist) {
      return
    }

    try {
      await applyOptionUpdate({
        [VAE_BY_FAMILY_OPTION_KEY]: serializeVaeByFamilyOption(nextVaeByFamily),
      })
    } catch (error) {
      if (
        typeof expectedAssetSnapshotEpoch === 'number'
        && Number.isSafeInteger(expectedAssetSnapshotEpoch)
        && expectedAssetSnapshotEpoch !== assetSnapshotEpoch
      ) {
        return
      }
      if (
        typeof expectedAssetSnapshotEpoch !== 'number'
        || !Number.isSafeInteger(expectedAssetSnapshotEpoch)
        || expectedAssetSnapshotEpoch === assetSnapshotEpoch
      ) {
        currentVae.value = previousCurrentVae
        saveVaeToStorage(previousCurrentVae)
        vaeByFamily.value = previousVaeByFamily
        saveVaeByFamilyToStorage(previousVaeByFamily)
      }
      throw error
    }
  }

  async function restoreVaeForFamilyOwner(
    family: string,
    persistedLabel: string,
    options?: { expectedAssetSnapshotEpoch?: number; persist?: boolean },
  ): Promise<void> {
    const normalizedFamily = String(family || '').trim().toLowerCase()
    const normalizedPersisted = normalizeVaeSelection(persistedLabel) || ''
    const expectedAssetSnapshotEpoch = options?.expectedAssetSnapshotEpoch
    const persist = options?.persist !== false
    if (
      typeof expectedAssetSnapshotEpoch === 'number'
      && Number.isSafeInteger(expectedAssetSnapshotEpoch)
      && expectedAssetSnapshotEpoch !== assetSnapshotEpoch
    ) {
      return
    }
    if (!isVaeFamily(normalizedFamily)) {
      currentVae.value = normalizedPersisted || DEFAULT_VAE_SELECTION
      saveVaeToStorage(currentVae.value)
      return
    }

    const previousCurrentVae = currentVae.value
    const previousVaeByFamily = { ...vaeByFamily.value }
    const nextVaeByFamily: Partial<Record<VaeFamily, string>> = { ...vaeByFamily.value }
    if (normalizedPersisted) {
      nextVaeByFamily[normalizedFamily] = normalizedPersisted
    } else {
      delete nextVaeByFamily[normalizedFamily]
    }
    const nextCurrentVae = normalizedPersisted || DEFAULT_VAE_SELECTION

    currentVae.value = nextCurrentVae
    saveVaeToStorage(nextCurrentVae)
    vaeByFamily.value = nextVaeByFamily
    saveVaeByFamilyToStorage(nextVaeByFamily)
    if (!persist) {
      return
    }

    try {
      await applyOptionUpdate({
        [VAE_BY_FAMILY_OPTION_KEY]: serializeVaeByFamilyOption(nextVaeByFamily),
      })
    } catch (error) {
      if (
        typeof expectedAssetSnapshotEpoch === 'number'
        && Number.isSafeInteger(expectedAssetSnapshotEpoch)
        && expectedAssetSnapshotEpoch !== assetSnapshotEpoch
      ) {
        return
      }
      if (
        typeof expectedAssetSnapshotEpoch !== 'number'
        || !Number.isSafeInteger(expectedAssetSnapshotEpoch)
        || expectedAssetSnapshotEpoch === assetSnapshotEpoch
      ) {
        currentVae.value = previousCurrentVae
        saveVaeToStorage(previousCurrentVae)
        vaeByFamily.value = previousVaeByFamily
        saveVaeByFamilyToStorage(previousVaeByFamily)
      }
      throw error
    }
  }

  function requireVaeSelection(label?: string | null): string {
    const normalized = normalizeVaeSelection(label ?? currentVae.value)
    if (!normalized) {
      throw new Error('Select a VAE before generating.')
    }
    return normalized
  }

  async function setMainDevice(value: string): Promise<void> {
    const normalizedValue = normalizeMainDeviceSetting(value)
    const previousMainDevice = mainDevice.value
    const previousCurrentDevice = currentDevice.value
    const previousAppliedNowMessages = lastAppliedNowMessages.value.slice()
    const previousRestartRequiredMessages = lastRestartRequiredMessages.value.slice()
    mainDevice.value = normalizedValue
    try {
      await applyOptionUpdate({ codex_main_device: normalizedValue })
      const mainDeviceRestartRequired = lastRestartRequiredMessages.value.some((message) =>
        String(message || '').startsWith('codex_main_device:'),
      )
      if (normalizedValue === 'auto' || mainDeviceRestartRequired) {
        await syncCurrentDeviceFromBackendAuthority()
        return
      }
      currentDevice.value = normalizedValue
      saveDeviceToStorage(normalizedValue)
    } catch (error) {
      mainDevice.value = previousMainDevice
      currentDevice.value = previousCurrentDevice
      lastAppliedNowMessages.value = previousAppliedNowMessages
      lastRestartRequiredMessages.value = previousRestartRequiredMessages
      if (previousCurrentDevice) {
        saveDeviceToStorage(previousCurrentDevice)
      } else {
        clearDeviceStorage()
      }
      throw error
    }
  }

  async function setCoreDtype(value: string): Promise<void> {
    coreDtype.value = value
    await applyOptionUpdate({ codex_core_dtype: value })
  }

  async function setCoreComputeDtype(value: string): Promise<void> {
    coreComputeDtype.value = value
    await applyOptionUpdate({ codex_core_compute_dtype: value })
  }

  async function setTeDtype(value: string): Promise<void> {
    teDtype.value = value
    await applyOptionUpdate({ codex_te_dtype: value })
  }

  async function setTeComputeDtype(value: string): Promise<void> {
    teComputeDtype.value = value
    await applyOptionUpdate({ codex_te_compute_dtype: value })
  }

  async function setVaeDtype(value: string): Promise<void> {
    vaeDtype.value = value
    await applyOptionUpdate({ codex_vae_dtype: value })
  }

  async function setVaeComputeDtype(value: string): Promise<void> {
    vaeComputeDtype.value = value
    await applyOptionUpdate({ codex_vae_compute_dtype: value })
  }

  async function setSmartOffload(value: boolean): Promise<void> {
    smartOffload.value = value
    await applyOptionUpdate({ codex_smart_offload: value })
  }

  async function setSmartFallback(value: boolean): Promise<void> {
    smartFallback.value = value
    await applyOptionUpdate({ codex_smart_fallback: value })
  }

  async function setSmartCache(value: boolean): Promise<void> {
    smartCache.value = value
    await applyOptionUpdate({ codex_smart_cache: value })
  }

  async function setCoreStreaming(value: boolean): Promise<void> {
    coreStreaming.value = value
    await applyOptionUpdate({ codex_core_streaming: value })
  }

  return {
    models,
    modelsFreshness,
    currentModel,
    vaeChoices,
    inventoryVaesSnapshot,
    pathsConfigSnapshot,
    currentVae,
    textEncoderChoices,
    currentTextEncoders,
    ipAdapterModelChoices,
    ipAdapterImageEncoderChoices,
    deviceChoices,
    currentDevice,
    init,
    invalidateModelsList,
    refreshModelsList,
    bumpAssetSnapshotEpoch,
    getAssetSnapshotEpoch,
    setVae,
    setVaeForFamily,
    restoreVaeForFamilyOwner,
    getVaeForFamily,
    getPersistedVaeForFamily,
    mainDevice,
    coreDtype,
    coreComputeDtype,
    teDtype,
    teComputeDtype,
    vaeDtype,
    vaeComputeDtype,
    dtypeChoices,
    setMainDevice,
    setCoreDtype,
    setCoreComputeDtype,
    setTeDtype,
    setTeComputeDtype,
    setVaeDtype,
    setVaeComputeDtype,
    smartOffload,
    smartFallback,
    smartCache,
    coreStreaming,
    lastAppliedNowMessages,
    lastRestartRequiredMessages,
    setSmartOffload,
    setSmartFallback,
    setSmartCache,
    setCoreStreaming,
    getSettingsRevision,
    refreshSettingsRevision,
    clearOptionApplyMessages,
    // SHA maps for asset resolution
    textEncoderShaMap,
    resolveTextEncoderSha,
    resolveTextEncoderSlot,
    resolveModelInfo,
    requireModelInfo,
    resolveFlux2CheckpointVariant,
    resolveLtxCheckpointExecutionMetadata,
    resolveModelSha,
    resolveVaeSha,
    requireVaeSelection,
    resolveLoraSha,
    resolveWanGgufSha,
    resolveWanGgufVariant,
    isModelCoreOnly,
    hydrateLoraShaMap,
    hydratePathsSnapshot,
    hydrateVaeInventorySnapshot,
    fetchInventoryWithLoraHydration,
    vaeShaMap,
    loraShaMap,
    wanGgufShaMap,
    wanGgufVariantMap,
  }
})
