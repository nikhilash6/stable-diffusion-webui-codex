/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Pinia store for backend engine capability gating.
Fetches `/api/engines/capabilities` and exposes cached capability + family + asset-contract + backend-owned dependency-check maps so views/components can gate
UI features, required asset selection, family-specific behavior, readiness indicators, family-scoped sampler/scheduler filtering, and the LTX-only
execution-profile/default surface from a single contract surface. The dependency-check helpers also resolve mode-scoped readiness rows
so SDXL `fooocus_inpaint` can stay code-supported while still blocking only that runtime mode when its dedicated assets are missing.
Exact engine-id and sampling-default truth is parsed from backend capabilities; frontend catalog filters validate explicit/default selections but never synthesize executable sampler defaults.

Symbols (top-level; keep in sync; no ghosts):
- `parseCapabilityInpaintMode` (function): Validates one inpaint-mode token from backend capability payloads against the canonical UI enum.
- `asEngineDependencyCheckRow` (function): Validates/coerces one dependency-check row from unknown payload data.
- `asEngineDependencyStatus` (function): Validates/coerces one dependency status payload per semantic engine.
- `parseDependencyChecks` (function): Parses strict `dependency_checks` map from capabilities response.
- `getApplicableDependencyChecks` / `isDependencyReady` / `firstDependencyError` (store helpers): Resolve global + mode-scoped dependency readiness for one engine surface.
- `SamplingDefaults` (interface): Backend-owned sampler/scheduler default pair returned only when both values are present.
- `parseEngineIdToSemanticMap` (function): Parses strict `engine_id_to_semantic_engine` map from capabilities response.
- `parseParkedExactEngines` (function): Parses strict `parked_exact_engines` map from capabilities response.
- `parseExactEngineInpaintModes` (function): Parses strict `exact_engine_inpaint_modes` map from capabilities response.
- `asLtxExecutionSurface` (function): Parses the optional nested LTX execution-profile/default surface from one engine capability row.
- `filterSamplersForFamilyCapabilities` (function): Applies family `supported_samplers`/`excluded_samplers` constraints to executable sampler rows.
- `filterSchedulersForFamilyCapabilities` (function): Applies family `supported_schedulers`/`excluded_schedulers` constraints to executable scheduler rows.
- `filterSchedulersForSampler` (function): Filters scheduler rows by sampler `allowed_schedulers` compatibility.
- `normalizeSamplerSchedulerSelection` (function): Resolves a valid sampler/scheduler pair against executable catalogs + family + sampler compatibility constraints.
- `parseFamilyCapabilities` (function): Parses strict `families` capability map from capabilities response.
- `useEngineCapabilitiesStore` (store): Pinia store exposing engine capabilities, load state, and lookup helpers (including `getLtxExecutionSurface(...)`).
*/

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  EngineAssetContract,
  EngineAssetContractVariants,
  EngineCapabilitiesResponse,
  EngineCapabilities,
  FamilyCapabilities,
  EngineDependencyStatus,
  EngineDependencyCheckRow,
  LtxExecutionSurface,
  ParkedExactEngineStatus,
  SamplerInfo,
  SchedulerInfo,
} from '../api/types'
import { fetchEngineCapabilities } from '../api/client'
import { normalizeSemanticEngine, resolveSemanticEngineForEngineId } from '../utils/engine_taxonomy'
import { parseInpaintMode, type InpaintMode } from '../utils/image_params'

const CAPABILITIES_CONTRACT_ERROR_PREFIX = "Invalid '/api/engines/capabilities' response:"

type FamilySamplingListKey =
  | 'supported_samplers'
  | 'supported_schedulers'
  | 'excluded_samplers'
  | 'excluded_schedulers'

export interface SamplingDefaults {
  sampler: string
  scheduler: string
}

function toUniqueNonEmptyStrings(values: Array<string | null | undefined>): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const rawValue of values) {
    const value = String(rawValue || '').trim()
    if (!value || seen.has(value)) continue
    seen.add(value)
    out.push(value)
  }
  return out
}

function parseCapabilityInpaintMode(
  rawValue: string,
  context: string,
): InpaintMode {
  const mode = parseInpaintMode(rawValue.trim())
  if (mode) return mode
  throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} ${context} has unsupported inpaint mode '${rawValue}'.`)
}

function parseFamilySamplingList(
  row: Record<string, unknown>,
  family: string,
  field: FamilySamplingListKey,
): string[] | null | undefined {
  const raw = row[field]
  if (typeof raw === 'undefined') return undefined
  if (raw === null) return null
  if (!Array.isArray(raw)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} family capability '${family}' has non-array '${field}'.`)
  }
  const normalized: string[] = []
  const seen = new Set<string>()
  for (const [index, entry] of raw.entries()) {
    if (typeof entry !== 'string') {
      throw new Error(
        `${CAPABILITIES_CONTRACT_ERROR_PREFIX} family capability '${family}' has non-string '${field}[${index}]'.`,
      )
    }
    const value = entry.trim()
    if (!value) {
      throw new Error(
        `${CAPABILITIES_CONTRACT_ERROR_PREFIX} family capability '${family}' has empty '${field}[${index}]'.`,
      )
    }
    if (seen.has(value)) continue
    seen.add(value)
    normalized.push(value)
  }
  return normalized
}

function capabilitySet(values: string[] | null | undefined): Set<string> | null {
  if (!Array.isArray(values)) return null
  if (values.length === 0) return null
  return new Set(values)
}

function asLtxExecutionSurface(
  value: unknown,
  engine: string,
): LtxExecutionSurface | null {
  if (value == null) return null
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has non-object 'ltx_execution_surface'.`)
  }
  const row = value as Record<string, unknown>
  const allowedExecutionProfiles = row.allowed_execution_profiles
  const defaultExecutionProfile = row.default_execution_profile
  const defaultStepsByProfile = row.default_steps_by_profile
  const defaultGuidanceByProfile = row.default_guidance_scale_by_profile
  if (!Array.isArray(allowedExecutionProfiles) || allowedExecutionProfiles.some((entry) => typeof entry !== 'string' || !entry.trim())) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has invalid 'ltx_execution_surface.allowed_execution_profiles'.`)
  }
  if (typeof defaultExecutionProfile !== 'string' || !defaultExecutionProfile.trim()) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has invalid 'ltx_execution_surface.default_execution_profile'.`)
  }
  if (defaultStepsByProfile === null || typeof defaultStepsByProfile !== 'object' || Array.isArray(defaultStepsByProfile)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has invalid 'ltx_execution_surface.default_steps_by_profile'.`)
  }
  if (defaultGuidanceByProfile === null || typeof defaultGuidanceByProfile !== 'object' || Array.isArray(defaultGuidanceByProfile)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has invalid 'ltx_execution_surface.default_guidance_scale_by_profile'.`)
  }
  const stepsOut: Record<string, number> = {}
  for (const [profile, rawValue] of Object.entries(defaultStepsByProfile as Record<string, unknown>)) {
    const normalizedProfile = String(profile || '').trim()
    if (!normalizedProfile) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has empty LTX profile key in 'default_steps_by_profile'.`)
    }
    if (typeof rawValue !== 'number' || !Number.isFinite(rawValue) || !Number.isInteger(rawValue) || rawValue <= 0) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has invalid default steps for LTX profile '${normalizedProfile}'.`)
    }
    stepsOut[normalizedProfile] = rawValue
  }
  const guidanceOut: Record<string, number> = {}
  for (const [profile, rawValue] of Object.entries(defaultGuidanceByProfile as Record<string, unknown>)) {
    const normalizedProfile = String(profile || '').trim()
    if (!normalizedProfile) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has empty LTX profile key in 'default_guidance_scale_by_profile'.`)
    }
    if (typeof rawValue !== 'number' || !Number.isFinite(rawValue) || rawValue < 0) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has invalid default guidance for LTX profile '${normalizedProfile}'.`)
    }
    guidanceOut[normalizedProfile] = rawValue
  }
  return {
    allowed_execution_profiles: allowedExecutionProfiles.map((entry) => String(entry).trim()),
    default_execution_profile: defaultExecutionProfile.trim(),
    default_steps_by_profile: stepsOut,
    default_guidance_scale_by_profile: guidanceOut,
  }
}

export function filterSamplersForFamilyCapabilities(
  samplers: SamplerInfo[],
  familyCapabilities: FamilyCapabilities | null | undefined,
): SamplerInfo[] {
  const supported = capabilitySet(familyCapabilities?.supported_samplers)
  const excluded = capabilitySet(familyCapabilities?.excluded_samplers)
  return samplers.filter((entry) => {
    if (supported && !supported.has(entry.name)) return false
    if (excluded && excluded.has(entry.name)) return false
    return true
  })
}

export function filterSchedulersForFamilyCapabilities(
  schedulers: SchedulerInfo[],
  familyCapabilities: FamilyCapabilities | null | undefined,
): SchedulerInfo[] {
  const supported = capabilitySet(familyCapabilities?.supported_schedulers)
  const excluded = capabilitySet(familyCapabilities?.excluded_schedulers)
  return schedulers.filter((entry) => {
    if (supported && !supported.has(entry.name)) return false
    if (excluded && excluded.has(entry.name)) return false
    return true
  })
}

export function filterSchedulersForSampler(
  schedulers: SchedulerInfo[],
  sampler: SamplerInfo | null | undefined,
): SchedulerInfo[] {
  if (!sampler) return schedulers.slice()
  const allowed = Array.isArray(sampler.allowed_schedulers)
    ? sampler.allowed_schedulers.map((entry) => String(entry || '').trim()).filter((entry) => entry.length > 0)
    : []
  if (allowed.length === 0) return schedulers.slice()
  const allowedSet = new Set(allowed)
  return schedulers.filter((entry) => allowedSet.has(entry.name))
}

export function normalizeSamplerSchedulerSelection(opts: {
  samplers: SamplerInfo[]
  schedulers: SchedulerInfo[]
  familyCapabilities: FamilyCapabilities | null | undefined
  sampler: string | null | undefined
  scheduler: string | null | undefined
  preferredSamplers?: Array<string | null | undefined>
  preferredSchedulers?: Array<string | null | undefined>
}): { sampler: string; scheduler: string } | null {
  const familySamplers = filterSamplersForFamilyCapabilities(opts.samplers, opts.familyCapabilities)
  const familySchedulers = filterSchedulersForFamilyCapabilities(opts.schedulers, opts.familyCapabilities)
  if (familySamplers.length === 0 || familySchedulers.length === 0) return null

  const samplerByName = new Map(familySamplers.map((entry) => [entry.name, entry]))
  const tryPair = (
    candidateSamplerRaw: string | null | undefined,
    candidateSchedulerRaw: string | null | undefined,
  ): { sampler: string; scheduler: string } | null => {
    const candidateSampler = String(candidateSamplerRaw || '').trim()
    const candidateScheduler = String(candidateSchedulerRaw || '').trim()
    if (!candidateSampler || !candidateScheduler) return null
    const samplerSpec = samplerByName.get(candidateSampler)
    if (!samplerSpec) return null
    const allowedSchedulers = filterSchedulersForSampler(familySchedulers, samplerSpec)
    if (allowedSchedulers.length === 0) return null
    const allowedSchedulerSet = new Set(allowedSchedulers.map((entry) => entry.name))
    if (!allowedSchedulerSet.has(candidateScheduler)) return null
    return { sampler: samplerSpec.name, scheduler: candidateScheduler }
  }

  const explicitPair = tryPair(opts.sampler, opts.scheduler)
  if (explicitPair) return explicitPair

  const preferredSamplers = toUniqueNonEmptyStrings(opts.preferredSamplers ?? [])
  const preferredSchedulers = toUniqueNonEmptyStrings(opts.preferredSchedulers ?? [])
  const preferredCount = Math.max(preferredSamplers.length, preferredSchedulers.length)
  for (let index = 0; index < preferredCount; index += 1) {
    const preferredPair = tryPair(preferredSamplers[index], preferredSchedulers[index])
    if (preferredPair) return preferredPair
  }

  return null
}

function asEngineDependencyCheckRow(
  value: unknown,
  context: { index: number; engine: string },
): EngineDependencyCheckRow {
  const { index, engine } = context
  if (value === null || typeof value !== 'object') {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency check row #${index + 1} for '${engine}' must be an object.`)
  }
  const row = value as Record<string, unknown>
  const id = typeof row.id === 'string' ? row.id.trim() : ''
  const label = typeof row.label === 'string' ? row.label.trim() : ''
  const message = typeof row.message === 'string' ? row.message.trim() : ''
  if (!id) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency check row #${index + 1} for '${engine}' has missing 'id'.`)
  }
  if (!label) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency check row '${id}' for '${engine}' has missing 'label'.`)
  }
  if (typeof row.ok !== 'boolean') {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency check row '${id}' for '${engine}' has non-boolean 'ok'.`)
  }
  if (!message) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency check row '${id}' for '${engine}' has missing 'message'.`)
  }
  let inpaintModes: InpaintMode[] | undefined
  if (row.inpaint_modes !== undefined) {
    if (!Array.isArray(row.inpaint_modes)) {
      throw new Error(
        `${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency check row '${id}' for '${engine}' has non-array 'inpaint_modes'.`,
      )
    }
    inpaintModes = toUniqueNonEmptyStrings(
      row.inpaint_modes.map((entry, modeIndex) => {
        if (typeof entry !== 'string') {
          throw new Error(
            `${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency check row '${id}' for '${engine}' has non-string 'inpaint_modes[${modeIndex}]'.`,
          )
        }
        return parseCapabilityInpaintMode(
          entry,
          `dependency check row '${id}' for '${engine}' 'inpaint_modes[${modeIndex}]'`,
        )
      }),
    ) as InpaintMode[]
  }
  return { id, label, ok: row.ok, message, inpaint_modes: inpaintModes }
}

function asEngineDependencyStatus(
  value: unknown,
  context: { engine: string },
): EngineDependencyStatus {
  const { engine } = context
  if (value === null || typeof value !== 'object') {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency status for '${engine}' must be an object.`)
  }
  const status = value as Record<string, unknown>
  if (typeof status.ready !== 'boolean') {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency status for '${engine}' has non-boolean 'ready'.`)
  }
  if (!Array.isArray(status.checks)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency status for '${engine}' has missing 'checks' array.`)
  }
  const checks = status.checks.map((row, index) => asEngineDependencyCheckRow(row, { index, engine }))
  const derivedGlobalReady = checks
    .filter((row) => !Array.isArray(row.inpaint_modes) || row.inpaint_modes.length === 0)
    .every((row) => row.ok)
  if (derivedGlobalReady !== status.ready) {
    throw new Error(
      `${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency status for '${engine}' is inconsistent (ready=${String(status.ready)} but global checks imply ready=${String(derivedGlobalReady)}).`,
    )
  }
  return { ready: derivedGlobalReady, checks }
}

function parseDependencyChecks(payload: unknown): Record<string, EngineDependencyStatus> {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} missing 'dependency_checks' object.`)
  }
  const raw = payload as Record<string, unknown>
  const out: Record<string, EngineDependencyStatus> = {}
  for (const [engine, status] of Object.entries(raw)) {
    out[engine] = asEngineDependencyStatus(status, { engine })
  }
  return out
}

function parseEngineIdToSemanticMap(payload: unknown): Record<string, string> {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} missing 'engine_id_to_semantic_engine' object.`)
  }
  const raw = payload as Record<string, unknown>
  const out: Record<string, string> = {}
  for (const [keyRaw, valueRaw] of Object.entries(raw)) {
    const engineId = String(keyRaw || '').trim().toLowerCase()
    const semanticRaw = String(valueRaw || '').trim()
    if (!engineId) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine_id_to_semantic_engine has an empty key.`)
    }
    if (Object.prototype.hasOwnProperty.call(out, engineId)) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine_id_to_semantic_engine has duplicate normalized key '${engineId}'.`)
    }
    const semantic = normalizeSemanticEngine(semanticRaw)
    if (!semantic) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} invalid semantic engine '${semanticRaw}' for engine id '${engineId}'.`)
    }
    out[engineId] = semantic
  }
  return out
}

function parseParkedExactEngines(
  payload: unknown,
  activeExactEngineIds: ReadonlySet<string>,
): Set<string> {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} missing 'parked_exact_engines' object.`)
  }
  const raw = payload as Record<string, unknown>
  const out = new Set<string>()
  for (const [keyRaw, valueRaw] of Object.entries(raw)) {
    const engineId = String(keyRaw || '').trim().toLowerCase()
    if (!engineId) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} parked_exact_engines has an empty key.`)
    }
    if (out.has(engineId)) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} parked_exact_engines has duplicate normalized key '${engineId}'.`)
    }
    if (activeExactEngineIds.has(engineId)) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} parked_exact_engines['${engineId}'] overlaps an active engine id.`)
    }
    if (valueRaw === null || typeof valueRaw !== 'object' || Array.isArray(valueRaw)) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} parked_exact_engines['${engineId}'] must be an object.`)
    }
    const row = valueRaw as Partial<ParkedExactEngineStatus>
    if (row.status !== 'not_implemented') {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} parked_exact_engines['${engineId}'].status must be 'not_implemented'.`)
    }
    const detail = typeof row.detail === 'string' ? row.detail.trim() : ''
    if (!detail) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} parked_exact_engines['${engineId}'].detail must not be empty.`)
    }
    out.add(engineId)
  }
  return out
}

function parseExactEngineInpaintModes(
  payload: unknown,
  engineIdToSemanticEngine: Record<string, string>,
  parkedExactEngineIds: ReadonlySet<string>,
): Record<string, InpaintMode[]> {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} missing 'exact_engine_inpaint_modes' object.`)
  }
  const raw = payload as Record<string, unknown>
  const out: Record<string, InpaintMode[]> = {}
  const activeExactEngineIds = new Set(Object.keys(engineIdToSemanticEngine))
  for (const [keyRaw, valueRaw] of Object.entries(raw)) {
    const engineId = String(keyRaw || '').trim().toLowerCase()
    if (!engineId) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} exact_engine_inpaint_modes has an empty key.`)
    }
    if (Object.prototype.hasOwnProperty.call(out, engineId)) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} exact_engine_inpaint_modes has duplicate normalized key '${engineId}'.`)
    }
    if (!activeExactEngineIds.has(engineId) && !parkedExactEngineIds.has(engineId)) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} exact_engine_inpaint_modes has unadvertised exact engine id '${engineId}'.`)
    }
    if (!Array.isArray(valueRaw)) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} exact_engine_inpaint_modes['${engineId}'] must be an array.`)
    }
    const modes = valueRaw.map((entry, index) => {
      if (typeof entry !== 'string') {
        throw new Error(
          `${CAPABILITIES_CONTRACT_ERROR_PREFIX} exact_engine_inpaint_modes['${engineId}'][${index}] must be a string.`,
        )
      }
      const mode = entry.trim()
      if (!mode) {
        throw new Error(
          `${CAPABILITIES_CONTRACT_ERROR_PREFIX} exact_engine_inpaint_modes['${engineId}'][${index}] must not be empty.`,
        )
      }
      return parseCapabilityInpaintMode(
        mode,
        `exact_engine_inpaint_modes['${engineId}'][${index}]`,
      )
    })
    out[engineId] = toUniqueNonEmptyStrings(modes) as InpaintMode[]
  }
  for (const engineId of activeExactEngineIds) {
    if (!Object.prototype.hasOwnProperty.call(out, engineId)) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} exact_engine_inpaint_modes is missing backend engine id '${engineId}'.`)
    }
  }
  return out
}

function parseFamilyCapabilities(payload: unknown): Record<string, FamilyCapabilities> {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} missing 'families' object.`)
  }
  const raw = payload as Record<string, unknown>
  const out: Record<string, FamilyCapabilities> = {}
  for (const [family, value] of Object.entries(raw)) {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} family capability '${family}' must be an object.`)
    }
    const row = value as Record<string, unknown>
    const supportsNegative = row.supports_negative_prompt
    const showsClipSkip = row.shows_clip_skip
    const supportedSamplers = parseFamilySamplingList(row, family, 'supported_samplers')
    const supportedSchedulers = parseFamilySamplingList(row, family, 'supported_schedulers')
    const excludedSamplers = parseFamilySamplingList(row, family, 'excluded_samplers')
    const excludedSchedulers = parseFamilySamplingList(row, family, 'excluded_schedulers')
    if (typeof supportsNegative !== 'boolean') {
      throw new Error(
        `${CAPABILITIES_CONTRACT_ERROR_PREFIX} family capability '${family}' has non-boolean 'supports_negative_prompt'.`,
      )
    }
    if (typeof showsClipSkip !== 'boolean') {
      throw new Error(
        `${CAPABILITIES_CONTRACT_ERROR_PREFIX} family capability '${family}' has non-boolean 'shows_clip_skip'.`,
      )
    }
    out[family] = {
      supports_negative_prompt: supportsNegative,
      shows_clip_skip: showsClipSkip,
      supported_samplers: supportedSamplers,
      supported_schedulers: supportedSchedulers,
      excluded_samplers: excludedSamplers,
      excluded_schedulers: excludedSchedulers,
    }
  }
  return out
}

export const useEngineCapabilitiesStore = defineStore('engineCapabilities', () => {
  const engines = ref<Record<string, EngineCapabilities>>({})
  const families = ref<Record<string, FamilyCapabilities>>({})
  const assetContracts = ref<Record<string, EngineAssetContractVariants>>({})
  const dependencyChecks = ref<Record<string, EngineDependencyStatus>>({})
  const engineIdToSemanticEngine = ref<Record<string, string>>({})
  const exactEngineInpaintModes = ref<Record<string, InpaintMode[]>>({})
  const loaded = ref(false)
  const loading = ref(false)
  const error = ref<string | null>(null)
  let initPromise: Promise<void> | null = null

  async function init(opts: { force?: boolean } = {}): Promise<void> {
    const force = Boolean(opts.force)
    if (!force && loaded.value) return
    if (initPromise) return initPromise

    initPromise = (async () => {
      loading.value = true
      error.value = null
      try {
        const res: EngineCapabilitiesResponse = await fetchEngineCapabilities()
        if (!res || typeof res !== 'object') {
          throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} payload must be an object.`)
        }
        if (!res.engines || typeof res.engines !== 'object' || Array.isArray(res.engines)) {
          throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} missing 'engines' object.`)
        }
        const nextDependencyChecks = parseDependencyChecks(res.dependency_checks)
        const nextEngineMap = parseEngineIdToSemanticMap(res.engine_id_to_semantic_engine)
        const nextParkedExactEngineIds = parseParkedExactEngines(
          res.parked_exact_engines,
          new Set(Object.keys(nextEngineMap)),
        )
        const nextFamilies = parseFamilyCapabilities(res.families)
        const nextInpaintModes = parseExactEngineInpaintModes(
          res.exact_engine_inpaint_modes,
          nextEngineMap,
          nextParkedExactEngineIds,
        )

        engines.value = res.engines
        families.value = nextFamilies
        assetContracts.value = res.asset_contracts ?? {}
        dependencyChecks.value = nextDependencyChecks
        engineIdToSemanticEngine.value = nextEngineMap
        exactEngineInpaintModes.value = nextInpaintModes
        loaded.value = true
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : String(e)
        error.value = message
        loaded.value = false
        throw e
      } finally {
        loading.value = false
      }
    })()

    try {
      await initPromise
    } finally {
      initPromise = null
      }
    }

  function semanticEngineForId(engineId: string | null | undefined): string | null {
    if (!engineId) return null
    return resolveSemanticEngineForEngineId(engineId, engineIdToSemanticEngine.value)
  }

  function get(engine: string | null | undefined): EngineCapabilities | null {
    const semantic = semanticEngineForId(engine)
    if (!semantic) return null
    return engines.value[semantic] ?? null
  }

  function getInpaintModes(engineId: string | null | undefined): InpaintMode[] {
    const key = String(engineId || '').trim().toLowerCase()
    if (!key) return []
    const modes = exactEngineInpaintModes.value[key]
    return Array.isArray(modes) ? modes.slice() : []
  }

  function getAssetVariants(engine: string | null | undefined): EngineAssetContractVariants | null {
    const semantic = semanticEngineForId(engine)
    if (!semantic) return null
    return assetContracts.value[semantic] ?? null
  }

  function getDependencyStatus(engine: string | null | undefined): EngineDependencyStatus | null {
    const semantic = semanticEngineForId(engine)
    if (!semantic) return null
    return dependencyChecks.value[semantic] ?? null
  }

  function getApplicableDependencyChecks(
    engine: string | null | undefined,
    opts: { inpaintMode?: string | null } = {},
  ): EngineDependencyCheckRow[] {
    const status = getDependencyStatus(engine)
    if (!status) return []
    const activeInpaintMode = String(opts.inpaintMode || '').trim()
    return status.checks.filter((row) => {
      const scopedModes = Array.isArray(row.inpaint_modes) ? row.inpaint_modes : []
      if (scopedModes.length === 0) return true
      if (!activeInpaintMode) return false
      return scopedModes.includes(activeInpaintMode)
    })
  }

  function isDependencyReady(
    engine: string | null | undefined,
    opts: { inpaintMode?: string | null } = {},
  ): boolean {
    const status = getDependencyStatus(engine)
    if (!status) return false
    return getApplicableDependencyChecks(engine, opts).every((row) => row.ok)
  }

  function getFamily(family: string | null | undefined): FamilyCapabilities | null {
    const key = String(family || '').trim().toLowerCase()
    if (!key) return null
    return families.value[key] ?? null
  }

  function getFamilyForEngine(engine: string | null | undefined): FamilyCapabilities | null {
    const semantic = semanticEngineForId(engine)
    if (!semantic) return null
    return getFamily(semantic)
  }

  function firstDependencyError(
    engine: string | null | undefined,
    opts: { inpaintMode?: string | null } = {},
  ): string {
    const status = getDependencyStatus(engine)
    if (!status) return "Dependency checks are not available for this engine."
    const first = getApplicableDependencyChecks(engine, opts).find((row) => !row.ok)
    return first?.message || ''
  }

  function getAssetContract(
    engine: string | null | undefined,
    opts: { checkpointCoreOnly: boolean }
  ): EngineAssetContract | null {
    const variants = getAssetVariants(engine)
    if (!variants) return null
    return opts?.checkpointCoreOnly ? variants.core_only : variants.base
  }

  function resolveSamplingDefaults(engineId: string | null | undefined): SamplingDefaults | null {
    const surface = get(engineId)
    const sampler = String(surface?.default_sampler || '').trim()
    const scheduler = String(surface?.default_scheduler || '').trim()
    if (!sampler || !scheduler) return null
    return { sampler, scheduler }
  }

  function getLtxExecutionSurface(engine: string | null | undefined): LtxExecutionSurface | null {
    const semantic = semanticEngineForId(engine)
    if (!semantic) return null
    return asLtxExecutionSurface(engines.value[semantic]?.ltx_execution_surface, semantic)
  }

  const knownEngines = computed(() => Object.keys(engines.value))
  const notReadyEngines = computed(() =>
    Object.entries(dependencyChecks.value)
      .filter(([, status]) => !status.ready)
      .map(([engine]) => engine),
  )

  return {
    engines,
    families,
    assetContracts,
    dependencyChecks,
    engineIdToSemanticEngine,
    exactEngineInpaintModes,
    knownEngines,
    notReadyEngines,
    loaded,
    loading,
    error,
    init,
    semanticEngineForId,
    get,
    getInpaintModes,
    getFamily,
    getFamilyForEngine,
    getAssetVariants,
    getDependencyStatus,
    getApplicableDependencyChecks,
    isDependencyReady,
    firstDependencyError,
    getAssetContract,
    getLtxExecutionSurface,
    resolveSamplingDefaults,
  }
})
