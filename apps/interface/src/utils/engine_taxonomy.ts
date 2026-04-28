/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical frontend engine/tab taxonomy helpers.
Centralizes tab-family aliases, exact video-lane detection, image request engine-id resolution, backend engine-id -> semantic-engine resolution,
and sampler/scheduler fallback defaults so stores/composables stop duplicating mapping tables. Fallback sampler/scheduler pairs must stay aligned
to the live executable surface (for example SD15 now defaults to `ddim` / `ddim`). FLUX.2 stays first-class in frontend taxonomy (no FLUX.1
aliasing), while backend-only semantic engines such as `netflix_void` remain valid semantic ids but intentionally resolve to no UI tab family.

Symbols (top-level; keep in sync; no ghosts):
- `TabFamily` (type): Canonical model tab families used by the UI.
- `VideoTabFamily` (type): Exact video tab families supported by the routed video workspace.
- `SemanticEngine` (type): Backend semantic engine ids from `/api/engines/capabilities`.
- `EngineRequestId` (type): Engine ids used in frontend payload dispatch (`flux1_kontext`, `flux1_chroma`, etc.).
- `normalizeTabFamily` (function): Normalizes raw alias values into `TabFamily` or `null`.
- `normalizeSemanticEngine` (function): Normalizes raw semantic-engine values into canonical `SemanticEngine` or `null`.
- `isWanTabFamily` (function): Returns whether a tab family is an exact WAN lane.
- `isVideoTabFamily` (function): Returns whether a tab family is a routed exact video lane.
- `tabFamilyFromSemanticEngine` (function): Converts semantic engine id to tab family when representable.
- `resolveImageRequestEngineId` (function): Canonical image request tab/mode -> engine-id mapper.
- `KNOWN_ENGINE_IDS` (constant): Known engine ids that must have valid semantic mapping.
- `isKnownEngineId` (function): Type guard for `KNOWN_ENGINE_IDS`.
- `resolveSemanticEngineForEngineId` (function): Resolves engine id to semantic id using backend map, failing loud for missing known mappings.
- `SamplingDefaults` (interface): Sampler/scheduler pair.
- `fallbackSamplingDefaultsForTabFamily` (function): Stable frontend fallback sampler/scheduler defaults by tab family.
*/

export type TabFamily = 'sd15' | 'sdxl' | 'flux1' | 'flux2' | 'chroma' | 'wan22_14b' | 'wan22_5b' | 'zimage' | 'anima' | 'ltx2'
export type VideoTabFamily = Extract<TabFamily, 'wan22_14b' | 'wan22_5b' | 'ltx2'>

export type SemanticEngine =
  | 'sd15'
  | 'sdxl'
  | 'flux1'
  | 'flux2'
  | 'zimage'
  | 'anima'
  | 'chroma'
  | 'wan22'
  | 'ltx2'
  | 'netflix_void'
  | 'hunyuan_video'
  | 'svd'

export type EngineRequestId =
  | SemanticEngine
  | 'flux1_chroma'
  | 'flux1_kontext'
  | 'flux1_fill'
  | 'wan22_14b'
  | 'wan22_5b'
  | 'ltx2'

const TAB_FAMILY_ALIASES: Readonly<Record<string, TabFamily>> = Object.freeze({
  sd15: 'sd15',
  sdxl: 'sdxl',
  flux1: 'flux1',
  flux2: 'flux2',
  chroma: 'chroma',
  zimage: 'zimage',
  anima: 'anima',
  ltx2: 'ltx2',
  wan22_14b: 'wan22_14b',
  wan22_5b: 'wan22_5b',
  flux1_chroma: 'chroma',
})

const SEMANTIC_ENGINE_SET: ReadonlySet<string> = new Set<string>([
  'sd15',
  'sdxl',
  'flux1',
  'flux2',
  'zimage',
  'anima',
  'chroma',
  'wan22',
  'ltx2',
  'netflix_void',
  'hunyuan_video',
  'svd',
])

const ENGINE_ID_SET: ReadonlySet<string> = new Set<string>([
  'sd15',
  'sdxl',
  'flux1',
  'flux2',
  'flux1_chroma',
  'flux1_kontext',
  'flux1_fill',
  'zimage',
  'anima',
  'ltx2',
  'wan22',
  'wan22_14b',
  'wan22_5b',
])

const TAB_FAMILY_FALLBACK_SAMPLING: Readonly<Record<TabFamily, SamplingDefaults>> = Object.freeze({
  sd15: { sampler: 'ddim', scheduler: 'ddim' },
  sdxl: { sampler: 'euler', scheduler: 'euler_discrete' },
  flux1: { sampler: 'euler', scheduler: 'simple' },
  flux2: { sampler: 'euler', scheduler: 'simple' },
  chroma: { sampler: 'euler', scheduler: 'simple' },
  zimage: { sampler: 'euler', scheduler: 'simple' },
  anima: { sampler: 'euler', scheduler: 'simple' },
  ltx2: { sampler: 'euler', scheduler: 'simple' },
  wan22_14b: { sampler: 'uni-pc', scheduler: 'simple' },
  wan22_5b: { sampler: 'uni-pc', scheduler: 'simple' },
})

function normalizeKey(value: unknown): string {
  return String(value || '').trim().toLowerCase()
}

export function normalizeSemanticEngine(value: unknown): SemanticEngine | null {
  const key = normalizeKey(value)
  if (!key) return null
  return SEMANTIC_ENGINE_SET.has(key) ? (key as SemanticEngine) : null
}

export const KNOWN_ENGINE_IDS: readonly EngineRequestId[] = Object.freeze(Array.from(ENGINE_ID_SET) as EngineRequestId[])

export function isKnownEngineId(value: unknown): value is EngineRequestId {
  const key = normalizeKey(value)
  if (!key) return false
  return ENGINE_ID_SET.has(key)
}

export function normalizeTabFamily(value: unknown): TabFamily | null {
  const key = normalizeKey(value)
  if (!key) return null
  return TAB_FAMILY_ALIASES[key] ?? null
}

export function isWanTabFamily(value: unknown): value is Extract<TabFamily, 'wan22_14b' | 'wan22_5b'> {
  return value === 'wan22_14b' || value === 'wan22_5b'
}

export function isVideoTabFamily(value: unknown): value is VideoTabFamily {
  return value === 'ltx2' || isWanTabFamily(value)
}

export function tabFamilyFromSemanticEngine(value: unknown): TabFamily | null {
  const semantic = normalizeSemanticEngine(value)
  if (!semantic) return null
  if (semantic === 'wan22') return null
  if (semantic === 'hunyuan_video' || semantic === 'svd' || semantic === 'netflix_void') return null
  return semantic
}

export function resolveImageRequestEngineId(tabType: string, useInitImage: boolean): EngineRequestId {
  const family = normalizeTabFamily(tabType)
  if (!family) {
    throw new Error(`Unsupported image tab type '${String(tabType)}'.`)
  }
  if (isWanTabFamily(family) || family === 'ltx2') {
    throw new Error(`Unsupported image tab type '${String(tabType)}'.`)
  }
  if (family === 'chroma') return 'flux1_chroma'
  if (family === 'flux1' && useInitImage) return 'flux1_kontext'
  return family
}

export function resolveSemanticEngineForEngineId(
  engineId: unknown,
  map: Record<string, string>,
): SemanticEngine | null {
  const id = normalizeKey(engineId)
  if (!id) return null

  const semanticDirect = normalizeSemanticEngine(id)
  if (semanticDirect) return semanticDirect

  const mappedRaw = typeof map[id] === 'string' ? map[id] : ''
  const mappedSemantic = normalizeSemanticEngine(mappedRaw)
  if (mappedSemantic) return mappedSemantic

  if (isKnownEngineId(id)) {
    throw new Error(`Missing or invalid semantic-engine mapping for known engine id '${id}'.`)
  }
  return null
}

export interface SamplingDefaults {
  sampler: string
  scheduler: string
}

export function fallbackSamplingDefaultsForTabFamily(family: TabFamily): SamplingDefaults {
  return TAB_FAMILY_FALLBACK_SAMPLING[family]
}
