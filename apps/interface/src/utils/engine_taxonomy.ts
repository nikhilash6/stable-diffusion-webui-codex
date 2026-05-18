/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical frontend engine/tab taxonomy helpers.
Centralizes tab-family aliases, exact video-lane detection, image request engine-id resolution, exact backend engine-id -> semantic-engine resolution,
and semantic/tab conversion so stores/composables stop duplicating mapping tables. FLUX.2 stays first-class in frontend taxonomy (no FLUX.1
aliasing), and Qwen Image is a first-class capability-gated image tab using the single canonical `qwen_image` id.

Symbols (top-level; keep in sync; no ghosts):
- `TabFamily` (type): Canonical model tab families used by the UI.
- `VideoTabFamily` (type): Exact video tab families supported by the routed video workspace.
- `SemanticEngine` (type): Backend semantic engine ids from `/api/engines/capabilities`.
- `EngineRequestId` (type): Exact backend request engine ids used in frontend payload dispatch (`flux1_kontext`, `flux1_chroma`, etc.).
- `normalizeTabFamily` (function): Normalizes raw alias values into `TabFamily` or `null`.
- `normalizeSemanticEngine` (function): Normalizes raw semantic-engine values into canonical `SemanticEngine` or `null`.
- `isWanTabFamily` (function): Returns whether a tab family is an exact WAN lane.
- `isVideoTabFamily` (function): Returns whether a tab family is a routed exact video lane.
- `tabFamilyFromSemanticEngine` (function): Converts semantic engine id to tab family when representable.
- `resolveImageRequestEngineId` (function): Canonical image request tab/mode -> engine-id mapper.
- `resolveSemanticEngineForEngineId` (function): Resolves engine id to semantic id using only the backend capability map.
*/

export type TabFamily = 'sd15' | 'sdxl' | 'flux1' | 'flux2' | 'chroma' | 'qwen_image' | 'wan22_14b' | 'wan22_5b' | 'zimage' | 'anima' | 'ltx2'
export type VideoTabFamily = Extract<TabFamily, 'wan22_14b' | 'wan22_5b' | 'ltx2'>

export type SemanticEngine =
  | 'sd15'
  | 'sdxl'
  | 'flux1'
  | 'flux2'
  | 'qwen_image'
  | 'zimage'
  | 'anima'
  | 'chroma'
  | 'wan22'
  | 'ltx2'
  | 'netflix_void'
  | 'hunyuan_video'
  | 'svd'

export type EngineRequestId =
  | 'sd15'
  | 'sd20'
  | 'sdxl'
  | 'sdxl_refiner'
  | 'flux1'
  | 'flux1_kontext'
  | 'flux1_fill'
  | 'flux2'
  | 'qwen_image'
  | 'flux1_chroma'
  | 'zimage'
  | 'anima'
  | 'wan22_5b'
  | 'wan22_14b'
  | 'wan22_14b_animate'
  | 'ltx2'

const TAB_FAMILY_ALIASES: Readonly<Record<string, TabFamily>> = Object.freeze({
  sd15: 'sd15',
  sdxl: 'sdxl',
  flux1: 'flux1',
  flux2: 'flux2',
  chroma: 'chroma',
  qwen_image: 'qwen_image',
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
  'qwen_image',
  'zimage',
  'anima',
  'chroma',
  'wan22',
  'ltx2',
  'netflix_void',
  'hunyuan_video',
  'svd',
])

function normalizeKey(value: unknown): string {
  return String(value || '').trim().toLowerCase()
}

export function normalizeSemanticEngine(value: unknown): SemanticEngine | null {
  const key = normalizeKey(value)
  if (!key) return null
  return SEMANTIC_ENGINE_SET.has(key) ? (key as SemanticEngine) : null
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

  const mappedRaw = typeof map[id] === 'string' ? map[id] : ''
  const mappedSemantic = normalizeSemanticEngine(mappedRaw)
  if (mappedSemantic) return mappedSemantic
  return null
}
