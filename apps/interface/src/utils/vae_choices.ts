/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared family-scoped VAE choice filtering and canonicalization helpers.
Used by frontend restore/quicksettings owners to validate family-compatible VAE selections, build family-scoped VAE choice lists, and
canonicalize saved selections through exact/sentinel/SHA matches without silently laundering unavailable assets.

Symbols (top-level; keep in sync; no ghosts):
- `InventoryVaeChoice` / `VaePathsConfig` / `VaeCanonicalizationReason` / `VaeCanonicalizationResult` (types): Shared VAE inventory/path/canonicalization contracts.
- `isVaeChoiceForFamily` (function): Family-aware VAE compatibility check backed by path roots plus known scaling/name heuristics.
- `withBuiltInVaeChoice` (function): Prepends canonical `built-in` while removing legacy aliases/duplicates from a VAE choice list.
- `buildFamilyVaeChoices` (function): Builds the active family's filtered VAE choices from inventory and path-root truth.
- `buildFamilyVaeValidationChoices` (function): Builds a fresh family-scoped VAE validation list directly from uncached inventory/path truth.
- `resolveInventoryVaeSha` (function): Resolves VAE SHA directly from fresh inventory rows without relying on quicksettings caches.
- `canonicalizeVaeChoice` (function): Canonicalizes a saved/current VAE selection through exact, sentinel, or SHA-equivalent matches.
*/

import type { InventoryResponse, PathsResponse } from '../api/types'

export type InventoryVaeChoice = InventoryResponse['vaes'][number]
export type VaePathsConfig = PathsResponse['paths']
export type VaeCanonicalizationReason = 'exact' | 'sentinel' | 'sha' | 'fallback'
export type VaeCanonicalizationResult = { value: string; reason: VaeCanonicalizationReason }

function pathBelongsToKey(path: string, key: string, pathsConfig: VaePathsConfig): boolean {
  const normalizedPath = String(path || '').replace(/\\+/g, '/').replace(/\/+$/g, '')
  for (const rawRoot of pathsConfig[key] || []) {
    const normalizedRoot = String(rawRoot || '').replace(/\\+/g, '/').replace(/\/+$/g, '')
    if (!normalizedRoot) continue
    if (normalizedPath === normalizedRoot || normalizedPath.startsWith(`${normalizedRoot}/`)) {
      return true
    }
    const relativeRoot = normalizedRoot.startsWith('/') ? normalizedRoot.slice(1) : normalizedRoot
    if (!relativeRoot) continue
    if (normalizedPath.includes(`/${relativeRoot}/`) || normalizedPath.endsWith(`/${relativeRoot}`)) {
      return true
    }
  }
  return false
}

export function isVaeChoiceForFamily(entry: Pick<InventoryVaeChoice, 'name' | 'path' | 'scaling_factor'>, family: string, pathsConfig: VaePathsConfig): boolean {
  const scale = entry.scaling_factor ?? null
  const name = String(entry.name || '')
  const path = String(entry.path || '')
  if (family === 'sdxl') return (scale !== null) ? Math.abs(Number(scale) - 0.13025) < 1e-3 : /sdxl|xl/i.test(name)
  if (family === 'sd15') return (scale !== null) ? Math.abs(Number(scale) - 0.18215) < 5e-3 : /sd1|1\\.5|sd15|v1-5/i.test(name)
  if (family === 'flux1') return pathBelongsToKey(path, 'flux1_vae', pathsConfig)
  if (family === 'flux2') return pathBelongsToKey(path, 'flux2_vae', pathsConfig)
  if (family === 'chroma') return pathBelongsToKey(path, 'flux1_vae', pathsConfig)
  if (family === 'ltx2') return pathBelongsToKey(path, 'ltx2_vae', pathsConfig)
  if (family === 'zimage') return pathBelongsToKey(path, 'zimage_vae', pathsConfig) || pathBelongsToKey(path, 'flux1_vae', pathsConfig)
  if (family === 'anima') return pathBelongsToKey(path, 'anima_vae', pathsConfig)
  return true
}

export function withBuiltInVaeChoice(values: string[]): string[] {
  const out: string[] = ['built-in']
  const seen = new Set<string>(out.map((value) => value.toLowerCase()))
  for (const value of values) {
    const trimmed = String(value || '').trim()
    if (!trimmed) continue
    const lower = trimmed.toLowerCase()
    if (lower === 'automatic' || lower === 'built in' || lower === 'built-in') continue
    if (seen.has(trimmed)) continue
    seen.add(trimmed)
    out.push(trimmed)
  }
  return out
}

export function buildFamilyVaeChoices(
  family: string,
  inventoryVaes: readonly InventoryVaeChoice[],
  allChoices: readonly string[],
  pathsConfig: VaePathsConfig,
): string[] {
  if (family === 'flux1' || family === 'flux2' || family === 'chroma') {
    const pathKey = family === 'flux2' ? 'flux2_vae' : 'flux1_vae'
    return withBuiltInVaeChoice(
      inventoryVaes
        .filter((entry) => typeof entry.path === 'string' && pathBelongsToKey(entry.path, pathKey, pathsConfig))
        .map((entry) => String(entry.path || '')),
    )
  }
  if (family === 'zimage') {
    return withBuiltInVaeChoice(
      inventoryVaes
        .filter((entry) => typeof entry.path === 'string' && (
          pathBelongsToKey(entry.path, 'zimage_vae', pathsConfig)
          || pathBelongsToKey(entry.path, 'flux1_vae', pathsConfig)
        ))
        .map((entry) => String(entry.path || '')),
    )
  }
  const familyChoices = allChoices.filter((value) => {
    const normalized = String(value || '').trim().toLowerCase()
    if (normalized === 'automatic' || normalized === 'built in' || normalized === 'built-in') return true
    if (normalized === 'none') return true
    const entry = inventoryVaes.find((candidate) => candidate.name === value || candidate.path.endsWith(`/${value}`))
    if (!entry) return false
    return isVaeChoiceForFamily(entry, family, pathsConfig)
  })
  return withBuiltInVaeChoice([...familyChoices])
}

export function buildFamilyVaeValidationChoices(
  family: string,
  inventoryVaes: readonly InventoryVaeChoice[],
  pathsConfig: VaePathsConfig,
): string[] {
  if (family === 'flux1' || family === 'flux2' || family === 'chroma') {
    const pathKey = family === 'flux2' ? 'flux2_vae' : 'flux1_vae'
    return withBuiltInVaeChoice(
      inventoryVaes
        .filter((entry) => typeof entry.path === 'string' && pathBelongsToKey(entry.path, pathKey, pathsConfig))
        .map((entry) => String(entry.path || '').trim())
        .filter((value) => value.length > 0),
    )
  }
  if (family === 'zimage') {
    return withBuiltInVaeChoice(
      inventoryVaes
        .filter((entry) => typeof entry.path === 'string' && (
          pathBelongsToKey(entry.path, 'zimage_vae', pathsConfig)
          || pathBelongsToKey(entry.path, 'flux1_vae', pathsConfig)
        ))
        .map((entry) => String(entry.path || '').trim())
        .filter((value) => value.length > 0),
    )
  }

  const seen = new Set<string>()
  const familyChoices: string[] = ['none']
  seen.add('none')
  for (const entry of inventoryVaes) {
    if (!isVaeChoiceForFamily(entry, family, pathsConfig)) continue
    const label = String(entry.name || '').trim()
    if (!label || seen.has(label)) continue
    seen.add(label)
    familyChoices.push(label)
  }
  return withBuiltInVaeChoice(familyChoices)
}

export function resolveInventoryVaeSha(
  label: string | null | undefined,
  inventoryVaes: readonly InventoryVaeChoice[],
): string | undefined {
  const raw = String(label || '').trim()
  if (!raw) return undefined

  const lower = raw.toLowerCase()
  if (lower.length === 64 && /^[0-9a-f]+$/.test(lower)) {
    return lower
  }

  const normalized = raw.replace(/\\+/g, '/')
  const withoutPrefix = normalized.includes('/') ? normalized.split('/').slice(1).join('/') : normalized
  const tail = normalized.split('/').pop() || ''
  for (const entry of inventoryVaes) {
    const sha = String(entry?.sha256 || '').trim().toLowerCase()
    if (!sha) continue
    const entryName = String(entry?.name || '').trim()
    const entryRawPath = String(entry?.path || '').trim()
    const entryNormalizedPath = entryRawPath.replace(/\\+/g, '/')
    const entryTail = entryNormalizedPath.split('/').pop() || entryName
    if (
      raw === entryName
      || raw === entryRawPath
      || normalized === entryNormalizedPath
      || withoutPrefix === entryNormalizedPath
      || tail === entryTail
    ) {
      return sha
    }
  }
  return undefined
}

export function canonicalizeVaeChoice(
  current: string,
  choices: readonly string[],
  resolveVaeSha: (label: string | null | undefined) => string | undefined,
): VaeCanonicalizationResult | null {
  if (!Array.isArray(choices) || choices.length === 0) return null

  const rawCurrent = String(current || '').trim()
  const defaultChoice = choices.includes('built-in') ? 'built-in' : String(choices[0] || '')
  if (!rawCurrent) {
    return { value: defaultChoice, reason: 'fallback' }
  }
  if (choices.includes(rawCurrent)) return { value: rawCurrent, reason: 'exact' }

  const currentLower = rawCurrent.toLowerCase()
  if (currentLower === 'automatic' || currentLower === 'built in' || currentLower === 'built-in') {
    return { value: defaultChoice, reason: 'sentinel' }
  }
  if (currentLower === 'none' && choices.includes('none')) {
    return { value: 'none', reason: 'sentinel' }
  }

  const currentSha = resolveVaeSha(rawCurrent)
  if (currentSha) {
    const normalizedCurrentSha = String(currentSha).trim().toLowerCase()
    for (const choice of choices) {
      const candidateSha = resolveVaeSha(choice)
      if (!candidateSha) continue
      if (String(candidateSha).trim().toLowerCase() === normalizedCurrentSha) {
        return { value: choice, reason: 'sha' }
      }
    }
  }

  return { value: defaultChoice, reason: 'fallback' }
}
