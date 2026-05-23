/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared model-family filtering helpers for quicksettings/checkpoint selectors.
Applies the same family/root heuristics used by QuickSettings so every checkpoint selector (including swap-model inputs) shows identical values.
Qwen Image and Z-Image L2P checkpoint selection are root-only through `qwen_image_ckpt` / `zimage_l2p_ckpt`; no filename/title fallback is used
for those families.

Symbols (top-level; keep in sync; no ghosts):
- `enginePrefixForFamily` (function): Maps UI tab family to path prefix key (`*_ckpt` roots).
- `modelMatchesFamily` (function): Returns whether a model entry belongs to a tab family using path roots + metadata/title/filename heuristics.
- `filterModelTitlesForFamily` (function): Filters model titles for one family using shared matching rules.
*/

import type { ModelInfo } from '../api/types'
import { isWanTabFamily, type TabFamily } from './engine_taxonomy'

function normalizePath(path: string): string {
  return String(path || '').replace(/\\/g, '/').replace(/\/+/g, '/').replace(/\/$/, '').toLowerCase().trim()
}

export function enginePrefixForFamily(
  family: TabFamily,
): 'sd15' | 'sdxl' | 'flux1' | 'flux2' | 'qwen_image' | 'wan22' | 'zimage' | 'zimage_l2p' | 'anima' | 'ltx2' {
  if (isWanTabFamily(family)) return 'wan22'
  if (family === 'chroma') return 'flux1'
  if (family === 'flux2') return 'flux2'
  if (family === 'qwen_image') return 'qwen_image'
  if (family === 'zimage_l2p') return 'zimage_l2p'
  if (family === 'anima') return 'anima'
  if (family === 'ltx2') return 'ltx2'
  return family
}

function fileInPaths(file: string, key: string, pathsByKey: Record<string, string[]>): boolean {
  if (!file) return false
  const roots = pathsByKey[key] || []
  if (roots.length === 0) return false
  const fileNorm = normalizePath(file)
  for (const root of roots) {
    const rootNorm = normalizePath(root)
    if (!rootNorm) continue
    if (fileNorm === rootNorm || fileNorm.startsWith(rootNorm + '/')) return true
    const rel = rootNorm.startsWith('/') ? rootNorm.slice(1) : rootNorm
    if (fileNorm.includes('/' + rel + '/') || fileNorm.endsWith('/' + rel)) return true
  }
  return false
}

export function modelMatchesFamily(
  model: ModelInfo,
  family: TabFamily,
  pathsByKey: Record<string, string[]>,
): boolean {
  const prefix = enginePrefixForFamily(family)
  const rootsKey = `${prefix}_ckpt`
  if ((pathsByKey[rootsKey] || []).length > 0) {
    return fileInPaths(model.filename, rootsKey, pathsByKey)
  }
  if (family === 'qwen_image' || family === 'zimage_l2p') return false

  const metadata = (model.metadata || {}) as Record<string, unknown>
  const metadataFamily = String((metadata.family as string) || (metadata.model_family as string) || '').toLowerCase()
  const title = String(model.title || '').toLowerCase()
  const filename = String(model.filename || '').toLowerCase()

  if (metadataFamily) return metadataFamily.includes(family)
  if (family === 'sdxl') return title.includes('sdxl') || filename.includes('sdxl')
  if (family === 'sd15') return title.includes('1.5') || title.includes('sd15') || filename.includes('sd15') || filename.includes('v1-5')
  if (family === 'flux2') return title.includes('flux2') || title.includes('flux.2') || filename.includes('flux2') || filename.includes('flux.2')
  if (family === 'flux1') return false
  if (family === 'chroma') return title.includes('chroma') || filename.includes('chroma')
  if (isWanTabFamily(family)) return title.includes('wan') || filename.includes('wan')
  if (family === 'ltx2') return title.includes('ltx') || filename.includes('ltx')
  if (family === 'zimage') {
    return (
      title.includes('zimage') ||
      title.includes('z-image') ||
      title.includes('z_image') ||
      filename.includes('zimage') ||
      filename.includes('z-image') ||
      filename.includes('z_image')
    )
  }
  return true
}

export function filterModelTitlesForFamily(
  models: ModelInfo[],
  family: TabFamily,
  pathsByKey: Record<string, string[]>,
): string[] {
  return models
    .filter((entry) => modelMatchesFamily(entry, family, pathsByKey))
    .map((entry) => String(entry.title || '').trim())
    .filter((title) => title.length > 0)
}
