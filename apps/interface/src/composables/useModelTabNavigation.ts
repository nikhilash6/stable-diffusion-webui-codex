/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Navigation helper for model tabs (`/models/:tabId`).
Ensures a tab exists for a given type, optionally applies init-image params for supported families, fails loud on unsupported init-image
flows, and navigates to the tab route.

Symbols (top-level; keep in sync; no ghosts):
- `useModelTabNavigation` (function): Provides `openModelTab(...)` helper for routing to a model tab.
*/

import { useRouter } from 'vue-router'
import { useModelTabsStore, type BaseTabType } from '../stores/model_tabs'
import { isWanTabFamily } from '../utils/engine_taxonomy'

export function useModelTabNavigation(): {
  openModelTab: (type: BaseTabType, options?: { initImage?: { dataUrl: string; name: string } }) => Promise<void>
} {
  const router = useRouter()
  const tabs = useModelTabsStore()

  async function openModelTab(type: BaseTabType, options?: { initImage?: { dataUrl: string; name: string } }): Promise<void> {
    await tabs.load()
    const existing = tabs.orderedTabs.find(t => t.type === type)
    const id = existing?.id || (await tabs.create(type))
    if (!id) throw new Error('failed to resolve a model tab id')

    if (options?.initImage) {
      if (isWanTabFamily(type)) {
        throw new Error('WAN init-image navigation is not implemented in useModelTabNavigation.')
      }
      if (type === 'ltx2') {
        const patch: Record<string, unknown> = {
          mode: 'img2vid',
          useInitImage: true,
          initImageData: options.initImage.dataUrl,
          initImageName: options.initImage.name,
        }
        await tabs.updateParams(id, patch)
      } else {
        const patch: Record<string, unknown> = {
          useInitImage: true,
          initSource: {
            mode: 'img',
          },
          initImageData: options.initImage.dataUrl,
          initImageName: options.initImage.name,
        }
        await tabs.updateParams(id, patch)
      }
    }

    await router.push(`/models/${id}`)
  }

  return { openModelTab }
}
