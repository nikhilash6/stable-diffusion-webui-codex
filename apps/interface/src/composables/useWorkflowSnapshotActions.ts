/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared workflow snapshot + current-params actions for generation views.
Composes the existing Results toast/clipboard owner with the shared `Save snapshot` and `Copy params`
actions used by image, WAN video, and LTX video runtime views while keeping domain-specific history/info
actions local to each owner.

Symbols (top-level; keep in sync; no ghosts):
- `WorkflowSnapshotTabLike` (type): Minimal tab metadata contract required to save a workflow snapshot.
- `UseWorkflowSnapshotActionsOptions` (type): Configuration contract for current-params copy and workflow snapshot actions.
- `useWorkflowSnapshotActions` (function): Returns shared Results notice helpers plus workflow snapshot/copy-current actions.
*/

import { ref } from 'vue'

import { useResultsCard } from './useResultsCard'
import type { ApiTab } from '../api/types'
import { useWorkflowsStore } from '../stores/workflows'

export type WorkflowSnapshotTabLike = {
  id: string
  title: string
  type: ApiTab['type']
}

export type UseWorkflowSnapshotActionsOptions<TTab extends WorkflowSnapshotTabLike> = {
  getTab: () => TTab | null
  getWorkflowParamsSnapshot: () => Record<string, unknown> | null
  getCopyCurrentParamsSnapshot?: () => Record<string, unknown> | null
  copyCurrentParamsMessage?: string
  noticeDurationMs?: number
  onBeforeCopyCurrentParams?: () => void
}

export function useWorkflowSnapshotActions<TTab extends WorkflowSnapshotTabLike>(
  options: UseWorkflowSnapshotActionsOptions<TTab>,
) {
  const workflows = useWorkflowsStore()
  const workflowBusy = ref(false)
  const { notice, toast, clearNotice, copyText, copyJson, formatJson } = useResultsCard({
    noticeDurationMs: options.noticeDurationMs,
  })

  async function sendToWorkflows(): Promise<void> {
    const tab = options.getTab()
    const paramsSnapshot = options.getWorkflowParamsSnapshot()
    if (!tab || !paramsSnapshot) return
    workflowBusy.value = true
    try {
      const result = await workflows.saveSnapshot({
        name: `${tab.title} — ${new Date().toLocaleString()}`,
        source_tab_id: tab.id,
        type: tab.type,
        params_snapshot: paramsSnapshot,
      })
      toast(result.action === 'updated' ? 'Snapshot updated in Workflows.' : 'Snapshot saved to Workflows.')
    } catch (error) {
      toast(error instanceof Error ? error.message : String(error))
    } finally {
      workflowBusy.value = false
    }
  }

  async function copyCurrentParams(): Promise<void> {
    const paramsSnapshot = options.getCopyCurrentParamsSnapshot?.() ?? options.getWorkflowParamsSnapshot()
    if (!paramsSnapshot) return
    options.onBeforeCopyCurrentParams?.()
    await copyJson(paramsSnapshot, options.copyCurrentParamsMessage ?? 'Copied params.')
  }

  return {
    notice,
    toast,
    clearNotice,
    copyText,
    copyJson,
    formatJson,
    workflowBusy,
    sendToWorkflows,
    copyCurrentParams,
  }
}
