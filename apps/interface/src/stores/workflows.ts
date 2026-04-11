/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Workflow snapshots store for the WebUI.
Fetches, saves, updates, and deletes workflow snapshots via the backend API and keeps the workflows list reactive for the Workflows view.

Symbols (top-level; keep in sync; no ghosts):
- `useWorkflowsStore` (store): Pinia store for listing and mutating workflows (refresh/saveSnapshot/updateSnapshot/remove).
*/

import { defineStore } from 'pinia'
import { ref } from 'vue'

import { createWorkflow, deleteWorkflow, fetchWorkflows, updateWorkflow } from '../api/client'
import type { WorkflowItem } from '../api/types'

type SaveSnapshotPayload = {
  name: string
  source_tab_id: string
  type: WorkflowItem['type']
  params_snapshot: Record<string, unknown>
}

type UpdateSnapshotPayload = {
  name?: string
  source_tab_id?: string
  params_snapshot?: Record<string, unknown>
}

function workflowsForSourceTab(items: WorkflowItem[], sourceTabId: string): WorkflowItem[] {
  return items.filter((item) => item.source_tab_id === sourceTabId)
}

export const useWorkflowsStore = defineStore('workflows', () => {
  const items = ref<WorkflowItem[]>([])
  const isLoading = ref(false)
  const error = ref('')

  async function refresh(options?: { throwOnError?: boolean }): Promise<WorkflowItem[]> {
    isLoading.value = true
    error.value = ''
    try {
      const res = await fetchWorkflows()
      items.value = (res.workflows || []) as WorkflowItem[]
      return items.value
    } catch (err) {
      const nextError = err instanceof Error ? err.message : String(err)
      error.value = nextError
      if (options?.throwOnError) {
        throw err instanceof Error ? err : new Error(nextError)
      }
      return items.value
    } finally {
      isLoading.value = false
    }
  }

  async function saveSnapshot(payload: SaveSnapshotPayload): Promise<{ id: string; action: 'created' | 'updated' }> {
    const currentItems = await refresh({ throwOnError: true })
    const matches = workflowsForSourceTab(currentItems, payload.source_tab_id)
    if (matches.length > 1) {
      throw new Error(
        `Workflow snapshot ownership is ambiguous for source_tab_id '${payload.source_tab_id}'; remove duplicate workflow bindings before saving again.`,
      )
    }
    const existing = matches[0] ?? null
    if (existing) {
      const response = await updateWorkflow(existing.id, {
        name: payload.name,
        source_tab_id: payload.source_tab_id,
        params_snapshot: payload.params_snapshot,
      })
      await refresh({ throwOnError: true })
      return { id: response.updated, action: 'updated' }
    }
    const response = await createWorkflow(payload)
    await refresh({ throwOnError: true })
    return { id: response.id, action: 'created' }
  }

  async function updateSnapshot(id: string, payload: UpdateSnapshotPayload): Promise<string> {
    const response = await updateWorkflow(id, payload)
    await refresh({ throwOnError: true })
    return response.updated
  }

  async function remove(id: string): Promise<void> {
    await deleteWorkflow(id)
    await refresh({ throwOnError: true })
  }

  return {
    items,
    isLoading,
    error,
    refresh,
    saveSnapshot,
    updateSnapshot,
    remove,
  }
})
