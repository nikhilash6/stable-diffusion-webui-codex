<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Workflows list and snapshot management view.
Lists saved workflows from the backend and provides actions to restore, rebind, or delete a workflow snapshot.

Symbols (top-level; keep in sync; no ghosts):
- `WorkflowsList` (component): Workflows route view component.
-->

<template>
  <section class="panel-stack">
    <div class="panel">
      <div class="panel-header">Workflows</div>
      <div class="panel-body">
        <div v-if="workflows.error || actionError" class="panel-error">{{ workflows.error || actionError }}</div>
        <p v-else-if="!items.length" class="caption">No workflows yet. Use “Save snapshot” from a model tab.</p>
        <ul v-else class="cdx-list">
          <li v-for="wf in items" :key="wf.id" class="cdx-list-item">
            <div class="cdx-list-main">
              <div class="cdx-list-title">{{ wf.name }}</div>
              <div class="cdx-list-meta">Type: {{ wf.type.toUpperCase() }} · Created: {{ new Date(wf.created_at).toLocaleString() }}</div>
            </div>
            <div class="cdx-list-actions">
              <button class="btn btn-sm btn-outline" type="button" @click="restore(wf.id)">Restore</button>
              <button class="btn btn-sm btn-destructive" type="button" @click="remove(wf.id)">Delete</button>
            </div>
          </li>
        </ul>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useWorkflowsStore } from '../stores/workflows'
import { defaultImageParamsForType, useModelTabsStore, type ImageTabType } from '../stores/model_tabs'
import { isWanTabFamily } from '../utils/engine_taxonomy'

const router = useRouter()
const tabs = useModelTabsStore()
const workflows = useWorkflowsStore()
const items = computed(() => workflows.items)
const actionError = ref('')

onMounted(() => { void workflows.refresh() })

async function remove(id: string): Promise<void> {
  actionError.value = ''
  await workflows.remove(id)
}

async function restore(itemId: string): Promise<void> {
  actionError.value = ''
  const wf = items.value.find(w => w.id === itemId)
  if (!wf) return
  await tabs.load()
  const sourceTab = tabs.tabs.find((tab) => tab.id === wf.source_tab_id) ?? null
  let targetTabId = sourceTab?.id ?? wf.source_tab_id
  let createdTabId = ''
  let restoreCommitted = false
  try {
    if (!sourceTab) {
      createdTabId = await tabs.create(wf.type, wf.name)
      targetTabId = createdTabId
    } else if (sourceTab.type !== wf.type) {
      throw new Error(
        `Workflow '${wf.name}' is bound to tab '${sourceTab.id}' of type '${sourceTab.type}', expected '${wf.type}'. Repair the stored workflow binding before restoring.`,
      )
    }
    const paramsSnapshot = { ...wf.params_snapshot }
    if (
      !isWanTabFamily(wf.type) &&
      wf.type !== 'ltx2' &&
      !Object.prototype.hasOwnProperty.call(paramsSnapshot, 'inpaintMode')
    ) {
      paramsSnapshot.inpaintMode = defaultImageParamsForType(wf.type as ImageTabType).inpaintMode
    }
    await tabs.updateParams(targetTabId, paramsSnapshot)
    if (wf.source_tab_id !== targetTabId) {
      await workflows.updateSnapshot(wf.id, { source_tab_id: targetTabId })
    }
    restoreCommitted = true
    tabs.setActive(targetTabId)
    await router.push(`/models/${targetTabId}`)
  } catch (error) {
    let message = error instanceof Error ? error.message : String(error)
    if (createdTabId && !restoreCommitted) {
      try {
        await tabs.remove(createdTabId)
      } catch (cleanupError) {
        const cleanupMessage = cleanupError instanceof Error ? cleanupError.message : String(cleanupError)
        message = `${message} Cleanup failed for temporary tab '${createdTabId}': ${cleanupMessage}`
      }
    }
    actionError.value = message
  }
}
</script>
