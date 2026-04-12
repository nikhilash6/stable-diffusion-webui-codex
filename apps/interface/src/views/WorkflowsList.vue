<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Workflows list and snapshot management view.
Lists saved workflows from the backend and provides actions to restore, rebind, or delete a workflow snapshot while preserving the
saved image-tab VAE owner instead of silently leaking the current quicksettings selection into restore.

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
              <button class="btn btn-sm btn-outline" type="button" :disabled="Boolean(restoringId)" @click="restore(wf.id)">
                {{ restoringId === wf.id ? 'Restoring…' : 'Restore' }}
              </button>
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
import { fetchFreshModelInventory, fetchFreshPaths, invalidateModelCatalogCaches } from '../api/client'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useWorkflowsStore } from '../stores/workflows'
import { defaultImageParamsForType, useModelTabsStore, type ImageTabType } from '../stores/model_tabs'
import { isWanTabFamily } from '../utils/engine_taxonomy'
import { buildFamilyVaeValidationChoices, canonicalizeVaeChoice, resolveInventoryVaeSha } from '../utils/vae_choices'

const router = useRouter()
const tabs = useModelTabsStore()
const quicksettings = useQuicksettingsStore()
const workflows = useWorkflowsStore()
const items = computed(() => workflows.items)
const actionError = ref('')
const restoringId = ref('')
const ASSET_RESTORE_SUPERSEDED_ERROR = 'A newer image asset restore/apply operation superseded this one.'

onMounted(() => { void workflows.refresh() })

async function remove(id: string): Promise<void> {
  actionError.value = ''
  await workflows.remove(id)
}

async function restore(itemId: string): Promise<void> {
  if (restoringId.value) return
  actionError.value = ''
  const wf = items.value.find(w => w.id === itemId)
  if (!wf) return
  restoringId.value = itemId
  const restoresImageQuicksettings = !isWanTabFamily(wf.type) && wf.type !== 'ltx2'
  let previousPersistedFamilyVae = ''
  let sourceTab = tabs.tabs.find((tab) => tab.id === wf.source_tab_id) ?? null
  let targetTabId = sourceTab?.id ?? wf.source_tab_id
  let createdTabId = ''
  let restoreCommitted = false
  let restoredFamilyVae = false
  const snapshotEpoch = restoresImageQuicksettings ? quicksettings.bumpAssetSnapshotEpoch() : 0
  const assertSnapshotEpochCurrent = (): void => {
    if (snapshotEpoch !== 0 && quicksettings.getAssetSnapshotEpoch() !== snapshotEpoch) {
      throw new Error(ASSET_RESTORE_SUPERSEDED_ERROR)
    }
  }
  try {
    await tabs.load()
    assertSnapshotEpochCurrent()
    if (restoresImageQuicksettings) {
      await quicksettings.init()
      assertSnapshotEpochCurrent()
      previousPersistedFamilyVae = quicksettings.getPersistedVaeForFamily(wf.type)
    }
    sourceTab = tabs.tabs.find((tab) => tab.id === wf.source_tab_id) ?? null
    targetTabId = sourceTab?.id ?? wf.source_tab_id
    if (!sourceTab) {
      createdTabId = await tabs.create(wf.type, wf.name)
      assertSnapshotEpochCurrent()
      targetTabId = createdTabId
    } else if (sourceTab.type !== wf.type) {
      throw new Error(
        `Workflow '${wf.name}' is bound to tab '${sourceTab.id}' of type '${sourceTab.type}', expected '${wf.type}'. Repair the stored workflow binding before restoring.`,
      )
    }
    const rawParamsSnapshot = { ...wf.params_snapshot }
    const snapshotVae = restoresImageQuicksettings && typeof rawParamsSnapshot.vae === 'string'
      ? rawParamsSnapshot.vae.trim()
      : ''
    delete rawParamsSnapshot.vae
    const paramsSnapshot = rawParamsSnapshot
    if (
      !isWanTabFamily(wf.type) &&
      wf.type !== 'ltx2' &&
      !Object.prototype.hasOwnProperty.call(paramsSnapshot, 'inpaintMode')
    ) {
      paramsSnapshot.inpaintMode = defaultImageParamsForType(wf.type as ImageTabType).inpaintMode
    }
    if (restoresImageQuicksettings && snapshotVae) {
      invalidateModelCatalogCaches()
      const [inventory, pathsResponse] = await Promise.all([fetchFreshModelInventory(), fetchFreshPaths()])
      assertSnapshotEpochCurrent()
      quicksettings.hydrateVaeInventorySnapshot(inventory)
      quicksettings.hydratePathsSnapshot((pathsResponse.paths || {}) as Record<string, string[]>)
      const canonicalVae = canonicalizeVaeChoice(
        snapshotVae,
        buildFamilyVaeValidationChoices(
          wf.type,
          inventory.vaes,
          pathsResponse.paths || {},
        ),
        (label) => resolveInventoryVaeSha(label, inventory.vaes),
      )
      if (!canonicalVae || canonicalVae.reason === 'fallback') {
        throw new Error(
          `Workflow '${wf.name}' saved VAE '${snapshotVae}' is no longer available for '${wf.type}'. Repair the workflow asset selection before restoring.`,
        )
      }
      assertSnapshotEpochCurrent()
      await quicksettings.setVaeForFamily(wf.type, canonicalVae.value, {
        expectedAssetSnapshotEpoch: snapshotEpoch,
      })
      assertSnapshotEpochCurrent()
      restoredFamilyVae = true
    }
    assertSnapshotEpochCurrent()
    await tabs.updateParams(targetTabId, paramsSnapshot)
    assertSnapshotEpochCurrent()
    if (wf.source_tab_id !== targetTabId) {
      assertSnapshotEpochCurrent()
      await workflows.updateSnapshot(wf.id, { source_tab_id: targetTabId })
      assertSnapshotEpochCurrent()
    }
    restoreCommitted = true
    assertSnapshotEpochCurrent()
    tabs.setActive(targetTabId)
    invalidateModelCatalogCaches()
    assertSnapshotEpochCurrent()
    await router.push(`/models/${targetTabId}`)
  } catch (error) {
    let message = error instanceof Error ? error.message : String(error)
    if (message === ASSET_RESTORE_SUPERSEDED_ERROR) {
      if (createdTabId && !restoreCommitted) {
        try {
          await tabs.remove(createdTabId)
        } catch {
          // Ignore cleanup failure for a superseded restore.
        }
      }
      return
    }
    if (
      !restoreCommitted
      && restoredFamilyVae
      && quicksettings.getAssetSnapshotEpoch() === snapshotEpoch
    ) {
      try {
        await quicksettings.restoreVaeForFamilyOwner(wf.type, previousPersistedFamilyVae, {
          expectedAssetSnapshotEpoch: snapshotEpoch,
        })
      } catch (rollbackError) {
        const rollbackMessage = rollbackError instanceof Error ? rollbackError.message : String(rollbackError)
        message = `${message} VAE rollback failed for '${wf.type}': ${rollbackMessage}`
      }
    }
    if (createdTabId && !restoreCommitted) {
      try {
        await tabs.remove(createdTabId)
      } catch (cleanupError) {
        const cleanupMessage = cleanupError instanceof Error ? cleanupError.message : String(cleanupError)
        message = `${message} Cleanup failed for temporary tab '${createdTabId}': ${cleanupMessage}`
      }
    }
    actionError.value = message
  } finally {
    if (restoringId.value === itemId) {
      restoringId.value = ''
    }
  }
}
</script>
