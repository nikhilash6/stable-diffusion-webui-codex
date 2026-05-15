<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA picker + token insertion modal.
Wraps the shared `PromptAssetInsertModal.vue` shell, fetches LoRAs via the backend API, and emits `<lora:filename:weight>` tokens targeting
positive/negative prompt inputs. Refreshes quicksettings LoRA SHA mappings from the same inventory payload used by the modal list.

Symbols (top-level; keep in sync; no ghosts):
- `LoraModal` (component): Modal for browsing LoRAs and emitting insertion tokens.
- `loadItems` (function): Loads cached LoRA inventory into the modal list.
- `resolveTokenName` (function): Resolves token filename from inventory row data.
- `normalizeInsertWeight` (function): Normalizes user-entered LoRA weight to a finite numeric value.
- `isAbortError` (function): Detects abort-shaped refresh failures so user-facing errors stay clean.
- `cancelActiveRefreshTask` (function): Cancels any in-flight refresh task SSE subscription.
- `refreshList` (function): Runs async inventory refresh and applies LoRA/quicksettings updates on terminal success.
- `toggleInsert` (function): Toggles add/remove insertion state for a LoRA token target (`positive`/`negative`).
-->

<template>
  <PromptAssetInsertModal
    :model-value="modelValue"
    title="LoRA Selector"
    panel-class="prompt-asset-modal-panel"
    :show-footer="false"
    :show-refresh="true"
    count-label="LoRAs"
    :items="items"
    :loading="loading"
    :loaded="loaded"
    :error-message="loadError"
    :weight-step="0.05"
    :get-item-label="getItemLabel"
    @update:modelValue="emit('update:modelValue', $event)"
    @ensure-loaded="loadItems"
    @refresh="refreshList"
  >
    <template #items="{ filteredItems, weight }">
      <ul class="lora-modal-list" role="listbox">
        <li v-for="item in asLoraItems(filteredItems)" :key="item.path || item.name" class="lora-modal-item">
          <span class="lora-modal-item__name" :title="item.name">{{ item.name }}</span>
          <span class="lora-modal-item__actions">
            <button
              :class="['btn', 'btn-sm', 'btn-secondary', 'lora-modal-action', { 'is-active': isSelected(item, 'positive') }]"
              type="button"
              :aria-pressed="isSelected(item, 'positive') ? 'true' : 'false'"
              :title="isSelected(item, 'positive') ? 'Remove from Prompt' : 'Insert into Prompt'"
              @click.stop="toggleInsert(item, 'positive', weight)"
            >
              {{ isSelected(item, 'positive') ? 'Prompt ✓' : 'Prompt' }}
            </button>
            <button
              v-if="showNegativeTarget"
              :class="['btn', 'btn-sm', 'btn-outline', 'lora-modal-action', { 'is-active': isSelected(item, 'negative') }]"
              type="button"
              :aria-pressed="isSelected(item, 'negative') ? 'true' : 'false'"
              :title="isSelected(item, 'negative') ? 'Remove from Negative Prompt' : 'Insert into Negative Prompt'"
              @click.stop="toggleInsert(item, 'negative', weight)"
            >
              {{ isSelected(item, 'negative') ? 'Negative ✓' : 'Negative' }}
            </button>
          </span>
        </li>
      </ul>
    </template>
  </PromptAssetInsertModal>
</template>

<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'
import type { InventoryResponse } from '../../api/types'
import { useQuicksettingsStore } from '../../stores/quicksettings'
import PromptAssetInsertModal from './PromptAssetInsertModal.vue'

type PromptTarget = 'positive' | 'negative'
type InsertAction = 'add' | 'remove'

const props = withDefaults(defineProps<{
  modelValue: boolean
  showNegativeTarget?: boolean
}>(), {
  showNegativeTarget: true,
})
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'insert', payload: { token: string; target: PromptTarget; action: InsertAction }): void
}>()
const quicksettings = useQuicksettingsStore()

interface LoraItem {
  name: string
  path: string
}
const items = ref<LoraItem[]>([])
const loading = ref(false)
const loaded = ref(false)
const loadError = ref('')
const selectedPositive = ref<Record<string, string>>({})
const selectedNegative = ref<Record<string, string>>({})
let activeRefreshAbortController: AbortController | null = null

watch(
  () => props.modelValue,
  (isOpen) => {
    if (!isOpen) {
      cancelActiveRefreshTask()
      selectedPositive.value = {}
      selectedNegative.value = {}
      return
    }
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  cancelActiveRefreshTask()
})

function sortedLoraItems(inv: Pick<InventoryResponse, 'loras'>): LoraItem[] {
  if (!Array.isArray(inv.loras)) {
    throw new Error('LoRA inventory payload is missing required loras[] array')
  }
  return (inv.loras as LoraItem[]).slice().sort((left, right) => left.name.localeCompare(right.name))
}

function applyInventorySnapshot(inv: InventoryResponse): void {
  items.value = sortedLoraItems(inv)
  loaded.value = true
}

async function loadItems(): Promise<void> {
  if (loading.value) return
  loading.value = true
  loadError.value = ''
  try {
    const inv = await quicksettings.fetchInventoryWithLoraHydration()
    applyInventorySnapshot(inv)
  } catch (error) {
    items.value = []
    loaded.value = false
    loadError.value = error instanceof Error ? error.message : String(error)
  } finally {
    loading.value = false
  }
}

function cancelActiveRefreshTask(): void {
  const controller = activeRefreshAbortController
  if (!controller) return
  activeRefreshAbortController = null
  controller.abort()
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

function getItemLabel(item: unknown): string {
  return isLoraItem(item) ? item.name : ''
}

function isLoraItem(item: unknown): item is LoraItem {
  return typeof item === 'object' && item !== null && 'name' in item && 'path' in item
}

function asLoraItems(items: readonly unknown[]): LoraItem[] {
  return items.filter(isLoraItem)
}

async function refreshList(): Promise<void> {
  if (loading.value) return
  loading.value = true
  loadError.value = ''
  const refreshAbortController = new AbortController()
  activeRefreshAbortController = refreshAbortController
  try {
    const refreshedInventory = await quicksettings.fetchInventoryWithLoraHydration({
      refresh: true,
      signal: refreshAbortController.signal,
    })
    applyInventorySnapshot(refreshedInventory)
  } catch (error) {
    if (!isAbortError(error)) {
      loadError.value = error instanceof Error ? error.message : String(error)
    }
  } finally {
    if (activeRefreshAbortController === refreshAbortController) {
      activeRefreshAbortController = null
    }
    loading.value = false
  }
}

function resolveTokenName(item: LoraItem): string {
  const name = String(item.name || '').trim()
  if (name) return name
  const normalizedPath = String(item.path || '').trim().replace(/\\+/g, '/')
  return normalizedPath ? (normalizedPath.split('/').pop() || '').trim() : ''
}

function normalizeInsertWeight(rawWeight: unknown): number {
  const numeric = Number(rawWeight)
  if (!Number.isFinite(numeric)) return 1.0
  return numeric
}

function selectionKey(item: LoraItem): string {
  const byPath = String(item.path || '').trim()
  if (byPath) return `path:${byPath}`
  return `name:${String(item.name || '').trim()}`
}

function isSelected(item: LoraItem, target: PromptTarget): boolean {
  const key = selectionKey(item)
  if (target === 'negative') return Boolean(selectedNegative.value[key] || '')
  return Boolean(selectedPositive.value[key] || '')
}

function selectedToken(item: LoraItem, target: PromptTarget): string {
  const key = selectionKey(item)
  if (target === 'negative') return String(selectedNegative.value[key] || '')
  return String(selectedPositive.value[key] || '')
}

function setSelected(item: LoraItem, target: PromptTarget, token: string): void {
  const key = selectionKey(item)
  if (target === 'negative') {
    selectedNegative.value[key] = token
    return
  }
  selectedPositive.value[key] = token
}

function emitInsert(token: string, target: PromptTarget, action: InsertAction): void {
  if (!token) return
  emit('insert', { token, target, action })
}

function toggleInsert(item: LoraItem, target: PromptTarget, weight: number): void {
  const currentToken = selectedToken(item, target)
  if (currentToken) {
    emitInsert(currentToken, target, 'remove')
    setSelected(item, target, '')
    return
  }

  const nextToken = buildToken(item, weight)
  if (!nextToken) return
  emitInsert(nextToken, target, 'add')
  setSelected(item, target, nextToken)
}

function buildToken(item: LoraItem, currentWeight: number): string {
  const tokenName = resolveTokenName(item)
  if (!tokenName) {
    loadError.value = `LoRA '${item.name || item.path}' has no valid filename; refresh and retry.`
    return ''
  }
  const resolvedWeight = normalizeInsertWeight(currentWeight)
  return `<lora:${tokenName}:${resolvedWeight.toFixed(2)}>`
}
</script>
