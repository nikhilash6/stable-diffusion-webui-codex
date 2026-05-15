<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared prompt-asset insertion modal shell.
Owns the common modal chrome, search query, optional weight input, lazy ensure-loaded trigger, optional refresh button, count label, error banner,
and filtered list surface so LoRA and Textual Inversion wrappers only keep asset-specific inventory and token semantics.

Symbols (top-level; keep in sync; no ghosts):
- `PromptAssetInsertModal` (component): Shared prompt-asset modal shell for LoRA and Textual Inversion wrappers.
- `filteredItems` (const): Items filtered by the current query against the provided label getter.
-->

<template>
  <Modal
    :model-value="modelValue"
    :title="title"
    :panel-class="panelClass"
    :show-footer="showFooter"
    @update:modelValue="emit('update:modelValue', $event)"
  >
    <div class="prompt-asset-modal-toolbar">
      <div class="prompt-asset-modal-field">
        <label class="label-muted">Search</label>
        <input class="ui-input" v-model="query" :placeholder="searchPlaceholder" />
      </div>
      <div v-if="showWeight" class="prompt-asset-modal-field prompt-asset-modal-field--weight">
        <label class="label-muted">Weight</label>
        <input
          class="ui-input prompt-asset-modal-weight-input"
          type="number"
          :step="weightStep"
          :min="weightMin"
          v-model.number="weight"
        />
      </div>
      <button
        v-if="showRefresh"
        class="btn btn-sm btn-secondary prompt-asset-modal-refresh-btn"
        type="button"
        :disabled="loading"
        @click="emit('refresh')"
      >
        {{ loading ? 'Refreshing…' : refreshLabel }}
      </button>
      <span class="caption prompt-asset-modal-count">
        {{ filteredItems.length }} / {{ items.length }} {{ countLabel }}
      </span>
    </div>

    <p v-if="errorMessage" class="panel-error">Error: {{ errorMessage }}</p>

    <div class="panel-section modal-list-section prompt-asset-modal-list-section">
      <slot name="items" :filtered-items="filteredItems" :weight="weight" />
    </div>
  </Modal>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import Modal from '../ui/Modal.vue'

const props = withDefaults(defineProps<{
  modelValue: boolean
  title: string
  items: readonly unknown[]
  loading?: boolean
  loaded?: boolean
  errorMessage?: string
  countLabel: string
  panelClass?: string
  showFooter?: boolean
  showRefresh?: boolean
  refreshLabel?: string
  showWeight?: boolean
  weightStep?: number
  weightMin?: number
  weightDefault?: number
  searchPlaceholder?: string
  getItemLabel?: (item: unknown) => string
}>(), {
  loading: false,
  loaded: false,
  errorMessage: '',
  panelClass: '',
  showFooter: true,
  showRefresh: false,
  refreshLabel: 'Refresh',
  showWeight: true,
  weightStep: 0.1,
  weightMin: 0,
  weightDefault: 1.0,
  searchPlaceholder: 'type to filter...',
  getItemLabel: (item: unknown) => String(item ?? ''),
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'ensure-loaded'): void
  (e: 'refresh'): void
}>()

const query = ref('')
const weight = ref(Number(props.weightDefault))

const filteredItems = computed(() => {
  const normalizedQuery = query.value.toLowerCase().trim()
  return props.items.filter((item) => props.getItemLabel(item).toLowerCase().includes(normalizedQuery))
})

watch(
  () => props.modelValue,
  (isOpen) => {
    if (!isOpen) return
    if (props.loaded || props.loading) return
    emit('ensure-loaded')
  },
  { immediate: true },
)
</script>
