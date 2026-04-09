<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Textual Inversion picker + insertion modal.
Wraps the shared `PromptAssetInsertModal.vue` shell, fetches embeddings via the backend API, and emits TI tokens (optionally weighted)
for prompt insertion.

Symbols (top-level; keep in sync; no ghosts):
- `TextualInversionModal` (component): Modal for selecting embeddings and emitting insertion tokens.
- `loadItems` (function): Loads embedding inventory for the modal list.
- `insert` (function): Formats and emits an embedding token for insertion into a prompt.
-->

<template>
  <PromptAssetInsertModal
    :model-value="modelValue"
    title="Textual Inversion"
    panel-class="prompt-asset-modal-panel"
    count-label="Embeddings"
    :items="names"
    :loading="loading"
    :loaded="loaded"
    :error-message="loadError"
    @update:modelValue="emit('update:modelValue', $event)"
    @ensure-loaded="loadItems"
  >
    <template #items="{ filteredItems, weight }">
      <ul class="list" role="listbox">
        <li v-for="name in asEmbeddingNames(filteredItems)" :key="name" class="cdx-list-item clickable" @click="insert(name, weight)">
          {{ name }}
        </li>
      </ul>
    </template>
  </PromptAssetInsertModal>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { fetchEmbeddings } from '../../api/client'
import PromptAssetInsertModal from './PromptAssetInsertModal.vue'

const props = defineProps<{ modelValue: boolean }>()
const emit = defineEmits<{ (e: 'update:modelValue', value: boolean): void; (e:'insert', token: string): void }>()

const names = ref<string[]>([])
const loading = ref(false)
const loaded = ref(false)
const loadError = ref('')

async function loadItems(): Promise<void> {
  if (loaded.value || loading.value) return
  loading.value = true
  loadError.value = ''
  try {
    const response = await fetchEmbeddings()
    names.value = Object.keys(response.loaded || {}).sort((left, right) => left.localeCompare(right))
    loaded.value = true
  } catch (error) {
    names.value = []
    loaded.value = false
    loadError.value = error instanceof Error ? error.message : String(error)
  } finally {
    loading.value = false
  }
}

function insert(name: string, weight: number): void {
  const token = weight && weight !== 1.0 ? `(${name}:${weight.toFixed(2)})` : name
  emit('insert', token)
}

function asEmbeddingNames(items: readonly unknown[]): string[] {
  return items.filter((item): item is string => typeof item === 'string')
}
</script>
