<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Presentational history-details modal shared by image and WAN generation surfaces.
Renders shared modal chrome for run-history details (preview, meta rows, summary, optional text sections, params snapshot,
and footer actions) while leaving state, section construction, and history persistence/actions in the calling owners.

Symbols (top-level; keep in sync; no ghosts):
- `RunHistoryDetailsModal` (component): Shared presentational modal for image/WAN run-history details.
-->

<template>
  <Modal :modelValue="modelValue" :title="title" @update:modelValue="emit('update:modelValue', $event)">
    <div class="cdx-history-modal">
      <div class="cdx-history-modal__top">
        <img v-if="previewUrl" class="cdx-history-modal__preview" :src="previewUrl" :alt="previewAlt || title">
        <div v-else class="cdx-history-modal__preview cdx-history-modal__preview--empty">No preview</div>
        <div class="cdx-history-modal__meta">
          <div class="cdx-history-modal__meta-row"><span>Mode</span><strong>{{ modeLabel || '—' }}</strong></div>
          <div class="cdx-history-modal__meta-row"><span>Created</span><strong>{{ createdAtLabel || '—' }}</strong></div>
          <div class="cdx-history-modal__meta-row"><span>Status</span><strong>{{ status || '—' }}</strong></div>
          <div class="cdx-history-modal__meta-row"><span>Task</span><code>{{ taskId || '—' }}</code></div>
        </div>
      </div>

      <div class="cdx-history-modal__section">
        <p class="label-muted">Summary</p>
        <p class="cdx-history-modal__summary">{{ summary || '—' }}</p>
      </div>

      <div v-for="section in visibleSections" :key="section.key" class="cdx-history-modal__section">
        <p class="label-muted">{{ section.label }}</p>
        <pre class="text-xs break-words">{{ section.text }}</pre>
      </div>

      <div v-if="errorMessage" class="cdx-history-modal__section">
        <p class="label-muted">Error</p>
        <pre class="text-xs break-words">{{ errorMessage }}</pre>
      </div>

      <details class="accordion">
        <summary>Params snapshot</summary>
        <div class="accordion-body">
          <pre class="text-xs break-words">{{ formatJson(paramsSnapshot) }}</pre>
        </div>
      </details>
    </div>
    <template #footer>
      <button class="btn btn-sm btn-secondary" type="button" :disabled="loadDisabled" @click="emit('load')">
        {{ loadLabel }}
      </button>
      <button class="btn btn-sm btn-outline" type="button" :disabled="applyDisabled" @click="emit('apply')">Apply</button>
      <button class="btn btn-sm btn-outline" type="button" :disabled="copyDisabled" @click="emit('copy')">Copy</button>
      <button class="btn btn-sm btn-outline" type="button" @click="emit('update:modelValue', false)">Close</button>
    </template>
  </Modal>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import { formatJson } from '../../composables/useResultsCard'
import Modal from '../ui/Modal.vue'

type RunHistoryDetailsSection = {
  key: string
  label: string
  text: string
}

const props = withDefaults(defineProps<{
  modelValue: boolean
  title: string
  previewUrl?: string
  previewAlt?: string
  modeLabel?: string
  createdAtLabel?: string
  status?: string
  taskId?: string
  summary?: string
  errorMessage?: string
  paramsSnapshot?: unknown
  sections?: RunHistoryDetailsSection[]
  loadDisabled?: boolean
  loadLabel?: string
  applyDisabled?: boolean
  copyDisabled?: boolean
}>(), {
  previewUrl: '',
  previewAlt: '',
  modeLabel: '',
  createdAtLabel: '—',
  status: '',
  taskId: '',
  summary: '',
  errorMessage: '',
  sections: () => [],
  loadDisabled: false,
  loadLabel: 'Load',
  applyDisabled: false,
  copyDisabled: false,
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  load: []
  apply: []
  copy: []
}>()

const visibleSections = computed(() => props.sections.filter((section) => String(section.text || '').trim()))
</script>
