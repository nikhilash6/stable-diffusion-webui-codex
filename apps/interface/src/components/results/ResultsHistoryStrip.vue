<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared thumbnail-strip body for generation history sections.
Renders the canonical Results history strip with thumbnail buttons, selected-state highlighting, and the shared empty-state used by generation tabs.

Symbols (top-level; keep in sync; no ghosts):
- `ResultsHistoryStrip` (component): Shared Results history strip for image and video generation tabs.
- `HistoryStripItem` (type): Minimum item contract consumed by the shared history strip.
-->

<template>
  <div v-if="props.items.length" class="cdx-history-list">
    <button
      v-for="item in props.items"
      :key="item.taskId"
      type="button"
      :class="['cdx-history-item', { 'is-selected': item.taskId === props.selectedTaskId }]"
      :aria-label="`Open history details for ${formatItemTitle(item)}`"
      @click="emit('select', item)"
    >
      <img
        v-if="item.thumbnail"
        class="cdx-history-thumb"
        :src="props.toDataUrl(item.thumbnail)"
        :alt="formatItemTitle(item)"
        loading="lazy"
      >
      <div v-else class="cdx-history-thumb cdx-history-thumb--empty">
        <span>No preview</span>
      </div>
    </button>
  </div>
  <div v-else class="caption">{{ props.emptyText }}</div>
</template>

<script setup lang="ts">
import type { GeneratedImage } from '../../api/types'

export interface HistoryStripItem {
  taskId: string
  thumbnail?: GeneratedImage | null
}

const props = withDefaults(defineProps<{
  items: HistoryStripItem[]
  selectedTaskId?: string
  emptyText?: string
  formatTitle: (item: any) => string
  toDataUrl: (image: GeneratedImage) => string
}>(), {
  selectedTaskId: '',
  emptyText: 'No runs yet.',
})

const emit = defineEmits<{
  (e: 'select', item: HistoryStripItem): void
}>()

function formatItemTitle(item: HistoryStripItem): string {
  return props.formatTitle(item)
}
</script>
