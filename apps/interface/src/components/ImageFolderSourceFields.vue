<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Presentational folder-source controls shared by image automation surfaces.
Renders folder path input, `all|count` amount selection, `random|sorted` ordering, sort-key selection, and an optional crop toggle against one nested source owner while emitting source-patch deltas back to the parent.

Symbols (top-level; keep in sync; no ghosts):
- `ImageFolderSourceFields` (component): Shared folder-source configuration block.
- `onFolderPathInput` (function): Emits the folder path string as the user types.
-->

<template>
  <div class="cdx-image-source-fields">
    <div class="field">
      <label class="label-muted">{{ pathLabel }}</label>
      <input
        class="ui-input"
        type="text"
        :disabled="disabled"
        :value="source.folderPath"
        :placeholder="pathPlaceholder"
        @input="onFolderPathInput"
      />
    </div>

    <div class="cdx-image-source-fields__row">
      <div class="field">
        <label class="label-muted">Selection</label>
        <CompactSegmentedControl
          :modelValue="source.selectionMode"
          :options="selectionOptions"
          :disabled="disabled"
          ariaLabel="Folder selection mode"
          @update:modelValue="(value) => emit('patch:source', { selectionMode: value as ImageFolderSelectionMode })"
        />
      </div>

      <div class="field">
        <label class="label-muted">Order</label>
        <CompactSegmentedControl
          :modelValue="source.order"
          :options="orderOptions"
          :disabled="disabled"
          ariaLabel="Folder order mode"
          @update:modelValue="(value) => emit('patch:source', { order: value as ImageFolderOrderMode })"
        />
      </div>

      <div v-if="showUseCrop" class="field cdx-image-source-fields__toggle-field">
        <label class="label-muted">Crop</label>
        <button
          :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', source.useCrop ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
          type="button"
          :aria-pressed="source.useCrop"
          :disabled="disabled"
          @click="emit('patch:source', { useCrop: !source.useCrop })"
        >
          Use crop
        </button>
      </div>
    </div>

    <div class="cdx-image-source-fields__row">
      <div v-if="source.selectionMode === 'count'" class="field cdx-image-source-fields__count-field">
        <label class="label-muted">{{ countLabel }}</label>
        <NumberStepperInput
          :modelValue="source.count"
          :disabled="disabled"
          :min="1"
          :step="1"
          :nudgeStep="1"
          size="sm"
          inputClass="cdx-input-w-xs"
          @update:modelValue="(value) => emit('patch:source', { count: value })"
        />
      </div>

      <div v-if="source.order === 'sorted'" class="field cdx-image-source-fields__sort-field">
        <label class="label-muted">Sort by</label>
        <select
          class="select-md"
          :disabled="disabled"
          :value="source.sortBy"
          @change="emit('patch:source', { sortBy: ($event.target as HTMLSelectElement).value as ImageFolderSortBy })"
        >
          <option value="name">Name</option>
          <option value="size">Size</option>
          <option value="created_at">Created</option>
          <option value="modified_at">Modified</option>
        </select>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ImageFolderOrderMode, ImageFolderSelectionMode, ImageFolderSortBy } from '../stores/model_tabs'
import CompactSegmentedControl from './ui/CompactSegmentedControl.vue'
import NumberStepperInput from './ui/NumberStepperInput.vue'

type FolderSourceValue = {
  folderPath: string
  selectionMode: ImageFolderSelectionMode
  count: number
  order: ImageFolderOrderMode
  sortBy: ImageFolderSortBy
  useCrop?: boolean
}

withDefaults(defineProps<{
  source: FolderSourceValue
  showUseCrop?: boolean
  disabled?: boolean
  pathLabel?: string
  pathPlaceholder?: string
  countLabel?: string
}>(), {
  showUseCrop: false,
  disabled: false,
  pathLabel: 'Folder path',
  pathPlaceholder: 'input/img2img-source',
  countLabel: 'Images to generate',
})

const emit = defineEmits<{
  (e: 'patch:source', value: Partial<FolderSourceValue>): void
}>()

const selectionOptions = computed(() => [
  { value: 'all', label: 'All' },
  { value: 'count', label: 'Count' },
])

const orderOptions = computed(() => [
  { value: 'random', label: 'Random' },
  { value: 'sorted', label: 'Order' },
])

function onFolderPathInput(event: Event): void {
  emit('patch:source', { folderPath: (event.target as HTMLInputElement).value })
}
</script>
