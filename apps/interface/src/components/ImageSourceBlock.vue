<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared live owner for frontend image-source mode controls.
Renders the shared `DIR|IMG` source-mode toggle, constrained image picker, and folder-source controls used by live init-image
and IP-Adapter surfaces, while leaving additive behavior such as inpaint overlays, frame-guide editing, and `sameAsInit`
shortcuts explicit through slots and props.

Symbols (top-level; keep in sync; no ghosts):
- `ImageSourceBlock` (component): Shared source-mode owner for live frontend image-input surfaces.
- `SOURCE_MODE_OPTIONS` (constant): Segmented-control options for `DIR|IMG` source switching.
-->

<template>
  <div class="cdx-image-source-block">
    <div v-if="showSourceModeToggle || $slots['controls-extra']" class="cdx-image-source-fields__row cdx-image-source-block__controls">
      <div v-if="showSourceModeToggle" class="field cdx-image-source-block__mode-field">
        <label v-if="showSourceModeLabel" class="label-muted">{{ sourceModeLabel }}</label>
        <CompactSegmentedControl
          :modelValue="mode"
          :options="SOURCE_MODE_OPTIONS"
          :disabled="disabled"
          :ariaLabel="sourceModeAriaLabel"
          @update:modelValue="(value) => emit('update:mode', value as SourceMode)"
        />
      </div>
      <slot name="controls-extra" />
    </div>

    <InitialImageCard
      v-if="mode === 'img' && showImagePicker"
      :label="imageLabel"
      :src="imageSrc"
      :has-image="hasImage"
      :disabled="disabled"
      :placeholder="imagePlaceholder"
      :dropzone="dropzone"
      :thumbnail="thumbnail"
      :zoomable="zoomable"
      :preview-click-action="previewClickAction"
      :zoom-frame-guide="zoomFrameGuide"
      @set="(file) => emit('set:image', file)"
      @clear="() => emit('clear:image')"
      @rejected="(payload) => emit('reject:image', payload)"
      @preview-click="emit('preview-click')"
      @update:zoom-frame-guide="(value) => emit('update:zoomFrameGuide', value)"
    >
      <template v-if="$slots['dropzone-actions']" #dropzone-actions>
        <slot name="dropzone-actions" />
      </template>
      <template v-if="$slots['preview-overlay']" #preview-overlay>
        <slot name="preview-overlay" />
      </template>
      <template v-if="$slots.footer" #footer>
        <slot name="footer" />
      </template>
    </InitialImageCard>

    <slot v-else-if="mode === 'img'" name="img-empty" />

    <ImageFolderSourceFields
      v-else
      :source="folderSource"
      :showUseCrop="showUseCrop"
      :disabled="disabled"
      :pathLabel="folderPathLabel"
      :pathPlaceholder="folderPathPlaceholder"
      :countLabel="folderCountLabel"
      @patch:source="(value) => emit('patch:folderSource', value)"
    />
  </div>
</template>

<script setup lang="ts">
import type { ImageFolderOrderMode, ImageFolderSelectionMode, ImageFolderSortBy } from '../stores/model_tabs'
import type { WanImg2VidFrameGuideConfig } from '../utils/wan_img2vid_frame_projection'
import CompactSegmentedControl from './ui/CompactSegmentedControl.vue'
import ImageFolderSourceFields from './ImageFolderSourceFields.vue'
import InitialImageCard from './InitialImageCard.vue'

type SourceMode = 'dir' | 'img'

type FolderSourceValue = {
  folderPath: string
  selectionMode: ImageFolderSelectionMode
  count: number
  order: ImageFolderOrderMode
  sortBy: ImageFolderSortBy
  useCrop?: boolean
}

const SOURCE_MODE_OPTIONS = [
  { value: 'dir', label: 'DIR' },
  { value: 'img', label: 'IMG' },
] as const

withDefaults(defineProps<{
  mode: SourceMode
  folderSource: FolderSourceValue
  disabled?: boolean
  showSourceModeToggle?: boolean
  showSourceModeLabel?: boolean
  sourceModeLabel?: string
  sourceModeAriaLabel?: string
  showImagePicker?: boolean
  imageLabel?: string
  imageSrc?: string
  hasImage?: boolean
  imagePlaceholder?: string
  dropzone?: boolean
  thumbnail?: boolean
  zoomable?: boolean
  previewClickAction?: 'zoom' | 'emit'
  zoomFrameGuide?: WanImg2VidFrameGuideConfig | null
  showUseCrop?: boolean
  folderPathLabel?: string
  folderPathPlaceholder?: string
  folderCountLabel?: string
}>(), {
  disabled: false,
  showSourceModeToggle: false,
  showSourceModeLabel: false,
  sourceModeLabel: 'Source',
  sourceModeAriaLabel: 'Image source mode',
  showImagePicker: true,
  imageLabel: 'Image',
  imageSrc: '',
  hasImage: false,
  imagePlaceholder: 'Select an image to start.',
  dropzone: true,
  thumbnail: true,
  zoomable: true,
  previewClickAction: 'zoom',
  zoomFrameGuide: null,
  showUseCrop: false,
  folderPathLabel: 'Folder path',
  folderPathPlaceholder: 'input/img2img-source',
  folderCountLabel: 'Images to generate',
})

const emit = defineEmits<{
  (e: 'update:mode', value: SourceMode): void
  (e: 'patch:folderSource', value: Partial<FolderSourceValue>): void
  (e: 'set:image', value: File): void
  (e: 'clear:image'): void
  (e: 'reject:image', payload: { reason: string; files: File[] }): void
  (e: 'preview-click'): void
  (e: 'update:zoomFrameGuide', value: WanImg2VidFrameGuideConfig): void
}>()
</script>
