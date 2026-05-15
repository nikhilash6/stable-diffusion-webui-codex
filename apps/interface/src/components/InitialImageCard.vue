<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared image picker/preview primitive for frontend image-input surfaces.
Provides a file input, preview, and remove action and emits the selected `File` back to the parent.
In dropzone mode, top-right actions render inside the dotted zone (including `Remove`) in a dedicated top row so picker controls stay close to the preview without covering it.
Exposes `dropzone-actions` and `preview-overlay` slots for caller-defined preview actions/overlays; `preview-overlay` now renders inside an image-bounds media wrapper so overlays align to the actual contained preview image.
Supports optional pass-through WAN frame-guide config for zoom-overlay no-stretch projection metadata.

Symbols (top-level; keep in sync; no ghosts):
- `InitialImageCard` (component): Shared image picker/preview primitive.
- `zoomFrameGuide` (prop): Optional WAN frame-guide config forwarded to `ImageZoomOverlay`.
- `previewClickAction` (prop): Controls preview click behavior (`zoom` or external emit hook).
- `onZoomFrameGuideUpdate` (function): Forwards zoom-overlay guide edits to parent state.
- `onFile` (function): Handles file-input selection and emits `set`.
- `onDropFiles` (function): Handles dropzone selection and emits `set`.
- `onPreviewClick` (function): Routes preview clicks to zoom overlay or external `preview-click` emit.
-->

<template>
  <div class="panel-section">
    <label class="label-muted">{{ label }}</label>
    <div class="init-picker">
      <template v-if="dropzone">
        <Dropzone
          :accept="accept"
          :disabled="disabled"
          :label="placeholder"
          hint="Drop an image or click to browse."
          @select="onDropFiles"
          @rejected="onDropRejected"
        >
          <div class="init-dropzone-slot">
            <div v-if="hasImage" class="init-dropzone-actions">
              <slot name="dropzone-actions" />
              <button
                class="btn btn-sm btn-ghost init-dropzone-remove"
                type="button"
                :disabled="disabled || !hasImage"
                @click.stop.prevent="$emit('clear')"
              >
                Remove
              </button>
            </div>
            <div
              v-if="src"
              :class="[
                'init-preview',
                thumbnail ? 'init-preview--thumb' : '',
                canPreviewClick ? 'init-preview--clickable' : '',
              ]"
              @click.stop="onPreviewClick"
            >
              <div class="init-preview__media">
                <img :src="src" alt="Initial" />
                <div class="init-preview__overlay">
                  <slot name="preview-overlay" />
                </div>
              </div>
            </div>
            <p v-else class="caption">{{ placeholder }}</p>
          </div>
        </Dropzone>
      </template>
      <template v-else>
        <div class="toolbar">
          <input class="ui-input" :disabled="disabled" type="file" :accept="accept" @change="onFile" />
          <button class="btn btn-sm btn-ghost" type="button" :disabled="disabled || !hasImage" @click="$emit('clear')">Remove</button>
        </div>
        <div
          v-if="src"
          :class="[
            'init-preview',
            thumbnail ? 'init-preview--thumb' : '',
            canPreviewClick ? 'init-preview--clickable' : '',
          ]"
          @click.stop="onPreviewClick"
        >
          <div class="init-preview__media">
            <img :src="src" alt="Initial" />
            <div class="init-preview__overlay">
              <slot name="preview-overlay" />
            </div>
          </div>
        </div>
        <p v-else class="caption">{{ placeholder }}</p>
      </template>
      <slot name="footer" />
    </div>
    <ImageZoomOverlay
      v-model="zoomOpen"
      :src="src"
      alt="Initial image preview"
      :wanFrameGuide="zoomFrameGuide"
      @update:wan-frame-guide="onZoomFrameGuideUpdate"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import Dropzone from './ui/Dropzone.vue'
import ImageZoomOverlay from './ui/ImageZoomOverlay.vue'
import type { WanImg2VidFrameGuideConfig } from '../utils/wan_img2vid_frame_projection'

const props = withDefaults(defineProps<{
  label?: string
  accept?: string
  src?: string
  hasImage?: boolean
  disabled?: boolean
  placeholder?: string
  dropzone?: boolean
  thumbnail?: boolean
  zoomable?: boolean
  previewClickAction?: 'zoom' | 'emit'
  zoomFrameGuide?: WanImg2VidFrameGuideConfig | null
}>(), {
  label: 'Initial Image',
  accept: 'image/*',
  src: '',
  hasImage: false,
  disabled: false,
  placeholder: 'Select an image to start.',
  dropzone: false,
  thumbnail: false,
  zoomable: false,
  previewClickAction: 'zoom',
  zoomFrameGuide: null,
})

const emit = defineEmits<{
  (e: 'set', file: File): void
  (e: 'clear'): void
  (e: 'rejected', payload: { reason: string; files: File[] }): void
  (e: 'preview-click'): void
  (e: 'update:zoomFrameGuide', value: WanImg2VidFrameGuideConfig): void
}>()
const zoomOpen = ref(false)
const canZoom = computed(() => Boolean(props.zoomable && props.src))
const canPreviewClick = computed(() => Boolean(props.src) && (props.previewClickAction === 'emit' || canZoom.value))

function onFile(e: Event): void {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) emit('set', file)
  input.value = ''
}

function onDropFiles(files: File[]): void {
  const file = files[0]
  if (file) emit('set', file)
}

function onDropRejected(payload: { reason: string; files: File[] }): void {
  emit('rejected', payload)
}

function onPreviewClick(): void {
  if (!props.src) return
  if (props.previewClickAction === 'emit') {
    emit('preview-click')
    return
  }
  if (!canZoom.value) return
  zoomOpen.value = true
}

function onZoomFrameGuideUpdate(value: WanImg2VidFrameGuideConfig): void {
  emit('update:zoomFrameGuide', value)
}
</script>

<!-- uses .init-picker styles from src/styles.css -->
