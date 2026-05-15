<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Full-screen video zoom overlay with dedicated pan/zoom controls.
Provides a WAN-exported-video overlay that opens only for valid `modelValue && src`, supports wheel zoom plus explicit pan-mode drag zoning,
closes on `Escape` or outside-click, and blocks browser/native fullscreen on double-click.

Symbols (top-level; keep in sync; no ghosts):
- `VideoZoomOverlay` (component): Dedicated full-screen video zoom overlay with close semantics for Escape and outside-click.
- `close` (function): Closes the overlay and clears transient listeners/state.
- `computeFitZoom` (function): Computes fit-to-viewport zoom for the current video source.
- `applyFitView` (function): Applies fit zoom and resets pan offsets.
- `setActualSize` (function): Resets pan and applies `1:1` zoom.
- `togglePanMode` (function): Toggles explicit drag-pan mode (controls-safe by default).
- `onOverlayWheel` (function): Handles wheel-based zoom on the main overlay region.
- `onPanZoneMouseDown` (function): Starts drag pan only from the explicit pan zone when pan mode is enabled.
- `onWindowKeydown` (function): Handles keyboard close semantics (`Escape` closes).
-->

<template>
  <div v-if="isOpen" class="video-zoom-overlay" @wheel="onOverlayWheel">
    <div ref="mainEl" class="video-zoom-main" @click="onMainClick">
      <div class="video-zoom-canvas" :style="zoomStyle" @click.stop>
        <video
          ref="videoEl"
          class="video-zoom-media"
          :src="src"
          :aria-label="resolvedAriaLabel"
          controls
          @loadedmetadata="onVideoMetadata"
          @dblclick.prevent.stop
        />
        <div
          :class="['video-zoom-pan-zone', panMode ? 'video-zoom-pan-zone--active' : '']"
          aria-hidden="true"
          @mousedown="onPanZoneMouseDown"
        />
      </div>
    </div>

    <div class="video-zoom-toolbar" @click.stop>
      <div class="toolbar-group">
        <button class="btn btn-sm btn-outline" type="button" @click="applyFitView">Fit</button>
        <button class="btn btn-sm btn-outline" type="button" @click="setActualSize">1:1</button>
        <button class="btn btn-sm btn-outline" type="button" @click="togglePanMode">{{ panMode ? 'Pan: On' : 'Pan: Off' }}</button>
        <button class="btn btn-sm btn-outline" type="button" @click="zoomIn">+</button>
        <button class="btn btn-sm btn-outline" type="button" @click="zoomOut">-</button>
        <button class="btn btn-sm btn-secondary" type="button" @click="close">Close</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch, type CSSProperties } from 'vue'

const props = withDefaults(defineProps<{
  modelValue: boolean
  src: string
  ariaLabel?: string
}>(), {
  ariaLabel: 'Zoomed video',
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
}>()

const ZOOM_MIN = 0.25
const ZOOM_MAX = 8
const FIT_PADDING_PX = 24

const isOpen = computed(() => Boolean(props.modelValue) && Boolean(props.src))
const src = computed(() => props.src)
const resolvedAriaLabel = computed(() => String(props.ariaLabel || '').trim() || 'Zoomed video')

const mainEl = ref<HTMLElement | null>(null)
const videoEl = ref<HTMLVideoElement | null>(null)
const zoom = ref(1)
const offsetX = ref(0)
const offsetY = ref(0)
const panMode = ref(false)

let panState: { startX: number; startY: number; originX: number; originY: number } | null = null

function clearPanListeners(): void {
  window.removeEventListener('mousemove', onPanMove)
  window.removeEventListener('mouseup', onPanEnd)
}

function close(): void {
  emit('update:modelValue', false)
  panMode.value = false
  panState = null
  clearPanListeners()
}

function readVideoWidth(): number {
  const intrinsic = Number(videoEl.value?.videoWidth || 0)
  if (Number.isFinite(intrinsic) && intrinsic > 0) return intrinsic
  const fallback = Number(videoEl.value?.clientWidth || 0)
  if (Number.isFinite(fallback) && fallback > 0) return fallback
  return 1
}

function readVideoHeight(): number {
  const intrinsic = Number(videoEl.value?.videoHeight || 0)
  if (Number.isFinite(intrinsic) && intrinsic > 0) return intrinsic
  const fallback = Number(videoEl.value?.clientHeight || 0)
  if (Number.isFinite(fallback) && fallback > 0) return fallback
  return 1
}

function computeFitZoom(): number {
  const main = mainEl.value
  if (!main) return 1
  const sourceWidth = readVideoWidth()
  const sourceHeight = readVideoHeight()
  const availableWidth = Math.max(1, main.clientWidth - FIT_PADDING_PX * 2)
  const availableHeight = Math.max(1, main.clientHeight - FIT_PADDING_PX * 2)
  const fit = Math.min(availableWidth / sourceWidth, availableHeight / sourceHeight)
  if (!Number.isFinite(fit) || fit <= 0) return 1
  return Math.min(1, fit)
}

function clampZoom(value: number): number {
  const minZoom = Math.min(ZOOM_MIN, computeFitZoom())
  if (!Number.isFinite(value)) return minZoom
  return Math.max(minZoom, Math.min(ZOOM_MAX, value))
}

function setActualSize(): void {
  zoom.value = clampZoom(1)
  offsetX.value = 0
  offsetY.value = 0
}

function applyFitView(): void {
  zoom.value = clampZoom(computeFitZoom())
  offsetX.value = 0
  offsetY.value = 0
}

function adjustZoom(delta: number): void {
  zoom.value = clampZoom(zoom.value + delta)
}

function togglePanMode(): void {
  panMode.value = !panMode.value
  if (!panMode.value) {
    panState = null
    clearPanListeners()
  }
}

function zoomIn(): void {
  adjustZoom(0.25)
}

function zoomOut(): void {
  adjustZoom(-0.25)
}

const zoomStyle = computed<CSSProperties>(() => ({
  position: 'relative',
  left: `${offsetX.value}px`,
  top: `${offsetY.value}px`,
  transform: `scale(${zoom.value})`,
}))

function onPanStart(event: MouseEvent): void {
  if (!isOpen.value || !panMode.value || event.button !== 0) return
  event.preventDefault()
  panState = {
    startX: event.clientX,
    startY: event.clientY,
    originX: offsetX.value,
    originY: offsetY.value,
  }
  window.addEventListener('mousemove', onPanMove)
  window.addEventListener('mouseup', onPanEnd)
}

function onPanZoneMouseDown(event: MouseEvent): void {
  onPanStart(event)
}

function onPanMove(event: MouseEvent): void {
  if (!panState) return
  const dx = event.clientX - panState.startX
  const dy = event.clientY - panState.startY
  offsetX.value = panState.originX + dx
  offsetY.value = panState.originY + dy
}

function onPanEnd(): void {
  panState = null
  clearPanListeners()
}

function onOverlayWheel(event: WheelEvent): void {
  if (!isOpen.value) return
  const target = event.target as HTMLElement | null
  if (!target) return
  if (target.closest('.video-zoom-toolbar')) return
  if (!target.closest('.video-zoom-main') && !target.closest('.video-zoom-canvas')) return
  event.preventDefault()
  const delta = event.deltaY < 0 ? 0.25 : -0.25
  adjustZoom(delta)
}

function onMainClick(event: MouseEvent): void {
  if (event.target !== event.currentTarget) return
  close()
}

function onWindowKeydown(event: KeyboardEvent): void {
  if (event.key !== 'Escape') return
  if (!isOpen.value) return
  close()
}

function onVideoMetadata(): void {
  if (!isOpen.value) return
  if (!Number.isFinite(zoom.value) || zoom.value <= 0) {
    setActualSize()
  }
}

watch(isOpen, (open) => {
  if (open) {
    window.addEventListener('keydown', onWindowKeydown)
    void nextTick(() => {
      setActualSize()
    })
    return
  }
  window.removeEventListener('keydown', onWindowKeydown)
  panMode.value = false
  panState = null
  clearPanListeners()
}, { immediate: true })

watch(src, () => {
  if (!isOpen.value) return
  setActualSize()
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onWindowKeydown)
  panMode.value = false
  panState = null
  clearPanListeners()
})
</script>
