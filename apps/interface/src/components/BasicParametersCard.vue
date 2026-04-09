<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared basic generation parameters card (sampler/scheduler/steps/seed/CFG/dimensions).
Reusable card used across model tabs to edit common fields, with optional resolution presets, CLIP skip, init-image dimension sync, img2img denoise control,
backend recommendation-aware sampler/scheduler selector grouping, and optional advanced CFG/APG controls (gated per-engine by capabilities).

Symbols (top-level; keep in sync; no ghosts):
- `BasicParametersCard` (component): Basic params card SFC; wires selectors/sliders and emits `update:*` events plus seed actions/sync hooks.
- `clampFloat` (function): Clamps a numeric value to a `[min,max]` range.
- `clampInt` (function): Clamps and truncates a numeric value to an integer range.
- `clampIntToStep` (function): Clamps and snaps an integer value to a step size (used for width/height constraints).
- `aspectRatioLocked` (ref): Local lock state for Width/Height proportional editing.
- `toggleAspectRatioLock` (function): Toggles aspect-ratio lock and captures current ratio anchor.
- `onWidthDimensionChange` (function): Handles width updates and keeps height in sync when lock is enabled.
- `onHeightDimensionChange` (function): Handles height updates and keeps width in sync when lock is enabled.
- `onSeedChange` (function): Handles manual seed input changes and emits a normalized integer seed.
- `recommendedSamplers` / `recommendedSchedulers` (const): Optional recommendation arrays forwarded into selector components.
- `patchGuidanceAdvanced` (function): Emits partial updates for nested advanced-guidance state.
- `toggleGuidanceAdvanced` (function): Toggles Advanced guidance mode and auto-syncs APG/CFG trunc activation flags when supported.
- `swapWH` (function): Swaps width/height while respecting min/max and step constraints.
- `applyResolutionPreset` (function): Applies a preset (W,H) pair to the width/height controls while respecting constraints.
-->

<template>
  <div class="gen-card">
    <WanSubHeader v-if="sectionTitle" :title="sectionTitle" />
    <div class="gc-stack">
      <div class="gc-row">
        <SamplerSelector
          class="gc-col"
          :samplers="samplers"
          :recommended-names="recommendedSamplers"
          :modelValue="sampler"
          :label="samplerLabel"
          :allow-empty="allowEmptySampler"
          :emptyLabel="samplerEmptyLabel"
          :disabled="disabled"
          @update:modelValue="(v) => emit('update:sampler', v)"
        />
        <SchedulerSelector
          class="gc-col"
          :schedulers="schedulers"
          :recommended-names="recommendedSchedulers"
          :modelValue="scheduler"
          :label="schedulerLabel"
          :allow-empty="allowEmptyScheduler"
          :emptyLabel="schedulerEmptyLabel"
          :disabled="disabled"
          @update:modelValue="(v) => emit('update:scheduler', v)"
        />
        <SliderField
          class="gc-col gc-col--wide"
          label="Steps"
          :modelValue="steps"
          :min="minSteps"
          :max="maxSteps"
          :step="1"
          :inputStep="1"
          :nudgeStep="1"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => emit('update:steps', clampInt(v, minSteps, maxSteps))"
        />
      </div>

      <div class="gc-row">
        <SliderField
          :class="['gc-col', { 'gc-col--wide': resolutionPresets.length === 0 }]"
          :label="widthLabel"
          :modelValue="width"
          :min="minWidth"
          :max="maxWidth"
          :step="widthStep"
          :inputStep="widthInputStep"
          :nudgeStep="widthInputStep"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="onWidthDimensionChange"
        >
          <template #right>
            <NumberStepperInput
              :modelValue="width"
              :min="minWidth"
              :max="maxWidth"
              :step="widthInputStep"
              :nudgeStep="widthInputStep"
              inputClass="cdx-input-w-md"
              :disabled="disabled"
              @update:modelValue="onWidthDimensionChange"
            />
            <button class="btn-swap" type="button" :disabled="disabled" title="Swap width/height" @click="swapWH">
              <span class="btn-swap-icon" aria-hidden="true">⇵</span>
            </button>
            <button
              class="btn-swap"
              :class="{ 'btn-swap--active': aspectRatioLocked }"
              type="button"
              :disabled="disabled"
              :title="aspectRatioLocked ? 'Unlock aspect ratio' : 'Lock aspect ratio'"
              :aria-label="aspectRatioLocked ? 'Unlock aspect ratio' : 'Lock aspect ratio'"
              @click="toggleAspectRatioLock"
            >
              <span aria-hidden="true">{{ aspectRatioLocked ? '🔒' : '🔓' }}</span>
            </button>
            <button
              v-if="showInitImageDims"
              class="btn-swap"
              type="button"
              :disabled="disabled"
              title="Use init image dimensions"
              @click="emit('sync-init-image-dims')"
            >
              <span aria-hidden="true">🖼</span>
            </button>
          </template>
        </SliderField>

        <SliderField
          :class="['gc-col', { 'gc-col--wide': resolutionPresets.length === 0 }]"
          :label="heightLabel"
          :modelValue="height"
          :min="minHeight"
          :max="maxHeight"
          :step="heightStep"
          :inputStep="heightInputStep"
          :nudgeStep="heightInputStep"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="onHeightDimensionChange"
        />

        <div v-if="resolutionPresets.length" class="gc-col gc-col--presets">
          <DimensionPresetsGrid :presets="resolutionPresets" :disabled="disabled" @apply="applyResolutionPreset" />
        </div>
      </div>
      <div class="gc-row">
        <div class="gc-col field">
          <label class="label-muted">{{ seedLabel }}</label>
          <div class="number-with-controls w-full">
            <input class="ui-input ui-input-sm pad-right" type="number" :disabled="disabled" :value="seed" @change="onSeedChange" />
            <div class="stepper">
              <button class="step-btn" type="button" :disabled="disabled" title="Random seed" @click="emit('random-seed')">🎲</button>
              <button class="step-btn" type="button" :disabled="disabled" title="Reuse seed" @click="emit('reuse-seed')">↺</button>
            </div>
          </div>
        </div>

        <div v-if="showClipSkip" class="gc-col field">
          <label class="label-muted">{{ clipSkipLabel }}</label>
          <NumberStepperInput
            :modelValue="clipSkip"
            :min="minClipSkip"
            :max="maxClipSkip"
            :step="1"
            :nudgeStep="1"
            inputClass="cdx-input-w-xs"
            :disabled="disabled"
            @update:modelValue="(v) => emit('update:clipSkip', clampInt(v, minClipSkip, maxClipSkip))"
          />
        </div>

        <SliderField
          v-if="showCfg"
          class="gc-col gc-col--wide"
          :label="cfgLabel"
          :modelValue="cfgScale"
          :min="minCfg"
          :max="maxCfg"
          :step="cfgStep"
          :inputStep="cfgStep"
          :nudgeStep="cfgStep"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => emit('update:cfgScale', clampFloat(v, minCfg, maxCfg))"
        >
          <template #right>
            <NumberStepperInput
              :modelValue="cfgScale"
              :min="minCfg"
              :max="maxCfg"
              :step="cfgStep"
              :nudgeStep="cfgStep"
              inputClass="cdx-input-w-md"
              :disabled="disabled"
              @update:modelValue="(v) => emit('update:cfgScale', clampFloat(v, minCfg, maxCfg))"
            />
            <button
              v-if="showGuidanceAdvancedToggle"
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', guidanceAdvanced.enabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :disabled="disabled"
              title="Show advanced guidance controls"
              @click="toggleGuidanceAdvanced"
            >
              Advanced
            </button>
          </template>
        </SliderField>

        <SliderField
          v-if="showDenoise"
          class="gc-col"
          label="Denoise"
          :modelValue="denoiseStrength"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-xs"
          :disabled="disabled"
          @update:modelValue="(v) => emit('update:denoiseStrength', clampFloat(v, 0, 1))"
        />
      </div>

      <AdvancedGuidanceFields
        v-if="showGuidanceAdvancedRow"
        variant="basic"
        :guidance-advanced="guidanceAdvanced"
        :guidance-support="guidanceSupport"
        :max-apg-start-step="maxSteps"
        :disabled="disabled"
        :show-tooltips="true"
        @patch="patchGuidanceAdvanced"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { GuidanceAdvancedCapabilities, SamplerInfo, SchedulerInfo } from '../api/types'
import { DEFAULT_GUIDANCE_ADVANCED_PARAMS, type GuidanceAdvancedParams } from '../stores/model_tabs'
import { buildGuidanceAdvancedTogglePatch, hasAnyGuidanceSupport } from '../utils/guidance_advanced'
import AdvancedGuidanceFields from './AdvancedGuidanceFields.vue'
import NumberStepperInput from './ui/NumberStepperInput.vue'
import DimensionPresetsGrid from './ui/DimensionPresetsGrid.vue'
import SliderField from './ui/SliderField.vue'
import SamplerSelector from './SamplerSelector.vue'
import SchedulerSelector from './SchedulerSelector.vue'
import WanSubHeader from './wan/WanSubHeader.vue'

const props = withDefaults(defineProps<{
  samplers: SamplerInfo[]
  schedulers: SchedulerInfo[]
  recommendedSamplers?: string[] | null
  recommendedSchedulers?: string[] | null
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  denoiseStrength?: number
  seed: number
  width: number
  height: number
  disabled?: boolean
  sectionTitle?: string

  // Labels
  samplerLabel?: string
  schedulerLabel?: string
  seedLabel?: string
  cfgLabel?: string
  widthLabel?: string
  heightLabel?: string
  clipSkipLabel?: string

  // Options
  allowEmptySampler?: boolean
  allowEmptyScheduler?: boolean
  samplerEmptyLabel?: string
  schedulerEmptyLabel?: string

  // Ranges / steps
  minSteps?: number
  maxSteps?: number
  showCfg?: boolean
  showDenoise?: boolean
  minCfg?: number
  maxCfg?: number
  cfgStep?: number
  minWidth?: number
  maxWidth?: number
  minHeight?: number
  maxHeight?: number
  widthStep?: number
  widthInputStep?: number
  heightStep?: number
  heightInputStep?: number
  showClipSkip?: boolean
  clipSkip?: number
  minClipSkip?: number
  maxClipSkip?: number

  resolutionPresets?: [number, number][]
  showInitImageDims?: boolean
  guidanceAdvanced?: GuidanceAdvancedParams
  guidanceSupport?: GuidanceAdvancedCapabilities | null
}>(), {
  disabled: false,
  samplerLabel: 'Sampler',
  schedulerLabel: 'Scheduler',
  seedLabel: 'Seed',
  cfgLabel: 'CFG',
  widthLabel: 'Width',
  heightLabel: 'Height',
  clipSkipLabel: 'CLIP Skip',
  allowEmptySampler: false,
  allowEmptyScheduler: false,
  samplerEmptyLabel: 'Select',
  schedulerEmptyLabel: 'Select',
  minSteps: 1,
  maxSteps: 150,
  showCfg: true,
  showDenoise: false,
  denoiseStrength: 0.5,
  minCfg: 0,
  maxCfg: 30,
  cfgStep: 0.5,
  minWidth: 64,
  maxWidth: 8192,
  minHeight: 64,
  maxHeight: 8192,
  widthStep: 64,
  widthInputStep: 8,
  heightStep: 64,
  heightInputStep: 8,
  showClipSkip: false,
  clipSkip: 1,
  minClipSkip: 1,
  maxClipSkip: 12,
  sectionTitle: '',
  resolutionPresets: () => [],
  showInitImageDims: false,
  guidanceAdvanced: () => ({ ...DEFAULT_GUIDANCE_ADVANCED_PARAMS }),
  guidanceSupport: null,
})

const emit = defineEmits<{
  (e: 'update:sampler', value: string): void
  (e: 'update:scheduler', value: string): void
  (e: 'update:steps', value: number): void
  (e: 'update:cfgScale', value: number): void
  (e: 'update:denoiseStrength', value: number): void
  (e: 'update:seed', value: number): void
  (e: 'update:width', value: number): void
  (e: 'update:height', value: number): void
  (e: 'update:clipSkip', value: number): void
  (e: 'update:guidanceAdvanced', patch: Partial<GuidanceAdvancedParams>): void
  (e: 'random-seed'): void
  (e: 'reuse-seed'): void
  (e: 'sync-init-image-dims'): void
}>()

const showCfg = computed(() => props.showCfg !== false)
const showClipSkip = computed(() => props.showClipSkip === true)
const recommendedSamplers = computed(() => (Array.isArray(props.recommendedSamplers) ? props.recommendedSamplers : null))
const recommendedSchedulers = computed(() => (Array.isArray(props.recommendedSchedulers) ? props.recommendedSchedulers : null))

const minSteps = computed(() => Number.isFinite(props.minSteps) ? Math.trunc(Number(props.minSteps)) : 1)
const maxSteps = computed(() => Number.isFinite(props.maxSteps) ? Math.trunc(Number(props.maxSteps)) : 150)

const minCfg = computed(() => Number.isFinite(props.minCfg) ? Number(props.minCfg) : 0)
const maxCfg = computed(() => Number.isFinite(props.maxCfg) ? Number(props.maxCfg) : 30)
const cfgStep = computed(() => Number.isFinite(props.cfgStep) ? Number(props.cfgStep) : 0.5)

const minWidth = computed(() => Number.isFinite(props.minWidth) ? Math.trunc(Number(props.minWidth)) : 64)
const maxWidth = computed(() => Number.isFinite(props.maxWidth) ? Math.trunc(Number(props.maxWidth)) : 8192)
const minHeight = computed(() => Number.isFinite(props.minHeight) ? Math.trunc(Number(props.minHeight)) : 64)
const maxHeight = computed(() => Number.isFinite(props.maxHeight) ? Math.trunc(Number(props.maxHeight)) : 8192)

const widthStep = computed(() => Number.isFinite(props.widthStep) ? Math.trunc(Number(props.widthStep)) : 64)
const widthInputStep = computed(() => Number.isFinite(props.widthInputStep) ? Math.trunc(Number(props.widthInputStep)) : 8)
const heightStep = computed(() => Number.isFinite(props.heightStep) ? Math.trunc(Number(props.heightStep)) : 64)
const heightInputStep = computed(() => Number.isFinite(props.heightInputStep) ? Math.trunc(Number(props.heightInputStep)) : 8)

const minClipSkip = computed(() => Number.isFinite(props.minClipSkip) ? Math.trunc(Number(props.minClipSkip)) : 1)
const maxClipSkip = computed(() => Number.isFinite(props.maxClipSkip) ? Math.trunc(Number(props.maxClipSkip)) : 12)

const resolutionPresets = computed(() => (Array.isArray(props.resolutionPresets) ? props.resolutionPresets : []))
const guidanceAdvanced = computed(() => props.guidanceAdvanced ?? DEFAULT_GUIDANCE_ADVANCED_PARAMS)
const guidanceSupport = computed(() => props.guidanceSupport ?? null)
const showGuidanceAdvancedToggle = computed(() => {
  if (!showCfg.value) return false
  return hasAnyGuidanceSupport(guidanceSupport.value)
})
const showGuidanceAdvancedRow = computed(() => showGuidanceAdvancedToggle.value && guidanceAdvanced.value.enabled)
const aspectRatioLocked = ref(false)
const aspectRatio = ref(1)

function clampFloat(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}

function clampInt(value: number, min: number, max: number): number {
  const n = Number.isFinite(value) ? Math.trunc(value) : min
  return Math.min(max, Math.max(min, n))
}

function clampIntToStep(value: number, min: number, max: number, step: number): number {
  const clamped = clampInt(value, min, max)
  if (!Number.isFinite(step) || step <= 0) return clamped
  const snapped = Math.round(clamped / step) * step
  return Math.min(max, Math.max(min, snapped))
}

function syncAspectRatioFromValues(widthValue: number, heightValue: number): void {
  const safeWidth = Math.max(1, clampInt(Number(widthValue), 1, Number.MAX_SAFE_INTEGER))
  const safeHeight = Math.max(1, clampInt(Number(heightValue), 1, Number.MAX_SAFE_INTEGER))
  aspectRatio.value = safeWidth / safeHeight
}

watch(
  () => [props.width, props.height] as const,
  ([nextWidth, nextHeight]) => {
    if (aspectRatioLocked.value) return
    syncAspectRatioFromValues(nextWidth, nextHeight)
  },
  { immediate: true },
)

function onSeedChange(event: Event): void {
  const raw = Number((event.target as HTMLInputElement).value)
  if (!Number.isFinite(raw)) return
  emit('update:seed', Math.trunc(raw))
}

function emitDimensions(widthValue: number, heightValue: number): void {
  emit('update:width', widthValue)
  emit('update:height', heightValue)
}

function deriveLockedDimensions(nextValue: number, axis: 'width' | 'height'): { width: number; height: number } {
  const ratio = Number.isFinite(aspectRatio.value) && aspectRatio.value > 0 ? aspectRatio.value : 1
  if (axis === 'width') {
    const widthValue = clampIntToStep(nextValue, minWidth.value, maxWidth.value, widthInputStep.value)
    const heightValue = clampIntToStep(Math.round(widthValue / ratio), minHeight.value, maxHeight.value, heightInputStep.value)
    return { width: widthValue, height: heightValue }
  }
  const heightValue = clampIntToStep(nextValue, minHeight.value, maxHeight.value, heightInputStep.value)
  const widthValue = clampIntToStep(Math.round(heightValue * ratio), minWidth.value, maxWidth.value, widthInputStep.value)
  return { width: widthValue, height: heightValue }
}

function toggleAspectRatioLock(): void {
  aspectRatioLocked.value = !aspectRatioLocked.value
  if (aspectRatioLocked.value) syncAspectRatioFromValues(props.width, props.height)
}

function onWidthDimensionChange(value: number): void {
  if (!aspectRatioLocked.value) {
    emit('update:width', clampIntToStep(value, minWidth.value, maxWidth.value, widthInputStep.value))
    return
  }
  const next = deriveLockedDimensions(value, 'width')
  emitDimensions(next.width, next.height)
}

function onHeightDimensionChange(value: number): void {
  if (!aspectRatioLocked.value) {
    emit('update:height', clampIntToStep(value, minHeight.value, maxHeight.value, heightInputStep.value))
    return
  }
  const next = deriveLockedDimensions(value, 'height')
  emitDimensions(next.width, next.height)
}

function patchGuidanceAdvanced(patch: Partial<GuidanceAdvancedParams>): void {
  emit('update:guidanceAdvanced', patch)
}

function toggleGuidanceAdvanced(): void {
  patchGuidanceAdvanced(buildGuidanceAdvancedTogglePatch(!guidanceAdvanced.value.enabled, guidanceSupport.value))
}

function swapWH(): void {
  const nextWidth = clampIntToStep(props.height, minWidth.value, maxWidth.value, widthInputStep.value)
  const nextHeight = clampIntToStep(props.width, minHeight.value, maxHeight.value, heightInputStep.value)
  syncAspectRatioFromValues(nextWidth, nextHeight)
  emitDimensions(nextWidth, nextHeight)
}

function applyResolutionPreset(pair: [number, number]): void {
  const w = clampIntToStep(pair[0], minWidth.value, maxWidth.value, widthInputStep.value)
  const h = clampIntToStep(pair[1], minHeight.value, maxHeight.value, heightInputStep.value)
  syncAspectRatioFromValues(w, h)
  emitDimensions(w, h)
}
</script>
