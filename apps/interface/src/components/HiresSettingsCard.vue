<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Hires (second pass) settings panel.
Renders hires controls in a Basic Parameters-like row organization (sampler/scheduler/steps, scale/width/height,
upscaler/cfg/denoise, tile controls with desktop row-alignment hooks and right-anchored presets, `swapModel` selector, prompt overrides), backend recommendation-aware sampler/scheduler selector grouping, plus an optional native second-pass refiner block.
Upscaler values are stable ids (`latent:*` / `spandrel:*`), not legacy display labels. Uses the shared `WanSubHeader`
title pattern with full-row click toggle parity to match the BASIC PARAMETERS card header style.

Symbols (top-level; keep in sync; no ghosts):
- `HiresSettingsCard` (component): Hires settings block for supported image tabs.
- `toggle` (function): Toggles the hires enabled state.
- `swapResize` (function): Swaps hires width/height overrides.
- `recommendedSamplers` / `recommendedSchedulers` (const): Optional recommendation arrays forwarded into selector components.
- `hideNegativePromptByCfg` (const): Hides the hires negative prompt field when effective hires CFG is `<= 1`.
- `onMinTileChange` (function): Normalizes/clamps min-tile updates before emitting.
-->

<template>
  <div class="gen-card hires-card">
    <WanSubHeader
      title="Hires (second pass)"
      :clickable="true"
      :aria-pressed="enabled"
      :aria-expanded="enabled"
      @header-click="toggle"
    >
      <button
        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', enabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :aria-pressed="enabled"
        @click.stop="toggle"
      >
        {{ enabled ? 'Enabled' : 'Disabled' }}
      </button>
    </WanSubHeader>
    <div v-if="enabled" class="gc-stack">
      <div class="gc-row">
        <SamplerSelector
          class="gc-col"
          :samplers="samplers"
          :recommended-names="recommendedSamplers"
          :modelValue="samplerValue"
          label="Sampler"
          :allow-empty="false"
          :disabled="disabled || !enabled"
          @update:modelValue="(v) => emit('update:sampler', v)"
        />
        <SchedulerSelector
          class="gc-col"
          :schedulers="schedulers"
          :recommended-names="recommendedSchedulers"
          :modelValue="schedulerValue"
          label="Scheduler"
          :allow-empty="false"
          :disabled="disabled || !enabled"
          @update:modelValue="(v) => emit('update:scheduler', v)"
        />
        <SliderField
          class="gc-col gc-col--wide"
          label="Hires steps"
          :modelValue="steps"
          :min="0"
          :max="150"
          :step="1"
          :inputStep="1"
          :nudgeStep="1"
          inputClass="cdx-input-w-md"
          :disabled="disabled || !enabled"
          @update:modelValue="(v) => emit('update:steps', Math.max(0, Math.trunc(v)))"
        >
          <template #below>
            <p class="hr-hint">0 = reuse base steps</p>
          </template>
        </SliderField>
      </div>

      <div class="gc-row">
        <SliderField
          :class="['gc-col', { 'hr-scale-pseudo-disabled': isScaleOverrideActive }]"
          label="Scale"
          :modelValue="scale"
          :min="1"
          :max="4"
          :step="0.1"
          :inputStep="0.1"
          :nudgeStep="0.1"
          inputClass="cdx-input-w-md"
          :disabled="disabled || !enabled"
          @update:modelValue="(v) => emit('update:scale', v)"
        >
          <template #below>
            <p class="hr-hint" v-if="targetWidth && targetHeight">Target ~ {{ targetWidth }}×{{ targetHeight }}</p>
            <p class="hr-hint" v-if="isScaleOverrideActive">Width/height override active. Scale remains editable.</p>
          </template>
        </SliderField>

        <SliderField
          class="gc-col"
          label="Width"
          :modelValue="resizeXValue"
          :min="0"
          :max="8192"
          :step="64"
          :inputStep="8"
          :nudgeStep="8"
          inputClass="cdx-input-w-md"
          :disabled="disabled || !enabled"
          @update:modelValue="(v) => emit('update:resizeX', v)"
        >
          <template #right>
            <NumberStepperInput
              :modelValue="resizeXValue"
              :min="0"
              :max="8192"
              :step="8"
              :nudgeStep="8"
              inputClass="cdx-input-w-md"
              :disabled="disabled || !enabled"
              @update:modelValue="(v) => emit('update:resizeX', v)"
            />
            <button class="btn-swap" type="button" :disabled="disabled || !enabled" title="Swap width/height override" @click="swapResize">
              <span class="btn-swap-icon" aria-hidden="true">⇵</span>
            </button>
          </template>
        </SliderField>

        <SliderField
          class="gc-col"
          label="Height"
          :modelValue="resizeYValue"
          :min="0"
          :max="8192"
          :step="64"
          :inputStep="8"
          :nudgeStep="8"
          inputClass="cdx-input-w-md"
          :disabled="disabled || !enabled"
          @update:modelValue="(v) => emit('update:resizeY', v)"
        />
      </div>

      <div class="gc-row">
        <div class="gc-col field">
          <label class="label-muted">Upscaler</label>
          <select class="select-md" :value="upscaler" :disabled="disabled || !enabled || upscalersLoading" @change="onUpscalerChange">
            <option v-if="upscalersLoading" :value="upscaler">Loading…</option>
            <option v-else-if="upscaler && !isUpscalerKnown" :value="upscaler">Invalid selection: {{ upscaler }}</option>
            <option v-else value="" disabled>Select</option>
            <optgroup v-if="spandrelUpscalers.length" label="Spandrel (pixel SR)">
              <option v-for="u in spandrelUpscalers" :key="u.id" :value="u.id">{{ u.label }}</option>
            </optgroup>
            <optgroup v-if="latentUpscalers.length" label="Latent">
              <option v-for="u in latentUpscalers" :key="u.id" :value="u.id">{{ u.label }}</option>
            </optgroup>
          </select>
          <p class="hr-hint" v-if="upscalersError">Error: {{ upscalersError }}</p>
          <p class="hr-hint" v-else-if="upscaler && !isUpscalerKnown">Select an upscaler id from `GET /api/upscalers`.</p>
        </div>

        <SliderField
          class="gc-col"
          :label="cfgLabel"
          :modelValue="cfgValue"
          :min="0"
          :max="30"
          :step="0.5"
          :inputStep="0.5"
          :nudgeStep="0.5"
          inputClass="cdx-input-w-md"
          :disabled="disabled || !enabled"
          @update:modelValue="(v) => emit('update:cfg', v)"
        />

        <SliderField
          class="gc-col"
          label="Denoise"
          :modelValue="denoise"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled || !enabled"
          @update:modelValue="(v) => emit('update:denoise', v)"
        />
      </div>

      <div class="gc-row hr-tile-row">
        <SliderField
          class="gc-col hr-tile-slider"
          label="Overlap"
          :modelValue="tileConfig.overlap"
          :min="0"
          :max="Math.max(0, tileConfig.tile - 1)"
          :step="4"
          :inputStep="4"
          :nudgeStep="4"
          inputClass="cdx-input-w-sm"
          :disabled="disabled || !enabled || !isSpandrelSelected"
          @update:modelValue="onTileOverlap"
        />

        <SliderField
          class="gc-col hr-tile-slider"
          label="Min tile"
          :modelValue="minTile"
          :min="1"
          :max="Math.max(1, tileConfig.tile)"
          :step="8"
          :inputStep="8"
          :nudgeStep="8"
          inputClass="cdx-input-w-sm"
          :disabled="disabled || !enabled || !isSpandrelSelected"
          @update:modelValue="onMinTileChange"
        />

        <div class="gc-col hr-tile-col">
          <label class="label-muted">Tile</label>
          <div class="cdx-res-presets hr-tile-presets" aria-label="Tile presets">
            <button
              v-for="preset in tilePresets"
              :key="preset"
              class="btn btn-sm btn-outline"
              type="button"
              :disabled="disabled || !enabled || !isSpandrelSelected"
              @click="onTileSize(preset)"
            >
              {{ preset }}
            </button>
          </div>
        </div>
      </div>

      <p class="hr-hint" v-if="upscaler && !isSpandrelSelected">
        Tile settings apply to Spandrel (pixel SR) upscalers only.
      </p>

      <div v-if="showSwapModel" class="gc-row">
        <div class="gc-col field hr-field--full">
          <label class="label-muted">Second-Pass Model</label>
          <select class="select-md" :value="swapModelValue" :disabled="disabled || !enabled" @change="onSwapModelChange">
            <option value="">Keep current model</option>
            <option v-if="showCurrentSwapModelOption" :value="swapModelValue">{{ swapModelValue }}</option>
            <option v-for="entry in normalizedSwapModelChoices" :key="entry" :value="entry">{{ entry }}</option>
          </select>
        </div>
      </div>

      <div class="gc-row">
        <div :class="['gc-col', 'field', { 'hr-field--full': hideNegativePromptByCfg }]">
          <label class="label-muted">Hires Prompt</label>
          <textarea
            class="ui-textarea h-prompt-sm"
            rows="4"
            :disabled="disabled || !enabled"
            :value="promptValue"
            @input="onPromptInput"
          />
          <p class="hr-hint">Leave blank to reuse the base prompt.</p>
        </div>
        <div v-if="!hideNegativePromptByCfg" class="gc-col field">
          <label class="label-muted">Hires Negative Prompt</label>
          <textarea
            class="ui-textarea h-prompt-sm"
            rows="4"
            :disabled="disabled || !enabled || !supportsNegative"
            :value="negativePromptValue"
            @input="onNegativePromptInput"
          />
          <p class="hr-hint" v-if="supportsNegative">Leave blank to reuse the base negative prompt.</p>
          <p class="hr-hint" v-else>Current engine ignores negative prompts.</p>
        </div>
      </div>

    </div>
    <div v-if="enabled && showRefiner" class="hr-refiner">
      <RefinerSettingsCard
        label="Second-Pass Refiner"
        :dense="true"
        :max-steps="refinerStepLimit"
        :model-choices="refinerModelChoices"
        :guidance-advanced="guidanceAdvanced"
        :guidance-support="guidanceSupport"
        v-model:enabled="refinerEnabled"
        v-model:swapAtStep="refinerSwapAtStep"
        v-model:cfg="refinerCfg"
        v-model:model="refinerModel"
        @update:guidanceAdvanced="(patch) => emit('update:guidanceAdvanced', patch)"
      />
      <p class="hr-hint">Refiner switch uses step-pointer semantics in the second pass.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
// tags: hires, settings, grid
import { computed } from 'vue'
import type { GuidanceAdvancedCapabilities, SamplerInfo, SchedulerInfo, UpscalerDefinition, UpscalerKind } from '../api/types'
import type { GuidanceAdvancedParams } from '../stores/model_tabs'
import RefinerSettingsCard from './RefinerSettingsCard.vue'
import NumberStepperInput from './ui/NumberStepperInput.vue'
import SamplerSelector from './SamplerSelector.vue'
import SchedulerSelector from './SchedulerSelector.vue'
import SliderField from './ui/SliderField.vue'
import WanSubHeader from './wan/WanSubHeader.vue'

type TileConfigState = { tile: number; overlap: number }

const props = defineProps<{
  disabled?: boolean
  enabled: boolean
  denoise: number
  scale: number
  steps: number
  cfg?: number
  cfgLabel?: string
  resizeX?: number
  resizeY?: number
  swapModel?: string
  swapModelChoices?: string[]
  showSwapModel?: boolean
  prompt?: string
  negativePrompt?: string
  supportsNegative?: boolean
  samplers?: SamplerInfo[]
  schedulers?: SchedulerInfo[]
  recommendedSamplers?: string[] | null
  recommendedSchedulers?: string[] | null
  sampler?: string
  scheduler?: string
  upscaler: string
  tile?: TileConfigState
  minTile?: number
  upscalers?: UpscalerDefinition[]
  upscalersLoading?: boolean
  upscalersError?: string
  baseWidth?: number
  baseHeight?: number
  refinerEnabled?: boolean
  refinerSwapAtStep?: number
  refinerCfg?: number
  refinerModel?: string
  refinerModelChoices?: string[]
  refinerMaxSteps?: number
  guidanceAdvanced?: GuidanceAdvancedParams
  guidanceSupport?: GuidanceAdvancedCapabilities | null
}>()

const emit = defineEmits<{
  (e: 'update:enabled', value: boolean): void
  (e: 'update:denoise', value: number): void
  (e: 'update:scale', value: number): void
  (e: 'update:steps', value: number): void
  (e: 'update:cfg', value: number): void
  (e: 'update:resizeX', value: number): void
  (e: 'update:resizeY', value: number): void
  (e: 'update:swapModel', value: string): void
  (e: 'update:prompt', value: string): void
  (e: 'update:negativePrompt', value: string): void
  (e: 'update:sampler', value: string): void
  (e: 'update:scheduler', value: string): void
  (e: 'update:upscaler', value: string): void
  (e: 'update:tile', value: TileConfigState): void
  (e: 'update:minTile', value: number): void
  (e: 'update:refinerEnabled', value: boolean): void
  (e: 'update:refinerSwapAtStep', value: number): void
  (e: 'update:refinerCfg', value: number): void
  (e: 'update:refinerModel', value: string): void
  (e: 'update:guidanceAdvanced', patch: Partial<GuidanceAdvancedParams>): void
}>()

const disabled = computed(() => Boolean(props.disabled))
const upscalersLoading = computed(() => Boolean(props.upscalersLoading))
const upscalersError = computed(() => String(props.upscalersError ?? '').trim())
const tilePresets = [128, 256, 512, 768] as const

const samplers = computed(() => Array.isArray(props.samplers) ? props.samplers : [])
const schedulers = computed(() => Array.isArray(props.schedulers) ? props.schedulers : [])
const recommendedSamplers = computed(() => (Array.isArray(props.recommendedSamplers) ? props.recommendedSamplers : null))
const recommendedSchedulers = computed(() => (Array.isArray(props.recommendedSchedulers) ? props.recommendedSchedulers : null))
const samplerValue = computed(() => String(props.sampler || '').trim())
const schedulerValue = computed(() => String(props.scheduler || '').trim())
const cfgLabel = computed(() => String(props.cfgLabel || 'CFG'))
const cfgValue = computed(() => Number.isFinite(props.cfg) ? Number(props.cfg) : 7)
const resizeXValue = computed(() => {
  const value = Number(props.resizeX)
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.trunc(value))
})
const resizeYValue = computed(() => {
  const value = Number(props.resizeY)
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.trunc(value))
})
const showSwapModel = computed(() => Boolean(props.showSwapModel ?? true))
const swapModelValue = computed(() => String(props.swapModel || '').trim())
const normalizedSwapModelChoices = computed(() => {
  const choices = Array.isArray(props.swapModelChoices) ? props.swapModelChoices : []
  return Array.from(new Set(choices.map((entry) => String(entry || '').trim()).filter((entry) => entry.length > 0)))
})
const showCurrentSwapModelOption = computed(() => {
  const current = swapModelValue.value
  if (!current) return false
  return !normalizedSwapModelChoices.value.includes(current)
})
const promptValue = computed(() => String(props.prompt ?? ''))
const negativePromptValue = computed(() => String(props.negativePrompt ?? ''))
const supportsNegative = computed(() => Boolean(props.supportsNegative ?? true))
const hideNegativePromptByCfg = computed(() => supportsNegative.value && cfgValue.value <= 1)

const upscalers = computed(() => Array.isArray(props.upscalers) ? props.upscalers : [])
const spandrelUpscalers = computed(() => upscalers.value.filter((u) => u.kind === 'spandrel'))
const latentUpscalers = computed(() => upscalers.value.filter((u) => u.kind === 'latent'))
const isUpscalerKnown = computed(() => upscalers.value.some((u) => u.id === props.upscaler))
const selectedUpscalerKind = computed<UpscalerKind | null>(() => {
  const found = upscalers.value.find((u) => u.id === props.upscaler)
  if (found) return found.kind
  const id = String(props.upscaler || '')
  if (id.startsWith('spandrel:')) return 'spandrel'
  if (id.startsWith('latent:')) return 'latent'
  return null
})
const isSpandrelSelected = computed(() => selectedUpscalerKind.value === 'spandrel')

const tileConfig = computed<TileConfigState>(() => {
  const v = props.tile
  if (!v) return { tile: 256, overlap: 16 }
  const tile = Number.isFinite(v.tile) ? Math.max(1, Math.trunc(v.tile)) : 256
  const overlap = Number.isFinite(v.overlap) ? Math.max(0, Math.trunc(v.overlap)) : 16
  return { tile, overlap: Math.min(tile - 1, overlap) }
})

const minTile = computed(() => {
  const raw = props.minTile
  const v = (typeof raw === 'number' && Number.isFinite(raw)) ? Math.max(1, Math.trunc(raw)) : 128
  return Math.min(tileConfig.value.tile, v)
})

const targetWidth = computed(() => {
  if (!props.baseWidth || props.scale <= 1) return null
  return Math.round(props.baseWidth * props.scale)
})

const targetHeight = computed(() => {
  if (!props.baseHeight || props.scale <= 1) return null
  return Math.round(props.baseHeight * props.scale)
})
const isScaleOverrideActive = computed(() => resizeXValue.value > 0 || resizeYValue.value > 0)

const showRefiner = computed(() => props.refinerEnabled !== undefined)
const refinerEnabled = computed({
  get: () => Boolean(props.refinerEnabled),
  set: (value: boolean) => emit('update:refinerEnabled', value),
})
const refinerSwapAtStep = computed({
  get: () => {
    const value = Number(props.refinerSwapAtStep)
    if (!Number.isFinite(value) || value < 1) return 1
    return Math.trunc(value)
  },
  set: (value: number) => emit('update:refinerSwapAtStep', value),
})
const refinerStepLimit = computed(() => {
  const value = Number(props.refinerMaxSteps)
  if (!Number.isFinite(value) || value < 1) return 150
  return Math.trunc(value)
})
const refinerCfg = computed({
  get: () => Number.isFinite(props.refinerCfg) ? Number(props.refinerCfg) : 7,
  set: (value: number) => emit('update:refinerCfg', value),
})
const refinerModel = computed({
  get: () => props.refinerModel ?? '',
  set: (value: string) => emit('update:refinerModel', value),
})
const refinerModelChoices = computed(() => Array.isArray(props.refinerModelChoices) ? props.refinerModelChoices : [])
const guidanceAdvanced = computed(() => props.guidanceAdvanced)
const guidanceSupport = computed(() => props.guidanceSupport ?? null)

function toggle(): void {
  emit('update:enabled', !props.enabled)
}

function onUpscalerChange(event: Event): void {
  emit('update:upscaler', (event.target as HTMLSelectElement).value)
}

function onSwapModelChange(event: Event): void {
  emit('update:swapModel', (event.target as HTMLSelectElement).value)
}

function onPromptInput(event: Event): void {
  emit('update:prompt', (event.target as HTMLTextAreaElement).value)
}

function onNegativePromptInput(event: Event): void {
  emit('update:negativePrompt', (event.target as HTMLTextAreaElement).value)
}

function swapResize(): void {
  emit('update:resizeX', resizeYValue.value)
  emit('update:resizeY', resizeXValue.value)
}

function onTileSize(value: number): void {
  const v = Math.max(1, Math.trunc(Number(value)))
  if (!Number.isFinite(v)) return
  emit('update:tile', { tile: v, overlap: Math.min(v - 1, tileConfig.value.overlap) })
}

function onTileOverlap(value: number): void {
  const v = Math.max(0, Math.trunc(Number(value)))
  if (!Number.isFinite(v)) return
  emit('update:tile', { tile: tileConfig.value.tile, overlap: Math.min(tileConfig.value.tile - 1, v) })
}

function onMinTileChange(value: number): void {
  const v = Math.trunc(Number(value))
  if (!Number.isFinite(v)) return
  emit('update:minTile', Math.max(1, Math.min(tileConfig.value.tile, v)))
}
</script>

<!-- styles in styles/components/hires-settings-card.css -->
