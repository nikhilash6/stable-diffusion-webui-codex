<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDXL refiner configuration card (global or hires second pass).
Renders a compact enable switch, refiner-model selector, and slider-based refiner controls (`Swap At Step` + `CFG`) in a single row,
with optional advanced guidance controls (CFG/APG) gated by capabilities and emitted as guidance-advanced patches.
Uses the shared `WanSubHeader` title pattern with full-row click toggle parity to match the BASIC PARAMETERS card header style.

Symbols (top-level; keep in sync; no ghosts):
- `RefinerSettingsCard` (component): SDXL refiner settings panel component.
- `toggle` (function): Toggles `enabled` via `update:enabled`.
- `patchGuidanceAdvanced` (function): Emits partial updates for nested advanced-guidance state.
- `toggleGuidanceAdvanced` (function): Toggles local advanced-panel visibility for this card only.
-->

<template>
  <div :class="['gen-card', 'refiner-card', { 'refiner-card--dense': dense } ]">
    <WanSubHeader
      :title="label"
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
    <div v-if="enabled" class="rf-grid">
      <div class="field rf-field--full">
        <label class="label-muted">Refiner Model</label>
        <select class="select-md" :value="model" @change="onModelChange">
          <option value="">Keep current model</option>
          <option v-if="showCurrentModelOption" :value="model">{{ model }}</option>
          <option v-for="choice in normalizedModelChoices" :key="choice" :value="choice">{{ choice }}</option>
        </select>
      </div>

      <div class="rf-row">
        <SliderField
          class="rf-col"
          label="Swap At Step"
          :modelValue="normalizedSwapAtStep"
          :min="1"
          :max="swapAtStepMax"
          :step="1"
          :inputStep="1"
          :nudgeStep="1"
          inputClass="cdx-input-w-md"
          @update:modelValue="onSwapAtStepUpdate"
        >
          <template #right>
            <NumberStepperInput
              :modelValue="normalizedSwapAtStep"
              :min="1"
              :max="swapAtStepMax"
              :step="1"
              :nudgeStep="1"
              inputClass="cdx-input-w-md"
              @update:modelValue="onSwapAtStepUpdate"
            />
          </template>
        </SliderField>

        <SliderField
          class="rf-col"
          label="CFG"
          :modelValue="normalizedCfg"
          :min="cfgMin"
          :max="cfgMax"
          :step="cfgStepValue"
          :inputStep="cfgStepValue"
          :nudgeStep="cfgStepValue"
          inputClass="cdx-input-w-md"
          @update:modelValue="onCfgUpdate"
        >
          <template #right>
            <NumberStepperInput
              :modelValue="normalizedCfg"
              :min="cfgMin"
              :max="cfgMax"
              :step="cfgStepValue"
              :nudgeStep="cfgStepValue"
              inputClass="cdx-input-w-md"
              @update:modelValue="onCfgUpdate"
            />
            <button
              v-if="showGuidanceAdvancedToggle"
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', advancedOpen ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              title="Show advanced guidance controls"
              :aria-pressed="advancedOpen"
              @click="toggleGuidanceAdvanced"
            >
              Advanced
            </button>
          </template>
        </SliderField>
      </div>

      <AdvancedGuidanceFields
        v-if="showGuidanceAdvancedRow"
        variant="refiner"
        :guidance-advanced="guidanceAdvanced"
        :guidance-support="guidanceSupport"
        :max-apg-start-step="swapAtStepMax"
        @patch="patchGuidanceAdvanced"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
// tags: refiner, settings, grid
import { computed, ref, watch } from 'vue'
import type { GuidanceAdvancedCapabilities } from '../api/types'
import { DEFAULT_GUIDANCE_ADVANCED_PARAMS, type GuidanceAdvancedParams } from '../stores/model_tabs'
import { hasAnyGuidanceSupport } from '../utils/guidance_advanced'

import AdvancedGuidanceFields from './AdvancedGuidanceFields.vue'
import NumberStepperInput from './ui/NumberStepperInput.vue'
import SliderField from './ui/SliderField.vue'
import WanSubHeader from './wan/WanSubHeader.vue'

const props = withDefaults(defineProps<{
  enabled: boolean
  swapAtStep: number
  cfg: number
  model?: string
  modelChoices?: string[]
  guidanceAdvanced?: GuidanceAdvancedParams
  guidanceSupport?: GuidanceAdvancedCapabilities | null
  label?: string
  dense?: boolean
  maxSteps?: number
  minCfg?: number
  maxCfg?: number
  cfgStep?: number
}>(), {
  label: 'Refiner',
  dense: false,
  guidanceSupport: null,
  maxSteps: 150,
  minCfg: 0,
  maxCfg: 30,
  cfgStep: 0.5,
})

const emit = defineEmits<{
  (e: 'update:enabled', value: boolean): void
  (e: 'update:swapAtStep', value: number): void
  (e: 'update:cfg', value: number): void
  (e: 'update:guidanceAdvanced', patch: Partial<GuidanceAdvancedParams>): void
  (e: 'update:model', value: string): void
}>()

const normalizedModelChoices = computed(() => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const raw of props.modelChoices || []) {
    const value = String(raw || '').trim()
    if (!value || seen.has(value)) continue
    seen.add(value)
    out.push(value)
  }
  return out
})

const showCurrentModelOption = computed(() => {
  const current = String(props.model || '').trim()
  if (!current) return false
  return !normalizedModelChoices.value.includes(current)
})

function toggle(): void {
  emit('update:enabled', !props.enabled)
}

const normalizedSwapAtStep = computed(() => {
  const v = Number(props.swapAtStep)
  if (!Number.isFinite(v) || v < 1) return 1
  return Math.trunc(v)
})

const swapAtStepMax = computed(() => {
  const value = Number(props.maxSteps)
  if (!Number.isFinite(value) || value < 1) return 150
  return Math.trunc(value)
})

const cfgMin = computed(() => {
  const value = Number(props.minCfg)
  if (!Number.isFinite(value)) return 0
  return value
})

const cfgMax = computed(() => {
  const value = Number(props.maxCfg)
  if (!Number.isFinite(value) || value <= cfgMin.value) return 30
  return value
})

const cfgStepValue = computed(() => {
  const value = Number(props.cfgStep)
  if (!Number.isFinite(value) || value <= 0) return 0.5
  return value
})

const normalizedCfg = computed(() => {
  const value = Number(props.cfg)
  if (!Number.isFinite(value)) return cfgMin.value
  return clampFloat(value, cfgMin.value, cfgMax.value)
})

const guidanceAdvanced = computed(() => props.guidanceAdvanced ?? DEFAULT_GUIDANCE_ADVANCED_PARAMS)
const guidanceSupport = computed(() => props.guidanceSupport ?? null)
const advancedOpen = ref(false)

watch(() => props.enabled, (isEnabled) => {
  if (!isEnabled) advancedOpen.value = false
})

const showGuidanceAdvancedToggle = computed(() => {
  return hasAnyGuidanceSupport(guidanceSupport.value)
})

const showGuidanceAdvancedRow = computed(() => showGuidanceAdvancedToggle.value && advancedOpen.value)

function clampFloat(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, Math.trunc(value)))
}

function onSwapAtStepUpdate(value: number): void {
  emit('update:swapAtStep', clampInt(value, 1, swapAtStepMax.value))
}

function onCfgUpdate(value: number): void {
  emit('update:cfg', clampFloat(value, cfgMin.value, cfgMax.value))
}

function patchGuidanceAdvanced(patch: Partial<GuidanceAdvancedParams>): void {
  emit('update:guidanceAdvanced', patch)
}

function toggleGuidanceAdvanced(): void {
  advancedOpen.value = !advancedOpen.value
}

function onModelChange(event: Event): void {
  emit('update:model', (event.target as HTMLSelectElement).value)
}
</script>

<!-- styles in styles/components/refiner-settings-card.css -->
