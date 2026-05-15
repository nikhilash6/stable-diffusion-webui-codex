<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Global first-pass model-swap configuration card.
Renders the generic `swap_model` stage for the first/base pass, including the enable toggle, swapped-model selector, and step-pointer/CFG
controls, without reusing SDXL-native refiner wording or ownership.

Symbols (top-level; keep in sync; no ghosts):
- `SwapStageSettingsCard` (component): First-pass `swap_model` settings panel component.
- `toggle` (function): Toggles `enabled` via `update:enabled`.
- `onSwapAtStepUpdate` (function): Emits a clamped integer swap step.
- `onCfgUpdate` (function): Emits a clamped CFG update.
- `onModelChange` (function): Emits the selected model label.
-->

<template>
  <div class="gen-card refiner-card">
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
        <label class="label-muted">Swap Model</label>
        <select class="select-md" :value="model" @change="onModelChange">
          <option value="">Keep current model</option>
          <option v-if="showCurrentModelOption" :value="model">{{ model }}</option>
          <option v-for="choice in normalizedModelChoices" :key="choice" :value="choice">{{ choice }}</option>
        </select>
        <p class="hr-hint">Continue the first pass with this model after the selected step.</p>
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
          </template>
        </SliderField>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import NumberStepperInput from './ui/NumberStepperInput.vue'
import SliderField from './ui/SliderField.vue'
import WanSubHeader from './wan/WanSubHeader.vue'

const props = withDefaults(defineProps<{
  enabled: boolean
  swapAtStep: number
  cfg: number
  model?: string
  modelChoices?: string[]
  label?: string
  maxSteps?: number
  minCfg?: number
  maxCfg?: number
  cfgStep?: number
}>(), {
  label: 'First-Pass Model Swap',
  maxSteps: 150,
  minCfg: 0,
  maxCfg: 30,
  cfgStep: 0.5,
})

const emit = defineEmits<{
  (e: 'update:enabled', value: boolean): void
  (e: 'update:swapAtStep', value: number): void
  (e: 'update:cfg', value: number): void
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
  const value = Number(props.swapAtStep)
  if (!Number.isFinite(value) || value < 1) return 1
  return Math.trunc(value)
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

function onModelChange(event: Event): void {
  emit('update:model', (event.target as HTMLSelectElement).value)
}
</script>
