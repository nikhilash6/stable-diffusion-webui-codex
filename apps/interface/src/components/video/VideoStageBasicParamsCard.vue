<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Generic stage-basic parameters card for video workspaces.
Renders optional sampler/scheduler controls plus the shared steps/seed/CFG fields for one video stage,
with optional collapse/header actions and no family-owned runtime state.

Symbols (top-level; keep in sync; no ghosts):
- `VideoStageBasicParamsCard` (component): Generic video stage parameters card.
- `isOpen` (const): Normalized open-state used when the card is collapsible.
- `toggleOpen` (function): Emits the next open-state for the owner.
-->

<template>
  <div class="gen-card cdx-video-card">
    <div class="cdx-video-card-header">
      <div class="cdx-video-card-header__left">
        <span class="cdx-video-card-header__title">{{ title }}</span>
      </div>
      <div class="cdx-video-card-header__right">
        <slot name="header-actions" />
        <button
          v-if="collapsible"
          class="btn-icon"
          type="button"
          :disabled="disabled"
          :aria-expanded="isOpen ? 'true' : 'false'"
          :title="isOpen ? 'Collapse' : 'Expand'"
          :aria-label="`Toggle ${title}`"
          @click="toggleOpen"
        >
          <span aria-hidden="true">{{ isOpen ? '▾' : '▸' }}</span>
        </button>
      </div>
    </div>

    <div v-if="caption" class="caption mt-2">{{ caption }}</div>

    <div v-if="!collapsible || isOpen" class="mt-2 cdx-video-card-body">
      <div class="gc-row">
        <SamplerSelector
          v-if="showSampler"
          class="gc-col"
          :samplers="samplers"
          :recommended-names="recommendedSamplers"
          :modelValue="sampler"
          :label="samplerLabel"
          :allow-empty="true"
          emptyLabel="Inherit"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:sampler', value)"
        />
        <SchedulerSelector
          v-if="showScheduler"
          class="gc-col"
          :schedulers="schedulers"
          :recommended-names="recommendedSchedulers"
          :modelValue="scheduler"
          :label="schedulerLabel"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:scheduler', value)"
        />
        <SliderField
          class="gc-col gc-col--wide"
          label="Steps"
          :modelValue="steps"
          :min="stepsMin"
          :max="stepsMax"
          :step="stepsStep"
          :inputStep="stepsInputStep"
          :nudgeStep="stepsNudgeStep"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:steps', Math.trunc(value))"
        />
      </div>
      <div class="gc-row">
        <div class="gc-col gc-col--wide field">
          <label class="label-muted">Seed</label>
          <div class="number-with-controls w-full">
            <NumberStepperInput
              :modelValue="seed"
              :min="-1"
              :step="1"
              :nudgeStep="1"
              inputClass="cdx-input-w-md"
              :disabled="disabled"
              updateOnInput
              @update:modelValue="(value) => emit('update:seed', Math.trunc(value))"
            />
            <div v-if="showSeedActions" class="stepper">
              <button class="step-btn" type="button" :disabled="disabled" title="Random seed" @click="emit('randomize-seed')">🎲</button>
              <button class="step-btn" type="button" :disabled="disabled || !canReuseSeed" title="Reuse seed" @click="emit('reuse-seed')">↺</button>
            </div>
          </div>
        </div>
        <SliderField
          class="gc-col gc-col--wide"
          label="CFG"
          :modelValue="cfgScale"
          :min="cfgMin"
          :max="cfgMax"
          :step="cfgStep"
          :inputStep="cfgInputStep"
          :nudgeStep="cfgNudgeStep"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:cfgScale', value)"
        />
      </div>
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import type { SamplerInfo, SchedulerInfo } from '../../api/types'
import NumberStepperInput from '../ui/NumberStepperInput.vue'
import SliderField from '../ui/SliderField.vue'
import SamplerSelector from '../SamplerSelector.vue'
import SchedulerSelector from '../SchedulerSelector.vue'

const props = withDefaults(defineProps<{
  title: string
  samplers?: SamplerInfo[]
  schedulers?: SchedulerInfo[]
  recommendedSamplers?: string[] | null
  recommendedSchedulers?: string[] | null
  sampler?: string
  scheduler?: string
  showSampler?: boolean
  showScheduler?: boolean
  steps: number
  cfgScale: number
  seed: number
  disabled?: boolean
  collapsible?: boolean
  open?: boolean
  caption?: string
  samplerLabel?: string
  schedulerLabel?: string
  stepsMin?: number
  stepsMax?: number
  stepsStep?: number
  stepsInputStep?: number
  stepsNudgeStep?: number
  cfgMin?: number
  cfgMax?: number
  cfgStep?: number
  cfgInputStep?: number
  cfgNudgeStep?: number
  showSeedActions?: boolean
  canReuseSeed?: boolean
}>(), {
  samplers: () => [],
  schedulers: () => [],
  recommendedSamplers: null,
  recommendedSchedulers: null,
  sampler: '',
  scheduler: '',
  showSampler: true,
  showScheduler: true,
  disabled: false,
  collapsible: false,
  open: true,
  caption: '',
  samplerLabel: 'Sampler',
  schedulerLabel: 'Scheduler',
  stepsMin: 1,
  stepsMax: 150,
  stepsStep: 1,
  stepsInputStep: 1,
  stepsNudgeStep: 1,
  cfgMin: 0,
  cfgMax: 30,
  cfgStep: 0.5,
  cfgInputStep: 0.5,
  cfgNudgeStep: 0.5,
  showSeedActions: true,
  canReuseSeed: false,
})

const emit = defineEmits<{
  (e: 'update:sampler', value: string): void
  (e: 'update:scheduler', value: string): void
  (e: 'update:steps', value: number): void
  (e: 'update:cfgScale', value: number): void
  (e: 'update:seed', value: number): void
  (e: 'randomize-seed'): void
  (e: 'reuse-seed'): void
  (e: 'update:open', value: boolean): void
}>()

const isOpen = computed(() => !props.collapsible || props.open)

function toggleOpen(): void {
  emit('update:open', !isOpen.value)
}
</script>
