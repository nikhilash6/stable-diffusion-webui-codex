<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN stage parameter panel (High/Low).
Renders sampler/scheduler/steps/cfg/seed controls for a WAN stage, forwards optional sampler/scheduler recommendation lists into shared selectors, keeps scheduler selection explicit/non-empty, and preserves the same grouped recommendation/risk UI contract used by image tabs before emitting stage patches to the parent view.

Symbols (top-level; keep in sync; no ghosts):
- `WanStagePanel` (component): High/Low stage panel for WAN generation parameters.
- `recommendedSamplers` / `recommendedSchedulers` (props): Optional recommendation arrays forwarded into shared selectors.
- `updateStage` (function): Emits a patch for the stage params (`update:stage`).
- `randomizeSeed` (function): Stores the previous seed and sets seed to `-1` (random).
- `reuseSeed` (function): Reuses the last non-random seed when available.
-->

<template>
  <div :class="['gen-card', { 'gen-card--embedded': embedded }]">
    <div v-if="!embedded" class="row-split">
      <span class="label-muted">{{ title }}</span>
    </div>

    <div class="gc-row">
      <SamplerSelector
        class="gc-col"
        :samplers="samplers"
        :recommended-names="recommendedSamplers"
        :modelValue="stage.sampler"
        :label="samplerLabel"
        :allow-empty="true"
        emptyLabel="Inherit"
        :disabled="disabled"
        @update:modelValue="(v) => updateStage({ sampler: v })"
      />
      <SchedulerSelector
        class="gc-col"
        :schedulers="schedulers"
        :recommended-names="recommendedSchedulers"
        :modelValue="stage.scheduler"
        :label="schedulerLabel"
        :disabled="disabled"
        @update:modelValue="(v) => updateStage({ scheduler: v })"
      />
      <SliderField
        class="gc-col gc-col--wide"
        label="Steps"
        :modelValue="stage.steps"
        :min="1"
        :max="150"
        :step="1"
        :inputStep="1"
        :nudgeStep="1"
        inputClass="cdx-input-w-md"
        :disabled="disabled"
        @update:modelValue="(v) => updateStage({ steps: Math.trunc(v) })"
      />
    </div>
    <div class="gc-row">
      <div class="gc-col gc-col--wide field">
        <label class="label-muted">Seed</label>
        <div class="number-with-controls w-full">
          <input class="ui-input ui-input-sm pad-right" type="number" :disabled="disabled" :value="stage.seed" @change="updateStage({ seed: toInt($event, stage.seed) })" />
          <div class="stepper">
            <button class="step-btn" type="button" :disabled="disabled" title="Random seed" @click="randomizeSeed">🎲</button>
            <button class="step-btn" type="button" :disabled="disabled || lastSeed === null" title="Reuse seed" @click="reuseSeed">↺</button>
          </div>
        </div>
      </div>
      <SliderField
        class="gc-col gc-col--wide"
        label="CFG"
        :modelValue="stage.cfgScale"
        :min="0"
        :max="30"
        :step="0.5"
        :inputStep="0.5"
        :nudgeStep="0.5"
        inputClass="cdx-input-w-md"
        :disabled="disabled"
        @update:modelValue="(v) => updateStage({ cfgScale: v })"
      />
    </div>
    <div v-if="showModelDir" class="gc-row">
      <div class="gc-col field">
        <label class="label-muted">Model Dir</label>
        <input class="ui-input" type="text" :disabled="disabled" :value="stage.modelDir" @change="updateStage({ modelDir: ($event.target as HTMLInputElement).value })" placeholder="/path/to/high-or-low" />
      </div>
    </div>

    <div v-if="showModelDir && !stage.modelDir" class="panel-error">{{ title }}: model directory is empty.</div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

import type { SamplerInfo, SchedulerInfo } from '../../api/types'
import type { WanStageParams } from '../../stores/model_tabs'

import SamplerSelector from '../SamplerSelector.vue'
import SchedulerSelector from '../SchedulerSelector.vue'
import SliderField from '../ui/SliderField.vue'

const props = withDefaults(defineProps<{
  title: string
  stage: WanStageParams
  samplers: SamplerInfo[]
  schedulers: SchedulerInfo[]
  recommendedSamplers?: string[] | null
  recommendedSchedulers?: string[] | null
  showModelDir?: boolean
  embedded?: boolean
  disabled?: boolean
  samplerLabel?: string
  schedulerLabel?: string
}>(), {
  showModelDir: false,
  embedded: false,
  disabled: false,
  samplerLabel: 'Sampler',
  schedulerLabel: 'Scheduler',
})

const emit = defineEmits<{
  (e: 'update:stage', patch: Partial<WanStageParams>): void
}>()

const lastSeed = ref<number | null>(null)

const samplerLabel = computed(() => props.samplerLabel)
const schedulerLabel = computed(() => props.schedulerLabel)

function updateStage(patch: Partial<WanStageParams>): void {
  emit('update:stage', patch)
}

function toInt(e: Event, fallback: number): number {
  const v = Number((e.target as HTMLInputElement).value)
  return Number.isFinite(v) ? Math.trunc(v) : fallback
}

function randomizeSeed(): void {
  if (props.stage.seed !== -1) lastSeed.value = props.stage.seed
  updateStage({ seed: -1 })
}

function reuseSeed(): void {
  if (lastSeed.value !== null) updateStage({ seed: lastSeed.value })
}
</script>
