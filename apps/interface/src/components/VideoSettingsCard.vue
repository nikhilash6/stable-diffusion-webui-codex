<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared video generation settings card (frames + FPS).
Renders video frame-count and FPS controls plus an approximate duration label while leaving frame-alignment ownership to the calling family/workspace.

Symbols (top-level; keep in sync; no ghosts):
- `VideoSettingsCard` (component): Video settings card for video generation parameters.
- `durationLabel` (const): Computed duration label derived from frames/fps.
- `frameHintText` (const): Computed frame-contract hint text shown below the frame control.
- `coerceFrameInput` (function): Clamps/truncates frame-count input within the configured bounds without applying family-specific alignment rules.
- `onFramesUpdate` (function): Emits sanitized frame values for slider/input updates.
-->

<template>
  <div :class="['vid-card', { 'vid-card--embedded': embedded }]">
    <div class="vc-grid">
      <SliderField
        label="Frames"
        :modelValue="frames"
        :min="minFrames"
        :max="maxFrames"
        :step="frameStep"
        :inputStep="frameInputStep"
        :nudgeStep="frameNudgeStep"
        inputClass="cdx-input-w-sm"
        @update:modelValue="onFramesUpdate"
      >
        <template #right>
          <NumberStepperInput
            :modelValue="frames"
            :min="minFrames"
            :max="maxFrames"
            :step="frameInputStep"
            :nudgeStep="frameNudgeStep"
            :inputClass="'cdx-input-w-sm'"
            @update:modelValue="onFramesUpdate"
          />
        </template>
        <template #below>
          <span class="caption">{{ frameHintText }}</span>
        </template>
      </SliderField>
      <SliderField
        label="FPS"
        :modelValue="fps"
        :min="minFps"
        :max="maxFps"
        :step="1"
        :inputStep="1"
        :nudgeStep="1"
        inputClass="cdx-input-w-sm"
        @update:modelValue="(v) => emit('update:fps', v)"
      >
        <template #below>
          <span class="caption vc-duration">~ {{ durationLabel }}</span>
        </template>
      </SliderField>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import SliderField from './ui/SliderField.vue'
import NumberStepperInput from './ui/NumberStepperInput.vue'

const props = withDefaults(defineProps<{
  frames: number
  fps: number
  embedded?: boolean
  minFrames?: number
  maxFrames?: number
  frameStep?: number
  frameInputStep?: number
  frameNudgeStep?: number
  frameRuleLabel?: string
  minFps?: number
  maxFps?: number
}>(), {
  embedded: false,
  minFrames: 1,
  maxFrames: 1000,
  frameStep: 1,
  frameInputStep: 1,
  frameNudgeStep: 1,
  frameRuleLabel: '',
  minFps: 8,
  maxFps: 60,
})

const emit = defineEmits({
  'update:frames': (v: number) => true,
  'update:fps': (v: number) => true,
})

const durationLabel = computed(() => {
  const f = Number(props.frames) || 0
  const fr = Number(props.fps) || 1
  const seconds = fr > 0 ? f / fr : 0
  return seconds.toFixed(2) + 's'
})

const minFrames = computed(() => props.minFrames)
const maxFrames = computed(() => props.maxFrames)
const frameStep = computed(() => Math.max(1, Math.trunc(Number(props.frameStep) || 1)))
const frameInputStep = computed(() => Math.max(1, Math.trunc(Number(props.frameInputStep) || 1)))
const frameNudgeStep = computed(() => Math.max(1, Math.trunc(Number(props.frameNudgeStep) || frameStep.value)))
const frameHintText = computed(() => {
  const rule = String(props.frameRuleLabel || '').trim()
  if (!rule) return `min ${minFrames.value} · max ${maxFrames.value}`
  return `${rule} · min ${minFrames.value} · max ${maxFrames.value}`
})
const minFps = computed(() => props.minFps)
const maxFps = computed(() => props.maxFps)

function coerceFrameInput(rawValue: number): number {
  const min = Number(minFrames.value) || 1
  const max = Number(maxFrames.value) || min
  const numeric = Number.isFinite(rawValue) ? Math.trunc(rawValue) : min
  return Math.min(max, Math.max(min, numeric))
}

function onFramesUpdate(value: number): void {
  emit('update:frames', coerceFrameInput(value))
}
</script>

<!-- Uses shared styles (gen-card layout avoided to keep focus on video-only params) -->
