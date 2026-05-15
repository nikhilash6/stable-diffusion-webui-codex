<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Generic core video-parameter card for video workspaces.
Renders width/height controls, shared frames/FPS settings, and optional family-owned footer/temporal sections
without owning any family-specific runtime behavior.

Symbols (top-level; keep in sync; no ghosts):
- `VideoCoreParamsCard` (component): Generic video core-params card.
-->

<template>
  <div class="gen-card cdx-video-card">
    <div class="cdx-video-card-header">
      <div class="cdx-video-card-header__left">
        <span class="cdx-video-card-header__title">{{ title }}</span>
      </div>
      <div v-if="$slots['header-actions']" class="cdx-video-card-header__right">
        <slot name="header-actions" />
      </div>
    </div>

    <div class="gc-row mt-2">
      <SliderField
        class="gc-col gc-col--wide"
        label="Width (px)"
        :modelValue="width"
        :min="widthMin"
        :max="widthMax"
        :step="widthStep"
        :inputStep="widthInputStep"
        :nudgeStep="widthNudgeStep"
        :disabled="disabled"
        inputClass="cdx-input-w-md"
        @update:modelValue="(value) => emit('update:width', value)"
      >
        <template v-if="$slots['width-right']" #right>
          <slot name="width-right" />
        </template>
        <template v-if="$slots['width-below']" #below>
          <slot name="width-below" />
        </template>
      </SliderField>
      <SliderField
        class="gc-col gc-col--wide"
        label="Height (px)"
        :modelValue="height"
        :min="heightMin"
        :max="heightMax"
        :step="heightStep"
        :inputStep="heightInputStep"
        :nudgeStep="heightNudgeStep"
        :disabled="disabled"
        inputClass="cdx-input-w-md"
        @update:modelValue="(value) => emit('update:height', value)"
      />
    </div>

    <VideoSettingsCard
      embedded
      :frames="frames"
      :fps="fps"
      :minFrames="minFrames"
      :maxFrames="maxFrames"
      :frameStep="frameStep"
      :frameInputStep="frameInputStep"
      :frameNudgeStep="frameNudgeStep"
      :frameRuleLabel="frameRuleLabel"
      :minFps="minFps"
      :maxFps="maxFps"
      @update:frames="(value) => emit('update:frames', value)"
      @update:fps="(value) => emit('update:fps', value)"
    />

    <div v-if="$slots.default" class="mt-2 cdx-video-card-body">
      <slot />
    </div>
    <div v-if="showTemporalSection && $slots.temporal" class="mt-2 cdx-video-card-body">
      <slot name="temporal" />
    </div>
  </div>
</template>

<script setup lang="ts">
import VideoSettingsCard from '../VideoSettingsCard.vue'
import SliderField from '../ui/SliderField.vue'

withDefaults(defineProps<{
  title?: string
  width: number
  height: number
  widthMin?: number
  widthMax?: number
  widthStep?: number
  widthInputStep?: number
  widthNudgeStep?: number
  heightMin?: number
  heightMax?: number
  heightStep?: number
  heightInputStep?: number
  heightNudgeStep?: number
  frames: number
  fps: number
  minFrames?: number
  maxFrames?: number
  frameStep?: number
  frameInputStep?: number
  frameNudgeStep?: number
  frameRuleLabel?: string
  minFps?: number
  maxFps?: number
  disabled?: boolean
  showTemporalSection?: boolean
}>(), {
  title: 'Video',
  widthMin: 1,
  widthMax: 4096,
  widthStep: 1,
  widthInputStep: 1,
  widthNudgeStep: 1,
  heightMin: 1,
  heightMax: 4096,
  heightStep: 1,
  heightInputStep: 1,
  heightNudgeStep: 1,
  minFrames: 1,
  maxFrames: 1000,
  frameStep: 1,
  frameInputStep: 1,
  frameNudgeStep: 1,
  frameRuleLabel: '',
  minFps: 1,
  maxFps: 60,
  disabled: false,
  showTemporalSection: false,
})

const emit = defineEmits<{
  (e: 'update:width', value: number): void
  (e: 'update:height', value: number): void
  (e: 'update:frames', value: number): void
  (e: 'update:fps', value: number): void
}>()
</script>
