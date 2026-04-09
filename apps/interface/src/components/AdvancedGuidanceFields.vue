<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared advanced CFG/APG field rows for image and refiner cards.
Renders the repeated three-row advanced-guidance field matrix (`Guidance Rescale`, `CFG Trunc Ratio`, `Renorm CFG`,
`APG Start`, `APG Eta`, `APG Rescale`, `APG Momentum`, `APG Norm`) with variant-aware layout classes, capability gating,
optional tooltip copy, and clamped partial-patch emits.

Symbols (top-level; keep in sync; no ghosts):
- `AdvancedGuidanceFields` (component): Shared advanced-guidance field owner for Basic Parameters and Refiner cards.
- `ADVANCED_GUIDANCE_TOOLTIPS` (constant): Optional tooltip copy for the advanced-guidance sliders.
- `tooltipProps` (function): Returns tooltip props only when the caller enables them.
-->

<template>
  <div
    v-if="showPrimaryRow"
    :class="primaryRowClass"
  >
    <SliderField
      v-if="hasGuidanceSupport(guidanceSupport, 'guidance_rescale')"
      :class="columnClass"
      label="Guidance Rescale"
      :modelValue="guidanceAdvanced.guidanceRescale"
      :min="0"
      :max="1"
      :step="0.01"
      :inputStep="0.01"
      :nudgeStep="0.01"
      inputClass="cdx-input-w-md"
      :disabled="disabled"
      v-bind="tooltipProps('Guidance Rescale', ADVANCED_GUIDANCE_TOOLTIPS.guidanceRescale)"
      @update:modelValue="(value) => emit('patch', { guidanceRescale: clampFloat(value, 0, 1) })"
    />

    <SliderField
      v-if="hasGuidanceSupport(guidanceSupport, 'cfg_trunc_ratio')"
      :class="columnClass"
      label="CFG Trunc Ratio"
      :modelValue="guidanceAdvanced.cfgTruncRatio"
      :min="0"
      :max="1"
      :step="0.01"
      :inputStep="0.01"
      :nudgeStep="0.01"
      inputClass="cdx-input-w-md"
      :disabled="disabled"
      v-bind="tooltipProps('CFG Trunc Ratio', ADVANCED_GUIDANCE_TOOLTIPS.cfgTruncRatio)"
      @update:modelValue="(value) => emit('patch', { cfgTruncRatio: clampFloat(value, 0, 1) })"
    />

    <SliderField
      v-if="hasGuidanceSupport(guidanceSupport, 'renorm_cfg')"
      :class="columnClass"
      label="Renorm CFG"
      :modelValue="guidanceAdvanced.renormCfg"
      :min="0"
      :max="4"
      :step="0.05"
      :inputStep="0.05"
      :nudgeStep="0.05"
      inputClass="cdx-input-w-md"
      :disabled="disabled"
      v-bind="tooltipProps('Renorm CFG', ADVANCED_GUIDANCE_TOOLTIPS.renormCfg)"
      @update:modelValue="(value) => emit('patch', { renormCfg: clampFloat(value, 0, 4) })"
    />
  </div>

  <div
    v-if="showSecondaryRow"
    :class="secondaryRowClass"
  >
    <SliderField
      v-if="hasGuidanceSupport(guidanceSupport, 'apg_start_step')"
      :class="columnClass"
      label="APG Start"
      :modelValue="guidanceAdvanced.apgStartStep"
      :min="0"
      :max="apgStartMax"
      :step="1"
      :inputStep="1"
      :nudgeStep="1"
      inputClass="cdx-input-w-md"
      :disabled="disabled"
      v-bind="tooltipProps('APG Start', ADVANCED_GUIDANCE_TOOLTIPS.apgStartStep)"
      @update:modelValue="(value) => emit('patch', { apgStartStep: clampInt(value, 0, apgStartMax) })"
    />

    <SliderField
      v-if="hasGuidanceSupport(guidanceSupport, 'apg_eta')"
      :class="columnClass"
      label="APG Eta"
      :modelValue="guidanceAdvanced.apgEta"
      :min="-1"
      :max="1"
      :step="0.01"
      :inputStep="0.01"
      :nudgeStep="0.01"
      inputClass="cdx-input-w-md"
      :disabled="disabled"
      v-bind="tooltipProps('APG Eta', ADVANCED_GUIDANCE_TOOLTIPS.apgEta)"
      @update:modelValue="(value) => emit('patch', { apgEta: clampFloat(value, -1, 1) })"
    />

    <SliderField
      v-if="hasGuidanceSupport(guidanceSupport, 'apg_rescale')"
      :class="columnClass"
      label="APG Rescale"
      :modelValue="guidanceAdvanced.apgRescale"
      :min="0"
      :max="1"
      :step="0.01"
      :inputStep="0.01"
      :nudgeStep="0.01"
      inputClass="cdx-input-w-md"
      :disabled="disabled"
      v-bind="tooltipProps('APG Rescale', ADVANCED_GUIDANCE_TOOLTIPS.apgRescale)"
      @update:modelValue="(value) => emit('patch', { apgRescale: clampFloat(value, 0, 1) })"
    />
  </div>

  <div
    v-if="showTertiaryRow"
    :class="tertiaryRowClass"
  >
    <SliderField
      v-if="hasGuidanceSupport(guidanceSupport, 'apg_momentum')"
      :class="columnClass"
      label="APG Momentum"
      :modelValue="guidanceAdvanced.apgMomentum"
      :min="0"
      :max="0.99"
      :step="0.01"
      :inputStep="0.01"
      :nudgeStep="0.01"
      inputClass="cdx-input-w-md"
      :disabled="disabled"
      v-bind="tooltipProps('APG Momentum', ADVANCED_GUIDANCE_TOOLTIPS.apgMomentum)"
      @update:modelValue="(value) => emit('patch', { apgMomentum: clampFloat(value, 0, 0.99) })"
    />

    <SliderField
      v-if="hasGuidanceSupport(guidanceSupport, 'apg_norm_threshold')"
      :class="columnClass"
      label="APG Norm"
      :modelValue="guidanceAdvanced.apgNormThreshold"
      :min="0"
      :max="40"
      :step="0.1"
      :inputStep="0.1"
      :nudgeStep="0.1"
      inputClass="cdx-input-w-md"
      :disabled="disabled"
      v-bind="tooltipProps('APG Norm', ADVANCED_GUIDANCE_TOOLTIPS.apgNormThreshold)"
      @update:modelValue="(value) => emit('patch', { apgNormThreshold: clampFloat(value, 0, 40) })"
    />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { GuidanceAdvancedCapabilities } from '../api/types'
import { DEFAULT_GUIDANCE_ADVANCED_PARAMS, type GuidanceAdvancedParams } from '../stores/model_tabs'
import { hasGuidanceSupport } from '../utils/guidance_advanced'
import SliderField from './ui/SliderField.vue'

type AdvancedGuidanceVariant = 'basic' | 'refiner'

const ADVANCED_GUIDANCE_TOOLTIPS = {
  apgStartStep: [
    'Sets the step where APG starts influencing guidance.',
    'Increase: APG starts later, so early composition stays closer to plain CFG.',
    'Decrease: APG starts earlier, so APG shapes structure sooner.',
    'Example: for 30 steps, APG Start 24 = late subtle change; APG Start 4 = early visible change.',
    'Neutral value: set it equal to total Steps (APG is almost off).',
  ],
  apgEta: [
    'Controls APG steering aggressiveness.',
    'Increase: stronger prompt steering, usually cleaner subject intent but can look over-shaped.',
    'Decrease: weaker steering, usually softer and more natural transitions.',
    'Example: on portraits, higher Eta can tighten facial features; lower Eta keeps looser detail.',
    'Neutral value: 0.00.',
  ],
  apgMomentum: [
    'Controls how much APG carries memory from previous steps.',
    'Increase: smoother, more stable behavior across steps.',
    'Decrease: faster reaction, but can introduce jitter between nearby details.',
    'Example: higher Momentum can stabilize repeating textures; lower Momentum can make them flicker more.',
    'Neutral value: 0.00.',
  ],
  apgRescale: [
    'Final strength multiplier for APG.',
    'Increase: APG has more visible influence on the image.',
    'Decrease: APG influence fades out.',
    'Example: for anime line-art, higher Rescale can sharpen contour intent; lower Rescale keeps softer edges.',
    'Neutral value: 0.00.',
  ],
  apgNormThreshold: [
    'Threshold that limits extreme guidance spikes.',
    'Increase: less limiting, more freedom (and more risk of overshoot artifacts).',
    'Decrease: clamps earlier, more controlled highlights/contrast.',
    'Example: if bright areas blow out, lowering Norm Threshold can recover smoother lighting.',
    'Neutral value: 40.0 (near no clamping).',
  ],
  guidanceRescale: [
    'Rebalances final CFG to reduce overcooked outputs.',
    'Increase: tones down harsh CFG artifacts and oversaturation.',
    'Decrease: keeps raw CFG punch and stronger contrast swings.',
    'Example: if skin turns plastic or neon, raising Guidance Rescale often makes it more natural.',
    'Neutral value: 0.00.',
  ],
  cfgTruncRatio: [
    'Truncates part of CFG to reduce instability.',
    'Increase: less truncation, behavior closer to regular CFG.',
    'Decrease: more truncation, safer but more conservative results.',
    'Example: if high-CFG runs create halos/ringing, lowering Trunc Ratio can calm those artifacts.',
    'Neutral value: 1.00 (no truncation).',
  ],
  renormCfg: [
    'Renormalizes CFG magnitude for stability.',
    'Increase: stronger normalization, usually fewer contrast explosions.',
    'Decrease: weaker normalization, more raw CFG response.',
    'Example: if edges look crunchy at high CFG, increasing Renorm CFG can soften those spikes.',
    'Neutral value: 0.00.',
  ],
} as const

const props = withDefaults(defineProps<{
  guidanceAdvanced?: GuidanceAdvancedParams
  guidanceSupport?: GuidanceAdvancedCapabilities | null
  maxApgStartStep?: number
  disabled?: boolean
  variant?: AdvancedGuidanceVariant
  showTooltips?: boolean
}>(), {
  guidanceAdvanced: () => ({ ...DEFAULT_GUIDANCE_ADVANCED_PARAMS }),
  guidanceSupport: null,
  maxApgStartStep: 150,
  disabled: false,
  variant: 'basic',
  showTooltips: false,
})

const emit = defineEmits<{
  (e: 'patch', patch: Partial<GuidanceAdvancedParams>): void
}>()

const guidanceAdvanced = computed(() => props.guidanceAdvanced ?? DEFAULT_GUIDANCE_ADVANCED_PARAMS)
const guidanceSupport = computed(() => props.guidanceSupport ?? null)
const apgStartMax = computed(() => {
  const value = Number(props.maxApgStartStep)
  if (!Number.isFinite(value) || value < 0) return 150
  return Math.trunc(value)
})

const columnClass = computed(() => (props.variant === 'refiner' ? 'rf-col' : 'gc-col'))
const primaryRowClass = computed(() => (props.variant === 'refiner' ? 'rf-row rf-row--advanced' : 'gc-row cfg-advanced-row'))
const secondaryRowClass = computed(() => (props.variant === 'refiner' ? 'rf-row rf-row--advanced' : 'gc-row cfg-advanced-row cfg-advanced-row--secondary'))
const tertiaryRowClass = computed(() => (props.variant === 'refiner' ? 'rf-row rf-row--advanced-secondary' : 'gc-row cfg-advanced-row cfg-advanced-row--secondary'))

const showPrimaryRow = computed(() => (
  hasGuidanceSupport(guidanceSupport.value, 'guidance_rescale')
    || hasGuidanceSupport(guidanceSupport.value, 'cfg_trunc_ratio')
    || hasGuidanceSupport(guidanceSupport.value, 'renorm_cfg')
))

const showSecondaryRow = computed(() => (
  hasGuidanceSupport(guidanceSupport.value, 'apg_start_step')
    || hasGuidanceSupport(guidanceSupport.value, 'apg_eta')
    || hasGuidanceSupport(guidanceSupport.value, 'apg_rescale')
))

const showTertiaryRow = computed(() => (
  hasGuidanceSupport(guidanceSupport.value, 'apg_momentum')
    || hasGuidanceSupport(guidanceSupport.value, 'apg_norm_threshold')
))

function clampFloat(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, Math.trunc(value)))
}

function tooltipProps(title: string, content: readonly string[]): { tooltip?: readonly string[]; tooltipTitle?: string } {
  if (!props.showTooltips) return {}
  return { tooltip: content, tooltipTitle: title }
}
</script>
