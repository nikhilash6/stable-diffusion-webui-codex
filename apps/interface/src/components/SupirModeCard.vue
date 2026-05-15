<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Presentational SUPIR mode configuration card for native SDXL img2img/inpaint.
Renders the SUPIR-specific nested-owner controls that stay outside Basic Parameters: variant, control/restoration scales,
color-fix mode, and the bounded advanced restore-window knob owned by the local runtime. The body collapses when SUPIR mode is off;
sampler/scheduler stay owned by `Img2ImgBasicParametersCard.vue`, and the main SUPIR toggle stays in `QuickSettingsBar.vue`.

Symbols (top-level; keep in sync; no ghosts):
- `SupirModeCard` (component): Dedicated SUPIR mode UI card for truthful SDXL img2img/inpaint surfaces.
- `defaultSupirMode` / `COLOR_FIX_OPTIONS` / `SUPIR_PARAMETER_TOOLTIPS` (constant): Canonical fallback state instance plus truthful tooltip/select copy for the public SUPIR surface.
- `supir` / `variantChoices` / `blockingReason` (const): Normalized props used by the dedicated SUPIR control card.
- `supirAdvancedDirty` / `supirAdvancedOpen` (const): Derived advanced-surface state used to auto-open the advanced block when non-default runtime knobs are active.
- `clampFloat` (function): Clamps a numeric value to a `[min,max]` range.
-->

<template>
  <div class="gen-card refiner-card">
    <WanSubHeader title="SUPIR Mode">
      <div class="qs-row">
        <span :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', supir.enabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']">
          {{ supir.enabled ? 'Enabled' : 'Disabled' }}
        </span>
        <HoverTooltip
          v-if="supir.enabled"
          class="cdx-slider-field__label-tooltip"
          title="SUPIR Advanced"
          :content="SUPIR_PARAMETER_TOOLTIPS.advancedToggle"
          :wrapperFocusable="false"
        >
          <button
            :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', supirAdvancedOpen ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
            type="button"
            :disabled="disabled"
            :aria-pressed="supirAdvancedOpen"
            @click="supirAdvancedOpen = !supirAdvancedOpen"
          >
            Advanced {{ supirAdvancedOpen ? 'Visible' : 'Hidden' }}
          </button>
        </HoverTooltip>
      </div>
    </WanSubHeader>

    <p v-if="!supir.enabled && blockingReason" class="caption hr-hint">{{ blockingReason }}</p>

    <div v-if="supir.enabled" class="rf-grid">
      <p v-if="blockingReason" class="caption hr-hint">{{ blockingReason }}</p>

      <div class="gc-row">
        <div class="gc-col field">
          <label class="label-muted">
            <HoverTooltip
              class="cdx-slider-field__label-tooltip"
              title="SUPIR Variant"
              :content="SUPIR_PARAMETER_TOOLTIPS.variant"
            >
              <span class="cdx-slider-field__label-trigger">
                <span>Variant</span>
                <span class="cdx-slider-field__label-help" aria-hidden="true">?</span>
              </span>
            </HoverTooltip>
          </label>
          <select
            class="select-md"
            :disabled="disabled"
            :value="supir.variant"
            @change="emit('patch:supir', { variant: ($event.target as HTMLSelectElement).value as SupirModeFormState['variant'] })"
          >
            <option v-if="variantChoices.length === 0" value="">No SUPIR variants reported</option>
            <option
              v-for="choice in variantChoices"
              :key="choice.value"
              :value="choice.value"
              :disabled="!choice.available"
            >
              {{ choice.available ? choice.label : `${choice.label} (missing)` }}
            </option>
          </select>
        </div>

        <SliderField
          class="gc-col gc-col--wide"
          label="Control Scale"
          :tooltip="SUPIR_PARAMETER_TOOLTIPS.controlScale"
          tooltipTitle="SUPIR Control Scale"
          :modelValue="supir.controlScale"
          :min="0.01"
          :max="2"
          :step="0.01"
          :inputStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(value) => emit('patch:supir', { controlScale: value })"
        />

        <SliderField
          class="gc-col gc-col--wide"
          label="Restoration Scale"
          :tooltip="SUPIR_PARAMETER_TOOLTIPS.restorationScale"
          tooltipTitle="SUPIR Restoration Scale"
          :modelValue="supir.restorationScale"
          :min="0.01"
          :max="6"
          :step="0.05"
          :inputStep="0.05"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(value) => emit('patch:supir', { restorationScale: value })"
        />
      </div>

      <div class="gc-row">
        <div class="gc-col field">
          <label class="label-muted">
            <HoverTooltip
              class="cdx-slider-field__label-tooltip"
              title="SUPIR Color Fix"
              :content="SUPIR_PARAMETER_TOOLTIPS.colorFix"
            >
              <span class="cdx-slider-field__label-trigger">
                <span>Color fix</span>
                <span class="cdx-slider-field__label-help" aria-hidden="true">?</span>
              </span>
            </HoverTooltip>
          </label>
          <select
            class="select-md"
            :disabled="disabled"
            :value="supir.colorFix"
            @change="emit('patch:supir', { colorFix: ($event.target as HTMLSelectElement).value as SupirModeFormState['colorFix'] })"
          >
            <option v-for="choice in COLOR_FIX_OPTIONS" :key="choice.value" :value="choice.value">
              {{ choice.label }}
            </option>
          </select>
        </div>
      </div>

      <div v-if="supirAdvancedOpen" class="gc-row">
        <SliderField
          class="gc-col"
          label="Restore End (Sigma)"
          :tooltip="SUPIR_PARAMETER_TOOLTIPS.restoreCfgSTmin"
          tooltipTitle="SUPIR Restore End"
          :modelValue="supir.restoreCfgSTmin"
          :min="0"
          :max="5"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(value) => emit('patch:supir', { restoreCfgSTmin: clampFloat(value, 0, 5) })"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import type { SupirVariantChoice } from '../composables/useSupirDiagnostics'
import { createDefaultSupirModeFormState, type SupirModeFormState } from '../stores/model_tabs'
import HoverTooltip from './ui/HoverTooltip.vue'
import SliderField from './ui/SliderField.vue'
import WanSubHeader from './wan/WanSubHeader.vue'

const COLOR_FIX_OPTIONS = [
  { value: 'None', label: 'None' },
  { value: 'AdaIN', label: 'AdaIN' },
  { value: 'Wavelet', label: 'Wavelet' },
] as const

const SUPIR_PARAMETER_TOOLTIPS = {
  variant: [
    'Selects the SUPIR checkpoint branch.',
    '`v0Q` is the upstream quality-oriented branch; `v0F` is the fidelity-oriented branch.',
    'This UI only exposes variants that the backend reports under `/api/supir/models`.',
  ],
  controlScale: [
    'Strength of the SUPIR Stage-2 control signal.',
    'Higher values let the model intervene more strongly; lower values preserve more of the input image.',
    'This native surface keeps the runtime-owned positive range only.',
  ],
  restorationScale: [
    'Controls the strength of the restore blend inside the active restore window.',
    'Lower values pull harder toward the Stage-1 anchor; higher values soften that pull across the same sigma window.',
    'The restore cutoff itself is owned separately by `Restore End (Sigma)`.',
  ],
  colorFix: [
    'Optional post-decode color correction.',
    '`Wavelet` adjusts tonal distribution, `AdaIN` matches color statistics, and `None` leaves the decoded output unchanged.',
  ],
  advancedToggle: [
    'Shows the extra SUPIR control that is actually owned by this native runtime.',
    'Right now this advanced block only exposes `Restore End (Sigma)`.',
  ],
  restoreCfgSTmin: [
    'Sigma cutoff for the restore window.',
    'When the sampler sigma drops below this threshold, the extra restore blend stops.',
    'Lower values keep restore active deeper into the denoise process; higher values stop it earlier.',
  ],
} as const

const props = withDefaults(defineProps<{
  disabled?: boolean
  supir?: SupirModeFormState
  variantChoices?: readonly SupirVariantChoice[]
  blockingReason?: string
}>(), {
  disabled: false,
  supir: () => createDefaultSupirModeFormState(),
  variantChoices: () => [],
  blockingReason: '',
})

const defaultSupirMode = createDefaultSupirModeFormState()

const emit = defineEmits<{
  (e: 'patch:supir', value: Partial<SupirModeFormState>): void
}>()

const supir = computed(() => props.supir ?? defaultSupirMode)
const variantChoices = computed(() => (Array.isArray(props.variantChoices) ? props.variantChoices : []))
const blockingReason = computed(() => String(props.blockingReason || '').trim())
const supirAdvancedDirty = computed(() => Number(supir.value.restoreCfgSTmin) !== defaultSupirMode.restoreCfgSTmin)
const supirAdvancedOpen = ref(false)

watch(
  () => [Boolean(supir.value.enabled), supirAdvancedDirty.value] as const,
  ([enabled, dirty]) => {
    if (!enabled) {
      supirAdvancedOpen.value = false
      return
    }
    if (dirty) supirAdvancedOpen.value = true
  },
  { immediate: true },
)

function clampFloat(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}
</script>
