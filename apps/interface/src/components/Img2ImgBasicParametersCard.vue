<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Img2img-focused Basic Parameters card with hires-like structure.
Renders sampler/scheduler/steps, dimensions, optional resize-mode + upscaler controls, and seed/CFG with optional denoise
for init-image mode without hires-only prompt/checkpoint swap controls, backend recommendation-aware sampler/scheduler selector grouping, plus optional advanced CFG/APG controls
gated by per-engine capabilities. When native SDXL SUPIR mode is enabled, the card swaps the generic sampler/scheduler row for the truthful SUPIR sampler surface
and shows the locked native scheduler derived from backend SUPIR diagnostics, while the SUPIR-specific knobs stay on the dedicated `SupirModeCard.vue`.
The resize-type selector can now receive an engine-scoped truthful option subset and hides the upscaler field when the active engine does not expose an upscaler-backed resize mode.

Symbols (top-level; keep in sync; no ghosts):
- `Img2ImgBasicParametersCard` (component): Img2img parameters card used when init image mode is active.
- `hasGuidanceSupport` (function): Returns whether a specific advanced-guidance control is supported by the active engine capability contract.
- `clampFloat` (function): Clamps a numeric value to a `[min,max]` range.
- `clampInt` (function): Clamps and truncates a numeric value to an integer range.
- `clampIntToStep` (function): Clamps and snaps an integer value to a step size.
- `aspectRatioLocked` (ref): Local lock state for Width/Height proportional editing.
- `toggleAspectRatioLock` (function): Toggles aspect-ratio lock and captures current ratio anchor.
- `onWidthDimensionChange` (function): Handles width updates and keeps height in sync when lock is enabled.
- `onHeightDimensionChange` (function): Handles height updates and keeps width in sync when lock is enabled.
- `onSeedChange` (function): Handles manual seed input changes and emits a normalized integer seed.
- `onResizeModeChange` (function): Emits normalized resize-mode updates from the resize-type select.
- `onUpscalerChange` (function): Emits upscaler selection updates.
- `showsUpscalerResizeMode` (const): Whether the active resize-mode option subset exposes an upscaler-backed mode.
- `recommendedSamplers` / `recommendedSchedulers` (const): Optional recommendation arrays forwarded into selector components.
- `patchGuidanceAdvanced` (function): Emits partial updates for nested advanced-guidance state.
- `toggleGuidanceAdvanced` (function): Toggles Advanced guidance mode and auto-syncs APG/CFG trunc activation flags when supported.
- `defaultSupirMode` / `SUPIR_PARAMETER_TOOLTIPS` (const): Canonical fallback SUPIR state instance and bounded tooltip copy for the SUPIR sampler/scheduler row.
- `supir` / `supirEnabled` / `supirLockedScheduler` / `supirHasStaleSamplerSelection` (const): Derived SUPIR state used to swap the Basic Parameters sampler surface when SUPIR mode is active and keep stale saved selections visible.
- `swapWH` (function): Swaps width/height while respecting min/max and step constraints.
-->

<template>
  <div class="gen-card img2img-basic-card">
    <WanSubHeader title="Basic Parameters" />
    <div class="gc-stack">
      <div class="gc-row">
        <template v-if="supirEnabled">
          <div class="gc-col field">
            <label class="label-muted">
              <HoverTooltip
                class="cdx-slider-field__label-tooltip"
                title="SUPIR Sampler"
                :content="SUPIR_PARAMETER_TOOLTIPS.sampler"
              >
                <span class="cdx-slider-field__label-trigger">
                  <span>Sampler</span>
                  <span class="cdx-slider-field__label-help" aria-hidden="true">?</span>
                </span>
              </HoverTooltip>
            </label>
            <select
              class="select-md"
              :disabled="disabled"
              :value="supir.sampler"
              @change="emit('patch:supir', { sampler: ($event.target as HTMLSelectElement).value })"
            >
              <option v-if="supirSamplerChoices.length === 0" value="">No SUPIR samplers reported</option>
              <option v-if="supirHasStaleSamplerSelection" :value="supir.sampler">Invalid selection: {{ supir.sampler }}</option>
              <option v-for="choice in supirSamplerChoices" :key="choice.id" :value="choice.id">
                {{ choice.label }}
              </option>
            </select>
          </div>
          <div class="gc-col field">
            <label class="label-muted">
              <HoverTooltip
                class="cdx-slider-field__label-tooltip"
                title="SUPIR Scheduler"
                :content="SUPIR_PARAMETER_TOOLTIPS.scheduler"
              >
                <span class="cdx-slider-field__label-trigger">
                  <span>Scheduler</span>
                  <span class="cdx-slider-field__label-help" aria-hidden="true">?</span>
                </span>
              </HoverTooltip>
            </label>
            <input class="ui-input ui-input-sm" type="text" :value="supirLockedScheduler || 'Unavailable'" disabled readonly />
          </div>
        </template>
        <template v-else>
          <SamplerSelector
            class="gc-col"
            :samplers="samplers"
            :recommended-names="recommendedSamplers"
            :modelValue="sampler"
            label="Sampler"
            :allow-empty="false"
            :disabled="disabled"
            @update:modelValue="(value) => emit('update:sampler', value)"
          />
          <SchedulerSelector
            class="gc-col"
            :schedulers="schedulers"
            :recommended-names="recommendedSchedulers"
            :modelValue="scheduler"
            label="Scheduler"
            :allow-empty="false"
            :disabled="disabled"
            @update:modelValue="(value) => emit('update:scheduler', value)"
          />
        </template>
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
          @update:modelValue="(value) => emit('update:steps', clampInt(value, minSteps, maxSteps))"
        />
      </div>
      <p v-if="supirEnabled && supirBlockingReason" class="hr-hint">{{ supirBlockingReason }}</p>

      <div class="gc-row">
        <SliderField
          class="gc-col"
          label="Width"
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
          class="gc-col"
          label="Height"
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
      </div>

      <div v-if="props.showResizeMode !== false" class="gc-row">
        <div class="gc-col field">
          <label class="label-muted">Resize type</label>
          <select class="select-md" :disabled="disabled" :value="resizeModeValue" @change="onResizeModeChange">
            <option v-for="option in resizeModeOptions" :key="option.value" :value="option.value">
              {{ option.label }}
            </option>
          </select>
        </div>

        <div v-if="showsUpscalerResizeMode" class="gc-col field">
          <label class="label-muted">Upscaler</label>
          <select
            class="select-md"
            :value="upscaler"
            :disabled="disabled || !isUpscalerResizeMode || upscalersLoading"
            @change="onUpscalerChange"
          >
            <option v-if="upscalersLoading" :value="upscaler">Loading…</option>
            <option v-else-if="upscaler && !isUpscalerKnown" :value="upscaler">Invalid selection: {{ upscaler }}</option>
            <option v-else value="" disabled>Select</option>
            <optgroup v-if="spandrelUpscalers.length" label="Spandrel (pixel SR)">
              <option v-for="entry in spandrelUpscalers" :key="entry.id" :value="entry.id">{{ entry.label }}</option>
            </optgroup>
            <optgroup v-if="latentUpscalers.length" label="Latent">
              <option v-for="entry in latentUpscalers" :key="entry.id" :value="entry.id">{{ entry.label }}</option>
            </optgroup>
          </select>
          <p class="hr-hint" v-if="upscalersError">Error: {{ upscalersError }}</p>
          <p class="hr-hint" v-else-if="isUpscalerResizeMode && upscaler && !isUpscalerKnown">Select an upscaler id from `GET /api/upscalers`.</p>
        </div>
      </div>

      <div class="gc-row">
        <div class="gc-col field">
          <label class="label-muted">Seed</label>
          <div class="number-with-controls w-full">
            <input class="ui-input ui-input-sm pad-right" type="number" :disabled="disabled" :value="seed" @change="onSeedChange" />
            <div class="stepper">
              <button class="step-btn" type="button" :disabled="disabled" title="Random seed" @click="emit('random-seed')">🎲</button>
              <button class="step-btn" type="button" :disabled="disabled" title="Reuse seed" @click="emit('reuse-seed')">↺</button>
            </div>
          </div>
        </div>

        <div v-if="showClipSkip" class="gc-col field">
          <label class="label-muted">CLIP Skip</label>
          <NumberStepperInput
            :modelValue="clipSkip"
            :min="minClipSkip"
            :max="maxClipSkip"
            :step="1"
            :nudgeStep="1"
            inputClass="cdx-input-w-xs"
            :disabled="disabled"
            @update:modelValue="(value) => emit('update:clipSkip', clampInt(value, minClipSkip, maxClipSkip))"
          />
        </div>

        <SliderField
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
          @update:modelValue="(value) => emit('update:cfgScale', clampFloat(value, minCfg, maxCfg))"
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
              @update:modelValue="(value) => emit('update:cfgScale', clampFloat(value, minCfg, maxCfg))"
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
          @update:modelValue="(value) => emit('update:denoiseStrength', clampFloat(value, 0, 1))"
        />
      </div>

      <div
        v-if="showGuidanceAdvancedRow && (hasGuidanceSupport('guidance_rescale') || hasGuidanceSupport('cfg_trunc_ratio') || hasGuidanceSupport('renorm_cfg'))"
        class="gc-row cfg-advanced-row"
      >
        <SliderField
          v-if="hasGuidanceSupport('guidance_rescale')"
          class="gc-col"
          label="Guidance Rescale"
          :tooltip="ADVANCED_GUIDANCE_TOOLTIPS.guidanceRescale"
          tooltipTitle="Guidance Rescale"
          :modelValue="guidanceAdvanced.guidanceRescale"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ guidanceRescale: clampFloat(v, 0, 1) })"
        />

        <SliderField
          v-if="hasGuidanceSupport('cfg_trunc_ratio')"
          class="gc-col"
          label="CFG Trunc Ratio"
          :tooltip="ADVANCED_GUIDANCE_TOOLTIPS.cfgTruncRatio"
          tooltipTitle="CFG Trunc Ratio"
          :modelValue="guidanceAdvanced.cfgTruncRatio"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ cfgTruncRatio: clampFloat(v, 0, 1) })"
        />

        <SliderField
          v-if="hasGuidanceSupport('renorm_cfg')"
          class="gc-col"
          label="Renorm CFG"
          :tooltip="ADVANCED_GUIDANCE_TOOLTIPS.renormCfg"
          tooltipTitle="Renorm CFG"
          :modelValue="guidanceAdvanced.renormCfg"
          :min="0"
          :max="4"
          :step="0.05"
          :inputStep="0.05"
          :nudgeStep="0.05"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ renormCfg: clampFloat(v, 0, 4) })"
        />
      </div>

      <div
        v-if="showGuidanceAdvancedRow && (hasGuidanceSupport('apg_start_step') || hasGuidanceSupport('apg_eta') || hasGuidanceSupport('apg_rescale'))"
        class="gc-row cfg-advanced-row cfg-advanced-row--secondary"
      >
        <SliderField
          v-if="hasGuidanceSupport('apg_start_step')"
          class="gc-col"
          label="APG Start"
          :tooltip="ADVANCED_GUIDANCE_TOOLTIPS.apgStartStep"
          tooltipTitle="APG Start"
          :modelValue="guidanceAdvanced.apgStartStep"
          :min="0"
          :max="maxSteps"
          :step="1"
          :inputStep="1"
          :nudgeStep="1"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ apgStartStep: clampInt(v, 0, maxSteps) })"
        />

        <SliderField
          v-if="hasGuidanceSupport('apg_eta')"
          class="gc-col"
          label="APG Eta"
          :tooltip="ADVANCED_GUIDANCE_TOOLTIPS.apgEta"
          tooltipTitle="APG Eta"
          :modelValue="guidanceAdvanced.apgEta"
          :min="-1"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ apgEta: clampFloat(v, -1, 1) })"
        />

        <SliderField
          v-if="hasGuidanceSupport('apg_rescale')"
          class="gc-col"
          label="APG Rescale"
          :tooltip="ADVANCED_GUIDANCE_TOOLTIPS.apgRescale"
          tooltipTitle="APG Rescale"
          :modelValue="guidanceAdvanced.apgRescale"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ apgRescale: clampFloat(v, 0, 1) })"
        />
      </div>

      <div
        v-if="showGuidanceAdvancedRow && (hasGuidanceSupport('apg_momentum') || hasGuidanceSupport('apg_norm_threshold'))"
        class="gc-row cfg-advanced-row cfg-advanced-row--secondary"
      >
        <SliderField
          v-if="hasGuidanceSupport('apg_momentum')"
          class="gc-col"
          label="APG Momentum"
          :tooltip="ADVANCED_GUIDANCE_TOOLTIPS.apgMomentum"
          tooltipTitle="APG Momentum"
          :modelValue="guidanceAdvanced.apgMomentum"
          :min="0"
          :max="0.99"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ apgMomentum: clampFloat(v, 0, 0.99) })"
        />

        <SliderField
          v-if="hasGuidanceSupport('apg_norm_threshold')"
          class="gc-col"
          label="APG Norm"
          :tooltip="ADVANCED_GUIDANCE_TOOLTIPS.apgNormThreshold"
          tooltipTitle="APG Norm"
          :modelValue="guidanceAdvanced.apgNormThreshold"
          :min="0"
          :max="40"
          :step="0.1"
          :inputStep="0.1"
          :nudgeStep="0.1"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ apgNormThreshold: clampFloat(v, 0, 40) })"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type {
  GuidanceAdvancedCapabilities,
  SamplerInfo,
  SchedulerInfo,
  SupirSamplerInfo,
  UpscalerDefinition,
} from '../api/types'
import { createDefaultSupirModeFormState, type GuidanceAdvancedParams, type SupirModeFormState } from '../stores/model_tabs'

import NumberStepperInput from './ui/NumberStepperInput.vue'
import HoverTooltip from './ui/HoverTooltip.vue'
import SliderField from './ui/SliderField.vue'
import SamplerSelector from './SamplerSelector.vue'
import SchedulerSelector from './SchedulerSelector.vue'
import WanSubHeader from './wan/WanSubHeader.vue'
import {
  IMG2IMG_RESIZE_MODE_OPTIONS,
  normalizeImg2ImgResizeModeFromOptions,
  type Img2ImgResizeModeOption,
  type Img2ImgResizeMode,
} from '../utils/img2img_resize'

const DEFAULT_GUIDANCE_ADVANCED: GuidanceAdvancedParams = {
  enabled: false,
  apgEnabled: false,
  apgStartStep: 0,
  apgEta: 0,
  apgMomentum: 0,
  apgNormThreshold: 15,
  apgRescale: 0,
  guidanceRescale: 0,
  cfgTruncEnabled: false,
  cfgTruncRatio: 0.8,
  renormCfg: 0,
}

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

const SUPIR_PARAMETER_TOOLTIPS = {
  sampler: [
    'Selects the repo-owned SUPIR restore sampler surface.',
    'The current public inventory is the stable sampler set reported by `/api/supir/models`.',
    'The locked scheduler shown below comes from backend diagnostics for the selected SUPIR sampler.',
  ],
  scheduler: [
    'Read-only.',
    'Native SUPIR mode maps each public SUPIR sampler to one backend-owned SDXL sampler/scheduler tuple.',
    'In the current stable public surface, every exposed SUPIR sampler maps to the `karras` scheduler.',
  ],
} as const

const props = withDefaults(defineProps<{
  samplers: SamplerInfo[]
  schedulers: SchedulerInfo[]
  recommendedSamplers?: string[] | null
  recommendedSchedulers?: string[] | null
  upscalers?: UpscalerDefinition[]
  upscalersLoading?: boolean
  upscalersError?: string
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  denoiseStrength: number
  showDenoise?: boolean
  seed: number
  width: number
  height: number
  upscaler: string
  resizeMode: Img2ImgResizeMode
  resizeModeOptions?: readonly Img2ImgResizeModeOption[]
  showResizeMode?: boolean
  dimensionSnapMode?: 'nearest' | 'floor'
  disabled?: boolean
  minSteps?: number
  maxSteps?: number
  minCfg?: number
  maxCfg?: number
  cfgStep?: number
  cfgLabel?: string
  minWidth?: number
  maxWidth?: number
  minHeight?: number
  maxHeight?: number
  widthStep?: number
  widthInputStep?: number
  heightStep?: number
  heightInputStep?: number
  showInitImageDims?: boolean
  showClipSkip?: boolean
  clipSkip?: number
  minClipSkip?: number
  maxClipSkip?: number
  guidanceAdvanced?: GuidanceAdvancedParams
  guidanceSupport?: GuidanceAdvancedCapabilities | null
  supir?: SupirModeFormState
  supirSamplerChoices?: readonly SupirSamplerInfo[]
  supirSelectedSamplerInfo?: SupirSamplerInfo | null
  supirBlockingReason?: string
}>(), {
  disabled: false,
  upscalers: () => [],
  upscalersLoading: false,
  upscalersError: '',
  minSteps: 1,
  maxSteps: 150,
  minCfg: 0,
  maxCfg: 30,
  cfgStep: 0.5,
  showDenoise: true,
  cfgLabel: 'CFG',
  minWidth: 64,
  maxWidth: 8192,
  minHeight: 64,
  maxHeight: 8192,
  widthStep: 64,
  widthInputStep: 8,
  heightStep: 64,
  heightInputStep: 8,
  showInitImageDims: false,
  showClipSkip: false,
  clipSkip: 0,
  minClipSkip: 0,
  maxClipSkip: 12,
  guidanceAdvanced: () => ({
    enabled: false,
    apgEnabled: false,
    apgStartStep: 0,
    apgEta: 0,
    apgMomentum: 0,
    apgNormThreshold: 15,
    apgRescale: 0,
    guidanceRescale: 0,
    cfgTruncEnabled: false,
    cfgTruncRatio: 0.8,
    renormCfg: 0,
  }),
  guidanceSupport: null,
  supir: () => createDefaultSupirModeFormState(),
  supirSamplerChoices: () => [],
  supirSelectedSamplerInfo: null,
  supirBlockingReason: '',
  resizeModeOptions: () => [...IMG2IMG_RESIZE_MODE_OPTIONS],
  showResizeMode: true,
  dimensionSnapMode: 'nearest',
})

const defaultSupirMode = createDefaultSupirModeFormState()

const emit = defineEmits<{
  (e: 'update:sampler', value: string): void
  (e: 'update:scheduler', value: string): void
  (e: 'update:steps', value: number): void
  (e: 'update:cfgScale', value: number): void
  (e: 'update:denoiseStrength', value: number): void
  (e: 'update:seed', value: number): void
  (e: 'update:width', value: number): void
  (e: 'update:height', value: number): void
  (e: 'update:upscaler', value: string): void
  (e: 'update:resizeMode', value: Img2ImgResizeMode): void
  (e: 'update:clipSkip', value: number): void
  (e: 'update:guidanceAdvanced', patch: Partial<GuidanceAdvancedParams>): void
  (e: 'patch:supir', patch: Partial<SupirModeFormState>): void
  (e: 'random-seed'): void
  (e: 'reuse-seed'): void
  (e: 'sync-init-image-dims'): void
}>()

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
const minClipSkip = computed(() => Number.isFinite(props.minClipSkip) ? Math.trunc(Number(props.minClipSkip)) : 0)
const maxClipSkip = computed(() => Number.isFinite(props.maxClipSkip) ? Math.trunc(Number(props.maxClipSkip)) : 12)
const showClipSkip = computed(() => props.showClipSkip === true)
const recommendedSamplers = computed(() => (Array.isArray(props.recommendedSamplers) ? props.recommendedSamplers : null))
const recommendedSchedulers = computed(() => (Array.isArray(props.recommendedSchedulers) ? props.recommendedSchedulers : null))
const guidanceAdvanced = computed(() => props.guidanceAdvanced ?? DEFAULT_GUIDANCE_ADVANCED)
const guidanceSupport = computed(() => props.guidanceSupport ?? null)
const supir = computed(() => props.supir ?? defaultSupirMode)
const supirEnabled = computed(() => Boolean(supir.value.enabled))
const supirSamplerChoices = computed(() => (Array.isArray(props.supirSamplerChoices) ? props.supirSamplerChoices : []))
const supirSelectedSamplerInfo = computed(() => props.supirSelectedSamplerInfo ?? null)
const supirBlockingReason = computed(() => String(props.supirBlockingReason || '').trim())
const supirLockedScheduler = computed(() => String(supirSelectedSamplerInfo.value?.native_scheduler || '').trim())
const supirHasStaleSamplerSelection = computed(() => (
  Boolean(String(supir.value.sampler || '').trim())
  && supirSelectedSamplerInfo.value === null
))
const showGuidanceAdvancedToggle = computed(() => {
  if (supirEnabled.value) return false
  const support = guidanceSupport.value
  if (!support) return false
  return Object.values(support).some((flag) => flag === true)
})
const showGuidanceAdvancedRow = computed(() => showGuidanceAdvancedToggle.value && guidanceAdvanced.value.enabled)
const aspectRatioLocked = ref(false)
const aspectRatio = ref(1)

const resizeModeOptions = computed<readonly Img2ImgResizeModeOption[]>(() => {
  if (Array.isArray(props.resizeModeOptions) && props.resizeModeOptions.length > 0) {
    return props.resizeModeOptions
  }
  return IMG2IMG_RESIZE_MODE_OPTIONS
})
const resizeModeValue = computed<Img2ImgResizeMode>(() => normalizeImg2ImgResizeModeFromOptions(props.resizeMode, resizeModeOptions.value))
const showsUpscalerResizeMode = computed(() => resizeModeOptions.value.some((option) => option.value === 'upscaler'))
const isUpscalerResizeMode = computed(() => resizeModeValue.value === 'upscaler')

const upscalers = computed(() => Array.isArray(props.upscalers) ? props.upscalers : [])
const spandrelUpscalers = computed(() => upscalers.value.filter((entry) => entry.kind === 'spandrel'))
const latentUpscalers = computed(() => upscalers.value.filter((entry) => entry.kind === 'latent'))
const isUpscalerKnown = computed(() => upscalers.value.some((entry) => entry.id === props.upscaler))

function clampFloat(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}

function clampInt(value: number, min: number, max: number): number {
  const numberValue = Number.isFinite(value) ? Math.trunc(value) : min
  return Math.min(max, Math.max(min, numberValue))
}

function clampIntToStep(
  value: number,
  min: number,
  max: number,
  step: number,
  mode: 'nearest' | 'floor' = 'nearest',
): number {
  const clamped = clampInt(value, min, max)
  if (!Number.isFinite(step) || step <= 0) return clamped
  const snapped = (mode === 'floor' ? Math.floor(clamped / step) : Math.round(clamped / step)) * step
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
    const widthValue = clampIntToStep(
      nextValue,
      minWidth.value,
      maxWidth.value,
      widthInputStep.value,
      props.dimensionSnapMode === 'floor' ? 'floor' : 'nearest',
    )
    const heightValue = clampIntToStep(
      Math.round(widthValue / ratio),
      minHeight.value,
      maxHeight.value,
      heightInputStep.value,
      props.dimensionSnapMode === 'floor' ? 'floor' : 'nearest',
    )
    return { width: widthValue, height: heightValue }
  }
  const heightValue = clampIntToStep(
    nextValue,
    minHeight.value,
    maxHeight.value,
    heightInputStep.value,
    props.dimensionSnapMode === 'floor' ? 'floor' : 'nearest',
  )
  const widthValue = clampIntToStep(
    Math.round(heightValue * ratio),
    minWidth.value,
    maxWidth.value,
    widthInputStep.value,
    props.dimensionSnapMode === 'floor' ? 'floor' : 'nearest',
  )
  return { width: widthValue, height: heightValue }
}

function toggleAspectRatioLock(): void {
  aspectRatioLocked.value = !aspectRatioLocked.value
  if (aspectRatioLocked.value) syncAspectRatioFromValues(props.width, props.height)
}

function onWidthDimensionChange(value: number): void {
  if (!aspectRatioLocked.value) {
    emit(
      'update:width',
      clampIntToStep(
        value,
        minWidth.value,
        maxWidth.value,
        widthInputStep.value,
        props.dimensionSnapMode === 'floor' ? 'floor' : 'nearest',
      ),
    )
    return
  }
  const next = deriveLockedDimensions(value, 'width')
  emitDimensions(next.width, next.height)
}

function onHeightDimensionChange(value: number): void {
  if (!aspectRatioLocked.value) {
    emit(
      'update:height',
      clampIntToStep(
        value,
        minHeight.value,
        maxHeight.value,
        heightInputStep.value,
        props.dimensionSnapMode === 'floor' ? 'floor' : 'nearest',
      ),
    )
    return
  }
  const next = deriveLockedDimensions(value, 'height')
  emitDimensions(next.width, next.height)
}

function onResizeModeChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  emit('update:resizeMode', normalizeImg2ImgResizeModeFromOptions(value, resizeModeOptions.value))
}

function onUpscalerChange(event: Event): void {
  emit('update:upscaler', (event.target as HTMLSelectElement).value)
}

function hasGuidanceSupport(control: keyof GuidanceAdvancedCapabilities): boolean {
  return Boolean(guidanceSupport.value?.[control])
}

function patchGuidanceAdvanced(patch: Partial<GuidanceAdvancedParams>): void {
  emit('update:guidanceAdvanced', patch)
}

function toggleGuidanceAdvanced(): void {
  const nextEnabled = !guidanceAdvanced.value.enabled
  const patch: Partial<GuidanceAdvancedParams> = { enabled: nextEnabled }
  if (hasGuidanceSupport('apg_enabled')) patch.apgEnabled = nextEnabled
  if (hasGuidanceSupport('cfg_trunc_ratio')) patch.cfgTruncEnabled = nextEnabled
  patchGuidanceAdvanced(patch)
}

function swapWH(): void {
  const nextWidth = clampIntToStep(
    props.height,
    minWidth.value,
    maxWidth.value,
    widthInputStep.value,
    props.dimensionSnapMode === 'floor' ? 'floor' : 'nearest',
  )
  const nextHeight = clampIntToStep(
    props.width,
    minHeight.value,
    maxHeight.value,
    heightInputStep.value,
    props.dimensionSnapMode === 'floor' ? 'floor' : 'nearest',
  )
  syncAspectRatioFromValues(nextWidth, nextHeight)
  emitDimensions(nextWidth, nextHeight)
}
</script>

<!-- styles in styles/components/img2img-basic-parameters-card.css -->
