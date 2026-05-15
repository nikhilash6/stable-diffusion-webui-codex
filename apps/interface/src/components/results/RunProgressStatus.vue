<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared run-status panel block.
Renders progress and non-progress run statuses (error/warning/info/success) in a single reusable panel for Run cards across views,
including severity-aware visuals and animated status icons.

Symbols (top-level; keep in sync; no ghosts):
- `RunProgressStatus` (component): Shared status panel for run progress and run notices/errors.
- `normalizeStatusVariant` (function): Normalizes incoming status variants into the supported union.
- `resolvedVariant` (const): Normalized status variant that drives panel styling and icon selection.
- `formatElapsedSeconds` (function): Formats elapsed seconds into `mm:ss` or `hh:mm:ss`.
- `displayElapsedSeconds` (const): Resolved elapsed seconds value (external prop or internal run timer).
- `elapsedLabel` (const): Formatted elapsed time string rendered on the progress meta row.
- `normalizedPercent` (const): Safe normalized sampling-step percent value for the lower progress bar.
- `normalizedTotalPercent` (const): Safe normalized total-pipeline percent value for the upper progress bar.
- `displayPercent` (const): Header percent display (prefers total percent when available).
- `totalPhaseLabel` (const): Normalized total-phase label (`encode`/`sampling`/`decode`) shown on the total bar.
-->

<template>
  <div
    :class="[
      'panel-status',
      'run-progress-status',
      {
        'run-progress-status--progress': resolvedVariant === 'progress',
        'run-progress-status--error': resolvedVariant === 'error',
        'run-progress-status--warning': resolvedVariant === 'warning',
        'run-progress-status--info': resolvedVariant === 'info',
        'run-progress-status--success': resolvedVariant === 'success',
      },
    ]"
    :data-variant="resolvedVariant"
    :role="ariaRole"
    :aria-live="ariaLive"
  >
    <div class="run-progress-status__header">
      <span class="run-progress-status__icon" aria-hidden="true">
        <svg
          v-if="resolvedVariant === 'progress'"
          class="run-progress-status__icon-svg run-progress-status__icon-svg--spinner"
          viewBox="0 0 24 24"
          fill="none"
        >
          <circle class="run-progress-status__spinner-track" cx="12" cy="12" r="9"></circle>
          <circle class="run-progress-status__spinner-head" cx="12" cy="12" r="9"></circle>
        </svg>
        <svg
          v-else-if="resolvedVariant === 'error'"
          class="run-progress-status__icon-svg"
          viewBox="0 0 24 24"
          fill="none"
        >
          <path d="M12 3L21 19H3L12 3Z"></path>
          <path d="M12 9V13"></path>
          <circle cx="12" cy="17" r="1"></circle>
        </svg>
        <svg
          v-else-if="resolvedVariant === 'warning'"
          class="run-progress-status__icon-svg"
          viewBox="0 0 24 24"
          fill="none"
        >
          <path d="M12 3L21 19H3L12 3Z"></path>
          <path d="M12 9V14"></path>
          <circle cx="12" cy="17.25" r="1"></circle>
        </svg>
        <svg
          v-else-if="resolvedVariant === 'success'"
          class="run-progress-status__icon-svg"
          viewBox="0 0 24 24"
          fill="none"
        >
          <circle cx="12" cy="12" r="9"></circle>
          <path d="M8.5 12.5L10.75 14.75L15.5 10"></path>
        </svg>
        <svg
          v-else
          class="run-progress-status__icon-svg"
          viewBox="0 0 24 24"
          fill="none"
        >
          <circle cx="12" cy="12" r="9"></circle>
          <path d="M12 10V16"></path>
          <circle cx="12" cy="7.5" r="1"></circle>
        </svg>
      </span>

      <div class="run-progress-status__headline">
        <p class="run-progress-status__title">{{ resolvedTitle }}</p>
        <p v-if="messageText" class="run-progress-status__message">{{ messageText }}</p>
        <p v-else-if="isProgressVariant" class="run-progress-status__message"><strong>Stage:</strong> {{ stageLabel }}</p>
      </div>

      <div v-if="isProgressVariant && displayPercent !== null" class="run-progress-status__percent">
        {{ displayPercent.toFixed(1) }}%
      </div>
    </div>

    <div v-if="isProgressVariant && showProgressBar" class="run-progress-status__bars">
      <div v-if="normalizedTotalPercent !== null" class="run-progress-status__bar-group">
        <div class="run-progress-status__bar-caption">
          <span>Total<span v-if="totalPhaseLabel"> · {{ totalPhaseLabel }}</span></span>
          <span>{{ normalizedTotalPercent.toFixed(1) }}%</span>
        </div>
        <progress
          class="run-progress-status__bar run-progress-status__bar--total"
          :value="normalizedTotalPercent"
          max="100"
        ></progress>
        <div
          v-if="totalPhaseStep !== null && totalPhaseTotalSteps !== null"
          class="run-progress-status__bar-meta"
        >
          {{ totalPhaseStep }} / {{ totalPhaseTotalSteps }}
        </div>
      </div>

      <div v-if="normalizedPercent !== null" class="run-progress-status__bar-group">
        <div class="run-progress-status__bar-caption">
          <span>Steps</span>
          <span>{{ normalizedPercent.toFixed(1) }}%</span>
        </div>
        <progress
          class="run-progress-status__bar run-progress-status__bar--steps"
          :value="normalizedPercent"
          max="100"
        ></progress>
      </div>
    </div>

    <div v-if="isProgressVariant" class="run-progress-status__meta">
      <div class="run-progress-status__meta-left">
        <span v-if="step !== null && totalSteps !== null" class="run-progress-status__meta-item">Step {{ step }} / {{ totalSteps }}</span>
        <span v-if="etaSeconds !== null" class="run-progress-status__meta-item">ETA ~ {{ etaSeconds.toFixed(0) }}s</span>
        <span v-if="queueLabel" class="run-progress-status__meta-item">{{ queueLabel }}</span>
        <slot name="extra" />
      </div>
      <div class="run-progress-status__meta-right">
        <span class="run-progress-status__meta-item run-progress-status__meta-item--elapsed">Elapsed {{ elapsedLabel }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'

type RunStatusVariant = 'progress' | 'error' | 'warning' | 'info' | 'success'

const props = withDefaults(defineProps<{
  variant?: RunStatusVariant | string
  title?: string
  message?: string
  stage?: string | null
  percent?: number | null
  step?: number | null
  totalSteps?: number | null
  etaSeconds?: number | null
  totalPercent?: number | null
  totalPhase?: string | null
  totalPhaseStep?: number | null
  totalPhaseTotalSteps?: number | null
  elapsedSeconds?: number | null
  queueLabel?: string
  showProgressBar?: boolean
}>(), {
  variant: 'progress',
  title: '',
  message: '',
  stage: 'running',
  percent: null,
  step: null,
  totalSteps: null,
  etaSeconds: null,
  totalPercent: null,
  totalPhase: null,
  totalPhaseStep: null,
  totalPhaseTotalSteps: null,
  elapsedSeconds: null,
  queueLabel: '',
  showProgressBar: true,
})

function normalizeStatusVariant(rawVariant: string): RunStatusVariant {
  const variant = String(rawVariant || '').trim().toLowerCase()
  if (variant === 'error' || variant === 'warning' || variant === 'info' || variant === 'success') return variant
  return 'progress'
}

function formatElapsedSeconds(totalSeconds: number): string {
  const safeSeconds = Number.isFinite(totalSeconds) ? Math.max(0, Math.trunc(totalSeconds)) : 0
  const hours = Math.trunc(safeSeconds / 3600)
  const minutes = Math.trunc((safeSeconds % 3600) / 60)
  const seconds = safeSeconds % 60
  const mm = String(minutes).padStart(2, '0')
  const ss = String(seconds).padStart(2, '0')
  if (hours > 0) return `${String(hours).padStart(2, '0')}:${mm}:${ss}`
  return `${mm}:${ss}`
}

const resolvedVariant = computed<RunStatusVariant>(() => normalizeStatusVariant(String(props.variant || 'progress')))
const isProgressVariant = computed(() => resolvedVariant.value === 'progress')
const messageText = computed(() => String(props.message || '').trim())
const stageLabel = computed(() => String(props.stage || 'running'))
const resolvedTitle = computed(() => {
  const customTitle = String(props.title || '').trim()
  if (customTitle) return customTitle
  if (resolvedVariant.value === 'error') return 'Run failed'
  if (resolvedVariant.value === 'warning') return 'Warning'
  if (resolvedVariant.value === 'info') return 'Info'
  if (resolvedVariant.value === 'success') return 'Success'
  return 'Running'
})
const normalizedPercent = computed(() => {
  if (props.percent === null || props.percent === undefined) return null
  if (!Number.isFinite(props.percent)) return null
  return Math.max(0, Math.min(100, props.percent))
})
const normalizedTotalPercent = computed(() => {
  if (props.totalPercent === null || props.totalPercent === undefined) return null
  if (!Number.isFinite(props.totalPercent)) return null
  return Math.max(0, Math.min(100, props.totalPercent))
})
const displayPercent = computed(() => {
  if (normalizedTotalPercent.value !== null) return normalizedTotalPercent.value
  return normalizedPercent.value
})
const step = computed(() => props.step)
const totalSteps = computed(() => props.totalSteps)
const etaSeconds = computed(() => props.etaSeconds)
const totalPhaseStep = computed(() => {
  if (props.totalPhaseStep === null || props.totalPhaseStep === undefined) return null
  if (!Number.isFinite(props.totalPhaseStep)) return null
  return Math.max(0, Math.trunc(props.totalPhaseStep))
})
const totalPhaseTotalSteps = computed(() => {
  if (props.totalPhaseTotalSteps === null || props.totalPhaseTotalSteps === undefined) return null
  if (!Number.isFinite(props.totalPhaseTotalSteps)) return null
  return Math.max(0, Math.trunc(props.totalPhaseTotalSteps))
})
const totalPhaseLabel = computed(() => {
  const normalized = String(props.totalPhase || '').trim().toLowerCase()
  if (normalized === 'encode') return 'encode'
  if (normalized === 'decode') return 'decode'
  if (normalized === 'sampling') return 'sampling'
  return normalized || null
})
const externalElapsedSeconds = computed(() => {
  if (props.elapsedSeconds === null || props.elapsedSeconds === undefined) return null
  if (!Number.isFinite(props.elapsedSeconds)) return null
  return Math.max(0, Math.trunc(props.elapsedSeconds))
})
const queueLabel = computed(() => String(props.queueLabel || '').trim())
const showProgressBar = computed(() => Boolean(props.showProgressBar))
const ariaRole = computed(() => (resolvedVariant.value === 'error' ? 'alert' : 'status'))
const ariaLive = computed(() => (resolvedVariant.value === 'error' ? 'assertive' : 'polite'))

const internalElapsedSeconds = ref(0)
let internalTimerId: number | null = null
let internalStartAtMs = 0

function startInternalTimer(): void {
  if (internalTimerId !== null) return
  internalTimerId = window.setInterval(() => {
    const elapsedMs = Date.now() - internalStartAtMs
    internalElapsedSeconds.value = Math.max(0, Math.trunc(elapsedMs / 1000))
  }, 1000)
}

function stopInternalTimer(): void {
  if (internalTimerId !== null) window.clearInterval(internalTimerId)
  internalTimerId = null
}

const shouldUseInternalTimer = computed(() => isProgressVariant.value && externalElapsedSeconds.value === null)

watch(shouldUseInternalTimer, (next) => {
  if (!next) {
    stopInternalTimer()
    return
  }
  internalElapsedSeconds.value = 0
  internalStartAtMs = Date.now()
  startInternalTimer()
}, {
  immediate: true,
})

onBeforeUnmount(() => {
  stopInternalTimer()
})

const displayElapsedSeconds = computed(() => {
  if (externalElapsedSeconds.value !== null) return externalElapsedSeconds.value
  return internalElapsedSeconds.value
})

const elapsedLabel = computed(() => formatElapsedSeconds(displayElapsedSeconds.value))
</script>
