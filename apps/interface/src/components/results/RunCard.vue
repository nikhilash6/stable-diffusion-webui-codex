<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Results run card wrapper (Generate/Infinite CTA + batch controls popovers).
Renders a `ResultsCard` with a primary run CTA, an optional split-button action menu (`Generate`/`Infinite`), an optional center-adjacent slot (for run badges), and an optional batch settings panel (count/size) that can be reused across image/video views.
When `isRunning=true`, the center CTA switches to a destructive cancel-confirm flow (click once -> `Are you sure?`, click again within timeout -> emit `cancel`).

Symbols (top-level; keep in sync; no ghosts):
- `RunCard` (component): Run/results card with generate CTA and optional batch controls.
- `setBatchCount` (function): Emits a clamped batch-count update.
- `setBatchSize` (function): Emits a clamped batch-size update.
- `onPrimaryAction` (function): Handles the center CTA click (generate or two-click cancel confirm while running).
- `selectActionMode` (function): Persists the current split-button mode and closes the action menu.
- `armCancelConfirm` (function): Enables temporary cancel-confirm mode and schedules auto-reset.
- `clearCancelConfirm` (function): Resets cancel-confirm mode and clears pending timeout.
- `toggleActionMenu` (function): Toggles the run-action dropdown.
- `openActionMenu` (function): Opens the run-action dropdown and schedules positioning.
- `closeActionMenu` (function): Closes the run-action dropdown and clears handlers.
- `toggleBatchMenu` (function): Toggles the batch settings popover.
- `openBatchMenu` (function): Opens the batch settings popover and schedules positioning.
- `closeBatchMenu` (function): Closes the batch settings popover and clears handlers.
- `isEventInsideMenu` (function): Checks whether a DOM event target is inside a menu panel/toggle.
- `onDocumentPointerDown` (function): Outside-click handler that closes the popover.
- `onDocumentKeyDown` (function): Keydown handler (Escape closes the popover).
- `updateActionMenuPosition` (function): Computes and applies the run-action menu position style.
- `updateBatchMenuPosition` (function): Computes and applies the popover position style.
- `scheduleMenuPositionUpdate` (function): Debounced/nextTick positioning update helper.
- `clampInt` (function): Clamps and truncates numeric values to an integer range.
-->

<template>
  <ResultsCard
    :title="props.title"
    headerClass="three-cols run-sticky"
    headerCenterClass="run-header-center"
    headerRightClass="run-controls"
    :showGenerate="false"
  >
    <template #header-center>
      <div class="run-primary-split" :class="{ 'run-primary-split--split': showActionMenuButton }">
        <button
          :id="props.generateId || undefined"
          :class="[primaryButtonClass, 'run-primary-split__primary']"
          type="button"
          :disabled="primaryButtonDisabled"
          :title="primaryButtonTitle"
          @click="onPrimaryAction"
        >
          {{ primaryButtonLabel }}
        </button>
        <button
          v-if="showActionMenuButton"
          ref="actionMenuToggleEl"
          class="btn btn-md btn-primary run-primary-split__menu-toggle"
          type="button"
          :disabled="inputsDisabled"
          :aria-expanded="isActionMenuOpen ? 'true' : 'false'"
          aria-haspopup="dialog"
          title="Choose run action"
          @click="toggleActionMenu"
        >
          <svg class="run-primary-split__menu-icon" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
            <path
              d="M4 6.25L8 10.25L12 6.25"
              fill="none"
              stroke="currentColor"
              stroke-width="1.75"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
          </svg>
        </button>
      </div>

      <Teleport to="body">
        <div
          v-if="isActionMenuOpen"
          ref="actionMenuPanelEl"
          class="run-primary-split__menu panel"
          :style="actionMenuStyle"
          role="dialog"
          aria-label="Run action menu"
        >
          <button
            class="btn btn-sm run-primary-split__menu-button"
            type="button"
            :disabled="inputsDisabled"
            @click="selectActionMode('generate')"
          >
            {{ props.generateLabel }}
          </button>
          <button
            class="btn btn-sm run-primary-split__menu-button"
            type="button"
            :disabled="inputsDisabled"
            @click="selectActionMode('infinite')"
          >
            {{ props.infiniteLabel }}
          </button>
        </div>
      </Teleport>
    </template>

    <template #header-center-after>
      <slot name="header-center-after" />
    </template>

    <template #header-right>
      <template v-if="props.showBatchControls">
        <div class="run-control run-batch-menu">
          <button
            ref="batchMenuToggleEl"
            class="btn btn-sm btn-outline run-batch-menu__toggle"
            type="button"
            :disabled="inputsDisabled"
            :aria-expanded="isBatchMenuOpen ? 'true' : 'false'"
            aria-haspopup="dialog"
            title="Batch settings"
            @click="toggleBatchMenu"
          >
            Batch {{ props.batchCount }}×{{ props.batchSize }}
          </button>

          <Teleport to="body">
            <div
              v-if="isBatchMenuOpen"
              ref="batchMenuPanelEl"
              class="run-batch-menu__panel panel"
              :style="batchMenuStyle"
              role="dialog"
              aria-label="Batch settings"
            >
              <div class="run-batch-menu__rows">
                <div class="run-batch-menu__row">
                  <span class="caption">Batch count</span>
                  <NumberStepperInput
                    :modelValue="props.batchCount"
                    :min="minBatchCount"
                    :max="maxBatchCount"
                    :step="1"
                    :nudgeStep="1"
                    size="sm"
                    inputClass="cdx-input-w-xs"
                    :disabled="inputsDisabled"
                    updateOnInput
                    @update:modelValue="setBatchCount"
                  />
                </div>
                <div class="run-batch-menu__row">
                  <span class="caption">Batch size</span>
                  <NumberStepperInput
                    :modelValue="props.batchSize"
                    :min="minBatchSize"
                    :max="maxBatchSize"
                    :step="1"
                    :nudgeStep="1"
                    size="sm"
                    inputClass="cdx-input-w-xs"
                    :disabled="inputsDisabled"
                    updateOnInput
                    @update:modelValue="setBatchSize"
                  />
                </div>
              </div>

              <div class="run-batch-menu__actions">
                <button class="btn btn-sm btn-primary" type="button" :disabled="inputsDisabled" @click="closeBatchMenu">OK</button>
              </div>
            </div>
          </Teleport>
        </div>
      </template>
      <slot name="header-right" />
    </template>

    <slot />
  </ResultsCard>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import NumberStepperInput from '../ui/NumberStepperInput.vue'
import ResultsCard from './ResultsCard.vue'

const props = withDefaults(defineProps<{
  title?: string
  generateId?: string
  generateButtonClass?: string
  cancelButtonClass?: string
  generateLabel?: string
  infiniteLabel?: string
  runningLabel?: string
  cancelLabel?: string
  cancelConfirmLabel?: string
  cancelConfirmWindowMs?: number
  generateDisabled?: boolean
  cancelDisabled?: boolean
  generateTitle?: string
  infiniteTitle?: string
  cancelTitle?: string
  isRunning?: boolean
  actionMode?: 'generate' | 'infinite'
  showActionMenu?: boolean
  showBatchControls?: boolean
  batchCount?: number
  batchSize?: number
  disabled?: boolean
  minBatchCount?: number
  maxBatchCount?: number
  minBatchSize?: number
  maxBatchSize?: number
}>(), {
  title: 'Run',
  generateId: '',
  generateButtonClass: 'btn btn-md btn-primary',
  cancelButtonClass: 'btn btn-md btn-destructive',
  generateLabel: 'Generate',
  infiniteLabel: 'Infinite',
  runningLabel: 'Running…',
  cancelLabel: 'Cancel',
  cancelConfirmLabel: 'Are you sure?',
  cancelConfirmWindowMs: 4000,
  generateDisabled: false,
  cancelDisabled: false,
  generateTitle: '',
  infiniteTitle: 'Keep generating until you stop the run.',
  cancelTitle: 'Click to cancel the current run.',
  isRunning: false,
  actionMode: 'generate',
  showActionMenu: false,
  showBatchControls: true,
  batchCount: 1,
  batchSize: 1,
  disabled: false,
  minBatchCount: 1,
  maxBatchCount: 999,
  minBatchSize: 1,
  maxBatchSize: 999,
})

const emit = defineEmits<{
  (e: 'generate', mode: 'generate' | 'infinite'): void
  (e: 'cancel'): void
  (e: 'update:actionMode', value: 'generate' | 'infinite'): void
  (e: 'update:batchCount', value: number): void
  (e: 'update:batchSize', value: number): void
}>()

const inputsDisabled = computed(() => Boolean(props.disabled || props.generateDisabled))

const minBatchCount = computed(() => Number.isFinite(props.minBatchCount) ? Math.trunc(Number(props.minBatchCount)) : 1)
const maxBatchCount = computed(() => Number.isFinite(props.maxBatchCount) ? Math.trunc(Number(props.maxBatchCount)) : 999)
const minBatchSize = computed(() => Number.isFinite(props.minBatchSize) ? Math.trunc(Number(props.minBatchSize)) : 1)
const maxBatchSize = computed(() => Number.isFinite(props.maxBatchSize) ? Math.trunc(Number(props.maxBatchSize)) : 999)

const actionMenuToggleEl = ref<HTMLElement | null>(null)
const actionMenuPanelEl = ref<HTMLElement | null>(null)
const isActionMenuOpen = ref(false)
const actionMenuStyle = ref<Record<string, string> | undefined>(undefined)
const batchMenuToggleEl = ref<HTMLElement | null>(null)
const batchMenuPanelEl = ref<HTMLElement | null>(null)
const isBatchMenuOpen = ref(false)
const batchMenuStyle = ref<Record<string, string> | undefined>(undefined)
let menuRAF: number | null = null

const isCancelConfirmArmed = ref(false)
let cancelConfirmTimerId: number | null = null

watch(() => props.showActionMenu, (show) => {
  if (!show) closeActionMenu()
})

watch(() => props.showBatchControls, (show) => {
  if (!show) closeBatchMenu()
})

watch(inputsDisabled, (disabled) => {
  if (disabled) closeActionMenu()
  if (disabled) closeBatchMenu()
})

watch(() => props.isRunning, (running) => {
  if (running) closeActionMenu()
  if (!running) clearCancelConfirm()
})

const primaryButtonClass = computed(() => {
  if (props.isRunning) return props.cancelButtonClass
  return props.generateButtonClass
})

const runningButtonLabel = computed(() => {
  if (!props.isRunning) return props.runningLabel
  return isCancelConfirmArmed.value ? props.cancelConfirmLabel : props.cancelLabel
})

const primaryButtonLabel = computed(() => {
  if (props.isRunning) return runningButtonLabel.value
  return props.actionMode === 'infinite' ? props.infiniteLabel : props.generateLabel
})

const primaryButtonDisabled = computed(() => {
  if (!props.isRunning) return Boolean(props.generateDisabled)
  return Boolean(props.cancelDisabled)
})

const primaryButtonTitle = computed(() => {
  if (!props.isRunning) return props.actionMode === 'infinite' ? props.infiniteTitle : props.generateTitle
  if (isCancelConfirmArmed.value) return 'Click again to confirm cancellation.'
  return props.cancelTitle
})

const showActionMenuButton = computed(() => Boolean(props.showActionMenu && !props.isRunning))

const cancelConfirmWindowMs = computed(() => {
  const value = Number(props.cancelConfirmWindowMs)
  if (!Number.isFinite(value)) return 4000
  return Math.max(500, Math.trunc(value))
})

function onPrimaryAction(): void {
  if (!props.isRunning) {
    emit('generate', props.actionMode)
    return
  }
  if (primaryButtonDisabled.value) return
  if (!isCancelConfirmArmed.value) {
    armCancelConfirm()
    return
  }
  clearCancelConfirm()
  emit('cancel')
}

function toggleActionMenu(): void {
  if (isActionMenuOpen.value) {
    closeActionMenu()
    return
  }
  openActionMenu()
}

function openActionMenu(): void {
  if (inputsDisabled.value || !showActionMenuButton.value) return
  actionMenuStyle.value = hiddenMenuStyle()
  isActionMenuOpen.value = true
  void nextTick(() => {
    window.requestAnimationFrame(() => {
      updateActionMenuPosition()
    })
  })
}

function closeActionMenu(): void {
  isActionMenuOpen.value = false
}

function selectActionMode(value: 'generate' | 'infinite'): void {
  emit('update:actionMode', value)
  closeActionMenu()
}

function armCancelConfirm(): void {
  clearCancelConfirm()
  isCancelConfirmArmed.value = true
  cancelConfirmTimerId = window.setTimeout(() => {
    isCancelConfirmArmed.value = false
    cancelConfirmTimerId = null
  }, cancelConfirmWindowMs.value)
}

function clearCancelConfirm(): void {
  isCancelConfirmArmed.value = false
  if (cancelConfirmTimerId !== null) window.clearTimeout(cancelConfirmTimerId)
  cancelConfirmTimerId = null
}

function setBatchCount(value: number): void {
  emit('update:batchCount', clampInt(value, minBatchCount.value, maxBatchCount.value))
}

function setBatchSize(value: number): void {
  emit('update:batchSize', clampInt(value, minBatchSize.value, maxBatchSize.value))
}

function toggleBatchMenu(): void {
  if (isBatchMenuOpen.value) {
    closeBatchMenu()
    return
  }

  openBatchMenu()
}

function openBatchMenu(): void {
  if (inputsDisabled.value) return

  batchMenuStyle.value = hiddenMenuStyle()
  isBatchMenuOpen.value = true

  void nextTick(() => {
    window.requestAnimationFrame(() => {
      updateBatchMenuPosition()
      const firstInput = batchMenuPanelEl.value?.querySelector<HTMLInputElement>('input')
      if (firstInput) {
        try {
          firstInput.focus({ preventScroll: true })
        } catch {
          firstInput.focus()
        }
      }
    })
  })
}

function closeBatchMenu(): void {
  isBatchMenuOpen.value = false
}

function isEventInsideMenu(event: Event, toggle: HTMLElement | null, panel: HTMLElement | null): boolean {
  const target = event.target
  if (!(target instanceof Node)) return false
  return Boolean((toggle && toggle.contains(target)) || (panel && panel.contains(target)))
}

function onDocumentPointerDown(event: PointerEvent): void {
  if (isActionMenuOpen.value && !isEventInsideMenu(event, actionMenuToggleEl.value, actionMenuPanelEl.value)) {
    closeActionMenu()
  }
  if (!isBatchMenuOpen.value) return
  if (isEventInsideMenu(event, batchMenuToggleEl.value, batchMenuPanelEl.value)) return
  closeBatchMenu()
}

function onDocumentKeyDown(event: KeyboardEvent): void {
  if (event.key !== 'Escape') return
  if (isActionMenuOpen.value) {
    event.preventDefault()
    closeActionMenu()
  }
  if (!isBatchMenuOpen.value) return
  event.preventDefault()
  closeBatchMenu()
}

function updateActionMenuPosition(): void {
  const toggle = actionMenuToggleEl.value
  const panel = actionMenuPanelEl.value
  if (!toggle || !panel) return
  const rect = toggle.getBoundingClientRect()
  const panelWidth = Math.max(panel.offsetWidth, panel.getBoundingClientRect().width)
  const viewportPadding = 8
  const gap = 6
  const top = rect.bottom + gap
  const left = Math.min(
    Math.max(viewportPadding, rect.left),
    window.innerWidth - panelWidth - viewportPadding,
  )
  const maxHeight = Math.max(120, window.innerHeight - top - viewportPadding)

  actionMenuStyle.value = {
    position: 'fixed',
    top: `${top}px`,
    left: `${left}px`,
    maxHeight: `${maxHeight}px`,
    visibility: 'visible',
  }
}

function updateBatchMenuPosition(): void {
  const toggle = batchMenuToggleEl.value
  if (!toggle) return

  const rect = toggle.getBoundingClientRect()
  const viewportPadding = 8
  const gap = 6
  const top = rect.bottom + gap
  const right = Math.max(viewportPadding, window.innerWidth - rect.right)
  const maxHeight = Math.max(160, window.innerHeight - top - viewportPadding)

  batchMenuStyle.value = {
    position: 'fixed',
    top: `${top}px`,
    right: `${right}px`,
    maxHeight: `${maxHeight}px`,
    visibility: 'visible',
  }
}

function hiddenMenuStyle(): Record<string, string> {
  return {
    position: 'fixed',
    top: '0px',
    left: '-9999px',
    maxHeight: '0px',
    visibility: 'hidden',
  }
}

function scheduleMenuPositionUpdate(): void {
  if (!isActionMenuOpen.value && !isBatchMenuOpen.value) return
  if (menuRAF !== null) return

  menuRAF = window.requestAnimationFrame(() => {
    menuRAF = null
    if (isActionMenuOpen.value) updateActionMenuPosition()
    if (isBatchMenuOpen.value) updateBatchMenuPosition()
  })
}

onMounted(() => {
  document.addEventListener('pointerdown', onDocumentPointerDown)
  document.addEventListener('keydown', onDocumentKeyDown)
  window.addEventListener('resize', scheduleMenuPositionUpdate)
  window.addEventListener('scroll', scheduleMenuPositionUpdate, true)
})

onBeforeUnmount(() => {
  document.removeEventListener('pointerdown', onDocumentPointerDown)
  document.removeEventListener('keydown', onDocumentKeyDown)
  window.removeEventListener('resize', scheduleMenuPositionUpdate)
  window.removeEventListener('scroll', scheduleMenuPositionUpdate, true)
  if (menuRAF !== null) window.cancelAnimationFrame(menuRAF)
  clearCancelConfirm()
})

function clampInt(value: number, min: number, max: number): number {
  const n = Number.isFinite(value) ? Math.trunc(value) : min
  return Math.min(max, Math.max(min, n))
}
</script>
