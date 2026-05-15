<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Reusable quicksettings add-path modal (scan + add-to-library).
Provides add-path workflows for checkpoint/VAE/text-encoder path keys by scanning a user-supplied path (no hash on scan),
then adding selected/all files with SHA computed only at add-time, explicit per-row FSM state transitions, and honest byte-progress
semantics that fail loud on invalid `size_bytes` metadata.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsAddPathModal` (component): Modal for scanning and adding model files into a target paths.json key.
- `AddPathRowFsmState` (type): Explicit row finite-state machine (`queued|adding|added|already_in_library|error`).
- `RowStatus` (interface): Per-row runtime state record (FSM + SHA/error/size metadata).
- `InvalidSizeBytesError` (class): Fail-loud error for invalid or missing `size_bytes` metadata.
- `sanitizePathInput` (function): Sanitizes path input (trim, quote removal, slash normalization, repeated separator collapse).
- `scanCandidates` (function): Calls backend scan endpoint and populates candidate rows (no SHA).
- `addOne` (function): Adds one candidate file to library key and records per-row SHA/result state.
- `addAllSequential` (function): Adds all scanned candidates sequentially and reports deterministic byte progress when metadata is complete.
- `normalizeSizeBytes` (function): Validates backend `size_bytes` metadata (`number|null`) and throws on invalid/missing contract values.
- `setRowFsm` (function): Enforces legal row FSM transitions and throws on invalid transitions.
- `formatBytes` (function): Formats byte counts for UI progress/status labels.
- `planAddAllRun` (function): Builds one sequential add-all plan of pending rows with byte-total planning.
- `rowSizeLabel` (function): Formats per-row metadata (`Size …` and optional short SHA after add).
-->

<template>
  <Modal v-model="open" :title="title" panel-class="qs-add-path-modal-panel" :show-footer="false">
    <div class="qs-add-path-modal">
      <label class="label-muted" for="qs-add-path-input">{{ label }}</label>
      <div class="qs-add-path-input-row">
        <input
          id="qs-add-path-input"
          ref="pathInputEl"
          class="ui-input"
          type="text"
          v-model="pathInput"
          :placeholder="placeholderExample"
          @keydown.enter.prevent="scanCandidates"
        />
        <button class="btn btn-sm btn-secondary" type="button" :disabled="!canScan" @click="scanCandidates">
          <span v-if="scanLoading" class="qs-add-path-spinner" aria-hidden="true"></span>
          {{ scanLoading ? 'Scanning…' : 'Scan' }}
        </button>
      </div>

      <div class="qs-add-path-actions">
        <button class="btn btn-sm btn-secondary" type="button" :disabled="!canAddAll" @click="addAllSequential">
          <span v-if="addAllRunning" class="qs-add-path-spinner" aria-hidden="true"></span>
          <span>{{ addAllButtonLabel }}</span>
        </button>
      </div>
      <p v-if="addAllRunning" class="caption qs-add-path-progress-caption">{{ addAllProgressCaption }}</p>
      <progress
        v-if="addAllRunning && addAllProgressPercent !== null"
        class="qs-add-path-progress-bar"
        :value="addAllProgressPercent"
        max="100"
      ></progress>

      <p v-if="scanError" class="panel-error">Error: {{ scanError }}</p>
      <p v-else-if="scanned && scanResults.length === 0 && !scanLoading" class="caption">No supported files found.</p>

      <div v-if="scanResults.length > 0" class="panel-section modal-list-section qs-add-path-list-section">
        <div class="qs-add-path-list" role="list">
          <div
            v-for="item in scanResults"
            :key="item.path"
            :class="[
              'qs-add-path-row',
              isRowErrored(item) ? 'is-error' : '',
              isRowAdding(item) ? 'is-adding' : '',
            ]"
            role="listitem"
          >
            <div class="qs-add-path-row__name" :title="item.path">
              <div class="qs-add-path-row__title">{{ displayName(item) }}</div>
              <div class="qs-add-path-row__size">{{ rowSizeLabel(item) }}</div>
              <div v-if="rowErrorText(item)" class="qs-add-path-row__status qs-add-path-row__status--error">{{ rowErrorText(item) }}</div>
            </div>
            <div class="qs-add-path-row__actions">
              <button
                class="btn btn-sm btn-outline"
                type="button"
                :disabled="isRowActionDisabled(item)"
                @click="addOne(item)"
              >
                <span>{{ rowActionLabel(item) }}</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </Modal>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'

import { addModelPathItem, scanModelPath } from '../../api/client'
import type { ModelPathLibraryKind, ModelPathScanItem } from '../../api/types'
import Modal from '../ui/Modal.vue'

type AddPathRowFsmState = 'queued' | 'adding' | 'added' | 'already_in_library' | 'error'

interface RowStatus {
  fsm: AddPathRowFsmState
  sha256: string
  error: string
  sizeBytes: number | null
}

interface AddAllRunEntry {
  item: ModelPathScanItem
  plannedSizeBytes: number | null
}

interface AddAllRunPlan {
  entries: AddAllRunEntry[]
  totalBytes: number | null
}

class InvalidSizeBytesError extends Error {
  constructor(filePath: string, raw: unknown) {
    super(`invalid size_bytes for ${filePath}: ${String(raw)}`)
    this.name = 'InvalidSizeBytesError'
  }
}

const ROW_FSM_TRANSITIONS: Record<AddPathRowFsmState, ReadonlyArray<AddPathRowFsmState>> = {
  queued: ['adding'],
  adding: ['added', 'already_in_library', 'error'],
  added: [],
  already_in_library: [],
  error: ['adding'],
}

const props = withDefaults(defineProps<{
  modelValue: boolean
  title: string
  label: string
  targetKey: string
  targetKind: ModelPathLibraryKind
  placeholder?: string
}>(), {
  placeholder: '',
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'added', payload: { addedCount: number }): void
  (e: 'error', message: string): void
}>()

const open = computed({
  get: () => props.modelValue,
  set: (value: boolean) => emit('update:modelValue', value),
})

const pathInput = ref('')
const pathInputEl = ref<HTMLInputElement | null>(null)
const scanLoading = ref(false)
const scanError = ref('')
const scanned = ref(false)
const scanResults = ref<ModelPathScanItem[]>([])
const rowStatuses = ref<Record<string, RowStatus>>({})
const addAllRunning = ref(false)
const addAllActivePath = ref('')
const addAllTotalCount = ref(0)
const addAllCompletedCount = ref(0)
const addAllPlannedTotalBytes = ref<number | null>(null)
const addAllProcessedBytes = ref(0)

const placeholderExample = computed(() => {
  const explicit = String(props.placeholder || '').trim()
  if (explicit) return explicit

  const windows = isWindowsClient()
  const suffix = props.targetKind === 'checkpoint'
    ? 'checkpoints'
    : (props.targetKind === 'vae' ? 'vae' : 'text_encoders')
  if (windows) return `C:\\models\\${suffix}`
  return `/home/user/models/${suffix}`
})

const sanitizedInput = computed(() => sanitizePathInput(pathInput.value))
const hasRowAddInFlight = computed(() => scanResults.value.some((item) => isRowAdding(item)))
const canScan = computed(() => !scanLoading.value && !addAllRunning.value && Boolean(sanitizedInput.value))
const pendingAddCount = computed(() => scanResults.value.filter((item) => {
  const state = rowState(item)
  return isRowPendingForAdd(state)
}).length)
const canAddAll = computed(() => !scanLoading.value && !addAllRunning.value && !hasRowAddInFlight.value && pendingAddCount.value > 0)
const addAllProgressPercent = computed<number | null>(() => {
  const totalBytes = addAllPlannedTotalBytes.value
  if (totalBytes === null) return null
  if (totalBytes === 0) {
    if (addAllTotalCount.value <= 0) return 0
    return Math.min(100, Math.max(0, (addAllCompletedCount.value / addAllTotalCount.value) * 100))
  }
  return Math.min(100, Math.max(0, (addAllProcessedBytes.value / totalBytes) * 100))
})
const addAllProgressCaption = computed(() => {
  if (!addAllRunning.value) return ''
  const completedLabel = `${addAllCompletedCount.value}/${addAllTotalCount.value} files`
  const totalBytes = addAllPlannedTotalBytes.value
  if (totalBytes === null) {
    return `Processing ${completedLabel} · byte progress unavailable (size metadata missing)`
  }
  const processed = Math.min(addAllProcessedBytes.value, totalBytes)
  const percent = addAllProgressPercent.value
  const percentText = percent === null ? '' : ` (${percent.toFixed(1)}%)`
  return `Processed ${formatBytes(processed)} / ${formatBytes(totalBytes)}${percentText} · ${completedLabel}`
})
const addAllButtonLabel = computed(() => addAllRunning.value ? 'Adding…' : 'Add whole folder')

watch(open, (isOpen) => {
  if (!isOpen) {
    resetState()
    return
  }
  void nextTick(() => {
    pathInputEl.value?.focus()
    pathInputEl.value?.select()
  })
})

function resetState(): void {
  pathInput.value = ''
  scanError.value = ''
  scanned.value = false
  scanLoading.value = false
  scanResults.value = []
  rowStatuses.value = {}
  addAllRunning.value = false
  addAllActivePath.value = ''
  addAllTotalCount.value = 0
  addAllCompletedCount.value = 0
  addAllPlannedTotalBytes.value = null
  addAllProcessedBytes.value = 0
}

function isWindowsClient(): boolean {
  if (typeof navigator === 'undefined') return false
  const uaData = (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData
  const platform = String(uaData?.platform || navigator.platform || navigator.userAgent || '').toLowerCase()
  return platform.includes('win')
}

function sanitizePathInput(raw: string): string {
  let value = String(raw ?? '').trim()
  if (!value) return ''

  while ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
    value = value.slice(1, -1).trim()
    if (!value) return ''
  }

  value = value.replace(/\\/g, '/')

  const hasUncPrefix = value.startsWith('//')
  const driveMatch = value.match(/^[A-Za-z]:/)
  if (driveMatch) {
    const drive = driveMatch[0]
    let rest = value.slice(drive.length)
    rest = rest.replace(/\/+/g, '/')
    if (rest && !rest.startsWith('/')) rest = `/${rest}`
    return `${drive}${rest}`.trim()
  }

  value = value.replace(/\/+/g, '/')
  if (hasUncPrefix) value = `//${value.replace(/^\/+/, '')}`
  if (value.length > 1 && /\/$/.test(value) && !/^[A-Za-z]:\/$/.test(value)) {
    value = value.replace(/\/+$/, '')
  }
  return value.trim()
}

function displayName(item: ModelPathScanItem): string {
  const base = String(item.name || '').trim()
  const ext = String(item.ext || '').trim().toLowerCase()
  if (base && ext && base.toLowerCase().endsWith(ext)) {
    return base.slice(0, base.length - ext.length)
  }
  return base || item.path
}

function shortSha(sha: string): string {
  const normalized = String(sha || '').trim().toLowerCase()
  if (normalized.length <= 10) return normalized
  return normalized.slice(0, 10)
}

function normalizeSizeBytes(raw: unknown, filePath: string): number | null {
  if (raw === null) return null
  if (raw === undefined) {
    throw new InvalidSizeBytesError(filePath, raw)
  }
  if (typeof raw !== 'number' || !Number.isFinite(raw) || raw < 0 || !Number.isInteger(raw)) {
    throw new InvalidSizeBytesError(filePath, raw)
  }
  return raw
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let amount = value
  let unitIndex = 0
  while (amount >= 1024 && unitIndex < units.length - 1) {
    amount /= 1024
    unitIndex += 1
  }
  const digits = unitIndex <= 1 ? 0 : 1
  return `${amount.toFixed(digits)} ${units[unitIndex]}`
}

function rowSizeLabel(item: ModelPathScanItem): string {
  const state = rowState(item)
  const sizeBytes = state.sizeBytes
  const sizeLabel = sizeBytes === null ? 'Size unavailable' : `Size ${formatBytes(sizeBytes)}`
  if (state.sha256) {
    return `${sizeLabel} - SHA ${shortSha(state.sha256)}`
  }
  return sizeLabel
}

function buildRowStatus(item: ModelPathScanItem): RowStatus {
  const alreadyInLibrary = item.already_in_library === true
  return {
    fsm: alreadyInLibrary ? 'already_in_library' : 'queued',
    sha256: '',
    error: '',
    sizeBytes: normalizeSizeBytes(item.size_bytes, item.path),
  }
}

function setRowFsm(state: RowStatus, next: AddPathRowFsmState, filePath: string): void {
  const allowed = ROW_FSM_TRANSITIONS[state.fsm]
  if (!allowed.includes(next)) {
    throw new Error(`invalid row state transition for ${filePath}: ${state.fsm} -> ${next}`)
  }
  state.fsm = next
}

function isRowPendingForAdd(state: RowStatus): boolean {
  return state.fsm === 'queued' || state.fsm === 'error'
}

function isRowAdding(item: ModelPathScanItem): boolean {
  return rowState(item).fsm === 'adding'
}

function isRowErrored(item: ModelPathScanItem): boolean {
  const state = rowState(item)
  return state.fsm === 'error' && Boolean(state.error)
}

function rowErrorText(item: ModelPathScanItem): string {
  const state = rowState(item)
  return state.fsm === 'error' ? state.error : ''
}

function rowState(item: ModelPathScanItem): RowStatus {
  const existing = rowStatuses.value[item.path]
  if (existing) return existing
  const created = buildRowStatus(item)
  rowStatuses.value[item.path] = created
  return created
}

function validatePlannedSizeBytes(value: unknown, filePath: string): number | null {
  if (value === null) return null
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
    throw new InvalidSizeBytesError(filePath, value)
  }
  return value
}

function planAddAllRun(): AddAllRunPlan {
  const entries: AddAllRunEntry[] = []
  let totalBytes = 0
  let hasUnknownSize = false
  for (const item of scanResults.value) {
    const state = rowState(item)
    if (!isRowPendingForAdd(state)) continue
    const plannedSizeBytes = validatePlannedSizeBytes(state.sizeBytes, item.path)
    if (plannedSizeBytes === null) {
      hasUnknownSize = true
    } else {
      totalBytes += plannedSizeBytes
    }
    entries.push({ item, plannedSizeBytes })
  }
  return {
    entries,
    totalBytes: hasUnknownSize ? null : totalBytes,
  }
}

function isRowActionDisabled(item: ModelPathScanItem): boolean {
  const state = rowState(item)
  if (state.fsm === 'adding') return true
  if (addAllRunning.value) return true
  return state.fsm === 'added' || state.fsm === 'already_in_library'
}

function rowActionLabel(item: ModelPathScanItem): string {
  const state = rowState(item)
  if (state.fsm === 'adding') {
    if (addAllRunning.value && addAllActivePath.value === item.path && addAllProgressPercent.value !== null) {
      return `Adding ${addAllProgressPercent.value.toFixed(0)}%`
    }
    return 'Adding…'
  }
  if (state.fsm === 'added') return 'Added'
  if (state.fsm === 'already_in_library') return 'Already in library'
  return 'Add to library'
}

async function scanCandidates(): Promise<void> {
  if (!canScan.value) return
  const sanitized = sanitizedInput.value
  if (!sanitized) return

  pathInput.value = sanitized
  scanLoading.value = true
  scanError.value = ''
  scanned.value = true
  scanResults.value = []
  rowStatuses.value = {}
  addAllTotalCount.value = 0
  addAllCompletedCount.value = 0
  addAllPlannedTotalBytes.value = null
  addAllProcessedBytes.value = 0

  try {
    const response = await scanModelPath({
      path: sanitized,
      key: props.targetKey,
      kind: props.targetKind,
    })
    const normalizedItems = response.items.map((item) => ({
      ...item,
      size_bytes: normalizeSizeBytes(item.size_bytes, item.path),
      already_in_library: item.already_in_library === true,
    }))
    scanResults.value = [...normalizedItems].sort((left, right) => {
      const byName = left.name.localeCompare(right.name)
      if (byName !== 0) return byName
      return left.path.localeCompare(right.path)
    })
    const next: Record<string, RowStatus> = {}
    for (const item of scanResults.value) {
      next[item.path] = buildRowStatus(item)
    }
    rowStatuses.value = next
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    scanError.value = message
    emit('error', message)
  } finally {
    scanLoading.value = false
  }
}

async function addOne(item: ModelPathScanItem, options?: { silent?: boolean }): Promise<{ ok: boolean; added: boolean; fatal: boolean }> {
  const state = rowState(item)
  if (state.fsm === 'added') return { ok: true, added: true, fatal: false }
  if (state.fsm === 'already_in_library') return { ok: true, added: false, fatal: false }
  if (state.fsm !== 'queued' && state.fsm !== 'error') {
    throw new Error(`cannot add row in state ${state.fsm}: ${item.path}`)
  }
  setRowFsm(state, 'adding', item.path)
  state.error = ''

  try {
    const response = await addModelPathItem({
      key: props.targetKey,
      kind: props.targetKind,
      path: item.path,
    })
    const addedToLibrary = response.item.added === true
    state.sha256 = String(response.item.sha256 || '')
    const responseSize = normalizeSizeBytes(response.item.size_bytes, item.path)
    state.sizeBytes = responseSize
    setRowFsm(state, addedToLibrary ? 'added' : 'already_in_library', item.path)
    if (addedToLibrary && !options?.silent) {
      emit('added', { addedCount: 1 })
    }
    return { ok: true, added: addedToLibrary, fatal: false }
  } catch (error) {
    if (ROW_FSM_TRANSITIONS[state.fsm].includes('error')) {
      setRowFsm(state, 'error', item.path)
    }
    const fatal = error instanceof InvalidSizeBytesError
    const message = error instanceof Error ? error.message : String(error)
    state.error = message
    if (!options?.silent) {
      emit('error', message)
    }
    return { ok: false, added: false, fatal }
  }
}

async function addAllSequential(): Promise<void> {
  if (!canAddAll.value) return
  let runPlan: AddAllRunPlan
  try {
    runPlan = planAddAllRun()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    scanError.value = message
    emit('error', message)
    return
  }
  if (runPlan.entries.length === 0) return

  scanError.value = ''
  addAllRunning.value = true
  addAllActivePath.value = ''
  addAllTotalCount.value = runPlan.entries.length
  addAllCompletedCount.value = 0
  addAllPlannedTotalBytes.value = runPlan.totalBytes
  addAllProcessedBytes.value = 0

  let addedCount = 0
  let fatalMessage = ''
  try {
    for (const entry of runPlan.entries) {
      const { item, plannedSizeBytes } = entry
      const state = rowState(item)
      addAllActivePath.value = item.path
      const result = await addOne(item, { silent: true })
      if (result.added) addedCount += 1
      if (plannedSizeBytes !== null) {
        addAllProcessedBytes.value += plannedSizeBytes
        if (addAllPlannedTotalBytes.value !== null) {
          addAllProcessedBytes.value = Math.min(addAllProcessedBytes.value, addAllPlannedTotalBytes.value)
        }
      }
      addAllCompletedCount.value += 1
      if (!result.ok) {
        if (result.fatal) {
          fatalMessage = state.error || `invalid size_bytes for ${item.path}`
          break
        }
        if (state.error) {
          emit('error', state.error)
        }
      }
    }
  } finally {
    addAllRunning.value = false
    addAllActivePath.value = ''
  }

  if (fatalMessage) {
    if (addedCount > 0) {
      emit('added', { addedCount })
    }
    scanError.value = fatalMessage
    emit('error', fatalMessage)
    return
  }

  if (addedCount > 0) {
    emit('added', { addedCount })
  }
}

watch(
  () => props.targetKey,
  () => {
    if (!open.value) return
    resetState()
  },
)

watch(
  () => props.targetKind,
  () => {
    if (!open.value) return
    resetState()
  },
)

watch(pathInput, (value) => {
  if (!value) return
  const sanitized = sanitizePathInput(value)
  if (sanitized !== value) {
    pathInput.value = sanitized
  }
})
</script>
