<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Renderless LTX runtime helper for the canonical video tab view.
Mounts the existing LTX composable/watcher runtime under a view-local seam and exposes reactive slot props to `VideoModelTab.vue`,
while keeping the route-owned video view as the only live body/layout owner and wiring compact LTX results/history actions plus truthful
save-or-update workflow snapshot notices into the shared WAN-baseline Results surface.

Symbols (top-level; keep in sync; no ghosts):
- `VideoModelTabLtxRuntime` (component): Renderless LTX runtime helper for `VideoModelTab.vue`.
- `ExecutionProfileOption` (type): Workspace-facing execution-profile selector row.
- `normalizeExecutionProfileName` (function): Normalizes raw execution-profile names for selector/display checks.
- `executionProfileLabel` (function): Formats a user-facing label for a known or stale execution profile.
- `ensureExecutionProfileVisible` (function): Preserves stale persisted execution-profile values in the local selector option list.
- `normalizeDimensionInput` / `normalizeFrameInput` (functions): Bound LTX geometry/frame edits to the truthful numeric domain without silently snapping alignment.
- `setVideoZoomOpen` / `openResultVideoZoom` (functions): Parent-facing exported-video zoom visibility bridge setters.
- `sendToWorkflows` / `copyCurrentParams` / `onSelectHistoryItem` (functions): Parent-facing Results header/history actions exposed to the shared WAN-baseline Results owner, including truthful save-vs-update Workflow notices.
- `slotProps` (const): Reactive slot-prop bundle exposed to `VideoModelTab.vue`.
-->

<template>
  <slot v-bind="slotProps" />
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import type { GeneratedImage } from '../../api/types'
import {
  LTX_DIM_ALIGNMENT,
  LTX_DIM_MAX,
  LTX_DIM_MIN,
  LTX_FRAME_ALIGNMENT,
  LTX_FRAMES_MAX,
  LTX_FRAMES_MIN,
  LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT,
  resolveLtxDimAlignmentForExecutionProfile,
} from '../../api/payloads_ltx_video'
import { useLtxVideoGeneration, type LtxRunHistoryItem } from '../../composables/useLtxVideoGeneration'
import { useWorkflowSnapshotActions } from '../../composables/useWorkflowSnapshotActions'
import { useModelTabsStore, type LtxTabParams } from '../../stores/model_tabs'
import { readFileAsDataURL } from '../../utils/image_io'

const props = defineProps<{ tabId: string }>()

const store = useModelTabsStore()

const {
  status,
  progress,
  frames,
  info,
  videoUrl,
  errorMessage,
  isRunning,
  tab,
  params,
  mode,
  history,
  selectedTaskId,
  historyLoadingTaskId,
  ltxExecutionSurface,
  checkpointExecutionMetadata,
  blockedReason,
  generate,
  cancel,
  loadHistory,
  clearHistory,
  resumeNotice,
} = useLtxVideoGeneration(props.tabId)

const {
  notice: copyNotice,
  copyJson,
  formatJson,
  toast,
  workflowBusy,
  sendToWorkflows,
  copyCurrentParams,
} = useWorkflowSnapshotActions({
  getTab: () => tab.value ?? null,
  getWorkflowParamsSnapshot: () => (params.value as unknown as Record<string, unknown> | null) ?? null,
  onBeforeCopyCurrentParams: () => {
    resumeNotice.value = ''
  },
})

type ExecutionProfileOption = {
  value: string
  label: string
  supported: boolean
}

function normalizeExecutionProfileName(rawValue: string): string {
  return String(rawValue || '').trim().toLowerCase()
}

function executionProfileLabel(value: string): string {
  const normalized = normalizeExecutionProfileName(value)
  if (normalized === 'one_stage') return 'One-stage'
  if (normalized === 'two_stage') return 'Two-stage'
  if (normalized === 'distilled') return 'Distilled'
  return value || 'Unknown'
}

function ensureExecutionProfileVisible(options: ExecutionProfileOption[], currentValue: string): ExecutionProfileOption[] {
  const current = String(currentValue || '').trim()
  if (!current) return options
  if (options.some((entry) => entry.value === current)) return options
  const normalizedCurrent = normalizeExecutionProfileName(current)
  const canonicalMatch = options.find((entry) => normalizeExecutionProfileName(entry.value) === normalizedCurrent)
  return [{
    value: current,
    label: canonicalMatch
      ? `${executionProfileLabel(current)} (stored raw value; reselect the canonical profile)`
      : `${executionProfileLabel(current)} (unsupported; reselect a supported profile)`,
    supported: false,
  }, ...options]
}

const executionProfileOptions = computed<ExecutionProfileOption[]>(() => {
  const checkpointExecution = checkpointExecutionMetadata.value
  const checkpointAllowed = checkpointExecution?.allowedExecutionProfiles ?? []
  const surfaceAllowed = ltxExecutionSurface.value?.allowed_execution_profiles ?? []
  const allowed = checkpointExecution ? checkpointAllowed : surfaceAllowed
  const base = allowed.map((value: string) => ({
    value,
    label: executionProfileLabel(value),
    supported: true,
  }))
  return ensureExecutionProfileVisible(base, String(params.value?.executionProfile || ''))
})

const selectedExecutionProfile = computed(() => String(params.value?.executionProfile || '').trim())
const dimensionAlignment = computed(() => {
  if (selectedExecutionProfile.value !== 'two_stage') return LTX_DIM_ALIGNMENT
  return resolveLtxDimAlignmentForExecutionProfile(selectedExecutionProfile.value)
})
const dimensionWarning = computed(() => {
  const current = params.value
  const profile = selectedExecutionProfile.value
  if (!current || !profile) return ''
  const requiredAlignment = dimensionAlignment.value
  if (current.width % requiredAlignment === 0 && current.height % requiredAlignment === 0) {
    return ''
  }
  if (profile === 'two_stage') {
    return `two_stage requires final width and height divisible by ${LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT}. Current size ${current.width}×${current.height} is blocking.`
  }
  return `${executionProfileLabel(profile)} requires width and height divisible by ${requiredAlignment}. Current size ${current.width}×${current.height} is blocking.`
})
const frameWarning = computed(() => {
  const current = params.value
  if (!current) return ''
  if ((current.frames - 1) % LTX_FRAME_ALIGNMENT === 0) return ''
  return `LTX requires frame counts aligned to 8n+1. Current frame count ${current.frames} is blocking.`
})
const executionProfileWarning = computed(() => {
  const currentProfile = selectedExecutionProfile.value
  const currentOption = executionProfileOptions.value.find((entry) => entry.value === currentProfile)
  if (currentProfile && currentOption && !currentOption.supported) {
    const normalized = normalizeExecutionProfileName(currentProfile)
    const canonicalMatch = executionProfileOptions.value.find(
      (entry) => entry.supported && normalizeExecutionProfileName(entry.value) === normalized,
    )
    if (canonicalMatch) {
      return `Stored raw profile '${currentProfile}' is blocking because the canonical supported value is '${canonicalMatch.value}'. Re-select the supported profile instead of relying on silent remapping.`
    }
    return `Stored profile '${currentProfile}' is unsupported for the selected checkpoint. Re-select a supported profile.`
  }
  const message = String(blockedReason.value || '')
  if (
    message.includes('execution profile')
    || message.includes('checkpoint metadata')
    || message.includes('not executable')
  ) {
    return message
  }
  return ''
})

const hideNegativePrompt = computed(() => Number(params.value?.cfgScale) === 1)
const promptModeLabel = computed(() => (mode.value === 'img2vid' ? 'IMG2VID' : 'TXT2VID'))
const runGenerateDisabled = computed(() => isRunning.value || Boolean(blockedReason.value))
const runGenerateTitle = computed(() => (isRunning.value ? '' : blockedReason.value))
const runSummary = computed(() => {
  const current = params.value
  if (!current) return ''
  const profile = String(current.executionProfile || '').trim()
  const profileLabel = profile ? executionProfileLabel(profile) : 'Profile unresolved'
  return `${current.width}×${current.height} · ${current.frames}f @ ${current.fps}fps · ${profileLabel} · steps ${current.steps} · cfg ${current.cfgScale}`
})
const successMessage = computed(() => {
  const parts: string[] = []
  if (videoUrl.value) parts.push('Video ready')
  if (frames.value.length > 0) parts.push(`${frames.value.length} frame${frames.value.length === 1 ? '' : 's'} returned`)
  return parts.join(' · ') || 'Task finished.'
})
const videoZoomOpen = ref(false)

watch(videoUrl, (currentVideoUrl) => {
  if (!currentVideoUrl) videoZoomOpen.value = false
})

watch(
  () => {
    const metadata = checkpointExecutionMetadata.value
    return {
      checkpoint: String(params.value?.checkpoint || '').trim(),
      checkpointKind: String(metadata?.checkpointKind || '').trim(),
      defaultProfile: String(metadata?.defaultExecutionProfile || '').trim(),
      defaultStepsKey: typeof metadata?.defaultSteps === 'number' ? String(metadata.defaultSteps) : '',
      defaultGuidanceKey: typeof metadata?.defaultGuidanceScale === 'number' ? String(metadata.defaultGuidanceScale) : '',
      allowedProfilesKey: (metadata?.allowedExecutionProfiles ?? []).join('|'),
    }
  },
  (nextState, previousState) => {
    const current = params.value
    const isInitialRun = previousState === undefined

    const metadata = checkpointExecutionMetadata.value
    if (!current || !metadata) return
    if (metadata.checkpointKind === 'unknown') return
    const defaultProfile = String(metadata.defaultExecutionProfile || '').trim()
    if (!defaultProfile) return
    const metadataArrived = !String(previousState?.checkpointKind || '').trim() && Boolean(nextState.checkpointKind)
    const defaultsReady = nextState.defaultProfile !== '' && nextState.defaultStepsKey !== '' && nextState.defaultGuidanceKey !== ''
    const previousDefaultsReady =
      String(previousState?.defaultProfile || '').trim() !== ''
      && String(previousState?.defaultStepsKey || '').trim() !== ''
      && String(previousState?.defaultGuidanceKey || '').trim() !== ''
    const defaultsCompleted = defaultsReady && !previousDefaultsReady
    const previousCheckpoint = String(previousState?.checkpoint || '').trim()
    const checkpointChanged = previousCheckpoint !== nextState.checkpoint
    const currentProfile = String(current.executionProfile || '').trim()
    if (currentProfile) return
    const shouldApplyDefaults = isInitialRun || metadataArrived || checkpointChanged || defaultsCompleted
    if (!shouldApplyDefaults) return

    const patch: Partial<LtxTabParams> = {}
    if (current.executionProfile !== defaultProfile) patch.executionProfile = defaultProfile
    if (typeof metadata.defaultSteps === 'number' && current.steps !== metadata.defaultSteps) patch.steps = metadata.defaultSteps
    if (
      typeof metadata.defaultGuidanceScale === 'number'
      && Number(current.cfgScale) !== Number(metadata.defaultGuidanceScale)
    ) {
      patch.cfgScale = metadata.defaultGuidanceScale
    }
    if (Object.keys(patch).length > 0) updateParamsPatch(patch)
  },
  { immediate: true },
)

function normalizePositiveInt(rawValue: number, fallback: number, minimum = 1, maximum?: number): number {
  const numeric = Number.isFinite(rawValue) ? Math.trunc(rawValue) : Math.max(minimum, Math.trunc(fallback))
  const lowerClamped = Math.max(minimum, numeric)
  if (maximum === undefined) return lowerClamped
  return Math.min(maximum, lowerClamped)
}

function normalizeFiniteNumber(rawValue: number, fallback: number, minimum?: number, maximum?: number): number {
  const numeric = Number.isFinite(rawValue) ? Number(rawValue) : Number(fallback)
  let next = Number.isFinite(numeric) ? numeric : 0
  if (minimum !== undefined) next = Math.max(minimum, next)
  if (maximum !== undefined) next = Math.min(maximum, next)
  return next
}

function normalizeDimensionInput(rawValue: number, fallback: number): number {
  return normalizePositiveInt(rawValue, fallback, LTX_DIM_MIN, LTX_DIM_MAX)
}

function normalizeFrameInput(rawValue: number, fallback: number): number {
  return normalizePositiveInt(rawValue, fallback, LTX_FRAMES_MIN, LTX_FRAMES_MAX)
}

function updateParamsPatch(patch: Partial<LtxTabParams>): void {
  store.updateParams(props.tabId, patch as Partial<Record<string, unknown>>).catch((error) => {
    toast(error instanceof Error ? error.message : String(error))
  })
}

async function onInitImageFile(file: File): Promise<void> {
  try {
    const dataUrl = await readFileAsDataURL(file)
    updateParamsPatch({
      mode: 'img2vid',
      initImageData: dataUrl,
      initImageName: file.name,
    })
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error))
  }
}

function onInitImageRejected(payload: { reason: string; files: File[] }): void {
  const suffix = payload.files.length > 0 ? ` (${payload.files.map((file) => file.name).join(', ')})` : ''
  toast(`${payload.reason}${suffix}`)
}

function clearInit(): void {
  updateParamsPatch({ initImageData: '', initImageName: '' })
}

function openResultVideoZoom(): void {
  if (!videoUrl.value) return
  videoZoomOpen.value = true
}

function setVideoZoomOpen(value: boolean): void {
  videoZoomOpen.value = value
}

function toDataUrl(image: GeneratedImage): string {
  return `data:image/${image.format};base64,${image.data}`
}

function formatVideoModeLabel(modeValue: unknown): string {
  const normalized = String(modeValue ?? '').trim().toLowerCase()
  if (normalized === 'img2vid') return 'Img2Vid'
  if (normalized === 'txt2vid') return 'Txt2Vid'
  return `Unsupported (${normalized || 'unknown'})`
}

function formatHistoryTitle(item: LtxRunHistoryItem): string {
  const dt = new Date(item.createdAtMs || Date.now())
  const hh = dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  return `${formatVideoModeLabel(item.mode)} · ${hh}`
}

async function onSelectHistoryItem(item: { taskId: string }): Promise<void> {
  await loadHistory(item.taskId)
}

const slotProps = computed(() => ({
  tab: tab.value,
  params: params.value,
  mode: mode.value,
  status: status.value,
  progress: progress.value,
  frames: frames.value,
  info: info.value,
  videoUrl: videoUrl.value,
  videoZoomOpen: videoZoomOpen.value,
  errorMessage: errorMessage.value,
  isRunning: isRunning.value,
  history: history.value,
  selectedTaskId: selectedTaskId.value,
  historyLoadingTaskId: historyLoadingTaskId.value,
  checkpointExecutionMetadata: checkpointExecutionMetadata.value,
  copyNotice: copyNotice.value,
  resumeNotice: resumeNotice.value,
  workflowBusy: workflowBusy.value,
  dimensionAlignment: dimensionAlignment.value,
  executionProfileOptions: executionProfileOptions.value,
  dimensionWarning: dimensionWarning.value,
  frameWarning: frameWarning.value,
  executionProfileWarning: executionProfileWarning.value,
  hideNegativePrompt: hideNegativePrompt.value,
  promptModeLabel: promptModeLabel.value,
  runGenerateDisabled: runGenerateDisabled.value,
  runGenerateTitle: runGenerateTitle.value,
  runSummary: runSummary.value,
  successMessage: successMessage.value,
  generate,
  cancel,
  updateParamsPatch,
  onInitImageFile,
  onInitImageRejected,
  clearInit,
  openResultVideoZoom,
  setVideoZoomOpen,
  sendToWorkflows,
  copyCurrentParams,
  onSelectHistoryItem,
  clearHistory,
  formatHistoryTitle,
  normalizeDimensionInput,
  normalizeFrameInput,
  normalizePositiveInt,
  normalizeFiniteNumber,
  copyJson,
  formatJson,
  toDataUrl,
  ltxDimMin: LTX_DIM_MIN,
  ltxDimMax: LTX_DIM_MAX,
  ltxFramesMin: LTX_FRAMES_MIN,
  ltxFramesMax: LTX_FRAMES_MAX,
  ltxFrameAlignment: LTX_FRAME_ALIGNMENT,
  LTX_DIM_MIN,
  LTX_DIM_MAX,
  LTX_FRAMES_MIN,
  LTX_FRAMES_MAX,
  LTX_FRAME_ALIGNMENT,
}))

</script>
