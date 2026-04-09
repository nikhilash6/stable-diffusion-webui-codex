<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Upscale route view.
Standalone upscaling workspace (Spandrel SR) with tile controls (tile/overlap/min_tile), explicit OOM fallback toggle, HF-backed weight downloads, and task streaming.
Persists a minimal resume marker to `localStorage` and auto-reattaches to in-flight upscale tasks after reload (SSE replay via `after` / `lastEventId`).
Run status rendering is standardized through the shared `RunProgressStatus` panel (`progress/error/info` variants + Stage/Progress/Step/ETA metadata).
Run cancellation is owned by the `RunCard` center CTA (destructive two-click confirm), so the Results header no longer carries a separate cancel button.
The remote download modal:
- surfaces backend safeweights mode (`CODEX_SAFE_WEIGHTS`) and allowed suffixes,
- shows manifest issues explicitly, and
- splits remote weights into Curated (manifest metadata) vs Other files (raw listing), with raw manifest JSON behind a hamburger (Advanced) action.

Symbols (top-level; keep in sync; no ghosts):
- `Upscale` (component): Upscale route view component.
- `setMinTile` (function): Updates the persisted `min_tile` preference used as the tiled OOM fallback lower bound.
- `start` (function): Starts an upscale task (`/api/upscale`) and subscribes to SSE events.
- `cancel` (function): Cancels the active upscale task.
- `downloadSelectedRemote` (function): Starts an upscaler download task (`/api/upscalers/download`) and subscribes to SSE events.
- `handleTaskEvent` (function): Task SSE handler for upscale runs.
- `handleDownloadEvent` (function): Task SSE handler for download runs.
-->

<template>
  <section class="panels">
    <!-- Left column: controls -->
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header">Upscale</div>
        <div class="panel-body">
          <InitialImageCard
            label="Source Image"
            :disabled="isRunning"
            :src="imagePreview"
            :hasImage="Boolean(imageFile)"
            :thumbnail="true"
            accept="image/*"
            placeholder="Select an image to upscale."
            @set="onImageSet"
            @clear="clearImage"
          >
            <template #footer>
              <p v-if="imageFile" class="caption">Selected: {{ imageFile.name }}</p>
            </template>
          </InitialImageCard>

          <div class="panel-section">
            <label class="label-muted">Upscaler</label>
            <div class="toolbar">
              <select class="select-md" :disabled="isRunning || isLoadingUpscalers" v-model="upscalerId">
                <option value="" disabled>Select</option>
                <optgroup label="Spandrel (pixel SR)">
                  <option v-for="u in spandrelUpscalers" :key="u.id" :value="u.id">{{ u.label }}</option>
                </optgroup>
                <optgroup v-if="latentUpscalers.length" label="Latent (hires-only)">
                  <option v-for="u in latentUpscalers" :key="u.id" :value="u.id" disabled>{{ u.label }}</option>
                </optgroup>
              </select>
              <button class="btn btn-sm btn-outline" type="button" :disabled="isRunning" @click="openRemoteModal">
                Download…
              </button>
            </div>
            <p v-if="upscalersError" class="caption">Error: {{ upscalersError }}</p>
            <p v-else-if="!isLoadingUpscalers && spandrelUpscalers.length === 0" class="caption">
              No upscaler weights installed. Use Download or drop `*.safetensors|*.pth|*.pt` into `models/upscale_models/` or `models/upscalers/`.
            </p>
          </div>

          <div class="panel-section">
            <label class="label-muted">Scale</label>
            <div class="toolbar">
              <button class="btn qs-toggle-btn qs-toggle-btn--sm" :class="scale === 2 ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off'" type="button" :disabled="isRunning" @click="setScale(2)">2×</button>
              <button class="btn qs-toggle-btn qs-toggle-btn--sm" :class="scale === 3 ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off'" type="button" :disabled="isRunning" @click="setScale(3)">3×</button>
              <button class="btn qs-toggle-btn qs-toggle-btn--sm" :class="scale === 4 ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off'" type="button" :disabled="isRunning" @click="setScale(4)">4×</button>
              <NumberStepperInput
                :modelValue="scale"
                :min="0.1"
                :max="16"
                :step="0.5"
                :nudgeStep="0.5"
                size="sm"
                inputClass="cdx-input-w-sm"
                :disabled="isRunning"
                updateOnInput
                @update:modelValue="setScale"
              />
            </div>
          </div>

          <div class="panel-section">
            <label class="label-muted">Tile</label>
            <UpscalerTileControls
              :tileSize="tileSize"
              :overlap="overlap"
              :minTile="Math.min(tileSize, minTile)"
              :fallbackOnOom="fallbackOnOom"
              :disabled="isRunning"
              @update:tileSize="setTileSize"
              @update:overlap="setOverlap"
              @update:minTile="setMinTile"
              @update:fallbackOnOom="setFallbackOnOom"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Right column: run + results -->
    <div class="panel-stack panel-stack--sticky">
      <RunCard
        title="Run"
        :showBatchControls="false"
        :generateDisabled="isRunning || !canRun"
        :isRunning="isRunning"
        :disabled="isRunning"
        generateLabel="Upscale"
        runningLabel="Upscaling…"
        :generateTitle="generateTitle"
        @generate="start"
        @cancel="cancel"
      >
        <RunProgressStatus
          v-if="errorMessage"
          variant="error"
          title="Upscale failed"
          :message="errorMessage"
          :show-progress-bar="false"
        />
        <RunProgressStatus
          v-else-if="isRunning"
          :stage="progress?.stage || 'starting'"
          :percent="progress?.percent ?? null"
          :step="progress?.step ?? null"
          :total-steps="progress?.totalSteps ?? null"
          :eta-seconds="progress?.etaSeconds ?? null"
        />
        <RunProgressStatus
          v-if="notice"
          variant="info"
          title="Notice"
          :message="notice"
          :show-progress-bar="false"
        />
      </RunCard>

      <ResultsCard :showGenerate="false" headerClass="three-cols results-sticky" headerRightClass="results-actions">
        <template #header-right>
          <input class="ui-input" list="upscale-preset-list" v-model="presetName" placeholder="Preset" />
          <datalist id="upscale-preset-list"><option v-for="p in presetNames" :key="p" :value="p" /></datalist>
          <button class="btn btn-sm btn-secondary" type="button" :disabled="isRunning" @click="savePreset(presetName)">Save</button>
          <button class="btn btn-sm btn-outline" type="button" :disabled="isRunning" @click="applyPreset(presetName)">Apply</button>
        </template>

        <ResultViewer
          mode="image"
          :images="images"
          :isRunning="isRunning"
          emptyText="Upscaled image(s) will appear here."
        >
          <template #image-actions="{ image, index }">
            <button class="gallery-action" type="button" title="Download Image" @click="download(image, index)">
              Download
            </button>
          </template>
        </ResultViewer>
      </ResultsCard>

      <div class="panel" v-if="info">
        <div class="panel-header">Upscale Info</div>
        <div class="panel-body">
          <pre class="text-xs break-words">{{ formatJson(info) }}</pre>
        </div>
      </div>
    </div>
  </section>

  <Modal title="Download Upscalers" v-model="remoteModalOpen">
    <div class="panel-section">
      <div class="toolbar">
        <button class="btn btn-sm btn-outline" type="button" :disabled="remoteLoading || downloadBusy" @click="refreshRemote">Refresh</button>
        <button class="btn btn-sm btn-secondary" type="button" :disabled="remoteLoading || downloadBusy || !remoteSelectable.length" @click="selectAllRemote">
          Select all
        </button>
        <button class="btn btn-sm btn-outline" type="button" :disabled="remoteLoading || downloadBusy || !remoteSelected.size" @click="clearRemoteSelection">
          Clear
        </button>
        <button
          class="btn-icon"
          type="button"
          :disabled="remoteLoading || downloadBusy || !remote?.manifest"
          aria-label="Raw manifest JSON"
          title="Raw manifest JSON"
          @click="openRawManifest"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 7H20M4 12H20M4 17H20" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" />
          </svg>
        </button>
      </div>

      <p v-if="remoteError" class="caption">Error: {{ remoteError }}</p>
      <p v-else-if="remoteLoading" class="caption">Loading remote list…</p>
      <template v-else-if="remote">
        <p class="caption">
          Safeweights: <strong>{{ remote.safeweights_enabled ? 'ON' : 'OFF' }}</strong>
          <span v-if="remote.allowed_weight_suffixes?.length"> (allowed: {{ remote.allowed_weight_suffixes.join(', ') }})</span>
          <span v-else-if="remote.safeweights_enabled"> (allowed: .safetensors)</span>
          <span v-else> (allowed: .safetensors, .pt, .pth)</span>
        </p>
        <p v-if="remote.manifest_found && remote.manifest_error" class="caption">
          Manifest issues: {{ remote.manifest_error }}
        </p>
        <details v-if="remote.manifest_found && remote.manifest_errors?.length" class="accordion">
          <summary>Manifest issues (details)</summary>
          <div class="accordion-body">
            <ul class="caption">
              <li v-for="(e, idx) in remote.manifest_errors" :key="idx">{{ e }}</li>
            </ul>
          </div>
        </details>
        <p v-else-if="!remote.manifest_found" class="caption">
          Manifest not found; showing raw `upscalers/**` listing (no metadata).
        </p>

        <div class="panel-section">
          <label class="label-muted">Curated (manifest)</label>
          <div v-if="curatedRemoteWeights.length" class="form-grid">
            <div v-for="(w, idx) in curatedRemoteWeights" :key="w.hf_path" class="form-row">
              <div class="toolbar">
                <input
                  :id="`remote-curated-${idx}`"
                  type="checkbox"
                  :checked="remoteSelected.has(w.hf_path)"
                  :disabled="downloadBusy"
                  @change="toggleRemote(w.hf_path, ($event.target as HTMLInputElement).checked)"
                />
                <label class="caption" :for="`remote-curated-${idx}`">{{ w.label }}</label>
              </div>
              <p class="caption">
                Arch: <strong>{{ w.meta.arch }}</strong> · Scale: <strong>{{ w.meta.scale }}×</strong>
              </p>
              <p class="caption">
                License:
                <a :href="w.meta.license_url" target="_blank" rel="noreferrer">{{ w.meta.license_name }}</a>
                <span v-if="w.meta.license_spdx"> ({{ w.meta.license_spdx }})</span>
              </p>
              <p class="caption">SHA256: <code>{{ w.meta.sha256 }}</code></p>
              <p v-if="w.meta.tags?.length" class="caption">Tags: {{ w.meta.tags.join(', ') }}</p>
              <p v-if="w.meta.notes" class="caption">Notes: {{ w.meta.notes }}</p>
              <p class="caption">HF path: <code>{{ w.hf_path }}</code></p>
            </div>
          </div>
          <p v-else class="caption">No curated weights available.</p>
        </div>

        <div class="panel-section">
          <label class="label-muted">Other files</label>
          <div v-if="otherRemoteWeights.length" class="form-grid">
            <div v-for="(w, idx) in otherRemoteWeights" :key="w.hf_path" class="form-row">
              <div class="toolbar">
                <input
                  :id="`remote-other-${idx}`"
                  type="checkbox"
                  :checked="remoteSelected.has(w.hf_path)"
                  :disabled="downloadBusy"
                  @change="toggleRemote(w.hf_path, ($event.target as HTMLInputElement).checked)"
                />
                <label class="caption" :for="`remote-other-${idx}`">{{ w.label }}</label>
              </div>
              <p class="caption">HF path: <code>{{ w.hf_path }}</code></p>
              <p class="caption">No manifest metadata for this file.</p>
            </div>
          </div>
          <p v-else class="caption">No other weights found in the remote repo.</p>
        </div>

        <div class="panel-section">
          <div class="toolbar">
            <button class="btn btn-md btn-primary" type="button" :disabled="downloadBusy || remoteSelected.size === 0" @click="downloadSelectedRemote">
              {{ downloadBusy ? 'Downloading…' : `Download (${remoteSelected.size})` }}
            </button>
            <button class="btn btn-md btn-outline" type="button" :disabled="downloadBusy" @click="remoteModalOpen = false">Close</button>
          </div>
          <p v-if="downloadStatusLine" class="caption">{{ downloadStatusLine }}</p>
          <p v-if="downloadError" class="caption">Error: {{ downloadError }}</p>
        </div>
      </template>
    </div>
  </Modal>

  <Modal title="Upscalers manifest (raw)" v-model="rawManifestModalOpen">
    <div v-if="!remote?.manifest" class="card text-sm">
      No manifest JSON available (missing or invalid).
    </div>

    <div v-else class="card text-sm cdx-json-scroll">
      <JsonTreeView :value="remote.manifest" :default-open-depth="1" :max-depth="12" />
    </div>

    <template #footer>
      <button class="btn btn-md btn-outline" type="button" @click="rawManifestModalOpen = false">Close</button>
    </template>
  </Modal>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { GeneratedImage, RemoteUpscalerWeight, RemoteUpscalersResponse, TaskEvent } from '../api/types'
import {
  cancelTask,
  downloadUpscalers,
  fetchTaskResult,
  fetchRemoteUpscalers,
  startUpscale,
  subscribeTask,
} from '../api/client'
import { formatJson, useResultsCard } from '../composables/useResultsCard'
import InitialImageCard from '../components/InitialImageCard.vue'
import { usePresetsStore } from '../stores/presets'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useUpscalersStore } from '../stores/upscalers'
import Modal from '../components/ui/Modal.vue'
import JsonTreeView from '../components/ui/JsonTreeView.vue'
import NumberStepperInput from '../components/ui/NumberStepperInput.vue'
import UpscalerTileControls from '../components/ui/UpscalerTileControls.vue'
import ResultViewer from '../components/ResultViewer.vue'
import ResultsCard from '../components/results/ResultsCard.vue'
import RunCard from '../components/results/RunCard.vue'
import RunProgressStatus from '../components/results/RunProgressStatus.vue'
import { readFileAsDataURL } from '../utils/image_io'

const presets = usePresetsStore()
const presetName = ref('')
const presetNames = computed(() => presets.names('upscale'))

const quicksettings = useQuicksettingsStore()
const { notice, toast } = useResultsCard()

const upscalersStore = useUpscalersStore()
const {
  upscalers,
  loading: isLoadingUpscalers,
  error: upscalersError,
  spandrelUpscalers,
  latentUpscalers,
  fallbackOnOom,
  minTile,
} = storeToRefs(upscalersStore)

const imageFile = ref<File | null>(null)
const imagePreview = ref('')

const upscalerId = ref<string>('')

const scale = ref<number>(2)
const tileSize = ref<number>(256)
const overlap = ref<number>(16)

const images = ref<GeneratedImage[]>([])
const info = ref<unknown | null>(null)
const errorMessage = ref('')
const taskId = ref('')
const isRunning = ref(false)
const progress = ref<{ stage: string; percent: number | null; etaSeconds: number | null; step: number | null; totalSteps: number | null } | null>(null)
let unsubTask: (() => void) | null = null

const RESUME_STORAGE_KEY = 'codex.resume.upscale'
const resumeToastShown = new Set<string>()

type ResumeState = {
  taskId: string
  lastEventId: number
  createdAtMs: number
  paramsSnapshot: Record<string, unknown>
}

function loadResumeState(): ResumeState | null {
  try {
    const raw = localStorage.getItem(RESUME_STORAGE_KEY)
    if (!raw) return null
    const obj = JSON.parse(raw) as any
    if (!obj || typeof obj !== 'object') return null
    if (typeof obj.taskId !== 'string' || !obj.taskId.trim()) return null
    const lastEventId = typeof obj.lastEventId === 'number' && Number.isFinite(obj.lastEventId) ? Math.trunc(obj.lastEventId) : 0
    const createdAtMs = typeof obj.createdAtMs === 'number' && Number.isFinite(obj.createdAtMs) ? Math.trunc(obj.createdAtMs) : 0
    const paramsSnapshot = obj.paramsSnapshot && typeof obj.paramsSnapshot === 'object' ? (obj.paramsSnapshot as Record<string, unknown>) : {}
    return { taskId: obj.taskId, lastEventId: Math.max(0, lastEventId), createdAtMs, paramsSnapshot }
  } catch {
    return null
  }
}

function saveResumeState(state: ResumeState): void {
  try {
    localStorage.setItem(RESUME_STORAGE_KEY, JSON.stringify(state))
  } catch {
    // ignore localStorage failures (private mode/quota)
  }
}

function clearResumeState(): void {
  try {
    localStorage.removeItem(RESUME_STORAGE_KEY)
  } catch {
    // ignore
  }
}

function updateResumeEventId(eventId: number): void {
  const v = Math.trunc(Number(eventId))
  if (!Number.isFinite(v) || v <= 0) return
  const cur = loadResumeState()
  if (!cur) return
  if (v <= cur.lastEventId) return
  saveResumeState({ ...cur, lastEventId: v })
}

async function refreshTaskSnapshot(id: string): Promise<void> {
  try {
    const res = await fetchTaskResult(id)
    if (res.status !== 'running') return
    const p = res.progress
    if (p && typeof p === 'object') {
      progress.value = {
        stage: String(p.stage ?? progress.value?.stage ?? 'running'),
        percent: p.percent ?? null,
        etaSeconds: p.eta_seconds ?? null,
        step: p.step ?? null,
        totalSteps: p.total_steps ?? null,
      }
    } else if (typeof res.stage === 'string' && res.stage.trim()) {
      progress.value = { stage: res.stage, percent: null, etaSeconds: null, step: null, totalSteps: null }
    }
  } catch {
    // ignore snapshot refresh failures
  }
}

async function tryAutoResume(): Promise<void> {
  const saved = loadResumeState()
  if (!saved) return

  let res
  try {
    res = await fetchTaskResult(saved.taskId)
  } catch {
    clearResumeState()
    return
  }

  if (res.status === 'running') {
    stopStreams()
    isRunning.value = true
    taskId.value = saved.taskId
    errorMessage.value = ''
    const p = res.progress
    if (p && typeof p === 'object') {
      progress.value = {
        stage: String(p.stage ?? 'running'),
        percent: p.percent ?? null,
        etaSeconds: p.eta_seconds ?? null,
        step: p.step ?? null,
        totalSteps: p.total_steps ?? null,
      }
    } else if (typeof res.stage === 'string' && res.stage.trim()) {
      progress.value = { stage: res.stage, percent: null, etaSeconds: null, step: null, totalSteps: null }
    }
    unsubTask = subscribeTask(saved.taskId, handleTaskEvent, undefined, {
      after: saved.lastEventId,
      onMeta: ({ eventId }) => {
        if (typeof eventId === 'number') updateResumeEventId(eventId)
      },
    })
    if (!resumeToastShown.has(saved.taskId)) {
      toast('Reconnected (resumed task).')
      resumeToastShown.add(saved.taskId)
    }
    return
  }

  clearResumeState()
  isRunning.value = false

  if (res.status === 'completed' && res.result) {
    images.value = res.result.images || []
    info.value = res.result.info ?? null
    return
  }
  if (res.status === 'error') {
    errorMessage.value = String(res.error || 'Task failed.')
  }
}

const remoteModalOpen = ref(false)
const remote = ref<RemoteUpscalersResponse | null>(null)
const remoteLoading = ref(false)
const remoteError = ref('')
const remoteSelected = ref<Set<string>>(new Set())
const rawManifestModalOpen = ref(false)
const downloadBusy = ref(false)
const downloadProgress = ref<{ stage: string; percent: number | null; step: number | null; totalSteps: number | null } | null>(null)
const downloadError = ref('')
let unsubDownload: (() => void) | null = null

const canRun = computed(() => Boolean(imageFile.value) && Boolean(upscalerId.value) && spandrelUpscalers.value.some((u) => u.id === upscalerId.value))
const generateTitle = computed(() => {
  if (!imageFile.value) return 'Select an image to upscale.'
  if (!upscalerId.value) return 'Select an upscaler.'
  if (!spandrelUpscalers.value.some((u) => u.id === upscalerId.value)) return 'Select a Spandrel (pixel SR) upscaler.'
  return ''
})

const remoteSelectable = computed<RemoteUpscalerWeight[]>(() => remote.value?.weights ?? [])

function isCuratedRemoteWeight(w: RemoteUpscalerWeight): w is Extract<RemoteUpscalerWeight, { curated: true }> {
  return w.curated
}

const curatedRemoteWeights = computed(() => remoteSelectable.value.filter(isCuratedRemoteWeight))
const otherRemoteWeights = computed(() => remoteSelectable.value.filter((w) => !w.curated))

const downloadStatusLine = computed(() => {
  const p = downloadProgress.value
  if (!p) return ''
  const pct = p.percent !== null && Number.isFinite(p.percent) ? `${p.percent.toFixed(1)}%` : ''
  const steps = (p.step !== null && p.totalSteps !== null) ? `${p.step}/${p.totalSteps}` : ''
  const parts = [p.stage, pct, steps].filter((s) => String(s || '').trim())
  return parts.join(' · ')
})

function setFallbackOnOom(value: boolean): void {
  fallbackOnOom.value = Boolean(value)
}

function setMinTile(value: number): void {
  const v = Math.max(1, Math.trunc(Number(value)))
  if (!Number.isFinite(v)) return
  minTile.value = v
}

async function loadUpscalers(): Promise<void> {
  await upscalersStore.load({ refresh: true })
  if (!upscalerId.value) {
    const first = upscalers.value.find((u) => u.kind === 'spandrel')
    if (first) upscalerId.value = first.id
  }
}

function snapshotParams(): Record<string, unknown> {
  return {
    upscalerId: upscalerId.value,
    scale: scale.value,
    tile: {
      tile: tileSize.value,
      overlap: overlap.value,
      fallbackOnOom: Boolean(fallbackOnOom.value),
      minTile: minTile.value,
    },
  }
}

function applyParams(v: Record<string, unknown>): void {
  if (typeof v.upscalerId === 'string') upscalerId.value = v.upscalerId
  if (typeof v.scale === 'number' && Number.isFinite(v.scale) && v.scale > 0) scale.value = v.scale
  const tile = v.tile
  if (tile && typeof tile === 'object') {
    const t = tile as any
    if (typeof t.tile === 'number' && Number.isFinite(t.tile)) setTileSize(t.tile)
    if (typeof t.overlap === 'number' && Number.isFinite(t.overlap)) setOverlap(t.overlap)
    if (typeof t.fallbackOnOom === 'boolean') fallbackOnOom.value = t.fallbackOnOom
    if (typeof t.minTile === 'number' && Number.isFinite(t.minTile)) minTile.value = Math.max(1, Math.trunc(t.minTile))
  }
}

function savePreset(name: string): void {
  const trimmed = String(name || '').trim()
  if (!trimmed) {
    toast('Preset name is required.')
    return
  }
  presets.upsert('upscale', trimmed, snapshotParams())
  presetName.value = trimmed
  toast(`Saved preset '${trimmed}'.`)
}

function applyPreset(name: string): void {
  const trimmed = String(name || '').trim()
  if (!trimmed) return
  const v = presets.get('upscale', trimmed)
  if (!v) {
    toast(`Preset '${trimmed}' not found.`)
    return
  }
  applyParams(v)
  toast(`Applied preset '${trimmed}'.`)
}

function setScale(value: number): void {
  const v = Number(value)
  if (!Number.isFinite(v) || v <= 0) return
  scale.value = Math.max(0.1, Math.min(16, v))
}

function setTileSize(value: number): void {
  const v = Math.trunc(Number(value))
  if (!Number.isFinite(v) || v <= 0) return
  tileSize.value = v
  // Keep invariants aligned with backend validation.
  if (overlap.value >= tileSize.value) overlap.value = Math.max(0, tileSize.value - 1)
}

function setOverlap(value: number): void {
  const v = Math.trunc(Number(value))
  if (!Number.isFinite(v)) return
  overlap.value = Math.max(0, Math.min(tileSize.value - 1, v))
}

async function onImageSet(file: File): Promise<void> {
  try {
    imageFile.value = file
    imagePreview.value = await readFileAsDataURL(file)
  } catch (err) {
    imageFile.value = null
    imagePreview.value = ''
    toast(err instanceof Error ? err.message : String(err))
  }
}

function clearImage(): void {
  imageFile.value = null
  imagePreview.value = ''
}

function stopStreams(): void {
  if (unsubTask) unsubTask()
  unsubTask = null
  if (unsubDownload) unsubDownload()
  unsubDownload = null
}

function handleTaskEvent(event: TaskEvent): void {
  switch (event.type) {
    case 'status':
      progress.value = { stage: event.stage, percent: null, etaSeconds: null, step: null, totalSteps: null }
      break
    case 'progress':
      progress.value = {
        stage: event.stage,
        percent: event.percent ?? null,
        etaSeconds: event.eta_seconds ?? null,
        step: event.step ?? null,
        totalSteps: event.total_steps ?? null,
      }
      break
    case 'result':
      images.value = event.images || []
      info.value = event.info ?? null
      break
    case 'gap':
      if (taskId.value) void refreshTaskSnapshot(taskId.value)
      break
    case 'error':
      errorMessage.value = event.message
      isRunning.value = false
      stopStreams()
      clearResumeState()
      break
    case 'end':
      clearResumeState()
      isRunning.value = false
      if (unsubTask) unsubTask()
      unsubTask = null
      break
  }
}

function handleDownloadEvent(event: TaskEvent): void {
  switch (event.type) {
    case 'status':
      downloadProgress.value = { stage: event.stage, percent: null, step: null, totalSteps: null }
      break
    case 'progress':
      downloadProgress.value = {
        stage: event.stage,
        percent: event.percent ?? null,
        step: event.step ?? null,
        totalSteps: event.total_steps ?? null,
      }
      break
    case 'result': {
      const files = (event.info as any)?.files
      const count = Array.isArray(files) ? files.length : null
      toast(count !== null ? `Downloaded ${count} file(s).` : 'Download completed.')
      void loadUpscalers()
      break
    }
    case 'error':
      downloadError.value = event.message
      break
    case 'end':
      downloadBusy.value = false
      break
  }
}

async function start(): Promise<void> {
  if (!canRun.value || !imageFile.value) return

  stopStreams()
  images.value = []
  info.value = null
  errorMessage.value = ''
  progress.value = null

  isRunning.value = true

  let payload: Record<string, unknown>
  try {
    payload = {
      device: String(quicksettings.currentDevice || '').trim().toLowerCase(),
      upscaler_id: upscalerId.value,
      scale: Number(scale.value),
      tile: {
        tile: Math.trunc(tileSize.value),
        overlap: Math.trunc(overlap.value),
        fallback_on_oom: Boolean(fallbackOnOom.value),
        min_tile: Math.trunc(Math.min(tileSize.value, minTile.value)),
      },
    }
    if (!payload.device) throw new Error("Missing device (QuickSettings).")
  } catch (err) {
    isRunning.value = false
    errorMessage.value = err instanceof Error ? err.message : String(err)
    return
  }

  try {
    const { task_id } = await startUpscale(imageFile.value, payload)
    taskId.value = task_id
  } catch (err) {
    isRunning.value = false
    errorMessage.value = err instanceof Error ? err.message : String(err)
    return
  }

  const createdAtMs = Date.now()
  const paramsSnapshot = {
    device: payload.device,
    upscaler_id: payload.upscaler_id,
    scale: payload.scale,
    tile: payload.tile,
  }
  saveResumeState({ taskId: taskId.value, lastEventId: 0, createdAtMs, paramsSnapshot })

  unsubTask = subscribeTask(taskId.value, handleTaskEvent, (err) => {
    if (err) console.warn('[upscale] task stream error', err)
  }, {
    onMeta: ({ eventId }) => {
      if (typeof eventId === 'number') updateResumeEventId(eventId)
    },
  })
}

async function cancel(): Promise<void> {
  if (!taskId.value) return
  try {
    await cancelTask(taskId.value, 'immediate')
    toast('Cancellation requested.')
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err))
  }
}

function toDataUrl(image: GeneratedImage): string {
  return `data:image/${image.format};base64,${image.data}`
}

function download(image: GeneratedImage, index: number): void {
  const link = document.createElement('a')
  link.href = toDataUrl(image)
  link.download = `upscale_${index + 1}.png`
  link.click()
}

function openRemoteModal(): void {
  remoteModalOpen.value = true
}

function openRawManifest(): void {
  if (!remote.value?.manifest) return
  rawManifestModalOpen.value = true
}

async function refreshRemote(): Promise<void> {
  remoteLoading.value = true
  remoteError.value = ''
  remote.value = null
  remoteSelected.value = new Set()
  try {
    remote.value = await fetchRemoteUpscalers()
  } catch (err) {
    remoteError.value = err instanceof Error ? err.message : String(err)
  } finally {
    remoteLoading.value = false
  }
}

watch(remoteModalOpen, (open) => {
  if (!open) return
  if (remote.value || remoteLoading.value) return
  void refreshRemote()
})

function toggleRemote(hfPath: string, enabled: boolean): void {
  const s = new Set(remoteSelected.value)
  if (enabled) s.add(hfPath)
  else s.delete(hfPath)
  remoteSelected.value = s
}

function selectAllRemote(): void {
  remoteSelected.value = new Set(remoteSelectable.value.map((w) => w.hf_path))
}

function clearRemoteSelection(): void {
  remoteSelected.value = new Set()
}

async function downloadSelectedRemote(): Promise<void> {
  if (!remoteSelected.value.size) return

  stopStreams()
  downloadBusy.value = true
  downloadError.value = ''
  downloadProgress.value = null

  const files = Array.from(remoteSelected.value)
  try {
    const { task_id } = await downloadUpscalers({ files })
    const id = task_id
    unsubDownload = subscribeTask(id, handleDownloadEvent, (err) => {
      if (err) console.warn('[upscale] download stream error', err)
    })
  } catch (err) {
    downloadBusy.value = false
    downloadError.value = err instanceof Error ? err.message : String(err)
  }
}

onMounted(() => {
  void loadUpscalers()
  void tryAutoResume()
})

onBeforeUnmount(() => {
  stopStreams()
})
</script>
