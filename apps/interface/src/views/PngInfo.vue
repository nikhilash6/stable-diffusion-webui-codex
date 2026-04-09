<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: PNG info inspection view.
Inspect uploaded PNG metadata, parse common infotext formats, and bridge extracted parameters into model tabs and workflow snapshots
(including family-aware sampler/scheduler patching that never applies family-invalid values).

Symbols (top-level; keep in sync; no ghosts):
- `PngInfo` (component): PNG info route view component.
- `clearLoadedPng` (function): Resets the currently loaded PNG preview/analysis state from the dropzone.
-->

<template>
  <section class="panels">
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header">Drop PNG</div>
        <div class="panel-body">
          <Dropzone
            accept="image/png,.png"
            label="Drop a PNG here, or click to browse"
            hint="We extract text metadata server-side, then parse infotext locally."
            @select="onDropFiles"
            @rejected="onDropRejected"
          >
            <div class="pnginfo-dropzone-slot">
              <div v-if="previewDataUrl" class="pnginfo-preview">
                <button
                  class="btn btn-icon pnginfo-clear-button"
                  type="button"
                  aria-label="Clear loaded PNG"
                  title="Clear"
                  @click.stop="clearLoadedPng"
                  @keydown.enter.stop
                  @keydown.space.stop
                >
                  ✕
                </button>
                <img :src="previewDataUrl" alt="PNG preview" />
              </div>
              <div class="pnginfo-dropzone-meta">
                <div class="pnginfo-dropzone-title">
                  {{ selectedFile ? selectedFile.name : 'Drop a PNG here, or click to browse' }}
                </div>
                <div class="caption">
                  <span v-if="analysis">{{ analysis.width }}×{{ analysis.height }} px</span>
                  <span v-else>PNG only · no upload storage</span>
                </div>
              </div>
            </div>
          </Dropzone>
        </div>
      </div>
    </div>

    <div class="panel-stack">
      <ResultsCard :showGenerate="false" headerClass="three-cols results-sticky" headerRightClass="pnginfo-header-actions" title="PNG Info">
        <template #header-right>
          <div class="pnginfo-header-controls">
            <div class="pnginfo-header-selects">
              <select class="select-md pnginfo-select" v-model="targetTabId" :disabled="!compatibleTabs.length">
                <option value="" disabled>Select tab</option>
                <option v-for="t in compatibleTabs" :key="t.id" :value="t.id">
                  {{ t.title }} ({{ t.type }})
                </option>
              </select>

              <select class="select-md pnginfo-select" v-model="targetMode" :disabled="!targetTab">
                <option value="txt2img">txt2img</option>
                <option value="img2img">img2img</option>
              </select>
            </div>

            <div class="pnginfo-header-buttons">
              <button class="btn btn-sm btn-secondary" type="button" :disabled="!canSaveSnapshot" @click="saveSnapshot">
                {{ workflowBusy ? 'Saving…' : 'Save snapshot' }}
              </button>

              <button class="btn btn-sm btn-primary" type="button" :disabled="!canSendTo" @click="sendTo">
                {{ sendBusy ? 'Sending…' : 'Send to' }}
              </button>
            </div>
          </div>
        </template>

        <div v-if="notice" class="pnginfo-notice-row">
          <div class="caption">{{ notice }}</div>
          <RouterLink v-if="lastSentTabId" class="btn btn-sm btn-outline" :to="`/models/${lastSentTabId}`">Open tab</RouterLink>
        </div>

        <div v-if="error" class="panel-error">{{ error }}</div>

        <div v-else-if="!selectedFile" class="viewer-card">
          <div class="viewer-empty">Drop a PNG on the left to inspect metadata and parse infotext.</div>
        </div>

        <div v-else class="pnginfo-body">
          <div v-if="allWarnings.length" class="pnginfo-warnings">
            <div class="pnginfo-warnings-title">Warnings</div>
            <ul class="pnginfo-warnings-list">
              <li v-for="(w, idx) in allWarnings" :key="idx">{{ w }}</li>
            </ul>
          </div>

          <div class="pnginfo-section">
            <div class="pnginfo-section-title">Infotext</div>
            <textarea
              v-model="infotext"
              class="ui-textarea h-prompt-sm"
              placeholder="Infotext (e.g. A1111 'parameters'). Edit to re-parse."
            />
          </div>

          <div class="pnginfo-grid">
            <div class="pnginfo-card">
              <div class="pnginfo-card-title">Parsed</div>
              <div v-if="!hasAnyParsedField" class="caption">No parsed fields yet. (Some PNGs only include provenance.)</div>
              <dl v-else class="pnginfo-kv">
                <template v-if="parsed.prompt.trim()">
                  <dt>Prompt</dt>
                  <dd class="pnginfo-kv-pre">{{ parsed.prompt }}</dd>
                </template>
                <template v-if="parsed.hasNegativePrompt">
                  <dt>Negative</dt>
                  <dd class="pnginfo-kv-pre">{{ parsed.negativePrompt }}</dd>
                </template>
                <template v-if="parsed.width && parsed.height">
                  <dt>Size</dt>
                  <dd>{{ parsed.width }}×{{ parsed.height }}</dd>
                </template>
                <template v-else-if="analysis && analysis.width && analysis.height">
                  <dt>Image</dt>
                  <dd>{{ analysis.width }}×{{ analysis.height }}</dd>
                </template>
                <template v-if="parsed.steps !== undefined">
                  <dt>Steps</dt>
                  <dd>{{ parsed.steps }}</dd>
                </template>
                <template v-if="parsed.cfgScale !== undefined">
                  <dt>CFG</dt>
                  <dd>{{ parsed.cfgScale }}</dd>
                </template>
                <template v-if="parsed.seed !== undefined">
                  <dt>Seed</dt>
                  <dd>{{ parsed.seed }}</dd>
                </template>
                <template v-if="mappedCheckpoint">
                  <dt>Checkpoint</dt>
                  <dd>{{ mappedCheckpoint }}</dd>
                </template>
                <template v-else-if="parsed.model || parsed.modelHash">
                  <dt>Checkpoint</dt>
                  <dd class="caption">Not applied (unknown or ambiguous).</dd>
                </template>
                <template v-if="resolvedVaeLabel">
                  <dt>VAE</dt>
                  <dd>{{ resolvedVaeLabel }}</dd>
                </template>
                <template v-else-if="parsed.vae">
                  <dt>VAE</dt>
                  <dd class="caption">Not applied (unknown or ambiguous).</dd>
                </template>
                <template v-if="mappedSampler && mappedScheduler">
                  <dt>Sampler / Scheduler</dt>
                  <dd>{{ mappedSampler }} / {{ mappedScheduler }}</dd>
                </template>
                <template v-else-if="parsed.sampler || parsed.scheduler">
                  <dt>Sampler / Scheduler</dt>
                  <dd class="caption">Not applied (incompatible or unknown).</dd>
                </template>
                <template v-if="parsed.clipSkip !== undefined">
                  <dt>CLIP Skip</dt>
                  <dd>{{ parsed.clipSkip }}</dd>
                </template>
                <template v-if="parsed.denoiseStrength !== undefined">
                  <dt>Denoise</dt>
                  <dd>{{ parsed.denoiseStrength }}</dd>
                </template>
                <template v-if="parsed.rng">
                  <dt>RNG</dt>
                  <dd>{{ parsed.rng }}</dd>
                </template>
                <template v-if="parsed.eta !== undefined">
                  <dt>Eta</dt>
                  <dd>{{ parsed.eta }}</dd>
                </template>
                <template v-if="parsed.ngms !== undefined">
                  <dt>NGMS</dt>
                  <dd>{{ parsed.ngms }}</dd>
                </template>
                <template v-if="parsed.hiresModule1">
                  <dt>Hires module</dt>
                  <dd>{{ parsed.hiresModule1 }}</dd>
                </template>
                <template v-if="parsed.version">
                  <dt>Version</dt>
                  <dd>{{ parsed.version }}</dd>
                </template>
              </dl>
            </div>

            <div class="pnginfo-card">
              <div class="pnginfo-card-title">Raw metadata</div>
              <details class="accordion" :open="true">
                <summary>Text chunks</summary>
                <div class="accordion-body">
                  <div v-if="analysis && Object.keys(analysis.metadata || {}).length" class="pnginfo-metadata">
                    <JsonTreeView :value="analysis.metadata" :default-open-depth="1" :max-depth="8" />
                  </div>
                  <div v-else class="caption">No text chunks found.</div>
                </div>
              </details>

              <details v-if="comfyPromptGraph" class="accordion">
                <summary>ComfyUI prompt</summary>
                <div class="accordion-body">
                  <JsonTreeView :value="comfyPromptGraph" :default-open-depth="1" :max-depth="12" />
                </div>
              </details>

              <details v-if="comfyWorkflowGraph" class="accordion">
                <summary>ComfyUI workflow</summary>
                <div class="accordion-body">
                  <JsonTreeView :value="comfyWorkflowGraph" :default-open-depth="1" :max-depth="12" />
                </div>
              </details>
            </div>
          </div>
        </div>
      </ResultsCard>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { analyzePngInfo, fetchModelInventory, fetchSamplers, fetchSchedulers } from '../api/client'
import type { InventoryResponse, PngInfoAnalyzeResponse, SamplerInfo, SchedulerInfo } from '../api/types'
import { useResultsCard } from '../composables/useResultsCard'
import {
  filterSamplersForFamilyCapabilities,
  filterSchedulersForFamilyCapabilities,
  useEngineCapabilitiesStore,
} from '../stores/engine_capabilities'
import { useModelTabsStore } from '../stores/model_tabs'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useWorkflowsStore } from '../stores/workflows'
import { resolveImageRequestEngineId, type TabFamily } from '../utils/engine_taxonomy'
import { readFileAsDataURL } from '../utils/image_io'
import { buildUseInitImagePatch } from '../utils/image_params'
import { mapCheckpointTitle, mapSamplerScheduler, parseComfyPromptJson, parseInfotext, type ParsedInfotext } from '../utils/pnginfo'
import ResultsCard from '../components/results/ResultsCard.vue'
import Dropzone from '../components/ui/Dropzone.vue'
import JsonTreeView from '../components/ui/JsonTreeView.vue'

type TargetMode = 'txt2img' | 'img2img'

const tabs = useModelTabsStore()
const workflows = useWorkflowsStore()
const quicksettings = useQuicksettingsStore()
const engineCaps = useEngineCapabilitiesStore()
const { notice, toast } = useResultsCard()

const selectedFile = ref<File | null>(null)
const previewDataUrl = ref('')
const analysis = ref<PngInfoAnalyzeResponse | null>(null)
const infotext = ref('')
const error = ref('')
const analyzeRequestEpoch = ref(0)

const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])
const inventory = ref<InventoryResponse | null>(null)

const workflowBusy = ref(false)
const sendBusy = ref(false)
const lastSentTabId = ref<string>('')

const compatibleTabs = computed(() => tabs.orderedTabs.filter(t => t.type !== 'wan' && t.type !== 'anima' && t.type !== 'ltx2'))
const targetTabId = ref('')
const targetTab = computed(() => compatibleTabs.value.find(t => t.id === targetTabId.value) || null)
const targetMode = ref<TargetMode>('txt2img')
const targetTabFamily = computed<TabFamily | null>(() => {
  const type = targetTab.value?.type
  if (!type || type === 'wan' || type === 'anima' || type === 'ltx2') return null
  return type
})
const targetSamplingEngineId = computed(() => {
  const family = targetTabFamily.value
  if (!family) return null
  return resolveImageRequestEngineId(family, targetMode.value === 'img2img')
})
const targetFamilyCapabilities = computed(() => {
  const engineId = targetSamplingEngineId.value
  if (!engineId) return null
  return engineCaps.getFamilyForEngine(engineId)
})

const parsedResult = computed(() => parseInfotext(infotext.value))
const parsed = computed<ParsedInfotext>(() => parsedResult.value.parsed)
const parseWarnings = computed(() => parsedResult.value.warnings)

const checkpointResult = computed(() => mapCheckpointTitle(parsed.value, quicksettings.models))
const mappedCheckpoint = computed(() => checkpointResult.value.checkpoint || '')
const checkpointWarnings = computed(() => checkpointResult.value.warnings)

type VaeResolution = { label?: string; warnings: string[] }
const resolvedVae = computed<VaeResolution>(() => {
  const raw = String(parsed.value.vae || '').trim()
  if (!raw) return { warnings: [] }
  const tab = targetTab.value
  if (!tab) return { warnings: [`VAE '${raw}' found, but no target tab selected; leaving unchanged.`] }

  const sha = quicksettings.resolveVaeSha(raw)
  if (!sha) return { warnings: [`VAE '${raw}' not recognized; leaving unchanged.`] }

  const matches = (inventory.value?.vaes || []).filter((v) => String(v.sha256 || '').trim().toLowerCase() === sha.toLowerCase())
  if (matches.length === 0) return { warnings: [`VAE '${raw}' resolved but not present in inventory; leaving unchanged.`] }
  if (matches.length > 1) return { warnings: [`VAE '${raw}' is ambiguous (${matches.length} matches); leaving unchanged.`] }

  const entry = matches[0]
  const wantsPath = tab.type === 'flux1' || tab.type === 'flux2' || tab.type === 'chroma' || tab.type === 'zimage'
  const label = wantsPath ? String(entry.path || '').replace(/\\+/g, '/') : String(entry.name || '').trim()
  if (!label) return { warnings: [`VAE '${raw}' resolved, but label is empty; leaving unchanged.`] }
  return { label, warnings: [] }
})
const resolvedVaeLabel = computed(() => resolvedVae.value.label || '')
const vaeWarnings = computed(() => resolvedVae.value.warnings)

const samplingMapSamplers = computed<SamplerInfo[]>(() => {
  if (!targetTabFamily.value) return samplers.value
  if (!targetFamilyCapabilities.value) return []
  return filterSamplersForFamilyCapabilities(samplers.value, targetFamilyCapabilities.value)
})
const samplingMapSchedulers = computed<SchedulerInfo[]>(() => {
  if (!targetTabFamily.value) return schedulers.value
  if (!targetFamilyCapabilities.value) return []
  return filterSchedulersForFamilyCapabilities(schedulers.value, targetFamilyCapabilities.value)
})
const samplingMappingCapabilityWarning = computed(() => {
  if (!targetTabFamily.value) return ''
  if (targetFamilyCapabilities.value) return ''
  const engineId = targetSamplingEngineId.value
  if (!engineId) return ''
  return `Family sampling capabilities for '${engineId}' are unavailable; sampler/scheduler import is skipped.`
})
const mappingResult = computed(() => {
  if (targetTabFamily.value && !targetFamilyCapabilities.value) {
    return { warnings: [] as string[] }
  }
  return mapSamplerScheduler(parsed.value.sampler, parsed.value.scheduler, samplingMapSamplers.value, samplingMapSchedulers.value)
})
const mappedSampler = computed(() => mappingResult.value.sampler || '')
const mappedScheduler = computed(() => mappingResult.value.scheduler || '')
const mappingWarnings = computed(() => mappingResult.value.warnings)

const comfyPromptGraph = ref<Record<string, unknown> | null>(null)
const comfyWorkflowGraph = ref<Record<string, unknown> | null>(null)
const comfyWarnings = ref<string[]>([])
const initWarnings = ref<string[]>([])

const allWarnings = computed(() => [
  ...initWarnings.value,
  ...parseWarnings.value,
  ...checkpointWarnings.value,
  ...vaeWarnings.value,
  ...(samplingMappingCapabilityWarning.value ? [samplingMappingCapabilityWarning.value] : []),
  ...mappingWarnings.value,
  ...comfyWarnings.value,
])

const hasAnyParsedField = computed(() => {
  const p = parsed.value
  return Boolean(
    p.prompt.trim()
      || p.hasNegativePrompt
      || p.model
      || p.modelHash
      || p.vae
      || p.steps !== undefined
      || p.cfgScale !== undefined
      || p.seed !== undefined
      || p.width !== undefined
      || p.height !== undefined
      || p.sampler
      || p.scheduler
      || p.clipSkip !== undefined
      || p.denoiseStrength !== undefined
      || p.rng
      || p.eta !== undefined
      || p.ngms !== undefined
      || p.hiresModule1
      || p.version,
  )
})

const canSaveSnapshot = computed(() => Boolean(selectedFile.value && targetTab.value) && !workflowBusy.value)
const canSendTo = computed(() => {
  if (!selectedFile.value) return false
  if (!targetTab.value) return false
  if (sendBusy.value) return false
  if (targetMode.value === 'img2img' && !previewDataUrl.value) return false
  return true
})

function metadataValue(metadata: Record<string, string> | undefined | null, key: string): string {
  if (!metadata) return ''
  const direct = metadata[key]
  if (typeof direct === 'string' && direct.trim()) return direct
  const want = key.toLowerCase()
  for (const [k, v] of Object.entries(metadata)) {
    if (k.toLowerCase() === want && typeof v === 'string' && v.trim()) return v
  }
  return ''
}

async function analyzeSelectedFile(): Promise<void> {
  const file = selectedFile.value
  if (!file) return
  const requestEpoch = ++analyzeRequestEpoch.value
  error.value = ''
  analysis.value = null
  lastSentTabId.value = ''
  comfyPromptGraph.value = null
  comfyWorkflowGraph.value = null
  comfyWarnings.value = []

  try {
    const res = await analyzePngInfo(file)
    if (requestEpoch !== analyzeRequestEpoch.value) return
    if (selectedFile.value !== file) return
    analysis.value = res
    const params = metadataValue(res.metadata, 'parameters')
    if (params) {
      infotext.value = params
    } else {
      // ComfyUI often stores structured JSON under `prompt` and `workflow`.
      const rawPrompt = metadataValue(res.metadata, 'prompt')
      if (rawPrompt) {
        const out = parseComfyPromptJson(rawPrompt)
        comfyPromptGraph.value = out.graph
        if (out.warnings.length) comfyWarnings.value.push(...out.warnings)
        if (!infotext.value.trim() && out.extracted && Object.keys(out.extracted).length > 0) {
          const ex = out.extracted
          const lines: string[] = []
          const prompt = String(ex.prompt || '').trim()
          const neg = String(ex.negativePrompt || '').trim()
          if (prompt) lines.push(prompt)
          if (neg) lines.push(`Negative prompt: ${neg}`)
          const kv: string[] = []
          if (typeof ex.steps === 'number') kv.push(`Steps: ${ex.steps}`)
          if (typeof ex.sampler === 'string' && ex.sampler.trim()) kv.push(`Sampler: ${ex.sampler.trim()}`)
          if (typeof ex.scheduler === 'string' && ex.scheduler.trim()) kv.push(`Schedule type: ${ex.scheduler.trim()}`)
          if (typeof ex.cfgScale === 'number') kv.push(`CFG scale: ${ex.cfgScale}`)
          if (typeof ex.seed === 'number') kv.push(`Seed: ${ex.seed}`)
          if (typeof ex.width === 'number' && typeof ex.height === 'number') kv.push(`Size: ${ex.width}x${ex.height}`)
          if (typeof ex.denoiseStrength === 'number') kv.push(`Denoising strength: ${ex.denoiseStrength}`)
          if (kv.length) lines.push(kv.join(', '))
          if (lines.length) {
            infotext.value = lines.join('\n')
            comfyWarnings.value.push('Infotext was populated from ComfyUI prompt JSON; verify before sending.')
          }
        }
      }
      const rawWorkflow = metadataValue(res.metadata, 'workflow')
      if (rawWorkflow) {
        try {
          const value = JSON.parse(rawWorkflow) as unknown
          if (value && typeof value === 'object' && !Array.isArray(value)) {
            comfyWorkflowGraph.value = value as Record<string, unknown>
          } else {
            comfyWarnings.value.push("ComfyUI: 'workflow' JSON is not an object; skipping.")
          }
        } catch {
          comfyWarnings.value.push("ComfyUI: failed to parse 'workflow' JSON; skipping.")
        }
      }
      if (!infotext.value.trim()) infotext.value = ''
    }
  } catch (err) {
    if (requestEpoch !== analyzeRequestEpoch.value) return
    error.value = err instanceof Error ? err.message : String(err)
  }
}

async function onDropFiles(files: File[]): Promise<void> {
  const file = files[0]
  if (!file) return
  const requestEpoch = ++analyzeRequestEpoch.value

  selectedFile.value = file
  error.value = ''
  lastSentTabId.value = ''

  try {
    const dataUrl = await readFileAsDataURL(file)
    if (requestEpoch !== analyzeRequestEpoch.value) return
    if (selectedFile.value !== file) return
    previewDataUrl.value = dataUrl
  } catch (err) {
    if (requestEpoch !== analyzeRequestEpoch.value) return
    if (selectedFile.value !== file) return
    previewDataUrl.value = ''
    error.value = err instanceof Error ? err.message : String(err)
    return
  }

  if (requestEpoch !== analyzeRequestEpoch.value) return
  if (selectedFile.value !== file) return
  await analyzeSelectedFile()
}

function onDropRejected(payload: { reason: string; files: File[] }): void {
  const list = payload.files.map(f => f.name).join(', ')
  error.value = list ? `${payload.reason} (${list})` : payload.reason
}

function clearLoadedPng(): void {
  analyzeRequestEpoch.value += 1
  selectedFile.value = null
  previewDataUrl.value = ''
  analysis.value = null
  infotext.value = ''
  error.value = ''
  lastSentTabId.value = ''
  comfyPromptGraph.value = null
  comfyWorkflowGraph.value = null
  comfyWarnings.value = []
}

function buildImageParamsPatch(options: { mode: TargetMode; includeInitImage: boolean }): { patch: Record<string, unknown>; warnings: string[] } {
  const p = parsed.value
  const patch: Record<string, unknown> = {}

  if (mappedCheckpoint.value) patch.checkpoint = mappedCheckpoint.value
  if (p.prompt.trim()) patch.prompt = p.prompt
  if (p.hasNegativePrompt) patch.negativePrompt = p.negativePrompt

  const width = p.width ?? analysis.value?.width
  const height = p.height ?? analysis.value?.height
  if (Number.isFinite(width) && Number.isFinite(height) && Number(width) > 0 && Number(height) > 0) {
    patch.width = Math.trunc(Number(width))
    patch.height = Math.trunc(Number(height))
  }

  if (p.steps !== undefined) patch.steps = p.steps
  if (p.cfgScale !== undefined) patch.cfgScale = p.cfgScale
  if (p.seed !== undefined) patch.seed = p.seed
  if (p.clipSkip !== undefined) patch.clipSkip = p.clipSkip
  if (p.denoiseStrength !== undefined) patch.denoiseStrength = p.denoiseStrength

  const canApplySamplingPatch = !targetTabFamily.value || Boolean(targetFamilyCapabilities.value)
  if (canApplySamplingPatch) {
    if (mappingResult.value.sampler) patch.sampler = mappingResult.value.sampler
    if (mappingResult.value.scheduler) patch.scheduler = mappingResult.value.scheduler
  }

  if (options.mode === 'txt2img') {
    Object.assign(patch, buildUseInitImagePatch(false))
  } else if (options.includeInitImage) {
    Object.assign(patch, buildUseInitImagePatch(true))
    patch.initSource = {
      ...(targetTab.value?.params.initSource ?? {
        mode: 'img',
        folderPath: '',
        selectionMode: 'all',
        count: 1,
        order: 'random',
        sortBy: 'name',
        useCrop: false,
      }),
      mode: 'img',
    }
    patch.initImageData = previewDataUrl.value
    patch.initImageName = selectedFile.value?.name || ''
  } else {
    Object.assign(patch, buildUseInitImagePatch(false))
  }

  return { patch, warnings: allWarnings.value }
}

async function maybeApplyVae(): Promise<{ appliedLabel?: string; error?: string }> {
  const label = resolvedVaeLabel.value
  if (!label) return {}
  const family = targetTabFamily.value
  if (!family) return { error: 'Selected target tab does not support family-owned VAE apply.' }
  try {
    await quicksettings.setVaeForFamily(family, label)
    return { appliedLabel: label }
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) }
  }
}

async function saveSnapshot(): Promise<void> {
  if (!targetTab.value) return
  if (!selectedFile.value) return

  workflowBusy.value = true
  try {
    const snapshotMode = targetMode.value
    const { patch } = buildImageParamsPatch({
      mode: snapshotMode,
      includeInitImage: snapshotMode === 'img2img',
    })
    const result = await workflows.saveSnapshot({
      name: `${selectedFile.value.name} — ${new Date().toLocaleString()}`,
      source_tab_id: targetTab.value.id,
      type: targetTab.value.type,
      engine_semantics: targetTab.value.type === 'wan' ? 'wan22' : targetTab.value.type,
      params_snapshot: patch,
    })
    const vae = await maybeApplyVae()
    if (vae.error) {
      toast(`${result.action === 'updated' ? 'Snapshot updated' : 'Snapshot saved'}. VAE not applied: ${vae.error}`)
    } else if (vae.appliedLabel) {
      toast(`${result.action === 'updated' ? 'Snapshot updated' : 'Snapshot saved'}. VAE: ${vae.appliedLabel}`)
    } else {
      toast(result.action === 'updated' ? 'Snapshot updated in Workflows.' : 'Snapshot saved to Workflows.')
    }
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err))
  } finally {
    workflowBusy.value = false
  }
}

async function sendTo(): Promise<void> {
  if (!targetTab.value) return
  if (!selectedFile.value) return

  sendBusy.value = true
  try {
    await tabs.load()
    const { patch } = buildImageParamsPatch({ mode: targetMode.value, includeInitImage: true })
    await tabs.updateParams(targetTab.value.id, patch)
    lastSentTabId.value = targetTab.value.id
    const vae = await maybeApplyVae()
    if (vae.error) {
      toast(`Sent to ${targetTab.value.title}. VAE not applied: ${vae.error}`)
    } else if (vae.appliedLabel) {
      toast(`Sent to ${targetTab.value.title}. VAE: ${vae.appliedLabel}`)
    } else {
      toast(`Sent to ${targetTab.value.title}.`)
    }
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err))
  } finally {
    sendBusy.value = false
  }
}

onMounted(async () => {
  initWarnings.value = []

  try { await tabs.load() } catch {}
  try { await quicksettings.init() } catch (err) {
    initWarnings.value.push(`QuickSettings: failed to initialize (${err instanceof Error ? err.message : String(err)}).`)
  }
  try { await engineCaps.init() } catch (err) {
    initWarnings.value.push(`Capabilities: failed to initialize (${err instanceof Error ? err.message : String(err)}).`)
  }
  try { inventory.value = await fetchModelInventory() } catch (err) {
    initWarnings.value.push(`Inventory: failed to load (${err instanceof Error ? err.message : String(err)}).`)
  }

  try {
    const [samp, sched] = await Promise.all([fetchSamplers(), fetchSchedulers()])
    samplers.value = samp.samplers
    schedulers.value = sched.schedulers
  } catch (err) {
    initWarnings.value.push(`Sampling: failed to load (${err instanceof Error ? err.message : String(err)}).`)
  }
})

watch([compatibleTabs, () => tabs.activeTab], () => {
  if (targetTabId.value && compatibleTabs.value.some(t => t.id === targetTabId.value)) return
  const active = tabs.activeTab
  if (active && compatibleTabs.value.some((tab) => tab.id === active.id)) {
    targetTabId.value = active.id
    return
  }
  targetTabId.value = compatibleTabs.value[0]?.id ?? ''
}, { immediate: true })
</script>
