<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tools view (GGUF converter + Safetensors merger + file browser modal).
Starts GGUF conversion jobs (`/api/tools/convert-gguf`), starts Safetensors merge jobs
(`POST /api/tools/merge-safetensors`), polls job status, and provides a modal file browser
to pick source/weights/output paths without manual typing.

Symbols (top-level; keep in sync; no ghosts):
- `ToolsTab` (component): Tools page SFC; owns GGUF + merger form state and the shared file browser modal.
- `GGUFConverterModelComponent` (interface): Convertible component entry (config dir + unified profile id).
- `GGUFConverterModelMetadata` (interface): Vendored model metadata entry returned by `/api/tools/gguf-converter/presets`.
- `QuantizationSurfaceDescriptor` (interface): Backend-owned recipe/policy support descriptor for one GGUF component.
- `QuantizationRecipeOption` (interface): Backend-owned public GGUF recipe option descriptor.
- `QuantizationSuffix` (type): K-quant suffix switch discriminator (`L|M|S`).
- `QuantizationBaseOption` (interface): Compact UI recipe base option derived from backend recipe descriptors.
- `QuantizationSuffixOption` (interface): Single suffix switch state for the selected compact recipe base.
- `GGUFForm` (interface): GGUF converter form state (model metadata + component + recipe/policy + overwrite).
- `SafetensorsMergeForm` (interface): Safetensors merge form state (source path + output path + overwrite).
- `ToolJobStatus` (interface): Polled tools job status payload (status + progress + current tensor + error).
- `BrowserItem` (interface): Single file browser entry (file/directory + optional size).
- `BrowserData` (interface): File browser listing payload (current path + items).
- `BrowserMode` (type): Active file browser mode across GGUF + merger flows.
- `formatComponentLabel` (function): Formats component options in the selector.
- `loadModelMetadata` (function): Loads vendored model metadata for the selector.
- `startConversion` (function): Starts a conversion job and begins polling.
- `cancelConversion` (function): Requests cancellation of the current conversion job (cooperative).
- `pollStatus` (function): Polls job status and stops polling when complete/error/cancelled.
- `startMerge` (function): Starts a Safetensors merge job and begins polling.
- `pollMergeStatus` (function): Polls merge job status and stops polling when complete/error/cancelled.
- `browseForSafetensors` (function): Opens the file browser in GGUF safetensors selection mode.
- `browseForOutputDir` (function): Opens the file browser in GGUF output-folder selection mode.
- `browseForMergeSource` (function): Opens the file browser in merger source selection mode.
- `browseForMergeOutputDir` (function): Opens the file browser in merger output-folder selection mode.
- `openFileBrowser` (function): Opens the modal and loads the current path listing.
- `closeFileBrowser` (function): Closes the modal.
- `loadBrowserPath` (function): Fetches the directory listing for the current browser path.
- `goToParent` (function): Navigates the browser up one directory.
- `selectItem` (function): Selects a browser row item.
- `openItem` (function): Opens a directory (or confirms selection for files).
- `confirmSelection` (function): Applies the selected path to the active form field and closes the modal.
- `formatSize` (function): Formats byte sizes for display.
- `quantizationBaseGroups` (computed): Compact recipe dropdown groups, collapsing K suffix recipes by base.
- `quantizationSuffixOptions` (computed): L/M/S switch availability and selected state for the current recipe base.
- `selectQuantizationBaseFromEvent` (function): Applies a compact dropdown base change to the final recipe id.
- `selectQuantizationSuffix` (function): Applies an L/M/S suffix switch to the final recipe id.
- `policyPresetSupported` (computed): Whether the selected quantization uses profile policy presets.
- `outputFileName` (computed): Generated output filename derived from the safetensors path (base `.gguf`).
- `outputFullPath` (computed): Output full path (folder + generated filename).
- `mergeOutputFileName` (computed): Deterministic merge output filename derived from merge source path (`-merged.safetensors`).
-->

<template>
  <section class="panel-stack cdx-tools">
    <div class="panel">
      <div class="panel-header">Tools</div>
      <div class="panel-body">
        <p class="subtitle">Utilities for model conversion and management</p>

        <div class="gen-card">
          <div>
            <div class="h3">GGUF Converter</div>
            <p class="caption">Convert Safetensors weights to GGUF format</p>
          </div>

          <div class="field">
            <label class="label-muted">Model Metadata (vendored Hugging Face)</label>
            <select class="select-md" v-model="ggufForm.modelId" :disabled="isConverting || metadataLoading">
              <option value="" disabled>Select a vendored model…</option>
              <option v-for="m in modelMetadata" :key="m.id" :value="m.id">{{ m.label }}</option>
            </select>
            <p class="caption">
              Uses the vendored Hugging Face mirror under <code>apps/backend/huggingface/**</code>.
            </p>
            <p v-if="metadataLoading" class="caption">Loading vendored model metadata…</p>
            <p v-if="metadataError" class="cdx-tools-error">{{ metadataError }}</p>
          </div>

          <div v-if="selectedModel" class="field">
            <label class="label-muted">Component</label>
            <select class="select-md" v-model="ggufForm.componentId" :disabled="isConverting">
              <option v-for="c in selectedModel.components" :key="c.id" :value="c.id">{{ formatComponentLabel(c) }}</option>
            </select>
            <p class="caption">Uses the vendored config directory for the selected model.</p>
          </div>

          <div class="field">
            <label class="label-muted">Safetensors File or Folder</label>
            <div class="row-inline">
              <input class="ui-input cdx-tools-grow" type="text" v-model="ggufForm.safetensorsPath" placeholder="Path to .safetensors file, .safetensors.index.json, or folder" :disabled="isConverting" />
              <button class="btn-icon" type="button" @click="browseForSafetensors" :disabled="isConverting" aria-label="Browse for safetensors file">…</button>
            </div>
            <p class="caption">For sharded weights, select the folder that contains <code>*.safetensors.index.json</code>.</p>
          </div>

          <div class="field">
            <label class="label-muted">Quantization</label>
            <div class="row-inline cdx-tools-quant-row">
              <select
                class="select-md cdx-tools-grow"
                :value="selectedQuantizationBaseId"
                :disabled="isConverting || quantizationBaseGroups.length === 0"
                @change="selectQuantizationBaseFromEvent"
              >
                <optgroup v-for="group in quantizationBaseGroups" :key="group.label" :label="group.label">
                  <option v-for="option in group.options" :key="option.id" :value="option.id">
                    {{ option.label }} — {{ option.description }}
                  </option>
                </optgroup>
              </select>
              <div v-if="hasQuantizationSuffixSwitches" class="cdx-tools-quant-suffixes" role="radiogroup" aria-label="K quant suffix">
                <button
                  v-for="suffix in quantizationSuffixOptions"
                  :key="suffix.id"
                  :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', suffix.selected ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                  type="button"
                  role="radio"
                  :aria-checked="suffix.selected ? 'true' : 'false'"
                  :disabled="isConverting || !suffix.available"
                  :title="suffix.available ? suffix.recipeLabel : `${suffix.id} is not available for ${selectedQuantizationBaseId}`"
                  @click="selectQuantizationSuffix(suffix.id)"
                >
                  {{ suffix.id }}
                </button>
              </div>
            </div>
            <p class="caption">
              Selected recipe: <code>{{ selectedQuantizationRecipeLabel }}</code>; profile support is backend-owned per selected component.
            </p>
          </div>

          <div v-if="policyPresetSupported" class="field">
            <label class="label-muted">Profile Policy</label>
            <select class="select-md" v-model="ggufForm.quantPolicyPreset" :disabled="isConverting">
              <option v-for="preset in profilePolicyOptions" :key="preset.id" :value="preset.id">{{ preset.label }}</option>
            </select>
            <p class="caption">
              Controls which optional profile tensors are preserved or precision-bumped. Preserved tensors keep their source dtype.
            </p>
          </div>

          <div class="field">
            <label class="label-muted">Output Folder</label>
            <div class="row-inline">
              <input
                class="ui-input cdx-tools-grow"
                type="text"
                v-model="ggufForm.outputDir"
                placeholder="Output folder path"
                :disabled="isConverting"
              />
              <button class="btn-icon" type="button" @click="browseForOutputDir" :disabled="isConverting" aria-label="Browse for output folder">…</button>
            </div>
            <p class="caption">Output file name is generated automatically: <code>{{ outputFileName }}</code></p>
            <div class="row-inline cdx-tools-actions">
              <button
                :class="['btn', 'qs-toggle-btn', ggufForm.overwrite ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="ggufForm.overwrite"
                :disabled="isConverting"
                title="Allow overwriting the output file if it already exists"
                @click="ggufForm.overwrite = !ggufForm.overwrite"
              >
                Overwrite
              </button>
            </div>
            <p class="caption">Overwrite: when off, conversion fails if the output file already exists.</p>
          </div>

	          <div class="row-inline cdx-tools-actions">
	            <button class="btn btn-md btn-primary" type="button" @click="startConversion" :disabled="!canConvert || isConverting">
	              <span v-if="!isConverting">Convert to GGUF</span>
	              <span v-else>Converting…</span>
	            </button>
            <button
              v-if="isConverting && currentJobId"
              class="btn btn-md btn-secondary"
              type="button"
              :disabled="conversionStatus?.status === 'cancelling'"
              @click="cancelConversion"
            >
              Cancel
            </button>
          </div>

          <div v-if="conversionStatus" class="panel-progress">
            <div class="cdx-tools-progress-head">
              <span class="cdx-tools-status" :data-status="conversionStatus.status">{{ conversionStatus.status }}</span>
              <span v-if="conversionStatus.current_tensor" class="cdx-tools-current-tensor">{{ conversionStatus.current_tensor }}</span>
            </div>
            <progress class="cdx-tools-progress" :value="conversionStatus.progress" max="100"></progress>
            <div class="caption">{{ Math.round(conversionStatus.progress) }}%</div>
            <div v-if="conversionStatus.error" class="cdx-tools-error">{{ conversionStatus.error }}</div>
          </div>
        </div>

        <div class="gen-card">
          <div>
            <div class="h3">Safetensors Merger</div>
            <p class="caption">Merge Safetensors file, sharded index, or folder inputs into one output file</p>
          </div>

          <div class="field">
            <label class="label-muted">Source Path (file/sharded-index/folder)</label>
            <div class="row-inline">
              <input
                class="ui-input cdx-tools-grow"
                type="text"
                v-model="mergeForm.sourcePath"
                placeholder="Path to .safetensors file, .safetensors.index.json, or folder"
                :disabled="isMerging"
              />
              <button class="btn-icon" type="button" @click="browseForMergeSource" :disabled="isMerging" aria-label="Browse for merge source">…</button>
            </div>
            <p class="caption">Supports <code>.safetensors</code>, <code>.safetensors.index.json</code>, or a folder.</p>
          </div>

          <div class="field">
            <label class="label-muted">Output Path</label>
            <div class="row-inline">
              <input
                class="ui-input cdx-tools-grow"
                type="text"
                v-model="mergeForm.outputPath"
                placeholder="Path to merged .safetensors output"
                :disabled="isMerging"
              />
              <button class="btn-icon" type="button" @click="browseForMergeOutputDir" :disabled="isMerging" aria-label="Browse for merge output directory">…</button>
            </div>
            <p class="caption">Browse selects a directory and fills <code>{{ mergeOutputFileName }}</code>.</p>
            <div class="row-inline cdx-tools-actions">
              <button
                :class="['btn', 'qs-toggle-btn', mergeForm.overwrite ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="mergeForm.overwrite"
                :disabled="isMerging"
                title="Allow overwriting the output file if it already exists"
                @click="mergeForm.overwrite = !mergeForm.overwrite"
              >
                Overwrite
              </button>
            </div>
            <p class="caption">Overwrite: when off, merge fails if the output path already exists.</p>
          </div>

          <div class="row-inline cdx-tools-actions">
            <button class="btn btn-md btn-primary" type="button" @click="startMerge" :disabled="!canMerge || isMerging">
              <span v-if="!isMerging">Merge Safetensors</span>
              <span v-else>Merging…</span>
            </button>
          </div>

          <div v-if="mergeStatus" class="panel-progress">
            <div class="cdx-tools-progress-head">
              <span class="cdx-tools-status" :data-status="mergeStatus.status">{{ mergeStatus.status }}</span>
              <span v-if="mergeStatus.current_tensor" class="cdx-tools-current-tensor">{{ mergeStatus.current_tensor }}</span>
            </div>
            <progress class="cdx-tools-progress" :value="mergeStatus.progress" max="100"></progress>
            <div class="caption">{{ Math.round(mergeStatus.progress) }}%</div>
            <div v-if="mergeStatus.error" class="cdx-tools-error">{{ mergeStatus.error }}</div>
          </div>
        </div>
      </div>
    </div>

    <Modal v-model="showFileBrowser" :title="browserTitle">
      <div class="cdx-tools-pathbar">
        <button class="btn btn-sm btn-secondary" type="button" @click="goToParent" :disabled="!browserData.parent">Up</button>
        <input class="ui-input cdx-tools-grow" type="text" v-model="browserPath" @keyup.enter="loadBrowserPath" />
        <button class="btn btn-sm btn-secondary" type="button" @click="loadBrowserPath">Go</button>
      </div>
      <div class="cdx-tools-file-list">
        <div
          v-for="item in browserItems"
          :key="item.name"
          class="cdx-tools-file-item"
          :class="{ 'is-selected': selectedItem && selectedItem.name === item.name && selectedItem.type === item.type }"
          :data-type="item.type"
          @click="selectItem(item)"
          @dblclick="openItem(item)"
        >
          <span aria-hidden="true">{{ item.type === 'directory' ? '📁' : '📄' }}</span>
          <span class="cdx-tools-file-name">{{ item.name }}</span>
          <span v-if="item.size" class="cdx-tools-file-size">{{ formatSize(item.size) }}</span>
        </div>
      </div>

      <template #footer>
        <button class="btn btn-md btn-secondary" type="button" @click="closeFileBrowser">Cancel</button>
        <button
          class="btn btn-md btn-primary"
          type="button"
          @click="confirmSelection"
          :disabled="browserRequiresSelection && !selectedItem"
        >
          Select
        </button>
      </template>
    </Modal>
  </section>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import Modal from '../components/ui/Modal.vue'

interface QuantizationSurfaceDescriptor {
  profile_id: string
  supported_recipes: string[]
  default_recipe: string
  policy_presets_by_recipe: Record<string, string[]>
  default_policy_preset_by_recipe: Record<string, string | null>
}

interface QuantizationRecipeOption {
  id: string
  label: string
  group: string
  description: string
  default_tensor_type: string
  file_type: string
  family: string
  tier: string
}

type QuantizationSuffix = 'L' | 'M' | 'S'

interface QuantizationBaseOption {
  id: string
  label: string
  group: string
  description: string
  representativeRecipe: QuantizationRecipeOption
  suffixRecipes: Partial<Record<QuantizationSuffix, QuantizationRecipeOption>>
}

interface QuantizationSuffixOption {
  id: QuantizationSuffix
  available: boolean
  selected: boolean
  recipeLabel: string
}

interface GGUFConverterModelComponent {
  id: string
  label: string
  config_dir: string
  kind: string
  profile_id: string | null
  quantization: QuantizationSurfaceDescriptor | null
}

interface GGUFConverterModelMetadata {
  id: string
  label: string
  org: string
  repo: string
  components: GGUFConverterModelComponent[]
}

interface GGUFForm {
  modelId: string
  componentId: string
  safetensorsPath: string
  quantization: string
  quantPolicyPreset: 'HQ' | 'MQ' | 'LQ' | ''
  outputDir: string
  overwrite: boolean
}

interface SafetensorsMergeForm {
  sourcePath: string
  outputPath: string
  overwrite: boolean
}

interface ToolJobStatus {
  status: string
  progress: number
  current_tensor: string
  error: string | null
}

interface BrowserItem {
  name: string
  type: 'file' | 'directory'
  size?: number
}

interface BrowserData {
  path: string
  exists: boolean
  parent: string
  items: BrowserItem[]
}

type BrowserMode = 'gguf_safetensors' | 'gguf_output_dir' | 'merge_source' | 'merge_output_dir'

const modelMetadata = ref<GGUFConverterModelMetadata[]>([])
const quantizationRecipes = ref<QuantizationRecipeOption[]>([])
const metadataLoading = ref(false)
const metadataError = ref<string | null>(null)
const QUANTIZATION_SUFFIXES: QuantizationSuffix[] = ['L', 'M', 'S']

const ggufForm = ref<GGUFForm>({
  modelId: '',
  componentId: '',
  safetensorsPath: '',
  quantization: 'Q4_K_M',
  quantPolicyPreset: 'MQ',
  outputDir: '',
  overwrite: false,
})

const mergeForm = ref<SafetensorsMergeForm>({
  sourcePath: '',
  outputPath: '',
  overwrite: false,
})

const conversionStatus = ref<ToolJobStatus | null>(null)
const currentJobId = ref<string | null>(null)
const pollInterval = ref<number | null>(null)
const mergeStatus = ref<ToolJobStatus | null>(null)
const currentMergeJobId = ref<string | null>(null)
const mergePollInterval = ref<number | null>(null)

// File browser
const showFileBrowser = ref(false)
const browserPath = ref('')
const browserData = ref<BrowserData>({ path: '', exists: false, parent: '', items: [] })
const browserMode = ref<BrowserMode>('gguf_safetensors')
const selectedItem = ref<BrowserItem | null>(null)

const selectedModel = computed(() => modelMetadata.value.find((m) => m.id === ggufForm.value.modelId) ?? null)
const selectedComponent = computed(() => {
  const model = selectedModel.value
  if (!model) return null
  return model.components.find((c) => c.id === ggufForm.value.componentId) ?? null
})

const effectiveProfileId = computed(() => selectedComponent.value?.profile_id ?? null)
const selectedQuantizationSurface = computed(() => selectedComponent.value?.quantization ?? null)

const supportedQuantizationRecipes = computed(() => {
  const surface = selectedQuantizationSurface.value
  if (!surface) return []
  const supported = new Set(surface.supported_recipes || [])
  return quantizationRecipes.value.filter((recipe) => supported.has(recipe.id))
})

const quantizationBaseOptions = computed(() => {
  const options: QuantizationBaseOption[] = []
  const byId = new Map<string, QuantizationBaseOption>()
  for (const recipe of supportedQuantizationRecipes.value) {
    const parts = _quantizationRecipeParts(recipe.id)
    const optionId = parts.suffix ? parts.baseId : recipe.id
    let option = byId.get(optionId)
    if (!option) {
      option = {
        id: optionId,
        label: parts.suffix ? optionId : recipe.label,
        group: recipe.group,
        description: parts.suffix ? 'Select L/M/S with the suffix buttons.' : recipe.description,
        representativeRecipe: recipe,
        suffixRecipes: {},
      }
      byId.set(optionId, option)
      options.push(option)
    }
    if (parts.suffix) {
      option.suffixRecipes[parts.suffix] = recipe
      if (parts.suffix === 'M') {
        option.representativeRecipe = recipe
      }
    }
  }
  return options
})

const quantizationBaseGroups = computed(() => {
  const groups: { label: string; options: QuantizationBaseOption[] }[] = []
  for (const option of quantizationBaseOptions.value) {
    let group = groups.find((entry) => entry.label === option.group)
    if (!group) {
      group = { label: option.group, options: [] }
      groups.push(group)
    }
    group.options.push(option)
  }
  return groups
})

const selectedQuantizationParts = computed(() => _quantizationRecipeParts(ggufForm.value.quantization))
const selectedQuantizationBaseId = computed(() => selectedQuantizationParts.value.baseId)
const selectedQuantizationBaseOption = computed(() => {
  const baseId = selectedQuantizationBaseId.value
  return quantizationBaseOptions.value.find((option) => option.id === baseId) ?? null
})
const selectedQuantizationRecipeOption = computed(
  () => supportedQuantizationRecipes.value.find((recipe) => recipe.id === ggufForm.value.quantization) ?? null,
)
const selectedQuantizationRecipeLabel = computed(() => {
  const recipe = selectedQuantizationRecipeOption.value
  if (!recipe) return String(ggufForm.value.quantization || 'F16').trim() || 'F16'
  return _formatQuantizationRecipeLabel(recipe)
})
const hasQuantizationSuffixSwitches = computed(() => {
  const option = selectedQuantizationBaseOption.value
  return Boolean(option && Object.keys(option.suffixRecipes).length > 0)
})
const quantizationSuffixOptions = computed<QuantizationSuffixOption[]>(() => {
  const option = selectedQuantizationBaseOption.value
  const selectedSuffix = selectedQuantizationParts.value.suffix
  return QUANTIZATION_SUFFIXES.map((suffix) => {
    const recipe = option?.suffixRecipes[suffix]
    return {
      id: suffix,
      available: Boolean(recipe),
      selected: selectedSuffix === suffix && Boolean(recipe),
      recipeLabel: recipe ? _formatQuantizationRecipeLabel(recipe) : `${suffix} unavailable`,
    }
  })
})

const selectedRecipePolicyPresets = computed(() => {
  const surface = selectedQuantizationSurface.value
  const recipe = String(ggufForm.value.quantization || '').trim()
  if (!surface || !recipe) return []
  const values = surface.policy_presets_by_recipe?.[recipe]
  return Array.isArray(values) ? values : []
})

const policyPresetSupported = computed(() => selectedRecipePolicyPresets.value.length > 0)

const profilePolicyOptions = computed(() => {
  const labels: Record<string, string> = {
    HQ: 'HQ — conservative, larger files',
    MQ: 'MQ — balanced default',
    LQ: 'LQ — compact, more aggressive',
  }
  return selectedRecipePolicyPresets.value.map((id) => ({ id, label: labels[id] ?? id }))
})

function _quantizationRecipeParts(recipeId: string): { baseId: string; suffix: QuantizationSuffix | null } {
  const raw = String(recipeId || '').trim()
  const match = raw.match(/^(Q[2-5]_K)_(L|M|S)$/)
  if (match) {
    return { baseId: match[1], suffix: match[2] as QuantizationSuffix }
  }
  if (raw === 'Q2_K') {
    return { baseId: 'Q2_K', suffix: 'M' }
  }
  return { baseId: raw, suffix: null }
}

function _preferredRecipeForBaseOption(option: QuantizationBaseOption, desiredSuffix: QuantizationSuffix | null): string {
  if (Object.keys(option.suffixRecipes).length === 0) {
    return option.representativeRecipe.id
  }

  const preference: QuantizationSuffix[] = []
  if (desiredSuffix) preference.push(desiredSuffix)
  if (!preference.includes('M')) preference.push('M')
  for (const suffix of QUANTIZATION_SUFFIXES) {
    if (!preference.includes(suffix)) preference.push(suffix)
  }
  for (const suffix of preference) {
    const recipe = option.suffixRecipes[suffix]
    if (recipe) return recipe.id
  }
  return option.representativeRecipe.id
}

function _formatQuantizationRecipeLabel(recipe: QuantizationRecipeOption): string {
  const description = recipe.description.trim().replace(/[.!?]+$/, '')
  return description ? `${recipe.label} — ${description}` : recipe.label
}

function _fallbackQuantizationRecipe(surface: QuantizationSurfaceDescriptor): string {
  const supported = new Set(surface.supported_recipes || [])
  if (supported.has('Q4_K_M')) return 'Q4_K_M'

  const mediumRecipe = supportedQuantizationRecipes.value.find((recipe) => _quantizationRecipeParts(recipe.id).suffix === 'M')
  if (mediumRecipe) return mediumRecipe.id

  const backendDefault = String(surface.default_recipe || '').trim()
  if (backendDefault && supported.has(backendDefault)) return backendDefault

  return supportedQuantizationRecipes.value[0]?.id || 'F16'
}

function selectQuantizationBase(baseId: string): void {
  const option = quantizationBaseOptions.value.find((entry) => entry.id === String(baseId || '').trim())
  if (!option) return
  ggufForm.value.quantization = _preferredRecipeForBaseOption(option, selectedQuantizationParts.value.suffix)
}

function selectQuantizationBaseFromEvent(event: Event): void {
  const target = event.target as HTMLSelectElement | null
  selectQuantizationBase(target?.value || '')
}

function selectQuantizationSuffix(suffix: QuantizationSuffix): void {
  const option = selectedQuantizationBaseOption.value
  const recipe = option?.suffixRecipes[suffix]
  if (!recipe) return
  ggufForm.value.quantization = recipe.id
}

function _titleizeWords(raw: string): string {
  return String(raw || '')
    .trim()
    .split(/[_-]+/)
    .filter((part) => Boolean(part))
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function formatComponentLabel(component: GGUFConverterModelComponent): string {
  const kind = String(component.kind || '')
  const baseByKind: Record<string, string> = {
    flux_transformer: 'Denoiser',
    zimage_transformer: 'Denoiser',
    zimage_l2p_denoiser: 'Denoiser',
    wan22_transformer: 'Denoiser',
    ltx2_transformer: 'Denoiser',
    gemma3_tenc: 'Text Encoder',
    qwen_image_tenc: 'Text Encoder',
    zimage_l2p_tenc: 'Text Encoder',
  }
  const base = baseByKind[kind] ?? _titleizeWords(kind || 'Component')
  const label = _titleizeWords(component.label || '')
  const suffix = label && !['Root', 'Denoiser', 'Text Encoder', base].includes(label) ? ` (${label})` : ''
  return `${base}${suffix}`
}

const isConverting = computed(() => {
  if (!currentJobId.value) return false
  const status = conversionStatus.value?.status
  if (!status) return true
  return !['complete', 'error', 'cancelled'].includes(status)
})

const canConvert = computed(() => {
  const surface = selectedQuantizationSurface.value
  const recipe = String(ggufForm.value.quantization || '').trim()
  const recipeSupported = Boolean(surface && surface.supported_recipes.includes(recipe))
  return Boolean(selectedComponent.value && recipeSupported && ggufForm.value.safetensorsPath && ggufForm.value.outputDir)
})

const isMerging = computed(() => {
  if (!currentMergeJobId.value) return false
  const status = mergeStatus.value?.status
  if (!status) return true
  return !['complete', 'error', 'cancelled'].includes(status)
})

const canMerge = computed(() => {
  return Boolean(String(mergeForm.value.sourcePath || '').trim() && String(mergeForm.value.outputPath || '').trim())
})

const browserTitle = computed(() => {
  if (browserMode.value === 'gguf_safetensors') return 'Choose Weights'
  if (browserMode.value === 'gguf_output_dir') return 'Choose Output Folder'
  if (browserMode.value === 'merge_source') return 'Choose Safetensors Source'
  if (browserMode.value === 'merge_output_dir') return 'Choose Output Folder'
  return 'Browse Files'
})

const browserItems = computed(() => {
  if (browserMode.value === 'gguf_output_dir' || browserMode.value === 'merge_output_dir') {
    return browserData.value.items.filter((it) => it.type === 'directory')
  }
  return browserData.value.items
})

const browserRequiresSelection = computed(() => {
  return browserMode.value === 'gguf_safetensors' || browserMode.value === 'merge_source'
})

function _sanitizeOutputStem(raw: string): string {
  const s = String(raw || '').trim()
  if (!s) return 'model'
  // Keep stable/portable: collapse whitespace and remove weird separators.
  const cleaned = s.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '')
  return cleaned || 'model'
}

function _basename(path: string): string {
  const p = String(path || '').replace(/[\\/]+$/, '').replace(/\\/g, '/')
  const parts = p.split('/')
  return parts[parts.length - 1] || ''
}

function _joinPath(dir: string, file: string): string {
  const d = String(dir || '').trim()
  if (!d) return file
  const sep = d.includes('\\') && !d.includes('/') ? '\\' : '/'
  return d.replace(/[\\/]+$/, '') + sep + file
}

function _dirname(path: string): string {
  const p = String(path || '').trim().replace(/[\\/]+$/, '')
  if (!p) return ''
  const slashPos = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'))
  if (slashPos < 0) return ''
  return p.slice(0, slashPos)
}

function _deriveSourceStem(path: string): string {
  const raw = String(path || '').trim()
  if (!raw) return 'model'
  const name = _basename(raw)
  if (name.toLowerCase().endsWith('.safetensors.index.json')) {
    return name.slice(0, -'.safetensors.index.json'.length)
  }
  if (name.toLowerCase().endsWith('.safetensors')) {
    return name.slice(0, -'.safetensors'.length)
  }
  // If user picked a folder, use its leaf.
  return name
}

function _deriveOutputStem(): string {
  return _deriveSourceStem(ggufForm.value.safetensorsPath)
}

function _detailToMessage(detail: unknown): string | null {
  if (typeof detail === 'string' && detail.trim()) return detail.trim()
  if (!Array.isArray(detail)) return null
  const lines = detail
    .map((entry) => {
      if (typeof entry === 'string') return entry
      if (!entry || typeof entry !== 'object') return ''
      const obj = entry as Record<string, unknown>
      const msg = typeof obj.msg === 'string' ? obj.msg : ''
      const loc = Array.isArray(obj.loc) ? obj.loc.map((v) => String(v)).join('.') : ''
      if (!msg) return ''
      return loc ? `${loc}: ${msg}` : msg
    })
    .filter((s) => Boolean(String(s).trim()))
  if (lines.length === 0) return null
  return lines.join('; ')
}

function _httpErrorMessage(response: Response, data: unknown): string {
  const detail = _detailToMessage((data as any)?.detail)
  if (detail) return detail
  return `${response.status} ${response.statusText}`
}

function _normalizeToolJobStatus(data: any): ToolJobStatus {
  const rawProgress = Number(data?.progress)
  const progress = Number.isFinite(rawProgress) ? Math.max(0, Math.min(100, rawProgress)) : 0
  return {
    status: String(data?.status || 'unknown'),
    progress,
    current_tensor: String(data?.current_tensor || ''),
    error: data?.error ? String(data.error) : null,
  }
}

function syncQuantizationSelection() {
  const surface = selectedQuantizationSurface.value
  if (!surface) {
    ggufForm.value.quantPolicyPreset = ''
    return
  }

  const supported = new Set(surface.supported_recipes || [])
  const fallbackRecipe = _fallbackQuantizationRecipe(surface)
  if (!supported.has(String(ggufForm.value.quantization || '').trim())) {
    ggufForm.value.quantization = fallbackRecipe
  }

  const recipe = String(ggufForm.value.quantization || '').trim()
  const presets = Array.isArray(surface.policy_presets_by_recipe?.[recipe])
    ? surface.policy_presets_by_recipe[recipe]
    : []
  if (presets.length === 0) {
    ggufForm.value.quantPolicyPreset = ''
    return
  }

  const current = String(ggufForm.value.quantPolicyPreset || '').trim()
  if (!presets.includes(current)) {
    const defaultPreset = surface.default_policy_preset_by_recipe?.[recipe]
    ggufForm.value.quantPolicyPreset = (defaultPreset && presets.includes(defaultPreset) ? defaultPreset : presets[0]) as
      | 'HQ'
      | 'MQ'
      | 'LQ'
  }
}

const outputFileName = computed(() => {
  const stem = _sanitizeOutputStem(_deriveOutputStem())
  const quant = String(ggufForm.value.quantization || 'F16').trim() || 'F16'
  const policySuffix = policyPresetSupported.value && ggufForm.value.quantPolicyPreset ? `-${ggufForm.value.quantPolicyPreset}` : ''
  const base = `${stem}-${quant}${policySuffix}-Codex`
  return `${base}.gguf`
})

const outputFullPath = computed(() => _joinPath(ggufForm.value.outputDir, outputFileName.value))
const mergeOutputFileName = computed(() => `${_sanitizeOutputStem(_deriveSourceStem(mergeForm.value.sourcePath))}-merged.safetensors`)

async function loadModelMetadata() {
  metadataLoading.value = true
  try {
    const res = await fetch('/api/tools/gguf-converter/presets')
    const data = await res.json().catch(() => ({}))
    if (!res.ok) {
      throw new Error((data as any)?.detail || `${res.status} ${res.statusText}`)
    }

    const models = Array.isArray((data as any)?.models) ? ((data as any).models as GGUFConverterModelMetadata[]) : []
    const recipes = Array.isArray((data as any)?.quantization_recipes)
      ? ((data as any).quantization_recipes as QuantizationRecipeOption[])
      : []
    modelMetadata.value = models
    quantizationRecipes.value = recipes

    if (!ggufForm.value.modelId && models.length > 0) {
      ggufForm.value.modelId = models[0].id
      ggufForm.value.componentId = models[0].components[0]?.id || ''
    }
    syncQuantizationSelection()

    metadataError.value = null
  } catch (e: any) {
    modelMetadata.value = []
    quantizationRecipes.value = []
    metadataError.value = String(e?.message || e)
  } finally {
    metadataLoading.value = false
  }
}

watch(
  () => ggufForm.value.modelId,
  () => {
    const model = selectedModel.value
    if (!model) {
      ggufForm.value.componentId = ''
      return
    }
    if (!model.components.find((c) => c.id === ggufForm.value.componentId)) {
      ggufForm.value.componentId = model.components[0]?.id || ''
    }
    syncQuantizationSelection()
  },
)

watch(
  () => ggufForm.value.componentId,
  () => {
    syncQuantizationSelection()
  },
)

watch(
  () => ggufForm.value.quantization,
  () => {
    syncQuantizationSelection()
  },
)

async function startConversion() {
  try {
    const component = selectedComponent.value
    if (!component) {
      throw new Error('Select a vendored model component first.')
    }
    const payload: Record<string, any> = {
      config_path: component.config_dir,
      safetensors_path: ggufForm.value.safetensorsPath,
      output_path: outputFullPath.value,
      overwrite: ggufForm.value.overwrite,
      quantization: ggufForm.value.quantization,
    }
    if (policyPresetSupported.value) {
      payload.quant_policy_preset = ggufForm.value.quantPolicyPreset
    }

    const profileId = effectiveProfileId.value
    if (profileId) {
      payload.profile_id = profileId
    }

    const response = await fetch('/api/tools/convert-gguf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    const data = await response.json().catch(() => ({}))

    if (!response.ok) {
      conversionStatus.value = {
        status: 'error',
        progress: 0,
        current_tensor: '',
        error: _httpErrorMessage(response, data),
      }
      return
    }

    const jobId = String((data as any)?.job_id || '').trim()
    if (!jobId) {
      throw new Error('convert-gguf response missing job_id.')
    }

    currentJobId.value = jobId
    conversionStatus.value = { status: 'pending', progress: 0, current_tensor: '', error: null }

    if (pollInterval.value) {
      clearInterval(pollInterval.value)
      pollInterval.value = null
    }
    pollInterval.value = window.setInterval(pollStatus, 500)
  } catch (e: any) {
    conversionStatus.value = {
      status: 'error',
      progress: 0,
      current_tensor: '',
      error: String(e?.message || e),
    }
  }
}

async function cancelConversion() {
  if (!currentJobId.value) return
  try {
    const res = await fetch(`/api/tools/convert-gguf/${currentJobId.value}/cancel`, { method: 'POST' })
    if (!res.ok) {
      const data = await res.json().catch(() => ({}))
      throw new Error(_httpErrorMessage(res, data))
    }
    if (conversionStatus.value) {
      conversionStatus.value = { ...conversionStatus.value, status: 'cancelling' }
    }
  } catch (e: any) {
    if (conversionStatus.value) {
      conversionStatus.value = { ...conversionStatus.value, error: String(e?.message || e) }
    } else {
      conversionStatus.value = { status: 'error', progress: 0, current_tensor: '', error: String(e?.message || e) }
    }
  }
}

function _stopConversionPolling() {
  if (pollInterval.value) {
    clearInterval(pollInterval.value)
    pollInterval.value = null
  }
}

async function pollStatus() {
  if (!currentJobId.value) return

  try {
    const response = await fetch(`/api/tools/convert-gguf/${currentJobId.value}`)
    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      conversionStatus.value = {
        status: 'error',
        progress: 0,
        current_tensor: '',
        error: _httpErrorMessage(response, data),
      }
      _stopConversionPolling()
      return
    }
    const status = _normalizeToolJobStatus(data)
    conversionStatus.value = status

    if (status.status === 'complete' || status.status === 'error' || status.status === 'cancelled') {
      _stopConversionPolling()
    }
  } catch (e: any) {
    conversionStatus.value = {
      status: 'error',
      progress: 0,
      current_tensor: '',
      error: String(e?.message || e),
    }
    _stopConversionPolling()
  }
}

async function startMerge() {
  try {
    const sourcePath = String(mergeForm.value.sourcePath || '').trim()
    const outputPath = String(mergeForm.value.outputPath || '').trim()
    if (!sourcePath) {
      throw new Error('Source path is required.')
    }
    if (!outputPath) {
      throw new Error('Output path is required.')
    }

    const response = await fetch('/api/tools/merge-safetensors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source_path: sourcePath,
        output_path: outputPath,
        overwrite: mergeForm.value.overwrite,
      }),
    })
    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      mergeStatus.value = {
        status: 'error',
        progress: 0,
        current_tensor: '',
        error: _httpErrorMessage(response, data),
      }
      return
    }

    const jobId = String((data as any)?.job_id || '').trim()
    if (!jobId) {
      throw new Error('merge-safetensors response missing job_id.')
    }

    currentMergeJobId.value = jobId
    mergeStatus.value = { status: 'pending', progress: 0, current_tensor: '', error: null }

    if (mergePollInterval.value) {
      clearInterval(mergePollInterval.value)
      mergePollInterval.value = null
    }
    mergePollInterval.value = window.setInterval(pollMergeStatus, 500)
  } catch (e: any) {
    mergeStatus.value = {
      status: 'error',
      progress: 0,
      current_tensor: '',
      error: String(e?.message || e),
    }
  }
}

function _stopMergePolling() {
  if (mergePollInterval.value) {
    clearInterval(mergePollInterval.value)
    mergePollInterval.value = null
  }
}

async function pollMergeStatus() {
  if (!currentMergeJobId.value) return

  try {
    const response = await fetch(`/api/tools/merge-safetensors/${currentMergeJobId.value}`)
    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      mergeStatus.value = {
        status: 'error',
        progress: 0,
        current_tensor: '',
        error: _httpErrorMessage(response, data),
      }
      _stopMergePolling()
      return
    }

    const status = _normalizeToolJobStatus(data)
    mergeStatus.value = status

    if (status.status === 'complete' || status.status === 'error' || status.status === 'cancelled') {
      _stopMergePolling()
    }
  } catch (e: any) {
    mergeStatus.value = {
      status: 'error',
      progress: 0,
      current_tensor: '',
      error: String(e?.message || e),
    }
    _stopMergePolling()
  }
}

// File browser functions
function browseForSafetensors() {
  browserMode.value = 'gguf_safetensors'
  browserPath.value = ggufForm.value.safetensorsPath || ''
  openFileBrowser()
}

function browseForOutputDir() {
  browserMode.value = 'gguf_output_dir'
  browserPath.value = ggufForm.value.outputDir || ''
  openFileBrowser()
}

function browseForMergeSource() {
  browserMode.value = 'merge_source'
  browserPath.value = mergeForm.value.sourcePath || ''
  openFileBrowser()
}

function browseForMergeOutputDir() {
  browserMode.value = 'merge_output_dir'
  const rawOutput = String(mergeForm.value.outputPath || '').trim()
  browserPath.value = _dirname(rawOutput)
  openFileBrowser()
}

async function openFileBrowser() {
  showFileBrowser.value = true
  selectedItem.value = null
  await loadBrowserPath()
}

function closeFileBrowser() {
  showFileBrowser.value = false
}

async function loadBrowserPath() {
  try {
    let ext = ''
    if (browserMode.value === 'gguf_safetensors' || browserMode.value === 'merge_source') {
      ext = '.safetensors,.safetensors.index.json'
    }

    const response = await fetch(
      `/api/tools/browse-files?path=${encodeURIComponent(browserPath.value)}&extensions=${encodeURIComponent(ext)}`,
    )
    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new Error(_httpErrorMessage(response, data))
    }
    browserData.value = data
    browserPath.value = browserData.value.path
  } catch (e: any) {
    console.error('Failed to browse:', e)
    if (conversionStatus.value) {
      conversionStatus.value = { ...conversionStatus.value, error: String(e?.message || e) }
    }
    if (mergeStatus.value) {
      mergeStatus.value = { ...mergeStatus.value, error: String(e?.message || e) }
    }
  }
}

function goToParent() {
  if (browserData.value.parent) {
    browserPath.value = browserData.value.parent
    loadBrowserPath()
  }
}

function selectItem(item: BrowserItem) {
  selectedItem.value = item
}

function openItem(item: BrowserItem) {
  if (item.type === 'directory') {
    browserPath.value = _joinPath(browserPath.value, item.name)
    loadBrowserPath()
    selectedItem.value = null
  } else {
    confirmSelection()
  }
}

function confirmSelection() {
  if (!selectedItem.value && browserRequiresSelection.value) return

  if (browserMode.value === 'gguf_safetensors') {
    if (!selectedItem.value) return
    const fullPath = _joinPath(browserPath.value, selectedItem.value.name)
    ggufForm.value.safetensorsPath = fullPath
  } else if (browserMode.value === 'gguf_output_dir') {
    if (!selectedItem.value) {
      ggufForm.value.outputDir = browserPath.value
    } else if (selectedItem.value.type === 'directory') {
      const fullPath = _joinPath(browserPath.value, selectedItem.value.name)
      ggufForm.value.outputDir = fullPath
    }
  } else if (browserMode.value === 'merge_source') {
    if (!selectedItem.value) return
    const fullPath = _joinPath(browserPath.value, selectedItem.value.name)
    mergeForm.value.sourcePath = fullPath
  } else if (browserMode.value === 'merge_output_dir') {
    let targetDir = String(browserPath.value || '').trim()
    if (selectedItem.value?.type === 'directory') {
      targetDir = _joinPath(browserPath.value, selectedItem.value.name)
    }
    mergeForm.value.outputPath = _joinPath(targetDir, mergeOutputFileName.value)
  }

  closeFileBrowser()
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + ' MB'
  return (bytes / 1024 / 1024 / 1024).toFixed(2) + ' GB'
}

onMounted(() => {
  loadModelMetadata()
})

onUnmounted(() => {
  _stopConversionPolling()
  _stopMergePolling()
})
</script>
