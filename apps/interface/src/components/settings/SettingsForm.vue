<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Settings form renderer from the backend schema.
Renders settings fields from the `/api/settings/schema` model and applies changes via `/api/options`, tracking pending/dirty changes locally and
showing apply metadata feedback (`applied_now[]` vs `restart_required[]`) after each save. Settings writes are revision-aware and fail loud on stale
pages instead of silently overwriting newer option state.

Symbols (top-level; keep in sync; no ghosts):
- `SettingsForm` (component): Dynamic settings form used in the Settings view.
- `onChange` (function): Updates local model state and tracks dirty keys.
- `applyChanges` (function): Persists pending changes via the backend options endpoint.
-->

<template>
  <div class="settings-form">
    <div v-if="fields.length === 0" class="card caption">No settings in this section.</div>
    <div v-else class="settings-grid-col">
      <div v-for="f in fields" :key="f.key" class="settings-form-row">
        <template v-if="f.type === 'slider'">
          <SliderField
            :label="f.label"
            :modelValue="asNumber(model[f.key], f.default)"
            :min="f.min ?? 0"
            :max="f.max ?? 100"
            :step="f.step ?? 1"
            :inputStep="f.step ?? 1"
            :numberUpdateOnInput="true"
            :numberSize="'md'"
            :showButtons="false"
            inputClass="cdx-input-w-sm"
            @update:modelValue="(v) => onChange(f.key, v)"
          />
        </template>
        <template v-else>
          <label class="settings-form-label">{{ f.label }}</label>
          <div class="settings-form-control">
            <template v-if="f.type === 'checkbox'">
              <input type="checkbox" :checked="asBool(model[f.key])" @change="onChange(f.key, ($event.target as HTMLInputElement).checked)" />
            </template>
            <template v-else-if="f.type === 'radio' && f.choices && f.choices.length">
              <div class="settings-radio-group">
                <label v-for="opt in f.choices" :key="String(opt)" class="settings-radio-item">
                  <input type="radio" :name="'rad-'+f.key" :checked="String(model[f.key])===String(opt)" @change="onChange(f.key, opt)" />
                  <span>{{ String(opt) }}</span>
                </label>
              </div>
            </template>
            <template v-else-if="f.type === 'dropdown'">
              <select class="select-md" :value="String(model[f.key] ?? '')" @change="onChange(f.key, ($event.target as HTMLSelectElement).value)">
                <option v-for="opt in (f.choices ?? [])" :key="String(opt)" :value="String(opt)">{{ String(opt) }}</option>
              </select>
            </template>
            <template v-else-if="f.type === 'number'">
              <input type="number" class="ui-input" :min="f.min ?? undefined" :max="f.max ?? undefined" :step="f.step ?? 1" :value="asNumber(model[f.key], f.default)" @input="onChange(f.key, asNumber(($event.target as HTMLInputElement).value))" />
            </template>
            <template v-else-if="f.type === 'color'">
              <input type="color" class="ui-input" :value="String(model[f.key] ?? f.default ?? '#000000')" @input="onChange(f.key, ($event.target as HTMLInputElement).value)" />
            </template>
            <template v-else-if="f.type === 'html'">
              <div class="card caption" v-html="String(f.default ?? '')" />
            </template>
            <template v-else>
              <input type="text" class="ui-input" :value="String(model[f.key] ?? f.default ?? '')" @input="onChange(f.key, ($event.target as HTMLInputElement).value)" />
            </template>
          </div>
        </template>
      </div>
      <div class="settings-form-actions">
        <button class="btn btn-sm btn-primary" :disabled="pending || changedCount===0" @click="applyChanges">Apply</button>
        <span class="caption" v-if="changedCount>0">{{ changedCount }} change(s) pending</span>
      </div>
      <div v-if="lastErrorMessage" class="settings-apply-alert settings-apply-alert--warn" role="alert">
        <div class="caption">{{ lastErrorMessage }}</div>
      </div>
      <div v-else-if="lastRestartRequired.length > 0" class="settings-apply-alert settings-apply-alert--warn" role="alert">
        <div class="caption">Restart required for:</div>
        <ul class="settings-apply-alert-list">
          <li v-for="message in lastRestartRequired" :key="message">{{ message }}</li>
        </ul>
      </div>
      <div v-else-if="lastAppliedNow.length > 0" class="settings-apply-alert settings-apply-alert--ok" role="status">
        <div class="caption">Applied immediately:</div>
        <ul class="settings-apply-alert-list">
          <li v-for="message in lastAppliedNow" :key="message">{{ message }}</li>
        </ul>
      </div>
    </div>
  </div>
  
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { SettingsField } from '../../api/types'
import { updateOptions } from '../../api/client'
import { formatSettingsRevisionConflictMessage, resolveSettingsRevisionConflict } from '../../composables/settings_revision_conflict'
import SliderField from '../ui/SliderField.vue'

const props = defineProps<{ fields: SettingsField[]; values: Record<string, unknown>; revision: number }>()

const model = ref<Record<string, unknown>>({})
const dirty = ref<Record<string, unknown>>({})
const pending = ref(false)
const lastAppliedNow = ref<string[]>([])
const lastRestartRequired = ref<string[]>([])
const lastErrorMessage = ref('')
const currentRevision = ref(0)

watch(
  () => ({ values: props.values, revision: props.revision }),
  ({ values, revision }) => {
    const normalizedRevision = Number.isFinite(revision) ? Math.max(0, Math.trunc(revision)) : 0
    currentRevision.value = normalizedRevision
    const v = values
    model.value = { ...(v || {}) }
    dirty.value = {}
    lastErrorMessage.value = ''
  },
  { immediate: true, deep: true },
)

function onChange(key: string, value: unknown) {
  model.value[key] = value
  dirty.value[key] = value
  lastErrorMessage.value = ''
}

const changedCount = computed(() => Object.keys(dirty.value).length)

async function applyChanges() {
  if (pending.value || changedCount.value === 0) return
  pending.value = true
  lastErrorMessage.value = ''
  try {
    const response = await updateOptions(dirty.value, { expectedRevision: currentRevision.value })
    const appliedNowRaw = (response as any).applied_now
    const restartRequiredRaw = (response as any).restart_required
    lastAppliedNow.value = Array.isArray(appliedNowRaw) ? appliedNowRaw.map((item) => String(item)) : []
    lastRestartRequired.value = Array.isArray(restartRequiredRaw) ? restartRequiredRaw.map((item) => String(item)) : []
    currentRevision.value = Number.isFinite((response as any).revision)
      ? Math.max(0, Math.trunc((response as any).revision))
      : currentRevision.value
    dirty.value = {}
  } catch (error) {
    const conflictRevision = resolveSettingsRevisionConflict(error)
    if (conflictRevision !== null) {
      currentRevision.value = conflictRevision
      lastAppliedNow.value = []
      lastRestartRequired.value = []
      lastErrorMessage.value = formatSettingsRevisionConflictMessage(
        conflictRevision,
        'retry applying your pending changes manually',
      )
      return
    }
    lastAppliedNow.value = []
    lastRestartRequired.value = []
    lastErrorMessage.value = error instanceof Error ? error.message : String(error)
  } finally {
    pending.value = false
  }
}

function asBool(v: unknown, fallback = false) {
  if (typeof v === 'boolean') return v
  if (typeof v === 'string') return v === 'true' || v === '1'
  if (typeof v === 'number') return v !== 0
  return fallback
}

function asNumber(v: unknown, def?: unknown) {
  if (typeof v === 'number') return v
  if (typeof v === 'string' && v !== '') return Number(v)
  if (typeof def === 'number') return def
  return 0
}
</script>
