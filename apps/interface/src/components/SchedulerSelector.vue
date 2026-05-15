<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Scheduler dropdown selector.
Renders a scheduler dropdown from the global scheduler catalog, groups options by backend recommendations, and emits the selected scheduler name.

Symbols (top-level; keep in sync; no ghosts):
- `SchedulerSelector` (component): Scheduler selector component.
- `recommendedOptions` / `riskOptions` (const): Recommendation-grouped scheduler lists for select optgroups.
- `showRiskWarning` (const): Indicates whether the selected scheduler is outside the recommendation set.
- `onChange` (function): Emits `update:modelValue` for the selected scheduler.
-->

<template>
  <div class="form-field">
    <label class="label-muted">{{ labelText }}</label>
    <select class="select-md" :disabled="disabled" :value="modelValue" @change="onChange">
      <option v-if="allowEmpty" value="">{{ emptyLabelText }}</option>
      <template v-if="showRecommendationGroups">
        <optgroup v-if="recommendedOptions.length" label="Recommended">
          <option v-for="entry in recommendedOptions" :key="entry.name" :value="entry.name">
            {{ entry.label ?? entry.name }}
          </option>
        </optgroup>
        <optgroup v-if="riskOptions.length" label="Use at your own risk">
          <option v-for="entry in riskOptions" :key="entry.name" :value="entry.name">
            {{ entry.label ?? entry.name }}
          </option>
        </optgroup>
      </template>
      <template v-else>
        <option v-for="entry in schedulers" :key="entry.name" :value="entry.name">
          {{ entry.label ?? entry.name }}
        </option>
      </template>
    </select>
    <p v-if="showRiskWarning" class="panel-status">{{ riskWarningText }}</p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { SchedulerInfo } from '../api/types'

const props = defineProps<{
  schedulers: SchedulerInfo[]
  modelValue: string
  label?: string
  allowEmpty?: boolean
  emptyLabel?: string
  disabled?: boolean
  recommendedNames?: string[] | null
}>()

const emit = defineEmits({
  'update:modelValue': (value: string) => true,
})

const labelText = computed(() => props.label ?? 'Scheduler')
const allowEmpty = computed(() => props.allowEmpty === true)
const emptyLabelText = computed(() => props.emptyLabel ?? 'Select')
const disabled = computed(() => props.disabled === true)
const selectedName = computed(() => String(props.modelValue || '').trim())
const recommendedSet = computed(() => {
  const list = Array.isArray(props.recommendedNames) ? props.recommendedNames : []
  return new Set(list.map((entry) => String(entry || '').trim()).filter((entry) => entry.length > 0))
})
const showRecommendationGroups = computed(() => recommendedSet.value.size > 0)
const recommendedOptions = computed(() => {
  if (!showRecommendationGroups.value) return []
  return props.schedulers.filter((entry) => recommendedSet.value.has(String(entry.name)))
})
const riskOptions = computed(() => {
  if (!showRecommendationGroups.value) return props.schedulers
  return props.schedulers.filter((entry) => !recommendedSet.value.has(String(entry.name)))
})
const showRiskWarning = computed(() => {
  if (!showRecommendationGroups.value) return false
  const selected = selectedName.value
  if (!selected) return false
  return !recommendedSet.value.has(selected)
})
const riskWarningText = computed(() => {
  return `Warning: selected scheduler '${selectedName.value}' is outside engine recommendations; model behavior may be unpredictable.`
})

function onChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  emit('update:modelValue', value)
}
</script>
