<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Compact segmented control primitive for short mutually-exclusive choices.
Renders a pill-group button row, highlights the active choice, and emits `update:modelValue` for presentational parents.

Symbols (top-level; keep in sync; no ghosts):
- `CompactSegmentedControl` (component): Small segmented button group for short option sets.
- `onSelect` (function): Emits the chosen option value when enabled.
-->

<template>
  <div class="cdx-segmented-control" :aria-label="ariaLabel || undefined" role="group">
    <button
      v-for="option in options"
      :key="option.value"
      :class="[
        'btn',
        'qs-toggle-btn',
        'qs-toggle-btn--sm',
        'cdx-segmented-control__button',
        option.value === modelValue ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off',
      ]"
      type="button"
      :aria-pressed="option.value === modelValue"
      :disabled="disabled || option.disabled"
      @click="onSelect(option.value)"
    >
      {{ option.label }}
    </button>
  </div>
</template>

<script setup lang="ts">
type SegmentedOption = {
  value: string
  label: string
  disabled?: boolean
}

const props = withDefaults(defineProps<{
  modelValue: string
  options: readonly SegmentedOption[]
  disabled?: boolean
  ariaLabel?: string
}>(), {
  disabled: false,
  ariaLabel: '',
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
}>()

function onSelect(value: string): void {
  if (props.disabled) return
  emit('update:modelValue', value)
}
</script>
