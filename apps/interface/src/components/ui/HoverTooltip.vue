<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Reusable hover/focus tooltip primitive for compact UI hints.
Wraps trigger content and renders a floating tooltip panel with a title and multi-line body, shown on hover or keyboard focus.

Symbols (top-level; keep in sync; no ghosts):
- `HoverTooltip` (component): Wrapper component that displays a floating tooltip for its slot trigger.
- `TooltipEntry` (type): Normalized tooltip line with an optional emphasized lead-in and body text.
- `tooltipEntries` (computed): Normalized tooltip body lines with optional emphasized lead-ins.
-->

<template>
  <span class="cdx-hover-tooltip" :tabindex="wrapperFocusable ? 0 : undefined">
    <slot />
    <span class="cdx-hover-tooltip__panel" role="tooltip">
      <span v-if="title" class="cdx-hover-tooltip__title">{{ title }}</span>
      <span
        v-for="(entry, index) in tooltipEntries"
        :key="`${entry.lead}-${entry.body}-${index}`"
        class="cdx-hover-tooltip__line"
      >
        <span v-if="entry.lead" class="cdx-hover-tooltip__line-lead">{{ entry.lead }}</span>
        <span>{{ entry.body }}</span>
      </span>
    </span>
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue'

type TooltipEntry = {
  lead: string
  body: string
}

const props = withDefaults(defineProps<{
  title?: string
  content: string | readonly string[]
  wrapperFocusable?: boolean
}>(), {
  title: '',
  wrapperFocusable: true,
})

const tooltipEntries = computed<TooltipEntry[]>(() => {
  const lines = typeof props.content === 'string'
    ? props.content.split('\n')
    : Array.from(props.content)

  return lines
    .map((line: string) => line.trim())
    .filter((line: string) => line.length > 0)
    .map((line: string) => {
      const match = line.match(/^\[\[([^[]+?:)\]\]\s*(.+)$/)
      if (!match) return { lead: '', body: line }
      return {
        lead: match[1].trim(),
        body: match[2].trim(),
      }
    })
})
</script>
