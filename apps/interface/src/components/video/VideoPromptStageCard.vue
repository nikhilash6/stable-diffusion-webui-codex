<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Generic prompt-stage card for video workspaces.
Renders a neutral video-card header, optional collapse toggle, shared prompt/negative fields, and optional footer content
without owning any family-specific prompt/runtime state.

Symbols (top-level; keep in sync; no ghosts):
- `VideoPromptStageCard` (component): Generic video prompt card.
- `isOpen` (const): Normalized open-state used when the card is collapsible.
- `toggleOpen` (function): Emits the next open-state for the owner.
-->

<template>
  <div class="gen-card cdx-video-card">
    <div class="cdx-video-card-header">
      <div class="cdx-video-card-header__left">
        <span class="cdx-video-card-header__title">{{ title }}</span>
      </div>
      <div class="cdx-video-card-header__right">
        <span v-if="cornerLabel" class="caption">{{ cornerLabel }}</span>
        <slot name="header-actions" />
        <button
          v-if="collapsible"
          class="btn-icon"
          type="button"
          :disabled="disabled"
          :aria-expanded="isOpen ? 'true' : 'false'"
          :title="isOpen ? 'Collapse' : 'Expand'"
          :aria-label="`Toggle ${title}`"
          @click="toggleOpen"
        >
          <span aria-hidden="true">{{ isOpen ? '▾' : '▸' }}</span>
        </button>
      </div>
    </div>

    <div v-if="!collapsible || isOpen" class="mt-2 cdx-video-card-body">
      <PromptFields
        :prompt="prompt"
        :negative="negative"
        :hide-negative="hideNegative"
        :token-engine="tokenEngine"
        @update:prompt="(value) => emit('update:prompt', value)"
        @update:negative="(value) => emit('update:negative', value)"
      />
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import PromptFields from '../prompt/PromptFields.vue'

const props = withDefaults(defineProps<{
  title: string
  prompt: string
  negative: string
  hideNegative?: boolean
  tokenEngine?: string
  collapsible?: boolean
  open?: boolean
  cornerLabel?: string
  disabled?: boolean
}>(), {
  hideNegative: false,
  tokenEngine: '',
  collapsible: false,
  open: true,
  cornerLabel: '',
  disabled: false,
})

const emit = defineEmits<{
  (e: 'update:prompt', value: string): void
  (e: 'update:negative', value: string): void
  (e: 'update:open', value: boolean): void
}>()

const isOpen = computed(() => !props.collapsible || props.open)

function toggleOpen(): void {
  emit('update:open', !isOpen.value)
}
</script>
