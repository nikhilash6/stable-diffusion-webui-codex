<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Prompt panel wrapper with toolbars and modals.
Renders `PromptFields` and an optional toolbar for asset insertion (LoRA/TI) and styles creation/application.
Supports forced negative-field hiding via `hideNegative` while preserving capability-based hiding.

Symbols (top-level; keep in sync; no ghosts):
- `PromptCard` (component): Prompt panel with prompt/negative fields, optional assets/styles controls, and insertion modals.
-->

<template>
  <div class="panel">
    <div class="panel-header">{{ title }}
      <div class="toolbar prompt-toolbar">
        <template v-if="enableAssets">
          <button class="btn btn-sm btn-secondary" type="button" @click="showTI = true">Textual Inversion</button>
          <button class="btn btn-sm btn-secondary" type="button" @click="showLora = true">LoRA</button>
        </template>

        <template v-if="enableStyles">
          <label class="label-muted styles-label">{{ stylesLabel }}</label>
          <div class="cdx-input-with-actions">
            <input class="ui-input styles-input" :list="styleListId" v-model="styleName" placeholder="Filter styles" />
            <datalist :id="styleListId">
              <option v-for="s in styleNames" :key="s" :value="s" />
            </datalist>
            <div class="cdx-input-actions">
              <button class="btn btn-sm btn-secondary" type="button" @click="showStyle = true">New</button>
              <button class="btn btn-sm btn-outline" type="button" @click="applyStyle(styleName)">Apply</button>
            </div>
          </div>
        </template>

        <template v-else-if="toolbarLabel">
          <label class="label-muted styles-label">{{ toolbarLabel }}</label>
        </template>
      </div>
    </div>

    <div class="panel-body">
      <div v-if="fieldsId" :id="fieldsId">
        <PromptFields
          v-model:prompt="innerPrompt"
          v-model:negative="innerNegative"
          :hide-negative="hideNegative"
          :token-engine="tokenEngine"
        />
      </div>
      <PromptFields
        v-else
        v-model:prompt="innerPrompt"
        v-model:negative="innerNegative"
        :hide-negative="hideNegative"
        :token-engine="tokenEngine"
      />

      <slot />
    </div>

    <LoraModal v-if="enableAssets" v-model="showLora" :show-negative-target="!hideNegative" @insert="onInsertToken" />
    <TextualInversionModal v-if="enableAssets" v-model="showTI" @insert="onInsertToken" />
    <StyleEditorModal v-if="enableStyles" v-model="showStyle" @saved="onStyleSaved" />
  </div>
</template>

<script setup lang="ts">
import { computed, getCurrentInstance } from 'vue'

import { usePromptCard } from '../../composables/usePromptCard'
import LoraModal from '../modals/LoraModal.vue'
import StyleEditorModal from '../modals/StyleEditorModal.vue'
import TextualInversionModal from '../modals/TextualInversionModal.vue'
import PromptFields from './PromptFields.vue'

const props = withDefaults(defineProps<{
  prompt: string
  negative: string
  title?: string
  enableAssets?: boolean
  enableStyles?: boolean
  stylesLabel?: string
  toolbarLabel?: string
  supportsNegative?: boolean
  hideNegative?: boolean
  tokenEngine?: string
  fieldsId?: string
}>(), {
  title: 'Prompt',
  enableAssets: true,
  enableStyles: true,
  stylesLabel: 'Styles',
  toolbarLabel: '',
  supportsNegative: true,
  hideNegative: false,
  tokenEngine: '',
  fieldsId: '',
})

const emit = defineEmits<{
  (e: 'update:prompt', value: string): void
  (e: 'update:negative', value: string): void
}>()

const innerPrompt = computed({
  get: () => props.prompt,
  set: (value: string) => emit('update:prompt', value),
})

const innerNegative = computed({
  get: () => props.negative,
  set: (value: string) => emit('update:negative', value),
})

const {
  hideNegative: hideNegativeByCapability,
  showLora,
  showTI,
  showStyle,
  styleName,
  styleNames,
  applyStyle,
  onInsertToken,
  onStyleSaved,
} = usePromptCard({
  prompt: innerPrompt,
  negative: innerNegative,
  supportsNegative: props.supportsNegative,
})

const hideNegative = computed(() => hideNegativeByCapability.value || props.hideNegative === true)
const tokenEngine = computed(() => String(props.tokenEngine || '').trim())

const instance = getCurrentInstance()
const styleListId = `style-list-${instance?.uid ?? Math.floor(Math.random() * 1_000_000_000)}`
</script>
