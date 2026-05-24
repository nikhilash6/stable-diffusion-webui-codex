<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared non-WAN quicksettings asset selectors.
Renders the common checkpoint/VAE/text-encoder selector block for non-WAN families, with explicit props for family-owned labels,
formatting, dual-text-encoder rows, disabled states, optional no-VAE families, and additive inline controls such as the Z-Image Turbo slot.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsAssetBlock` (component): Shared non-WAN checkpoint/VAE/text-encoder selector block used by `QuickSettingsBar.vue`.
- `ChoiceLabelMode` (type): Supported display strategies for selector choices (`raw`, `truncate`, `sentinel`).
- `MetadataKind` (type): Supported metadata discriminants emitted by the shared asset block.
- `truncatePath` (function): Truncates absolute paths for compact dropdown labels.
- `isVaeSentinel` (function): Returns whether a VAE value is a sentinel selection (built-in/none) without metadata.
- `vaeLabel` (function): Formats canonical VAE sentinel values for dropdown display.
- `textEncoderLabel` (function): Builds a compact `family/basename` label for text encoder dropdown values.
- `formatChoice` (function): Applies the configured choice-label mode to selector values.
- `metadataTargetLabel` (function): Resolves the right metadata noun (`model` vs `checkpoint`) for button titles.
-->

<template>
  <div class="quicksettings-group qs-group-checkpoint">
    <label class="label-muted">{{ checkpointLabel }}</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select
          class="select-md"
          :value="checkpoint"
          :disabled="disabled"
          @change="$emit('update:checkpoint', ($event.target as HTMLSelectElement).value)"
        >
          <option v-if="checkpoints.length === 0" value="">No models found</option>
          <option v-for="model in checkpoints" :key="model" :value="model">
            {{ formatChoice(model, checkpointChoiceMode) }}
          </option>
        </select>
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="disabled || !checkpoint"
          :title="`Show ${metadataTargetLabel(checkpointLabel)} metadata`"
          :aria-label="`Show ${metadataTargetLabel(checkpointLabel)} metadata`"
          @click="$emit('showMetadata', { kind: 'checkpoint', value: checkpoint })"
        >
          i
        </button>
        <button class="btn qs-btn-outline qs-inline-btn" type="button" :disabled="disabled" @click="$emit('addCheckpointPath')">+</button>
      </div>
    </div>
  </div>

  <slot name="after-checkpoint" />

  <div v-if="showVae" class="quicksettings-group qs-group-vae">
    <label class="label-muted">VAE</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select
          class="select-md"
          :value="vae"
          :disabled="disabled"
          @change="$emit('update:vae', ($event.target as HTMLSelectElement).value)"
        >
          <option v-if="vaePlaceholderLabel" value="">{{ vaePlaceholderLabel }}</option>
          <option v-for="value in vaeChoices" :key="value" :value="value">
            {{ formatChoice(value, vaeChoiceMode) }}
          </option>
        </select>
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="disabled || !vae || isVaeSentinel(vae)"
          title="Show VAE metadata"
          aria-label="Show VAE metadata"
          @click="$emit('showMetadata', { kind: 'vae', value: vae })"
        >
          i
        </button>
        <button class="btn qs-btn-outline qs-inline-btn" type="button" :disabled="disabled" @click="$emit('addVaePath')">+</button>
      </div>
    </div>
  </div>

  <div v-if="showTextEncoder" :class="['quicksettings-group', textEncoderGroupClass]">
    <label class="label-muted">{{ textEncoderGroupLabel }}</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select
          class="select-md"
          :value="textEncoder"
          :disabled="disabled"
          @change="$emit('update:textEncoder', ($event.target as HTMLSelectElement).value)"
        >
          <option value="">{{ textEncoderAutomaticLabel }}</option>
          <option v-for="value in textEncoderChoices" :key="value" :value="value">
            {{ textEncoderLabel(value) }}
          </option>
        </select>
        <button
          v-if="showTextEncoderActions"
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="disabled || !textEncoder"
          title="Show text encoder metadata"
          aria-label="Show text encoder metadata"
          @click="$emit('showMetadata', { kind: textEncoderMetadataKind, value: textEncoder })"
        >
          i
        </button>
        <button
          v-if="showTextEncoderActions"
          class="btn qs-btn-outline qs-inline-btn"
          type="button"
          :disabled="disabled"
          @click="$emit('addTencPath')"
        >
          +
        </button>
      </div>

      <div v-if="showSecondaryTextEncoder" class="qs-pair">
        <select
          class="select-md"
          :value="secondaryTextEncoder"
          :disabled="disabled"
          @change="$emit('update:secondaryTextEncoder', ($event.target as HTMLSelectElement).value)"
        >
          <option value="">{{ secondaryTextEncoderAutomaticLabel }}</option>
          <option v-for="value in secondaryTextEncoderChoices" :key="`secondary-${value}`" :value="value">
            {{ textEncoderLabel(value) }}
          </option>
        </select>
        <button
          v-if="showSecondaryTextEncoderActions"
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="disabled || !secondaryTextEncoder"
          title="Show text encoder metadata"
          aria-label="Show text encoder metadata"
          @click="$emit('showMetadata', { kind: secondaryTextEncoderMetadataKind, value: secondaryTextEncoder })"
        >
          i
        </button>
        <button
          v-if="showSecondaryTextEncoderActions"
          class="btn qs-btn-outline qs-inline-btn"
          type="button"
          :disabled="disabled"
          @click="$emit('addTencPath')"
        >
          +
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
type ChoiceLabelMode = 'raw' | 'truncate' | 'sentinel'
type MetadataKind = 'checkpoint' | 'vae' | 'text_encoder' | 'text_encoder_primary' | 'text_encoder_secondary'

withDefaults(defineProps<{
  checkpoint: string
  checkpoints: string[]
  checkpointLabel?: string
  checkpointChoiceMode?: ChoiceLabelMode
  vae: string
  vaeChoices: string[]
  vaeChoiceMode?: ChoiceLabelMode
	  vaePlaceholderLabel?: string
	  showVae?: boolean
	  textEncoder: string
  textEncoderChoices: string[]
  textEncoderGroupLabel?: string
  textEncoderGroupClass?: string
  textEncoderAutomaticLabel?: string
  textEncoderMetadataKind?: MetadataKind
  showTextEncoder?: boolean
  showTextEncoderActions?: boolean
  secondaryTextEncoder?: string
  secondaryTextEncoderChoices?: string[]
  secondaryTextEncoderAutomaticLabel?: string
  secondaryTextEncoderMetadataKind?: MetadataKind
  showSecondaryTextEncoder?: boolean
  showSecondaryTextEncoderActions?: boolean
  disabled?: boolean
}>(), {
  checkpointLabel: 'Checkpoint',
  checkpointChoiceMode: 'raw',
	  vaeChoiceMode: 'sentinel',
	  vaePlaceholderLabel: undefined,
	  showVae: true,
  textEncoderGroupLabel: 'Text Encoder',
  textEncoderGroupClass: 'qs-group-text-encoder',
  textEncoderAutomaticLabel: 'Built-in',
  textEncoderMetadataKind: 'text_encoder',
  showTextEncoder: true,
  showTextEncoderActions: false,
  secondaryTextEncoder: '',
  secondaryTextEncoderChoices: () => [],
  secondaryTextEncoderAutomaticLabel: 'Built-in',
  secondaryTextEncoderMetadataKind: 'text_encoder_secondary',
  showSecondaryTextEncoder: false,
  showSecondaryTextEncoderActions: false,
  disabled: false,
})

defineEmits<{
  (e: 'update:checkpoint', value: string): void
  (e: 'update:vae', value: string): void
  (e: 'update:textEncoder', value: string): void
  (e: 'update:secondaryTextEncoder', value: string): void
  (e: 'addCheckpointPath'): void
  (e: 'addVaePath'): void
  (e: 'addTencPath'): void
  (e: 'showMetadata', payload: { kind: MetadataKind; value: string }): void
}>()

function truncatePath(path: string, maxLen = 40): string {
  if (!path || path.length <= maxLen) return path
  const parts = path.replace(/\\/g, '/').split('/')
  const name = parts[parts.length - 1] || path
  return name.length > maxLen ? `...${name.slice(-maxLen)}` : name
}

function isVaeSentinel(value: string): boolean {
  const normalized = String(value || '').trim().toLowerCase()
  return normalized === 'automatic' || normalized === 'built in' || normalized === 'built-in' || normalized === 'none'
}

function vaeLabel(value: string): string {
  const normalized = String(value || '').trim().toLowerCase()
  if (normalized === 'built in' || normalized === 'built-in') return 'Built-in'
  if (normalized === 'none') return 'None'
  return value
}

function textEncoderLabel(raw: unknown): string {
  const value = String(raw ?? '')
  if (!value.includes('/')) return value
  const parts = value.replace(/\\/g, '/').split('/').filter(Boolean)
  if (parts.length < 2) return value
  return `${parts[0]}/${parts[parts.length - 1]}`
}

function formatChoice(value: string, mode: ChoiceLabelMode): string {
  if (mode === 'truncate') return truncatePath(value)
  if (mode === 'sentinel') return vaeLabel(value)
  return value
}

function metadataTargetLabel(label: string): 'model' | 'checkpoint' {
  return String(label || '').trim().toLowerCase() === 'model' ? 'model' : 'checkpoint'
}
</script>
