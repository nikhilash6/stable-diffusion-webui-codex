<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 5B quicksettings selectors.
Renders the exact 5B header surface: the shared `TXT2VID|IMG2VID` input-mode toggle, one single-stage GGUF selector, and WAN text encoder/VAE selectors
for the dedicated `wan22_5b` tab family. The header never mirrors the 14B dual-stage owner shape.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsWan22_5b` (component): WAN 2.2 5B quicksettings row used by the main quicksettings bar.
- `dirLabel` (function): Produces compact directory/file labels from absolute paths.
- `encoderLabel` (function): Produces compact `family/basename` labels for WAN text encoder values.
-->

<template>
  <div class="quicksettings-group qs-group-wan-mode">
    <label class="label-muted">Mode</label>
    <div class="qs-row">
      <div class="qs-toggle-group" role="group" aria-label="WAN 2.2 5B mode">
        <button
          type="button"
          class="btn qs-toggle-btn"
          :class="mode === 'txt2vid' ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off'"
          :aria-pressed="mode === 'txt2vid'"
          @click="$emit('update:mode', 'txt2vid')"
        >
          TXT2VID
        </button>
        <button
          type="button"
          class="btn qs-toggle-btn"
          :class="mode === 'img2vid' ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off'"
          :aria-pressed="mode === 'img2vid'"
          @click="$emit('update:mode', 'img2vid')"
        >
          IMG2VID
        </button>
      </div>
    </div>
  </div>

  <div class="quicksettings-group qs-group-wan-model">
    <label class="label-muted">WAN Model</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select id="qs-wan-single" class="select-md" :value="model" @change="$emit('update:model', ($event.target as HTMLSelectElement).value)">
          <option value="">{{ builtInLabel }}</option>
          <option v-for="entry in modelChoices" :key="entry" :value="entry">{{ dirLabel(entry) }}</option>
        </select>
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!model"
          title="Show model metadata"
          aria-label="Show model metadata"
          @click="$emit('showMetadata', { kind: 'wan_model', value: model })"
        >
          i
        </button>
        <button class="btn qs-btn-outline qs-inline-btn" type="button" title="Browse WAN models…" aria-label="Browse WAN models…" @click="$emit('browseModels')">+</button>
      </div>
    </div>
  </div>

  <div class="quicksettings-group qs-group-wan-text-encoder">
    <label class="label-muted">WAN Text Encoder</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select id="qs-wan5b-text-encoder" class="select-md" :value="textEncoder" @change="$emit('update:textEncoder', ($event.target as HTMLSelectElement).value)">
          <option value="">{{ builtInLabel }}</option>
          <option v-for="te in textEncoderChoices" :key="te" :value="te">{{ encoderLabel(te) }}</option>
        </select>
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!textEncoder"
          title="Show text encoder metadata"
          aria-label="Show text encoder metadata"
          @click="$emit('showMetadata', { kind: 'wan_text_encoder', value: textEncoder })"
        >
          i
        </button>
        <button class="btn qs-btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseTe')">+</button>
      </div>
    </div>
  </div>

  <div class="quicksettings-group qs-group-wan-vae">
    <label class="label-muted">WAN VAE</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select id="qs-wan5b-vae" class="select-md" :value="vae" @change="$emit('update:vae', ($event.target as HTMLSelectElement).value)">
          <option value="">{{ builtInLabel }}</option>
          <option v-for="entry in vaeChoices" :key="entry" :value="entry">{{ dirLabel(entry) }}</option>
        </select>
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!vae"
          title="Show VAE metadata"
          aria-label="Show VAE metadata"
          @click="$emit('showMetadata', { kind: 'wan_vae', value: vae })"
        >
          i
        </button>
        <button class="btn qs-btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseVae')">+</button>
      </div>
    </div>
  </div>

  <div class="quicksettings-group qs-group-wan-refresh qs-group-wan-refresh--end">
    <label class="label-muted">Lists</label>
    <div class="qs-row">
      <button class="btn qs-btn-secondary qs-refresh-btn" type="button" title="Refresh lists" @click="$emit('refresh')">Refresh</button>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  mode: 'txt2vid' | 'img2vid'
  model: string
  modelChoices: string[]
  textEncoder: string
  textEncoderChoices: string[]
  vae: string
  vaeChoices: string[]
}>()

defineEmits<{
  (e: 'update:mode', value: 'txt2vid' | 'img2vid'): void
  (e: 'update:model', value: string): void
  (e: 'update:textEncoder', value: string): void
  (e: 'update:vae', value: string): void
  (e: 'browseModels'): void
  (e: 'browseTe'): void
  (e: 'browseVae'): void
  (e: 'refresh'): void
  (e: 'showMetadata', payload: { kind: 'wan_model' | 'wan_text_encoder' | 'wan_vae'; value: string }): void
}>()

const builtInLabel = 'Select…'

function dirLabel(path: string): string {
  const normalized = path.replace(/\\/g, '/')
  if (!normalized) return ''
  const index = normalized.lastIndexOf('/')
  return index >= 0 ? normalized.slice(index + 1) || normalized : normalized
}

function encoderLabel(value: string): string {
  const normalized = String(value || '').replace(/\\/g, '/')
  if (!normalized) return ''
  if (!normalized.includes('/')) return normalized
  const [family, ...rest] = normalized.split('/').filter(Boolean)
  if (!family || rest.length === 0) return normalized
  const tail = rest[rest.length - 1] || rest[0]
  return `${family}/${tail}`
}
</script>
