<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 14B quicksettings selectors.
Renders the exact 14B header surface: the shared `TXT2VID|IMG2VID` input-mode toggle, optional LightX2V toggle, high/low GGUF selectors,
text encoder, and VAE selectors for the dedicated `wan22_14b` tab family.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsWan` (component): WAN 2.2 14B quicksettings row used by the main quicksettings bar.
- `dirLabel` (function): Produces compact directory/file labels from absolute paths.
- `encoderLabel` (function): Produces compact `family/basename` labels for WAN text encoder values.
-->

<template>
  <div class="quicksettings-group qs-group-wan-mode">
    <label class="label-muted">Mode</label>
    <div class="qs-row">
      <div class="qs-toggle-group" role="group" aria-label="WAN 2.2 14B mode">
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

  <div class="quicksettings-group qs-group-wan-lightx2v">
    <div class="qs-row">
      <button
        :class="['btn', 'qs-toggle-btn', lightx2v ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :aria-pressed="lightx2v"
        title="Enable LightX2V runtime"
        @click="$emit('update:lightx2v', !lightx2v)"
      >
        LightX2V
      </button>
      <button
        class="btn qs-btn-outline qs-inline-btn"
        type="button"
        title="Browse WAN models…"
        aria-label="Browse WAN models…"
        @click="$emit('browseModels')"
      >
        +
      </button>
    </div>
  </div>

  <div class="quicksettings-group qs-group-wan-high">
    <label class="label-muted">WAN High model</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select id="qs-wan-high" class="select-md" :value="highModel" @change="$emit('update:highModel', ($event.target as HTMLSelectElement).value)">
          <option value="">{{ builtInLabel }}</option>
          <option v-for="m in highChoices" :key="m" :value="m">{{ dirLabel(m) }}</option>
        </select>
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!highModel"
          title="Show model metadata"
          aria-label="Show model metadata"
          @click="$emit('showMetadata', { kind: 'wan_high_model', value: highModel })"
        >
          i
        </button>
      </div>
    </div>
  </div>

  <div class="quicksettings-group qs-group-wan-low">
    <label class="label-muted">WAN Low model</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select id="qs-wan-low" class="select-md" :value="lowModel" @change="$emit('update:lowModel', ($event.target as HTMLSelectElement).value)">
          <option value="">{{ builtInLabel }}</option>
          <option v-for="m in lowChoices" :key="m" :value="m">{{ dirLabel(m) }}</option>
        </select>
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!lowModel"
          title="Show model metadata"
          aria-label="Show model metadata"
          @click="$emit('showMetadata', { kind: 'wan_low_model', value: lowModel })"
        >
          i
        </button>
      </div>
    </div>
  </div>

  <div class="quicksettings-group qs-group-wan-text-encoder">
    <label class="label-muted">WAN Text Encoder</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select id="qs-wan-text-encoder" class="select-md" :value="textEncoder" @change="$emit('update:textEncoder', ($event.target as HTMLSelectElement).value)">
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
        <select id="qs-wan-vae" class="select-md" :value="vae" @change="$emit('update:vae', ($event.target as HTMLSelectElement).value)">
          <option value="">{{ builtInLabel }}</option>
          <option v-for="v in vaeChoices" :key="v" :value="v">{{ dirLabel(v) }}</option>
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
  lightx2v: boolean
  highModel: string
  highChoices: string[]
  lowModel: string
  lowChoices: string[]
  textEncoder: string
  textEncoderChoices: string[]
  vae: string
  vaeChoices: string[]
}>()

defineEmits<{
  (e: 'update:mode', value: 'txt2vid' | 'img2vid'): void
  (e: 'update:lightx2v', value: boolean): void
  (e: 'update:highModel', value: string): void
  (e: 'update:lowModel', value: string): void
  (e: 'update:textEncoder', value: string): void
  (e: 'update:vae', value: string): void
  (e: 'browseModels'): void
  (e: 'browseTe'): void
  (e: 'browseVae'): void
  (e: 'refresh'): void
  (e: 'showMetadata', payload: { kind: 'wan_high_model' | 'wan_low_model' | 'wan_text_encoder' | 'wan_vae'; value: string }): void
}>()

const builtInLabel = 'Select…'

function dirLabel(path: string): string {
  const norm = path.replace(/\\/g, '/')
  if (!norm) return ''
  const idx = norm.lastIndexOf('/')
  return idx >= 0 ? norm.slice(idx + 1) || norm : norm
}

function encoderLabel(value: string): string {
  const norm = String(value || '').replace(/\\/g, '/')
  if (!norm) return ''
  if (!norm.includes('/')) return norm
  const [family, ...rest] = norm.split('/').filter(Boolean)
  if (!family || rest.length === 0) return norm
  const tail = rest[rest.length - 1] || rest[0]
  // For file labels like wan22//abs/path/to/file.safetensors, show wan22/file.safetensors.
  return `${family}/${tail}`
}
</script>
