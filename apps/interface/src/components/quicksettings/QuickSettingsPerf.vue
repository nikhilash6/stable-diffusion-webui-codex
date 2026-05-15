<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Performance quicksettings toggles (smart flags + streaming + safe VRAM cleanup action).
Renders toggle buttons for Smart Offload/Fallback/Cache/Core Streaming plus an explicit "Obliterate VRAM" action
(internal cleanup by default), emitting updates to the parent quicksettings bar.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsPerf` (component): Performance toggles for smart runtime flags/core streaming plus a safe VRAM cleanup action.
-->

<template>
  <!-- Smart Offload -->
  <div class="quicksettings-group qs-group-perf-offload">
    <div class="qs-row">
      <button
        :class="['btn', 'qs-toggle-btn', { 'qs-toggle-btn--on': smartOffload, 'qs-toggle-btn--off': !smartOffload }]"
        type="button"
        :aria-pressed="smartOffload"
        title="Unload TE/UNet/VAE between stages to save VRAM"
        @click="$emit('update:smartOffload', !smartOffload)"
      >
        Smart Offload
      </button>
    </div>
  </div>

  <!-- Smart Fallback -->
  <div class="quicksettings-group qs-group-perf-fallback">
    <div class="qs-row">
      <button
        :class="['btn', 'qs-toggle-btn', { 'qs-toggle-btn--on': smartFallback, 'qs-toggle-btn--off': !smartFallback }]"
        type="button"
        :aria-pressed="smartFallback"
        title="Fallback to CPU when GPU runs out of memory"
        @click="$emit('update:smartFallback', !smartFallback)"
      >
        Smart Fallback
      </button>
    </div>
  </div>

  <!-- Smart Cache -->
  <div class="quicksettings-group qs-group-perf-cache">
    <div class="qs-row">
      <button
        :class="['btn', 'qs-toggle-btn', { 'qs-toggle-btn--on': smartCache, 'qs-toggle-btn--off': !smartCache }]"
        type="button"
        :aria-pressed="smartCache"
        title="Cache text encoder embeddings for faster subsequent generations"
        @click="$emit('update:smartCache', !smartCache)"
      >
        Smart Cache
      </button>
    </div>
  </div>

  <!-- Core Streaming -->
  <div class="quicksettings-group qs-group-perf-streaming">
    <div class="qs-row">
      <button
        :class="['btn', 'qs-toggle-btn', { 'qs-toggle-btn--on': coreStreaming, 'qs-toggle-btn--off': !coreStreaming }]"
        type="button"
        :aria-pressed="coreStreaming"
        title="Stream model blocks from RAM for large quantized models (GGUF)"
        @click="$emit('update:coreStreaming', !coreStreaming)"
      >
        Core Streaming
      </button>
    </div>
  </div>

  <!-- Obliterate VRAM -->
  <div class="quicksettings-group qs-group-perf-obliterate">
    <div class="qs-row">
      <button
        class="btn qs-toggle-btn qs-toggle-btn--off"
        type="button"
        :disabled="obliterateBusy"
        :title="obliterateBusy ? 'Obliterate VRAM is running' : 'Run internal VRAM cleanup now (external process termination is disabled by default)'"
        @click="$emit('obliterateVram')"
      >
        {{ obliterateBusy ? 'Obliterating...' : 'Obliterate VRAM' }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  smartOffload: boolean
  smartFallback: boolean
  smartCache: boolean
  coreStreaming: boolean
  obliterateBusy: boolean
}>()

defineEmits<{
  (e: 'update:smartOffload', value: boolean): void
  (e: 'update:smartFallback', value: boolean): void
  (e: 'update:smartCache', value: boolean): void
  (e: 'update:coreStreaming', value: boolean): void
  (e: 'obliterateVram'): void
}>()
</script>
