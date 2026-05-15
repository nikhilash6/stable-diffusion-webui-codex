<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Editable list widget for path arrays.
Provides add/remove controls for a list of paths and a best-effort “Browse…” helper (uses the browser directory picker API when available).

Symbols (top-level; keep in sync; no ghosts):
- `PathList` (component): Path list editor used by `SettingsPaths.vue`.
- `add` (function): Adds a path from the input to the list (deduped).
- `remove` (function): Removes a path entry by index.
- `dedupe` (function): Deduplicates paths using a normalized key.
- `browse` (function): Best-effort directory picker helper (falls back to an alert when unsupported).
-->

<template>
    <div>
    <div class="pathlist-controls">
      <input class="ui-input" v-model="newPath" placeholder="Add path (or use Browse)" />
      <button class="btn btn-sm btn-secondary" type="button" @click="add">Add</button>
      <button class="btn btn-sm btn-outline" type="button" @click="browse">Browse…</button>
    </div>
    <ul class="cdx-list">
      <li v-for="(p, idx) in modelValue" :key="p+idx" class="cdx-list-item">
        <span class="text-sm break-all">{{ p }}</span>
        <button class="btn btn-sm btn-ghost" type="button" @click="remove(idx)">Remove</button>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ modelValue: string[] }>()
const emit = defineEmits<{ (e:'update:modelValue', v:string[]): void }>()
const newPath = ref('')

function add(): void {
  const v = newPath.value.trim()
  if (!v) return
  const list = [...props.modelValue, v]
  emit('update:modelValue', dedupe(list))
  newPath.value = ''
}
function remove(idx: number): void {
  const list = props.modelValue.slice()
  list.splice(idx, 1)
  emit('update:modelValue', list)
}
function dedupe(arr: string[]): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const p of arr) {
    const k = p.replace(/\\+/g, '/').replace(/\/$/, '')
    if (!seen.has(k)) { seen.add(k); out.push(p) }
  }
  return out
}

async function browse(): Promise<void> {
  try {
    // Chrome/Edge
    // @ts-expect-error experimental API
    if (window.showDirectoryPicker) {
      // @ts-expect-error experimental API
      const handle = await window.showDirectoryPicker()
      // NOTE: path is not directly available; we store name as a placeholder
      if (handle && handle.name) {
        const guess = handle.name
        const list = [...props.modelValue, guess]
        emit('update:modelValue', dedupe(list))
      }
      return
    }
  } catch {}
  // Fallback: ask user to paste path into input
  alert('Directory picker not supported in this browser. Paste the path in the input and click Add.')
}
</script>
