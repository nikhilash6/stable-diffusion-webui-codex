<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model tabs management view.
Lists model tabs and provides actions to create/open/duplicate/remove tabs (including capability-gated Z-Image L2P/Anima/LTX2 tabs when supported).

Symbols (top-level; keep in sync; no ghosts):
- `ModelsList` (component): View for managing model tabs.
-->

<template>
  <section class="panel-stack">
    <div class="panel">
      <div class="panel-header">Model Tabs
        <div class="toolbar">
          <select class="select-md" v-model="newType" aria-label="New tab type">
            <option value="sd15">SD 1.5</option>
            <option value="sdxl">SDXL</option>
            <option value="flux1">FLUX.1</option>
            <option value="flux2">FLUX.2</option>
            <option value="zimage">Z Image</option>
            <option v-if="showZImageL2POption" value="zimage_l2p">Z-Image L2P</option>
            <option v-if="showAnimaOption" value="anima">Anima</option>
            <option v-if="showLtx2Option" value="ltx2">LTX 2.3</option>
            <option value="wan22_14b">WAN 2.2 14B</option>
            <option value="wan22_5b">WAN 2.2 5B</option>
          </select>
          <button class="btn btn-sm btn-primary" type="button" @click="createTab">New Tab</button>
        </div>
      </div>
      <div class="panel-body">
        <p v-if="modelsNotice" class="caption">{{ modelsNotice }}</p>
        <p v-if="!tabs.length" class="caption">No tabs yet. Create one above.</p>
        <ul v-else class="cdx-list">
          <li v-for="t in tabs" :key="t.id" class="cdx-list-item">
            <div class="cdx-list-main">
              <div class="cdx-list-title">{{ t.title }}</div>
              <div class="cdx-list-meta">{{ t.type.toUpperCase() }} · {{ t.id }}</div>
            </div>
            <div class="cdx-list-actions">
              <RouterLink class="btn btn-sm btn-outline" :to="`/models/${t.id}`">Open</RouterLink>
              <button class="btn btn-sm btn-secondary" type="button" @click="dup(t.id)">Duplicate</button>
              <button class="btn btn-sm btn-destructive" type="button" @click="remove(t.id)">Remove</button>
            </div>
          </li>
        </ul>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
	import { ref, onMounted, computed } from 'vue'
	import { useRouter } from 'vue-router'
	import { useModelTabsStore, type BaseTabType } from '../stores/model_tabs'
	import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
  import { useResultsCard } from '../composables/useResultsCard'

	const router = useRouter()
	const store = useModelTabsStore()
	const engineCaps = useEngineCapabilitiesStore()
const newType = ref<BaseTabType>('sdxl')
  const { notice: modelsNotice, toast: modelsToast } = useResultsCard({ noticeDurationMs: 4000 })

	onMounted(async () => {
	  try {
	    await engineCaps.init()
	    await store.load()
	  } catch (error) {
	    modelsToast(error instanceof Error ? error.message : String(error))
	  }
	})

const tabs = computed(() => store.orderedTabs)
const showZImageL2POption = computed(() => Boolean(engineCaps.get('zimage_l2p')))
const showAnimaOption = computed(() => Boolean(engineCaps.get('anima')))
const showLtx2Option = computed(() => Boolean(engineCaps.get('ltx2')))

async function createTab(): Promise<void> {
  try {
    if (newType.value === 'anima' && !showAnimaOption.value) {
      const msg = "Cannot create Anima tab: '/api/engines/capabilities' does not expose 'anima'."
      console.error(`[ModelsList] ${msg}`)
      throw new Error(msg)
    }
    if (newType.value === 'zimage_l2p' && !showZImageL2POption.value) {
      const msg = "Cannot create Z-Image L2P tab: '/api/engines/capabilities' does not expose 'zimage_l2p'."
      console.error(`[ModelsList] ${msg}`)
      throw new Error(msg)
    }
    if (newType.value === 'ltx2' && !showLtx2Option.value) {
      const msg = "Cannot create LTX 2.3 tab: '/api/engines/capabilities' does not expose 'ltx2'."
      console.error(`[ModelsList] ${msg}`)
      throw new Error(msg)
    }
    const id = await store.create(newType.value)
    if (id) void router.push(`/models/${id}`)
  } catch (error) {
    modelsToast(error instanceof Error ? error.message : String(error))
  }
}

async function dup(id: string): Promise<void> {
  try {
    await store.duplicate(id)
  } catch (error) {
    modelsToast(error instanceof Error ? error.message : String(error))
  }
}

async function remove(id: string): Promise<void> {
  try {
    await store.remove(id)
  } catch (error) {
    modelsToast(error instanceof Error ? error.message : String(error))
  }
}
</script>
