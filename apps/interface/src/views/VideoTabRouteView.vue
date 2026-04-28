<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Temporary route-level selector for current video workspaces under `/models/:tabId`.
Keeps route selection separate from the baseline video workspace owner while current video families still resolve through the canonical video body,
and fails loud on impossible tab-type drift.

Symbols (top-level; keep in sync; no ghosts):
- `VideoTabRouteView` (component): Temporary route-level selector for current video workspaces.
- `VideoTabType` (type): Supported video tab types handled here.
- `videoTabType` (computed): Current video tab type when the selected tab belongs to the video lane.
-->

<template>
  <section v-if="tab">
    <VideoModelTab v-if="videoTabType" :tab-id="tab.id" :key="tab.id" />
    <div v-else class="panel">
      <div class="panel-body">Unsupported video tab type: {{ tab.type }}</div>
    </div>
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab não encontrada.</div></div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import VideoModelTab from './VideoModelTab.vue'
import { useModelTabsStore } from '../stores/model_tabs'
import { isVideoTabFamily, type VideoTabFamily } from '../utils/engine_taxonomy'

const props = defineProps<{
  tabId: string
}>()

const store = useModelTabsStore()

type VideoTabType = VideoTabFamily

const tab = computed(() => store.tabs.find((entry) => entry.id === props.tabId) || null)
const videoTabType = computed<VideoTabType | null>(() => {
  const value = tab.value?.type
  if (isVideoTabFamily(value)) return value
  return null
})
</script>
