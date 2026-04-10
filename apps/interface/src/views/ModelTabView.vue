<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Dynamic model tab view (`/models/:tabId`).
Loads the selected tab from the tabs store and mounts the current video route selector or `ImageModelTab` for image families,
while distinguishing stale route ids from deferred tab-load failures.

Symbols (top-level; keep in sync; no ghosts):
- `ModelTabView` (component): Route view that mounts the correct model tab workspace.
- `tabLoadFailed` (computed): Tracks whether deferred model-tab loading failed for the current route.
- `VideoTabType` (type): Video tab types supported by `VideoTabRouteView`.
- `videoTabType` (computed): Normalized video tab type passed to `VideoTabRouteView`.
- `ImageTabType` (type): Non-video tab types supported by `ImageModelTab`.
- `imageTabType` (computed): Normalized non-video type passed to `ImageModelTab`.
-->

<template>
  <section v-if="tab">
    <VideoTabRouteView v-if="videoTabType" :tab-id="tab.id" :key="tab.id" />
    <ImageModelTab v-else-if="imageTabType" :tab-id="tab.id" :key="tab.id" :type="imageTabType" />
    <div v-else class="panel">
      <div class="panel-body">Unsupported tab type: {{ tab.type }}</div>
    </div>
  </section>
  <section v-else-if="tabLoadFailed">
    <div class="panel">
      <div class="panel-body">
        <p>Falha ao carregar abas do modelo.</p>
        <p v-if="bootstrap.deferredMessage" class="caption">{{ bootstrap.deferredMessage }}</p>
      </div>
    </div>
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab não encontrada.</div></div>
  </section>
</template>

<script setup lang="ts">
import { computed, watch } from 'vue'
import { useRoute } from 'vue-router'
import ImageModelTab from './ImageModelTab.vue'
import VideoTabRouteView from './VideoTabRouteView.vue'
import { useBootstrapStore } from '../stores/bootstrap'
import { useModelTabsStore, type BaseTabType } from '../stores/model_tabs'
import { isWanTabFamily } from '../utils/engine_taxonomy'

const route = useRoute()
const bootstrap = useBootstrapStore()
const store = useModelTabsStore()

const id = computed(() => String(route.params.tabId || ''))
const tab = computed(() => store.tabs.find(t => t.id === id.value) || null)
const tabLoadFailed = computed(() => !tab.value && bootstrap.deferredStatus === 'error')

type VideoTabType = Extract<BaseTabType, 'wan22_14b' | 'wan22_5b' | 'ltx2'>
type ImageTabType = Exclude<BaseTabType, 'wan22_14b' | 'wan22_5b' | 'ltx2'>

const videoTabType = computed<VideoTabType | null>(() => {
  const t = tab.value?.type
  if (t === 'ltx2' || isWanTabFamily(t)) return t
  return null
})

const imageTabType = computed<ImageTabType | null>(() => {
  const t = tab.value?.type
  if (!t || isWanTabFamily(t) || t === 'ltx2') return null
  return t
})

watch(id, (nextId) => {
  if (nextId) store.setActive(nextId)
}, { immediate: true })
</script>
