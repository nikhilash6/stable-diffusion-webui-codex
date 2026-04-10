<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Engine-agnostic landing page and model-tab manager UI with global dependency overview.
Renders the Home workspace, exposes one dependency-check surface for all engines, creates new tabs, and manages existing tabs
(enable/disable, rename, load/unload, duplicate, remove) while linking users to model tabs (`/models/:tabId`), gallery/workflows,
and utilities; XYZ guidance points to the embedded Generation Parameters card (with `/xyz` compatibility route).

Symbols (top-level; keep in sync; no ghosts):
- `Home` (component): Home workspace + tab manager; drives tab CRUD and status actions via stores/API (contains nested UI helpers and dialogs).
- `HelpTopic` (type): Help topic selector for the contextual help panel.
- `dependencyChecks` (computed): Home dependency map sourced from engine capabilities store.
- `dependencyLabels` (computed): Engine label map for dependency panel display and deterministic ordering.
- `dependencyError` (ref): Fatal capabilities-init error shown in the global dependency panel.
- `onCreate` (function): Creates a new model tab for the selected engine type (optional title; includes capability-gated Anima/LTX2 when supported).
- `setTitleDraft` (function): Updates the in-memory title draft for a tab row (before persisting).
- `commitTitle` (function): Persists a tab title edit to the backend/store.
- `setEnabled` (function): Toggles a tab enabled/disabled state and persists the change.
- `load` (function): Loads/activates a tab (backend load when required).
- `unload` (function): Unloads/deactivates a tab (backend unload when supported).
- `dup` (function): Duplicates an existing tab (clones params into a new id).
- `remove` (function): Deletes a tab and removes it from the store.
- `setHelpTopic` (function): Switches the active help topic in the UI.
-->

<template>
  <section class="panels">
    <div class="panel">
      <div class="panel-header">Welcome</div>
      <div class="panel-body">
        <p class="subtitle">
          This home workspace is engine-agnostic. Use it to create and manage model tabs (SD 1.5, SDXL, FLUX.1, FLUX.2, Z Image, Anima, LTX 2.3, WAN 2.2)
          and to navigate to workflows or utilities. Generation happens in tabs and workflows, not here.
        </p>

        <ul class="cdx-list">
          <li class="cdx-list-item">
            <div class="cdx-list-main">
              <div class="cdx-list-title">Model Tabs</div>
              <div class="cdx-list-meta">
                Create one or more tabs per engine (e.g., several WAN 2.2 tabs for different model dirs) and open them under
                <code>/models/:tabId</code>. Tabs persist their own parameters.
              </div>
            </div>
          </li>
          <li class="cdx-list-item">
            <div class="cdx-list-main">
              <div class="cdx-list-title">Workflows</div>
              <div class="cdx-list-meta">Use the Workflows view to inspect or run saved workflows built from tab snapshots.</div>
            </div>
          </li>
          <li class="cdx-list-item">
            <div class="cdx-list-main">
              <div class="cdx-list-title">XYZ</div>
              <div class="cdx-list-meta">Run frontend-driven XYZ sweeps from Generation Parameters inside image model tabs (route <code>/xyz</code> remains available for compatibility).</div>
            </div>
          </li>
        </ul>
      </div>
    </div>

    <DependencyCheckPanel
      :statuses="dependencyChecks"
      :labels="dependencyLabels"
      :loading="dependencyLoading"
      :error="dependencyError"
      title="Dependency Check (All Engines)"
    />

    <div class="panel">
      <div class="panel-header">Create Model Tab</div>
      <div class="panel-body">
        <p class="subtitle">
          Choose an engine type and an optional title. Tabs are identified by a generated id and can be duplicated or removed later.
        </p>

        <form class="gen-card" @submit.prevent="onCreate">
          <div class="two-up">
            <div class="field">
              <label class="label-muted" for="engineType">Engine</label>
              <select id="engineType" class="select-md" v-model="newType">
                <option value="sd15">SD 1.5</option>
                <option value="sdxl">SDXL</option>
                <option value="flux1">FLUX.1</option>
                <option value="flux2">FLUX.2</option>
                <option value="zimage">Z Image</option>
                <option v-if="showAnimaOption" value="anima">Anima</option>
                <option v-if="showLtx2Option" value="ltx2">LTX 2.3</option>
                <option value="wan22_14b">WAN 2.2 14B</option>
                <option value="wan22_5b">WAN 2.2 5B</option>
              </select>
            </div>
            <div class="field">
              <label class="label-muted" for="tabTitle">Title (optional)</label>
              <input id="tabTitle" class="ui-input" type="text" v-model="newTitle" placeholder="e.g. WAN — main video rig" />
            </div>
          </div>
          <div class="row-inline">
            <button class="btn btn-md btn-primary" type="submit">Create Tab</button>
          </div>
        </form>
        <div v-if="createError" class="panel-error">{{ createError }}</div>

        <div class="panel-section">
          <div class="panel-section-title">Existing Tabs</div>
          <p v-if="!tabs.length" class="caption">No tabs yet. Create one above.</p>
          <div v-else class="panel-section">
            <div v-for="t in tabs" :key="t.id" class="gen-card">
              <div class="row-split">
                <div class="row-inline">
                  <div class="h3">{{ t.title }}</div>
                  <span class="caption">{{ t.type.toUpperCase() }}</span>
                </div>
                <button
                  :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', t.enabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                  type="button"
                  :aria-pressed="t.enabled"
                  :title="t.enabled ? 'Enabled' : 'Disabled'"
                  :disabled="tabBusy[t.id]"
                  @click="setEnabled(t.id, !t.enabled)"
                >
                  Enabled
                </button>
              </div>

              <div class="two-up">
                <div class="field">
                  <label class="label-muted">Title</label>
                  <input
                    class="ui-input"
                    type="text"
                    :disabled="tabBusy[t.id]"
                    :value="titleDraft[t.id] ?? t.title"
                    @input="setTitleDraft(t.id, ($event.target as HTMLInputElement).value)"
                    @change="commitTitle(t.id)"
                    @blur="commitTitle(t.id)"
                    placeholder="Tab title"
                    aria-label="Tab title"
                  />
                </div>
                <div class="field">
                  <label class="label-muted">Models</label>
                  <div class="row-inline">
                    <button class="btn btn-sm btn-secondary" type="button" :disabled="tabBusy[t.id]" @click="load(t.id)">Load</button>
                    <button class="btn btn-sm btn-secondary" type="button" :disabled="tabBusy[t.id]" @click="unload(t.id)">Unload</button>
                    <button class="btn btn-sm btn-outline" type="button" :disabled="tabBusy[t.id]" @click="void router.push(`/models/${t.id}`)">Open</button>
                  </div>
                </div>
              </div>

              <div v-if="tabError[t.id]" class="panel-error">{{ tabError[t.id] }}</div>

              <div class="row-inline">
                <button class="btn btn-sm btn-outline" type="button" :disabled="tabBusy[t.id]" @click="dup(t.id)">Duplicate</button>
                <button class="btn btn-sm btn-destructive" type="button" :disabled="tabBusy[t.id]" @click="remove(t.id)">Remove</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">Docs &amp; Help</div>
      <div class="panel-body">
        <p class="subtitle">
          The paths below refer to files in this repository and act as the canonical documentation for the Codex UI.
          Open them in your editor when you need deeper guidance. A short Markdown help snippet is also loaded from
          <code>apps/interface/public/help/home-overview.md</code>.
        </p>

        <div class="panel-section">
          <div class="panel-section-title">Design &amp; Flows</div>
          <ul class="cdx-list">
            <li class="cdx-list-item">
              <div class="cdx-list-main">
                <div class="cdx-list-title">Model tabs &amp; workflows spec</div>
                <div class="cdx-list-meta"><code>apps/interface/public/help/workflows-basics.md</code></div>
              </div>
            </li>
            <li class="cdx-list-item">
              <div class="cdx-list-main">
                <div class="cdx-list-title">Frontend architecture guide</div>
                <div class="cdx-list-meta"><code>apps/interface/README.md</code></div>
              </div>
            </li>
            <li class="cdx-list-item">
              <div class="cdx-list-main">
                <div class="cdx-list-title">Frontend style guide</div>
                <div class="cdx-list-meta"><code>apps/interface/src/styles/AGENTS.md</code></div>
              </div>
            </li>
          </ul>
        </div>

        <div class="panel-section">
          <div class="panel-section-title">Tabs &amp; Workflows Tasks</div>
          <ul class="cdx-list">
            <li class="cdx-list-item">
              <div class="cdx-list-main">
                <div class="cdx-list-title">Views/components ownership docs</div>
                <div class="cdx-list-meta">
                  <code>apps/interface/src/views/AGENTS.md</code>, <code>apps/interface/src/components/AGENTS.md</code>
                </div>
              </div>
            </li>
            <li class="cdx-list-item">
              <div class="cdx-list-main">
                <div class="cdx-list-title">State/api ownership docs</div>
                <div class="cdx-list-meta"><code>apps/interface/src/stores/AGENTS.md</code>, <code>apps/interface/src/api/AGENTS.md</code></div>
              </div>
            </li>
          </ul>
        </div>

        <div class="panel-section">
          <div class="panel-section-title">Operational References</div>
          <ul class="cdx-list">
            <li class="cdx-list-item">
              <div class="cdx-list-main">
                <div class="cdx-list-title">Repository structure</div>
                <div class="cdx-list-meta"><code>SUBSYSTEM-MAP.md</code></div>
              </div>
            </li>
            <li class="cdx-list-item">
              <div class="cdx-list-main">
                <div class="cdx-list-title">Top-level ownership docs</div>
                <div class="cdx-list-meta"><code>apps/AGENTS.md</code>, <code>apps/interface/AGENTS.md</code></div>
              </div>
            </li>
          </ul>
        </div>

        <div class="panel-section">
          <div class="panel-section-title">Inline help (Markdown)</div>
          <div class="toolbar">
            <button
              class="btn btn-sm"
              :class="helpTopic === 'home' ? 'btn-secondary' : 'btn-ghost'"
              type="button"
              @click="setHelpTopic('home')"
            >
              Home
            </button>
            <button
              class="btn btn-sm"
              :class="helpTopic === 'wan22' ? 'btn-secondary' : 'btn-ghost'"
              type="button"
              @click="setHelpTopic('wan22')"
            >
              WAN22 video
            </button>
            <button
              class="btn btn-sm"
              :class="helpTopic === 'workflows' ? 'btn-secondary' : 'btn-ghost'"
              type="button"
              @click="setHelpTopic('workflows')"
            >
              Workflows
            </button>
          </div>
          <MarkdownHelp :src="helpSrc" />
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref, computed, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { useModelTabsStore, type BaseTabType } from '../stores/model_tabs'
import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
import MarkdownHelp from '../components/MarkdownHelp.vue'
import DependencyCheckPanel from '../components/DependencyCheckPanel.vue'
import { loadModelsForTab, unloadModelsForTab } from '../api/client'

type HelpTopic = 'home' | 'wan22' | 'workflows'

const router = useRouter()
const store = useModelTabsStore()
const engineCaps = useEngineCapabilitiesStore()

const newType = ref<BaseTabType>('sdxl')
const newTitle = ref('')
const helpTopic = ref<HelpTopic>('home')
const createError = ref('')
const dependencyError = ref('')
const titleDraft = reactive<Record<string, string>>({})
const tabBusy = reactive<Record<string, boolean>>({})
const tabError = reactive<Record<string, string>>({})

onMounted(async () => {
  try {
    await engineCaps.init()
  } catch (error) {
    dependencyError.value = error instanceof Error ? error.message : String(error)
  }
  try {
    await store.load()
    for (const t of store.orderedTabs) titleDraft[t.id] = t.title
  } catch (error) {
    createError.value = error instanceof Error ? error.message : String(error)
  }
})

const tabs = computed(() => store.orderedTabs.filter((tab) => tab.type !== 'chroma'))
const showAnimaOption = computed(() => Boolean(engineCaps.get('anima')))
const showLtx2Option = computed(() => Boolean(engineCaps.get('ltx2')))
const dependencyChecks = computed(() => engineCaps.dependencyChecks)
const dependencyLoading = computed(() => !engineCaps.loaded && !dependencyError.value)
const dependencyLabels = computed<Record<string, string>>(() => {
  const out: Record<string, string> = {}
  for (const engine of Object.keys(engineCaps.dependencyChecks)) {
    out[engine] = engine
  }
  return out
})
const helpSrc = computed(() => {
  if (helpTopic.value === 'wan22') return '/help/wan22-quickstart.md'
  if (helpTopic.value === 'workflows') return '/help/workflows-basics.md'
  return '/help/home-overview.md'
})

async function onCreate(): Promise<void> {
  createError.value = ''
  try {
    if (newType.value === 'anima' && !showAnimaOption.value) {
      const msg = "Cannot create Anima tab: '/api/engines/capabilities' does not expose 'anima'."
      console.error(`[Home] ${msg}`)
      throw new Error(msg)
    }
    if (newType.value === 'ltx2' && !showLtx2Option.value) {
      const msg = "Cannot create LTX 2.3 tab: '/api/engines/capabilities' does not expose 'ltx2'."
      console.error(`[Home] ${msg}`)
      throw new Error(msg)
    }
    const id = await store.create(newType.value, newTitle.value.trim() || undefined)
    newTitle.value = ''
    if (id) void router.push(`/models/${id}`)
  } catch (error) {
    createError.value = error instanceof Error ? error.message : String(error)
  }
}

function setTitleDraft(id: string, v: string): void {
  titleDraft[id] = v
}

async function commitTitle(id: string): Promise<void> {
  const t = store.tabs.find(x => x.id === id)
  if (!t) return
  const next = String(titleDraft[id] ?? t.title).trim()
  if (!next || next === t.title) {
    titleDraft[id] = t.title
    return
  }
  tabError[id] = ''
  tabBusy[id] = true
  try {
    await store.rename(id, next)
  } catch (e) {
    tabError[id] = e instanceof Error ? e.message : String(e)
  } finally {
    tabBusy[id] = false
  }
}

async function setEnabled(id: string, enabled: boolean): Promise<void> {
  tabError[id] = ''
  tabBusy[id] = true
  try {
    await store.setEnabled(id, enabled)
  } catch (e) {
    tabError[id] = e instanceof Error ? e.message : String(e)
  } finally {
    tabBusy[id] = false
  }
}

async function load(id: string): Promise<void> {
  tabError[id] = ''
  tabBusy[id] = true
  try {
    await loadModelsForTab(id)
  } catch (e) {
    tabError[id] = e instanceof Error ? e.message : String(e)
  } finally {
    tabBusy[id] = false
  }
}

async function unload(id: string): Promise<void> {
  tabError[id] = ''
  tabBusy[id] = true
  try {
    await unloadModelsForTab(id)
  } catch (e) {
    tabError[id] = e instanceof Error ? e.message : String(e)
  } finally {
    tabBusy[id] = false
  }
}

async function dup(id: string): Promise<void> {
  tabError[id] = ''
  tabBusy[id] = true
  try {
    const newId = await store.duplicate(id)
    if (newId) titleDraft[newId] = store.tabs.find(t => t.id === newId)?.title ?? ''
  } catch (e) {
    tabError[id] = e instanceof Error ? e.message : String(e)
  } finally {
    tabBusy[id] = false
  }
}

async function remove(id: string): Promise<void> {
  tabError[id] = ''
  tabBusy[id] = true
  try {
    await store.remove(id)
  } catch (e) {
    tabError[id] = e instanceof Error ? e.message : String(e)
  } finally {
    tabBusy[id] = false
  }
}

function setHelpTopic(topic: HelpTopic): void {
  helpTopic.value = topic
}
</script>
