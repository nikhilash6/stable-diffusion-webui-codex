<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Settings route view.
Fetches the settings schema/options and renders the Settings UI (settings form + paths panel), passing the current `/api/options` revision into the
form so settings writes fail loud on stale pages instead of overwriting newer option state.

Symbols (top-level; keep in sync; no ghosts):
- `Settings` (component): Settings route view component.
-->

<template>
  <section class="panels settings-page">
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header">Settings</div>
        <div class="panel-body" v-if="loaded">
          <div class="subtabs">
            <button v-for="c in categories" :key="c.id" class="subtab" :class="{active: c.id===activeCategory}" @click="activeCategory=c.id">{{ c.label }}</button>
          </div>
          <div class="subtabs thin" v-if="filteredSections.length>1">
            <button v-for="s in filteredSections" :key="s.key" class="subtab" :class="{active: s.key===activeSection}" @click="activeSection=s.key">{{ s.label }}</button>
          </div>

          <div class="settings-layout">
            <div class="left">
              <div class="toolbar">
                <input class="ui-input" v-model="q" placeholder="Search settings" />
              </div>
              <SettingsForm :fields="visibleFields" :values="values" :revision="revision" />
            </div>
            <div class="right">
              <div class="panel subtle">
                <div class="panel-header">Paths</div>
                <div class="panel-body">
                  <SettingsPaths />
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="panel-body" v-else>
          <div class="card text-xs opacity-70">Loading settings…</div>
        </div>
        <div class="panel-body" v-if="loaded && loadError">
          <div class="card text-xs settings-error">
            Failed to load settings. {{ loadError }}
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import SettingsPaths from '../components/settings/SettingsPaths.vue'
import SettingsForm from '../components/settings/SettingsForm.vue'
import { fetchSettingsSchema, fetchOptions } from '../api/client'
import type { SettingsCategory, SettingsField, SettingsSection } from '../api/types'

const loaded = ref(false)
const loadError = ref<string | null>(null)
const q = ref('')
const categories = ref<SettingsCategory[]>([])
const sections = ref<SettingsSection[]>([])
const fields = ref<SettingsField[]>([])
const values = ref<Record<string, unknown>>({})
const revision = ref(0)

const activeCategory = ref<string>('sd')
const activeSection = ref<string>('sd')

onMounted(async () => {
  try {
    const [schema, opts] = await Promise.all([fetchSettingsSchema(), fetchOptions()])
    categories.value = schema.categories
    sections.value = schema.sections
    fields.value = schema.fields
    values.value = opts.values
    revision.value = Number.isFinite((opts as any).revision) ? Math.max(0, Math.trunc((opts as any).revision)) : 0
    // Default selection
    activeCategory.value = schema.categories[0]?.id ?? 'sd'
    const firstSection = schema.sections.find(s => s.category_id === activeCategory.value) || schema.sections[0]
    activeSection.value = firstSection?.key ?? 'sd'
  } catch (e: any) {
    console.error('[settings] failed to load schema/options', e)
    loadError.value = String(e?.message || e)
  } finally {
    loaded.value = true
  }
})

const filteredSections = computed(() => sections.value.filter(s => s.category_id === activeCategory.value))

const visibleFields = computed(() => {
  const needle = q.value.toLowerCase().trim()
  const inSection = fields.value.filter(f => f.section === activeSection.value)
  if (!needle) return inSection
  return inSection.filter(f => f.label.toLowerCase().includes(needle) || f.key.toLowerCase().includes(needle))
})

</script>

<!-- view styles moved to styles/views/settings.css -->
