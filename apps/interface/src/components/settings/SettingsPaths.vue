<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Settings panel for model search paths (`/api/paths`).
Edits engine-specific checkpoint/VAE/LoRA/text-encoder/connectors roots (`sd15/sdxl/flux1/flux2/anima/ltx2/wan22`) plus dedicated IP-Adapter model/image-encoder roots,
using `PathList` to manage per-key lists and persisting them via the backend paths API.

Symbols (top-level; keep in sync; no ghosts):
- `SettingsPaths` (component): Settings panel for model and asset path roots.
- `getList` (function): Normalizes a raw key list from the backend paths payload.
- `reload` (function): Fetches and populates current paths from the backend.
- `save` (function): Persists the edited paths back to the backend.
-->

<template>
  <div class="space-y-4">
    <div class="panel-section">
      <h3 class="label-muted">SD 1.5</h3>
      <div class="space-y-2">
        <div>
          <label class="label-muted">Checkpoints</label>
          <PathList v-model="paths.sd15.ckpt" />
        </div>
        <div>
          <label class="label-muted">VAE</label>
          <PathList v-model="paths.sd15.vae" />
        </div>
        <div>
          <label class="label-muted">LoRA</label>
          <PathList v-model="paths.sd15.loras" />
        </div>
        <div>
          <label class="label-muted">Text Encoders</label>
          <PathList v-model="paths.sd15.tenc" />
        </div>
      </div>
    </div>

    <div class="panel-section">
      <h3 class="label-muted">SDXL</h3>
      <div class="space-y-2">
        <div>
          <label class="label-muted">Checkpoints</label>
          <PathList v-model="paths.sdxl.ckpt" />
        </div>
        <div>
          <label class="label-muted">VAE</label>
          <PathList v-model="paths.sdxl.vae" />
        </div>
        <div>
          <label class="label-muted">LoRA</label>
          <PathList v-model="paths.sdxl.loras" />
        </div>
        <div>
          <label class="label-muted">Text Encoders</label>
          <PathList v-model="paths.sdxl.tenc" />
        </div>
      </div>
    </div>

    <div class="panel-section">
      <h3 class="label-muted">FLUX.1</h3>
      <div class="space-y-2">
        <div>
          <label class="label-muted">Checkpoints</label>
          <PathList v-model="paths.flux1.ckpt" />
        </div>
        <div>
          <label class="label-muted">VAE</label>
          <PathList v-model="paths.flux1.vae" />
        </div>
        <div>
          <label class="label-muted">LoRA</label>
          <PathList v-model="paths.flux1.loras" />
        </div>
        <div>
          <label class="label-muted">Text Encoders</label>
          <PathList v-model="paths.flux1.tenc" />
        </div>
      </div>
    </div>

    <div class="panel-section">
      <h3 class="label-muted">FLUX.2</h3>
      <div class="space-y-2">
        <div>
          <label class="label-muted">Checkpoints</label>
          <PathList v-model="paths.flux2.ckpt" />
        </div>
        <div>
          <label class="label-muted">VAE</label>
          <PathList v-model="paths.flux2.vae" />
        </div>
        <div>
          <label class="label-muted">LoRA</label>
          <PathList v-model="paths.flux2.loras" />
        </div>
        <div>
          <label class="label-muted">Text Encoders</label>
          <PathList v-model="paths.flux2.tenc" />
        </div>
      </div>
    </div>

    <div class="panel-section">
      <h3 class="label-muted">Anima</h3>
      <div class="space-y-2">
        <div>
          <label class="label-muted">Checkpoints</label>
          <PathList v-model="paths.anima.ckpt" />
        </div>
        <div>
          <label class="label-muted">VAE</label>
          <PathList v-model="paths.anima.vae" />
        </div>
        <div>
          <label class="label-muted">LoRA</label>
          <PathList v-model="paths.anima.loras" />
        </div>
        <div>
          <label class="label-muted">Text Encoders</label>
          <PathList v-model="paths.anima.tenc" />
        </div>
      </div>
    </div>

    <div class="panel-section">
      <h3 class="label-muted">LTX 2.3</h3>
      <div class="space-y-2">
        <div>
          <label class="label-muted">Checkpoints</label>
          <PathList v-model="paths.ltx2.ckpt" />
        </div>
        <div>
          <label class="label-muted">VAE</label>
          <PathList v-model="paths.ltx2.vae" />
        </div>
        <div>
          <label class="label-muted">Connectors</label>
          <PathList v-model="paths.ltx2.connectors" />
        </div>
        <div>
          <label class="label-muted">LoRA</label>
          <PathList v-model="paths.ltx2.loras" />
        </div>
        <div>
          <label class="label-muted">Text Encoders</label>
          <PathList v-model="paths.ltx2.tenc" />
        </div>
      </div>
    </div>

    <div class="panel-section">
      <h3 class="label-muted">WAN22</h3>
      <div class="space-y-2">
        <div>
          <label class="label-muted">Checkpoints</label>
          <PathList v-model="paths.wan22.ckpt" />
        </div>
        <div>
          <label class="label-muted">VAE</label>
          <PathList v-model="paths.wan22.vae" />
        </div>
        <div>
          <label class="label-muted">LoRA</label>
          <PathList v-model="paths.wan22.loras" />
        </div>
        <div>
          <label class="label-muted">Text Encoders</label>
          <PathList v-model="paths.wan22.tenc" />
        </div>
      </div>
    </div>

    <div class="panel-section">
      <h3 class="label-muted">IP-Adapter</h3>
      <div class="space-y-2">
        <div>
          <label class="label-muted">Models</label>
          <PathList v-model="ipAdapterPaths.models" />
        </div>
        <div>
          <label class="label-muted">Image Encoders</label>
          <PathList v-model="ipAdapterPaths.imageEncoders" />
        </div>
      </div>
    </div>

    <div class="settings-paths-actions">
      <button class="btn btn-md btn-outline" type="button" @click="reload">Reload</button>
      <button class="btn btn-md btn-primary" type="button" @click="save">Save</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive } from 'vue'
import { fetchPaths, updatePaths } from '../../api/client'
import PathList from './widgets/PathList.vue'

type EngineId = 'sd15' | 'sdxl' | 'flux1' | 'flux2' | 'anima' | 'ltx2' | 'wan22'
type EnginePaths = { ckpt: string[]; vae: string[]; loras: string[]; tenc: string[]; connectors: string[] }
type EnginePathsState = Record<EngineId, EnginePaths>
type IpAdapterPaths = { models: string[]; imageEncoders: string[] }
type RawPaths = Record<string, string[]>

const paths = reactive<EnginePathsState>({
  sd15: { ckpt: [], vae: [], loras: [], tenc: [], connectors: [] },
  sdxl: { ckpt: [], vae: [], loras: [], tenc: [], connectors: [] },
  flux1: { ckpt: [], vae: [], loras: [], tenc: [], connectors: [] },
  flux2: { ckpt: [], vae: [], loras: [], tenc: [], connectors: [] },
  anima: { ckpt: [], vae: [], loras: [], tenc: [], connectors: [] },
  ltx2: { ckpt: [], vae: [], loras: [], tenc: [], connectors: [] },
  wan22: { ckpt: [], vae: [], loras: [], tenc: [], connectors: [] },
})

const ipAdapterPaths = reactive<IpAdapterPaths>({
  models: [],
  imageEncoders: [],
})

const rawPaths = reactive<RawPaths>({})

function getList(raw: RawPaths, key: string): string[] {
  const value = raw[key]
  return Array.isArray(value) ? [...value] : []
}

async function reload(): Promise<void> {
  try {
    const res = await fetchPaths()
    const loaded = (res.paths || {}) as RawPaths

    // Reset rawPaths and repopulate.
    for (const key of Object.keys(rawPaths)) {
      delete rawPaths[key]
    }
    for (const [key, value] of Object.entries(loaded)) {
      rawPaths[key] = Array.isArray(value) ? [...value] : []
    }

    paths.sd15.ckpt = getList(loaded, 'sd15_ckpt')
    paths.sd15.vae = getList(loaded, 'sd15_vae')
    paths.sd15.loras = getList(loaded, 'sd15_loras')
    paths.sd15.tenc = getList(loaded, 'sd15_tenc')

    paths.sdxl.ckpt = getList(loaded, 'sdxl_ckpt')
    paths.sdxl.vae = getList(loaded, 'sdxl_vae')
    paths.sdxl.loras = getList(loaded, 'sdxl_loras')
    paths.sdxl.tenc = getList(loaded, 'sdxl_tenc')

    paths.flux1.ckpt = getList(loaded, 'flux1_ckpt')
    paths.flux1.vae = getList(loaded, 'flux1_vae')
    paths.flux1.loras = getList(loaded, 'flux1_loras')
    paths.flux1.tenc = getList(loaded, 'flux1_tenc')

    paths.flux2.ckpt = getList(loaded, 'flux2_ckpt')
    paths.flux2.vae = getList(loaded, 'flux2_vae')
    paths.flux2.loras = getList(loaded, 'flux2_loras')
    paths.flux2.tenc = getList(loaded, 'flux2_tenc')

    paths.anima.ckpt = getList(loaded, 'anima_ckpt')
    paths.anima.vae = getList(loaded, 'anima_vae')
    paths.anima.loras = getList(loaded, 'anima_loras')
    paths.anima.tenc = getList(loaded, 'anima_tenc')
    paths.anima.connectors = []

    paths.ltx2.ckpt = getList(loaded, 'ltx2_ckpt')
    paths.ltx2.vae = getList(loaded, 'ltx2_vae')
    paths.ltx2.loras = getList(loaded, 'ltx2_loras')
    paths.ltx2.tenc = getList(loaded, 'ltx2_tenc')
    paths.ltx2.connectors = getList(loaded, 'ltx2_connectors')

    paths.wan22.ckpt = getList(loaded, 'wan22_ckpt')
    paths.wan22.vae = getList(loaded, 'wan22_vae')
    paths.wan22.loras = getList(loaded, 'wan22_loras')
    paths.wan22.tenc = getList(loaded, 'wan22_tenc')
    paths.wan22.connectors = []

    ipAdapterPaths.models = getList(loaded, 'ip_adapter_models')
    ipAdapterPaths.imageEncoders = getList(loaded, 'ip_adapter_image_encoders')
  } catch {
    // Keep existing state on failure; errors are surfaced elsewhere.
  }
}

async function save(): Promise<void> {
  const next: RawPaths = {}

  // Preserve non-aggregated keys not managed explicitly here.
  for (const [key, value] of Object.entries(rawPaths)) {
    if (key === 'checkpoints' || key === 'vae' || key === 'lora' || key === 'text_encoders') continue
    next[key] = Array.isArray(value) ? [...value] : []
  }

  next.sd15_ckpt = [...paths.sd15.ckpt]
  next.sd15_vae = [...paths.sd15.vae]
  next.sd15_loras = [...paths.sd15.loras]
  next.sd15_tenc = [...paths.sd15.tenc]

  next.sdxl_ckpt = [...paths.sdxl.ckpt]
  next.sdxl_vae = [...paths.sdxl.vae]
  next.sdxl_loras = [...paths.sdxl.loras]
  next.sdxl_tenc = [...paths.sdxl.tenc]

  next.flux1_ckpt = [...paths.flux1.ckpt]
  next.flux1_vae = [...paths.flux1.vae]
  next.flux1_loras = [...paths.flux1.loras]
  next.flux1_tenc = [...paths.flux1.tenc]

  next.flux2_ckpt = [...paths.flux2.ckpt]
  next.flux2_vae = [...paths.flux2.vae]
  next.flux2_loras = [...paths.flux2.loras]
  next.flux2_tenc = [...paths.flux2.tenc]

  next.anima_ckpt = [...paths.anima.ckpt]
  next.anima_vae = [...paths.anima.vae]
  next.anima_loras = [...paths.anima.loras]
  next.anima_tenc = [...paths.anima.tenc]

  next.ltx2_ckpt = [...paths.ltx2.ckpt]
  next.ltx2_vae = [...paths.ltx2.vae]
  next.ltx2_loras = [...paths.ltx2.loras]
  next.ltx2_tenc = [...paths.ltx2.tenc]
  next.ltx2_connectors = [...paths.ltx2.connectors]

  next.wan22_ckpt = [...paths.wan22.ckpt]
  next.wan22_vae = [...paths.wan22.vae]
  next.wan22_loras = [...paths.wan22.loras]
  next.wan22_tenc = [...paths.wan22.tenc]

  next.ip_adapter_models = [...ipAdapterPaths.models]
  next.ip_adapter_image_encoders = [...ipAdapterPaths.imageEncoders]

  await updatePaths(next)
}

onMounted(() => {
  void reload()
})
</script>
