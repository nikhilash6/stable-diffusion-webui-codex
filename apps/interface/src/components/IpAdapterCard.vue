<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Presentational IP-Adapter configuration card for image tabs.
Provides a dedicated card with one nested `ipAdapter` owner prop, enable toggle, asset selectors, source-mode switching (`DIR|IMG`), same-as-init shortcut, optional reference-image dropzone, folder-source controls, weight range sliders, and an optional blocking-reason notice without owning store or API state.

Symbols (top-level; keep in sync; no ghosts):
- `IpAdapterCard` (component): Dedicated IP-Adapter UI card for supported image tabs.
-->

<template>
  <div class="gen-card ip-adapter-card">
    <WanSubHeader title="IP-Adapter">
      <button
        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', ipAdapter.enabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :aria-pressed="ipAdapter.enabled"
        @click="emit('patch:ipAdapter', { enabled: !ipAdapter.enabled })"
      >
        {{ ipAdapter.enabled ? 'Enabled' : 'Disabled' }}
      </button>
    </WanSubHeader>

    <p v-if="ipAdapter.enabled && blockingReason" class="caption ip-adapter-card__hint">
      {{ blockingReason }}
    </p>

    <div v-if="ipAdapter.enabled" class="ip-adapter-card__body">
      <div class="gc-row ip-adapter-card__selectors">
        <div class="field gc-col gc-col--wide">
          <label class="label-muted">Adapter model</label>
          <select
            class="select-md"
            :disabled="disabled"
            :value="ipAdapter.model"
            @change="emit('patch:ipAdapter', { model: ($event.target as HTMLSelectElement).value })"
          >
            <option value="">Select IP-Adapter model</option>
            <option v-for="choice in modelChoices" :key="choice.value" :value="choice.value">
              {{ choice.label }}
            </option>
          </select>
        </div>

        <div class="field gc-col gc-col--wide">
          <label class="label-muted">Image encoder</label>
          <select
            class="select-md"
            :disabled="disabled"
            :value="ipAdapter.imageEncoder"
            @change="emit('patch:ipAdapter', { imageEncoder: ($event.target as HTMLSelectElement).value })"
          >
            <option value="">Select image encoder</option>
            <option v-for="choice in imageEncoderChoices" :key="choice.value" :value="choice.value">
              {{ choice.label }}
            </option>
          </select>
        </div>
      </div>

      <div class="gc-row ip-adapter-card__sliders">
        <SliderField
          class="gc-col gc-col--wide"
          label="Weight"
          :modelValue="ipAdapter.weight"
          :min="0"
          :max="2"
          :step="0.01"
          :inputStep="0.01"
          inputClass="cdx-input-w-xs"
          :disabled="disabled"
          @update:modelValue="(value) => emit('patch:ipAdapter', { weight: value })"
        />
        <SliderField
          class="gc-col gc-col--wide"
          label="Start"
          :modelValue="ipAdapter.startAt"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          inputClass="cdx-input-w-xs"
          :disabled="disabled"
          @update:modelValue="(value) => emit('patch:ipAdapter', { startAt: value })"
        />
        <SliderField
          class="gc-col gc-col--wide"
          label="End"
          :modelValue="ipAdapter.endAt"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          inputClass="cdx-input-w-xs"
          :disabled="disabled"
          @update:modelValue="(value) => emit('patch:ipAdapter', { endAt: value })"
        />
      </div>

      <ImageSourceBlock
        :mode="ipAdapter.source.mode"
        :folder-source="ipAdapter.source"
        :disabled="disabled"
        show-source-mode-toggle
        show-source-mode-label
        source-mode-label="Source"
        source-mode-aria-label="IP-Adapter source mode"
        :show-image-picker="!ipAdapter.source.sameAsInit"
        image-label="Reference Image"
        :image-src="ipAdapter.source.referenceImageData"
        :has-image="Boolean(ipAdapter.source.referenceImageData)"
        :thumbnail="true"
        :zoomable="true"
        folder-path-label="Folder path"
        folder-path-placeholder="input/ip-adapter-source"
        folder-count-label="Reference images"
        @update:mode="(value) => emit('patch:ipAdapter', { source: { mode: value } })"
        @patch:folderSource="(value) => emit('patch:ipAdapter', { source: value })"
        @set:image="(file) => emit('set:referenceImage', file)"
        @clear:image="() => emit('clear:referenceImage')"
        @reject:image="(payload) => emit('reject:referenceImage', payload)"
      >
        <template #controls-extra>
          <div v-if="img2imgMode" class="field ip-adapter-card__same-init-field">
            <label class="label-muted">Shortcut</label>
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', ipAdapter.source.sameAsInit ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :aria-pressed="ipAdapter.source.sameAsInit"
              :disabled="disabled || ipAdapter.source.mode !== 'img'"
              @click="emit('patch:ipAdapter', { source: { sameAsInit: !ipAdapter.source.sameAsInit } })"
            >
              Same as init image
            </button>
          </div>
        </template>
        <template #img-empty>
          <p class="caption ip-adapter-card__hint">
            Uses the current init image as the IP-Adapter reference image.
          </p>
        </template>
      </ImageSourceBlock>
    </div>
  </div>
</template>

<script setup lang="ts">
import ImageSourceBlock from './ImageSourceBlock.vue'
import SliderField from './ui/SliderField.vue'
import WanSubHeader from './wan/WanSubHeader.vue'
import type { IpAdapterFormState } from '../stores/model_tabs'

type SelectChoice = {
  value: string
  label: string
}

type IpAdapterPatch = Partial<Omit<IpAdapterFormState, 'source'>> & {
  source?: Partial<IpAdapterFormState['source']>
}

withDefaults(defineProps<{
  disabled?: boolean
  img2imgMode?: boolean
  ipAdapter: IpAdapterFormState
  modelChoices: SelectChoice[]
  imageEncoderChoices: SelectChoice[]
  blockingReason?: string
}>(), {
  disabled: false,
  img2imgMode: false,
  blockingReason: '',
})

const emit = defineEmits<{
  (e: 'patch:ipAdapter', value: IpAdapterPatch): void
  (e: 'set:referenceImage', value: File): void
  (e: 'clear:referenceImage'): void
  (e: 'reject:referenceImage', payload: { reason: string; files: File[] }): void
}>()
</script>
