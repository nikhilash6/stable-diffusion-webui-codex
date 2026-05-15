<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Unified public owner for live initial-image workflow surfaces.
Groups the shared constrained init-image preview/upload block with optional source-mode controls, optional inpaint controls,
and optional WAN frame-guide support so img2img/inpaint and img2vid can mount one parameterized owner instead of separate card components.
When `Per-step blend` is active, the strength/step-limit sliders share one proportional desktop row instead of stacking as separate full-width rows.
Supports optional pass-through WAN zoom frame-guide config for init-image overlays.
Saved inpaint masks can preview their effective mask, outward blur-spill range, and effective masked-region crop directly on the init-image thumbnail, anchored to the same image-bounds wrapper as the base preview, with footer legend copy that makes the final blue crop box explicit in the UI.
Keeps `Split mask regions` / `Invert mask` interlocked in the card and suppresses preview/editor entry until truthful natural init-image + processing dimensions are available.
Init-image filename captions are centered in the footer area for clearer media identification.

Symbols (top-level; keep in sync; no ghosts):
- `InitialImageBlock` (component): Unified initial-image owner for img2img/inpaint and img2vid surfaces.
- `showSourceModeToggle` (prop): Controls whether the composed shared image-source surface exposes `DIR|IMG` source ownership.
- `showInpaintControls` (prop): Controls whether inpaint editor/runtime controls are available in the block.
- `showFrameGuideEditor` (prop): Controls whether WAN frame-guide metadata is forwarded to the zoom overlay.
- `initSource` (prop): Nested init-image source owner (`DIR|IMG` + folder settings) rendered when source-mode toggling is enabled.
- `INPAINT_PARAMETER_TOOLTIPS` (constant): Tooltip copy for inpaint select, slider, and split-toggle controls.
- `perStepBlendStrength` (prop): Scales how strongly `Per-step blend` restores preserved outside-mask content each outer sampling step.
- `perStepBlendSteps` (prop): Limits how many outer sampling steps `Per-step blend` stays active before the final preserved-content close.
- `zoomFrameGuide` (prop): Optional WAN frame-guide config forwarded to `InitialImageCard` zoom overlay.
- `onZoomFrameGuideUpdate` (function): Forwards zoom-overlay guide edits to parent WAN state.
- `onInpaintModeChange` (function): Emits raw inpaint-mode select updates for parent-side normalization.
- `onInpaintingFillChange` (function): Emits raw masked-content numeric updates for parent-side normalization.
- `onMaskEditorApply` (function): Emits edited mask data URL produced by the inpaint mask editor overlay.
- `onInitPreviewClick` (function): Opens the mask editor when inpaint mode is active and an init image is present.
- `onMaskEditorExternalReset` (function): Forwards editor reset notices for parent-side toasts.
- `loadMaskPreviewPlaneFromSource` (function): Decodes the saved mask PNG into a binary plane for preview geometry.
- `effectivePreviewMaskPlane` (computed): Display-only effective mask plane used by the thumbnail overlay/crop preview.
- `previewOverlaySource` (ref): Cached RGBA data URL for the inline thumbnail hard-mask + blur-spill preview.
- `previewHasBlurSpill` (ref): Tracks whether the current preview overlay actually contains outward blur spill beyond the hard mask.
- `schedulePreviewOverlaySourceRender` (function): Coalesces thumbnail overlay raster updates after mask/blur changes.
- `previewCropStyle` (computed): Expresses the effective masked-region crop box directly in image-space percentages inside the thumbnail wrapper.
- `showUnsupportedActiveInpaintMode` (computed): Surfaces an explicit disabled select option when the current exact-engine mode is no longer supported.
-->

<template>
  <div :class="['gen-card', { 'gen-card--embedded': embedded }, 'initial-image-block']">
    <div class="initial-image-block__header">
      <div class="initial-image-block__header-main">
        <span class="initial-image-block__title">{{ sectionTitle }}</span>
        <span v-if="sectionSubtitle" class="caption">{{ sectionSubtitle }}</span>
      </div>
      <div v-if="$slots['header-actions']" class="initial-image-block__header-actions">
        <slot name="header-actions" />
      </div>
    </div>

    <ImageSourceBlock
      :mode="effectiveInitSource.mode"
      :folder-source="effectiveInitSource"
      :disabled="disabled"
      :show-source-mode-toggle="showSourceModeToggle"
      source-mode-aria-label="Initial image source mode"
      :image-label="initImageLabel"
      :image-src="initImageData"
      :has-image="hasInitImage"
      :thumbnail="true"
      :zoomable="true"
      :preview-click-action="shouldUseMaskPreviewClick ? 'emit' : 'zoom'"
      :zoom-frame-guide="showFrameGuideEditor ? zoomFrameGuide : null"
      :show-use-crop="true"
      folder-path-label="Folder path"
      folder-path-placeholder="input/img2img-source"
      folder-count-label="Images to generate"
      @update:mode="(value) => emit('patch:initSource', { mode: value as InitSourceFormState['mode'] })"
      @patch:folderSource="(value) => emit('patch:initSource', value)"
      @set:image="(file) => emit('set:initImage', file)"
      @clear:image="() => emit('clear:initImage')"
      @reject:image="(payload) => emit('reject:initImage', payload)"
      @preview-click="onInitPreviewClick"
      @update:zoom-frame-guide="onZoomFrameGuideUpdate"
    >
      <template #dropzone-actions>
        <div v-if="shouldShowMaskUi" class="initial-image-block__mask-editor-actions">
          <button
            class="btn btn-sm btn-outline"
            type="button"
            :disabled="disabled || !maskImageData"
            @click.stop.prevent="emit('clear:maskImage')"
          >
            Clear mask
          </button>
        </div>
      </template>
      <template #preview-overlay>
        <div v-if="shouldShowMaskUi && (previewOverlaySource || previewCropStyle)" class="initial-image-block__mask-preview-layers">
          <img
            v-if="previewOverlaySource"
            class="initial-image-block__mask-preview-overlay-image"
            :src="previewOverlaySource"
            alt=""
            aria-hidden="true"
          >
          <div v-if="previewCropStyle" class="initial-image-block__mask-preview-crop-box" :style="previewCropStyle" />
        </div>
      </template>
      <template #footer>
        <div v-if="showPreviewLegend" class="initial-image-block__preview-legend">
          <span v-if="previewOverlaySource" class="caption initial-image-block__preview-legend__item">
            <span class="initial-image-block__preview-legend__swatch initial-image-block__preview-legend__swatch--mask" aria-hidden="true" />
            <span>Mask</span>
          </span>
          <span v-if="previewHasBlurSpill" class="caption initial-image-block__preview-legend__item">
            <span class="initial-image-block__preview-legend__swatch initial-image-block__preview-legend__swatch--blur" aria-hidden="true" />
            <span>Blur range</span>
          </span>
          <HoverTooltip
            v-if="previewCropStyle"
            class="initial-image-block__preview-legend__tooltip"
            :title="INPAINT_PARAMETER_TOOLTIPS.previewCrop.title"
            :content="INPAINT_PARAMETER_TOOLTIPS.previewCrop.content"
          >
            <span class="caption initial-image-block__preview-legend__item initial-image-block__preview-legend__item--interactive">
              <span class="initial-image-block__preview-legend__swatch initial-image-block__preview-legend__swatch--crop" aria-hidden="true" />
              <span>Final inpaint crop</span>
              <span class="cdx-slider-field__label-help" aria-hidden="true">?</span>
            </span>
          </HoverTooltip>
        </div>
        <p v-if="initImageName" class="caption initial-image-block__caption initial-image-block__caption--init-name">{{ initImageName }}</p>
      </template>
    </ImageSourceBlock>

    <slot />

    <div v-if="shouldShowMaskUi" class="initial-image-block__mask-stack">
      <WanSubHeader title="Inpaint Parameters">
        <template #subtitle>
          <span class="caption">Canvas mask tools + runtime parameters</span>
        </template>
      </WanSubHeader>

      <div class="gc-row initial-image-block__mask-grid">
        <div class="field">
          <label class="label-muted">
            <HoverTooltip
              class="cdx-slider-field__label-tooltip"
              :title="INPAINT_PARAMETER_TOOLTIPS.mode.title"
              :content="INPAINT_PARAMETER_TOOLTIPS.mode.content"
            >
              <span class="cdx-slider-field__label-trigger">
                <span>Inpaint mode</span>
                <span class="cdx-slider-field__label-help" aria-hidden="true">?</span>
              </span>
            </HoverTooltip>
          </label>
          <select class="select-md" :disabled="disabled || resolvedInpaintModeOptions.length === 0" :value="inpaintMode" @change="onInpaintModeChange">
            <option v-if="resolvedInpaintModeOptions.length === 0" value="" disabled>No inpaint modes available for this engine</option>
            <option v-else-if="showUnsupportedActiveInpaintMode" :value="inpaintMode" disabled>Current mode unavailable for this engine - reselect</option>
            <option v-for="option in resolvedInpaintModeOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
          </select>
        </div>

        <div class="field">
          <label class="label-muted">
            <HoverTooltip
              class="cdx-slider-field__label-tooltip"
              :title="INPAINT_PARAMETER_TOOLTIPS.maskedContent.title"
              :content="INPAINT_PARAMETER_TOOLTIPS.maskedContent.content"
            >
              <span class="cdx-slider-field__label-trigger">
                <span>Masked content</span>
                <span class="cdx-slider-field__label-help" aria-hidden="true">?</span>
              </span>
            </HoverTooltip>
          </label>
          <select class="select-md" :disabled="disabled" :value="inpaintingFill" @change="onInpaintingFillChange">
            <option :value="1">Original</option>
            <option :value="0">Fill</option>
            <option :value="2">Latent noise</option>
            <option :value="3">Latent nothing</option>
          </select>
        </div>

        <div class="gc-col gc-col--presets initial-image-block__mask-toggle-col">
          <div class="initial-image-block__mask-toggle-stack">
            <HoverTooltip
              class="cdx-slider-field__label-tooltip"
              :title="INPAINT_PARAMETER_TOOLTIPS.splitMaskRegions.title"
              :content="INPAINT_PARAMETER_TOOLTIPS.splitMaskRegions.content"
              :wrapperFocusable="false"
            >
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', maskRegionSplit ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="maskRegionSplit"
                :disabled="splitMaskRegionToggleDisabled"
                @click="emit('toggle:maskRegionSplit')"
              >
                Split mask regions
              </button>
            </HoverTooltip>

            <HoverTooltip
              class="cdx-slider-field__label-tooltip"
              :title="INPAINT_PARAMETER_TOOLTIPS.invertMask.title"
              :content="INPAINT_PARAMETER_TOOLTIPS.invertMask.content"
              :wrapperFocusable="false"
            >
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', maskInvert ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="maskInvert"
                :disabled="invertMaskToggleDisabled"
                @click="emit('toggle:maskInvert')"
              >
                Invert mask
              </button>
            </HoverTooltip>
          </div>
        </div>
      </div>

      <div v-if="inpaintMode === 'per_step_blend'" class="gc-row initial-image-block__mask-slider-row">
        <SliderField
          class="gc-col gc-col--wide"
          label="Per-step blend strength"
          :tooltip="INPAINT_PARAMETER_TOOLTIPS.perStepBlendStrength.content"
          :tooltipTitle="INPAINT_PARAMETER_TOOLTIPS.perStepBlendStrength.title"
          :modelValue="perStepBlendStrength"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          inputClass="cdx-input-w-xs"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:perStepBlendStrength', value)"
        />

        <SliderField
          class="gc-col gc-col--wide"
          label="Per-step blend steps"
          :tooltip="INPAINT_PARAMETER_TOOLTIPS.perStepBlendSteps.content"
          :tooltipTitle="INPAINT_PARAMETER_TOOLTIPS.perStepBlendSteps.title"
          :modelValue="perStepBlendSteps"
          :min="0"
          :step="1"
          :inputStep="1"
          inputClass="cdx-input-w-xs"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:perStepBlendSteps', value)"
        />
      </div>

      <div class="gc-row initial-image-block__mask-slider-row">
        <SliderField
          class="gc-col gc-col--wide"
          label="Masked padding"
          :tooltip="INPAINT_PARAMETER_TOOLTIPS.maskedPadding.content"
          :tooltipTitle="INPAINT_PARAMETER_TOOLTIPS.maskedPadding.title"
          :modelValue="inpaintFullResPadding"
          :min="0"
          :max="256"
          :step="1"
          :inputStep="1"
          inputClass="cdx-input-w-xs"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:inpaintFullResPadding', value)"
        />

        <SliderField
          class="gc-col gc-col--wide"
          label="Mask blur"
          :tooltip="INPAINT_PARAMETER_TOOLTIPS.maskBlur.content"
          :tooltipTitle="INPAINT_PARAMETER_TOOLTIPS.maskBlur.title"
          :modelValue="maskBlur"
          :min="0"
          :max="64"
          :step="1"
          :inputStep="1"
          inputClass="cdx-input-w-xs"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:maskBlur', value)"
        />
      </div>
    </div>

    <InpaintMaskEditorOverlay
      v-if="shouldShowMaskUi"
      v-model="maskEditorOpen"
      :init-image-data="initImageData"
      :initial-mask-data="maskImageData"
      :image-width="imageWidth"
      :image-height="imageHeight"
      :processing-width="effectiveProcessingWidth"
      :processing-height="effectiveProcessingHeight"
      :mask-blur="maskBlur"
      :masked-padding="inpaintFullResPadding"
      :mask-invert="maskInvert"
      @apply="onMaskEditorApply"
      @external-reset="onMaskEditorExternalReset"
      @update:maskBlur="(value) => emit('update:maskBlur', value)"
      @update:maskedPadding="(value) => emit('update:inpaintFullResPadding', value)"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch, type CSSProperties } from 'vue'
import ImageSourceBlock from './ImageSourceBlock.vue'
import InpaintMaskEditorOverlay from './ui/InpaintMaskEditorOverlay.vue'
import HoverTooltip from './ui/HoverTooltip.vue'
import SliderField from './ui/SliderField.vue'
import { rgbaToMaskPlane } from './ui/inpaint_mask_editor_engine'
import WanSubHeader from './wan/WanSubHeader.vue'
import type { InitSourceFormState } from '../stores/model_tabs'
import {
  computeInpaintMaskBlurSpillAlphaPlane,
  computeInpaintMaskPreviewGeometry,
  resolveInpaintDisplayMaskPlane,
  tintAlphaPlaneToRgba,
} from '../utils/inpaint_mask_preview'
import type { WanImg2VidFrameGuideConfig } from '../utils/wan_img2vid_frame_projection'

type InpaintMode = 'per_step_blend' | 'post_sample_blend' | 'fooocus_inpaint' | 'brushnet'

const PREVIEW_SPILL_TINT = {
  red: 255,
  green: 178,
  blue: 68,
  opacity: 0.62,
} as const

const PREVIEW_MASK_TINT = {
  red: 247,
  green: 73,
  blue: 58,
  alpha: 115,
} as const

const INPAINT_PARAMETER_TOOLTIPS = {
  mode: {
    title: 'Inpaint mode',
      content: [
        'Controls which inpaint runtime path owns the masked sampling pass.',
        '[[Per-step blend:]] keeps the generic masked runtime and blends preserved outside-mask content back on each step.',
        '[[Post-sample blend:]] keeps the generic masked runtime and restores preserved outside-mask content only after sampling finishes.',
        '[[Fooocus Inpaint:]] uses the SDXL-only Fooocus inpaint patch/runtime path and requires `fooocus_inpaint_head.pth` plus `inpaint_v26.fooocus.patch`.',
        '[[Fooocus Inpaint:]] does not support distilled, Turbo, Lightning, or Hyper SDXL variants.',
        '[[BrushNet:]] uses the SDXL-only BrushNet runtime path pinned to `random_mask_brushnet_ckpt_sdxl_v0` under the dedicated `sdxl_brushnet` root.',
      ],
    },
  maskedContent: {
    title: 'Masked content',
    content: [
      'Chooses how the masked area is initialized before sampling.',
      '[[Original:]] starts from the current image crop inside the working area.',
      '[[Fill:]] replaces the masked area with a fill scaffold synthesized from surrounding preserved pixels.',
      '[[Latent noise:]] starts from that same filled image scaffold, then replaces the masked latent with fresh noise for a stronger redraw.',
      '[[Latent nothing:]] starts from that same filled image scaffold, then zeros the masked latent for the most destructive reset.',
    ],
  },
  perStepBlendStrength: {
    title: 'Per-step blend strength',
    content: [
      'Controls how strongly `Per-step blend` pulls preserved outside-mask content back during sampling.',
      '[[1.0:]] preserves the current legacy `Per-step blend` behavior.',
      '[[0.0:]] disables the once-per-step pull-back, but the final preserved-content blend still happens at the end.',
      '[[Increase:]] pulls the result back toward preserved content more each outer sampling step.',
      '[[Decrease:]] gives the model more freedom between steps before the final preserved-content blend closes the result.',
    ],
  },
  perStepBlendSteps: {
    title: 'Per-step blend steps',
    content: [
      'Limits for how many outer sampling steps `Per-step blend` keeps pulling preserved outside-mask content back toward the source latent.',
      '[[0:]] applies that pull-back on every outer sampling step, which preserves the current default behavior.',
      '[[Positive values:]] apply the pull-back only on the first `N` outer sampling steps.',
      '[[Values above current sampling steps:]] are allowed; the runtime resolves the effective window instead of the UI silently clamping it.',
      '[[Increase:]] keeps the preserved-content pull-back active longer into sampling.',
      '[[Decrease:]] stops the pull-back earlier, leaving more late-step freedom before the final preserved-content close.',
    ],
  },
  splitMaskRegions: {
    title: 'Split mask regions',
    content: [
      'Runs disconnected mask islands as separate inpaint passes instead of one combined masked region.',
      '[[Enabled:]] each disconnected region gets its own masked crop/pass; the blue crop preview still shows the union box, not the per-region boxes.',
      '[[Disabled:]] the full effective mask is processed as one region, matching the blue crop preview.',
      '[[Requires:]] one output image total (`batchCount = 1` and `batchSize = 1`).',
      '[[Unavailable with Invert mask:]] this pair is blocked in the UI.',
      '[[Engine support:]] some engines, such as FLUX.2, still reject split-region execution.',
    ],
  },
  invertMask: {
    title: 'Invert mask',
    content: [
      'Swaps the editable and preserved sides of the current mask.',
      '[[Enabled:]] everything outside the painted mask becomes editable, and the thumbnail/editor previews switch to that effective mask immediately.',
      '[[Disabled:]] only the painted mask stays editable.',
      '[[Unavailable with Split mask regions:]] this pair is blocked in the UI.',
    ],
  },
  previewCrop: {
    title: 'Inpaint crop preview',
    content: [
      'The blue box previews the crop derived from the current effective mask.',
      '[[Single-region runs:]] this matches the crop used by the inpaint-only-masked pass.',
      '[[Includes:]] mask bounds, mask blur, masked padding, and the final aspect-ratio expansion for processing.',
      '[[Split mask regions:]] runtime can run multiple per-region crops even though the preview still shows the union box.',
    ],
  },
  maskedPadding: {
    title: 'Masked padding',
    content: [
      'Adds extra context after the blurred mask bounds are computed for the inpaint-only-masked pass.',
      '[[Blue box:]] shows the crop preview after mask blur, masked padding, and aspect-ratio expansion.',
      '[[Increase:]] gives the model more surrounding context and can reduce seam pressure, but pulls more nearby content into the working crop.',
      '[[Decrease:]] keeps the working crop tighter to the mask, but can starve surrounding context and make seams harsher.',
    ],
  },
  maskBlur: {
    title: 'Mask blur',
    content: [
      'Blurs the working mask itself before crop planning and latent mask generation.',
      '[[Blue box:]] can grow even when Masked padding is 0, because blur expands the effective mask before padding is applied.',
      '[[Increase:]] softens the whole mask, not just the edge, and can make the model less tied to `Masked content: Original` because more of that guide turns into a soft blur.',
      '[[Decrease:]] keeps more of the original masked structure intact and the edit tighter to the drawn mask, but edges can look harsher.',
    ],
  },
} as const

const props = withDefaults(defineProps<{
  disabled?: boolean
  embedded?: boolean
  sectionTitle?: string
  sectionSubtitle?: string
  initImageLabel?: string
  showSourceModeToggle?: boolean
  showInpaintControls?: boolean
  showFrameGuideEditor?: boolean
  initSource?: InitSourceFormState
  initImageData?: string
  initImageName?: string
  imageWidth?: number
  imageHeight?: number
  useMask?: boolean
  maskImageData?: string
  maskImageName?: string
  inpaintMode?: InpaintMode
  inpaintModeOptions?: Array<{ value: InpaintMode; label: string }>
  perStepBlendStrength?: number
  perStepBlendSteps?: number
  inpaintingFill?: number
  inpaintFullResPadding?: number
  maskBlur?: number
  maskInvert?: boolean
  maskRegionSplit?: boolean
  processingWidth?: number
  processingHeight?: number
  zoomFrameGuide?: WanImg2VidFrameGuideConfig | null
}>(), {
  disabled: false,
  embedded: false,
  sectionTitle: 'Initial Image',
  sectionSubtitle: 'Initial image',
  initImageLabel: 'Initial Image',
  showSourceModeToggle: false,
  showInpaintControls: false,
  showFrameGuideEditor: false,
  initSource: () => ({
    mode: 'img',
    folderPath: '',
    selectionMode: 'all',
    count: 1,
    order: 'sorted',
    sortBy: 'name',
    useCrop: false,
  }),
  initImageData: '',
  initImageName: '',
  imageWidth: 0,
  imageHeight: 0,
  useMask: false,
  maskImageData: '',
  maskImageName: '',
  inpaintMode: 'per_step_blend',
  inpaintModeOptions: () => [],
  perStepBlendStrength: 1,
  perStepBlendSteps: 0,
  inpaintingFill: 1,
  inpaintFullResPadding: 0,
  maskBlur: 0,
  maskInvert: false,
  maskRegionSplit: false,
  processingWidth: 0,
  processingHeight: 0,
  zoomFrameGuide: null,
})

const emit = defineEmits<{
  (e: 'set:initImage', value: File): void
  (e: 'clear:initImage'): void
  (e: 'reject:initImage', payload: { reason: string; files: File[] }): void
  (e: 'patch:initSource', value: Partial<InitSourceFormState>): void
  (e: 'clear:maskImage'): void
  (e: 'apply:maskImageData', value: string): void
  (e: 'notice:maskEditorReset', message: string): void
  (e: 'update:inpaintMode', value: string): void
  (e: 'update:perStepBlendStrength', value: number): void
  (e: 'update:perStepBlendSteps', value: number): void
  (e: 'update:inpaintingFill', value: number): void
  (e: 'update:inpaintFullResPadding', value: number): void
  (e: 'toggle:maskRegionSplit'): void
  (e: 'toggle:maskInvert'): void
  (e: 'update:maskBlur', value: number): void
  (e: 'update:zoomFrameGuide', value: WanImg2VidFrameGuideConfig): void
}>()

const maskEditorOpen = ref(false)
const previewMaskPlane = ref<Uint8Array | null>(null)
const previewMaskDecodeToken = ref(0)
const previewOverlaySource = ref('')
const previewHasBlurSpill = ref(false)
let previewOverlayRenderRafId = 0
let previewOverlayCanvas: HTMLCanvasElement | null = null

const effectiveInitSource = computed<InitSourceFormState>(() => {
  if (props.showSourceModeToggle) return props.initSource
  return {
    mode: 'img',
    folderPath: '',
    selectionMode: 'all',
    count: 1,
    order: 'sorted',
    sortBy: 'name',
    useCrop: false,
  }
})

const hasInitImage = computed(() => Boolean(String(props.initImageData || '').trim()))

const shouldShowMaskUi = computed(() => (
  props.showInpaintControls
    && props.useMask
    && effectiveInitSource.value.mode === 'img'
))

const shouldUseMaskPreviewClick = computed(() => shouldShowMaskUi.value)

const hasNaturalImageDimensions = computed(() => {
  const imageWidth = Math.trunc(Number(props.imageWidth))
  const imageHeight = Math.trunc(Number(props.imageHeight))
  return Number.isFinite(imageWidth) && imageWidth > 0 && Number.isFinite(imageHeight) && imageHeight > 0
})

const effectiveProcessingWidth = computed(() => {
  const width = Number(props.processingWidth)
  return Number.isFinite(width) && width > 0 ? Math.trunc(width) : 0
})

const effectiveProcessingHeight = computed(() => {
  const height = Number(props.processingHeight)
  return Number.isFinite(height) && height > 0 ? Math.trunc(height) : 0
})

const hasProcessingDimensions = computed(() => effectiveProcessingWidth.value > 0 && effectiveProcessingHeight.value > 0)

const splitMaskRegionToggleDisabled = computed(() => {
  if (props.disabled) return true
  return props.maskInvert && !props.maskRegionSplit
})

const invertMaskToggleDisabled = computed(() => {
  if (props.disabled) return true
  return props.maskRegionSplit && !props.maskInvert
})

const effectivePreviewMaskPlane = computed<Uint8Array | Uint8ClampedArray | null>(() => {
  const maskPlane = previewMaskPlane.value
  if (!maskPlane) return null
  return resolveInpaintDisplayMaskPlane(maskPlane, props.maskInvert)
})

const previewGeometry = computed(() => {
  const maskPlane = effectivePreviewMaskPlane.value
  if (!maskPlane) return null
  if (!hasNaturalImageDimensions.value || !hasProcessingDimensions.value) return null
  try {
    return computeInpaintMaskPreviewGeometry(maskPlane, {
      imageWidth: props.imageWidth,
      imageHeight: props.imageHeight,
      processingWidth: effectiveProcessingWidth.value,
      processingHeight: effectiveProcessingHeight.value,
      maskBlur: props.maskBlur,
      maskedPadding: props.inpaintFullResPadding,
    })
  } catch (error) {
    console.error('[InitialImageBlock] Failed to compute inpaint preview geometry.', error)
    return null
  }
})

const previewCropStyle = computed<CSSProperties | null>(() => {
  const geometry = previewGeometry.value
  const imageWidth = Math.max(1, Math.trunc(props.imageWidth))
  const imageHeight = Math.max(1, Math.trunc(props.imageHeight))
  if (!geometry || imageWidth <= 0 || imageHeight <= 0) return null
  return {
    left: `${(geometry.cropRegion.x1 / imageWidth) * 100}%`,
    top: `${(geometry.cropRegion.y1 / imageHeight) * 100}%`,
    width: `${(geometry.cropRegion.width / imageWidth) * 100}%`,
    height: `${(geometry.cropRegion.height / imageHeight) * 100}%`,
  }
})

const showPreviewLegend = computed(() => (
  shouldShowMaskUi.value
    && (Boolean(previewOverlaySource.value) || Boolean(previewCropStyle.value))
))

watch(
  [() => props.maskImageData, () => props.imageWidth, () => props.imageHeight],
  ([sourceMask, imageWidth, imageHeight]) => {
    const token = (previewMaskDecodeToken.value += 1)
    const source = String(sourceMask || '').trim()
    if (!source || imageWidth <= 0 || imageHeight <= 0) {
      cancelPreviewOverlaySourceRender()
      previewMaskPlane.value = null
      previewOverlaySource.value = ''
      previewHasBlurSpill.value = false
      return
    }
    cancelPreviewOverlaySourceRender()
    previewMaskPlane.value = null
    previewOverlaySource.value = ''
    previewHasBlurSpill.value = false
    void loadMaskPreviewPlaneFromSource(source, imageWidth, imageHeight)
      .then((maskPlane) => {
        if (previewMaskDecodeToken.value !== token) return
        previewMaskPlane.value = maskPlane
      })
      .catch((error) => {
        if (previewMaskDecodeToken.value !== token) return
        previewMaskPlane.value = null
        previewHasBlurSpill.value = false
        console.error('[InitialImageBlock] Failed to decode saved mask preview.', error)
      })
  },
  { immediate: true },
)

watch(
  [previewMaskPlane, () => props.maskBlur, () => props.maskInvert, () => props.imageWidth, () => props.imageHeight],
  () => {
    schedulePreviewOverlaySourceRender()
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  cancelPreviewOverlaySourceRender()
})

function cancelPreviewOverlaySourceRender(): void {
  if (previewOverlayRenderRafId <= 0) return
  window.cancelAnimationFrame(previewOverlayRenderRafId)
  previewOverlayRenderRafId = 0
}

function schedulePreviewOverlaySourceRender(): void {
  cancelPreviewOverlaySourceRender()
  previewOverlayRenderRafId = window.requestAnimationFrame(() => {
    previewOverlayRenderRafId = 0
    renderPreviewOverlaySource()
  })
}

function getPreviewOverlayCanvas(width: number, height: number): HTMLCanvasElement {
  if (!previewOverlayCanvas) {
    previewOverlayCanvas = document.createElement('canvas')
  }
  if (previewOverlayCanvas.width !== width) previewOverlayCanvas.width = width
  if (previewOverlayCanvas.height !== height) previewOverlayCanvas.height = height
  return previewOverlayCanvas
}

function renderPreviewOverlaySource(): void {
  const maskPlane = effectivePreviewMaskPlane.value
  const imageWidth = Math.trunc(props.imageWidth)
  const imageHeight = Math.trunc(props.imageHeight)
  if (!maskPlane || imageWidth <= 0 || imageHeight <= 0) {
    previewOverlaySource.value = ''
    previewHasBlurSpill.value = false
    return
  }

  try {
    const spillAlphaPlane = computeInpaintMaskBlurSpillAlphaPlane(maskPlane, {
      imageWidth,
      imageHeight,
      maskBlur: props.maskBlur,
    })
    const canvas = getPreviewOverlayCanvas(imageWidth, imageHeight)
    const context = canvas.getContext('2d')
    if (!context) {
      throw new Error('Failed to create canvas context for inpaint thumbnail preview.')
    }
    context.clearRect(0, 0, imageWidth, imageHeight)
    const rgba = spillAlphaPlane
      ? tintAlphaPlaneToRgba(spillAlphaPlane, imageWidth, imageHeight, PREVIEW_SPILL_TINT)
      : new Uint8ClampedArray(imageWidth * imageHeight * 4)

    let hasVisiblePixel = false
    let hasBlurSpill = false
    for (let pixel = 0; pixel < maskPlane.length; pixel += 1) {
      const baseIndex = pixel * 4
      if (maskPlane[pixel] > 0) {
        rgba[baseIndex] = PREVIEW_MASK_TINT.red
        rgba[baseIndex + 1] = PREVIEW_MASK_TINT.green
        rgba[baseIndex + 2] = PREVIEW_MASK_TINT.blue
        rgba[baseIndex + 3] = PREVIEW_MASK_TINT.alpha
        hasVisiblePixel = true
        continue
      }
      if (rgba[baseIndex + 3] > 0) {
        hasVisiblePixel = true
        hasBlurSpill = true
      }
    }

    if (!hasVisiblePixel) {
      previewOverlaySource.value = ''
      previewHasBlurSpill.value = false
      return
    }

    const imageData = context.createImageData(imageWidth, imageHeight)
    imageData.data.set(rgba)
    context.putImageData(imageData, 0, 0)
    previewOverlaySource.value = canvas.toDataURL('image/png')
    previewHasBlurSpill.value = hasBlurSpill
  } catch (error) {
    previewOverlaySource.value = ''
    previewHasBlurSpill.value = false
    console.error('[InitialImageBlock] Failed to render inpaint thumbnail preview source.', error)
  }
}

function onInpaintModeChange(event: Event): void {
  emit('update:inpaintMode', (event.target as HTMLSelectElement).value)
}

function onInpaintingFillChange(event: Event): void {
  emit('update:inpaintingFill', Number((event.target as HTMLSelectElement).value))
}

function onMaskEditorApply(maskDataUrl: string): void {
  emit('apply:maskImageData', maskDataUrl)
}

function onInitPreviewClick(): void {
  if (props.disabled || !shouldShowMaskUi.value || effectiveInitSource.value.mode !== 'img' || !props.initImageData) return
  if (!hasNaturalImageDimensions.value) {
    emit('notice:maskEditorReset', 'Mask editor unavailable: init image dimensions are unavailable.')
    return
  }
  if (!hasProcessingDimensions.value) {
    emit('notice:maskEditorReset', 'Mask editor unavailable: processing dimensions are not ready yet.')
    return
  }
  maskEditorOpen.value = true
}

function onMaskEditorExternalReset(message: string): void {
  emit('notice:maskEditorReset', message)
}

function onZoomFrameGuideUpdate(value: WanImg2VidFrameGuideConfig): void {
  if (!props.showFrameGuideEditor) return
  emit('update:zoomFrameGuide', value)
}

async function loadMaskPreviewPlaneFromSource(
  sourceMask: string,
  imageWidth: number,
  imageHeight: number,
): Promise<Uint8Array> {
  const maskImage = await loadMaskPreviewImage(sourceMask)
  const naturalWidth = maskImage.naturalWidth || maskImage.width
  const naturalHeight = maskImage.naturalHeight || maskImage.height
  if (naturalWidth !== imageWidth || naturalHeight !== imageHeight) {
    throw new Error(`Expected mask preview dimensions ${imageWidth}x${imageHeight}, got ${naturalWidth}x${naturalHeight}.`)
  }

  const canvas = document.createElement('canvas')
  canvas.width = imageWidth
  canvas.height = imageHeight
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) {
    throw new Error('Failed to create canvas context for saved mask preview decode.')
  }
  context.drawImage(maskImage, 0, 0, imageWidth, imageHeight)
  const rgba = context.getImageData(0, 0, imageWidth, imageHeight).data
  return rgbaToMaskPlane(rgba, imageWidth, imageHeight)
}

function loadMaskPreviewImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Failed to decode saved mask preview image.'))
    image.src = src
  })
}

const INPAINT_MODE_LABELS: Record<InpaintMode, string> = {
  per_step_blend: 'Per-step blend',
  post_sample_blend: 'Post-sample blend',
  fooocus_inpaint: 'Fooocus Inpaint',
  brushnet: 'BrushNet',
}

const resolvedInpaintModeOptions = computed(() => {
  const raw = Array.isArray(props.inpaintModeOptions) ? props.inpaintModeOptions : []
  return raw.map((entry) => ({
    value: entry.value,
    label: INPAINT_MODE_LABELS[entry.value] ?? entry.label,
  }))
})

const showUnsupportedActiveInpaintMode = computed(() => (
  resolvedInpaintModeOptions.value.length > 0
    && !resolvedInpaintModeOptions.value.some((option) => option.value === props.inpaintMode)
))
</script>

<!-- styles in styles/components/initial-image-block.css -->
