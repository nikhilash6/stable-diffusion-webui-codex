/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical img2img resize-mode options used by UI state and payload routing.
Defines the stable option ids for img2img resize behavior selectors, the truthful per-engine option
subsets, and normalization helpers for persisted/unknown values.

Symbols (top-level; keep in sync; no ghosts):
- `IMG2IMG_RESIZE_MODE_OPTIONS` (constant): Ordered resize-mode options exposed in img2img controls.
- `Img2ImgResizeMode` (type): Union of allowed resize-mode ids.
- `Img2ImgResizeModeOption` (type): Single resize-mode option object used by engine-scoped selectors.
- `IMG2IMG_PIXEL_RESIZE_MODE_OPTIONS` (constant): Pixel-space resize modes that map truthfully to the current backend path.
- `DEFAULT_IMG2IMG_RESIZE_MODE` (constant): Default resize mode for image-tab params.
- `normalizeImg2ImgResizeModeFromOptions` (function): Normalizes unknown values against a specific option subset.
- `normalizeImg2ImgResizeMode` (function): Normalizes unknown values to a valid resize-mode id.
- `img2imgResizeModeOptionsForEngine` (function): Returns the truthful resize-mode option subset for a given engine id.
- `normalizeImg2ImgResizeModeForEngine` (function): Normalizes a resize-mode value against the active engine contract.
*/

export const IMG2IMG_RESIZE_MODE_OPTIONS = [
  { value: 'just_resize', label: 'Just resize' },
  { value: 'crop_and_resize', label: 'Crop and resize' },
  { value: 'resize_and_fill', label: 'Resize and fill' },
  { value: 'just_resize_latent_upscale', label: 'Just resize (latent upscale)' },
  { value: 'upscaler', label: 'Upscaler' },
] as const

export type Img2ImgResizeMode = (typeof IMG2IMG_RESIZE_MODE_OPTIONS)[number]['value']
export type Img2ImgResizeModeOption = (typeof IMG2IMG_RESIZE_MODE_OPTIONS)[number]

export const IMG2IMG_PIXEL_RESIZE_MODE_OPTIONS = [
  { value: 'just_resize', label: 'Just resize' },
  { value: 'crop_and_resize', label: 'Crop and resize' },
  { value: 'resize_and_fill', label: 'Resize and fill' },
] as const satisfies readonly Img2ImgResizeModeOption[]

export const DEFAULT_IMG2IMG_RESIZE_MODE: Img2ImgResizeMode = 'just_resize'

export function normalizeImg2ImgResizeModeFromOptions(
  value: unknown,
  options: readonly Img2ImgResizeModeOption[],
): Img2ImgResizeMode {
  const raw = typeof value === 'string' ? value.trim() : ''
  for (const option of options) {
    if (option.value === raw) return option.value
  }
  return DEFAULT_IMG2IMG_RESIZE_MODE
}

export function normalizeImg2ImgResizeMode(value: unknown): Img2ImgResizeMode {
  return normalizeImg2ImgResizeModeFromOptions(value, IMG2IMG_RESIZE_MODE_OPTIONS)
}

export function img2imgResizeModeOptionsForEngine(engineId: string | null | undefined): readonly Img2ImgResizeModeOption[] {
  if (engineId === 'zimage') return IMG2IMG_PIXEL_RESIZE_MODE_OPTIONS
  return IMG2IMG_RESIZE_MODE_OPTIONS
}

export function normalizeImg2ImgResizeModeForEngine(
  engineId: string | null | undefined,
  value: unknown,
): Img2ImgResizeMode {
  return normalizeImg2ImgResizeModeFromOptions(value, img2imgResizeModeOptionsForEngine(engineId))
}
