/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Pure normalization helpers for image-tab parameter controls.
Centralizes normalization used by img2img/inpaint UI updates, init-image mode cleanup, inpaint-toggle interlocks, and capability-driven hires visibility so parent handlers stay explicit and unit-testable.

Symbols (top-level; keep in sync; no ghosts):
- `InpaintMode` (type): Allowed inpaint mode values.
- `InpaintMaskToggleState` (interface): Normalized interlock state for invert-mask and split-mask toggles.
- `UseInitImageModePatch` (interface): Canonical mode-toggle cleanup patch for txt2img/img2img init-image ownership.
- `parseInpaintMode` (function): Parses one strict inpaint-mode enum value or returns `null` for stale/invalid input.
- `normalizeInpaintingFill` (function): Clamps masked-content fill mode to backend-supported integer range `[0, 3]`.
- `normalizeNonNegativeInt` (function): Truncates and clamps any numeric input to `>= 0`.
- `buildUseInitImagePatch` (function): Builds the canonical init-image/mask cleanup patch when toggling img2img ownership.
- `normalizeInpaintMaskToggleState` (function): Preserves `maskInvert` while clearing the invalid `maskRegionSplit + maskInvert` combination.
- `resolveTextOverride` (function): Uses override text when non-blank; otherwise falls back to base text.
- `isHiresVisibleForMode` (function): Returns whether hires controls should be visible for the active mode/engine/mask combination.
- `resolveHiresModePolicy` (function): Resolves hires panel visibility and reset behavior for the active mode/engine/mask combination.
*/

export type InpaintMode = 'per_step_blend' | 'post_sample_blend' | 'fooocus_inpaint' | 'brushnet'
export interface InpaintMaskToggleState {
  maskInvert: boolean
  maskRegionSplit: boolean
}
export interface UseInitImageModePatch {
  useInitImage: boolean
  initImageData: string
  initImageName: string
  useMask: boolean
  maskImageData: string
  maskImageName: string
}

export function parseInpaintMode(value: unknown): InpaintMode | null {
  if (value === 'per_step_blend' || value === 'post_sample_blend' || value === 'fooocus_inpaint' || value === 'brushnet') {
    return value
  }
  return null
}

export function normalizeInpaintingFill(value: number): number {
  return Math.max(0, Math.min(3, Math.trunc(value)))
}

export function normalizeNonNegativeInt(value: number): number {
  return Math.max(0, Math.trunc(value))
}

export function buildUseInitImagePatch(useInitImage: boolean): Partial<UseInitImageModePatch> {
  if (useInitImage) {
    return { useInitImage: true }
  }
  return {
    useInitImage: false,
    initImageData: '',
    initImageName: '',
    useMask: false,
    maskImageData: '',
    maskImageName: '',
  }
}

export function normalizeInpaintMaskToggleState(
  maskInvert: boolean,
  maskRegionSplit: boolean,
): InpaintMaskToggleState {
  if (maskInvert && maskRegionSplit) {
    return {
      maskInvert: true,
      maskRegionSplit: false,
    }
  }
  return {
    maskInvert,
    maskRegionSplit,
  }
}

export function resolveTextOverride(baseText: string, overrideText?: string): string {
  const override = String(overrideText ?? '')
  if (override.trim().length > 0) return override
  return String(baseText ?? '')
}

export function isHiresVisibleForMode(useInitImage: boolean, supportsHires: boolean, useMask = false): boolean {
  if (!supportsHires) return false
  if (!useInitImage) return true
  return !useMask
}

export function resolveHiresModePolicy(
  useInitImage: boolean,
  supportsHiresForEngine: boolean,
  useMask = false,
): { showCard: boolean; resetState: boolean } {
  const maskedImg2Img = useInitImage && useMask
  return {
    showCard: isHiresVisibleForMode(useInitImage, supportsHiresForEngine, useMask),
    resetState: !supportsHiresForEngine || maskedImg2Img,
  }
}
