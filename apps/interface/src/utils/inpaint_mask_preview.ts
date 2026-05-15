/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Pure inpaint preview helpers shared by the img2img card and mask editor.
Mirrors the backend masked-img2img crop math needed for frontend previews and exposes pure frontend raster helpers for
mask-blur spill visualization: binary mask bbox, blur-support expansion, masked-padding expansion, aspect-preserving
crop expansion, blur-spill alpha generation, display/storage invert-mask resolution, and tint packing.

Symbols (top-level; keep in sync; no ghosts):
- `InpaintMaskPreviewInput` (interface): Input contract for computing preview geometry from a binary mask plane.
- `InpaintPreviewRect` (interface): Rect contract using image/container pixel coordinates with cached width/height.
- `InpaintMaskPreviewGeometry` (interface): Output contract for mask bounds, blur bounds, blur-support radius, and final crop region.
- `InpaintMaskBlurSpillInput` (interface): Input contract for computing the outward mask-blur spill alpha plane.
- `InpaintPreviewTint` (interface): RGB + opacity contract for packing a preview alpha plane into RGBA bytes.
- `computeInpaintMaskPreviewGeometry` (function): Computes blur-support and masked-padding preview geometry from a binary mask plane.
- `computeInpaintMaskBlurSpillAlphaPlane` (function): Computes the outward-only blur spill alpha plane for a binary mask.
- `resolveInpaintDisplayMaskPlane` (function): Returns the display-only effective mask plane, with optional grayscale inversion for preview semantics.
- `resolveInpaintStorageMaskPlane` (function): Returns the raw storage/export mask plane that corresponds to the visible effective mask.
- `tintAlphaPlaneToRgba` (function): Packs an alpha plane into an RGBA buffer using a shared preview tint.
*/

export interface InpaintMaskPreviewInput {
  imageWidth: number
  imageHeight: number
  processingWidth: number
  processingHeight: number
  maskBlur: number
  maskedPadding: number
}

export interface InpaintPreviewRect {
  x1: number
  y1: number
  x2: number
  y2: number
  width: number
  height: number
}

export interface InpaintMaskPreviewGeometry {
  maskBounds: InpaintPreviewRect
  blurBounds: InpaintPreviewRect
  blurSupportRadius: number
  cropRegion: InpaintPreviewRect
}

export interface InpaintMaskBlurSpillInput {
  imageWidth: number
  imageHeight: number
  maskBlur: number
}

export interface InpaintPreviewTint {
  red: number
  green: number
  blue: number
  opacity: number
}

function requirePositiveInt(name: string, value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Inpaint preview requires ${name} > 0 (got ${String(value)}).`)
  }
  const normalized = Math.trunc(Number(value))
  if (normalized <= 0) {
    throw new Error(`Inpaint preview requires ${name} >= 1 after normalization (got ${String(value)}).`)
  }
  return normalized
}

function normalizeNonNegativeInt(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 0
  return Math.max(0, Math.trunc(Number(value)))
}

function buildRect(x1: number, y1: number, x2: number, y2: number): InpaintPreviewRect {
  return {
    x1,
    y1,
    x2,
    y2,
    width: Math.max(0, x2 - x1),
    height: Math.max(0, y2 - y1),
  }
}

function expandRect(
  rect: InpaintPreviewRect,
  pad: number,
  imageWidth: number,
  imageHeight: number,
): InpaintPreviewRect {
  const normalizedPad = normalizeNonNegativeInt(pad)
  if (normalizedPad <= 0) return rect
  return buildRect(
    Math.max(0, rect.x1 - normalizedPad),
    Math.max(0, rect.y1 - normalizedPad),
    Math.min(imageWidth, rect.x2 + normalizedPad),
    Math.min(imageHeight, rect.y2 + normalizedPad),
  )
}

function gaussianSupportRadius(sigma: number): number {
  const normalizedSigma = Number(sigma)
  if (!Number.isFinite(normalizedSigma) || normalizedSigma <= 0) return 0
  return Math.max(0, Math.trunc(2.5 * normalizedSigma + 0.5))
}

function boxesForApproximateGaussian(sigma: number): number[] {
  const normalizedSigma = Number(sigma)
  if (!Number.isFinite(normalizedSigma) || normalizedSigma <= 0) return [1, 1, 1]

  const idealWidth = Math.sqrt((12 * normalizedSigma * normalizedSigma / 3) + 1)
  let lowerWidth = Math.floor(idealWidth)
  if (lowerWidth % 2 === 0) lowerWidth -= 1
  if (lowerWidth < 1) lowerWidth = 1
  const upperWidth = lowerWidth + 2
  const idealLowerCount =
    (12 * normalizedSigma * normalizedSigma - (3 * lowerWidth * lowerWidth) - (12 * lowerWidth) - 9) /
    ((-4 * lowerWidth) - 4)
  const lowerCount = Math.max(0, Math.min(3, Math.round(idealLowerCount)))
  return [0, 1, 2].map((index) => (index < lowerCount ? lowerWidth : upperWidth))
}

function boxBlurHorizontal(
  source: Float32Array,
  target: Float32Array,
  width: number,
  height: number,
  radius: number,
): void {
  if (radius <= 0) {
    target.set(source)
    return
  }

  const prefix = new Float32Array(width + 1)
  const windowSize = (radius * 2) + 1

  for (let y = 0; y < height; y += 1) {
    const rowBase = y * width
    prefix[0] = 0
    for (let x = 0; x < width; x += 1) {
      prefix[x + 1] = prefix[x] + source[rowBase + x]
    }

    const firstValue = source[rowBase]
    const lastValue = source[rowBase + width - 1]
    for (let x = 0; x < width; x += 1) {
      const left = x - radius
      const right = x + radius
      const clampedLeft = Math.max(0, left)
      const clampedRight = Math.min(width - 1, right)
      let sum = prefix[clampedRight + 1] - prefix[clampedLeft]
      if (left < 0) sum += firstValue * Math.abs(left)
      if (right >= width) sum += lastValue * (right - width + 1)
      target[rowBase + x] = sum / windowSize
    }
  }
}

function boxBlurVertical(
  source: Float32Array,
  target: Float32Array,
  width: number,
  height: number,
  radius: number,
): void {
  if (radius <= 0) {
    target.set(source)
    return
  }

  const prefix = new Float32Array(height + 1)
  const windowSize = (radius * 2) + 1

  for (let x = 0; x < width; x += 1) {
    prefix[0] = 0
    for (let y = 0; y < height; y += 1) {
      prefix[y + 1] = prefix[y] + source[(y * width) + x]
    }

    const firstValue = source[x]
    const lastValue = source[((height - 1) * width) + x]
    for (let y = 0; y < height; y += 1) {
      const top = y - radius
      const bottom = y + radius
      const clampedTop = Math.max(0, top)
      const clampedBottom = Math.min(height - 1, bottom)
      let sum = prefix[clampedBottom + 1] - prefix[clampedTop]
      if (top < 0) sum += firstValue * Math.abs(top)
      if (bottom >= height) sum += lastValue * (bottom - height + 1)
      target[(y * width) + x] = sum / windowSize
    }
  }
}

function clampByte(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(255, Math.round(value)))
}

function invertMaskPlane(maskPlane: Uint8Array | Uint8ClampedArray): Uint8Array {
  const invertedMask = new Uint8Array(maskPlane.length)
  for (let index = 0; index < maskPlane.length; index += 1) {
    invertedMask[index] = 255 - maskPlane[index]
  }
  return invertedMask
}

function computeMaskBounds(
  maskPlane: Uint8Array | Uint8ClampedArray,
  imageWidth: number,
  imageHeight: number,
): InpaintPreviewRect | null {
  let minX = imageWidth
  let minY = imageHeight
  let maxX = -1
  let maxY = -1

  for (let y = 0; y < imageHeight; y += 1) {
    const rowBase = y * imageWidth
    for (let x = 0; x < imageWidth; x += 1) {
      if (maskPlane[rowBase + x] <= 0) continue
      if (x < minX) minX = x
      if (y < minY) minY = y
      if (x > maxX) maxX = x
      if (y > maxY) maxY = y
    }
  }

  if (maxX < 0 || maxY < 0) return null
  return buildRect(minX, minY, maxX + 1, maxY + 1)
}

function expandCropRegion(
  cropRegion: InpaintPreviewRect,
  processingWidth: number,
  processingHeight: number,
  imageWidth: number,
  imageHeight: number,
): InpaintPreviewRect {
  let { x1, y1, x2, y2 } = cropRegion
  const cropWidth = Math.max(1, x2 - x1)
  const cropHeight = Math.max(1, y2 - y1)
  const ratioCrop = cropWidth / cropHeight
  const ratioProcessing = processingWidth / processingHeight

  if (ratioCrop > ratioProcessing) {
    const desiredHeight = cropWidth / ratioProcessing
    const diff = Math.trunc(desiredHeight - cropHeight)
    y1 -= Math.trunc(diff / 2)
    y2 += diff - Math.trunc(diff / 2)
    if (y2 >= imageHeight) {
      const overflow = y2 - imageHeight
      y2 -= overflow
      y1 -= overflow
    }
    if (y1 < 0) {
      y2 -= y1
      y1 = 0
    }
    if (y2 >= imageHeight) y2 = imageHeight
  } else {
    const desiredWidth = cropHeight * ratioProcessing
    const diff = Math.trunc(desiredWidth - cropWidth)
    x1 -= Math.trunc(diff / 2)
    x2 += diff - Math.trunc(diff / 2)
    if (x2 >= imageWidth) {
      const overflow = x2 - imageWidth
      x2 -= overflow
      x1 -= overflow
    }
    if (x1 < 0) {
      x2 -= x1
      x1 = 0
    }
    if (x2 >= imageWidth) x2 = imageWidth
  }

  return buildRect(x1, y1, x2, y2)
}

export function computeInpaintMaskPreviewGeometry(
  maskPlane: Uint8Array | Uint8ClampedArray,
  input: InpaintMaskPreviewInput,
): InpaintMaskPreviewGeometry | null {
  const imageWidth = requirePositiveInt('imageWidth', input.imageWidth)
  const imageHeight = requirePositiveInt('imageHeight', input.imageHeight)
  const expectedMaskLength = imageWidth * imageHeight
  if (maskPlane.length !== expectedMaskLength) {
    throw new Error(`Inpaint preview requires maskPlane length ${expectedMaskLength} (got ${maskPlane.length}).`)
  }

  const maskBounds = computeMaskBounds(maskPlane, imageWidth, imageHeight)
  if (!maskBounds) return null

  const processingWidth = requirePositiveInt('processingWidth', input.processingWidth)
  const processingHeight = requirePositiveInt('processingHeight', input.processingHeight)
  const blurSupportRadius = gaussianSupportRadius(input.maskBlur)
  const blurBounds = expandRect(maskBounds, blurSupportRadius, imageWidth, imageHeight)
  const paddedBounds = expandRect(blurBounds, input.maskedPadding, imageWidth, imageHeight)
  const cropRegion = expandCropRegion(
    paddedBounds,
    processingWidth,
    processingHeight,
    imageWidth,
    imageHeight,
  )

  return {
    maskBounds,
    blurBounds,
    blurSupportRadius,
    cropRegion,
  }
}

export function computeInpaintMaskBlurSpillAlphaPlane(
  maskPlane: Uint8Array | Uint8ClampedArray,
  input: InpaintMaskBlurSpillInput,
): Uint8ClampedArray | null {
  const imageWidth = requirePositiveInt('imageWidth', input.imageWidth)
  const imageHeight = requirePositiveInt('imageHeight', input.imageHeight)
  const expectedMaskLength = imageWidth * imageHeight
  if (maskPlane.length !== expectedMaskLength) {
    throw new Error(`Inpaint blur preview requires maskPlane length ${expectedMaskLength} (got ${maskPlane.length}).`)
  }

  const normalizedBlur = Number(input.maskBlur)
  if (!Number.isFinite(normalizedBlur) || normalizedBlur <= 0) return null

  const source = new Float32Array(expectedMaskLength)
  let hasFilledPixel = false
  let hasEmptyPixel = false
  for (let index = 0; index < expectedMaskLength; index += 1) {
    const filled = maskPlane[index] > 0
    source[index] = filled ? 1 : 0
    if (filled) hasFilledPixel = true
    else hasEmptyPixel = true
  }
  if (!hasFilledPixel || !hasEmptyPixel) return null

  const blurred = new Float32Array(source)
  const scratch = new Float32Array(expectedMaskLength)
  for (const size of boxesForApproximateGaussian(normalizedBlur)) {
    const radius = Math.max(0, Math.trunc((size - 1) / 2))
    boxBlurHorizontal(blurred, scratch, imageWidth, imageHeight, radius)
    boxBlurVertical(scratch, blurred, imageWidth, imageHeight, radius)
  }

  const spillAlpha = new Uint8ClampedArray(expectedMaskLength)
  let hasVisibleSpill = false
  for (let index = 0; index < expectedMaskLength; index += 1) {
    if (maskPlane[index] > 0) continue
    const alpha = clampByte(blurred[index] * 255)
    spillAlpha[index] = alpha
    if (alpha > 0) hasVisibleSpill = true
  }
  return hasVisibleSpill ? spillAlpha : null
}

export function resolveInpaintDisplayMaskPlane(
  maskPlane: Uint8Array | Uint8ClampedArray,
  invertMask: boolean,
): Uint8Array | Uint8ClampedArray {
  if (!invertMask) return maskPlane
  return invertMaskPlane(maskPlane)
}

export function resolveInpaintStorageMaskPlane(
  maskPlane: Uint8Array | Uint8ClampedArray,
  invertMask: boolean,
): Uint8Array | Uint8ClampedArray {
  if (!invertMask) return maskPlane
  return invertMaskPlane(maskPlane)
}

export function tintAlphaPlaneToRgba(
  alphaPlane: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  tint: InpaintPreviewTint,
): Uint8ClampedArray {
  const normalizedWidth = requirePositiveInt('width', width)
  const normalizedHeight = requirePositiveInt('height', height)
  const expectedLength = normalizedWidth * normalizedHeight
  if (alphaPlane.length !== expectedLength) {
    throw new Error(`Inpaint preview tint requires alphaPlane length ${expectedLength} (got ${alphaPlane.length}).`)
  }

  const red = clampByte(tint.red)
  const green = clampByte(tint.green)
  const blue = clampByte(tint.blue)
  const opacity = Number.isFinite(tint.opacity) ? Math.max(0, Math.min(1, tint.opacity)) : 1
  const rgba = new Uint8ClampedArray(expectedLength * 4)

  for (let pixel = 0; pixel < expectedLength; pixel += 1) {
    const baseIndex = pixel * 4
    rgba[baseIndex] = red
    rgba[baseIndex + 1] = green
    rgba[baseIndex + 2] = blue
    rgba[baseIndex + 3] = clampByte(alphaPlane[pixel] * opacity)
  }

  return rgba
}
