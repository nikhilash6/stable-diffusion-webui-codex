/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared frontend image file I/O helpers.
Provides deterministic file-to-data-URL loading and image-dimension probing for live frontend image workflows without owning
any caller-specific toast, rejection, or normalization policy.

Symbols (top-level; keep in sync; no ghosts):
- `readFileAsDataURL` (function): Reads a `File` as a data URL string.
- `readImageDimensions` (function): Loads image width/height from a source URL or data URL.
*/

export function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result || ''))
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read file.'))
    reader.readAsDataURL(file)
  })
}

export function readImageDimensions(src: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve({ width: image.naturalWidth || image.width, height: image.naturalHeight || image.height })
    image.onerror = () => reject(new Error('Failed to load image.'))
    image.src = src
  })
}
