/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared Results helpers (toast + clipboard + JSON formatting).
Provides clipboard copy helpers, a short-lived notice/toast state, and JSON formatting utilities used by results panels.

Symbols (top-level; keep in sync; no ghosts):
- `formatJson` (function): Formats a value as pretty JSON (fallback to string).
- `copyToClipboard` (function): Copies text via the Clipboard API, with a class-based textarea fallback when the secure API is unavailable.
- `useResultsCard` (function): Returns notice state and helpers for results UI (toast/clipboard).
*/

import { onBeforeUnmount, ref } from 'vue'

export function formatJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

async function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text)
    return
  }

  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.className = 'cdx-clipboard-fallback'
  textarea.setAttribute('readonly', '')
  textarea.setAttribute('aria-hidden', 'true')
  textarea.tabIndex = -1

  document.body.appendChild(textarea)
  try {
    textarea.focus()
    textarea.select()
    textarea.setSelectionRange(0, textarea.value.length)
    if (!document.execCommand('copy')) throw new Error('Clipboard copy failed.')
  } finally {
    textarea.remove()
  }
}

export function useResultsCard(options: { noticeDurationMs?: number } = {}) {
  const notice = ref('')
  let noticeTimer: number | null = null

  const noticeDurationMs = Number.isFinite(options.noticeDurationMs)
    ? Math.max(0, Number(options.noticeDurationMs))
    : 2000

  function clearNotice(): void {
    notice.value = ''
    if (noticeTimer !== null) window.clearTimeout(noticeTimer)
    noticeTimer = null
  }

  function toast(message: string): void {
    notice.value = message
    if (noticeTimer !== null) window.clearTimeout(noticeTimer)
    noticeTimer = window.setTimeout(() => {
      notice.value = ''
      noticeTimer = null
    }, noticeDurationMs)
  }

  async function copyText(text: string, successMessage = 'Copied to clipboard.'): Promise<void> {
    try {
      await copyToClipboard(text)
      toast(successMessage)
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err))
    }
  }

  async function copyJson(value: unknown, successMessage = 'Copied JSON.'): Promise<void> {
    try {
      await copyText(JSON.stringify(value, null, 2), successMessage)
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err))
    }
  }

  onBeforeUnmount(() => {
    if (noticeTimer !== null) window.clearTimeout(noticeTimer)
  })

  return {
    notice,
    toast,
    clearNotice,
    copyText,
    copyJson,
    formatJson,
  }
}
