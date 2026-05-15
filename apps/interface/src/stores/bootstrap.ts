/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Application bootstrap orchestrator and fatal-error funnel for the interface.
Owns two-phase startup sequencing: a minimal required phase (`engine_capabilities`) that gates initial UI readiness, and a deferred phase
(`model_tabs`) that continues in the background after minimal readiness. Required failures are hard-fatal; deferred failures are explicit but non-fatal.

Symbols (top-level; keep in sync; no ghosts):
- `BootstrapCriticalStatus` (type): Minimal required bootstrap state (`idle|loading|ready|fatal`).
- `BootstrapDeferredStatus` (type): Deferred bootstrap state (`idle|loading|ready|error`).
- `useBootstrapStore` (store): Pinia store exposing bootstrap lifecycle + global error funnel helpers.
*/

import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { useEngineCapabilitiesStore } from './engine_capabilities'
import { useModelTabsStore } from './model_tabs'

export type BootstrapCriticalStatus = 'idle' | 'loading' | 'ready' | 'fatal'
export type BootstrapDeferredStatus = 'idle' | 'loading' | 'ready' | 'error'

function normalizeErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message || error.name || 'Unknown error'
  if (typeof error === 'string') return error
  return String(error)
}

export const useBootstrapStore = defineStore('bootstrap', () => {
  const criticalStatus = ref<BootstrapCriticalStatus>('idle')
  const deferredStatus = ref<BootstrapDeferredStatus>('idle')
  const fatalContext = ref<string>('')
  const fatalMessage = ref<string>('')
  const deferredContext = ref<string>('')
  const deferredMessage = ref<string>('')
  let criticalBootstrapPromise: Promise<void> | null = null
  let deferredBootstrapPromise: Promise<void> | null = null
  let globalHandlersInstalled = false

  const isReady = computed(() => criticalStatus.value === 'ready')
  const isLoading = computed(() => criticalStatus.value === 'loading')
  const isFatal = computed(() => criticalStatus.value === 'fatal')
  const isDeferredLoading = computed(() => deferredStatus.value === 'loading')
  const hasDeferredError = computed(() => deferredStatus.value === 'error')

  function clearFatalState(): void {
    fatalContext.value = ''
    fatalMessage.value = ''
  }

  function clearDeferredState(): void {
    deferredContext.value = ''
    deferredMessage.value = ''
  }

  function reportFatal(error: unknown, context: string): void {
    const message = normalizeErrorMessage(error)
    if (isFatal.value) {
      console.error('[bootstrap] additional fatal error', { context, error })
      return
    }
    console.error('[bootstrap] fatal error', { context, error })
    fatalContext.value = String(context || 'Fatal error')
    fatalMessage.value = message
    criticalStatus.value = 'fatal'
  }

  async function runRequired<T>(context: string, fn: () => Promise<T>): Promise<T> {
    try {
      return await fn()
    } catch (error: unknown) {
      reportFatal(error, context)
      throw error
    }
  }

  async function startDeferred(opts: { force?: boolean } = {}): Promise<void> {
    const force = Boolean(opts.force)
    if (criticalStatus.value !== 'ready') return
    if (!force && deferredStatus.value === 'ready') return
    if (deferredBootstrapPromise) return deferredBootstrapPromise

    if (force || deferredStatus.value === 'error') {
      clearDeferredState()
    }
    deferredStatus.value = 'loading'

    const tabsStore = useModelTabsStore()
    deferredBootstrapPromise = (async () => {
      try {
        await tabsStore.load()
        deferredStatus.value = 'ready'
      } catch (error: unknown) {
        deferredContext.value = 'Failed to load model tabs'
        deferredMessage.value = normalizeErrorMessage(error)
        deferredStatus.value = 'error'
        console.error('[bootstrap] deferred startup failed', {
          context: deferredContext.value,
          error,
        })
      }
    })()

    try {
      await deferredBootstrapPromise
    } finally {
      deferredBootstrapPromise = null
    }
  }

  async function start(opts: { force?: boolean } = {}): Promise<void> {
    const force = Boolean(opts.force)
    if (!force && isReady.value) {
      void startDeferred()
      return
    }
    if (criticalBootstrapPromise) return criticalBootstrapPromise

    if (force || criticalStatus.value === 'fatal') {
      clearFatalState()
    }
    if (force) {
      clearDeferredState()
      deferredStatus.value = 'idle'
    }
    criticalStatus.value = 'loading'

    const engineCaps = useEngineCapabilitiesStore()
    criticalBootstrapPromise = (async () => {
      await runRequired('Failed to load engine capabilities', async () => {
        await engineCaps.init({ force })
      })
      criticalStatus.value = 'ready'
      void startDeferred({ force })
    })()

    try {
      await criticalBootstrapPromise
    } finally {
      criticalBootstrapPromise = null
    }
  }

  async function retry(): Promise<void> {
    await start({ force: true })
  }

  async function retryDeferred(): Promise<void> {
    await startDeferred({ force: true })
  }

  function installGlobalErrorHandlers(): void {
    if (globalHandlersInstalled) return
    globalHandlersInstalled = true

    window.addEventListener('error', (event: ErrorEvent) => {
      const reason = event.error ?? event.message ?? 'Unhandled runtime error'
      reportFatal(reason, 'Unhandled runtime error')
    })

    window.addEventListener('unhandledrejection', (event: PromiseRejectionEvent) => {
      reportFatal(event.reason, 'Unhandled promise rejection')
    })
  }

  return {
    criticalStatus,
    deferredStatus,
    fatalContext,
    fatalMessage,
    deferredContext,
    deferredMessage,
    isReady,
    isLoading,
    isFatal,
    isDeferredLoading,
    hasDeferredError,
    start,
    startDeferred,
    retry,
    retryDeferred,
    reportFatal,
    runRequired,
    installGlobalErrorHandlers,
  }
})
