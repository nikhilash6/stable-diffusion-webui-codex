/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared frontend task lifecycle shell for task-backed generation composables.
Owns task-stream attach/stop, resume-marker persistence hooks, auto-resume reattach, history load/clear gating, and resume-notice
coordination shared by image/WAN/LTX generation composables, while leaving payload building, queueing, result shaping, and
domain-specific snapshot handling in the caller-owned composables.

Symbols (top-level; keep in sync; no ghosts):
- `TaskLifecycleStateBase` (interface): Minimal shared reactive state contract required by the lifecycle owner.
- `ResumeLoadResult` (interface): Structured resume-load result with optional parse error message.
- `TaskStreamAttachOptions` (interface): Optional replay offset and event-id hook for a stream attach.
- `UseTaskRunLifecycleOptions` (interface): Callback-driven contract for the shared lifecycle owner.
- `useTaskRunLifecycle` (function): Shared lifecycle owner for resume persistence, stream attach/stop, auto-resume, and history load/clear.
*/

import { ref, type Ref } from 'vue'

import { fetchTaskResult, subscribeTask } from '../api/client'
import type { TaskEvent } from '../api/types'

type TaskSnapshot = Awaited<ReturnType<typeof fetchTaskResult>>

export interface TaskLifecycleStateBase<HistoryItem, CurrentRun> {
  status: string
  taskId: string
  errorMessage: string
  history: HistoryItem[]
  selectedTaskId: string
  historyLoadingTaskId: string
  currentRun: CurrentRun | null
}

export interface ResumeLoadResult<ResumeState> {
  state: ResumeState | null
  error: string | null
}

export interface TaskStreamAttachOptions {
  after?: number
  onEventId?: (eventId: number) => void
}

export interface UseTaskRunLifecycleOptions<
  State extends TaskLifecycleStateBase<HistoryItem, CurrentRun>,
  HistoryItem,
  CurrentRun,
  ResumeState extends { taskId: string },
> {
  tabId: string
  state: Ref<State>
  resumeKey: string
  unsubscribers: Map<string, () => void>
  resumeAttempts: Set<string>
  loadResumeState: (key: string) => ResumeLoadResult<ResumeState>
  saveResumeState: (key: string, state: ResumeState) => void
  clearResumeState: (key: string) => void
  updateResumeEventId: (key: string, eventId: number) => void
  onTaskEvent: (event: TaskEvent) => void
  isSnapshotRunning: (snapshot: TaskSnapshot) => boolean
  onResumeRunning: (saved: ResumeState, snapshot: TaskSnapshot) => void
  onResumeTerminal: (saved: ResumeState, snapshot: TaskSnapshot) => void
  onResumeFetchError?: (saved: ResumeState) => void
  onResumeLoadError?: (message: string) => void
  onHistoryLoaded: (taskId: string, snapshot: TaskSnapshot) => void
  onHistoryLoadError?: (taskId: string, error: unknown) => void
  stopStreamBeforeHistoryLoad?: boolean
  resumeNotice?: Ref<string>
  resumeToastShown?: Set<string>
  getResumeNoticeKey?: (saved: ResumeState) => string
  resumeNoticeMessage?: string
  getResumeAttachOptions?: (saved: ResumeState) => TaskStreamAttachOptions
  onStopStream?: () => void
}

export function useTaskRunLifecycle<
  State extends TaskLifecycleStateBase<HistoryItem, CurrentRun>,
  HistoryItem,
  CurrentRun,
  ResumeState extends { taskId: string },
>(options: UseTaskRunLifecycleOptions<State, HistoryItem, CurrentRun, ResumeState>) {
  const resumeNotice = options.resumeNotice ?? ref('')

  function stopStream(): void {
    const unsub = options.unsubscribers.get(options.tabId)
    if (unsub) {
      unsub()
      options.unsubscribers.delete(options.tabId)
    }
    options.onStopStream?.()
  }

  function saveResume(saved: ResumeState): void {
    options.saveResumeState(options.resumeKey, saved)
  }

  function clearResume(): void {
    options.clearResumeState(options.resumeKey)
  }

  function patchResumeEventId(eventId: number): void {
    options.updateResumeEventId(options.resumeKey, eventId)
  }

  function attachStream(taskId: string, attachOptions?: TaskStreamAttachOptions): void {
    const unsub = subscribeTask(taskId, options.onTaskEvent, undefined, {
      after: attachOptions?.after,
      onMeta: ({ eventId }) => {
        if (typeof eventId !== 'number') return
        attachOptions?.onEventId?.(eventId)
        patchResumeEventId(eventId)
      },
    })
    options.unsubscribers.set(options.tabId, unsub)
  }

  async function tryAutoResume(): Promise<void> {
    if (options.resumeAttempts.has(options.tabId)) return
    options.resumeAttempts.add(options.tabId)

    const loaded = options.loadResumeState(options.resumeKey)
    if (loaded.error) {
      clearResume()
      options.onResumeLoadError?.(loaded.error)
      return
    }
    const saved = loaded.state
    if (!saved) return

    let snapshot: TaskSnapshot
    try {
      snapshot = await fetchTaskResult(saved.taskId)
    } catch {
      clearResume()
      options.onResumeFetchError?.(saved)
      return
    }

    if (options.isSnapshotRunning(snapshot)) {
      stopStream()
      options.onResumeRunning(saved, snapshot)
      const attachOptions = options.getResumeAttachOptions?.(saved)
      attachStream(saved.taskId, attachOptions)

      const noticeKey = options.getResumeNoticeKey?.(saved) ?? saved.taskId
      if (!options.resumeToastShown || !options.resumeToastShown.has(noticeKey)) {
        resumeNotice.value = options.resumeNoticeMessage ?? 'Reconnected (resumed task).'
        options.resumeToastShown?.add(noticeKey)
      }
      return
    }

    clearResume()
    options.onResumeTerminal(saved, snapshot)
  }

  async function loadHistory(taskId: string): Promise<void> {
    if (!taskId || options.state.value.status === 'running') return
    if (options.stopStreamBeforeHistoryLoad) stopStream()
    options.state.value.historyLoadingTaskId = taskId
    try {
      const snapshot = await fetchTaskResult(taskId)
      options.onHistoryLoaded(taskId, snapshot)
    } catch (error) {
      options.onHistoryLoadError?.(taskId, error)
    } finally {
      options.state.value.historyLoadingTaskId = ''
    }
  }

  function clearHistory(): void {
    options.state.value.history = []
    options.state.value.selectedTaskId = ''
  }

  return {
    resumeNotice,
    stopStream,
    saveResume,
    clearResume,
    patchResumeEventId,
    attachStream,
    tryAutoResume,
    loadHistory,
    clearHistory,
  }
}
