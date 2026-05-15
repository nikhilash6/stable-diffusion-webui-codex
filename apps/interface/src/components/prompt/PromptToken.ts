/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Prompt token node + prompt parsing/serialization for the editor.
Defines the Tiptap `PromptToken` node view and implements best-effort conversion between the editor JSON document and the legacy prompt string format used by the backend.

Symbols (top-level; keep in sync; no ghosts):
- `PromptTokenAttrs` (interface): Attributes stored on a prompt token node (kind/name/weight/enabled).
- `PromptToken` (const): Tiptap node definition rendered via `PromptTokenChip`.
- `parseStrictFiniteNumber` (function): Parses numeric strings using strict finite-float rules (no partial/garbage suffix acceptance).
- `nodeTypeName` (function): Best-effort node type resolver for ProseMirror/Tiptap nodes.
- `forEachChild` (function): Child iterator for ProseMirror/Tiptap node content.
- `serializePrompt` (function): Serializes an editor JSON document into a legacy prompt string (tokens + text).
- `parsePromptToTiptap` (function): Parses a legacy prompt string into a minimal Tiptap doc with prompt token nodes.
*/

// tags: prompt, serialization, tiptap
import { Node, mergeAttributes } from '@tiptap/core'
import { VueNodeViewRenderer } from '@tiptap/vue-3'
import PromptTokenChip from './PromptTokenChip.vue'

export interface PromptTokenAttrs {
  kind: 'lora' | 'ti' | 'style'
  name: string
  weight: number
  enabled: boolean
}

export const PromptToken = Node.create({
  name: 'promptToken',
  inline: true,
  group: 'inline',
  atom: true,

  addAttributes() {
    return {
      kind: { default: 'lora' },
      name: { default: '' },
      weight: { default: 1.0 },
      enabled: { default: true },
    }
  },

  parseHTML() {
    return [
      { tag: 'span[data-token]' },
    ]
  },

  renderHTML({ HTMLAttributes }) {
    return ['span', mergeAttributes(HTMLAttributes, { 'data-token': '1' })]
  },

  addNodeView() {
    return VueNodeViewRenderer(PromptTokenChip)
  },
})

function parseStrictFiniteNumber(raw: string): number | null {
  const candidate = String(raw || '').trim()
  if (!candidate) return null
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(candidate)) return null
  const value = Number(candidate)
  if (!Number.isFinite(value)) return null
  return value
}

function nodeTypeName(node: any): string | undefined {
  if (!node || typeof node !== 'object') return undefined
  if (typeof node.type === 'string') return node.type
  if (typeof node.type?.name === 'string') return node.type.name
  return undefined
}

function forEachChild(node: any, visit: (child: any) => void): void {
  const content = (node as any)?.content
  if (!content) return
  if (Array.isArray(content)) {
    content.forEach(visit)
  } else if (typeof content.forEach === 'function') {
    content.forEach(visit)
  }
}

export function serializePrompt(doc: any): string {
  // Walk the ProseMirror document and build the legacy prompt string
  const parts: string[] = []
  function walk(node: any) {
    const type = nodeTypeName(node)
    if (type === 'text') {
      const text = typeof node.text === 'string' ? node.text : typeof node.textContent === 'string' ? node.textContent : ''
      if (text) parts.push(text)
    } else if (type === 'promptToken') {
      const { kind, name, weight, enabled } = (node as { attrs?: Partial<PromptTokenAttrs> }).attrs ?? {}
      if (enabled === false) return
      const safeWeight = Number.isFinite(Number(weight)) ? Number(weight) : 1
      if (kind === 'lora' && name) parts.push(`<lora:${name}:${safeWeight.toFixed(2)}>`)
      else if (kind === 'ti') parts.push(`(${name}:${safeWeight.toFixed(2)})`)
      else if (name) parts.push(String(name))
    }
    forEachChild(node, walk)
  }
  walk(doc)
  return parts.join('')
}

export function parsePromptToTiptap(prompt: string) {
  // Very small parser for <lora:name:w> and (name:w); leave everything else as text
  const nodes: any[] = []
  let i = 0
  const pushText = (s: string) => { if (s) nodes.push({ type: 'text', text: s }) }
  while (i < prompt.length) {
    // lora pattern
    if (prompt[i] === '<' && prompt.slice(i, i + 6) === '<lora:') {
      const end = prompt.indexOf('>', i + 6)
      if (end > -1) {
        const body = prompt.slice(i + 6, end)
        const splitAt = body.lastIndexOf(':')
        if (splitAt > 0) {
          const name = body.slice(0, splitAt).trim()
          const weightRaw = body.slice(splitAt + 1).trim()
          const parsedWeight = weightRaw ? parseStrictFiniteNumber(weightRaw) : 1
          if (name && parsedWeight !== null) {
            nodes.push({ type: 'promptToken', attrs: { kind: 'lora', name, weight: parsedWeight, enabled: true } })
            i = end + 1
            continue
          }
        }
      }
    }
    // TI pattern
    if (prompt[i] === '(') {
      const end = prompt.indexOf(')', i + 1)
      const colon = prompt.indexOf(':', i + 1)
      if (end > -1 && colon > -1 && colon < end) {
        const name = prompt.slice(i + 1, colon)
        const weightRaw = prompt.slice(colon + 1, end).trim()
        const parsedWeight = weightRaw ? parseStrictFiniteNumber(weightRaw) : 1
        if (parsedWeight !== null) {
          nodes.push({ type: 'promptToken', attrs: { kind: 'ti', name, weight: parsedWeight, enabled: true } })
          i = end + 1
          continue
        }
      }
    }
    // default: add single char and continue (coalesce later)
    pushText(prompt[i])
    i++
  }
  return { type: 'doc', content: [{ type: 'paragraph', content: nodes }] }
}
