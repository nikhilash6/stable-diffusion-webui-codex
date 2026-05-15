<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tiptap-based prompt editor with token insertion.
Provides a rich prompt editor backed by Tiptap, serializing `PromptToken` nodes back into the legacy prompt string format and exposing `insertToken(...)` for UI actions.

Symbols (top-level; keep in sync; no ghosts):
- `PromptEditor` (component): Prompt editor that emits serialized prompt strings and exposes token insertion.
- `insertToken` (function): Inserts a `promptToken` node into the editor (LoRA/TI/style) with a weight.
-->

<template>
  <div>
    <EditorContent :editor="editor" />
  </div>
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch, nextTick } from 'vue'
import { EditorContent, useEditor } from '@tiptap/vue-3'
import StarterKit from '@tiptap/starter-kit'
import { PromptToken, parsePromptToTiptap, serializePrompt } from './PromptToken'

const props = defineProps<{ modelValue: string }>()
const emit = defineEmits<{ (e:'update:modelValue', v:string): void }>()

const editor = useEditor({
  extensions: [StarterKit.configure({ history: {} }), PromptToken],
  content: parsePromptToTiptap(props.modelValue || ''),
  onUpdate: ({ editor }) => {
    const json = editor.getJSON()
    emit('update:modelValue', serializePrompt(json))
  },
})

watch(() => props.modelValue, (v) => {
  const ed = editor.value
  if (!ed) return
  const current = serializePrompt(ed.getJSON())
  if ((v || '') !== current) ed.commands.setContent(parsePromptToTiptap(v || ''))
})

onBeforeUnmount(() => { editor.value?.destroy() })

function insertToken(kind: 'lora'|'ti'|'style', name: string, weight = 1.0): void {
  const ed = editor.value
  if (!ed) return
  ed.chain().focus().insertContent({ type: 'promptToken', attrs: { kind, name, weight, enabled: true } }).run()
}

defineExpose({ insertToken })
</script>
