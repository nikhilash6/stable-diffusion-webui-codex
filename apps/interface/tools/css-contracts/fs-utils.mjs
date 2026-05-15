/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared filesystem and path helpers for the frontend CSS verifier.
Provides explicit repo-relative path normalization, file walking, and line/column
resolution helpers used by the verifier modules.

Symbols (top-level; keep in sync; no ghosts):
- `normalizePath` (function): Normalizes a repo-relative path to forward-slash form without leading `./`.
- `resolveRepoPath` (function): Resolves one repo-relative path against the repo root.
- `toRelativeRepoPath` (function): Converts one absolute path back to normalized repo-relative form.
- `collectFiles` (function): Recursively collects files with matching extensions under one root.
- `readText` (function): Reads one UTF-8 text file.
- `readJson` (function): Reads one UTF-8 JSON file.
- `writeJson` (function): Writes one JSON file with deterministic indentation and a trailing newline.
- `getLineColumnFromOffset` (function): Resolves line/column metadata for one byte offset in a UTF-8 string.
*/

import fs from 'node:fs/promises'
import path from 'node:path'

export function normalizePath(value) {
  return String(value || '')
    .trim()
    .replace(/\\/g, '/')
    .replace(/^\.\//, '')
}

export function resolveRepoPath(repoRoot, relativePath) {
  return path.resolve(repoRoot, normalizePath(relativePath))
}

export function toRelativeRepoPath(repoRoot, absolutePath) {
  return normalizePath(path.relative(repoRoot, absolutePath))
}

export async function collectFiles(root, extensions) {
  const matches = []
  const allowed = new Set((extensions || []).map((entry) => String(entry || '').trim()).filter(Boolean))

  async function walk(current) {
    const entries = await fs.readdir(current, { withFileTypes: true })
    for (const entry of entries) {
      const absolutePath = path.join(current, entry.name)
      if (entry.isDirectory()) {
        await walk(absolutePath)
        continue
      }
      if (!entry.isFile()) continue
      if (allowed.size > 0 && !allowed.has(path.extname(entry.name))) continue
      matches.push(absolutePath)
    }
  }

  await walk(root)
  return matches.sort((left, right) => left.localeCompare(right))
}

export async function readText(filePath) {
  return fs.readFile(filePath, 'utf8')
}

export async function readJson(filePath) {
  return JSON.parse(await readText(filePath))
}

export async function writeJson(filePath, value) {
  const body = `${JSON.stringify(value, null, 2)}\n`
  await fs.mkdir(path.dirname(filePath), { recursive: true })
  await fs.writeFile(filePath, body, 'utf8')
}

export function getLineColumnFromOffset(content, offset) {
  const safeOffset = Math.max(0, Math.min(Number(offset) || 0, content.length))
  const before = content.slice(0, safeOffset)
  const lines = before.split('\n')
  return {
    line: lines.length,
    column: lines[lines.length - 1].length + 1,
  }
}
