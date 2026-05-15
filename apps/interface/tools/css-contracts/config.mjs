/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Config loader and validator for the frontend CSS verifier.
Loads `css-contracts.config.json`, validates the required path/contract keys,
and normalizes the typed exception tables used by the verifier.

Symbols (top-level; keep in sync; no ghosts):
- `loadConfig` (function): Loads and validates the verifier config plus the resolved repo-root metadata.
*/

import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { normalizePath, readJson, resolveRepoPath } from './fs-utils.mjs'

function assert(condition, message) {
  if (!condition) {
    throw new Error(message)
  }
}

function readString(value, key) {
  const normalized = String(value || '').trim()
  assert(normalized.length > 0, `missing/invalid string: ${key}`)
  return normalized
}

function readStringArray(value, key) {
  assert(Array.isArray(value), `missing/invalid string array: ${key}`)
  return value.map((entry, index) => readString(entry, `${key}[${index}]`))
}

function readNumber(value, key) {
  const numeric = Number(value)
  assert(Number.isFinite(numeric), `missing/invalid number: ${key}`)
  return numeric
}

function readPatternChecks(value, key) {
  assert(Array.isArray(value), `missing/invalid array: ${key}`)
  return value.map((entry, index) => ({
    pattern: readString(entry?.pattern, `${key}[${index}].pattern`),
    files: readStringArray(entry?.files, `${key}[${index}].files`).map(normalizePath),
    reason: readString(entry?.reason, `${key}[${index}].reason`),
  }))
}

function readTypedExceptionEntries(value, key, targetKey) {
  assert(Array.isArray(value), `missing/invalid array: ${key}`)
  return value.map((entry, index) => ({
    file: normalizePath(readString(entry?.file, `${key}[${index}].file`)),
    target: readString(entry?.[targetKey], `${key}[${index}].${targetKey}`),
    rationale: readString(entry?.rationale, `${key}[${index}].rationale`),
  }))
}

export async function loadConfig() {
  const moduleDir = path.dirname(fileURLToPath(import.meta.url))
  const toolsDir = path.resolve(moduleDir, '..')
  const repoRoot = path.resolve(toolsDir, '..', '..', '..')
  const configPath = path.resolve(toolsDir, 'css-contracts.config.json')
  const rawConfig = await readJson(configPath)

  const config = {
    repoRoot,
    configPath,
    configPathRelative: normalizePath(path.relative(repoRoot, configPath)),
    paths: {
      bootstrapFile: normalizePath(readString(rawConfig?.paths?.bootstrapFile, 'paths.bootstrapFile')),
      runtimeCssEntryFile: normalizePath(readString(rawConfig?.paths?.runtimeCssEntryFile, 'paths.runtimeCssEntryFile')),
      stylesRoot: normalizePath(readString(rawConfig?.paths?.stylesRoot, 'paths.stylesRoot')),
      sourceRoot: normalizePath(readString(rawConfig?.paths?.sourceRoot, 'paths.sourceRoot')),
      sourceExtensions: readStringArray(rawConfig?.paths?.sourceExtensions, 'paths.sourceExtensions'),
      topologySource: normalizePath(readString(rawConfig?.paths?.topologySource, 'paths.topologySource')),
      reportOutputFile: normalizePath(readString(rawConfig?.paths?.reportOutputFile, 'paths.reportOutputFile')),
    },
    contracts: {
      docsAudit: {
        authorityFiles: readStringArray(rawConfig?.contracts?.docsAudit?.authorityFiles, 'contracts.docsAudit.authorityFiles').map(normalizePath),
        forbiddenPatterns: readPatternChecks(rawConfig?.contracts?.docsAudit?.forbiddenPatterns, 'contracts.docsAudit.forbiddenPatterns'),
      },
      plainCssGuard: {
        forbiddenAtRules: readStringArray(rawConfig?.contracts?.plainCssGuard?.forbiddenAtRules, 'contracts.plainCssGuard.forbiddenAtRules').map((value) => value.toLowerCase()),
        forbiddenImportPatterns: readStringArray(rawConfig?.contracts?.plainCssGuard?.forbiddenImportPatterns, 'contracts.plainCssGuard.forbiddenImportPatterns'),
      },
      budgets: {
        maxUnusedClasses: readNumber(rawConfig?.contracts?.budgets?.maxUnusedClasses, 'contracts.budgets.maxUnusedClasses'),
        maxMissingClasses: readNumber(rawConfig?.contracts?.budgets?.maxMissingClasses, 'contracts.budgets.maxMissingClasses'),
        maxDefinedOnlyVariables: readNumber(rawConfig?.contracts?.budgets?.maxDefinedOnlyVariables, 'contracts.budgets.maxDefinedOnlyVariables'),
        maxReferencedOnlyVariables: readNumber(rawConfig?.contracts?.budgets?.maxReferencedOnlyVariables, 'contracts.budgets.maxReferencedOnlyVariables'),
        maxDuplicateDeclarations: readNumber(rawConfig?.contracts?.budgets?.maxDuplicateDeclarations, 'contracts.budgets.maxDuplicateDeclarations'),
        maxConflictDeclarations: readNumber(rawConfig?.contracts?.budgets?.maxConflictDeclarations, 'contracts.budgets.maxConflictDeclarations'),
        maxNoEffectDeclarations: readNumber(rawConfig?.contracts?.budgets?.maxNoEffectDeclarations, 'contracts.budgets.maxNoEffectDeclarations'),
        maxSelectorDuplicates: readNumber(rawConfig?.contracts?.budgets?.maxSelectorDuplicates, 'contracts.budgets.maxSelectorDuplicates'),
        maxInlineStyleAttributes: readNumber(rawConfig?.contracts?.budgets?.maxInlineStyleAttributes, 'contracts.budgets.maxInlineStyleAttributes'),
        maxScopedStyleBlocks: readNumber(rawConfig?.contracts?.budgets?.maxScopedStyleBlocks, 'contracts.budgets.maxScopedStyleBlocks'),
        maxUnownedBoundStyleBindings: readNumber(rawConfig?.contracts?.budgets?.maxUnownedBoundStyleBindings, 'contracts.budgets.maxUnownedBoundStyleBindings'),
        maxUnownedDomStyleWrites: readNumber(rawConfig?.contracts?.budgets?.maxUnownedDomStyleWrites, 'contracts.budgets.maxUnownedDomStyleWrites'),
      },
      typedExceptions: {
        boundStyleBindings: readTypedExceptionEntries(rawConfig?.contracts?.typedExceptions?.boundStyleBindings, 'contracts.typedExceptions.boundStyleBindings', 'expression'),
        domStyleWrites: readTypedExceptionEntries(rawConfig?.contracts?.typedExceptions?.domStyleWrites, 'contracts.typedExceptions.domStyleWrites', 'target'),
        cssVariableWrites: readTypedExceptionEntries(rawConfig?.contracts?.typedExceptions?.cssVariableWrites, 'contracts.typedExceptions.cssVariableWrites', 'target'),
      },
    },
  }

  const topologyAbsolutePath = resolveRepoPath(repoRoot, config.paths.topologySource)
  const topologyModule = await import(pathToFileURL(topologyAbsolutePath).href)
  assert(Array.isArray(topologyModule.STYLE_TOPOLOGY), 'topology module must export STYLE_TOPOLOGY')

  config.absolutePaths = {
    bootstrapFile: resolveRepoPath(repoRoot, config.paths.bootstrapFile),
    runtimeCssEntryFile: resolveRepoPath(repoRoot, config.paths.runtimeCssEntryFile),
    stylesRoot: resolveRepoPath(repoRoot, config.paths.stylesRoot),
    sourceRoot: resolveRepoPath(repoRoot, config.paths.sourceRoot),
    topologySource: topologyAbsolutePath,
    reportOutputFile: resolveRepoPath(repoRoot, config.paths.reportOutputFile),
  }

  config.topology = topologyModule
  return config
}
