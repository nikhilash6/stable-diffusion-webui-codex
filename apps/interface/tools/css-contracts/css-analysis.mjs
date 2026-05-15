/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Plain-CSS analysis for the frontend CSS verifier.
Parses the runtime stylesheet tree with PostCSS, extracts defined classes and
custom properties, and reports fail-loud drift such as forbidden at-rules,
forbidden imports, duplicated/conflicting declarations, and selector shadowing.

Symbols (top-level; keep in sync; no ghosts):
- `analyzeCssContracts` (function): Parses runtime CSS owners and returns the CSS-side contract report.
*/

import postcss from 'postcss'
import { normalizePath, readText, resolveRepoPath } from './fs-utils.mjs'

const CLASS_NAME_PATTERN = /\.(-?[_a-zA-Z]+[_a-zA-Z0-9-]*)/g
const CSS_VARIABLE_REFERENCE_PATTERN = /var\(\s*(--[_a-zA-Z0-9-]+)/g

function buildLocation(file, node) {
  return {
    file,
    line: node?.source?.start?.line ?? 1,
    column: node?.source?.start?.column ?? 1,
  }
}

function buildDeclarationMessage(selector, property, value) {
  return `${selector} -> ${property}: ${value}`
}

function normalizeCssValue(value) {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

function extractClassNames(selector) {
  const classes = []
  let match = CLASS_NAME_PATTERN.exec(selector)
  while (match) {
    classes.push(match[1])
    match = CLASS_NAME_PATTERN.exec(selector)
  }
  CLASS_NAME_PATTERN.lastIndex = 0
  return [...new Set(classes)]
}

function extractVariableReferences(value) {
  const references = []
  let match = CSS_VARIABLE_REFERENCE_PATTERN.exec(value)
  while (match) {
    references.push(match[1])
    match = CSS_VARIABLE_REFERENCE_PATTERN.exec(value)
  }
  CSS_VARIABLE_REFERENCE_PATTERN.lastIndex = 0
  return references
}

function getRuleContext(ruleNode) {
  const contexts = []
  let cursor = ruleNode?.parent
  while (cursor) {
    if (cursor.type === 'atrule') {
      const name = String(cursor.name || '').trim()
      const params = String(cursor.params || '').trim()
      contexts.push(params ? `@${name} ${params}` : `@${name}`)
    }
    cursor = cursor.parent
  }
  return contexts.reverse().join(' | ') || '<root>'
}

function isKeyframesRule(ruleNode) {
  let cursor = ruleNode?.parent
  while (cursor) {
    if (cursor.type === 'atrule' && String(cursor.name || '').toLowerCase().includes('keyframes')) {
      return true
    }
    cursor = cursor.parent
  }
  return false
}

export async function analyzeCssContracts(config) {
  const cssFiles = [config.paths.runtimeCssEntryFile, ...config.topology.RUNTIME_CSS_TOPOLOGY.map((entry) => normalizePath(entry.sourcePath))]
  const fileSet = Array.from(new Set(cssFiles)).sort((left, right) => left.localeCompare(right))
  const definedClassNames = new Set()
  const variableDefinitions = new Map()
  const variableReferences = new Map()
  const selectorOccurrences = new Map()
  const duplicateDeclarations = []
  const conflictDeclarations = []
  const noEffectDeclarations = []
  const forbiddenAtRuleFindings = []
  const forbiddenImportFindings = []

  for (const relativeFile of fileSet) {
    const absoluteFile = resolveRepoPath(config.repoRoot, relativeFile)
    const content = await readText(absoluteFile)
    const root = postcss.parse(content, { from: absoluteFile })

    root.walkAtRules((atRule) => {
      const ruleName = `@${String(atRule.name || '').toLowerCase()}`
      if (config.contracts.plainCssGuard.forbiddenAtRules.includes(ruleName)) {
        forbiddenAtRuleFindings.push({
          file: relativeFile,
          line: atRule.source?.start?.line ?? 1,
          column: atRule.source?.start?.column ?? 1,
          atRule: `@${String(atRule.name || '')}`,
          message: `${relativeFile}:${atRule.source?.start?.line ?? 1} uses forbidden at-rule ${ruleName}`,
        })
      }

      if (String(atRule.name || '').toLowerCase() !== 'import') return
      const params = String(atRule.params || '')
      for (const pattern of config.contracts.plainCssGuard.forbiddenImportPatterns) {
        if (!params.includes(pattern)) continue
        forbiddenImportFindings.push({
          file: relativeFile,
          line: atRule.source?.start?.line ?? 1,
          column: atRule.source?.start?.column ?? 1,
          pattern,
          message: `${relativeFile}:${atRule.source?.start?.line ?? 1} imports forbidden pattern ${pattern}`,
        })
      }
    })

    root.walkDecls((declNode) => {
      const property = String(declNode.prop || '').trim()
      const value = normalizeCssValue(declNode.value)
      const relativeLocation = buildLocation(relativeFile, declNode)
      const selector =
        declNode.parent?.type === 'rule'
          ? String(declNode.parent.selector || '').trim()
          : `<${declNode.parent?.type || 'root'}>`
      const context = getRuleContext(declNode.parent?.type === 'rule' ? declNode.parent : declNode)

      if (property.startsWith('--')) {
        const list = variableDefinitions.get(property) ?? []
        list.push({ value, location: relativeLocation, selector, context })
        variableDefinitions.set(property, list)
      }

      for (const variableName of extractVariableReferences(value)) {
        const list = variableReferences.get(variableName) ?? []
        list.push({ value, location: relativeLocation, selector, context })
        variableReferences.set(variableName, list)
      }
    })

    root.walkRules((rule) => {
      if (isKeyframesRule(rule)) return
      const selector = String(rule.selector || '').trim()
      if (!selector) return

      for (const className of extractClassNames(selector)) {
        definedClassNames.add(className)
      }

      const selectorKey = selector.replace(/\s+/g, ' ').trim()
      const occurrenceList = selectorOccurrences.get(selectorKey) ?? []
      occurrenceList.push({ file: relativeFile, line: rule.source?.start?.line ?? 1 })
      selectorOccurrences.set(selectorKey, occurrenceList)

      const byProperty = new Map()
      rule.nodes
        .filter((node) => node.type === 'decl')
        .forEach((declNode) => {
          const property = String(declNode.prop || '').trim()
          const value = normalizeCssValue(declNode.value)
          const location = buildLocation(relativeFile, declNode)

          const prior = byProperty.get(property) ?? []
          if (prior.some((entry) => entry.value === value)) {
            duplicateDeclarations.push({
              file: relativeFile,
              selector,
              property,
              value,
              location,
              context: getRuleContext(rule),
              message: buildDeclarationMessage(selector, property, value),
            })
          }
          if (prior.some((entry) => entry.value !== value)) {
            conflictDeclarations.push({
              file: relativeFile,
              selector,
              property,
              value,
              location,
              context: getRuleContext(rule),
              message: buildDeclarationMessage(selector, property, value),
            })
          }
          if (prior.length > 0 && prior[prior.length - 1].value === value) {
            noEffectDeclarations.push({
              file: relativeFile,
              selector,
              property,
              value,
              location,
              context: getRuleContext(rule),
              message: buildDeclarationMessage(selector, property, value),
            })
          }
          prior.push({ value, location })
          byProperty.set(property, prior)
        })
    })
  }

  const definedCssVariables = Array.from(variableDefinitions.keys()).sort((left, right) => left.localeCompare(right))
  const referencedCssVariables = Array.from(variableReferences.keys()).sort((left, right) => left.localeCompare(right))
  const definedOnlyVariables = definedCssVariables.filter((name) => !variableReferences.has(name))
  const referencedOnlyVariables = referencedCssVariables.filter((name) => !variableDefinitions.has(name))
  const selectorDuplicates = Array.from(selectorOccurrences.entries())
    .map(([selector, locations]) => ({
      selector,
      locations: locations.slice().sort((left, right) => {
        if (left.file !== right.file) return left.file.localeCompare(right.file)
        return left.line - right.line
      }),
      fileCount: new Set(locations.map((entry) => entry.file)).size,
    }))
    .filter((entry) => entry.fileCount > 1)
    .sort((left, right) => left.selector.localeCompare(right.selector))

  return {
    files: fileSet,
    definedClassNames: Array.from(definedClassNames).sort((left, right) => left.localeCompare(right)),
    definedCssVariables,
    referencedCssVariables,
    definedOnlyVariables,
    referencedOnlyVariables,
    duplicateDeclarations,
    conflictDeclarations,
    noEffectDeclarations,
    selectorDuplicates,
    forbiddenAtRuleFindings,
    forbiddenImportFindings,
  }
}
