/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Summary/report helpers for the frontend CSS verifier.
Formats the fail-loud verifier summary, evaluates the configured budgets, and
writes the deterministic JSON artifact consumed by maintainers.

Symbols (top-level; keep in sync; no ghosts):
- `evaluateBudgets` (function): Compares measured metrics against configured CSS budgets.
- `formatSummary` (function): Renders the one-line verifier summary.
- `writeReport` (function): Writes the JSON report artifact with deterministic formatting.
*/

import { writeJson } from './fs-utils.mjs'

function compareLimit(metricName, actual, limit) {
  return actual <= limit
    ? null
    : {
        metric: metricName,
        actual,
        limit,
        message: `${metricName} exceeded budget (${actual} > ${limit})`,
      }
}

export function evaluateBudgets(metrics, budgets) {
  const failures = [
    compareLimit('unusedClasses', metrics.unusedClasses, budgets.maxUnusedClasses),
    compareLimit('missingClasses', metrics.missingClasses, budgets.maxMissingClasses),
    compareLimit('definedOnlyVariables', metrics.definedOnlyVariables, budgets.maxDefinedOnlyVariables),
    compareLimit('referencedOnlyVariables', metrics.referencedOnlyVariables, budgets.maxReferencedOnlyVariables),
    compareLimit('duplicateDeclarations', metrics.duplicateDeclarations, budgets.maxDuplicateDeclarations),
    compareLimit('conflictDeclarations', metrics.conflictDeclarations, budgets.maxConflictDeclarations),
    compareLimit('noEffectDeclarations', metrics.noEffectDeclarations, budgets.maxNoEffectDeclarations),
    compareLimit('selectorDuplicates', metrics.selectorDuplicates, budgets.maxSelectorDuplicates),
    compareLimit('inlineStyleAttributes', metrics.inlineStyleAttributes, budgets.maxInlineStyleAttributes),
    compareLimit('scopedStyleBlocks', metrics.scopedStyleBlocks, budgets.maxScopedStyleBlocks),
    compareLimit('unownedBoundStyleBindings', metrics.unownedBoundStyleBindings, budgets.maxUnownedBoundStyleBindings),
    compareLimit('unownedDomStyleWrites', metrics.unownedDomStyleWrites, budgets.maxUnownedDomStyleWrites),
  ].filter(Boolean)

  return {
    ok: failures.length === 0,
    failures,
  }
}

export function formatSummary(ok, metrics, budgets) {
  const pairs = [
    `unused=${metrics.unusedClasses}/${budgets.maxUnusedClasses}`,
    `missing=${metrics.missingClasses}/${budgets.maxMissingClasses}`,
    `definedOnlyVars=${metrics.definedOnlyVariables}/${budgets.maxDefinedOnlyVariables}`,
    `referencedOnlyVars=${metrics.referencedOnlyVariables}/${budgets.maxReferencedOnlyVariables}`,
    `duplicates=${metrics.duplicateDeclarations}/${budgets.maxDuplicateDeclarations}`,
    `conflicts=${metrics.conflictDeclarations}/${budgets.maxConflictDeclarations}`,
    `noEffect=${metrics.noEffectDeclarations}/${budgets.maxNoEffectDeclarations}`,
    `selectorDupes=${metrics.selectorDuplicates}/${budgets.maxSelectorDuplicates}`,
    `inline=${metrics.inlineStyleAttributes}/${budgets.maxInlineStyleAttributes}`,
    `scoped=${metrics.scopedStyleBlocks}/${budgets.maxScopedStyleBlocks}`,
    `bound=${metrics.unownedBoundStyleBindings}/${budgets.maxUnownedBoundStyleBindings}`,
    `domStyle=${metrics.unownedDomStyleWrites}/${budgets.maxUnownedDomStyleWrites}`,
  ]
  return `[verify:css-contracts] ${ok ? 'OK' : 'FAIL'} ${pairs.join(' | ')}`
}

export async function writeReport(reportPath, report) {
  await writeJson(reportPath, report)
}
