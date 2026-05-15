/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical frontend CSS contract verifier.
Loads the runtime CSS topology/config owners, analyzes CSS + source usage + docs,
writes a JSON report, and fails loud when topology or budget contracts drift.

Symbols (top-level; keep in sync; no ghosts):
- `main` (function): Runs the verifier and exits non-zero on contract drift.
*/

import { loadConfig } from './css-contracts/config.mjs'
import { analyzeTopologyContracts } from './css-contracts/topology.mjs'
import { analyzeCssContracts } from './css-contracts/css-analysis.mjs'
import { analyzeSourceContracts } from './css-contracts/source-analysis.mjs'
import { formatSummary, evaluateBudgets, writeReport } from './css-contracts/report.mjs'
import { readText } from './css-contracts/fs-utils.mjs'

function buildMessageList(items, mapper) {
  return (items || []).map(mapper)
}

async function analyzeDocsAudit(config) {
  const findings = []
  for (const rule of config.contracts.docsAudit.forbiddenPatterns) {
    for (const file of rule.files) {
      const content = await readText(file.startsWith('/') ? file : `${config.repoRoot}/${file}`)
      if (!content.includes(rule.pattern)) continue
      findings.push({
        file,
        pattern: rule.pattern,
        reason: rule.reason,
        message: `${file} still contains forbidden pattern \`${rule.pattern}\` (${rule.reason})`,
      })
    }
  }
  return findings
}

function buildMetrics(cssAnalysis, sourceAnalysis) {
  const runtimeWrittenVariables = new Set(
    sourceAnalysis.cssVariableWrites.filter((entry) => entry.owned).map((entry) => entry.target),
  )
  const frameworkOwnedDefinedVariables = new Set(
    cssAnalysis.definedOnlyVariables.filter(
      (name) => name.startsWith('--breakpoint-') || name.startsWith('--color-'),
    ),
  )

  return {
    unusedClasses: sourceAnalysis.unusedClasses.length,
    missingClasses: sourceAnalysis.missingClasses.length,
    definedOnlyVariables: cssAnalysis.definedOnlyVariables.filter((name) => !frameworkOwnedDefinedVariables.has(name))
      .length,
    referencedOnlyVariables: cssAnalysis.referencedOnlyVariables.filter((name) => !runtimeWrittenVariables.has(name))
      .length,
    duplicateDeclarations: cssAnalysis.duplicateDeclarations.length,
    conflictDeclarations: cssAnalysis.conflictDeclarations.length,
    noEffectDeclarations: cssAnalysis.noEffectDeclarations.length,
    selectorDuplicates: cssAnalysis.selectorDuplicates.length,
    inlineStyleAttributes: sourceAnalysis.inlineStyleAttributes.length,
    scopedStyleBlocks: sourceAnalysis.scopedStyleBlocks.length,
    unownedBoundStyleBindings: sourceAnalysis.unownedBoundStyleBindings.length,
    unownedDomStyleWrites: sourceAnalysis.unownedDomStyleWrites.length,
  }
}

function printFailures(title, failures) {
  if (!failures || failures.length === 0) return
  console.error(`[verify:css-contracts] ${title}`)
  for (const failure of failures) {
    console.error(`- ${failure.message}`)
  }
}

function buildUnmatchedTypedExceptionFailures(entries, bucketName, targetLabel) {
  return buildMessageList(entries, (entry) => ({
    message: `unmatched typed exception ${bucketName}: ${entry.file} -> ${targetLabel} \`${entry.target}\``,
  }))
}

export async function main() {
  const config = await loadConfig()
  const topology = await analyzeTopologyContracts(config)
  const cssAnalysis = await analyzeCssContracts(config)
  const sourceAnalysis = await analyzeSourceContracts(config, cssAnalysis)
  const docsAudit = await analyzeDocsAudit(config)
  const metrics = buildMetrics(cssAnalysis, sourceAnalysis)
  const budgetEvaluation = evaluateBudgets(metrics, config.contracts.budgets)

  const contractFailures = [
    ...topology.findings,
    ...docsAudit,
    ...buildMessageList(cssAnalysis.forbiddenAtRuleFindings, (finding) => ({ message: finding.message })),
    ...buildMessageList(cssAnalysis.forbiddenImportFindings, (finding) => ({ message: finding.message })),
    ...buildUnmatchedTypedExceptionFailures(sourceAnalysis.unmatchedBoundStyleBindings, 'boundStyleBindings', 'expression'),
    ...buildUnmatchedTypedExceptionFailures(sourceAnalysis.unmatchedDomStyleWrites, 'domStyleWrites', 'target'),
    ...buildUnmatchedTypedExceptionFailures(sourceAnalysis.unmatchedCssVariableWrites, 'cssVariableWrites', 'target'),
    ...buildMessageList(sourceAnalysis.unownedCssVariableWrites, (finding) => ({
      message: `${finding.file}:${finding.line} unowned CSS variable write \`${finding.target}\``,
    })),
  ]

  const ok = contractFailures.length === 0 && budgetEvaluation.ok
  const summary = formatSummary(ok, metrics, config.contracts.budgets)
  console.log(summary)

  const report = {
    generatedAt: new Date().toISOString(),
    ok,
    summary,
    configPath: config.configPathRelative,
    topology,
    metrics,
    budgets: config.contracts.budgets,
    budgetFailures: budgetEvaluation.failures,
    docsAudit: { findings: docsAudit },
    plainCssGuard: {
      forbiddenAtRuleFindings: cssAnalysis.forbiddenAtRuleFindings,
      forbiddenImportFindings: cssAnalysis.forbiddenImportFindings,
    },
    cssAnalysis: {
      files: cssAnalysis.files,
      definedClassNames: cssAnalysis.definedClassNames,
      definedOnlyVariables: cssAnalysis.definedOnlyVariables,
      referencedOnlyVariables: cssAnalysis.referencedOnlyVariables,
      duplicateDeclarations: cssAnalysis.duplicateDeclarations,
      conflictDeclarations: cssAnalysis.conflictDeclarations,
      noEffectDeclarations: cssAnalysis.noEffectDeclarations,
      selectorDuplicates: cssAnalysis.selectorDuplicates,
    },
    sourceAnalysis: {
      sourceFiles: sourceAnalysis.sourceFiles,
      usedClassNames: sourceAnalysis.usedClassNames,
      unusedClasses: sourceAnalysis.unusedClasses,
      missingClasses: sourceAnalysis.missingClasses,
      inlineStyleAttributes: sourceAnalysis.inlineStyleAttributes,
      boundStyleBindings: sourceAnalysis.boundStyleBindings,
      unownedBoundStyleBindings: sourceAnalysis.unownedBoundStyleBindings,
      unmatchedBoundStyleBindings: sourceAnalysis.unmatchedBoundStyleBindings,
      domStyleWrites: sourceAnalysis.domStyleWrites,
      unownedDomStyleWrites: sourceAnalysis.unownedDomStyleWrites,
      unmatchedDomStyleWrites: sourceAnalysis.unmatchedDomStyleWrites,
      cssVariableWrites: sourceAnalysis.cssVariableWrites,
      unownedCssVariableWrites: sourceAnalysis.unownedCssVariableWrites,
      unmatchedCssVariableWrites: sourceAnalysis.unmatchedCssVariableWrites,
      styleBlocks: sourceAnalysis.styleBlocks,
      scopedStyleBlocks: sourceAnalysis.scopedStyleBlocks,
    },
    contractFailures,
  }

  await writeReport(config.absolutePaths.reportOutputFile, report)

  printFailures('Contract failures', contractFailures)
  printFailures('Budget failures', budgetEvaluation.failures)

  if (!ok) {
    process.exitCode = 1
  }
}

main().catch((error) => {
  console.error('[verify:css-contracts] fatal error')
  console.error(error instanceof Error ? error.stack || error.message : error)
  process.exitCode = 1
})
