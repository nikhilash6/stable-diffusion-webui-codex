/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime-topology contract checks for the frontend CSS verifier.
Validates the only allowed bootstrap path (`src/main.ts -> src/styles.css`),
checks the ordered local imports declared by `src/styles.css`, and fails loud
when reference-only sheets or stale topology owners drift into runtime.

Symbols (top-level; keep in sync; no ghosts):
- `analyzeTopologyContracts` (function): Collects topology findings and the resolved runtime/reference inventories.
*/

import path from 'node:path'
import { collectFiles, normalizePath, readText, toRelativeRepoPath } from './fs-utils.mjs'

function buildFinding(type, message, extra = {}) {
  return { type, message, ...extra }
}

function toRelativeImport(fromFileRelative, specifier) {
  if (!String(specifier || '').startsWith('.')) return null
  return normalizePath(path.join(path.dirname(fromFileRelative), specifier))
}

function collectJsCssImports(content) {
  const imports = []
  const importRegex = /import\s+(?:[^;]*?from\s+)?['"]([^'"]+\.css)['"]/g
  let match = importRegex.exec(content)
  while (match) {
    imports.push(String(match[1] || '').trim())
    match = importRegex.exec(content)
  }
  return imports
}

function collectCssImports(content) {
  const imports = []
  const importRegex = /@import\s+(?:url\()?['"]([^'"]+)['"]\)?/g
  let match = importRegex.exec(content)
  while (match) {
    imports.push(String(match[1] || '').trim())
    match = importRegex.exec(content)
  }
  return imports
}

function compareOrderedLists(actual, expected) {
  if (actual.length !== expected.length) return false
  return actual.every((entry, index) => entry === expected[index])
}

export async function analyzeTopologyContracts(config) {
  const bootstrapContent = await readText(config.absolutePaths.bootstrapFile)
  const runtimeEntryContent = await readText(config.absolutePaths.runtimeCssEntryFile)

  const bootstrapCssImports = collectJsCssImports(bootstrapContent).map((specifier) =>
    toRelativeImport(config.paths.bootstrapFile, specifier),
  )

  const runtimeCssImports = collectCssImports(runtimeEntryContent)
    .map((specifier) => toRelativeImport(config.paths.runtimeCssEntryFile, specifier))
    .filter(Boolean)

  const styleTopology = Array.isArray(config.topology.STYLE_TOPOLOGY) ? config.topology.STYLE_TOPOLOGY : []
  const runtimeTopology = Array.isArray(config.topology.RUNTIME_CSS_TOPOLOGY)
    ? config.topology.RUNTIME_CSS_TOPOLOGY
    : []
  const referenceOnlyTopology = Array.isArray(config.topology.REFERENCE_ONLY_CSS_TOPOLOGY)
    ? config.topology.REFERENCE_ONLY_CSS_TOPOLOGY
    : []

  const expectedBootstrapImport = config.paths.runtimeCssEntryFile
  const expectedRuntimeImports = runtimeTopology.map((entry) => normalizePath(entry.sourcePath))
  const referenceOnlyImports = referenceOnlyTopology.map((entry) => normalizePath(entry.sourcePath))
  const styleRootPrefix = `${config.paths.stylesRoot}/`
  const styleTreeFiles = (await collectFiles(config.absolutePaths.stylesRoot, [])).map((absolutePath) =>
    toRelativeRepoPath(config.repoRoot, absolutePath),
  )
  const styleTreeCssFiles = styleTreeFiles.filter((file) => path.extname(file) === '.css')
  const zoneIdentifierFiles = styleTreeFiles.filter((file) => file.includes(':Zone.Identifier'))
  const topologyStylesRootFiles = styleTopology
    .map((entry) => normalizePath(entry.sourcePath))
    .filter((file) => file.startsWith(styleRootPrefix))
  const topologyStylesRootSet = new Set(topologyStylesRootFiles)
  const styleTreeCssFileSet = new Set(styleTreeCssFiles)
  const findings = []

  if (styleTopology.length === 0) {
    findings.push(
      buildFinding('missing-style-topology', 'STYLE_TOPOLOGY is empty; the runtime CSS inventory lost its owner.'),
    )
  }

  const rootEntry = styleTopology[0]
  if (!rootEntry || normalizePath(rootEntry.sourcePath) !== config.paths.runtimeCssEntryFile || rootEntry.kind !== 'root-entry') {
    findings.push(
      buildFinding(
        'invalid-root-entry',
        'style-topology.mjs must start with the runtime root entry for apps/interface/src/styles.css.',
        { expectedRootEntry: config.paths.runtimeCssEntryFile, actualRootEntry: rootEntry?.sourcePath ?? null },
      ),
    )
  }

  if (bootstrapCssImports.length !== 1 || bootstrapCssImports[0] !== expectedBootstrapImport) {
    findings.push(
      buildFinding(
        'bootstrap-css-drift',
        'src/main.ts must import exactly one runtime stylesheet: ./styles.css.',
        { actualImports: bootstrapCssImports, expectedImports: [expectedBootstrapImport] },
      ),
    )
  }

  if (!compareOrderedLists(runtimeCssImports, expectedRuntimeImports)) {
    findings.push(
      buildFinding(
        'runtime-import-order-drift',
        'src/styles.css local imports no longer match the ordered runtime inventory declared in style-topology.mjs.',
        { actualImports: runtimeCssImports, expectedImports: expectedRuntimeImports },
      ),
    )
  }

  const leakedReferenceOnlyImports = runtimeCssImports.filter((entry) => referenceOnlyImports.includes(entry))
  if (leakedReferenceOnlyImports.length > 0) {
    findings.push(
      buildFinding(
        'reference-only-runtime-leak',
        'Reference-only stylesheets leaked into src/styles.css runtime imports.',
        { imports: leakedReferenceOnlyImports },
      ),
    )
  }

  const topologyPaths = new Set(styleTopology.map((entry) => normalizePath(entry.sourcePath)))
  const missingRuntimeFiles = expectedRuntimeImports.filter((entry) => !topologyPaths.has(entry))
  if (missingRuntimeFiles.length > 0) {
    findings.push(
      buildFinding(
        'runtime-topology-gap',
        'RUNTIME_CSS_TOPOLOGY contains entries not present in STYLE_TOPOLOGY.',
        { missingRuntimeFiles },
      ),
    )
  }

  const topologyFilesOutsideStylesRoot = styleTopology
    .map((entry) => normalizePath(entry.sourcePath))
    .filter((file) => file !== config.paths.runtimeCssEntryFile && !file.startsWith(styleRootPrefix))
  if (topologyFilesOutsideStylesRoot.length > 0) {
    findings.push(
      buildFinding(
        'topology-owner-drift',
        'style-topology.mjs contains non-root stylesheet entries outside apps/interface/src/styles/**.',
        { files: topologyFilesOutsideStylesRoot },
      ),
    )
  }

  const stylesTreeMissingFromTopology = styleTreeCssFiles.filter((file) => !topologyStylesRootSet.has(file))
  if (stylesTreeMissingFromTopology.length > 0) {
    findings.push(
      buildFinding(
        'styles-tree-missing-from-topology',
        'Tracked CSS files under apps/interface/src/styles/** are missing from style-topology.mjs.',
        { files: stylesTreeMissingFromTopology },
      ),
    )
  }

  const topologyFilesMissingOnDisk = topologyStylesRootFiles.filter((file) => !styleTreeCssFileSet.has(file))
  if (topologyFilesMissingOnDisk.length > 0) {
    findings.push(
      buildFinding(
        'topology-files-missing-on-disk',
        'style-topology.mjs references CSS files that are missing under apps/interface/src/styles/**.',
        { files: topologyFilesMissingOnDisk },
      ),
    )
  }

  if (zoneIdentifierFiles.length > 0) {
    findings.push(
      buildFinding(
        'styles-tree-sidecar-drift',
        'apps/interface/src/styles/** contains forbidden Zone.Identifier sidecar files.',
        { files: zoneIdentifierFiles },
      ),
    )
  }

  return {
    bootstrapCssImports,
    runtimeCssImports,
    expectedBootstrapImport,
    expectedRuntimeImports,
    referenceOnlyImports,
    styleTreeCssFiles,
    topologyStylesRootFiles,
    zoneIdentifierFiles,
    findings,
  }
}
