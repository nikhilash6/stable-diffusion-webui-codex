/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Source-side contract analysis for the frontend CSS verifier.
Scans Vue/TS source usage for class ownership, inline style escapes, typed
`:style` exceptions, and JS-side DOM style writes so CSS contracts fail loud at
call sites instead of drifting silently.

Symbols (top-level; keep in sync; no ghosts):
- `analyzeSourceContracts` (function): Collects source-side CSS contract findings from Vue and script files.
*/

import ts from 'typescript'
import { NodeTypes, parse as parseTemplate } from '@vue/compiler-dom'
import { parse as parseSfc } from 'vue/compiler-sfc'
import path from 'node:path'
import { collectFiles, getLineColumnFromOffset, normalizePath, readText } from './fs-utils.mjs'

const CLASS_TOKEN_PATTERN = /^-?[_a-zA-Z]+[_a-zA-Z0-9:-]*$/
const CLASS_LITERAL_PATTERN = /(['"`])([^'"`]+)\1/g
const HTML_CLASS_ATTRIBUTE_PATTERN = /class\s*=\s*(['"])([^'"]+)\1/g
const FRAMEWORK_UTILITY_PATTERN =
  /^(?:container|group|absolute|relative|fixed|sticky|block|inline|inline-block|inline-flex|flex|grid|hidden|sr-only|items-.+|justify-.+|content-.+|self-.+|place-.+|gap-.+|space-[xy]-.+|p[trblxy]?-.+|m[trblxy]?-.+|w-.+|h-.+|min-.+|max-.+|top-.+|right-.+|bottom-.+|left-.+|z-.+|overflow-.+|object-.+|text-.+|font-.+|tracking-.+|leading-.+|bg-.+|border(?:-.+)?|rounded(?:-.+)?|shadow(?:-.+)?|opacity-.+|transition(?:-.+)?|duration-.+|ease-.+|origin-.+|transform|scale-.+|rotate-.+|translate-.+|cursor-.+|select-.+|break-.+|whitespace-.+|truncate)$/
const RUNTIME_OWNED_CLASS_NAMES = new Set(['router-link-active', 'ProseMirror', 'light'])

function splitClassTokens(value) {
  return String(value || '')
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => CLASS_TOKEN_PATTERN.test(token))
}

function buildLocation(relativeFile, line, column, snippet) {
  return { file: relativeFile, line, column, snippet: String(snippet || '').trim() }
}

function getLineSnippet(content, lineNumber) {
  return String(content || '').split(/\r?\n/)[Math.max(0, lineNumber - 1)] ?? ''
}

function getLocationFromOffset(relativeFile, content, offset) {
  const { line, column } = getLineColumnFromOffset(content, offset)
  return buildLocation(relativeFile, line, column, getLineSnippet(content, line))
}

function createSourceFile(fileName, content) {
  const extension = path.extname(fileName).toLowerCase()
  const scriptKind = extension === '.js' || extension === '.mjs' ? ts.ScriptKind.JS : ts.ScriptKind.TS
  return ts.createSourceFile(fileName, content, ts.ScriptTarget.Latest, true, scriptKind)
}

function collectClassTokensFromExpressionText(expressionText) {
  const tokens = new Set()
  const wrappedSource = createSourceFile('__class_expr__.ts', `const __cdx_expr = (${expressionText});`)
  const statement = wrappedSource.statements[0]
  const expression =
    statement && ts.isVariableStatement(statement)
      ? statement.declarationList.declarations[0]?.initializer ?? null
      : null

  function visit(node) {
    if (!node) return

    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      for (const token of splitClassTokens(node.text)) tokens.add(token)
      return
    }

    if (ts.isTemplateExpression(node)) {
      for (const span of node.templateSpans) {
        visit(span.expression)
      }
      return
    }

    if (ts.isArrayLiteralExpression(node)) {
      node.elements.forEach(visit)
      return
    }

    if (ts.isObjectLiteralExpression(node)) {
      node.properties.forEach((property) => {
        if (ts.isPropertyAssignment(property) || ts.isShorthandPropertyAssignment(property)) {
          const propertyName = property.name && ts.isIdentifier(property.name)
            ? property.name.text
            : property.name && ts.isStringLiteral(property.name)
              ? property.name.text
              : null
          if (propertyName && CLASS_TOKEN_PATTERN.test(propertyName)) tokens.add(propertyName)
        }
        if (ts.isPropertyAssignment(property)) visit(property.initializer)
        if (ts.isSpreadAssignment(property)) visit(property.expression)
      })
      return
    }

    if (ts.isConditionalExpression(node)) {
      visit(node.whenTrue)
      visit(node.whenFalse)
      return
    }

    if (ts.isBinaryExpression(node)) {
      visit(node.left)
      visit(node.right)
      return
    }

    if (ts.isParenthesizedExpression(node) || ts.isPrefixUnaryExpression(node) || ts.isAsExpression(node)) {
      visit(node.expression)
      return
    }

    if (ts.isCallExpression(node)) {
      node.arguments.forEach(visit)
      return
    }
  }

  visit(expression)

  let match = CLASS_LITERAL_PATTERN.exec(expressionText)
  while (match) {
    for (const token of splitClassTokens(match[2])) tokens.add(token)
    match = CLASS_LITERAL_PATTERN.exec(expressionText)
  }
  CLASS_LITERAL_PATTERN.lastIndex = 0

  return Array.from(tokens).sort((left, right) => left.localeCompare(right))
}

function isFrameworkOwnedClass(token) {
  if (!token) return false
  if (token.includes(':')) return true
  return FRAMEWORK_UTILITY_PATTERN.test(token)
}

function findVueSections(absoluteFile, content) {
  if (path.extname(absoluteFile).toLowerCase() !== '.vue') {
    return {
      templateContent: null,
      templateOffset: 0,
      styleBlocks: [],
      scriptBlocks: [{ fileName: absoluteFile, content, offset: 0 }],
    }
  }

  const { descriptor } = parseSfc(content, { filename: absoluteFile })
  return {
    templateContent: descriptor.template?.content ?? null,
    templateOffset: descriptor.template?.loc.start.offset ?? 0,
    styleBlocks: descriptor.styles.map((block) => ({
      scoped: Boolean(block.scoped),
      offset: block.loc.start.offset,
      content: block.content,
    })),
    scriptBlocks: [descriptor.script, descriptor.scriptSetup]
      .filter(Boolean)
      .map((block) => ({
        fileName: absoluteFile,
        content: block.content,
        offset: block.loc.start.offset,
      })),
  }
}

function buildTypedExceptionSet(entries) {
  return new Set((entries || []).map((entry) => `${normalizePath(entry.file)}::${String(entry.target || '').trim()}`))
}

function createTypedExceptionState(entries) {
  return {
    allowedSet: buildTypedExceptionSet(entries),
    matchedKeys: new Set(),
    entries: (entries || []).map((entry) => ({
      file: normalizePath(entry.file),
      target: String(entry.target || '').trim(),
      rationale: String(entry.rationale || '').trim(),
    })),
  }
}

function markTypedExceptionUsage(state, relativeFile, target) {
  const key = `${relativeFile}::${String(target || '').trim()}`
  const owned = state.allowedSet.has(key)
  if (owned) state.matchedKeys.add(key)
  return owned
}

function collectUnmatchedTypedExceptions(state) {
  return state.entries.filter((entry) => !state.matchedKeys.has(`${entry.file}::${entry.target}`))
}

function getLocationFromBlockOffset(relativeFile, fullContent, blockOffset, localOffset) {
  return getLocationFromOffset(relativeFile, fullContent, blockOffset + localOffset)
}

function buildTemplatePropLocation(relativeFile, fullContent, templateOffset, prop) {
  return getLocationFromBlockOffset(relativeFile, fullContent, templateOffset, prop.loc.start.offset)
}

function isClassLikeAttributeName(name) {
  const normalized = String(name || '').trim().toLowerCase()
  return normalized === 'class' || normalized.endsWith('class')
}

export async function analyzeSourceContracts(config, cssAnalysis) {
  const sourceFiles = await collectFiles(config.absolutePaths.sourceRoot, config.paths.sourceExtensions)
  const definedClassNames = new Set(cssAnalysis.definedClassNames || [])
  const usedClasses = new Set()
  const inlineStyleAttributes = []
  const boundStyleBindings = []
  const domStyleWrites = []
  const cssVariableWrites = []
  const styleBlocks = []
  const scopedStyleBlocks = []

  const boundStyleBindingState = createTypedExceptionState(config.contracts.typedExceptions.boundStyleBindings)
  const domStyleWriteState = createTypedExceptionState(config.contracts.typedExceptions.domStyleWrites)
  const cssVariableWriteState = createTypedExceptionState(config.contracts.typedExceptions.cssVariableWrites)

  for (const absoluteFile of sourceFiles) {
    const relativeFile = normalizePath(path.relative(config.repoRoot, absoluteFile))
    const content = await readText(absoluteFile)
    const sections = findVueSections(absoluteFile, content)

    let htmlClassMatch = HTML_CLASS_ATTRIBUTE_PATTERN.exec(content)
    while (htmlClassMatch) {
      for (const token of splitClassTokens(htmlClassMatch[2])) usedClasses.add(token)
      htmlClassMatch = HTML_CLASS_ATTRIBUTE_PATTERN.exec(content)
    }
    HTML_CLASS_ATTRIBUTE_PATTERN.lastIndex = 0

    for (const styleBlock of sections.styleBlocks) {
      const location = getLocationFromOffset(relativeFile, content, styleBlock.offset)
      styleBlocks.push(location)
      if (styleBlock.scoped) scopedStyleBlocks.push(location)
    }

    if (sections.templateContent) {
      const templateAst = parseTemplate(sections.templateContent)
      const walk = (node) => {
        if (!node) return

        if (node.type === NodeTypes.ELEMENT) {
          for (const prop of node.props || []) {
            if (prop.type === NodeTypes.ATTRIBUTE) {
              if (isClassLikeAttributeName(prop.name) && prop.value?.content) {
                for (const token of splitClassTokens(prop.value.content)) usedClasses.add(token)
              }
              if (prop.name === 'style' && prop.value?.content) {
                inlineStyleAttributes.push({
                  ...buildTemplatePropLocation(relativeFile, content, sections.templateOffset, prop),
                  value: prop.value.content,
                })
              }
            }

            if (prop.type === NodeTypes.DIRECTIVE && prop.name === 'bind' && prop.arg?.type === NodeTypes.SIMPLE_EXPRESSION) {
              const argument = String(prop.arg.content || '').trim()
              const expression = String(prop.exp?.content || '').trim()
              if (isClassLikeAttributeName(argument) && expression) {
                for (const token of collectClassTokensFromExpressionText(expression)) usedClasses.add(token)
              }
              if (argument === 'style' && expression) {
                boundStyleBindings.push({
                  ...buildTemplatePropLocation(relativeFile, content, sections.templateOffset, prop),
                  expression,
                  owned: markTypedExceptionUsage(boundStyleBindingState, relativeFile, expression),
                })
              }
            }
          }
        }

        for (const child of node.children || []) walk(child)
        if (node.branches) {
          for (const branch of node.branches) walk(branch)
        }
      }
      walk(templateAst)
    }

    for (const block of sections.scriptBlocks) {
      const sourceFile = createSourceFile(absoluteFile, block.content)
      const visit = (node) => {
        if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression)) {
          const propertyName = node.expression.name.text
          const location = getLocationFromBlockOffset(relativeFile, content, block.offset, node.getStart(sourceFile))

          if (
            ['add', 'remove', 'toggle', 'replace', 'contains'].includes(propertyName) &&
            ts.isPropertyAccessExpression(node.expression.expression) &&
            node.expression.expression.name.text === 'classList'
          ) {
            for (const argument of node.arguments) {
              if (!ts.isStringLiteral(argument) && !ts.isNoSubstitutionTemplateLiteral(argument)) continue
              for (const token of splitClassTokens(argument.text)) usedClasses.add(token)
            }
          }

          if (propertyName === 'setAttribute' && node.arguments.length >= 2) {
            const [nameArgument, valueArgument] = node.arguments
            if (
              (ts.isStringLiteral(nameArgument) || ts.isNoSubstitutionTemplateLiteral(nameArgument)) &&
              nameArgument.text === 'class' &&
              (ts.isStringLiteral(valueArgument) || ts.isNoSubstitutionTemplateLiteral(valueArgument))
            ) {
              for (const token of splitClassTokens(valueArgument.text)) usedClasses.add(token)
            }
          }

          if (
            propertyName === 'setProperty' &&
            ts.isPropertyAccessExpression(node.expression.expression) &&
            node.expression.expression.name.text === 'style' &&
            node.arguments.length >= 1 &&
            (ts.isStringLiteral(node.arguments[0]) || ts.isNoSubstitutionTemplateLiteral(node.arguments[0]))
          ) {
            const target = node.arguments[0].text.trim()
            const bucket = target.startsWith('--') ? cssVariableWrites : domStyleWrites
            const exceptionState = target.startsWith('--') ? cssVariableWriteState : domStyleWriteState
            bucket.push({
              ...location,
              target,
              owned: markTypedExceptionUsage(exceptionState, relativeFile, target),
            })
          }
        }

        if (
          ts.isBinaryExpression(node) &&
          node.operatorToken.kind === ts.SyntaxKind.EqualsToken &&
          ts.isPropertyAccessExpression(node.left)
        ) {
          const location = getLocationFromBlockOffset(relativeFile, content, block.offset, node.getStart(sourceFile))
          if (node.left.name.text === 'className' && (ts.isStringLiteral(node.right) || ts.isNoSubstitutionTemplateLiteral(node.right))) {
            for (const token of splitClassTokens(node.right.text)) usedClasses.add(token)
          }
          if (
            ts.isPropertyAccessExpression(node.left.expression) &&
            node.left.expression.name.text === 'style'
          ) {
            const target = node.left.name.text
            const bucket = target.startsWith('--') ? cssVariableWrites : domStyleWrites
            const exceptionState = target.startsWith('--') ? cssVariableWriteState : domStyleWriteState
            bucket.push({
              ...location,
              target,
              owned: markTypedExceptionUsage(exceptionState, relativeFile, target),
            })
          }
        }

        if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
          for (const token of splitClassTokens(node.text)) {
            if (definedClassNames.has(token)) usedClasses.add(token)
          }
        }

        ts.forEachChild(node, visit)
      }

      visit(sourceFile)
    }
  }

  const usedClassNames = Array.from(usedClasses).sort((left, right) => left.localeCompare(right))
  const unusedClasses = Array.from(definedClassNames)
    .filter((className) => !RUNTIME_OWNED_CLASS_NAMES.has(className))
    .filter((className) => !usedClasses.has(className))
    .sort((left, right) => left.localeCompare(right))
    .map((className) => ({ className }))
  const missingClasses = usedClassNames
    .filter((className) => !definedClassNames.has(className))
    .filter((className) => !isFrameworkOwnedClass(className))
    .filter((className) => className.startsWith('cdx-') || className.includes('-'))
    .sort((left, right) => left.localeCompare(right))
    .map((className) => ({ className }))
  const unmatchedBoundStyleBindings = collectUnmatchedTypedExceptions(boundStyleBindingState)
  const unmatchedDomStyleWrites = collectUnmatchedTypedExceptions(domStyleWriteState)
  const unmatchedCssVariableWrites = collectUnmatchedTypedExceptions(cssVariableWriteState)

  return {
    sourceFiles: sourceFiles.map((file) => normalizePath(path.relative(config.repoRoot, file))),
    usedClassNames,
    unusedClasses,
    missingClasses,
    inlineStyleAttributes,
    boundStyleBindings,
    unownedBoundStyleBindings: boundStyleBindings.filter((entry) => !entry.owned),
    unmatchedBoundStyleBindings,
    domStyleWrites,
    unownedDomStyleWrites: domStyleWrites.filter((entry) => !entry.owned),
    unmatchedDomStyleWrites,
    cssVariableWrites,
    unownedCssVariableWrites: cssVariableWrites.filter((entry) => !entry.owned),
    unmatchedCssVariableWrites,
    styleBlocks,
    scopedStyleBlocks,
  }
}
