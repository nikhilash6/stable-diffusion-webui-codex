/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: PNG Info infotext parsing + send-to mapping helpers.
Parses common A1111/Forge-style `parameters` infotext into structured fields (including legacy JSON fallback) and maps sampler/scheduler
strings into Codex canonical names with truthful partial-mapping warnings when only one side is recognized.

Symbols (top-level; keep in sync; no ghosts):
- `ParsedInfotext` (interface): Structured subset of parsed infotext fields.
- `ModelLike` (interface): Minimal model entry used to resolve checkpoints from infotext.
- `ComfyParseResult` (interface): Parsed ComfyUI prompt JSON result (extracted fields + graph + warnings).
- `SamplerLike` (interface): Minimal sampler entry used for name + scheduler-compatibility mapping.
- `SchedulerLike` (interface): Minimal scheduler entry used for name matching.
- `parseInfotext` (function): Parses infotext into structured fields + raw kv map.
- `tokenizeKvParts` (function): Tokenizes `Steps:` KV blocks while preserving quoted/bracketed comma values.
- `parseLegacyJsonInfotext` (function): Parses legacy JSON metadata blobs into `ParsedInfotext`.
- `parseComfyPromptJson` (function): Parses ComfyUI prompt JSON and extracts common generation fields when unambiguous.
- `mapCheckpointTitle` (function): Resolves a checkpoint title from parsed infotext against a models list.
- `mapSamplerScheduler` (function): Maps raw sampler/scheduler strings to canonical names with sampler/scheduler compatibility validation.
*/

export interface ParsedInfotext {
  prompt: string
  negativePrompt: string
  hasNegativePrompt: boolean
  model?: string
  modelHash?: string
  vae?: string
  steps?: number
  sampler?: string
  scheduler?: string
  cfgScale?: number
  seed?: number
  width?: number
  height?: number
  clipSkip?: number
  denoiseStrength?: number
  rng?: string
  eta?: number
  ngms?: number
  version?: string
  hiresModule1?: string
  rawKv: Record<string, string>
}

export interface ModelLike {
  title: string
  hash?: string | null
  filename?: string
  model_name?: string
  name?: string
}

export interface SamplerLike {
  name: string
  allowed_schedulers?: string[]
}

export interface SchedulerLike {
  name: string
}

export interface ComfyParseResult {
  graph: Record<string, unknown> | null
  extracted: Partial<ParsedInfotext>
  warnings: string[]
}

function normalizeComparable(value: string): string {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[_]+/g, ' ')
    .replace(/\s+/g, ' ')
}

function parseIntStrict(raw: string): number | null {
  const text = String(raw || '').trim()
  if (!text) return null
  if (!/^-?\d+$/.test(text)) return null
  const n = Number(text)
  return Number.isFinite(n) ? n : null
}

function parseFloatStrict(raw: string): number | null {
  const text = String(raw || '').trim()
  if (!text) return null
  if (!/^-?\d+(\.\d+)?$/.test(text)) return null
  const n = Number(text)
  return Number.isFinite(n) ? n : null
}

function parseSize(raw: string): { width: number; height: number } | null {
  const text = String(raw || '').trim()
  if (!text) return null
  const m = text.match(/^(\d+)\s*[x×]\s*(\d+)$/i)
  if (!m) return null
  const w = parseIntStrict(m[1])
  const h = parseIntStrict(m[2])
  if (w === null || h === null) return null
  if (w <= 0 || h <= 0) return null
  return { width: w, height: h }
}

function parseHex(raw: string): string | null {
  const text = String(raw || '').trim().toLowerCase()
  if (!text) return null
  if (!/^[0-9a-f]+$/.test(text)) return null
  return text
}

function tryParseJson(raw: string): unknown | null {
  const text = String(raw || '').trim()
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

function tokenizeKvParts(input: string): string[] {
  const parts: string[] = []
  let current = ''
  let inSingleQuote = false
  let inDoubleQuote = false
  let escapeNext = false
  let roundDepth = 0
  let squareDepth = 0
  let curlyDepth = 0

  const flush = (): void => {
    const token = current.trim()
    if (token) parts.push(token)
    current = ''
  }

  for (const character of String(input || '')) {
    if (escapeNext) {
      current += character
      escapeNext = false
      continue
    }
    if ((inSingleQuote || inDoubleQuote) && character === '\\') {
      current += character
      escapeNext = true
      continue
    }
    if (!inSingleQuote && character === '"') {
      inDoubleQuote = !inDoubleQuote
      current += character
      continue
    }
    if (!inDoubleQuote && character === '\'') {
      inSingleQuote = !inSingleQuote
      current += character
      continue
    }
    if (!inSingleQuote && !inDoubleQuote) {
      if (character === '(') roundDepth += 1
      else if (character === ')') roundDepth = Math.max(0, roundDepth - 1)
      else if (character === '[') squareDepth += 1
      else if (character === ']') squareDepth = Math.max(0, squareDepth - 1)
      else if (character === '{') curlyDepth += 1
      else if (character === '}') curlyDepth = Math.max(0, curlyDepth - 1)

      const topLevel = roundDepth === 0 && squareDepth === 0 && curlyDepth === 0
      if (topLevel && (character === ',' || character === '\n' || character === '\r')) {
        flush()
        continue
      }
    }
    current += character
  }
  flush()
  return parts
}

function parseRawKvBlock(kvBlock: string): Record<string, string> {
  const rawKv: Record<string, string> = {}
  for (const part of tokenizeKvParts(kvBlock)) {
    const idx = part.indexOf(':')
    if (idx === -1) continue
    const key = part.slice(0, idx).trim()
    const value = part.slice(idx + 1).trim()
    if (!key) continue
    rawKv[key] = value
  }
  return rawKv
}

function parseTextInfotext(rawText: string): { parsed: ParsedInfotext; warnings: string[] } {
  const warnings: string[] = []
  const normalized = String(rawText || '').replace(/\r\n/g, '\n').trim()
  if (!normalized) {
    return {
      parsed: { prompt: '', negativePrompt: '', hasNegativePrompt: false, rawKv: {} },
      warnings: [],
    }
  }

  const lines = normalized.split('\n')
  const stepsLineIndex = lines.findIndex((line) => /^\s*steps\s*:/i.test(line))
  if (stepsLineIndex === -1) {
    warnings.push("Infotext: couldn't find 'Steps:' block; treating content as prompt only.")
    return {
      parsed: { prompt: normalized, negativePrompt: '', hasNegativePrompt: false, rawKv: {} },
      warnings,
    }
  }

  const head = lines.slice(0, stepsLineIndex).join('\n').trimEnd()
  const kvBlock = lines.slice(stepsLineIndex).join('\n').trim()

  const negMarker = 'negative prompt:'
  const negIdx = head.toLowerCase().indexOf(negMarker)
  const hasNegativePrompt = negIdx !== -1
  const prompt = (hasNegativePrompt ? head.slice(0, negIdx) : head).trimEnd()
  const negativePrompt = hasNegativePrompt ? head.slice(negIdx + negMarker.length).trim() : ''

  const rawKv = parseRawKvBlock(kvBlock)
  const parsed: ParsedInfotext = {
    prompt,
    negativePrompt,
    hasNegativePrompt,
    rawKv,
  }

  const get = (key: string): string | undefined => {
    const exact = rawKv[key]
    if (exact !== undefined) return exact
    const low = key.toLowerCase()
    for (const [k, v] of Object.entries(rawKv)) {
      if (k.toLowerCase() === low) return v
    }
    return undefined
  }

  const steps = get('Steps')
  if (steps !== undefined) {
    const n = parseIntStrict(steps)
    if (n === null || n < 0) warnings.push(`Invalid Steps value: ${steps}`)
    else parsed.steps = n
  }

  const sampler = get('Sampler')
  if (sampler !== undefined && sampler.trim()) parsed.sampler = sampler.trim()

  const scheduler = get('Schedule type') ?? get('Scheduler')
  if (scheduler !== undefined && scheduler.trim()) parsed.scheduler = scheduler.trim()

  const cfg = get('CFG scale') ?? get('CFG')
  if (cfg !== undefined) {
    const n = parseFloatStrict(cfg)
    if (n === null) warnings.push(`Invalid CFG scale value: ${cfg}`)
    else parsed.cfgScale = n
  }

  const seed = get('Seed')
  if (seed !== undefined) {
    const n = parseIntStrict(seed)
    if (n === null) warnings.push(`Invalid Seed value: ${seed}`)
    else parsed.seed = n
  }

  const size = get('Size')
  if (size !== undefined) {
    const dims = parseSize(size)
    if (!dims) warnings.push(`Invalid Size value: ${size}`)
    else {
      parsed.width = dims.width
      parsed.height = dims.height
    }
  }

  const clipSkip = get('Clip skip')
  if (clipSkip !== undefined) {
    const n = parseIntStrict(clipSkip)
    if (n === null || n < 0) warnings.push(`Invalid Clip skip value: ${clipSkip}`)
    else parsed.clipSkip = n
  }

  const denoise = get('Denoising strength') ?? get('Denoising Strength')
  if (denoise !== undefined) {
    const n = parseFloatStrict(denoise)
    if (n === null || n < 0 || n > 1) warnings.push(`Invalid Denoising strength value: ${denoise}`)
    else parsed.denoiseStrength = n
  }

  const model = get('Model')
  if (model !== undefined && model.trim()) parsed.model = model.trim()
  const modelHash = get('Model hash') ?? get('Model Hash')
  if (modelHash !== undefined && modelHash.trim()) parsed.modelHash = modelHash.trim()
  const vae = get('VAE')
  if (vae !== undefined && vae.trim()) parsed.vae = vae.trim()

  const rng = get('RNG')
  if (rng !== undefined && rng.trim()) parsed.rng = rng.trim()

  const eta = get('Eta')
  if (eta !== undefined) {
    const n = parseFloatStrict(eta)
    if (n === null) warnings.push(`Invalid Eta value: ${eta}`)
    else parsed.eta = n
  }

  const ngms = get('NGMS')
  if (ngms !== undefined) {
    const n = parseFloatStrict(ngms)
    if (n === null) warnings.push(`Invalid NGMS value: ${ngms}`)
    else parsed.ngms = n
  }

  const version = get('Version')
  if (version !== undefined && version.trim()) parsed.version = version.trim()

  const hiresModule1 = get('Hires Module 1') ?? get('Module 1')
  if (hiresModule1 !== undefined && hiresModule1.trim()) parsed.hiresModule1 = hiresModule1.trim()

  return { parsed, warnings }
}

function parseLegacyJsonInfotext(rawText: string): { parsed: ParsedInfotext; warnings: string[] } | null {
  const jsonValue = tryParseJson(rawText)
  if (!jsonValue || typeof jsonValue !== 'object' || Array.isArray(jsonValue)) return null

  const obj = jsonValue as Record<string, unknown>
  const parametersField = obj.parameters
  if (typeof parametersField === 'string' && parametersField.trim()) {
    const normalizedParameters = parametersField
      .replace(/\\r\\n/g, '\n')
      .replace(/\\n/g, '\n')
      .replace(/\\r/g, '\n')
    const out = parseTextInfotext(normalizedParameters)
    out.warnings.unshift("Infotext: parsed from JSON metadata field 'parameters'.")
    return out
  }

  const warnings: string[] = ["Infotext: parsed from legacy JSON metadata blob."]
  const parsed: ParsedInfotext = {
    prompt: '',
    negativePrompt: '',
    hasNegativePrompt: false,
    rawKv: {},
  }

  const pickString = (...keys: string[]): string => {
    for (const key of keys) {
      const value = obj[key]
      if (typeof value === 'string' && value.trim()) return value.trim()
    }
    return ''
  }

  const pickInt = (...keys: string[]): number | undefined => {
    for (const key of keys) {
      const value = obj[key]
      const parsedValue = parseIntStrict(String(value ?? ''))
      if (parsedValue !== null) return parsedValue
    }
    return undefined
  }

  const pickFloat = (...keys: string[]): number | undefined => {
    for (const key of keys) {
      const value = obj[key]
      const parsedValue = parseFloatStrict(String(value ?? ''))
      if (parsedValue !== null) return parsedValue
    }
    return undefined
  }

  for (const [key, value] of Object.entries(obj)) {
    if (value === null || value === undefined) continue
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      parsed.rawKv[key] = String(value)
    }
  }

  const prompt = pickString('prompt')
  if (prompt) parsed.prompt = prompt

  const negativePrompt = pickString('negative_prompt', 'negativePrompt')
  if (negativePrompt) {
    parsed.negativePrompt = negativePrompt
    parsed.hasNegativePrompt = true
  }

  const steps = pickInt('steps')
  if (steps !== undefined && steps >= 0) parsed.steps = steps

  const sampler = pickString('sampler_name', 'sampler')
  if (sampler) parsed.sampler = sampler

  const scheduler = pickString('schedule_type', 'scheduler')
  if (scheduler) parsed.scheduler = scheduler

  const cfgScale = pickFloat('cfg_scale', 'guidance_scale', 'cfg')
  if (cfgScale !== undefined) parsed.cfgScale = cfgScale

  const seed = pickInt('seed')
  if (seed !== undefined) parsed.seed = seed

  const width = pickInt('width')
  const height = pickInt('height')
  if (width !== undefined && width > 0) parsed.width = width
  if (height !== undefined && height > 0) parsed.height = height
  if ((parsed.width === undefined || parsed.height === undefined)) {
    const sizeRaw = pickString('size')
    if (sizeRaw) {
      const dims = parseSize(sizeRaw)
      if (dims) {
        parsed.width = dims.width
        parsed.height = dims.height
      }
    }
  }

  const clipSkip = pickInt('clip_skip', 'clipSkip')
  if (clipSkip !== undefined && clipSkip >= 0) parsed.clipSkip = clipSkip

  const denoiseStrength = pickFloat('denoising_strength', 'denoise_strength', 'denoiseStrength')
  if (denoiseStrength !== undefined && denoiseStrength >= 0 && denoiseStrength <= 1) {
    parsed.denoiseStrength = denoiseStrength
  }

  const model = pickString('model', 'sd_model_checkpoint')
  if (model) parsed.model = model

  const modelHash = pickString('model_hash')
  if (modelHash) parsed.modelHash = modelHash

  const vae = pickString('vae', 'vae_name')
  if (vae) parsed.vae = vae

  const rng = pickString('rng', 'rng_source')
  if (rng) parsed.rng = rng

  const eta = pickFloat('eta')
  if (eta !== undefined) parsed.eta = eta

  const ngms = pickFloat('ngms')
  if (ngms !== undefined) parsed.ngms = ngms

  const version = pickString('version')
  if (version) parsed.version = version

  const hiresModule1 = pickString('hires_module_1')
  if (hiresModule1) parsed.hiresModule1 = hiresModule1

  return { parsed, warnings }
}

export function parseInfotext(infotext: string): { parsed: ParsedInfotext; warnings: string[] } {
  const raw = String(infotext || '').trim()
  if (!raw) {
    return {
      parsed: { prompt: '', negativePrompt: '', hasNegativePrompt: false, rawKv: {} },
      warnings: [],
    }
  }

  const fromLegacyJson = parseLegacyJsonInfotext(raw)
  if (fromLegacyJson) return fromLegacyJson

  return parseTextInfotext(raw)
}

function comfyNodeInputs(node: unknown): Record<string, unknown> | null {
  if (!node || typeof node !== 'object') return null
  const inputs = (node as any).inputs
  if (!inputs || typeof inputs !== 'object' || Array.isArray(inputs)) return null
  return inputs as Record<string, unknown>
}

function comfyNodeType(node: unknown): string {
  const raw = (node as any)?.class_type
  return typeof raw === 'string' ? raw : ''
}

function comfyLinkNodeId(value: unknown): string | null {
  if (!Array.isArray(value) || value.length < 1) return null
  const nodeId = value[0]
  if (typeof nodeId === 'string' && nodeId.trim()) return nodeId.trim()
  if (typeof nodeId === 'number' && Number.isFinite(nodeId)) return String(nodeId)
  return null
}

export function parseComfyPromptJson(rawJson: string): ComfyParseResult {
  const parsed = tryParseJson(rawJson)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { graph: null, extracted: {}, warnings: ["ComfyUI: 'prompt' JSON is missing or invalid."] }
  }

  const graph = parsed as Record<string, unknown>
  const nodes = Object.entries(graph)
    .map(([id, node]) => ({ id, node }))
    .filter(({ id, node }) => typeof id === 'string' && id.trim() && node && typeof node === 'object')

  const samplers = nodes.filter(({ node }) => {
    const t = comfyNodeType(node).trim()
    return t === 'KSampler' || t === 'KSamplerAdvanced'
  })
  if (samplers.length === 0) {
    return { graph, extracted: {}, warnings: ["ComfyUI: no 'KSampler' node found; cannot extract fields."] }
  }
  if (samplers.length !== 1) {
    return { graph, extracted: {}, warnings: [`ComfyUI: multiple KSampler nodes found (${samplers.length}); not extracting fields.`] }
  }

  const samplerNode = samplers[0].node
  const inputs = comfyNodeInputs(samplerNode)
  if (!inputs) {
    return { graph, extracted: {}, warnings: ["ComfyUI: KSampler node has no inputs; cannot extract fields."] }
  }

  const extracted: Partial<ParsedInfotext> = {}
  const warnings: string[] = []

  const seed = inputs.seed
  if (typeof seed === 'number' && Number.isFinite(seed)) extracted.seed = Math.trunc(seed)
  const steps = inputs.steps
  if (typeof steps === 'number' && Number.isFinite(steps)) extracted.steps = Math.trunc(steps)
  const cfg = inputs.cfg
  if (typeof cfg === 'number' && Number.isFinite(cfg)) extracted.cfgScale = cfg
  const samplerName = inputs.sampler_name ?? inputs.sampler
  if (typeof samplerName === 'string' && samplerName.trim()) extracted.sampler = samplerName.trim()
  const scheduler = inputs.scheduler
  if (typeof scheduler === 'string' && scheduler.trim()) extracted.scheduler = scheduler.trim()
  const denoise = inputs.denoise
  if (typeof denoise === 'number' && Number.isFinite(denoise)) extracted.denoiseStrength = denoise

  const resolveText = (nodeIdValue: unknown): string | null => {
    const nodeId = comfyLinkNodeId(nodeIdValue)
    if (!nodeId) return null
    const target = graph[nodeId]
    if (!target || typeof target !== 'object') return null
    if (comfyNodeType(target) !== 'CLIPTextEncode') return null
    const i = comfyNodeInputs(target)
    const text = i?.text
    return typeof text === 'string' ? text : null
  }

  const positiveText = resolveText(inputs.positive)
  if (positiveText && positiveText.trim()) extracted.prompt = positiveText
  const negativeText = resolveText(inputs.negative)
  if (negativeText && negativeText.trim()) {
    extracted.negativePrompt = negativeText
    extracted.hasNegativePrompt = true
  }

  const latentNodeId = comfyLinkNodeId(inputs.latent_image)
  if (latentNodeId) {
    const latentNode = graph[latentNodeId]
    if (latentNode && typeof latentNode === 'object' && comfyNodeType(latentNode) === 'EmptyLatentImage') {
      const li = comfyNodeInputs(latentNode)
      const w = li?.width
      const h = li?.height
      if (typeof w === 'number' && Number.isFinite(w) && typeof h === 'number' && Number.isFinite(h)) {
        extracted.width = Math.trunc(w)
        extracted.height = Math.trunc(h)
      }
    }
  }

  if (!extracted.prompt?.trim()) {
    warnings.push("ComfyUI: couldn't resolve a single positive prompt; prompt will stay unchanged.")
  }

  return { graph, extracted, warnings }
}

export function mapCheckpointTitle(
  parsed: Pick<ParsedInfotext, 'model' | 'modelHash'>,
  models: ModelLike[],
): { checkpoint?: string; warnings: string[] } {
  const warnings: string[] = []
  const list = Array.isArray(models) ? models : []
  if (list.length === 0) return { warnings: [] }

  const hashRaw = parsed.modelHash ? parseHex(parsed.modelHash) : null
  if (hashRaw) {
    const matches = list.filter(m => {
      const h = typeof m.hash === 'string' ? m.hash.toLowerCase() : ''
      if (!h) return false
      return h === hashRaw || h.startsWith(hashRaw) || hashRaw.startsWith(h)
    })
    if (matches.length === 1) return { checkpoint: matches[0].title, warnings }
    if (matches.length > 1) {
      warnings.push(`Model hash '${parsed.modelHash}' is ambiguous (${matches.length} matches); leaving checkpoint unchanged.`)
      return { warnings }
    }
    warnings.push(`Model hash '${parsed.modelHash}' not found; leaving checkpoint unchanged.`)
  }

  const modelRaw = String(parsed.model || '').trim()
  if (!modelRaw) return { warnings }

  const needle = normalizeComparable(modelRaw.replace(/\\+/g, '/'))
  const candidates = list.filter(m => {
    const title = normalizeComparable(String(m.title || ''))
    const name = normalizeComparable(String(m.name || ''))
    const modelName = normalizeComparable(String(m.model_name || ''))
    const filename = normalizeComparable(String(m.filename || '').replace(/\\+/g, '/'))
    const tail = filename ? filename.split('/').pop() || '' : ''
    return title === needle || name === needle || modelName === needle || filename === needle || tail === needle
  })

  if (candidates.length === 1) return { checkpoint: candidates[0].title, warnings }
  if (candidates.length > 1) {
    warnings.push(`Model '${modelRaw}' is ambiguous (${candidates.length} matches); leaving checkpoint unchanged.`)
    return { warnings }
  }

  warnings.push(`Model '${modelRaw}' not recognized; leaving checkpoint unchanged.`)
  return { warnings }
}

export function mapSamplerScheduler(
  rawSampler: string | undefined,
  rawScheduler: string | undefined,
  samplers: SamplerLike[],
  schedulers: SchedulerLike[],
): { sampler?: string; scheduler?: string; warnings: string[] } {
  const warnings: string[] = []

  const samplerMap = new Map<string, string>()
  for (const s of samplers) {
    if (!s?.name) continue
    samplerMap.set(normalizeComparable(s.name), s.name)
  }

  const schedulerMap = new Map<string, string>()
  const schedulerLabels: string[] = []
  for (const sch of schedulers) {
    if (!sch?.name) continue
    const canonical = sch.name
    const key = normalizeComparable(canonical)
    schedulerMap.set(key, canonical)
    const label = normalizeComparable(canonical.replace(/_/g, ' '))
    schedulerMap.set(label, canonical)
    schedulerLabels.push(label)
  }
  schedulerLabels.sort((a, b) => b.length - a.length)

  const resolveSampler = (value: string): string | null => {
    let key = normalizeComparable(value)
    if (key === 'euler ancestral') key = 'euler a'
    if (key.startsWith('dpmpp ')) key = `dpm++ ${key.slice('dpmpp '.length)}`
    const found = samplerMap.get(key)
    return found ?? null
  }

  const resolveScheduler = (value: string): string | null => {
    const key = normalizeComparable(value)
    const found = schedulerMap.get(key)
    return found ?? null
  }

  let sampler = rawSampler ? resolveSampler(rawSampler) : null
  let scheduler = rawScheduler ? resolveScheduler(rawScheduler) : null

  // Heuristic: when schedule type isn't present, it may be appended to sampler name
  // e.g. "DPM++ 2M Karras" → sampler="dpm++ 2m", scheduler="karras".
  if (scheduler === null && rawScheduler && normalizeComparable(rawScheduler) === 'automatic') {
    warnings.push("Scheduler is 'Automatic' (not reproducible); leaving scheduler unchanged.")
  }

  if (scheduler === null && rawScheduler && normalizeComparable(rawScheduler) !== 'automatic') {
    warnings.push(`Scheduler '${rawScheduler}' not recognized; leaving scheduler unchanged.`)
  }

  if ((sampler === null || scheduler === null) && rawSampler) {
    const rawKey = normalizeComparable(rawSampler)
    for (const label of schedulerLabels) {
      const suffix = ` ${label}`
      if (!rawKey.endsWith(suffix)) continue
      const samplerPart = rawKey.slice(0, -suffix.length).trim()
      const inferredSampler = samplerMap.get(samplerPart) ?? null
      const inferredScheduler = schedulerMap.get(label) ?? null
      if (inferredSampler && inferredScheduler) {
        sampler = inferredSampler
        scheduler = inferredScheduler
      }
      break
    }
  }

  if (sampler === null && rawSampler) {
    warnings.push(`Sampler '${rawSampler}' not recognized; leaving sampler unchanged.`)
  }

  if (sampler && scheduler) {
    const samplerEntry = samplers.find(s => s.name === sampler) ?? null
    const allowed = samplerEntry?.allowed_schedulers ?? []
    if (Array.isArray(allowed) && allowed.length > 0 && !allowed.includes(scheduler)) {
      warnings.push(`Sampler/scheduler pair '${sampler}' / '${scheduler}' is not supported; leaving unchanged.`)
      return { warnings }
    }
  }

  const result: { sampler?: string; scheduler?: string; warnings: string[] } = { warnings }
  if (sampler) result.sampler = sampler
  if (scheduler) result.scheduler = scheduler
  return result
}
