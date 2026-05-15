/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Zod-validated payload schemas + builders for the generic LTX video lane (`/api/txt2vid` + `/api/img2vid`).
Defines the strict payload contracts used by the dedicated LTX frontend path, validating exact device/settings revision, profile-aware geometry,
and frame-count inputs against the real backend-supported domain while sending the explicit checkpoint-aware execution profile as the only public
lane owner instead of exposing a second raw sampler/scheduler authority seam on the wire.

Symbols (top-level; keep in sync; no ghosts):
- `LTX_DIM_ALIGNMENT` (const): Spatial alignment required by the active LTX generic-video contract.
- `LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT` (const): Final-output spatial alignment required by the `two_stage` LTX execution profile.
- `LTX_DIM_MIN` / `LTX_DIM_MAX` (const): Supported LTX dimension bounds.
- `LTX_FRAME_ALIGNMENT` (const): Frame-count alignment required by the active LTX generic-video contract.
- `LTX_FRAMES_MIN` / `LTX_FRAMES_MAX` (const): Supported LTX frame-count bounds.
- `LTX_ALLOWED_EXECUTION_PROFILES` (const): Truthful public execution profiles exposed by the current LTX frontend contract.
- `LtxTxt2VidPayloadSchema` (const): Zod schema for generic `/api/txt2vid` LTX requests.
- `LtxImg2VidPayloadSchema` (const): Zod schema for generic `/api/img2vid` LTX requests.
- `LtxTxt2VidPayload` (type): Zod-inferred txt2vid payload type.
- `LtxImg2VidPayload` (type): Zod-inferred img2vid payload type.
- `LtxVideoCommonInput` (interface): Shared UI-friendly input shape used by LTX payload builders.
- `LtxTxt2VidInput` (interface): Common txt2vid input.
- `LtxImg2VidInput` (interface): Img2vid input including init image data.
- `normalizeDevice` (function): Validates and normalizes a device token.
- `requireSettingsRevision` (function): Validates unknown revision input as a non-negative integer without synthesizing fallback values.
- `requireLtxDim` (function): Validates that a dimension already satisfies the strict LTX alignment contract (`32px` baseline, `64px` for `two_stage` final output).
- `requireLtxFrameCount` (function): Validates that a frame count already satisfies the strict LTX `8n+1` contract.
- `requirePositiveInt` (function): Validates an already-normalized positive integer field without dispatch-time fallback coercion.
- `requireFiniteNumber` (function): Validates an already-normalized finite numeric field without dispatch-time fallback coercion.
- `requireLtxSeed` (function): Validates the LTX seed field as an integer, preserving the visible negative random sentinel.
- `normalizeLtxExecutionProfile` (function): Enforces the currently supported LTX execution profiles.
- `resolveLtxDimAlignmentForExecutionProfile` (function): Resolves the profile-aware LTX final-output dimension alignment requirement.
- `buildLtxTxt2VidPayload` (function): Builds a validated generic txt2vid payload for engine `ltx2`.
- `buildLtxImg2VidPayload` (function): Builds a validated generic img2vid payload for engine `ltx2`.
*/

import { z } from 'zod'

const DEVICE_VALUES = ['cuda', 'cpu'] as const
const DeviceEnum = z.enum(DEVICE_VALUES)
export const LTX_DIM_ALIGNMENT = 32
export const LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT = 64
export const LTX_DIM_MIN = 32
export const LTX_DIM_MAX = 8192
export const LTX_FRAME_ALIGNMENT = 8
export const LTX_FRAMES_MIN = 9
export const LTX_FRAMES_MAX = 401

export const LTX_ALLOWED_EXECUTION_PROFILES = ['one_stage', 'two_stage', 'distilled'] as const

const LtxExecutionProfileEnum = z.enum(LTX_ALLOWED_EXECUTION_PROFILES)
type LtxExecutionProfile = z.infer<typeof LtxExecutionProfileEnum>
const Sha256Schema = z
  .string()
  .transform((value) => value.trim().toLowerCase())
  .refine((value) => /^[0-9a-f]{64}$/.test(value), { message: 'Expected sha256 (64 lowercase hex chars)' })
const PromptSchema = z
  .string()
  .transform((value) => value.trim())
  .refine((value) => value.length > 0, { message: 'Prompt must not be empty' })
const NegativePromptSchema = z.string().transform((value) => value.trim())
const LtxDimSchema = z
  .number()
  .int()
  .min(LTX_DIM_MIN)
  .max(LTX_DIM_MAX)
  .refine((value) => value % LTX_DIM_ALIGNMENT === 0, {
    message: `Expected dimension aligned to ${LTX_DIM_ALIGNMENT}px`,
  })
const LtxFrameCountSchema = z
  .number()
  .int()
  .min(LTX_FRAMES_MIN)
  .max(LTX_FRAMES_MAX)
  .refine((value) => (value - 1) % LTX_FRAME_ALIGNMENT === 0, {
    message: `Expected 8n+1 frame count in [${LTX_FRAMES_MIN}, ${LTX_FRAMES_MAX}]`,
  })

const CommonLtxVideoPayloadSchema = z.object({
  device: DeviceEnum,
  settings_revision: z.number().int().min(0),
  engine: z.literal('ltx2'),
  model: z.string().min(1),
  model_sha: Sha256Schema.optional(),
  tenc_sha: Sha256Schema,
  vae_sha: Sha256Schema.optional(),
  video_save_output: z.literal(true),
  video_save_metadata: z.literal(true),
  video_return_frames: z.boolean(),
  ltx_execution_profile: LtxExecutionProfileEnum,
}).strict()

export const LtxTxt2VidPayloadSchema = CommonLtxVideoPayloadSchema.extend({
  txt2vid_prompt: PromptSchema,
  txt2vid_neg_prompt: NegativePromptSchema.default(''),
  txt2vid_width: LtxDimSchema,
  txt2vid_height: LtxDimSchema,
  txt2vid_steps: z.number().int().min(1),
  txt2vid_fps: z.number().int().min(1).max(240),
  txt2vid_num_frames: LtxFrameCountSchema,
  txt2vid_seed: z.number().int(),
  txt2vid_cfg_scale: z.number().min(0),
}).strict().superRefine((payload, ctx) => {
  if (payload.ltx_execution_profile !== 'two_stage') return
  if (payload.txt2vid_width % LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT !== 0) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ['txt2vid_width'],
      message: `Expected two_stage final output dimension aligned to ${LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT}px`,
    })
  }
  if (payload.txt2vid_height % LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT !== 0) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ['txt2vid_height'],
      message: `Expected two_stage final output dimension aligned to ${LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT}px`,
    })
  }
})

export const LtxImg2VidPayloadSchema = CommonLtxVideoPayloadSchema.extend({
  img2vid_prompt: PromptSchema,
  img2vid_neg_prompt: NegativePromptSchema.default(''),
  img2vid_width: LtxDimSchema,
  img2vid_height: LtxDimSchema,
  img2vid_steps: z.number().int().min(1),
  img2vid_fps: z.number().int().min(1).max(240),
  img2vid_num_frames: LtxFrameCountSchema,
  img2vid_seed: z.number().int(),
  img2vid_cfg_scale: z.number().min(0),
  img2vid_init_image: z.string().min(1),
}).strict().superRefine((payload, ctx) => {
  if (payload.ltx_execution_profile !== 'two_stage') return
  if (payload.img2vid_width % LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT !== 0) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ['img2vid_width'],
      message: `Expected two_stage final output dimension aligned to ${LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT}px`,
    })
  }
  if (payload.img2vid_height % LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT !== 0) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ['img2vid_height'],
      message: `Expected two_stage final output dimension aligned to ${LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT}px`,
    })
  }
})

export type LtxTxt2VidPayload = z.infer<typeof LtxTxt2VidPayloadSchema>
export type LtxImg2VidPayload = z.infer<typeof LtxImg2VidPayloadSchema>

export interface LtxVideoCommonInput {
  device: string
  settingsRevision: unknown
  model: string
  modelSha?: string | null | undefined
  prompt: string
  negativePrompt?: string
  width: number
  height: number
  fps: number
  frames: number
  steps: number
  cfgScale: number
  executionProfile: string
  seed: number
  textEncoderSha: string
  vaeSha?: string | null | undefined
  videoReturnFrames?: boolean
}

export interface LtxTxt2VidInput extends LtxVideoCommonInput {}

export interface LtxImg2VidInput extends LtxVideoCommonInput {
  initImageData: string
}

export function normalizeDevice(device: string): LtxTxt2VidPayload['device'] {
  const normalized = device.trim().toLowerCase()
  if (DEVICE_VALUES.includes(normalized as (typeof DEVICE_VALUES)[number])) {
    return normalized as LtxTxt2VidPayload['device']
  }
  throw new Error(`Unsupported device '${device}'`)
}

export function requireSettingsRevision(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value) && Number.isInteger(value) && value >= 0) {
    return value
  }
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (/^\d+$/.test(trimmed)) return Number(trimmed)
  }
  throw new Error(`settings_revision must be a non-negative integer, got ${String(value)}.`)
}

export function requireLtxDim(rawValue: number, fieldName = 'dimension', alignment = LTX_DIM_ALIGNMENT): number {
  if (typeof rawValue !== 'number' || !Number.isFinite(rawValue)) {
    throw new Error(`${fieldName} must be a finite integer, got ${String(rawValue)}.`)
  }
  if (!Number.isInteger(rawValue)) {
    throw new Error(`${fieldName} must be an integer, got ${String(rawValue)}.`)
  }
  const value = rawValue
  if (value < LTX_DIM_MIN || value > LTX_DIM_MAX) {
    throw new Error(`${fieldName} must be within [${LTX_DIM_MIN}, ${LTX_DIM_MAX}], got ${value}.`)
  }
  if (value % alignment !== 0) {
    const profileContext = alignment === LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT
      ? ' for two_stage final output dimensions'
      : ''
    throw new Error(`${fieldName} must be divisible by ${alignment}${profileContext}, got ${value}.`)
  }
  return value
}

export function requireLtxFrameCount(rawValue: number): number {
  if (typeof rawValue !== 'number' || !Number.isFinite(rawValue)) {
    throw new Error(`frame count must be a finite integer, got ${String(rawValue)}.`)
  }
  if (!Number.isInteger(rawValue)) {
    throw new Error(`frame count must be an integer, got ${String(rawValue)}.`)
  }
  const value = rawValue
  if (value < LTX_FRAMES_MIN || value > LTX_FRAMES_MAX) {
    throw new Error(`frame count must be within [${LTX_FRAMES_MIN}, ${LTX_FRAMES_MAX}], got ${value}.`)
  }
  if ((value - 1) % LTX_FRAME_ALIGNMENT !== 0) {
    throw new Error(`frame count must satisfy 8n+1, got ${value}.`)
  }
  return value
}

export function requirePositiveInt(rawValue: number, fieldName: string, minimum = 1, maximum?: number): number {
  if (typeof rawValue !== 'number' || !Number.isFinite(rawValue)) {
    throw new Error(`${fieldName} must be a finite integer, got ${String(rawValue)}.`)
  }
  if (!Number.isInteger(rawValue)) {
    throw new Error(`${fieldName} must be an integer, got ${String(rawValue)}.`)
  }
  const value = rawValue
  if (value < minimum) {
    throw new Error(`${fieldName} must be >= ${minimum}, got ${value}.`)
  }
  if (maximum !== undefined && value > maximum) {
    throw new Error(`${fieldName} must be <= ${maximum}, got ${value}.`)
  }
  return value
}

export function requireFiniteNumber(rawValue: number, fieldName: string, minimum?: number, maximum?: number): number {
  if (typeof rawValue !== 'number' || !Number.isFinite(rawValue)) {
    throw new Error(`${fieldName} must be a finite number, got ${String(rawValue)}.`)
  }
  const value = rawValue
  if (minimum !== undefined && value < minimum) {
    throw new Error(`${fieldName} must be >= ${minimum}, got ${value}.`)
  }
  if (maximum !== undefined && value > maximum) {
    throw new Error(`${fieldName} must be <= ${maximum}, got ${value}.`)
  }
  return value
}

export function requireLtxSeed(rawValue: number): number {
  if (typeof rawValue !== 'number' || !Number.isFinite(rawValue)) {
    throw new Error(`seed must be a finite integer, got ${String(rawValue)}.`)
  }
  if (!Number.isInteger(rawValue)) {
    throw new Error(`seed must be an integer, got ${String(rawValue)}.`)
  }
  return rawValue
}

export function normalizeLtxExecutionProfile(rawValue: string): LtxExecutionProfile {
  const normalized = String(rawValue || '').trim()
  if (LTX_ALLOWED_EXECUTION_PROFILES.includes(normalized as (typeof LTX_ALLOWED_EXECUTION_PROFILES)[number])) {
    return normalized as LtxExecutionProfile
  }
  throw new Error(
    `Unsupported LTX execution profile '${rawValue}'. LTX currently accepts only ${LTX_ALLOWED_EXECUTION_PROFILES.map((value) => `'${value}'`).join(', ')}.`,
  )
}

export function resolveLtxDimAlignmentForExecutionProfile(rawValue: string): number {
  return normalizeLtxExecutionProfile(rawValue) === 'two_stage'
    ? LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT
    : LTX_DIM_ALIGNMENT
}

function normalizeOptionalSha(rawValue: string | null | undefined): string | undefined {
  const normalized = String(rawValue || '').trim().toLowerCase()
  if (!normalized) return undefined
  return normalized
}

function buildCommonFields(input: LtxVideoCommonInput): Omit<LtxTxt2VidPayload, 'txt2vid_prompt' | 'txt2vid_neg_prompt' | 'txt2vid_width' | 'txt2vid_height' | 'txt2vid_steps' | 'txt2vid_fps' | 'txt2vid_num_frames' | 'txt2vid_seed' | 'txt2vid_cfg_scale'> {
  const model = String(input.model || '').trim()
  if (!model) throw new Error('Select a checkpoint to generate.')
  const textEncoderSha = normalizeOptionalSha(input.textEncoderSha)
  if (!textEncoderSha) throw new Error('LTX requests require a text encoder sha.')

  const common: Omit<LtxTxt2VidPayload, 'txt2vid_prompt' | 'txt2vid_neg_prompt' | 'txt2vid_width' | 'txt2vid_height' | 'txt2vid_steps' | 'txt2vid_fps' | 'txt2vid_num_frames' | 'txt2vid_seed' | 'txt2vid_cfg_scale'> = {
    device: normalizeDevice(String(input.device || '')),
    settings_revision: requireSettingsRevision(input.settingsRevision),
    engine: 'ltx2',
    model,
    tenc_sha: textEncoderSha,
    video_save_output: true,
    video_save_metadata: true,
    video_return_frames: Boolean(input.videoReturnFrames),
    ltx_execution_profile: normalizeLtxExecutionProfile(input.executionProfile),
  }
  const modelSha = normalizeOptionalSha(input.modelSha)
  if (modelSha) common.model_sha = modelSha
  const vaeSha = normalizeOptionalSha(input.vaeSha)
  if (vaeSha) common.vae_sha = vaeSha
  return common
}

export function buildLtxTxt2VidPayload(input: LtxTxt2VidInput): LtxTxt2VidPayload {
  const common = buildCommonFields(input)
  const requiredDimAlignment = resolveLtxDimAlignmentForExecutionProfile(common.ltx_execution_profile)
  const payload: LtxTxt2VidPayload = {
    ...common,
    txt2vid_prompt: String(input.prompt || '').trim(),
    txt2vid_neg_prompt: String(input.negativePrompt || '').trim(),
    txt2vid_width: requireLtxDim(input.width, 'txt2vid_width', requiredDimAlignment),
    txt2vid_height: requireLtxDim(input.height, 'txt2vid_height', requiredDimAlignment),
    txt2vid_steps: requirePositiveInt(input.steps, 'txt2vid_steps'),
    txt2vid_fps: requirePositiveInt(input.fps, 'txt2vid_fps', 1, 240),
    txt2vid_num_frames: requireLtxFrameCount(input.frames),
    txt2vid_seed: requireLtxSeed(input.seed),
    txt2vid_cfg_scale: requireFiniteNumber(input.cfgScale, 'txt2vid_cfg_scale', 0),
  }
  return LtxTxt2VidPayloadSchema.parse(payload)
}

export function buildLtxImg2VidPayload(input: LtxImg2VidInput): LtxImg2VidPayload {
  const initImageData = String(input.initImageData || '').trim()
  if (!initImageData) throw new Error('Select an initial image for img2vid.')
  const common = buildCommonFields(input)
  const requiredDimAlignment = resolveLtxDimAlignmentForExecutionProfile(common.ltx_execution_profile)

  const payload: LtxImg2VidPayload = {
    ...common,
    img2vid_prompt: String(input.prompt || '').trim(),
    img2vid_neg_prompt: String(input.negativePrompt || '').trim(),
    img2vid_width: requireLtxDim(input.width, 'img2vid_width', requiredDimAlignment),
    img2vid_height: requireLtxDim(input.height, 'img2vid_height', requiredDimAlignment),
    img2vid_steps: requirePositiveInt(input.steps, 'img2vid_steps'),
    img2vid_fps: requirePositiveInt(input.fps, 'img2vid_fps', 1, 240),
    img2vid_num_frames: requireLtxFrameCount(input.frames),
    img2vid_seed: requireLtxSeed(input.seed),
    img2vid_cfg_scale: requireFiniteNumber(input.cfgScale, 'img2vid_cfg_scale', 0),
    img2vid_init_image: initImageData,
  }
  return LtxImg2VidPayloadSchema.parse(payload)
}
