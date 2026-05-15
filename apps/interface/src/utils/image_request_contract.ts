/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Pure helper for explicit frontend image request contract resolution.
Resolves checkpoint metadata, FLUX.2 guidance mode, asset-contract-backed text-encoder/VAE selectors, and canonical image `extras`
without importing Pinia/Vue stores directly. Callers inject store-backed resolver callbacks and remain responsible for translating thrown
contract `Error`s into UI state.

Symbols (top-level; keep in sync; no ghosts):
- `ImageRequestGuidanceMode` (type): Canonical frontend guidance mode for image payloads (`cfg` or `distilled_cfg`).
- `ImageRequestContractResolvers` (interface): Injected pure resolver surface used by `buildExplicitImageRequestContract(...)`.
- `BuildExplicitImageRequestContractArgs` (interface): Input contract for explicit image selector resolution + `extras` assembly.
- `ExplicitImageRequestContract` (interface): Resolved guidance mode + canonical `extras` payload for one image request.
- `buildExplicitImageRequestContract` (function): Resolves fail-loud image request selectors and required asset SHAs into canonical `extras`.
*/

import type { EngineAssetContract, ModelInfo } from '../api/types'

export type ImageRequestGuidanceMode = 'cfg' | 'distilled_cfg'

export interface ImageRequestContractResolvers {
  requireModelInfo: (label: string) => ModelInfo
  resolveFlux2CheckpointVariant: (source: string | ModelInfo) => 'base' | 'distilled' | null
  resolveTextEncoderSha: (label: string | null | undefined) => string | undefined
  resolveTextEncoderSlot: (label: string | null | undefined) => string | undefined
  requireVaeSelection: (label?: string | null) => string
  resolveVaeSha: (label: string | null | undefined) => string | undefined
  getAssetContract: (
    engine: string | null | undefined,
    opts: { checkpointCoreOnly: boolean },
  ) => EngineAssetContract | null
}

export interface BuildExplicitImageRequestContractArgs {
  modelLabel: string
  engineKey: string
  textEncoderLabels?: Array<string | null | undefined>
  selectedVaeLabel?: string | null
  zimageTurbo?: boolean
  fallbackGuidanceMode?: ImageRequestGuidanceMode
  resolvers: ImageRequestContractResolvers
}

export interface ExplicitImageRequestContract {
  guidanceMode: ImageRequestGuidanceMode | undefined
  extras: Record<string, unknown>
}

function requiredTextEncoderMessage(contract: EngineAssetContract): string {
  const requiredCount = Math.max(0, Math.trunc(Number(contract.tenc_count ?? 0)))
  const kindLabel = String(contract.tenc_kind_label || contract.tenc_kind || '').trim()
  if (kindLabel) {
    return `This engine requires exactly ${requiredCount} text encoder(s) (${kindLabel}).`
  }
  return `This engine requires exactly ${requiredCount} text encoder(s).`
}

function usesExplicitSdxlTextEncoderSelectors(
  engineKey: string,
  checkpointCoreOnly: boolean,
): boolean {
  return checkpointCoreOnly && (engineKey === 'sdxl' || engineKey === 'sdxl_refiner')
}

export function buildExplicitImageRequestContract(
  args: BuildExplicitImageRequestContractArgs,
): ExplicitImageRequestContract {
  const modelLabel = String(args.modelLabel || '').trim()
  if (!modelLabel) {
    throw new Error('Select a checkpoint to generate.')
  }

  const modelInfo = args.resolvers.requireModelInfo(modelLabel)
  const modelSha = String(modelInfo.hash || '').trim().toLowerCase()
  if (!modelSha) {
    throw new Error('Selected checkpoint is missing hash metadata. Refresh model inventory and retry.')
  }

  const modelFormat = String(modelInfo.format || '').trim().toLowerCase()
  if (modelFormat !== 'checkpoint' && modelFormat !== 'diffusers' && modelFormat !== 'gguf') {
    throw new Error('Selected checkpoint is missing format metadata. Refresh model inventory and retry.')
  }

  const checkpointCoreOnly = modelInfo.core_only
  if (typeof checkpointCoreOnly !== 'boolean') {
    throw new Error('Selected checkpoint is missing core-only metadata. Refresh model inventory and retry.')
  }

  let guidanceMode = args.fallbackGuidanceMode
  if (args.engineKey === 'flux2') {
    const variant = args.resolvers.resolveFlux2CheckpointVariant(modelInfo)
    if (!variant) {
      throw new Error('Unsupported FLUX.2 checkpoint variant. Only Klein 4B/base-4B is supported.')
    }
    guidanceMode = variant === 'base' ? 'cfg' : 'distilled_cfg'
  }

  const assetContract = args.resolvers.getAssetContract(args.engineKey, {
    checkpointCoreOnly,
  })
  if (!assetContract) {
    throw new Error(`Asset contract for '${args.engineKey}' is not available.`)
  }

  const extras: Record<string, unknown> = {
    model_sha: modelSha,
    checkpoint_core_only: checkpointCoreOnly,
    model_format: modelFormat,
  }

  const textEncoderLabels = Array.isArray(args.textEncoderLabels)
    ? args.textEncoderLabels
        .map((value) => String(value || '').trim())
        .filter((value) => value.length > 0)
    : []
  const requiredTencCount = Math.max(0, Math.trunc(Number(assetContract.tenc_count ?? 0)))
  if (requiredTencCount > 0) {
    if (textEncoderLabels.length === 0) {
      throw new Error(requiredTextEncoderMessage(assetContract))
    }
    if (usesExplicitSdxlTextEncoderSelectors(args.engineKey, checkpointCoreOnly)) {
      const expectedSlots = Array.isArray(assetContract.tenc_slots)
        ? assetContract.tenc_slots
            .map((value) => String(value || '').trim())
            .filter((value) => value.length > 0)
        : []
      if (expectedSlots.length !== requiredTencCount) {
        throw new Error(`SDXL core-only asset contract for '${args.engineKey}' is missing truthful text-encoder slots.`)
      }
      const slotToSha = new Map<string, string>()
      for (const label of textEncoderLabels) {
        const sha = args.resolvers.resolveTextEncoderSha(label)
        if (!sha) {
          throw new Error(`Text encoder SHA not found for '${label}'.`)
        }
        const slot = String(args.resolvers.resolveTextEncoderSlot(label) || '').trim()
        if (!slot) {
          throw new Error(`Text encoder slot metadata not found for '${label}'. Refresh inventory and retry.`)
        }
        if (!expectedSlots.includes(slot)) {
          continue
        }
        const previousSha = slotToSha.get(slot)
        if (previousSha && previousSha !== sha) {
          throw new Error(`Multiple text encoders selected for SDXL slot '${slot}'.`)
        }
        slotToSha.set(slot, sha)
      }
      const missingSlots = expectedSlots.filter((slot) => !slotToSha.has(slot))
      if (missingSlots.length > 0) {
        throw new Error(requiredTextEncoderMessage(assetContract))
      }
      expectedSlots.forEach((slot, index) => {
        extras[`tenc${index + 1}_sha`] = slotToSha.get(slot)
      })
    } else {
      const shas: string[] = []
      for (const label of textEncoderLabels) {
        const sha = args.resolvers.resolveTextEncoderSha(label)
        if (!sha) {
          throw new Error(`Text encoder SHA not found for '${label}'.`)
        }
        shas.push(sha)
      }
      if (shas.length !== requiredTencCount) {
        throw new Error(requiredTextEncoderMessage(assetContract))
      }
      extras.tenc_sha = shas.length === 1 ? shas[0] : shas
    }
  }

  const selectedVae = args.resolvers.requireVaeSelection(args.selectedVaeLabel)
  const selectedVaeIsSentinel = selectedVae === 'built-in' || selectedVae === 'none'
  const resolvedVaeSha = selectedVaeIsSentinel
    ? ''
    : String(args.resolvers.resolveVaeSha(selectedVae) || '').trim().toLowerCase()
  if (!selectedVaeIsSentinel && !resolvedVaeSha) {
    throw new Error('Selected VAE is invalid or stale. Re-select a VAE and retry.')
  }
  extras.vae_source = resolvedVaeSha ? 'external' : 'built_in'
  if (assetContract.requires_vae) {
    if (!resolvedVaeSha) {
      throw new Error('Select a VAE so the request can include vae_sha.')
    }
    extras.vae_sha = resolvedVaeSha
  } else if (resolvedVaeSha) {
    extras.vae_sha = resolvedVaeSha
  }

  if (args.engineKey === 'zimage') {
    extras.zimage_variant = args.zimageTurbo === false ? 'base' : 'turbo'
  }

  return {
    guidanceMode,
    extras,
  }
}
