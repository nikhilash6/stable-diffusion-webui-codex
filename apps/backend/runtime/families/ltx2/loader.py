"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LTX2 bundle planning helpers for loader/runtime handoff.
Converts parser-normalized LTX2 checkpoints into a typed bundle-planning contract with immutable parser-owned component
names, vendored metadata paths, one resolved external Gemma3 text-encoder asset, and the real split-pack side assets
required by LTX 2.3 GGUF core-only checkpoints.

Symbols (top-level; keep in sync; no ghosts):
- `prepare_ltx2_bundle_inputs` (function): Build the typed LTX2 bundle-planning contract from loader-side parser output.
- `build_ltx2_bundle_metadata` (function): Serialize stable loader metadata for future runtime/engine assembly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from apps.backend.infra.config.paths import get_paths_for
from apps.backend.runtime.checkpoint.io import load_torch_file, read_safetensors_metadata
from apps.backend.runtime.model_parser.families.ltx2 import split_ltx2_transformer_and_connectors_state
from apps.backend.runtime.model_parser.specs import CodexEstimatedConfig
from apps.backend.runtime.model_registry.specs import ModelFamily, ModelSignature

from .audio import (
    is_ltx2_wrapped_vocoder_state,
    split_ltx2_audio_bundle_state,
    validate_ltx2_audio_bundle_contract,
)
from .config import LTX2_COMPONENT_NAMES, LTX2_REQUIRED_TEXT_ENCODER_SLOT, resolve_ltx2_vendor_paths
from .model import Ltx2BundleInputs, Ltx2ComponentStates
from .text_encoder import resolve_ltx2_text_encoder_asset
from .vae import validate_ltx2_video_vae_contract


_SIDE_ASSET_SUFFIXES = (".safetensor", ".safetensors", ".pt", ".bin")


def _load_ltx2_state_dict(path: str) -> Mapping[str, object]:
    state_dict = load_torch_file(path, device="cpu")
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(f"LTX2 side asset must resolve to a state_dict mapping, got {type(state_dict).__name__}: {path}")
    return state_dict


def _resolve_unique_side_asset_path(
    *,
    key: str,
    label: str,
    required_name_fragment: str,
) -> str:
    candidates: list[str] = []
    seen: set[str] = set()
    fragment = required_name_fragment.lower()

    for raw_root in get_paths_for(key):
        root = Path(os.path.expanduser(str(raw_root).strip()))
        if root.is_file():
            lower_name = root.name.lower()
            if root.suffix.lower() in _SIDE_ASSET_SUFFIXES and fragment in lower_name:
                resolved = str(root.resolve(strict=False))
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(resolved)
            continue
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*"), key=lambda item: str(item).lower()):
            lower_name = path.name.lower()
            if not path.is_file():
                continue
            if path.suffix.lower() not in _SIDE_ASSET_SUFFIXES:
                continue
            if fragment not in lower_name:
                continue
            resolved = str(path.resolve(strict=False))
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)

    if not candidates:
        raise RuntimeError(
            f"LTX2 {label} resolution failed: no matching file containing {required_name_fragment!r} was found under paths.json[{key!r}]."
        )
    if len(candidates) != 1:
        raise RuntimeError(
            f"LTX2 {label} resolution failed: expected exactly one file containing {required_name_fragment!r} under paths.json[{key!r}], "
            f"got {len(candidates)} candidates: {candidates!r}"
        )
    return candidates[0]

def _read_ltx2_audio_bundle_vocoder_config(audio_bundle_path: str) -> Mapping[str, object] | None:
    metadata = read_safetensors_metadata(audio_bundle_path)
    raw_config_json = str(metadata.get("config") or "").strip()
    if not raw_config_json:
        return None
    try:
        payload = json.loads(raw_config_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "LTX2 audio bundle metadata contains invalid JSON in the `config` field: "
            f"path={audio_bundle_path!r} error={exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            "LTX2 audio bundle metadata `config` must decode to a mapping. "
            f"Got {type(payload).__name__} from {audio_bundle_path!r}."
        )
    raw_vocoder_config = payload.get("vocoder")
    if raw_vocoder_config is None:
        return None
    if not isinstance(raw_vocoder_config, Mapping):
        raise RuntimeError(
            "LTX2 audio bundle metadata `config.vocoder` must be a mapping. "
            f"Got {type(raw_vocoder_config).__name__} from {audio_bundle_path!r}."
        )
    return dict(raw_vocoder_config)


def prepare_ltx2_bundle_inputs(
    *,
    model_ref: str,
    estimated_config: CodexEstimatedConfig,
    signature: ModelSignature,
    text_encoder_override_paths: Mapping[str, str],
    vae_path: str | None,
    backend_root: Path,
) -> Ltx2BundleInputs:
    if signature.family is not ModelFamily.LTX2:
        raise RuntimeError(
            "LTX2 bundle planning requires `ModelFamily.LTX2`; "
            f"got {getattr(signature.family, 'value', signature.family)!r}."
        )

    component_map = {name: component.state_dict for name, component in estimated_config.components.items()}
    parser_split = str(estimated_config.extras.get("parser_split") or "").strip()

    if parser_split == "ltx2_core_only_gguf":
        if not isinstance(vae_path, str) or not vae_path.strip():
            raise RuntimeError(
                "LTX2 core-only GGUF bundle planning requires an external video VAE path. "
                "Provide one via request extras.vae_sha so the API can pass a valid vae_path."
            )
        resolved_vae_path = os.path.expanduser(vae_path.strip())
        if not os.path.isfile(resolved_vae_path):
            raise RuntimeError(f"LTX2 external video VAE path not found: {resolved_vae_path}")

        transformer_bundle = component_map.get("transformer")
        if transformer_bundle is None:
            raise RuntimeError(
                "LTX2 core-only GGUF bundle planning requires a parser-owned `transformer` component."
            )

        transformer_state, embedded_connectors_state = split_ltx2_transformer_and_connectors_state(transformer_bundle)
        if not transformer_state:
            raise RuntimeError("LTX2 core-only GGUF bundle planning produced an empty transformer state after connector separation.")
        if embedded_connectors_state:
            raise RuntimeError(
                "LTX2 core-only GGUF bundle planning requires the parser-owned `transformer` component to stay connector-free. "
                "Connector tensors must resolve from the external embeddings sidecar, not from the core transformer checkpoint."
            )

        connectors_path = _resolve_unique_side_asset_path(
            key="ltx2_connectors",
            label="embeddings connectors side asset",
            required_name_fragment="embeddings_connectors",
        )
        audio_bundle_path = _resolve_unique_side_asset_path(
            key="ltx2_vae",
            label="audio bundle side asset",
            required_name_fragment="audio_vae",
        )

        connectors_sidecar_state = _load_ltx2_state_dict(connectors_path)
        video_vae_state = _load_ltx2_state_dict(resolved_vae_path)
        audio_bundle_state = _load_ltx2_state_dict(audio_bundle_path)
        audio_vae_state, vocoder_state = split_ltx2_audio_bundle_state(audio_bundle_state)
        vocoder_config = _read_ltx2_audio_bundle_vocoder_config(audio_bundle_path)
        if is_ltx2_wrapped_vocoder_state(vocoder_state) and vocoder_config is None:
            raise RuntimeError(
                "LTX2 core-only GGUF bundle planning requires wrapped vocoder metadata in the audio bundle. "
                f"Missing `config.vocoder` in SafeTensors metadata for {audio_bundle_path!r}."
            )
        connectors_state = connectors_sidecar_state

        validate_ltx2_video_vae_contract(video_vae_state)
        validate_ltx2_audio_bundle_contract(audio_vae_state=audio_vae_state, vocoder_state=vocoder_state)
        components = Ltx2ComponentStates.from_component_map(
            {
                "transformer": transformer_state,
                "connectors": connectors_state,
                "vae": video_vae_state,
                "audio_vae": audio_vae_state,
                "vocoder": vocoder_state,
            }
        )
        estimated_config = estimated_config.replace_components(
            {
                "transformer": transformer_state,
                "connectors": connectors_state,
                "vae": video_vae_state,
                "audio_vae": audio_vae_state,
                "vocoder": vocoder_state,
            }
        )
    else:
        components = Ltx2ComponentStates.from_component_map(component_map)
        validate_ltx2_video_vae_contract(components.vae)
        validate_ltx2_audio_bundle_contract(
            audio_vae_state=components.audio_vae,
            vocoder_state=components.vocoder,
        )
        vocoder_config = None
        if is_ltx2_wrapped_vocoder_state(components.vocoder):
            raise RuntimeError(
                "LTX2 bundle planning encountered a wrapped 2.3 vocoder state without an audio-bundle metadata source. "
                "Only the real split audio bundle path is supported for wrapped vocoder layouts."
            )

    vendor_paths = resolve_ltx2_vendor_paths(
        backend_root=backend_root,
        repo_id=str(estimated_config.repo_id or "").strip(),
    )
    text_encoder = resolve_ltx2_text_encoder_asset(
        override_paths=text_encoder_override_paths,
        vendor_paths=vendor_paths,
    )
    if text_encoder.alias != LTX2_REQUIRED_TEXT_ENCODER_SLOT:
        raise RuntimeError(
            "LTX2 bundle planning resolved an unexpected text-encoder alias; "
            f"expected {LTX2_REQUIRED_TEXT_ENCODER_SLOT!r}, got {text_encoder.alias!r}."
        )

    return Ltx2BundleInputs(
        model_ref=model_ref,
        signature=signature,
        estimated_config=estimated_config,
        components=components,
        text_encoder=text_encoder,
        vendor_paths=vendor_paths,
        vocoder_config=vocoder_config,
    )


def build_ltx2_bundle_metadata(inputs: Ltx2BundleInputs) -> dict[str, object]:
    return {
        "engine_key": "ltx2",
        "asset_repo_id": str(inputs.estimated_config.repo_id),
        "parser_split": str(inputs.estimated_config.extras.get("parser_split", "")),
        "component_names": LTX2_COMPONENT_NAMES,
        "tenc_path": inputs.text_encoder.path,
        "tenc_kind": inputs.text_encoder.kind,
        "tenc_alias": inputs.text_encoder.alias,
        "tokenizer_dir": inputs.text_encoder.tokenizer_dir,
        "model_index_path": inputs.vendor_paths.model_index_path,
        "connectors_config_path": inputs.vendor_paths.connectors_config_path,
        "vendor_repo_dir": inputs.vendor_paths.repo_dir,
        "vocoder_config": inputs.vocoder_config,
    }
