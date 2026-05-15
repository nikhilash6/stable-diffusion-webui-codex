"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: IP-Adapter reference-image preprocessing and embedding preparation.
Decodes the already-selected concrete reference image, converts it into the clip-vision runtime tensor contract, and prepares the
conditional/unconditional IP-Adapter tokens without owning folder selection or pairing.

Symbols (top-level; keep in sync; no ghosts):
- `prepare_ip_adapter_embeddings` (function): Resolve the selected reference image and build the conditional/unconditional adapter tokens.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from PIL import Image

from apps.backend.runtime.adapters.ip_adapter.types import IpAdapterConfig, PreparedIpAdapterAssets, PreparedIpAdapterEmbeddings
from apps.backend.runtime.logging import get_backend_logger
from apps.backend.services.media_service import MediaService

logger = get_backend_logger(__name__)
_MEDIA_SERVICE = MediaService()


def prepare_ip_adapter_embeddings(
    *,
    processing,
    config: IpAdapterConfig,
    assets: PreparedIpAdapterAssets,
) -> PreparedIpAdapterEmbeddings:
    image = _resolve_reference_image(processing=processing, config=config)
    _, _, _, condition, uncondition = _prepare_ip_adapter_conditioning(
        image=image,
        assets=assets,
        crop=True,
    )
    logger.debug(
        "Prepared IP-Adapter embeddings layout=%s cond_shape=%s uncond_shape=%s",
        assets.layout.value,
        tuple(condition.shape),
        tuple(uncondition.shape),
    )
    return PreparedIpAdapterEmbeddings(condition=condition, uncondition=uncondition)


def _resolve_reference_image(*, processing, config: IpAdapterConfig) -> Image.Image:
    kind = str(config.source.kind or "").strip()
    if kind == "uploaded":
        if not config.source.reference_image_data:
            raise RuntimeError("IP-Adapter uploaded source requires reference_image_data.")
        return _MEDIA_SERVICE.decode_image(config.source.reference_image_data).convert("RGB")
    if kind == "same_as_init":
        init_image = getattr(processing, "init_image", None)
        if init_image is None:
            raise RuntimeError("IP-Adapter source.kind='same_as_init' requires img2img init_image.")
        if isinstance(init_image, Image.Image):
            return init_image.convert("RGB")
        raise RuntimeError(
            f"IP-Adapter source.kind='same_as_init' expected PIL.Image.Image init_image, got {type(init_image).__name__}."
        )
    if kind == "server_folder":
        raise RuntimeError(
            "IP-Adapter source.kind='server_folder' must be materialized by image automation before request preparation."
        )
    raise RuntimeError(f"Unsupported IP-Adapter source.kind '{kind}'.")


def _image_to_bhwc_tensor(image: Image.Image) -> torch.Tensor:
    rgb_image = image.convert("RGB")
    array = np.asarray(rgb_image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(f"IP-Adapter reference image must decode to HWC RGB; got shape={array.shape}.")
    return torch.from_numpy(array / 255.0).unsqueeze(0)


def _prepare_ip_adapter_conditioning(
    *,
    image: Image.Image,
    assets: PreparedIpAdapterAssets,
    crop: bool,
) -> tuple[torch.Tensor, torch.Tensor, object, torch.Tensor, torch.Tensor]:
    source_pixels = _image_to_bhwc_tensor(image)
    projector = copy.deepcopy(assets.image_projector)
    encoder = assets.image_encoder_runtime
    projector.to(device=encoder.load_device, dtype=encoder.runtime_dtype)
    projector.eval()
    with torch.inference_mode():
        processed = encoder.prepare_pixels(source_pixels, crop=bool(crop))
        encoded = encoder.encode_pixels(processed)
        if assets.uses_hidden_states:
            uncondition_encoded = encoder.encode_pixels(torch.zeros_like(processed))
            condition_inputs = encoded.penultimate_hidden_states
            uncondition_inputs = uncondition_encoded.penultimate_hidden_states
        else:
            condition_inputs = encoded.image_embeds
            uncondition_inputs = torch.zeros_like(condition_inputs)
        condition = projector(condition_inputs.to(device=encoder.load_device, dtype=encoder.runtime_dtype))
        uncondition = projector(uncondition_inputs.to(device=encoder.load_device, dtype=encoder.runtime_dtype))
    return source_pixels, processed, encoded, condition, uncondition
