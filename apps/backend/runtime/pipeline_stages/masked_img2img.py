"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Masked img2img (“inpaint”) helpers for request-driven latent-mask pipelines.
Normalizes masks (RGBA alpha semantics), applies invert/blur/round options, optionally builds an inpaint-full-res crop plan
(Forge-style zoom-crop + paste-back overlay), and produces latent-space masks for sampler enforcement.
Keeps init/mask at the original image resolution and resizes only the cropped patch to `processing.width/height` for
sampling (Forge/A1111 “Only masked” semantics).

Symbols (top-level; keep in sync; no ghosts):
- `InpaintFullResPlan` (dataclass): Full-res inpaint plan (crop region + overlay composite inputs).
- `MaskedImg2ImgBundle` (dataclass): Prepared init tensor/latents + latent masks + optional full-res plan.
- `MaskEnforcerHooks` (dataclass): Centralized generic masked-runtime hook set returned by `resolve_mask_enforcer_hooks(...)`.
- `LatentMaskEnforcer` (class): Latent masking helper implementing post-sample/per-step blend hooks.
- `compute_mask_connected_component_bboxes` (function): Returns connected-component bounding boxes for a binary inpaint mask (for multi-region flows).
- `resolve_mask_enforcer_hooks` (function): Resolve the runtime hook set for a validated generic inpaint mode.
- `prepare_masked_img2img_bundle` (function): Build a masked img2img bundle from a processing object and sampling plan.
- `apply_inpaint_full_res_composite` (function): Paste-back + overlay composite for full-res inpaint outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

from apps.backend.runtime.processing.conditioners import (
    encode_image_batch,
    normalize_torch_manual_seed,
    resolve_processing_encode_seed,
)
from apps.backend.runtime.pipeline_stages.image_io import pil_to_tensor

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
_RESAMPLE_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST

InpaintMode = str
INPAINT_MODE_POST_SAMPLE_BLEND = "post_sample_blend"
INPAINT_MODE_PER_STEP_BLEND = "per_step_blend"
ALLOWED_INPAINT_MODES = frozenset({INPAINT_MODE_POST_SAMPLE_BLEND, INPAINT_MODE_PER_STEP_BLEND})


@dataclass(slots=True)
class InpaintFullResPlan:
    """Full-res “Only masked” plan (Forge-style zoom-crop + paste-back)."""

    crop_region: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in init-image coordinates
    paste_to: Tuple[int, int, int, int]  # (x, y, w, h) in init-image coordinates
    overlay: Image.Image  # RGBA overlay with original unmasked pixels


@dataclass(slots=True)
class MaskedImg2ImgBundle:
    """Prepared init tensor/latents + latent masks for a masked img2img run."""

    init_tensor: torch.Tensor
    init_latent: torch.Tensor
    image_conditioning: torch.Tensor | None
    latent_masked: torch.Tensor  # 1 inside mask, 0 outside (shape 1x1xHlatentxWlatent)
    latent_unmasked: torch.Tensor  # 1 outside mask, 0 inside (shape 1x1xHlatentxWlatent)
    full_res: InpaintFullResPlan | None = None


@dataclass(slots=True, frozen=True)
class MaskEnforcerHooks:
    """Centralized masked-enforcement hook set."""

    pre_denoiser: Callable[[torch.Tensor, torch.Tensor, int, int | None], torch.Tensor] | None = None
    post_denoiser: Callable[[torch.Tensor, torch.Tensor, int, int | None], torch.Tensor] | None = None
    post_step: Callable[[torch.Tensor, int, int | None], None] | None = None
    post_sample: Callable[[torch.Tensor], torch.Tensor] | None = None


class LatentMaskEnforcer:
    """Applies latent masking constraints (post-sample blend and/or per-step blend)."""

    def __init__(
        self,
        *,
        init_latent: torch.Tensor,
        latent_masked: torch.Tensor,
        latent_unmasked: torch.Tensor,
        per_step_blend_strength: float,
        per_step_blend_steps: int | None,
        noise_scaling: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        noise_seeds: Sequence[int] | None = None,
    ) -> None:
        strength = float(per_step_blend_strength)
        if not math.isfinite(strength):
            raise ValueError("per_step_blend_strength must be finite")
        if strength < 0.0 or strength > 1.0:
            raise ValueError("per_step_blend_strength must be between 0.0 and 1.0")
        configured_steps = self._parse_configured_steps(per_step_blend_steps)
        self._init_latent = init_latent
        self._latent_masked = latent_masked
        self._latent_unmasked = latent_unmasked
        self._per_step_blend_strength = strength
        self._per_step_blend_steps = configured_steps
        self._noise_scaling = noise_scaling
        raw_seeds = [int(seed) for seed in (noise_seeds or [0])]
        if not raw_seeds:
            raw_seeds = [0]
        self._noise_seeds = tuple(normalize_torch_manual_seed(seed) for seed in raw_seeds)
        self._cache: dict[
            tuple[torch.device, torch.dtype, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._pre_denoiser_noise_calls = 0

    @staticmethod
    def _expand_batch(tensor: torch.Tensor, *, batch: int) -> torch.Tensor:
        current_batch = int(tensor.shape[0])
        if current_batch <= 0:
            raise ValueError("tensor batch must be >= 1")
        if current_batch == batch:
            return tensor
        repeats = (batch + current_batch - 1) // current_batch
        return tensor.repeat((repeats, 1, 1, 1))[:batch]

    @staticmethod
    def _new_generator(*, device: torch.device, seed: int) -> torch.Generator:
        if device.type == "cuda":
            generator = torch.Generator(device=device)
        else:
            generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _build_noise_sample(self, init_latent: torch.Tensor, *, call_index: int) -> torch.Tensor:
        batch = int(init_latent.shape[0])
        samples: list[torch.Tensor] = []
        for index in range(batch):
            seed = (self._noise_seeds[index % len(self._noise_seeds)] + int(call_index)) % (1 << 64)
            generator = self._new_generator(device=init_latent.device, seed=seed)
            sample = torch.randn(
                tuple(init_latent[index : index + 1].shape),
                generator=generator,
                device=init_latent.device,
                dtype=init_latent.dtype,
            )
            samples.append(sample)
        return torch.cat(samples, dim=0)

    def _next_pre_denoiser_noise(self, init_latent: torch.Tensor) -> torch.Tensor:
        call_index = self._pre_denoiser_noise_calls
        self._pre_denoiser_noise_calls += 1
        return self._build_noise_sample(init_latent, call_index=call_index)

    def _materialize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(x.shape[0])
        key = (x.device, x.dtype, batch)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        init_latent = self._init_latent
        if init_latent.device != x.device or init_latent.dtype != x.dtype:
            init_latent = init_latent.to(device=x.device, dtype=x.dtype)
        init_latent = self._expand_batch(init_latent, batch=batch)
        if tuple(init_latent.shape[1:]) != tuple(x.shape[1:]):
            raise ValueError(
                f"init_latent shape mismatch: got={tuple(init_latent.shape)} expected batchx{tuple(x.shape[1:])}"
            )

        masked = self._latent_masked
        if masked.device != x.device or masked.dtype != x.dtype:
            masked = masked.to(device=x.device, dtype=x.dtype)
        masked = self._expand_batch(masked, batch=batch)
        if tuple(masked.shape[2:]) != tuple(x.shape[2:]):
            raise ValueError(
                f"mask spatial shape mismatch: got={tuple(masked.shape)} expected batchx?x{tuple(x.shape[2:])}"
            )
        if int(masked.shape[1]) not in (1, int(x.shape[1])):
            raise ValueError(
                f"mask channel mismatch: got={int(masked.shape[1])} expected 1 or {int(x.shape[1])}"
            )

        unmasked = self._latent_unmasked
        if unmasked.device != x.device or unmasked.dtype != x.dtype:
            unmasked = unmasked.to(device=x.device, dtype=x.dtype)
        unmasked = self._expand_batch(unmasked, batch=batch)
        if tuple(unmasked.shape) != tuple(masked.shape):
            raise ValueError(
                f"unmasked mask shape mismatch: got={tuple(unmasked.shape)} expected {tuple(masked.shape)}"
            )

        init_unmasked = init_latent * unmasked
        self._cache[key] = (masked, unmasked, init_latent, init_unmasked)
        return self._cache[key]

    @staticmethod
    def _sigma_to_latent_shape(sigma: torch.Tensor, *, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if sigma.ndim == 0:
            sigma = sigma.view(1)
        if sigma.ndim != 1:
            raise ValueError(f"sigma must be 1D or scalar; got shape={tuple(sigma.shape)}")
        if int(sigma.shape[0]) not in (1, batch):
            raise ValueError(f"sigma batch mismatch: got={int(sigma.shape[0])} expected 1 or {batch}")
        if int(sigma.shape[0]) == 1 and batch != 1:
            sigma = sigma.expand(batch)
        return sigma.to(device=device, dtype=dtype).view(batch, 1, 1, 1)

    @staticmethod
    def _parse_configured_steps(per_step_blend_steps: int | None) -> int:
        if per_step_blend_steps is None:
            raise ValueError("per_step_blend_steps must be provided explicitly")
        if isinstance(per_step_blend_steps, bool):
            raise ValueError("per_step_blend_steps must be an integer >= 0, got boolean")
        if isinstance(per_step_blend_steps, int):
            parsed = per_step_blend_steps
        elif isinstance(per_step_blend_steps, float):
            if not per_step_blend_steps.is_integer():
                raise ValueError(f"per_step_blend_steps must be an integer >= 0, got {per_step_blend_steps!r}")
            parsed = int(per_step_blend_steps)
        else:
            raise ValueError(
                f"per_step_blend_steps must be an integer >= 0, got {type(per_step_blend_steps).__name__}"
            )
        if parsed < 0:
            raise ValueError(f"per_step_blend_steps must be >= 0, got {parsed}")
        return parsed

    @staticmethod
    def _parse_step_index(step: int) -> int:
        if isinstance(step, bool):
            raise ValueError("step must be an integer >= 1, got boolean")
        parsed = int(step)
        if parsed < 1:
            raise ValueError(f"step must be >= 1, got {parsed}")
        return parsed

    @staticmethod
    def _parse_total_steps(steps: int | None) -> int | None:
        if steps is None:
            return None
        if isinstance(steps, bool):
            raise ValueError("steps must be an integer >= 1 when provided, got boolean")
        parsed = int(steps)
        if parsed < 1:
            raise ValueError(f"steps must be >= 1 when provided, got {parsed}")
        return parsed

    def resolve_active_step_limit(self, steps: int | None) -> int | None:
        total_steps = self._parse_total_steps(steps)
        if self._per_step_blend_steps == 0:
            return total_steps
        if total_steps is None:
            return self._per_step_blend_steps
        return min(self._per_step_blend_steps, total_steps)

    def is_step_active(self, *, step: int, steps: int | None) -> bool:
        current_step = self._parse_step_index(step)
        active_limit = self.resolve_active_step_limit(steps)
        if active_limit is None:
            return True
        return current_step <= active_limit

    def uses_total_per_step_blend(self) -> bool:
        return self._per_step_blend_strength == 1.0 and self._per_step_blend_steps == 0

    @staticmethod
    def _blend_toward_target(
        current: torch.Tensor,
        *,
        target: torch.Tensor,
        unmasked: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        if strength <= 0.0:
            return current
        current.add_((target - current) * unmasked * strength)
        return current

    def pre_denoiser(self, x: torch.Tensor, sigma: torch.Tensor, step: int, steps: int | None) -> torch.Tensor:  # noqa: ARG002
        masked, unmasked, init_latent, _init_unmasked = self._materialize(x)
        noise_sample = self._next_pre_denoiser_noise(init_latent)
        sigma_view = self._sigma_to_latent_shape(
            sigma,
            batch=int(x.shape[0]),
            device=x.device,
            dtype=x.dtype,
        )
        if self._noise_scaling is not None:
            noisy_init = self._noise_scaling(sigma_view, noise_sample, init_latent)
        else:
            noisy_init = init_latent + sigma_view * noise_sample
        x.mul_(masked)
        x.add_(noisy_init * unmasked)
        return x

    def post_denoiser(self, denoised: torch.Tensor, sigma: torch.Tensor, step: int, steps: int | None) -> torch.Tensor:  # noqa: ARG002
        masked, _unmasked, _init_latent, init_unmasked = self._materialize(denoised)
        denoised.mul_(masked)
        denoised.add_(init_unmasked)
        return denoised

    def post_step(self, x: torch.Tensor, step: int, steps: int | None) -> None:
        if not self.is_step_active(step=step, steps=steps):
            return
        _masked, unmasked, init_latent, _init_unmasked = self._materialize(x)
        self._blend_toward_target(
            x,
            target=init_latent,
            unmasked=unmasked,
            strength=self._per_step_blend_strength,
        )

    def post_sample(self, x: torch.Tensor) -> torch.Tensor:
        masked, _unmasked, _init_latent, init_unmasked = self._materialize(x)
        x.mul_(masked)
        x.add_(init_unmasked)
        return x


def resolve_mask_enforcer_hooks(
    enforcer: LatentMaskEnforcer,
    *,
    enforce_mode: InpaintMode,
) -> MaskEnforcerHooks:
    enforcement_value = str(enforce_mode).strip()
    if enforcement_value == INPAINT_MODE_PER_STEP_BLEND:
        if enforcer.uses_total_per_step_blend():
            return MaskEnforcerHooks(
                pre_denoiser=enforcer.pre_denoiser,
                post_denoiser=enforcer.post_denoiser,
                post_sample=enforcer.post_sample,
            )
        return MaskEnforcerHooks(
            post_step=enforcer.post_step,
            post_sample=enforcer.post_sample,
        )
    if enforcement_value == INPAINT_MODE_POST_SAMPLE_BLEND:
        return MaskEnforcerHooks(post_sample=enforcer.post_sample)
    raise ValueError(f"Unknown mask enforcement '{enforcement_value}' (internal validation bug)")


def _create_binary_mask(mask: Image.Image, *, round_mask: bool) -> Image.Image:
    if mask.mode == "RGBA":
        extrema = mask.getextrema()
        alpha_extrema = extrema[-1] if isinstance(extrema, tuple) and len(extrema) == 4 else None
        if alpha_extrema is not None and alpha_extrema != (255, 255):
            alpha = mask.split()[-1].convert("L")
            return alpha.point(lambda x: 255 if x > 128 else 0) if round_mask else alpha

    gray = mask.convert("L")
    return gray.point(lambda x: 255 if x > 128 else 0) if round_mask else gray


def _gaussian_kernel_1d(*, sigma: float, device: torch.device) -> tuple[torch.Tensor, int]:
    sigma = float(sigma)
    if sigma <= 0.0:
        kernel = torch.tensor([1.0], device=device, dtype=torch.float32)
        return kernel, 0
    radius = int(2.5 * sigma + 0.5)
    if radius <= 0:
        kernel = torch.tensor([1.0], device=device, dtype=torch.float32)
        return kernel, 0
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel, radius


def _blur_mask(mask: Image.Image, *, sigma_x: float, sigma_y: float) -> Image.Image:
    sigma_x = float(sigma_x)
    sigma_y = float(sigma_y)
    if sigma_x <= 0.0 and sigma_y <= 0.0:
        return mask

    array = np.array(mask.convert("L"), dtype=np.float32)
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0) / 255.0

    if sigma_x > 0.0:
        kernel, radius = _gaussian_kernel_1d(sigma=sigma_x, device=tensor.device)
        if radius > 0:
            tensor = F.pad(tensor, (radius, radius, 0, 0), mode="replicate")
        weight = kernel.view(1, 1, 1, -1)
        tensor = F.conv2d(tensor, weight)

    if sigma_y > 0.0:
        kernel, radius = _gaussian_kernel_1d(sigma=sigma_y, device=tensor.device)
        if radius > 0:
            tensor = F.pad(tensor, (0, 0, radius, radius), mode="replicate")
        weight = kernel.view(1, 1, -1, 1)
        tensor = F.conv2d(tensor, weight)

    out = tensor.squeeze(0).squeeze(0).clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
    return Image.fromarray(out, mode="L")


def _get_crop_region(mask: Image.Image, *, pad: int) -> tuple[int, int, int, int] | None:
    if pad < 0:
        raise ValueError("pad must be >= 0")
    box = mask.getbbox()
    if box is None:
        return None
    x1, y1, x2, y2 = box
    if pad == 0:
        return int(x1), int(y1), int(x2), int(y2)
    return (
        max(int(x1) - pad, 0),
        max(int(y1) - pad, 0),
        min(int(x2) + pad, mask.size[0]),
        min(int(y2) + pad, mask.size[1]),
    )


def _expand_crop_region(
    crop_region: tuple[int, int, int, int],
    *,
    processing_width: int,
    processing_height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = crop_region
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    if processing_width <= 0 or processing_height <= 0:
        raise ValueError("processing_width/processing_height must be positive")
    ratio_crop = float(crop_w) / float(crop_h)
    ratio_processing = float(processing_width) / float(processing_height)

    if ratio_crop > ratio_processing:
        desired_h = crop_w / ratio_processing
        diff = int(desired_h - crop_h)
        y1 -= diff // 2
        y2 += diff - diff // 2
        if y2 >= image_height:
            overflow = y2 - image_height
            y2 -= overflow
            y1 -= overflow
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_w = crop_h * ratio_processing
        diff = int(desired_w - crop_w)
        x1 -= diff // 2
        x2 += diff - diff // 2
        if x2 >= image_width:
            overflow = x2 - image_width
            x2 -= overflow
            x1 -= overflow
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if x2 >= image_width:
            x2 = image_width

    return int(x1), int(y1), int(x2), int(y2)


def compute_mask_connected_component_bboxes(
    raw_mask: Image.Image,
    *,
    round_mask: bool,
    min_component_pixels: int = 8,
    max_components: int = 16,
    max_foreground_pixels: int = 250_000,
) -> list[tuple[int, int, int, int]]:
    """Return connected-component bounding boxes for a raw inpaint mask.

    This is a best-effort helper intended for ADetailer-like multi-region flows.

    Notes:
    - Runs on a binary view of the mask (alpha semantics supported for RGBA inputs).
    - Uses 8-connectivity.
    - Caps work for extremely dense masks; in that case, returns a single bbox to avoid worst-case runtime.
    """

    if min_component_pixels <= 0:
        raise ValueError("min_component_pixels must be >= 1")
    if max_components <= 0:
        raise ValueError("max_components must be >= 1")
    if max_foreground_pixels <= 0:
        raise ValueError("max_foreground_pixels must be >= 1")

    mask = _create_binary_mask(raw_mask, round_mask=round_mask).convert("L")
    array = np.array(mask, dtype=np.uint8)
    foreground = array > 0
    foreground_pixels = int(np.count_nonzero(foreground))
    if foreground_pixels <= 0:
        return []
    if foreground_pixels > max_foreground_pixels:
        box = mask.getbbox()
        return [tuple(int(v) for v in box)] if box is not None else []

    height, width = foreground.shape
    ys, xs = np.nonzero(foreground)
    if ys.size == 0:
        return []

    # Mutate `foreground` in-place to avoid a separate visited array.
    components: list[tuple[int, int, int, int, int]] = []
    stack: list[tuple[int, int]] = []
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    for y0, x0 in zip(ys, xs):
        y0_i = int(y0)
        x0_i = int(x0)
        if not foreground[y0_i, x0_i]:
            continue
        foreground[y0_i, x0_i] = False
        stack.append((y0_i, x0_i))
        min_x = max_x = x0_i
        min_y = max_y = y0_i
        area = 0

        while stack:
            y, x = stack.pop()
            area += 1
            if x < min_x:
                min_x = x
            elif x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            elif y > max_y:
                max_y = y

            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if not foreground[ny, nx]:
                    continue
                foreground[ny, nx] = False
                stack.append((ny, nx))

        if area < min_component_pixels:
            continue
        components.append((min_x, min_y, max_x + 1, max_y + 1, int(area)))
        if len(components) >= max_components:
            break

    components.sort(key=lambda item: item[4], reverse=True)
    return [(x1, y1, x2, y2) for x1, y1, x2, y2, _area in components]


def _overlay_from_mask(*, image: Image.Image, mask_for_overlay: Image.Image) -> Image.Image:
    return _preserved_region_scaffold(image=image, mask=mask_for_overlay).convert("RGBA")


def _preserved_region_scaffold(*, image: Image.Image, mask: Image.Image) -> Image.Image:
    """Build a premultiplied-alpha preserved-region scaffold for soft mask edges."""
    preserved = Image.new("RGBa", image.size)
    preserved.paste(
        image.convert("RGBA").convert("RGBa"),
        mask=ImageOps.invert(mask.convert("L")),
    )
    return preserved.convert("RGBa")


def _fill_masked_regions(image: Image.Image, *, mask: Image.Image) -> Image.Image:
    """Forge/A1111-parity masked-region fill (blur-smear) for `inpainting_fill`."""
    base = Image.new("RGBA", image.size, (0, 0, 0, 0))
    scaffold = _preserved_region_scaffold(image=image, mask=mask)

    for radius, repeats in ((256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)):
        blurred = scaffold.filter(ImageFilter.GaussianBlur(radius)).convert("RGBA")
        for _ in range(repeats):
            base.alpha_composite(blurred)
    return base.convert("RGB")


def _latent_mask_from_image(
    mask: Image.Image,
    *,
    latent_width: int,
    latent_height: int,
    round_mask: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if latent_width <= 0 or latent_height <= 0:
        raise ValueError("latent_width/latent_height must be positive")
    resized = mask.convert("RGB").resize((latent_width, latent_height), resample=_RESAMPLE_LANCZOS)
    array = np.array(resized, dtype=np.float32) / 255.0
    plane = array[..., 0]
    if round_mask:
        plane = np.around(plane)
    tensor = torch.from_numpy(plane).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    return tensor


def _validate_inpaint_mode(mode: Any) -> InpaintMode:
    value = str(mode or "").strip()
    if value not in ALLOWED_INPAINT_MODES:
        raise ValueError(
            f"Invalid inpaint_mode '{value}'. Allowed: {sorted(ALLOWED_INPAINT_MODES)}"
        )
    return value


def prepare_masked_img2img_bundle(
    processing: Any,
    plan: Any,
    *,
    enforce_mode: Any,
    include_image_conditioning: bool = True,
) -> tuple[MaskedImg2ImgBundle, LatentMaskEnforcer]:
    """Prepare masked img2img inputs (mask processing + optional full-res plan + latent masks).

    Notes:
    - Requires `processing.init_image` and `processing.mask`.
    - Init image + mask are kept full-res; only the crop patch is resized to `processing.width/height` for sampling
      (Forge/A1111 “Only masked” semantics).
    """
    init_image = getattr(processing, "init_image", None)
    if init_image is None:
        raise ValueError("masked img2img requires processing.init_image")

    raw_mask = getattr(processing, "mask", None)
    if raw_mask is None:
        raise ValueError("masked img2img requires processing.mask")

    _validate_inpaint_mode(enforce_mode)

    width = int(getattr(processing, "width", 0) or 0)
    height = int(getattr(processing, "height", 0) or 0)
    if width <= 0 or height <= 0:
        raise ValueError("processing.width/height must be set for img2img")

    if raw_mask.size != init_image.size:
        raise ValueError(
            f"Mask size must match init image size; got mask={raw_mask.size} init={init_image.size}"
        )

    mask_round = bool(getattr(processing, "mask_round", True))
    invert = bool(int(getattr(processing, "inpainting_mask_invert", 0) or 0))
    blur_x = float(getattr(processing, "mask_blur_x", getattr(processing, "mask_blur", 0)) or 0)
    blur_y = float(getattr(processing, "mask_blur_y", getattr(processing, "mask_blur", 0)) or 0)

    mask = _create_binary_mask(raw_mask, round_mask=mask_round)
    if invert:
        mask = ImageOps.invert(mask)
        processing.update_extra_param("Mask mode", "Inpaint not masked")

    if blur_x > 0.0 or blur_y > 0.0:
        mask = _blur_mask(mask, sigma_x=blur_x, sigma_y=blur_y)
        processing.update_extra_param("Mask blur", int(getattr(processing, "mask_blur", 0) or 0))

    full_res_plan: InpaintFullResPlan | None = None
    mask_for_sampling = mask
    image_for_sampling = init_image.convert("RGB")

    pad = int(getattr(processing, "inpaint_full_res_padding", 0) or 0)
    crop_region = _get_crop_region(mask.convert("L"), pad=pad)
    if crop_region is None:
        raise ValueError('Unable to perform "Inpaint only masked" because mask is blank')
    crop_region = _expand_crop_region(
        crop_region,
        processing_width=width,
        processing_height=height,
        image_width=init_image.size[0],
        image_height=init_image.size[1],
    )
    x1, y1, x2, y2 = crop_region
    paste_to = (x1, y1, x2 - x1, y2 - y1)
    full_res_plan = InpaintFullResPlan(
        crop_region=crop_region,
        paste_to=paste_to,
        overlay=_overlay_from_mask(image=init_image.convert("RGB"), mask_for_overlay=mask),
    )
    mask_crop = mask.crop(crop_region)
    mask_for_sampling = mask_crop.resize((width, height), resample=_RESAMPLE_LANCZOS)
    image_crop = image_for_sampling.crop(crop_region)
    image_for_sampling = image_crop.resize((width, height), resample=_RESAMPLE_LANCZOS)
    processing.update_extra_param("Inpaint area", "Only masked")
    processing.update_extra_param("Masked area padding", int(pad))

    inpainting_fill = int(getattr(processing, "inpainting_fill", 0) or 0)
    if inpainting_fill != 1:
        image_for_sampling = _fill_masked_regions(image_for_sampling, mask=mask_for_sampling)
        if inpainting_fill == 0:
            processing.update_extra_param("Masked content", "fill")

    init_tensor = pil_to_tensor([image_for_sampling])
    round_conditioning_mask = bool(getattr(processing, "mask_round", True))
    per_step_blend_strength = float(getattr(processing, "per_step_blend_strength"))
    per_step_blend_steps = LatentMaskEnforcer._parse_configured_steps(getattr(processing, "per_step_blend_steps", None))
    if not math.isfinite(per_step_blend_strength):
        raise ValueError("processing.per_step_blend_strength must be finite")
    if per_step_blend_strength < 0.0 or per_step_blend_strength > 1.0:
        raise ValueError("processing.per_step_blend_strength must be between 0.0 and 1.0")
    init_latent = encode_image_batch(
        processing.sd_model,
        init_tensor,
        encode_seed=resolve_processing_encode_seed(processing),
        stage="runtime.pipeline_stages.masked_img2img.prepare_masked_img2img_bundle.encode",
    )

    latent_h = int(init_latent.shape[2])
    latent_w = int(init_latent.shape[3])
    latent_masked = _latent_mask_from_image(
        mask_for_sampling,
        latent_width=latent_w,
        latent_height=latent_h,
        round_mask=mask_round,
        device=init_latent.device,
        dtype=init_latent.dtype,
    )
    latent_unmasked = (1.0 - latent_masked).to(device=init_latent.device, dtype=init_latent.dtype)
    noise_seeds: Sequence[int] = list(getattr(plan, "seeds", []) or getattr(processing, "seeds", []) or [])
    if not noise_seeds:
        noise_seeds = [int(getattr(processing, "seed", 0) or 0)]

    if inpainting_fill == 2:
        gens = []
        for seed in noise_seeds:
            gen = torch.Generator(device=init_latent.device)
            gen.manual_seed(normalize_torch_manual_seed(int(seed)))
            gens.append(torch.randn(tuple(init_latent.shape[1:]), generator=gen, device=init_latent.device, dtype=init_latent.dtype))
        noise = torch.stack(gens, dim=0)
        init_latent = init_latent * latent_unmasked + noise * latent_masked
        processing.update_extra_param("Masked content", "latent noise")
    elif inpainting_fill == 3:
        init_latent = init_latent * latent_unmasked
        processing.update_extra_param("Masked content", "latent nothing")

    image_conditioning: torch.Tensor | None = None
    if include_image_conditioning:
        from apps.backend.runtime.processing.conditioners import img2img_conditioning

        image_conditioning = img2img_conditioning(
            processing.sd_model,
            init_tensor,
            init_latent,
            image_mask=mask_for_sampling,
            round_mask=round_conditioning_mask,
        )

    bundle = MaskedImg2ImgBundle(
        init_tensor=init_tensor,
        init_latent=init_latent,
        image_conditioning=image_conditioning,
        latent_masked=latent_masked,
        latent_unmasked=latent_unmasked,
        full_res=full_res_plan,
    )
    predictor = getattr(getattr(getattr(processing.sd_model, "codex_objects", None), "denoiser", None), "model", None)
    predictor = getattr(predictor, "predictor", None)
    noise_scaling = None
    if predictor is not None and callable(getattr(predictor, "noise_scaling", None)):
        def _predictor_noise_scaling(sigma: torch.Tensor, noise: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
            return predictor.noise_scaling(
                sigma,
                noise,
                latent,
                max_denoise=False,
            )

        noise_scaling = _predictor_noise_scaling

    if str(enforce_mode).strip() == INPAINT_MODE_PER_STEP_BLEND:
        processing.update_extra_param("Per-step blend strength", float(per_step_blend_strength))
        processing.update_extra_param(
            "Per-step blend steps",
            "all" if per_step_blend_steps == 0 else int(per_step_blend_steps),
        )

    enforcer = LatentMaskEnforcer(
        init_latent=init_latent,
        latent_masked=latent_masked,
        latent_unmasked=latent_unmasked,
        per_step_blend_strength=per_step_blend_strength,
        per_step_blend_steps=per_step_blend_steps,
        noise_scaling=noise_scaling,
        noise_seeds=noise_seeds,
    )
    return bundle, enforcer


def _uncrop(image: Image.Image, *, dest_size: tuple[int, int], paste_to: tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = paste_to
    base = Image.new("RGBA", dest_size, (0, 0, 0, 0))
    resized = image.resize((w, h), resample=_RESAMPLE_LANCZOS).convert("RGBA")
    base.paste(resized, (x, y))
    return base


def apply_inpaint_full_res_composite(
    images: Sequence[Image.Image],
    *,
    plan: InpaintFullResPlan,
) -> list[Image.Image]:
    """Apply paste-back + overlay compositing for full-res inpaint outputs."""
    if plan.overlay.mode != "RGBA":
        raise ValueError("full-res overlay must be RGBA")
    out: list[Image.Image] = []
    for img in images:
        base = _uncrop(img, dest_size=(plan.overlay.width, plan.overlay.height), paste_to=plan.paste_to)
        base.alpha_composite(plan.overlay)
        out.append(base.convert("RGB"))
    return out


__all__ = [
    "ALLOWED_INPAINT_MODES",
    "compute_mask_connected_component_bboxes",
    "InpaintFullResPlan",
    "LatentMaskEnforcer",
    "MaskEnforcerHooks",
    "INPAINT_MODE_PER_STEP_BLEND",
    "INPAINT_MODE_POST_SAMPLE_BLEND",
    "MaskedImg2ImgBundle",
    "apply_inpaint_full_res_composite",
    "prepare_masked_img2img_bundle",
    "resolve_mask_enforcer_hooks",
]
