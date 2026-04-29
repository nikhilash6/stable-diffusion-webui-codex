"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Processing model dataclasses (txt2img/img2img) used by engine runtimes and orchestration.
Defines the stable “processing” parameter containers (hires, first-pass swap stages, refiner, IP-Adapter, SUPIR mode + common fields) and helpers
that normalize list-like inputs so per-batch runs have consistent lengths.
`CodexProcessingImg2Img` now carries masked runtime selection under `inpaint_mode`, including exact-engine SDXL accessory branches like Fooocus.

Symbols (top-level; keep in sync; no ghosts):
- `_repeat_to_length` (function): Expands/truncates a sequence to a target length (used for per-batch list normalization).
- `_require_native_refiner_selection` (function): Rejects generic swap-only selectors that must not leak into SDXL-native refiner owners.
- `SwapModelConfig` (dataclass): Typed generic model-selection owner for selector-only swap surfaces.
- `SwapStageConfig` (dataclass): Typed first-pass model-swap stage configuration (step pointer + CFG/seed + nested model selection).
- `RefinerConfig` (dataclass): Typed SDXL refiner-stage configuration (stage controls + nested model selector) with override serialization.
- `CodexHiresConfig` (dataclass): Hires configuration (target scale/steps/denoise + upscaler tile config) with override serialization.
- `CodexProcessingBase` (dataclass): Shared processing fields for image generation runs (prompt/negative/seed/steps/cfg/dims + hi-res/refiner).
- `CodexProcessingTxt2Img` (dataclass): Txt2img processing container (extends base with txt2img-specific fields).
- `CodexProcessingImg2Img` (dataclass): Img2img processing container (extends base with init image/mask/strength, SUPIR mode, and inpaint mask controls).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple
import logging
import math

from apps.backend.runtime.adapters.ip_adapter.types import IpAdapterConfig
from apps.backend.runtime.families.supir.config import SupirModeConfig
from apps.backend.runtime.vision.upscalers.specs import TileConfig, default_tile_config, tile_config_from_payload

logger = get_backend_logger(__name__)


def _repeat_to_length(values: Sequence[Any], length: int, *, default: Any) -> List[Any]:
    if length <= 0:
        return []
    if not values:
        return [default for _ in range(length)]
    result = list(values)
    if len(result) >= length:
        return result[:length]
    if len(result) == 1:
        result = result * length
    else:
        factor = math.ceil(length / len(result))
        result = (result * factor)[:length]
    if len(result) < length:
        result.extend(default for _ in range(length - len(result)))
    return result[:length]


def _swap_model_config_from_mapping(payload: Mapping[str, Any]) -> SwapModelConfig | None:
    model_raw = payload.get("model")
    model = str(model_raw).strip() if model_raw is not None else ""
    model_sha_raw = payload.get("model_sha")
    model_sha = str(model_sha_raw).strip().lower() if model_sha_raw is not None else ""
    if not model and not model_sha:
        return None

    raw_tenc_path = payload.get("tenc_path")
    tenc_path: str | Tuple[str, ...] | None = None
    if isinstance(raw_tenc_path, str):
        normalized_path = raw_tenc_path.strip()
        tenc_path = normalized_path or None
    elif isinstance(raw_tenc_path, (list, tuple)):
        normalized_paths = tuple(
            str(entry).strip()
            for entry in raw_tenc_path
            if isinstance(entry, str) and str(entry).strip()
        )
        tenc_path = normalized_paths or None

    text_encoder_override_raw = payload.get("text_encoder_override")
    text_encoder_override: Dict[str, Any] | None = None
    if isinstance(text_encoder_override_raw, Mapping):
        text_encoder_override = {str(key): value for key, value in text_encoder_override_raw.items()}

    model_format_raw = payload.get("model_format")
    model_format = str(model_format_raw).strip().lower() if isinstance(model_format_raw, str) else None
    if model_format not in {None, "checkpoint", "diffusers", "gguf"}:
        raise ValueError(f"Invalid swap-model model_format: {model_format_raw!r}.")

    zimage_variant_raw = payload.get("zimage_variant")
    zimage_variant = str(zimage_variant_raw).strip().lower() if isinstance(zimage_variant_raw, str) else None
    if zimage_variant not in {None, "turbo", "base"}:
        raise ValueError(f"Invalid swap-model zimage_variant: {zimage_variant_raw!r}.")

    vae_source_raw = payload.get("vae_source")
    vae_source = str(vae_source_raw).strip().lower() if isinstance(vae_source_raw, str) else None
    if vae_source not in {None, "built_in", "external"}:
        raise ValueError(f"Invalid swap-model vae_source: {vae_source_raw!r}.")

    checkpoint_core_only = payload.get("checkpoint_core_only")
    if checkpoint_core_only is not None and not isinstance(checkpoint_core_only, bool):
        raise ValueError("swap_model.checkpoint_core_only must be a boolean when provided.")

    vae_path_raw = payload.get("vae_path")
    vae_path = str(vae_path_raw).strip() if isinstance(vae_path_raw, str) else None

    return SwapModelConfig(
        model=model or None,
        model_sha=model_sha or None,
        checkpoint_core_only=checkpoint_core_only,
        model_format=model_format,  # type: ignore[arg-type]
        zimage_variant=zimage_variant,  # type: ignore[arg-type]
        vae_source=vae_source,  # type: ignore[arg-type]
        vae_path=vae_path or None,
        tenc_path=tenc_path,
        text_encoder_override=text_encoder_override,
    )


def _require_native_refiner_selection(selection: SwapModelConfig | None, *, context: str) -> SwapModelConfig:
    if selection is None:
        return SwapModelConfig()
    if selection.zimage_variant is not None:
        raise ValueError(f"'{context}.zimage_variant' is unsupported.")
    return selection


@dataclass
class SwapModelConfig:
    """Typed generic model-selection owner for selector-only swap surfaces."""

    model: Optional[str] = None
    model_sha: Optional[str] = None
    checkpoint_core_only: bool | None = None
    model_format: Literal["checkpoint", "diffusers", "gguf"] | None = None
    zimage_variant: Literal["turbo", "base"] | None = None
    vae_source: Literal["built_in", "external"] | None = None
    vae_path: Optional[str] = None
    tenc_path: str | Tuple[str, ...] | None = None
    text_encoder_override: Dict[str, Any] | None = None

    def is_configured(self) -> bool:
        return bool(isinstance(self.model, str) and self.model.strip())

    def require_model_ref(self, *, context: str) -> str:
        model_ref = str(self.model or "").strip()
        if not model_ref:
            raise RuntimeError(f"{context} is missing 'model'.")
        return model_ref

    def as_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.model:
            payload["model"] = self.model
        if self.model_sha:
            payload["model_sha"] = self.model_sha
        if self.checkpoint_core_only is not None:
            payload["checkpoint_core_only"] = bool(self.checkpoint_core_only)
        if self.model_format:
            payload["model_format"] = self.model_format
        if self.zimage_variant:
            payload["zimage_variant"] = self.zimage_variant
        if self.vae_source:
            payload["vae_source"] = self.vae_source
        if self.vae_path:
            payload["vae_path"] = self.vae_path
        if isinstance(self.tenc_path, tuple) and self.tenc_path:
            payload["tenc_path"] = list(self.tenc_path)
        elif isinstance(self.tenc_path, str) and self.tenc_path:
            payload["tenc_path"] = self.tenc_path
        if self.text_encoder_override:
            payload["text_encoder_override"] = dict(self.text_encoder_override)
        return payload


@dataclass
class SwapStageConfig:
    """Configuration for a first-pass mid-generation model swap stage."""

    enabled: bool = False
    swap_at_step: int = 0
    cfg: float = 7.0
    seed: int = -1
    selection: SwapModelConfig = field(default_factory=SwapModelConfig)

    def as_override(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enable": False}
        data: Dict[str, Any] = {
            "enable": True,
            "switch_at_step": int(self.swap_at_step),
            "cfg": float(self.cfg),
            "seed": int(self.seed),
        }
        data.update(_require_native_refiner_selection(self.selection, context="refiner").as_payload())
        return data


@dataclass
class RefinerConfig:
    """Configuration for an SDXL-native refiner stage."""

    enabled: bool = False
    swap_at_step: int = 0
    cfg: float = 7.0
    seed: int = -1
    selection: SwapModelConfig = field(default_factory=SwapModelConfig)

    def as_override(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enable": False}
        data: Dict[str, Any] = {
            "enable": True,
            "switch_at_step": int(self.swap_at_step),
            "cfg": float(self.cfg),
            "seed": int(self.seed),
        }
        data.update(_require_native_refiner_selection(self.selection, context="refiner").as_payload())
        return data


@dataclass
class CodexHiresConfig:
    """Configuration for hires (second-pass) rendering."""

    enabled: bool = False
    scale: float = 1.0
    denoise: float = 0.0
    upscaler: Optional[str] = None
    tile: TileConfig = field(default_factory=default_tile_config)
    second_pass_steps: int = 0
    resize_x: int = 0
    resize_y: int = 0
    prompt: str = ""
    negative_prompt: str = ""
    cfg: float = 7.0
    distilled_cfg: float = 3.5
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    swap_model: "SwapModelConfig | None" = None
    refiner: "RefinerConfig | None" = None

    def require_upscaler_id(self) -> str:
        raw_upscaler = self.upscaler
        if not isinstance(raw_upscaler, str) or not raw_upscaler.strip():
            raise ValueError(
                "Hires is enabled but 'hires.upscaler' is missing. "
                "Choose an upscaler id from GET /api/upscalers (e.g. 'latent:bicubic-aa')."
            )
        return raw_upscaler.strip()

    def resolve_target_dimensions(self, *, base_width: int, base_height: int) -> tuple[int, int]:
        resize_x = int(self.resize_x) if self.resize_x is not None else 0
        resize_y = int(self.resize_y) if self.resize_y is not None else 0
        if resize_x < 0:
            raise ValueError("Hires is enabled but 'hires.resize_x' must be >= 0 (0 means fallback to scale).")
        if resize_y < 0:
            raise ValueError("Hires is enabled but 'hires.resize_y' must be >= 0 (0 means fallback to scale).")

        scale = float(self.scale) if self.scale is not None else None
        if resize_x > 0:
            target_width = resize_x
        else:
            if scale is None or scale <= 0.0:
                raise ValueError(
                    "Hires is enabled but neither 'hires.resize_x' nor a valid positive 'hires.scale' is set. "
                    "Provide explicit dimensions or a scale."
                )
            target_width = int(base_width * scale)

        if resize_y > 0:
            target_height = resize_y
        else:
            if scale is None or scale <= 0.0:
                raise ValueError(
                    "Hires is enabled but neither 'hires.resize_y' nor a valid positive 'hires.scale' is set. "
                    "Provide explicit dimensions or a scale."
                )
            target_height = int(base_height * scale)

        return target_width, target_height

    def resolve_second_pass_steps(self, *, base_steps: int) -> int:
        second_pass_steps = int(self.second_pass_steps) if self.second_pass_steps is not None else 0
        if second_pass_steps < 0:
            raise ValueError("Hires is enabled but 'hires.steps' must be >= 0 (0 means reuse first-pass steps).")
        steps = second_pass_steps if second_pass_steps > 0 else int(base_steps)
        if steps <= 0:
            raise ValueError("Hires is enabled but resolved 'steps' must be > 0.")
        return steps

    def as_dict(self) -> Dict[str, Any]:
        result = {
            "enabled": self.enabled,
            "scale": self.scale,
            "denoise": self.denoise,
            "upscaler": self.upscaler,
            "tile": {
                "tile": int(self.tile.tile),
                "overlap": int(self.tile.overlap),
                "fallback_on_oom": bool(self.tile.fallback_on_oom),
                "min_tile": int(self.tile.min_tile),
            },
            "second_pass_steps": self.second_pass_steps,
            "resize_x": self.resize_x,
            "resize_y": self.resize_y,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "cfg": self.cfg,
            "distilled_cfg": self.distilled_cfg,
            "sampler_name": self.sampler_name,
            "scheduler": self.scheduler,
        }
        if self.swap_model is not None and self.swap_model.is_configured():
            result["swap_model"] = self.swap_model.as_payload()
        if self.refiner is not None:
            result["refiner"] = self.refiner.as_override()
        return result

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        self.enabled = bool(payload.get("enabled", self.enabled))
        self.scale = float(payload.get("scale", self.scale))
        self.denoise = float(payload.get("denoise", self.denoise))
        self.upscaler = payload.get("upscaler", self.upscaler)
        if "tile" in payload:
            self.tile = tile_config_from_payload(payload.get("tile"), context="hires.tile")
        self.second_pass_steps = int(payload.get("second_pass_steps", self.second_pass_steps))
        self.resize_x = int(payload.get("resize_x", self.resize_x))
        self.resize_y = int(payload.get("resize_y", self.resize_y))
        self.prompt = str(payload.get("prompt", self.prompt))
        self.negative_prompt = str(payload.get("negative_prompt", self.negative_prompt))
        self.cfg = float(payload.get("cfg", self.cfg))
        self.distilled_cfg = float(payload.get("distilled_cfg", self.distilled_cfg))
        self.sampler_name = payload.get("sampler_name", self.sampler_name)
        self.scheduler = payload.get("scheduler", self.scheduler)
        if "swap_model" in payload:
            swap_payload = payload["swap_model"]
            if isinstance(swap_payload, SwapModelConfig):
                self.swap_model = swap_payload if swap_payload.is_configured() else None
            elif isinstance(swap_payload, Mapping):
                self.swap_model = _swap_model_config_from_mapping(swap_payload)
        if "refiner" in payload and payload["refiner"] is not None:
            ref_payload = payload["refiner"]
            if isinstance(ref_payload, RefinerConfig):
                _require_native_refiner_selection(ref_payload.selection, context="hires.refiner")
                self.refiner = ref_payload
            elif isinstance(ref_payload, Mapping):
                self.refiner = RefinerConfig(
                    enabled=bool(ref_payload.get("enable", False)),
                    swap_at_step=int(ref_payload.get("switch_at_step", 0) or 0),
                    cfg=float(ref_payload.get("cfg", self.cfg)),
                    seed=int(ref_payload.get("seed", -1)),
                    selection=_require_native_refiner_selection(
                        _swap_model_config_from_mapping(ref_payload),
                        context="hires.refiner",
                    ),
                )


@dataclass
class CodexProcessingBase:
    """Reusable description of a generation run.

    Unlike the legacy ``modules.processing`` classes this dataclass keeps state
    lightweight and free of side effects. Higher-level orchestration fills in
    runtime-only attributes (sampler, rng, etc.) explicitly.
    """

    prompt: str = ""
    negative_prompt: str = ""
    prompts: Sequence[str] = field(default_factory=list)
    negative_prompts: Sequence[str] = field(default_factory=list)
    styles: Sequence[str] = field(default_factory=list)
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance_scale: float = 7.0
    distilled_guidance_scale: float = 3.5
    batch_size: int = 1
    iterations: int = 1
    seed: int = -1
    subseed: int = -1
    seeds: Sequence[int] = field(default_factory=list)
    subseeds: Sequence[int] = field(default_factory=list)
    subseed_strength: float = 0.0
    seed_resize_from_h: int = 0
    seed_resize_from_w: int = 0
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    user: str = "api"
    disable_extra_networks: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra_generation_params: Dict[str, Any] = field(default_factory=dict)
    override_settings: Dict[str, Any] = field(default_factory=dict)
    ip_adapter: IpAdapterConfig | None = None
    eta_noise_seed_delta: int = 0
    # Smart runtime flags (per-job effective settings; callers decide defaults).
    smart_offload: bool = False
    smart_fallback: bool = False
    smart_cache: bool = False

    # Runtime-assigned attributes (populated by orchestrator/use-cases)
    sd_model: Any = None
    sampler: Any = None
    rng: Any = None
    scripts: Any = None
    script_args: Sequence[Any] = field(default_factory=tuple)
    modified_noise: Any = None

    # Derived collections populated by ``prepare_prompt_data``
    all_prompts: List[str] = field(default_factory=list, init=False)
    all_negative_prompts: List[str] = field(default_factory=list, init=False)
    all_seeds: List[int] = field(default_factory=list, init=False)
    all_subseeds: List[int] = field(default_factory=list, init=False)
    prompts_prepared: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.prompts = list(self.prompts)
        self.negative_prompts = list(self.negative_prompts)
        self.styles = list(self.styles)
        self.seeds = list(self.seeds)
        self.subseeds = list(self.subseeds)

    @property
    def batch_total(self) -> int:
        return max(1, int(self.batch_size) * max(1, int(self.iterations)))

    @property
    def primary_prompt(self) -> str:
        if self.prompts:
            return self.prompts[0]
        return self.prompt

    @property
    def primary_negative_prompt(self) -> str:
        if self.negative_prompts:
            return self.negative_prompts[0]
        return self.negative_prompt

    def prepare_prompt_data(self) -> None:
        total = self.batch_total
        prompts = _repeat_to_length(self.prompts, total, default=self.prompt)
        negatives = _repeat_to_length(self.negative_prompts, total, default=self.negative_prompt)
        seeds = _repeat_to_length(self.seeds, total, default=self.seed)
        subseeds = _repeat_to_length(self.subseeds, total, default=self.subseed)
        self.all_prompts = prompts
        self.all_negative_prompts = negatives
        self.all_seeds = seeds
        self.all_subseeds = subseeds
        self.prompts_prepared = True

    def iteration_slice(self, iteration_index: int) -> slice:
        if iteration_index < 0:
            raise ValueError("iteration index must be non-negative")
        start = iteration_index * max(1, self.batch_size)
        end = start + max(1, self.batch_size)
        return slice(start, end)

    def get_prompts_for_iteration(self, iteration_index: int) -> Tuple[List[str], List[str]]:
        if not self.prompts_prepared:
            self.prepare_prompt_data()
        span = self.iteration_slice(iteration_index)
        return self.all_prompts[span], self.all_negative_prompts[span]

    def get_seeds_for_iteration(self, iteration_index: int) -> Tuple[List[int], List[int]]:
        if not self.prompts_prepared:
            self.prepare_prompt_data()
        span = self.iteration_slice(iteration_index)
        return self.all_seeds[span], self.all_subseeds[span]

    def iter_batches(self) -> Iterable[Tuple[int, str, str, int, int]]:
        if not self.prompts_prepared:
            self.prepare_prompt_data()
        for idx in range(self.batch_total):
            yield (
                idx,
                self.all_prompts[idx],
                self.all_negative_prompts[idx],
                self.all_seeds[idx],
                self.all_subseeds[idx],
            )

    def set_scripts(self, scripts: Any, script_args: Optional[Sequence[Any]] = None) -> None:
        self.scripts = scripts
        if script_args is not None:
            self.script_args = list(script_args)

    def update_override(self, key: str, value: Any) -> None:
        self.override_settings[str(key)] = value

    def update_extra_param(self, key: str, value: Any) -> None:
        self.extra_generation_params[str(key)] = value


@dataclass
class CodexProcessingTxt2Img(CodexProcessingBase):
    """Processing description for txt2img tasks."""

    hires: CodexHiresConfig = field(default_factory=CodexHiresConfig)
    swap_model: "SwapStageConfig | None" = None
    refiner: "RefinerConfig | None" = None
    firstpass_image: Any = None
    latent_scale_mode: Optional[Dict[str, Any]] = None
    hires_prompts: List[str] = field(default_factory=list, init=False)
    hires_negative_prompts: List[str] = field(default_factory=list, init=False)
    firstpass_use_distilled_cfg_scale: bool = False

    def enable_hires(self, *, cfg: CodexHiresConfig) -> None:
        self.hires = cfg
        if cfg.enabled:
            self.update_extra_param("Hires Distilled CFG Scale", cfg.distilled_cfg)
            return
        self.hires_prompts = []
        self.hires_negative_prompts = []

    def ensure_hires_prompts(self) -> None:
        if not self.hires.enabled:
            self.hires_prompts = []
            self.hires_negative_prompts = []
            return
        total = self.batch_total
        self.hires_prompts = _repeat_to_length(
            [self.hires.prompt] if self.hires.prompt else [],
            total,
            default=self.primary_prompt,
        )
        self.hires_negative_prompts = _repeat_to_length(
            [self.hires.negative_prompt] if self.hires.negative_prompt else [],
            total,
            default=self.primary_negative_prompt,
        )


@dataclass
class CodexProcessingImg2Img(CodexProcessingBase):
    """Processing description for img2img tasks."""

    hires: CodexHiresConfig = field(default_factory=CodexHiresConfig)
    supir: SupirModeConfig | None = None
    init_image: Any = None
    init_images: Sequence[Any] = field(default_factory=list)
    denoising_strength: float = 0.75
    image_cfg_scale: Optional[float] = None
    mask: Any = None
    per_step_blend_strength: float = 1.0
    per_step_blend_steps: int | None = None
    mask_blur: int = 4
    mask_blur_x: int = 4
    mask_blur_y: int = 4
    mask_round: bool = True
    inpainting_fill: int = 0
    inpaint_full_res_padding: int = 0
    inpainting_mask_invert: int = 0
    inpaint_mode: Optional[str] = None
    mask_region_split: bool = False
    initial_noise_multiplier: Optional[float] = None
    latent_mask: Any = None
    resize_mode: int = 0

    def enable_hires(self, cfg: CodexHiresConfig) -> None:
        self.hires = cfg

    def has_mask(self) -> bool:
        return self.mask is not None

    def set_mask(self, mask: Any) -> None:
        self.mask = mask
