"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Per-model-family runtime specification (capabilities + latent/normalization defaults).
Defines UI-facing capability flags and runtime defaults per `ModelFamily` (latent channels, prediction kind, normalization hints),
acting as the single source of truth for both backend assembly and frontend conditional UI (flow-shift is left unset when variant-specific).
Includes Anima (`ModelFamily.ANIMA`) as a flow-based image family with fixed flow shift defaults, FLUX.2 as a
32-channel flow family whose scheduler shift remains dynamic/variant-aware (`flow_shift=None` here; runtime owns the truthful bridge),
explicit WAN22 family variants (`WAN22_5B`/`WAN22_14B`/`WAN22_ANIMATE`) with independent defaults, and the
CogVideoX-Fun-backed Netflix VOID vid2vid scaffold (`ModelFamily.NETFLIX_VOID`).

Symbols (top-level; keep in sync; no ghosts):
- `FamilyCapabilities` (dataclass): UI-facing capability flags (what controls should be shown/hidden; supported/excluded samplers/schedulers).
- `FamilyRuntimeSpec` (dataclass): Runtime defaults for a `ModelFamily` (latent channels, prediction kind, and related family invariants).
- `get_family_spec` (function): Returns the `FamilyRuntimeSpec` for a known `ModelFamily` (raises on missing entries; no fallbacks).
- `get_family_spec_or_default` (function): Returns `FamilyRuntimeSpec` for a family, with an explicit caller-provided default fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from apps.backend.runtime.model_registry.specs import ModelFamily, PredictionKind


@dataclass(frozen=True)
class FamilyCapabilities:
    """UI-facing capabilities for a model family.
    
    Used by frontend to dynamically show/hide UI elements based on
    what the model actually supports.
    """
    # Conditioning support
    supports_negative_prompt: bool = True   # False for Flux/Chroma
    supports_cfg: bool = True               # False for distilled-guidance models
    
    # UI elements visibility
    shows_clip_skip: bool = True            # False for T5-only models
    shows_guidance_scale: bool = True       # False if CFG not used
    shows_steps: bool = True
    shows_seed: bool = True
    shows_width_height: bool = True
    
    # Sampler/scheduler support (empty tuple = all supported)
    supported_samplers: Tuple[str, ...] = ()
    supported_schedulers: Tuple[str, ...] = ()
    excluded_samplers: Tuple[str, ...] = ()     # Samplers to hide
    excluded_schedulers: Tuple[str, ...] = ()   # Schedulers to hide
    
    # Resolution constraints
    min_resolution: int = 256
    max_resolution: int = 4096
    resolution_step: int = 64  # Width/height must be multiple of this
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict for API."""
        return {
            "supports_negative_prompt": self.supports_negative_prompt,
            "supports_cfg": self.supports_cfg,
            "shows_clip_skip": self.shows_clip_skip,
            "shows_guidance_scale": self.shows_guidance_scale,
            "shows_steps": self.shows_steps,
            "shows_seed": self.shows_seed,
            "shows_width_height": self.shows_width_height,
            "supported_samplers": list(self.supported_samplers),
            "supported_schedulers": list(self.supported_schedulers),
            "excluded_samplers": list(self.excluded_samplers),
            "excluded_schedulers": list(self.excluded_schedulers),
            "min_resolution": self.min_resolution,
            "max_resolution": self.max_resolution,
            "resolution_step": self.resolution_step,
        }


# Pre-defined capabilities for common model types
CAPABILITIES_STANDARD = FamilyCapabilities()  # SD15, SD20, SDXL

CAPABILITIES_XL = FamilyCapabilities(
    shows_clip_skip=True,  # SDXL uses dual CLIP
)

CAPABILITIES_FLOW_NO_CFG = FamilyCapabilities(
    supports_negative_prompt=False,
    supports_cfg=False,
    shows_clip_skip=False,
    shows_guidance_scale=False,  # Flux uses distilled guidance
)

CAPABILITIES_KONTEXT = FamilyCapabilities(
    supports_negative_prompt=False,
    supports_cfg=False,
    shows_clip_skip=False,
    shows_guidance_scale=False,
    # Kontext checkpoints are trained on sizes divisible by 16; allow the UI to hit preferred sizes.
    resolution_step=16,
)

CAPABILITIES_FLOW_WITH_CFG = FamilyCapabilities(
    supports_negative_prompt=True,
    supports_cfg=True,
    shows_clip_skip=False,  # Uses T5
)

CAPABILITIES_NETFLIX_VOID = FamilyCapabilities(
    supports_negative_prompt=True,
    supports_cfg=True,
    shows_clip_skip=False,
    supported_samplers=("ddim",),
    supported_schedulers=("ddim",),
    resolution_step=8,
)

# Family-level FLUX.2 capabilities cover the supported Klein 4B/base-4B slice as a whole.
# Distilled-vs-base guidance and negative-prompt behavior are resolved from the selected checkpoint at runtime/UI seams.
CAPABILITIES_FLUX2 = FamilyCapabilities(
    supports_negative_prompt=True,
    supports_cfg=True,
    shows_clip_skip=False,
    shows_guidance_scale=True,
    supported_samplers=("euler", "dpm++ 2m"),
    supported_schedulers=("simple",),
    resolution_step=16,
)

CAPABILITIES_TURBO = FamilyCapabilities(
    supports_cfg=False,  # Turbo models don't use CFG
    shows_guidance_scale=False,
    shows_clip_skip=False,
)


@dataclass(frozen=True)
class FamilyRuntimeSpec:
    """Runtime parameters for a model family.
    
    These parameters determine how latents are generated, how the VAE
    normalizes/denormalizes, and other inference-time settings.
    """
    family: ModelFamily
    
    # Latent space configuration
    latent_channels: int  # 4 for SD/SDXL, 16 for Flux/SD3
    latent_scale_factor: int  # Downscale factor: 8 for most
    
    # VAE normalization (applied during encode/decode)
    vae_scaling_factor: float  # Multiply latent before decode
    vae_shift_factor: float  # Shift latent before decode (0 for SD, non-zero for Flux)
    
    # Conditioning
    context_dim: int  # Cross-attention dimension: 768, 2048, 4096
    uses_pooled_output: bool  # SDXL/SD3 use pooled embeddings
    uses_guidance_embed: bool  # Flux dev uses guidance embedding
    
    # Sampling defaults
    default_cfg: float  # Default CFG scale
    prediction: PredictionKind  # Prediction type (epsilon, v, flow)
    
    # Sampling parameters (new)
    default_steps: int = 20  # Default sampling steps
    flow_shift: Optional[float] = None  # Flow-match shift (None for non-flow models)
    scheduler_default: str = "karras"  # Default scheduler name (canonical)
    
    # Text encoding parameters (new)
    t5_min_length: Optional[int] = None  # T5 padding length (256 for Flux/SD3, None if no T5)
    clip_skip_default: int = 1  # Default CLIP skip
    uses_t5: bool = False  # Whether model uses T5 encoder
    
    # Architecture flags (new - replaces is_sdxl checks)
    is_xl_variant: bool = False  # True for SDXL, SD35, Flux (affects AYS tables, etc.)
    uses_adm: bool = False  # ADM/y conditioning (SDXL, SD35)
    uses_time_ids: bool = False  # SDXL-style time_ids
    
    # Resolution hints (new)
    preferred_width: int = 1024  # Default width
    preferred_height: int = 1024  # Default height
    patch_size: int = 1  # DiT patch size (2 for Flux/SD3, 1 for UNet)
    
    # Optional overrides
    sigma_max: Optional[float] = None
    sigma_min: Optional[float] = None
    
    # UI capabilities for frontend
    capabilities: FamilyCapabilities = field(default_factory=FamilyCapabilities)


# -----------------------------------------------------------------------------
# Family Runtime Specs Registry
# -----------------------------------------------------------------------------

FAMILY_RUNTIME_SPECS: Dict[ModelFamily, FamilyRuntimeSpec] = {
    ModelFamily.SD15: FamilyRuntimeSpec(
        family=ModelFamily.SD15,
        latent_channels=4,
        latent_scale_factor=8,
        vae_scaling_factor=0.18215,
        vae_shift_factor=0.0,
        context_dim=768,
        uses_pooled_output=False,
        uses_guidance_embed=False,
        default_cfg=7.5,
        prediction=PredictionKind.EPSILON,
        # New fields
        default_steps=20,
        scheduler_default="karras",
        clip_skip_default=1,
        preferred_width=512,
        preferred_height=512,
    ),
    ModelFamily.SD20: FamilyRuntimeSpec(
        family=ModelFamily.SD20,
        latent_channels=4,
        latent_scale_factor=8,
        vae_scaling_factor=0.18215,
        vae_shift_factor=0.0,
        context_dim=1024,
        uses_pooled_output=False,
        uses_guidance_embed=False,
        default_cfg=7.5,
        prediction=PredictionKind.V_PREDICTION,
        # New fields
        default_steps=20,
        scheduler_default="karras",
        clip_skip_default=1,
        preferred_width=768,
        preferred_height=768,
    ),
    ModelFamily.SDXL: FamilyRuntimeSpec(
        family=ModelFamily.SDXL,
        latent_channels=4,
        latent_scale_factor=8,
        vae_scaling_factor=0.13025,
        vae_shift_factor=0.0,
        context_dim=2048,
        uses_pooled_output=True,
        uses_guidance_embed=False,
        default_cfg=7.0,
        prediction=PredictionKind.EPSILON,
        # New fields
        default_steps=25,
        scheduler_default="karras",
        clip_skip_default=2,
        is_xl_variant=True,
        uses_adm=True,
        uses_time_ids=True,
        preferred_width=1024,
        preferred_height=1024,
    ),
    ModelFamily.SDXL_REFINER: FamilyRuntimeSpec(
        family=ModelFamily.SDXL_REFINER,
        latent_channels=4,
        latent_scale_factor=8,
        vae_scaling_factor=0.13025,
        vae_shift_factor=0.0,
        context_dim=1280,
        uses_pooled_output=True,
        uses_guidance_embed=False,
        default_cfg=5.0,
        prediction=PredictionKind.EPSILON,
        # New fields
        default_steps=20,
        scheduler_default="karras",
        clip_skip_default=2,
        is_xl_variant=True,
        uses_adm=True,
        uses_time_ids=True,
        preferred_width=1024,
        preferred_height=1024,
    ),
    ModelFamily.SD3: FamilyRuntimeSpec(
        family=ModelFamily.SD3,
        latent_channels=16,
        latent_scale_factor=8,
        vae_scaling_factor=1.5305,
        vae_shift_factor=0.0609,
        context_dim=4096,
        uses_pooled_output=True,
        uses_guidance_embed=False,
        default_cfg=5.0,
        prediction=PredictionKind.FLOW,
        # New fields
        default_steps=28,
        flow_shift=3.0,
        scheduler_default="simple",
        t5_min_length=256,
        uses_t5=True,
        is_xl_variant=True,
        uses_adm=True,
        patch_size=2,
        capabilities=CAPABILITIES_FLOW_WITH_CFG,
    ),
    ModelFamily.SD35: FamilyRuntimeSpec(
        family=ModelFamily.SD35,
        latent_channels=16,
        latent_scale_factor=8,
        vae_scaling_factor=1.5305,
        vae_shift_factor=0.0609,
        context_dim=4096,
        uses_pooled_output=True,
        uses_guidance_embed=False,
        default_cfg=4.5,
        prediction=PredictionKind.FLOW,
        # New fields
        default_steps=28,
        flow_shift=3.0,
        scheduler_default="simple",
        t5_min_length=256,
        uses_t5=True,
        is_xl_variant=True,
        uses_adm=True,
        patch_size=2,
        capabilities=CAPABILITIES_FLOW_WITH_CFG,
    ),
    ModelFamily.FLUX: FamilyRuntimeSpec(
        family=ModelFamily.FLUX,
        latent_channels=16,
        latent_scale_factor=8,
        vae_scaling_factor=0.3611,
        vae_shift_factor=0.1159,
        context_dim=4096,
        uses_pooled_output=False,
        uses_guidance_embed=True,  # Flux dev; Schnell is False
        default_cfg=1.0,
        prediction=PredictionKind.FLOW,
        # New fields
        default_steps=20,
        # NOTE: Flux uses dynamic resolution-dependent shifting (use_dynamic_shifting=true).
        # Source of truth is the diffusers scheduler_config.json (base/max shift + seq_len); do not hard-code a single value.
        flow_shift=None,
        scheduler_default="simple",
        t5_min_length=256,
        uses_t5=True,
        is_xl_variant=True,
        patch_size=2,
        capabilities=CAPABILITIES_FLOW_NO_CFG,
    ),
    ModelFamily.FLUX_KONTEXT: FamilyRuntimeSpec(
        family=ModelFamily.FLUX_KONTEXT,
        latent_channels=16,
        latent_scale_factor=8,
        vae_scaling_factor=0.3611,
        vae_shift_factor=0.1159,
        context_dim=4096,
        uses_pooled_output=False,
        uses_guidance_embed=True,
        default_cfg=1.0,
        prediction=PredictionKind.FLOW,
        # New fields
        default_steps=20,
        # NOTE: Flux uses dynamic resolution-dependent shifting (use_dynamic_shifting=true).
        flow_shift=None,
        scheduler_default="simple",
        t5_min_length=256,
        uses_t5=True,
        is_xl_variant=True,
        patch_size=2,
        capabilities=CAPABILITIES_KONTEXT,
    ),
    ModelFamily.FLUX2: FamilyRuntimeSpec(
        family=ModelFamily.FLUX2,
        latent_channels=32,
        latent_scale_factor=8,
        # AutoencoderKLFlux2 latent normalization is batch-norm based over patchified latents, so scalar process_in/out
        # must stay identity. FLUX.2 encode/decode overrides handle the truthful latent contract explicitly.
        vae_scaling_factor=1.0,
        vae_shift_factor=0.0,
        context_dim=7680,
        uses_pooled_output=False,
        uses_guidance_embed=False,
        default_cfg=4.0,
        prediction=PredictionKind.FLOW,
        default_steps=20,
        # FLUX.2 scheduler config uses dynamic shifting (base/max shift + image seq len); runtime computes the truthful bridge.
        flow_shift=None,
        scheduler_default="simple",
        clip_skip_default=1,
        uses_t5=False,
        is_xl_variant=True,
        preferred_width=1024,
        preferred_height=1024,
        patch_size=2,
        capabilities=CAPABILITIES_FLUX2,
    ),
    ModelFamily.CHROMA: FamilyRuntimeSpec(
        family=ModelFamily.CHROMA,
        latent_channels=16,
        latent_scale_factor=8,
        vae_scaling_factor=0.3611,
        vae_shift_factor=0.1159,
        context_dim=4096,
        uses_pooled_output=False,
        uses_guidance_embed=False,
        default_cfg=1.0,
        prediction=PredictionKind.FLOW,
        # New fields
        default_steps=20,
        # NOTE: Chroma follows Flux flow-shift semantics (dynamic shifting in diffusers configs).
        flow_shift=None,
        scheduler_default="simple",
        t5_min_length=256,
        uses_t5=True,
        is_xl_variant=True,
        patch_size=2,
        capabilities=CAPABILITIES_FLOW_NO_CFG,
    ),
    ModelFamily.AURA: FamilyRuntimeSpec(
        family=ModelFamily.AURA,
        latent_channels=16,
        latent_scale_factor=8,
        vae_scaling_factor=0.5,
        vae_shift_factor=0.0,
        context_dim=2048,
        uses_pooled_output=False,
        uses_guidance_embed=False,
        default_cfg=3.5,
        prediction=PredictionKind.V_PREDICTION,
        # New fields
        default_steps=20,
        scheduler_default="karras",
        patch_size=2,
    ),
    ModelFamily.WAN22_14B: FamilyRuntimeSpec(
        family=ModelFamily.WAN22_14B,
        latent_channels=16,
        latent_scale_factor=8,
        vae_scaling_factor=1.0,
        vae_shift_factor=0.0,
        context_dim=4096,
        uses_pooled_output=False,
        uses_guidance_embed=False,
        default_cfg=5.0,
        prediction=PredictionKind.FLOW,
        # New fields
        default_steps=20,
        # NOTE: WAN22 flow_shift is model-variant specific (T2V/I2V vs TI2V/Animate).
        # Source of truth is the diffusers scheduler_config.json; do not hard-code here.
        flow_shift=None,
        scheduler_default="simple",
        t5_min_length=512,  # UMT5-XXL
        uses_t5=True,
        capabilities=CAPABILITIES_FLOW_WITH_CFG,
    ),
    ModelFamily.WAN22_5B: FamilyRuntimeSpec(
        family=ModelFamily.WAN22_5B,
        latent_channels=16,
        latent_scale_factor=8,
        vae_scaling_factor=1.0,
        vae_shift_factor=0.0,
        context_dim=4096,
        uses_pooled_output=False,
        uses_guidance_embed=False,
        default_cfg=6.0,
        prediction=PredictionKind.FLOW,
        default_steps=16,
        flow_shift=None,
        scheduler_default="simple",
        t5_min_length=512,
        uses_t5=True,
        capabilities=CAPABILITIES_FLOW_WITH_CFG,
    ),
    ModelFamily.WAN22_ANIMATE: FamilyRuntimeSpec(
        family=ModelFamily.WAN22_ANIMATE,
        latent_channels=16,
        latent_scale_factor=8,
        vae_scaling_factor=1.0,
        vae_shift_factor=0.0,
        context_dim=4096,
        uses_pooled_output=False,
        uses_guidance_embed=False,
        default_cfg=5.0,
        prediction=PredictionKind.FLOW,
        default_steps=20,
        flow_shift=None,
        scheduler_default="simple",
        t5_min_length=512,
        uses_t5=True,
        capabilities=CAPABILITIES_FLOW_WITH_CFG,
    ),
    ModelFamily.NETFLIX_VOID: FamilyRuntimeSpec(
        family=ModelFamily.NETFLIX_VOID,
        latent_channels=16,
        latent_scale_factor=8,
        vae_scaling_factor=1.15258426,
        vae_shift_factor=0.0,
        context_dim=4096,
        uses_pooled_output=False,
        uses_guidance_embed=False,
        default_cfg=1.0,
        prediction=PredictionKind.EPSILON,
        default_steps=30,
        scheduler_default="ddim",
        t5_min_length=226,
        uses_t5=True,
        preferred_width=672,
        preferred_height=384,
        patch_size=2,
        capabilities=CAPABILITIES_NETFLIX_VOID,
    ),
    ModelFamily.ZIMAGE: FamilyRuntimeSpec(
        family=ModelFamily.ZIMAGE,
        latent_channels=16,
        latent_scale_factor=8,
        vae_scaling_factor=0.3611,
        vae_shift_factor=0.1159,
        context_dim=2560,
        uses_pooled_output=False,
        uses_guidance_embed=False,
        default_cfg=5.0,  # Diffusers ZImagePipeline default guidance_scale=5.0 (classic CFG).
        prediction=PredictionKind.FLOW,
        # New fields
        default_steps=9,  # Diffusers ZImagePipeline recommends 9 (≈8 effective; last dt=0)
        # NOTE: Z-Image flow_shift is variant-specific (Turbo shift=3.0, Base shift=6.0).
        # Source of truth is the diffusers scheduler_config.json; do not hard-code here.
        flow_shift=None,
        scheduler_default="simple",
        is_xl_variant=True,
        patch_size=2,
        capabilities=CAPABILITIES_FLOW_WITH_CFG,
    ),
    ModelFamily.ANIMA: FamilyRuntimeSpec(
        family=ModelFamily.ANIMA,
        latent_channels=16,
        latent_scale_factor=8,
        # Anima uses the WanVAE-style 3D-conv image VAE weights (`qwen_image_vae.safetensors`) in its reference asset bundle.
        vae_scaling_factor=1.0,
        vae_shift_factor=0.0,
        context_dim=1024,  # Qwen3 0.6B hidden size
        uses_pooled_output=False,
        uses_guidance_embed=False,
        default_cfg=4.0,
        prediction=PredictionKind.FLOW,
        # New fields
        default_steps=30,
        flow_shift=3.0,
        scheduler_default="simple",
        uses_t5=False,
        preferred_width=1024,
        preferred_height=1024,
        patch_size=2,
        capabilities=CAPABILITIES_FLOW_WITH_CFG,
    ),
}


def get_family_spec(family: ModelFamily) -> FamilyRuntimeSpec:
    """Get the runtime spec for a model family.
    
    Args:
        family: The model family enum value.
        
    Returns:
        The FamilyRuntimeSpec for this family.
        
    Raises:
        KeyError: If the family is not in the registry.
    """
    if family not in FAMILY_RUNTIME_SPECS:
        raise KeyError(f"No runtime spec defined for family {family}")
    return FAMILY_RUNTIME_SPECS[family]


def get_family_spec_or_default(
    family: ModelFamily, 
    default: Optional[ModelFamily] = ModelFamily.SDXL
) -> FamilyRuntimeSpec:
    """Get the runtime spec for a model family, with fallback.
    
    Args:
        family: The model family to look up.
        default: The fallback family if not found. If None, raises KeyError.
        
    Returns:
        The FamilyRuntimeSpec for this family or the default.
    """
    if family in FAMILY_RUNTIME_SPECS:
        return FAMILY_RUNTIME_SPECS[family]
    if default is not None and default in FAMILY_RUNTIME_SPECS:
        return FAMILY_RUNTIME_SPECS[default]
    raise KeyError(f"No runtime spec defined for family {family} and no valid default")


__all__ = [
    "FamilyCapabilities",
    "CAPABILITIES_STANDARD",
    "CAPABILITIES_XL",
    "CAPABILITIES_FLOW_NO_CFG",
    "CAPABILITIES_FLOW_WITH_CFG",
    "CAPABILITIES_FLUX2",
    "CAPABILITIES_TURBO",
    "FamilyRuntimeSpec",
    "FAMILY_RUNTIME_SPECS",
    "get_family_spec",
    "get_family_spec_or_default",
]
