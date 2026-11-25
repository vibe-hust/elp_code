"""Modality-specific reversible inference wrappers for frozen transformers."""

from .image import AdaptiveLoopedDiTSampler, CachedLoopGuidance, ImageLoopConfig, patch_image_blocks
from .text import ElasticRecurrencePatch, TextLoopConfig, WindowLoopPatch, geodesic_extrapolate, iterate_window
from .video import (
    CausalWorldModelLoopPatch,
    VideoDiffusionLoopConfig,
    VideoWorldModelLoopConfig,
    patch_video_diffusion_blocks,
)

__all__ = [
    "AdaptiveLoopedDiTSampler",
    "CachedLoopGuidance",
    "CausalWorldModelLoopPatch",
    "ElasticRecurrencePatch",
    "ImageLoopConfig",
    "TextLoopConfig",
    "VideoDiffusionLoopConfig",
    "VideoWorldModelLoopConfig",
    "WindowLoopPatch",
    "geodesic_extrapolate",
    "iterate_window",
    "patch_image_blocks",
    "patch_video_diffusion_blocks",
]
