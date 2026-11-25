"""Image diffusion backbone adapters and reversible loop patches."""

from .looped_dit import LoopedDiTAdapter, LoopedDiTLoopPatch, LoopedDiTSampleHandle, euler_sample_with_context
from .rae import RAEAdapter, RAELoopPatch, reset_rae_model_fn, shifted_euler_sample, wrap_rae_model_fn
from .scale_rae import (
    ScaleRAEAdapter,
    ScaleRAELoopPatch,
    resolve_scale_rae_backbone,
    resolve_scale_rae_blocks,
    set_sampler_step_count,
)

__all__ = [
    "LoopedDiTAdapter",
    "LoopedDiTLoopPatch",
    "LoopedDiTSampleHandle",
    "RAEAdapter",
    "RAELoopPatch",
    "ScaleRAEAdapter",
    "ScaleRAELoopPatch",
    "euler_sample_with_context",
    "reset_rae_model_fn",
    "resolve_scale_rae_backbone",
    "resolve_scale_rae_blocks",
    "set_sampler_step_count",
    "shifted_euler_sample",
    "wrap_rae_model_fn",
]
