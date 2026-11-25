"""Text decoder backbone adapters and reversible loop patches."""

from .dory import DoryAdapter, DoryLoopConfig, DoryLoopPatch, remap_dory_state_dict
from .dory_model import DoryConfig, DoryForCausalLM, DoryModel, DoryRecurrentCache, load_dory_model
from .hf_decoder import (
    DecoderLayerAdapter,
    HFDecoderAdapter,
    HFDecoderLoopPatch,
    pinned_moe_routing,
    preserve_dynamic_cache,
)

__all__ = [
    "DecoderLayerAdapter",
    "DoryAdapter",
    "DoryConfig",
    "DoryForCausalLM",
    "DoryLoopConfig",
    "DoryLoopPatch",
    "DoryModel",
    "DoryRecurrentCache",
    "HFDecoderAdapter",
    "HFDecoderLoopPatch",
    "load_dory_model",
    "pinned_moe_routing",
    "preserve_dynamic_cache",
    "remap_dory_state_dict",
]
