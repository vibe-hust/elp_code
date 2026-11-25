"""Unified registry mapping all Image, Text, and Video backbones to their ELP adapters."""
from __future__ import annotations

from dataclasses import dataclass

from .base import BackboneAdapter
from .image import LoopedDiTAdapter, RAEAdapter, ScaleRAEAdapter
from .text import DoryAdapter, HFDecoderAdapter
from .video import CausalWorldModelAdapter, Wan21Adapter


@dataclass(frozen=True)
class _BackboneEntry:
    adapter: BackboneAdapter
    loading: str
    upstream: str
    description: str


def _factories() -> dict[str, _BackboneEntry]:
    entries = [
        _BackboneEntry(
            ScaleRAEAdapter(),
            loading="bundled_upstream",
            upstream="Scale-RAE",
            description="Scale-RAE unlooped image diffusion (`third_party/scale_rae` or `SCALE_RAE_REPO`)",
        ),
        _BackboneEntry(
            RAEAdapter(),
            loading="bundled_upstream",
            upstream="diffusion-bench (RAE-SigLIP2-B/DDT)",
            description="RAE sparse/modality-split image diffusion (`third_party/rae` or `DIFFUSION_BENCH_REPO`)",
        ),
        _BackboneEntry(
            LoopedDiTAdapter(),
            loading="bundled_upstream",
            upstream="OpenSenseNova/Looped-DiT",
            description="Looped-DiT-B32 pretrained recurrent image diffusion (`third_party/looped_dit` or `LOOPED_DIT_REPO`)",
        ),
        _BackboneEntry(
            HFDecoderAdapter(),
            loading="transformers",
            upstream="HuggingFace AutoModelForCausalLM",
            description="Qwen3 / compatible unlooped causal decoder LLM",
        ),
        _BackboneEntry(
            DoryAdapter(),
            loading="builtin_safetensors",
            upstream="Dory1.0-4B (nGPT Looped LM)",
            description="Dory1.0-4B weight-tied looped LLM (built-in safetensors loader)",
        ),
        _BackboneEntry(
            CausalWorldModelAdapter("longlive2"),
            loading="bundled_upstream",
            upstream="NVlabs/LongLive (main)",
            description="LongLive 2.0 (5B) causal video world model (`third_party/longlive2` or `LONGLIVE2_REPO`)",
        ),
        _BackboneEntry(
            CausalWorldModelAdapter("longlive1"),
            loading="bundled_upstream",
            upstream="NVlabs/LongLive (v1.0)",
            description="LongLive 1.0 (1.3B) causal video world model (`third_party/longlive1` or `LONGLIVE1_REPO`)",
        ),
        _BackboneEntry(
            CausalWorldModelAdapter("self_forcing"),
            loading="bundled_upstream",
            upstream="guandeh17/Self-Forcing",
            description="Self-Forcing (1.3B) causal video world model (`third_party/self_forcing` or `SELF_FORCING_REPO`)",
        ),
        _BackboneEntry(
            Wan21Adapter(),
            loading="diffusers_or_bundled_upstream",
            upstream="Wan-Video/Wan2.1",
            description="Wan2.1-1.3B short-video diffusion (`diffusers` or `third_party/wan21` / `WAN21_REPO`)",
        ),
    ]
    return {entry.adapter.name: entry for entry in entries}


_ALIASES: dict[str, str] = {
    "hf_decoder": "qwen3",
    "qwen3-4b": "qwen3",
    "dory1.0": "dory",
    "looped-dit": "looped_dit",
    "scale-rae": "scale_rae",
    "self-forcing": "self_forcing",
    "wan2.1": "wan21",
}


def list_backbones() -> list[dict[str, str]]:
    """Return metadata records for all registered multimodal backbones."""
    return [
        {
            "name": entry.adapter.name,
            "modality": entry.adapter.modality,
            "loading": entry.loading,
            "upstream": entry.upstream,
            "description": entry.description,
        }
        for entry in _factories().values()
    ]


def get_adapter(name: str) -> BackboneAdapter:
    """Resolve a backbone adapter instance by canonical name or alias."""
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    catalog = _factories()
    if key not in catalog:
        known = ", ".join(sorted(catalog))
        raise KeyError(f"Unknown backbone {name!r}; registered backbones: {known}")
    return catalog[key].adapter


__all__ = [
    "get_adapter",
    "list_backbones",
]
