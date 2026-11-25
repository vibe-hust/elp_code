"""Reversible step-windowed block loop wrapper for short-video diffusion transformers."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Iterator, Sequence

from torch import Tensor, nn


@dataclass(frozen=True)
class VideoDiffusionLoopConfig:
    """Configuration for step-windowed residual Euler looping on video DiT blocks."""

    blocks: tuple[int, ...]
    repeats: int = 2
    strength: float = 1.0
    start_frac: float = 0.0
    end_frac: float = 0.45

    def __post_init__(self) -> None:
        if not self.blocks or len(set(self.blocks)) != len(self.blocks) or any(b < 0 for b in self.blocks):
            raise ValueError("blocks must be non-empty distinct non-negative integers")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        if not math.isfinite(self.strength) or not 0.0 <= self.start_frac <= self.end_frac <= 1.0:
            raise ValueError("Invalid loop strength or active fraction window")


@contextmanager
def patch_video_diffusion_blocks(
    blocks: Sequence[nn.Module],
    config: VideoDiffusionLoopConfig,
    context: dict[str, Any],
) -> Iterator[None]:
    """Reversibly apply residual Euler substeps to `config.blocks` inside `[start_frac, end_frac)`."""
    saved: list[tuple[nn.Module, Any, bool]] = []
    try:
        for idx in config.blocks:
            block = blocks[idx]
            had_attr = "forward" in vars(block)
            orig = block.forward
            saved.append((block, orig, had_attr))

            def forward(x: Tensor, *args: Any, _orig=orig, **kwargs: Any) -> Any:
                step_count = max(1, int(context.get("step_count", 1)))
                progress = float(context.get("step_index", 0)) / float(step_count)
                if not config.start_frac <= progress < config.end_frac:
                    return _orig(x, *args, **kwargs)
                if config.repeats == 1 and config.strength == 1.0:
                    return _orig(x, *args, **kwargs)
                scale = config.strength / float(config.repeats)
                cur = x
                tail = None
                for _ in range(config.repeats):
                    res = _orig(cur, *args, **kwargs)
                    if isinstance(res, tuple):
                        y, tail = res[0], res[1:]
                    else:
                        y, tail = res, None
                    cur = cur + scale * (y - cur)
                return cur if tail is None else (cur, *tail)

            block.forward = forward  # type: ignore[method-assign]
        yield
    finally:
        for block, orig, had_attr in reversed(saved):
            if had_attr:
                block.forward = orig
            elif "forward" in vars(block):
                delattr(block, "forward")
