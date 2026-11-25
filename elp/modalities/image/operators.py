"""Block-level ODE loop operators, reversible DiT hooks, and cached LoopGuidance."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Callable, Iterator, Literal, Sequence

import torch
from torch import Tensor, nn

OperatorKind = Literal["euler", "heun", "midpoint"]


@dataclass(frozen=True)
class ImageLoopConfig:
    """Configuration for training-free layerwise loops on unlooped image DiTs."""

    blocks: tuple[int, ...]
    num_loops: int = 2
    operator: OperatorKind = "euler"
    step_scale: float = 0.5
    start_frac: float = 0.0
    end_frac: float = 0.5
    loopguidance_weight: float = 3.0
    reference_refresh_interval: int = 1
    reuse_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.blocks or len(set(self.blocks)) != len(self.blocks) or any(b < 0 for b in self.blocks):
            raise ValueError("blocks must be non-empty distinct non-negative integers")
        if self.num_loops < 1:
            raise ValueError("num_loops must be >= 1")
        if self.operator not in {"euler", "heun", "midpoint"}:
            raise ValueError(f"Unsupported operator: {self.operator!r}")
        if not math.isfinite(self.step_scale) or self.step_scale <= 0:
            raise ValueError("step_scale must be finite and positive")
        if not (0.0 <= self.start_frac <= self.end_frac <= 1.0):
            raise ValueError("window must satisfy 0 <= start_frac <= end_frac <= 1")
        if self.reference_refresh_interval < 1:
            raise ValueError("reference_refresh_interval must be >= 1")


def _extract_tensor(result: Any) -> tuple[Tensor, Any]:
    if isinstance(result, tuple):
        return result[0], result[1:]
    return result, None


def _reassemble(tensor: Tensor, tail: Any) -> Any:
    if tail is None:
        return tensor
    return (tensor, *tail)


def euler_loop(block_fn: Callable[[Tensor], Tensor], x: Tensor, num_loops: int, step_scale: float) -> Tensor:
    """Iterated residual Euler updates: x <- x + step_scale * (B(x) - x)."""
    cur = x
    for _ in range(num_loops):
        y = block_fn(cur)
        cur = cur + step_scale * (y - cur)
    return cur


def heun_loop(block_fn: Callable[[Tensor], Tensor], x: Tensor, num_loops: int, step_scale: float) -> Tensor:
    """Explicit Heun / RK2 residual loop."""
    cur = x
    for _ in range(num_loops):
        k1 = block_fn(cur) - cur
        probe = cur + step_scale * k1
        k2 = block_fn(probe) - probe
        cur = cur + 0.5 * step_scale * (k1 + k2)
    return cur


def midpoint_loop(block_fn: Callable[[Tensor], Tensor], x: Tensor, num_loops: int, step_scale: float) -> Tensor:
    """Explicit Midpoint / RK2 residual loop."""
    cur = x
    for _ in range(num_loops):
        k1 = block_fn(cur) - cur
        mid = cur + 0.5 * step_scale * k1
        k2 = block_fn(mid) - mid
        cur = cur + step_scale * k2
    return cur


_OPERATORS: dict[OperatorKind, Callable[[Callable[[Tensor], Tensor], Tensor, int, float], Tensor]] = {
    "euler": euler_loop,
    "heun": heun_loop,
    "midpoint": midpoint_loop,
}


def loopguidance_combine(pred_ref: Tensor, pred_loop: Tensor, weight: float) -> Tensor:
    """Combine unlooped reference and looped predictions: D_ref + w * (D_loop - D_ref)."""
    return pred_ref + float(weight) * (pred_loop - pred_ref)


class CachedLoopGuidance:
    """Interval-refreshed LoopGuidance skipping reference passes via cached correction reuse."""

    def __init__(self, weight: float = 3.0, refresh_interval: int = 2, reuse_scale: float = 1.0) -> None:
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be >= 1")
        self.weight = float(weight)
        self.refresh_interval = int(refresh_interval)
        self.reuse_scale = float(reuse_scale)
        self.delta_cache: Tensor | None = None
        self.last_refresh_step: int = -10**6
        self.reference_calls: int = 0
        self.loop_calls: int = 0

    def reset(self) -> None:
        """Reset cached residual state before a new sample."""
        self.delta_cache = None
        self.last_refresh_step = -10**6
        self.reference_calls = 0
        self.loop_calls = 0

    @torch.no_grad()
    def predict(
        self,
        step_idx: int,
        *,
        pred_loop_fn: Callable[[], Tensor],
        pred_ref_fn: Callable[[], Tensor],
        active: bool = True,
    ) -> Tensor:
        """Return guided velocity estimate using either fresh D_ref or cached delta_loop."""
        if not active:
            self.reference_calls += 1
            return pred_ref_fn()

        need_refresh = self.delta_cache is None or (step_idx - self.last_refresh_step) >= self.refresh_interval
        if need_refresh:
            d_ref = pred_ref_fn()
            self.reference_calls += 1
            d_loop = pred_loop_fn()
            self.loop_calls += 1
            self.delta_cache = (d_loop - d_ref).detach()
            self.last_refresh_step = step_idx
            return d_ref + self.weight * self.delta_cache

        d_loop = pred_loop_fn()
        self.loop_calls += 1
        return d_loop + (self.weight - 1.0) * self.reuse_scale * self.delta_cache


@contextmanager
def patch_image_blocks(
    blocks: Sequence[nn.Module],
    config: ImageLoopConfig,
    step_state: dict[str, Any],
) -> Iterator[None]:
    """Reversibly wrap selected image-DiT blocks inside `[start_frac, end_frac]`."""
    loop_fn = _OPERATORS[config.operator]
    saved: list[tuple[nn.Module, Any, bool]] = []
    try:
        for idx in config.blocks:
            block = blocks[idx]
            had_attr = "forward" in vars(block)
            orig = block.forward
            saved.append((block, orig, had_attr))

            def wrapped_forward(x: Tensor, *args: Any, _orig=orig, **kwargs: Any) -> Any:
                progress = float(step_state.get("progress", 0.0))
                in_window = config.start_frac <= progress <= config.end_frac
                if step_state.get("reference_mode", False) or not in_window:
                    return _orig(x, *args, **kwargs)
                tail_box: list[Any] = [None]

                def single_call(h: Tensor) -> Tensor:
                    out, tail = _extract_tensor(_orig(h, *args, **kwargs))
                    tail_box[0] = tail
                    return out

                updated = loop_fn(single_call, x, config.num_loops, config.step_scale)
                return _reassemble(updated, tail_box[0])

            block.forward = wrapped_forward  # type: ignore[method-assign]
        yield
    finally:
        for block, orig, had_attr in reversed(saved):
            if had_attr:
                block.forward = orig
            elif "forward" in vars(block):
                delattr(block, "forward")
