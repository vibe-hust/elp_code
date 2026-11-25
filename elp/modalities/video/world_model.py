"""Causal autoregressive video world-model recurrence with final-pass KV commits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import torch
from torch import Tensor, nn

LoopKind = Literal["layerwise", "windowwise"]


@dataclass(frozen=True)
class VideoWorldModelLoopConfig:
    """Layerwise or contiguous windowwise Euler recurrence with deferred KV updates."""

    kind: LoopKind = "windowwise"
    start_layer: int = 11
    end_layer: int = 18
    k: int = 6
    active_steps: tuple[int, ...] = (0, 1, 2)
    active_chunks: tuple[int, ...] = tuple(range(12))

    def __post_init__(self) -> None:
        if self.kind not in {"layerwise", "windowwise"}:
            raise ValueError(f"Unsupported world-model loop kind: {self.kind!r}")
        if self.k < 2:
            raise ValueError("k must be >= 2")
        if not 0 <= self.start_layer <= self.end_layer:
            raise ValueError("start_layer and end_layer must satisfy 0 <= start <= end")
        if self.kind == "windowwise" and self.start_layer == self.end_layer:
            raise ValueError("windowwise loop must span at least two layers")
        if not self.active_steps or len(set(self.active_steps)) != len(self.active_steps):
            raise ValueError("active_steps must be non-empty distinct integers")
        if not self.active_chunks or len(set(self.active_chunks)) != len(self.active_chunks):
            raise ValueError("active_chunks must be non-empty distinct integers")


def expanded_layer_plan(num_layers: int, cfg: VideoWorldModelLoopConfig) -> list[int]:
    """Return the exact physical layer execution sequence for one active forward."""
    if cfg.end_layer >= num_layers:
        raise ValueError(f"end_layer={cfg.end_layer} exceeds depth {num_layers}")
    window = list(range(cfg.start_layer, cfg.end_layer + 1))
    if cfg.kind == "layerwise":
        body = [layer for layer in window for _ in range(cfg.k)]
    else:
        body = window * cfg.k
    return list(range(cfg.start_layer)) + body + list(range(cfg.end_layer + 1, num_layers))


class CausalWorldModelLoopPatch:
    """Execute middle-band unit-budget Euler loops while deferring causal KV writes to final pass."""

    def __init__(self, blocks: Sequence[nn.Module], config: VideoWorldModelLoopConfig) -> None:
        if config.end_layer >= len(blocks):
            raise ValueError(f"end_layer={config.end_layer} exceeds block count {len(blocks)}")
        self.blocks = blocks
        self.config = config
        self.last_committed_updates: dict[int, Any] = {}
        self.execution_trace: list[int] = []

    def is_active(self, step_index: int, chunk_index: int, *, is_clean_refresh: bool = False) -> bool:
        """Return whether recurrence applies at `(step_index, chunk_index)`."""
        return (
            not is_clean_refresh
            and step_index in self.config.active_steps
            and chunk_index in self.config.active_chunks
        )

    @torch.no_grad()
    def forward_blocks(
        self,
        x: Tensor,
        *,
        step_index: int,
        chunk_index: int,
        is_clean_refresh: bool = False,
        kv_cache: dict[int, Any] | None = None,
        block_kwargs_fn: Callable[[int], dict[str, Any]] | None = None,
    ) -> Tensor:
        """Run all physical blocks with optional recurrence and final-pass KV commit."""
        cfg = self.config
        num_layers = len(self.blocks)
        committed: dict[int, Any] = {}
        self.execution_trace = []

        def run_one(layer_idx: int, state: Tensor) -> tuple[Tensor, Any]:
            self.execution_trace.append(layer_idx)
            kw = block_kwargs_fn(layer_idx) if block_kwargs_fn is not None else {}
            res = self.blocks[layer_idx](state, **kw)
            if isinstance(res, tuple):
                return res[0], res[1]
            return res, None

        if not self.is_active(step_index, chunk_index, is_clean_refresh=is_clean_refresh):
            for layer_idx in range(num_layers):
                x, upd = run_one(layer_idx, x)
                if upd is not None:
                    committed[layer_idx] = upd
        else:
            for layer_idx in range(cfg.start_layer):
                x, upd = run_one(layer_idx, x)
                if upd is not None:
                    committed[layer_idx] = upd

            if cfg.kind == "layerwise":
                for layer_idx in range(cfg.start_layer, cfg.end_layer + 1):
                    for _ in range(cfg.k):
                        y, upd = run_one(layer_idx, x)
                        x = x + (y - x) / float(cfg.k)
                        if upd is not None:
                            committed[layer_idx] = upd  # final pass overwrites prior pass
            else:
                for _ in range(cfg.k):
                    win_in = x
                    for layer_idx in range(cfg.start_layer, cfg.end_layer + 1):
                        x, upd = run_one(layer_idx, x)
                        if upd is not None:
                            committed[layer_idx] = upd  # keep last logical pass per physical layer
                    x = win_in + (x - win_in) / float(cfg.k)

            for layer_idx in range(cfg.end_layer + 1, num_layers):
                x, upd = run_one(layer_idx, x)
                if upd is not None:
                    committed[layer_idx] = upd

        self.last_committed_updates = committed
        if kv_cache is not None:
            for layer_idx, upd in committed.items():
                kv_cache[layer_idx] = upd
        return x
