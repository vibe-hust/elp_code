"""Read-only built-in and programmable runtime diagnostics via PyTorch hooks (Eq. 3)."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
import math
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .program import load_program_callable, validate_program_source


SUPPORTED_FEATURES = frozenset({"activation", "residual", "alignment", "repeat", "cache"})


@dataclass(frozen=True)
class ProbeSpec:
    """Specify layers, built-in features, and optional programmable `collect(...)` source."""

    layers: tuple[int, ...]
    features: tuple[str, ...] = ("activation", "residual", "alignment", "repeat")
    max_tokens: int = 256
    collector_code: str | None = None

    def __post_init__(self) -> None:
        if not self.layers or any(type(idx) is not int or idx < 0 for idx in self.layers):
            raise ValueError("layers must be non-empty non-negative integers")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must be distinct")
        if not set(self.features) <= SUPPORTED_FEATURES:
            raise ValueError(f"Unsupported diagnostic features: {self.features!r}")
        if self.collector_code is None and not self.features:
            raise ValueError("Specify at least one diagnostic feature or provide `collector_code`")
        if self.collector_code is not None:
            validate_program_source(self.collector_code, required_entrypoint="collect")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")


def _resolve_block_list(model: nn.Module) -> nn.ModuleList | list[nn.Module]:
    for attr in ("blocks", "layers", "dit_blocks"):
        candidate = getattr(model, attr, None)
        if candidate is not None and len(candidate) > 0:
            return candidate
    for nested_attr in ("transformer", "diff_head", "generator", "model"):
        nested = getattr(model, nested_attr, None)
        if nested is not None and isinstance(nested, nn.Module) and nested is not model:
            try:
                return _resolve_block_list(nested)
            except ValueError:
                continue
    raise ValueError("Could not locate transformer block list (`blocks` or `layers`) on model")


def _sample_flat(tensor: Tensor, max_tokens: int) -> Tensor:
    t = tensor.detach().float()
    if t.ndim >= 3:
        stride = max(1, (t.shape[1] + max_tokens - 1) // max_tokens)
        return t[:, ::stride, ...].flatten()
    if t.ndim == 2:
        stride = max(1, (t.shape[0] + max_tokens - 1) // max_tokens)
        return t[::stride, ...].flatten()
    return t.flatten()


def _coerce_custom_metrics(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise TypeError("Custom collector `collect(...)` must return a dict of metric names to floats")
    if len(raw) > 64:
        raise ValueError("Custom collector returned more than 64 metric keys")
    out: dict[str, float] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not k:
            raise ValueError(f"Custom collector metric key must be a non-empty str, got {k!r}")
        if isinstance(v, Tensor):
            if v.numel() != 1:
                raise ValueError(f"Custom collector tensor metric {k!r} must be scalar")
            val = float(v.detach().item())
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            val = float(v)
        else:
            raise TypeError(f"Custom collector metric {k!r} must be numeric, got {type(v).__name__}")
        if not math.isfinite(val):
            raise ValueError(f"Custom collector metric {k!r} is non-finite: {val!r}")
        out[k] = val
    return out


class DiagnosticCollector:
    """Attach temporary read-only hooks measuring built-in stats and programmable `collect(...)`."""

    def __init__(self, model: nn.Module, spec: ProbeSpec) -> None:
        self.model = model
        self.spec = spec
        self.blocks = _resolve_block_list(model)
        if max(spec.layers) >= len(self.blocks):
            raise ValueError(f"Layer index {max(spec.layers)} out of bounds for depth {len(self.blocks)}")
        self._custom_collector: Callable[..., Any] | None = (
            load_program_callable(spec.collector_code, entrypoint="collect")
            if spec.collector_code is not None
            else None
        )
        self._stack: ExitStack | None = None
        self._repeats: Counter[int] = Counter()
        self._previous: dict[int, Tensor] = {}
        self._stats: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(dict)

    def reset_pass_state(self) -> None:
        """Reset per-layer pass counters (called automatically at model entry)."""
        self._repeats.clear()
        self._previous.clear()

    def _accumulate(self, key: tuple[int, int], metrics: dict[str, float]) -> None:
        bucket = self._stats[key]
        for name, value in metrics.items():
            entry = bucket.setdefault(name, [0.0, 0.0])
            entry[0] += float(value)
            entry[1] += 1.0

    def _on_block_forward(self, layer_idx: int):
        features = set(self.spec.features)
        custom_fn = self._custom_collector

        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
            if not args or not isinstance(args[0], Tensor):
                return
            source = args[0]
            out = result[0] if isinstance(result, tuple) else result
            cache_update = (
                result[1]
                if isinstance(result, tuple) and len(result) > 1 and isinstance(result[1], tuple) and len(result[1]) >= 2
                else None
            )
            if not isinstance(out, Tensor) or out.shape != source.shape:
                return

            self._repeats[layer_idx] += 1
            pass_idx = self._repeats[layer_idx]
            metrics: dict[str, float] = {}

            if features - {"cache"}:
                x = _sample_flat(source, self.spec.max_tokens)
                y = _sample_flat(out, self.spec.max_tokens)
                r = y - x
                x_rms = x.square().mean().sqrt()
                y_rms = y.square().mean().sqrt()
                r_rms = r.square().mean().sqrt()
                if "activation" in features:
                    metrics["input_rms"] = float(x_rms)
                    metrics["output_rms"] = float(y_rms)
                if "residual" in features:
                    metrics["residual_rms"] = float(r_rms)
                    metrics["residual_over_input_rms"] = float(r_rms / x_rms.clamp_min(1e-12))
                if "alignment" in features:
                    metrics["input_residual_cosine"] = float(F.cosine_similarity(x, r, dim=0))
                if "repeat" in features:
                    if layer_idx in self._previous:
                        prev = self._previous[layer_idx]
                        metrics["previous_pass_residual_cosine"] = float(F.cosine_similarity(prev, r, dim=0))
                        metrics["residual_change_over_previous_rms"] = float(
                            (r - prev).square().mean().sqrt() / prev.square().mean().sqrt().clamp_min(1e-12)
                        )
                    self._previous[layer_idx] = r

            if "cache" in features:
                kv = kwargs.get("kv_cache")
                if isinstance(kv, dict) and "global_end_index" in kv:
                    val = kv["global_end_index"]
                    metrics["kv_read_global_end"] = float(val.item() if isinstance(val, Tensor) else val)
                if cache_update is not None:
                    metrics["kv_proposed_global_end"] = float(cache_update[0])
                    metrics["kv_proposed_local_end"] = float(cache_update[1])

            if custom_fn is not None:
                x_ro = source.detach().clone()
                y_ro = out.detach().clone()
                raw_custom = custom_fn(
                    layer_idx,
                    pass_idx,
                    x_ro,
                    y_ro,
                    {"max_tokens": self.spec.max_tokens},
                )
                metrics.update(_coerce_custom_metrics(raw_custom))

            if metrics:
                self._accumulate((layer_idx, pass_idx), metrics)

        return hook

    def __enter__(self) -> "DiagnosticCollector":
        self._stack = ExitStack()
        self._stack.callback(self.model.register_forward_pre_hook(lambda _m, _a: self.reset_pass_state()).remove)
        for layer_idx in self.spec.layers:
            handle = self.blocks[layer_idx].register_forward_hook(
                self._on_block_forward(layer_idx),
                with_kwargs=True,
            )
            self._stack.callback(handle.remove)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def summary(self) -> dict[tuple[int, int], dict[str, float]]:
        """Return mean observed metrics indexed by (layer_idx, pass_idx)."""
        return {
            key: {name: total / count for name, (total, count) in metrics.items()}
            for key, metrics in sorted(self._stats.items())
        }

    def rows(self) -> list[dict[str, Any]]:
        """Return JSON-serializable diagnostic summary rows."""
        return [
            {"layer": layer, "pass": pass_idx, "metrics": metrics}
            for (layer, pass_idx), metrics in self.summary().items()
        ]
