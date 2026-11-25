"""Reversible inference wrappers for unlooped dual-window LLMs and weight-tied looped LLMs."""
from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn

from .strategies import TextLoopConfig, geodesic_extrapolate, iterate_window


def _find_decoder_layers(model: nn.Module) -> nn.ModuleList | list[nn.Module]:
    for attr in ("layers", "blocks"):
        seq = getattr(model, attr, None)
        if seq is not None and len(seq) > 0:
            return seq
    nested = getattr(model, "model", None)
    if nested is not None and nested is not model:
        return _find_decoder_layers(nested)
    raise ValueError("Could not locate decoder layer list on model")


class WindowLoopPatch:
    """Reversibly wrap one or more non-overlapping layer windows with read-only loop refinement."""

    def __init__(self, model: nn.Module, windows: Sequence[TextLoopConfig] | TextLoopConfig) -> None:
        if model.training:
            raise ValueError("Call model.eval() before applying WindowLoopPatch")
        cfgs = [windows] if isinstance(windows, TextLoopConfig) else list(windows)
        if not cfgs:
            raise ValueError("At least one TextLoopConfig window is required")
        cfgs.sort(key=lambda c: c.start_layer)
        for prev, nxt in zip(cfgs, cfgs[1:]):
            if prev.end_layer >= nxt.start_layer:
                raise ValueError("Decoder loop windows must be non-overlapping")
        self.model = model
        self.layers = _find_decoder_layers(model)
        if cfgs[-1].end_layer >= len(self.layers):
            raise ValueError(f"Window end_layer={cfgs[-1].end_layer} exceeds depth {len(self.layers)}")
        self.windows = tuple(cfgs)
        self.stats: dict[str, int] = {"window_invocations": 0, "stage_evaluations": 0}
        self._saved: list[tuple[int, Any, bool]] = []
        self._skip_returns_tuple: dict[int, bool] = {}
        self._restored = False

    def __enter__(self) -> "WindowLoopPatch":
        for cfg in self.windows:
            orig_forwards = [self.layers[idx].forward for idx in range(cfg.start_layer, cfg.end_layer + 1)]
            for idx in range(cfg.start_layer, cfg.end_layer + 1):
                layer = self.layers[idx]
                had_attr = "forward" in vars(layer)
                self._saved.append((idx, layer.forward, had_attr))

            start_layer = self.layers[cfg.start_layer]

            def entry_forward(
                hidden_states: Tensor,
                *args: Any,
                _cfg: TextLoopConfig = cfg,
                _origs: list[Any] = orig_forwards,
                **kwargs: Any,
            ) -> Any:
                # Strip KV cache flags for intermediate read-only RK/Euler probes
                probe_kwargs = {k: v for k, v in kwargs.items() if k not in {"past_key_value", "past_key_values"}}
                if "use_cache" in probe_kwargs:
                    probe_kwargs["use_cache"] = False

                tail_box: list[Any] = [None]
                returns_tuple_box = [False]

                def eval_window(state: Tensor) -> Tensor:
                    self.stats["stage_evaluations"] += 1
                    cur = state
                    for fn in _origs:
                        out = fn(cur, *args, **probe_kwargs)
                        if isinstance(out, tuple):
                            returns_tuple_box[0] = True
                            cur, tail_box[0] = out[0], out[1:]
                        else:
                            returns_tuple_box[0] = False
                            cur, tail_box[0] = out, None
                    return cur

                with torch.no_grad():
                    refined = iterate_window(eval_window, hidden_states, _cfg)
                self.stats["window_invocations"] += 1
                for skip_idx in range(_cfg.start_layer + 1, _cfg.end_layer + 1):
                    self._skip_returns_tuple[skip_idx] = returns_tuple_box[0]
                return refined if tail_box[0] is None else (refined, *tail_box[0])

            start_layer.forward = entry_forward  # type: ignore[method-assign]

            for idx in range(cfg.start_layer + 1, cfg.end_layer + 1):
                tail_layer = self.layers[idx]

                def passthrough(
                    hidden_states: Tensor,
                    *args: Any,
                    _idx: int = idx,
                    _orig=self.layers[idx].forward,
                    **kwargs: Any,
                ) -> Any:
                    if _idx in self._skip_returns_tuple:
                        ret_tuple = self._skip_returns_tuple.pop(_idx)
                        return (hidden_states,) if ret_tuple else hidden_states
                    return _orig(hidden_states, *args, **kwargs)

                tail_layer.forward = passthrough  # type: ignore[method-assign]
        return self

    def restore(self) -> None:
        if self._restored:
            return
        for idx, orig, had_attr in reversed(self._saved):
            layer = self.layers[idx]
            if had_attr:
                layer.forward = orig
            elif "forward" in vars(layer):
                delattr(layer, "forward")
        self._saved.clear()
        self._skip_returns_tuple.clear()
        self._restored = True

    def __exit__(self, *exc: Any) -> None:
        self.restore()


class ElasticRecurrencePatch:
    """Reversibly configure loop depth, KV policy, and geodesic extrapolation on looped LLMs."""

    def __init__(
        self,
        model: nn.Module,
        *,
        num_loops: int = 2,
        kv_cache_mode: str = "last_loop",
        geodesic_scale: float = 0.0,
        max_angle: float = 0.35,
    ) -> None:
        if num_loops < 1:
            raise ValueError("num_loops must be >= 1")
        if kv_cache_mode not in {"per_loop", "last_loop"}:
            raise ValueError(f"Unsupported kv_cache_mode: {kv_cache_mode!r}")
        self.model = model
        self.num_loops = int(num_loops)
        self.kv_cache_mode = kv_cache_mode
        self.geodesic_scale = float(geodesic_scale)
        self.max_angle = float(max_angle)
        self._orig_loops = getattr(getattr(model, "config", None), "n_recurrent_loops", None)
        self._orig_kv_mode = getattr(getattr(model, "config", None), "recurrent_kv_cache_mode", None)
        inner = getattr(model, "model", model)
        self._inner = inner
        self._orig_runner = getattr(inner, "_loop_runner", None)
        self._restored = False

    def __enter__(self) -> "ElasticRecurrencePatch":
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            if hasattr(cfg, "n_recurrent_loops"):
                cfg.n_recurrent_loops = self.num_loops
            if hasattr(cfg, "recurrent_kv_cache_mode"):
                cfg.recurrent_kv_cache_mode = self.kv_cache_mode

        if hasattr(self._inner, "_loop_runner"):
            scale, cap = self.geodesic_scale, self.max_angle
            k_loops = self.num_loops

            def geodesic_runner(step_fn: Callable[[Tensor, int], Tensor], hidden_states: Tensor) -> Tensor:
                prev = hidden_states
                cur = hidden_states
                for loop_idx in range(k_loops):
                    prev = cur
                    cur = step_fn(cur, loop_idx)
                if k_loops >= 2 and scale != 0.0:
                    cur = geodesic_extrapolate(prev, cur, scale=scale, max_angle=cap)
                return cur

            self._inner._loop_runner = geodesic_runner
        return self

    def restore(self) -> None:
        if self._restored:
            return
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            if self._orig_loops is not None:
                cfg.n_recurrent_loops = self._orig_loops
            if self._orig_kv_mode is not None:
                cfg.recurrent_kv_cache_mode = self._orig_kv_mode
        if hasattr(self._inner, "_loop_runner"):
            self._inner._loop_runner = self._orig_runner
        self._restored = True

    def __exit__(self, *exc: Any) -> None:
        self.restore()
