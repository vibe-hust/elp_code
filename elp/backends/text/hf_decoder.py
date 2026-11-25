"""Reversible non-overlapping decoder loop patch, transactional KV rollback, and HF LLM adapter."""
from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager, nullcontext
import inspect
import time
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor, nn

from elp.core.diagnostics import DiagnosticCollector, ProbeSpec
from elp.core.program import LoopProgram, invoke_loop_program, load_program_callable
from elp.modalities.text.strategies import TextLoopConfig, iterate_window

from ..base import BackboneAdapter, BlockLoopStats, InferenceResult, resolve_block_container, resolve_dtype


class DecoderLayerAdapter:
    """Thin compatibility shim across single-tensor and legacy-tuple HF decoder layers."""

    def __init__(self, layer: nn.Module) -> None:
        params = inspect.signature(layer.forward).parameters
        self.legacy = "past_key_value" in params and "past_key_values" not in params
        self.cache_key = "past_key_value" if self.legacy else "past_key_values"
        self.tuple_output = self.legacy

    def pack(self, hidden: Tensor, kwargs: dict[str, Any], tail: Any = None) -> Any:
        if tail is not None:
            return (hidden, *tail)
        if self.legacy and kwargs.get("use_cache"):
            return (hidden, kwargs.get(self.cache_key))
        return (hidden,) if self.tuple_output else hidden

    @staticmethod
    def hidden(output: Any) -> tuple[Tensor, Any]:
        if isinstance(output, tuple):
            return output[0], output[1:]
        return output, None


def _cache_has_prefix(cache: Any, indices: Sequence[int]) -> bool:
    if cache is None:
        return False
    if hasattr(cache, "get_seq_length"):
        try:
            for idx in indices:
                if int(cache.get_seq_length(idx)) > 0:
                    return True
            return False
        except Exception:
            pass
    if isinstance(cache, dict):
        return any(idx in cache for idx in indices)
    return False


def _cache_seq_len(cache: Any) -> int:
    if cache is None or not hasattr(cache, "get_seq_length"):
        return 0
    try:
        return int(cache.get_seq_length())
    except Exception:
        return 0


@contextmanager
def preserve_dynamic_cache(
    cache: Any,
    indices: Sequence[int],
    stats: BlockLoopStats | None = None,
) -> Iterator[None]:
    """Snapshot selected KV cache layers before read-only probes and restore cleanly."""
    if cache is None:
        yield
        return

    if hasattr(cache, "layers") and isinstance(cache.layers, list):
        layers_snap = list(cache.layers)
        states = {i: dict(layers_snap[i].__dict__) for i in indices if i < len(layers_snap)}
        try:
            yield
        finally:
            for i, state in states.items():
                layers_snap[i].__dict__.clear()
                layers_snap[i].__dict__.update(state)
            cache.layers[:] = layers_snap
            if stats is not None:
                stats.kv_rollbacks += 1
        return

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        key_snap = list(cache.key_cache)
        val_snap = list(cache.value_cache)
        seen_snap = getattr(cache, "_seen_tokens", None)
        try:
            yield
        finally:
            cache.key_cache[:] = key_snap
            cache.value_cache[:] = val_snap
            if seen_snap is not None:
                cache._seen_tokens = seen_snap
            if stats is not None:
                stats.kv_rollbacks += 1
        return

    if isinstance(cache, dict):
        saved = {k: (v.clone() if isinstance(v, Tensor) else v) for k, v in cache.items()}
        try:
            yield
        finally:
            cache.clear()
            cache.update(saved)
            if stats is not None:
                stats.kv_rollbacks += 1
        return

    yield


@contextmanager
def pinned_moe_routing(layers: Sequence[nn.Module], enabled: bool) -> Iterator[None]:
    """Pin `layer.mlp.gate.forward` outputs across loop substeps within one window pass."""
    saved: list[tuple[nn.Module, Any, bool]] = []
    try:
        if enabled:
            for layer in layers:
                mlp = getattr(layer, "mlp", None)
                gate = getattr(mlp, "gate", None) if mlp is not None else None
                if gate is None:
                    continue
                had_attr = "forward" in vars(gate)
                orig = gate.forward
                saved.append((gate, orig, had_attr))

                def make_pinned(original_fn: Any):
                    captured: list[Any] = []

                    def forward(*args: Any, **kwargs: Any) -> Any:
                        if not captured:
                            captured.append(original_fn(*args, **kwargs))
                        return captured[0]

                    return forward

                gate.forward = make_pinned(orig)  # type: ignore[method-assign]
        yield
    finally:
        for gate, orig, had_attr in reversed(saved):
            if had_attr:
                gate.forward = orig
            elif "forward" in vars(gate):
                delattr(gate, "forward")


def _windows_from_program(program: LoopProgram, depth: int) -> list[TextLoopConfig]:
    meta = program.metadata
    params = program.params
    raw_windows = meta.get("windows")
    if isinstance(raw_windows, (list, tuple)) and raw_windows:
        configs: list[TextLoopConfig] = []
        for item in raw_windows:
            if isinstance(item, dict):
                s = int(item["start_layer"])
                e = int(item["end_layer"])
            else:
                s, e = int(item[0]), int(item[1])
            configs.append(TextLoopConfig(start_layer=s, end_layer=e, num_loops=int(params.get("k", 2)), strategy="naive"))
        return configs

    if "start_layer" in meta or "start_layer" in params:
        s = int(meta.get("start_layer", params.get("start_layer", 0)))
        e = int(meta.get("end_layer", params.get("end_layer", s)))
        return [TextLoopConfig(start_layer=s, end_layer=e, num_loops=int(params.get("k", 2)), strategy="naive")]

    if "layers" in meta and meta["layers"]:
        layer_ids = sorted(int(i) for i in meta["layers"])
        return [
            TextLoopConfig(
                start_layer=layer_ids[0],
                end_layer=layer_ids[-1],
                num_loops=int(params.get("k", 2)),
                strategy="naive",
            )
        ]

    start = depth // 3
    end = min(depth - 1, start + max(0, min(3, depth - start - 1)))
    return [TextLoopConfig(start_layer=start, end_layer=end, num_loops=int(params.get("k", 2)), strategy="naive")]


class HFDecoderLoopPatch:
    """Reversible multi-window inference patch supporting `TextLoopConfig` and `LoopProgram`."""

    def __init__(
        self,
        model: nn.Module,
        windows: Sequence[TextLoopConfig] | TextLoopConfig | LoopProgram,
        *,
        pin_routing: bool = False,
        kv_commit: str = "last_pass",
        fp32_state: bool = True,
    ) -> None:
        if getattr(model, "training", False):
            raise ValueError("Call model.eval() before installing HFDecoderLoopPatch")
        if kv_commit not in {"last_pass", "none"}:
            raise ValueError(f"kv_commit must be 'last_pass' or 'none', got {kv_commit!r}")

        self.model = model
        self.layers = resolve_block_container(model, attrs=("layers", "blocks"))

        if isinstance(windows, LoopProgram):
            self.program: LoopProgram | None = windows
            self._program_fn = load_program_callable(windows.code)
            cfgs = _windows_from_program(windows, len(self.layers))
            pin_routing = bool(windows.metadata.get("pin_routing", pin_routing))
            kv_commit = str(windows.metadata.get("kv_commit", kv_commit))
            fp32_state = bool(windows.metadata.get("fp32_state", fp32_state))
        else:
            self.program = None
            self._program_fn = None
            cfgs = [windows] if isinstance(windows, TextLoopConfig) else list(windows)

        if not cfgs:
            raise ValueError("At least one TextLoopConfig window is required")
        cfgs.sort(key=lambda c: c.start_layer)
        for prev, nxt in zip(cfgs, cfgs[1:]):
            if prev.end_layer >= nxt.start_layer:
                raise ValueError("Decoder loop windows must be non-overlapping")
        if cfgs[-1].end_layer >= len(self.layers):
            raise ValueError(f"Window end_layer={cfgs[-1].end_layer} exceeds depth {len(self.layers)}")

        self.windows = tuple(cfgs)
        self.pin_routing = bool(pin_routing)
        self.kv_commit = kv_commit
        self.fp32_state = bool(fp32_state)
        self.layer_adapter = DecoderLayerAdapter(self.layers[0])
        self.stats = BlockLoopStats()
        self._memories: dict[int, dict[str, Any]] = {}
        self._saved: list[tuple[int, Any, bool]] = []
        self._passthrough_tails: dict[int, tuple[bool, Any]] = {}
        self._restored = False

    def __enter__(self) -> "HFDecoderLoopPatch":
        for win_idx, cfg in enumerate(self.windows):
            indices = list(range(cfg.start_layer, cfg.end_layer + 1))
            orig_forwards = [self.layers[idx].forward for idx in indices]
            for idx in indices:
                layer = self.layers[idx]
                self._saved.append((idx, layer.forward, "forward" in vars(layer)))

            start_layer = self.layers[cfg.start_layer]

            def entry_forward(
                hidden_states: Tensor,
                *args: Any,
                _win_idx: int = win_idx,
                _cfg: TextLoopConfig = cfg,
                _indices: list[int] = indices,
                _origs: list[Any] = orig_forwards,
                **kwargs: Any,
            ) -> Any:
                cache = kwargs.get(self.layer_adapter.cache_key)
                if cache is None:
                    cache = kwargs.get("past_key_values")
                if cache is None:
                    cache = kwargs.get("past_key_value")

                has_prefix = _cache_has_prefix(cache, _indices)
                probe_kwargs = dict(kwargs)
                if not has_prefix:
                    probe_kwargs.pop("past_key_values", None)
                    probe_kwargs.pop("past_key_value", None)
                    if "use_cache" in probe_kwargs:
                        probe_kwargs["use_cache"] = False
                    q_len = hidden_states.shape[1]
                    mask = probe_kwargs.get("attention_mask")
                    if isinstance(mask, Tensor) and mask.shape[-1] > q_len:
                        probe_kwargs["attention_mask"] = mask[..., -q_len:]

                last_tail: list[Any] = [None]
                last_was_tuple: list[bool] = [False]
                in_dtype = hidden_states.dtype

                def run_layers(state: Tensor, call_kwargs: dict[str, Any]) -> Tensor:
                    cur = state.to(in_dtype) if state.dtype != in_dtype else state
                    for fn in _origs:
                        self.stats.block_calls += 1
                        out = fn(cur, *args, **call_kwargs)
                        last_was_tuple[0] = isinstance(out, tuple)
                        cur, last_tail[0] = DecoderLayerAdapter.hidden(out)
                    return cur.float() if (self.fp32_state and cur.dtype != torch.float32) else cur

                def g(state: Tensor) -> Tensor:
                    if has_prefix:
                        with preserve_dynamic_cache(cache, _indices, self.stats):
                            return run_layers(state, kwargs)
                    return run_layers(state, probe_kwargs)

                window_layers = [self.layers[i] for i in _indices]
                with torch.no_grad(), pinned_moe_routing(window_layers, self.pin_routing):
                    self.stats.loop_invocations += 1
                    x_init = (
                        hidden_states.float()
                        if (self.fp32_state and hidden_states.dtype != torch.float32)
                        else hidden_states
                    )
                    if self._program_fn is not None and self.program is not None:
                        endpoint = invoke_loop_program(
                            self._program_fn,
                            x_init,
                            g,
                            info={
                                "window": _win_idx,
                                "start_layer": _cfg.start_layer,
                                "end_layer": _cfg.end_layer,
                                "past_length": _cache_seq_len(cache),
                                "modality": "text",
                            },
                            params=dict(self.program.params),
                            memory=self._memories.setdefault(_win_idx, {}),
                        )
                    else:
                        endpoint = iterate_window(g, x_init, _cfg)
                    endpoint_out = endpoint.to(in_dtype) if endpoint.dtype != in_dtype else endpoint
                    if cache is not None and self.kv_commit == "last_pass":
                        run_layers(endpoint_out, kwargs)
                        self.stats.kv_commits += 1

                for skip_idx in _indices[1:]:
                    self._passthrough_tails[skip_idx] = (last_was_tuple[0], last_tail[0])
                return self.layer_adapter.pack(endpoint_out, kwargs, last_tail[0])

            start_layer.forward = entry_forward  # type: ignore[method-assign]

            for idx in indices[1:]:
                tail_layer = self.layers[idx]

                def passthrough(
                    hidden_states: Tensor,
                    *args: Any,
                    _idx: int = idx,
                    _orig=tail_layer.forward,
                    **kwargs: Any,
                ) -> Any:
                    if _idx in self._passthrough_tails:
                        was_tuple, tail = self._passthrough_tails.pop(_idx)
                        if tail is not None:
                            return (hidden_states, *tail)
                        return (hidden_states,) if was_tuple else hidden_states
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
        self._passthrough_tails.clear()
        self._restored = True

    def __exit__(self, *exc: Any) -> None:
        self.restore()


class HFDecoderAdapter(BackboneAdapter):
    """Backbone adapter for unlooped causal HuggingFace decoder LLMs (e.g., Qwen3-4B)."""

    name = "qwen3"
    modality = "text"

    def load(
        self,
        model_path: str,
        *,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype | None = None,
        local_files_only: bool = True,
        **kwargs: Any,
    ) -> Any:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Loading HuggingFace text models requires `transformers`. "
                "Install `elp[text]` or pass an already instantiated nn.Module directly."
            ) from exc
        torch_dtype = resolve_dtype(dtype)
        load_kw: dict[str, Any] = {"local_files_only": local_files_only}
        if torch_dtype is not None:
            load_kw["torch_dtype"] = torch_dtype
        load_kw.update(kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)
        model.to(device=device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model._elp_tokenizer = tokenizer
        return model

    def install_program(self, model: Any, program: Any) -> AbstractContextManager[Any]:
        if program is None:
            return nullcontext()
        prog = self.resolve_program(program)
        if isinstance(prog, LoopProgram):
            return HFDecoderLoopPatch(model, prog)
        if isinstance(prog, TextLoopConfig):
            return HFDecoderLoopPatch(model, (prog,))
        if isinstance(prog, (tuple, list)) and all(isinstance(x, TextLoopConfig) for x in prog):
            return HFDecoderLoopPatch(model, prog)
        raise TypeError(f"Unsupported program type for HFDecoderAdapter: {type(program).__name__}")

    @torch.no_grad()
    def infer(
        self,
        model: Any,
        inputs: Any,
        *,
        program: Any = None,
        probe: ProbeSpec | None = None,
        max_new_tokens: int = 32,
        **kwargs: Any,
    ) -> InferenceResult:
        if isinstance(model, nn.Module):
            model.eval()
        tokenizer = kwargs.pop("tokenizer", None) or getattr(model, "_elp_tokenizer", None)

        call_inputs: dict[str, Any]
        if isinstance(inputs, str):
            if tokenizer is None:
                raise ValueError("Passing raw prompt text requires `tokenizer` or model._elp_tokenizer")
            enc = {k: v for k, v in tokenizer(inputs, return_tensors="pt").items() if k != "token_type_ids"}
            first_param = next(model.parameters(), None)
            if first_param is not None:
                enc = {k: v.to(first_param.device) if isinstance(v, Tensor) else v for k, v in enc.items()}
            call_inputs = enc
        elif isinstance(inputs, Tensor):
            call_inputs = {"input_ids": inputs}
        elif isinstance(inputs, dict):
            if "max_new_tokens" in inputs:
                max_new_tokens = int(inputs["max_new_tokens"])
            if "prompt" in inputs and "input_ids" not in inputs:
                if tokenizer is None:
                    raise ValueError("Passing {'prompt': ...} requires `tokenizer` or model._elp_tokenizer")
                enc = {
                    k: v
                    for k, v in tokenizer(inputs["prompt"], return_tensors="pt").items()
                    if k != "token_type_ids"
                }
                first_param = next(model.parameters(), None)
                if first_param is not None:
                    enc = {k: v.to(first_param.device) if isinstance(v, Tensor) else v for k, v in enc.items()}
                call_inputs = enc
            else:
                call_inputs = {k: v for k, v in inputs.items() if k != "max_new_tokens"}
        else:
            raise TypeError(f"Unsupported inputs type for HFDecoderAdapter: {type(inputs).__name__}")

        started = time.perf_counter()
        patch_ctx = self.install_program(model, program)
        diag = DiagnosticCollector(model, probe) if probe is not None else nullcontext()

        decoded_text: str | None = None
        with patch_ctx as patch, diag as collector:
            if max_new_tokens > 0 and tokenizer is not None and hasattr(model, "generate"):
                out = model.generate(**call_inputs, max_new_tokens=int(max_new_tokens), **kwargs)
                decoded_text = tokenizer.decode(out[0], skip_special_tokens=True)
            else:
                out = model(**call_inputs, **kwargs)

        latency = time.perf_counter() - started
        stats = patch.stats.to_dict() if isinstance(patch, HFDecoderLoopPatch) else {}
        diag_rows = collector.rows() if isinstance(collector, DiagnosticCollector) else []
        meta = {"text": decoded_text} if decoded_text is not None else {}
        return InferenceResult(
            output=out,
            latency_s=latency,
            step_count=int(max_new_tokens) if decoded_text is not None else 1,
            loop_stats=stats,
            diagnostics=diag_rows,
            metadata=meta,
        )


__all__ = [
    "DecoderLayerAdapter",
    "HFDecoderAdapter",
    "HFDecoderLoopPatch",
    "pinned_moe_routing",
    "preserve_dynamic_cache",
]
