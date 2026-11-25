"""Reversible loop depth, checkpoint remapping, LoopProgram & geodesic adapter for Dory1.0-4B."""
from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
import time
from typing import Any, Callable

import torch
from torch import Tensor, nn

from elp.core.diagnostics import DiagnosticCollector, ProbeSpec
from elp.core.program import LoopProgram, invoke_loop_program, load_program_callable
from elp.modalities.text.strategies import geodesic_extrapolate
from elp.modalities.text.wrapper import ElasticRecurrencePatch

from ..base import BackboneAdapter, BlockLoopStats, InferenceResult, resolve_dtype
from .dory_model import DoryRecurrentCache, load_dory_model, remap_dory_state_dict


@dataclass(frozen=True)
class DoryLoopConfig:
    """Configuration for Dory1.0-4B weight-tied elastic recurrence & geodesic extrapolation."""

    num_loops: int = 2
    recurrent_kv_cache_mode: str = "last_loop"
    geodesic_scale: float = 0.0
    max_angle: float = 0.35
    disabled: bool = False

    def __post_init__(self) -> None:
        if type(self.num_loops) is not int or self.num_loops < 1:
            raise ValueError(f"num_loops must be a positive int, got {self.num_loops!r}")
        if self.recurrent_kv_cache_mode not in {"per_loop", "last_loop"}:
            raise ValueError(f"Invalid recurrent_kv_cache_mode: {self.recurrent_kv_cache_mode!r}")


class DoryLoopPatch:
    """Reversibly install `DoryLoopConfig`, custom runner, or `LoopProgram` onto Dory1.0-4B."""

    def __init__(
        self,
        model: nn.Module,
        config: DoryLoopConfig | None = None,
        *,
        program: LoopProgram | None = None,
        loop_runner: Callable[..., Tensor] | None = None,
    ) -> None:
        if getattr(model, "training", False):
            raise ValueError("Call model.eval() before applying DoryLoopPatch")
        self.model = model
        self.program = program
        self._program_fn = load_program_callable(program.code) if program is not None else None
        if config is None:
            params = program.params if program is not None else {}
            meta = program.metadata if program is not None else {}
            config = DoryLoopConfig(
                num_loops=int(params.get("k", meta.get("num_loops", 2))),
                recurrent_kv_cache_mode=str(params.get("kv_cache_mode", meta.get("kv_cache_mode", "last_loop"))),
                geodesic_scale=float(params.get("geodesic_scale", 0.0)),
                max_angle=float(params.get("max_angle", 0.35)),
            )
        self.config = config
        self.custom_runner = loop_runner
        self.stats = BlockLoopStats()
        self._memories: dict[int, dict[str, Any]] = {}
        self._orig_n_loops = getattr(getattr(model, "config", None), "n_recurrent_loops", None)
        self._orig_kv_mode = getattr(getattr(model, "config", None), "recurrent_kv_cache_mode", None)
        self._inner = getattr(model, "model", model)
        self._has_runner_attr = hasattr(self._inner, "_loop_runner")
        self._orig_runner = getattr(self._inner, "_loop_runner", None)
        self._restored = False

    def __enter__(self) -> "DoryLoopPatch":
        if self.config.disabled:
            return self
        cfg = self.config
        stats = self.stats
        model_cfg = getattr(self.model, "config", None)
        if model_cfg is not None:
            if hasattr(model_cfg, "n_recurrent_loops"):
                model_cfg.n_recurrent_loops = cfg.num_loops
            if hasattr(model_cfg, "recurrent_kv_cache_mode"):
                model_cfg.recurrent_kv_cache_mode = cfg.recurrent_kv_cache_mode

        if self._has_runner_attr:
            def tracked_runner(*args: Any, **kwargs: Any) -> Tensor:
                if self.custom_runner is not None:
                    return self.custom_runner(*args, **kwargs)
                if "model" in kwargs and "hidden_states" in kwargs:
                    return self._run_real_dory_loop(**kwargs)
                if len(args) == 2 and callable(args[0]):
                    step_fn, hidden_states = args
                    stats.loop_invocations += 1
                    if self._program_fn is not None and self.program is not None:
                        idx_box = [0]

                        def evaluate(h: Tensor) -> Tensor:
                            i = idx_box[0]
                            idx_box[0] += 1
                            stats.block_calls += 1
                            return step_fn(h, i)

                        return invoke_loop_program(
                            self._program_fn,
                            hidden_states,
                            evaluate,
                            info={"modality": "text", "backbone": "dory"},
                            params=dict(self.program.params),
                            memory=self._memories.setdefault(0, {}),
                        )
                    prev = hidden_states
                    cur = hidden_states
                    for loop_idx in range(cfg.num_loops):
                        prev = cur
                        stats.block_calls += 1
                        cur = step_fn(cur, loop_idx)
                    if cfg.num_loops >= 2 and cfg.geodesic_scale != 0.0:
                        cur = geodesic_extrapolate(prev, cur, scale=cfg.geodesic_scale, max_angle=cfg.max_angle)
                    return cur
                raise TypeError("Unsupported _loop_runner invocation signature")

            self._inner._loop_runner = tracked_runner
        return self

    def _run_real_dory_loop(
        self,
        *,
        model: nn.Module,
        hidden_states: Tensor,
        input_hidden_states: Tensor,
        recurrent_layers: Any,
        layer_inputs: dict[str, Any],
        past_key_values: Any = None,
    ) -> Tensor:
        cfg = self.config
        stats = self.stats
        stats.loop_invocations += 1

        if self._program_fn is not None and self.program is not None:
            pass_counter = [0]

            def evaluate(h: Tensor) -> Tensor:
                idx = pass_counter[0]
                pass_counter[0] += 1
                cur_h = model.norm(h) + input_hidden_states
                kv_snap = past_key_values.snapshot() if isinstance(past_key_values, DoryRecurrentCache) else None
                for layer in recurrent_layers:
                    stats.block_calls += 1
                    cur_h = layer(
                        cur_h,
                        past_key_values=past_key_values,
                        loop_idx=idx,
                        is_last_loop=False,
                        **layer_inputs[layer.layer_type],
                    )
                if kv_snap is not None:
                    past_key_values.restore(kv_snap)
                    stats.kv_rollbacks += 1
                return cur_h

            endpoint = invoke_loop_program(
                self._program_fn,
                hidden_states,
                evaluate,
                info={"modality": "text", "backbone": "dory", "recurrent_depth": len(recurrent_layers)},
                params=dict(self.program.params),
                memory=self._memories.setdefault(0, {}),
            )
            if past_key_values is not None:
                commit_h = model.norm(endpoint) + input_hidden_states
                for layer in recurrent_layers:
                    stats.block_calls += 1
                    commit_h = layer(
                        commit_h,
                        past_key_values=past_key_values,
                        loop_idx=max(0, pass_counter[0] - 1),
                        is_last_loop=True,
                        **layer_inputs[layer.layer_type],
                    )
                stats.kv_commits += 1
            return endpoint

        prev = hidden_states
        cur = hidden_states
        for loop_idx in range(cfg.num_loops):
            prev = cur
            cur = model.norm(cur) + input_hidden_states
            is_last = loop_idx == (cfg.num_loops - 1)
            for layer in recurrent_layers:
                stats.block_calls += 1
                cur = layer(
                    cur,
                    past_key_values=past_key_values,
                    loop_idx=loop_idx,
                    is_last_loop=is_last,
                    **layer_inputs[layer.layer_type],
                )
            if past_key_values is not None and is_last:
                stats.kv_commits += 1
        if cfg.num_loops >= 2 and cfg.geodesic_scale != 0.0:
            cur = geodesic_extrapolate(prev, cur, scale=cfg.geodesic_scale, max_angle=cfg.max_angle)
        return cur

    def restore(self) -> None:
        if self._restored:
            return
        model_cfg = getattr(self.model, "config", None)
        if model_cfg is not None:
            if self._orig_n_loops is not None and hasattr(model_cfg, "n_recurrent_loops"):
                model_cfg.n_recurrent_loops = self._orig_n_loops
            if self._orig_kv_mode is not None and hasattr(model_cfg, "recurrent_kv_cache_mode"):
                model_cfg.recurrent_kv_cache_mode = self._orig_kv_mode
        if self._has_runner_attr:
            self._inner._loop_runner = self._orig_runner
        self._restored = True

    def __exit__(self, *exc: Any) -> None:
        self.restore()


class DoryAdapter(BackboneAdapter):
    """Backbone adapter for weight-tied looped Dory1.0-4B language models."""

    name = "dory"
    modality = "text"

    def load(
        self,
        model_path: str,
        *,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype | None = torch.bfloat16,
        local_files_only: bool = True,
        n_recurrent_loops: int | None = None,
        recurrent_kv_cache_mode: str = "last_loop",
        **kwargs: Any,
    ) -> Any:
        del kwargs
        try:
            import safetensors  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Loading Dory checkpoints requires `transformers` and `safetensors` (`elp[text]`). "
                "Alternatively construct `DoryForCausalLM` directly in PyTorch."
            ) from exc
        torch_dtype = resolve_dtype(dtype) or torch.bfloat16
        model, tokenizer = load_dory_model(
            model_path,
            device=device,
            dtype=torch_dtype,
            n_recurrent_loops=n_recurrent_loops,
            recurrent_kv_cache_mode=recurrent_kv_cache_mode,
            local_files_only=local_files_only,
        )
        model._elp_tokenizer = tokenizer
        return model

    def install_program(self, model: Any, program: Any) -> AbstractContextManager[Any]:
        if program is None:
            return nullcontext()
        prog = self.resolve_program(program)
        if isinstance(prog, LoopProgram):
            return DoryLoopPatch(model, program=prog)
        if isinstance(prog, ElasticRecurrencePatch):
            cfg = DoryLoopConfig(
                num_loops=prog.num_loops,
                recurrent_kv_cache_mode=prog.kv_cache_mode,
                geodesic_scale=prog.geodesic_scale,
                max_angle=prog.max_angle,
            )
            return DoryLoopPatch(model, cfg)
        if isinstance(prog, DoryLoopConfig):
            return DoryLoopPatch(model, prog)
        if isinstance(prog, dict):
            normalized = dict(prog)
            if "kv_cache_mode" in normalized and "recurrent_kv_cache_mode" not in normalized:
                normalized["recurrent_kv_cache_mode"] = normalized.pop("kv_cache_mode")
            return DoryLoopPatch(model, DoryLoopConfig(**normalized))
        raise TypeError(f"Unsupported program type for DoryAdapter: {type(program).__name__}")

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

        def encode_prompt(prompt: str) -> dict[str, Any]:
            if tokenizer is None:
                raise ValueError("Passing raw prompt text requires `tokenizer` or model._elp_tokenizer")
            enc = tokenizer(prompt, return_tensors="pt")
            enc_dict = {k: v for k, v in enc.items() if k != "token_type_ids"}
            first_param = next(model.parameters(), None)
            if first_param is not None:
                enc_dict = {
                    k: v.to(first_param.device) if isinstance(v, Tensor) else v
                    for k, v in enc_dict.items()
                }
            return enc_dict

        if isinstance(inputs, str):
            call_inputs = encode_prompt(inputs)
        elif isinstance(inputs, Tensor):
            call_inputs = {"input_ids": inputs}
        elif isinstance(inputs, dict):
            if "max_new_tokens" in inputs:
                max_new_tokens = int(inputs["max_new_tokens"])
            if "prompt" in inputs and "input_ids" not in inputs:
                call_inputs = encode_prompt(str(inputs["prompt"]))
            else:
                call_inputs = {k: v for k, v in inputs.items() if k != "max_new_tokens"}
        else:
            raise TypeError(f"Unsupported inputs type for DoryAdapter: {type(inputs).__name__}")

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
        stats = patch.stats.to_dict() if isinstance(patch, DoryLoopPatch) else {}
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
    "DoryAdapter",
    "DoryLoopConfig",
    "DoryLoopPatch",
    "remap_dory_state_dict",
]
