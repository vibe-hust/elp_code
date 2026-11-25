"""Causal autoregressive video world-model driver (LongLive 1.0/2.0, Self-Forcing)."""
from __future__ import annotations

import ast
from contextlib import AbstractContextManager, nullcontext
import copy
import math
from pathlib import Path
import re
import time
from typing import Any, Sequence
import weakref

import torch
from torch import Tensor, nn

from elp.core.diagnostics import DiagnosticCollector, ProbeSpec
from elp.core.program import LoopProgram, invoke_loop_program, load_program_callable
from elp.modalities.video.world_model import VideoWorldModelLoopConfig

from ..base import (
    BackboneAdapter,
    BlockLoopStats,
    InferenceResult,
    activate_upstream_repo,
    resolve_block_container,
    resolve_dtype,
    resolve_upstream_repo,
)

_WORLD_MODEL_REPOS: dict[str, tuple[str, str, str, str]] = {
    "longlive2": ("LONGLIVE2_REPO", "NVlabs/LongLive (main)", "longlive2", "configs/inference.yaml"),
    "longlive1": ("LONGLIVE1_REPO", "NVlabs/LongLive (v1.0)", "longlive1", "configs/longlive_inference.yaml"),
    "self_forcing": ("SELF_FORCING_REPO", "guandeh17/Self-Forcing", "self_forcing", "configs/self_forcing_dmd.yaml"),
}


def _uses_residual_dict_contract(source: str) -> bool:
    tree = ast.parse(source)
    return any(isinstance(node, ast.Constant) and node.value == "residuals" for node in ast.walk(tree))


def _resolve_world_model_root(model_or_pipe: Any) -> Any:
    """Unwrap pipeline -> generator -> model -> Peft base model if present."""
    target = getattr(model_or_pipe, "generator", model_or_pipe)
    inner = getattr(target, "model", target)
    if hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
        inner = inner.base_model.model
    return inner


def validate_world_model_action(
    action: dict[str, Any],
    *,
    num_blocks: int = 30,
    max_chunks: int = 12,
) -> None:
    """Validate discrete step/block/chunk/component selectors and unit residual budget."""
    action_id = str(action.get("id", ""))
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", action_id):
        raise ValueError(f"Invalid world-model action id: {action_id!r}")
    steps = action["denoising_steps"]
    if not (1 <= len(steps) <= 6) or steps[0] != 1000 or any(type(s) is not int or not (1 <= s <= 1000) for s in steps):
        raise ValueError(f"Invalid denoising_steps: {steps!r}")
    if any(a <= b for a, b in zip(steps, steps[1:])):
        raise ValueError("denoising_steps must be strictly decreasing")

    spec = action.get("loop")
    if not spec:
        return
    if spec.get("total_lambda_pct", 100) != 100:
        raise ValueError("total_lambda_pct must equal 100")
    for key, bound in (("blocks", num_blocks), ("active_steps", len(steps)), ("chunks", max_chunks)):
        vals = spec[key]
        if not vals or len(vals) != len(set(vals)) or any(type(v) is not int or not (0 <= v < bound) for v in vals):
            raise ValueError(f"Invalid selector `{key}`: {vals!r}")
    components = spec.get("components", ["block"])
    if not components or len(components) != len(set(components)) or not set(components) <= {"block", "self", "cross", "ffn"}:
        raise ValueError(f"Invalid loop components: {components!r}")
    if "block" in components and len(components) != 1:
        raise ValueError("Whole-block looping cannot overlap component-level loops")


class CausalWorldModelLoopDriver:
    """Reversible world-model driver handling per-layer KV snapshot/rollback across Self-Forcing & LongLive."""

    def __init__(
        self,
        model_or_pipe: Any,
        *,
        config: VideoWorldModelLoopConfig | None = None,
        action: dict[str, Any] | None = None,
        algorithm: LoopProgram | str | None = None,
        chunk_size: int = 3,
        frame_seq_length: int | None = None,
    ) -> None:
        if config is None and action is None and not isinstance(algorithm, LoopProgram):
            raise ValueError("Specify `config`, (`action`, `algorithm`), or `LoopProgram`")

        self.target = model_or_pipe
        self.model_root = _resolve_world_model_root(model_or_pipe)
        self.blocks = resolve_block_container(self.model_root, attrs=("blocks", "layers"))
        depth = len(self.blocks)
        self.config = config
        self.chunk_size = max(1, int(getattr(model_or_pipe, "num_frame_per_block", chunk_size)))
        self.frame_seq_length = int(getattr(model_or_pipe, "frame_seq_length", frame_seq_length or 1560))
        self.context_noise = float(getattr(getattr(model_or_pipe, "args", None), "context_noise", 0.0))

        self.algorithm_fn = None
        self.algorithm_params: dict[str, Any] = {}
        self.residual_dict_contract = False

        if isinstance(algorithm, LoopProgram):
            self.algorithm_fn = load_program_callable(algorithm.code)
            self.algorithm_params = copy.deepcopy(algorithm.params)
            self.residual_dict_contract = (action is not None) or _uses_residual_dict_contract(algorithm.code)
        elif isinstance(algorithm, str):
            self.algorithm_fn = load_program_callable(algorithm)
            self.algorithm_params = copy.deepcopy((action or {}).get("params", {}))
            self.residual_dict_contract = (action is not None) or _uses_residual_dict_contract(algorithm)

        if action is not None:
            validate_world_model_action(action, num_blocks=depth, max_chunks=max(12, int(action.get("max_chunks", 12))))
        self.action = copy.deepcopy(action) if action is not None else None

        if config is not None and config.end_layer >= depth:
            raise ValueError(f"end_layer={config.end_layer} exceeds depth {depth}")

        self.phase_name = "context"
        self.current: dict[str, Any] = {
            "denoise_index": 0,
            "total_steps": len((self.action or {}).get("denoising_steps", [1000, 750, 500, 250])),
            "chunk_index": 0,
            "chunk_start": 0,
            "timestep": 1000.0,
        }
        self._manual_phase = False
        self._auto_chunk_start: int | None = None
        self._auto_timesteps: list[float] = []
        self._active_call: dict[str, Any] | None = None
        self.memories: dict[tuple[int, str], dict[str, Any]] = {}
        self._provenance: dict[int, tuple[weakref.ref[Tensor], int]] = {}
        self.stats = BlockLoopStats()
        self._saved_model_forward: tuple[Any, bool] | None = None
        self._saved: list[tuple[int, Any, bool]] = []
        self._passthrough_tails: dict[int, tuple[bool, Any]] = {}
        self._restored = False

    def begin_video(self) -> None:
        self.phase_name = "context"
        self._manual_phase = False
        self._auto_chunk_start = None
        self._auto_timesteps.clear()
        self.memories.clear()
        self._provenance.clear()

    def phase(self, name: str, denoise_index: int, chunk_start: int, timestep: float) -> None:
        if self.current.get("chunk_start") != chunk_start:
            self.memories.clear()
            self._provenance.clear()
        total_steps = len((self.action or {}).get("denoising_steps", [1000, 750, 500, 250]))
        self.phase_name = name
        self._manual_phase = True
        self.current = {
            "denoise_index": int(denoise_index),
            "total_steps": total_steps,
            "chunk_index": int(chunk_start) // self.chunk_size,
            "chunk_start": int(chunk_start),
            "timestep": float(timestep),
        }

    def _auto_update_phase(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if self._manual_phase:
            return
        t_val = kwargs.get("t") if "t" in kwargs else (args[1] if len(args) > 1 else None)
        if not isinstance(t_val, Tensor) or t_val.numel() == 0:
            return
        ts = float(t_val.detach().flatten()[0].item())
        cur_tok = int(kwargs.get("current_start", 0))
        chunk_start = cur_tok // max(1, self.frame_seq_length)
        if self._auto_chunk_start != chunk_start:
            self._auto_chunk_start = chunk_start
            self._auto_timesteps = []
            self.memories.clear()
            self._provenance.clear()

        if ts <= self.context_noise:
            self.phase_name = "context"
            denoise_idx = -1
        else:
            self.phase_name = "denoise"
            if not any(abs(prev - ts) < 1e-4 for prev in self._auto_timesteps):
                self._auto_timesteps.append(ts)
            denoise_idx = next(i for i, prev in enumerate(self._auto_timesteps) if abs(prev - ts) < 1e-4)

        total_steps = len((self.action or {}).get("denoising_steps", [1000, 750, 500, 250]))
        self.current = {
            "denoise_index": denoise_idx,
            "total_steps": max(total_steps, len(self._auto_timesteps)),
            "chunk_index": chunk_start // self.chunk_size,
            "chunk_start": chunk_start,
            "timestep": ts,
        }

    def _is_active(self, layer_idx: int) -> bool:
        if self.phase_name != "denoise":
            return False
        if self.config is not None:
            return (
                self.config.start_layer <= layer_idx <= self.config.end_layer
                and self.current["denoise_index"] in self.config.active_steps
                and self.current["chunk_index"] in self.config.active_chunks
            )
        spec = (self.action or {}).get("loop") or {}
        return bool(
            self.algorithm_fn is not None
            and layer_idx in spec.get("blocks", [])
            and self.current["denoise_index"] in spec.get("active_steps", [])
            and self.current["chunk_index"] in spec.get("chunks", [])
        )

    @staticmethod
    def _snapshot_single_kv(kv_cache: Any, x_len: int) -> Any:
        if not isinstance(kv_cache, dict):
            return None
        if all(k in kv_cache for k in ("k", "v", "local_end_index", "global_end_index")):
            k_buf = kv_cache["k"]
            if isinstance(k_buf, Tensor):
                cap = int(k_buf.shape[1]) if k_buf.ndim >= 2 else 0
                end = (
                    int(kv_cache["local_end_index"].item())
                    if isinstance(kv_cache["local_end_index"], Tensor)
                    else int(kv_cache["local_end_index"])
                )
                extra = {
                    k: (kv_cache[k].clone() if isinstance(kv_cache[k], Tensor) else copy.deepcopy(kv_cache[k]))
                    for k in ("global_end_index", "local_end_index", "pinned_start", "pinned_len")
                    if k in kv_cache
                }
                if x_len > 0 and end + x_len <= cap:
                    start = max(0, end - x_len)
                    stop = min(cap, end + x_len)
                else:
                    # Rolling eviction shifts the entire buffer in-place, so snapshot full capacity.
                    start = 0
                    stop = cap
                return (
                    "sliced",
                    start,
                    stop,
                    {k: kv_cache[k][:, start:stop].clone() for k in ("k", "v")},
                    extra,
                )
            # Quantized list-backed LongLive 2.0 cache is not mutated inside block forward anyway
            return ("none",)
        return ("dict", copy.deepcopy(kv_cache))

    def _snapshot_caches(self, indices: Sequence[int], fallback_kv: Any, x_len: int) -> Any:
        active = self._active_call
        if active is not None and isinstance(active.get("kv_cache_list"), (list, tuple)):
            kv_list = active["kv_cache_list"]
            return ("multi", {j: self._snapshot_single_kv(kv_list[j], x_len) for j in indices if j < len(kv_list)}, kv_list)
        if isinstance(fallback_kv, dict):
            return ("single", self._snapshot_single_kv(fallback_kv, x_len), fallback_kv)
        return None

    def _restore_single_kv(self, kv_cache: Any, snap: Any) -> bool:
        if kv_cache is None or snap is None:
            return False
        kind = snap[0]
        if kind == "sliced":
            _, start, end, tensors, indices = snap
            for key, tensor in tensors.items():
                kv_cache[key][:, start:end].copy_(tensor)
            for key, val in indices.items():
                if isinstance(kv_cache.get(key), Tensor) and isinstance(val, Tensor):
                    kv_cache[key].copy_(val)
                else:
                    kv_cache[key] = val
            return True
        if kind == "dict":
            kv_cache.clear()
            kv_cache.update(snap[1])
            return True
        return False

    def _restore_caches(self, snap_bundle: Any) -> None:
        if snap_bundle is None:
            return
        mode = snap_bundle[0]
        restored = False
        if mode == "multi":
            _, per_layer, kv_list = snap_bundle
            for j, snap in per_layer.items():
                if self._restore_single_kv(kv_list[j], snap):
                    restored = True
        elif mode == "single":
            _, snap, kv_cache = snap_bundle
            restored = self._restore_single_kv(kv_cache, snap)
        if restored:
            self.stats.kv_rollbacks += 1

    def _kwargs_for_layer(self, layer_idx: int, base_kwargs: dict[str, Any]) -> dict[str, Any]:
        active = self._active_call
        if active is None:
            return base_kwargs
        out = dict(base_kwargs)
        kv_list = active.get("kv_cache_list")
        if isinstance(kv_list, (list, tuple)) and layer_idx < len(kv_list):
            out["kv_cache"] = kv_list[layer_idx]
        ca_list = active.get("crossattn_cache_list")
        if isinstance(ca_list, (list, tuple)) and layer_idx < len(ca_list):
            out["crossattn_cache"] = ca_list[layer_idx]
        return out

    def _remember(self, tensor: Tensor) -> Tensor:
        self._provenance[id(tensor)] = (weakref.ref(tensor), tensor._version)
        return tensor

    def _verify_residual(self, tensor: Tensor) -> None:
        item = self._provenance.get(id(tensor))
        if item is None or item[0]() is not tensor or item[1] != tensor._version:
            raise ValueError("Residual term must be an unmodified evaluator return value")

    def _run_algorithm(self, x: Tensor, call_fn: Any, layer_idx: int, *, has_kv: bool = False) -> Tensor:
        assert self.algorithm_fn is not None
        calls = 0

        def evaluate(h: Tensor, indices: Tensor | None = None, base: Tensor | None = None) -> Tensor:
            nonlocal calls
            del indices, base
            calls += 1
            if calls > 8:
                raise ValueError("At most 8 native evaluations permitted per loop call")
            if h.device != x.device or h.dtype != x.dtype or h.shape != x.shape:
                raise ValueError("Evaluated hidden state shape/dtype/device mismatch")
            self.stats.block_calls += 1
            out = call_fn(h, allow_kv_commit=False)
            if self.residual_dict_contract:
                return self._remember(out - h)
            return out

        info = dict(self.current, component="block", layer=layer_idx)
        memory = self.memories.setdefault((layer_idx, "block"), {})
        out = invoke_loop_program(
            self.algorithm_fn,
            x,
            evaluate,
            info=info,
            params=copy.deepcopy(self.algorithm_params),
            memory=memory,
            enforce_unit_budget=self.residual_dict_contract,
            verify_residual=self._verify_residual if self.residual_dict_contract else None,
        )
        self.stats.loop_invocations += 1
        # Commit causal KV caches (and/or capture final tuple tails) at the numerical endpoint.
        _ = call_fn(out, allow_kv_commit=True)
        if has_kv:
            self.stats.kv_commits += 1
        return out

    def __enter__(self) -> "CausalWorldModelLoopDriver":
        if isinstance(self.model_root, nn.Module) and hasattr(self.model_root, "forward"):
            had_fwd = "forward" in vars(self.model_root)
            orig_model_forward = self.model_root.forward
            self._saved_model_forward = (orig_model_forward, had_fwd)

            def hooked_model_forward(*args: Any, **kwargs: Any) -> Any:
                self._auto_update_phase(args, kwargs)
                prev_active = self._active_call
                self._active_call = {
                    "kv_cache_list": kwargs.get("kv_cache"),
                    "crossattn_cache_list": kwargs.get("crossattn_cache"),
                }
                try:
                    return orig_model_forward(*args, **kwargs)
                finally:
                    self._active_call = prev_active
                    self._manual_phase = False

            self.model_root.forward = hooked_model_forward  # type: ignore[method-assign]

        if self.config is not None and self.config.kind == "windowwise":
            self._install_windowwise(self.config)
        elif self.config is not None:
            self._install_layerwise(self.config)
        else:
            spec = (self.action or {}).get("loop") or {}
            block_ids = list(spec.get("blocks", []))
            if self.action and self.action.get("kind") == "windowwise" and len(block_ids) >= 2:
                self._install_algorithm_window(block_ids)
            else:
                for layer_idx in block_ids:
                    self._install_algorithm_block(layer_idx)
        return self

    def _install_windowwise(self, cfg: VideoWorldModelLoopConfig) -> None:
        indices = list(range(cfg.start_layer, cfg.end_layer + 1))
        orig_forwards = {idx: self.blocks[idx].forward for idx in indices}
        for idx in indices:
            blk = self.blocks[idx]
            self._saved.append((idx, blk.forward, "forward" in vars(blk)))

        entry_blk = self.blocks[cfg.start_layer]

        def window_forward(x: Tensor, *args: Any, **kwargs: Any) -> Any:
            if not self._is_active(cfg.start_layer):
                self.stats.block_calls += 1
                return orig_forwards[cfg.start_layer](x, *args, **kwargs)

            kv_fallback = kwargs.get("kv_cache")
            snap = self._snapshot_caches(indices, kv_fallback, x.shape[1] if x.ndim >= 2 else 0)
            self.stats.loop_invocations += 1
            cur = x
            tails: dict[int, tuple[bool, Any]] = {}
            for pass_idx in range(cfg.k):
                is_final = pass_idx == cfg.k - 1
                win_in = cur
                for j in indices:
                    self.stats.block_calls += 1
                    blk_kw = self._kwargs_for_layer(j, kwargs)
                    res = orig_forwards[j](cur, *args, **blk_kw)
                    if isinstance(res, tuple):
                        cur = res[0]
                        if is_final:
                            tails[j] = (True, res[1:])
                    else:
                        cur = res
                        if is_final:
                            tails[j] = (False, None)
                cur = win_in + (cur - win_in) / float(cfg.k)
                if not is_final:
                    self._restore_caches(snap)

            if self._active_call is not None and isinstance(self._active_call.get("kv_cache_list"), (list, tuple)):
                self.stats.kv_commits += 1
            elif kv_fallback is not None:
                self.stats.kv_commits += 1

            for skip_idx in indices[1:]:
                if skip_idx in tails:
                    self._passthrough_tails[skip_idx] = tails[skip_idx]
            was_tup, entry_tail = tails.get(cfg.start_layer, (False, None))
            if entry_tail is not None:
                return (cur, *entry_tail)
            return (cur,) if was_tup else cur

        entry_blk.forward = window_forward  # type: ignore[method-assign]

        for idx in indices[1:]:
            blk = self.blocks[idx]

            def passthrough(x: Tensor, *args: Any, _idx=idx, _orig=orig_forwards[idx], **kwargs: Any) -> Any:
                if _idx in self._passthrough_tails:
                    was_tup, tail = self._passthrough_tails.pop(_idx)
                    if tail is not None:
                        return (x, *tail)
                    return (x,) if was_tup else x
                self.stats.block_calls += 1
                return _orig(x, *args, **kwargs)

            blk.forward = passthrough  # type: ignore[method-assign]

    def _install_layerwise(self, cfg: VideoWorldModelLoopConfig) -> None:
        for idx in range(cfg.start_layer, cfg.end_layer + 1):
            blk = self.blocks[idx]
            orig = blk.forward
            self._saved.append((idx, orig, "forward" in vars(blk)))

            def layer_forward(x: Tensor, *args: Any, _idx=idx, _orig=orig, **kwargs: Any) -> Any:
                if not self._is_active(_idx):
                    self.stats.block_calls += 1
                    return _orig(x, *args, **kwargs)
                kv_fallback = kwargs.get("kv_cache")
                snap = self._snapshot_caches((_idx,), kv_fallback, x.shape[1] if x.ndim >= 2 else 0)
                self.stats.loop_invocations += 1
                cur = x
                tail = None
                was_tup = False
                for pass_idx in range(cfg.k):
                    is_final = pass_idx == cfg.k - 1
                    self.stats.block_calls += 1
                    res = _orig(cur, *args, **kwargs)
                    if isinstance(res, tuple):
                        y, tail = res[0], res[1:]
                        was_tup = True
                    else:
                        y, tail = res, None
                        was_tup = False
                    cur = cur + (y - cur) / float(cfg.k)
                    if not is_final:
                        self._restore_caches(snap)
                if kv_fallback is not None:
                    self.stats.kv_commits += 1
                if tail is not None:
                    return (cur, *tail)
                return (cur,) if was_tup else cur

            blk.forward = layer_forward  # type: ignore[method-assign]

    def _install_algorithm_block(self, idx: int) -> None:
        blk = self.blocks[idx]
        orig = blk.forward
        self._saved.append((idx, orig, "forward" in vars(blk)))

        def algo_forward(x: Tensor, *args: Any, _idx=idx, _orig=orig, **kwargs: Any) -> Any:
            if not self._is_active(_idx):
                self.stats.block_calls += 1
                return _orig(x, *args, **kwargs)
            kv_fallback = kwargs.get("kv_cache")
            snap = self._snapshot_caches((_idx,), kv_fallback, x.shape[1] if x.ndim >= 2 else 0)
            tail_box: list[tuple[bool, Any]] = [(False, None)]

            def call_fn(h: Tensor, *, allow_kv_commit: bool) -> Tensor:
                res = _orig(h, *args, **kwargs)
                if isinstance(res, tuple):
                    out = res[0]
                    tail_box[0] = (True, res[1:])
                else:
                    out = res
                    tail_box[0] = (False, None)
                if not allow_kv_commit:
                    self._restore_caches(snap)
                return out

            has_kv = snap is not None
            updated = self._run_algorithm(x, call_fn, _idx, has_kv=has_kv)
            was_tup, tail = tail_box[0]
            if tail is not None:
                return (updated, *tail)
            return (updated,) if was_tup else updated

        blk.forward = algo_forward  # type: ignore[method-assign]

    def _install_algorithm_window(self, indices: list[int]) -> None:
        orig_forwards = {idx: self.blocks[idx].forward for idx in indices}
        for idx in indices:
            blk = self.blocks[idx]
            self._saved.append((idx, blk.forward, "forward" in vars(blk)))

        entry_idx = indices[0]

        def algo_window_forward(x: Tensor, *args: Any, **kwargs: Any) -> Any:
            if not self._is_active(entry_idx):
                self.stats.block_calls += 1
                return orig_forwards[entry_idx](x, *args, **kwargs)

            kv_fallback = kwargs.get("kv_cache")
            snap = self._snapshot_caches(indices, kv_fallback, x.shape[1] if x.ndim >= 2 else 0)
            tails: dict[int, tuple[bool, Any]] = {}

            def call_fn(h: Tensor, *, allow_kv_commit: bool) -> Tensor:
                cur = h
                for j in indices:
                    blk_kw = self._kwargs_for_layer(j, kwargs)
                    res = orig_forwards[j](cur, *args, **blk_kw)
                    if isinstance(res, tuple):
                        cur = res[0]
                        if allow_kv_commit:
                            tails[j] = (True, res[1:])
                    else:
                        cur = res
                        if allow_kv_commit:
                            tails[j] = (False, None)
                if not allow_kv_commit:
                    self._restore_caches(snap)
                return cur

            has_kv = snap is not None
            updated = self._run_algorithm(x, call_fn, entry_idx, has_kv=has_kv)
            for skip_idx in indices[1:]:
                if skip_idx in tails:
                    self._passthrough_tails[skip_idx] = tails[skip_idx]
            was_tup, entry_tail = tails.get(entry_idx, (False, None))
            if entry_tail is not None:
                return (updated, *entry_tail)
            return (updated,) if was_tup else updated

        self.blocks[entry_idx].forward = algo_window_forward  # type: ignore[method-assign]

        for idx in indices[1:]:
            blk = self.blocks[idx]

            def passthrough(x: Tensor, *args: Any, _idx=idx, _orig=orig_forwards[idx], **kwargs: Any) -> Any:
                if _idx in self._passthrough_tails:
                    was_tup, tail = self._passthrough_tails.pop(_idx)
                    if tail is not None:
                        return (x, *tail)
                    return (x,) if was_tup else x
                self.stats.block_calls += 1
                return _orig(x, *args, **kwargs)

            blk.forward = passthrough  # type: ignore[method-assign]

    def restore(self) -> None:
        if self._restored:
            return
        if self._saved_model_forward is not None:
            orig_mf, had_mf = self._saved_model_forward
            if had_mf:
                self.model_root.forward = orig_mf
            elif "forward" in vars(self.model_root):
                delattr(self.model_root, "forward")
            self._saved_model_forward = None
        for idx, orig, had_attr in reversed(self._saved):
            blk = self.blocks[idx]
            if had_attr:
                blk.forward = orig
            elif "forward" in vars(blk):
                delattr(blk, "forward")
        self._saved.clear()
        self._passthrough_tails.clear()
        self._restored = True

    def __exit__(self, *exc: Any) -> None:
        self.restore()


class CausalWorldModelAdapter(BackboneAdapter):
    """Unified adapter for causal autoregressive video world models (LongLive 1.0/2.0 & Self-Forcing)."""

    modality = "video"

    def __init__(self, name: str = "longlive2") -> None:
        self.name = name

    def load(
        self,
        model_path: str,
        *,
        upstream_repo: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype | None = torch.bfloat16,
        local_files_only: bool = True,
        **kwargs: Any,
    ) -> Any:
        del local_files_only, kwargs
        env_var, label, bundled_subdir, default_cfg_rel = _WORLD_MODEL_REPOS[self.name]
        repo = resolve_upstream_repo(upstream_repo, env_var, label, bundled_subdir=bundled_subdir)
        pkg_root = (repo / "LongLive2.0") if (self.name == "longlive2" and (repo / "LongLive2.0").is_dir()) else repo
        activate_upstream_repo(pkg_root)

        try:
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError(
                f"Loading {label} requires upstream dependencies (`omegaconf`, `diffusers`, `easydict`) installed."
            ) from exc

        cfg_candidate = Path(config_path) if config_path else (pkg_root / default_cfg_rel)
        if not cfg_candidate.is_file():
            raise FileNotFoundError(
                f"Expected inference YAML config for {label} at {cfg_candidate}; pass `config_path=...`."
            )
        cfg = OmegaConf.load(str(cfg_candidate))

        if self.name == "longlive2":
            from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline

            pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(device))
        else:
            from pipeline.causal_inference import CausalInferencePipeline

            pipe = CausalInferencePipeline(cfg, device=torch.device(device))

        ckpt_file = Path(model_path)
        if ckpt_file.is_dir():
            candidates = sorted(ckpt_file.glob("*.pt")) + sorted(ckpt_file.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"No checkpoint (.pt/.safetensors) found under {ckpt_file}")
            ckpt_file = candidates[0]
        if ckpt_file.is_file():
            state = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
            sd = state.get("generator_ema", state.get("generator", state.get("model", state)))
            pipe.generator.load_state_dict(sd, strict=False)

        torch_dtype = resolve_dtype(dtype)
        pipe.to(device=device)
        if torch_dtype is not None:
            pipe.generator.to(dtype=torch_dtype)
        pipe.eval()
        return pipe

    def install_program(self, model: Any, program: Any) -> AbstractContextManager[Any]:
        if program is None:
            return nullcontext()
        prog = self.resolve_program(program)
        if isinstance(prog, VideoWorldModelLoopConfig):
            return CausalWorldModelLoopDriver(model, config=prog)
        if isinstance(prog, dict) and "action" in prog and "algorithm" in prog:
            return CausalWorldModelLoopDriver(model, action=prog["action"], algorithm=prog["algorithm"])
        if isinstance(prog, LoopProgram):
            meta = prog.metadata
            blocks_seq = meta.get("blocks")
            if blocks_seq is None:
                root = _resolve_world_model_root(model)
                depth = len(resolve_block_container(root, attrs=("blocks", "layers")))
                blocks_seq = [depth // 2]
            action: dict[str, Any] = {
                "id": prog.id,
                "kind": meta.get("kind", "block"),
                "denoising_steps": meta.get("denoising_steps", [1000, 750, 500, 250]),
                "loop": {
                    "blocks": list(blocks_seq),
                    "active_steps": list(meta.get("active_steps", [0, 1])),
                    "chunks": list(meta.get("chunks", list(range(12)))),
                    "components": ["block"],
                    "total_lambda_pct": 100,
                },
                "params": prog.params,
            }
            return CausalWorldModelLoopDriver(model, action=action, algorithm=prog)
        raise TypeError(f"Unsupported program type for CausalWorldModelAdapter: {type(program).__name__}")

    @torch.no_grad()
    def infer(
        self,
        model: Any,
        inputs: Any,
        *,
        program: Any = None,
        probe: ProbeSpec | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        if isinstance(model, nn.Module):
            model.eval()
        started = time.perf_counter()
        patch_ctx = self.install_program(model, program)
        diag = DiagnosticCollector(_resolve_world_model_root(model), probe) if probe is not None else nullcontext()

        steps = 1
        with patch_ctx as driver, diag as collector:
            if isinstance(driver, CausalWorldModelLoopDriver):
                driver.begin_video()
            if hasattr(model, "inference") and isinstance(inputs, dict) and "noise" in inputs and "text_prompts" in inputs:
                out = model.inference(inputs["noise"], inputs["text_prompts"], **kwargs)
                steps = len(getattr(model, "denoising_step_list", [1000, 750, 500, 250]))
            elif isinstance(inputs, dict) and "x" in inputs:
                denoise_steps = inputs.get("steps", [0])
                chunks = inputs.get("chunks", [0])
                kv_cache = inputs.get("kv_cache")
                steps = len(denoise_steps)
                cur = inputs["x"]
                for chunk_idx in chunks:
                    for step_idx in denoise_steps:
                        if isinstance(driver, CausalWorldModelLoopDriver):
                            driver.phase("denoise", step_idx, chunk_idx * 3, 1000.0 - step_idx * 250.0)
                        call_kw = {k: v for k, v in inputs.items() if k not in {"x", "steps", "chunks"}}
                        call_kw.update(kwargs)
                        if kv_cache is not None:
                            call_kw["kv_cache"] = kv_cache
                        cur = model(cur, **call_kw)
                out = cur
            elif isinstance(inputs, Tensor):
                if isinstance(driver, CausalWorldModelLoopDriver):
                    driver.phase("denoise", 0, 0, 1000.0)
                out = model(inputs, **kwargs)
            elif isinstance(inputs, dict):
                out = model(**inputs, **kwargs)
            else:
                out = model(inputs, **kwargs)

        latency = time.perf_counter() - started
        stats = driver.stats.to_dict() if isinstance(driver, CausalWorldModelLoopDriver) else {}
        diag_rows = collector.rows() if isinstance(collector, DiagnosticCollector) else []
        return InferenceResult(
            output=out,
            latency_s=latency,
            step_count=int(steps),
            loop_stats=stats,
            diagnostics=diag_rows,
        )


__all__ = [
    "CausalWorldModelAdapter",
    "CausalWorldModelLoopDriver",
    "validate_world_model_action",
]
