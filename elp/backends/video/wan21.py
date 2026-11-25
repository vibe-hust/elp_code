"""Reversible step-windowed loop adapter for Wan2.1 short-video diffusion (diffusers & native)."""
from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor, nn

from elp.core.diagnostics import DiagnosticCollector, ProbeSpec
from elp.core.program import LoopProgram, invoke_loop_program, load_program_callable
from elp.modalities.video.diffusion import VideoDiffusionLoopConfig

from ..base import (
    BackboneAdapter,
    BlockLoopStats,
    InferenceResult,
    activate_upstream_repo,
    resolve_block_container,
    resolve_dtype,
    resolve_upstream_repo,
)


class Wan21LoopPatch:
    """Reversibly install `VideoDiffusionLoopConfig` or `LoopProgram` onto Wan2.1 blocks."""

    def __init__(
        self,
        model_or_pipe: Any,
        config: VideoDiffusionLoopConfig | None = None,
        context: dict[str, Any] | None = None,
        *,
        program: LoopProgram | None = None,
    ) -> None:
        if config is None and program is None:
            raise ValueError("Provide either `config` or `program` to Wan21LoopPatch")
        self.target = model_or_pipe
        self.blocks = resolve_block_container(model_or_pipe, attrs=("blocks", "layers"))
        depth = len(self.blocks)
        self.program = program
        self._program_fn = load_program_callable(program.code) if program is not None else None

        if config is None:
            assert program is not None
            meta = program.metadata
            params = program.params
            default_blocks = tuple(range(depth // 3, min(depth, depth // 3 + max(1, min(4, depth)))))
            config = VideoDiffusionLoopConfig(
                blocks=tuple(int(b) for b in meta.get("blocks", default_blocks)),
                repeats=int(params.get("repeats", params.get("k", 2))),
                strength=float(params.get("strength", 1.0)),
                start_frac=float(params.get("start_frac", meta.get("start_frac", 0.0))),
                end_frac=float(params.get("end_frac", meta.get("end_frac", 0.5))),
            )

        if max(config.blocks) >= depth:
            raise ValueError(f"Block index {max(config.blocks)} exceeds depth {depth}")
        self.config = config
        self.context = context if context is not None else {"step_index": 0, "step_count": int(program.metadata.get("steps", 26)) if program else 26}
        self.stats = BlockLoopStats()
        self._memories: dict[int, dict[str, Any]] = {}
        self._saved: list[tuple[int, Any, bool]] = []
        self._restored = False

    def __enter__(self) -> "Wan21LoopPatch":
        cfg = self.config
        ctx = self.context
        stats = self.stats
        prog_fn = self._program_fn
        prog_params = dict(self.program.params) if self.program is not None else {}
        scale = cfg.strength / float(cfg.repeats)

        for idx in cfg.blocks:
            block = self.blocks[idx]
            self._saved.append((idx, block.forward, "forward" in vars(block)))
            orig = block.forward

            def forward(x: Tensor, *args: Any, _idx=idx, _orig=orig, **kwargs: Any) -> Any:
                step_count = max(1, int(ctx.get("step_count", 1)))
                progress = float(ctx.get("step_index", 0)) / float(step_count)
                if not cfg.start_frac <= progress < cfg.end_frac:
                    stats.block_calls += 1
                    return _orig(x, *args, **kwargs)

                if prog_fn is not None:
                    stats.loop_invocations += 1
                    tail_box: list[tuple[bool, Any]] = [(False, None)]

                    def single_call(h: Tensor) -> Tensor:
                        stats.block_calls += 1
                        res = _orig(h, *args, **kwargs)
                        if isinstance(res, tuple):
                            tail_box[0] = (True, res[1:])
                            return res[0]
                        tail_box[0] = (False, None)
                        return res

                    updated = invoke_loop_program(
                        prog_fn,
                        x,
                        single_call,
                        info={"layer": _idx, "progress": progress, "modality": "video"},
                        params=dict(prog_params),
                        memory=self._memories.setdefault(_idx, {}),
                    )
                    was_tup, tail = tail_box[0]
                    if tail is not None:
                        return (updated, *tail)
                    return (updated,) if was_tup else updated

                if cfg.repeats == 1 and cfg.strength == 1.0:
                    stats.block_calls += 1
                    return _orig(x, *args, **kwargs)

                stats.loop_invocations += 1
                cur = x
                tail = None
                for _ in range(cfg.repeats):
                    stats.block_calls += 1
                    res = _orig(cur, *args, **kwargs)
                    if isinstance(res, tuple):
                        y, tail = res[0], res[1:]
                    else:
                        y, tail = res, None
                    cur = cur + scale * (y - cur)
                return cur if tail is None else (cur, *tail)

            block.forward = forward  # type: ignore[method-assign]
        return self

    def restore(self) -> None:
        if self._restored:
            return
        for idx, orig, had_attr in reversed(self._saved):
            block = self.blocks[idx]
            if had_attr:
                block.forward = orig
            elif "forward" in vars(block):
                delattr(block, "forward")
        self._saved.clear()
        self._restored = True

    def __exit__(self, *exc: Any) -> None:
        self.restore()


class Wan21Adapter(BackboneAdapter):
    """Backbone adapter for Wan2.1-1.3B text-to-video diffusion."""

    name = "wan21"
    modality = "video"

    def load(
        self,
        model_path: str,
        *,
        upstream_repo: str | Path | None = None,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype | None = None,
        local_files_only: bool = True,
        **kwargs: Any,
    ) -> Any:
        torch_dtype = resolve_dtype(dtype)
        repo_hint = upstream_repo or os.environ.get("WAN21_REPO", "").strip()
        ckpt_path = Path(model_path)
        is_native_wan = bool(repo_hint) or (ckpt_path.is_dir() and not (ckpt_path / "model_index.json").is_file() and (ckpt_path / "config.json").is_file())

        if is_native_wan:
            repo = resolve_upstream_repo(
                upstream_repo,
                "WAN21_REPO",
                "Wan2.1",
                bundled_subdir="wan21",
            )
            activate_upstream_repo(repo)
            from safetensors.torch import load_file
            from wan.modules.model import WanModel

            if not ckpt_path.is_dir():
                from huggingface_hub import snapshot_download

                ckpt_path = Path(snapshot_download(model_path, local_files_only=local_files_only))
            cfg = json.loads((ckpt_path / "config.json").read_text(encoding="utf-8"))
            model = WanModel.from_config(cfg)
            index_path = ckpt_path / "diffusion_pytorch_model.safetensors.index.json"
            files = (
                sorted(set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values()))
                if index_path.is_file()
                else ["diffusion_pytorch_model.safetensors"]
            )
            weights: dict[str, Tensor] = {}
            for name in files:
                weights.update(load_file(str(ckpt_path / name), device="cpu"))
            model.load_state_dict(weights, strict=True, assign=True)
            model.eval().to(device=device)
            if torch_dtype is not None:
                model.to(dtype=torch_dtype)
            return model

        try:
            from diffusers import WanPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Loading Wan2.1 requires `diffusers` (`WanPipeline`, `elp[video]`) or native `WAN21_REPO`."
            ) from exc
        load_kw: dict[str, Any] = {"local_files_only": local_files_only}
        if torch_dtype is not None:
            load_kw["torch_dtype"] = torch_dtype
        load_kw.update(kwargs)
        pipe = WanPipeline.from_pretrained(model_path, **load_kw)
        pipe.to(device)
        return pipe

    def install_program(self, model: Any, program: Any) -> AbstractContextManager[Any]:
        if program is None:
            return nullcontext()
        prog = self.resolve_program(program)
        if isinstance(prog, LoopProgram):
            return Wan21LoopPatch(model, program=prog)
        if isinstance(prog, VideoDiffusionLoopConfig):
            return Wan21LoopPatch(model, prog)
        if isinstance(prog, dict):
            return Wan21LoopPatch(model, VideoDiffusionLoopConfig(**prog))
        raise TypeError(f"Unsupported program type for Wan21Adapter: {type(program).__name__}")

    @torch.no_grad()
    def infer(
        self,
        model: Any,
        inputs: Any,
        *,
        program: Any = None,
        probe: ProbeSpec | None = None,
        step_count: int = 26,
        **kwargs: Any,
    ) -> InferenceResult:
        if isinstance(model, nn.Module):
            model.eval()
        started = time.perf_counter()
        patch_ctx = self.install_program(model, program)
        diag_target = getattr(model, "transformer", model)
        diag = DiagnosticCollector(diag_target, probe) if probe is not None else nullcontext()

        with patch_ctx as patch, diag as collector:
            if callable(model) and not isinstance(model, nn.Module) and isinstance(inputs, dict) and "prompt" in inputs:
                def on_step_end(_pipe: Any, step_index: int, _t: Any, cb_kwargs: dict[str, Any]) -> dict[str, Any]:
                    if isinstance(patch, Wan21LoopPatch):
                        patch.context["step_index"] = step_index + 1
                        patch.context["step_count"] = step_count
                    return cb_kwargs

                if isinstance(patch, Wan21LoopPatch):
                    patch.context["step_index"] = 0
                    patch.context["step_count"] = step_count
                call_kw = dict(inputs)
                call_kw.update(kwargs)
                call_kw.setdefault("num_inference_steps", step_count)
                call_kw.setdefault("callback_on_step_end", on_step_end)
                out = model(**call_kw)
            elif isinstance(inputs, dict) and "x" in inputs:
                cur = inputs["x"]
                n_steps = int(inputs.get("steps", step_count))
                for s in range(n_steps):
                    if isinstance(patch, Wan21LoopPatch):
                        patch.context["step_index"] = s
                        patch.context["step_count"] = n_steps
                    cur = model(cur, **kwargs)
                out = cur
                step_count = n_steps
            elif isinstance(inputs, Tensor):
                if isinstance(patch, Wan21LoopPatch):
                    patch.context["step_index"] = 0
                    patch.context["step_count"] = step_count
                out = model(inputs, **kwargs)
            else:
                out = model(**inputs, **kwargs) if isinstance(inputs, dict) else model(inputs, **kwargs)

        latency = time.perf_counter() - started
        stats = patch.stats.to_dict() if isinstance(patch, Wan21LoopPatch) else {}
        diag_rows = collector.rows() if isinstance(collector, DiagnosticCollector) else []
        return InferenceResult(
            output=out,
            latency_s=latency,
            step_count=int(step_count),
            loop_stats=stats,
            diagnostics=diag_rows,
        )


__all__ = [
    "Wan21Adapter",
    "Wan21LoopPatch",
]
