"""Reversible Looped-DiT-B32 (`LoopedMMDiT`) adapter supporting LoopProgram and adaptive schedules."""
from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
import time
from typing import Any, Callable, Iterator

import torch
from torch import Tensor, nn

from elp.core.diagnostics import DiagnosticCollector, ProbeSpec
from elp.core.program import LoopProgram, invoke_loop_program, load_program_callable
from elp.modalities.image.looped_dit import AdaptiveLoopedDiTSampler, LoopedDiTScheduleConfig

from ..base import (
    BackboneAdapter,
    BlockLoopStats,
    InferenceResult,
    activate_upstream_repo,
    resolve_dtype,
    resolve_upstream_repo,
)


def _is_native_looped_mmdit(model: Any) -> bool:
    return all(hasattr(model, attr) for attr in ("img_embed", "txt_embed", "mask_token", "pos_embed", "txt_blocks", "blocks", "loop_blocks", "decode"))


def _extract_stream_callables(
    model: nn.Module,
    *,
    cache_text_preamble: bool = True,
) -> tuple[
    Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
    Callable[[Tensor, Tensor, int], tuple[Tensor, Tensor]],
    Callable[[Tensor, Tensor], Tensor],
]:
    """Adapt either upstream `LoopedMMDiT` or explicit `(pre_blocks, core_step, post_blocks)` models."""
    if _is_native_looped_mmdit(model):
        cached_txt: dict[tuple[int, int, str, Any], Tensor] = {}

        def pre_blocks(x: Tensor, text: Tensor, text_mask: Tensor) -> tuple[Tensor, Tensor]:
            key = (text.data_ptr(), text_mask.data_ptr(), str(text.device), text.dtype)
            txt = cached_txt.get(key) if cache_text_preamble else None
            if txt is None:
                masked = torch.where(
                    text_mask.to(torch.bool)[:, :, None],
                    text,
                    model.mask_token.to(device=text.device, dtype=text.dtype),
                )
                txt = model.txt_embed(masked)
                for tb in model.txt_blocks:
                    txt = tb(txt)
                if cache_text_preamble:
                    cached_txt[key] = txt
            img = model.img_embed(x) + model.pos_embed.to(device=x.device, dtype=x.dtype)
            for block in model.blocks[: model.pre]:
                img, txt = block(img, txt)
            return img, txt

        def core_step(img: Tensor, txt: Tensor, loop_idx_1based: int = 1) -> tuple[Tensor, Tensor]:
            n_avail = max(1, int(getattr(model, "num_loops", 1)))
            r = 1 if getattr(model, "share_loop_weights", True) else min(loop_idx_1based, n_avail)
            for block in model.loop_blocks(r):
                img, txt = block(img, txt)
            return img, txt

        def post_blocks(img: Tensor, txt: Tensor) -> Tensor:
            return model.decode(img, txt)

        return pre_blocks, core_step, post_blocks

    if all(hasattr(model, attr) for attr in ("pre_blocks", "core_step", "post_blocks")):
        return (
            lambda x, text, text_mask: model.pre_blocks(x, text, text_mask),
            lambda img, txt, _r=1: model.core_step(img, txt),
            lambda img, txt: model.post_blocks(img, txt),
        )

    raise TypeError(
        "LoopedDiTAdapter expects an OpenSenseNova `LoopedMMDiT` module or a module exposing "
        "`pre_blocks`, `core_step`, `post_blocks`."
    )


class LoopedDiTLoopPatch:
    """Reversibly override `LoopedMMDiT.forward` to execute an AST-validated `LoopProgram`."""

    def __init__(self, model: nn.Module, program: LoopProgram) -> None:
        if model.training:
            raise ValueError("Call model.eval() before installing LoopedDiTLoopPatch")
        self.model = model
        self.program = program
        self._loop_fn = load_program_callable(program.code)
        self._pre_fn, self._core_fn, self._post_fn = _extract_stream_callables(model)
        self.stats = BlockLoopStats()
        self._memories: dict[int, dict[str, Any]] = {}
        self._saved_forward: tuple[Any, bool] | None = None
        self._restored = False

    def __enter__(self) -> "LoopedDiTLoopPatch":
        had_attr = "forward" in vars(self.model)
        self._saved_forward = (self.model.forward, had_attr)

        def patched_forward(
            x: Tensor,
            text: Tensor,
            text_mask: Tensor,
            num_loops: int | None = None,
            exit_loops: tuple[int, ...] = (),
            **_kwargs: Any,
        ) -> Tensor:
            del num_loops, exit_loops
            img, txt = self._pre_fn(x, text, text_mask)
            li = img.shape[1]
            packed = torch.cat([img, txt], dim=1)
            pass_counter = [0]

            def evaluate(state: Tensor) -> Tensor:
                pass_counter[0] += 1
                self.stats.block_calls += 1
                s_img, s_txt = state[:, :li], state[:, li:]
                n_img, n_txt = self._core_fn(s_img, s_txt, pass_counter[0])
                return torch.cat([n_img, n_txt], dim=1)

            self.stats.loop_invocations += 1
            updated = invoke_loop_program(
                self._loop_fn,
                packed,
                evaluate,
                info={"modality": "image", "backbone": "looped_dit", "img_tokens": li, "txt_tokens": txt.shape[1]},
                params=dict(self.program.params),
                memory=self._memories.setdefault(0, {}),
            )
            return self._post_fn(updated[:, :li], updated[:, li:])

        self.model.forward = patched_forward  # type: ignore[method-assign]
        return self

    def restore(self) -> None:
        if self._restored or self._saved_forward is None:
            return
        orig, had_attr = self._saved_forward
        if had_attr:
            self.model.forward = orig
        elif "forward" in vars(self.model):
            delattr(self.model, "forward")
        self._saved_forward = None
        self._restored = True

    def __exit__(self, *exc: Any) -> None:
        self.restore()


@torch.no_grad()
def euler_sample_with_context(
    model: nn.Module,
    text: Tensor,
    text_mask: Tensor,
    x_init: Tensor,
    *,
    step_count: int = 100,
    cfg_scale: float = 6.0,
    num_loops: int | None = 4,
    context: dict[str, Any] | None = None,
) -> Tensor:
    """Default fused-CFG Euler flow-matching sampler from t=0 (noise) to t=1 (image)."""
    was_training = model.training
    model.eval()
    x = x_init
    device = x.device
    ts = torch.linspace(0.0, 1.0, int(step_count) + 1, device=device)
    null_mask = torch.zeros_like(text_mask)
    amp_dtype = text.dtype if text.dtype in (torch.float16, torch.bfloat16) else torch.float32
    if cfg_scale != 1.0:
        text_both = torch.cat([text, text], dim=0)
        mask_both = torch.cat([text_mask, null_mask], dim=0)

    try:
        for i in range(int(step_count)):
            t0, t1 = ts[i], ts[i + 1]
            if context is not None:
                context["step_index"] = i
                context["t0"] = float(t0)
                context["t1"] = float(t1)
                context["dt"] = float(t1 - t0)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=x.is_cuda and amp_dtype != torch.float32):
                if cfg_scale != 1.0:
                    x_both = torch.cat([x, x], dim=0)
                    pred_both = model(x_both, text_both, mask_both, num_loops=num_loops)
                    pred, pred_uncond = pred_both.chunk(2, dim=0)
                    pred = pred_uncond + (pred - pred_uncond) * cfg_scale
                else:
                    pred = model(x, text, text_mask, num_loops=num_loops)
            denom = (1.0 - t0).clamp_min(0.05)
            v = (pred - x) / denom
            x = x + v * (t1 - t0)
    finally:
        model.train(was_training)
    return x.clamp(-1.0, 1.0)


class LoopedDiTSampleHandle:
    """Reversible runtime handle installing adaptive Looped-DiT sampling onto a frozen model."""

    def __init__(self, sampler: AdaptiveLoopedDiTSampler) -> None:
        self.sampler = sampler
        self.stats = BlockLoopStats()

    @torch.no_grad()
    def sample_batch(
        self,
        model: nn.Module,
        text: Tensor,
        text_mask: Tensor,
        x_init: Tensor,
        *,
        cfg_scale: float = 6.0,
    ) -> Tensor:
        was_training = model.training
        model.eval()
        pre_raw, core_raw, post_raw = _extract_stream_callables(model, cache_text_preamble=True)
        try:
            null_mask = torch.zeros_like(text_mask)
            use_cfg = cfg_scale != 1.0
            pass_counter = [0]

            def pre_fn(x: Tensor, txt_cond: Tensor) -> tuple[Tensor, Tensor]:
                pass_counter[0] = 0
                if not use_cfg:
                    return pre_raw(x, txt_cond, text_mask)
                img_c, txt_c = pre_raw(x, txt_cond, text_mask)
                img_u, txt_u = pre_raw(x, txt_cond, null_mask)
                return torch.cat([img_c, img_u], dim=0), torch.cat([txt_c, txt_u], dim=0)

            def core_fn(img: Tensor, txt: Tensor) -> tuple[Tensor, Tensor]:
                pass_counter[0] += 1
                self.stats.loop_invocations += 1
                return core_raw(img, txt, pass_counter[0])

            def post_fn(img: Tensor, txt: Tensor) -> Tensor:
                if not use_cfg:
                    return post_raw(img, txt)
                pred_c, pred_u = post_raw(img, txt).chunk(2, dim=0)
                return pred_u + (pred_c - pred_u) * cfg_scale

            out, sampler_stats = self.sampler.sample(
                x_init,
                text,
                pre_blocks=pre_fn,
                core_step=core_fn,
                post_blocks=post_fn,
            )
            self.stats.block_calls += int(sampler_stats["core_passes"])
            return out.clamp(-1.0, 1.0)
        finally:
            model.train(was_training)


class LoopedDiTAdapter(BackboneAdapter):
    """Backbone adapter for OpenSenseNova pretrained recurrent `LoopedMMDiT` image generation."""

    name = "looped_dit"
    modality = "image"

    def load(
        self,
        model_path: str,
        *,
        upstream_repo: str | Path | None = None,
        text_encoder_path: str | None = None,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype | None = torch.bfloat16,
        weights: str = "ema",
        local_files_only: bool = True,
        **kwargs: Any,
    ) -> Any:
        del local_files_only, kwargs
        repo = resolve_upstream_repo(
            upstream_repo,
            "LOOPED_DIT_REPO",
            "OpenSenseNova/Looped-DiT",
            bundled_subdir="looped_dit",
        )
        activate_upstream_repo(repo)
        from looped_dit.pipeline import TextEncoder, load_model

        torch_dtype = resolve_dtype(dtype) or torch.bfloat16
        target_dev = torch.device(device)
        model, cfg = load_model(model_path, target_dev, dtype=torch_dtype, weights=weights)
        model._elp_model_cfg = cfg
        if text_encoder_path:
            model._elp_text_encoder = TextEncoder(text_encoder_path, cfg.prompt_length, target_dev)
        return model

    def install_program(self, model: Any, program: Any) -> AbstractContextManager[Any]:
        if program is None:
            return nullcontext()
        prog = self.resolve_program(program)
        if isinstance(prog, LoopProgram):
            return LoopedDiTLoopPatch(model, prog)
        if isinstance(prog, AdaptiveLoopedDiTSampler):
            sampler = prog
        elif isinstance(prog, LoopedDiTScheduleConfig):
            sampler = AdaptiveLoopedDiTSampler(prog)
        else:
            raise TypeError(f"Unsupported program type for LoopedDiTAdapter: {type(program).__name__}")

        @contextmanager
        def _ctx() -> Iterator[LoopedDiTSampleHandle]:
            yield LoopedDiTSampleHandle(sampler)

        return _ctx()

    def _prepare_conditioning(
        self,
        model: Any,
        inputs: dict[str, Any],
        *,
        noise_scale: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if {"x_init", "text", "text_mask"} <= set(inputs):
            return inputs["x_init"], inputs["text"], inputs["text_mask"]
        if "prompt" not in inputs:
            raise ValueError("LoopedDiTAdapter.infer expects {'x_init', 'text', 'text_mask'} or {'prompt'}")

        text_enc = getattr(model, "_elp_text_encoder", None)
        if text_enc is None:
            raise ValueError("Looped-DiT prompt inference requires `text_encoder_path` passed to `load(...)`")
        prompt = inputs["prompt"]
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        input_ids, mask = text_enc.tokenize(prompts)
        param = next(model.parameters())
        text = text_enc.encode(input_ids, mask).to(device=param.device, dtype=param.dtype)
        mask = mask.to(device=param.device)
        cfg = getattr(model, "_elp_model_cfg", None)
        image_size = int(inputs.get("image_size", getattr(cfg, "image_size", 512)))
        seed = inputs.get("seed")
        gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None
        x_init = torch.randn((len(prompts), 3, image_size, image_size), generator=gen, dtype=param.dtype) * float(noise_scale)
        x_init = x_init.to(device=param.device)
        return x_init, text, mask

    @torch.no_grad()
    def infer(
        self,
        model: Any,
        inputs: Any,
        *,
        program: Any = None,
        probe: ProbeSpec | None = None,
        step_count: int = 100,
        cfg_scale: float = 6.0,
        num_loops: int | None = None,
        noise_scale: float = 2.0,
        **kwargs: Any,
    ) -> InferenceResult:
        del kwargs
        if isinstance(model, nn.Module):
            model.eval()
        if not isinstance(inputs, dict):
            raise ValueError("LoopedDiTAdapter.infer expects dict inputs")

        x_init, text, text_mask = self._prepare_conditioning(model, inputs, noise_scale=noise_scale)
        used_loops = num_loops if num_loops is not None else int(getattr(model, "num_loops", 4))
        started = time.perf_counter()

        patch_ctx = self.install_program(model, program)
        diag = DiagnosticCollector(model, probe) if probe is not None else nullcontext()
        with patch_ctx as handle, diag as collector:
            if isinstance(handle, LoopedDiTSampleHandle):
                out = handle.sample_batch(model, text, text_mask, x_init, cfg_scale=cfg_scale)
                used_steps = handle.sampler.config.step_count
                stats = handle.stats.to_dict()
            else:
                used_steps = int(program.metadata.get("steps", step_count)) if isinstance(program, LoopProgram) else step_count
                out = euler_sample_with_context(
                    model,
                    text,
                    text_mask,
                    x_init,
                    step_count=used_steps,
                    cfg_scale=cfg_scale,
                    num_loops=used_loops,
                )
                stats = handle.stats.to_dict() if isinstance(handle, LoopedDiTLoopPatch) else {}

        latency = time.perf_counter() - started
        diag_rows = collector.rows() if isinstance(collector, DiagnosticCollector) else []
        return InferenceResult(
            output=out,
            latency_s=latency,
            step_count=int(used_steps),
            loop_stats=stats,
            diagnostics=diag_rows,
        )


__all__ = [
    "LoopedDiTAdapter",
    "LoopedDiTLoopPatch",
    "LoopedDiTSampleHandle",
    "euler_sample_with_context",
]
