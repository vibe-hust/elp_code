"""Reversible block looping and shifted-Euler sampler adapter for RAE DiTs (diffusion-bench)."""
from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from functools import wraps
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Callable

import torch
from torch import Tensor, nn

from elp.core.diagnostics import DiagnosticCollector, ProbeSpec
from elp.core.program import LoopProgram, invoke_loop_program, load_program_callable
from elp.modalities.image.operators import (
    _OPERATORS,
    CachedLoopGuidance,
    ImageLoopConfig,
    _extract_tensor,
    _reassemble,
)

from ..base import (
    BackboneAdapter,
    BlockLoopStats,
    InferenceResult,
    activate_upstream_repo,
    resolve_block_container,
    resolve_dtype,
    resolve_upstream_repo,
)


def shifted_euler_sample(
    model_fn: Callable[..., Tensor],
    noise: Tensor,
    context: Tensor,
    attn_mask: Tensor | None = None,
    steps: int = 50,
    *,
    shift: float = (196608.0 / 4096.0) ** 0.5,
) -> Tensor:
    """Upstream shifted-Euler sampler matching DiffusionBench RAE inference."""
    grid = torch.linspace(1.0, 0.0, int(steps) + 1)
    grid = shift * grid / (1.0 + (shift - 1.0) * grid)
    times = grid[:-1, None].expand(-1, noise.shape[0]).contiguous().to(device=noise.device, dtype=noise.dtype)
    widths = (grid[:-1] - grid[1:]).to(device=noise.device, dtype=noise.dtype)
    x = noise
    kw: dict[str, Any] = {"context": context}
    if attn_mask is not None:
        kw["attn_mask"] = attn_mask
    for idx in range(int(steps)):
        pred = model_fn(x, times[idx], **kw)
        x = x - widths[idx] * pred
    return x


class RAELoopPatch:
    """Reversibly install block looping (`ImageLoopConfig` or `LoopProgram`) on RAE DiTs."""

    def __init__(
        self,
        model: nn.Module,
        loop_cfg: ImageLoopConfig | None = None,
        guidance_cfg: CachedLoopGuidance | None = None,
        *,
        program: LoopProgram | None = None,
        step_state: dict[str, Any] | None = None,
    ) -> None:
        if model.training:
            raise ValueError("Call model.eval() before installing RAELoopPatch")
        if loop_cfg is None and program is None:
            raise ValueError("Provide either `loop_cfg` or `program` to RAELoopPatch")
        self.model = model
        self.blocks = resolve_block_container(model)
        depth = len(self.blocks)

        self.program = program
        self._program_fn = load_program_callable(program.code) if program is not None else None
        if loop_cfg is None:
            assert program is not None
            meta = program.metadata
            params = program.params
            default_blocks = tuple(range(max(0, depth - 10), max(1, depth - 2)))
            blocks = tuple(int(b) for b in meta.get("blocks", default_blocks))
            loop_cfg = ImageLoopConfig(
                blocks=blocks,
                num_loops=int(params.get("k", 2)),
                operator=str(params.get("operator", "euler")),  # type: ignore[arg-type]
                step_scale=float(params.get("step_scale", 0.5)),
                start_frac=float(params.get("start_frac", meta.get("start_frac", 0.0))),
                end_frac=float(params.get("end_frac", meta.get("end_frac", 1.0))),
                loopguidance_weight=float(params.get("loopguidance_weight", 1.0)),
                reference_refresh_interval=int(params.get("refresh_interval", 1)),
                reuse_scale=float(params.get("reuse_scale", 1.0)),
            )
            if guidance_cfg is None and float(params.get("loopguidance_weight", 1.0)) > 1.0:
                guidance_cfg = CachedLoopGuidance(
                    weight=loop_cfg.loopguidance_weight,
                    refresh_interval=loop_cfg.reference_refresh_interval,
                    reuse_scale=loop_cfg.reuse_scale,
                )

        if max(loop_cfg.blocks) >= depth:
            raise ValueError(f"Block index {max(loop_cfg.blocks)} out of range for depth {depth}")
        self.loop_cfg = loop_cfg
        self.guidance_cfg = guidance_cfg
        self.step_state = step_state if step_state is not None else {"progress": 0.0, "reference_mode": False}
        self.stats = BlockLoopStats()
        self._memories: dict[int, dict[str, Any]] = {}
        self._saved: list[tuple[nn.Module, Any, bool]] = []
        self._restored = False

    def __enter__(self) -> "RAELoopPatch":
        cfg = self.loop_cfg
        loop_fn = _OPERATORS[cfg.operator]
        prog_fn = self._program_fn
        prog_params = dict(self.program.params) if self.program is not None else {}
        step_state = self.step_state
        stats = self.stats
        memories = self._memories

        for idx in cfg.blocks:
            block = self.blocks[idx]
            had_attr = "forward" in vars(block)
            orig = block.forward
            self._saved.append((block, orig, had_attr))

            def wrapped_forward(x: Tensor, *args: Any, _idx: int = idx, _orig=orig, **kwargs: Any) -> Any:
                progress = float(step_state.get("progress", 0.0))
                in_window = cfg.start_frac <= progress <= cfg.end_frac
                if step_state.get("reference_mode", False) or not in_window:
                    stats.block_calls += 1
                    return _orig(x, *args, **kwargs)
                tail_box: list[Any] = [None]

                def single_call(h: Tensor) -> Tensor:
                    stats.block_calls += 1
                    out, tail = _extract_tensor(_orig(h, *args, **kwargs))
                    tail_box[0] = tail
                    return out

                stats.loop_invocations += 1
                if prog_fn is not None:
                    updated = invoke_loop_program(
                        prog_fn,
                        x,
                        single_call,
                        info={"layer": _idx, "progress": progress, "modality": "image"},
                        params=dict(prog_params),
                        memory=memories.setdefault(_idx, {}),
                    )
                else:
                    updated = loop_fn(single_call, x, cfg.num_loops, cfg.step_scale)
                return _reassemble(updated, tail_box[0])

            block.forward = wrapped_forward  # type: ignore[method-assign]
        return self

    def restore(self) -> None:
        if self._restored:
            return
        for block, orig, had_attr in reversed(self._saved):
            if had_attr:
                block.forward = orig
            elif "forward" in vars(block):
                delattr(block, "forward")
        self._saved.clear()
        self._restored = True

    def __exit__(self, *exc: Any) -> None:
        self.restore()


def wrap_rae_model_fn(
    model_fn: Callable[..., Tensor],
    patch: RAELoopPatch,
    total_steps: int,
) -> Callable[..., Tensor]:
    """Wrap an ODE denoiser callable to update step progress and apply cached LoopGuidance."""
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    guidance = patch.guidance_cfg
    if guidance is not None:
        guidance.reset()

    @wraps(model_fn)
    def tracked(*args: Any, **kwargs: Any) -> Tensor:
        step_index = tracked._rae_step_index
        if step_index >= total_steps:
            raise RuntimeError("RAE tracked model_fn exceeded total_steps; call reset_rae_model_fn")
        progress = step_index / float(max(1, total_steps - 1)) if total_steps > 1 else 0.0
        patch.step_state["progress"] = progress
        tracked._rae_step_index += 1

        active = guidance is not None and patch.loop_cfg.start_frac <= progress <= patch.loop_cfg.end_frac
        if active and guidance is not None:
            def eval_mode(is_ref: bool) -> Tensor:
                prev = bool(patch.step_state.get("reference_mode", False))
                patch.step_state["reference_mode"] = is_ref
                try:
                    return model_fn(*args, **kwargs)
                finally:
                    patch.step_state["reference_mode"] = prev

            prev_ref, prev_loop = guidance.reference_calls, guidance.loop_calls
            out = guidance.predict(
                step_index,
                pred_loop_fn=lambda: eval_mode(False),
                pred_ref_fn=lambda: eval_mode(True),
                active=True,
            )
            patch.stats.guidance_ref_calls += guidance.reference_calls - prev_ref
            patch.stats.guidance_loop_calls += guidance.loop_calls - prev_loop
            return out
        return model_fn(*args, **kwargs)

    tracked._rae_step_index = 0
    tracked._rae_patch = patch
    return tracked


def reset_rae_model_fn(model_fn: Callable[..., Any]) -> None:
    """Reset tracked ODE step index and guidance state before the next batch."""
    if not hasattr(model_fn, "_rae_step_index"):
        raise TypeError("model_fn was not created by wrap_rae_model_fn")
    model_fn._rae_step_index = 0
    guidance = model_fn._rae_patch.guidance_cfg
    if guidance is not None:
        guidance.reset()
    model_fn._rae_patch.step_state["progress"] = 0.0


def _build_rae_decoder(
    repo: Path,
    decoder_path: str | Path,
    *,
    stats_path: str | Path | None = None,
    decoder_config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> nn.Module:
    from stage1.decoders import GeneralDecoder
    from transformers import AutoConfig

    cfg_dir = Path(decoder_config_path) if decoder_config_path is not None else (repo / "configs/decoder/ViTXL")
    config = AutoConfig.from_pretrained(str(cfg_dir))
    config.hidden_size, config.patch_size, config.image_size = 768, 16, 256

    class _DecodeOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = GeneralDecoder(config, num_patches=256)
            state = torch.load(str(decoder_path), map_location="cpu", weights_only=False)
            sd = state.get("state_dict", state) if isinstance(state, dict) else state
            self.decoder.load_state_dict(sd, strict=True, assign=True)
            if stats_path is not None:
                stats = torch.load(str(stats_path), map_location="cpu", weights_only=False)
                mean = stats.get("mean", torch.tensor(0.0))
                var = stats.get("var", torch.tensor(1.0))
            else:
                mean, var = torch.tensor(0.0), torch.tensor(1.0)
            self.register_buffer("latent_mean", mean)
            self.register_buffer("latent_var", var)

        def forward(self, z: Tensor) -> Tensor:
            z = z * torch.sqrt(self.latent_var + 1e-5) + self.latent_mean
            b, c, h, w = z.shape
            z = z.view(b, c, h * w).transpose(1, 2)
            return self.decoder.unpatchify(self.decoder(z, drop_cls_token=False).logits)

    return _DecodeOnly().requires_grad_(False).eval().to(device=device)


class RAEAdapter(BackboneAdapter):
    """Backbone adapter for sparse/modality-split RAE-SigLIP2-B/DDT diffusion models."""

    name = "rae"
    modality = "image"

    def load(
        self,
        model_path: str,
        *,
        upstream_repo: str | Path | None = None,
        decoder_path: str | Path | None = None,
        stats_path: str | Path | None = None,
        decoder_config_path: str | Path | None = None,
        text_encoder_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype | None = None,
        local_files_only: bool = True,
        **kwargs: Any,
    ) -> Any:
        repo = resolve_upstream_repo(
            upstream_repo,
            "DIFFUSION_BENCH_REPO",
            "diffusion-bench (RAE)",
            bundled_subdir="rae",
        )
        activate_upstream_repo(repo, extra_subdirs=("src",))
        from stage2.models.DDT import DiTwDDTHead

        init_kwargs = {
            "condition_type": "text",
            "context_dim": 1024,
            "cond_arch": SimpleNamespace(num_t_tokens=4, num_c_tokens=256),
            **kwargs,
        }
        model = DiTwDDTHead(**init_kwargs)
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        sd = state["ema"] if isinstance(state, dict) and "ema" in state else state
        model.load_state_dict(sd, strict=True, assign=True)
        model.requires_grad_(False).eval().to(device=device)
        torch_dtype = resolve_dtype(dtype)
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)
        if decoder_path is not None:
            model._elp_rae_decoder = _build_rae_decoder(
                repo,
                decoder_path,
                stats_path=stats_path,
                decoder_config_path=decoder_config_path,
                device=device,
            )
        if text_encoder_path is not None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(text_encoder_path), local_files_only=local_files_only)
            enc_kw: dict[str, Any] = {"local_files_only": local_files_only}
            enc_kw["torch_dtype"] = torch_dtype if torch_dtype is not None else "auto"
            txt_enc = AutoModelForCausalLM.from_pretrained(str(text_encoder_path), **enc_kw)
            txt_enc.requires_grad_(False).eval().to(device=device)
            model._elp_tokenizer = tok
            model._elp_text_encoder = txt_enc
        return model

    def install_program(self, model: Any, program: Any) -> AbstractContextManager[Any]:
        if program is None:
            return nullcontext()
        prog = self.resolve_program(program)
        if isinstance(prog, tuple) and len(prog) == 2 and isinstance(prog[0], ImageLoopConfig):
            return RAELoopPatch(model, prog[0], prog[1])
        if isinstance(prog, ImageLoopConfig):
            return RAELoopPatch(model, prog)
        if isinstance(prog, LoopProgram):
            return RAELoopPatch(model, program=prog)
        raise TypeError(f"Unsupported program type for RAEAdapter: {type(program).__name__}")

    @torch.no_grad()
    def infer(
        self,
        model: Any,
        inputs: Any,
        *,
        program: Any = None,
        probe: ProbeSpec | None = None,
        total_steps: int = 50,
        **kwargs: Any,
    ) -> InferenceResult:
        if isinstance(model, nn.Module):
            model.eval()
        started = time.perf_counter()
        patch_ctx = self.install_program(model, program)
        diag = DiagnosticCollector(model, probe) if probe is not None else nullcontext()

        with patch_ctx as patch, diag as collector:
            if isinstance(inputs, dict) and callable(inputs.get("sampler_fn")) and callable(inputs.get("model_fn")):
                model_fn = inputs["model_fn"]
                if isinstance(patch, RAELoopPatch):
                    model_fn = wrap_rae_model_fn(model_fn, patch, total_steps=total_steps)
                out = inputs["sampler_fn"](
                    model_fn,
                    **{k: v for k, v in inputs.items() if k not in {"sampler_fn", "model_fn"}},
                )
            elif isinstance(inputs, dict) and {"noise", "context"} <= set(inputs):
                steps = int(inputs.get("steps", total_steps))
                model_fn = wrap_rae_model_fn(model, patch, total_steps=steps) if isinstance(patch, RAELoopPatch) else model
                out = shifted_euler_sample(
                    model_fn,
                    inputs["noise"],
                    inputs["context"],
                    attn_mask=inputs.get("attn_mask"),
                    steps=steps,
                )
                total_steps = steps
            elif isinstance(inputs, dict) and "prompt" in inputs:
                txt_enc = getattr(model, "_elp_text_encoder", None)
                tok = getattr(model, "_elp_tokenizer", None)
                if txt_enc is None or tok is None:
                    raise ValueError(
                        "RAE prompt inference requires `text_encoder_path` in `load(...)` "
                        "(or pass precomputed `{'noise', 'context'}` tensors)."
                    )
                prompts = [inputs["prompt"]] if isinstance(inputs["prompt"], str) else list(inputs["prompt"])
                enc_dev = next(txt_enc.parameters()).device
                tok_in = tok(prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
                tok_in = {k: v.to(enc_dev) if isinstance(v, Tensor) else v for k, v in tok_in.items()}
                if hasattr(txt_enc, "model"):
                    hidden = txt_enc.model(**tok_in, use_cache=False, return_dict=True).last_hidden_state
                else:
                    hidden = txt_enc(**tok_in, use_cache=False, output_hidden_states=True, return_dict=True).hidden_states[-1]
                model_param = next(model.parameters())
                hidden = hidden.to(device=model_param.device, dtype=model_param.dtype)
                attn_mask = tok_in.get("attention_mask")
                if isinstance(attn_mask, Tensor):
                    attn_mask = attn_mask.to(device=model_param.device)
                seed = inputs.get("seed")
                gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None
                in_ch = int(getattr(model, "in_channels", 768))
                noise = torch.randn((len(prompts), in_ch, 16, 16), generator=gen, dtype=model_param.dtype).to(model_param.device)
                steps = int(inputs.get("steps", total_steps))
                model_fn = wrap_rae_model_fn(model, patch, total_steps=steps) if isinstance(patch, RAELoopPatch) else model
                out = shifted_euler_sample(model_fn, noise, hidden, attn_mask=attn_mask, steps=steps)
                decoder = getattr(model, "_elp_rae_decoder", None)
                if decoder is not None:
                    out = decoder(out)
                total_steps = steps
            elif isinstance(inputs, Tensor):
                out = model(inputs, **kwargs)
            elif isinstance(inputs, dict):
                call_kw = {k: v for k, v in inputs.items() if k != "x"}
                call_kw.update(kwargs)
                out = model(inputs["x"], **call_kw) if "x" in inputs else model(**call_kw)
            else:
                out = model(inputs, **kwargs)

        latency = time.perf_counter() - started
        stats = patch.stats.to_dict() if isinstance(patch, RAELoopPatch) else {}
        diag_rows = collector.rows() if isinstance(collector, DiagnosticCollector) else []
        return InferenceResult(
            output=out,
            latency_s=latency,
            step_count=int(total_steps),
            loop_stats=stats,
            diagnostics=diag_rows,
        )


__all__ = [
    "RAEAdapter",
    "RAELoopPatch",
    "reset_rae_model_fn",
    "shifted_euler_sample",
    "wrap_rae_model_fn",
]
