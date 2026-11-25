"""Reversible block looping, sampler hooks, and LoopGuidance adapter for Scale-RAE DiTs."""
from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
import os
from pathlib import Path
import time
from typing import Any

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
    resolve_upstream_repo,
)


def resolve_scale_rae_backbone(model: Any) -> nn.Module:
    """Locate the inner DiT backbone owning `.layers`, `.dit_blocks`, or `.blocks`."""
    candidate = model
    if hasattr(candidate, "diff_head"):
        candidate = candidate.diff_head
    if hasattr(candidate, "model"):
        candidate = candidate.model
    if (
        getattr(candidate, "layers", None) is None
        and getattr(candidate, "dit_blocks", None) is None
        and getattr(candidate, "blocks", None) is None
    ):
        raise TypeError("Scale-RAE adapter requires a backbone exposing .layers, .dit_blocks, or .blocks")
    return candidate


def resolve_scale_rae_blocks(backbone: Any) -> Any:
    """Return block list container across Scale-RAE variants."""
    for attr in ("layers", "dit_blocks", "blocks"):
        blocks = getattr(backbone, attr, None)
        if blocks is not None:
            return blocks
    raise TypeError("Scale-RAE backbone missing block container")


def set_sampler_step_count(model: Any, step_count: int | None) -> dict[str, Any]:
    """Adjust Rectified Flow `inference_flow.used_timesteps` cleanly or restore defaults."""
    diff_head = getattr(model, "diff_head", None)
    flow = getattr(diff_head, "inference_flow", None)
    if flow is None or not hasattr(flow, "used_timesteps"):
        return {"requested_step_count": step_count, "observed_step_count": step_count or 0}

    if not hasattr(flow, "_elp_orig_used_timesteps"):
        flow._elp_orig_used_timesteps = list(flow.used_timesteps)
    original = list(flow._elp_orig_used_timesteps)
    default_steps = max(0, len(original) - 1)
    if step_count is None:
        flow.used_timesteps = original
        return {
            "requested_step_count": None,
            "observed_step_count": default_steps,
            "default_step_count": default_steps,
        }
    if step_count < 1 or step_count > max(1, default_steps):
        raise ValueError(f"step_count must be in 1..{max(1, default_steps)}, got {step_count}")
    last = len(original) - 1
    indices = [round(i * last / step_count) for i in range(step_count + 1)]
    if len(set(indices)) != len(indices):
        raise ValueError(f"step_count={step_count} does not map to unique timesteps")
    flow.used_timesteps = [original[idx] for idx in indices]
    return {
        "requested_step_count": int(step_count),
        "observed_step_count": max(0, len(flow.used_timesteps) - 1),
        "default_step_count": default_steps,
    }


def _install_dtype_preserving_scale_rae_loader() -> Any:
    """Patch Scale-RAE's vendored loader so checkpoint tensor dtypes stay intact."""
    try:
        import utils.load_model as load_model_module
    except ImportError:
        load_model_module = None
    from scale_rae.model import builder

    if getattr(builder, "_elp_dtype_patched", False):
        return builder

    existing = list(getattr(builder.ScaleRAEQwenForCausalLM, "_no_split_modules", None) or [])
    for cls_name in ("LightningDiT", "LightningDiTBlock", "LuminaNextDiT2DModel", "LuminaNextDiTBlock", "ResBlock"):
        if cls_name not in existing:
            existing.append(cls_name)
    builder.ScaleRAEQwenForCausalLM._no_split_modules = existing

    def load_pretrained_model(
        model_path: str,
        model_base: str | None = None,
        model_name: str = "",
        load_8bit: bool = False,
        load_4bit: bool = False,
        device_map: Any = "auto",
        device: str = "cuda",
        use_flash_attn: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, Any, Any, int]:
        del model_base, model_name
        load_kwargs = {"device_map": device_map, **kwargs}
        if "torch_dtype" in load_kwargs:
            load_kwargs["torch_dtype"] = None
        if device != "cuda":
            load_kwargs["device_map"] = {"": device}
        if load_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["quantization_config"] = builder.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if use_flash_attn:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        tokenizer = builder.AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = builder.ScaleRAEQwenForCausalLM.from_pretrained(model_path, **load_kwargs)
        for vision_tower in model.get_vision_tower_aux_list():
            vision_tower.load_model()
        image_processor = [vt.image_processor for vt in model.get_vision_tower_aux_list()]
        context_len = int(getattr(model.config, "max_sequence_length", 2048))
        return tokenizer, model, image_processor, context_len

    builder.load_pretrained_model = load_pretrained_model
    if load_model_module is not None:
        load_model_module.load_pretrained_model = load_pretrained_model
    builder._elp_dtype_patched = True
    return builder


class ScaleRAELoopPatch:
    """Reversibly install block looping (`ImageLoopConfig` or `LoopProgram`) on Scale-RAE."""

    def __init__(
        self,
        model: nn.Module,
        loop_cfg: ImageLoopConfig | None = None,
        guidance_cfg: CachedLoopGuidance | None = None,
        *,
        program: LoopProgram | None = None,
        step_count: int | None = None,
        step_state: dict[str, Any] | None = None,
    ) -> None:
        if model.training:
            raise ValueError("Call model.eval() before installing ScaleRAELoopPatch")
        if loop_cfg is None and program is None:
            raise ValueError("Provide either `loop_cfg` or `program` to ScaleRAELoopPatch")
        self.model = model
        self.backbone = resolve_scale_rae_backbone(model)
        self.blocks = resolve_scale_rae_blocks(self.backbone)
        depth = len(self.blocks)

        self.program = program
        self._program_fn = load_program_callable(program.code) if program is not None else None
        if loop_cfg is None:
            assert program is not None
            meta = program.metadata
            params = program.params
            default_blocks = tuple(range(depth // 3, max(depth // 3 + 1, (2 * depth) // 3)))
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
            if step_count is None and "steps" in meta:
                step_count = int(meta["steps"])

        if max(loop_cfg.blocks) >= depth:
            raise ValueError(f"Block index {max(loop_cfg.blocks)} out of range for depth {depth}")

        self.loop_cfg = loop_cfg
        self.guidance_cfg = guidance_cfg
        self.step_count = step_count
        self.step_state = step_state if step_state is not None else {"progress": 0.0, "reference_mode": False}
        self.stats = BlockLoopStats()
        self._memories: dict[int, dict[str, Any]] = {}
        self._saved_blocks: list[tuple[nn.Module, Any, bool]] = []
        self._saved_flow: list[tuple[Any, str, Any, bool]] = []
        self._had_step_override = False
        self._restored = False

    def __enter__(self) -> "ScaleRAELoopPatch":
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
            self._saved_blocks.append((block, orig, had_attr))

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

        if self.step_count is not None:
            set_sampler_step_count(self.model, self.step_count)
            self._had_step_override = True

        diff_head = getattr(self.model, "diff_head", None)
        flow = getattr(diff_head, "inference_flow", None)
        if flow is not None:
            self._hook_flow(flow)
        return self

    def _hook_flow(self, flow: Any) -> None:
        guidance = self.guidance_cfg
        if guidance is not None:
            guidance.reset()
        cfg = self.loop_cfg
        step_state = self.step_state
        stats = self.stats
        flow_state = {"idx": 0}

        def save_attr(name: str, new_val: Any) -> None:
            existed = name in vars(flow)
            self._saved_flow.append((flow, name, vars(flow).get(name), existed))
            setattr(flow, name, new_val)

        def total_steps() -> int:
            if hasattr(flow, "used_timesteps"):
                return max(1, len(flow.used_timesteps) - 1)
            return max(1, self.step_count or 1)

        def advance_progress() -> tuple[int, float, bool]:
            total = total_steps()
            step_idx = flow_state["idx"] % total
            progress = step_idx / float(max(1, total - 1)) if total > 1 else 0.0
            step_state["progress"] = progress
            flow_state["idx"] = step_idx + 1
            active = guidance is not None and cfg.start_frac <= progress <= cfg.end_frac
            return step_idx, progress, active

        if hasattr(flow, "p_sample_loop"):
            orig_loop = flow.p_sample_loop

            def hooked_p_sample_loop(*args: Any, **kwargs: Any) -> Any:
                flow_state["idx"] = 0
                if guidance is not None:
                    guidance.reset()
                return orig_loop(*args, **kwargs)

            save_attr("p_sample_loop", hooked_p_sample_loop)

        if hasattr(flow, "euler_forward"):
            orig_ef = flow.euler_forward

            def hooked_euler_forward(model_arg: Any, x_t: Tensor, t_curr: Tensor, t_next: Tensor, denoised_fn: Any, model_kwargs: Any) -> Any:
                step_idx, _progress, active = advance_progress()
                if not active or guidance is None:
                    return orig_ef(model_arg, x_t, t_curr, t_next, denoised_fn, model_kwargs)

                def eval_pred(is_ref: bool) -> Tensor:
                    prev = bool(step_state.get("reference_mode", False))
                    step_state["reference_mode"] = is_ref
                    try:
                        pred = model_arg(x_t, t_curr, **(model_kwargs or {}))
                        return denoised_fn(pred) if denoised_fn is not None else pred
                    finally:
                        step_state["reference_mode"] = prev

                prev_ref, prev_loop = guidance.reference_calls, guidance.loop_calls
                model_pred = guidance.predict(
                    step_idx,
                    pred_loop_fn=lambda: eval_pred(False),
                    pred_ref_fn=lambda: eval_pred(True),
                    active=True,
                )
                stats.guidance_ref_calls += guidance.reference_calls - prev_ref
                stats.guidance_loop_calls += guidance.loop_calls - prev_loop
                delta_t = (t_next - t_curr).view(-1, *([1] * (x_t.ndim - 1))).to(x_t.device)
                return (x_t + delta_t * model_pred,)

            save_attr("euler_forward", hooked_euler_forward)
            orig_step_fn = getattr(flow, "step_fn", None)
            if orig_step_fn == orig_ef:
                save_attr("step_fn", hooked_euler_forward)
            elif orig_step_fn is not None:
                def hooked_step_fn(*args: Any, **kwargs: Any) -> Any:
                    advance_progress()
                    return orig_step_fn(*args, **kwargs)

                save_attr("step_fn", hooked_step_fn)

        if hasattr(flow, "euler_sample"):
            orig_es = flow.euler_sample

            def hooked_euler_sample(model_arg: Any, x_t: Tensor, u_t: Tensor, u_s: Tensor, clip_denoised: bool, denoised_fn: Any, cond_fn: Any, model_kwargs: Any) -> Any:
                step_idx, _progress, active = advance_progress()
                if not active or guidance is None:
                    return orig_es(model_arg, x_t, u_t, u_s, clip_denoised, denoised_fn, cond_fn, model_kwargs)

                sigma_t = flow.get_sigmas(u_t)
                sigma_s = flow.get_sigmas(u_s)

                def eval_pred(is_ref: bool) -> Tensor:
                    prev = bool(step_state.get("reference_mode", False))
                    step_state["reference_mode"] = is_ref
                    try:
                        pred = model_arg(x_t, sigma_t, **(model_kwargs or {}))
                        return denoised_fn(pred) if denoised_fn is not None else pred
                    finally:
                        step_state["reference_mode"] = prev

                prev_ref, prev_loop = guidance.reference_calls, guidance.loop_calls
                model_pred = guidance.predict(
                    step_idx,
                    pred_loop_fn=lambda: eval_pred(False),
                    pred_ref_fn=lambda: eval_pred(True),
                    active=True,
                )
                stats.guidance_ref_calls += guidance.reference_calls - prev_ref
                stats.guidance_loop_calls += guidance.loop_calls - prev_loop
                delta_t = (sigma_s - sigma_t).view(-1, *([1] * (x_t.ndim - 1))).to(x_t.device)
                return x_t + delta_t * model_pred

            save_attr("euler_sample", hooked_euler_sample)

    def restore(self) -> None:
        if self._restored:
            return
        for obj, name, orig, existed in reversed(self._saved_flow):
            if existed:
                setattr(obj, name, orig)
            elif name in vars(obj):
                delattr(obj, name)
        self._saved_flow.clear()
        if self._had_step_override:
            set_sampler_step_count(self.model, None)
            self._had_step_override = False
        for block, orig, had_attr in reversed(self._saved_blocks):
            if had_attr:
                block.forward = orig
            elif "forward" in vars(block):
                delattr(block, "forward")
        self._saved_blocks.clear()
        self._restored = True

    def __exit__(self, *exc: Any) -> None:
        self.restore()


class ScaleRAEAdapter(BackboneAdapter):
    """Backbone adapter for Scale-RAE DiT / MLLM image generation models."""

    name = "scale_rae"
    modality = "image"

    def load(
        self,
        model_path: str,
        *,
        upstream_repo: str | Path | None = None,
        decoder_config: dict[str, Any] | None = None,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype | None = None,
        local_files_only: bool = True,
        **kwargs: Any,
    ) -> Any:
        repo = resolve_upstream_repo(upstream_repo, "SCALE_RAE_REPO", "Scale-RAE", bundled_subdir="scale_rae")
        activate_upstream_repo(repo, extra_subdirs=("inference",))
        builder = _install_dtype_preserving_scale_rae_loader()
        del dtype, local_files_only  # upstream loader preserves checkpoint tensor dtypes explicitly
        tokenizer, model, _image_processor, _ctx_len = builder.load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=Path(model_path).name,
            device=str(device),
            **kwargs,
        )
        model.eval()
        model._elp_tokenizer = tokenizer
        if decoder_config is not None:
            from scale_rae.model.multimodal_decoder import MultimodalDecoder

            decoder = MultimodalDecoder(**decoder_config).to(model.device)
            model._elp_decoder = decoder
        return model

    def install_program(self, model: Any, program: Any) -> AbstractContextManager[Any]:
        if program is None:
            return nullcontext()
        prog = self.resolve_program(program)
        if isinstance(prog, tuple) and len(prog) == 2 and isinstance(prog[0], ImageLoopConfig):
            return ScaleRAELoopPatch(model, prog[0], prog[1])
        if isinstance(prog, ImageLoopConfig):
            return ScaleRAELoopPatch(model, prog)
        if isinstance(prog, LoopProgram):
            return ScaleRAELoopPatch(model, program=prog)
        raise TypeError(f"Unsupported program type for ScaleRAEAdapter: {type(program).__name__}")

    def _run_prompt_generation(self, model: Any, prompt: str, *, guidance_level: float = 1.0, max_new_tokens: int = 512) -> Any:
        from scale_rae.constants import IMAGE_TOKEN_INDEX
        from scale_rae.conversation import conv_templates
        from scale_rae.mm_utils import tokenizer_image_token

        tokenizer = getattr(model, "_elp_tokenizer", None)
        if tokenizer is None:
            raise ValueError("Scale-RAE prompt inference requires an attached tokenizer (`_elp_tokenizer`)")
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_ids = input_ids.to(next(model.parameters()).device)
        output_ids, image_embeds = model.generate(
            input_ids,
            images=None,
            output_image=True,
            do_sample=True,
            temperature=0.0,
            use_customize_greedy=True,
            top_p=None,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            start_image_token_id=tokenizer.convert_tokens_to_ids("<im_start>"),
            end_image_token_id=tokenizer.convert_tokens_to_ids("<im_end>"),
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            guidance_level=guidance_level,
        )
        decoder = getattr(model, "_elp_decoder", None)
        if decoder is not None and image_embeds is not None and image_embeds.ndim > 1:
            emb = image_embeds.unsqueeze(0)
            cls_tok = torch.zeros((emb.shape[0], 1, emb.shape[-1]), device=emb.device, dtype=emb.dtype)
            imgs = decoder(torch.cat([cls_tok, emb], dim=1)).permute(0, 2, 3, 1).clamp(0, 1)
            return imgs
        return {"output_ids": output_ids, "image_embeds": image_embeds}

    @torch.no_grad()
    def infer(
        self,
        model: Any,
        inputs: Any,
        *,
        program: Any = None,
        probe: ProbeSpec | None = None,
        step_count: int | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        if isinstance(model, nn.Module):
            model.eval()
        started = time.perf_counter()
        patch_ctx = self.install_program(model, program)
        diag = DiagnosticCollector(model, probe) if probe is not None else nullcontext()

        with patch_ctx as patch, diag as collector:
            if isinstance(inputs, Tensor):
                out = model(inputs, **kwargs)
            elif isinstance(inputs, dict) and "x" in inputs:
                call_kw = {k: v for k, v in inputs.items() if k != "x"}
                call_kw.update(kwargs)
                out = model(inputs["x"], **call_kw)
            elif isinstance(inputs, dict) and "prompt" in inputs and getattr(model, "_elp_tokenizer", None) is not None:
                out = self._run_prompt_generation(model, str(inputs["prompt"]), **kwargs)
            elif isinstance(inputs, dict) and hasattr(model, "sample"):
                out = model.sample(**inputs, **kwargs)
            elif hasattr(model, "generate"):
                out = model.generate(inputs, **kwargs)
            else:
                out = model(inputs, **kwargs)

        latency = time.perf_counter() - started
        stats = patch.stats.to_dict() if isinstance(patch, ScaleRAELoopPatch) else {}
        diag_rows = collector.rows() if isinstance(collector, DiagnosticCollector) else []
        used_steps = step_count or (patch.step_count if isinstance(patch, ScaleRAELoopPatch) and patch.step_count else 1)
        return InferenceResult(
            output=out,
            latency_s=latency,
            step_count=int(used_steps),
            loop_stats=stats,
            diagnostics=diag_rows,
        )


__all__ = [
    "ScaleRAEAdapter",
    "ScaleRAELoopPatch",
    "resolve_scale_rae_backbone",
    "resolve_scale_rae_blocks",
    "set_sampler_step_count",
]
