"""Isolated subprocess evaluator, diagnostic probe & control worker (`python -m elp.agent.worker`)."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from elp.core import (
    DiagnosticCollector,
    LoopProgram,
    ProbeSpec,
    invoke_loop_program,
    load_program_callable,
    validate_program_source,
)


class _ToyLayer(nn.Module):
    """Deterministic residual transformer-like block for CPU closed-loop search."""

    def __init__(self, dim: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.proj = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            delta = 0.08 * torch.randn(dim, dim, generator=gen)
            self.proj.weight.copy_(torch.eye(dim) * 0.96 + delta)

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.35 * torch.tanh(self.proj(x) - x)


class _ToyBackbone(nn.Module):
    def __init__(self, depth: int = 4, dim: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer(dim, seed=100 + i) for i in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        cur = x
        for layer in self.layers:
            cur = layer(cur)
        return cur


def _toy_data(batch: int = 8, seq: int = 12, dim: int = 16) -> tuple[Tensor, Tensor]:
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(batch, seq, dim, generator=gen)
    # Deeper contraction acts as clean teacher target for quality evaluation
    teacher = _ToyBackbone(depth=4, dim=dim).eval()
    target = x
    with torch.no_grad():
        for idx, layer in enumerate(teacher.layers):
            reps = 2 if idx in (1, 2) else 1
            for _ in range(reps):
                y = layer(target)
                target = target + 0.5 * (y - target)
    return x, target


def load_scorer(spec_str: str) -> Callable[..., Any]:
    """Load a scorer callable from `module.path:attr` specification."""
    raw = str(spec_str or "").strip()
    if ":" not in raw:
        raise ValueError(f"Scorer specification must use `module.path:callable` format, got {spec_str!r}")
    module_name, attr_path = raw.split(":", 1)
    module_name = module_name.strip()
    attr_path = attr_path.strip()
    if not module_name or not attr_path:
        raise ValueError(f"Invalid scorer specification: {spec_str!r}")
    obj: Any = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"Scorer target {spec_str!r} is not callable (got {type(obj).__name__})")
    return obj


def _load_eval_inputs(spec: dict[str, Any]) -> Any:
    raw_path = spec.get("eval_inputs_path")
    if raw_path:
        path = Path(str(raw_path))
        if not path.is_file():
            raise FileNotFoundError(f"eval_inputs_path does not exist: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"eval_inputs_path is empty: {path}")
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        return json.loads(text)
    return spec.get("inputs", {"prompt": "Describe how iterative refinement works."})


def _hash_object(hasher: "hashlib._Hash", obj: Any) -> None:
    if isinstance(obj, Tensor):
        t = obj.detach().cpu().contiguous()
        hasher.update(f"T:{tuple(t.shape)}:{t.dtype}".encode("utf-8"))
        if t.is_floating_point() and t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        flat = t.reshape(-1)
        if flat.numel() <= 16384:
            hasher.update(json.dumps(flat.tolist()).encode("utf-8"))
        else:
            step = max(1, flat.numel() // 4096)
            sample = flat[::step][:4096]
            f_vals = flat.to(torch.float64)
            stats = [
                float(f_vals.sum()),
                float(torch.linalg.vector_norm(f_vals)),
                float(f_vals.min()),
                float(f_vals.max()),
            ]
            hasher.update(json.dumps({"sample": sample.tolist(), "stats": stats}).encode("utf-8"))
        return
    if obj is None or isinstance(obj, (bool, int, float, str)):
        hasher.update(json.dumps(obj).encode("utf-8"))
        return
    if isinstance(obj, (bytes, bytearray)):
        hasher.update(bytes(obj))
        return
    if isinstance(obj, dict):
        hasher.update(b"{")
        for key in sorted(obj.keys(), key=str):
            if str(key) == "past_key_values":
                continue
            hasher.update(str(key).encode("utf-8"))
            _hash_object(hasher, obj[key])
        hasher.update(b"}")
        return
    if isinstance(obj, (list, tuple)):
        hasher.update(b"[")
        for item in obj:
            _hash_object(hasher, item)
        hasher.update(b"]")
        return
    if hasattr(obj, "logits") and isinstance(getattr(obj, "logits"), Tensor):
        hasher.update(b"logits:")
        _hash_object(hasher, getattr(obj, "logits"))
        return
    if hasattr(obj, "tobytes"):
        # e.g., PIL Image or numpy buffer
        try:
            hasher.update(f"B:{getattr(obj, 'size', '')}".encode("utf-8"))
            hasher.update(obj.tobytes())
            return
        except Exception:
            pass
    hasher.update(type(obj).__qualname__.encode("utf-8"))


def _compute_control_digest(model: Any, output: Any) -> str:
    hasher = hashlib.sha256()
    _hash_object(hasher, output)
    if isinstance(model, nn.Module):
        for name, param in sorted(model.named_parameters()):
            p = param.detach().cpu()
            p_f = p.to(torch.float64)
            hasher.update(name.encode("utf-8"))
            hasher.update(
                json.dumps([
                    list(p.shape),
                    float(p_f.sum().item()) if p_f.numel() else 0.0,
                    float(torch.linalg.vector_norm(p_f).item()) if p_f.numel() else 0.0,
                ]).encode("utf-8")
            )
    return hasher.hexdigest()


def _infer_with_inputs(
    adapter: Any,
    model: Any,
    inputs: Any,
    *,
    program: Any = None,
    probe: ProbeSpec | None = None,
):
    from elp.backends.base import InferenceResult

    if isinstance(inputs, list):
        if not inputs:
            raise ValueError("Evaluation input list must not be empty")
        if len(inputs) == 1:
            return adapter.infer(model, inputs[0], program=program, probe=probe)
        results = [adapter.infer(model, item, program=program, probe=probe) for item in inputs]
        agg_stats: dict[str, Any] = {}
        for res in results:
            for k, v in res.loop_stats.items():
                if isinstance(v, (int, float)):
                    agg_stats[k] = float(agg_stats.get(k, 0.0)) + float(v)
        agg_diags: list[dict[str, Any]] = []
        for idx, res in enumerate(results):
            for row in res.diagnostics:
                entry = dict(row)
                entry.setdefault("item_index", idx)
                agg_diags.append(entry)
        return InferenceResult(
            output=[res.output for res in results],
            latency_s=sum(res.latency_s for res in results),
            step_count=sum(res.step_count for res in results),
            loop_stats=agg_stats,
            diagnostics=agg_diags,
            metadata={"per_item": results, "num_items": len(results)},
        )
    return adapter.infer(model, inputs, program=program, probe=probe)


def _evaluate_toy(spec: dict[str, Any]) -> dict[str, Any]:
    prog_dict = spec["program"]
    program = LoopProgram(
        id=prog_dict["id"],
        code=prog_dict["code"],
        params=prog_dict.get("params", {}),
        hypothesis=prog_dict.get("hypothesis", ""),
        metadata=prog_dict.get("metadata", {}),
    )
    loop_fn = load_program_callable(program.code)
    model = _ToyBackbone(depth=4, dim=16).eval()
    x, target = _toy_data()

    op_calls = 0
    started = time.perf_counter()
    with torch.no_grad():
        cur = x
        for idx, layer in enumerate(model.layers):
            if idx in (1, 2):
                memory: dict[str, Any] = {}

                def evaluate(state: Tensor, _layer=layer) -> Tensor:
                    nonlocal op_calls
                    op_calls += 1
                    return _layer(state)

                cur = invoke_loop_program(
                    loop_fn,
                    cur,
                    evaluate,
                    info={"layer": idx},
                    params=dict(program.params),
                    memory=memory,
                )
            else:
                op_calls += 1
                cur = layer(cur)
    elapsed_s = max(1e-6, time.perf_counter() - started)

    mse = float(F.mse_loss(cur, target))
    cos = float(F.cosine_similarity(cur.flatten(), target.flatten(), dim=0))
    quality = max(0.0, round(cos - 0.5 * mse, 6))
    baseline_calls = 4.0
    speedup = round(baseline_calls / max(1.0, float(op_calls)), 6)
    return {
        "program_id": program.id,
        "status": "complete",
        "objectives": [quality, speedup],
        "metrics": {
            "quality": quality,
            "cosine": round(cos, 6),
            "mse": round(mse, 6),
            "operator_calls": float(op_calls),
            "speedup": speedup,
            "wall_seconds": round(elapsed_s, 6),
        },
    }


def _diagnose_toy(spec: dict[str, Any]) -> dict[str, Any]:
    probe_cfg = spec.get("probe", {})
    layers = tuple(int(i) for i in probe_cfg.get("layers", [0, 1, 2, 3]))
    features = tuple(probe_cfg.get("features", ["activation", "residual", "alignment", "repeat"]))
    collector_code = probe_cfg.get("collector_code") or spec.get("collector_code")
    model = _ToyBackbone(depth=4, dim=16).eval()
    x, _ = _toy_data()
    probe = ProbeSpec(
        layers=layers,
        features=features,
        max_tokens=int(probe_cfg.get("max_tokens", 128)),
        collector_code=collector_code,
    )
    with torch.no_grad(), DiagnosticCollector(model, probe) as diag:
        _ = model(x)
    return {
        "status": "complete",
        "question": spec.get("question", ""),
        "rows": diag.rows(),
    }


def _control_toy(spec: dict[str, Any]) -> dict[str, Any]:
    del spec
    model = _ToyBackbone(depth=4, dim=16).eval()
    x, _ = _toy_data()
    with torch.no_grad():
        out = model(x)
    return {
        "status": "complete",
        "digest": _compute_control_digest(model, out),
    }


def _normalize_scorer_output(scored: Any, default_speedup: float) -> tuple[list[float], dict[str, float]]:
    if isinstance(scored, bool):
        raise TypeError("Scorer returned bool; expected float/int or dict with 'objectives'/'quality'")
    if isinstance(scored, (int, float)):
        q = float(scored)
        if not math.isfinite(q):
            raise ValueError(f"Scorer quality must be finite, got {q!r}")
        return [q, default_speedup], {"quality": q}
    if not isinstance(scored, dict):
        raise TypeError(f"Unsupported scorer return type: {type(scored).__name__}")

    raw_metrics = scored.get("metrics") or {}
    if not isinstance(raw_metrics, dict):
        raise TypeError("Scorer 'metrics' field must be a dict")
    metrics: dict[str, float] = {}
    for k, v in raw_metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            val = float(v)
            if not math.isfinite(val):
                raise ValueError(f"Non-finite scorer metric {k!r}={val!r}")
            metrics[str(k)] = val

    if "objectives" in scored and scored["objectives"] is not None:
        raw_objs = scored["objectives"]
        if not isinstance(raw_objs, (list, tuple)) or not raw_objs:
            raise ValueError("Scorer 'objectives' must be a non-empty list/tuple of floats")
        objectives = [float(v) for v in raw_objs]
    elif "quality" in scored:
        q = float(scored["quality"])
        sp = float(scored.get("speedup", metrics.get("speedup", default_speedup)))
        objectives = [q, sp]
        metrics.setdefault("quality", q)
    elif "quality" in metrics:
        q = float(metrics["quality"])
        sp = float(metrics.get("speedup", default_speedup))
        objectives = [q, sp]
    else:
        raise ValueError("Scorer dict output must contain `'objectives'` or `'quality'`")

    if any(not math.isfinite(v) for v in objectives):
        raise ValueError(f"Scorer returned non-finite objective values: {objectives!r}")
    metrics.setdefault("quality", objectives[0])
    return objectives, metrics


def _evaluate_real_backbone(spec: dict[str, Any]) -> dict[str, Any]:
    from elp.backends import get_adapter

    backbone_name = spec["backbone"]
    model_path = spec.get("model_path")
    if not model_path:
        raise ValueError(f"Real backbone {backbone_name!r} requires `model_path` in worker spec")
    scorer_spec = spec.get("scorer")
    if not scorer_spec:
        raise ValueError("Real backbone search requires a `scorer` (`module:callable`) to evaluate task quality")
    scorer_fn = load_scorer(str(scorer_spec))

    prog_dict = spec["program"]
    program = LoopProgram(
        id=prog_dict["id"],
        code=prog_dict["code"],
        params=prog_dict.get("params", {}),
        hypothesis=prog_dict.get("hypothesis", ""),
        metadata=prog_dict.get("metadata", {}),
    )
    validate_program_source(program.code)

    adapter = get_adapter(backbone_name)
    model = adapter.load(
        model_path,
        device=spec.get("device", "cpu"),
        **spec.get("load_kwargs", {}),
    )
    inputs = _load_eval_inputs(spec)
    base_res = _infer_with_inputs(adapter, model, inputs, program=None)
    prog_res = _infer_with_inputs(adapter, model, inputs, program=program)
    speedup = round(max(1e-6, base_res.latency_s) / max(1e-6, prog_res.latency_s), 6)

    scored = scorer_fn(
        adapter=adapter,
        model=model,
        program=program,
        baseline_result=base_res,
        program_result=prog_res,
        inputs=inputs,
    )
    objectives, scorer_metrics = _normalize_scorer_output(scored, speedup)
    metrics: dict[str, float] = {
        "speedup": speedup,
        "latency_s": round(prog_res.latency_s, 6),
        "baseline_latency_s": round(base_res.latency_s, 6),
        "step_count": float(prog_res.step_count),
        **{k: float(v) for k, v in prog_res.loop_stats.items() if isinstance(v, (int, float))},
        **scorer_metrics,
    }
    return {
        "program_id": program.id,
        "status": "complete",
        "objectives": objectives,
        "metrics": metrics,
    }


def _diagnose_real_backbone(spec: dict[str, Any]) -> dict[str, Any]:
    from elp.backends import get_adapter

    backbone_name = spec["backbone"]
    model_path = spec.get("model_path")
    if not model_path:
        raise ValueError(f"Real backbone {backbone_name!r} diagnostic probe requires `model_path`")
    adapter = get_adapter(backbone_name)
    model = adapter.load(
        model_path,
        device=spec.get("device", "cpu"),
        **spec.get("load_kwargs", {}),
    )
    probe_cfg = spec.get("probe", {})
    collector_code = probe_cfg.get("collector_code") or spec.get("collector_code")
    probe = ProbeSpec(
        layers=tuple(int(i) for i in probe_cfg.get("layers", [0, 1])),
        features=tuple(probe_cfg.get("features", ["activation", "residual", "alignment", "repeat"])),
        max_tokens=int(probe_cfg.get("max_tokens", 128)),
        collector_code=collector_code,
    )
    inputs = _load_eval_inputs(spec)
    res = _infer_with_inputs(adapter, model, inputs, program=None, probe=probe)
    return {
        "status": "complete",
        "question": spec.get("question", ""),
        "rows": res.diagnostics,
    }


def _control_real_backbone(spec: dict[str, Any]) -> dict[str, Any]:
    from elp.backends import get_adapter

    backbone_name = spec["backbone"]
    model_path = spec.get("model_path")
    if not model_path:
        raise ValueError(f"Real backbone {backbone_name!r} control check requires `model_path`")
    torch.manual_seed(0)
    adapter = get_adapter(backbone_name)
    model = adapter.load(
        model_path,
        device=spec.get("device", "cpu"),
        **spec.get("load_kwargs", {}),
    )
    inputs = _load_eval_inputs(spec)
    base_res = _infer_with_inputs(adapter, model, inputs, program=None)
    return {
        "status": "complete",
        "digest": _compute_control_digest(model, base_res.output),
    }


def run_worker(mode: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Execute one isolated worker action (`evaluate`, `diagnose`, or `control`) cleanly."""
    backbone = str(spec.get("backbone", "toy"))
    try:
        if mode == "evaluate":
            if backbone == "toy":
                return _evaluate_toy(spec)
            return _evaluate_real_backbone(spec)
        if mode == "diagnose":
            if backbone == "toy":
                return _diagnose_toy(spec)
            return _diagnose_real_backbone(spec)
        if mode == "control":
            if backbone == "toy":
                return _control_toy(spec)
            return _control_real_backbone(spec)
        raise ValueError(f"Unsupported worker mode: {mode!r}")
    except Exception as exc:
        prog_id = (spec.get("program") or {}).get("id", "unknown")
        return {
            "program_id": prog_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated ELP candidate evaluation, probe & control worker.")
    parser.add_argument("--mode", choices=("evaluate", "diagnose", "control"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = run_worker(args.mode, spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
