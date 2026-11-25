"""Unified CLI entrypoint (`elp list`, `elp infer`, `elp probe`, `elp search`)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor

from elp.agent import AgentSearchRunner
from elp.backends import get_adapter, list_backbones, resolve_dtype
from elp.core import ProbeSpec, SearchConfig, load_frozen_program


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    vals = [part.strip() for part in raw.split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers")
    return tuple(int(v) for v in vals)


def _cmd_list(args: argparse.Namespace) -> int:
    payload = {"backbones": list_backbones()}
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print("Registered Backbones:")
    for item in payload["backbones"]:
        print(
            f"  - {item['name']:14s} [{item['modality']:5s}]  "
            f"loading={item['loading']}  upstream={item['upstream']}  :: {item['description']}"
        )
    return 0


def _resolve_program_arg(raw: str | None) -> Any:
    if raw is None:
        return None
    if raw.strip().lower() in {"none", "baseline"}:
        return None
    candidate_path = Path(raw)
    if not candidate_path.is_file():
        raise FileNotFoundError(
            f"Frozen program JSON file not found: {raw!r}. "
            "Pass a frozen JSON produced by `elp search` or omit `--program` / pass `baseline`."
        )
    return load_frozen_program(candidate_path)


def _tensor_to_uint8_hwc(tensor: Tensor) -> Tensor:
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 3 and t.shape[0] in (1, 3, 4):
        t = t.permute(1, 2, 0)
    if t.is_floating_point():
        t = t.float()
        if float(t.min().item()) < -0.05:
            t = (t.clamp(-1.0, 1.0) + 1.0) * 127.5
        elif float(t.max().item()) <= 1.5:
            t = t.clamp(0.0, 1.0) * 255.0
        else:
            t = t.clamp(0.0, 255.0)
        t = t.round().to(torch.uint8)
    else:
        t = t.clamp(0, 255).to(torch.uint8)
    if t.ndim == 3 and t.shape[-1] == 1:
        t = t.repeat(1, 1, 3)
    return t


def _save_inference_output(output: Any, out_path: Path, modality: str) -> str:
    """Persist inference artifacts (`.txt`, `.png`/`.jpg`, `.mp4`, or `.pt`)."""
    del modality
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    if suffix == ".pt":
        torch.save(output, out_path)
        return str(out_path)

    if isinstance(output, str) or suffix in {".txt", ".md"}:
        text = str(output)
        out_path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
        return str(out_path)

    if hasattr(output, "save") and callable(output.save):
        output.save(out_path)
        return str(out_path)

    if isinstance(output, Tensor) and suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Saving image outputs as PNG/JPG requires `Pillow`; install `Pillow` or save to `.pt`."
            ) from exc
        hwc = _tensor_to_uint8_hwc(output)
        Image.fromarray(hwc.numpy()).save(out_path)
        return str(out_path)

    if suffix in {".mp4", ".gif"}:
        if isinstance(output, Tensor):
            try:
                import imageio.v3 as iio  # type: ignore[import-not-found]
            except ImportError:
                iio = None
            if iio is not None:
                vid = output.detach().cpu()
                if vid.ndim == 5:
                    vid = vid[0]
                if vid.ndim == 4 and vid.shape[0] in (1, 3, 4):
                    vid = vid.permute(1, 0, 2, 3)
                frames = [_tensor_to_uint8_hwc(frame).numpy() for frame in vid]
                iio.imwrite(out_path, frames, fps=16)
                return str(out_path)
            try:
                import torchvision.io as tvio  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "Saving video outputs as .mp4/.gif requires `imageio` or `torchvision`; "
                    "pass `--save-output out.pt` to save raw video tensors."
                ) from exc
            vid = output.detach().cpu()
            if vid.ndim == 5:
                vid = vid[0]
            if vid.ndim == 4 and vid.shape[0] in (1, 3, 4):
                vid = vid.permute(1, 0, 2, 3)
            frames = torch.stack([_tensor_to_uint8_hwc(frame) for frame in vid], dim=0)
            tvio.write_video(str(out_path), frames, fps=16)
            return str(out_path)
        raise RuntimeError(f"Cannot export {type(output).__name__} to {suffix}; use `--save-output out.pt`.")

    if isinstance(output, Tensor):
        torch.save(output, out_path)
        return str(out_path)

    out_path.write_text(json.dumps({"output": repr(output)}, indent=2) + "\n", encoding="utf-8")
    return str(out_path)


def _cmd_infer(args: argparse.Namespace) -> int:
    adapter = get_adapter(args.backbone)
    torch_dtype = resolve_dtype(args.dtype)
    load_kw: dict[str, Any] = {
        "device": args.device,
        "dtype": torch_dtype,
        "local_files_only": not args.allow_download,
    }
    if args.upstream_repo:
        load_kw["upstream_repo"] = args.upstream_repo
    model = adapter.load(args.model_path, **load_kw)
    program = _resolve_program_arg(args.program)

    inputs: Any = {"prompt": args.prompt}
    extra: dict[str, Any] = {}
    if adapter.modality == "text":
        extra["max_new_tokens"] = int(args.max_new_tokens)
    elif args.steps is not None:
        extra["step_count"] = int(args.steps)

    result = adapter.infer(model, inputs, program=program, **extra)
    summary: dict[str, Any] = {
        "backbone": adapter.name,
        "modality": adapter.modality,
        "latency_s": round(result.latency_s, 6),
        "step_count": result.step_count,
        "loop_stats": result.loop_stats,
        "metadata": result.metadata,
    }
    if args.save_output:
        summary["saved_output"] = _save_inference_output(result.output, Path(args.save_output), adapter.modality)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    adapter = get_adapter(args.backbone)
    torch_dtype = resolve_dtype(args.dtype)
    load_kw: dict[str, Any] = {
        "device": args.device,
        "dtype": torch_dtype,
        "local_files_only": not args.allow_download,
    }
    if args.upstream_repo:
        load_kw["upstream_repo"] = args.upstream_repo
    model = adapter.load(args.model_path, **load_kw)
    layers = _parse_int_csv(args.layers)
    features = tuple(part.strip() for part in args.features.split(",") if part.strip())
    collector_code = None
    if args.collector_file:
        cpath = Path(args.collector_file)
        if not cpath.is_file():
            raise FileNotFoundError(f"Collector file not found: {cpath}")
        collector_code = cpath.read_text(encoding="utf-8")
    probe = ProbeSpec(layers=layers, features=features, collector_code=collector_code)
    inputs: Any = {"prompt": args.prompt}
    extra = {"max_new_tokens": 1} if adapter.modality == "text" else {}

    res = adapter.infer(model, inputs, program=None, probe=probe, **extra)
    print(json.dumps({"backbone": adapter.name, "rows": res.diagnostics}, indent=2, ensure_ascii=False))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    rounds = _parse_int_csv(args.rounds)
    if args.backbone != "toy":
        adapter = get_adapter(args.backbone)
        modality = adapter.modality
        canonical_backbone = adapter.name
        if not args.model_path:
            raise ValueError(f"`--model-path` is required when searching real backbone {canonical_backbone!r}")
    else:
        modality = "toy"
        canonical_backbone = "toy"

    runner = AgentSearchRunner(
        workspace=args.workspace,
        backbone=canonical_backbone,
        modality=modality,
        agent_backend=args.agent,
        agent_model=args.model,
        cli_bin=args.cli_bin,
        base_url=args.base_url,
        model_path=args.model_path,
        device=args.device,
        search_config=SearchConfig(round_sizes=rounds, reference_point=(0.0, 0.0)),
        freeze_dir=args.freeze_dir,
        worker_timeout_s=int(args.worker_timeout),
        scorer=args.scorer,
        eval_inputs_path=args.eval_inputs,
        upstream_repo=args.upstream_repo,
        resume=bool(args.resume),
        control_check=bool(args.control_check),
    )
    res = runner.run()
    summary = {
        "backbone": canonical_backbone,
        "agent": args.agent,
        "workspace": str(Path(args.workspace).resolve()),
        "completed": len(res.completed),
        "diagnostics": len(res.diagnostics),
        "pareto_frontier": res.pareto_frontier,
        "hypervolume_history": res.hypervolume_history,
        "representatives": res.representatives,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elp",
        description="Evolving Loop Programs (ELP): Closed-loop search, diagnostics, and frozen-program replay.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List registered backbones.")
    p_list.add_argument("--json", action="store_true", help="Print JSON output.")

    p_infer = sub.add_parser("infer", help="Run frozen-weight inference with baseline or a searched frozen JSON.")
    p_infer.add_argument("--backbone", required=True, help="Backbone name (see `elp list`).")
    p_infer.add_argument("--model-path", required=True, help="HuggingFace repo id or local model directory.")
    p_infer.add_argument("--program", default=None, help="Path to searched frozen JSON program, or `baseline`.")
    p_infer.add_argument("--prompt", default="A crisp technical studio photograph of an hourglass.")
    p_infer.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_infer.add_argument("--dtype", default=None, help="float32, float16, or bfloat16.")
    p_infer.add_argument("--max-new-tokens", type=int, default=64)
    p_infer.add_argument("--steps", type=int, default=None, help="Optional denoising step override for image/video backbones.")
    p_infer.add_argument("--upstream-repo", default=None, help="Optional local path to cloned upstream model repository.")
    p_infer.add_argument("--allow-download", action="store_true", help="Allow HF Hub downloads.")
    p_infer.add_argument("--output", default=None, help="Optional path to save JSON inference summary.")
    p_infer.add_argument("--save-output", default=None, help="Optional path to save generated output (.txt/.png/.mp4/.pt).")

    p_probe = sub.add_parser("probe", help="Collect read-only layer diagnostic statistics (Eq. 3).")
    p_probe.add_argument("--backbone", required=True)
    p_probe.add_argument("--model-path", required=True)
    p_probe.add_argument("--layers", default="0,4,8", help="Comma-separated layer indices.")
    p_probe.add_argument("--features", default="activation,residual,alignment,repeat")
    p_probe.add_argument("--collector-file", default=None, help="Optional Python file defining `collect(layer_idx, pass_idx, x, y, info)`.")
    p_probe.add_argument("--prompt", default="Step-by-step check of layer activations.")
    p_probe.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_probe.add_argument("--dtype", default=None)
    p_probe.add_argument("--upstream-repo", default=None, help="Optional local path to cloned upstream model repository.")
    p_probe.add_argument("--allow-download", action="store_true")

    p_search = sub.add_parser("search", help="Launch a sandboxed closed-loop ELP agent search.")
    p_search.add_argument("--backbone", default="toy", help="Registered backbone name or `toy` CPU demo.")
    p_search.add_argument("--agent", choices=("codex", "openai", "cli", "mock"), default="mock")
    p_search.add_argument("--workspace", required=True, help="Directory for isolated workspace, logs, and frozen programs.")
    p_search.add_argument("--rounds", default="4,4", help="Comma-separated per-round candidate counts.")
    p_search.add_argument("--model", default=None, help="Search LLM model name (for `openai` backend).")
    p_search.add_argument("--cli-bin", default=None, help="Executable name/path (for `codex` or `cli` backend).")
    p_search.add_argument("--base-url", default=None, help="Custom base URL for OpenAI-compatible API.")
    p_search.add_argument("--model-path", default=None, help="Target model path/HF repo (required if backbone != toy).")
    p_search.add_argument("--upstream-repo", default=None, help="Optional local path to cloned upstream model repository.")
    p_search.add_argument("--scorer", default=None, help="Required for real backbones: `module.path:callable` returning quality/objectives.")
    p_search.add_argument("--eval-inputs", default=None, help="Optional JSON/JSONL file path with evaluation inputs.")
    p_search.add_argument("--freeze-dir", default=None, help="Override directory for frozen Pareto representatives.")
    p_search.add_argument("--device", default="cpu")
    p_search.add_argument("--worker-timeout", type=int, default=300)
    p_search.add_argument("--resume", action="store_true", help="Resume from persisted state/trials.jsonl and diagnostics.jsonl.")
    p_search.add_argument("--control-check", action="store_true", help="Verify unlooped baseline digest invariance before/after each round.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return _cmd_list(args)
    if args.command == "infer":
        return _cmd_infer(args)
    if args.command == "probe":
        return _cmd_probe(args)
    if args.command == "search":
        return _cmd_search(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
