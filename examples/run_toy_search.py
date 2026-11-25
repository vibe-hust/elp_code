"""End-to-end CPU demonstration of ELP closed-loop search, programmable diagnostics, and freeze/replay."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
from typing import Any

# Allow running directly as `python examples/run_toy_search.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import Tensor, nn

from elp import (
    DiagnosticCollector,
    DiagnosticRequest,
    EvalResult,
    EvaluateRequest,
    LoopProgram,
    ProbeSpec,
    SearchConfig,
    SearchCoordinator,
    invoke_loop_program,
    load_frozen_program,
    load_program_callable,
)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, scale: float) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim) * (1.0 - scale) + 0.05 * torch.randn(dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        return 0.7 * x + 0.3 * torch.tanh(self.proj(x))


class ToyTransformer(nn.Module):
    def __init__(self, depth: int = 6, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(dim, scale=0.25) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


_SINGLE_PASS_EULER = '''import torch

def loop(x, evaluate, params):
    alpha = float(params["alpha"])
    r0 = evaluate(x) - x
    r1 = evaluate(x + alpha * r0) - (x + alpha * r0)
    return x + alpha * r0 + (1.0 - alpha) * r1
'''

_DAMPED_TWO_PASS = '''import math
import torch

def loop(x, evaluate, params):
    h = float(params["h"])
    w = float(params["blend"])
    r0 = evaluate(x) - x
    r1 = evaluate(x + h * r0) - (x + h * r0)
    return x + w * r0 + (1.0 - w) * r1
'''

_CUSTOM_COLLECTOR = '''import torch

def collect(layer_idx, pass_idx, x, y, info):
    del info, pass_idx
    delta = y - x
    return {
        "custom_delta_rms": float(torch.sqrt(torch.mean(delta * delta))),
        "custom_layer_id": float(layer_idx),
    }
'''


class ToyEvaluator:
    def __init__(self, model: ToyTransformer, inputs: Tensor, target: Tensor) -> None:
        self.model = model
        self.inputs = inputs
        self.target = target

    @torch.no_grad()
    def _run_with_program(self, program: LoopProgram) -> tuple[Tensor, int]:
        fn = load_program_callable(program.code, entrypoint="loop")
        target_layers = set(program.metadata.get("layers", []))
        eval_count = 0
        x = self.inputs
        for idx, block in enumerate(self.model.blocks):
            if idx not in target_layers:
                eval_count += 1
                x = block(x)
                continue

            def evaluate(state: Tensor, _b=block) -> Tensor:
                nonlocal eval_count
                eval_count += 1
                return _b(state)

            x = invoke_loop_program(fn, x, evaluate, info={"layer": idx}, params=program.params, memory={})
        return x, eval_count

    @torch.no_grad()
    def control_digest(self) -> str:
        out = self.model(self.inputs)
        return hashlib.sha256(str(out.detach().cpu().float().sum().item()).encode("utf-8")).hexdigest()

    @torch.no_grad()
    def diagnose(self, request: DiagnosticRequest) -> dict[str, Any]:
        spec = ProbeSpec(
            layers=tuple(request.probe["layers"]),
            features=tuple(request.probe["features"]),
            collector_code=request.collector_code,
        )
        with DiagnosticCollector(self.model, spec) as diag:
            _ = self.model(self.inputs)
        return {"question": request.question, "rows": diag.rows()}

    def evaluate(self, program: LoopProgram) -> EvalResult:
        out, calls = self._run_with_program(program)
        mse = float((out - self.target).square().mean())
        quality = 1.0 / (1.0 + mse)
        speed = len(self.model.blocks) / float(calls)
        return EvalResult(
            program_id=program.id,
            status="complete",
            objectives=(quality, speed),
            metrics={"quality": quality, "speed": speed, "block_calls": float(calls)},
        )


class ToyProposer:
    def __init__(self) -> None:
        self._did_diag = False

    def propose(self, observation: dict[str, Any], requested_count: int) -> DiagnosticRequest | EvaluateRequest:
        r = observation["round"]
        if r == 0 and not self._did_diag:
            self._did_diag = True
            return DiagnosticRequest(
                round_index=0,
                question="Measure residual-to-input ratios and custom delta RMS across blocks 0..5.",
                probe={"layers": [0, 1, 2, 3, 4, 5], "features": ["activation", "residual", "alignment"]},
                collector_code=_CUSTOM_COLLECTOR,
            )

        if r == 0:
            candidates = [
                LoopProgram(
                    id="r0_mid_euler_a04",
                    code=_SINGLE_PASS_EULER,
                    params={"alpha": 0.4},
                    hypothesis="Test middle blocks [2, 3] with alpha=0.4",
                    metadata={"layers": [2, 3]},
                ),
                LoopProgram(
                    id="r0_mid_euler_a05",
                    code=_SINGLE_PASS_EULER,
                    params={"alpha": 0.5},
                    hypothesis="Test middle blocks [2, 3] with symmetric alpha=0.5",
                    metadata={"layers": [2, 3]},
                ),
                LoopProgram(
                    id="r0_late_euler_a05",
                    code=_SINGLE_PASS_EULER,
                    params={"alpha": 0.5},
                    hypothesis="Test single late block [4] for lower overhead",
                    metadata={"layers": [4]},
                ),
                LoopProgram(
                    id="r0_band_euler_a03",
                    code=_SINGLE_PASS_EULER,
                    params={"alpha": 0.3},
                    hypothesis="Test broader band [1, 2, 3, 4] for maximum refinement",
                    metadata={"layers": [1, 2, 3, 4]},
                ),
            ]
            return EvaluateRequest(round_index=0, rationale="Initial structured band exploration.", programs=candidates[:requested_count])

        candidates = [
            LoopProgram(
                id="r1_mid_blend_03",
                code=_DAMPED_TWO_PASS,
                params={"h": 0.5, "blend": 0.5},
                hypothesis="Looped midpoint refinement across middle blocks [2, 3]",
                metadata={"layers": [2, 3]},
            ),
            LoopProgram(
                id="r1_early_late_blend",
                code=_DAMPED_TWO_PASS,
                params={"h": 0.5, "blend": 0.5},
                hypothesis="Refine [2] only for maximum speed trade-off",
                metadata={"layers": [2]},
            ),
            LoopProgram(
                id="r1_full_band_ref",
                code=_DAMPED_TWO_PASS,
                params={"h": 0.5, "blend": 0.5},
                hypothesis="Refine [0..5] matching deep target fixed point",
                metadata={"layers": [0, 1, 2, 3, 4, 5]},
            ),
            LoopProgram(
                id="r1_deep_blend",
                code=_DAMPED_TWO_PASS,
                params={"h": 0.5, "blend": 0.5},
                hypothesis="Mid-band refinement across [1, 2, 3, 4] for balanced frontier",
                metadata={"layers": [1, 2, 3, 4]},
            ),
        ]
        return EvaluateRequest(round_index=r, rationale="Refine update weights on frontier layers.", programs=candidates[:requested_count])


def main() -> None:
    torch.manual_seed(0)
    model = ToyTransformer(depth=6, dim=16).eval()
    inputs = torch.randn(4, 12, 16)
    with torch.no_grad():
        base_out = model(inputs)
        # Synthesize target matching two-pass 0.5-blend refinement on blocks 0..5
        target = inputs
        for blk in model.blocks:
            r0 = blk(target) - target
            r1 = blk(target + 0.5 * r0) - (target + 0.5 * r0)
            target = target + 0.5 * r0 + 0.5 * r1

    base_mse = float((base_out - target).square().mean())
    evaluator = ToyEvaluator(model, inputs, target)
    coordinator = SearchCoordinator(
        proposer=ToyProposer(),
        evaluator=evaluator,
        config=SearchConfig(round_sizes=(4, 4), reference_point=(0.0, 0.0)),
        baseline_metrics={"quality": 1.0 / (1.0 + base_mse), "speed": 1.0},
        control_check=evaluator.control_digest,
    )

    with tempfile.TemporaryDirectory() as tmp:
        result = coordinator.run(freeze_dir=tmp)
        print(f"Completed programs : {len(result.completed)}")
        print(f"Diagnostic turns   : {len(result.diagnostics)}")
        print(f"Hypervolume / rnd  : {[round(x, 6) for x in result.hypervolume_history]}")
        print(f"Pareto frontier    : {result.pareto_frontier}")
        print(f"Representatives    : {result.representatives}")

        bal_id = result.representatives["balanced"]
        frozen_path = Path(tmp) / f"balanced_{bal_id}.json"
        reloaded = load_frozen_program(frozen_path)
        replay_eval = evaluator.evaluate(reloaded)
        assert replay_eval.objectives == result.completed[bal_id][1].objectives
        print(f"Frozen replay OK   : {reloaded.id} objectives={replay_eval.objectives}")


if __name__ == "__main__":
    main()
