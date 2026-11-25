"""Closed-loop ELP search coordinator with diagnostics, control guards, and resume support."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .pareto import hypervolume, pareto_frontier, select_representatives
from .program import (
    LoopProgram,
    freeze_program,
    program_fingerprint,
    validate_program_source,
)


@dataclass(frozen=True)
class DiagnosticRequest:
    """On-demand diagnostic probe request executed outside candidate budget slots."""

    round_index: int
    question: str
    probe: dict[str, Any]
    actions: tuple[dict[str, Any], ...] = ()
    collector_code: str | None = None


@dataclass(frozen=True)
class EvaluateRequest:
    """Batch of candidate LoopPrograms proposed for evaluation in the current round."""

    round_index: int
    rationale: str
    programs: list[LoopProgram]


ProposalResponse = DiagnosticRequest | EvaluateRequest


@dataclass(frozen=True)
class EvalResult:
    """Evaluation outcome for one candidate LoopProgram."""

    program_id: str
    status: str  # "complete" | "failed"
    objectives: tuple[float, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None


class Proposer(Protocol):
    """Search-agent interface proposing diagnostics or LoopProgram batches (Eq. 4)."""

    def propose(self, observation: dict[str, Any], requested_count: int) -> ProposalResponse:
        ...


class Evaluator(Protocol):
    """Target evaluator executing candidate programs and optional diagnostic probes."""

    def evaluate(self, program: LoopProgram) -> EvalResult:
        ...

    def diagnose(self, request: DiagnosticRequest) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class SearchConfig:
    """Fixed-budget closed-loop search schedule and hypervolume reference point."""

    round_sizes: tuple[int, ...] = (4, 4)
    reference_point: tuple[float, ...] = (0.0, 0.0)
    max_turns_per_round: int = 12


@dataclass
class SearchResult:
    """Complete record returned at the conclusion of an ELP search run."""

    completed: dict[str, tuple[LoopProgram, EvalResult]]
    pareto_frontier: list[str]
    hypervolume_history: list[float]
    diagnostics: list[dict[str, Any]]
    representatives: dict[str, str]


class SearchCoordinator:
    """Execute ELP's four-component closed-loop offline search (Eq. 4)."""

    def __init__(
        self,
        proposer: Proposer,
        evaluator: Evaluator,
        config: SearchConfig | None = None,
        *,
        baseline_metrics: dict[str, float] | None = None,
        control_check: Callable[[], str] | None = None,
        initial_completed: Sequence[tuple[int, LoopProgram, EvalResult]] | None = None,
        initial_diagnostics: Sequence[dict[str, Any]] | None = None,
        on_trial_complete: Callable[[int, LoopProgram, EvalResult], None] | None = None,
        on_diagnostic_complete: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.proposer = proposer
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.baseline_metrics = baseline_metrics or {}
        self.control_check = control_check
        self.on_trial_complete = on_trial_complete
        self.on_diagnostic_complete = on_diagnostic_complete

        self.completed: dict[str, tuple[LoopProgram, EvalResult]] = {}
        self.diagnostics: list[dict[str, Any]] = list(initial_diagnostics or ())
        self.hypervolume_history: list[float] = []
        self._used_ids: set[str] = set()
        self._used_fingerprints: set[str] = set()
        self._completed_by_round: Counter[int] = Counter()

        for round_idx, prog, res in initial_completed or ():
            if res.status == "complete" and prog.id not in self.completed:
                self.completed[prog.id] = (prog, res)
                self._completed_by_round[int(round_idx)] += 1
            self._used_ids.add(prog.id)
            try:
                self._used_fingerprints.add(program_fingerprint(prog))
            except ValueError:
                pass

    def _build_observation(self, round_index: int, corrections: list[str]) -> dict[str, Any]:
        points = {cid: res.objectives for cid, (_, res) in self.completed.items()}
        front = pareto_frontier(points)
        hv = hypervolume(list(points.values()), self.config.reference_point)
        return {
            "round": round_index,
            "baseline_metrics": dict(self.baseline_metrics),
            "completed_candidates": {
                cid: {
                    "params": prog.params,
                    "metadata": prog.metadata,
                    "hypothesis": prog.hypothesis,
                    "objectives": res.objectives,
                    "metrics": res.metrics,
                }
                for cid, (prog, res) in self.completed.items()
            },
            "pareto_frontier": front,
            "hypervolume": hv,
            "diagnostics": self.diagnostics[-2:],
            "corrections": list(corrections),
        }

    def run(self, *, freeze_dir: str | Path | None = None) -> SearchResult:
        """Run remaining search rounds, check control invariants, and optionally freeze outputs."""
        for round_index, round_size in enumerate(self.config.round_sizes):
            valid_in_round = int(self._completed_by_round.get(round_index, 0))
            if valid_in_round >= round_size:
                points = {cid: res.objectives for cid, (_, res) in self.completed.items()}
                self.hypervolume_history.append(hypervolume(list(points.values()), self.config.reference_point))
                continue

            pre_control = self.control_check() if self.control_check is not None else None
            corrections: list[str] = []
            turns = 0
            while valid_in_round < round_size and turns < self.config.max_turns_per_round:
                turns += 1
                remaining = round_size - valid_in_round
                obs = self._build_observation(round_index, corrections)
                response = self.proposer.propose(obs, remaining)

                if isinstance(response, DiagnosticRequest):
                    if response.round_index != round_index:
                        corrections.append(f"Expected diagnostic round {round_index}, got {response.round_index}")
                        continue
                    diag_out = self.evaluator.diagnose(response)
                    diag_entry = {
                        "round": round_index,
                        "question": response.question,
                        "result": diag_out,
                    }
                    self.diagnostics.append(diag_entry)
                    if self.on_diagnostic_complete is not None:
                        self.on_diagnostic_complete(diag_entry)
                    continue

                if response.round_index != round_index:
                    corrections.append(f"Expected evaluate round {round_index}, got {response.round_index}")
                    continue
                if len(response.programs) != remaining:
                    corrections.append(f"Expected {remaining} candidates, got {len(response.programs)}")
                    continue

                validated: list[tuple[LoopProgram, str]] = []
                try:
                    batch_ids: set[str] = set()
                    batch_fps: set[str] = set()
                    for prog in response.programs:
                        validate_program_source(prog.code)
                        fp = program_fingerprint(prog)
                        if (
                            prog.id in self._used_ids
                            or prog.id in batch_ids
                            or fp in self._used_fingerprints
                            or fp in batch_fps
                        ):
                            raise ValueError(f"Duplicate candidate id or canonical action: {prog.id}")
                        batch_ids.add(prog.id)
                        batch_fps.add(fp)
                        validated.append((prog, fp))
                except (ValueError, SyntaxError, TypeError) as exc:
                    corrections.append(f"{type(exc).__name__}: {exc}")
                    continue

                for prog, fp in validated:
                    self._used_ids.add(prog.id)
                    self._used_fingerprints.add(fp)
                    result = self.evaluator.evaluate(prog)
                    if result.status == "complete":
                        self.completed[prog.id] = (prog, result)
                        self._completed_by_round[round_index] += 1
                        valid_in_round += 1
                        if self.on_trial_complete is not None:
                            self.on_trial_complete(round_index, prog, result)
                    else:
                        corrections.append(f"{prog.id} failed: {result.error or 'unknown error'}")

            if pre_control is not None and self.control_check is not None:
                post_control = self.control_check()
                if post_control != pre_control:
                    raise RuntimeError(
                        f"No-loop control check failed after round {round_index}: {pre_control} != {post_control}"
                    )

            if valid_in_round != round_size:
                raise RuntimeError(f"Round {round_index} collected {valid_in_round}/{round_size} candidates")

            points = {cid: res.objectives for cid, (_, res) in self.completed.items()}
            self.hypervolume_history.append(hypervolume(list(points.values()), self.config.reference_point))

        points = {cid: res.objectives for cid, (_, res) in self.completed.items()}
        front = pareto_frontier(points)
        reps = select_representatives(points)
        if freeze_dir is not None:
            out_dir = Path(freeze_dir)
            for role, cid in reps.items():
                freeze_program(self.completed[cid][0], out_dir / f"{role}_{cid}.json")
        return SearchResult(
            completed=dict(self.completed),
            pareto_frontier=front,
            hypervolume_history=list(self.hypervolume_history),
            diagnostics=list(self.diagnostics),
            representatives=reps,
        )
