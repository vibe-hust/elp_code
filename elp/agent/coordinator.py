"""Persistent closed-loop search coordinator isolating agent sessions from target workers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from elp.core import (
    DiagnosticRequest,
    EvalResult,
    EvaluateRequest,
    LoopProgram,
    SearchConfig,
    SearchCoordinator,
    SearchResult,
    program_fingerprint,
)

from .backends import AgentSession, create_agent_session, utc_now, write_json_atomic
from .prompts import build_system_prompt
from .sandbox import WorkspaceTool, apply_workspace_edits, validate_relative_file

_STRIPPED_WORKER_ENV = frozenset({
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
})


def _strip_worker_env(env: dict[str, str]) -> dict[str, str]:
    return {
        k: v
        for k, v in env.items()
        if k not in _STRIPPED_WORKER_ENV and not k.endswith(("_API_KEY", "_TOKEN"))
    }


def _next_counter(directory: Path | None, prefix: str) -> int:
    if directory is None or not directory.is_dir():
        return 0
    max_idx = -1
    for entry in directory.iterdir():
        name = entry.name
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :].split("_", 1)[0]
        if rest.isdigit():
            max_idx = max(max_idx, int(rest))
    return max_idx + 1


class SessionProposer:
    """Adapt an `AgentSession` + editable code workspace to `elp.core.search.Proposer`."""

    def __init__(
        self,
        session: AgentSession,
        code_workspace: Path,
        *,
        turns_dir: Path | None = None,
        max_edit_hops: int = 6,
    ) -> None:
        self.session = session
        self.code_workspace = code_workspace.resolve()
        self.code_workspace.mkdir(parents=True, exist_ok=True)
        self.turns_dir = turns_dir
        self.max_edit_hops = int(max_edit_hops)
        self._turn_counter = _next_counter(turns_dir, "turn_")

    def _next_turn_dir(self) -> Path | None:
        if self.turns_dir is None:
            return None
        path = self.turns_dir / f"turn_{self._turn_counter:04d}"
        self._turn_counter += 1
        return path

    def _resolve_candidate_code(self, cand: dict[str, Any]) -> str:
        if isinstance(cand.get("code"), str) and cand["code"].strip():
            return cand["code"]
        code_file = cand.get("code_file")
        if isinstance(code_file, str) and code_file.strip():
            path = validate_relative_file(self.code_workspace, code_file)
            return path.read_text(encoding="utf-8")
        raise ValueError(f"Candidate {cand.get('id', '?')!r} must include 'code' or 'code_file'")

    def propose(self, observation: dict[str, Any], requested_count: int) -> DiagnosticRequest | EvaluateRequest:
        edit_feedback: list[dict[str, Any]] = []
        round_idx = int(observation.get("round", 0))

        for _ in range(self.max_edit_hops):
            payload = dict(observation)
            payload["requested_count"] = int(requested_count)
            if edit_feedback:
                payload["applied_edits"] = edit_feedback

            reply = self.session.turn(
                json.dumps(payload, ensure_ascii=False),
                turn_dir=self._next_turn_dir(),
            )
            action = str(reply.get("action", "")).strip()

            if action == "edit_code":
                edits = reply.get("edits") or []
                audit = apply_workspace_edits(self.code_workspace, edits)
                edit_feedback.extend(audit)
                continue

            if action == "run_diagnostic":
                probe = dict(reply.get("probe", {"layers": [0, 1]}))
                collector_code = reply.get("collector_code") or probe.pop("collector_code", None)
                return DiagnosticRequest(
                    round_index=int(reply.get("round", round_idx)),
                    question=str(reply.get("question", "")),
                    probe=probe,
                    actions=tuple(reply.get("actions", ())),
                    collector_code=collector_code,
                )

            if action == "evaluate_candidates":
                raw_candidates = reply.get("candidates")
                if not isinstance(raw_candidates, list):
                    raise ValueError("'evaluate_candidates' response requires a list field 'candidates'")
                programs = [
                    LoopProgram(
                        id=str(item["id"]),
                        code=self._resolve_candidate_code(item),
                        params=dict(item.get("params", {})),
                        hypothesis=str(item.get("hypothesis", "")),
                        metadata=dict(item.get("metadata", {})),
                    )
                    for item in raw_candidates
                ]
                return EvaluateRequest(
                    round_index=int(reply.get("round", round_idx)),
                    rationale=str(reply.get("rationale", "")),
                    programs=programs,
                )

            raise ValueError(f"Unsupported agent action: {action!r}")

        raise RuntimeError(f"Exceeded max consecutive edit_code hops ({self.max_edit_hops})")


class SubprocessWorkerEvaluator:
    """Evaluate candidates, diagnostic probes, and control checks in isolated worker subprocesses."""

    def __init__(
        self,
        *,
        backbone: str,
        work_dir: Path,
        model_path: str | None = None,
        device: str = "cpu",
        load_kwargs: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        scorer: str | None = None,
        eval_inputs_path: str | Path | None = None,
        timeout_s: int = 300,
    ) -> None:
        self.backbone = backbone
        self.work_dir = work_dir.resolve()
        self.calls_dir = self.work_dir / "worker_calls"
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.device = device
        self.load_kwargs = dict(load_kwargs or {})
        self.inputs = dict(inputs or {})
        self.scorer = scorer
        self.eval_inputs_path = str(Path(eval_inputs_path).resolve()) if eval_inputs_path is not None else None
        self.timeout_s = int(timeout_s)
        self._call_counter = _next_counter(self.calls_dir, "call_")
        self._package_root = str(Path(__file__).resolve().parents[2])
        self._caller_cwd = str(Path.cwd().resolve())

    def _run_subprocess(self, mode: str, spec: dict[str, Any]) -> dict[str, Any]:
        call_dir = self.calls_dir / f"call_{self._call_counter:04d}_{mode}"
        self._call_counter += 1
        call_dir.mkdir(parents=True, exist_ok=True)
        spec_path = call_dir / "spec.json"
        out_path = call_dir / "result.json"
        write_json_atomic(spec_path, spec)

        env = _strip_worker_env(dict(os.environ))
        pp_entries: list[str] = [self._package_root, self._caller_cwd]
        existing_pp = env.get("PYTHONPATH", "")
        if existing_pp:
            pp_entries.extend(p for p in existing_pp.split(os.pathsep) if p)
        deduped_pp: list[str] = []
        for entry in pp_entries:
            if entry and entry not in deduped_pp:
                deduped_pp.append(entry)
        env["PYTHONPATH"] = os.pathsep.join(deduped_pp)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        cmd = [
            sys.executable,
            "-m",
            "elp.agent.worker",
            "--mode",
            mode,
            "--spec",
            str(spec_path),
            "--out",
            str(out_path),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(self.work_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=self.timeout_s,
            check=False,
        )
        if not out_path.is_file():
            err_tail = (proc.stderr or proc.stdout or f"returncode={proc.returncode}").strip()[-400:]
            raise RuntimeError(f"Worker failed to write {out_path.name}: {err_tail}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    def _attach_shared_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        if self.model_path:
            spec["model_path"] = self.model_path
        if self.load_kwargs:
            spec["load_kwargs"] = self.load_kwargs
        if self.inputs:
            spec["inputs"] = self.inputs
        if self.eval_inputs_path:
            spec["eval_inputs_path"] = self.eval_inputs_path
        return spec

    def diagnose(self, request: DiagnosticRequest) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "backbone": self.backbone,
            "question": request.question,
            "probe": request.probe,
            "actions": list(request.actions),
            "device": self.device,
        }
        if request.collector_code:
            spec["collector_code"] = request.collector_code
        self._attach_shared_spec(spec)
        return self._run_subprocess("diagnose", spec)

    def evaluate(self, program: LoopProgram) -> EvalResult:
        spec: dict[str, Any] = {
            "backbone": self.backbone,
            "device": self.device,
            "program": {
                "id": program.id,
                "code": program.code,
                "params": program.params,
                "hypothesis": program.hypothesis,
                "metadata": program.metadata,
            },
        }
        if self.scorer:
            spec["scorer"] = self.scorer
        self._attach_shared_spec(spec)

        payload = self._run_subprocess("evaluate", spec)
        if payload.get("status") != "complete":
            return EvalResult(
                program_id=program.id,
                status="failed",
                error=str(payload.get("error") or "Worker evaluation failed"),
            )
        return EvalResult(
            program_id=program.id,
            status="complete",
            objectives=tuple(float(v) for v in payload["objectives"]),
            metrics={k: float(v) for k, v in payload.get("metrics", {}).items()},
        )

    def control_digest(self) -> str:
        """Compute deterministic unlooped baseline digest inside an isolated worker subprocess."""
        spec = self._attach_shared_spec({
            "backbone": self.backbone,
            "device": self.device,
        })
        payload = self._run_subprocess("control", spec)
        digest = payload.get("digest")
        if payload.get("status") != "complete" or not isinstance(digest, str) or not digest:
            err = payload.get("error") or "missing digest"
            raise RuntimeError(f"Worker control check failed: {err}")
        return digest


class AgentSearchRunner:
    """High-level launcher tying together agent backend, workspace sandbox, and worker pool."""

    def __init__(
        self,
        *,
        workspace: Path | str,
        backbone: str = "toy",
        modality: str = "generic",
        agent_backend: str = "mock",
        agent_model: str | None = None,
        cli_bin: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model_path: str | None = None,
        device: str = "cpu",
        search_config: SearchConfig | None = None,
        freeze_dir: Path | str | None = None,
        baseline_metrics: dict[str, float] | None = None,
        worker_timeout_s: int = 300,
        scorer: str | None = None,
        eval_inputs_path: str | Path | None = None,
        upstream_repo: str | Path | None = None,
        load_kwargs: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        resume: bool = False,
        control_check: bool = False,
    ) -> None:
        self.root = Path(workspace).resolve()
        self.code_dir = self.root / "code"
        self.state_dir = self.root / "state"
        self.logs_dir = self.root / "logs"
        self.freeze_dir = Path(freeze_dir).resolve() if freeze_dir is not None else (self.root / "frozen")
        for d in (self.code_dir, self.state_dir, self.logs_dir, self.freeze_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.trials_path = self.state_dir / "trials.jsonl"
        self.diagnostics_path = self.state_dir / "diagnostics.jsonl"
        self.backbone = backbone
        self.modality = modality
        self.agent_backend = agent_backend
        self.resume = bool(resume)
        self.control_check = bool(control_check)
        self.search_config = search_config or SearchConfig(round_sizes=(4, 4), reference_point=(0.0, 0.0))

        effective_load_kwargs = dict(load_kwargs or {})
        if upstream_repo is not None:
            effective_load_kwargs["upstream_repo"] = str(Path(upstream_repo).resolve())

        self.tool = WorkspaceTool(self.code_dir, state_dir=self.state_dir)
        self.instructions = build_system_prompt(backbone, modality=modality)
        self.session = create_agent_session(
            agent_backend,
            workspace=self.code_dir,
            state_dir=self.state_dir / "agent",
            log_dir=self.logs_dir / "agent",
            instructions=self.instructions,
            model=agent_model,
            cli_bin=cli_bin,
            base_url=base_url,
            api_key=api_key,
            tool_handler=self.tool,
        )
        self.proposer = SessionProposer(
            self.session,
            self.code_dir,
            turns_dir=self.state_dir / "turns",
        )
        self.evaluator = SubprocessWorkerEvaluator(
            backbone=backbone,
            work_dir=self.state_dir,
            model_path=model_path,
            device=device,
            load_kwargs=effective_load_kwargs,
            inputs=inputs,
            scorer=scorer,
            eval_inputs_path=eval_inputs_path,
            timeout_s=worker_timeout_s,
        )
        self.baseline_metrics = baseline_metrics or {"quality": 0.0, "speedup": 1.0}

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()

    def _on_trial_complete(self, round_index: int, prog: LoopProgram, result: EvalResult) -> None:
        try:
            fp = program_fingerprint(prog)
        except ValueError:
            fp = ""
        self._append_jsonl(
            self.trials_path,
            {
                "at": utc_now(),
                "round_index": int(round_index),
                "program_id": prog.id,
                "program": {
                    "id": prog.id,
                    "code": prog.code,
                    "params": dict(prog.params),
                    "hypothesis": prog.hypothesis,
                    "metadata": dict(prog.metadata),
                },
                "fingerprint": fp,
                "status": result.status,
                "objectives": list(result.objectives),
                "metrics": dict(result.metrics),
            },
        )

    def _on_diagnostic_complete(self, entry: dict[str, Any]) -> None:
        self._append_jsonl(
            self.diagnostics_path,
            {"at": utc_now(), **entry},
        )

    def _load_resume_state(
        self,
    ) -> tuple[list[tuple[int, LoopProgram, EvalResult]], list[dict[str, Any]]]:
        completed: list[tuple[int, LoopProgram, EvalResult]] = []
        diagnostics: list[dict[str, Any]] = []
        if self.trials_path.is_file():
            for raw_line in self.trials_path.read_text(encoding="utf-8").splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                rec = json.loads(raw_line)
                if rec.get("status") != "complete":
                    continue
                prog_data = rec.get("program")
                if not isinstance(prog_data, dict):
                    continue
                prog = LoopProgram(
                    id=str(prog_data["id"]),
                    code=str(prog_data["code"]),
                    params=dict(prog_data.get("params", {})),
                    hypothesis=str(prog_data.get("hypothesis", "")),
                    metadata=dict(prog_data.get("metadata", {})),
                )
                res = EvalResult(
                    program_id=prog.id,
                    status="complete",
                    objectives=tuple(float(v) for v in rec.get("objectives", ())),
                    metrics={k: float(v) for k, v in rec.get("metrics", {}).items()},
                )
                completed.append((int(rec.get("round_index", 0)), prog, res))
        if self.diagnostics_path.is_file():
            for raw_line in self.diagnostics_path.read_text(encoding="utf-8").splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                rec = json.loads(raw_line)
                rec.pop("at", None)
                diagnostics.append(rec)
        return completed, diagnostics

    def run(self) -> SearchResult:
        """Execute full closed-loop search and persist experiment state & frozen representatives."""
        initial_completed: list[tuple[int, LoopProgram, EvalResult]] = []
        initial_diagnostics: list[dict[str, Any]] = []
        if self.resume:
            initial_completed, initial_diagnostics = self._load_resume_state()
        try:
            coordinator = SearchCoordinator(
                self.proposer,
                self.evaluator,
                self.search_config,
                baseline_metrics=self.baseline_metrics,
                control_check=self.evaluator.control_digest if self.control_check else None,
                initial_completed=initial_completed,
                initial_diagnostics=initial_diagnostics,
                on_trial_complete=self._on_trial_complete,
                on_diagnostic_complete=self._on_diagnostic_complete,
            )
            result = coordinator.run(freeze_dir=self.freeze_dir)
            write_json_atomic(
                self.state_dir / "pareto_frontier.json",
                {
                    "pareto_frontier": result.pareto_frontier,
                    "hypervolume_history": result.hypervolume_history,
                    "representatives": result.representatives,
                },
            )
            write_json_atomic(
                self.state_dir / "experiment.json",
                {
                    "status": "complete",
                    "completed_at": utc_now(),
                    "backbone": self.backbone,
                    "modality": self.modality,
                    "agent_backend": self.agent_backend,
                    "resumed": self.resume,
                    "control_check": self.control_check,
                    "completed_count": len(result.completed),
                    "pareto_frontier": result.pareto_frontier,
                    "hypervolume_history": result.hypervolume_history,
                    "representatives": result.representatives,
                    "diagnostics": result.diagnostics,
                },
            )
            return result
        finally:
            self.session.close()


__all__ = [
    "AgentSearchRunner",
    "SessionProposer",
    "SubprocessWorkerEvaluator",
]
