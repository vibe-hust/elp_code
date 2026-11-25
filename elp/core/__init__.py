"""Core ELP abstractions: AST sandbox, programmable diagnostics, Pareto metrics, and search loop."""

from .diagnostics import DiagnosticCollector, ProbeSpec
from .pareto import hypervolume, pareto_frontier, select_representatives
from .program import (
    LoopProgram,
    freeze_program,
    invoke_loop_program,
    load_frozen_program,
    load_program_callable,
    program_fingerprint,
    validate_program_source,
)
from .search import (
    DiagnosticRequest,
    EvalResult,
    EvaluateRequest,
    Evaluator,
    Proposer,
    SearchConfig,
    SearchCoordinator,
    SearchResult,
)

__all__ = [
    "DiagnosticCollector",
    "DiagnosticRequest",
    "EvalResult",
    "EvaluateRequest",
    "Evaluator",
    "LoopProgram",
    "ProbeSpec",
    "Proposer",
    "SearchConfig",
    "SearchCoordinator",
    "SearchResult",
    "freeze_program",
    "hypervolume",
    "invoke_loop_program",
    "load_frozen_program",
    "load_program_callable",
    "pareto_frontier",
    "program_fingerprint",
    "select_representatives",
    "validate_program_source",
]
