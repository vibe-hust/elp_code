"""Safe AST sandbox, canonical fingerprinting, unified invocation, and freeze/replay persistence."""
from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

import torch
from torch import Tensor


_BANNED_NAMES = frozenset({
    "eval", "exec", "compile", "open", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "type", "object", "input", "__import__",
    "breakpoint", "help", "dir", "memoryview",
})

_BANNED_TORCH_ATTRS = frozenset({
    "cuda", "load", "save", "distributed", "multiprocessing", "hub",
    "utils", "serialization", "profiler", "compile", "jit", "set_default_device",
    "set_default_dtype", "set_num_threads", "manual_seed", "random",
    "data", "numpy", "storage", "untyped_storage", "data_ptr",
    "frombuffer", "from_numpy", "as_strided",
})

_SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "isinstance", "len", "list", "max", "min", "range", "round", "set",
    "sorted", "str", "sum", "tuple", "zip", "ValueError", "RuntimeError",
)

_ID_PATTERN = re.compile(r"[a-zA-Z0-9_-]{1,64}")


@dataclass(frozen=True)
class LoopProgram:
    """Pair an executable AST-checked loop algorithm with parameters and metadata."""

    id: str
    code: str
    params: dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ID_PATTERN.fullmatch(self.id):
            raise ValueError(f"Invalid program id: {self.id!r}")
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dict")
        # Ensure JSON serializability of parameters
        json.dumps(self.params, allow_nan=False)


def validate_program_source(source: str, *, required_entrypoint: str = "loop") -> ast.Module:
    """Validate candidate loop or diagnostic source against ELP's pure-tensor sandbox."""
    if not isinstance(source, str) or not source.strip() or len(source) > 120_000:
        raise ValueError("Invalid candidate source size")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            raise ValueError("Use `import torch` or `import math`; from-import is disallowed")
        if isinstance(node, ast.Import):
            if any(alias.name not in {"torch", "math"} for alias in node.names):
                raise ValueError("Only `torch` and `math` imports are allowed")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.With, ast.AsyncWith)):
            raise ValueError("Candidate modules must contain pure functions without classes or global state")
        if isinstance(node, ast.FunctionDef):
            defaults = node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
            if node.decorator_list or any(not isinstance(d, ast.Constant) for d in defaults):
                raise ValueError("Decorators and non-constant default arguments are disallowed")
        if isinstance(node, ast.Name) and (node.id in _BANNED_NAMES or node.id.startswith("__")):
            raise ValueError(f"Forbidden candidate identifier: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr.endswith("_") or node.attr in _BANNED_TORCH_ATTRS:
                raise ValueError(f"Forbidden candidate attribute: {node.attr}")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Attribute) for t in targets):
                raise ValueError("Attribute mutation is disallowed inside candidate programs")
    if any(not isinstance(stmt, (ast.FunctionDef, ast.Import, ast.Expr)) for stmt in tree.body):
        raise ValueError("Top-level executable or mutable state is disallowed")
    if any(
        isinstance(stmt, ast.Expr)
        and not (isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str))
        for stmt in tree.body
    ):
        raise ValueError("Top-level expressions other than docstrings are disallowed")
    if not any(isinstance(stmt, ast.FunctionDef) and stmt.name == required_entrypoint for stmt in tree.body):
        raise ValueError(f"Missing required entrypoint function {required_entrypoint!r}")
    return tree


def load_program_callable(source: str, *, entrypoint: str = "loop") -> Callable[..., Any]:
    """Compile and return the validated entrypoint callable from candidate source."""
    validate_program_source(source, required_entrypoint=entrypoint)
    safe_env = {name: getattr(builtins, name) for name in _SAFE_BUILTINS}

    def restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in {"torch", "math"}:
            raise ImportError(f"Blocked import: {name}")
        return importlib.import_module(name)

    safe_env["__import__"] = restricted_import
    namespace: dict[str, Any] = {"__builtins__": safe_env}
    exec(compile(source, "<elp-program>", "exec"), namespace)
    return namespace[entrypoint]


def invoke_loop_program(
    loop_fn: Callable[..., Any],
    x: Tensor,
    evaluate: Callable[[Tensor], Tensor],
    *,
    info: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    memory: dict[str, Any] | None = None,
    enforce_unit_budget: bool = False,
    verify_residual: Callable[[Tensor], None] | None = None,
) -> Tensor:
    """Invoke a sandbox-compiled `loop` callable uniformly across 2/3/4/5-arg signatures.

    Supports both direct tensor return (`def loop(x, evaluate, params) -> Tensor`)
    and residual combination dicts (`{"residuals": [...], "weights": [...]}`).
    """
    sig = inspect.signature(loop_fn)
    n_pos = sum(
        1
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    info_arg = {} if info is None else info
    params_arg = {} if params is None else params
    memory_arg = {} if memory is None else memory

    before_ver = x._version
    if n_pos >= 5:
        res = loop_fn(x, evaluate, info_arg, params_arg, memory_arg)
    elif n_pos == 4:
        res = loop_fn(x, evaluate, info_arg, params_arg)
    elif n_pos == 3:
        res = loop_fn(x, evaluate, params_arg)
    elif n_pos == 2:
        res = loop_fn(x, params_arg)
    else:
        res = loop_fn(x)

    if x._version != before_ver:
        raise ValueError("Loop program mutated input tensor in-place")

    if isinstance(res, dict) and "residuals" in res and "weights" in res:
        residuals = res["residuals"]
        weights = res["weights"]
        if not isinstance(residuals, (list, tuple)) or not isinstance(weights, (list, tuple)):
            raise ValueError("Loop residuals and weights must be sequences")
        if not residuals or len(residuals) != len(weights) or len(weights) > 8:
            raise ValueError("Invalid residual/weight count returned by loop program")
        w_floats = [float(w) for w in weights]
        if any(not math.isfinite(w) or abs(w) > 8.0 for w in w_floats):
            raise ValueError("Residual weights must be finite and bounded in [-8, 8]")
        if enforce_unit_budget and abs(sum(w_floats) - 1.0) > 1e-8:
            raise ValueError("Residual weights must sum to 1.0 under unit-budget constraint")
        out = x
        for r, w in zip(residuals, w_floats):
            if verify_residual is not None:
                verify_residual(r)
            if not isinstance(r, Tensor) or r.shape != x.shape or r.device != x.device:
                raise ValueError("Residual tensor shape/device mismatch")
            out = out + w * (r.to(out.dtype) if r.dtype != out.dtype else r)
        return out

    if isinstance(res, Tensor):
        if res.shape != x.shape or res.device != x.device:
            raise ValueError("Loop program output shape/device mismatch")
        return res.to(x.dtype) if res.dtype != x.dtype else res

    raise TypeError(f"Unsupported loop program return type: {type(res).__name__}")


def program_fingerprint(program: LoopProgram) -> str:
    """Compute a whitespace/comment-invariant canonical SHA-256 fingerprint."""
    normalized_ast = ast.dump(validate_program_source(program.code))
    payload = {
        "ast": normalized_ast,
        "params": program.params,
        "metadata": program.metadata,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_program(program: LoopProgram, path: str | Path) -> None:
    """Persist a validated, self-verifying frozen loop program snapshot."""
    validate_program_source(program.code)
    source_sha = hashlib.sha256(program.code.encode("utf-8")).hexdigest()
    record = {
        "id": program.id,
        "code": program.code,
        "params": program.params,
        "hypothesis": program.hypothesis,
        "metadata": program.metadata,
        "source_sha256": source_sha,
        "fingerprint": program_fingerprint(program),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(target)


def load_frozen_program(path: str | Path) -> LoopProgram:
    """Load and verify a frozen loop program snapshot for deterministic replay."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    source_sha = hashlib.sha256(record["code"].encode("utf-8")).hexdigest()
    if record.get("source_sha256") != source_sha:
        raise ValueError("Frozen program source checksum mismatch")
    program = LoopProgram(
        id=record["id"],
        code=record["code"],
        params=record.get("params", {}),
        hypothesis=record.get("hypothesis", ""),
        metadata=record.get("metadata", {}),
    )
    if record.get("fingerprint") != program_fingerprint(program):
        raise ValueError("Frozen program fingerprint mismatch")
    return program
