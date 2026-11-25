"""Common adapter contract, upstream resolver, stats tracking, and module helpers for ELP backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import torch


@dataclass
class BlockLoopStats:
    """Runtime invocation and cache-transaction counters across patched inference."""

    block_calls: int = 0
    loop_invocations: int = 0
    guidance_ref_calls: int = 0
    guidance_loop_calls: int = 0
    kv_rollbacks: int = 0
    kv_commits: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class InferenceResult:
    """Standardized return object produced by `BackboneAdapter.infer`."""

    output: Any
    latency_s: float
    step_count: int = 1
    loop_stats: dict[str, int] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    """Convert string dtype names (`float32`, `float16`, `bfloat16`) to `torch.dtype`."""
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    key = str(dtype).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype!r}")
    return mapping[key]


def resolve_block_container(
    root: Any,
    attrs: Sequence[str] = ("layers", "dit_blocks", "blocks"),
    _seen: set[int] | None = None,
) -> Any:
    """Locate the primary sequential transformer block container on a model or pipeline."""
    if _seen is None:
        _seen = set()
    if id(root) in _seen or root is None:
        raise ValueError("Could not locate transformer block list on model")
    _seen.add(id(root))

    for attr in attrs:
        seq = getattr(root, attr, None)
        if seq is not None and hasattr(seq, "__len__") and len(seq) > 0:
            return seq

    for nested_attr in ("transformer", "diff_head", "generator", "model"):
        nested = getattr(root, nested_attr, None)
        if nested is not None and nested is not root:
            try:
                return resolve_block_container(nested, attrs=attrs, _seen=_seen)
            except ValueError:
                continue
    raise ValueError(f"Could not locate transformer block container ({attrs}) on {type(root).__name__}")


BUNDLED_THIRD_PARTY_DIR = Path(__file__).resolve().parents[2] / "third_party"


def _is_other_bundled_repo(candidate: Path, active_repo: Path) -> bool:
    tp_root = BUNDLED_THIRD_PARTY_DIR.resolve()
    try:
        rel = candidate.resolve().relative_to(tp_root)
    except (OSError, ValueError):
        return False
    if not rel.parts:
        return False
    bundled_root = tp_root / rel.parts[0]
    return not active_repo.resolve().is_relative_to(bundled_root)


def resolve_upstream_repo(
    upstream_repo: str | Path | None,
    env_var: str,
    label: str,
    *,
    bundled_subdir: str | None = None,
) -> Path:
    """Resolve an upstream checkout from explicit arg, env var, or bundled `third_party/`."""
    raw = str(upstream_repo).strip() if upstream_repo is not None else os.environ.get(env_var, "").strip()
    if raw:
        repo = Path(raw).expanduser().resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"Upstream {label} repository directory not found: {repo}")
        return repo
    if bundled_subdir:
        bundled = (BUNDLED_THIRD_PARTY_DIR / bundled_subdir).resolve()
        if bundled.is_dir():
            return bundled
    raise RuntimeError(
        f"Loading {label} from checkpoint requires bundled `third_party/{bundled_subdir or label}` "
        f"or an explicit `upstream_repo` (or {env_var} env var)."
    )


def prepend_sys_path(path: Path) -> None:
    """Prepend `path` to `sys.path` if not already at index 0."""
    resolved = str(path.resolve())
    if resolved in sys.path:
        sys.path.remove(resolved)
    sys.path.insert(0, resolved)


def activate_upstream_repo(repo: Path, *, extra_subdirs: tuple[str, ...] = ()) -> None:
    """Isolate bundled `third_party` packages on `sys.path`/`sys.modules` and activate `repo`."""
    resolved_repo = repo.resolve()
    sys.path[:] = [
        entry
        for entry in sys.path
        if not (entry and _is_other_bundled_repo(Path(entry), resolved_repo))
    ]
    stale_modules: list[str] = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file and _is_other_bundled_repo(Path(mod_file), resolved_repo):
            stale_modules.append(mod_name)
            continue
        mod_path = getattr(mod, "__path__", None)
        if mod_path is not None:
            try:
                if any(_is_other_bundled_repo(Path(p), resolved_repo) for p in mod_path):
                    stale_modules.append(mod_name)
            except TypeError:
                pass
    for mod_name in stale_modules:
        sys.modules.pop(mod_name, None)

    prepend_sys_path(resolved_repo)
    for sub in reversed(extra_subdirs):
        sub_path = resolved_repo / sub
        if sub_path.is_dir():
            prepend_sys_path(sub_path)


def _unwrap_program(program: Any) -> Any:
    if program is None:
        return None
    if isinstance(program, (str, Path)):
        from elp.core.program import load_frozen_program

        candidate = Path(program)
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Frozen program JSON file not found: {program!r}. "
                "Built-in presets are not shipped; pass a frozen JSON file, LoopProgram, or modality config."
            )
        return load_frozen_program(candidate)
    return program


class BackboneAdapter(ABC):
    """Unified reversible-patch + inference contract across Image, Text, and Video models."""

    name: str
    modality: str

    @abstractmethod
    def load(
        self,
        model_path: str,
        *,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype | None = None,
        local_files_only: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Load a frozen pretrained model from HuggingFace Hub ID or local directory."""

    @abstractmethod
    def install_program(self, model: Any, program: Any) -> AbstractContextManager[Any]:
        """Return a context manager reversibly installing `program` without mutating weights."""

    @abstractmethod
    def infer(
        self,
        model: Any,
        inputs: Any,
        *,
        program: Any = None,
        probe: Any = None,
        **kwargs: Any,
    ) -> InferenceResult:
        """Execute inference (with optional ELP program and/or read-only `DiagnosticCollector`)."""

    @staticmethod
    def resolve_program(program: Any) -> Any:
        return _unwrap_program(program)

    @staticmethod
    def null_context() -> AbstractContextManager[None]:
        return nullcontext()


__all__ = [
    "BUNDLED_THIRD_PARTY_DIR",
    "BackboneAdapter",
    "BlockLoopStats",
    "InferenceResult",
    "activate_upstream_repo",
    "prepend_sys_path",
    "resolve_block_container",
    "resolve_dtype",
    "resolve_upstream_repo",
]
