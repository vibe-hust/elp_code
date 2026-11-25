"""Sandboxed workspace file editing and bounded local shell execution for search agents."""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Sequence


WORKSPACE_EXEC_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": "workspace_exec",
    "description": (
        "Execute a bounded local shell command inside the search workspace with no network "
        "or GPU access. Use this to inspect candidate source or summarize diagnostic traces."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
}

_STRIPPED_ENV_KEYS = frozenset({
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
})


def _strip_secret_env(env: dict[str, str]) -> dict[str, str]:
    return {
        k: v
        for k, v in env.items()
        if k not in _STRIPPED_ENV_KEYS and not k.endswith(("_API_KEY", "_TOKEN"))
    }

_DEFAULT_ALLOWED_PREFIXES = ("candidate.py", "algorithms/", "notes/")


def validate_relative_file(root: Path, rel_path: str) -> Path:
    """Validate that `rel_path` is strictly relative and resolves inside `root`."""
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError("Relative file path must be a non-empty string")
    norm = rel_path.replace("\\", "/")
    if norm.startswith("/") or ":" in norm:
        raise ValueError(f"Absolute or drive-qualified path is disallowed: {rel_path!r}")
    posix = PurePosixPath(norm)
    if posix.is_absolute() or any(part in ("..", "") for part in posix.parts):
        raise ValueError(f"Path traversal or empty segment is disallowed: {rel_path!r}")
    root_resolved = root.resolve()
    target = (root_resolved / Path(*posix.parts)).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace root: {rel_path!r}") from exc
    return target


def _is_allowed_edit_path(norm_path: str, allowed_prefixes: Sequence[str]) -> bool:
    for prefix in allowed_prefixes:
        p = prefix.replace("\\", "/")
        if p.endswith("/"):
            if norm_path.startswith(p):
                return True
        elif norm_path == p:
            return True
    return False


def apply_workspace_edits(
    workspace: Path,
    edits: list[dict[str, Any]],
    *,
    allowed_prefixes: Sequence[str] = _DEFAULT_ALLOWED_PREFIXES,
    max_edits: int = 30,
    max_bytes: int = 500_000,
) -> list[dict[str, Any]]:
    """Validate and apply structured text/code edits inside an isolated agent workspace."""
    if not isinstance(edits, list) or not (1 <= len(edits) <= max_edits):
        raise ValueError(f"Expected 1..{max_edits} edits")

    pending: dict[str, str] = {}
    for edit in edits:
        if not isinstance(edit, dict) or "path" not in edit:
            raise ValueError("Each edit must be a dict containing 'path'")
        raw_name = str(edit["path"])
        norm_name = raw_name.replace("\\", "/")
        if not _is_allowed_edit_path(norm_name, allowed_prefixes):
            raise ValueError(
                f"Editable paths are restricted to {tuple(allowed_prefixes)}; got {raw_name!r}"
            )
        target = validate_relative_file(workspace, norm_name)
        before = pending.get(
            norm_name,
            target.read_text(encoding="utf-8") if target.exists() else "",
        )
        if "content" in edit:
            after = edit["content"]
        else:
            old = edit.get("old")
            new = edit.get("new")
            if not isinstance(old, str) or not old or not isinstance(new, str):
                raise ValueError("Patch edit requires non-empty string 'old' and string 'new'")
            if before.count(old) != 1:
                raise ValueError(f"Exact replacement in {norm_name} must match exactly once")
            after = before.replace(old, new, 1)

        if not isinstance(after, str) or len(after.encode("utf-8")) > max_bytes:
            raise ValueError(f"Edited file {norm_name} is invalid or exceeds {max_bytes} bytes")
        if target.suffix == ".py":
            ast.parse(after, filename=norm_name)
        pending[norm_name] = after

    audit: list[dict[str, Any]] = []
    for norm_name, after in pending.items():
        target = validate_relative_file(workspace, norm_name)
        before_sha = (
            hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(after, encoding="utf-8")
        tmp.replace(target)
        after_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        audit.append({
            "path": norm_name,
            "before_sha256": before_sha,
            "after_sha256": after_sha,
        })
    return audit


class WorkspaceTool:
    """Execute bounded shell commands inside a workspace with sanitized offline env."""

    def __init__(
        self,
        workspace: Path,
        readonly: Sequence[Path] = (),
        *,
        python: Path | None = None,
        state_dir: Path | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.readonly = tuple(Path(p).resolve() for p in readonly)
        self.python = (python or Path(sys.executable)).resolve()
        self.state_dir = state_dir.resolve() if state_dir is not None else None
        self.count = 0

    def _sanitized_env(self) -> dict[str, str]:
        scratch = self.workspace / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        env = _strip_secret_env(dict(os.environ))
        env.update({
            "TMPDIR": str(scratch),
            "TEMP": str(scratch),
            "TMP": str(scratch),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        })
        if os.name != "nt":
            env["HOME"] = str(scratch)
            env["LANG"] = "C.UTF-8"
            env["LC_ALL"] = "C.UTF-8"
            env["PATH"] = f"{self.python.parent}:/usr/bin:/bin"
        else:
            existing_path = env.get("PATH", "")
            env["PATH"] = f"{self.python.parent}{os.pathsep}{existing_path}" if existing_path else str(self.python.parent)
        return env

    def _build_command(self, shell_command: str) -> list[str]:
        if os.name == "nt":
            pwsh = shutil.which("powershell") or shutil.which("pwsh")
            if pwsh:
                return [pwsh, "-NoProfile", "-Command", shell_command]
            return [os.environ.get("COMSPEC", "cmd.exe"), "/c", shell_command]

        bash_bin = "/bin/bash" if Path("/bin/bash").exists() else (shutil.which("sh") or "/bin/sh")
        base_cmd = (
            [bash_bin, "--noprofile", "--norc", "-c", shell_command]
            if bash_bin.endswith("bash")
            else [bash_bin, "-c", shell_command]
        )
        landlock_helper = os.environ.get("ELP_LANDLOCK_HELPER", "").strip()
        if landlock_helper and Path(landlock_helper).is_file() and self.state_dir is not None:
            import json

            self.state_dir.mkdir(parents=True, exist_ok=True)
            system_ro = [
                p for p in (
                    "/usr", "/bin", "/lib", "/lib64", "/etc/ld.so.cache",
                    "/etc/passwd", "/etc/group", "/etc/localtime",
                    "/proc/self", "/proc/thread-self", "/proc/meminfo",
                    "/proc/cpuinfo", "/proc/stat",
                )
                if Path(p).exists()
            ]
            read_paths = list(dict.fromkeys(
                [str(p) for p in self.readonly]
                + [str(self.python.parent.parent)]
                + system_ro
            ))
            policy = {
                "read_only": read_paths,
                "read_write": [str(self.workspace), "/dev/null", "/dev/urandom"],
                "cwd": str(self.workspace),
                "env": self._sanitized_env(),
            }
            policy_path = self.state_dir / "agent_tool_policy.json"
            policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
            return [str(self.python), landlock_helper, "--policy", str(policy_path), "--", *base_cmd]
        return base_cmd

    def __call__(self, params: dict[str, Any]) -> dict[str, Any]:
        if "tool" in params:
            if params["tool"] != "workspace_exec" or params.get("namespace"):
                raise ValueError(f"Unknown isolated tool: {params.get('tool')!r}")
            args = params.get("arguments", {})
        else:
            args = params

        if not isinstance(args, dict) or set(args) - {"command", "timeout_seconds"}:
            raise ValueError("Expected arguments with 'command' and optional 'timeout_seconds'")
        command_str = args.get("command")
        if not isinstance(command_str, str) or not command_str.strip():
            raise ValueError("Expected non-empty shell command string")
        timeout = args.get("timeout_seconds", 60)
        if type(timeout) is not int or not (1 <= timeout <= 120):
            raise ValueError(f"timeout_seconds must be an int in 1..120, got {timeout!r}")

        self.count += 1
        cmd = self._build_command(command_str)
        env = self._sanitized_env()
        started = time.monotonic()
        popen_kwargs: dict[str, Any] = {
            "cwd": str(self.workspace),
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(cmd, **popen_kwargs)
        timed_out = False
        try:
            raw_out = proc.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
            raw_out = proc.communicate()[0]

        decoded = raw_out[:40_000].decode("utf-8", errors="replace")
        return {
            "returncode": proc.returncode,
            "timeout": timed_out,
            "seconds": round(time.monotonic() - started, 4),
            "output": decoded,
            "truncated": len(raw_out) > 40_000,
        }
