"""Search-agent session adapters (Codex CLI, OpenAI-compatible HTTP API, generic CLI, mock)."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

from .prompts import DIAGNOSTIC_STEP_PROGRAM, evaluate_program_template
from .sandbox import WORKSPACE_EXEC_TOOL_SCHEMA

_STRIPPED_SUBPROCESS_ENV = frozenset({
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
})


def _sanitized_cli_env() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k not in _STRIPPED_SUBPROCESS_ENV and not k.endswith(("_API_KEY", "_TOKEN"))
    }


def utc_now() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON atomically via a temporary sibling file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def public_event(value: Any) -> Any:
    """Redact hidden chain-of-thought and encrypted fields before persisting event logs."""
    if isinstance(value, dict):
        method = value.get("method")
        if method in {"item/reasoning/textDelta", "codex/event/agent_reasoning_raw_content"}:
            return {"method": method, "omitted": "raw_reasoning_not_collected"}
        drop_keys = {"encrypted_content", "encryptedContent"}
        if value.get("type") == "reasoning":
            drop_keys = drop_keys | {"content"}
        return {k: public_event(v) for k, v in value.items() if k not in drop_keys}
    if isinstance(value, list):
        return [public_event(item) for item in value]
    return value


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the final JSON object from a raw text or Markdown-fenced model response."""
    clean = text.strip()
    if not clean:
        raise ValueError("Empty agent response text")
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", clean, flags=re.DOTALL | re.IGNORECASE)
    for block in reversed(fenced):
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    starts = [match.start() for match in re.finditer(r"\{", clean)]
    for pos in reversed(starts):
        try:
            parsed, _ = decoder.raw_decode(clean[pos:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not extract JSON object from agent response: {clean[:240]!r}")


class AgentSession(Protocol):
    """Persistent multi-turn search-agent interface returning structured JSON dicts."""

    def turn(self, prompt: str, *, turn_dir: Path | None = None, timeout: int = 600) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


class CodexCLISession:
    """Persistent Codex app-server JSON-RPC client with event logging and dynamic tools."""

    def __init__(
        self,
        workspace: Path,
        state_dir: Path,
        log_dir: Path,
        instructions: str,
        *,
        launch_argv: list[str] | None = None,
        config_overrides: dict[str, Any] | None = None,
        permission_profile: str | None = None,
        dynamic_tools: list[dict[str, Any]] | None = None,
        dynamic_tool_handler: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.workspace = workspace
        self.state_dir = state_dir
        self.log_dir = log_dir
        state_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.session_path = state_dir / "session.json"
        self.events_path = log_dir / "session_events.jsonl"
        self.turn_dir: Path | None = None
        self.counter = 0
        self.usage: Any = None
        self.dynamic_tool_handler = dynamic_tool_handler
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._stderr = (log_dir / "agent_server.stderr.log").open("a", encoding="utf-8")

        env = _sanitized_cli_env()
        argv = launch_argv or [shutil.which("codex") or "codex", "app-server"]
        self.process = subprocess.Popen(
            argv,
            cwd=str(workspace),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            encoding="utf-8",
            env=env,
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        def reader() -> None:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._queue.put(msg)
                safe = json.dumps(public_event(msg), ensure_ascii=False)
                with self.events_path.open("a", encoding="utf-8") as fh:
                    fh.write(safe + "\n")
                if self.turn_dir is not None:
                    with (self.turn_dir / "events.jsonl").open("a", encoding="utf-8") as fh:
                        fh.write(safe + "\n")
            self._queue.put(None)

        threading.Thread(target=reader, daemon=True).start()
        self.call(
            "initialize",
            {
                "clientInfo": {"name": "elp_search_coordinator", "version": "1.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        self.notify("initialized", {})

        if self.session_path.exists():
            saved = json.loads(self.session_path.read_text(encoding="utf-8"))
            self.thread_id = saved["thread_id"]
            self.call("thread/resume", {"threadId": self.thread_id})
        else:
            params: dict[str, Any] = {
                "cwd": str(workspace),
                "approvalPolicy": "never",
                "baseInstructions": instructions,
                "personality": "pragmatic",
            }
            if permission_profile:
                params["sandbox"] = permission_profile
            if config_overrides:
                params["config"] = config_overrides
            if dynamic_tools:
                params["dynamicTools"] = dynamic_tools
            res = self.call("thread/start", params)
            self.thread_id = res["thread"]["id"]
            write_json_atomic(self.session_path, {"thread_id": self.thread_id, "created_at": utc_now()})

    def _send(self, msg: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def call(self, method: str, params: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
        self.counter += 1
        req_id = self.counter
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            msg = self._queue.get(timeout=max(1.0, deadline - time.monotonic()))
            if msg is None:
                raise RuntimeError("Codex app-server terminated unexpectedly")
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(str(msg["error"]))
                return msg["result"]

    def turn(self, prompt: str, *, turn_dir: Path | None = None, timeout: int = 600) -> dict[str, Any]:
        self.turn_dir = turn_dir
        if turn_dir is not None:
            turn_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(turn_dir / "prompt.json", {"prompt": prompt, "at": utc_now()})
        turn_id = self.call(
            "turn/start",
            {"threadId": self.thread_id, "input": [{"type": "text", "text": prompt}]},
        )["turn"]["id"]

        last_text = ""
        deadline = time.monotonic() + timeout
        while True:
            msg = self._queue.get(timeout=max(1.0, deadline - time.monotonic()))
            if msg is None:
                raise RuntimeError("Codex app-server exited before turn completed")
            method = msg.get("method")
            params = msg.get("params", {})
            if "id" in msg and method == "item/tool/call":
                try:
                    if self.dynamic_tool_handler is None:
                        raise RuntimeError("No dynamic tool handler configured")
                    output = self.dynamic_tool_handler(params)
                    result = {
                        "success": True,
                        "contentItems": [{"type": "inputText", "text": json.dumps(output)}],
                    }
                except Exception as exc:  # pragma: no cover - defensive RPC bridge
                    result = {
                        "success": False,
                        "contentItems": [{"type": "inputText", "text": f"{type(exc).__name__}: {exc}"}],
                    }
                self._send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
            elif method == "thread/tokenUsage/updated" and params.get("turnId") == turn_id:
                self.usage = params.get("tokenUsage")
            elif method == "item/completed":
                item = params.get("item", {})
                if item.get("type") == "agentMessage":
                    last_text = item.get("text", "") or last_text
            elif method == "turn/completed" and params.get("turn", {}).get("id") == turn_id:
                if params["turn"].get("status") != "completed":
                    raise RuntimeError(str(params["turn"]))
                parsed = extract_json_object(last_text)
                if turn_dir is not None:
                    write_json_atomic(turn_dir / "response.json", {"parsed": parsed, "at": utc_now()})
                self.turn_dir = None
                return parsed

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5)
        except Exception:
            self.process.kill()
        finally:
            self._stderr.close()


class OpenAICompatibleSession:
    """Zero-dependency HTTP chat-completions session supporting JSON replies & tool calls."""

    def __init__(
        self,
        state_dir: Path,
        log_dir: Path,
        instructions: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        tool_handler: Callable[[dict[str, Any]], Any] | None = None,
        max_tool_steps: int = 8,
    ) -> None:
        self.state_dir = state_dir
        self.log_dir = log_dir
        state_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or os.environ.get("ELP_OPENAI_MODEL") or "gpt-4.1"
        resolved_base = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.endpoint = (
            resolved_base if resolved_base.endswith("/chat/completions") else f"{resolved_base}/chat/completions"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OpenAICompatibleSession requires OPENAI_API_KEY or explicit `api_key`")
        self.tool_handler = tool_handler
        self.max_tool_steps = int(max_tool_steps)
        self.events_path = log_dir / "session_events.jsonl"
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}]

    def _post(self, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {self.endpoint}: {detail[:400]}") from exc

    def turn(self, prompt: str, *, turn_dir: Path | None = None, timeout: int = 600) -> dict[str, Any]:
        if turn_dir is not None:
            turn_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(turn_dir / "prompt.json", {"prompt": prompt, "at": utc_now()})
        self.messages.append({"role": "user", "content": prompt})

        tools_payload = None
        if self.tool_handler is not None:
            tools_payload = [{
                "type": "function",
                "function": {
                    "name": WORKSPACE_EXEC_TOOL_SCHEMA["name"],
                    "description": WORKSPACE_EXEC_TOOL_SCHEMA["description"],
                    "parameters": WORKSPACE_EXEC_TOOL_SCHEMA["inputSchema"],
                },
            }]

        for _ in range(self.max_tool_steps):
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": self.messages,
                "temperature": 0.2,
            }
            if tools_payload is not None:
                payload["tools"] = tools_payload
            resp = self._post(payload, timeout=timeout)
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(public_event({"at": utc_now(), "usage": resp.get("usage")})) + "\n")
            choice = resp["choices"][0]["message"]
            self.messages.append(choice)
            tool_calls = choice.get("tool_calls") or []
            if tool_calls and self.tool_handler is not None:
                for call in tool_calls:
                    fn = call["function"]
                    args = json.loads(fn.get("arguments") or "{}")
                    result = self.tool_handler({"tool": fn["name"], "arguments": args})
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                continue
            parsed = extract_json_object(choice.get("content") or "")
            if turn_dir is not None:
                write_json_atomic(turn_dir / "response.json", {"parsed": parsed, "at": utc_now()})
            return parsed

        raise RuntimeError(f"Exceeded max tool steps ({self.max_tool_steps}) without a final JSON reply")

    def close(self) -> None:
        return None


class GenericCLISession:
    """Stateless-or-resumable adapter launching any configurable local agent CLI binary."""

    def __init__(
        self,
        workspace: Path,
        state_dir: Path,
        log_dir: Path,
        instructions: str,
        *,
        cli_bin: str | None = None,
        extra_args: list[str] | None = None,
        prompt_flag: str | None = None,
    ) -> None:
        resolved = cli_bin or os.environ.get("ELP_AGENT_CLI", "")
        if not resolved:
            raise RuntimeError("GenericCLISession requires `cli_bin` or ELP_AGENT_CLI env var")
        exe = shutil.which(resolved) or resolved
        if not Path(exe).exists() and shutil.which(resolved) is None:
            raise FileNotFoundError(f"Could not locate agent CLI binary: {resolved!r}")
        self.workspace = workspace
        self.state_dir = state_dir
        self.log_dir = log_dir
        self.instructions = instructions
        self.cli_bin = exe
        self.extra_args = list(extra_args or [])
        self.prompt_flag = prompt_flag
        self.history: list[dict[str, str]] = []
        state_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    def turn(self, prompt: str, *, turn_dir: Path | None = None, timeout: int = 600) -> dict[str, Any]:
        if turn_dir is not None:
            turn_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(turn_dir / "prompt.json", {"prompt": prompt, "at": utc_now()})

        parts = [f"SYSTEM INSTRUCTIONS:\n{self.instructions.strip()}"]
        if self.history:
            recent = self.history[-2:]
            parts.append("RECENT HISTORY:\n" + json.dumps(recent, ensure_ascii=False))
        parts.append("CURRENT OBSERVATION:\n" + prompt)
        full_prompt = "\n\n".join(parts)

        argv = [self.cli_bin, *self.extra_args]
        stdin_text: str | None = full_prompt
        if self.prompt_flag:
            argv.extend([self.prompt_flag, full_prompt])
            stdin_text = None

        env = _sanitized_cli_env()
        proc = subprocess.run(
            argv,
            cwd=str(self.workspace),
            input=stdin_text,
            text=True,
            encoding="utf-8",
            capture_output=True,
            env=env,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Agent CLI exited with {proc.returncode}: {(proc.stderr or proc.stdout)[:400]}"
            )
        parsed = extract_json_object(proc.stdout)
        self.history.append({"prompt": prompt[:500], "response": json.dumps(parsed)[:500]})
        if turn_dir is not None:
            write_json_atomic(turn_dir / "response.json", {"parsed": parsed, "at": utc_now()})
        return parsed

    def close(self) -> None:
        return None


_MOCK_COLLECTOR_CODE = """import torch

def collect(layer_idx, pass_idx, x, y, info):
    del info
    r = y - x
    r_rms = torch.sqrt(torch.mean(r * r))
    x_rms = torch.sqrt(torch.mean(x * x)).clamp(min=1e-12)
    return {
        "custom_rel_residual": float(r_rms / x_rms),
        "custom_layer_index": float(layer_idx),
    }
"""


class MockAgentSession:
    """Deterministic offline session exercising programmable diagnostics, edits, and evolution."""

    def __init__(self, *, exercise_edit: bool = True, exercise_diagnostic: bool = True) -> None:
        self.exercise_edit = exercise_edit
        self.exercise_diagnostic = exercise_diagnostic
        self._did_diag = False
        self._did_edit = False
        self._counter = 0

    def turn(self, prompt: str, *, turn_dir: Path | None = None, timeout: int = 600) -> dict[str, Any]:
        del timeout
        obs = json.loads(prompt)
        round_idx = int(obs.get("round", 0))
        requested = int(obs.get("requested_count", 2))

        if self.exercise_diagnostic and not self._did_diag and round_idx == 0:
            self._did_diag = True
            out = {
                "action": "run_diagnostic",
                "round": round_idx,
                "question": "Probe initial residual RMS, repeat cosines, and custom relative residual before editing.",
                "probe": {"layers": [0, 1], "features": ["activation", "residual", "alignment", "repeat"]},
                "collector_code": _MOCK_COLLECTOR_CODE,
            }
            if turn_dir is not None:
                write_json_atomic(turn_dir / "response.json", out)
            return out

        if self.exercise_edit and not self._did_edit and round_idx == 0:
            self._did_edit = True
            out = {
                "action": "edit_code",
                "rationale": "Stage baseline notes and helper note inside workspace before proposing programs.",
                "edits": [
                    {
                        "path": "notes/search_plan.txt",
                        "content": "Explore damped Euler vs two-stage Heun operators over layers 1..2.\n",
                    }
                ],
            }
            if turn_dir is not None:
                write_json_atomic(turn_dir / "response.json", out)
            return out

        candidates = []
        for idx in range(requested):
            self._counter += 1
            k = 1 if (self._counter % 3 == 1 and round_idx > 0) else 2 + (idx % 2)
            step_scale = round(0.35 + 0.15 * ((idx + round_idx) % 4), 2)
            op = "heun" if (round_idx > 0 and idx % 2 == 1) else "euler"
            cid = f"r{round_idx:02d}_c{idx:02d}_{self._counter}"
            candidates.append({
                "id": cid,
                "code": evaluate_program_template(operator=op, k=k, step_scale=step_scale),
                "params": {"operator": op, "k": k, "step_scale": step_scale, "trial": self._counter},
                "hypothesis": f"Round {round_idx} {op} K={k}, step_scale={step_scale} Pareto probe.",
                "metadata": {"operator": op, "k": k, "step_scale": step_scale},
            })
        out = {
            "action": "evaluate_candidates",
            "round": round_idx,
            "rationale": f"Propose {len(candidates)} candidate loop programs for round {round_idx}.",
            "candidates": candidates,
        }
        if turn_dir is not None:
            write_json_atomic(turn_dir / "response.json", out)
        return out

    def close(self) -> None:
        return None


def create_agent_session(
    backend: str,
    *,
    workspace: Path,
    state_dir: Path,
    log_dir: Path,
    instructions: str,
    model: str | None = None,
    cli_bin: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    tool_handler: Callable[[dict[str, Any]], Any] | None = None,
    exercise_edit: bool = True,
    exercise_diagnostic: bool = True,
) -> AgentSession:
    """Factory creating a search-agent session (`codex`, `openai`, `cli`, or `mock`)."""
    kind = backend.strip().lower()
    if kind == "mock":
        return MockAgentSession(
            exercise_edit=exercise_edit,
            exercise_diagnostic=exercise_diagnostic,
        )
    if kind == "codex":
        argv = [cli_bin, "app-server"] if cli_bin else None
        return CodexCLISession(
            workspace=workspace,
            state_dir=state_dir,
            log_dir=log_dir,
            instructions=instructions,
            launch_argv=argv,
            dynamic_tools=[WORKSPACE_EXEC_TOOL_SCHEMA] if tool_handler is not None else None,
            dynamic_tool_handler=tool_handler,
        )
    if kind == "openai":
        return OpenAICompatibleSession(
            state_dir=state_dir,
            log_dir=log_dir,
            instructions=instructions,
            model=model,
            base_url=base_url,
            api_key=api_key,
            tool_handler=tool_handler,
        )
    if kind == "cli":
        return GenericCLISession(
            workspace=workspace,
            state_dir=state_dir,
            log_dir=log_dir,
            instructions=instructions,
            cli_bin=cli_bin,
        )
    raise ValueError(f"Unknown agent backend {backend!r}; expected codex, openai, cli, or mock")


__all__ = [
    "AgentSession",
    "CodexCLISession",
    "DIAGNOSTIC_STEP_PROGRAM",
    "GenericCLISession",
    "MockAgentSession",
    "OpenAICompatibleSession",
    "create_agent_session",
    "extract_json_object",
    "public_event",
    "utc_now",
    "write_json_atomic",
]
