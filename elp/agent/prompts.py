"""System prompts, structured JSON protocol specs, and safe loop templates for ELP agents."""
from __future__ import annotations

DIAGNOSTIC_STEP_PROGRAM = """import torch

def loop(x, evaluate, params):
    step_scale = float(params.get("step_scale", 0.5))
    k = int(params.get("k", 2))
    cur = x
    for _ in range(k):
        y = evaluate(cur)
        cur = cur + step_scale * (y - cur)
    return cur
"""


def evaluate_program_template(*, operator: str = "euler", k: int = 2, step_scale: float = 0.5) -> str:
    """Return a validated pure-tensor loop program snippet for demos or initial seeds."""
    if operator == "heun":
        return f'''import torch

def loop(x, evaluate, params):
    """Damped Heun residual refinement (K={k}, step_scale={step_scale})."""
    step_scale = float(params.get("step_scale", {step_scale}))
    k = int(params.get("k", {k}))
    cur = x
    for _ in range(k):
        k1 = evaluate(cur) - cur
        probe = cur + step_scale * k1
        k2 = evaluate(probe) - probe
        cur = cur + 0.5 * step_scale * (k1 + k2)
    return cur
'''
    return f'''import torch

def loop(x, evaluate, params):
    """Damped Euler residual refinement (K={k}, step_scale={step_scale})."""
    step_scale = float(params.get("step_scale", {step_scale}))
    k = int(params.get("k", {k}))
    cur = x
    for _ in range(k):
        y = evaluate(cur)
        cur = cur + step_scale * (y - cur)
    return cur
'''


def build_system_prompt(
    backbone: str,
    *,
    modality: str,
    allow_diagnostics: bool = True,
    allow_code_edits: bool = True,
    objective_names: tuple[str, ...] = ("quality", "speedup"),
) -> str:
    """Construct the closed-loop search prompt enforcing AST and workspace contracts."""
    actions = ['"evaluate_candidates"']
    if allow_diagnostics:
        actions.insert(0, '"run_diagnostic"')
    if allow_code_edits:
        actions.insert(0, '"edit_code"')
    action_list = ", ".join(actions)
    objs = ", ".join(objective_names)

    return f"""You are an automated research engineer running Evolving Loop Programs (ELP) offline search on a frozen `{backbone}` backbone (modality: {modality}).

Goal:
Discover executable loop programs `(code, params)` that strictly advance the non-dominated Pareto frontier over ({objs}) with zero weight updates.

Reply strictly with a single valid JSON object. Allowed `"action"` values: {action_list}.

1. Evaluation batch format:
{{
  "action": "evaluate_candidates",
  "round": <int>,
  "rationale": "<short hypothesis explaining why these edits improve structure/dynamics/control>",
  "candidates": [
    {{
      "id": "<unique alphanumeric/underscore id>",
      "code": "<Python defining `def loop(x, evaluate, params):` using only `torch` and `math`; `evaluate(h)` returns the full layer/window output F(h) so the residual is `evaluate(h) - h`>",
      "params": {{"k": 2, "step_scale": 0.5, ...}},
      "hypothesis": "<concise note>",
      "metadata": {{}}
    }}
  ]
}}

2. Read-only diagnostic probe format (does not consume candidate budget):
{{
  "action": "run_diagnostic",
  "round": <int>,
  "question": "<what residual/alignment/repeat/custom statistic you want to measure>",
  "probe": {{"layers": [0, 1], "features": ["activation", "residual", "alignment", "repeat"]}},
  "collector_code": "<optional Python defining `def collect(layer_idx, pass_idx, x, y, info): -> dict[str, float]`>"
}}

3. Workspace edit format (restricted to `candidate.py`, `algorithms/`, `notes/`):
{{
  "action": "edit_code",
  "rationale": "<summary of edit>",
  "edits": [{{"path": "notes/hypothesis.txt", "content": "..."}}]
}}

Safety rules for `code` and `collector_code`:
- `evaluate(h)` returns full layer/window output `F(h)` (the residual update is `evaluate(h) - h`).
- Only `import torch` and `import math` (no `from ... import ...`, no `os`/`sys`/`eval`, no private `_attr` or in-place `attr_` calls, no classes, no module-level state).
- Never mutate input tensors in-place or modify model parameters (`collector_code` receives detached cloned tensors and must return finite scalar floats).
"""
