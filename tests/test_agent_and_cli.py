"""Unit tests for sandbox, subprocess scorer, programmable diagnostics, resume, control guard & CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elp import load_frozen_program
from elp.agent import (
    AgentSearchRunner,
    WorkspaceTool,
    apply_workspace_edits,
    extract_json_object,
    public_event,
    validate_relative_file,
)
from elp.agent.worker import load_scorer, run_worker
from elp.backends.text import DoryConfig, DoryForCausalLM
from elp.cli import _resolve_program_arg, _save_inference_output, main as cli_main
from elp.core import SearchConfig


def dummy_real_scorer(*, adapter, model, program, baseline_result, program_result, inputs) -> dict:
    """Module-level scorer imported by worker tests via `tests.test_agent_and_cli:dummy_real_scorer`."""
    del adapter, model, program
    base_outs = baseline_result.output if isinstance(baseline_result.output, list) else [baseline_result.output]
    prog_outs = program_result.output if isinstance(program_result.output, list) else [program_result.output]
    diffs = [
        float((p.logits.float() - b.logits.float()).abs().mean().item())
        for b, p in zip(base_outs, prog_outs)
    ]
    diff = sum(diffs) / max(1, len(diffs))
    n_items = float(len(inputs)) if isinstance(inputs, list) else 1.0
    quality = round(0.5 + min(0.4, diff + 0.05 * n_items), 6)
    return {
        "quality": quality,
        "metrics": {"custom_diff": diff, "custom_score": quality, "eval_items": n_items},
    }


def _create_tiny_local_dory_dir(target_dir: Path) -> Path:
    """Save a tiny self-contained local Dory safetensors checkpoint + offline tokenizer."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2, "hello": 3, "world": 4, "loop": 5}
    tok_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok_backend, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
    tok.save_pretrained(str(target_dir))

    cfg = DoryConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        n_input_layers=1,
        n_recurrent_layers=2,
        n_output_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_recurrent_loops=2,
    )
    (target_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    model = DoryForCausalLM(cfg).eval()
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(target_dir / "model.safetensors"))
    return target_dir


class TestAgentSandboxAndHelpers(unittest.TestCase):
    def test_validate_relative_and_apply_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            ok = validate_relative_file(ws, "algorithms/op.py")
            self.assertEqual(ok, (ws / "algorithms" / "op.py").resolve())

            for bad in ("/etc/passwd", "../escape.py", "C:/Windows/win.ini", ""):
                with self.assertRaises(ValueError):
                    validate_relative_file(ws, bad)

            audit1 = apply_workspace_edits(
                ws,
                [{"path": "candidate.py", "content": "def loop(x, eval_fn, p):\n    return eval_fn(x)\n"}],
            )
            self.assertEqual(len(audit1), 1)
            self.assertIsNone(audit1[0]["before_sha256"])

            audit2 = apply_workspace_edits(
                ws,
                [{"path": "candidate.py", "old": "return eval_fn(x)", "new": "return 0.5 * (x + eval_fn(x))"}],
            )
            self.assertEqual(audit2[0]["before_sha256"], audit1[0]["after_sha256"])

            with self.assertRaises(ValueError):
                apply_workspace_edits(ws, [{"path": "elp/hack.py", "content": "x = 1\n"}])
            with self.assertRaises(SyntaxError):
                apply_workspace_edits(ws, [{"path": "candidate.py", "content": "def broken(:\n"}])

    def test_workspace_tool_and_json_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            tool = WorkspaceTool(ws, state_dir=ws / "state")
            (ws / "probe_env.py").write_text(
                "import os\nprint(os.environ.get('OPENAI_API_KEY', 'STRIPPED'), os.environ.get('HF_HUB_OFFLINE'))\n",
                encoding="utf-8",
            )
            prev = os.environ.get("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = "secret-test-key"
            try:
                res = tool({"tool": "workspace_exec", "arguments": {"command": "python probe_env.py", "timeout_seconds": 15}})
                self.assertEqual(res["returncode"], 0, msg=res.get("output"))
                self.assertFalse(res["timeout"])
                self.assertIn("STRIPPED 1", res["output"])
            finally:
                if prev is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = prev

            with self.assertRaises(ValueError):
                tool({"tool": "other_tool", "arguments": {"command": "echo hi"}})
            with self.assertRaises(ValueError):
                tool({"tool": "workspace_exec", "arguments": {"command": "echo hi", "timeout_seconds": 999}})

        parsed = extract_json_object('Sure, here is the proposal:\n```json\n{"action": "run_diagnostic", "round": 0}\n```')
        self.assertEqual(parsed["action"], "run_diagnostic")

        redacted = public_event({"method": "item/reasoning/textDelta", "content": "secret"})
        self.assertEqual(redacted["omitted"], "raw_reasoning_not_collected")
        nested = public_event({"type": "reasoning", "content": "hidden", "encrypted_content": "enc", "id": "r1"})
        self.assertEqual(nested, {"type": "reasoning", "id": "r1"})


class TestCoordinatorWorkerScorerAndCLI(unittest.TestCase):
    def test_end_to_end_mock_search_programmable_diag_control_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            # First run Round 0 only (2 candidates) with control_check=True
            r0_runner = AgentSearchRunner(
                workspace=ws,
                backbone="toy",
                modality="toy",
                agent_backend="mock",
                search_config=SearchConfig(round_sizes=(2,), reference_point=(0.0, 0.0)),
                control_check=True,
            )
            r0_res = r0_runner.run()
            self.assertEqual(len(r0_res.completed), 2)
            self.assertEqual(len(r0_res.diagnostics), 1)
            diag_metrics = r0_res.diagnostics[0]["result"]["rows"][0]["metrics"]
            self.assertIn("custom_rel_residual", diag_metrics)
            self.assertIn("custom_layer_index", diag_metrics)

            diag_lines = (ws / "state" / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(diag_lines), 1)
            trial_lines = (ws / "state" / "trials.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(trial_lines), 2)

            # Record worker calls count before resume
            calls_before = len(list((ws / "state" / "worker_calls").iterdir()))

            # Resume with expanded round_sizes=(2, 2): Round 0 must be skipped, only Round 1 evaluated
            resumed_runner = AgentSearchRunner(
                workspace=ws,
                backbone="toy",
                modality="toy",
                agent_backend="mock",
                search_config=SearchConfig(round_sizes=(2, 2), reference_point=(0.0, 0.0)),
                resume=True,
                control_check=True,
            )
            resumed_res = resumed_runner.run()
            self.assertEqual(len(resumed_res.completed), 4)
            trial_lines_after = (ws / "state" / "trials.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(trial_lines_after), 4)

            calls_after = len(list((ws / "state" / "worker_calls").iterdir()))
            # Control check (pre+post = 2) + Round 1 evaluations (2) = 4 new calls (NO Round 0 re-eval or extra diag)
            self.assertEqual(calls_after - calls_before, 4)

            exp = json.loads((ws / "state" / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(exp["status"], "complete")
            self.assertTrue(exp["resumed"])
            self.assertTrue(exp["control_check"])
            for role, cid in resumed_res.representatives.items():
                loaded = load_frozen_program(ws / "frozen" / f"{role}_{cid}.json")
                self.assertEqual(loaded.id, cid)

    def test_real_backbone_worker_requires_and_invokes_scorer(self) -> None:
        prog_dict = {
            "id": "cand_real_01",
            "code": "import torch\ndef loop(x, evaluate, params):\n    y = evaluate(x)\n    return x + 0.5 * (y - x)\n",
            "params": {},
        }
        missing = run_worker(
            "evaluate",
            {"backbone": "dory", "model_path": "/nonexistent", "program": prog_dict},
        )
        self.assertEqual(missing["status"], "failed")
        self.assertIn("scorer", missing["error"].lower())

        with self.assertRaises(ValueError):
            load_scorer("invalid_no_colon")

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = _create_tiny_local_dory_dir(Path(tmp) / "tiny_dory")
            eval_jsonl = Path(tmp) / "prompts.jsonl"
            eval_jsonl.write_text(
                json.dumps({"prompt": "hello world", "max_new_tokens": 0}) + "\n"
                + json.dumps({"prompt": "loop hello", "max_new_tokens": 0}) + "\n",
                encoding="utf-8",
            )
            res = run_worker(
                "evaluate",
                {
                    "backbone": "dory",
                    "model_path": str(ckpt_dir),
                    "program": prog_dict,
                    "scorer": "tests.test_agent_and_cli:dummy_real_scorer",
                    "eval_inputs_path": str(eval_jsonl),
                    "load_kwargs": {"dtype": "float32", "local_files_only": True},
                },
            )
            self.assertEqual(res["status"], "complete", msg=res.get("error"))
            self.assertGreater(res["metrics"]["custom_diff"], 0.0)
            self.assertEqual(res["metrics"]["eval_items"], 2.0)
            self.assertNotEqual(res["objectives"][0], 1.0)
            self.assertAlmostEqual(res["objectives"][0], res["metrics"]["custom_score"], places=6)

            ctrl1 = run_worker(
                "control",
                {
                    "backbone": "dory",
                    "model_path": str(ckpt_dir),
                    "eval_inputs_path": str(eval_jsonl),
                    "load_kwargs": {"dtype": "float32", "local_files_only": True},
                },
            )
            ctrl2 = run_worker(
                "control",
                {
                    "backbone": "dory",
                    "model_path": str(ckpt_dir),
                    "eval_inputs_path": str(eval_jsonl),
                    "load_kwargs": {"dtype": "float32", "local_files_only": True},
                },
            )
            self.assertEqual(ctrl1["status"], "complete", msg=ctrl1.get("error"))
            self.assertEqual(ctrl1["digest"], ctrl2["digest"])

    def test_cli_list_search_program_errors_and_save_output(self) -> None:
        self.assertEqual(cli_main(["list", "--json"]), 0)
        self.assertIsNone(_resolve_program_arg(None))
        self.assertIsNone(_resolve_program_arg("baseline"))
        with self.assertRaises(FileNotFoundError):
            _resolve_program_arg("nonexistent_preset_name")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "exports"
            txt_p = Path(_save_inference_output("decoded answer", out_dir / "out.txt", "text"))
            self.assertEqual(txt_p.read_text(encoding="utf-8").strip(), "decoded answer")

            pt_p = Path(_save_inference_output(torch.ones(2, 3), out_dir / "out.pt", "video"))
            self.assertTrue(torch.equal(torch.load(pt_p, weights_only=True), torch.ones(2, 3)))

            png_p = Path(_save_inference_output(torch.zeros(1, 3, 8, 8), out_dir / "out.png", "image"))
            self.assertTrue(png_p.read_bytes().startswith(b"\x89PNG"))

            ws = Path(tmp) / "cli_ws"
            rc = cli_main(["search", "--backbone", "toy", "--agent", "mock", "--workspace", str(ws), "--rounds", "2,2", "--control-check"])
            self.assertEqual(rc, 0)
            self.assertTrue((ws / "state" / "experiment.json").is_file())


if __name__ == "__main__":
    unittest.main()
