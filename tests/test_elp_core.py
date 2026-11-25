"""Unit tests for ELP core AST sandbox, invoke_loop_program, programmable probes, Pareto & search."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elp import (
    DiagnosticCollector,
    DiagnosticRequest,
    EvalResult,
    EvaluateRequest,
    LoopProgram,
    ProbeSpec,
    SearchConfig,
    SearchCoordinator,
    freeze_program,
    hypervolume,
    invoke_loop_program,
    load_frozen_program,
    load_program_callable,
    pareto_frontier,
    program_fingerprint,
    select_representatives,
    validate_program_source,
)
from elp.backends.text import DoryLoopConfig, DoryLoopPatch
from elp.modalities.image import (
    AdaptiveLoopedDiTSampler,
    CachedLoopGuidance,
    ImageLoopConfig,
    LoopedDiTScheduleConfig,
    euler_loop,
    heun_loop,
    loopguidance_combine,
    midpoint_loop,
    patch_image_blocks,
    warped_snr_schedule,
)
from elp.modalities.text import (
    TextLoopConfig,
    WindowLoopPatch,
    geodesic_extrapolate,
    iterate_window,
)
from elp.modalities.video import (
    CausalWorldModelLoopPatch,
    VideoDiffusionLoopConfig,
    VideoWorldModelLoopConfig,
    expanded_layer_plan,
    patch_video_diffusion_blocks,
)


class DummyBlock(nn.Module):
    def __init__(self, dim: int = 8, scale: float = 0.25, with_cache: bool = False) -> None:
        super().__init__()
        self.scale = scale
        self.with_cache = with_cache
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim) * (1.0 + scale))
        self.calls = 0

    def forward(self, x: Tensor, **kwargs):
        self.calls += 1
        out = x + self.scale * (self.linear(x) - x)
        if self.with_cache:
            step_val = int(kwargs.get("cache_tick", self.calls))
            return out, (step_val, step_val * 2, {"pass": self.calls})
        return out


class TinyStack(nn.Module):
    def __init__(self, depth: int = 6, dim: int = 8, attr: str = "blocks", with_cache: bool = False) -> None:
        super().__init__()
        setattr(self, attr, nn.ModuleList([DummyBlock(dim, with_cache=with_cache) for _ in range(depth)]))
        self._attr = attr

    def forward(self, x: Tensor) -> Tensor:
        for blk in getattr(self, self._attr):
            res = blk(x)
            x = res[0] if isinstance(res, tuple) else res
        return x


class TestProgramSandbox(unittest.TestCase):
    VALID_CODE = '''import math
import torch

def helper(t, s):
    return t * s

def loop(x, evaluate, params):
    r0 = evaluate(x)
    r1 = evaluate(x + helper(r0, float(params["alpha"])))
    return x + 0.5 * (r0 + r1)
'''

    def test_valid_sandbox_and_freeze_roundtrip(self) -> None:
        validate_program_source(self.VALID_CODE)
        fn = load_program_callable(self.VALID_CODE)
        x = torch.ones(2, 4)
        out = fn(x, lambda h: h * 0.5, {"alpha": 0.5})
        self.assertEqual(out.shape, x.shape)

        prog_a = LoopProgram(id="cand_a", code=self.VALID_CODE, params={"alpha": 0.5, "b": 1})
        prog_b = LoopProgram(id="cand_b", code=self.VALID_CODE + "\n# trailing comment\n", params={"b": 1, "alpha": 0.5})
        self.assertEqual(program_fingerprint(prog_a), program_fingerprint(prog_b))

        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "frozen.json"
            freeze_program(prog_a, snap)
            loaded = load_frozen_program(snap)
            self.assertEqual(loaded, prog_a)
            snap.write_text(snap.read_text(encoding="utf-8").replace("0.5", "0.9"), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_frozen_program(snap)

    def test_rejects_unsafe_constructs(self) -> None:
        bad_sources = [
            "import os\ndef loop(x):\n    return x\n",
            "from math import sqrt\ndef loop(x):\n    return x\n",
            "import torch\ndef loop(x):\n    return x.cuda()\n",
            "import torch\ndef loop(x):\n    x.tag = 1\n    return x\n",
            "import torch\ndef loop(x):\n    return eval('x')\n",
            "import torch\ndef loop(x):\n    return x.__class__\n",
            "import torch\nX = 1\ndef loop(x):\n    return x\n",
        ]
        for src in bad_sources:
            with self.assertRaises(ValueError, msg=f"Should reject:\n{src}"):
                validate_program_source(src)

    def test_invoke_loop_program_variants_budget_and_mutation_guard(self) -> None:
        x = torch.ones(2, 4)
        eval_fn = lambda h: 2.0 * h

        # 2-arg, 3-arg, 5-arg invocation
        fn2 = load_program_callable("import torch\ndef loop(x, params):\n    return x * float(params['s'])\n")
        self.assertTrue(torch.allclose(invoke_loop_program(fn2, x, eval_fn, params={"s": 3.0}), 3.0 * x))

        fn3 = load_program_callable("import torch\ndef loop(x, evaluate, params):\n    return evaluate(x)\n")
        self.assertTrue(torch.allclose(invoke_loop_program(fn3, x, eval_fn, params={}), 2.0 * x))

        fn5 = load_program_callable(
            "import torch\ndef loop(x, evaluate, info, params, memory):\n"
            "    memory['seen'] = info['layer']\n"
            "    r0 = evaluate(x) - x\n"
            "    r1 = evaluate(x + 0.5 * r0) - (x + 0.5 * r0)\n"
            "    return {'residuals': [r0, r1], 'weights': [0.4, 0.6]}\n"
        )
        mem: dict = {}
        out_dict = invoke_loop_program(
            fn5,
            x,
            eval_fn,
            info={"layer": 3},
            params={},
            memory=mem,
            enforce_unit_budget=True,
        )
        self.assertEqual(mem["seen"], 3)
        self.assertEqual(out_dict.shape, x.shape)

        # Reject invalid unit budget when enforce_unit_budget=True
        bad_budget = load_program_callable(
            "import torch\ndef loop(x, evaluate, info, params, memory):\n"
            "    return {'residuals': [evaluate(x) - x], 'weights': [0.5]}\n"
        )
        with self.assertRaises(ValueError):
            invoke_loop_program(bad_budget, x, eval_fn, enforce_unit_budget=True)

        # Reject in-place input tensor mutation
        def mutating_loop(inp, _ev, _p):
            inp.zero_()
            return inp

        with self.assertRaises(ValueError):
            invoke_loop_program(mutating_loop, x.clone(), eval_fn, params={})


class TestParetoDiagnosticsAndSearch(unittest.TestCase):
    def test_pareto_hypervolume_and_representatives(self) -> None:
        pts = {
            "q": (0.90, 0.20),
            "b": (0.80, 0.70),
            "s": (0.50, 0.90),
            "dom": (0.75, 0.60),
        }
        front = pareto_frontier(pts)
        self.assertEqual(set(front), {"q", "b", "s"})
        hv = hypervolume([pts[k] for k in pts], reference=(0.0, 0.0))
        expected_hv = 0.50 * 0.90 + (0.80 - 0.50) * 0.70 + (0.90 - 0.80) * 0.20
        self.assertAlmostEqual(hv, expected_hv, places=7)
        reps = select_representatives(pts)
        self.assertEqual(reps, {"quality": "q", "speed": "s", "balanced": "b"})

    def test_diagnostic_collector_builtin_and_programmable_collector_code(self) -> None:
        model = TinyStack(depth=4, dim=8, attr="blocks", with_cache=True).eval()
        x = torch.randn(2, 16, 8)
        collector_src = '''import torch

def collect(layer_idx, pass_idx, x, y, info):
    del info
    diff = y - x
    norm = torch.sqrt(torch.mean(diff * diff))
    return {
        "prog_delta_rms": float(norm),
        "prog_pass": float(pass_idx),
        "prog_layer": float(layer_idx),
    }
'''
        probe = ProbeSpec(
            layers=(1, 2),
            features=("activation", "residual", "alignment", "repeat", "cache"),
            max_tokens=8,
            collector_code=collector_src,
        )
        x_orig = x.clone()
        with DiagnosticCollector(model, probe) as diag:
            _ = model(x)
            diag._repeats[1] = 1
            blk = model.blocks[1]
            _ = blk(x, kv_cache={"global_end_index": 100}, cache_tick=120)

        self.assertTrue(torch.equal(x, x_orig))
        rows = diag.summary()
        self.assertIn((1, 1), rows)
        self.assertIn((1, 2), rows)
        self.assertIn("residual_over_input_rms", rows[(1, 1)])
        self.assertIn("previous_pass_residual_cosine", rows[(1, 2)])
        self.assertIn("kv_proposed_global_end", rows[(1, 2)])
        self.assertIn("prog_delta_rms", rows[(1, 1)])
        self.assertEqual(rows[(1, 2)]["prog_pass"], 2.0)

        # Reject missing entrypoint, unsafe import, or in-place suffix calls in collector_code
        for bad in (
            "import os\ndef collect(l, p, x, y, i):\n    return {'a': 1.0}\n",
            "import torch\ndef wrong_name(l, p, x, y, i):\n    return {'a': 1.0}\n",
            "import torch\ndef collect(l, p, x, y, i):\n    x.zero_()\n    return {'a': 1.0}\n",
        ):
            with self.assertRaises(ValueError):
                ProbeSpec(layers=(0,), collector_code=bad)

    def test_search_coordinator_diagnostics_dedup_resume_and_control_check(self) -> None:
        src = "import torch\ndef loop(x, p):\n    return x * float(p['k'])\n"

        class MiniProposer:
            def __init__(self) -> None:
                self.calls = 0
                self.seen_rounds: list[int] = []

            def propose(self, obs, remaining):
                self.calls += 1
                self.seen_rounds.append(int(obs["round"]))
                if self.calls == 1 and obs["round"] == 0:
                    return DiagnosticRequest(0, "Check layers?", {"layers": [0, 1]})
                if self.calls == 2 and obs["round"] == 0:
                    return EvaluateRequest(
                        0,
                        "dup batch check",
                        [
                            LoopProgram("a1", src, {"k": 1.0}),
                            LoopProgram("a2", src, {"k": 1.0}),
                        ],
                    )
                return EvaluateRequest(
                    obs["round"],
                    "valid batch",
                    [
                        LoopProgram(f"r{obs['round']}_c{i}", src, {"k": 1.0 + i + obs["round"] * 0.5})
                        for i in range(remaining)
                    ],
                )

        class MiniEval:
            def __init__(self, control_seq: list[str] | None = None) -> None:
                self.control_calls = 0
                self._seq = list(control_seq or ["digest_ok", "digest_ok", "digest_ok", "digest_ok"])

            def diagnose(self, req):
                return {"ok": True, "layers": req.probe["layers"]}

            def evaluate(self, prog):
                k = float(prog.params["k"])
                return EvalResult(prog.id, "complete", objectives=(k, 1.0 / k), metrics={"k": k})

            def control_digest(self) -> str:
                val = self._seq[min(self.control_calls, len(self._seq) - 1)]
                self.control_calls += 1
                return val

        with tempfile.TemporaryDirectory() as tmp:
            ev = MiniEval()
            coord = SearchCoordinator(
                MiniProposer(),
                ev,
                SearchConfig(round_sizes=(2, 2)),
                control_check=ev.control_digest,
            )
            res = coord.run(freeze_dir=tmp)
            self.assertEqual(len(res.completed), 4)
            self.assertEqual(len(res.diagnostics), 1)
            self.assertEqual(ev.control_calls, 4)
            self.assertLessEqual(res.hypervolume_history[0], res.hypervolume_history[1] + 1e-12)
            for role, cid in res.representatives.items():
                self.assertTrue((Path(tmp) / f"{role}_{cid}.json").is_file())

            # Resume skipping Round 0 completely
            r0_items = [
                (0, prog, eval_res)
                for cid, (prog, eval_res) in res.completed.items()
                if cid.startswith("r0_")
            ]
            resumed_proposer = MiniProposer()
            ev_resume = MiniEval()
            coord_resume = SearchCoordinator(
                resumed_proposer,
                ev_resume,
                SearchConfig(round_sizes=(2, 2)),
                control_check=ev_resume.control_digest,
                initial_completed=r0_items,
                initial_diagnostics=res.diagnostics,
            )
            res_resumed = coord_resume.run()
            self.assertEqual(resumed_proposer.seen_rounds, [1])
            self.assertEqual(ev_resume.control_calls, 2)
            self.assertEqual(len(res_resumed.completed), 4)

            # Control digest drift must raise RuntimeError
            drifted_eval = MiniEval(["before_digest", "after_drifted_digest"])
            coord_drift = SearchCoordinator(
                MiniProposer(),
                drifted_eval,
                SearchConfig(round_sizes=(2,)),
                control_check=drifted_eval.control_digest,
            )
            with self.assertRaises(RuntimeError):
                coord_drift.run()


class TestImageModality(unittest.TestCase):
    def test_operators_and_reversible_patch(self) -> None:
        x = torch.ones(2, 4, 8)
        fn = lambda h: 2.0 * h
        out_euler = euler_loop(fn, x, num_loops=2, step_scale=0.5)
        self.assertTrue(torch.allclose(out_euler, 2.25 * x))
        self.assertTrue(torch.allclose(heun_loop(fn, x, 1, 0.5), 1.625 * x))
        self.assertTrue(torch.allclose(midpoint_loop(fn, x, 1, 0.5), 1.625 * x))

        model = TinyStack(depth=32, dim=8, attr="blocks").eval()
        loop_cfg = ImageLoopConfig(blocks=(12, 13, 14, 15), num_loops=2, step_scale=0.5, start_frac=0.0, end_frac=0.5)
        guidance = CachedLoopGuidance(weight=3.0, refresh_interval=2)
        step_state = {"progress": 0.1, "reference_mode": False}
        orig_forward_ref = model.blocks[12].forward
        with torch.no_grad():
            step_state["reference_mode"] = True
            ref_out = model(x)
            step_state["reference_mode"] = False
            with patch_image_blocks(model.blocks, loop_cfg, step_state):
                looped_out = model(x)
                step_state["reference_mode"] = True
                bypass_out = model(x)
                self.assertTrue(torch.allclose(ref_out, bypass_out))
                self.assertFalse(torch.allclose(ref_out, looped_out))
            self.assertEqual(model.blocks[12].forward, orig_forward_ref)
            self.assertNotIn("forward", vars(model.blocks[12]))
            restored_out = model(x)
            self.assertTrue(torch.allclose(ref_out, restored_out))

        guidance.reset()
        p0 = guidance.predict(0, pred_loop_fn=lambda: x * 2.0, pred_ref_fn=lambda: x, active=True)
        self.assertTrue(torch.allclose(p0, loopguidance_combine(x, x * 2.0, 3.0)))
        p1 = guidance.predict(1, pred_loop_fn=lambda: x * 2.0, pred_ref_fn=lambda: x, active=True)
        self.assertEqual((guidance.reference_calls, guidance.loop_calls), (1, 2))
        self.assertTrue(torch.allclose(p1, x * 4.0))

    def test_looped_dit_adaptive_sampler(self) -> None:
        sched = warped_snr_schedule(14, curvature=1.38)
        self.assertEqual(len(sched), 14)
        self.assertAlmostEqual(sum(dt for _, dt in sched), 1.0, places=6)

        sampler = AdaptiveLoopedDiTSampler(LoopedDiTScheduleConfig(step_count=14, refresh_steps=(0, 3, 7, 10)))
        x = torch.randn(1, 8, 8)
        txt = torch.randn(1, 4, 8)
        out, stats = sampler.sample(
            x,
            txt,
            pre_blocks=lambda xi, ti: (xi, ti),
            core_step=lambda xi, ti: (0.9 * xi, 0.9 * ti),
            post_blocks=lambda xi, _ti: xi,
        )
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(stats["steps"], 14)
        self.assertGreater(stats["core_passes"], 14)


class TestTextModality(unittest.TestCase):
    def test_strategies_and_geodesic_extrapolate(self) -> None:
        x = torch.ones(2, 4, 8)
        g = lambda u: 1.5 * u
        for strategy in ("euler", "anchored_euler", "heun", "midpoint", "rk4", "heavy_ball", "anderson", "norm_stab"):
            cfg = TextLoopConfig(
                start_layer=0,
                end_layer=1,
                num_loops=2,
                strategy=strategy,
                anchor_weight=0.5 if strategy == "anchored_euler" else 0.0,
            )
            y = iterate_window(g, x, cfg)
            self.assertEqual(y.shape, x.shape)
            self.assertTrue(torch.isfinite(y).all())

        h_prev = torch.randn(2, 5, 16)
        h_curr = h_prev + 0.2 * torch.randn(2, 5, 16)
        extrap = geodesic_extrapolate(h_prev, h_curr, scale=0.25, max_angle=0.35)
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(extrap, dim=-1),
                torch.linalg.vector_norm(h_curr, dim=-1),
                atol=1e-5,
            )
        )

    def test_dual_window_and_dory_patch_restore(self) -> None:
        model = TinyStack(depth=36, dim=8, attr="blocks").eval()
        x = torch.randn(2, 6, 8)
        with torch.no_grad():
            base = model(x)

        w1 = TextLoopConfig(start_layer=12, end_layer=14, num_loops=2, strategy="rk4")
        w2 = TextLoopConfig(start_layer=22, end_layer=24, num_loops=2, strategy="euler")
        with WindowLoopPatch(model, (w1, w2)) as patch:
            looped = model(x)
            self.assertFalse(torch.allclose(base, looped))
            self.assertEqual(patch.stats["window_invocations"], 2)
            self.assertEqual(patch.stats["stage_evaluations"], 10)
        self.assertTrue(torch.allclose(base, model(x)))

        class DummyDory(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = type("Cfg", (), {"n_recurrent_loops": 4, "recurrent_kv_cache_mode": "per_loop"})()
                self.model = type("Inner", (), {"_loop_runner": None})()

        dory = DummyDory().eval()
        cfg = DoryLoopConfig(num_loops=2, recurrent_kv_cache_mode="last_loop", geodesic_scale=0.18)
        with DoryLoopPatch(dory, cfg):
            self.assertEqual(dory.config.n_recurrent_loops, 2)
            self.assertEqual(dory.config.recurrent_kv_cache_mode, "last_loop")
            self.assertIsNotNone(dory.model._loop_runner)
            h_in = torch.randn(1, 4, 8)
            out = dory.model._loop_runner(lambda s, _k: s + 0.1, h_in)
            self.assertEqual(out.shape, h_in.shape)
        self.assertEqual(dory.config.n_recurrent_loops, 4)
        self.assertEqual(dory.config.recurrent_kv_cache_mode, "per_loop")
        self.assertIsNone(dory.model._loop_runner)


class TestVideoModality(unittest.TestCase):
    def test_causal_world_model_layerwise_windowwise_and_kv_commit(self) -> None:
        model = TinyStack(depth=30, dim=8, attr="blocks", with_cache=True).eval()
        x = torch.randn(1, 8, 8)

        cfg_ww = VideoWorldModelLoopConfig(kind="windowwise", start_layer=11, end_layer=18, k=6, active_steps=(0, 1), active_chunks=(1, 2))
        cfg_k4 = VideoWorldModelLoopConfig(kind="windowwise", start_layer=12, end_layer=16, k=4, active_steps=(0,), active_chunks=(0,))
        self.assertEqual(cfg_k4.k, 4)
        self.assertEqual(len(expanded_layer_plan(30, cfg_ww)), 11 + 8 * 6 + 11)

        cfg_lw = VideoWorldModelLoopConfig(kind="layerwise", start_layer=12, end_layer=14, k=3, active_steps=(0, 1), active_chunks=(0, 1))
        patch_lw = CausalWorldModelLoopPatch(model.blocks, cfg_lw)
        kv_cache: dict[int, tuple[int, int, dict]] = {}
        out_lw = patch_lw.forward_blocks(x, step_index=0, chunk_index=0, kv_cache=kv_cache)
        self.assertEqual(patch_lw.execution_trace, expanded_layer_plan(30, cfg_lw))
        self.assertEqual(out_lw.shape, x.shape)
        self.assertEqual(kv_cache[0][2]["pass"], 1)
        self.assertEqual(kv_cache[12][2]["pass"], 3)

        before_calls = model.blocks[11].calls
        patch_ww = CausalWorldModelLoopPatch(model.blocks, cfg_ww)
        _ = patch_ww.forward_blocks(x, step_index=0, chunk_index=0, is_clean_refresh=True)
        self.assertEqual(patch_ww.execution_trace, list(range(30)))
        self.assertEqual(model.blocks[11].calls - before_calls, 1)

        kv_ww: dict[int, tuple] = {}
        _ = patch_ww.forward_blocks(x, step_index=1, chunk_index=2, is_clean_refresh=False, kv_cache=kv_ww)
        self.assertEqual(patch_ww.execution_trace, expanded_layer_plan(30, cfg_ww))
        self.assertEqual(model.blocks[11].calls - before_calls, 1 + 6)
        self.assertEqual(kv_ww[11][2]["pass"], model.blocks[11].calls)

    def test_video_diffusion_reversible_patch(self) -> None:
        model = TinyStack(depth=12, dim=8, attr="blocks").eval()
        x = torch.randn(1, 6, 8)
        with torch.no_grad():
            base = model(x)
        cfg = VideoDiffusionLoopConfig(blocks=(4, 5), repeats=3, strength=1.0, start_frac=0.0, end_frac=0.5)
        ctx = {"step_index": 0, "step_count": 10}
        with patch_video_diffusion_blocks(model.blocks, cfg, ctx):
            out_active = model(x)
            ctx["step_index"] = 8
            out_inactive = model(x)
            self.assertFalse(torch.allclose(base, out_active))
            self.assertTrue(torch.allclose(base, out_inactive))
        self.assertTrue(torch.allclose(base, model(x)))


if __name__ == "__main__":
    unittest.main()
