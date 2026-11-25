"""Faithful CPU structural contract tests for all 9 ELP backbone adapters, patches, and KV rollback."""
from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elp.backends import (
    BUNDLED_THIRD_PARTY_DIR,
    CausalWorldModelAdapter,
    DoryAdapter,
    DoryConfig,
    DoryForCausalLM,
    DoryLoopConfig,
    DoryLoopPatch,
    DoryRecurrentCache,
    HFDecoderAdapter,
    HFDecoderLoopPatch,
    LoopedDiTAdapter,
    RAEAdapter,
    ScaleRAEAdapter,
    Wan21Adapter,
    activate_upstream_repo,
    get_adapter,
    list_backbones,
    pinned_moe_routing,
    preserve_dynamic_cache,
    remap_dory_state_dict,
    resolve_upstream_repo,
)
from elp.core import LoopProgram, ProbeSpec
from elp.modalities.image import CachedLoopGuidance, ImageLoopConfig, LoopedDiTScheduleConfig
from elp.modalities.text import TextLoopConfig
from elp.modalities.video import VideoDiffusionLoopConfig, VideoWorldModelLoopConfig


_STANDARD_EULER_PROGRAM = """import torch

def loop(x, evaluate, params):
    step_scale = float(params.get("step_scale", 0.5))
    k = int(params.get("k", 2))
    cur = x
    for _ in range(k):
        y = evaluate(cur)
        cur = cur + step_scale * (y - cur)
    return cur
"""


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int = 8, scale: float = 0.25) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim) * (1.0 + scale))
        self.calls = 0

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        self.calls += 1
        kv = kwargs.get("kv_cache")
        if isinstance(kv, dict) and "local_end_index" in kv:
            end = int(kv["local_end_index"].item())
            start = max(0, end - x.shape[1])
            kv["k"][:, start:end] += 1.0
            kv["v"][:, start:end] += 2.0
            kv["local_end_index"].add_(1)
            kv["global_end_index"].add_(1)
        elif isinstance(kv, dict):
            kv["writes"] = kv.get("writes", 0) + 1
        return self.proj(x)


class _ScaleRAEFlow:
    def __init__(self) -> None:
        self.used_timesteps = list(range(26))

    def get_sigmas(self, u: Tensor) -> Tensor:
        return u

    def euler_forward(self, model_fn, x_t: Tensor, t_curr: Tensor, t_next: Tensor, denoised_fn, model_kwargs):
        pred = model_fn(x_t, t_curr, **(model_kwargs or {}))
        if denoised_fn is not None:
            pred = denoised_fn(pred)
        return (x_t + (t_next - t_curr).view(-1, 1, 1) * pred,)

    step_fn = euler_forward

    def euler_sample(self, model_fn, x_t: Tensor, u_t: Tensor, u_s: Tensor, clip_denoised, denoised_fn, cond_fn, model_kwargs) -> Tensor:
        del clip_denoised, cond_fn
        sigma_t = self.get_sigmas(u_t)
        sigma_s = self.get_sigmas(u_s)
        pred = model_fn(x_t, sigma_t, **(model_kwargs or {}))
        if denoised_fn is not None:
            pred = denoised_fn(pred)
        return x_t + (sigma_s - sigma_t).view(-1, 1, 1) * pred

    def p_sample_loop(self, model_fn, x: Tensor, **model_kwargs) -> Tensor:
        cur = x
        steps = self.used_timesteps
        n = max(1, len(steps) - 1)
        for i in range(n):
            t0 = torch.tensor([float(i) / n], device=x.device)
            t1 = torch.tensor([float(i + 1) / n], device=x.device)
            if i == n - 1:
                cur = self.euler_sample(model_fn, cur, t0, t1, False, None, None, model_kwargs)
            else:
                (cur,) = self.step_fn(model_fn, cur, t0, t1, None, model_kwargs)
        return cur


class _MockScaleRAE(nn.Module):
    def __init__(self, depth: int = 16, dim: int = 8) -> None:
        super().__init__()
        self.diff_head = nn.Module()
        inner = nn.Module()
        inner.layers = nn.ModuleList([_ResidualBlock(dim, scale=0.1) for _ in range(depth)])
        self.diff_head.model = inner
        self.diff_head.inference_flow = _ScaleRAEFlow()

    def forward(self, x: Tensor, t: Tensor | None = None, **_kwargs) -> Tensor:
        del t
        cur = x
        for layer in self.diff_head.model.layers:
            cur = layer(cur)
        return cur

    def sample(self, x: Tensor) -> Tensor:
        return self.diff_head.inference_flow.p_sample_loop(self, x)


class _MockNativeLoopedMMDiT(nn.Module):
    """Exact tensor-shape & attribute match for OpenSenseNova `LoopedMMDiT` ([B, 3, H, W] -> [B, 3, H, W])."""

    def __init__(self, image_size: int = 4, patch_size: int = 2, in_ch: int = 3, tok_dim: int = 12, pre: int = 1, core: int = 2, post: int = 1) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_ch
        self.grid = image_size // patch_size
        self.pre = pre
        self.core = core
        self.post = post
        self.num_loops = 4
        self.share_loop_weights = True
        patch_dim = in_ch * patch_size * patch_size
        self.img_embed = nn.Conv2d(in_ch, tok_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.txt_embed = nn.Linear(8, tok_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 8))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid * self.grid, tok_dim))
        self.out_proj = nn.Linear(tok_dim, patch_dim, bias=False)
        with torch.no_grad():
            self.img_embed.weight.fill_(0.05)
            self.txt_embed.weight.fill_(0.05)
            self.out_proj.weight.fill_(0.05)

        class _ImgFlat(nn.Module):
            def __init__(self, conv: nn.Conv2d) -> None:
                super().__init__()
                self.conv = conv

            def forward(self, x: Tensor) -> Tensor:
                # [B, C, H, W] -> [B, N, D]
                return self.conv(x).flatten(2).transpose(1, 2)

        self._raw_conv = self.img_embed
        self.img_embed = _ImgFlat(self._raw_conv)

        class _TxtBlk(nn.Module):
            def __init__(self, d: int) -> None:
                super().__init__()
                self.lin = nn.Linear(d, d, bias=False)
                with torch.no_grad():
                    self.lin.weight.copy_(torch.eye(d) * 0.98)

            def forward(self, txt: Tensor) -> Tensor:
                return self.lin(txt)

        class _JointBlk(nn.Module):
            def __init__(self, d: int) -> None:
                super().__init__()
                self.lin = nn.Linear(d, d, bias=False)
                with torch.no_grad():
                    self.lin.weight.copy_(torch.eye(d) * 0.95)

            def forward(self, img: Tensor, txt: Tensor) -> tuple[Tensor, Tensor]:
                return self.lin(img), txt

        self.txt_blocks = nn.ModuleList([_TxtBlk(tok_dim)])
        self.blocks = nn.ModuleList([_JointBlk(tok_dim) for _ in range(pre + core + post)])

    def loop_blocks(self, r: int = 1):
        del r
        return self.blocks[self.pre : self.pre + self.core]

    def decode(self, img: Tensor, txt: Tensor) -> Tensor:
        cur_img, cur_txt = img, txt
        for blk in self.blocks[self.pre + self.core :]:
            cur_img, cur_txt = blk(cur_img, cur_txt)
        patches = self.out_proj(cur_img)
        b, n, _ = patches.shape
        p, c, g = self.patch_size, self.in_channels, int(n**0.5)
        return patches.view(b, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4).contiguous().view(b, c, g * p, g * p)

    def forward(self, x: Tensor, text: Tensor, text_mask: Tensor, num_loops: int | None = None, **_kwargs) -> Tensor:
        masked = torch.where(text_mask.to(torch.bool)[:, :, None], text, self.mask_token.to(x.dtype))
        txt = self.txt_embed(masked)
        for tb in self.txt_blocks:
            txt = tb(txt)
        img = self.img_embed(x) + self.pos_embed
        for blk in self.blocks[: self.pre]:
            img, txt = blk(img, txt)
        loops = self.num_loops if num_loops is None else num_loops
        for r in range(1, loops + 1):
            for blk in self.loop_blocks(r):
                img, txt = blk(img, txt)
        return self.decode(img, txt)


class _EmptyFalsyCache:
    """Mimics HF `DynamicCache()` where empty `bool(cache)` is False at pass 0."""

    def __init__(self) -> None:
        self.key_cache: list[Tensor] = []
        self.value_cache: list[Tensor] = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self.key_cache):
            return int(self.key_cache[layer_idx].shape[-2])
        return 0

    def __len__(self) -> int:
        return len(self.key_cache)


def copy_kv(kv: dict) -> dict:
    return {k: (v.clone() if isinstance(v, Tensor) else v) for k, v in kv.items()}


class TestRegistryWithoutPresets(unittest.TestCase):
    def test_registry_covers_all_9_models_and_no_presets_exist(self) -> None:
        items = list_backbones()
        names = {item["name"] for item in items}
        self.assertEqual(
            names,
            {
                "scale_rae", "rae", "looped_dit",
                "qwen3", "dory",
                "longlive2", "longlive1", "self_forcing", "wan21",
            },
        )
        for item in items:
            self.assertEqual(set(item.keys()), {"name", "modality", "loading", "upstream", "description"})
            self.assertNotIn("default_program", item)
            self.assertTrue(item["loading"])
            self.assertTrue(item["upstream"])

        self.assertEqual(get_adapter("scale-rae").name, "scale_rae")
        self.assertEqual(get_adapter("hf_decoder").name, "qwen3")
        self.assertEqual(get_adapter("wan2.1").name, "wan21")
        with self.assertRaises(KeyError):
            get_adapter("nonexistent_model")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("elp.programs")


class TestImageBackendsContract(unittest.TestCase):
    def test_scale_rae_loop_program_step_override_guidance_and_restore(self) -> None:
        model = _MockScaleRAE(depth=16, dim=8).eval()
        orig_weights = {k: v.clone() for k, v in model.state_dict().items()}
        x = torch.randn(1, 4, 8)
        adapter = ScaleRAEAdapter()
        base = adapter.infer(model, {"x": x}, program=None)

        prog = LoopProgram(
            id="sr_test_loop",
            code=_STANDARD_EULER_PROGRAM,
            params={"k": 2, "step_scale": 0.5, "loopguidance_weight": 3.0, "refresh_interval": 2},
            metadata={"blocks": [6, 7, 8, 9], "steps": 6},
        )
        with adapter.install_program(model, prog) as patch:
            self.assertEqual(len(model.diff_head.inference_flow.used_timesteps), 7)
            sampled = model.sample(x)
            self.assertEqual(sampled.shape, x.shape)
            self.assertGreater(patch.stats.loop_invocations, 0)
            self.assertEqual(patch.stats.block_calls % 2, 0)
            self.assertGreater(patch.stats.guidance_ref_calls, 0)
            self.assertGreater(patch.stats.guidance_loop_calls, 0)

        self.assertEqual(len(model.diff_head.inference_flow.used_timesteps), 26)
        restored = adapter.infer(model, {"x": x}, program=None, probe=ProbeSpec(layers=(6, 9)))
        self.assertTrue(torch.allclose(base.output, restored.output))
        self.assertEqual(len(restored.diagnostics), 2)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(orig_weights[k], v))

    def test_rae_dit_ddt_head_config_and_loop_program_restore(self) -> None:
        class _MiniRAE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([_ResidualBlock(8, 0.1) for _ in range(10)])

            def forward(self, x: Tensor) -> Tensor:
                cur = x
                for blk in self.blocks:
                    cur = blk(cur)
                return cur

        model = _MiniRAE().eval()
        orig_weights = {k: v.clone() for k, v in model.state_dict().items()}
        adapter = RAEAdapter()
        x = torch.randn(1, 4, 8)
        ref = adapter.infer(model, x, program=None)

        def sampler(wrapped_fn, z: Tensor) -> Tensor:
            for _ in range(4):
                z = z + 0.1 * wrapped_fn(z)
            return z

        cfg = ImageLoopConfig(blocks=(4, 5), num_loops=2, step_scale=0.5, end_frac=0.6)
        guidance = CachedLoopGuidance(weight=2.0, refresh_interval=2)
        res_cfg = adapter.infer(
            model,
            {"sampler_fn": sampler, "model_fn": model, "z": x},
            program=(cfg, guidance),
            total_steps=4,
        )
        self.assertGreater(res_cfg.loop_stats["guidance_ref_calls"], 0)

        prog = LoopProgram(
            id="rae_loop_prog",
            code=_STANDARD_EULER_PROGRAM,
            params={"k": 2, "step_scale": 0.5, "end_frac": 0.5},
            metadata={"blocks": [3, 4]},
        )
        res_prog = adapter.infer(model, x, program=prog, total_steps=4)
        self.assertEqual(res_prog.loop_stats["loop_invocations"], 2)
        self.assertEqual(res_prog.loop_stats["block_calls"], 4)
        self.assertFalse(torch.allclose(ref.output, res_prog.output))

        restored = adapter.infer(model, x, program=None)
        self.assertTrue(torch.allclose(ref.output, restored.output))
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(orig_weights[k], v))

    def test_looped_dit_native_signature_prompt_schedule_loop_program_and_restore(self) -> None:
        model = _MockNativeLoopedMMDiT(image_size=4, patch_size=2, in_ch=3, tok_dim=12).eval()
        orig_weights = {k: v.clone() for k, v in model.state_dict().items()}
        model._elp_model_cfg = type("Cfg", (), {"image_size": 4, "patch_size": 2, "in_channels": 3})()

        class _DummyTextEncoder:
            def tokenize(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
                return torch.ones(len(prompts), 2, dtype=torch.long), torch.ones(len(prompts), 2)

            def encode(self, input_ids: Tensor, mask: Tensor) -> Tensor:
                del mask
                return torch.ones(input_ids.shape[0], input_ids.shape[1], 8)

        model._elp_text_encoder = _DummyTextEncoder()
        adapter = LoopedDiTAdapter()

        inputs = {
            "x_init": torch.randn(1, 3, 4, 4),
            "text": torch.randn(1, 2, 8),
            "text_mask": torch.ones(1, 2),
        }
        base = adapter.infer(model, inputs, program=None, step_count=3, num_loops=2, cfg_scale=2.0)
        self.assertEqual(base.step_count, 3)

        prompt_res = adapter.infer(model, {"prompt": "a glass prism", "seed": 7}, program=None, step_count=2, cfg_scale=1.0)
        self.assertEqual(prompt_res.output.shape, (1, 3, 4, 4))

        sched_res = adapter.infer(
            model,
            inputs,
            program=LoopedDiTScheduleConfig(step_count=4, refresh_steps=(0, 2)),
        )
        self.assertEqual(sched_res.step_count, 4)
        self.assertGreater(sched_res.loop_stats["block_calls"], 8)

        prog = LoopProgram(
            id="ldit_prog",
            code=_STANDARD_EULER_PROGRAM,
            params={"k": 2, "step_scale": 0.5},
            metadata={"steps": 3},
        )
        prog_res = adapter.infer(model, inputs, program=prog, cfg_scale=1.0)
        self.assertEqual(prog_res.step_count, 3)
        self.assertEqual(prog_res.loop_stats["loop_invocations"], 3)
        self.assertEqual(prog_res.loop_stats["block_calls"], 6)

        restored = adapter.infer(model, inputs, program=None, step_count=3, num_loops=2, cfg_scale=2.0)
        self.assertTrue(torch.allclose(base.output, restored.output))
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(orig_weights[k], v))


class TestTextBackendsContract(unittest.TestCase):
    def test_hf_decoder_empty_cache_mask_slice_loop_program_and_restore(self) -> None:
        observed_mask_widths: list[int] = []

        class _MoELayer(nn.Module):
            def __init__(self, dim: int = 8) -> None:
                super().__init__()
                self.lin = nn.Linear(dim, dim, bias=False)
                self.mlp = nn.Module()
                gate = nn.Module()
                gate.calls = 0

                def gate_forward(h: Tensor) -> Tensor:
                    gate.calls += 1
                    return torch.ones(h.shape[0], h.shape[1], 1)

                gate.forward = gate_forward  # type: ignore[method-assign]
                self.mlp.gate = gate

            def forward(
                self,
                hidden_states: Tensor,
                *,
                past_key_values=None,
                attention_mask: Tensor | None = None,
                use_cache: bool = False,
            ) -> Tensor:
                if attention_mask is not None:
                    observed_mask_widths.append(int(attention_mask.shape[-1]))
                weight = self.mlp.gate(hidden_states)
                out = hidden_states + 0.1 * self.lin(hidden_states) * weight
                if use_cache and isinstance(past_key_values, dict):
                    past_key_values["commits"] = past_key_values.get("commits", 0) + 1
                elif use_cache and isinstance(past_key_values, _EmptyFalsyCache):
                    k = torch.zeros(hidden_states.shape[0], 1, hidden_states.shape[1], hidden_states.shape[2])
                    past_key_values.key_cache.append(k)
                    past_key_values.value_cache.append(k)
                return out

        class _MockDecoder(nn.Module):
            def __init__(self, depth: int = 8) -> None:
                super().__init__()
                self.layers = nn.ModuleList([_MoELayer(8) for _ in range(depth)])

            def forward(
                self,
                hidden_states: Tensor,
                *,
                past_key_values=None,
                attention_mask: Tensor | None = None,
                use_cache: bool = False,
            ) -> Tensor:
                cur = hidden_states
                for layer in self.layers:
                    cur = layer(
                        cur,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        use_cache=use_cache,
                    )
                return cur

        model = _MockDecoder(depth=8).eval()
        x = torch.randn(1, 4, 8)
        adapter = HFDecoderAdapter()
        base = adapter.infer(model, {"hidden_states": x}, program=None, max_new_tokens=0)

        # Empty falsy cache + wider attention_mask gets sliced on uncached probe and committed once at end
        falsy_cache = _EmptyFalsyCache()
        wide_mask = torch.ones(1, 10)
        prog = LoopProgram(
            id="qwen_loop_prog",
            code=_STANDARD_EULER_PROGRAM,
            params={"k": 2, "step_scale": 0.5},
            metadata={"start_layer": 2, "end_layer": 3},
        )
        with HFDecoderLoopPatch(model, prog) as patch:
            _ = model(x, past_key_values=falsy_cache, attention_mask=wide_mask, use_cache=True)
            self.assertEqual(patch.stats.kv_commits, 1)
            self.assertGreater(len(falsy_cache.key_cache), 0)
            self.assertIn(x.shape[1], observed_mask_widths)

        # Dict prefix rollback + pinned MoE routing
        kv = {0: torch.tensor([1.0])}
        with preserve_dynamic_cache(kv, [0]):
            kv[0] = torch.tensor([99.0])
        self.assertEqual(float(kv[0].item()), 1.0)

        cache_dict = {2: torch.tensor([0.0]), "commits": 0}
        win = (
            TextLoopConfig(start_layer=1, end_layer=2, num_loops=2, strategy="heun"),
            TextLoopConfig(start_layer=4, end_layer=5, num_loops=2, strategy="euler"),
        )
        before_gate_calls = model.layers[1].mlp.gate.calls
        with HFDecoderLoopPatch(model, win, pin_routing=True) as patch:
            _ = model(x, past_key_values=cache_dict, use_cache=True)
            self.assertEqual(model.layers[1].mlp.gate.calls - before_gate_calls, 1)
            self.assertGreater(patch.stats.kv_rollbacks, 0)
            self.assertEqual(patch.stats.kv_commits, 2)
            self.assertEqual(cache_dict["commits"], 8)

        restored = adapter.infer(model, {"hidden_states": x}, program=None, max_new_tokens=0)
        self.assertTrue(torch.allclose(base.output, restored.output))

        with pinned_moe_routing([model.layers[0]], enabled=True):
            pass

    def test_dory_real_model_forward_remap_loop_program_and_restore(self) -> None:
        raw_sd = {
            "backbone.embeddings.weight": torch.zeros(4, 4),
            "backbone.layers.0.attn_alpha": torch.ones(4),
            "backbone.layers.0.mixer.g_proj.weight": torch.zeros(4, 4),
            "backbone.layers.0.mixer.q_proj.weight": torch.zeros(4, 4),
            "backbone.layers.1.mlp_alpha": torch.ones(4),
            "backbone.layers.1.mixer.up_proj.weight": torch.zeros(4, 4),
            "lm_head.weight": torch.zeros(4, 4),
        }
        mapped = remap_dory_state_dict(raw_sd)
        self.assertIn("model.embed_tokens.weight", mapped)
        self.assertIn("model.layers.0.lambda_1", mapped)
        self.assertIn("model.layers.0.self_attn.gate_proj.weight", mapped)
        self.assertIn("model.layers.0.lambda_2", mapped)
        self.assertIn("model.layers.0.mlp.up_proj.weight", mapped)

        cfg = DoryConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            n_input_layers=1,
            n_recurrent_layers=2,
            n_output_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            n_recurrent_loops=2,
        )
        model = DoryForCausalLM(cfg).eval()
        orig_weights = {k: v.clone() for k, v in model.state_dict().items()}

        class _DummyTok:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text: str, return_tensors: str = "pt"):
                del text, return_tensors
                return {"input_ids": torch.tensor([[3, 7, 11]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

        model._elp_tokenizer = _DummyTok()
        adapter = DoryAdapter()
        base = adapter.infer(model, {"prompt": "hello"}, program=None, max_new_tokens=0)
        self.assertEqual(base.output.logits.shape, (1, 3, 64))

        # Geodesic config on genuine DoryForCausalLM
        loop_res = adapter.infer(
            model,
            {"input_ids": torch.tensor([[3, 7, 11]])},
            program=DoryLoopConfig(num_loops=3, recurrent_kv_cache_mode="last_loop", geodesic_scale=0.18),
            max_new_tokens=0,
        )
        self.assertEqual(loop_res.loop_stats["block_calls"], 6)
        self.assertFalse(torch.allclose(base.output.logits, loop_res.output.logits))

        # LoopProgram on genuine DoryForCausalLM with DoryRecurrentCache rollback + commit
        cache = DoryRecurrentCache(num_layers=cfg.num_hidden_layers, sliding_window=16)
        dory_prog = LoopProgram(id="dory_prog", code=_STANDARD_EULER_PROGRAM, params={"k": 2, "step_scale": 0.5})
        with DoryLoopPatch(model, program=dory_prog) as patch:
            prog_out = model(input_ids=torch.tensor([[3, 7, 11]]), past_key_values=cache, use_cache=True)
            self.assertGreater(patch.stats.kv_rollbacks, 0)
            self.assertEqual(patch.stats.kv_commits, 1)
            self.assertEqual(prog_out.logits.shape, (1, 3, 64))

        self.assertEqual(model.config.n_recurrent_loops, 2)
        restored = adapter.infer(model, {"prompt": "hello"}, program=None, max_new_tokens=0)
        self.assertTrue(torch.allclose(base.output.logits, restored.output.logits))
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(orig_weights[k], v))


class TestVideoBackendsContract(unittest.TestCase):
    def test_causal_world_models_multilayer_kv_longlive_tuples_and_auto_phase(self) -> None:
        class _CausalBlock(nn.Module):
            def __init__(self, idx: int, dim: int = 8) -> None:
                super().__init__()
                self.idx = idx
                self.proj = nn.Linear(dim, dim, bias=False)
                with torch.no_grad():
                    self.proj.weight.copy_(torch.eye(dim) * (1.05 + 0.01 * idx))
                self.calls = 0

            def forward(self, x: Tensor, **kwargs):
                self.calls += 1
                kv = kwargs.get("kv_cache")
                if isinstance(kv, dict) and "local_end_index" in kv:
                    if kv.get("owner_layer") != self.idx:
                        raise AssertionError(f"Layer {self.idx} received wrong kv_cache owner {kv.get('owner_layer')}")
                    end = int(kv["local_end_index"].item())
                    kv_len = x.shape[1]
                    start = max(0, end)
                    stop = min(kv["k"].shape[1], start + kv_len)
                    kv["k"][:, start:stop] += 1.0 + self.idx
                    kv["v"][:, start:stop] += 2.0 + self.idx
                    kv["local_end_index"].fill_(stop)
                    kv["global_end_index"].add_(kv_len)
                    kv["layer_writes"] = kv.get("layer_writes", 0) + 1
                out = self.proj(x)
                return out, (end + kv_len if isinstance(kv, dict) else 0, self.idx, self.calls)

        class _CausalWanMock(nn.Module):
            def __init__(self, depth: int = 6) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([_CausalBlock(i, dim=8) for i in range(depth)])

            def forward(self, x: Tensor, t: Tensor | None = None, *, kv_cache=None, current_start: int = 0) -> Tensor:
                del t, current_start
                cur = x
                updates = []
                for idx, blk in enumerate(self.blocks):
                    layer_kv = kv_cache[idx] if isinstance(kv_cache, list) else kv_cache
                    res = blk(cur, kv_cache=layer_kv)
                    cur, upd = res
                    updates.append(upd)
                return cur

        def make_kv_list(depth: int = 6) -> list[dict]:
            out = []
            for j in range(depth):
                k_buf = torch.zeros(1, 6, 8)
                v_buf = torch.zeros(1, 6, 8)
                # Pre-populate prefix slots 0..2 to assert prefix slice immutability across rollbacks
                k_buf[:, :2] = 7.0 + j
                v_buf[:, :2] = 9.0 + j
                out.append({
                    "owner_layer": j,
                    "k": k_buf,
                    "v": v_buf,
                    "local_end_index": torch.tensor(2),
                    "global_end_index": torch.tensor(2),
                    "layer_writes": 0,
                })
            return out

        model = _CausalWanMock(depth=6).eval()
        orig_weights = {k: v.clone() for k, v in model.state_dict().items()}
        x = torch.randn(1, 2, 8)

        for name in ("longlive2", "longlive1", "self_forcing"):
            adapter = CausalWorldModelAdapter(name)
            base_kv = make_kv_list(6)
            base_out = model(x, t=torch.tensor([1000.0]), kv_cache=base_kv, current_start=0)

            elp_kv = make_kv_list(6)
            cfg = VideoWorldModelLoopConfig(
                kind="windowwise", start_layer=1, end_layer=3, k=3, active_steps=(0,), active_chunks=(0,)
            )
            with adapter.install_program(model, cfg) as driver:
                driver.begin_video()
                denoise_out = model(x, t=torch.tensor([1000.0]), kv_cache=elp_kv, current_start=0)
                self.assertEqual(driver.stats.kv_rollbacks, 2)
                self.assertEqual(driver.stats.kv_commits, 1)
                self.assertFalse(torch.allclose(base_out, denoise_out))
                # Verify prefix slots untouched and layer 2 received its own kv_cache[2] cleanly
                self.assertTrue(torch.allclose(elp_kv[2]["k"][:, :2], torch.full((1, 2, 8), 9.0)))
                self.assertEqual(elp_kv[2]["k"][0, 2, 0].item(), 3.0)
                # Context noise step (t=0) must bypass looping via auto phase tracking
                before_calls = model.blocks[1].calls
                _ = model(x, t=torch.tensor([0.0]), kv_cache=make_kv_list(6), current_start=0)
                self.assertEqual(model.blocks[1].calls - before_calls, 1)

            # Standard LoopProgram AND Self-Forcing residual-budget LoopProgram
            std_prog = LoopProgram(
                id=f"{name}_std_prog",
                code=_STANDARD_EULER_PROGRAM,
                params={"k": 2, "step_scale": 0.5},
                metadata={"blocks": [2], "active_steps": [0], "chunks": [0]},
            )
            res_std = adapter.infer(
                model,
                {"x": x, "steps": [0], "chunks": [0], "kv_cache": make_kv_list(6)},
                program=std_prog,
            )
            self.assertEqual(res_std.loop_stats["kv_commits"], 1)

            rb_prog = LoopProgram(
                id=f"{name}_rb_prog",
                code="import torch\ndef loop(x, evaluate, info, params, memory):\n    r0 = evaluate(x)\n    r1 = evaluate(x + 0.5 * r0)\n    return {'residuals': [r0, r1], 'weights': [0.5, 0.5]}\n",
                params={},
                metadata={"blocks": [3], "active_steps": [0], "chunks": [0]},
            )
            res_rb = adapter.infer(
                model,
                {"x": x, "steps": [0], "chunks": [0], "kv_cache": make_kv_list(6)},
                program=rb_prog,
            )
            self.assertGreater(res_rb.loop_stats["kv_rollbacks"], 0)
            self.assertEqual(res_rb.loop_stats["kv_commits"], 1)

            restored_out = model(x, t=torch.tensor([1000.0]), kv_cache=make_kv_list(6), current_start=0)
            self.assertTrue(torch.allclose(base_out, restored_out))

        bad_prog = LoopProgram(
            id="wm_bad_sum",
            code="import torch\ndef loop(x, evaluate, info, params, memory):\n    r0 = evaluate(x)\n    return {'residuals': [r0], 'weights': [0.5]}\n",
            metadata={"blocks": [3], "active_steps": [0], "chunks": [0]},
        )
        with self.assertRaises(ValueError):
            CausalWorldModelAdapter("self_forcing").infer(
                model, {"x": x, "steps": [0], "chunks": [0]}, program=bad_prog
            )
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(orig_weights[k], v))

    def test_wan21_loop_config_and_loop_program_restore(self) -> None:
        class _WanStack(nn.Module):
            def __init__(self, depth: int = 8) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([_ResidualBlock(8, scale=0.1) for _ in range(depth)])

            def forward(self, x: Tensor, **kwargs) -> Tensor:
                cur = x
                for blk in self.blocks:
                    cur = blk(cur, **kwargs)
                return cur

        model = _WanStack(depth=8).eval()
        orig_weights = {k: v.clone() for k, v in model.state_dict().items()}
        x = torch.randn(1, 2, 8)
        adapter = Wan21Adapter()
        base = adapter.infer(model, {"x": x, "steps": 4}, program=None, step_count=4)

        cfg = VideoDiffusionLoopConfig(blocks=(2, 3), repeats=3, strength=1.0, start_frac=0.0, end_frac=0.5)
        elp = adapter.infer(model, {"x": x, "steps": 4}, program=cfg, step_count=4)
        self.assertEqual(elp.loop_stats["loop_invocations"], 4)
        self.assertGreater(elp.loop_stats["block_calls"], base.step_count * 2)

        wan_prog = LoopProgram(
            id="wan_prog_euler",
            code=_STANDARD_EULER_PROGRAM,
            params={"k": 2, "step_scale": 0.5, "start_frac": 0.0, "end_frac": 0.5},
            metadata={"blocks": [2, 3]},
        )
        prog_res = adapter.infer(model, {"x": x, "steps": 4}, program=wan_prog, step_count=4)
        self.assertEqual(prog_res.loop_stats["loop_invocations"], 4)
        self.assertFalse(torch.allclose(base.output, prog_res.output))

        restored = adapter.infer(model, {"x": x, "steps": 4}, program=None, step_count=4)
        self.assertTrue(torch.allclose(base.output, restored.output))
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(orig_weights[k], v))


class TestBundledThirdPartyRepos(unittest.TestCase):
    def test_bundled_third_party_repos_parse_resolve_and_isolate(self) -> None:
        self.assertTrue(BUNDLED_THIRD_PARTY_DIR.is_dir())
        expected = [
            ("scale_rae", "SCALE_RAE_REPO", "Scale-RAE"),
            ("rae", "DIFFUSION_BENCH_REPO", "diffusion-bench (RAE)"),
            ("looped_dit", "LOOPED_DIT_REPO", "OpenSenseNova/Looped-DiT"),
            ("self_forcing", "SELF_FORCING_REPO", "guandeh17/Self-Forcing"),
            ("longlive1", "LONGLIVE1_REPO", "NVlabs/LongLive (v1.0)"),
            ("longlive2", "LONGLIVE2_REPO", "NVlabs/LongLive (main)"),
            ("wan21", "WAN21_REPO", "Wan2.1"),
        ]
        cleaned_env = {
            k: v
            for k, v in os.environ.items()
            if k not in {env_var for _, env_var, _ in expected}
        }
        with patch.dict(os.environ, cleaned_env, clear=True):
            for sub, env_var, label in expected:
                resolved = resolve_upstream_repo(None, env_var, label, bundled_subdir=sub)
                self.assertEqual(resolved, (BUNDLED_THIRD_PARTY_DIR / sub).resolve())
                self.assertTrue(resolved.is_dir())

        py_files = list(BUNDLED_THIRD_PARTY_DIR.rglob("*.py"))
        self.assertGreater(len(py_files), 300)
        for py_path in py_files:
            ast.parse(py_path.read_text(encoding="utf-8", errors="ignore"), filename=str(py_path))

        # Bundled config sanity across RAE decoder & world-model YAMLs
        vi_cfg = json.loads(
            (BUNDLED_THIRD_PARTY_DIR / "rae/configs/decoder/ViTXL/config.json").read_text(encoding="utf-8")
        )
        self.assertIn("architectures", vi_cfg)
        import yaml

        for cfg_rel in (
            "longlive2/configs/inference.yaml",
            "longlive1/configs/longlive_inference.yaml",
            "self_forcing/configs/self_forcing_dmd.yaml",
        ):
            parsed = yaml.safe_load((BUNDLED_THIRD_PARTY_DIR / cfg_rel).read_text(encoding="utf-8"))
            self.assertIsInstance(parsed, dict)

        # Switching active repo must prune stale third_party sys.path and sys.modules entries cleanly
        orig_sys_path = list(sys.path)
        try:
            looped_repo = resolve_upstream_repo(None, "LOOPED_DIT_REPO", "Looped-DiT", bundled_subdir="looped_dit")
            activate_upstream_repo(looped_repo)
            self.assertEqual(Path(sys.path[0]).resolve(), looped_repo)
            mod = importlib.import_module("looped_dit.model")
            self.assertTrue(Path(mod.__file__).resolve().is_relative_to(looped_repo))

            sf_repo = resolve_upstream_repo(None, "SELF_FORCING_REPO", "Self-Forcing", bundled_subdir="self_forcing")
            activate_upstream_repo(sf_repo)
            self.assertNotIn("looped_dit.model", sys.modules)
            self.assertNotIn("looped_dit", sys.modules)
            self.assertFalse(any(Path(p).resolve() == looped_repo for p in sys.path if p))
        finally:
            sys.path[:] = orig_sys_path
            for m in ("looped_dit", "looped_dit.model"):
                sys.modules.pop(m, None)

    def test_real_vendored_looped_mmdit_end_to_end(self) -> None:
        orig_sys_path = list(sys.path)
        try:
            looped_repo = resolve_upstream_repo(None, "LOOPED_DIT_REPO", "Looped-DiT", bundled_subdir="looped_dit")
            activate_upstream_repo(looped_repo)
            from looped_dit.model import LoopedMMDiT

            model = LoopedMMDiT(
                image_size=8,
                patch_size=4,
                in_channels=3,
                hidden_size=16,
                num_heads=2,
                head_dim=8,
                pca_channels=8,
                text_dim=16,
                text_preamble_depth=1,
                loop_split=(1, 1, 1),
                num_loops=2,
            ).eval()
            orig_weights = {k: v.clone() for k, v in model.state_dict().items()}
            inputs = {
                "x_init": torch.randn(1, 3, 8, 8),
                "text": torch.randn(1, 4, 16),
                "text_mask": torch.ones(1, 4),
            }
            adapter = LoopedDiTAdapter()
            base = adapter.infer(model, inputs, program=None, step_count=2, num_loops=2, cfg_scale=2.0)
            self.assertEqual(base.output.shape, (1, 3, 8, 8))

            sched = adapter.infer(
                model,
                inputs,
                program=LoopedDiTScheduleConfig(step_count=3, refresh_steps=(0, 1)),
            )
            self.assertEqual(sched.output.shape, (1, 3, 8, 8))
            self.assertEqual(sched.step_count, 3)

            prog = LoopProgram(
                id="real_vendored_ldit_euler",
                code=_STANDARD_EULER_PROGRAM,
                params={"k": 2, "step_scale": 0.5},
                metadata={"steps": 2},
            )
            prog_res = adapter.infer(model, inputs, program=prog, cfg_scale=1.0)
            self.assertEqual(prog_res.output.shape, (1, 3, 8, 8))
            self.assertEqual(prog_res.loop_stats["loop_invocations"], 2)
            self.assertEqual(prog_res.loop_stats["block_calls"], 4)

            restored = adapter.infer(model, inputs, program=None, step_count=2, num_loops=2, cfg_scale=2.0)
            self.assertTrue(torch.allclose(base.output, restored.output))
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(orig_weights[k], v))
        finally:
            sys.path[:] = orig_sys_path
            for m in ("looped_dit", "looped_dit.model"):
                sys.modules.pop(m, None)


if __name__ == "__main__":
    unittest.main()
