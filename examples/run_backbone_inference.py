"""End-to-end CPU demonstration of ELP adapters across Image, Text, and Video backbones."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elp.backends import (
    CausalWorldModelAdapter,
    DoryAdapter,
    HFDecoderAdapter,
    LoopedDiTAdapter,
    RAEAdapter,
    ScaleRAEAdapter,
    Wan21Adapter,
)
from elp.backends.text import DoryConfig, DoryForCausalLM, DoryLoopConfig
from elp.core import LoopProgram, ProbeSpec
from elp.modalities.image import ImageLoopConfig, LoopedDiTScheduleConfig
from elp.modalities.text import TextLoopConfig
from elp.modalities.video import VideoDiffusionLoopConfig, VideoWorldModelLoopConfig


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.net.weight.copy_(torch.eye(dim) * 1.15)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        del kwargs
        return self.net(x)


class _SimpleStack(nn.Module):
    def __init__(self, depth: int, dim: int, attr: str = "layers") -> None:
        super().__init__()
        setattr(self, attr, nn.ModuleList([_ResidualBlock(dim) for _ in range(depth)]))
        self._attr = attr

    def forward(self, x: Tensor | None = None, *, hidden_states: Tensor | None = None, **kwargs) -> Tensor:
        cur = hidden_states if x is None else x
        for blk in getattr(self, self._attr):
            cur = blk(cur, **kwargs)
        return cur


_EULER_LOOP_CODE = """import torch

def loop(x, evaluate, params):
    step_scale = float(params.get("step_scale", 0.5))
    k = int(params.get("k", 2))
    cur = x
    for _ in range(k):
        y = evaluate(cur)
        cur = cur + step_scale * (y - cur)
    return cur
"""

_CUSTOM_PROBE_CODE = """import torch

def collect(layer_idx, pass_idx, x, y, info):
    del info, pass_idx
    diff = y - x
    rel = torch.sqrt(torch.mean(diff * diff)) / torch.sqrt(torch.mean(x * x)).clamp(min=1e-12)
    return {"rel_delta": float(rel), "layer": float(layer_idx)}
"""


def demo_multimodal_adapters() -> None:
    torch.manual_seed(0)

    # 1. Image: Scale-RAE block loop + restoration verification + programmable probe
    img_model = _SimpleStack(depth=16, dim=8, attr="layers").eval()
    img_adapter = ScaleRAEAdapter()
    x_img = torch.randn(1, 4, 8)
    img_base = img_adapter.infer(img_model, x_img, program=None)
    img_elp = img_adapter.infer(
        img_model,
        x_img,
        program=ImageLoopConfig(blocks=(6, 7, 8, 9), num_loops=2, step_scale=0.5),
        probe=ProbeSpec(layers=(6, 9), features=("activation", "residual"), collector_code=_CUSTOM_PROBE_CODE),
    )
    img_restored = img_adapter.infer(img_model, x_img, program=None)
    assert torch.allclose(img_base.output, img_restored.output)
    assert "rel_delta" in img_elp.diagnostics[0]["metrics"]
    print(f"[Image / Scale-RAE] block_calls={img_elp.loop_stats['block_calls']} probe_rows={len(img_elp.diagnostics)}")

    # 2. Image: RAE shifted-Euler LoopProgram + Looped-DiT adaptive schedule
    rae_model = _SimpleStack(depth=10, dim=8, attr="blocks").eval()
    rae_prog = LoopProgram(
        id="rae_demo_euler",
        code=_EULER_LOOP_CODE,
        params={"k": 2, "step_scale": 0.5},
        metadata={"blocks": [3, 4]},
    )
    rae_res = RAEAdapter().infer(rae_model, x_img, program=rae_prog)
    print(f"[Image / RAE] loop_invocations={rae_res.loop_stats['loop_invocations']} block_calls={rae_res.loop_stats['block_calls']}")

    class _MiniLoopedDiT(nn.Module):
        def __init__(self, dim: int = 8) -> None:
            super().__init__()
            self.proj = nn.Linear(dim, dim, bias=False)

        def pre_blocks(self, x: Tensor, txt: Tensor, _mask: Tensor) -> tuple[Tensor, Tensor]:
            return x, txt

        def core_step(self, img: Tensor, txt: Tensor) -> tuple[Tensor, Tensor]:
            return img - 0.1 * self.proj(img), txt

        def post_blocks(self, img: Tensor, _txt: Tensor) -> Tensor:
            return img

    dit_adapter = LoopedDiTAdapter()
    dit_res = dit_adapter.infer(
        _MiniLoopedDiT().eval(),
        {
            "x_init": torch.randn(1, 4, 8),
            "text": torch.randn(1, 2, 8),
            "text_mask": torch.ones(1, 2),
        },
        program=LoopedDiTScheduleConfig(step_count=4, refresh_steps=(0, 2)),
    )
    print(f"[Image / Looped-DiT] step_count={dit_res.step_count} core_passes={dit_res.loop_stats['block_calls']}")

    # 3. Text: Dual-window RK3/Euler decoder + genuine tiny DoryForCausalLM geodesic loop
    txt_model = _SimpleStack(depth=12, dim=8, attr="layers").eval()
    txt_adapter = HFDecoderAdapter()
    txt_res = txt_adapter.infer(
        txt_model,
        {"hidden_states": torch.randn(1, 4, 8)},
        program=(
            TextLoopConfig(start_layer=2, end_layer=4, num_loops=2, strategy="heun"),
            TextLoopConfig(start_layer=7, end_layer=9, num_loops=2, strategy="euler"),
        ),
    )
    print(f"[Text / HF Decoder] loop_invocations={txt_res.loop_stats['loop_invocations']} block_calls={txt_res.loop_stats['block_calls']}")

    tiny_cfg = DoryConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        n_input_layers=1,
        n_recurrent_layers=2,
        n_output_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    dory_model = DoryForCausalLM(tiny_cfg).eval()
    dory_res = DoryAdapter().infer(
        dory_model,
        {"input_ids": torch.tensor([[1, 2, 3, 4]])},
        program=DoryLoopConfig(num_loops=2, recurrent_kv_cache_mode="last_loop", geodesic_scale=0.2),
    )
    print(f"[Text / DoryForCausalLM] loop_invocations={dory_res.loop_stats['loop_invocations']} block_calls={dory_res.loop_stats['block_calls']}")

    # 4. Video: Causal World Model (final-pass KV commit) & Wan2.1 diffusion LoopProgram
    wm_model = _SimpleStack(depth=10, dim=8, attr="blocks").eval()
    wm_adapter = CausalWorldModelAdapter("longlive2")
    kv_cache: dict[str, int] = {}
    wm_res = wm_adapter.infer(
        wm_model,
        {"x": torch.randn(1, 4, 8), "steps": [0], "chunks": [0], "kv_cache": kv_cache},
        program=VideoWorldModelLoopConfig(
            kind="windowwise", start_layer=3, end_layer=5, k=4, active_steps=(0,), active_chunks=(0,)
        ),
    )
    print(f"[Video / World Model] kv_rollbacks={wm_res.loop_stats['kv_rollbacks']} kv_commits={wm_res.loop_stats['kv_commits']}")

    wan_model = _SimpleStack(depth=8, dim=8, attr="blocks").eval()
    wan_adapter = Wan21Adapter()
    wan_res = wan_adapter.infer(
        wan_model,
        {"x": torch.randn(1, 4, 8), "steps": 4},
        program=VideoDiffusionLoopConfig(blocks=(2, 3), repeats=3, strength=1.0, start_frac=0.0, end_frac=0.5),
    )
    print(f"[Video / Wan2.1] step_count={wan_res.step_count} block_calls={wan_res.loop_stats['block_calls']}")


if __name__ == "__main__":
    demo_multimodal_adapters()
