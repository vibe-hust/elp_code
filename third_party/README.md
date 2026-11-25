# Bundled Upstream Repositories (`third_party/`)

To make frozen-weight multimodal inference cleanly reproducible without separate `git clone` steps, ELP bundles stripped, read-only inference code and configs from all 7 upstream model repositories below. All binaries, wheels, checkpoints, media assets, `.git/` metadata, and bytecode caches were removed.

| Subdirectory | Upstream Repository | Commit | Bundled Paths | License |
|---|---|---|---|---|
| `scale_rae/` | https://github.com/ZitengWangNYU/Scale-RAE.git | `dab3800` | `inference/`, `scale_rae/`, `inference.py` (58 files) | Apache-2.0 / upstream LICENSE |
| `rae/` | https://github.com/End2End-Diffusion/diffusion-bench.git | `4848f0c` | `src/`, `configs/`, `pyproject.toml` (222 files) | Upstream repository license (no root LICENSE file) |
| `looped_dit/` | https://github.com/OpenSenseNova/Looped-DiT.git | `18ad35c` | `looped_dit/`, `configs/` (30 files) | Upstream LICENSE |
| `self_forcing/` | https://github.com/guandeh17/Self-Forcing.git | `33593df` | `wan/`, `pipeline/`, `model/`, `utils/`, `configs/` (53 files) | Upstream LICENSE |
| `longlive1/` | https://github.com/NVlabs/LongLive.git | `e52d9ef` | `wan/`, `pipeline/`, `model/`, `utils/`, `configs/`, `inference.py` (58 files) | Upstream LICENSE |
| `longlive2/` | https://github.com/NVlabs/LongLive.git | `31c70d9` | `wan_5b/`, `pipeline/`, `model/`, `utils/`, `configs/`, `fouroversix/`, `inference.py` (209 files) | Upstream LICENSE |
| `wan21/` | https://github.com/Wan-Video/Wan2.1.git | `9737cba` | `wan/` (31 files) | Apache-2.0 (LICENSE.txt) |

> **Note:** `qwen3` uses HuggingFace `transformers` (`AutoModelForCausalLM`) directly, and `dory` uses ELP's self-contained pure-PyTorch/safetensors implementation in `elp/backends/text/dory_model.py`.
