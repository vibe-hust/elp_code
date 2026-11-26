# Evolving Loop Programs (ELP)

<a id="authors"></a>

**👥 Authors:** [Yang Chen](https://github.com/cheyan9)<sup>1</sup> · [Kaitai Shen](https://github.com/Kaitai-Shen)<sup>1</sup> · [Xuemei Zhou](https://www.cwi.nl/en/people/xuemei-zhou/)<sup>2</sup> · [Teng Long](https://tenglon.github.io/)<sup>3</sup> · [Vincent Tao Hu](https://taohu.me/)<sup>1</sup>

<sup>1</sup> Huazhong University of Science and Technology<br>
<sup>2</sup> Centrum voor Wiskunde en Informatica (CWI)<br>
<sup>3</sup> University of Trento

[🌐 Project website](https://vibe-hust.github.io/elp_page/) · <a href="https://github.com/vibe-hust/elp_code"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/github-mark-white.svg"><img src="docs/assets/github-mark.svg" width="16" height="16" alt="GitHub"></picture> Code repository</a>

*In Submission · Training-Free Multimodal Search & Inference Across Image, Text, and Video*

**Evolving Loop Programs (ELP)** discovers executable inference-time loop programs `(code, params)` for frozen transformer backbones across **Image**, **Text**, and **Video** with **zero weight updates**. Offline closed-loop search jointly optimizes *loop structure*, *state-update dynamics*, and *execution controls* guided by read-only runtime diagnostics and Pareto hypervolume; discovered programs are frozen as standalone JSON artifacts (`frozen/*.json`) and replayed deterministically with zero search-LLM overhead.

---


<a id="method-in-motion"></a>

## 🎬 Method in motion

![ELP method animation](docs/assets/method-intro.gif)

An illustrated example of program proposal, execution, feedback, and frozen replay.

[Static overview](docs/assets/method-overview.png) · [Animation source and notes](docs/method-animation.md)

<a id="architecture"></a>

## 🗂️ Architecture

```text
elp_code/
├── pyproject.toml                       # Installs `elp` package and CLI entrypoint
├── LICENSE                              # MIT License
├── README.md
├── elp/
│   ├── core/                            # Search loop engine, AST sandbox, probes & Pareto
│   │   ├── program.py                   # AST-checked LoopProgram, invoke_loop_program, SHA-256 fingerprint, freeze/replay
│   │   ├── diagnostics.py               # Read-only built-in & programmable `collect(...)` hooks (Eq. 3)
│   │   ├── pareto.py                    # Non-dominated frontier, hypervolume & Speed/Balanced/Quality (Eq. 5)
│   │   └── search.py                    # Closed-loop SearchCoordinator with resume & control guard (Eq. 4)
│   ├── agent/                           # Agent sessions, isolated workspace & subprocess workers
│   │   ├── backends.py                  # AgentSession adapters: `codex`, `openai`, `cli`, `mock`
│   │   ├── sandbox.py                   # AST-checked workspace editor & offline WorkspaceTool shell
│   │   ├── prompts.py                   # Structured closed-loop JSON search & diagnostic contract
│   │   ├── worker.py                    # Isolated worker (`evaluate`, `diagnose`, `control`) + pluggable scorer
│   │   └── coordinator.py               # SessionProposer, SubprocessWorkerEvaluator & AgentSearchRunner
│   ├── backends/                        # Reversible inference patches & BackboneAdapters (weights frozen)
│   │   ├── base.py                      # Unified `BackboneAdapter`, `InferenceResult`, `BlockLoopStats`
│   │   ├── registry.py                  # Registry covering all 9 Image/Text/Video backbones
│   │   ├── image/                       # `scale_rae`, `rae`, `looped_dit`
│   │   ├── text/                        # `qwen3` (HF causal decoders), `dory` (`DoryForCausalLM` + safetensors)
│   │   └── video/                       # `longlive2`, `longlive1`, `self_forcing`, `wan21`
│   ├── modalities/                      # Pure tensor operators (Euler/Heun/RK, AB2, Geodesic, Window/KV)
│   └── cli.py                           # Unified CLI (`elp list | infer | probe | search`)
├── third_party/                         # Bundled stripped upstream inference trees & configs
│   ├── scale_rae/                       # Scale-RAE (`scale_rae/`, `inference/`)
│   ├── rae/                             # DiffusionBench RAE-SigLIP2-B/DDT (`src/`, `configs/`)
│   ├── looped_dit/                      # OpenSenseNova Looped-DiT (`looped_dit/`, `configs/`)
│   ├── self_forcing/                    # Self-Forcing (`wan/`, `pipeline/`, `model/`, `utils/`, `configs/`)
│   ├── longlive1/                       # LongLive v1.0 (`wan/`, `pipeline/`, `model/`, `utils/`, `configs/`)
│   ├── longlive2/                       # LongLive 2.0 (`wan_5b/`, `pipeline/`, `model/`, `utils/`, `configs/`, `fouroversix/`)
│   └── wan21/                           # Wan2.1 (`wan/`)
├── examples/
│   ├── run_toy_search.py                # Fast CPU closed-loop search, control check, probe & freeze/replay
│   └── run_backbone_inference.py        # Multimodal reversible patch demo across Image/Text/Video
└── tests/
    ├── test_elp_core.py                 # Core AST sandbox, invoke_loop_program, programmable probes, Pareto & resume
    ├── test_backends_contract.py        # Structural & KV-rollback contract verification across all 9 backbones
    └── test_agent_and_cli.py            # Worker scorer, programmable diagnostics, resume, control guard & CLI
```

---

<a id="backbone-integration-matrix"></a>

## 🧩 Backbone Integration Matrix

All upstream model inference trees are bundled cleanly inside [`third_party/`](third_party/README.md) with zero external cloning required (`--upstream-repo` / env vars remain supported as overrides).

| Backbone (`--backbone`) | Modality | Upstream Family | Loader Strategy (`loading`) | Bundled Source / Override |
|---|---|---|---|---|
| `scale_rae` | Image | Scale-RAE (Flow + DiT/RAE) | `bundled_upstream` | `third_party/scale_rae` (`SCALE_RAE_REPO` / `--upstream-repo`) |
| `rae` | Image | diffusion-bench (`DiTwDDTHead`) | `bundled_upstream` | `third_party/rae` (`DIFFUSION_BENCH_REPO` / `--upstream-repo`) |
| `looped_dit` | Image | OpenSenseNova `LoopedMMDiT` | `bundled_upstream` | `third_party/looped_dit` (`LOOPED_DIT_REPO` / `--upstream-repo`) |
| `qwen3` | Text | HF Causal Decoder (`Qwen3`, etc.) | `transformers` | HF Repo ID or local dir |
| `dory` | Text | Dory1.0-4B (nGPT Looped LM) | `builtin_safetensors` | Self-contained `DoryForCausalLM` + safetensors |
| `longlive2` | Video | NVlabs LongLive 2.0 (`main`) | `bundled_upstream` | `third_party/longlive2` (`LONGLIVE2_REPO` / `--upstream-repo`) |
| `longlive1` | Video | NVlabs LongLive 1.0 (`v1.0`) | `bundled_upstream` | `third_party/longlive1` (`LONGLIVE1_REPO` / `--upstream-repo`) |
| `self_forcing` | Video | Self-Forcing CausalWan | `bundled_upstream` | `third_party/self_forcing` (`SELF_FORCING_REPO` / `--upstream-repo`) |
| `wan21` | Video | Wan2.1 (`WanPipeline` / `WanModel`) | `diffusers_or_bundled_upstream` | `diffusers` or `third_party/wan21` (`WAN21_REPO` / `--upstream-repo`) |

---

<a id="paper-to-code-map"></a>

## 📝 Paper-to-Code Map

- **Section 3.1 (Executable Loop Operators & AST Safety)**: [program.py](elp/core/program.py) (`LoopProgram`, `validate_program_source`, `invoke_loop_program`, `program_fingerprint`, `freeze_program`, `load_frozen_program`).
- **Section 3.2 (Read-Only Diagnostic Probes, Eq. 3)**: [diagnostics.py](elp/core/diagnostics.py) (`ProbeSpec`, `DiagnosticCollector`, built-in `activation`/`residual`/`alignment`/`repeat`/`cache` stats + AST-checked `collector_code`).
- **Section 3.3 (Closed-Loop Search, Control Guard & Resume, Eq. 4)**: [search.py](elp/core/search.py) (`SearchCoordinator`, `control_check`, `initial_completed`, `initial_diagnostics`) and [coordinator.py](elp/agent/coordinator.py) (`AgentSearchRunner`, `SubprocessWorkerEvaluator`).
- **Section 3.4 (Pareto Hypervolume & Representative Selection, Eq. 5)**: [pareto.py](elp/core/pareto.py) (`pareto_frontier`, `hypervolume`, `select_representatives`).
- **Section 4 (Modality-Specific Dynamics & Reversible Runtime Patches)**: [modalities/](elp/modalities/) and [backends/](elp/backends/) (per-layer Causal KV rollback, DynamicCache transactional rollback, `DoryForCausalLM`, and `LoopedMMDiT` stream looping).

---

<a id="installation"></a>

## 🛠️ Installation

The core framework and toy search require Python >= 3.10 and PyTorch:

```bash
git clone https://github.com/vibe-hust/elp_code.git
cd elp_code
pip install -e .
```

Optional extras for HuggingFace, Diffusers, and image serialization:

```bash
pip install -e ".[image]"   # Pillow (.png / .jpg export)
pip install -e ".[text]"    # transformers, accelerate, safetensors (Qwen3, Dory)
pip install -e ".[video]"   # diffusers, transformers, accelerate (Wan2.1, world models)
```

The full CPU test suite also requires the `text` extra: its local Dory test
creates a tokenizer and a tiny safetensors checkpoint without downloading weights.

---

<a id="cli-guide"></a>

## 🚀 CLI Guide

### 1. List Supported Backbones

```bash
python -m elp list
python -m elp list --json
```

### 2. Run Closed-Loop Agent Search (`elp search`)

Candidate programs generated by the search agent execute strictly inside isolated subprocesses (`python -m elp.agent.worker`), never sharing process state with the coordinator. Real backbone search **requires** an explicit `--scorer module.path:callable` returning real quality metrics (never hardcoded).

```bash
# Offline CPU smoke test (zero API keys / zero GPU required)
python -m elp search \
  --backbone toy \
  --agent mock \
  --workspace ./tmp_search \
  --rounds 4,4 \
  --control-check

# Resume an interrupted search from state/trials.jsonl and state/diagnostics.jsonl
python -m elp search \
  --backbone toy \
  --agent mock \
  --workspace ./tmp_search \
  --rounds 4,4 \
  --resume \
  --control-check

# Real backbone search with Codex CLI, cloned upstream repo, and custom scorer
python -m elp search \
  --backbone longlive2 \
  --model-path /path/to/longlive2_ckpt.pt \
  --upstream-repo /path/to/LongLive \
  --scorer my_eval.video_scorer:score_vbench \
  --eval-inputs ./prompts.jsonl \
  --agent codex \
  --workspace ./longlive2_search \
  --rounds 16,8,8 \
  --control-check
```

Custom scorer signature:

```python
def score_candidate(*, adapter, model, program, baseline_result, program_result, inputs):
    # Return either float quality (paired with measured speedup)
    # or {"objectives": [quality, speedup], "metrics": {...}}.
    ...
```

### 3. Replay a Discovered Frozen JSON Program (`elp infer`)

```bash
# Unlooped baseline with saved image/text/tensor output
python -m elp infer \
  --backbone looped_dit \
  --model-path /path/to/looped_dit_ckpt \
  --upstream-repo /path/to/Looped-DiT \
  --program baseline \
  --steps 25 \
  --prompt "A studio photograph of a glass prism on dark slate." \
  --save-output ./out_base.png

# Replay a frozen JSON discovered by `elp search`
python -m elp infer \
  --backbone looped_dit \
  --model-path /path/to/looped_dit_ckpt \
  --upstream-repo /path/to/Looped-DiT \
  --program ./search_ws/frozen/balanced_cand01.json \
  --prompt "A studio photograph of a glass prism on dark slate." \
  --save-output ./out_elp.png
```

### 4. Run Read-Only Runtime Diagnostics (`elp probe`)

Collect built-in statistics (`activation`, `residual`, `alignment`, `repeat`, `cache`) and optionally execute a custom read-only `def collect(layer_idx, pass_idx, x, y, info): -> dict[str, float]` script:

```bash
python -m elp probe \
  --backbone qwen3 \
  --model-path Qwen/Qwen3-4B \
  --layers 14,16,25,27 \
  --features activation,residual,alignment,repeat \
  --collector-file ./custom_probe.py
```

---

<a id="cpu-verification"></a>

## 🧪 CPU Verification

Run the standalone examples and unit test suite on CPU without external weights or API keys:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python examples/run_toy_search.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python examples/run_backbone_inference.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -t . -v
```

<a id="acknowledgments"></a>

## 🙏 Acknowledgments

We thank the authors and maintainers of the models and software used by ELP:

- **Image backbones:** [Scale-RAE](https://github.com/ZitengWangNYU/Scale-RAE), [DiffusionBench / RAE](https://github.com/End2End-Diffusion/diffusion-bench), and [Looped-DiT](https://github.com/OpenSenseNova/Looped-DiT).
- **Text backbones:** [Qwen3](https://github.com/QwenLM/Qwen3) and [Dory1.0-4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16).
- **Video backbones:** [Self-Forcing](https://github.com/guandeh17/Self-Forcing), [LongLive](https://github.com/NVlabs/LongLive), and [Wan2.1](https://github.com/Wan-Video/Wan2.1).
- **Runtime libraries:** [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers), and [Diffusers](https://github.com/huggingface/diffusers).

The [bundled upstream inventory](third_party/README.md) records the source revisions and retained components.

<a id="citation"></a>

## 📚 Citation

If you use this work or code, please cite:

```bibtex
@misc{chen2026elp,
  title  = {{Evolving Loop Programs: Discovering How Frozen Transformers Should Loop}},
  author = {Chen, Yang and Shen, Kaitai and Zhou, Xuemei and Long, Teng and Hu, Vincent Tao},
  year   = {2026},
  url    = {https://github.com/vibe-hust/elp_code}
}
```

A machine-readable citation is available in [CITATION.cff](CITATION.cff).

<a id="license"></a>

## 📄 License

ELP's original code is released under the [MIT License](LICENSE).

Bundled third-party code retains its upstream terms. Consult the [upstream inventory](third_party/README.md) and the notices included with each component. Model checkpoints are covered by the terms in their respective model cards.
