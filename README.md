# Research Agent RL

A small tool-using research agent trained on HotpotQA: Qwen2.5-0.5B +
LoRA, fine-tuned first by SFT on synthetic interaction traces, then
refined by multi-turn GRPO whose rollouts **execute the real BM25
retrieval tool at every step** (not a virtual/self-generated trace).

The full write-up is in [report/main.pdf](report/main.pdf) and the
engineering post-mortem is in [TUNING_JOURNEY.md](TUNING_JOURNEY.md).

---

## Headline results (HotpotQA distractor, 100 validation questions)

| Method | EM (%) | Avg. steps |
|---|---|---|
| FixedStep(N=2) | 6.0 | 2.00 |
| FixedStep(N=3) | 16.0 | 2.50 |
| Confidence(τ=0.75) | 20.0 | 2.88 |
| Confidence(τ=0.85) | **26.0** | 3.20 |
| NeverStop(max=6) | 21.0 | 3.57 |
| SFT-only @ max_steps=5 | **42.0 / 44.0** | 5.00 |
| Multi-turn GRPO (ours) | **44.0** | 4.99 |

The 26% → 44% headline decomposes into **+16 points from aligning the
inference step budget with the SFT trajectory length** and
**~0 points from GRPO on top** (two independent SFT-only runs gave
42.0 and 44.0; GRPO sits inside that spread). The null result for
GRPO is analysed in the report — in short, 159 of 200 training epochs
had tied reward groups, so the effective update budget was ~41, not
200. The concrete next step is a reward term with continuous
intra-group variance on all-wrong groups.

---

## Repository layout

```
Research_Agent_RL-1/
├── config.yaml                     # all hyperparameters (model, LoRA, SFT, reward)
├── requirements.txt
│
├── agent/
│   ├── tools.py                    # BM25 ToolExecutor: search + read actions
│   ├── agent.py                    # inference-time agent: prompt + generation loop
│   └── stopping.py                 # five zero-shot stopping-policy baselines
│
├── data/
│   ├── prepare_sft_dataset.py      # HotpotQA → 5-step traces w/ real retrieval
│   └── sft_dataset.py              # PyTorch Dataset with observation-masked loss
│
├── sft/
│   ├── model.py                    # Qwen2.5-0.5B + LoRA (attention + MLP projections)
│   └── train_sft.py                # TRL SFTTrainer entry point
│
├── rl/
│   ├── grpo_rewards.py             # correctness + format + efficiency rewards
│   └── multi_turn_grpo.py          # rollouts, filter, grpo_train_step, train()
│
├── eval/
│   ├── evaluate.py                 # agent-level EM evaluator
│   └── metrics.py                  # normalisation + exact-match + substring
│
├── scripts/
│   ├── make_figures.py             # regenerates report figures from data
│   └── smoke_mt_grpo.py            # single-question GRPO smoke test
│
├── kaggle_week1_sft.ipynb          # week 1: SFT trace generation + fine-tuning
├── kaggle_week2_baselines.ipynb    # week 2: five zero-shot stopping heuristics
├── kaggle_week3_grpo.ipynb         # week 3: multi-turn GRPO with real tools
├── kaggle_sft_ablation.ipynb       # ablation: SFT-only @ max_steps=5
├── kaggle_ablation_500.ipynb       # 500-q paired SFT-vs-GRPO ablation (unused)
│
├── report/
│   ├── main.tex                    # the paper
│   ├── refs.bib                    # bibliography
│   ├── figures/                    # fig1_main.pdf, fig2_val.pdf
│   └── main.pdf                    # rendered output
│
└── TUNING_JOURNEY.md               # failure-mode post-mortem
```

---

## Reproducing the pipeline

Every stage is packaged as a self-contained Kaggle notebook. Each one
clones this repo, installs dependencies, and runs the relevant
training/eval step. Kaggle T4 (16 GB) + Internet ON is the target
environment; all four notebooks fit inside the 12 h kernel cap with
margin.

### Week 1 — SFT

Notebook: [kaggle_week1_sft.ipynb](kaggle_week1_sft.ipynb)

Generates 6000 5-step HotpotQA traces using per-example BM25
mini-indices (search → read → search → read → answer) and fine-tunes
Qwen2.5-0.5B with a LoRA adapter on attention + MLP projections. The
training loss masks out `Observation:` lines so the model is only
scored on its own JSON actions. Output: a 34 MB LoRA adapter.

Runtime: ~2 h on T4.

### Week 2 — zero-shot stopping heuristics

Notebook: [kaggle_week2_baselines.ipynb](kaggle_week2_baselines.ipynb)

Runs five fixed stopping policies over the SFT adapter on 100 val
questions: `FixedStep(N∈{2,3})`, `Confidence(τ∈{0.75, 0.85})`,
`NeverStop(max=6)`. No additional training. These are the "what can
heuristics recover without RL?" baselines.

Runtime: ~30 min on T4.

### Week 3 — multi-turn GRPO with real tool execution

Notebook: [kaggle_week3_grpo.ipynb](kaggle_week3_grpo.ipynb)

For each question, runs G=4 rollouts where every `search`/`read`
action is dispatched to the real BM25 index and the real observation
is fed back before the next step. Within-group advantage
normalisation, combined reward (correctness + format + efficiency +
low-info-gain JSD penalty), per-episode backward for T4 memory.

Key hyperparameters (see [rl/multi_turn_grpo.py](rl/multi_turn_grpo.py)):

| Flag | Value | Why |
|---|---|---|
| `G` | 4 | within-group advantage needs ≥3 rollouts |
| `max_steps` | 5 | must match the SFT trajectory length |
| `temperature` | 0.5 | at 0.9 the 0.5B model loses JSON format |
| `n_epochs` | 200 | |
| `batch_size` | 1 | per-episode backward; peak VRAM = 1 trajectory |
| `lr` | 5e-6 | standard for LoRA+RL |

Runtime: ~3 h on T4 (filter + 200 updates + checkpoint vals + final eval).

### SFT-only ablation

Notebook: [kaggle_sft_ablation.ipynb](kaggle_sft_ablation.ipynb)

Evaluates the SFT adapter alone at `max_steps=5, T=0.1` on the same
100 val questions Week 2 and Week 3 use. Isolates the "step-budget
alignment" contribution from the "GRPO training" contribution.
Device-adaptive (GPU or CPU).

Runtime: ~5 min on T4, ~45 min on CPU.

---

## Configuration

All hyperparameters live in [config.yaml](config.yaml). Key sections:

| Section | Settings |
|---|---|
| `model` | `Qwen/Qwen2.5-0.5B-Instruct`, `max_seq_length=1024` |
| `lora` | `r=16`, `alpha=32`, targets = all attention + MLP projections, dropout 0.05 |
| `dataset` | `train_size=6000`, `val_size=500`, `max_steps_per_trace=4` |
| `training` (SFT) | 5 epochs, `lr=2e-4`, cosine, warmup 75, batch 2 × accum 8, fp16 |
| `reward` (GRPO) | `alpha=0.1` (step), `beta=0.05` (JSD), `epsilon=0.05` (JSD threshold) |

GRPO-specific flags (`G`, `max_steps`, `temperature`, `n_epochs`) are
passed at the call-site in the Week 3 notebook rather than through
the YAML.

---

## Related documents

- [report/main.pdf](report/main.pdf) — the paper, 15 pages, honest
  decomposition and null-result discussion.
- [TUNING_JOURNEY.md](TUNING_JOURNEY.md) — the three Week 3 failure
  modes and their fixes, with commit hashes (rollout temperature,
  `max_steps` vs SFT trajectory length, `torch.no_grad()`
  $\neq$ `model.eval()`). Worth reading before reproducing.
- [scripts/make_figures.py](scripts/make_figures.py) — regenerates
  both report figures from the embedded experiment data.
