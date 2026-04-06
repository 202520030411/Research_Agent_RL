# Research Agent RL

**Efficient Research Agents via Hybrid Training: SFT + RL-Based Decision Policies**

A two-component system for building efficient open-domain QA agents:

1. **SFT LLM** (Qwen2.5-0.5B-Instruct, QLoRA) — learns structured reasoning and calibrated confidence estimation
2. **RL Policy Network** (MLP, Week 3) — learns when to stop searching, framed as an optimal stopping problem

---

## Project Structure

```
Research_Agent_RL-1/
├── config.yaml                    # All hyperparameters
├── requirements.txt
├── data/
│   ├── prepare_sft_dataset.py     # HotpotQA → structured reasoning traces (JSONL)
│   └── sft_dataset.py             # PyTorch Dataset with instruction masking
└── sft/
    ├── model.py                   # Qwen2.5-0.5B + 4-bit QLoRA setup
    └── train_sft.py               # TRL SFTTrainer entry point
```

---

## Week 1: SFT Dataset & Fine-tuning

### Prerequisites

**On Kaggle:** enable the T4 GPU accelerator and turn on internet access in notebook settings.

**Install dependencies:**
```bash
pip install -r requirements.txt
```

### Step 1 — Build the SFT dataset

Downloads HotpotQA (distractor split) and converts examples into multi-step reasoning traces:

```bash
python data/prepare_sft_dataset.py
```

This writes:
- `data/sft_traces/train.jsonl` — 8 000 training traces
- `data/sft_traces/val.jsonl` — 500 validation traces

Each JSONL line is a `{"question": ..., "answer": ..., "trace": [...]}` record. Each step in `trace` is a JSON object the model must learn to produce:

```json
{"thought": "I need to find ...", "action": "search", "query": "...", "confidence": 0.31}
{"thought": "The document says ...", "action": "read", "document": "...", "confidence": 0.55}
{"thought": "Based on both sources ...", "action": "answer", "answer": "...", "confidence": 0.89}
```

### Step 2 — Fine-tune

```bash
python sft/train_sft.py
# or with an explicit config path:
python sft/train_sft.py --config config.yaml
```

Checkpoints are saved to `checkpoints/qwen-sft/` every 200 steps.  
The best checkpoint (lowest eval loss) is saved to `checkpoints/qwen-sft/final/`.

**Expected Kaggle T4 runtime:** ~2–3 hours for 3 epochs over 8k samples.

---

## Configuration

All hyperparameters live in `config.yaml`. Key sections:

| Section | Key settings |
|---|---|
| `model` | `name`, `max_seq_length` |
| `quantization` | 4-bit NF4, double quant |
| `lora` | `r=16`, `alpha=32`, targets `q/k/v/o_proj` |
| `dataset` | `train_size=8000`, `val_size=500` |
| `training` | `lr=2e-4`, `epochs=3`, batch 4 × grad_accum 4 |
| `reward` | `alpha`, `beta`, `epsilon` (used in Week 3) |

---

## Timeline

| Week | Goal |
|------|------|
| 1 | SFT dataset preparation & fine-tuning *(current)* |
| 2 | Baseline agent implementation (fixed-step, confidence-threshold) |
| 3 | RL policy network training (PPO/REINFORCE) |
| 4 | Evaluation & analysis |

---

## Reward Design (Week 3 Preview)

The RL stopping policy is trained with:

```
R = +1 (correct answer)
  - α · t                          (step penalty)
  - β · Σ 1[JSD(P_ans^t ∥ P_ans^{t-1}) < ε]  (low-info-gain penalty)
```

The Jensen–Shannon divergence between consecutive answer distributions is computed directly from the LLM's `confidence` output — no oracle labels required.
