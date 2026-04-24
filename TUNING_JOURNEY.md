# Tuning Journey: Research Agent RL

A record of the concrete failure modes we hit and the fixes that resolved
them, for Week 1 SFT and Week 3 multi-turn GRPO. Kept as a post-mortem so
the pattern-of-mistakes is visible even after the commit messages scroll
out of sight.

Terminology: **Kaggle kernel** = a single run of a notebook on Kaggle's
hosted GPU (T4). Hard-capped at 12 hours of wall time; anything the kernel
hasn't written to `/kaggle/working/` by the cap is lost when it's killed.
This budget shaped a lot of the decisions below.

---

## Week 1 — SFT

**Goal:** fine-tune Qwen2.5-0.5B-Instruct with LoRA so it produces JSON
actions (`search`, `read`, `answer`) in the format the Week 2/3 tool
executor expects.

### Starting point
Initial SFT followed a minimal recipe: generate synthetic traces from
HotpotQA supporting facts, train with a plain causal-LM loss, save
adapter. Week 2 baselines over this adapter peaked at ~6% EM
(FixedStep(N=2)), with ~50% parse-error rate across every baseline. The
model had learned the JSON schema shape but not how to hold format across
multiple interactive turns.

### Issue 1: train / inference format mismatch

**Symptom.** At inference time the model sees real `Observation: <retrieved
text>` lines injected between its own actions; at training time the SFT
traces had never contained those lines. Cross-entropy was computed over
the entire trace including actions that followed where observations would
appear at inference time. The model had no idea what to do after an
observation.

**Fix.** Rewrote [data/sft_dataset.py](data/sft_dataset.py):

- Added `OBS_PREVIEW = 300` and rendered real `Observation:` lines into
  the training text, so the training context matches inference.
- Added `compute_observation_char_spans()` to identify which character
  ranges are observations.
- In `SFTTraceDataset.__getitem__`, used the tokenizer's
  `return_offsets_mapping=True` to map character spans → token positions
  and masked observation tokens out of the loss with
  `labels[i] = IGNORE_INDEX`.

Net effect: the model reads observations in context but is only trained
to predict its own actions. Commit: `6da2a6c`.

### Issue 2: LoRA capacity too small

**Symptom.** Format-error rate stayed high even after observation masking.
The original LoRA targeted only attention projections (`q/k/v/o_proj`),
which is fine for style transfer but weak for structural behavior like
"always close every brace" or "always emit `action` before `query`".

**Fix.** Expanded `target_modules` in [config.yaml](config.yaml) to
include the MLP projections (`gate_proj`, `up_proj`, `down_proj`). Total
trainable parameter count roughly tripled; final adapter is 34 MB. Commit:
`6da2a6c`.

### Issue 3: not enough epochs

Bumped `num_train_epochs: 3 → 5` in the same commit. At 5 epochs the best
`eval_loss` was 0.197 at step 940, a real improvement over 3-epoch runs
that plateaued around 0.28.

### Issue 4: dataset prep timed out on Kaggle (the one that stung)

**Symptom.** We rewrote `data/prepare_sft_dataset.py` to use a **global
BM25 index over all 90k HotpotQA train documents** so traces would be
built against the same retrieval distribution the model would see at
inference. The kernel died at 43,200 s (the exact 12 h cap) having never
reached the `write_jsonl` call. Nothing saved.

**Diagnosis.** BM25 scoring is pure Python. Common query tokens
(`film`, `born`, `director`) had posting lists of roughly 50,000 documents
each. Each `search()` call took ~500 ms. At 2 searches per trace × 6,000
traces, the hot loop was 10.5 h minimum, with no headroom for
tokenization overhead. And because the only `write_jsonl` call was at the
end of both splits, a timeout wiped every trace.

**Fix (two-part).** In [data/prepare_sft_dataset.py](data/prepare_sft_dataset.py):

1. **Per-example mini-indices.** Each HotpotQA example ships with its own
   ~10 candidate paragraphs (2 gold + 8 distractor). Replaced the global
   index with `build_mini_tool(example)` which builds a `ToolExecutor`
   over just those 10 docs. BM25 scoring is now O(10 docs) per search
   instead of O(50k). Per-example cost dropped from ~1 s to <1 ms.
2. **Incremental JSONL flushing.** Added `FLUSH_EVERY = 500` so records
   are written to disk every 500 traces instead of only at the end. Any
   future timeout loses at most 500 records, not the whole run.

Post-fix wall time: 2 minutes instead of 10.5 hours for 6,000 traces.
Commit: `d215feb`.

### Issue 5: tight budget

Reduced `train_size: 8000 → 6000` so the whole Week 1 kernel (prep +
training) fit comfortably under 12 h even with conservative estimates.
Commit: `3d46b3c`.

### Final Week 1 outcome

- Adapter: 34 MB, LoRA r=16, α=32, all 7 attention + MLP projections
- 5 epochs over 6,000 traces, final step 940
- Best `eval_loss = 0.197`
- Week 2 best baseline (Confidence τ=0.85): **26% EM** — a ~20-point lift
  over the first-attempt SFT's 6%, entirely from the format-fragility fixes.

### Lessons

- **Train/inference format mismatch** is a silent killer. The loss can
  look fine while the model has never been exposed to the actual
  inference-time context.
- **Benchmark hot loops locally before pushing to Kaggle.** A Kaggle
  kernel is an all-or-nothing 12 h budget; a wrong time estimate wipes
  the run. Rule saved in project memory: before any BM25 / tokenization
  loop goes to Kaggle, run one iteration locally and extrapolate, and
  demand ≥2× headroom on the 12 h cap.
- **Never let durable output hang off a single end-of-run write.** Flush
  every N records.

---

## Week 3 — Multi-turn GRPO

**Goal:** use GRPO with **real tool execution at each step** (not virtual
traces) to push the SFT policy past the 26% baseline ceiling on HotpotQA.

### Starting point
First GRPO attempt on the old weak SFT: the learnable-zone filter
rejected 96% of questions because *all* G=3 rollouts were wrong. GRPO
advantages `(R_i − mean(R)) / std(R)` collapse to zero on tied groups,
so there was almost no gradient signal. This was the motivator for
strengthening SFT first (§Week 1 above). Then we retried GRPO.

### Issue 1: rollout temperature hardcoded to 0.9

**Symptom.** First retry on Kaggle: the filter stage printed
`{0: 1000, 1: 0, 2: 0, 3: 0}` — every single question had 0 out of 3
rollouts correct. Not one of 1,000 questions produced a single working
rollout. Filter ran 9.6 h before the 12 h timeout killed the kernel; no
training ever started.

**Diagnosis.** `collect_episodes_batched` in
[rl/multi_turn_grpo.py:254](rl/multi_turn_grpo.py#L254) had
`temperature=0.9` as default. `filter_learnable_questions` called it
without passing `temperature`. Week 2 baselines, which worked at 26% EM,
ran at `temperature=0.1`. At 0.9, the 0.5B model can't hold JSON format
across a multi-step generation — parse fails on the first step, rollout
gives up.

**Secondary symptom explaining the timeout.** Because no rollout emitted
a valid `action: "answer"`, none of them short-circuited — every rollout
walked the full `max_steps` burning 150 tokens per step. That's ~35 s per
question × 1000 questions = 9.7 h on filter alone.

**Fix.** Threaded a `temperature` parameter through
`filter_learnable_questions` → `grpo_train_step` → `train` →
`collect_episodes_batched`, defaulting to 0.5. Notebook call sites pass
`temperature=0.5` explicitly. 0.5 is close enough to the T=0.1 baseline
to preserve JSON format while still giving rollout-to-rollout variance
(which GRPO needs for non-degenerate advantages).

Also trimmed `N_TRAIN: 1000 → 200` since filter cost scales linearly and
the learnable set was only going to be a fraction of whatever N_TRAIN
was. Commit: `d14b63b`.

### Issue 2: `max_steps=3` too tight for 2-hop HotpotQA

**Symptom.** Local smoke test at the new T=0.5 on 10 real HotpotQA val
questions the Week 2 baseline had gotten right: `{0: 10, 1: 0, 2: 0, 3: 0}`.
All 30 rollouts wrong.

**Diagnosis — inspecting the raw rollout transcripts.** The model's JSON
was *perfect*. Every step was a clean `{"thought": ..., "action": ...,
"confidence": ...}`. And the sequence was exactly what a 2-hop HotpotQA
agent should do:

```
Step 1: search("Big Stone Gap director romantic comedy")
Step 2: read(doc_00050)
Step 3: search("Adriana Trigiani ...")
# rollout ends here, max_steps=3, never emits answer
```

With `max_steps=3` the rollout was cut off on the *second* search — the
model never got to `read → answer` for the second hop. The SFT training
traces are literally 5 steps long (`search → read → search → read →
answer`); at inference it was trying to follow that same 5-step pattern
and being killed at step 3.

**Fix.** Raised `max_steps: 3 → 5` in both the filter and training calls
in [kaggle_week3_grpo.ipynb](kaggle_week3_grpo.ipynb) cells 5b and 6. Same
commit as Issue 1: `d14b63b`.

### Issue 3: LoRA dropout corrupting training-time rollouts

**Symptom.** First Kaggle run with Issues 1+2 fixed started training, but
every epoch logged:

```
Epoch  2/200  loss=+0.0000  acc=0.00  steps=1.0  (54s)
Epoch  3/200  loss=+0.0000  acc=0.00  steps=1.0  (51s)
```

`steps=1.0` on every training rollout — impossible if rollouts were
reaching the `answer` action at step 5 like the val rollouts were.
Meanwhile `[VAL] acc=0.350, steps=5.0` — at eval time the rollouts
worked fine.

**Diagnosis.** `grpo_train_step` calls `model.train()` at the top, which
enables **LoRA dropout (0.05)**. The rollout call (`collect_episodes_batched`)
has `@torch.no_grad()`, but `torch.no_grad()` only disables gradient
tracking — it does **not** disable dropout. Dropout is controlled by
`model.train()` vs `model.eval()`. With dropout active during generation
at T=0.5, the first step's tokens got corrupted enough that
`_canonicalize_step_output` returned `{}`. Then at
[rl/multi_turn_grpo.py:332](rl/multi_turn_grpo.py#L332):

```python
action = step_json.get("action", "answer")  # defaults to "answer" on empty dict
if action == "answer" or not action:
    done[i] = True    # rollout terminates at step 1 with empty pred_answer
```

Every training rollout died at step 1 with no answer, scored zero
correctness and identical format rewards, so advantages across the G=4
rollouts collapsed to zero. Loss accumulated zero. **The model parameters
never updated.** The `[VAL] acc=0.350` at every validation was just the
unchanged SFT adapter being evaluated in eval mode.

The val loop had `model.eval()` → `model.train()` bracketing, which is
why val rollouts produced clean 5-step trajectories. Only the training
loop was broken.

**Fix.** Added `model.eval()` before `collect_episodes_batched` and
`model.train()` after, inside the `for q in questions` loop in
[rl/multi_turn_grpo.py:601](rl/multi_turn_grpo.py#L601). Backward pass
still runs in train mode (required for LoRA gradient checkpointing).
Commit: `a9d65d8`.

### Final Week 3 outcome

After the three fixes the next Kaggle run trained cleanly:

- 41 of 200 epochs (~20%) produced non-zero loss. Remaining epochs had
  tied reward groups — expected and not harmful.
- `[VAL]` accuracies over the run: 0.50, 0.45, 0.35, 0.35, 0.40, 0.45,
  0.40, 0.50, 0.40, 0.50 (every 20 epochs, 20-question samples with
  ~11 pp standard error — noisy).
- **Final eval on 100 val questions: 0.44 EM.**

### Scoreboard

| Configuration | EM | Notes |
|---|---|---|
| Week 2 FixedStep(N=2) | 6.0% | simple floor |
| Week 2 Confidence(τ=0.85) | **26.0%** | best zero-shot baseline |
| SFT-only @ max_steps=5, T=0.1 | ~35% | SFT alone once the max_steps-fix is applied |
| **Week 3 GRPO final** | **44.0%** | **+18 points over Week 2 best** |

### Lessons

- **`torch.no_grad()` ≠ `model.eval()`.** The two do different things.
  No-grad turns off gradient tracking; eval turns off dropout/batchnorm.
  For RL rollouts of a LoRA-trained policy you need both.
- **Default kwarg values that match training-time needs but not
  inference-time needs are landmines.** `collect_episodes_batched`
  worked fine in early tests because those tests happened to set
  `temperature` explicitly. It broke the moment a new caller
  (`filter_learnable_questions`, `grpo_train_step`) used the default.
- **Match `max_steps` to the SFT trajectory length.** The model will
  follow the pattern it was trained on; if its natural trajectory is
  5 steps and you cap it at 3, every rollout fails deterministically.
- **Read the raw rollout text when diagnosing "0% correct".** Two very
  different bugs produce the same top-line number: "model emits garbage"
  (symptom: parse fails) and "model emits clean JSON but runs out of
  steps" (symptom: no `answer` action). You can only tell them apart by
  looking at an actual trajectory.
- **`steps=1.0` on every training epoch is a smoking gun.** If every
  rollout terminates at step 1, something is making the parser give up
  deterministically. In our case: dropout.

---

## Common thread

Three out of four of the Week 3 issues were variations on the same
pattern: **a configuration choice that worked for the original caller
silently broke when a new caller used the defaults.**

- `temperature=0.9` was the default for `collect_episodes_batched` at
  a time when only generation-heavy test code called it; it became
  wrong the moment filter and training called it.
- `max_steps=3` was correct for 1-hop tests; it became wrong for 2-hop
  HotpotQA trajectories.
- `model.train()` at the top of `grpo_train_step` was correct for the
  *training* forward pass; it was wrong for the *rollout* inside the
  training function, where dropout should be off.

The useful rule in hindsight: **any function whose behavior depends on
`model.training` or on sampling temperature should take those as
explicit parameters, not inherit them from call-site state.**
