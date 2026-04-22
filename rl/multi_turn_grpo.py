"""
Multi-turn GRPO trainer with real tool execution.

Instead of the LLM generating a full virtual trace in one shot, each
training episode runs an interactive loop:

    Step 1:  LLM generates JSON action  →  real tool executes  →  real result
    Step 2:  LLM sees result, generates next action  →  real tool  →  result
    ...
    Step N:  LLM generates {"action": "answer", "answer": "..."}
    Score:   correctness + format + efficiency rewards

GRPO advantage (within-group normalisation):
    Run G=4 episodes for the same question.
    A_i = (R_i - mean(R)) / (std(R) + 1e-8)

Policy gradient loss (vanilla REINFORCE + GRPO baseline, no PPO clipping):
    L = -Σ_t  A  *  Σ_{token t}  log π_θ(token | context)
    Gradient only through LLM-generated tokens — tool observations are masked.

One forward pass over the full concatenated trajectory lets us extract all
token log-probs in a single call, keeping T4 VRAM usage manageable.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from rl.grpo_rewards import (
    correctness_reward,
    format_reward,
    efficiency_reward,
    parse_trace,
    _extract_json_objects,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    """One LLM turn: the generated text and its character span in full_text.

    Character offsets (not token offsets) are the source of truth. Token
    positions are resolved later from the tokenizer's offset_mapping on the
    full trajectory — this avoids the fragment-tokenization drift that
    independent per-turn tokenization would introduce.
    """
    text:       str
    char_start: int   # byte/char index in Episode.full_text (inclusive)
    char_end:   int   # exclusive


@dataclass
class Episode:
    """Full multi-turn trajectory for one question."""
    question:     str
    gold_answer:  str
    turns:        list[TurnRecord] = field(default_factory=list)
    full_text:    str = ""   # prompt + all steps + all observations
    reward:       float = 0.0
    correct:      bool = False
    n_steps:      int = 0


# ---------------------------------------------------------------------------
# Prompt builder (mirrors agent/agent.py but returns plain string)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_FALLBACK = (
    "You are a research agent. For each question, reason step by step. "
    "At each step output a single JSON object with keys: "
    "thought, action (search|read|answer), confidence (0-1), "
    "and either query/document/answer depending on action. "
    "Always end with action=answer."
)


def _build_turn_prompt(
    question: str,
    history: list[dict],   # list of {step_json, observation}
    tokenizer,
    system_prompt: str,
    step_idx: int,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Question: {question}"},
    ]
    base = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    instruction = (
        "Return exactly one next step as a single JSON object only. "
        "Do not output multiple steps, observations, explanations, or any text "
        "before or after the JSON."
    )
    if not history:
        return base + instruction + f"\nStep {step_idx + 1}: "

    lines = []
    for i, entry in enumerate(history):
        lines.append(f"Step {i + 1}: {json.dumps(entry['step_json'])}")
        if entry.get("observation"):
            lines.append(f"Observation: {entry['observation'][:300]}")
    lines.append(instruction)
    lines.append(f"Step {step_idx + 1}: ")
    return base + "\n".join(lines)


def _canonicalize_step_output(raw_text: str) -> tuple[str, dict]:
    """
    Clamp a raw generation to its first valid JSON step.

    The SFT model often continues by generating multiple future steps in one
    shot. For multi-turn interaction we only keep the first valid JSON object
    and ignore everything after it.

    Returns:
        canonical_text : compact JSON string if parse succeeds, else raw first line
        step_json      : parsed first JSON object, or {} if parsing failed
    """
    objs = _extract_json_objects(raw_text)
    if objs:
        step_json = objs[0]
        return json.dumps(step_json, ensure_ascii=True), step_json

    first_line = raw_text.strip().splitlines()[0].strip() if raw_text.strip() else ""
    return first_line, {}


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_episode(
    question:      str,
    gold_answer:   str,
    model,
    tokenizer,
    tool_executor,
    system_prompt: str,
    max_steps:     int = 5,
    max_new_tokens: int = 256,
    temperature:   float = 0.9,
    device:        str = "cuda",
) -> Episode:
    """
    Run one interactive episode with real tool calls.
    Returns an Episode with the full trajectory text and per-turn char spans.
    """
    history: list[dict] = []
    turn_records: list[TurnRecord] = []
    pred_answer = ""

    # Build the base prompt once — it's the prefix of every turn's input and
    # also the prefix of the final full_text we score.
    base_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user",   "content": f"Question: {question}"}],
        tokenize=False, add_generation_prompt=True,
    )

    # full_text is grown as we go; char offsets into it are the source of truth
    # for which spans count as "LLM-generated" during loss computation.
    full_text = base_prompt

    for step_idx in range(max_steps):
        prompt = _build_turn_prompt(question, history, tokenizer, system_prompt, step_idx)

        enc = tokenizer(prompt, return_tensors="pt").to(device)
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids = out_ids[0][enc["input_ids"].shape[1]:]
        raw_step_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        step_text, step_json = _canonicalize_step_output(raw_step_text)

        # Record the char span of this LLM turn inside full_text BEFORE we
        # append the trailing newline / observation, so char_end points
        # exactly one past the final generated character.
        char_start = len(full_text)
        full_text += step_text
        char_end = len(full_text)
        turn_records.append(TurnRecord(
            text=step_text,
            char_start=char_start,
            char_end=char_end,
        ))
        full_text += "\n"

        action = step_json.get("action", "answer")

        if action == "answer" or not action:
            pred_answer = step_json.get("answer", "")
            if not pred_answer:
                import re
                m = re.search(r'"answer"\s*:\s*"([^"]+)"', raw_step_text)
                pred_answer = m.group(1) if m else step_text[:80]
            history.append({"step_json": step_json, "observation": ""})
            break

        observation = ""
        if action == "search":
            query = step_json.get("query", question)
            observation = tool_executor.search(query)
        elif action == "read":
            doc = step_json.get("document", "")
            observation = tool_executor.read(doc)

        history.append({"step_json": step_json, "observation": observation})
        if observation:
            full_text += f"\nObservation: {observation[:300]}\n"

    correct = bool(pred_answer) and _check_correct(pred_answer, gold_answer)

    return Episode(
        question=question,
        gold_answer=gold_answer,
        turns=turn_records,
        full_text=full_text,
        correct=correct,
        n_steps=len(history),
    )


@torch.no_grad()
def collect_episodes_batched(
    question:      str,
    gold_answer:   str,
    model,
    tokenizer,
    tool_executor,
    system_prompt: str,
    G:             int = 2,
    max_steps:     int = 3,
    max_new_tokens: int = 150,
    temperature:   float = 0.9,
    device:        str = "cuda",
) -> list[Episode]:
    """
    Run G parallel rollouts for one question in a single batched generate per step.

    This is the key optimisation for multi-turn GRPO on a free-tier T4. The naive
    loop calls `collect_episode` G times sequentially, which means G separate
    `model.generate()` calls per step. Here we stack the G active contexts into
    one padded batch and call generate ONCE per step — same wall-time as a
    single-turn rollout.

    Rollouts terminate independently when they emit `action: "answer"`. Finished
    rollouts are dropped from the batch; active ones continue until max_steps.

    Returns a list of G Episodes with correct char spans in each `full_text`.
    """
    base_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user",   "content": f"Question: {question}"}],
        tokenize=False, add_generation_prompt=True,
    )

    histories:   list[list[dict]]        = [[] for _ in range(G)]
    full_texts:  list[str]               = [base_prompt for _ in range(G)]
    turn_lists:  list[list[TurnRecord]]  = [[] for _ in range(G)]
    pred_answers: list[str]              = ["" for _ in range(G)]
    done:        list[bool]              = [False] * G

    # Batched generate requires left-padding so every prompt's last real token
    # sits at the rightmost position — otherwise the model continues from pad
    # tokens on short rows. Save and restore to avoid leaking this into eval.
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    try:
        for step_idx in range(max_steps):
            active = [i for i in range(G) if not done[i]]
            if not active:
                break

            prompts = [
                _build_turn_prompt(question, histories[i], tokenizer, system_prompt, step_idx)
                for i in active
            ]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TRAJECTORY_TOKENS,
            ).to(device)

            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_id,
            )
            prompt_len = enc["input_ids"].shape[1]
            new_ids = out_ids[:, prompt_len:]  # (B, new_T)

            for b, i in enumerate(active):
                raw_step_text = tokenizer.decode(new_ids[b], skip_special_tokens=True).strip()
                step_text, step_json = _canonicalize_step_output(raw_step_text)

                char_start = len(full_texts[i])
                full_texts[i] += step_text
                char_end = len(full_texts[i])
                turn_lists[i].append(TurnRecord(
                    text=step_text, char_start=char_start, char_end=char_end,
                ))
                full_texts[i] += "\n"

                action = step_json.get("action", "answer")

                if action == "answer" or not action:
                    pred = step_json.get("answer", "")
                    if not pred:
                        import re
                        m = re.search(r'"answer"\s*:\s*"([^"]+)"', raw_step_text)
                        pred = m.group(1) if m else step_text[:80]
                    pred_answers[i] = pred
                    histories[i].append({"step_json": step_json, "observation": ""})
                    done[i] = True
                    continue

                observation = ""
                if action == "search":
                    query = step_json.get("query", question)
                    observation = tool_executor.search(query)
                elif action == "read":
                    doc = step_json.get("document", "")
                    observation = tool_executor.read(doc)

                histories[i].append({"step_json": step_json, "observation": observation})
                if observation:
                    full_texts[i] += f"\nObservation: {observation[:300]}\n"
    finally:
        tokenizer.padding_side = orig_padding_side

    episodes: list[Episode] = []
    for i in range(G):
        correct = bool(pred_answers[i]) and _check_correct(pred_answers[i], gold_answer)
        episodes.append(Episode(
            question=question,
            gold_answer=gold_answer,
            turns=turn_lists[i],
            full_text=full_texts[i],
            correct=correct,
            n_steps=len(histories[i]),
        ))
    return episodes


@torch.no_grad()
def filter_learnable_questions(
    questions:     list[dict],
    model,
    tokenizer,
    tool_executor,
    system_prompt: str,
    K:             int  = 3,
    max_steps:     int  = 3,
    min_correct:   int  = 1,
    max_correct:   int | None = None,
    temperature:   float = 0.5,
    device:        str  = "cuda",
    verbose:       bool = True,
) -> list[dict]:
    """
    Keep only questions where the model's rollout accuracy is in a target zone.

    For each question, runs K parallel rollouts (batched, real tools) and
    counts how many produce a correct answer. A question is kept iff
    `min_correct <= n_correct <= max_correct`.

    Defaults (K=3, min=1, max=2) implement the "not too easy, not too hard"
    learnable zone: rollouts disagree, so GRPO advantage normalisation
    produces a non-zero gradient signal instead of a tied group.

    Args:
        questions    : list of {"question": str, "answer": str} dicts
        K            : rollouts per question (default 3)
        max_steps    : cap on tool-call steps per rollout
        min_correct  : lower bound on correct rollouts (inclusive)
        max_correct  : upper bound on correct rollouts (inclusive, default K-1)
        verbose      : print progress every 25 questions

    Returns a list of the kept question dicts, each with extra fields
    `_n_correct` and `_K` recording the filter decision for inspection.
    """
    if max_correct is None:
        max_correct = K - 1

    kept: list[dict] = []
    n_total = len(questions)
    zone_counts = {k: 0 for k in range(K + 1)}

    model.eval()
    for idx, q in enumerate(questions):
        episodes = collect_episodes_batched(
            question=q["question"],
            gold_answer=q.get("answer", ""),
            model=model,
            tokenizer=tokenizer,
            tool_executor=tool_executor,
            system_prompt=system_prompt,
            G=K,
            max_steps=max_steps,
            temperature=temperature,
            device=device,
        )
        n_correct = sum(int(ep.correct) for ep in episodes)
        zone_counts[n_correct] += 1

        if min_correct <= n_correct <= max_correct:
            kept.append({**q, "_n_correct": n_correct, "_K": K})

        if verbose and (idx + 1) % 25 == 0:
            print(
                f"  [{idx + 1:4d}/{n_total}]  kept={len(kept):4d}  "
                f"zone_hist={dict(zone_counts)}",
                flush=True,
            )

    if verbose:
        pct = 100.0 * len(kept) / max(n_total, 1)
        print(
            f"filter: kept {len(kept)}/{n_total} ({pct:.1f}%) "
            f"in zone [{min_correct},{max_correct}]/{K}   "
            f"full histogram: {dict(zone_counts)}",
            flush=True,
        )
    return kept


def _check_correct(pred: str, gold: str) -> bool:
    import string, unicodedata, re
    def norm(s):
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(c for c in s if c not in string.punctuation)
        return re.sub(r"\s+", " ", s).strip()
    p, g = norm(pred), norm(gold)
    return p == g or p in g or g in p


# ---------------------------------------------------------------------------
# Log-prob computation (single forward pass over full trajectory)
# ---------------------------------------------------------------------------

MAX_TRAJECTORY_TOKENS = 1536


def compute_trajectory_log_probs(
    model,
    tokenizer,
    episode: Episode,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute the sum of log-probs of LLM-generated tokens in this episode.

    One forward pass over the full trajectory; log-probs are extracted only at
    positions inside a turn's char span. Char→token resolution uses the
    tokenizer's offset_mapping, which is the only reliable way to align a
    subrange of the full text with its tokens (fragment-tokenization drifts).

    Returns a scalar tensor (requires_grad=True).
    """
    enc = tokenizer(
        episode.full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TRAJECTORY_TOKENS,
        return_offsets_mapping=True,
    )
    input_ids     = enc["input_ids"].to(device)            # (1, T)
    offset_map    = enc["offset_mapping"][0].tolist()      # [(c_start, c_end), ...]
    T = input_ids.shape[1]

    # Warn once per overlong episode — signals that max_length needs raising or
    # max_steps/obs clipping tightening. Loss still computes on surviving turns.
    orig_len = len(tokenizer(episode.full_text, add_special_tokens=True)["input_ids"])
    if orig_len > MAX_TRAJECTORY_TOKENS:
        print(
            f"[grpo] trajectory truncated {orig_len} → {T} tokens; "
            f"turns beyond the cap will be dropped from loss",
            flush=True,
        )

    # Build generation mask via char-span overlap with each token's offset_mapping.
    # A token belongs to a turn iff its char span intersects the turn's char span.
    # Note: special tokens report (0, 0) in offset_mapping and will correctly
    # fail to intersect any turn (turns always have char_end > char_start).
    gen_mask = torch.zeros(T, dtype=torch.bool, device=device)
    turn_spans = [(rec.char_start, rec.char_end) for rec in episode.turns]
    for t_idx, (tok_cs, tok_ce) in enumerate(offset_map):
        if tok_ce <= tok_cs:
            continue  # special / padding token
        for ts, te in turn_spans:
            if tok_cs < te and tok_ce > ts:
                gen_mask[t_idx] = True
                break

    with torch.enable_grad():
        outputs   = model(input_ids=input_ids)
        logits    = outputs.logits[0]                       # (T, V)
        log_probs = F.log_softmax(logits, dim=-1)

        # log_prob[t] predicts token[t+1] given token[0..t], so the target at
        # position t is input_ids[t+1] and it counts if token[t+1] is generated.
        shift_log_probs = log_probs[:-1]                    # (T-1, V)
        shift_targets   = input_ids[0, 1:]                  # (T-1,)
        shift_mask      = gen_mask[1:]                      # (T-1,)

        token_log_probs = shift_log_probs[
            torch.arange(T - 1, device=device), shift_targets
        ]                                                   # (T-1,)

        return (token_log_probs * shift_mask.float()).sum()


# ---------------------------------------------------------------------------
# GRPO training step
# ---------------------------------------------------------------------------

def grpo_train_step(
    questions:    list[dict],
    model,
    tokenizer,
    tool_executor,
    optimizer:    torch.optim.Optimizer,
    system_prompt: str,
    G:            int   = 2,
    max_steps:    int   = 3,
    alpha:        float = 0.1,
    beta:         float = 0.05,
    epsilon:      float = 0.05,
    temperature:  float = 0.5,
    device:       str   = "cuda",
) -> dict:
    """
    One GRPO update step over a batch of questions.

    For each question:
      1. Collect G episodes (real tool calls)
      2. Score each with reward functions
      3. Normalise rewards within the group → advantages
      4. Compute policy gradient loss

    Returns a dict with loss and metrics.
    """
    model.train()
    total_correct   = 0
    total_steps     = 0
    total_episodes  = 0
    running_loss    = 0.0

    # Pre-count expected episodes so we can pre-scale each per-episode loss
    # by 1/N. This lets us call .backward() per episode — the activations
    # for that episode's forward pass are released immediately, so peak
    # memory is ONE trajectory worth of activations instead of
    # batch_size * G trajectories. Gradients accumulate in .grad in-place,
    # and optimizer.step() runs once at the end.
    n_expected = max(len(questions) * G, 1)

    optimizer.zero_grad()

    for q in questions:
        question    = q["question"]
        gold_answer = q.get("answer", "")

        # --- Collect G episodes in ONE batched generate per step ---
        group: list[Episode] = collect_episodes_batched(
            question, gold_answer, model, tokenizer, tool_executor,
            system_prompt=system_prompt,
            G=G,
            max_steps=max_steps,
            temperature=temperature,
            device=device,
        )

        # --- Score ---
        completions = [ep.full_text for ep in group]
        r_corr = correctness_reward(completions, answer=[gold_answer] * G)
        r_fmt  = format_reward(completions)
        r_eff  = efficiency_reward(completions, alpha=alpha, beta=beta, epsilon=epsilon)
        rewards = [c + f + e for c, f, e in zip(r_corr, r_fmt, r_eff)]

        for ep, r in zip(group, rewards):
            ep.reward = r

        # --- GRPO advantages (within-group normalisation) ---
        r_mean = sum(rewards) / G
        r_std  = (sum((r - r_mean) ** 2 for r in rewards) / G) ** 0.5 + 1e-8
        advantages = [(r - r_mean) / r_std for r in rewards]

        # --- Per-episode forward + backward: graph for each trajectory is
        # released immediately after backward(), so peak VRAM stays O(1) in
        # the number of rollouts instead of O(batch_size * G). ---
        for ep, adv in zip(group, advantages):
            if abs(adv) < 1e-8:
                # Tied group — advantage is zero, gradient would be zero anyway.
                # Skip the forward pass to save compute and memory.
                total_correct  += int(ep.correct)
                total_steps    += ep.n_steps
                total_episodes += 1
                continue

            log_prob = compute_trajectory_log_probs(model, tokenizer, ep, device)
            loss = -(log_prob * adv) / n_expected
            loss.backward()
            running_loss += loss.item()

            total_correct  += int(ep.correct)
            total_steps    += ep.n_steps
            total_episodes += 1

            del log_prob, loss
            torch.cuda.empty_cache()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss":      running_loss,
        "accuracy":  total_correct / max(total_episodes, 1),
        "avg_steps": total_steps   / max(total_episodes, 1),
    }


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(
    model,
    tokenizer,
    tool_executor,
    train_questions: list[dict],
    val_questions:   list[dict],
    system_prompt:   str,
    output_dir:      str = "checkpoints/mt-grpo",
    n_epochs:        int = 30,
    batch_size:      int = 2,     # questions per update (G=2 → 4 episodes/update)
    G:               int = 2,
    lr:              float = 5e-6,
    max_steps:       int = 3,
    alpha:           float = 0.1,
    beta:            float = 0.05,
    epsilon:         float = 0.05,
    temperature:     float = 0.5,
    val_every:       int = 5,
    save_every:      int = 10,
    device:          str = "cuda",
    seed:            int = 42,
):
    import json as _json

    random.seed(seed)
    torch.manual_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    log = []

    print(f"Multi-turn GRPO training: {n_epochs} epochs, "
          f"batch={batch_size} questions × G={G} episodes\n")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        batch = random.sample(train_questions, min(batch_size, len(train_questions)))

        metrics = grpo_train_step(
            questions=batch,
            model=model,
            tokenizer=tokenizer,
            tool_executor=tool_executor,
            optimizer=optimizer,
            system_prompt=system_prompt,
            G=G,
            max_steps=max_steps,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            temperature=temperature,
            device=device,
        )

        elapsed = time.time() - t0
        entry = {"epoch": epoch, **metrics, "elapsed_s": round(elapsed, 1)}
        log.append(entry)
        print(f"Epoch {epoch:3d}/{n_epochs}  "
              f"loss={metrics['loss']:+.4f}  "
              f"acc={metrics['accuracy']:.2f}  "
              f"steps={metrics['avg_steps']:.1f}  "
              f"({elapsed:.0f}s)")

        # --- Validation ---
        if epoch % val_every == 0:
            model.eval()
            val_batch = random.sample(val_questions, min(20, len(val_questions)))
            v_correct, v_steps = 0, 0
            for q in val_batch:
                ep = collect_episode(
                    q["question"], q.get("answer", ""),
                    model, tokenizer, tool_executor,
                    system_prompt=system_prompt,
                    max_steps=max_steps, temperature=0.1, device=device,
                )
                v_correct += int(ep.correct)
                v_steps   += ep.n_steps
            n_v = len(val_batch)
            print(f"  [VAL] acc={v_correct/n_v:.3f}  steps={v_steps/n_v:.1f}")
            model.train()

        # --- Checkpoint ---
        if epoch % save_every == 0:
            ckpt = f"{output_dir}/epoch{epoch:03d}"
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  Saved → {ckpt}")

    # Final save
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    with open(f"{output_dir}/training_log.json", "w") as f:
        _json.dump(log, f, indent=2)
    print(f"\nDone. Model saved to {output_dir}/final")
    return log
