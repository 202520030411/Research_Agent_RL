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
    """One LLM turn: the generated text and its token span in the full sequence."""
    text:       str
    token_start: int   # index in the full concatenated token sequence
    token_end:   int


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
    Returns an Episode with the full trajectory text and turn token spans.
    """
    history: list[dict] = []
    full_parts: list[str] = []   # alternating: [prompt, step1, obs1, step2, ...]
    turn_records: list[TurnRecord] = []
    pred_answer = ""
    token_cursor = 0

    for step_idx in range(max_steps):
        prompt = _build_turn_prompt(question, history, tokenizer, system_prompt, step_idx)

        # Tokenise the prompt up to this point for generation
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids = out_ids[0][enc["input_ids"].shape[1]:]
        raw_step_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        step_text, step_json = _canonicalize_step_output(raw_step_text)

        # Track token span for this LLM turn in the FULL sequence
        # (computed after full concatenation, approximated here via re-tokenisation)
        turn_records.append(TurnRecord(
            text=step_text,
            token_start=token_cursor,  # will be recalculated after concat
            token_end=token_cursor,
        ))
        full_parts.append(step_text + "\n")

        action    = step_json.get("action", "answer")

        if action == "answer" or not action:
            pred_answer = step_json.get("answer", "")
            if not pred_answer:
                import re
                m = re.search(r'"answer"\s*:\s*"([^"]+)"', raw_step_text)
                pred_answer = m.group(1) if m else step_text[:80]
            history.append({"step_json": step_json, "observation": ""})
            break

        # Execute real tool
        observation = ""
        if action == "search":
            query      = step_json.get("query", question)
            observation = tool_executor.search(query)
        elif action == "read":
            doc        = step_json.get("document", "")
            observation = tool_executor.read(doc)

        history.append({"step_json": step_json, "observation": observation})
        if observation:
            full_parts.append(f"\nObservation: {observation[:300]}\n")

    # Build full trajectory text
    from data.sft_dataset import SYSTEM_PROMPT as SYS
    base_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user",   "content": f"Question: {question}"}],
        tokenize=False, add_generation_prompt=True,
    )
    full_text = base_prompt + "".join(full_parts)

    # Recompute token spans by tokenising the full text
    full_enc = tokenizer(full_text, return_tensors="pt")
    prompt_enc = tokenizer(base_prompt, return_tensors="pt")
    prompt_len = prompt_enc["input_ids"].shape[1]

    # Walk through parts to find each turn's token span
    running_text = base_prompt
    cursor = prompt_len
    for i, rec in enumerate(turn_records):
        turn_enc  = tokenizer(rec.text, return_tensors="pt")
        turn_len  = turn_enc["input_ids"].shape[1]
        rec.token_start = cursor
        rec.token_end   = cursor + turn_len
        cursor   += turn_len
        # Skip observation tokens
        if i < len(history) and history[i].get("observation"):
            obs_text = f"\nObservation: {history[i]['observation'][:300]}\n"
            obs_enc  = tokenizer(obs_text, return_tensors="pt")
            cursor  += obs_enc["input_ids"].shape[1]

    correct = bool(pred_answer) and _check_correct(pred_answer, gold_answer)

    return Episode(
        question=question,
        gold_answer=gold_answer,
        turns=turn_records,
        full_text=full_text,
        correct=correct,
        n_steps=len(history),
    )


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

def compute_trajectory_log_probs(
    model,
    tokenizer,
    episode: Episode,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute the sum of log-probs of LLM-generated tokens in this episode.

    We run ONE forward pass over the full concatenated trajectory and extract
    the log-probs only at positions where the LLM generated tokens (masking
    out the prompt and tool observation tokens).

    Returns a scalar tensor (requires_grad=True) — the sum of
    log π_θ(a_t | context_t) over all generated tokens.
    """
    enc = tokenizer(
        episode.full_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    input_ids = enc["input_ids"]           # (1, T)
    T = input_ids.shape[1]

    # Build generation mask: 1 at LLM-generated token positions, 0 elsewhere
    gen_mask = torch.zeros(T, dtype=torch.bool, device=device)
    for rec in episode.turns:
        s = min(rec.token_start, T - 1)
        e = min(rec.token_end,   T)
        if e > s:
            gen_mask[s:e] = True

    # Forward pass
    with torch.enable_grad():
        outputs    = model(input_ids=input_ids)
        logits     = outputs.logits[0]          # (T, V)
        log_probs  = F.log_softmax(logits, dim=-1)

        # Shift: log_prob[t] = log P(token[t+1] | token[0..t])
        shift_log_probs = log_probs[:-1]        # (T-1, V)
        shift_targets   = input_ids[0, 1:]      # (T-1,)
        shift_mask      = gen_mask[1:]          # (T-1,)

        token_log_probs = shift_log_probs[
            torch.arange(T - 1, device=device), shift_targets
        ]                                        # (T-1,)

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
    G:            int   = 4,
    max_steps:    int   = 5,
    alpha:        float = 0.1,
    beta:         float = 0.05,
    epsilon:      float = 0.05,
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
    total_loss      = torch.tensor(0.0, device=device, requires_grad=True)
    total_correct   = 0
    total_steps     = 0
    total_episodes  = 0

    for q in questions:
        question    = q["question"]
        gold_answer = q.get("answer", "")

        # --- Collect G episodes ---
        group: list[Episode] = []
        for _ in range(G):
            ep = collect_episode(
                question, gold_answer, model, tokenizer, tool_executor,
                system_prompt=system_prompt,
                max_steps=max_steps,
                device=device,
            )
            group.append(ep)

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

        # --- Policy gradient loss ---
        for ep, adv in zip(group, advantages):
            log_prob  = compute_trajectory_log_probs(model, tokenizer, ep, device)
            total_loss = total_loss - log_prob * adv

            total_correct  += int(ep.correct)
            total_steps    += ep.n_steps
            total_episodes += 1

    if total_episodes > 0:
        total_loss = total_loss / total_episodes

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss":      total_loss.item(),
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
    batch_size:      int = 2,     # questions per update (G=4 → 8 episodes/update)
    G:               int = 4,
    lr:              float = 5e-6,
    max_steps:       int = 5,
    alpha:           float = 0.1,
    beta:            float = 0.05,
    epsilon:         float = 0.05,
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
