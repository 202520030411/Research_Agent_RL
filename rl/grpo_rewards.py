"""
Reward functions for GRPO training of the research agent LLM.

GRPO trains the LLM end-to-end: given a question, the model generates a
complete multi-step reasoning trace and we score it with these rewards.

The model generates the FULL trace in one shot (no live tool execution during
training — the model must reason from its parametric knowledge). This is the
standard approach for GRPO on T4 / free compute.

Reward components:
  1. correctness_reward   (+1.0 / 0.0)   — did the final answer match gold?
  2. format_reward        (+0.0 to +0.3)  — was the output well-structured JSON?
  3. step_penalty         (-0.1 * steps)  — penalise verbosity
  4. jsd_penalty          (-0.05 * n)     — penalise low-info-gain steps

GRPO expects each reward function to have signature:
    fn(completions: list[str], **kwargs) -> list[float]

The `prompts` list and any extra kwargs (e.g. `answer`) are passed through
GRPOTrainer and available via **kwargs.
"""

from __future__ import annotations

import json
import math
import re
import string
import unicodedata


# ---------------------------------------------------------------------------
# JSD utility
# ---------------------------------------------------------------------------

def _kl_bernoulli(p: float, q: float, eps: float = 1e-8) -> float:
    p = max(min(p, 1 - eps), eps)
    q = max(min(q, 1 - eps), eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def jsd_bernoulli(p: float, q: float) -> float:
    """Jensen-Shannon divergence between Bernoulli(p) and Bernoulli(q)."""
    m = (p + q) / 2.0
    return 0.5 * _kl_bernoulli(p, m) + 0.5 * _kl_bernoulli(q, m)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_json_objects(text: str) -> list[dict]:
    """Return all top-level JSON objects found in text."""
    objects = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    objects.append(json.loads(text[start : i + 1]))
                except json.JSONDecodeError:
                    pass
                start = -1
    return objects


def _normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation/articles, collapse whitespace."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def _is_correct(pred: str, gold: str) -> bool:
    if not pred or not pred.strip():
        return False
    pred_n = _normalize_answer(pred)
    gold_n = _normalize_answer(gold)
    if pred_n == gold_n:
        return True
    # substring match (handles "New York City" vs "New York")
    return pred_n in gold_n or gold_n in pred_n


def parse_trace(completion: str) -> dict:
    """
    Parse a model completion into its components.

    Returns a dict with:
        steps        : list of parsed step dicts
        final_answer : str (may be empty if not found)
        n_steps      : int
        confidences  : list[float]
        has_answer   : bool
    """
    objs = _extract_json_objects(completion)
    steps = []
    final_answer = ""
    confidences = []

    for obj in objs:
        action = obj.get("action", "")
        try:
            conf = float(obj.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        confidences.append(conf)
        steps.append(obj)
        if action == "answer":
            final_answer = str(obj.get("answer", ""))
            break  # stop at first answer action

    # Fallback: regex for "answer": "..."
    if not final_answer:
        m = re.search(r'"answer"\s*:\s*"([^"]+)"', completion)
        if m:
            final_answer = m.group(1)

    return {
        "steps":        steps,
        "final_answer": final_answer,
        "n_steps":      len(steps),
        "confidences":  confidences,
        "has_answer":   bool(final_answer),
    }


# ---------------------------------------------------------------------------
# Individual reward functions (GRPO-compatible signatures)
# ---------------------------------------------------------------------------

def correctness_reward(completions: list[str], **kwargs) -> list[float]:
    """
    +1.0 if the parsed final answer matches the gold answer, else 0.0.
    Gold answers come from kwargs['answer'] (list of strings, one per prompt).
    """
    gold_answers = kwargs.get("answer", [""] * len(completions))
    rewards = []
    for comp, gold in zip(completions, gold_answers):
        trace = parse_trace(comp)
        rewards.append(1.0 if _is_correct(trace["final_answer"], gold) else 0.0)
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Reward for producing a well-structured agent trace.

    Scoring:
      +0.1  any valid JSON object in the output
      +0.1  at least one step with required keys (thought, action, confidence)
      +0.1  ends with an "answer" action
    Max: 0.3
    """
    rewards = []
    required_keys = {"thought", "action", "confidence"}
    for comp in completions:
        score = 0.0
        objs = _extract_json_objects(comp)
        if objs:
            score += 0.1
        if any(required_keys.issubset(o.keys()) for o in objs):
            score += 0.1
        if any(o.get("action") == "answer" for o in objs):
            score += 0.1
        rewards.append(score)
    return rewards


def efficiency_reward(
    completions: list[str],
    alpha: float = 0.1,
    beta:  float = 0.05,
    epsilon: float = 0.05,
    **kwargs,
) -> list[float]:
    """
    Penalise verbose and low-info-gain traces.

    Score = -alpha * n_steps - beta * n_low_info_gain_steps
    """
    rewards = []
    for comp in completions:
        trace = parse_trace(comp)
        step_pen = -alpha * trace["n_steps"]
        confs = trace["confidences"]
        n_low = sum(
            1 for i in range(1, len(confs))
            if jsd_bernoulli(confs[i], confs[i - 1]) < epsilon
        )
        jsd_pen = -beta * n_low
        rewards.append(step_pen + jsd_pen)
    return rewards


# ---------------------------------------------------------------------------
# Combined reward (used as the single GRPOTrainer reward_funcs entry)
# ---------------------------------------------------------------------------

def combined_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Sum of correctness + format + efficiency rewards.
    This is the single reward function to pass to GRPOTrainer if you prefer
    one combined score over separate functions.
    """
    corr = correctness_reward(completions, **kwargs)
    fmt  = format_reward(completions, **kwargs)
    eff  = efficiency_reward(completions, **kwargs)
    return [c + f + e for c, f, e in zip(corr, fmt, eff)]
