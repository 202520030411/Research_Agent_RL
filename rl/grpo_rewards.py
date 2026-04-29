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
from collections import Counter


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


def _token_set(s: str) -> set[str]:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = "".join(c if c not in string.punctuation else " " for c in s)
    return {tok for tok in re.sub(r"\s+", " ", s).split() if tok}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _extract_doc_refs(completion: str) -> set[str]:
    """
    Extract document references touched by a trajectory.

    Includes explicit read(document=...) arguments and any doc_id strings in
    observations/search results. This intentionally works on the full episode
    text because GRPO rollouts contain real BM25 observations.
    """
    refs = {m.group(0) for m in re.finditer(r"\bdoc_\d+\b", completion)}
    for obj in _extract_json_objects(completion):
        if obj.get("action") == "read":
            doc = str(obj.get("document", "")).strip()
            if doc:
                refs.add(_normalize_answer(doc))
    return refs


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

    Kept small relative to correctness_reward (max +1.0) so format cannot
    dominate the learning signal when correctness is 0 across a group.

    Scoring:
      +0.03  any valid JSON object in the output
      +0.03  at least one step with required keys (thought, action, confidence)
      +0.04  ends with an "answer" action
    Max: 0.10
    """
    rewards = []
    required_keys = {"thought", "action", "confidence"}
    for comp in completions:
        score = 0.0
        objs = _extract_json_objects(comp)
        if objs:
            score += 0.03
        if any(required_keys.issubset(o.keys()) for o in objs):
            score += 0.03
        if any(o.get("action") == "answer" for o in objs):
            score += 0.04
        rewards.append(score)
    return rewards


def dense_format_reward(
    completions: list[str],
    max_score: float = 0.20,
    **kwargs,
) -> list[float]:
    """
    Continuous format/action-quality reward.

    Unlike `format_reward`, this does not saturate as soon as a valid JSON
    object exists. It scores every parsed step for schema quality, action
    validity, confidence validity, and action-specific fields. This gives GRPO
    usable within-group variance even when all rollouts are wrong.
    """
    rewards = []
    valid_actions = {"search", "read", "answer"}

    for comp in completions:
        objs = _extract_json_objects(comp)
        if not objs:
            rewards.append(0.0)
            continue

        step_scores = []
        seen_answer = False
        for obj in objs:
            action = obj.get("action", "")
            score = 0.0

            if "thought" in obj and str(obj.get("thought", "")).strip():
                score += 0.15
            if action in valid_actions:
                score += 0.20

            try:
                conf = float(obj.get("confidence"))
                if 0.0 <= conf <= 1.0:
                    score += 0.20
            except (TypeError, ValueError):
                pass

            if action == "search" and str(obj.get("query", "")).strip():
                score += 0.25
            elif action == "read" and str(obj.get("document", "")).strip():
                score += 0.25
            elif action == "answer" and str(obj.get("answer", "")).strip():
                score += 0.25
                seen_answer = True

            # Prefer one JSON object per turn/action; unknown extra fields are
            # allowed but do not add reward.
            step_scores.append(min(score, 1.0))

            if action == "answer":
                break

        trace_score = sum(step_scores) / max(len(step_scores), 1)
        if seen_answer:
            trace_score += 0.10

        rewards.append(max_score * min(trace_score, 1.0))

    return rewards


def query_novelty_reward(
    completions: list[str],
    max_score: float = 0.10,
    **kwargs,
) -> list[float]:
    """
    Reward non-redundant search queries within a trajectory.

    Each search receives novelty = 1 - max lexical Jaccard overlap with any
    previous query in the same rollout. Repeated/rephrased queries therefore
    get lower reward, while second-hop queries with new entities/relations get
    higher reward.
    """
    rewards = []
    for comp in completions:
        queries = [
            str(obj.get("query", "")).strip()
            for obj in _extract_json_objects(comp)
            if obj.get("action") == "search" and str(obj.get("query", "")).strip()
        ]

        if not queries:
            rewards.append(0.0)
            continue

        novelties = []
        previous: list[str] = []
        for query in queries:
            if not previous:
                novelty = 1.0
            else:
                novelty = 1.0 - max(_jaccard(query, prev) for prev in previous)
            novelties.append(max(0.0, novelty))
            previous.append(query)

        # HotpotQA is 2-hop: one decent query is not enough. Scale by progress
        # toward at least two searches, then reward novelty among them.
        two_hop_progress = min(len(queries), 2) / 2.0
        rewards.append(max_score * two_hop_progress * (sum(novelties) / len(novelties)))

    return rewards


def trajectory_shape_reward(
    completions: list[str],
    max_score: float = 0.10,
    **kwargs,
) -> list[float]:
    """
    Reward progress toward the SFT/HotpotQA 2-hop action pattern.

    This is intentionally a soft prefix score, not a hard template: partial
    progress still gets partial reward, which creates variance in all-wrong
    groups.
    """
    target = ["search", "read", "search", "read", "answer"]
    rewards = []
    for comp in completions:
        actions = [
            str(obj.get("action", "")).strip()
            for obj in _extract_json_objects(comp)
        ]
        prefix = 0
        for got, want in zip(actions, target):
            if got != want:
                break
            prefix += 1
        rewards.append(max_score * (prefix / len(target)))
    return rewards


def cross_rollout_doc_overlap_reward(
    completions: list[str],
    max_score: float = 0.10,
    **kwargs,
) -> list[float]:
    """
    Reward evidence agreement across rollouts for the same question.

    GRPO passes the G completions for one question together, so a document
    touched by multiple rollouts is a weak unsupervised signal of relevance.
    This term creates continuous intra-group variance for all-wrong groups
    without requiring gold document labels.
    """
    doc_sets = [_extract_doc_refs(comp) for comp in completions]
    if len(doc_sets) <= 1:
        return [0.0 for _ in completions]

    doc_counts = Counter(doc for docs in doc_sets for doc in docs)
    denom = max(len(doc_sets) - 1, 1)
    rewards = []
    for docs in doc_sets:
        if not docs:
            rewards.append(0.0)
            continue
        agreement = sum((doc_counts[doc] - 1) / denom for doc in docs) / len(docs)
        rewards.append(max_score * agreement)
    return rewards


def continuous_auxiliary_reward(
    completions: list[str],
    format_weight: float = 1.0,
    query_weight: float = 1.0,
    shape_weight: float = 1.0,
    doc_weight: float = 1.0,
    **kwargs,
) -> list[float]:
    """
    Dense tie-breaking reward for multi-turn GRPO.

    Max contribution with default weights is 0.50, so a correct answer (+1)
    remains dominant, but all-wrong groups can still produce non-zero
    advantages from continuous structure/retrieval signals.
    """
    r_fmt_dense = dense_format_reward(completions, **kwargs)
    r_query = query_novelty_reward(completions, **kwargs)
    r_shape = trajectory_shape_reward(completions, **kwargs)
    r_doc = cross_rollout_doc_overlap_reward(completions, **kwargs)
    return [
        format_weight * f + query_weight * q + shape_weight * s + doc_weight * d
        for f, q, s, d in zip(r_fmt_dense, r_query, r_shape, r_doc)
    ]


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
